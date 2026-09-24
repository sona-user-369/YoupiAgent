# Agent de support client pour un événement : gestion des inscriptions et
# des renseignements sur le programme, avec une mémoire long-terme (Mem0)
# qui retient les informations utiles sur chaque participant d'une
# conversation à l'autre.

import json
import operator
from datetime import datetime, timezone
from typing import Annotated, Optional

import tiktoken
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field
from rodiumai import RodiumAI
from rodiumai.integrations.langchain import ChatRodiumAI

import database
import settings_yp
from memory import MemoryStore, get_memory_store
from tools import BUSINESS_TOOLS

from langgraph.graph import StateGraph

MAX_TOOL_ITERATIONS = 10

# Encodage approximatif pour compter les tokens envoyés au LLM (à titre de
# diagnostic dans l'écran de logs) : RodiumAI/gpt-4o-mini n'a pas d'encodage
# tiktoken dédié, cl100k_base donne une estimation suffisamment fidèle.
_TOKENIZER = tiktoken.get_encoding("cl100k_base")
_TOKENS_PER_MESSAGE_OVERHEAD = 4  # approx. rôle + formatage, façon ChatML


def _count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_TOKENIZER.encode(text))


def _count_message_tokens(messages: list) -> int:
    return sum(
        _count_tokens(getattr(m, "content", "") or "") + _TOKENS_PER_MESSAGE_OVERHEAD
        for m in messages
    )

SYSTEM_PROMPT_TEMPLATE = """Tu es l'assistant de support client d'un événement. Tu aides les \
participants à s'inscrire, à connaître le programme (sessions, horaires, salles, places \
disponibles) et à répondre aux questions pratiques (lieu, date, parking, restauration, badge).

Règles :
- Utilise les outils à ta disposition pour toute inscription, annulation, consultation du \
programme ou question pratique : ne devine jamais une information que tu peux vérifier avec un outil.
- Demande le nom et l'email du participant avant de l'inscrire si tu ne les as pas déjà.
- Avant d'inscrire un participant, vérifie avec check_registration s'il est déjà inscrit : si c'est \
le cas, dis-le-lui au lieu de l'inscrire une seconde fois.
- Reste concis, courtois, et réponds en français sauf si le participant écrit dans une autre langue.

{memory_section}"""


def _format_memory_section(memories: list[dict]) -> str:
    if not memories:
        return ""
    bullet_list = "\n".join(f"- {m['memory']}" for m in memories)
    return f"Informations déjà connues sur ce participant :\n{bullet_list}"


def _session_messages_to_langchain(messages: list[dict]) -> list:
    """Convertit le dernier échange de session (dicts) en messages LangChain."""
    out = []
    for m in messages:
        if m["role"] == "user":
            out.append(HumanMessage(content=m["content"]))
        else:
            out.append(AIMessage(content=m["content"]))
    return out


class State(BaseModel):
    messages: Annotated[list, operator.add] = Field(default_factory=list)
    user_id: str = "anonymous"
    session_id: str = "default"


class YoupiAgent:
    def __init__(self, memory: Optional[MemoryStore] = None):
        database.init_db()
        self.llm = RodiumAI(api_key=settings_yp.RODIUMAI_APIKEY)
        self.tools = list(BUSINESS_TOOLS)
        self.tools_by_name = {t.name: t for t in self.tools}
        self.chat_model = ChatRodiumAI(client=self.llm, model=settings_yp.RODIUMAI_MODEL)
        self.memory = memory or get_memory_store()

    async def _run_tool_call(self, tool_call: dict) -> ToolMessage:
        tool_obj = self.tools_by_name.get(tool_call["name"])
        if tool_obj is None:
            content = f"Erreur : outil '{tool_call['name']}' introuvable."
        else:
            try:
                result = tool_obj.invoke(tool_call["args"])
            except Exception as e:
                result = {"status": "error", "message": str(e)}
            content = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
        return ToolMessage(content=content, tool_call_id=tool_call["id"])

    async def call(self, state: State):
        # On ne reçoit ici que le nouveau message de l'utilisateur : on ne
        # passe jamais tout l'historique brut de la conversation à l'agent.
        current_turn = list(state.messages)

        last_user_text = next(
            (m.content for m in reversed(current_turn) if m.type == "human"),
            "",
        )
        relevant_memories = self.memory.search(query=last_user_text, user_id=state.user_id)
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            memory_section=_format_memory_section(relevant_memories)
        )

        # Contexte court-terme : uniquement le dernier échange de la session
        # (au-delà, on compte sur Mem0 plutôt que sur l'historique brut).
        last_exchange_raw = self.memory.get_messages(state.session_id)
        last_exchange = _session_messages_to_langchain(last_exchange_raw)

        conversation = [SystemMessage(system_prompt)] + last_exchange + current_turn
        model_with_tools = self.chat_model.bind_tools(self.tools)

        turn_start = len(conversation)
        for _ in range(MAX_TOOL_ITERATIONS):
            response = await model_with_tools.ainvoke(conversation)
            conversation.append(response)

            if not response.tool_calls:
                break

            for tool_call in response.tool_calls:
                conversation.append(await self._run_tool_call(tool_call))

        new_messages = conversation[turn_start:]

        final_ai_text = next(
            (m.content for m in reversed(new_messages) if isinstance(m, AIMessage) and m.content),
            "",
        )

        mem0_events = self.memory.remember_exchange(state.user_id, last_user_text, final_ai_text)
        self.memory.add_message(state.session_id, "user", last_user_text)
        self.memory.add_message(state.session_id, "assistant", final_ai_text)

        # Diagnostic pour l'écran de logs : ce qui a réellement été envoyé au
        # LLM pour ce tour (system prompt + dernier échange + message courant),
        # et ce que Mem0 a cherché/retenu.
        system_tokens = _count_tokens(system_prompt) + _TOKENS_PER_MESSAGE_OVERHEAD
        short_term_tokens = _count_message_tokens(last_exchange)
        current_tokens = _count_message_tokens(current_turn)
        self.memory.log_turn(
            state.session_id,
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "user_id": state.user_id,
                "user_text": last_user_text,
                "mem0_search": {"query": last_user_text, "results": relevant_memories},
                "mem0_events": mem0_events,
                "short_term_window": last_exchange_raw,
                "tokens": {
                    "system_prompt": system_tokens,
                    "short_term": short_term_tokens,
                    "current_message": current_tokens,
                    "sent_to_llm": system_tokens + short_term_tokens + current_tokens,
                },
            },
        )

        return {"messages": new_messages}

    def construct(self):
        g = StateGraph(State)
        g.add_node("assistant", self.call)
        g.set_entry_point("assistant")
        g.set_finish_point("assistant")
        return g.compile()
