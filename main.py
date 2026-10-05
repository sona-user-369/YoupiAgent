# Agent de support client pour un événement : gestion des inscriptions et
# des renseignements sur le programme, avec une mémoire long-terme (Mem0)
# qui retient les informations utiles sur chaque participant d'une
# conversation à l'autre.

import json
import operator
import uuid
from datetime import datetime, timezone
from typing import Annotated, Optional

import tiktoken
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field
from rodiumai import RodiumAI
from rodiumai.integrations.langchain import ChatRodiumAI

import database
import settings_yp
from errors import record_error
from memory import DEFAULT_MEMORY_TYPE, get_store_for
from tools import BUSINESS_TOOLS

from langgraph.graph import StateGraph

MAX_TOOL_ITERATIONS = 10
RAG_TOOL_NAME = "search_event_documents"

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

SYSTEM_PROMPT_TEMPLATE = """Tu es l'assistant de support client de l'événement organisé ici. Il n'y a \
qu'UN SEUL événement : celui décrit dans les documents officiels accessibles via l'outil \
search_event_documents. Ces documents sont ta seule source de vérité sur l'événement (nom, \
description, dates, lieu, programme, sessions, intervenants, tarifs, accès, parking, \
restauration, badge, règles...).

Règles sur les informations de l'événement :
- Pour toute question concernant l'événement, appelle d'abord search_event_documents avec une \
requête précise, puis réponds uniquement à partir des passages retournés. Si la première \
recherche est insuffisante, relance-en une autre avec une formulation différente.
- Ne demande JAMAIS au participant de préciser de quel événement il parle : il s'agit toujours de \
l'événement des documents. Quand tu présentes ou nommes l'événement, utilise son nom exact tel \
qu'il figure dans les documents.
- Si une question est vague (« c'est quand ? », « où ça se passe ? », « parle-moi de l'événement »), \
interprète-la comme portant sur cet événement et recherche l'information au lieu de demander \
des précisions. Ne pose une question de clarification que si, après recherche, plusieurs \
réponses différentes restent possibles (ex : plusieurs sessions correspondant à la demande).
- N'invente jamais un nom, une date, un horaire, un prix ou un intervenant. Si l'information \
n'apparaît pas dans les documents, dis-le simplement (« Je n'ai pas cette information ») et \
propose de contacter les organisateurs, sans supposer de réponse.
- Réponds directement à la question posée, sans reformuler la demande ni proposer d'aide générique.

Règles sur les inscriptions :
- Demande le nom et l'email du participant avant de l'inscrire si tu ne les as pas déjà.
- Avant d'inscrire un participant, vérifie avec check_registration s'il est déjà inscrit : si c'est \
le cas, dis-le-lui au lieu de l'inscrire une seconde fois.
- Utilise les outils d'inscription (register_participant, cancel_registration, get_registration, \
check_session_availability) pour toute action ou consultation d'inscription ; ne devine jamais un \
état d'inscription ou un nombre de places.

Style : concis, courtois, en français sauf si le participant écrit dans une autre langue.

{memory_section}"""


def _format_history_section(history: list[dict]) -> str:
    """Section mémoire du mode « classique » : derniers messages injectés tels quels."""
    if not history:
        return ""
    lines = "\n".join(
        f"{'Participant' if m['role'] == 'user' else 'Assistant'} : {m['content']}" for m in history
    )
    return f"Derniers messages des conversations avec ce participant :\n{lines}"


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


def _last_user_text(messages: list) -> str:
    return next((m.content for m in reversed(messages) if m.type == "human"), "")


class State(BaseModel):
    messages: Annotated[list, operator.add] = Field(default_factory=list)
    user_id: str = "anonymous"
    session_id: str = "default"
    turn_id: Optional[str] = None
    memory_type: str = DEFAULT_MEMORY_TYPE  # "classic" ou "mem0", fixé au début de la conversation
    # Contexte préparé par le nœud "prepare", lu par les nœuds suivants.
    system_prompt: str = ""
    memories: list = Field(default_factory=list)
    last_exchange_raw: list = Field(default_factory=list)


class YoupiAgent:
    def __init__(self):
        database.init_db()
        self.llm = RodiumAI(api_key=settings_yp.RODIUMAI_APIKEY)
        self.tools = list(BUSINESS_TOOLS)
        self.tools_by_name = {t.name: t for t in self.tools}
        self.chat_model = ChatRodiumAI(client=self.llm, model=settings_yp.RODIUMAI_MODEL)
        self.model_with_tools = self.chat_model.bind_tools(self.tools)

    async def _run_tool_call(self, tool_call: dict) -> ToolMessage:
        tool_obj = self.tools_by_name.get(tool_call["name"])
        if tool_obj is None:
            content = f"Erreur : outil '{tool_call['name']}' introuvable."
            record_error(f"tool:{tool_call['name']}", content)
        else:
            try:
                result = tool_obj.invoke(tool_call["args"])
            except Exception as e:
                record_error(f"tool:{tool_call['name']}", f"L'outil {tool_call['name']} a échoué", exc=e)
                result = {"status": "error", "message": str(e)}
            content = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
        return ToolMessage(content=content, tool_call_id=tool_call["id"], name=tool_call["name"])

    async def prepare(self, state: State):
        """Prépare le contexte du tour : souvenirs Mem0 + dernier échange de la session."""
        # On ne reçoit ici que le nouveau message de l'utilisateur : on ne
        # passe jamais tout l'historique brut de la conversation à l'agent.
        last_user_text = _last_user_text(state.messages)
        memory = get_store_for(state.memory_type)
        if state.memory_type == "classic":
            memories, last_exchange_raw = [], []
            memory_section = _format_history_section(memory.get_history(state.user_id))
        else:
            memories = memory.search(query=last_user_text, user_id=state.user_id)
            last_exchange_raw = memory.get_messages(state.session_id)
            memory_section = _format_memory_section(memories)
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(memory_section=memory_section)
        # Contexte court-terme : uniquement le dernier échange de la session
        # (au-delà, on compte sur Mem0 plutôt que sur l'historique brut).
        return {
            "system_prompt": system_prompt,
            "memories": memories,
            "last_exchange_raw": last_exchange_raw,
        }

    async def agent(self, state: State):
        """Étape « Reason » : le LLM répond ou demande des outils."""
        conversation = (
            [SystemMessage(state.system_prompt)]
            + _session_messages_to_langchain(state.last_exchange_raw)
            + state.messages
        )
        response = await self.model_with_tools.ainvoke(conversation)
        return {"messages": [response]}

    async def act(self, state: State):
        """Étape « Act » : exécute les appels d'outils du dernier message du LLM."""
        tool_messages = [await self._run_tool_call(tc) for tc in state.messages[-1].tool_calls]
        return {"messages": tool_messages}

    def route(self, state: State) -> str:
        """Boucle vers les outils tant que le LLM en demande, dans la limite de MAX_TOOL_ITERATIONS."""
        llm_calls = sum(isinstance(m, AIMessage) for m in state.messages)
        if state.messages[-1].tool_calls and llm_calls < MAX_TOOL_ITERATIONS:
            return "act"
        return "finalize"

    async def finalize(self, state: State):
        """Journalise le tour, met à jour les mémoires et renvoie l'état final."""
        memory = get_store_for(state.memory_type)
        current_turn = [m for m in state.messages if m.type != "system"]
        last_user_text = _last_user_text(current_turn)
        last_exchange_raw = state.last_exchange_raw
        last_exchange = _session_messages_to_langchain(last_exchange_raw)

        tool_counts: dict[str, int] = {}
        rag_tokens = 0  # tokens des passages RAG renvoyés au LLM pendant ce tour
        for m in state.messages:
            if isinstance(m, AIMessage):
                for tc in m.tool_calls:
                    tool_counts[tc["name"]] = tool_counts.get(tc["name"], 0) + 1
            elif isinstance(m, ToolMessage) and m.name == RAG_TOOL_NAME:
                rag_tokens += _count_tokens(m.content)

        final_ai_text = next(
            (m.content for m in reversed(state.messages) if isinstance(m, AIMessage) and m.content),
            "",
        )

        if not final_ai_text:
            # Réponse vide : sans trace, l'utilisateur ne voit rien et ne sait pas pourquoi.
            if state.messages[-1].tool_calls:
                record_error(
                    "max_iterations",
                    f"Aucune réponse : {MAX_TOOL_ITERATIONS} tours d'outils sans réponse finale",
                    details=f"Outils appelés : {tool_counts}",
                )
            else:
                record_error(
                    "empty_reply",
                    "Le LLM a renvoyé une réponse vide",
                    details=f"Dernier message : {state.messages[-1]!r}\nOutils appelés : {tool_counts}",
                )

        mem0_events = memory.remember_exchange(
            state.session_id, state.user_id, last_user_text, final_ai_text
        )

        # Diagnostic pour l'écran de logs : ce qui a réellement été envoyé au
        # LLM pour ce tour (system prompt + dernier échange + message courant),
        # et ce que Mem0 a cherché/retenu.
        first_human = next(i for i, m in enumerate(state.messages) if m.type == "human")
        system_tokens = _count_tokens(state.system_prompt) + _TOKENS_PER_MESSAGE_OVERHEAD
        short_term_tokens = _count_message_tokens(last_exchange)
        current_tokens = _count_message_tokens(state.messages[: first_human + 1])
        turn_id = state.turn_id or str(uuid.uuid4())
        database.save_turn(
            turn_id=turn_id,
            session_id=state.session_id,
            user_id=state.user_id,
            user_text=last_user_text,
            assistant_text=final_ai_text,
            tokens_sent=system_tokens + short_term_tokens + current_tokens,
            rag_tokens=rag_tokens,
            tool_counts=tool_counts,
            memory_type=state.memory_type,
        )
        memory.log_turn(
            state.session_id,
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "user_id": state.user_id,
                "memory_type": state.memory_type,
                "user_text": last_user_text,
                "mem0_search": {"query": last_user_text, "results": state.memories},
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
        return {"turn_id": turn_id}

    def construct(self):
        """Graphe ReAct : prepare → agent ⇄ act → finalize."""
        g = StateGraph(State)
        g.add_node("prepare", self.prepare)
        g.add_node("agent", self.agent)
        g.add_node("act", self.act)
        g.add_node("finalize", self.finalize)
        g.set_entry_point("prepare")
        g.add_edge("prepare", "agent")
        g.add_conditional_edges("agent", self.route, {"act": "act", "finalize": "finalize"})
        g.add_edge("act", "agent")
        g.set_finish_point("finalize")
        return g.compile()
