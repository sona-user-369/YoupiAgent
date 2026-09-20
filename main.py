# Agent de support client pour un événement : gestion des inscriptions et
# des renseignements sur le programme, avec une mémoire long-terme (Mem0)
# qui retient les informations utiles sur chaque participant d'une
# conversation à l'autre.

import json
import operator
from typing import Annotated, Optional

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field
from rodiumai import RodiumAI
from rodiumai.integrations.langchain import ChatRodiumAI

import settings_yp
from memory import MemoryStore, get_memory_store
from tools import BUSINESS_TOOLS

from langgraph.graph import StateGraph

MAX_TOOL_ITERATIONS = 10

SYSTEM_PROMPT_TEMPLATE = """Tu es l'assistant de support client d'un événement. Tu aides les \
participants à s'inscrire, à connaître le programme (sessions, horaires, salles, places \
disponibles) et à répondre aux questions pratiques (lieu, date, parking, restauration, badge).

Règles :
- Utilise les outils à ta disposition pour toute inscription, annulation, consultation du \
programme ou question pratique : ne devine jamais une information que tu peux vérifier avec un outil.
- Demande le nom et l'email du participant avant de l'inscrire si tu ne les as pas déjà.
- Reste concis, courtois, et réponds en français sauf si le participant écrit dans une autre langue.

{memory_section}"""


def _format_memory_section(memories: list[str]) -> str:
    if not memories:
        return ""
    bullet_list = "\n".join(f"- {m}" for m in memories)
    return f"Informations déjà connues sur ce participant :\n{bullet_list}"


class State(BaseModel):
    messages: Annotated[list, operator.add] = Field(default_factory=list)
    user_id: str = "anonymous"
    session_id: str = "default"


class YoupiAgent:
    def __init__(self, memory: Optional[MemoryStore] = None):
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
        history = list(state.messages)

        last_user_text = next(
            (m.content for m in reversed(history) if m.type == "human"),
            "",
        )
        relevant_memories = self.memory.search(query=last_user_text, user_id=state.user_id)
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            memory_section=_format_memory_section(relevant_memories)
        )

        conversation = [SystemMessage(system_prompt)] + history
        model_with_tools = self.chat_model.bind_tools(self.tools)

        for _ in range(MAX_TOOL_ITERATIONS):
            response = await model_with_tools.ainvoke(conversation)
            conversation.append(response)

            if not response.tool_calls:
                break

            for tool_call in response.tool_calls:
                conversation.append(await self._run_tool_call(tool_call))

        new_messages = conversation[len(history) + 1 :]

        final_ai_text = next(
            (m.content for m in reversed(new_messages) if isinstance(m, AIMessage) and m.content),
            "",
        )

        self.memory.add_message(state.session_id, "user", last_user_text)
        self.memory.add_message(state.session_id, "assistant", final_ai_text)
        self.memory.remember_exchange(state.user_id, last_user_text, final_ai_text)

        return {"messages": new_messages}

    def construct(self):
        g = StateGraph(State)
        g.add_node("assistant", self.call)
        g.set_entry_point("assistant")
        g.set_finish_point("assistant")
        return g.compile()
