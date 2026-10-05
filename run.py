"""Point d'entrée exécutable de l'agent : construit le graphe LangGraph et
expose une fonction `chat()` réutilisable (par le CLI ci-dessous ou par
l'interface Streamlit dans app_streamlit.py).
"""

import asyncio
import uuid
from typing import Optional

from langchain_core.messages import HumanMessage

from errors import record_error, set_context
from memory import DEFAULT_MEMORY_TYPE
from main import State, YoupiAgent

_agent: Optional[YoupiAgent] = None
_graph = None


def get_graph():
    """Construit (une seule fois) et retourne le graphe compilé de l'agent."""
    global _agent, _graph
    if _graph is None:
        _agent = YoupiAgent()
        _graph = _agent.construct()
    return _graph


async def chat_turn(
    user_input: str, user_id: str, session_id: str, memory_type: str = DEFAULT_MEMORY_TYPE
) -> tuple[str, str]:
    """Envoie un message utilisateur à l'agent.

    On ne passe jamais l'historique brut de la conversation à l'agent : le
    contexte court-terme (dernier échange) et long-terme (faits Mem0) sont
    gérés en interne par MemoryStore, indexés sur session_id / user_id.

    Args:
        user_input: le texte tapé par l'utilisateur.
        user_id: identifiant stable du participant (sert de clé pour la mémoire long-terme).
        session_id: identifiant de la conversation en cours.
        memory_type: "classic" ou "mem0", choisi au début de la conversation.

    Returns:
        (réponse texte de l'agent, identifiant du tour pour l'annotation).
    """
    turn_id = str(uuid.uuid4())
    set_context(session_id, user_id, turn_id)
    state = State(
        messages=[HumanMessage(content=user_input)],
        user_id=user_id,
        session_id=session_id,
        turn_id=turn_id,
        memory_type=memory_type,
    )
    try:
        result = await get_graph().ainvoke(state)
    except Exception as e:
        record_error("agent", "L'agent a planté sur ce message", exc=e)
        raise

    reply = next(
        (m.content for m in reversed(result["messages"]) if m.type == "ai" and m.content),
        "",
    )
    return reply, result["turn_id"]


async def chat(user_input: str, user_id: str, session_id: str) -> str:
    """Comme chat_turn, mais ne retourne que la réponse texte."""
    reply, _ = await chat_turn(user_input, user_id, session_id)
    return reply


async def _cli_main():
    print("YoupiAgent - assistant de support d'événement (tapez 'exit' pour quitter)")
    user_id = input("Votre email (identifiant participant) : ").strip() or "anonymous"
    session_id = str(uuid.uuid4())

    while True:
        try:
            user_input = input("Vous > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        reply = await chat(user_input, user_id, session_id)
        print(f"Agent > {reply}")


if __name__ == "__main__":
    asyncio.run(_cli_main())
