"""Point d'entrée exécutable de l'agent : construit le graphe LangGraph et
expose une fonction `chat()` réutilisable (par le CLI ci-dessous ou par
l'interface Streamlit dans app_streamlit.py).
"""

import asyncio
import uuid
from typing import List, Optional, Tuple

from langchain_core.messages import BaseMessage, HumanMessage

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


async def chat(
    user_input: str,
    history: List[BaseMessage],
    user_id: str,
    session_id: str,
) -> Tuple[str, List[BaseMessage]]:
    """Envoie un message utilisateur à l'agent.

    Args:
        user_input: le texte tapé par l'utilisateur.
        history: l'historique des messages LangChain déjà échangés dans la session.
        user_id: identifiant stable du participant (sert de clé pour la mémoire long-terme).
        session_id: identifiant de la conversation en cours.

    Returns:
        Un tuple (réponse texte de l'agent, historique complet mis à jour).
    """
    graph = get_graph()
    state = State(
        messages=history + [HumanMessage(content=user_input)],
        user_id=user_id,
        session_id=session_id,
    )
    result = await graph.ainvoke(state)
    updated_history = result["messages"]

    reply = next(
        (m.content for m in reversed(updated_history) if m.type == "ai" and m.content),
        "",
    )
    return reply, updated_history


async def _cli_main():
    print("YoupiAgent - assistant de support d'événement (tapez 'exit' pour quitter)")
    user_id = input("Votre email (identifiant participant) : ").strip() or "anonymous"
    session_id = str(uuid.uuid4())
    history: List[BaseMessage] = []

    while True:
        try:
            user_input = input("Vous > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        reply, history = await chat(user_input, history, user_id, session_id)
        print(f"Agent > {reply}")


if __name__ == "__main__":
    asyncio.run(_cli_main())
