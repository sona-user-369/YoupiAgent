"""Journal des erreurs « sourdines » de l'agent.

Plusieurs composants avalent volontairement leurs exceptions pour ne pas
casser la conversation (mémoire Mem0 en best-effort, outils qui renvoient un
message d'erreur au LLM...), ce qui laisse l'utilisateur devant une réponse
vide sans explication. Tout ce qui est avalé est enregistré ici, en base, et
consultable dans l'onglet « Erreurs ».

Le contexte (session, participant, tour) est porté par une ContextVar pour que
memory.py, tools.py, etc. n'aient pas à le connaître.
"""

import traceback
from contextvars import ContextVar
from typing import Optional

import database

_context: ContextVar[dict] = ContextVar("error_context", default={})


def set_context(session_id: str, user_id: str, turn_id: Optional[str]) -> None:
    _context.set({"session_id": session_id, "user_id": user_id, "turn_id": turn_id})


def record_error(
    source: str,
    message: str,
    *,
    exc: Optional[BaseException] = None,
    level: str = "error",
    details: Optional[str] = None,
) -> None:
    """Enregistre une erreur ; ne lève jamais (le journal ne doit pas casser l'agent).

    Args:
        source: composant en cause (ex: "llm", "tool:search_event_documents", "mem0_search", "empty_reply").
        message: description courte, lisible.
        exc: exception d'origine, dont la stack trace sera conservée.
        level: "error" (l'utilisateur est impacté) ou "warning" (dégradation silencieuse).
        details: informations complémentaires (utilisé à la place de la stack trace).
    """
    if exc is not None:
        details = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        message = f"{message} : {type(exc).__name__}: {exc}"
    ctx = _context.get()
    try:
        database.log_error(
            source=source,
            level=level,
            message=message,
            details=details or "",
            session_id=ctx.get("session_id"),
            user_id=ctx.get("user_id"),
            turn_id=ctx.get("turn_id"),
        )
    except Exception:
        pass
