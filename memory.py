"""Mémoire de l'agent, en deux couches :

- Mémoire de session (courte durée) : uniquement le dernier échange
  (dernier message utilisateur + dernière réponse) de la session en cours,
  gardé en mémoire process. On ne repasse jamais tout l'historique brut à
  l'agent : au-delà du dernier échange, le contexte vient de Mem0.
- Mémoire long-terme (Mem0) : des faits extraits des échanges et persistés
  par participant (ex: préférences, restrictions alimentaires, sessions
  d'intérêt), retrouvables par recherche sémantique d'une conversation à
  l'autre.

Mem0 est configuré pour utiliser RodiumAI (déjà utilisé par l'agent, API
compatible OpenAI) à la fois comme LLM d'extraction de faits et comme
modèle d'embedding, et Qdrant en mode local (fichier sur disque, aucun
serveur requis) comme vector store.
"""

import threading
from typing import Dict, List, Optional

from mem0 import Memory

import settings_yp


def _build_mem0_config() -> dict:
    common_llm_embedder_kwargs = {
        "api_key": settings_yp.RODIUMAI_APIKEY,
        "openai_base_url": settings_yp.RODIUMAI_BASE_URL,
    }
    return {
        "llm": {
            "provider": "openai",
            "config": {
                "model": settings_yp.RODIUMAI_MODEL,
                **common_llm_embedder_kwargs,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": settings_yp.RODIUMAI_EMBEDDING_MODEL,
                "embedding_dims": settings_yp.RODIUMAI_EMBEDDING_DIMS,
                **common_llm_embedder_kwargs,
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "youpi_event_memories",
                "path": str(settings_yp.MEM0_STORAGE_DIR / "qdrant"),
                "embedding_model_dims": settings_yp.RODIUMAI_EMBEDDING_DIMS,
                "on_disk": True,
            },
        },
    }


def _extract_memory_texts(result) -> List[str]:
    entries = result.get("results", result) if isinstance(result, dict) else result
    return [entry.get("memory", "") for entry in entries if entry.get("memory")]


def _extract_search_entries(result) -> List[Dict]:
    """Faits trouvés par une recherche Mem0, avec leur score de similarité."""
    entries = result.get("results", result) if isinstance(result, dict) else result
    return [
        {"memory": e.get("memory", ""), "score": e.get("score")}
        for e in entries
        if e.get("memory")
    ]


def _extract_add_events(result) -> List[Dict]:
    """Évènements Mem0 (ADD/UPDATE/DELETE/NONE) suite à l'ajout d'un échange."""
    entries = result.get("results", result) if isinstance(result, dict) else result
    return [
        {"memory": e.get("memory", ""), "event": e.get("event", "")}
        for e in entries
        if e.get("memory")
    ]


class MemoryStore:
    """Gère la mémoire de session (dernier échange) et la mémoire long-terme (Mem0) de l'agent."""

    # Nombre de messages de session conservés (1 message utilisateur + 1 réponse
    # assistant = le dernier échange). Volontairement petit : le reste du
    # contexte vient de la recherche Mem0, pas de l'historique brut.
    _SESSION_WINDOW = 2

    # Nombre de tours de debug conservés par session pour l'écran de logs.
    _LOG_WINDOW = 50

    def __init__(self):
        self._lock = threading.Lock()
        self._sessions: Dict[str, List[Dict[str, str]]] = {}
        self._logs: Dict[str, List[Dict]] = {}
        self._memory = Memory.from_config(_build_mem0_config())

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Ajoute un message à la fenêtre de session (dernier échange uniquement)."""
        if not content:
            return
        with self._lock:
            messages = self._sessions.setdefault(session_id, [])
            messages.append({"role": role, "content": content})
            del messages[: -self._SESSION_WINDOW]

    def get_messages(self, session_id: str) -> List[Dict[str, str]]:
        """Récupère le dernier échange (courte fenêtre) de la session."""
        with self._lock:
            return list(self._sessions.get(session_id, []))

    def remember_exchange(self, user_id: str, user_text: str, assistant_text: str) -> List[Dict]:
        """Envoie un échange (question/réponse) à Mem0 pour extraction de faits durables.

        Retourne les évènements Mem0 (faits ajoutés/mis à jour/supprimés), utilisés
        uniquement pour l'écran de logs.
        """
        if not user_text or not assistant_text:
            return []
        try:
            result = self._memory.add(
                messages=[
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": assistant_text},
                ],
                user_id=user_id,
            )
            return _extract_add_events(result)
        except Exception:
            # La mémoire long-terme est un "best effort" : une panne ne doit
            # jamais casser la conversation en cours.
            return []

    def search(self, query: str, user_id: str, limit: int = 5) -> List[Dict]:
        """Recherche les faits mémorisés les plus pertinents pour une requête donnée.

        Retourne une liste de {"memory": ..., "score": ...}, triée par pertinence
        (le score de similarité est conservé pour l'écran de logs).
        """
        if not query or not user_id:
            return []
        try:
            result = self._memory.search(query=query, filters={"user_id": user_id}, top_k=limit)
        except Exception:
            return []
        return _extract_search_entries(result)

    def log_turn(self, session_id: str, entry: Dict) -> None:
        """Enregistre les infos de debug d'un tour (recherche Mem0, tokens envoyés...)."""
        with self._lock:
            logs = self._logs.setdefault(session_id, [])
            logs.append(entry)
            del logs[: -self._LOG_WINDOW]

    def get_logs(self, session_id: str) -> List[Dict]:
        """Récupère le journal de debug (recherche Mem0, tokens) de la session."""
        with self._lock:
            return list(self._logs.get(session_id, []))

    def flush_user(self, user_id: str) -> None:
        """Supprime toute la mémoire d'un participant : faits Mem0 et, pour toutes
        ses sessions, fenêtre court-terme et logs. Lève une exception si Mem0 échoue."""
        self._memory.delete_all(user_id=user_id)
        with self._lock:
            user_sessions = [
                session_id
                for session_id, logs in self._logs.items()
                if any(entry.get("user_id") == user_id for entry in logs)
            ]
            for session_id in user_sessions:
                self._sessions.pop(session_id, None)
                self._logs.pop(session_id, None)

    def get_all(self, user_id: str) -> List[str]:
        """Liste tous les faits mémorisés pour un participant."""
        try:
            result = self._memory.get_all(filters={"user_id": user_id})
        except Exception:
            return []
        return _extract_memory_texts(result)


_default_store: Optional[MemoryStore] = None
_default_store_lock = threading.Lock()


def get_memory_store() -> MemoryStore:
    """Retourne l'instance partagée de MemoryStore (créée à la demande)."""
    global _default_store
    if _default_store is None:
        with _default_store_lock:
            if _default_store is None:
                _default_store = MemoryStore()
    return _default_store
