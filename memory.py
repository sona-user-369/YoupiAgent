"""Mémoire de l'agent, en deux couches :

- Mémoire de session (courte durée) : l'historique brut des messages échangés
  pendant la conversation en cours, gardé en mémoire process.
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


class MemoryStore:
    """Gère l'historique de session et la mémoire long-terme (Mem0) de l'agent."""

    def __init__(self):
        self._lock = threading.Lock()
        self._sessions: Dict[str, List[Dict[str, str]]] = {}
        self._memory = Memory.from_config(_build_mem0_config())

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Ajoute un message à l'historique court-terme de la session."""
        if not content:
            return
        with self._lock:
            self._sessions.setdefault(session_id, []).append({"role": role, "content": content})

    def get_messages(self, session_id: str) -> List[Dict[str, str]]:
        """Récupère l'historique court-terme des messages de la session."""
        with self._lock:
            return list(self._sessions.get(session_id, []))

    def remember_exchange(self, user_id: str, user_text: str, assistant_text: str) -> None:
        """Envoie un échange (question/réponse) à Mem0 pour extraction de faits durables."""
        if not user_text or not assistant_text:
            return
        try:
            self._memory.add(
                messages=[
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": assistant_text},
                ],
                user_id=user_id,
            )
        except Exception:
            # La mémoire long-terme est un "best effort" : une panne ne doit
            # jamais casser la conversation en cours.
            pass

    def search(self, query: str, user_id: str, limit: int = 5) -> List[str]:
        """Recherche les faits mémorisés les plus pertinents pour une requête donnée."""
        if not query or not user_id:
            return []
        try:
            result = self._memory.search(query=query, filters={"user_id": user_id}, top_k=limit)
        except Exception:
            return []
        return _extract_memory_texts(result)

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
