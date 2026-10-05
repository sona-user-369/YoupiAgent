"""Mémoire « classique » (alternative à Mem0, pour comparaison).

Aucun extraction de faits ni recherche sémantique : on garde simplement en
cache (process) les derniers messages des conversations de chaque participant,
et l'agent les réinjecte tels quels dans la section mémoire du system prompt.

Expose la même interface que `MemoryStore` (memory.py) pour que l'agent puisse
utiliser l'un ou l'autre sans branchement côté stockage.
"""

import threading
from collections import deque
from typing import Deque, Dict, List, Optional

# Nombre de derniers messages (utilisateur + assistant) réinjectés dans le prompt.
HISTORY_WINDOW = 30


class CacheMemoryStore:
    _LOG_WINDOW = 50

    def __init__(self, window: int = HISTORY_WINDOW):
        self._lock = threading.Lock()
        self._window = window
        # Historique par participant (user_id), à travers ses conversations.
        self._history: Dict[str, Deque[Dict[str, str]]] = {}
        self._logs: Dict[str, List[Dict]] = {}

    def remember_exchange(
        self, session_id: str, user_id: str, user_text: str, assistant_text: str
    ) -> List[Dict]:
        """Met en cache l'échange dans l'historique du participant (`session_id` ignoré :
        l'historique traverse les sessions). Pas d'évènements : rien n'est extrait."""
        if not user_text or not assistant_text:
            return []
        with self._lock:
            history = self._history.setdefault(user_id, deque(maxlen=self._window))
            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": assistant_text})
        return []

    def get_history(self, user_id: str) -> List[Dict[str, str]]:
        """Les derniers messages (30 max) du participant, du plus ancien au plus récent."""
        with self._lock:
            return list(self._history.get(user_id, []))

    def log_turn(self, session_id: str, entry: Dict) -> None:
        with self._lock:
            logs = self._logs.setdefault(session_id, [])
            logs.append(entry)
            del logs[: -self._LOG_WINDOW]

    def get_logs(self, session_id: str) -> List[Dict]:
        with self._lock:
            return list(self._logs.get(session_id, []))

    def flush_user(self, user_id: str) -> None:
        with self._lock:
            self._history.pop(user_id, None)
            sessions = [
                sid for sid, logs in self._logs.items()
                if any(e.get("user_id") == user_id for e in logs)
            ]
            for sid in sessions:
                self._logs.pop(sid, None)

    def get_all(self, user_id: str) -> List[str]:
        return [m["content"] for m in self.get_history(user_id)]


_default_store: Optional[CacheMemoryStore] = None
_default_store_lock = threading.Lock()


def get_cache_memory_store() -> CacheMemoryStore:
    global _default_store
    if _default_store is None:
        with _default_store_lock:
            if _default_store is None:
                _default_store = CacheMemoryStore()
    return _default_store
