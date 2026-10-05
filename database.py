"""Accès à la base SQLite des inscriptions (voir settings_yp.EVENT_DB_FILE).

Cette couche ne connaît que le stockage : les règles métier (programme,
validation des emails, messages destinés à l'agent) restent dans tools.py.
"""

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Dict, List, Optional

import settings_yp

_WRITE_LOCK = threading.Lock()


class SessionFullError(Exception):
    def __init__(self, session_id: str):
        super().__init__(f"Session complète : {session_id}")
        self.session_id = session_id


@contextmanager
def _connect():
    conn = sqlite3.connect(settings_yp.EVENT_DB_FILE)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Crée les tables si besoin et reprend l'ancien event_store.json s'il existe."""
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS participants (
                email TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                registered_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS session_registrations (
                email TEXT NOT NULL REFERENCES participants(email) ON DELETE CASCADE,
                session_id TEXT NOT NULL,
                PRIMARY KEY (email, session_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS turns (
                turn_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                user_text TEXT NOT NULL,
                assistant_text TEXT NOT NULL,
                tokens_sent INTEGER NOT NULL DEFAULT 0,
                rag_tokens INTEGER NOT NULL DEFAULT 0,
                tool_counts TEXT NOT NULL DEFAULT '{}',
                label TEXT CHECK (label IN ('correct', 'incorrect'))
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS errors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                session_id TEXT,
                user_id TEXT,
                turn_id TEXT,
                source TEXT NOT NULL,
                level TEXT NOT NULL DEFAULT 'error',
                message TEXT NOT NULL,
                details TEXT NOT NULL DEFAULT ''
            )
            """
        )
        turn_columns = {row["name"] for row in conn.execute("PRAGMA table_info(turns)")}
        if "memory_type" not in turn_columns:
            conn.execute("ALTER TABLE turns ADD COLUMN memory_type TEXT NOT NULL DEFAULT 'mem0'")
    _migrate_legacy_json_store()


def _migrate_legacy_json_store() -> None:
    legacy_file = settings_yp.LEGACY_EVENT_STORE_FILE
    if not legacy_file.exists():
        return
    with _connect() as conn:
        is_empty = conn.execute("SELECT COUNT(*) FROM participants").fetchone()[0] == 0
        if is_empty:
            legacy_store = json.loads(legacy_file.read_text(encoding="utf-8"))
            for email, participant in legacy_store.get("participants", {}).items():
                conn.execute(
                    "INSERT INTO participants (email, name, registered_at) VALUES (?, ?, ?)",
                    (email, participant["name"], participant["registered_at"]),
                )
                conn.executemany(
                    "INSERT INTO session_registrations (email, session_id) VALUES (?, ?)",
                    [(email, session_id) for session_id in participant.get("sessions", [])],
                )
    legacy_file.unlink()


def get_participant(email: str) -> Optional[Dict]:
    """Retourne l'inscription (nom, email, sessions, date) ou None si inconnue."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT name, registered_at FROM participants WHERE email = ?", (email,)
        ).fetchone()
        if row is None:
            return None
        sessions = [
            r["session_id"]
            for r in conn.execute(
                "SELECT session_id FROM session_registrations WHERE email = ?", (email,)
            )
        ]
    return {
        "name": row["name"],
        "email": email,
        "sessions": sessions,
        "registered_at": row["registered_at"],
    }


def upsert_participant(
    email: str, name: str, sessions: List[str], capacities: Dict[str, int]
) -> None:
    """Crée ou remplace l'inscription d'un participant.

    Lève SessionFullError si l'une des sessions demandées a déjà atteint sa
    capacité (`capacities` : session_id -> nombre de places).
    """
    with _WRITE_LOCK, _connect() as conn:
        for session_id in sessions:
            taken = conn.execute(
                "SELECT COUNT(*) FROM session_registrations WHERE session_id = ? AND email != ?",
                (session_id, email),
            ).fetchone()[0]
            if taken >= capacities[session_id]:
                raise SessionFullError(session_id)

        conn.execute(
            """
            INSERT INTO participants (email, name, registered_at) VALUES (?, ?, ?)
            ON CONFLICT(email) DO UPDATE SET name = excluded.name, registered_at = excluded.registered_at
            """,
            (email, name, datetime.now(timezone.utc).isoformat()),
        )
        conn.execute("DELETE FROM session_registrations WHERE email = ?", (email,))
        conn.executemany(
            "INSERT INTO session_registrations (email, session_id) VALUES (?, ?)",
            [(email, session_id) for session_id in sessions],
        )


def delete_participant(email: str) -> bool:
    """Supprime l'inscription ; retourne False si le participant n'existait pas."""
    with _WRITE_LOCK, _connect() as conn:
        return conn.execute("DELETE FROM participants WHERE email = ?", (email,)).rowcount > 0


def count_session_registrations(session_id: str) -> int:
    with _connect() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM session_registrations WHERE session_id = ?", (session_id,)
        ).fetchone()[0]


# --- Historique des tours (métriques + annotations) -------------------------

def save_turn(
    turn_id: str,
    session_id: str,
    user_id: str,
    user_text: str,
    assistant_text: str,
    tokens_sent: int,
    rag_tokens: int,
    tool_counts: Dict[str, int],
    memory_type: str = "mem0",
) -> None:
    with _WRITE_LOCK, _connect() as conn:
        conn.execute(
            """
            INSERT INTO turns (turn_id, session_id, user_id, timestamp, user_text,
                               assistant_text, tokens_sent, rag_tokens, tool_counts, memory_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                turn_id,
                session_id,
                user_id,
                datetime.now(timezone.utc).isoformat(),
                user_text,
                assistant_text,
                tokens_sent,
                rag_tokens,
                json.dumps(tool_counts, ensure_ascii=False),
                memory_type,
            ),
        )


def get_turns(session_id: Optional[str] = None) -> List[Dict]:
    """Tours enregistrés (du plus ancien au plus récent), d'une session ou de toutes."""
    query = "SELECT * FROM turns"
    params: tuple = ()
    if session_id is not None:
        query += " WHERE session_id = ?"
        params = (session_id,)
    with _connect() as conn:
        rows = conn.execute(query + " ORDER BY timestamp", params).fetchall()
    turns = [dict(r) for r in rows]
    for t in turns:
        t["tool_counts"] = json.loads(t["tool_counts"])
    return turns


def set_turn_label(turn_id: str, label: Optional[str]) -> None:
    """Annote la réponse d'un tour : 'correct', 'incorrect' ou None pour effacer."""
    with _WRITE_LOCK, _connect() as conn:
        conn.execute("UPDATE turns SET label = ? WHERE turn_id = ?", (label, turn_id))


# --- Journal des erreurs -----------------------------------------------------

def log_error(
    source: str,
    level: str,
    message: str,
    details: str = "",
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    turn_id: Optional[str] = None,
) -> None:
    with _WRITE_LOCK, _connect() as conn:
        conn.execute(
            """
            INSERT INTO errors (timestamp, session_id, user_id, turn_id, source, level, message, details)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                session_id,
                user_id,
                turn_id,
                source,
                level,
                message,
                details,
            ),
        )


def get_errors(session_id: Optional[str] = None) -> List[Dict]:
    """Erreurs enregistrées, de la plus récente à la plus ancienne (session donnée ou toutes)."""
    query = "SELECT * FROM errors"
    params: tuple = ()
    if session_id is not None:
        query += " WHERE session_id = ?"
        params = (session_id,)
    with _connect() as conn:
        return [dict(r) for r in conn.execute(query + " ORDER BY id DESC", params)]


def clear_errors(session_id: Optional[str] = None) -> None:
    with _WRITE_LOCK, _connect() as conn:
        if session_id is None:
            conn.execute("DELETE FROM errors")
        else:
            conn.execute("DELETE FROM errors WHERE session_id = ?", (session_id,))
