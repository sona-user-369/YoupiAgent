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
