"""Local Firestore replacement using SQLite.

This connector stores session logs in a SQLite database allowing the
application to run without access to Google Cloud Firestore.  The API
mirrors the subset of functionality used by the rest of the code base.
"""

from __future__ import annotations

from abc import ABC
import sqlite3

from .core import DBConnector


class LocalFirestoreConnector(DBConnector, ABC):
    """Persist chat logs to a local SQLite table."""

    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        with self.conn:
            self.conn.execute(
                """CREATE TABLE IF NOT EXISTS session_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                user_id TEXT,
                user_question TEXT,
                bot_response TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )"""
            )

    def log_chat(
        self,
        session_id: str,
        user_question: str,
        bot_response: str,
        user_id: str = "TEST",
    ) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO session_logs (session_id, user_id, user_question, bot_response)"
                " VALUES (?, ?, ?, ?)",
                (session_id, user_id, user_question, bot_response),
            )

    def get_chat_logs_for_session(self, session_id: str):
        cur = self.conn.cursor()
        cur.execute(
            "SELECT user_question, bot_response, timestamp FROM session_logs"
            " WHERE session_id=? ORDER BY timestamp",
            (session_id,),
        )
        rows = cur.fetchall()
        return [
            {
                "user_question": r[0],
                "bot_response": r[1],
                "timestamp": r[2],
            }
            for r in rows
        ]


__all__ = ["LocalFirestoreConnector"]

