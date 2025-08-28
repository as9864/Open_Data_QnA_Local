"""Local BigQuery-like connector using SQLite.

The original :mod:`dbconnectors.BQConnector` relies on the Google Cloud
BigQuery client.  For local development and testing we provide a
minimal implementation backed by SQLite.  It supports the small subset
of functionality used by the application – executing SQL and returning
results as :class:`pandas.DataFrame` objects and recording audit
information.
"""

from __future__ import annotations

from abc import ABC
import sqlite3
from datetime import datetime

import pandas as pd

from .core import DBConnector


class LocalBQConnector(DBConnector, ABC):
    """Connector that executes SQL against a local SQLite database."""

    def __init__(self, db_path: str, audit_table: str = "audit_log"):
        self.db_path = db_path
        self.audit_table = audit_table
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._ensure_audit_table()

    def _ensure_audit_table(self) -> None:
        with self.conn:
            self.conn.execute(
                f"""CREATE TABLE IF NOT EXISTS {self.audit_table} (
                    source_type TEXT,
                    user_grouping TEXT,
                    model_used TEXT,
                    question TEXT,
                    generated_sql TEXT,
                    execution_time TEXT,
                    full_log TEXT
                )"""
            )

    def getconn(self):
        return self.conn

    def retrieve_df(self, query: str) -> pd.DataFrame:
        return pd.read_sql_query(query, self.conn)

    def make_audit_entry(
        self,
        source_type: str,
        user_grouping: str,
        model: str,
        question: str,
        generated_sql: str,
        found_in_vector: bool,
        need_rewrite: bool,
        failure_step: str,
        error_msg: str,
        full_log_text: str,
    ) -> str:
        """Persist a minimal audit record to the local SQLite DB."""

        with self.conn:
            self.conn.execute(
                f"INSERT INTO {self.audit_table} VALUES (?,?,?,?,?,?,?)",
                (
                    source_type,
                    user_grouping,
                    model,
                    question,
                    generated_sql,
                    datetime.utcnow().isoformat(),
                    full_log_text,
                ),
            )
        return "OK"


__all__ = ["LocalBQConnector"]

