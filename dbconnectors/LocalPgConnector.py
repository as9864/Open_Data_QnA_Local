"""Local PostgreSQL connector.

This module provides a lightweight connector implementation that uses
``sqlalchemy`` and ``psycopg2`` to talk to a locally running PostgreSQL
instance.  It mirrors the :class:`PgConnector` interface used for the
cloud implementation but avoids any dependency on Google Cloud specific
libraries so that the application can be executed completely offline.
"""

from __future__ import annotations

from abc import ABC

import pandas as pd
from sqlalchemy import create_engine, text

from .core import DBConnector


class LocalPgConnector(DBConnector, ABC):
    """Connector for a local PostgreSQL database.

    Parameters
    ----------
    conn_str:
        SQLAlchemy style connection string (e.g.
        ``postgresql+psycopg2://user:pass@localhost/db``).
    """

    def __init__(self, conn_str: str):
        self.conn_str = conn_str
        self.engine = create_engine(conn_str)

    def getconn(self):
        """Return a new connection object."""

        return self.engine.connect()

    def retrieve_df(self, query: str) -> pd.DataFrame:
        """Execute *query* and return the result as a :class:`DataFrame`."""

        with self.getconn() as conn:
            return pd.read_sql_query(text(query), conn)


__all__ = ["LocalPgConnector"]

