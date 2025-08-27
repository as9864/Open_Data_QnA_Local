"""Database connector factory.

The project can operate either against Google Cloud services or against
local databases for offline development.  This module exposes helper
functions that return the appropriate connector instances based on the
configuration loaded in :mod:`utilities`.
"""

from __future__ import annotations

from .core import DBConnector  # re-export for convenience
from .PgConnector import PgConnector, pg_specific_data_types
from .BQConnector import BQConnector, bq_specific_data_types
from .FirestoreConnector import FirestoreConnector
from .LocalPgConnector import LocalPgConnector
from .LocalBQConnector import LocalBQConnector
from .LocalFirestoreConnector import LocalFirestoreConnector

from utilities import (
    PROJECT_ID,
    PG_INSTANCE,
    PG_DATABASE,
    PG_USER,
    PG_PASSWORD,
    PG_REGION,
    BQ_REGION,
    BQ_OPENDATAQNA_DATASET_NAME,
    BQ_LOG_TABLE_NAME,
    CONNECTOR_BACKEND,
    LOCAL_PG_CONN,
    LOCAL_SQLITE_DB,
)


def get_pg_connector() -> DBConnector:
    if CONNECTOR_BACKEND.lower() == "local":
        return LocalPgConnector(LOCAL_PG_CONN)
    return PgConnector(PROJECT_ID, PG_REGION, PG_INSTANCE, PG_DATABASE, PG_USER, PG_PASSWORD)


def get_bq_connector() -> DBConnector:
    if CONNECTOR_BACKEND.lower() == "local":
        return LocalBQConnector(LOCAL_SQLITE_DB)
    return BQConnector(PROJECT_ID, BQ_REGION, BQ_OPENDATAQNA_DATASET_NAME, BQ_LOG_TABLE_NAME)


def get_firestore_connector() -> DBConnector:
    if CONNECTOR_BACKEND.lower() == "local":
        return LocalFirestoreConnector(LOCAL_SQLITE_DB)
    return FirestoreConnector(PROJECT_ID, "opendataqna-session-logs")


pgconnector = get_pg_connector()
bqconnector = get_bq_connector()
firestoreconnector = get_firestore_connector()

__all__ = [
    "DBConnector",
    "pgconnector",
    "pg_specific_data_types",
    "bqconnector",
    "bq_specific_data_types",
    "firestoreconnector",
]

