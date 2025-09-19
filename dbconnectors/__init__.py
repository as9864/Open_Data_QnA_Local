"""Database connector factory.

The project can operate either against Google Cloud services or against
local databases for offline development.  This module exposes helper
functions that return the appropriate connector instances based on the
configuration loaded in :mod:`utilities`.
"""

from __future__ import annotations

import re

from .core import DBConnector  # re-export for convenience
from .PgConnector import PgConnector, pg_specific_data_types
from .BQConnector import BQConnector, bq_specific_data_types
from .FirestoreConnector import FirestoreConnector
from .LocalPgConnector import LocalPgConnector
from .LocalBQConnector import LocalBQConnector
from .LocalFirestoreConnector import LocalFirestoreConnector

from sqlalchemy.exc import OperationalError

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
    PG_QUERY_CONN,
    PG_VECTOR_CONN,
    PG_AUDIT_CONN,
    PG_CLOUD_QUERY_CONN,
    PG_CLOUD_VECTOR_CONN,
    PG_CLOUD_AUDIT_CONN,
    LOCAL_SQLITE_DB,
)


def _parse_cloud_pg_conn(raw: str) -> dict[str, str]:
    raw = (raw or "").strip()
    if not raw:
        return {}
    if "://" in raw:
        return {"connection_string": raw}

    overrides: dict[str, str] = {}
    for part in re.split(r"[;,\n]+", raw):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            key, value = part.split("=", 1)
            overrides[key.strip().lower()] = value.strip()
        else:
            overrides["database"] = part
    return overrides


def _build_cloud_pg_connector(raw: str, *, ensure_audit_table: bool = False) -> DBConnector:
    overrides = _parse_cloud_pg_conn(raw)
    if "connection_string" in overrides:
        return _build_local_pg_connector(overrides["connection_string"], ensure_audit_table=ensure_audit_table)

    project_id = overrides.get("project", PROJECT_ID)
    region = overrides.get("region", PG_REGION)
    instance = overrides.get("instance", PG_INSTANCE)
    database = overrides.get("database", PG_DATABASE)
    user = overrides.get("user", PG_USER)
    password = overrides.get("password", PG_PASSWORD)

    try:
        return PgConnector(
            project_id,
            region,
            instance,
            database,
            user,
            password,
            ensure_audit_table=ensure_audit_table,
        )
    except OperationalError:
        if ensure_audit_table:
            return PgConnector(
                project_id,
                region,
                instance,
                database,
                user,
                password,
                ensure_audit_table=False,
            )
        raise


def _build_local_pg_connector(conn_str: str, *, ensure_audit_table: bool = False) -> DBConnector:
    connection = conn_str or LOCAL_PG_CONN
    try:
        return LocalPgConnector(connection, ensure_audit_table=ensure_audit_table)
    except OperationalError:
        if ensure_audit_table:
            return LocalPgConnector(connection, ensure_audit_table=False)
        raise


def get_data_pg_connector() -> DBConnector:
    if CONNECTOR_BACKEND.lower() == "local":
        return _build_local_pg_connector(PG_QUERY_CONN)
    return _build_cloud_pg_connector(PG_CLOUD_QUERY_CONN)


def get_vector_pg_connector() -> DBConnector:
    if CONNECTOR_BACKEND.lower() == "local":
        return _build_local_pg_connector(PG_VECTOR_CONN)
    return _build_cloud_pg_connector(PG_CLOUD_VECTOR_CONN)


def get_audit_pg_connector() -> DBConnector:
    if CONNECTOR_BACKEND.lower() == "local":
        return _build_local_pg_connector(PG_AUDIT_CONN, ensure_audit_table=True)
    return _build_cloud_pg_connector(PG_CLOUD_AUDIT_CONN, ensure_audit_table=True)


def get_bq_connector() -> DBConnector:
    if CONNECTOR_BACKEND.lower() == "local":
        return LocalBQConnector(LOCAL_SQLITE_DB)
    return BQConnector(PROJECT_ID, BQ_REGION, BQ_OPENDATAQNA_DATASET_NAME, BQ_LOG_TABLE_NAME)


def get_firestore_connector() -> DBConnector:
    if CONNECTOR_BACKEND.lower() == "local":
        return LocalFirestoreConnector(LOCAL_SQLITE_DB)
    return FirestoreConnector(PROJECT_ID, "opendataqna-session-logs")


data_pgconnector = get_data_pg_connector()
vector_pgconnector = get_vector_pg_connector()
audit_pgconnector = get_audit_pg_connector()
pgconnector = data_pgconnector
bqconnector = get_bq_connector()
firestoreconnector = get_firestore_connector()

__all__ = [
    "DBConnector",
    "pgconnector",
    "data_pgconnector",
    "vector_pgconnector",
    "audit_pgconnector",
    "pg_specific_data_types",
    "bqconnector",
    "bq_specific_data_types",
    "firestoreconnector",
]
