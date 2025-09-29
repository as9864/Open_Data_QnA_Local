"""Database connector helpers for local-only operation.

This module now focuses exclusively on the lightweight connectors used
for local development: PostgreSQL (via ``LocalPgConnector``), the
SQLite-backed BigQuery shim, and the SQLite-backed Firestore shim.  Each
helper returns a fully initialised connector using the configuration
values provided by :mod:`utilities`.
"""

from __future__ import annotations

from sqlalchemy.exc import OperationalError

from .core import DBConnector  # re-export for convenience
from .LocalPgConnector import LocalPgConnector
from .LocalBQConnector import LocalBQConnector
from .LocalFirestoreConnector import LocalFirestoreConnector

from utilities import (
    LOCAL_PG_CONN,
    PG_QUERY_CONN,
    PG_VECTOR_CONN,
    PG_AUDIT_CONN,
    LOCAL_SQLITE_DB,
)


def pg_specific_data_types() -> str:
    """Return a description of the most common PostgreSQL data types."""

    return """
    PostgreSQL offers a wide variety of datatypes to store different types of data effectively. Here's a breakdown of the available categories:

    Numeric datatypes -
    SMALLINT: Stores small-range integers between -32768 and 32767.
    INTEGER: Stores typical integers between -2147483648 and 2147483647.
    BIGINT: Stores large-range integers between -9223372036854775808 and 9223372036854775807.
    DECIMAL(p,s): Stores arbitrary precision numbers with a maximum of p digits and s digits to the right of the decimal point.
    NUMERIC: Similar to DECIMAL but with additional features like automatic scaling.
    REAL: Stores single-precision floating-point numbers with an approximate range of -3.4E+38 to 3.4E+38.
    DOUBLE PRECISION: Stores double-precision floating-point numbers with an approximate range of -1.7E+308 to 1.7E+308.


    Character datatypes -
    CHAR(n): Fixed-length character string with a specified length of n characters.
    VARCHAR(n): Variable-length character string with a maximum length of n characters.
    TEXT: Variable-length string with no maximum size limit.
    CHARACTER VARYING(n): Alias for VARCHAR(n).
    CHARACTER: Alias for CHAR.

    Monetary datatypes -
    MONEY: Stores monetary amounts with two decimal places.

    Date/Time datatypes -
    DATE: Stores dates without time information.
    TIME: Stores time of day without date information (optionally with time zone).
    TIMESTAMP: Stores both date and time information (optionally with time zone).
    INTERVAL: Stores time intervals between two points in time.

    Binary types -
    BYTEA: Stores variable-length binary data.
    BIT: Stores single bits.
    BIT VARYING: Stores variable-length bit strings.

    Other types -
    BOOLEAN: Stores true or false values.
    UUID: Stores universally unique identifiers.
    XML: Stores XML data.
    JSON: Stores JSON data.
    ENUM: Stores user-defined enumerated values.
    RANGE: Stores ranges of data values.

    This list covers the most common datatypes in PostgreSQL.
    """


def bq_specific_data_types() -> str:
    """Return a description of the most common (BigQuery-like) data types."""

    return """
    BigQuery offers a wide variety of datatypes to store different types of data effectively. Here's a breakdown of the available categories:
    Numeric Types -
    INTEGER (INT64): Stores whole numbers within the range of -9,223,372,036,854,775,808 to 9,223,372,036,854,775,807. Ideal for non-fractional values.
    FLOAT (FLOAT64): Stores approximate floating-point numbers with a range of -1.7E+308 to 1.7E+308. Suitable for decimals with a degree of imprecision.
    NUMERIC: Stores exact fixed-precision decimal numbers, with up to 38 digits of precision and 9 digits to the right of the decimal point. Useful for precise financial and accounting calculations.
    BIGNUMERIC: Similar to NUMERIC but with even larger scale and precision. Designed for extreme precision in calculations.

    Character Types -
    STRING: Stores variable-length Unicode character sequences. Enclosed using single, double, or triple quotes.

    Boolean Type -
    BOOLEAN: Stores logical values of TRUE or FALSE (case-insensitive).

    Date and Time Types -
    DATE: Stores dates without associated time information.
    TIME: Stores time information independent of a specific date.
    DATETIME: Stores both date and time information (without timezone information).
    TIMESTAMP: Stores an exact moment in time with microsecond precision, including a timezone component for global accuracy.

    Other Types
    BYTES: Stores variable-length binary data. Distinguished from strings by using 'B' or 'b' prefix in values.
    GEOGRAPHY: Stores points, lines, and polygons representing locations on the Earth's surface.
    ARRAY: Stores an ordered collection of zero or more elements of the same (non-ARRAY) data type.
    STRUCT: Stores an ordered collection of fields, each with its own name and data type (can be nested).

    This list covers the most common datatypes in BigQuery.
    """


def _build_local_pg_connector(
    conn_str: str | None,
    *,
    ensure_audit_table: bool = False,
) -> DBConnector:
    """Create a :class:`LocalPgConnector` using *conn_str* or the default."""

    connection = conn_str or LOCAL_PG_CONN
    try:
        return LocalPgConnector(connection, ensure_audit_table=ensure_audit_table)
    except OperationalError:
        if ensure_audit_table:
            return LocalPgConnector(connection, ensure_audit_table=False)
        raise


def get_data_pg_connector() -> DBConnector:
    """Return the connector used for primary data access."""

    return _build_local_pg_connector(PG_QUERY_CONN)


def get_vector_pg_connector() -> DBConnector:
    """Return the connector used for pgvector queries."""

    return _build_local_pg_connector(PG_VECTOR_CONN)


def get_audit_pg_connector() -> DBConnector:
    """Return the connector used for audit logging."""

    return _build_local_pg_connector(PG_AUDIT_CONN, ensure_audit_table=True)


def get_bq_connector() -> DBConnector:
    """Return the local SQLite-backed BigQuery connector."""

    return LocalBQConnector(LOCAL_SQLITE_DB)


def get_firestore_connector() -> DBConnector:
    """Return the local SQLite-backed Firestore connector."""

    return LocalFirestoreConnector(LOCAL_SQLITE_DB)


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
