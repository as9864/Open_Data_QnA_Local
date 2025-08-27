"""Utility helpers and configuration loading."""

from __future__ import annotations

import configparser
import os
import yaml


config = configparser.ConfigParser()


def is_root_dir() -> bool:
    """Return ``True`` if the current working directory is the project root."""

    current_dir = os.getcwd()
    notebooks_path = os.path.join(current_dir, "notebooks")
    agents_path = os.path.join(current_dir, "agents")
    return os.path.exists(notebooks_path) or os.path.exists(agents_path)


def load_yaml(file_path: str) -> dict:
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


if is_root_dir():
    root_dir = os.getcwd()
else:
    root_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))

config.read(os.path.join(root_dir, "config.ini"))

if not config.sections():
    raise FileNotFoundError("config.ini not found in current or parent directories.")


def format_prompt(context_prompt, **kwargs):
    """Formats a context prompt by replacing placeholders with values."""

    return context_prompt.format(**kwargs)


# [CONFIG]
EMBEDDING_MODEL = config["CONFIG"]["EMBEDDING_MODEL"]
DESCRIPTION_MODEL = config["CONFIG"]["DESCRIPTION_MODEL"]
VECTOR_STORE = config["CONFIG"]["VECTOR_STORE"]
LOGGING = config.getboolean("CONFIG", "LOGGING")
EXAMPLES = config.getboolean("CONFIG", "KGQ_EXAMPLES")
USE_SESSION_HISTORY = config.getboolean("CONFIG", "USE_SESSION_HISTORY")
USE_COLUMN_SAMPLES = config.getboolean("CONFIG", "USE_COLUMN_SAMPLES")

CONNECTOR_BACKEND = config["CONFIG"].get("CONNECTOR_BACKEND", "cloud")
LOCAL_PG_CONN = config.get("LOCAL", "PG_CONN", fallback="")
LOCAL_SQLITE_DB = config.get("LOCAL", "SQLITE_DB", fallback="opendataqna.db")

# [GCP]
PROJECT_ID = config["GCP"]["PROJECT_ID"]

# [PGCLOUDSQL]
PG_REGION = config["PGCLOUDSQL"]["PG_REGION"]
PG_INSTANCE = config["PGCLOUDSQL"]["PG_INSTANCE"]
PG_DATABASE = config["PGCLOUDSQL"]["PG_DATABASE"]
PG_USER = config["PGCLOUDSQL"]["PG_USER"]
PG_PASSWORD = config["PGCLOUDSQL"]["PG_PASSWORD"]

# [BIGQUERY]
BQ_REGION = config["BIGQUERY"]["BQ_DATASET_REGION"]
BQ_OPENDATAQNA_DATASET_NAME = config["BIGQUERY"]["BQ_OPENDATAQNA_DATASET_NAME"]
BQ_LOG_TABLE_NAME = config["BIGQUERY"]["BQ_LOG_TABLE_NAME"]

# [FIRESTORE]
FIRESTORE_REGION = config["CONFIG"]["FIRESTORE_REGION"]

# [PROMPTS]
PROMPTS = load_yaml(os.path.join(root_dir, "prompts.yaml"))


__all__ = [
    "EMBEDDING_MODEL",
    "DESCRIPTION_MODEL",
    "VECTOR_STORE",
    "LOGGING",
    "EXAMPLES",
    "PROJECT_ID",
    "PG_REGION",
    "PG_INSTANCE",
    "PG_DATABASE",
    "PG_USER",
    "PG_PASSWORD",
    "BQ_REGION",
    "BQ_OPENDATAQNA_DATASET_NAME",
    "BQ_LOG_TABLE_NAME",
    "FIRESTORE_REGION",
    "PROMPTS",
    "root_dir",
    "CONNECTOR_BACKEND",
    "LOCAL_PG_CONN",
    "LOCAL_SQLITE_DB",
    "USE_SESSION_HISTORY",
    "USE_COLUMN_SAMPLES",
]

