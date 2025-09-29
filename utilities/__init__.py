"""Utility helpers and configuration loading."""

from __future__ import annotations

import configparser
import os
import yaml


config = configparser.ConfigParser()


def is_root_dir() -> bool:
    """Return True if the current working directory is the project root."""
    current_dir = os.getcwd()
    notebooks_path = os.path.join(current_dir, "notebooks")
    agents_path = os.path.join(current_dir, "agents")
    return os.path.exists(notebooks_path) or os.path.exists(agents_path)


def load_yaml(file_path: str) -> dict:
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Determine root_dir and load config.ini
if is_root_dir():
    root_dir = os.getcwd()
else:
    root_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))

config_path = os.path.join(root_dir, "config.ini")
config.read(config_path, encoding="utf-8")  # 필요 시 "utf-8-sig"
if not config.sections():
    raise FileNotFoundError("config.ini not found in current or parent directories.")


def format_prompt(context_prompt, **kwargs):
    """Formats a context prompt by replacing placeholders with values."""
    return context_prompt.format(**kwargs)


# =========================
# [CONFIG] (안전한 fallback 포함)
# =========================
# MODE
MODE = config.get("CONFIG", "MODE", fallback="gcp").lower()

# 기본 옵션들
EMBEDDING_MODEL = config.get("CONFIG", "EMBEDDING_MODEL", fallback="")
# Optional path for locally hosted embedding models
EMBEDDING_MODEL_PATH = config.get("CONFIG", "EMBEDDING_MODEL_PATH", fallback="")
DESCRIPTION_MODEL = config.get("CONFIG", "DESCRIPTION_MODEL", fallback="")
VECTOR_STORE = config.get("CONFIG", "VECTOR_STORE", fallback="")

LOGGING = config.getboolean("CONFIG", "LOGGING", fallback=False)
EXAMPLES = config.getboolean("CONFIG", "KGQ_EXAMPLES", fallback=False)
USE_SESSION_HISTORY = config.getboolean("CONFIG", "USE_SESSION_HISTORY", fallback=True)
USE_COLUMN_SAMPLES = config.getboolean("CONFIG", "USE_COLUMN_SAMPLES", fallback=True)

# [GCP]
PROJECT_ID = config.get("GCP", "PROJECT_ID", fallback="")

# [PGCLOUDSQL]
PG_REGION = config.get("PGCLOUDSQL", "PG_REGION", fallback="")
PG_INSTANCE = config.get("PGCLOUDSQL", "PG_INSTANCE", fallback="")
PG_DATABASE = config.get("PGCLOUDSQL", "PG_DATABASE", fallback="")
PG_USER = config.get("PGCLOUDSQL", "PG_USER", fallback="")
PG_PASSWORD = config.get("PGCLOUDSQL", "PG_PASSWORD", fallback="")


# [BIGQUERY]
BQ_REGION = config.get("BIGQUERY", "BQ_DATASET_REGION", fallback="")
BQ_OPENDATAQNA_DATASET_NAME = config.get("BIGQUERY", "BQ_OPENDATAQNA_DATASET_NAME", fallback="")
BQ_LOG_TABLE_NAME = config.get("BIGQUERY", "BQ_LOG_TABLE_NAME", fallback="")

# [FIRESTORE]
FIRESTORE_REGION = config.get("CONFIG", "FIRESTORE_REGION", fallback="")

# PROMPTS
PROMPTS = load_yaml(os.path.join(root_dir, "prompts.yaml"))

# =========================================
# Backward-compat for dbconnectors imports
# =========================================
# If CONFIG.CONNECTOR_BACKEND is missing, derive it from MODE.
CONNECTOR_BACKEND = config.get(
    "CONFIG", "CONNECTOR_BACKEND",
    fallback=("cloud" if MODE == "gcp" else "local"),
)

# Support both PG_CONN and PG_CONN_STRING keys for local
LOCAL_PG_CONN = (
    config.get("LOCAL", "PG_CONN", fallback="")
    or config.get("LOCAL", "PG_CONN_STRING", fallback="")
)
_LOCAL_QUERY_CONN = config.get("LOCAL", "PG_QUERY_CONN", fallback="").strip()
_LOCAL_VECTOR_CONN = config.get("LOCAL", "PG_VECTOR_CONN", fallback="").strip()
_LOCAL_AUDIT_CONN = config.get("LOCAL", "PG_AUDIT_CONN", fallback="").strip()

PG_QUERY_CONN = _LOCAL_QUERY_CONN or LOCAL_PG_CONN
PG_VECTOR_CONN = _LOCAL_VECTOR_CONN or LOCAL_PG_CONN
PG_AUDIT_CONN = _LOCAL_AUDIT_CONN or LOCAL_PG_CONN

PG_CLOUD_QUERY_CONN = config.get("PGCLOUDSQL", "PG_QUERY_CONN", fallback="").strip()
PG_CLOUD_VECTOR_CONN = config.get("PGCLOUDSQL", "PG_VECTOR_CONN", fallback="").strip()
PG_CLOUD_AUDIT_CONN = config.get("PGCLOUDSQL", "PG_AUDIT_CONN", fallback="").strip()
LOCAL_SQLITE_DB = config.get("LOCAL", "SQLITE_DB", fallback="opendataqna.db")
PG_CONN_STRING = config.get("LOCAL","PG_CONN_STRING", fallback="")

PG_LOCAL = config.get("LOCAL", "pglocal", fallback="postgres")
LOCAL_USER_GROUPING =  config.get("LOCAL", "user_grouping", fallback="cdm")

CALL_BACK_URL = config.get("CONFIG","CALL_BACK_URL" , fallback="http://caus.re.kr:3010")

CHAT_MODEL = config.get("CONFIG","CHAT_MODEL" , fallback="timHan/llama3korean8B4QKM:latest")

CHAT_MODEL_URL = config.get("CONFIG","CHAT_MODEL_URL" , fallback="http://222.236.26.27:25123")

LOCAL_AUTH_TOKEN = config.get("LOCAL", "LOCAL_AUTH_TOKEN", fallback="")



__all__ = [
    "MODE",
    "EMBEDDING_MODEL",
    "EMBEDDING_MODEL_PATH",
    "DESCRIPTION_MODEL",
    "VECTOR_STORE",
    "LOGGING",
    "EXAMPLES",
    "USE_SESSION_HISTORY",
    "USE_COLUMN_SAMPLES",
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
    "format_prompt",
    # 호환 심볼
    "CONNECTOR_BACKEND",
    "LOCAL_PG_CONN",
    "PG_QUERY_CONN",
    "PG_VECTOR_CONN",
    "PG_AUDIT_CONN",
    "PG_CLOUD_QUERY_CONN",
    "PG_CLOUD_VECTOR_CONN",
    "PG_CLOUD_AUDIT_CONN",
    "LOCAL_SQLITE_DB",
    "PG_CONN_STRING",
    "CALL_BACK_URL",
    "CHAT_MODEL",
    "CHAT_MODEL_URL",
    "LOCAL_AUTH_TOKEN",
]
