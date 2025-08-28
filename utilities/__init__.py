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

config.read(os.path.join(root_dir, "config.ini"))
if not config.sections():
    raise FileNotFoundError("config.ini not found in current or parent directories.")


def format_prompt(context_prompt, **kwargs):
    """Formats a context prompt by replacing placeholders with values."""
    return context_prompt.format(**kwargs)

# [CONFIG]
EMBEDDING_MODEL = config['CONFIG']['EMBEDDING_MODEL']
# Optional path for locally hosted embedding models
EMBEDDING_MODEL_PATH = config['CONFIG'].get('EMBEDDING_MODEL_PATH', '')
DESCRIPTION_MODEL = config['CONFIG']['DESCRIPTION_MODEL']
# DATA_SOURCE = config['CONFIG']['DATA_SOURCE'] 
VECTOR_STORE = config['CONFIG']['VECTOR_STORE']

#CACHING = config.getboolean('CONFIG','CACHING')
#DEBUGGING = config.getboolean('CONFIG','DEBUGGING')
LOGGING = config.getboolean('CONFIG','LOGGING')
EXAMPLES = config.getboolean('CONFIG', 'KGQ_EXAMPLES')
USE_SESSION_HISTORY = config.getboolean('CONFIG', 'USE_SESSION_HISTORY')
USE_COLUMN_SAMPLES = config.getboolean('CONFIG','USE_COLUMN_SAMPLES')

#[GCP]
PROJECT_ID =  config['GCP']['PROJECT_ID']

#[PGCLOUDSQL]
PG_REGION = config['PGCLOUDSQL']['PG_REGION']
# PG_SCHEMA = config['PGCLOUDSQL']['PG_SCHEMA'] 
PG_INSTANCE = config['PGCLOUDSQL']['PG_INSTANCE']
PG_DATABASE = config['PGCLOUDSQL']['PG_DATABASE'] 
PG_USER = config['PGCLOUDSQL']['PG_USER'] 
PG_PASSWORD = config['PGCLOUDSQL']['PG_PASSWORD']

#[BIGQUERY]
BQ_REGION = config['BIGQUERY']['BQ_DATASET_REGION']
# BQ_DATASET_NAME = config['BIGQUERY']['BQ_DATASET_NAME']
BQ_OPENDATAQNA_DATASET_NAME = config['BIGQUERY']['BQ_OPENDATAQNA_DATASET_NAME']
BQ_LOG_TABLE_NAME = config['BIGQUERY']['BQ_LOG_TABLE_NAME']
# BQ_TABLE_LIST = config['BIGQUERY']['BQ_TABLE_LIST']

#[FIRESTORE]
FIRESTORE_REGION = config['CONFIG']['FIRESTORE_REGION']

#[PROMPTS]
PROMPTS = load_yaml(root_dir + '/prompts.yaml')

__all__ = [
    "EMBEDDING_MODEL",
    "EMBEDDING_MODEL_PATH",
    "DESCRIPTION_MODEL",
    # "DATA_SOURCE",
    "VECTOR_STORE",
    # "CACHING",
    # "DEBUGGING",
    "LOGGING",
    "EXAMPLES",
    "PROJECT_ID",
    "PG_REGION",
    # "PG_SCHEMA",
    "PG_INSTANCE",
    "PG_DATABASE",
    "PG_USER",
    "PG_PASSWORD",
    "BQ_REGION",
    # "BQ_DATASET_NAME",
    "BQ_OPENDATAQNA_DATASET_NAME",
    "BQ_LOG_TABLE_NAME",
    # "BQ_TABLE_LIST",
    "FIRESTORE_REGION",
    "PROMPTS",
    "root_dir",
    "save_config",
]
