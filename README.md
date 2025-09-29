# Open Data QnA Local

Open Data QnA Local provides the backend-only variant of the Open Data QnA project. It exposes
Flask APIs that allow you to chat with a PostgreSQL database using locally hosted language
models. All Google Cloud, Terraform, and frontend assets were removed so that the repository
contains only the components required to run the backend on a workstation or server you control.

![Chat teaser](utilities/imgs/Teaser.gif)

## Repository layout

The repository now focuses exclusively on the services that power the local backend:

| Directory | Description |
| --- | --- |
| `backend-apis/` | Flask application exposing chat and utility endpoints. |
| `agents/` | Agent implementations used for SQL generation, validation, and responses. |
| `embeddings/` | Helpers for generating and storing embeddings, including Known Good Query tooling. |
| `dbconnectors/` | Database connector abstractions for PostgreSQL/SQLite and audit logging. |
| `services/` | Business logic shared by the Flask routes. |
| `utilities/` | Configuration loading and shared helpers. |
| `data/` | Placeholder directory for local assets such as the Known Good SQL cache. |
| `tests/` | Automated tests for the backend components. |

Cloud-specific directories such as `frontend/`, `terraform/`, `docs/`, `notebooks/`, and `scripts/`
were deleted because they are not needed to run the local backend. If you need those workflows,
refer to the upstream GoogleCloudPlatform/Open_Data_QnA project.

## Quick start

1. **Create a Python environment** (Python 3.10 or later):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. **Install dependencies** using Poetry (preferred) or pip:
   ```bash
   pip install poetry
   poetry install
   # or: pip install -e .
   ```
3. **Configure the backend** by editing [`config.ini`](config.ini). At minimum provide:
   - `LOCAL.PG_CONN` or `LOCAL.PG_CONN_STRING` – SQLAlchemy style connection string for your database.
   - Optional agent/model overrides under the `CONFIG` section.
   - Optional `LOCAL.KNOWN_GOOD_SQL_PATH` if you store the Known Good SQL cache outside of `data/`.
4. **Run the API server**:
   ```bash
   python backend-apis/main.py
   ```
   The Flask app listens on `http://0.0.0.0:8080` by default.
5. **Execute tests** (optional):
   ```bash
   pytest
   ```

## Known Good SQL cache

The backend can warm-start SQL generation using a Known Good Query (KGQ) cache. Place a CSV file
with the columns `prompt`, `sql`, and `user_grouping` at `data/known_good_sql.csv` or set the
absolute path in `LOCAL.KNOWN_GOOD_SQL_PATH` in `config.ini`. The helper utilities will look for the
file in that location when populating the cache or running the KGQ embedding tools.

If you do not have KGQ data yet you can leave the file absent. The backend will log a clear error
message if KGQ-specific routines are invoked without the CSV present.

## Maintenance scope

This repository is now scoped to the local backend. Cloud deployment tooling, infrastructure-as-code
assets, notebooks, and frontend implementations are intentionally out of scope. The README contains
all the information required to run and maintain the remaining components; no additional docs need
to be kept in sync.

