# Backend API service

This directory contains the Flask application that powers the Open Data QnA Local
backend. The service can be run entirely on a workstation or server that you
control – no Google Cloud resources are required.

## Prerequisites

1. Follow the [repository quick start](../README.md#quick-start) to create a
   Python environment, install dependencies, and configure `config.ini` with
   your database connection details.
2. (Optional) Set environment variables before launching the server:
   - `PORT` – override the default port (`8080`).
   - `CHAT_HISTORY_LIMIT` – limit the number of cached chat turns (default `20`).
   - `LOCAL_AUTH_TOKEN` – if set in `config.ini`, clients must send an
     `Authorization: Bearer <token>` header to access protected routes.

### Configuration

The service reads [`config.ini`](../config.ini) at startup. Update the `LOCAL`
section with connection strings for your PostgreSQL instance (or SQLite file) and
any agent/model overrides. The same configuration file is used across the
project, so edits remain consistent with the quick-start guide.

## Running the Flask app locally

Launch the API server from the project root:

```bash
python backend-apis/main.py
```

By default the app listens on `http://0.0.0.0:8080`. Use `PORT=<value>` to run on
another port. Logs will indicate whether an authorization token is required.

## API reference

All endpoints accept and return JSON unless noted otherwise. URLs below assume a
local server on `http://localhost:8080`.

> **Auth note:** Routes decorated with `@jwt_authenticated` require the
> `Authorization` header only when `LOCAL_AUTH_TOKEN` is configured.

### `POST /api/chat`

Queue an asynchronous chat job handled by the background worker.

**Request**
```json
{
  "questionType": 1,
  "question": "Show total encounters by month",
  "chatId": "demo-chat-1",
  "sessionId": "optional-existing-session"
}
```

**Response** – HTTP 202
```json
{
  "chatId": "demo-chat-1",
  "chat_status": "PENDING"
}
```

### `GET /available_databases`

Return the databases configured for SQL generation.

**Response**
```json
{
  "ResponseCode": 200,
  "KnownDB": [
    {"table_schema": "imdb"},
    {"table_schema": "retail"}
  ],
  "Error": ""
}
```

### `POST /embed_sql`

Persist a manually reviewed SQL query into the Known Good Query store.

**Request**
```json
{
  "session_id": "123e4567",
  "user_grouping": "retail",
  "user_question": "Which city had the most sales?",
  "generated_sql": "SELECT city_id, COUNT(*) FROM retail.sales GROUP BY 1"
}
```

**Response** – HTTP 201
```json
{
  "ResponseCode": 201,
  "Message": "Example SQL has been accepted for embedding",
  "SessionID": "123e4567",
  "Error": ""
}
```

### `POST /run_query`

Execute provided SQL against the configured database.

**Request**
```json
{
  "session_id": "123e4567",
  "user_grouping": "retail",
  "user_question": "Which city had the most sales?",
  "generated_sql": "SELECT city_id, COUNT(*) AS total_sales FROM retail.sales GROUP BY 1"
}
```

**Response**
```json
{
  "ResponseCode": 200,
  "KnownDB": "[{\"city_id\": \"C014\", \"total_sales\": 152}]",
  "NaturalResponse": "City C014 recorded the highest number of sales.",
  "SessionID": "123e4567",
  "Error": ""
}
```

### `POST /chat`

Run the full chat flow synchronously: generate SQL, execute it, and return a
natural-language answer.

**Request**
```json
{
  "session_id": "",
  "user_grouping": "retail",
  "user_question": "Which city had the most sales?"
}
```

**Response**
```json
{
  "session_id": "123e4567",
  "sql": "SELECT city_id, COUNT(*) AS total_sales FROM retail.sales GROUP BY 1",
  "results": "[{\"city_id\": \"C014\", \"total_sales\": 152}]",
  "response": "City C014 recorded the highest number of sales.",
  "session_reset": true
}
```

### `POST /omop/concept_chat`

Retrieve OMOP CDM vocabulary information for a code or concept term.

**Request**
```json
{
  "question": "What concept represents ICD10 code E11?",
  "top_k": 5
}
```

**Response**
```json
{
  "answer": "...model generated OMOP concept summary..."
}
```

### `POST /get_known_sql`

Fetch Known Good Queries for the supplied user grouping.

**Request**
```json
{
  "user_grouping": "retail"
}
```

**Response**
```json
{
  "ResponseCode": 200,
  "KnownSQL": "[{\"example_user_question\": \"Which city had the most sales?\", ...}]",
  "Error": ""
}
```

### `POST /generate_sql`

Generate SQL for a user question without executing it.

**Request**
```json
{
  "session_id": "",
  "user_grouping": "retail",
  "user_question": "Which city had the most sales?",
  "user_id": "analyst@example.com"
}
```

**Response**
```json
{
  "ResponseCode": 200,
  "GeneratedSQL": "SELECT city_id, COUNT(*) AS total_sales FROM retail.sales GROUP BY 1",
  "SessionID": "123e4567",
  "Error": ""
}
```

### `POST /generate_viz`

Produce Google Charts JavaScript based on SQL results.

**Request**
```json
{
  "session_id": "123e4567",
  "user_question": "Top 5 product SKUs by orders",
  "generated_sql": "SELECT product_sku, SUM(total_ordered) FROM retail.sales GROUP BY 1",
  "sql_results": [
    {"product_sku": "SKU-001", "total_ordered": 456}
  ]
}
```

**Response**
```json
{
  "ResponseCode": 200,
  "GeneratedChartjs": {
    "chart_div": "google.charts.load('current', {packages: ['corechart']});..."
  },
  "SessionID": "123e4567",
  "Error": ""
}
```

### `POST /summarize_results`

Summarize a result set for a given question.

**Request**
```json
{
  "user_question": "Which city had the most sales?",
  "sql_results": "[{\"city_id\": \"C014\", \"total_sales\": 152}]"
}
```

**Response**
```json
{
  "ResponseCode": 200,
  "summary_response": "City C014 recorded the highest number of sales.",
  "Error": ""
}
```

### `POST /natural_response`

Combine SQL generation, execution, and summarization to return an answer only.

**Request**
```json
{
  "user_grouping": "retail",
  "user_question": "Which city had the most sales?"
}
```

**Response**
```json
{
  "ResponseCode": 200,
  "summary_response": "City C014 recorded the highest number of sales.",
  "Error": ""
}
```

### `POST /get_results`

Generate and execute SQL, returning the raw result rows.

**Request**
```json
{
  "user_database": "retail",
  "user_question": "Which city had the most sales?"
}
```

**Response**
```json
{
  "ResponseCode": 200,
  "GeneratedResults": "[{\"city_id\": \"C014\", \"total_sales\": 152}]",
  "Error": ""
}
```

### `POST /papers/embed`

Store paper documents and their embeddings in PostgreSQL.

**Request**
```json
[
  {
    "title": "Example title",
    "abstract": "Optional abstract",
    "content": "Full document text",
    "metadata": {"source": "demo"}
  }
]
```

**Response** – HTTP 201
```json
{
  "inserted": 1
}
```

### `POST /papers/search`

Perform a vector similarity search over embedded papers.

**Request**
```json
{
  "query": "climate change",
  "k": 5,
  "summarize": true
}
```

**Response**
```json
{
  "results": [
    {"id": 1, "title": "Example title", "abstract": "Optional abstract", "metadata": {"source": "demo"}}
  ],
  "summary": "Optional natural language summary"
}
```

## Next steps

To integrate these endpoints with the demo UI or other clients, continue using
the configuration established in the repository quick start. The backend README
and the root guide now describe the same local workflow.
