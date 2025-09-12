import asyncio
import pandas as pd

# Default arguments for run_pipeline. These mirror the previous hard-coded
# values used in the Streamlit app and API server.
_DEFAULT_PIPELINE_ARGS = dict(
    RUN_DEBUGGER=True,
    EXECUTE_FINAL_SQL=True,
    DEBUGGING_ROUNDS=2,
    LLM_VALIDATION=False,
    Embedder_model="local",
    SQLBuilder_model="gemini-1.5-pro",
    SQLChecker_model="gemini-1.5-pro",
    SQLDebugger_model="gemini-1.5-pro",
    num_table_matches=5,
    num_column_matches=10,
    table_similarity_threshold=0.1,
    column_similarity_threshold=0.1,
    example_similarity_threshold=0.1,
    num_sql_matches=3,
)

async def generate_sql_results(session_id: str | None, selected_schema: str, user_question: str, **overrides):
    """Run the pipeline and return the generated SQL, results and response.

    Args:
        session_id: Identifier for the conversation session. If ``None`` a new
            UUID will be generated.
        selected_schema: Database schema to query.
        user_question: Natural language question from the user.
        **overrides: Optional keyword arguments to override defaults passed to
            ``run_pipeline``.

    Returns:
        Tuple ``(final_sql, results_df, response)`` where ``results_df`` is
        always a :class:`~pandas.DataFrame`.
    """
    params = _DEFAULT_PIPELINE_ARGS.copy()
    params.update(overrides)

    # Import here to avoid heavy module imports at import time and to make the
    # function easier to mock in tests.
    from opendataqna import run_pipeline, generate_uuid

    session_id = session_id or generate_uuid()
    final_sql, results_df, response = await run_pipeline(
        session_id,
        user_question,
        selected_schema,
        **params,
    )
    if not isinstance(results_df, pd.DataFrame):
        results_df = pd.DataFrame([])
    return final_sql, results_df, response
