import pandas as pd
import pytest

import types
import sys
import pathlib
import asyncio

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from services import chat
from utilities import CHAT_MODEL


def test_generate_sql_results_returns_dataframe(monkeypatch):
    async def fake_run_pipeline(session_id, user_question, selected_schema, **kwargs):
        return "SQL", pd.DataFrame({"a": [1]}), "resp"

    fake_module = types.SimpleNamespace(run_pipeline=fake_run_pipeline, generate_uuid=lambda: "sess")
    monkeypatch.setitem(sys.modules, "opendataqna", fake_module)

    sql, df, resp = asyncio.run(chat.generate_sql_results("sess", "schema", "question"))
    assert sql == "SQL"
    assert isinstance(df, pd.DataFrame)
    assert resp == "resp"


def test_generate_sql_results_handles_non_dataframe(monkeypatch):
    async def fake_run_pipeline(*args, **kwargs):
        return "SQL", "notdf", "resp"

    fake_module = types.SimpleNamespace(run_pipeline=fake_run_pipeline, generate_uuid=lambda: "sess")
    monkeypatch.setitem(sys.modules, "opendataqna", fake_module)

    sql, df, resp = asyncio.run(chat.generate_sql_results("sess", "schema", "question"))
    assert df.empty
    assert list(df.columns) == []


def test_default_pipeline_uses_local_chat_model():
    assert chat._DEFAULT_PIPELINE_ARGS["SQLBuilder_model"] == CHAT_MODEL
    assert chat._DEFAULT_PIPELINE_ARGS["SQLChecker_model"] == CHAT_MODEL
    assert chat._DEFAULT_PIPELINE_ARGS["SQLDebugger_model"] == CHAT_MODEL
