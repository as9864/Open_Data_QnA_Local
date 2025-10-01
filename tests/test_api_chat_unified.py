import copy
import importlib.util
import itertools
import pathlib
import sys
import types

import pytest


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


class _FakeResponder:
    def __init__(self, *args, **kwargs):
        pass

    def run_paper(self, *args, **kwargs):  # pragma: no cover - safety stub
        return ""


_FAKE_AGENT_LOCAL_MODULE = types.ModuleType("agents.Agent_local")
setattr(_FAKE_AGENT_LOCAL_MODULE, "LocalOllamaResponder", _FakeResponder)

sys.modules.setdefault("agents.Agent_local", _FAKE_AGENT_LOCAL_MODULE)


class _FakeSentenceTransformer:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, inputs):  # pragma: no cover - safety stub
        return [0.0] if isinstance(inputs, str) else [[0.0] for _ in inputs]


_FAKE_SENTENCE_MODULE = types.ModuleType("sentence_transformers")
setattr(_FAKE_SENTENCE_MODULE, "SentenceTransformer", _FakeSentenceTransformer)
sys.modules.setdefault("sentence_transformers", _FAKE_SENTENCE_MODULE)


_MODULE_PATH = _REPO_ROOT / "backend-apis" / "main.py"
_SPEC = importlib.util.spec_from_file_location("backend_apis.main", _MODULE_PATH)
main = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(main)


@pytest.fixture(autouse=True)
def _reset_chat_state():
    main._session_store.clear()
    main._chat_sessions.clear()
    main._chat_histories.clear()


@pytest.fixture
def client():
    main.app.config.update(TESTING=True)
    with main.app.test_client() as client:
        yield client


@pytest.fixture
def fake_generate_uuid(monkeypatch):
    counter = itertools.count(1)

    def _fake_generate_uuid() -> str:
        return f"session-{next(counter)}"

    monkeypatch.setattr(main, "generate_uuid", _fake_generate_uuid)
    return _fake_generate_uuid


@pytest.fixture
def capture_enqueued(monkeypatch):
    captured: list[dict] = []

    def _capture(body: dict) -> None:
        captured.append(copy.deepcopy(body))

    monkeypatch.setattr(main, "enqueue_chat_task", _capture)
    return captured


def test_api_chat_unified_returns_new_session_id(client, fake_generate_uuid, capture_enqueued):
    response = client.post(
        "/api/chat",
        json={"questionType": 1, "question": "Hello?", "chatId": "chat-1"},
    )

    assert response.status_code == 202
    payload = response.get_json()
    assert payload == {
        "chatId": "chat-1",
        "sessionId": "session-1",
        "chat_status": "PENDING",
    }

    assert len(capture_enqueued) == 1
    enqueued = capture_enqueued[0]
    assert enqueued["session_id"] == "session-1"
    assert "sessionId" not in enqueued


def test_api_chat_unified_reuses_existing_session_id(client, fake_generate_uuid, capture_enqueued):
    first = client.post(
        "/api/chat",
        json={"questionType": 1, "question": "Hi", "chatId": "chat-42"},
    )

    assert first.status_code == 202
    first_payload = first.get_json()
    assert first_payload["sessionId"] == "session-1"

    second = client.post(
        "/api/chat",
        json={
            "questionType": 1,
            "question": "Follow-up?",
            "chatId": "chat-42",
            "sessionId": first_payload["sessionId"],
        },
    )

    assert second.status_code == 202
    second_payload = second.get_json()
    assert second_payload == {
        "chatId": "chat-42",
        "sessionId": "session-1",
        "chat_status": "PENDING",
    }

    assert len(capture_enqueued) == 2
    assert [payload["session_id"] for payload in capture_enqueued] == ["session-1", "session-1"]
    assert "sessionId" not in capture_enqueued[0]
    assert capture_enqueued[1]["sessionId"] == "session-1"
