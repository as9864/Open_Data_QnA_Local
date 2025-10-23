from __future__ import annotations

import json
import pathlib
import sys

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from services import answer_refiner
from services.answer_refiner import AnswerRefiner


def test_answer_refiner_skips_when_draft_is_empty():
    refiner = AnswerRefiner(model="test-model", host="http://example.com")

    # 공백 초안은 편집 모델을 호출하지 않고 그대로 반환한다.
    assert refiner.refine(question="질문", draft_response="   ") == "   "


def test_answer_refiner_invokes_model_with_expected_payload(monkeypatch):
    refiner = AnswerRefiner(model="test-model", host="http://example.com")

    captured: dict[str, object] = {}

    def _fake_post(url, json=None, timeout=None):  # pragma: no cover - patch
        captured["url"] = url
        captured["payload"] = json

        class _FakeResponse:
            content = b"{}"

            @staticmethod
            def raise_for_status() -> None:
                return None

            @staticmethod
            def json() -> dict[str, str]:
                return {"response": "정제된 답변"}

        return _FakeResponse()

    monkeypatch.setattr(answer_refiner.requests, "post", _fake_post)

    result = refiner.refine(
        question="질문",
        draft_response="초안",
        evidence={"key": "value"},
    )

    assert result == "정제된 답변"
    assert captured["url"] == "http://example.com/api/generate"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == "test-model"
    assert "요구사항" in payload["prompt"]
    assert "질문" in payload["prompt"]
    assert "초안" in payload["prompt"]
    assert json.dumps({"key": "value"}, ensure_ascii=False) in payload["prompt"]


def test_answer_refiner_failure_returns_draft(monkeypatch):
    refiner = AnswerRefiner(model="test-model", host="http://example.com")

    def _raise(*_, **__):  # pragma: no cover - patch
        raise RuntimeError("boom")

    monkeypatch.setattr(answer_refiner.requests, "post", _raise)

    assert refiner.refine(question="질문", draft_response="초안") == "초안"
