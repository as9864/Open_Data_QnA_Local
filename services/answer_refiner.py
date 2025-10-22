"""Lightweight helper for refining draft responses with an editing model."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

import requests

from utilities import (
    CHAT_MODEL_URL,
    RESPONSE_DRAFT_MODEL,
    RESPONSE_DRAFT_MODEL_URL,
    RESPONSE_EDITOR_MAX_TOKENS,
    RESPONSE_EDITOR_MODEL,
    RESPONSE_EDITOR_MODEL_URL,
    RESPONSE_EDITOR_TEMPERATURE,
)


log = logging.getLogger(__name__)


def _strip_or_default(value: Optional[str], default: str = "") -> str:
    if not value:
        return default
    return str(value).strip()


@dataclass
class AnswerRefinerConfig:
    model: str
    host: str
    temperature: float
    max_tokens: int
    timeout_sec: int = 60
    max_evidence_chars: int = 4000


class AnswerRefiner:
    """Generate an edited response from a draft answer and supporting evidence."""

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        host: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout_sec: int = 60,
        max_evidence_chars: int = 4000,
    ) -> None:
        resolved_model = _strip_or_default(
            model,
            _strip_or_default(
                RESPONSE_EDITOR_MODEL,
                _strip_or_default(RESPONSE_DRAFT_MODEL),
            ),
        )
        resolved_host = _strip_or_default(
            host,
            _strip_or_default(
                RESPONSE_EDITOR_MODEL_URL,
                _strip_or_default(RESPONSE_DRAFT_MODEL_URL, CHAT_MODEL_URL),
            ),
        )
        resolved_temperature = (
            temperature if temperature is not None else RESPONSE_EDITOR_TEMPERATURE
        )
        resolved_max_tokens = max_tokens or RESPONSE_EDITOR_MAX_TOKENS

        self.config = AnswerRefinerConfig(
            model=resolved_model,
            host=resolved_host.rstrip("/"),
            temperature=resolved_temperature,
            max_tokens=resolved_max_tokens,
            timeout_sec=timeout_sec,
            max_evidence_chars=max_evidence_chars,
        )

    @property
    def enabled(self) -> bool:
        return bool(self.config.model and self.config.host)

    def refine(
        self,
        *,
        question: str,
        draft_response: str,
        evidence: Any = None,
        instructions: Optional[str] = None,
    ) -> str:
        if not isinstance(draft_response, str) or not draft_response.strip():
            return draft_response

        if not self.enabled:
            return draft_response

        prompt = self._build_prompt(
            question=question,
            draft_response=draft_response,
            evidence=evidence,
            instructions=instructions,
        )

        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
            "stop": ["<think>", "</think>", "<analysis>", "</analysis>", "<Thought>", "</Thought>"],
            "stream": False,
        }

        url = f"{self.config.host}/api/generate"
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.config.timeout_sec,
            )
            response.raise_for_status()
            data = response.json() if response.content else {}
            refined = (data.get("response") or "").strip()
            if not refined:
                return draft_response
            return refined
        except Exception:
            log.exception("AnswerRefiner failed, returning draft response")
            return draft_response

    def _build_prompt(
        self,
        *,
        question: str,
        draft_response: str,
        evidence: Any,
        instructions: Optional[str],
    ) -> str:
        base_instruction = instructions or (
            "당신은 한국어 편집 도우미입니다. 제공된 근거를 벗어나지 않고, 초안 응답의 사실성을 확인하며, "
            "간결하고 자연스러운 한국어 문장으로 다듬어 주세요."
        )
        evidence_text = self._format_evidence(evidence)

        sections = [
            base_instruction.strip(),
            "요구사항:",
            "- 초안의 정보 중 근거와 일치하지 않는 내용은 삭제하거나 수정하세요.",
            "- 새로운 사실이나 추측은 추가하지 마세요.",
            "- 불필요한 반복을 줄이고 간결한 문장으로 정리하세요.",
            "- 한국어 맞춤법과 가독성을 개선하세요.",
            "- 최종 답변만 출력하세요.",
            "",
            "[사용자 질문]",
            question.strip() or "(질문 없음)",
            "",
            "[초안 응답]",
            draft_response.strip(),
            "",
            "[근거]",
            evidence_text,
        ]
        return "\n".join(sections)

    def _format_evidence(self, evidence: Any) -> str:
        if evidence is None:
            return "(제공된 근거 없음)"

        text: str
        if isinstance(evidence, str):
            text = evidence.strip()
        elif isinstance(evidence, (list, dict)):
            try:
                text = json.dumps(evidence, ensure_ascii=False)
            except (TypeError, ValueError):
                text = str(evidence)
        else:
            # Lazy import to avoid pandas dependency at import time
            try:
                import pandas as pd  # type: ignore

                if isinstance(evidence, pd.DataFrame):
                    preview = evidence.replace({pd.NA: None}).head(5)
                    text = preview.to_json(orient="records", force_ascii=False)
                else:
                    text = str(evidence)
            except Exception:
                text = str(evidence)

        if not text:
            text = "(근거 내용을 비울 수 없습니다)"

        if len(text) > self.config.max_evidence_chars:
            return text[: self.config.max_evidence_chars]
        return text


# Module level singleton used by the API server
ANSWER_REFINER = AnswerRefiner()

__all__ = ["AnswerRefiner", "ANSWER_REFINER"]
