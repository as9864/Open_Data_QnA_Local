from abc import ABC
import re
from typing import Optional

import requests

from utilities import (
    PROMPTS,
    format_prompt,
    CHAT_MODEL,
    CHAT_MODEL_URL,
    RESPONSE_DRAFT_MODEL,
    RESPONSE_DRAFT_MODEL_URL,
    RESPONSE_EDITOR_MODEL,
    RESPONSE_EDITOR_MODEL_URL,
    RESPONSE_EDITOR_TEMPERATURE,
    RESPONSE_EDITOR_MAX_TOKENS,
)

from .core import Agent

class ResponseAgent:
    """
    로컬 LLM 기반으로 SQL 결과를 자연어 응답으로 변환하는 ResponseAgentLocal 버전.
    BuildSQLAgent_Local.py를 참고해 /api/generate 엔드포인트를 호출합니다.
    """

    agentType: str = "ResponseAgentLocal"

    def __init__(
        self,
        model: Optional[str] = None,
        host: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        top_p: float = 1.0,
        timeout_sec: int = 300,
        editor_model: Optional[str] = None,
        editor_host: Optional[str] = None,
        editor_max_tokens: Optional[int] = None,
        editor_temperature: Optional[float] = None,
    ):
        super().__init__()
        self.draft_model = (model or RESPONSE_DRAFT_MODEL or CHAT_MODEL).strip()
        draft_host = host or RESPONSE_DRAFT_MODEL_URL or CHAT_MODEL_URL
        self.draft_host = draft_host.rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.timeout_sec = timeout_sec

        configured_editor_model = (
            editor_model if editor_model is not None else RESPONSE_EDITOR_MODEL
        )
        self.editor_model = configured_editor_model.strip() or None
        resolved_editor_host = editor_host or RESPONSE_EDITOR_MODEL_URL
        self.editor_host = (resolved_editor_host or self.draft_host).rstrip("/")
        self.editor_max_tokens = editor_max_tokens or RESPONSE_EDITOR_MAX_TOKENS
        self.editor_temperature = (
            editor_temperature
            if editor_temperature is not None
            else RESPONSE_EDITOR_TEMPERATURE
        )

    def _postprocess_text(self, text: str) -> str:
        text = re.sub(r"<think\b[^>]*>.*?</think>", "", text, flags=re.S)
        text = re.sub(r"<think\b[^>]*>.*", "", text, flags=re.S)
        text = re.sub(r"<analysis\b[^>]*>.*?</analysis>", "", text, flags=re.S)
        text = re.sub(r"<analysis\b[^>]*>.*", "", text, flags=re.S)
        text = re.sub(r"<Thought\b[^>]*>.*?</Thought>", "", text, flags=re.S)
        text = re.sub(r"<Thought\b[^>]*>.*", "", text, flags=re.S)
        return text.strip()

    def _generate_with_model(
        self,
        *,
        prompt: str,
        model: str,
        host: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> str:
        safe_prompt = (
            "중요: 사고 과정, 중간 추론, <think> 태그를 절대 출력하지 마세요. "
            "최종 답변만 한국어로 불릿 포인트로 출력하세요.\n\n" + prompt
        )

        payload = {
            "model": model,
            "prompt": safe_prompt,
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_tokens,
            },
            "stop": [
                "<think>",
                "</think>",
                "<analysis>",
                "</analysis>",
                "<Thought>",
                "</Thought>",
            ],
            "stream": False,
        }
        url_generate = f"{host}/api/generate"
        print(model, host, url_generate)
        resp = requests.post(url_generate, json=payload, timeout=self.timeout_sec)
        resp.raise_for_status()
        data = resp.json() if resp.content else {}
        text = (data.get("response") or "").strip()
        return self._postprocess_text(text)

    def _build_editor_prompt(
        self, user_question: str, sql_result, draft_response: str
    ) -> str:
        template = PROMPTS.get("nl_response_editor")
        if template:
            return format_prompt(
                template,
                user_question=user_question,
                sql_result=sql_result,
                draft_response=draft_response,
            )
        return (
            "당신은 데이터 분석 비서가 작성한 한국어 초안을 검토하는 편집 도우미입니다.\n"
            "다음 정보를 바탕으로 사실성을 확인하고, 간결하고 자연스러운 한국어 불릿 포인트로 다듬어 주세요.\n"
            f"사용자 질문: {user_question}\n"
            f"SQL 결과: {sql_result}\n"
            f"초안 응답: {draft_response}\n"
            "외부 지식은 추가하지 말고, 불필요한 반복을 제거한 후 최종 답변만 출력하세요."
        )

    def run(self, user_question, sql_result):
        context_prompt = PROMPTS["nl_reponse"]
        context_prompt = format_prompt(
            context_prompt,
            user_question=user_question,
            sql_result=sql_result,
        )

        print(f"Prompt for Natural Language Response (Local): \n{context_prompt}")
        draft_response = self._generate_with_model(
            prompt=context_prompt,
            model=self.draft_model,
            host=self.draft_host,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )

        if not self.editor_model:
            return draft_response

        editor_prompt = self._build_editor_prompt(
            user_question=user_question,
            sql_result=sql_result,
            draft_response=draft_response,
        )

        print(f"Prompt for Natural Language Editor (Local): \n{editor_prompt}")
        try:
            refined_response = self._generate_with_model(
                prompt=editor_prompt,
                model=self.editor_model,
                host=self.editor_host,
                temperature=self.editor_temperature,
                top_p=self.top_p,
                max_tokens=self.editor_max_tokens,
            )
        except Exception as exc:
            print(
                "Editor model invocation failed, returning draft response instead. "
                f"Reason: {exc}"
            )
            return draft_response

        return refined_response or draft_response
