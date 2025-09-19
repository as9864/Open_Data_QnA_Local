import json
import re
import requests
from abc import ABC
from utilities import PROMPTS, format_prompt
from .core import Agent
import json
import pandas as pd
from io import StringIO

class ValidateSQLAgentLocal():
    """
    로컬 LLM(예: Ollama 등 HTTP /api/generate 엔드포인트)로
    SQL 유효성(문법/의미) 검증을 수행하는 에이전트.

    - 입력: user_question, tables_schema, columns_schema, generated_sql
    - 출력: {"valid": bool, "errors": "설명 문자열"}
    """

    agentType: str = "ValidateSQLAgentLocal"

    def __init__(
        self,
        model: str = "hopephoto/Qwen3-4B-Instruct-2507_q8",
        host: str = "http://192.168.0.230:11434",
        max_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float = 1.0,
        timeout_sec: int = 300,
    ):
        # (선택) 내부에서도 동일 키로 쓰고 싶으면 아래처럼 정리
        self.model_id = model
        self.model = model

        self.host = host.rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.timeout_sec = timeout_sec
        self.url_generate = f"{self.host}/api/generate"

    # --- 내부 유틸: 로컬 LLM 호출 ---
    def _call_local_llm(self, prompt: str) -> str:
        """
        로컬 LLM(generation API) 호출해 텍스트 반환
        """
        # 응답을 'JSON만' 내도록 강제 지시
        safe_prompt = (
            "중요: 다음 지시를 따르세요.\n"
            "1) 사고 과정, 중간 추론, <think> 태그 등은 절대 출력하지 마세요.\n"
            "2) 오직 JSON만 출력하세요. 코드펜스(```)나 추가 설명 금지.\n"
            '3) 출력 형식은 {"valid": true/false, "errors": "<string>"} 입니다.\n\n'
            + prompt
        )

        payload = {
            "model": self.model,
            "prompt": safe_prompt,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_predict": self.max_tokens,
            },
            "stop": ["```", "<think>", "</think>", "<analysis>", "</analysis>", "<Thought>", "</Thought>"],
            "stream": False,
        }
        resp = requests.post(self.url_generate, json=payload, timeout=self.timeout_sec)
        resp.raise_for_status()
        data = resp.json() if resp.content else {}
        text = (data.get("response") or "").strip()

        # 안전: 코드펜스/태그 잔여물 제거
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
        text = re.sub(r"<think\b[^>]*>.*?</think>", "", text, flags=re.S)
        text = re.sub(r"<analysis\b[^>]*>.*?</analysis>", "", text, flags=re.S)
        text = re.sub(r"<Thought\b[^>]*>.*?</Thought>", "", text, flags=re.S)
        text = text.strip()
        return text

    # --- 퍼블릭 API: 유효성 검사 ---
    def check(self, source_type, user_question, tables_schema, columns_schema, generated_sql) -> dict:
        """
        returns dict: {"valid": bool, "errors": "<string>"}
        """
        # 기존 ValidateSQL 프롬프트 활용
        context_prompt = PROMPTS["validatesql"]
        context_prompt = format_prompt(
            context_prompt,
            source_type=source_type,
            user_question=user_question,
            tables_schema=tables_schema,
            columns_schema=columns_schema,
            generated_sql=generated_sql,
        )


        print(" chekc 1context_prompt" , context_prompt)

        print(" chekc 2 source_type", source_type)

        print(" chekc3 user_question", user_question)
        print(" chekc4 tables_schema", tables_schema)
        print(" chekc4 columns_schema", columns_schema)
        print(" chekc6 generated_sql", generated_sql)


        # LLM 호출
        raw = self._call_local_llm(context_prompt)

        print(" chekc7 raw", raw)

        # 1차: 그대로 JSON 파싱
        try:
            return json.loads(raw)
        except Exception:
            pass

        # 2차: 본문에서 첫 JSON 객체만 추출 후 파싱
        try:
            m = re.search(r"\{.*\}", raw, flags=re.S)
            if m:
                return json.loads(m.group(0))
        except Exception:
            pass

        # 3차: 실패 시, 포맷 강제
        return {"valid": False, "errors": f"Invalid JSON from local LLM: {raw[:500]} ..."}





