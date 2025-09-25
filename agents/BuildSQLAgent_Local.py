# BuildSQLAgent_Local.py
from __future__ import annotations

import json
import requests
from typing import List, Optional, Tuple

from utilities import PROMPTS, format_prompt , CHAT_MODEL, CHAT_MODEL_URL
# 기존 BuildSQLAgent가 상속하던 Agent의 기능(세션/LLM호출)을
# 로컬 LLM 호출 코드로 치환합니다. start_chat / generate_llm_response 없이 동작.

class BuildSQLAgent_Local:
    """
    로컬 LLM(예: Ollama Qwen2.5 3B Instruct)을 사용해 SQL을 생성하는 에이전트.
    - Vertex/GCP 의존성 완전 제거
    - 기존 BuildSQLAgent의 프롬프트 구성/이력 활용 방식을 최대한 유지
    """

    def __init__(
        self,
        model: str = CHAT_MODEL,
        host: str = CHAT_MODEL_URL,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        top_p: float = 1.0,
        # Ollama에는 top_k 옵션이 직접 없지만 인터페이스 유지를 위해 둠
        timeout_sec: int = 300,
        korean_output: bool = True,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.timeout_sec = timeout_sec
        self.korean_output = korean_output

        # 단순 generate 엔드포인트 사용 (스트리밍 off)
        self.url_generate = f"{self.host}/api/generate"

    # ─────────────────────────────────────────────────────
    # 내부: 로컬 LLM 호출 (Ollama /api/generate)
    # ─────────────────────────────────────────────────────
    def _llm_complete(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_predict": self.max_tokens,
            },
            "stream": False,
        }
        resp = requests.post(self.url_generate, json=payload, timeout=self.timeout_sec)
        resp.raise_for_status()
        data = resp.json() if resp.content else {}
        return (data.get("response") or "").strip()

    # ─────────────────────────────────────────────────────
    # 공개 API: SQL 생성
    # ─────────────────────────────────────────────────────
    def build_sql(
        self,
        source_type: str,
        user_grouping: str,
        user_question: str,
        session_history: Optional[List[dict]],
        tables_schema: str,
        columns_schema: str,
        similar_sql: str,
    ) -> str:
        """
        source_type: 'bigquery' | 'postgres'
        user_grouping: 스키마/데이터셋 명
        session_history: [{'user_question': ..., 'bot_response': ...}, ...] 또는 None
        tables_schema / columns_schema / similar_sql : 상위 파이프라인에서 준비된 컨텍스트 문자열들
        """

        not_related_msg = "select 'Question is not related to the dataset' as unrelated_answer;"

        # DB별 타입 예시 텍스트 주입
        if source_type == "bigquery":
            from dbconnectors import bq_specific_data_types
            specific_data_types = bq_specific_data_types()
        else:
            from dbconnectors import pg_specific_data_types
            specific_data_types = pg_specific_data_types()

        # 유스케이스 별 추가 프롬프트
        if f"usecase_{source_type}_{user_grouping}" in PROMPTS:
            usecase_context = PROMPTS[f"usecase_{source_type}_{user_grouping}"]
        else:
            usecase_context = "No extra context for the usecase is provided"

        # 빌드 템플릿
        context_template = PROMPTS.get(f"buildsql_{source_type}", "")

        context_prompt = format_prompt(
            context_template,
            specific_data_types=specific_data_types,
            not_related_msg=not_related_msg,
            usecase_context=usecase_context,
            similar_sql=similar_sql,
            tables_schema=tables_schema,
            columns_schema=columns_schema,
        )

        # 직전 질의/SQL (이력이 있으면)
        previous_question, previous_sql = self._get_last_sql(session_history)



        # 출력 언어
        # lang_suffix = (
        #     "\n\n응답은 한국어로, 가능한 간결하게 작성하세요."
        #     if self.korean_output else ""
        # )

        build_context_prompt = f"""
[CONTEXT]
{context_prompt}

[SESSION]
- Previous Question: {previous_question}
- Previous Generated SQL: {previous_sql}

[INSTRUCTION]
- 입력 질문에 대해 위 [CONTEXT]만을 근거로, 올바른 SQL을 생성하세요.
- DDL/설명은 쓰지 말고, 최종 실행 가능한 SQL만 출력하세요.
- 백틱(```) 코드펜스는 제거하고 순수 SQL만 출력하세요.
- 질문과 무관하면 아래 문장을 그대로 반환하세요:
  {not_related_msg}


[USER QUESTION]
{user_question}
        """.strip()

        sql = self._llm_complete(build_context_prompt)
        # 코드펜스 제거 방어

        sql = sql.replace("```sql", "").replace("```", "").strip()
        return sql

    # ─────────────────────────────────────────────────────
    # 공개 API: 히스토리 기반 질문 리라이트
    # ─────────────────────────────────────────────────────
    def rewrite_question(
        self, question: str, session_history: Optional[List[dict]]
    ) -> Tuple[str, str]:
        """
        session_history를 요약해 문맥을 보강한 self-contained 질문을 생성.
        return: (concat_questions, rewritten_question)
        """
        if not session_history:
            return "", question

        formatted_history = []
        concat_questions = []
        for i, row in enumerate(session_history, start=1):
            uq = row.get("user_question")
            sql = row.get("bot_response")
            formatted_history.append(f"User Q{str(i)}: {uq}\nSQL {str(i)}: {sql}")
            if uq:
                concat_questions.append(uq)

        history_text = "\n\n".join(formatted_history)
        concat_text = " ".join(concat_questions)

        prompt = f"""
당신은 사용자의 과거 대화를 바탕으로, 아래 질문을 맥락이 충분한 단일 문장으로 재작성합니다.
- 생략된 엔티티/조건을 과거 질의에서 추론 가능하면 포함하고,
- 불명확하면 원문을 유지하되 과도한 추측은 하지 마세요.
- 한국어로 간단명료하게 작성하세요.

[SESSION HISTORY]
{history_text}

[QUESTION]
{question}

[OUTPUT]
리라이트된 질문 한 문장:
        """.strip()

        rewritten = self._llm_complete(prompt).strip()

        return concat_text, rewritten

    # ─────────────────────────────────────────────────────
    # 내부: 세션의 마지막 SQL 찾기
    # ─────────────────────────────────────────────────────
    def _get_last_sql(
        self, session_history: Optional[List[dict]]
    ) -> Tuple[Optional[str], Optional[str]]:
        if not session_history:
            return None, None
        for entry in reversed(session_history):
            uq = entry.get("user_question")
            bs = entry.get("bot_response")
            if bs:
                return uq, bs
        return None, None
