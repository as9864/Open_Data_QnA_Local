# DebugSQLAgent_Local.py
from __future__ import annotations

import json
import requests
from typing import Optional, Tuple, List
import pandas as pd

from utilities import PROMPTS, format_prompt
from dbconnectors import pgconnector, bqconnector  # test_sql_plan_execution 재사용

class DebugSQLAgent_Local:
    """
    로컬 LLM(예: Ollama)으로 SQL 디버깅을 수행하는 에이전트.
    - Vertex / telemetry 의존성 제거
    - 기존 DebugSQLAgent의 프롬프트/흐름을 유지
    """
    agentType: str = "DebugSQLAgent"

    def __init__(
        self,
        model: str = "qwen3:4b-instruct",         # Ollama 모델명 (BuildSQLAgent_Local와 통일)
        host: str = "http://localhost:11434",    # Ollama 기본 호스트
        max_tokens: int = 1024,
        temperature: float = 0.2,
        top_p: float = 1.0,
        timeout_sec: int = 300,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.url_generate = f"{self.host}/api/generate"
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.timeout_sec = timeout_sec

    # ─────────────────────────────────────────────────────
    # 내부: 로컬 LLM 호출 (Ollama /api/generate, 비스트리밍)
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
    # 기존 init_chat 대체: 컨텍스트 프롬프트를 만들어 세션 dict로 반환
    # ─────────────────────────────────────────────────────
    def init_chat(
        self,
        source_type: str,
        user_grouping: str,
        tables_schema: str,
        columns_schema: str,
        similar_sql: str = "-No examples provided..-",
    ):
        usecase_context = PROMPTS.get(
            f"usecase_{source_type}_{user_grouping}",
            "No extra context for the usecase is provided"
        )
        context_template = PROMPTS[f"debugsql_{source_type}"]
        context_prompt = format_prompt(
            context_template,
            usecase_context=usecase_context,
            similar_sql=similar_sql,
            tables_schema=tables_schema,
            columns_schema=columns_schema,
        )
        # 로컬 버전에선 chat 객체 대신 dict에 컨텍스트를 담아 전달
        return {"context": context_prompt, "source_type": source_type}

    # ─────────────────────────────────────────────────────
    # SQL 재작성: 컨텍스트+오류메시지 기반으로 대안 SQL 생성
    # ─────────────────────────────────────────────────────
    def rewrite_sql_chat(self, chat_session, sql: str, question: str, error_df) -> str:
        error_text = error_df if isinstance(error_df, str) else pd.DataFrame(error_df).to_string(index=False)
        source_type = chat_session.get("source_type", "postgres")
        prompt = f"""{chat_session.get("context","")}

You are a senior SQL engineer. Produce an alternative SQL that addresses the error while preserving the intended answer.
Rules:
- Return only executable SQL (no code fences, no prose).
- Do not include EXPLAIN/ANALYZE.
- All selected columns must exist in referenced tables.
- Prefer ANSI SQL; use dialect-specific functions only when needed for {source_type}.

<Original SQL>
{sql}
</Original SQL>

<Original Question>
{question}
</Original Question>

<Error Message>
{error_text}
</Error Message>
"""
        out = self._llm_complete(prompt)
        return (
            out.replace("```sql", "")
               .replace("```", "")
               .replace("EXPLAIN ANALYZE ", "")
               .strip()
        )

    # ─────────────────────────────────────────────────────
    # 메인 디버거: 기존 로직과 동일하게 루프/리라이트/검증
    # ─────────────────────────────────────────────────────
    def start_debugger(
        self,
        source_type: str,
        user_grouping: str,
        query: str,
        user_question: str,
        SQLChecker,
        tables_schema: str,
        columns_schema: str,
        AUDIT_TEXT: str,
        similar_sql: str = "-No examples provided..-",
        DEBUGGING_ROUNDS: int = 2,
        LLM_VALIDATION: bool = False,
    ):
        i = 0
        STOP = False
        invalid_response = False
        print("debug_SQL 1-1 : ")
        chat_session = self.init_chat(
            source_type, user_grouping, tables_schema, columns_schema, similar_sql
        )
        print("debug_SQL 1-2 : ",chat_session)
        sql = (
            query.replace("```sql", "")
                 .replace("```", "")
                 .replace("EXPLAIN ANALYZE ", "")
        )
        print("debug_SQL 2 : ", sql)
        AUDIT_TEXT += "\n\nEntering the debugging steps!"
        while not STOP:
            json_syntax_result = {"valid": True, "errors": "None"}
            print("debug_SQL 3 : ")
            if LLM_VALIDATION:
                print("debug_SQL 4 : ", LLM_VALIDATION)
                json_syntax_result = SQLChecker.check(
                    source_type, user_question, tables_schema, columns_schema, sql
                )
                print("debug_SQL 5 : ", json_syntax_result)
            else:
                AUDIT_TEXT += "\nLLM Validation is deactivated. Jumping directly to dry run execution."
            print("debug_SQL 6 : ", json_syntax_result)
            if json_syntax_result.get("valid", True):
                AUDIT_TEXT += "\nGenerated SQL is syntactically correct as per LLM Validation!"
                connector = bqconnector if source_type == "bigquery" else pgconnector
                print("debug_SQL 7 : ", connector)
                correct_sql, exec_result_df = connector.test_sql_plan_execution(sql)
                print("debug_SQL 8 : ", correct_sql)
                if not correct_sql:
                    AUDIT_TEXT += "\nGenerated SQL failed on execution! Feedback from dry run/explain:\n" + str(exec_result_df)
                    rewrite_result = self.rewrite_sql_chat(chat_session, sql, user_question, exec_result_df)
                    AUDIT_TEXT += "\nRewritten and Cleaned SQL:\n" + str(rewrite_result)
                    sql = rewrite_result
                else:
                    STOP = True
            else:
                AUDIT_TEXT += "\nGenerated query failed on syntax check! Error Message from LLM: " + str(json_syntax_result) + "\nRewriting the query..."
                syntax_err_df = pd.read_json(json.dumps(json_syntax_result))
                rewrite_result = self.rewrite_sql_chat(chat_session, sql, user_question, syntax_err_df)
                AUDIT_TEXT += "\nRewritten SQL:\n" + str(rewrite_result)
                sql = rewrite_result

            i += 1
            if i > DEBUGGING_ROUNDS:
                AUDIT_TEXT += "\nExceeded the number of iterations for correction! The generated SQL can be invalid!"
                STOP = True
                invalid_response = True
            print(print("debug_SQL 9 : ", AUDIT_TEXT))
        return sql, invalid_response, AUDIT_TEXT
