# agents_local.py
from __future__ import annotations
import json
import pandas as pd
import requests

class LocalOllamaResponder:
    """
    로컬 LLM(Ollama) 기반 응답기.
    - Qwen2.5 3B Instruct 권장: qwen2.5:3b-instruct-q4_K_M
    - 결과 DF를 상위 N행으로 요약해 프롬프트에 넣음(추측 금지 지시 포함)
    """
    def __init__(self, model: str = "qwen3:4b-instruct",
                 max_tokens: int = 220,
                 temperature: float = 0.2,
                 preview_rows: int = 5,
                 host: str = "http://localhost:11434"):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.preview_rows = preview_rows
        self.url = f"{host.rstrip('/')}/api/generate"

    def _to_df(self, data) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data
        if data is None:
            return pd.DataFrame()
        if isinstance(data, list):
            return pd.DataFrame(data)
        if isinstance(data, dict):
            return pd.DataFrame([data])
        return pd.DataFrame()

    def _preview_json(self, df: pd.DataFrame) -> str:
        if df.empty:
            return "[]"
        # 너무 큰 값/NaN 방어
        safe = df.head(self.preview_rows).copy()
        safe = safe.replace({pd.NA: None}).astype(object)
        return json.dumps(safe.to_dict(orient="records"), ensure_ascii=False)

    def run(self, user_question: str, result_df) -> str:
        df = self._to_df(result_df)

        # 규칙 기반의 빠른 처리(속도↑):
        if "exists" in df.columns and len(df.index) >= 1:
            v = bool(df["exists"].iloc[0])
            return f"질문: {user_question}\n\n결과: {'예' if v else '아니요'} (해당 조건을 만족하는 레코드가 {'존재합니다' if v else '존재하지 않습니다'}.)"

        if df.shape == (1, 1):
            col = df.columns[0]
            val = df.iloc[0, 0]
            return f"질문: {user_question}\n\n{col}: {val}"

        # LLM 프롬프트
        preview = self._preview_json(df)
        prompt = (
            "당신은 데이터 분석 도우미입니다. 아래 JSON 결과만 근거로 한국어로 간결하게 답변하세요.\n"
            "추측하지 말고, 결과에 없는 정보는 모른다고 하세요. 핵심만 요약하세요.\n\n"
            f"질문: {user_question}\n"
            f"결과(JSON, 상위 {self.preview_rows}행 미리보기): {preview}\n"
        )

        payload = {
            "model": self.model,
            "prompt": prompt,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            },
            "stream": False
        }

        try:
            res = requests.post(self.url, json=payload, timeout=60)
            res.raise_for_status()
            text = (res.json() or {}).get("response", "").strip()
            return text or "결과를 요약할 수 없습니다."
        except Exception as e:
            # LLM 실패 시 템플릿 폴백
            if df.empty:
                return f"질문: {user_question}\n\n결과가 없습니다."
            nrows, ncols = df.shape
            return (f"질문: {user_question}\n\n"
                    f"총 {nrows}행, {ncols}열이 조회되었습니다. 상위 {self.preview_rows}행을 확인하세요.")
