# agents_local.py
from __future__ import annotations
import json
import pandas as pd
import requests

import re, unicodedata
import json as _json


def _normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"-\s*\n\s*", "", s)  # 하이픈 줄바꿈 연결
    s = re.sub(r"\s+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _build_context_from_df(df, max_chars=6000, max_rows=3):
    # title + abstract 위주로 한국어 요약에 유용한 부분만 제공
    rows = []
    for i in range(min(len(df), max_rows)):
        r = df.iloc[i].to_dict()
        title = _normalize_text(str(r.get("title", "")))
        abstract = _normalize_text(str(r.get("abstract", "")))[:2000]
        # content 컬럼이 있으면 약간만
        content = _normalize_text(str(r.get("content", "")))[:1000]
        block = f"Title: {title}\nAbstract: {abstract}"
        if content:
            block += f"\nSnippet: {content}"
        rows.append(block)
    ctx = ("\n\n---\n\n").join(rows)
    return ctx[:max_chars]

class LocalOllamaResponder:
    """
    로컬 LLM(Ollama) 기반 응답기.
    - Qwen2.5 3B Instruct 권장: qwen2.5:3b-instruct-q4_K_M
    - 결과 DF를 상위 N행으로 요약해 프롬프트에 넣음(추측 금지 지시 포함)
    """


    def __init__(self, model: str = "qwen3:8b",
                 max_tokens: int = 220,
                 temperature: float = 0.2,
                 preview_rows: int = 5,
                 host: str = "http://192.168.0.230:11434"):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.preview_rows = preview_rows
        self.url = f"{host.rstrip('/')}/api/generate"

    import json
    import pandas as pd
    from typing import Any

    def _to_df(self, data: Any) -> pd.DataFrame:
        # 이미 DataFrame이면 그대로
        if isinstance(data, pd.DataFrame):
            return data

        # None → 빈 DF
        if data is None:
            return pd.DataFrame()

        # 문자열(JSON 텍스트) 처리
        if isinstance(data, str):
            s = data.strip()
            # JSON 배열/객체로 보이면 파싱
            if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
                try:
                    parsed = json.loads(s)
                    return self._to_df(parsed)  # 재귀
                except Exception:
                    pass
            # JSON이 아니면 빈 DF
            return pd.DataFrame()

        # 딕트 처리: 표준 래퍼 키 우선
        if isinstance(data, dict):
            # 우선순위: results > data > items > records
            for key in ("results", "data", "items", "records"):
                if key in data and isinstance(data[key], list):
                    return self._to_df(data[key])  # 리스트로 재귀
            # 일반 딕트라면 단일행 DF
            return pd.DataFrame([data])

        # 리스트 처리
        if isinstance(data, list):
            if not data:
                return pd.DataFrame()
            # 리스트 원소가 문자열(JSON)인 경우 개별 파싱 시도
            if all(isinstance(x, str) for x in data):
                parsed_rows = []
                for x in data:
                    try:
                        parsed_rows.append(json.loads(x))
                    except Exception:
                        # 파싱 실패 시 텍스트 그대로 래핑
                        parsed_rows.append({"text": x})
                return pd.DataFrame(parsed_rows)
            # 리스트 원소가 딕트면 그대로 DF
            if all(isinstance(x, dict) for x in data):
                return pd.DataFrame(data)
            # 혼합형이면 문자열은 text 필드로 감싸기
            normalized = []
            for x in data:
                if isinstance(x, dict):
                    normalized.append(x)
                elif isinstance(x, str):
                    try:
                        normalized.append(json.loads(x))
                    except Exception:
                        normalized.append({"text": x})
                else:
                    normalized.append({"value": x})
            return pd.DataFrame(normalized)

        # 그 외 타입은 빈 DF
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



    def run_paper(self, user_question: str, result_df) -> str:
        print("result_df : ", result_df)
        df = self._to_df(result_df)
        print("df : " , df)
        # 규칙 기반 단답
        if "exists" in df.columns and len(df.index) >= 1:
            print("exists : ", df["exists"].iloc[0])
            v = bool(df["exists"].iloc[0])
            return f"질문: {user_question}\n\n결과: {'예' if v else '아니요'} (해당 조건을 만족하는 레코드가 {'존재합니다' if v else '존재하지 않습니다'}.)"

        if df.shape == (1, 1):
            print("df.columns[0] : ",df.columns[0])
            print("df.iloc[0, 0] : ", df.iloc[0, 0])
            col = df.columns[0]
            val = df.iloc[0, 0]
            return f"질문: {user_question}\n\n{col}: {val}"

        # LLM 컨텍스트 구성
        if df.empty:
            return f"질문: {user_question}\n\n결과가 없습니다."

        ctx = _build_context_from_df(df, max_chars=6000, max_rows=3)

        prompt = (
            "당신은 신중한 데이터 분석 도우미입니다. 아래 제공된 컨텍스트만 근거로 한국어로 간결하게 답하세요.\n"
            "반드시 컨텍스트 안에서 답을 구성하고, 불필요한 추측/장황한 설명은 금지합니다.\n"
            "컨텍스트에 충분한 근거가 없으면 '컨텍스트에 정보 없음'이라고만 답하세요.\n\n"
            f"질문: {user_question}\n\n[컨텍스트]\n{ctx}\n"
        )

        payload = {
            "model": self.model,  # qwen3:4b 등
            "prompt": prompt,
            "options": {
                "temperature": getattr(self, "temperature", 0.2),
                "num_predict": getattr(self, "max_tokens", 512),
            },
            "stream": False
        }

        try:
            res = requests.post(self.url, json=payload, timeout=60)
            res.raise_for_status()
            text = (res.json() or {}).get("response", "").strip()
            # LLM이 빈 응답/모른다를 내놓으면 폴백
            if not text or "모른다" in text or "정보 없음" in text:
                # DF에서 최소한의 요약을 규칙으로 생성
                # title 목록 + 첫 레코드의 abstract 요약
                titles = [str(t) for t in df["title"].dropna().head(3).tolist()] if "title" in df.columns else []
                abstract = str(df["abstract"].iloc[0])[:400] if "abstract" in df.columns and not df[
                    "abstract"].isna().all() else ""
                bullets = "\n".join(f"- {t}" for t in titles) if titles else "- (제목 없음)"
                return (
                    f"질문: {user_question}\n\n"
                    f"관련 문서(최대 3개):\n{bullets}\n\n"
                    f"요약: {abstract or '컨텍스트에 정보 없음'}"
                )
            return text
        except Exception:
            # 네트워크/서버 오류 폴백
            nrows, ncols = df.shape
            preview = df.head(getattr(self, "preview_rows", 5)).to_dict(orient="records")
            return (f"질문: {user_question}\n\n"
                    f"요약 생성에 실패했습니다. 총 {nrows}행, {ncols}열이 조회되었습니다. "
                    f"상위 {getattr(self, 'preview_rows', 5)}행 미리보기:\n{_json.dumps(preview, ensure_ascii=False, indent=2)}")

