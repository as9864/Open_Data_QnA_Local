from abc import ABC
import requests
from utilities import PROMPTS, format_prompt , CHAT_MODEL, CHAT_MODEL_URL
from .core import Agent
import re

class ResponseAgent:
    """
    로컬 LLM 기반으로 SQL 결과를 자연어 응답으로 변환하는 ResponseAgentLocal 버전.
    BuildSQLAgent_Local.py를 참고해 /api/generate 엔드포인트를 호출합니다.
    """

    agentType: str = "ResponseAgentLocal"

    def __init__(
        self,
        model: str = CHAT_MODEL,
        host: str = CHAT_MODEL_URL,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        top_p: float = 1.0,
        timeout_sec: int = 300,
    ):
        super().__init__()
        self.model = model
        self.host = host.rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.timeout_sec = timeout_sec
        self.url_generate = f"{self.host}/api/generate"

    def generate_llm_response(self, prompt: str) -> str:
        safe_prompt = (
                "중요: 사고 과정, 중간 추론, <think> 태그를 절대 출력하지 마세요. "
                "최종 답변만 한국어로 불릿 포인트로 출력하세요.\n\n" + prompt
        )

        payload = {
            "model": self.model,
            "prompt": safe_prompt,
            "options": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_predict": self.max_tokens,
            },
            # stop을 여러 개 추가
            "stop": ["<think>", "</think>", "<analysis>", "</analysis>", "<Thought>", "</Thought>"],
            "stream": False,
        }
        print(self.model, self.host, self.url_generate)
        resp = requests.post(self.url_generate, json=payload, timeout=self.timeout_sec)
        resp.raise_for_status()
        data = resp.json() if resp.content else {}
        text = (data.get("response") or "").strip()

        # 사후 필터(이중 안전장치)

        text = (data.get("response") or "").strip()

        # 1차: <think> ... </think> 형태 제거
        text = re.sub(r"<think\b[^>]*>.*?</think>", "", text, flags=re.S)

        # 2차: 닫는 태그가 없는 경우, <think> 이후 전부 제거
        text = re.sub(r"<think\b[^>]*>.*", "", text, flags=re.S)

        # 보조 태그들(<analysis> 등)도 동일 처리
        text = re.sub(r"<analysis\b[^>]*>.*?</analysis>", "", text, flags=re.S)
        text = re.sub(r"<analysis\b[^>]*>.*", "", text, flags=re.S)
        text = re.sub(r"<Thought\b[^>]*>.*?</Thought>", "", text, flags=re.S)
        text = re.sub(r"<Thought\b[^>]*>.*", "", text, flags=re.S)

        text = text.strip()
        return text

    def run(self, user_question, sql_result):
        context_prompt = PROMPTS["nl_reponse"]
        context_prompt = format_prompt(
            context_prompt,
            user_question=user_question,
            sql_result=sql_result,
        )

        print(f"Prompt for Natural Language Response (Local): \n{context_prompt}")
        return self.generate_llm_response(context_prompt)
