import requests
import json
import traceback


def build_prompt(question: str, draft_response: str, evidence: str = None, instructions: str = None) -> str:
    """_build_prompt()를 단순하게 대체"""
    return f"""초안 응답을 사실성, 간결성, 한국어 가독성 관점에서 검토하고 필요한 부분만 다듬어 주세요.
요구사항:
- 초안의 정보 중 근거와 일치하지 않는 내용은 삭제하거나 수정하세요.
- 새로운 사실이나 추측은 추가하지 마세요.
- 불필요한 반복을 줄이고 간결한 문장으로 정리하세요.
- 한국어 맞춤법과 가독성을 개선하세요.
- 최종 답변만 출력하세요.

[사용자 질문]
{question}

[초안 응답]
{draft_response}

[근거]
{evidence or "없음"}

{instructions or ""}
"""


def test_answer_refiner():
    # === 🔧 환경 설정 ===
    host = "http://222.236.26.27:25123"  # Ollama 서버 주소
    model = "timHan/llama3korean8B4QKM:latest"
    temperature = 0.1
    max_tokens = 512
    timeout_sec = 60

    # === 🧠 테스트 입력 ===
    question = "폐혈증 관련된 SNOMED 코드 알려줘"
    draft_response = """
• concept_id: 123456, concept_name: "Septicemia", vocabulary_id: SNOMED, domain_id: Condition, concept_class_id: Diagnosis  
  • 폐혈증은 혈액 감염으로 인해 발생하는 상태로, 심각한 시스템 적응반응을 유발함.
• concept_id: 123457, concept_name: "Sepsis", vocabulary_id: SNOMED, domain_id: Condition, concept_class_id: Diagnosis  
  • 세피스는 폐혈증의 심각한 형태임.
"""
    evidence = "SNOMED 기준으로 폐혈증은 Sepsis(128613002) 코드로 정의됨."

    # === 🧩 prompt 생성 ===
    prompt = build_prompt(question, draft_response, evidence)
    print("\n==============================")
    print("PROMPT:\n", prompt)
    print("==============================\n")

    # === 🚀 payload 구성 ===
    payload = {
        "model": model,
        "prompt": prompt,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
        "stop": ["<think>", "</think>", "<analysis>", "</analysis>", "<Thought>", "</Thought>"],
        "stream": False,
    }

    # === 🌐 요청 URL (trailing slash 포함 권장) ===
    url = f"{host.rstrip('/')}/api/generate/"
    print("➡️  URL:", url)
    print("➡️  Model:", model)
    print("➡️  Payload:", json.dumps(payload, ensure_ascii=False, indent=2))

    try:
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=timeout_sec,
        )

        print("\n🛰️  Response status:", response.status_code)
        print("🛰️  Response text:", response.text[:1000])  # 1000자까지만 출력

        response.raise_for_status()
        data = response.json() if response.content else {}

        refined = (data.get("response") or "").strip()
        if not refined:
            print("\n⚠️  refined 응답 비어 있음 — 서버에서 응답이 비정상일 수 있습니다.")
        else:
            print("\n✅ REFINED ANSWER:\n", refined)

    except Exception as e:
        print("\n❌ ERROR OCCURRED")
        print(traceback.format_exc())


if __name__ == "__main__":
    test_answer_refiner()