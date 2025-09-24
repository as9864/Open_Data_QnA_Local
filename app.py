# app.py (patched)
# 변경 요약
# 1) logging 포맷 오류 수정: logger.info("... %s", obj) 방식으로 통일
# 2) Streamlit bare mode 가드: in_streamlit_runtime() 도입, st.session_state 접근 시 가드
# 3) eval 제거: JSON/리터럴 안전 파서(safe_parse_list_or_dict) 도입
# 4) 채팅 메시지 흐름 버그 수정: msg 사용 순서/타입 오류 수정, DataFrame은 메시지에 직접 넣지 않음
# 5) 예외 상황 로깅 강화 및 기본값/폴백 처리

import json
import logging
import asyncio
import pandas as pd

import streamlit as st
from streamlit.components.v1 import html
from streamlit.logger import get_logger

# 외부 모듈
from opendataqna import generate_uuid, get_all_databases, get_kgq
from services.chat import generate_sql_results as chat_generate_sql_results

# ---------- 로깅 포맷 ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger = get_logger(__name__)


# ---------- Streamlit 런타임 가드 ----------
def in_streamlit_runtime() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


def set_session_default(key, value):
    if not in_streamlit_runtime():
        return
    if key not in st.session_state:
        st.session_state[key] = value


# ---------- 안전 파서 ----------
def safe_parse_list_or_dict(s: str):
    """
    문자열 s를 JSON -> ast.literal_eval 순으로 시도.
    list/dict가 아니면 None 반환.streamlit run app.py
    """
    import ast
    # 1) JSON
    try:
        obj = json.loads(s)
        if isinstance(obj, (list, dict)):
            return obj
    except Exception:
        pass
    # 2) 파이썬 리터럴
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, (list, dict)):
            return obj
    except Exception:
        pass
    return None


# ---------- 세션 초기화 ----------
if in_streamlit_runtime():
    # 최초 실행 시 세션 키들 초기화
    set_session_default("session_id", generate_uuid())
    set_session_default("kgq", [])
    set_session_default("user_grouping", None)
    set_session_default("messages", [{"role": "assistant", "content": "Frequently Asked Questions"}])
    logger.info("New Session Created  - %s", st.session_state.session_id)
else:
    logger.info("Running in bare mode: Streamlit session state is not available.")


# ---------- 데이터 소스 ----------
def get_known_databases():
    """
    가용 DB 스키마 목록을 백엔드에서 받아와 리스트로 반환.
    """
    logger.info("Getting list of all user databases")
    try:
        json_groupings, _ = get_all_databases()
        logger.info("json_groupings: %s", json_groupings)
        js = json.loads(json_groupings)
        logger.info("json_groupings2: %s", js)
        groupings = [item["table_schema"] for item in js if isinstance(item, dict) and "table_schema" in item]
        if not groupings:
            logger.warning("No groupings found, fallback to ['cdm']")
            groupings = ["cdm"]
    except Exception as e:
        logger.exception("Failed to retrieve/parse database groupings. Using fallback.")
        groupings = ["cdm"]
    logger.info("user_groupings - %s", groupings)
    return groupings


def get_known_sql(selected_schema: str) -> pd.DataFrame:
    """
    특정 스키마의 KGQ(사전 정의된 SQL 예시)를 DataFrame으로 반환.
    """
    try:
        data = get_kgq(selected_schema)
        # data가 문자열 리스트/튜플/문자열 등일 수 있으므로 방어적으로 처리
        payload = None
        if isinstance(data, (list, tuple)) and data:
            payload = data[0]
        elif isinstance(data, str):
            payload = data
        else:
            logger.error("Unexpected get_kgq() return type: %s", type(data).__name__)
            return pd.DataFrame([])

        if not isinstance(payload, str):
            logger.error("get_kgq payload is not a string. Type=%s", type(payload).__name__)
            return pd.DataFrame([])

        obj = safe_parse_list_or_dict(payload)
        if obj is None:
            logger.error("Failed to parse KGQ payload (not a list/dict). startswith=%r", payload[:120])
            return pd.DataFrame([])

        if isinstance(obj, dict):
            obj = [obj]
        df = pd.DataFrame(obj)
        return df
    except Exception:
        logger.exception("get_known_sql failed.")
        return pd.DataFrame([])


def generate_response(prompt: str):
    """
    채팅 UI에 생성된 SQL/자연어 응답/결과를 순서대로 표시.
    """
    # 기존 메시지 렌더
    for m in st.session_state.messages:
        st.chat_message(m["role"]).write(m["content"])

    # 사용자 메시지
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 진행 메시지
    progress_msg = "Generating Response"
    st.session_state.messages.append({"role": "assistant", "content": progress_msg})
    st.chat_message("assistant").write(progress_msg)

    # SQL 생성/실행
    query, results, response = asyncio.run(
        chat_generate_sql_results(
            st.session_state.session_id if in_streamlit_runtime() else None,
            st.session_state.user_grouping,
            prompt,
        )
    )

    # 생성된 SQL
    st.session_state.messages.append({"role": "assistant", "content": query})
    st.chat_message("assistant").write(query)

    # 자연어 응답
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)

    # 결과 테이블 (메시지 히스토리에는 DataFrame 객체 자체를 저장하지 않음)
    with st.chat_message("assistant"):
        st.dataframe(results)


# ---------- 페이지 레이아웃 ----------
st.set_page_config(page_title='Open Data QnA', page_icon="📊", initial_sidebar_state="expanded", layout='wide')
st.markdown("""
        <style>
               .block-container {
                    padding-top: 2rem;
                    padding-bottom: 0rem;
                    padding-left: 2rem;
                    padding-right: 2rem;
                }
        </style>
        """, unsafe_allow_html=True)

st.title("Open Data QnA")

# ---------- 사이드바 ----------
with st.sidebar:
    options = get_known_databases()
    default_index = 0 if options else None
    if in_streamlit_runtime():
        st.session_state.user_grouping = st.selectbox(
            'Select Table Groupings',
            options,
            index=default_index
        )
        if st.button("New Query"):
            st.session_state.session_id = generate_uuid()
            st.session_state.messages = [{"role": "assistant", "content": "Frequently Asked Questions"}]
            st.rerun()
    else:
        # bare mode일 때도 함수가 호출될 수 있으므로 로그만 남김
        logger.info("Sidebar not rendered in bare mode.")


# ---------- KGQ 로드 ----------
if in_streamlit_runtime() and st.session_state.get("user_grouping"):
    df = get_known_sql(st.session_state.user_grouping)
    # 중복 추가 방지
    if not df.empty and "example_user_question" in df.columns:
        existing = set(st.session_state.kgq)
        for _, row in df.iterrows():
            text = str(row["example_user_question"])
            if text not in existing:
                st.session_state.kgq.append(text)
                existing.add(text)

# ---------- 입력 처리 ----------
if in_streamlit_runtime():
    prompt = st.chat_input()
    if prompt:
        generate_response(prompt)
else:
    logger.info("No chat input in bare mode.")
