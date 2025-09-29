# -*- coding: utf-8 -*-


# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from flask import Flask, request, jsonify, render_template, Response
import asyncio
from collections.abc import Callable
import logging as log
import json
import datetime
import urllib
import re
import time
import textwrap
import pandas as pd
from flask_cors import CORS
import os
import sys
from functools import wraps

from typing import Any, Dict, List, Optional

from embeddings.store_papers import _prepare_records, store_papers, _pg_connect
from agents import EmbedderAgent
from pgvector.psycopg import register_vector
from dbconnectors import audit_pgconnector


from services.chat import generate_sql_results as chat_generate_sql_results
from services.omop_concept_chat import run as concept_chat_run


from agents.Agent_local import LocalOllamaResponder as ResponderClass


import os
import json
import logging as log
import requests

from utilities import CALL_BACK_URL , CHAT_MODEL, CHAT_MODEL_URL , LOCAL_AUTH_TOKEN




#Local Ollama Responder Model
Responder = ResponderClass(
    model=CHAT_MODEL,  # ← 권장(양자화로 CPU 쾌적)
    max_tokens=220,
    temperature=0.2,
    preview_rows=5,
    host=CHAT_MODEL_URL
)



from opendataqna import (
    get_all_databases,
    get_kgq,
    generate_sql,
    embed_sql,
    get_response,
    get_results,
    visualize,
    generate_uuid,
)


module_path = os.path.abspath(os.path.join('.'))
sys.path.append(module_path)


log.basicConfig(level=log.INFO, format="%(asctime)s %(levelname)s %(message)s")

SESSION_TIMEOUT_SECONDS = 1800
_session_store: dict[str, datetime.datetime] = {}

CHAT_HISTORY_LIMIT = int(os.environ.get("CHAT_HISTORY_LIMIT", "20"))
_chat_histories: dict[str, List[Dict[str, Any]]] = {}
_chat_sessions: dict[str, str] = {}


def _get_chat_history(chat_id: str) -> List[Dict[str, Any]]:
    history = _chat_histories.get(chat_id)
    if history is None:
        history = _load_persisted_history(chat_id)
        _chat_histories[chat_id] = history
    return history


def _load_persisted_history(chat_id: str) -> List[Dict[str, Any]]:
    if not chat_id:
        return []

    try:
        records = audit_pgconnector.get_chat_history(chat_id, CHAT_HISTORY_LIMIT)
    except Exception:
        log.exception("Failed to load chat history for chat_id=%s", chat_id)
        return []

    history: List[Dict[str, Any]] = []
    for record in records:
        timestamp = record.get("timestamp")
        timestamp_str = str(timestamp) if timestamp is not None else ""

        history.append(
            {
                "questionType": record.get("questionType"),
                "question": (record.get("question") or "").strip(),
                "answer": (record.get("answer") or "").strip(),
                "timestamp": timestamp_str,
                "sessionId": record.get("sessionId"),
            }
        )

    return history


def _remember_exchange(
    chat_id: str,
    question_type: int,
    question: Optional[str],
    answer: Optional[str],
    session_id: Optional[str],
) -> None:
    if not chat_id:
        return

    recorded_at = datetime.datetime.utcnow()
    entry = {
        "questionType": question_type,
        "question": (question or "").strip(),
        "answer": (answer or "").strip(),
        "timestamp": recorded_at.isoformat() + "Z",
        "sessionId": session_id,
    }

    history = _get_chat_history(chat_id)
    history.append(entry)
    if len(history) > CHAT_HISTORY_LIMIT:
        del history[:-CHAT_HISTORY_LIMIT]

    try:
        audit_pgconnector.make_chat_history_entry(
            chat_id=chat_id,
            session_id=session_id,
            question_type=question_type,
            question=entry["question"],
            answer=entry["answer"],
            created_at=recorded_at,
        )
    except Exception:
        log.exception("Failed to persist chat exchange for chat_id=%s", chat_id)


def _resolve_chat_session(chat_id: str, provided_session_id: Optional[str] = None) -> tuple[str, bool]:
    previous_session = _chat_sessions.get(chat_id)
    base_session = provided_session_id or previous_session
    session_id, new_session = validate_session(base_session)
    _chat_sessions[chat_id] = session_id

    if new_session and previous_session and previous_session != session_id:
        _chat_histories.pop(chat_id, None)

    return session_id, new_session


def _history_snippet(history: List[Dict[str, Any]], max_turns: int = 3) -> str:
    if not history:
        return ""
    turns = history[-max_turns:]
    lines: List[str] = []
    for turn in turns:
        q = (turn.get("question") or turn.get("user_question") or "").strip()
        a = (turn.get("answer") or turn.get("bot_response") or "").strip()
        if not q and not a:
            continue
        line = f"Q: {q}" if q else ""
        if a:
            line = f"{line}\nA: {a}" if line else f"A: {a}"
        if line:
            lines.append(line)
    return "\n\n".join(lines)


def _apply_history_to_question(question: str, history: List[Dict[str, Any]]) -> str:
    snippet = _history_snippet(history)
    if not snippet:
        return question
    return f"{question}\n\n[이전 대화 참고]\n{snippet}"

# ★ NEW: 콜백 베이스 URL (환경변수로도 재정의 가능)
CALLBACK_BASE_URL = os.environ.get("CALLBACK_BASE_URL", CALL_BACK_URL)

def _callback_url(chat_id: str) -> str:
    chat_id_enc = urllib.parse.quote(str(chat_id or ""), safe="")
    return f"{CALLBACK_BASE_URL}/api/chat/callback/{chat_id_enc}"

def _post_callback(chat_id: str, answer: str, status: str = "DONE") -> None:
    url = _callback_url(chat_id)
    payload = {"answer": answer or "", "chat_status": status}
    try:
        resp = requests.post(url, json=payload, timeout=10)
        log.info("Callback POST %s %s %s", url, resp.status_code, resp.text[:2000])
    except Exception as e:
        log.exception("Callback POST failed: %s", e)






def validate_session(session_id: str) -> tuple[str, bool]:
    now = datetime.datetime.utcnow()
    new_session = False
    expiry = _session_store.get(session_id)
    if not session_id or expiry is None or expiry < now:
        session_id = generate_uuid()
        new_session = True
    _session_store[session_id] = now + datetime.timedelta(seconds=SESSION_TIMEOUT_SECONDS)
    return session_id, new_session




def _authorization_token(header: Optional[str]) -> Optional[str]:
    if not header:
        return None
    parts = header.split()
    if len(parts) == 1:
        return parts[0]
    if len(parts) >= 2 and parts[0].lower() == "bearer":
        return parts[1]
    return parts[-1]


def jwt_authenticated(func: Callable[..., int]) -> Callable[..., int]:
    @wraps(func)
    async def decorated_function(*args, **kwargs):
        if LOCAL_AUTH_TOKEN:
            header = request.headers.get("Authorization")
            token = _authorization_token(header)
            if not token:
                return Response(status=401, response="Missing Authorization header")
            if token != LOCAL_AUTH_TOKEN:
                return Response(status=403, response="Invalid local auth token")
        request.uid = "local-user"
        return await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)

    return decorated_function

RUN_DEBUGGER = True
DEBUGGING_ROUNDS = 2 
LLM_VALIDATION = True
EXECUTE_FINAL_SQL = True
Embedder_model = 'local'
SQLBuilder_model = 'timHan/llama3korean8B4QKM:latest'
SQLChecker_model = 'timHan/llama3korean8B4QKM:latest'
SQLDebugger_model = 'timHan/llama3korean8B4QKM:latest'
num_table_matches = 5
num_column_matches = 10
table_similarity_threshold = 0.3
column_similarity_threshold = 0.3
example_similarity_threshold = 0.3
num_sql_matches = 3

app = Flask(__name__) 
cors = CORS(app, resources={r"/*": {"origins": "*"}})



# ------------------------------------------------------------
# ★ NEW: 통합 엔드포인트 - 사진 구조에 맞춘 입력/콜백
# Backend → SLLM Python Server
#   Body: { "questionType": int, "question": str, "chatId": str, ... }
#   1: text2sql, 2: concept_chat, 3: papers/search
# SLLM Python Server → Backend
#   POST {CALLBACK_BASE_URL}/api/chat/callback/{chatId}
#   Body: { "answer": str, "chat_status": "DONE" | "FAIL" }
# ------------------------------------------------------------
async def _process_chat_request(body: dict) -> None:
    try:
        # 오타 대비(qeustionType)도 함께 허용
        qtype = body.get("questionType", body.get("qeustionType"))
        question = body.get("question")
        chat_id = body.get("chatId")

        # 선택 파라미터(있으면 사용)
        user_grouping = body.get("user_grouping", 'cdm')  # text2sql에서 스키마/DB 구분 등에 사용 #cdm 고정
        top_k = body.get("top_k", 5) # top_k 5개 고정
        summarize = bool(body.get("summarize", True))

        answer_text = ""

        session_id: Optional[str] = None
        if chat_id:
            session_id, _ = _resolve_chat_session(chat_id, body.get("session_id"))

        if qtype == 1:
            # ----- /query/text2sql 로직 호출 -----
            history = _get_chat_history(chat_id)
            final_sql, results_df, response = await chat_generate_sql_results(
                session_id,
                user_grouping,
                question,
            )
            # answer: 자연어 응답 우선, 없으면 SQL/row수 보조
            if response:
                answer_text = str(response)
            else:
                rows = (len(results_df) if hasattr(results_df, "__len__") else 0)
                answer_text = f"SQL 생성 완료. rows={rows}\n{final_sql}"

        elif qtype == 2:
            # ----- /omop/concept_chat 로직 호출 -----
            history = _get_chat_history(chat_id)
            payload = await concept_chat_run(question, top_k=int(top_k), history=history)
            # payload에서 문자열 응답 필드 우선 추출
            if isinstance(payload, dict):
                answer_text = payload.get("answer") or payload.get("message") \
                              or payload.get("text") or json.dumps(payload, ensure_ascii=False)
            else:
                answer_text = str(payload)

        elif qtype == 3:
            # ----- /papers/search 로직 호출 (요약 있으면 사용) -----
            history = _get_chat_history(chat_id)
            embedder = EmbedderAgent("local", "BAAI/bge-m3")
            query_emb = embedder.create(question)

            with _pg_connect() as conn:
                register_vector(conn)
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT id, title, abstract, metadata
                        FROM papers_embeddings
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s;
                        """,
                        (query_emb, int(top_k)),
                    )
                    rows = cur.fetchall()

            results = [
                {"id": r[0], "title": r[1], "abstract": r[2], "metadata": r[3]}
                for r in rows
            ]

            # 요약이 켜져 있으면 요약 생성(가능하면)
            if summarize and results:
                try:
                    contextual_question = _apply_history_to_question(question, history)
                    answer_text = Responder.run_paper(contextual_question, json.dumps(results, ensure_ascii=False))
                except Exception:
                    # 요약 실패 시 타이틀 나열
                    titles = [x.get("title") for x in results if x.get("title")]
                    answer_text = "검색 결과:\n- " + "\n- ".join(titles)
            else:
                titles = [x.get("title") for x in results if x.get("title")]
                answer_text = "검색 결과:\n- " + "\n- ".join(titles)

        else:
            raise ValueError(f"Unsupported questionType: {qtype}")

        _remember_exchange(chat_id, qtype, question, answer_text, session_id)

        # ---- 콜백 전송 ----
        _post_callback(chat_id, answer_text, status="DONE")

    except Exception as e:
        log.exception("api_chat_unified failed")
        # 실패 콜백 전송(가능하면 chatId 포함)
        try:
            chat_id = (body or {}).get("chatId")
            if chat_id:
                _post_callback(chat_id, f"에러: {e}", status="FAIL")
        except Exception:
            pass


@app.route("/api/chat", methods=["POST"])
@jwt_authenticated
async def api_chat_unified():
    body = request.get_json(silent=True) or {}

    # 오타 대비(qeustionType)도 함께 허용
    qtype = body.get("questionType", body.get("qeustionType"))
    question = body.get("question")
    chat_id = body.get("chatId")

    # 기본 검증
    if qtype is None or question is None or chat_id is None:
        return jsonify({"Error": "questionType, question, chatId are required"}), 400

    try:
        qtype = int(qtype)
    except Exception:
        return jsonify({"Error": "questionType must be int"}), 400

    if qtype not in {1, 2, 3}:
        return jsonify({"Error": f"Unsupported questionType: {qtype}"}), 400

    body["questionType"] = qtype

    asyncio.create_task(_process_chat_request(body))

    return jsonify({"chatId": chat_id, "chat_status": "PENDING"}), 202



@app.route("/available_databases", methods=["GET"])
@jwt_authenticated
def getBDList():

    result,invalid_response=get_all_databases()
    
    if not invalid_response:
        responseDict = { 
                "ResponseCode" : 200, 
                "KnownDB" : result,
                "Error":""
                }

    else:
        responseDict = { 
                "ResponseCode" : 500, 
                "KnownDB" : "",
                "Error":result
                } 
    return jsonify(responseDict)




@app.route("/embed_sql", methods=["POST"])
@jwt_authenticated
async def embedSql():

    envelope = str(request.data.decode('utf-8'))
    envelope=json.loads(envelope)
    user_grouping=envelope.get('user_grouping')
    generated_sql = envelope.get('generated_sql')
    user_question = envelope.get('user_question')
    session_id = envelope.get('session_id')

    embedded, invalid_response=await embed_sql(session_id,user_grouping,user_question,generated_sql)

    if not invalid_response:
        responseDict = { 
                        "ResponseCode" : 201, 
                        "Message" : "Example SQL has been accepted for embedding",
                        "SessionID" : session_id,
                        "Error":""
                        } 
        return jsonify(responseDict)
    else:
        responseDict = { 
                   "ResponseCode" : 500, 
                   "KnownDB" : "",
                   "SessionID" : session_id,
                   "Error":embedded
                   } 
        return jsonify(responseDict)




@app.route("/run_query", methods=["POST"])
@jwt_authenticated
def getSQLResult():
    
    envelope = str(request.data.decode('utf-8'))
    envelope=json.loads(envelope)

    user_question = envelope.get('user_question')
    user_grouping = envelope.get('user_grouping')
    generated_sql = envelope.get('generated_sql')
    session_id = envelope.get('session_id')

    result_df,invalid_response=get_results(user_grouping,generated_sql)


    if not invalid_response:
        _resp,invalid_response=get_response(session_id,user_question,result_df.to_json(orient='records'))
        if not invalid_response:
            responseDict = { 
                    "ResponseCode" : 200, 
                    "KnownDB" : result_df.to_json(orient='records'),
                    "NaturalResponse" : _resp,
                    "SessionID" : session_id,
                    "Error":""
                    }
        else:
            responseDict = { 
                    "ResponseCode" : 500, 
                    "KnownDB" : result_df.to_json(orient='records'),
                    "NaturalResponse" : _resp,
                    "SessionID" : session_id,
                    "Error":""
                    }

    else:
        _resp=result_df
        responseDict = { 
                "ResponseCode" : 500, 
                "KnownDB" : "",
                "NaturalResponse" : _resp,
                "SessionID" : session_id,
                "Error":result_df
                } 
    return jsonify(responseDict)




@app.route("/chat", methods=["POST"])
@jwt_authenticated
async def chat():
    try:
        envelope = request.get_json()
        user_question = envelope.get("user_question")
        user_grouping = envelope.get("user_grouping")

        session_id, new_session = validate_session(envelope.get("session_id"))
        uid = getattr(request, "uid", "unknown")
        log.info("/chat request - uid=%s session_id=%s time=%s", uid, session_id, datetime.datetime.utcnow().isoformat())

        final_sql, results_df, response = await chat_generate_sql_results(
            session_id,
            user_grouping,
            user_question,
        )
        results_json = (
            results_df.to_json(orient="records")
            if isinstance(results_df, pd.DataFrame)
            else results_df
        )
        resp = {
            "session_id": session_id,
            "sql": final_sql,
            "response": response,
            "results": results_json,
        }
        if new_session:
            resp["session_reset"] = True
        log.info("/chat response - uid=%s session_id=%s time=%s", uid, session_id, datetime.datetime.utcnow().isoformat())
        return jsonify(resp)
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route("/omop/concept_chat", methods=["POST"])
async def omop_concept_chat():
    payload = request.get_json(silent=True) or {}
    question = payload.get("question", "")
    if not question:
        return jsonify({"Error": "question required"}), 400
    top_k = payload.get("top_k")
    try:
        top_k_int = int(top_k) if top_k is not None else 5
    except (TypeError, ValueError):
        top_k_int = 5
    answer_payload = await concept_chat_run(question, top_k=top_k_int)
    return jsonify(answer_payload)


@app.route("/get_known_sql", methods=["POST"])
@jwt_authenticated
def getKnownSQL():
    print("Extracting the known SQLs from the example embeddings.")
    envelope = str(request.data.decode('utf-8'))
    envelope=json.loads(envelope)
    
    user_grouping = envelope.get('user_grouping')


    result,invalid_response=get_kgq(user_grouping)
    
    if not invalid_response:
        responseDict = { 
                "ResponseCode" : 200, 
                "KnownSQL" : result,
                "Error":""
                }

    else:
        responseDict = { 
                "ResponseCode" : 500, 
                "KnownSQL" : "",
                "Error":result
                } 
    return jsonify(responseDict)



@app.route("/generate_sql", methods=["POST"])
@jwt_authenticated
async def generateSQL():
    print("Here is the request payload ")
    envelope = str(request.data.decode('utf-8'))
    print("Here is the request payload " + envelope)
    envelope=json.loads(envelope)

    user_question = envelope.get('user_question')
    user_grouping = envelope.get('user_grouping')
    session_id = envelope.get('session_id')
    user_id = envelope.get('user_id')
    generated_sql,session_id,invalid_response = await generate_sql(session_id,
                user_question,
                user_grouping,  
                RUN_DEBUGGER,
                DEBUGGING_ROUNDS, 
                LLM_VALIDATION,
                Embedder_model,
                SQLBuilder_model,
                SQLChecker_model,
                SQLDebugger_model,
                num_table_matches,
                num_column_matches,
                table_similarity_threshold,
                column_similarity_threshold,
                example_similarity_threshold,
                num_sql_matches,
                user_id=user_id)

    if not invalid_response:
        responseDict = { 
                        "ResponseCode" : 200, 
                        "GeneratedSQL" : generated_sql,
                        "SessionID" : session_id,
                        "Error":""
                        }
    else:
        responseDict = { 
                        "ResponseCode" : 500, 
                        "GeneratedSQL" : "",
                        "SessionID" : session_id,
                        "Error":generated_sql
                        }          

    return jsonify(responseDict)


@app.route("/generate_viz", methods=["POST"])
@jwt_authenticated
async def generateViz():
    envelope = str(request.data.decode('utf-8'))
    # print("Here is the request payload " + envelope)
    envelope=json.loads(envelope)

    user_question = envelope.get('user_question')
    generated_sql = envelope.get('generated_sql')
    sql_results = envelope.get('sql_results')
    session_id = envelope.get('session_id')
    chart_js=''

    try:
        chart_js, invalid_response = visualize(session_id,user_question,generated_sql,sql_results)
        
        if not invalid_response:
            responseDict = { 
            "ResponseCode" : 200, 
            "GeneratedChartjs" : chart_js,
            "Error":"",
            "SessionID":session_id
            }
        else:
            responseDict = { 
                "ResponseCode" : 500, 
                "GeneratedSQL" : "",
                "SessionID":session_id,
                "Error": chart_js
                } 


        return jsonify(responseDict)

    except Exception as e:
        # util.write_log_entry("Cannot generate the Visualization!!!, please check the logs!" + str(e))
        responseDict = { 
                "ResponseCode" : 500, 
                "GeneratedSQL" : "",
                "SessionID":session_id,
                "Error":"Issue was encountered while generating the Google Chart, please check the logs!"  + str(e)
                } 
        return jsonify(responseDict)

@app.route("/summarize_results", methods=["POST"])
@jwt_authenticated
async def getSummary():
    envelope = str(request.data.decode('utf-8'))
    envelope=json.loads(envelope)
   
    user_question = envelope.get('user_question')
    sql_results = envelope.get('sql_results')

    result,invalid_response=get_response(user_question,sql_results)
    
    if not invalid_response:
        responseDict = { 
                    "ResponseCode" : 200, 
                    "summary_response" : result,
                    "Error":""
                    } 

    else:
        responseDict = { 
                    "ResponseCode" : 500, 
                    "summary_response" : "",
                    "Error":result
                    } 
    return jsonify(responseDict)




@app.route("/natural_response", methods=["POST"])
@jwt_authenticated
async def getNaturalResponse():
   envelope = str(request.data.decode('utf-8'))
   #print("Here is the request payload " + envelope)
   envelope=json.loads(envelope)
   
   user_question = envelope.get('user_question')
   user_grouping = envelope.get('user_grouping')
   
   generated_sql,session_id,invalid_response = await generate_sql(user_question,
                user_grouping,  
                RUN_DEBUGGER,
                DEBUGGING_ROUNDS, 
                LLM_VALIDATION,
                Embedder_model,
                SQLBuilder_model,
                SQLChecker_model,
                SQLDebugger_model,
                num_table_matches,
                num_column_matches,
                table_similarity_threshold,
                column_similarity_threshold,
                example_similarity_threshold,
                num_sql_matches)
   
   if not invalid_response:

        result_df,invalid_response=get_results(user_grouping,generated_sql)
        
        if not invalid_response:
            result,invalid_response=get_response(user_question,result_df.to_json(orient='records'))

            if not invalid_response:
                responseDict = { 
                            "ResponseCode" : 200, 
                            "summary_response" : result,
                            "Error":""
                            } 

            else:
                responseDict = { 
                            "ResponseCode" : 500, 
                            "summary_response" : "",
                            "Error":result
                            } 


        else:
            responseDict = { 
                    "ResponseCode" : 500, 
                    "KnownDB" : "",
                    "Error":result_df
                    } 

   else:
        responseDict = { 
                        "ResponseCode" : 500, 
                        "GeneratedSQL" : "",
                        "Error":generated_sql
                        }

   return jsonify(responseDict)   


@app.route("/get_results", methods=["POST"])
async def getResultsResponse():
   envelope = str(request.data.decode('utf-8'))
   #print("Here is the request payload " + envelope)
   envelope=json.loads(envelope)
   
   user_question = envelope.get('user_question')
   user_database = envelope.get('user_database')
   
   generated_sql,invalid_response = await generate_sql(user_question,
                user_database,  
                RUN_DEBUGGER,
                DEBUGGING_ROUNDS, 
                LLM_VALIDATION,
                Embedder_model,
                SQLBuilder_model,
                SQLChecker_model,
                SQLDebugger_model,
                num_table_matches,
                num_column_matches,
                table_similarity_threshold,
                column_similarity_threshold,
                example_similarity_threshold,
                num_sql_matches)
   
   if not invalid_response:

        result_df,invalid_response=get_results(user_database,generated_sql)
        
        if not invalid_response:
            responseDict = { 
                            "ResponseCode" : 200, 
                            "GeneratedResults" : result_df.to_json(orient='records'),
                            "Error":""
                            } 

        else:
            responseDict = { 
                    "ResponseCode" : 500, 
                    "GeneratedResults" : "",
                    "Error":result_df
                    } 

   else:
        responseDict = { 
                        "ResponseCode" : 500, 
                        "GeneratedResults" : "",
                        "Error":generated_sql
                        }

   return jsonify(responseDict)  
   
@app.route("/papers/embed", methods=["POST"])
def embed_papers():
    """Embed paper documents and store them in Postgres."""
    docs = request.get_json(silent=True)
    if not docs:
        return jsonify({"Error": "No documents provided"}), 400
    if isinstance(docs, dict):
        docs = [docs]
    try:
        records = _prepare_records(docs)
        inserted = store_papers(records)
        return jsonify({"inserted": inserted}), 201
    except Exception as e:  # pragma: no cover - error path
        log.exception("Failed to store papers")
        return jsonify({"Error": str(e)}), 500


@app.route("/papers/search", methods=["POST"])
def search_papers():
    """Search embedded papers and optionally summarise results."""
    payload = request.get_json(silent=True) or {}
    query = payload.get("query")
    k = int(payload.get("k", 5))
    summarize = bool(payload.get("summarize"))
    if not query:
        return jsonify({"Error": "Query text required"}), 400

    embedder = EmbedderAgent("local", "BAAI/bge-m3")
    query_emb = embedder.create(query)

    with _pg_connect() as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, title, abstract, metadata
                FROM papers_embeddings
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """,
                (query_emb, k),
            )
            rows = cur.fetchall()

    results = [
        {
            "id": r[0],
            "title": r[1],
            "abstract": r[2],
            "metadata": r[3],
        }
        for r in rows
    ]

    response = {"results": results}
    if summarize and results:
        # responder = R

        print("check Results" , results)
        print("check query", query)
        response["summary"] = Responder.run_paper(query, json.dumps(results))
    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
