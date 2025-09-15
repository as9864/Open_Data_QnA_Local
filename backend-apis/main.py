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
import firebase_admin
from firebase_admin import credentials, auth
from functools import wraps

from embeddings.store_papers import _prepare_records, store_papers, _pg_connect
from agents import EmbedderAgent, ResponseAgent
from pgvector.psycopg import register_vector

firebase_admin.initialize_app()

from services.chat import generate_sql_results as chat_generate_sql_results

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


def validate_session(session_id: str) -> tuple[str, bool]:
    now = datetime.datetime.utcnow()
    new_session = False
    expiry = _session_store.get(session_id)
    if not session_id or expiry is None or expiry < now:
        session_id = generate_uuid()
        new_session = True
    _session_store[session_id] = now + datetime.timedelta(seconds=SESSION_TIMEOUT_SECONDS)
    return session_id, new_session


def jwt_authenticated(func: Callable[..., int]) -> Callable[..., int]:
    @wraps(func)
    async def decorated_function(*args, **kwargs):
        header = request.headers.get("Authorization", None)
        if header:
            token = header.split(" ")[1]
            try:
                
                print("TOKEN::"+str(token))
                decoded_token = firebase_admin.auth.verify_id_token(token)
            except Exception as e:
                log.exception(e)
                return Response(status=403, response=f"Error with authentication: {e}")
        else:
            return Response(status=401)
        
        request.uid = decoded_token["uid"]
        print("USER:: "+str(request.uid))
        return await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
    
    return decorated_function

RUN_DEBUGGER = True
DEBUGGING_ROUNDS = 2 
LLM_VALIDATION = True
EXECUTE_FINAL_SQL = True
Embedder_model = 'vertex'
SQLBuilder_model = 'gemini-1.5-pro'
SQLChecker_model = 'gemini-1.5-pro'
SQLDebugger_model = 'gemini-1.5-pro'
num_table_matches = 5
num_column_matches = 10
table_similarity_threshold = 0.3
column_similarity_threshold = 0.3
example_similarity_threshold = 0.3
num_sql_matches = 3

app = Flask(__name__) 
cors = CORS(app, resources={r"/*": {"origins": "*"}})



@app.route("/available_databases", methods=["GET"])
# @jwt_authenticated
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
# @jwt_authenticated
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
# @jwt_authenticated
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



@app.route("/get_known_sql", methods=["POST"])
# @jwt_authenticated
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
# @jwt_authenticated
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
# @jwt_authenticated
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
# @jwt_authenticated
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
# @jwt_authenticated
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
                ORDER BY embedding <=> %s
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
        responder = ResponseAgent("gemini-1.5-pro")
        response["summary"] = responder.run(query, json.dumps(results))
    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
