"""Local PostgreSQL connector.

This module provides a lightweight connector implementation that uses
``sqlalchemy`` and ``psycopg2`` to talk to a locally running PostgreSQL
instance.  It mirrors the :class:`PgConnector` interface used for the
cloud implementation but avoids any dependency on Google Cloud specific
libraries so that the application can be executed completely offline.
"""

from __future__ import annotations

from abc import ABC
import asyncpg
from pgvector.asyncpg import register_vector
import asyncio

import pandas as pd
from sqlalchemy import create_engine, text, bindparam
from pgvector.sqlalchemy import Vector
import scripts
from .core import DBConnector

from datetime import datetime

from utilities import root_dir
# from google.cloud.sql.connector import Connector

class LocalPgConnector(DBConnector, ABC):
    """Connector for a local PostgreSQL database.

    Parameters
    ----------
    conn_str:
        SQLAlchemy style connection string (e.g.
        ``postgresql+psycopg2://user:pass@localhost/db``).
    """

    def __init__(self, conn_str: str):
        self.conn_str = conn_str
        self.engine = create_engine(conn_str)
        self._ensure_audit_table()

    def getconn(self):
        """Return a new connection object."""

        return self.engine.connect()

    def retrieve_df(self, query: str) -> pd.DataFrame:
        """Execute *query* and return the result as a :class:`DataFrame`."""

        with self.getconn() as conn:
            return pd.read_sql_query(text(query), conn)

    def getExactMatches(self, query):
        """
        Checks if the exact question is already present in the example SQL set
        """
        check_history_sql=f"""SELECT example_user_question,example_generated_sql
        FROM example_prompt_sql_embeddings
        WHERE lower(example_user_question) = lower('{query}') LIMIT 1; """


        print("getExactMatches : " , check_history_sql)

        print("getExactMatches2 : ", query)




        exact_sql_history = self.retrieve_df(check_history_sql)

        print("getExactMatches3 : ", exact_sql_history)


        if exact_sql_history[exact_sql_history.columns[0]].count() != 0:
            sql_example_txt = ''
            exact_sql = ''
            for index, row in exact_sql_history.iterrows():
                example_user_question=row["example_user_question"]
                example_sql=row["example_generated_sql"]
                exact_sql=example_sql
                sql_example_txt = sql_example_txt + "\n Example_question: "+example_user_question+ "; Example_SQL: "+example_sql

            # print("Found a matching question from the history!" + str(sql_example_txt))
            final_sql=exact_sql

        else:
            print("No exact match found for the user prompt")
            final_sql = None

        return final_sql

    # async def cache_known_sql(self):
    #
    #     df = pd.read_csv(f"{root_dir}/{scripts}/known_good_sql.csv")
    #     df = df.loc[:, ["prompt", "sql", "database_name"]]
    #     df = df.dropna()
    #
    #     loop = asyncio.get_running_loop()
    #     async with Connector(loop=loop) as connector:
    #         # # Create connection to Cloud SQL database.
    #         conn: asyncpg.Connection = await connector.connect_async(
    #             f"{self.project_id}:{self.region}:{self.instance_name}",
    #             "asyncpg",
    #             user=f"{self.database_user}",
    #             password=f"{self.database_password}",
    #             db=f"{self.database_name}",
    #         )
    #
    #         await register_vector(conn)
    #
    #         # Delete the table if it exists.
    #         await conn.execute("DROP TABLE IF EXISTS query_example_embeddings CASCADE")
    #
    #         # Create the `query_example_embeddings` table.
    #         await conn.execute(
    #             """CREATE TABLE query_example_embeddings(
    #                                 prompt TEXT,
    #                                 sql TEXT,
    #                                 user_grouping TEXT)"""
    #         )
    #
    #         # Copy the dataframe to the 'query_example_embeddings' table.
    #         tuples = list(df.itertuples(index=False))
    #         await conn.copy_records_to_table(
    #             "query_example_embeddings", records=tuples, columns=list(df), timeout=10000
    #         )
    #
    #     await conn.close()

    async def cache_known_sql(self):
        """
        known_good_sql.csv -> query_example_embeddings (prompt, sql, user_grouping)
        로컬(Postgres) 전용. Cloud SQL Connector/ADC 불필요.
        """
        # 1) CSV 로드 & 정리
        df = pd.read_csv(f"{root_dir}/{scripts}/known_good_sql.csv")
        df = df.loc[:, ["prompt", "sql", "database_name"]].dropna()
        df = df.rename(columns={"database_name": "user_grouping"})

        # 2) 동기 DB 작업을 스레드로 실행 (스트림릿/비동기 충돌 방지)
        def _work():
            # 2-1) 연결 열기
            with self.getconn() as conn:
                # 2-2) 테이블 준비 (없으면 생성)
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS query_example_embeddings (
                        prompt        TEXT NOT NULL,
                        sql           TEXT NOT NULL,
                        user_grouping TEXT NOT NULL
                    );
                """))
                # 필요 시 완전 재적재하려면 TRUNCATE (DROP 대신 권장)
                conn.execute(text("TRUNCATE TABLE query_example_embeddings;"))

                # 2-3) bulk insert
                # SQLAlchemy의 executemany: 리스트[dict]를 한 번에 넣을 수 있음
                records = df.to_dict(orient="records")
                conn.execute(
                    text("""
                        INSERT INTO query_example_embeddings (prompt, sql, user_grouping)
                        VALUES (:prompt, :sql, :user_grouping)
                    """),
                    records
                )
                conn.commit()

        await asyncio.to_thread(_work)



    # async def retrieve_matches(self, mode, user_grouping, qe, similarity_threshold, limit):
    #     """
    #     This function retrieves the most similar table_schema and column_schema.
    #     Modes can be either 'table', 'column', or 'example'
    #     """
    #
    #     print("retrieve_matches 1")
    #     matches = []
    #
    #     loop = asyncio.get_running_loop()
    #     print("retrieve_matches 2",loop)
    #     async with Connector(loop=loop) as connector:
    #         print("retrieve_matches 3")
    #         # # Create connection to Cloud SQL database.
    #         conn: asyncpg.Connection = await connector.connect_async(
    #             f"{self.project_id}:{self.region}:{self.instance_name}",
    #             "asyncpg",
    #             user=f"{self.database_user}",
    #             password=f"{self.database_password}",
    #             db=f"{self.database_name}",
    #         )
    #         print("retrieve_matches 33")
    #         await register_vector(conn)
    #
    #         # Prepare the SQL depending on 'mode'
    #         if mode == 'table':
    #             sql = """
    #                 SELECT content as tables_content,
    #                 1 - (embedding <=> $1) AS similarity
    #                 FROM table_details_embeddings
    #                 WHERE 1 - (embedding <=> $1) > $2
    #                 AND user_grouping = $4
    #                 ORDER BY similarity DESC LIMIT $3
    #             """
    #
    #
    #         elif mode == 'column':
    #             sql = """
    #                 SELECT content as columns_content,
    #                 1 - (embedding <=> $1) AS similarity
    #                 FROM tablecolumn_details_embeddings
    #                 WHERE 1 - (embedding <=> $1) > $2
    #                 AND user_grouping = $4
    #                 ORDER BY similarity DESC LIMIT $3
    #             """
    #
    #         elif mode == 'example':
    #             sql = """
    #                 SELECT user_grouping, example_user_question, example_generated_sql,
    #                 1 - (embedding <=> $1) AS similarity
    #                 FROM example_prompt_sql_embeddings
    #                 WHERE 1 - (embedding <=> $1) > $2
    #                 AND user_grouping = $4
    #                 ORDER BY similarity DESC LIMIT $3
    #             """
    #
    #         else:
    #             ValueError("No valid mode. Must be either table, column, or example")
    #             name_txt = ''
    #         print("retrieve_matches 4", sql)
    #         # print(sql,qe,similarity_threshold,limit,user_grouping)
    #         print(sql, qe, similarity_threshold, limit,user_grouping)
    #         # FETCH RESULTS FROM POSTGRES DB
    #         results = await conn.fetch(
    #             sql,
    #             qe,
    #             similarity_threshold,
    #             limit,
    #             user_grouping
    #         )
    #         print("retrieve_matches 2" , results)
    #         # CHECK RESULTS
    #         if len(results) == 0:
    #             print(f"Did not find any results  for {mode}. Adjust the query parameters.")
    #         else:
    #             print(f"Found {len(results)} similarity matches for {mode}.")
    #
    #         if mode == 'table':
    #             name_txt = ''
    #             for r in results:
    #                 name_txt = name_txt + r["tables_content"] + "\n\n"
    #
    #         elif mode == 'column':
    #             name_txt = ''
    #             for r in results:
    #                 name_txt = name_txt + r["columns_content"] + "\n\n "
    #
    #         elif mode == 'example':
    #             name_txt = ''
    #             for r in results:
    #                 example_user_question = r["example_user_question"]
    #                 example_sql = r["example_generated_sql"]
    #                 # print(example_user_question+"\nThreshold::"+str(r["similarity"]))
    #                 name_txt = name_txt + "\n Example_question: " + example_user_question + "; Example_SQL: " + example_sql
    #
    #         else:
    #             ValueError("No valid mode. Must be either table, column, or example")
    #             name_txt = ''
    #         print("retrieve_matches 3", name_txt)
    #         matches.append(name_txt)
    #
    #     # Close the connection to the database.
    #     await conn.close()
    #
    #     return matches

    def _to_list(vec):
        import numpy as np
        if isinstance(vec, np.ndarray): return vec.astype(float).tolist()
        if isinstance(vec, (list, tuple)): return list(vec)
        return vec

    # async def retrieve_matches(
    #         self,
    #         mode: str,
    #         user_grouping: str,
    #         qe,  # 질문 임베딩 (list / np.ndarray)
    #         similarity_threshold: float,
    #         limit: int,
    #         filter_by_grouping: bool = False,  # 테이블에 user_grouping 컬럼이 있을 때만 True
    # ):
    #     """
    #     mode: 'table' | 'column' | 'example'
    #     - 'example' 테이블에는 보통 user_grouping이 있으니 filter_by_grouping=True 추천
    #     - 'table' / 'column' 에 user_grouping 컬럼이 없으면 False
    #     """
    #     print("retrieve_matches(local) 1")
    #     qe = LocalPgConnector._to_list(qe)
    #
    #     conn: asyncpg.Connection = await asyncpg.connect(
    #         host=self.host,
    #         port=self.port,
    #         user=self.database_user,
    #         password=self.database_password,
    #         database=self.database_name,
    #     )
    #     try:
    #         await register_vector(conn)
    #         print("retrieve_matches(local) 2 connected")
    #
    #         # 코사인 거리 d = embedding <=> qe / 유사도 = 1 - d
    #         if mode == 'table':
    #             where_group = "AND user_grouping = $4" if filter_by_grouping else ""
    #             sql = f"""
    #                 SELECT content AS tables_content,
    #                        1 - (embedding <=> $1) AS similarity
    #                 FROM table_details_embeddings
    #                 WHERE 1 - (embedding <=> $1) > $2
    #                   {where_group}
    #                 ORDER BY similarity DESC
    #                 LIMIT $3;
    #             """
    #             params = [qe, similarity_threshold, limit] + ([user_grouping] if filter_by_grouping else [])
    #
    #         elif mode == 'column':
    #             where_group = "AND user_grouping = $4" if filter_by_grouping else ""
    #             sql = f"""
    #                 SELECT content AS columns_content,
    #                        1 - (embedding <=> $1) AS similarity
    #                 FROM tablecolumn_details_embeddings
    #                 WHERE 1 - (embedding <=> $1) > $2
    #                   {where_group}
    #                 ORDER BY similarity DESC
    #                 LIMIT $3;
    #             """
    #             params = [qe, similarity_threshold, limit] + ([user_grouping] if filter_by_grouping else [])
    #
    #         elif mode == 'example':
    #             # example_prompt_sql_embeddings 는 user_grouping 컬럼 존재 전제
    #             sql = """
    #                 SELECT user_grouping, example_user_question, example_generated_sql,
    #                        1 - (embedding <=> $1) AS similarity
    #                 FROM example_prompt_sql_embeddings
    #                 WHERE 1 - (embedding <=> $1) > $2
    #                   AND user_grouping = $4
    #                 ORDER BY similarity DESC
    #                 LIMIT $3;
    #             """
    #             params = [qe, similarity_threshold, limit, user_grouping]
    #
    #         else:
    #             raise ValueError("mode must be 'table' | 'column' | 'example'")
    #
    #         print("retrieve_matches(local) SQL:", sql)
    #         rows = await conn.fetch(sql, *params)
    #         print(f"retrieve_matches(local) rows={len(rows)}")
    #
    #         # 파이프라인이 기대하는 문자열 포맷으로 변환
    #         if not rows:
    #             return [""]
    #
    #         if mode == 'table':
    #             lines = [r["tables_content"] for r in rows]
    #             return ["Schema(values): " + "\n\n".join(lines)]
    #
    #         if mode == 'column':
    #             lines = [r["columns_content"] for r in rows]
    #             return ["Column name(type): " + "\n\n".join(lines)]
    #
    #         if mode == 'example':
    #             parts = []
    #             for r in rows:
    #                 parts.append(
    #                     f"\n Example_question: {r['example_user_question']}; "
    #                     f"Example_SQL: {r['example_generated_sql']}"
    #                 )
    #             return ["".join(parts)]
    #
    #     finally:
    #         await conn.close()

    from sqlalchemy import text

    # def retrieve_matches(
    #         self,
    #         mode: str,
    #         user_grouping: str,
    #         qe,  # 질문 임베딩 (list / np.ndarray)
    #         similarity_threshold: float,
    #         limit: int,
    #         filter_by_grouping: bool = False,
    # ):
    #     """
    #     mode: 'table' | 'column' | 'example'
    #     로컬 DB (getconn) 연결 사용
    #     """
    #     qe = LocalPgConnector._to_list(qe)  # numpy → list 변환
    #     matches = []
    #     dim = 1024
    #     with self.getconn() as conn:
    #         # vector 확장 등록
    #         conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
    #
    #         if mode == "table":
    #             where_group = "AND user_grouping = :user_grouping" if filter_by_grouping else ""
    #             sql = f"""
    #                 SELECT content AS tables_content,
    #                        1 - (embedding <=> :qe) AS similarity
    #                 FROM table_details_embeddings
    #                 WHERE 1 - (embedding <=> :qe) > :threshold
    #                   {where_group}
    #                 ORDER BY similarity DESC
    #                 LIMIT :limit;
    #             """
    #             params = {"qe": qe, "threshold": similarity_threshold, "limit": limit}
    #             if filter_by_grouping:
    #                 params["user_grouping"] = user_grouping
    #
    #         elif mode == "column":
    #             where_group = "AND user_grouping = :user_grouping" if filter_by_grouping else ""
    #             sql = f"""
    #                 SELECT content AS columns_content,
    #                        1 - (embedding <=> :qe) AS similarity
    #                 FROM tablecolumn_details_embeddings
    #                 WHERE 1 - (embedding <=> :qe) > :threshold
    #                   {where_group}
    #                 ORDER BY similarity DESC
    #                 LIMIT :limit;
    #             """
    #             params = {"qe": qe, "threshold": similarity_threshold, "limit": limit}
    #             if filter_by_grouping:
    #                 params["user_grouping"] = user_grouping
    #
    #         elif mode == "example":
    #             sql = """
    #                 SELECT user_grouping, example_user_question, example_generated_sql,
    #                        1 - (embedding <=> :qe) AS similarity
    #                 FROM example_prompt_sql_embeddings
    #                 WHERE 1 - (embedding <=> :qe) > :threshold
    #                   AND user_grouping = :user_grouping
    #                 ORDER BY similarity DESC
    #                 LIMIT :limit;
    #             """
    #             params = {
    #                 "qe": qe,
    #                 "threshold": similarity_threshold,
    #                 "limit": limit,
    #                 "user_grouping": user_grouping,
    #             }
    #
    #         else:
    #             raise ValueError("mode must be 'table' | 'column' | 'example'")
    #
    #         rows = conn.execute(text(sql).bindparams(bindparam("qe", type_=Vector(dim))), params).fetchall()
    #
    #     # 결과 가공
    #     if not rows:
    #         return [""]
    #
    #     if mode == "table":
    #         lines = [r["tables_content"] for r in rows]
    #         return ["Schema(values): " + "\n\n".join(lines)]
    #
    #     if mode == "column":
    #         lines = [r["columns_content"] for r in rows]
    #         return ["Column name(type): " + "\n\n".join(lines)]
    #
    #     if mode == "example":
    #         parts = []
    #         for r in rows:
    #             parts.append(
    #                 f"\n Example_question: {r['example_user_question']}; Example_SQL: {r['example_generated_sql']}"
    #             )
    #         return ["".join(parts)]

    def retrieve_matches(
            self,
            mode: str,
            user_grouping: str,
            qe,  # list / np.ndarray
            similarity_threshold: float,
            limit: int,
            filter_by_grouping: bool = False,
    ):
        # pgvector: 바인딩 타입 맞추기
        qe_vec = list(qe)  # ndarray -> list
        dim = len(qe_vec)
        print(" mode : " , mode , type(qe))
        with self.getconn() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))

            if mode == "table":

                where_group = "AND user_grouping = :user_grouping" if filter_by_grouping else ""
                sql = f"""
                    SELECT content AS tables_content,
                           1 - (embedding <=> :qe) AS similarity
                    FROM table_details_embeddings
                    WHERE 1 - (embedding <=> :qe) > :threshold
                      {where_group}
                    ORDER BY similarity DESC
                    LIMIT :limit
                """
                print(" table : ", sql)
                stmt = (
                    text(sql)
                    .bindparams(bindparam("qe", type_=Vector(dim)))
                )
                params = {"qe": qe_vec, "threshold": similarity_threshold, "limit": limit}
                if filter_by_grouping:
                    params["user_grouping"] = user_grouping

            elif mode == "column":
                where_group = "AND user_grouping = :user_grouping" if filter_by_grouping else ""
                sql = f"""
                    SELECT content AS columns_content,
                           1 - (embedding <=> :qe) AS similarity
                    FROM tablecolumn_details_embeddings
                    WHERE 1 - (embedding <=> :qe) > :threshold
                      {where_group}
                    ORDER BY similarity DESC
                    LIMIT :limit
                """
                stmt = (
                    text(sql)
                    .bindparams(bindparam("qe", type_=Vector(dim)))
                )
                print(" column : ", sql)
                params = {"qe": qe_vec, "threshold": similarity_threshold, "limit": limit}
                if filter_by_grouping:
                    params["user_grouping"] = user_grouping

            elif mode == "example":
                sql = """
                    SELECT user_grouping, example_user_question, example_generated_sql,
                           1 - (embedding <=> :qe) AS similarity
                    FROM example_prompt_sql_embeddings
                    WHERE 1 - (embedding <=> :qe) > :threshold
                      AND user_grouping = :user_grouping
                    ORDER BY similarity DESC
                    LIMIT :limit
                """
                print(" example : ", sql)
                stmt = (
                    text(sql)
                    .bindparams(bindparam("qe", type_=Vector(dim)))
                )
                params = {
                    "qe": qe_vec,
                    "threshold": similarity_threshold,
                    "limit": limit,
                    "user_grouping": user_grouping,
                }
            else:
                raise ValueError("mode must be 'table' | 'column' | 'example'")
            print(" execute before ")
            # ✅ 딕셔너리 모드로 받기
            rows = conn.execute(stmt, params).mappings().all()
            print(" execute after " , rows)
        if not rows:
            return [""]

        if mode == "table":
            lines = [r["tables_content"] for r in rows]
            return ["Schema(values): " + "\n\n".join(lines)]
        if mode == "column":
            lines = [r["columns_content"] for r in rows]
            return ["Column name(type): " + "\n\n".join(lines)]
        if mode == "example":
            parts = [
                f"\n Example_question: {r['example_user_question']}; Example_SQL: {r['example_generated_sql']}"
                for r in rows
            ]
            return ["".join(parts)]

    async def getSimilarMatches(self, mode, user_grouping, qe, num_matches, similarity_threshold):

        print("localPGConnector getSimilarMatches")
        if mode == 'table':
            print("localPGConnector getSimilarMatches table")
            match_result = self.retrieve_matches(mode, user_grouping, qe, similarity_threshold, num_matches)
            match_result = match_result[0]

        elif mode == 'column':
            print("localPGConnector getSimilarMatches Column")
            match_result = self.retrieve_matches(mode, user_grouping, qe, similarity_threshold, num_matches)
            match_result = match_result[0]

        elif mode == 'example':
            print("localPGConnector getSimilarMatches Example")
            match_result = self.retrieve_matches(mode, user_grouping, qe, similarity_threshold, num_matches)
            print("localPGConnector getSimilarMatches match_result" , match_result)
            if len(match_result) == 0:
                match_result = None
            else:
                match_result = match_result[0]

        return match_result

    def test_sql_plan_execution(self, generated_sql):
        try:
            exec_result_df = pd.DataFrame()
            sql = f"""EXPLAIN ANALYZE {generated_sql}"""
            exec_result_df = self.retrieve_df(sql)

            if not exec_result_df.empty:
                if str(exec_result_df.iloc[0]).startswith('Error. Message'):
                    correct_sql = False

                else:
                    print('\n No need to rewrite the query. This seems to work fine and returned rows...')
                    correct_sql = True
            else:
                print('\n No need to rewrite the query. This seems to work fine but no rows returned...')
                correct_sql = True

            return correct_sql, exec_result_df

        except Exception as e:
            return False, str(e)

        def return_column_schema_sql(self, schema, table_names=None):
            """
            This SQL returns a df containing the cols table_schema, table_name, column_name, data_type, column_description, table_description, primary_key, column_constraints
            for the schema specified above, e.g. 'retail'
            - table_schema: e.g. retail
            - table_name: name of the table inside the schema, e.g. products
            - column_name: name of each col in each table in the schema, e.g. id_product
            - data_type: data type of each col
            - column_description: col descriptor, can be empty
            - table_description: text descriptor, can be empty
            - primary_key: whether the col is PK; if yes, the field contains the col_name
            - column_constraints: e.g. "Primary key for this table"
            """
            table_filter_clause = ""
            if table_names:
                # table_names = [name.strip() for name in table_names[1:-1].split(",")]  # Handle the string as a list
                formatted_table_names = [f"'{name}'" for name in table_names]
                table_filter_clause = f"""and table_name in ({', '.join(formatted_table_names)})"""

            column_schema_sql = f'''
            WITH
            columns_schema
            AS
            (select c.table_schema,c.table_name,c.column_name,c.data_type,d.description as column_description, obj_description(c1.oid) as table_description
            from information_schema.columns c
            inner join pg_class c1
            on c.table_name=c1.relname
            inner join pg_catalog.pg_namespace n
            on c.table_schema=n.nspname
            and c1.relnamespace=n.oid
            left join pg_catalog.pg_description d
            on d.objsubid=c.ordinal_position
            and d.objoid=c1.oid
            where
            c.table_schema='{schema}' {table_filter_clause}) ,
            pk_schema as
            (SELECT table_name, column_name AS primary_key
            FROM information_schema.key_column_usage
    WHERE TABLE_SCHEMA='{schema}' {table_filter_clause}
            AND CONSTRAINT_NAME like '%_pkey%'
            ORDER BY table_name, primary_key),
            fk_schema as
            (SELECT table_name, column_name AS foreign_key
            FROM information_schema.key_column_usage
            WHERE TABLE_SCHEMA='{schema}' {table_filter_clause}
            AND CONSTRAINT_NAME like '%_fkey%'
            ORDER BY table_name, foreign_key)

            select lr.*,
            case
            when primary_key is not null then 'Primary key for this table'
            when foreign_key is not null then CONCAT('Foreign key',column_description)
            else null
            END as column_constraints
            from
            (select l.*,r.primary_key
            from
            columns_schema l
            left outer join
            pk_schema r
            on
            l.table_name=r.table_name
            and
            l.column_name=r.primary_key) lr
            left outer join
            fk_schema rt
            on
            lr.table_name=rt.table_name
            and
            lr.column_name=rt.foreign_key
            ;
            '''

            return column_schema_sql

        def return_table_schema_sql(self, schema, table_names=None):
            """
            This SQL returns a df containing the cols table_schema, table_name, table_description, table_columns (with cols in the table)
            for the schema specified above, e.g. 'retail'
            - table_schema: e.g. retail
            - table_name: name of the table inside the schema, e.g. products
            - table_description: text descriptor, can be empty
            - table_columns: aggregate of the col names inside the table
            """

            table_filter_clause = ""

            if table_names:
                # Extract individual table names from the input string
                # table_names = [name.strip() for name in table_names[1:-1].split(",")]  # Handle the string as a list
                formatted_table_names = [f"'{name}'" for name in table_names]
                table_filter_clause = f"""and table_name in ({', '.join(formatted_table_names)})"""

            table_schema_sql = f'''
            SELECT table_schema, table_name,table_description, array_to_string(array_agg(column_name), ' , ') as table_columns
            FROM
            (select c.table_schema,c.table_name,c.column_name,c.ordinal_position,c.column_default,c.data_type,d.description, obj_description(c1.oid) as table_description
            from information_schema.columns c
            inner join pg_class c1
            on c.table_name=c1.relname
            inner join pg_catalog.pg_namespace n
            on c.table_schema=n.nspname
            and c1.relnamespace=n.oid
            left join pg_catalog.pg_description d
            on d.objsubid=c.ordinal_position
            and d.objoid=c1.oid
            where
            c.table_schema='{schema}' {table_filter_clause} ) data
            GROUP BY table_schema, table_name, table_description
            ORDER BY table_name;
            '''

            return table_schema_sql

        def get_column_samples(self, columns_df):
            sample_column_list = []

            for index, row in columns_df.iterrows():
                get_column_sample_sql = f'''SELECT most_common_vals AS sample_values FROM pg_stats WHERE tablename = '{row["table_name"]}' AND schemaname = '{row["table_schema"]}' AND attname = '{row["column_name"]}' '''

                column_samples_df = self.retrieve_df(get_column_sample_sql)
                # display(column_samples_df)
                sample_column_list.append(
                    column_samples_df['sample_values'].to_string(index=False).replace("{", "").replace("}", ""))

            columns_df["sample_values"] = sample_column_list
            return columns_df

    def _ensure_audit_table(self) -> None:
        with self.getconn() as conn:
            conn.execute(text(
                """
                CREATE TABLE IF NOT EXISTS audit_log (
                    id SERIAL PRIMARY KEY,
                    source_type TEXT,
                    user_grouping TEXT,
                    model_used TEXT,
                    question TEXT,
                    generated_sql TEXT,
                    execution_time TIMESTAMPTZ,
                    full_log TEXT
                );
                """)
            )
            conn.commit()

    def make_audit_entry(
        self,
        source_type: str,
        user_grouping: str,
        model: str,
        question: str,
        generated_sql: str,
        found_in_vector: bool,
        need_rewrite: bool,
        failure_step: str,
        error_msg: str,
        full_log_text: str,
    ) -> str:
        """Persist a minimal audit record to PostgreSQL."""



        with self.getconn() as conn:
            stmt = text("""
                    INSERT INTO audit_log
                        (source_type, user_grouping, model_used, question, generated_sql, execution_time, full_log)
                    VALUES
                        (:source_type, :user_grouping, :model, :question, :generated_sql, :execution_time, :full_log)
                """)
            params = {
                "source_type": source_type,
                "user_grouping": user_grouping,
                "model": model,
                "question": question,
                "generated_sql": generated_sql,
                "execution_time": datetime.utcnow(),
                "full_log": full_log_text,
            }
            conn.execute(stmt, params)
            conn.commit()
        return "OK"

__all__ = ["LocalPgConnector"]

