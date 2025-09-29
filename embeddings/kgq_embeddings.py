import os
import asyncio
from pathlib import Path

import asyncpg
import numpy as np
import pandas as pd
from pgvector.asyncpg import register_vector
from google.cloud.sql.connector import Connector
from google.cloud import bigquery
from agents import EmbedderAgent
from sqlalchemy.sql import text
from utilities import (
    PROJECT_ID,
    PG_INSTANCE,
    PG_DATABASE,
    PG_USER,
    PG_PASSWORD,
    PG_REGION,
    BQ_OPENDATAQNA_DATASET_NAME,
    BQ_REGION,
    EMBEDDING_MODEL,
    EMBEDDING_MODEL_PATH,
    KNOWN_GOOD_SQL_PATH,
    root_dir,
)


if EMBEDDING_MODEL == "local":
    embedder = EmbedderAgent("local", EMBEDDING_MODEL_PATH)
else:
    embedder = EmbedderAgent(EMBEDDING_MODEL)


async def setup_kgq_table( project_id,
                            instance_name,
                            database_name,
                            schema,
                            database_user,
                            database_password,
                            region,
                            VECTOR_STORE = "cloudsql-pgvector"):
    """ 
    This function sets up or refreshes the Vector Store for Known Good Queries (KGQ)
    """
    if VECTOR_STORE=='bigquery-vector':

        # Create BQ Client
        client=bigquery.Client(project=project_id)

        # Delete an old table
        # client.query_and_wait(f'''DROP TABLE IF EXISTS `{project_id}.{schema}.example_prompt_sql_embeddings`''')
        # Create a new emptry table
        client.query_and_wait(f'''CREATE TABLE IF NOT EXISTS `{project_id}.{schema}.example_prompt_sql_embeddings` (
                              user_grouping string NOT NULL, example_user_question string NOT NULL, example_generated_sql string NOT NULL,
                              embedding ARRAY<FLOAT64>)''')
        

    elif VECTOR_STORE=='cloudsql-pgvector':

        loop = asyncio.get_running_loop()
        async with Connector(loop=loop) as connector:
            # Create connection to Cloud SQL database
            conn: asyncpg.Connection = await connector.connect_async(
                f"{project_id}:{region}:{instance_name}",  # Cloud SQL instance connection name
                "asyncpg",
                user=f"{database_user}",
                password=f"{database_password}",
                db=f"{database_name}",
            )

            # Drop on old table
            # await conn.execute("DROP TABLE IF EXISTS example_prompt_sql_embeddings")
            # Create a new emptry table
            await conn.execute(
            """CREATE TABLE IF NOT EXISTS example_prompt_sql_embeddings(
                                user_grouping VARCHAR(1024) NOT NULL,
                                example_user_question text NOT NULL,
                                example_generated_sql text NOT NULL,
                                embedding vector(768))"""
            )

    else: raise ValueError("Not a valid parameter for a vector store.")

async def store_kgq_embeddings(df_kgq, 
                            project_id,
                            instance_name,
                            database_name,
                            schema,
                            database_user,
                            database_password,
                            region,
                            VECTOR_STORE = "cloudsql-pgvector"
                            ):
    """ 
    Create and save the Known Good Query Embeddings to Vector Store  
    """
    if VECTOR_STORE=='bigquery-vector':

        client=bigquery.Client(project=project_id)
        
        example_sql_details_chunked = []

        for _, row_aug in df_kgq.iterrows():

            example_user_question = str(row_aug['prompt'])
            example_generated_sql = str(row_aug['sql'])
            example_grouping = str(row_aug['user_grouping'])
            emb =  embedder.create(example_user_question)
            

            r = {"example_grouping":example_grouping,"example_user_question": example_user_question,"example_generated_sql": example_generated_sql,"embedding": emb}
            example_sql_details_chunked.append(r)

        example_prompt_sql_embeddings = pd.DataFrame(example_sql_details_chunked)

        client.query_and_wait(f'''CREATE TABLE IF NOT EXISTS `{project_id}.{schema}.example_prompt_sql_embeddings` (
            user_grouping string NOT NULL, example_user_question string NOT NULL, example_generated_sql string NOT NULL,
            embedding ARRAY<FLOAT64>)''')

        for _, row in example_prompt_sql_embeddings.iterrows():
                client.query_and_wait(f'''DELETE FROM `{project_id}.{schema}.example_prompt_sql_embeddings`
                            WHERE user_grouping= '{row["example_grouping"]}' and example_user_question= "{row["example_user_question"]}" '''
                                )
                    # embedding=np.array(row["embedding"])
                cleaned_sql = row["example_generated_sql"].replace("\r", " ").replace("\n", " ")
                client.query_and_wait(f'''INSERT INTO `{project_id}.{schema}.example_prompt_sql_embeddings` 
                    VALUES ("{row["example_grouping"]}","{row["example_user_question"]}" , 
                    "{cleaned_sql}",{row["embedding"]} )''')
                    
        


    elif VECTOR_STORE=='cloudsql-pgvector':

        loop = asyncio.get_running_loop()
        async with Connector(loop=loop) as connector:
            # Create connection to Cloud SQL database
            conn: asyncpg.Connection = await connector.connect_async(
                f"{project_id}:{region}:{instance_name}",  # Cloud SQL instance connection name
                "asyncpg",
                user=f"{database_user}",
                password=f"{database_password}",
                db=f"{database_name}",
            )


            example_sql_details_chunked = []
            
            for _, row_aug in df_kgq.iterrows():

                example_user_question =  str(row_aug['prompt'])
                example_generated_sql = str(row_aug['sql'])
                example_grouping = str(row_aug['user_grouping'])

                emb =  embedder.create(example_user_question)

                r = {"example_grouping":example_grouping,"example_user_question": example_user_question,"example_generated_sql": example_generated_sql,"embedding": emb}
                example_sql_details_chunked.append(r)

            example_prompt_sql_embeddings = pd.DataFrame(example_sql_details_chunked)
            
            for _, row in example_prompt_sql_embeddings.iterrows():
                await conn.execute(
                        "DELETE FROM example_prompt_sql_embeddings WHERE user_grouping= $1 and example_user_question=$2",
                        row["example_grouping"],
                        row["example_user_question"])
                cleaned_sql = row["example_generated_sql"].replace("\r", " ").replace("\n", " ")
                await conn.execute(
                    "INSERT INTO example_prompt_sql_embeddings (user_grouping, example_user_question, example_generated_sql, embedding) VALUES ($1, $2, $3, $4)",
                    row["example_grouping"],
                    row["example_user_question"],
                    cleaned_sql,
                    str(row["embedding"]),
                )

        await conn.close()

    else: raise ValueError("Not a valid parameter for a vector store.")


def _resolve_known_good_sql_path(path: str | os.PathLike[str] | None = None) -> Path:
    """Return the configured path to the known good SQL CSV file."""

    configured = Path(path or KNOWN_GOOD_SQL_PATH)
    if not configured.is_absolute():
        configured = Path(root_dir) / configured

    if not configured.exists():
        raise FileNotFoundError(
            "known_good_sql.csv file not found. "
            "Place your cache file at "
            f"{configured} or update LOCAL.KNOWN_GOOD_SQL_PATH in config.ini."
        )

    return configured


def load_kgq_df(path: str | os.PathLike[str] | None = None) -> pd.DataFrame:
    """Load and normalise the known good SQL cache CSV."""

    csv_path = _resolve_known_good_sql_path(path)
    df_kgq = pd.read_csv(csv_path)
    df_kgq = df_kgq.loc[:, ["prompt", "sql", "user_grouping"]]
    df_kgq = df_kgq.dropna()

    return df_kgq



if __name__ == '__main__': 
    from utilities import PROJECT_ID, PG_INSTANCE, PG_DATABASE, PG_USER, PG_PASSWORD, PG_REGION
    VECTOR_STORE = "cloudsql-pgvector"
    
    csv_path = _resolve_known_good_sql_path()

    print("Known Good SQL Found at Path :: " + str(csv_path))

    # Load the file
    df_kgq = pd.read_csv(csv_path)
    df_kgq = df_kgq.loc[:, ["prompt", "sql", "database_name"]]
    df_kgq = df_kgq.dropna()

    print('Known Good SQLs Loaded into a Dataframe')

    asyncio.run(setup_kgq_table(PROJECT_ID,
                            PG_INSTANCE,
                            PG_DATABASE,
                            PG_USER,
                            PG_PASSWORD,
                            PG_REGION,
                            VECTOR_STORE))

    asyncio.run(store_kgq_embeddings(df_kgq,
                            PROJECT_ID,
                            PG_INSTANCE,
                            PG_DATABASE,
                            PG_USER,
                            PG_PASSWORD,
                            PG_REGION,
                            VECTOR_STORE))
