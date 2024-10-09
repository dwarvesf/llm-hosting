from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json
from datetime import date, datetime
from modal import Image, App, asgi_app

image = Image.debian_slim().pip_install("duckdb==1.1.0", "fastapi")
app = App(name="quack")

class Query(BaseModel):
    sql: str

def json_serial(obj):
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

@app.function(image=image, keep_warm=1)
def preload_and_query_duckdb(query: str):
    import duckdb
    conn = duckdb.connect('vault.duckdb')
    
    # Check if the 'vault' table exists
    table_exists = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='vault'").fetchone() is not None
    
    # Preload the data only if the 'vault' table doesn't exist
    if not table_exists:
        conn.execute("INSTALL httpfs")
        conn.execute("LOAD httpfs")
        conn.execute("IMPORT DATABASE 'https://memo.d.foundation/db'")
        conn.execute("INSTALL fts")
        conn.execute("LOAD fts")
        conn.execute("PRAGMA create_fts_index('vault', 'file_path', 'title', 'md_content', 'tags', 'authors')")
        conn.execute("INSTALL vss")
        conn.execute("LOAD vss")
        conn.execute("SET hnsw_enable_experimental_persistence = true")
        conn.execute("CREATE INDEX emb_openai_hnsw_index ON vault USING HNSW (embeddings_openai)")
        conn.execute("CREATE INDEX emb_spr_custom_hnsw_index ON vault USING HNSW (embeddings_spr_custom)")
    
    try:
        result = conn.execute(query).fetchall()
        column_names = [desc[0] for desc in conn.description]
        formatted_result = [dict(zip(column_names, row)) for row in result]
        return formatted_result
    finally:
        conn.close()

# Create FastAPI app
web_app = FastAPI(
    title="Quack DuckDB Query API",
    version="1.0",
    description="API for querying DuckDB database.",
)

# Add CORS middleware
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@web_app.post("/")
async def query_duckdb(query: Query):
    try:
        result = preload_and_query_duckdb.remote(query.sql)
        json_compatible_result = json.loads(
            json.dumps(result, default=json_serial)
        )
        return {"result": json_compatible_result}
    except Exception as e:
        return {"error": str(e)}

@app.function(image=image)
@asgi_app()
def serve():
    return web_app
