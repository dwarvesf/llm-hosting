from pydantic import BaseModel
from fastapi.responses import JSONResponse
from modal import Image, App, web_endpoint
import json
from datetime import date, datetime

image = Image.debian_slim().pip_install("duckdb==1.1.0")
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

@app.function(image=image, concurrency_limit=5)
@web_endpoint(method="POST")
def query_duckdb(query: Query):
  try:
    result = preload_and_query_duckdb.remote(query.sql)

    json_compatible_result = json.loads(
        json.dumps(result, default=json_serial)
    )

    return JSONResponse(
        content={"result": json_compatible_result},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type"
        },
    )
  except Exception as e:
    return JSONResponse(content={"error": str(e)}, status_code=400)

@web_endpoint(method="OPTIONS")
def handle_options():
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type"
        },
    )
