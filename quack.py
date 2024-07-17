from pydantic import BaseModel
from fastapi.responses import JSONResponse
from modal import Image, App, web_endpoint
import json
from datetime import date, datetime

image = Image.debian_slim().pip_install("duckdb==1.0.0")
app = App(name="quack")

class Query(BaseModel):
  sql: str
  
def json_serial(obj):
  if isinstance(obj, (datetime, date)):
    return obj.isoformat()
  raise TypeError(f"Type {type(obj)} not serializable")

@app.function(image=image, concurrency_limit=5)
@web_endpoint(method="POST")
def query_duckdb(query: Query):
  import duckdb
  
  try:
    conn = duckdb.connect()
    result = conn.execute(query.sql).fetchall()
    column_names = [desc[0] for desc in conn.description]
    formatted_result = [dict(zip(column_names, row)) for row in result]
    conn.close()

    json_compatible_result = json.loads(
      json.dumps(formatted_result, default=json_serial)
    )

    return JSONResponse(content={"result": json_compatible_result})
  except Exception as e:
    return JSONResponse(content={"error": str(e)}, status_code=400)
