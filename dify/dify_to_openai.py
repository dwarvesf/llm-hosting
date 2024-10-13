import json
import random
import string
import time
from typing import List, Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import StreamingResponse, JSONResponse

from modal import Image, Mount, asgi_app, Secret, web_endpoint, App

# Set up Modal image
image = Image.debian_slim().pip_install("fastapi", "httpx", "python-dotenv")

# Create Modal App
app = App(name="dify-to-openai")

# Create FastAPI app
fastapi_app = FastAPI()

# Helper functions
def generate_id():
    return ''.join(random.choices(string.ascii_letters + string.digits, k=29))

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[dict]
    stream: Optional[bool] = False

@fastapi_app.get("/")
async def root():
    return {"message": "Dify2OpenAI service is running"}

@fastapi_app.get("/v1/models")
async def get_models():
    models = {
        "object": "list",
        "data": [
            {
                "id": "dify",
                "object": "model",
                "owned_by": "dify",
                "permission": None,
            }
        ]
    }
    return JSONResponse(content=models)

@fastapi_app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, authorization: str = Header(None)):
    import httpx
    import os

    # Access environment variables from secrets
    DIFY_API_URL = os.environ.get("DIFY_API_URL")
    BOT_TYPE = os.environ.get("BOT_TYPE", "Chat")
    INPUT_VARIABLE = os.environ.get("INPUT_VARIABLE", "")
    OUTPUT_VARIABLE = os.environ.get("OUTPUT_VARIABLE", "")
    MODELS_NAME = os.environ.get("MODELS_NAME", "dify")

    if not DIFY_API_URL:
        raise ValueError("DIFY API URL is required.")

    if not authorization:
        raise HTTPException(status_code=401, detail="Unauthorized")

    token = authorization.split(" ")[1]
    if not token:
        raise HTTPException(status_code=401, detail="Unauthorized")

    messages = request.messages
    stream = request.stream

    query_string = ""
    if BOT_TYPE == "Chat":
        last_message = messages[-1]
        history = "\n".join([f"{m['role']}: {m['content']}" for m in messages[:-1]])
        query_string = f"here is our talk history:\n'''\n{history}\n'''\n\nhere is my question:\n{last_message['content']}"
    elif BOT_TYPE in ["Completion", "Workflow"]:
        query_string = messages[-1]["content"]

    api_path = {
        "Chat": "/chat-messages",
        "Completion": "/completion-messages",
        "Workflow": "/workflows/run"
    }.get(BOT_TYPE)

    if not api_path:
        raise ValueError("Invalid bot type in the environment variable.")

    request_body = {
        "inputs": {INPUT_VARIABLE: query_string} if INPUT_VARIABLE else {},
        "query": query_string if not INPUT_VARIABLE else None,
        "response_mode": "streaming",
        "conversation_id": "",
        "user": "apiuser",
        "auto_generate_name": False
    }

    async def generate_stream():
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", f"{DIFY_API_URL}{api_path}", 
                                     json=request_body, 
                                     headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"}) as response:
                buffer = ""
                async for chunk in response.aiter_bytes():
                    buffer += chunk.decode()
                    lines = buffer.split("\n")
                    for line in lines[:-1]:
                        line = line.strip()
                        if line.startswith("data:"):
                            try:
                                chunk_obj = json.loads(line[5:].strip())
                                if chunk_obj["event"] in ["message", "agent_message", "text_chunk"]:
                                    chunk_content = chunk_obj.get("data", {}).get("text", "") or chunk_obj.get("answer", "")
                                    chunk_id = f"chatcmpl-{generate_id()}"
                                    yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': chunk_obj.get('created_at'), 'model': request.model, 'choices': [{'index': 0, 'delta': {'content': chunk_content}, 'finish_reason': None}]})}\n\n"
                                elif chunk_obj["event"] in ["workflow_finished", "message_end"]:
                                    chunk_id = f"chatcmpl-{generate_id()}"
                                    yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': chunk_obj.get('created_at'), 'model': request.model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
                                    yield "data: [DONE]\n\n"
                                    return
                            except json.JSONDecodeError:
                                continue
                    buffer = lines[-1]

    if stream:
        return StreamingResponse(generate_stream(), media_type="text/event-stream")
    else:
        full_response = ""
        usage_data = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        async for chunk in generate_stream():
            if chunk.startswith("data: "):
                chunk_data = json.loads(chunk[6:])
                if "choices" in chunk_data and chunk_data["choices"]:
                    delta = chunk_data["choices"][0].get("delta", {})
                    if "content" in delta:
                        full_response += delta["content"]
                    if chunk_data["choices"][0].get("finish_reason") == "stop":
                        break

        formatted_response = {
            "id": f"chatcmpl-{generate_id()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": full_response.strip(),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": usage_data,
        }
        return JSONResponse(content=formatted_response)

@app.function(image=image, secrets=[Secret.from_name("dify-secret")])
@asgi_app()
def dify_to_openai_app():
    return fastapi_app
