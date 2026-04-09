import os
import json
import asyncio

from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel
from dotenv import load_dotenv

from app.graph import run_graph_with_stream, graph as agent

load_dotenv()

app = FastAPI(
    title="Granted Agent API",
    description="A grant writer agent powered by LangGraph",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class FormRequest(BaseModel):
    org_name: str
    mission: str
    goals: str
    budget: str
    timeline: str

class ChatResponse(BaseModel):
    response: str

def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif "text" in item and isinstance(item["text"], str):
                    parts.append(item["text"])
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p).strip()
    return str(content)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/submit", response_model=ChatResponse)
async def submit_proposal(request: FormRequest):
    """Ingests form details and runs the LangGraph grant process (non-streaming)."""
    try:
        form_json = json.dumps(request.model_dump())
        result = agent.invoke({"user_input": form_json})
        response_text = result.get("final_summary") or str(result)
        return ChatResponse(response=_content_to_text(response_text))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/submit-stream")
async def submit_proposal_stream(request: FormRequest):
    """Streams real-time progress events via SSE as the agent runs."""
    form_json = json.dumps(request.model_dump())

    async def event_generator() -> AsyncGenerator[str, None]:
        queue: asyncio.Queue = asyncio.Queue()

        async def run_agent():
            try:
                await run_graph_with_stream({"user_input": form_json}, queue)
            except Exception as e:
                await queue.put({"type": "error", "message": str(e)})
            finally:
                await queue.put(None)  # sentinel to signal completion

        task = asyncio.create_task(run_agent())

        while True:
            item = await queue.get()
            if item is None:
                break
            yield f"data: {json.dumps(item)}\n\n"

        await task

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

@app.get("/download/{filename}")
async def download_file(filename: str):
    if filename != "Grant_Proposal_Submission.pdf":
        raise HTTPException(status_code=404, detail="File not found")
    file_path = os.path.join(os.getcwd(), filename)
    if os.path.exists(file_path):
        return FileResponse(path=file_path, filename=filename, media_type='application/pdf')
    raise HTTPException(status_code=404, detail="File not generated yet.")

@app.get("/view/{filename}")
async def view_file(filename: str):
    if filename != "Grant_Proposal_Submission.pdf":
        raise HTTPException(status_code=404, detail="File not found")
    file_path = os.path.join(os.getcwd(), filename)
    if os.path.exists(file_path):
        return FileResponse(
            path=file_path,
            media_type='application/pdf',
            headers={"Content-Disposition": "inline; filename=Grant_Proposal_Submission.pdf"}
        )
    raise HTTPException(status_code=404, detail="File not generated yet.")

app.mount("/ui", StaticFiles(directory="static", html=True), name="static")