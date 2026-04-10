import os
import json
import asyncio
import io
import re

from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel
from dotenv import load_dotenv

from app.graph import run_graph_with_stream, graph as agent
from app.agent import llm

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    from PyPDF2 import PdfReader

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

class OrgFieldsResponse(BaseModel):
    org_name: str
    mission: str
    goals: str
    budget: str
    timeline: str

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

def _clean_json_response(raw: str) -> str:
    cleaned = (raw or "").strip()
    if "```json" in cleaned:
        return cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
    if "```" in cleaned:
        return cleaned.split("```", 1)[1].split("```", 1)[0].strip()
    return cleaned

def _extract_pdf_text(pdf_bytes: bytes, max_pages: int = 35) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    chunks = []
    for page in reader.pages[:max_pages]:
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        page_text = page_text.strip()
        if page_text:
            chunks.append(page_text)
    text = "\n\n".join(chunks)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def _safe_field(value) -> str:
    if value is None:
        return ""
    return str(value).strip()

def _heuristic_org_name(pdf_text: str) -> str:
    lines = [line.strip() for line in pdf_text.splitlines() if line.strip()]
    prefix_patterns = [
        r"organization name\s*[:\-]\s*(.+)",
        r"nonprofit name\s*[:\-]\s*(.+)",
        r"legal name\s*[:\-]\s*(.+)",
    ]
    for line in lines[:120]:
        lower_line = line.lower()
        for pattern in prefix_patterns:
            match = re.search(pattern, lower_line, flags=re.IGNORECASE)
            if match:
                original_match = re.search(pattern, line, flags=re.IGNORECASE)
                if original_match:
                    return original_match.group(1).strip()
    for line in lines[:40]:
        if len(line.split()) <= 12 and len(line) <= 90 and not re.search(r"\d", line):
            if any(
                re.search(pattern, line, flags=re.IGNORECASE)
                for pattern in [
                    r"\bfoundation\b",
                    r"\bnonprofit\b",
                    r"\binc\.?\b",
                    r"\binitiative\b",
                    r"\bassociation\b",
                    r"\bsociety\b",
                    r"\bcenter\b",
                    r"\bcentre\b",
                    r"\borganization\b",
                ]
            ):
                return line
    return ""

def _extract_labeled_field(lines: list[str], patterns: list[str], lookahead: int = 4) -> str:
    for idx, line in enumerate(lines):
        for pattern in patterns:
            match = re.match(pattern, line, flags=re.IGNORECASE)
            if not match:
                continue
            inline_value = (match.group(1) or "").strip()
            if inline_value:
                return inline_value
            continuation = []
            for nxt in lines[idx + 1 : idx + 1 + lookahead]:
                if re.search(r":\s*$", nxt):
                    break
                if re.match(r"^[A-Z][A-Za-z\s/&\-]{2,40}$", nxt) and len(nxt.split()) <= 6:
                    break
                continuation.append(nxt.strip())
            if continuation:
                return " ".join(continuation).strip()
    return ""

def _heuristic_profile_fields(pdf_text: str) -> dict:
    lines = [line.strip() for line in pdf_text.splitlines() if line.strip()]
    mission = _extract_labeled_field(
        lines,
        [
            r"^\s*mission(?:\s+statement)?\s*[:\-]\s*(.*)$",
            r"^\s*our mission\s*[:\-]\s*(.*)$",
        ],
        lookahead=6,
    )
    goals = _extract_labeled_field(
        lines,
        [
            r"^\s*(?:project\s+)?goals?(?:\s*&\s*objectives?)?\s*[:\-]\s*(.*)$",
            r"^\s*objectives?\s*[:\-]\s*(.*)$",
        ],
        lookahead=6,
    )
    budget = _extract_labeled_field(
        lines,
        [
            r"^\s*(?:project\s+)?budget\s*[:\-]\s*(.*)$",
            r"^\s*funding\s+request\s*[:\-]\s*(.*)$",
            r"^\s*requested\s+amount\s*[:\-]\s*(.*)$",
        ],
    )
    timeline = _extract_labeled_field(
        lines,
        [
            r"^\s*(?:project\s+)?timeline\s*[:\-]\s*(.*)$",
            r"^\s*period\s+of\s+performance\s*[:\-]\s*(.*)$",
            r"^\s*project\s+duration\s*[:\-]\s*(.*)$",
        ],
    )
    return {
        "org_name": _heuristic_org_name(pdf_text),
        "mission": mission,
        "goals": goals,
        "budget": budget,
        "timeline": timeline,
    }

def _extract_org_fields_with_llm(pdf_text: str) -> dict:
    excerpt = (pdf_text or "")[:18000]
    prompt = f"""
You extract organization profile fields from raw PDF text.
Return ONLY a valid JSON object with exactly these keys:
org_name, mission, goals, budget, timeline

Rules:
- If a field is missing, use an empty string.
- Keep mission concise (1-3 sentences).
- Keep goals concise and specific to project intent if available.
- Preserve numbers/currency and date/timeline wording when present.
- Do not invent facts.

PDF TEXT:
{excerpt}
"""
    try:
        raw = llm.invoke(prompt).content
        parsed = json.loads(_clean_json_response(raw))
    except Exception:
        parsed = {}

    fields = {
        "org_name": _safe_field(parsed.get("org_name", "")) if isinstance(parsed, dict) else "",
        "mission": _safe_field(parsed.get("mission", "")) if isinstance(parsed, dict) else "",
        "goals": _safe_field(parsed.get("goals", "")) if isinstance(parsed, dict) else "",
        "budget": _safe_field(parsed.get("budget", "")) if isinstance(parsed, dict) else "",
        "timeline": _safe_field(parsed.get("timeline", "")) if isinstance(parsed, dict) else "",
    }

    fallback_fields = _heuristic_profile_fields(pdf_text)
    for key, fallback_value in fallback_fields.items():
        if not fields[key]:
            fields[key] = _safe_field(fallback_value)

    return fields

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

@app.post("/extract-org-details", response_model=OrgFieldsResponse)
async def extract_org_details(file: UploadFile = File(...)):
    filename = (file.filename or "").strip()
    is_pdf_filename = filename.lower().endswith(".pdf")
    is_pdf_content = (file.content_type or "").lower() in {"application/pdf", "application/x-pdf"}
    if not (is_pdf_filename or is_pdf_content):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(pdf_bytes) > 15 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="PDF is too large. Please upload a file under 15MB.")

    try:
        pdf_text = _extract_pdf_text(pdf_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not parse PDF text from this file.")

    if not pdf_text:
        raise HTTPException(status_code=400, detail="No readable text found in this PDF.")

    extracted = _extract_org_fields_with_llm(pdf_text)
    return OrgFieldsResponse(**extracted)

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
