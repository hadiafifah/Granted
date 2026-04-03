import os
import json
import io
import uuid
from urllib.parse import urlencode
from typing import Optional

import PyPDF2
from docx import Document

from fastapi import FastAPI, HTTPException, UploadFile, File, Request, Response
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.google_user_utils import get_connected_user_email

from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request as GoogleAuthRequest

from app.agent import agent, SCOPES, set_current_google_auth_context

load_dotenv()

app = FastAPI(
    title="AlgoRhythm Agent API",
    description="A content creator agent powered by LangGraph",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SESSION_COOKIE_NAME = "granted_sid"
SESSION_TOKENS = {}
SESSION_OAUTH_STATES = {}

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts =[]
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

def _client_secrets_path() -> str:
    return os.getenv("GOOGLE_OAUTH_CLIENT_PATH", "credentials.json")

def _external_base_url(request: Request) -> str:
    proto = request.headers.get("x-forwarded-proto", request.url.scheme).split(",")[0].strip()
    host = request.headers.get("x-forwarded-host", request.headers.get("host", request.url.netloc)).split(",")[0].strip()
    return f"{proto}://{host}"

def _redirect_uri(request: Request) -> str:
    configured = os.getenv("GOOGLE_OAUTH_REDIRECT_URI")
    if configured:
        return configured
    return f"{_external_base_url(request)}/auth/google/callback"

def _ensure_session_id(request: Request, response: Optional[Response] = None) -> str:
    sid = request.cookies.get(SESSION_COOKIE_NAME)
    if not sid:
        sid = str(uuid.uuid4())
    if response is not None:
        response.set_cookie(SESSION_COOKIE_NAME, sid, httponly=True, samesite="lax")
    return sid

def _refresh_token_if_needed(token_info: Optional[dict]) -> Optional[dict]:
    if not token_info:
        return None
    creds = Credentials.from_authorized_user_info(token_info, SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(GoogleAuthRequest())
        return json.loads(creds.to_json())
    return token_info

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: Request, payload: ChatRequest, response: Response):
    try:
        sid = _ensure_session_id(request, response)
        token_info = _refresh_token_if_needed(SESSION_TOKENS.get(sid))
        if token_info:
            SESSION_TOKENS[sid] = token_info
        auth_qs = urlencode({"next": "/ui"})
        auth_url = f"{_external_base_url(request)}/auth/google/start?{auth_qs}"
        set_current_google_auth_context(token_info=token_info, auth_url=auth_url)

        result = agent.invoke({
            "messages": [HumanMessage(content=payload.message)]
        })
        last_content = result["messages"][-1].content
        return ChatResponse(response=_content_to_text(last_content))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        set_current_google_auth_context(token_info=None, auth_url=None)

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Extracts text from an uploaded document to provide context to the agent."""
    try:
        content = await file.read()
        text = ""
        
        if file.filename.endswith(".pdf"):
            reader = PyPDF2.PdfReader(io.BytesIO(content))
            for page in reader.pages:
                text += page.extract_text() + "\n"
        elif file.filename.endswith(".docx"):
            doc = Document(io.BytesIO(content))
            text = "\n".join([p.text for p in doc.paragraphs])
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload PDF or DOCX.")
        
        return {"text": text.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/auth/google/start")
async def start_google_auth(request: Request, next: str = "/ui"):
    if not os.path.exists(_client_secrets_path()):
        raise HTTPException(status_code=500, detail="Missing Google OAuth client file (credentials.json).")

    response = RedirectResponse(url="/ui", status_code=302)
    sid = _ensure_session_id(request, response)
    flow = Flow.from_client_secrets_file(_client_secrets_path(), scopes=SCOPES)
    flow.redirect_uri = _redirect_uri(request)
    auth_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    SESSION_OAUTH_STATES[sid] = {"state": state, "next": next if next.startswith("/") else "/ui"}
    response.headers["Location"] = auth_url
    return response

@app.get("/auth/google/callback")
async def google_auth_callback(request: Request, state: str):
    sid = _ensure_session_id(request)
    expected = SESSION_OAUTH_STATES.get(sid, {})
    if not expected or expected.get("state") != state:
        raise HTTPException(status_code=400, detail="Invalid or expired OAuth state.")

    flow = Flow.from_client_secrets_file(_client_secrets_path(), scopes=SCOPES, state=state)
    flow.redirect_uri = _redirect_uri(request)
    flow.fetch_token(authorization_response=str(request.url))

    creds = flow.credentials
    SESSION_TOKENS[sid] = json.loads(creds.to_json())
    next_path = expected.get("next", "/ui")
    SESSION_OAUTH_STATES.pop(sid, None)

    response = RedirectResponse(url=next_path, status_code=302)
    response.set_cookie(SESSION_COOKIE_NAME, sid, httponly=True, samesite="lax")
    return response

@app.get("/auth/google/status")
async def google_auth_status(request: Request, response: Response):
    sid = _ensure_session_id(request, response)
    connected = sid in SESSION_TOKENS and SESSION_TOKENS[sid] is not None
    return {"connected": connected}

@app.post("/auth/google/disconnect")
async def google_auth_disconnect(request: Request, response: Response):
    sid = _ensure_session_id(request, response)
    SESSION_TOKENS.pop(sid, None)
    SESSION_OAUTH_STATES.pop(sid, None)
    return {"connected": False}

@app.get("/download/{filename}")
async def download_file(filename: str):
    if filename != "Grant_Proposal_Submission.pdf":
        raise HTTPException(status_code=404, detail="File not found")
    
    file_path = os.path.join(os.getcwd(), filename)
    if os.path.exists(file_path):
        return FileResponse(path=file_path, filename=filename, media_type='application/pdf')
    raise HTTPException(status_code=404, detail="File not generated yet.")

app.mount("/ui", StaticFiles(directory="static", html=True), name="static")
