import os
import json
import io

import PyPDF2
from docx import Document

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

<<<<<<< HEAD
from app.agent import agent

load_dotenv()

app = FastAPI(
    title="AlgoRhythm Agent API",
    description="A content creator agent powered by LangGraph",
=======
# from app.agent import agent
from app.graph import graph as agent

load_dotenv()
uploaded_context = ""

app = FastAPI(
    title="Granted Agent API",
    description="A grant writer agent powered by LangGraph",
>>>>>>> nero/pdf-uploader
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
<<<<<<< HEAD
        result = agent.invoke({
            "messages": [HumanMessage(content=request.message)]
        })
        last_content = result["messages"][-1].content
        return ChatResponse(response=_content_to_text(last_content))
=======
        message_text = request.message
        if uploaded_context:
            message_text = f"[BACKGROUND DOCUMENT INFORMATION]:\n{uploaded_context}\n\n[USER REQUEST]:\n{request.message}"
        result = agent.invoke({"user_input": message_text})

        response_text = result.get("final_summary") or str(result)
        return ChatResponse(response=_content_to_text(response_text))

>>>>>>> nero/pdf-uploader
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Extracts text from an uploaded document to provide context to the agent."""
<<<<<<< HEAD
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

@app.get("/download/{filename}")
async def download_file(filename: str):
    if filename != "Grant_Proposal_Submission.pdf":
        raise HTTPException(status_code=404, detail="File not found")
    
    file_path = os.path.join(os.getcwd(), filename)
    if os.path.exists(file_path):
        return FileResponse(path=file_path, filename=filename, media_type='application/pdf')
    raise HTTPException(status_code=404, detail="File not generated yet.")

=======
    global uploaded_context

    try:
        content = await file.read()
        text = ""
        
        if file.filename.endswith(".pdf"):
            reader = PyPDF2.PdfReader(io.BytesIO(content))
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
        elif file.filename.endswith(".docx"):
            doc = Document(io.BytesIO(content))
            text = "\n".join([p.text for p in doc.paragraphs])
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Please upload PDF or DOCX."
            )
        
        uploaded_context = text.strip() 
        return {"text": uploaded_context}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{filename}")
async def download_file(filename: str):
    if filename != "Grant_Proposal_Submission.pdf":
        raise HTTPException(status_code=404, detail="File not found")
    
    file_path = os.path.join(os.getcwd(), filename)
    if os.path.exists(file_path):
        return FileResponse(path=file_path, filename=filename, media_type='application/pdf')
    raise HTTPException(status_code=404, detail="File not generated yet.")

>>>>>>> nero/pdf-uploader
app.mount("/ui", StaticFiles(directory="static", html=True), name="static")