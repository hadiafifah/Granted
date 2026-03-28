import os
import json

from fastapi import FastAPI, HTTPException

from fastapi.responses import FileResponse

from pydantic import BaseModel

from langchain_core.messages import HumanMessage

from dotenv import load_dotenv

from app.agent import agent


load_dotenv()


app = FastAPI(

    title="AlgoRhythm Agent API",

    description="A content creator agent powered by LangGraph",

    version="1.0.0"

)


from fastapi.middleware.cors import CORSMiddleware


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


@app.post("/chat", response_model=ChatResponse)

async def chat(request: ChatRequest):

    try:

        result = agent.invoke({

            "messages": [HumanMessage(content=request.message)]

        })

        last_content = result["messages"][-1].content
        return ChatResponse(response=_content_to_text(last_content))

    except Exception as e:

        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Endpoint to download the generated grant proposal PDF"""
    # Simple security check to prevent directory traversal
    if filename != "Grant_Proposal_Submission.pdf":
        raise HTTPException(status_code=404, detail="File not found")
    
    file_path = os.path.join(os.getcwd(), filename)
    
    if os.path.exists(file_path):
        return FileResponse(
            path=file_path, 
            filename=filename, 
            media_type='application/pdf'
        )
    raise HTTPException(status_code=404, detail="File not generated yet.")

from fastapi.staticfiles import StaticFiles

app.mount("/ui", StaticFiles(directory="static", html=True), name="static")
