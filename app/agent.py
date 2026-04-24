### `agent.py`
import os
import json
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional, Dict
import base64
import resend
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

from dotenv import load_dotenv
from email.message import EmailMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent

# Google Calendar API imports
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from datetime import timedelta

from pathlib import Path
from app.pdf_utils import render_proposal_pdf

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

google_api_key = os.getenv("GOOGLE_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

if not google_api_key:
    raise RuntimeError("GOOGLE_API_KEY missing. Check .env")
if not tavily_api_key:
    raise RuntimeError("TAVILY_API_KEY missing. Check .env")

os.environ["GOOGLE_API_KEY"] = google_api_key
os.environ["TAVILY_API_KEY"] = tavily_api_key
print("✓ API keys configured successfully!")

# Guardrail: all proposal emails must go to this fixed recipient.
PROPOSAL_RECIPIENT_EMAIL = "chaner@whitman.edu"

def _is_env_present(name: str) -> bool:
    return bool((os.getenv(name) or "").strip())

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
reviewer_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

proposal_schema =[
    "Executive Summary",
    "Problem Statement / Needs Assessment",
    "Project Goals & Objectives",
    "Methodology / Activities / Timeline",
    "Evaluation Plan & Success Metrics",
    "Team Background",
    "Budget & Funding Request",
    "Conclusion / Call to Action"
]

# Tool 1: Web Search Tool
search_tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    name="web_search",
    time_range="year",
    description="Use this tool to search the web for active, real-world grant opportunities. Input should be a highly targeted search query."
)

# Tool 2: Custom Document Generation & PDF Output Tool
@tool
def generate_grant_and_save_pdf(funder_details: str, project_details: str) -> str:
    """
    Use this tool AFTER you have found a suitable grant from the web search.
    Pass the detailed funder information (Name, Mission, Priorities, Due Date, etc.) as `funder_details`.
    Pass the organization's name, mission, and project details as `project_details`.
    This tool will automatically generate the 8-section proposal using the LLM and save it directly as a PDF.
    """
    print("\n[Tool Executing] Generating Proposal Sections based on found grant and provided details...")
    full_proposal = {}
    
    # Generate each section utilizing the global LLM
    for section in proposal_schema:
        print(f" -> Generating: {section}")
        prompt = f"""
        You are an expert nonprofit grant writer. Write ONLY the section titled: {section}

        =============================
        FUNDER INFORMATION (from search)
        =============================
        {funder_details}

        =============================
        PROJECT INFORMATION
        =============================
        {project_details}

        INSTRUCTIONS:
        - Align strongly with the funder's mission and funding priorities.
        - Use persuasive and impact-driven language.
        - 300-500 words.
        - Do NOT generate other section titles.
        """
        response = llm.invoke(prompt)
        full_proposal[section] = response.content

    print("\n[Tool Executing] Formatting and saving to PDF...")
    date_str = datetime.now().strftime("%B %d, %Y")
    pdf_file_name = "Grant_Proposal_Submission.pdf"
    render_proposal_pdf(
        proposal_sections={str(k): str(v) for k, v in full_proposal.items()},
        output_path=pdf_file_name,
        title="Grant Proposal Submission",
        date_str=date_str,
    )
    
    return f"Success! The grant proposal has been generated and saved locally as {pdf_file_name}."


# Tool 3:Email text generator
class EmailDraftInput(BaseModel):
    org_name: str = Field(..., description="Organization/nonprofit name")
    goal: str = Field(..., description="What the email is trying to achieve")
    recipient_type: Optional[str] = Field("funder", description="Who this email is to (e.g., funder, partner, sponsor)")
    grant_name: Optional[str] = Field(None, description="Grant/funder name if relevant")
    context: Optional[str] = Field(None, description="Any extra context to include (optional)")

def _extract_subject_body_from_draft(draft_text: str) -> Dict[str, str]:
    """Parse draft text in the expected format:
    SUBJECT: ...
    BODY:
    ...
    """
    text = (draft_text or "").strip()
    subject = ""
    body = ""

    if "SUBJECT:" in text:
        subject = text.split("SUBJECT:", 1)[1].split("\n", 1)[0].strip()
    if "BODY:" in text:
        body = text.split("BODY:", 1)[1].strip()

    return {"subject": subject, "body": body}

@tool("generate_email_draft", args_schema=EmailDraftInput)
def generate_email_draft(
    org_name: str,
    goal: str,
    recipient_type: Optional[str] = "funder",
    grant_name: Optional[str] = None,
    context: Optional[str] = None
) -> str:
    """Generates an email draft for outreach."""
    print("\n[Tool Executing] Generating Email...")
    prompt = f"""
Write a professional, friendly outreach email draft.

Output MUST be exactly:
SUBJECT: ...
BODY:
...

Rules:
- 120-180 words
- Plain language
- Clear call-to-action
- Make the email specific to the organization and grant
- Do NOT use placeholders like [Your Name], [Role], [Phone], or [Website]
- End the email with a realistic organizational signature, not a personal placeholder
- Use a closing like:
  Best,
  {org_name} Team

Organization: {org_name}
Recipient type: {recipient_type}
Goal: {goal}
Grant name (optional): {grant_name or "N/A"}
Extra context (optional): {context or "N/A"}
"""
    resp = llm.invoke(prompt).content.strip()

    subject = ""
    body = resp

    if "SUBJECT:" in resp:
        subject = resp.split("SUBJECT:", 1)[1].split("\n", 1)[0].strip()
    if "BODY:" in resp:
        body = resp.split("BODY:", 1)[1].strip()

    print("\n================ EMAIL DRAFT ================")
    print("SUBJECT:", subject)
    print("\nBODY:\n" + body)
    print("============================================\n")

    return f"SUBJECT: {subject}\nBODY:\n{body}"

class SendEmailInput(BaseModel):
    to_email: str
    subject: Optional[str] = None
    body: Optional[str] = None
    draft: Optional[str] = Field(
        None,
        description="Optional full draft from generate_email_draft. If provided, subject/body are extracted from this.",
    )
    attach_proposal_pdf: Optional[bool] = Field(
        True,
        description="Attach the generated grant proposal PDF to the email.",
    )
    pdf_path: Optional[str] = Field(
        "Grant_Proposal_Submission.pdf",
        description="Path to the PDF file to attach when attach_proposal_pdf is true.",
    )

@tool("send_email", args_schema=SendEmailInput)
def send_email(
    to_email: str,
    subject: Optional[str] = None,
    body: Optional[str] = None,
    draft: Optional[str] = None,
    attach_proposal_pdf: Optional[bool] = True,
    pdf_path: Optional[str] = "Grant_Proposal_Submission.pdf",
) -> Dict[str, str]:
    """
    Sends an email using the Resend API.
    The recipient is always enforced by the guardrail email.
    """
    try:
        recipient_email = PROPOSAL_RECIPIENT_EMAIL

        if draft and (not subject or not body):
            parsed = _extract_subject_body_from_draft(draft)
            subject = subject or parsed["subject"]
            body = body or parsed["body"]

        if not subject or not body:
            return {
                "status": "error",
                "message": "Missing subject/body. Provide them directly or pass `draft` from generate_email_draft.",
            }

        resend_api_key = (os.getenv("RESEND_API_KEY") or "").strip()
        sender_email = (os.getenv("RESEND_FROM_EMAIL") or "").strip()

        if not resend_api_key:
            return {
                "status": "error",
                "message": "Missing RESEND_API_KEY in environment.",
            }

        if not sender_email:
            return {
                "status": "error",
                "message": "Missing RESEND_FROM_EMAIL in environment.",
            }

        resend.api_key = resend_api_key

        params = {
            "from": sender_email,
            "to": [recipient_email],
            "subject": subject,
            "text": body,
        }

        attached_file = None

        if attach_proposal_pdf:
            path_to_attach = (pdf_path or "Grant_Proposal_Submission.pdf").strip()

            if not os.path.exists(path_to_attach):
                return {
                    "status": "error",
                    "message": f"Attachment not found: {path_to_attach}",
                }

            with open(path_to_attach, "rb") as f:
                pdf_bytes = f.read()

            params["attachments"] = [
                {
                    "filename": os.path.basename(path_to_attach),
                    "content": base64.b64encode(pdf_bytes).decode("utf-8"),
                }
            ]
            attached_file = path_to_attach

        result = resend.Emails.send(params)

        response = {
            "status": "sent",
            "to": recipient_email,
            "requested_to": to_email,
            "note": "Recipient enforced by guardrail.",
            "provider": "resend",
        }

        if isinstance(result, dict) and result.get("id"):
            response["message_id"] = str(result["id"])

        if attached_file:
            response["attachment"] = attached_file

        return response

    except Exception as e:
        print("EMAIL ERROR:", repr(e))
        return {"status": "error", "message": str(e)}

# Tool 4: Google Calendar API tool and Email
SCOPES = [
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/gmail.send",
]

def _get_calendar_service():
    creds = None

    token_json_env = os.getenv("GOOGLE_OAUTH_TOKEN_JSON")
    client_json_env = os.getenv("GOOGLE_OAUTH_CLIENT_JSON")
    token_path_env = os.getenv("GOOGLE_OAUTH_TOKEN_PATH")
    client_path_env = os.getenv("GOOGLE_OAUTH_CLIENT_PATH")
    token_json_present = _is_env_present("GOOGLE_OAUTH_TOKEN_JSON")
    client_json_present = _is_env_present("GOOGLE_OAUTH_CLIENT_JSON")

    if token_json_env:
        try:
            creds = Credentials.from_authorized_user_info(json.loads(token_json_env), SCOPES)
        except Exception as e:
            raise RuntimeError(f"Invalid GOOGLE_OAUTH_TOKEN_JSON format: {str(e)}")

    if not creds and token_path_env and os.path.exists(token_path_env):
        creds = Credentials.from_authorized_user_file(token_path_env, SCOPES)

    if not creds and os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = None

            if client_json_env:
                try:
                    flow = InstalledAppFlow.from_client_config(json.loads(client_json_env), SCOPES)
                except Exception as e:
                    raise RuntimeError(f"Invalid GOOGLE_OAUTH_CLIENT_JSON format: {str(e)}")
            elif client_path_env and os.path.exists(client_path_env):
                flow = InstalledAppFlow.from_client_secrets_file(client_path_env, SCOPES)
            elif os.path.exists("credentials.json"):
                flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)

            if not flow:
                raise RuntimeError(
                    "Google Calendar OAuth credentials not configured."
                )

            creds = flow.run_local_server(port=0)

        token_save_path = token_path_env or "token.json"
        with open(token_save_path, "w") as token:
            token.write(creds.to_json())
    return build("calendar", "v3", credentials=creds)


@tool
def create_grant_deadline_event(
    deadline_date: str,
    title: str,
    application_url: str = "",
    timezone: str = "America/Los_Angeles",
    description: Optional[str] = None,
) -> str:
    """Create an all-day Google Calendar event for a grant deadline."""
    print("\n[Tool Executing] Creating calendar event for grant.")
    try:
        service = _get_calendar_service()
        start_date = deadline_date
        end_date = (datetime.fromisoformat(deadline_date) + timedelta(days=1)).date().isoformat()
        event_description = (description or "").strip()
        if not event_description:
            event_description = f"Grant deadline.\n\nApply: {application_url}" if application_url else "Grant deadline."

        event = {
            "summary": title,
            "description": event_description,
            "start": {"date": start_date, "timeZone": timezone},
            "end": {"date": end_date, "timeZone": timezone},
        }

        created = service.events().insert(calendarId="primary", body=event).execute()
        return f"Created event: {created.get('htmlLink', '(no link returned)')}"
    except Exception as e:
        return f"Calendar event was not created: {str(e)}"
