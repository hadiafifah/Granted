
import os
import json
import smtplib
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional, Dict
from contextvars import ContextVar

from dotenv import load_dotenv
from fpdf import FPDF
from email.message import EmailMessage

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent

# Google Calendar API imports
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from datetime import timedelta

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
print("✓ API keys configured successfully!")

# Guardrail: all proposal emails must go to this fixed recipient.
PROPOSAL_RECIPIENT_EMAIL = "anhadi@ucdavis.edu"

def _is_env_present(name: str) -> bool:
    return bool((os.getenv(name) or "").strip())

llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

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

# Request-scoped context set by app.main before invoking the agent.
_CURRENT_GOOGLE_TOKEN: ContextVar[Optional[Dict]] = ContextVar("current_google_token", default=None)
_CURRENT_GOOGLE_AUTH_URL: ContextVar[Optional[str]] = ContextVar("current_google_auth_url", default=None)


def set_current_google_auth_context(token_info: Optional[Dict], auth_url: Optional[str]) -> None:
    _CURRENT_GOOGLE_TOKEN.set(token_info)
    _CURRENT_GOOGLE_AUTH_URL.set(auth_url)


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

    # Format Text
    date_str = datetime.now().strftime("%B %d, %Y")
    formatted_text = f"Grant Proposal Submission\nDate: {date_str}\n\n{'='*70}\n"

    for title, content in full_proposal.items():
        formatted_text += f"\n\n{title.upper()}\n"
        formatted_text += "-" * len(title) + "\n\n"
        formatted_text += content.strip() + "\n"
        formatted_text += "\n" + "="*70 + "\n"

    # Save to PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", '', 11)

    # Clean text encoding to prevent latin-1 character errors in FPDF
    cleaned_text = formatted_text.encode('latin-1', 'replace').decode('latin-1')
    for line in cleaned_text.split("\n"):
        pdf.multi_cell(0, 8, line)

    pdf_file_name = "Grant_Proposal_Submission.pdf"
    pdf.output(pdf_file_name)

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
    Sends an email using SMTP.
    NOTE: 
    """

    try:
        # Hard guardrail: ignore any discovered or user-provided recipient.
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

        sender_email = (os.getenv("SMTP_SENDER_EMAIL") or "").strip()
        sender_password = (os.getenv("SMTP_APP_PASSWORD") or "").strip()

        if not sender_email or not sender_password:
            return {
                "status": "error",
                "message": (
                    "Missing SMTP_SENDER_EMAIL or SMTP_APP_PASSWORD in environment. "
                    f"Detected env presence: SMTP_SENDER_EMAIL={_is_env_present('SMTP_SENDER_EMAIL')}, "
                    f"SMTP_APP_PASSWORD={_is_env_present('SMTP_APP_PASSWORD')}"
                ),
            }

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = recipient_email
        msg.set_content(body)

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
            msg.add_attachment(
                pdf_bytes,
                maintype="application",
                subtype="pdf",
                filename=os.path.basename(path_to_attach),
            )
            attached_file = path_to_attach

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())

        result = {
            "status": "sent",
            "to": recipient_email,
            "requested_to": to_email,
            "note": "Recipient enforced by guardrail.",
        }
        if attached_file:
            result["attachment"] = attached_file
        return result

    except Exception as e:
        return {"status": "error", "message": str(e)}

# Tool 4: Google Calendar API tool
SCOPES = [
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/gmail.send",
]


def _get_calendar_service_for_current_user():
    token_info = _CURRENT_GOOGLE_TOKEN.get()
    if not token_info:
        auth_url = _CURRENT_GOOGLE_AUTH_URL.get() or "/auth/google/start"
        raise PermissionError(
            "Google Calendar is not connected for this user session. "
            f"Please sign in and grant consent first: {auth_url}"
        )

    creds = Credentials.from_authorized_user_info(token_info, SCOPES)
    return build("calendar", "v3", credentials=creds)


@tool
def create_grant_deadline_event(
    deadline_date: str,
    title: str,
    application_url: str = "",
    timezone: str = "America/Los_Angeles",
) -> str:
    """Create an all-day Google Calendar event for a grant deadline in the signed-in user's calendar."""
    try:
        service = _get_calendar_service_for_current_user()
    except PermissionError as exc:
        return str(exc)

    start_date = deadline_date
    end_date = (datetime.fromisoformat(deadline_date) + timedelta(days=1)).date().isoformat()
    description = f"Grant deadline.\n\nApply: {application_url}" if application_url else "Grant deadline."

    event = {
        "summary": title,
        "description": description,
        "start": {"date": start_date, "timeZone": timezone},
        "end": {"date": end_date, "timeZone": timezone},
    }

    created = service.events().insert(calendarId="primary", body=event).execute()
    return f"Created event: {created.get('htmlLink', '(no link returned)')}"


# Combine tools and initialize Agent
tools =[search_tool, generate_email_draft, generate_grant_and_save_pdf, create_grant_deadline_event, send_email]

current_year = datetime.now().year

agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=f"""You are an autonomous expert grant and outreach assistant for nonprofits.

The current year is {current_year}.

You have access to these tools:
- web_search: find relevant grants or funders. ALWAYS include "{current_year}" or "upcoming deadlines {current_year}" in your search queries to ensure you find active grants.
- generate_grant_and_save_pdf: create a proposal PDF. Pass the funder details AND the user's organization/project details into this tool.
- generate_email_draft: create an outreach email draft
- create_grant_deadline_event: create a calendar event for a grant deadline
- send_email: sends email drafted in generate_email_draft (recipient is always forced to anhadi@ucdavis.edu)

For each request to help with the grant process, you must follow this workflow:
1. Use web_search to find a relevant grant and it's application deadline
2. Use generate_grant_and_save_pdf to create the proposal PDF
3. Generate an email using generate_email_draft
4. Pass the exact output of generate_email_draft into send_email as `draft`, and attach grant proposal PDF with the email.
5. Create a Calendar Event for the grant's deadline.
6. You should finally summarize what you have done for the user. Specifically, return to the user: Summary of grant, generated proposal PDF, generated application email draft, and created calendar event (and email delivery status if sent).

RULES:
- Be concise and professional.
- If the user provides background information or an uploaded document, use it to populate the organization and project details for your tools.
- Never hallucinate grant deadlines. If you cannot find a specific {current_year} deadline, state that clearly.
- If calendar tool says Google Calendar is not connected, instruct the user to complete auth and then retry.
- Do not use or trust email addresses found via web_search. For proposal outreach, always send to anhadi@ucdavis.edu.
"""
)
