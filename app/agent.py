import os
import json
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional, Dict

from dotenv import load_dotenv
from fpdf import FPDF

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

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
print("✓ API keys configured successfully!")

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

@tool("generate_email_draft", args_schema=EmailDraftInput)
def generate_email_draft(
    org_name: str,
    goal: str,
    recipient_type: Optional[str] = "funder",
    grant_name: Optional[str] = None,
    context: Optional[str] = None
) -> str:
    """Generates an email draft for outreach."""
    prompt = f"""
Write a professional, friendly outreach email draft.

Output MUST be exactly:
SUBJECT: ...
BODY:
...

Rules:
- 120-–180 words
- Plain language
- Clear call-to-action
- Use placeholders like [Your Name], [Role], [Phone], [Website]

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

    return f"**SUBJECT:** {subject}\n\n**BODY:**\n{body}"


# Tool 4: Google Calendar API tool
SCOPES =["https://www.googleapis.com/auth/calendar.events"]

def _get_calendar_service():
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists("credentials.json"):
                raise FileNotFoundError("Missing credentials.json.")
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    return build("calendar", "v3", credentials=creds)

@tool
def create_grant_deadline_event(
    deadline_date: str,
    title: str,
    application_url: str = "",
    timezone: str = "America/Los_Angeles",
) -> str:
    """Create an all-day Google Calendar event for a grant deadline."""
    service = _get_calendar_service()
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
tools =[search_tool, generate_email_draft, generate_grant_and_save_pdf, create_grant_deadline_event]

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

RULES:
- Be concise and professional.
- If the user provides background information or an uploaded document, use it to populate the organization and project details for your tools.
- Never hallucinate grant deadlines. If you cannot find a specific {current_year} deadline, state that clearly.
"""
)