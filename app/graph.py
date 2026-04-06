import json
import asyncio
from datetime import datetime
from typing import TypedDict, Optional, Dict

from langgraph.graph import StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI

from app.agent import (
    search_tool,
    generate_grant_and_save_pdf,
    generate_email_draft,
    send_email,
    create_grant_deadline_event,
    llm,
    proposal_schema,
)

from fpdf import FPDF

# ── streaming queue (module-level, set before each run) ──────────────────────
_stream_queue: Optional[asyncio.Queue] = None

def _emit(event: dict):
    """Push an event onto the queue if streaming is active."""
    if _stream_queue is not None:
        try:
            _stream_queue.put_nowait(event)
        except Exception:
            pass

# ── State schema ─────────────────────────────────────────────────────────────
class GrantState(TypedDict):
    user_input: str
    project_details: Optional[str]
    funder_info: Optional[Dict]
    proposal_pdf_path: Optional[str]
    email_draft: Optional[str]
    email_result: Optional[Dict]
    calendar_event: Optional[str]
    final_summary: Optional[str]

# ── NODE 1: Parse Project Details ─────────────────────────────────────────────
def parse_project_node(state: GrantState):
    _emit({"type": "node_start", "node": "parse", "label": "Parsing project details…"})
    try:
        data = json.loads(state["user_input"])
        org_name  = data.get("org_name", "")
        mission   = data.get("mission", "")
        goals     = data.get("goals", "")
        budget    = data.get("budget", "")
        timeline  = data.get("timeline", "")

        project_details = (
            f"Organization Name: {org_name}\n"
            f"Mission Statement: {mission}\n"
            f"Project Goals & Objectives: {goals}\n"
            f"Project Budget: {budget}\n"
            f"Project Timeline: {timeline}"
        )

        prompt = f"""
        Generate a highly targeted 4-8 word Google search query to find active, real-world grant opportunities for this organization.
        Mission: {mission}
        Goals: {goals}
        Only return the search query text. Do not use quotes or introductory text.
        """
        search_query = llm.invoke(prompt).content.strip()
        _emit({"type": "node_done", "node": "parse", "label": "Project details parsed ✓"})
    except Exception as e:
        project_details = state.get("user_input", "")
        search_query    = state.get("user_input", "")[:50]
        _emit({"type": "node_done", "node": "parse", "label": f"Parsed (fallback): {e}"})

    return {"project_details": project_details, "user_input": search_query}


# ── NODE 2: Web-search for grants ─────────────────────────────────────────────
def search_node(state: GrantState):
    _emit({"type": "node_start", "node": "search", "label": "Searching the web for active grants…"})
    try:
        query  = f"{state['user_input']} grant {datetime.now().year} nonprofit funding"
        result = search_tool.invoke(query)
        _emit({"type": "node_done", "node": "search", "label": "Grant search complete ✓"})
        return {"funder_info": {"raw_result": str(result)}}
    except Exception as e:
        _emit({"type": "node_done", "node": "search", "label": f"Search failed: {e}"})
        return {"funder_info": {"raw_result": f"Search failed: {e}"}}


# ── NODE 3: Extract best funder ───────────────────────────────────────────────
def extract_funder_node(state: GrantState):
    _emit({"type": "node_start", "node": "extract", "label": "Identifying the best-matching grant opportunity…"})

    fi_state = state.get("funder_info") or {}
    if isinstance(fi_state, list):
        fi_state = fi_state[0] if fi_state else {}
    raw = fi_state.get("raw_result", "")

    prompt = f"""
You are a grant research assistant. From the search results below, identify the SINGLE most promising and active grant opportunity for this organization.

SEARCH RESULT:
{raw}

Pick only ONE funder — the best match. Return a single JSON object (not a list) with exactly these keys:
- funder_name: (string) Name of the grant program or foundation
- mission: (string) The funder's stated mission or focus area
- priorities: (string) Key funding priorities or eligibility criteria
- deadline: (string) Application deadline in YYYY-MM-DD format, or null if unknown
- summary: (string) 2-3 sentence summary of why this is the best match and what the grant covers

Rules:
- Return ONLY the raw JSON object. No markdown, no backticks, no explanation.
- Do NOT return a list. Return a single JSON object.
- If deadline is unclear or not mentioned, set it to null.
"""
    response = llm.invoke(prompt).content.strip()

    try:
        cleaned = response
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0].strip()
        data = json.loads(cleaned)
        if isinstance(data, list):
            data = data[0] if data else {}
    except Exception as e:
        data = {
            "funder_name": "Unknown Grant",
            "mission": "",
            "priorities": "",
            "deadline": None,
            "summary": response[:300],
        }

    _emit({
        "type": "node_done",
        "node": "extract",
        "label": f"Best match found: {data.get('funder_name', 'Unknown')} ✓",
        "funder": data.get("funder_name", ""),
        "deadline": data.get("deadline", ""),
    })
    return {"funder_info": data}


# ── NODE 4: Generate PDF proposal ─────────────────────────────────────────────
def pdf_node(state: GrantState):
    _emit({"type": "node_start", "node": "pdf", "label": "Starting proposal generation…"})

    funder_info     = state.get("funder_info") or {}
    project_details = state.get("project_details", "")
    funder_details  = json.dumps(funder_info, indent=2)

    full_proposal = {}
    for section in proposal_schema:
        _emit({"type": "section_start", "section": section})
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
        content  = response.content
        full_proposal[section] = content
        _emit({"type": "section_done", "section": section, "preview": content[:200]})

    # Format & save PDF
    _emit({"type": "node_start", "node": "pdf_save", "label": "Formatting and saving proposal PDF…"})
    date_str       = datetime.now().strftime("%B %d, %Y")
    formatted_text = f"Grant Proposal Submission\nDate: {date_str}\n\n{'='*70}\n"
    for title, content in full_proposal.items():
        formatted_text += f"\n\n{title.upper()}\n"
        formatted_text += "-" * len(title) + "\n\n"
        formatted_text += content.strip() + "\n"
        formatted_text += "\n" + "="*70 + "\n"

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", '', 11)
    cleaned_text = formatted_text.encode('latin-1', 'replace').decode('latin-1')
    for line in cleaned_text.split("\n"):
        pdf.multi_cell(0, 8, line)

    pdf_file_name = "Grant_Proposal_Submission.pdf"
    pdf.output(pdf_file_name)

    _emit({"type": "node_done", "node": "pdf", "label": "Proposal PDF saved ✓"})
    return {"proposal_pdf_path": pdf_file_name}


# ── NODE 5: Generate email draft ──────────────────────────────────────────────
def email_node(state: GrantState):
    _emit({"type": "node_start", "node": "email", "label": "Drafting outreach email…"})
    try:
        project_details = state.get("project_details", "") or ""
        org_name = "Organization"
        if project_details:
            first_line = project_details.split("\n")[0]
            org_name   = first_line.replace("Organization Name: ", "").strip()

        funder_info = state.get("funder_info") or {}
        if isinstance(funder_info, list):
            funder_info = funder_info[0] if funder_info else {}

        draft = generate_email_draft.invoke({
            "org_name": org_name,
            "goal": "Apply for grant funding",
            "grant_name": funder_info.get("funder_name", "Unknown Grant"),
            "context": project_details
        })
        _emit({"type": "node_done", "node": "email", "label": "Email draft ready ✓"})
        return {"email_draft": str(draft)}
    except Exception as e:
        _emit({"type": "node_done", "node": "email", "label": f"Email draft failed: {e}"})
        return {"email_draft": f"Failed to generate draft: {str(e)}"}


# ── NODE 6: Send email ────────────────────────────────────────────────────────
def send_node(state: GrantState):
    _emit({"type": "node_start", "node": "send", "label": "Sending outreach email…"})
    try:
        result = send_email.invoke({
            "to_email": "ignored@example.com",
            "draft": state.get("email_draft", "")
        })
        if not isinstance(result, dict):
            result = {"status": str(result)}
        status = result.get("status", "unknown")
        _emit({"type": "node_done", "node": "send",
               "label": f"Email {status} ✓" if status == "sent" else f"Email status: {status}"})
        return {"email_result": result}
    except Exception as e:
        _emit({"type": "node_done", "node": "send", "label": f"Email send error: {e}"})
        return {"email_result": {"status": "error", "message": str(e)}}


# ── NODE 7: Create calendar event ─────────────────────────────────────────────
def calendar_node(state: GrantState):
    _emit({"type": "node_start", "node": "calendar", "label": "Adding grant deadline to Google Calendar…"})
    try:
        funder_info = state.get("funder_info") or {}
        if isinstance(funder_info, list):
            funder_info = funder_info[0] if funder_info else {}

        deadline = funder_info.get("deadline")
        bad_deadlines = ["none", "n/a", "unknown", "tbd", ""]
        if not deadline or str(deadline).strip().lower() in bad_deadlines:
            _emit({"type": "node_done", "node": "calendar", "label": "No valid deadline — calendar skipped"})
            return {"calendar_event": "No valid deadline found. Event not created."}

        result = create_grant_deadline_event.invoke({
            "deadline_date": str(deadline).strip(),
            "title": f"{funder_info.get('funder_name', 'Grant')} Deadline",
            "application_url": ""
        })
        _emit({"type": "node_done", "node": "calendar", "label": "Calendar event created ✓"})
        return {"calendar_event": str(result)}
    except Exception as e:
        _emit({"type": "node_done", "node": "calendar", "label": f"Calendar error: {e}"})
        return {"calendar_event": f"Failed to create event: {str(e)}"}


# ── NODE 8: Build final summary ───────────────────────────────────────────────
def summary_node(state: GrantState):
    _emit({"type": "node_start", "node": "summary", "label": "Building final summary…"})
    try:
        funder_info = state.get("funder_info") or {}
        if isinstance(funder_info, list):
            funder_info = funder_info[0] if funder_info else {}
        if not isinstance(funder_info, dict):
            funder_info = {}

        email_result = state.get("email_result") or {}
        if isinstance(email_result, str):
            email_result = {"status": email_result}
        elif not isinstance(email_result, dict):
            email_result = {}

        raw_draft    = state.get("email_draft", "") or ""
        email_subject = ""
        email_body    = ""
        if "SUBJECT:" in raw_draft:
            email_subject = raw_draft.split("SUBJECT:", 1)[1].split("\n", 1)[0].strip()
        if "BODY:" in raw_draft:
            email_body = raw_draft.split("BODY:", 1)[1].strip()
        else:
            email_body = raw_draft

        summary = (
            "## Grant Summary\n\n"
            f"**Funder:** {funder_info.get('funder_name', 'N/A')}\n\n"
            f"**Deadline:** {funder_info.get('deadline', 'N/A')}\n\n"
            f"**Summary:** {funder_info.get('summary', 'N/A')}\n\n"
            "---\n\n"
            "## Proposal\n\n"
            "Your tailored proposal has been generated.\n\n"
            f"**File:** {state.get('proposal_pdf_path', 'Error generating PDF')}\n\n"
            "---\n\n"
            "## Email Draft\n\n"
            f"**Subject:** {email_subject}\n\n"
            f"{email_body}\n\n"
            "---\n\n"
            "## Email Status\n\n"
            f"**Status:** {(state.get('email_result') or {}).get('status', 'N/A')}\n\n"
            "---\n\n"
            "## Calendar\n\n"
            f"**Status:** {state.get('calendar_event', 'N/A')}\n"
        )
        _emit({"type": "node_done", "node": "summary", "label": "Done ✓"})
        _emit({"type": "final", "summary": summary.strip()})
        return {"final_summary": summary.strip()}
    except Exception as e:
        _emit({"type": "node_done", "node": "summary", "label": f"Summary error: {e}"})
        return {"final_summary": f"Failed to generate final summary: {e}"}


# ── Build graph ───────────────────────────────────────────────────────────────
builder = StateGraph(GrantState)

builder.add_node("parse",    parse_project_node)
builder.add_node("search",   search_node)
builder.add_node("extract",  extract_funder_node)
builder.add_node("pdf",      pdf_node)
builder.add_node("email",    email_node)
builder.add_node("send",     send_node)
builder.add_node("calendar", calendar_node)
builder.add_node("summary",  summary_node)

builder.set_entry_point("parse")
builder.add_edge("parse",    "search")
builder.add_edge("search",   "extract")
builder.add_edge("extract",  "pdf")
builder.add_edge("pdf",      "email")
builder.add_edge("email",    "send")
builder.add_edge("send",     "calendar")
builder.add_edge("calendar", "summary")

graph = builder.compile()


# ── Async streaming entrypoint (used by /submit-stream) ──────────────────────
async def run_graph_with_stream(initial_state: dict, queue: asyncio.Queue):
    """Run the compiled graph in a thread executor, emitting events into queue."""
    global _stream_queue
    _stream_queue = queue

    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, lambda: graph.invoke(initial_state))
    finally:
        _stream_queue = None