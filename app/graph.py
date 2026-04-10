import asyncio
import json
from datetime import datetime
from typing import Dict, Optional, TypedDict

from fpdf import FPDF
from langgraph.graph import StateGraph

from app.agent import (
    create_grant_deadline_event,
    generate_email_draft,
    llm,
    proposal_schema,
    search_tool,
    send_email,
)

# Retry policy defaults
MAX_SEARCH_RETRIES = 2
MAX_CONTENT_RETRIES = 2

# streaming queue (module-level, set before each run)
_stream_queue: Optional[asyncio.Queue] = None


def _emit(event: dict):
    """Push an event onto the queue if streaming is active."""
    if _stream_queue is not None:
        try:
            _stream_queue.put_nowait(event)
        except Exception:
            pass


def _coerce_dict(value) -> Dict:
    if isinstance(value, list):
        value = value[0] if value else {}
    return value if isinstance(value, dict) else {}


def _clean_json_response(raw: str) -> str:
    cleaned = (raw or "").strip()
    if "```json" in cleaned:
        return cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
    if "```" in cleaned:
        return cleaned.split("```", 1)[1].split("```", 1)[0].strip()
    return cleaned


def _parse_json_object(raw: str, fallback: Dict) -> Dict:
    try:
        data = json.loads(_clean_json_response(raw))
        if isinstance(data, list):
            data = data[0] if data else {}
        return data if isinstance(data, dict) else fallback
    except Exception:
        return fallback


def _normalize_verdict(value: str, default: str = "fail") -> str:
    v = (value or "").strip().lower()
    if v in {"pass", "approve", "approved", "good"}:
        return "pass"
    if v in {"fail", "reject", "rejected", "bad"}:
        return "fail"
    return default


def _listify_text(value) -> str:
    if isinstance(value, list):
        return "\n".join(f"- {str(item).strip()}" for item in value if str(item).strip())
    if isinstance(value, str):
        return value.strip()
    return ""


class GrantState(TypedDict):
    user_input: str
    project_details: Optional[str]
    funder_info: Optional[Dict]
    proposal_pdf_path: Optional[str]
    proposal_text: Optional[str]
    email_draft: Optional[str]
    email_result: Optional[Dict]
    calendar_event: Optional[str]
    final_summary: Optional[str]
    search_attempts: Optional[int]
    content_attempts: Optional[int]
    grant_review: Optional[Dict]
    content_review: Optional[Dict]
    refined_search_query: Optional[str]
    halt_reason: Optional[str]


def parse_project_node(state: GrantState):
    _emit({"type": "node_start", "node": "parse", "label": "Parsing project details..."})
    try:
        data = json.loads(state["user_input"])
        org_name = data.get("org_name", "")
        mission = data.get("mission", "")
        goals = data.get("goals", "")
        budget = data.get("budget", "")
        timeline = data.get("timeline", "")

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
        _emit({"type": "node_done", "node": "parse", "label": "Project details parsed"})
    except Exception as e:
        project_details = state.get("user_input", "")
        search_query = state.get("user_input", "")[:50]
        _emit({"type": "node_done", "node": "parse", "label": f"Parsed (fallback): {e}"})

    return {
        "project_details": project_details,
        "user_input": search_query,
        "search_attempts": 0,
        "content_attempts": 0,
        "grant_review": None,
        "content_review": None,
        "refined_search_query": None,
        "halt_reason": None,
        "proposal_text": "",
        "email_draft": "",
        "email_result": {},
        "calendar_event": "",
    }


def search_node(state: GrantState):
    _emit({"type": "node_start", "node": "search", "label": "Searching the web for active grants..."})
    attempts = int(state.get("search_attempts") or 0) + 1
    base_query = (state.get("refined_search_query") or state.get("user_input") or "").strip()
    query = f"{base_query} grant {datetime.now().year} nonprofit funding".strip()
    try:
        result = search_tool.invoke(query)
        _emit(
            {
                "type": "node_done",
                "node": "search",
                "label": f"Grant search complete (attempt {attempts}/{MAX_SEARCH_RETRIES})",
            }
        )
        return {
            "funder_info": {"raw_result": str(result), "query_used": query},
            "search_attempts": attempts,
            "refined_search_query": None,
        }
    except Exception as e:
        _emit({"type": "node_done", "node": "search", "label": f"Search failed: {e}"})
        return {
            "funder_info": {"raw_result": f"Search failed: {e}", "query_used": query},
            "search_attempts": attempts,
            "refined_search_query": None,
        }


def extract_funder_node(state: GrantState):
    _emit(
        {
            "type": "node_start",
            "node": "extract",
            "label": "Identifying the best-matching grant opportunity...",
        }
    )

    fi_state = _coerce_dict(state.get("funder_info") or {})
    raw = fi_state.get("raw_result", "")

    prompt = f"""
You are a grant research assistant. From the search results below, identify the SINGLE most promising and active grant opportunity for this organization.

SEARCH RESULT:
{raw}

Pick only ONE funder - the best match. Return a single JSON object (not a list) with exactly these keys:
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

    data = _parse_json_object(
        response,
        {
            "funder_name": "Unknown Grant",
            "mission": "",
            "priorities": "",
            "deadline": None,
            "summary": response[:300],
        },
    )
    data["source_query"] = fi_state.get("query_used", "")

    _emit(
        {
            "type": "node_done",
            "node": "extract",
            "label": f"Best match found: {data.get('funder_name', 'Unknown')}",
            "funder": data.get("funder_name", ""),
            "deadline": data.get("deadline", ""),
        }
    )
    return {"funder_info": data}


def grant_review_node(state: GrantState):
    _emit({"type": "node_start", "node": "grant_review", "label": "Reviewing grant relevance..."})
    project_details = state.get("project_details", "") or ""
    funder_info = _coerce_dict(state.get("funder_info") or {})
    attempts = int(state.get("search_attempts") or 0)
    original_query = (state.get("user_input") or "").strip()

    prompt = f"""
You are a strict grant-fit reviewer.
Evaluate if this selected grant is relevant and useful for the nonprofit profile.

PROJECT DETAILS:
{project_details}

SELECTED GRANT:
{json.dumps(funder_info, indent=2)}

Return exactly one JSON object with these keys:
- verdict: "pass" or "fail"
- score: integer from 0 to 10
- reasons: list of short strings explaining key strengths/weaknesses
- improvement_notes: concise tactical guidance for the next iteration
- refined_search_query: better search query text if verdict is fail, otherwise null

Rules:
- Be strict. Fail if fit is weak, ambiguous, stale, or not actionable.
- Return only raw JSON.
"""
    raw = llm.invoke(prompt).content.strip()
    review = _parse_json_object(
        raw,
        {
            "verdict": "fail",
            "score": 0,
            "reasons": ["Could not parse review output."],
            "improvement_notes": "Use a more specific query tied to mission and goals.",
            "refined_search_query": original_query,
        },
    )

    review["verdict"] = _normalize_verdict(review.get("verdict"), default="fail")
    review["score"] = int(review.get("score", 0) or 0)
    review["reasons"] = review.get("reasons") if isinstance(review.get("reasons"), list) else []
    review["improvement_notes"] = str(review.get("improvement_notes", "") or "").strip()

    refined_query = review.get("refined_search_query")
    if review["verdict"] == "pass":
        refined_query = None
    elif not isinstance(refined_query, str) or not refined_query.strip():
        refined_query = original_query
    else:
        refined_query = refined_query.strip()

    if review["verdict"] == "pass":
        next_action = "proceed"
    elif attempts < MAX_SEARCH_RETRIES:
        next_action = "retry_search"
    else:
        next_action = "proceed"

    _emit(
        {
            "type": "review_attempt",
            "node": "grant_review",
            "attempt": attempts,
            "max_attempts": MAX_SEARCH_RETRIES,
            "verdict": review["verdict"],
            "score": review["score"],
            "reasons": review["reasons"],
            "improvement_notes": review["improvement_notes"],
            "next_action": next_action,
        }
    )

    _emit(
        {
            "type": "node_done",
            "node": "grant_review",
            "label": (
                f"Grant review: PASS ({review['score']}/10)"
                if review["verdict"] == "pass"
                else f"Grant review: FAIL ({review['score']}/10), attempt {attempts}/{MAX_SEARCH_RETRIES}"
            ),
            "verdict": review["verdict"],
            "score": review["score"],
        }
    )
    return {"grant_review": review, "refined_search_query": refined_query}


def route_after_grant_review(state: GrantState) -> str:
    review = _coerce_dict(state.get("grant_review") or {})
    verdict = _normalize_verdict(review.get("verdict"), default="fail")
    attempts = int(state.get("search_attempts") or 0)

    if verdict == "pass":
        return "to_pdf"
    if attempts < MAX_SEARCH_RETRIES:
        return "retry_search"
    return "to_pdf"


def pdf_node(state: GrantState):
    _emit({"type": "node_start", "node": "pdf", "label": "Starting proposal generation..."})

    funder_info = _coerce_dict(state.get("funder_info") or {})
    project_details = state.get("project_details", "") or ""
    funder_details = json.dumps(funder_info, indent=2)

    grant_review = _coerce_dict(state.get("grant_review") or {})
    content_review = _coerce_dict(state.get("content_review") or {})
    reviewer_notes = []
    if grant_review.get("improvement_notes"):
        reviewer_notes.append(f"Grant review guidance: {grant_review.get('improvement_notes')}")
    if content_review.get("improvement_notes"):
        reviewer_notes.append(f"Content review guidance: {content_review.get('improvement_notes')}")
    notes_text = "\n".join(f"- {note}" for note in reviewer_notes) if reviewer_notes else "None"

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

=============================
REVIEWER GUIDANCE
=============================
{notes_text}

INSTRUCTIONS:
- Align strongly with the funder's mission and funding priorities.
- Use persuasive and impact-driven language.
- 300-500 words.
- Do NOT generate other section titles.
"""
        response = llm.invoke(prompt)
        content = response.content
        full_proposal[section] = content
        _emit({"type": "section_done", "section": section, "preview": str(content)[:200]})

    _emit({"type": "node_start", "node": "pdf_save", "label": "Formatting and saving proposal PDF..."})
    date_str = datetime.now().strftime("%B %d, %Y")
    formatted_text = f"Grant Proposal Submission\nDate: {date_str}\n\n{'=' * 70}\n"
    for title, content in full_proposal.items():
        formatted_text += f"\n\n{title.upper()}\n"
        formatted_text += "-" * len(title) + "\n\n"
        formatted_text += str(content).strip() + "\n"
        formatted_text += "\n" + "=" * 70 + "\n"

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", "", 11)
    cleaned_text = formatted_text.encode("latin-1", "replace").decode("latin-1")
    for line in cleaned_text.split("\n"):
        pdf.multi_cell(0, 8, line)

    pdf_file_name = "Grant_Proposal_Submission.pdf"
    pdf.output(pdf_file_name)

    low_confidence = (
        _normalize_verdict(grant_review.get("verdict"), default="fail") != "pass"
        and int(state.get("search_attempts") or 0) >= MAX_SEARCH_RETRIES
    )
    label = "Proposal PDF saved"
    if low_confidence:
        label = "Proposal PDF saved (low-confidence grant fit)"
    _emit({"type": "node_done", "node": "pdf", "label": label})

    return {"proposal_pdf_path": pdf_file_name, "proposal_text": formatted_text}


def email_node(state: GrantState):
    _emit({"type": "node_start", "node": "email", "label": "Drafting outreach email..."})
    try:
        project_details = state.get("project_details", "") or ""
        org_name = "Organization"
        if project_details:
            first_line = project_details.split("\n")[0]
            org_name = first_line.replace("Organization Name: ", "").strip() or "Organization"

        funder_info = _coerce_dict(state.get("funder_info") or {})
        content_review = _coerce_dict(state.get("content_review") or {})
        reviewer_guidance = str(content_review.get("improvement_notes", "") or "").strip()

        context_parts = [
            project_details,
            f"Funder: {funder_info.get('funder_name', 'Unknown Grant')}",
            f"Funder priorities: {funder_info.get('priorities', '')}",
            f"Proposal excerpt:\n{(state.get('proposal_text') or '')[:2500]}",
        ]
        if reviewer_guidance:
            context_parts.append(f"Reviewer guidance for this revision: {reviewer_guidance}")

        draft = generate_email_draft.invoke(
            {
                "org_name": org_name,
                "goal": "Apply for grant funding",
                "grant_name": funder_info.get("funder_name", "Unknown Grant"),
                "context": "\n\n".join(context_parts),
            }
        )
        _emit({"type": "node_done", "node": "email", "label": "Email draft ready"})
        return {"email_draft": str(draft)}
    except Exception as e:
        _emit({"type": "node_done", "node": "email", "label": f"Email draft failed: {e}"})
        return {"email_draft": f"Failed to generate draft: {str(e)}"}


def content_review_node(state: GrantState):
    _emit({"type": "node_start", "node": "content_review", "label": "Reviewing proposal + email quality..."})
    attempts = int(state.get("content_attempts") or 0) + 1
    project_details = state.get("project_details", "") or ""
    funder_info = _coerce_dict(state.get("funder_info") or {})
    proposal_text = state.get("proposal_text", "") or ""
    email_draft = state.get("email_draft", "") or ""

    prompt = f"""
You are a strict grant-communications reviewer.
Evaluate whether the proposal and outreach email are appropriate for both:
1) the selected grant priorities, and
2) the nonprofit organization profile.

PROJECT DETAILS:
{project_details}

FUNDER INFO:
{json.dumps(funder_info, indent=2)}

PROPOSAL TEXT:
{proposal_text[:6000]}

EMAIL DRAFT:
{email_draft}

Return exactly one JSON object with these keys:
- verdict: "pass" or "fail"
- score: integer from 0 to 10
- reasons: list of short strings
- improvement_notes: concise edits needed to pass

Rules:
- Fail if the proposal/email are generic, misaligned, or weakly tied to funder priorities.
- Return only raw JSON.
"""
    raw = llm.invoke(prompt).content.strip()
    review = _parse_json_object(
        raw,
        {
            "verdict": "fail",
            "score": 0,
            "reasons": ["Could not parse content review output."],
            "improvement_notes": "Increase alignment with funder priorities and organization mission.",
        },
    )
    review["verdict"] = _normalize_verdict(review.get("verdict"), default="fail")
    review["score"] = int(review.get("score", 0) or 0)
    review["reasons"] = review.get("reasons") if isinstance(review.get("reasons"), list) else []
    review["improvement_notes"] = str(review.get("improvement_notes", "") or "").strip()

    halt_reason = None
    if review["verdict"] == "pass":
        next_action = "proceed"
    elif attempts < MAX_CONTENT_RETRIES:
        next_action = "retry_pdf"
    else:
        next_action = "halt_manual_review"

    if review["verdict"] != "pass" and attempts >= MAX_CONTENT_RETRIES:
        halt_reason = (
            "Content review failed after max retries. Manual review required before sending outreach."
        )

    _emit(
        {
            "type": "review_attempt",
            "node": "content_review",
            "attempt": attempts,
            "max_attempts": MAX_CONTENT_RETRIES,
            "verdict": review["verdict"],
            "score": review["score"],
            "reasons": review["reasons"],
            "improvement_notes": review["improvement_notes"],
            "next_action": next_action,
        }
    )

    _emit(
        {
            "type": "node_done",
            "node": "content_review",
            "label": (
                f"Content review: PASS ({review['score']}/10)"
                if review["verdict"] == "pass"
                else f"Content review: FAIL ({review['score']}/10), attempt {attempts}/{MAX_CONTENT_RETRIES}"
            ),
            "verdict": review["verdict"],
            "score": review["score"],
        }
    )

    return {"content_review": review, "content_attempts": attempts, "halt_reason": halt_reason}


def route_after_content_review(state: GrantState) -> str:
    review = _coerce_dict(state.get("content_review") or {})
    verdict = _normalize_verdict(review.get("verdict"), default="fail")
    attempts = int(state.get("content_attempts") or 0)

    if verdict == "pass":
        return "to_send"
    if attempts < MAX_CONTENT_RETRIES:
        return "retry_pdf"
    return "to_summary"


def send_node(state: GrantState):
    _emit({"type": "node_start", "node": "send", "label": "Sending outreach email..."})
    try:
        result = send_email.invoke({"to_email": "ignored@example.com", "draft": state.get("email_draft", "")})
        if not isinstance(result, dict):
            result = {"status": str(result)}
        status = result.get("status", "unknown")
        _emit(
            {
                "type": "node_done",
                "node": "send",
                "label": f"Email {status}" if status == "sent" else f"Email status: {status}",
            }
        )
        return {"email_result": result}
    except Exception as e:
        _emit({"type": "node_done", "node": "send", "label": f"Email send error: {e}"})
        return {"email_result": {"status": "error", "message": str(e)}}


def calendar_node(state: GrantState):
    _emit({"type": "node_start", "node": "calendar", "label": "Adding grant deadline to Google Calendar..."})
    try:
        funder_info = _coerce_dict(state.get("funder_info") or {})
        deadline = funder_info.get("deadline")
        bad_deadlines = ["none", "n/a", "unknown", "tbd", ""]
        if not deadline or str(deadline).strip().lower() in bad_deadlines:
            _emit({"type": "node_done", "node": "calendar", "label": "No valid deadline; calendar skipped"})
            return {"calendar_event": "No valid deadline found. Event not created."}

        result = create_grant_deadline_event.invoke(
            {
                "deadline_date": str(deadline).strip(),
                "title": f"{funder_info.get('funder_name', 'Grant')} Deadline",
                "application_url": "",
            }
        )
        _emit({"type": "node_done", "node": "calendar", "label": "Calendar event created"})
        return {"calendar_event": str(result)}
    except Exception as e:
        _emit({"type": "node_done", "node": "calendar", "label": f"Calendar error: {e}"})
        return {"calendar_event": f"Failed to create event: {str(e)}"}


def summary_node(state: GrantState):
    _emit({"type": "node_start", "node": "summary", "label": "Building final summary..."})
    try:
        funder_info = _coerce_dict(state.get("funder_info") or {})
        grant_review = _coerce_dict(state.get("grant_review") or {})
        content_review = _coerce_dict(state.get("content_review") or {})

        email_result = state.get("email_result") or {}
        if isinstance(email_result, str):
            email_result = {"status": email_result}
        elif not isinstance(email_result, dict):
            email_result = {}

        raw_draft = state.get("email_draft", "") or ""
        email_subject = ""
        email_body = ""
        if "SUBJECT:" in raw_draft:
            email_subject = raw_draft.split("SUBJECT:", 1)[1].split("\n", 1)[0].strip()
        if "BODY:" in raw_draft:
            email_body = raw_draft.split("BODY:", 1)[1].strip()
        else:
            email_body = raw_draft

        grant_attempts = int(state.get("search_attempts") or 0)
        content_attempts = int(state.get("content_attempts") or 0)
        grant_verdict = _normalize_verdict(grant_review.get("verdict"), default="fail")
        content_verdict = _normalize_verdict(content_review.get("verdict"), default="fail")

        low_confidence = grant_verdict != "pass" and grant_attempts >= MAX_SEARCH_RETRIES
        halt_reason = (state.get("halt_reason") or "").strip()
        email_status = email_result.get("status", "N/A")
        calendar_status = state.get("calendar_event", "N/A")
        if halt_reason:
            email_status = "not sent (halted by content review)"
            calendar_status = "not created (halted before send)"

        summary = (
            "## Grant Summary\n\n"
            f"**Funder:** {funder_info.get('funder_name', 'N/A')}\n\n"
            f"**Deadline:** {funder_info.get('deadline', 'N/A')}\n\n"
            f"**Summary:** {funder_info.get('summary', 'N/A')}\n\n"
            "---\n\n"
            "## Review Gates\n\n"
            f"**Grant Review:** {grant_verdict.upper()} ({grant_review.get('score', 'N/A')}/10), attempts: {grant_attempts}\n\n"
            f"{_listify_text(grant_review.get('reasons'))}\n\n"
            f"**Grant Reviewer Notes:** {grant_review.get('improvement_notes', 'N/A')}\n\n"
            f"**Content Review:** {content_verdict.upper()} ({content_review.get('score', 'N/A')}/10), attempts: {content_attempts}\n\n"
            f"{_listify_text(content_review.get('reasons'))}\n\n"
            f"**Content Reviewer Notes:** {content_review.get('improvement_notes', 'N/A')}\n\n"
            f"**Low-Confidence Grant Fit:** {'Yes' if low_confidence else 'No'}\n\n"
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
            f"**Status:** {email_status}\n\n"
            "---\n\n"
            "## Calendar\n\n"
            f"**Status:** {calendar_status}\n\n"
            "---\n\n"
            "## Manual Review\n\n"
            f"**Required:** {'Yes' if bool(halt_reason) else 'No'}\n\n"
            f"**Reason:** {halt_reason or 'None'}\n"
        )
        _emit({"type": "node_done", "node": "summary", "label": "Done"})
        _emit({"type": "final", "summary": summary.strip()})
        return {"final_summary": summary.strip()}
    except Exception as e:
        _emit({"type": "node_done", "node": "summary", "label": f"Summary error: {e}"})
        return {"final_summary": f"Failed to generate final summary: {e}"}


builder = StateGraph(GrantState)

builder.add_node("parse", parse_project_node)
builder.add_node("search", search_node)
builder.add_node("extract", extract_funder_node)
builder.add_node("grant_review", grant_review_node)
builder.add_node("pdf", pdf_node)
builder.add_node("email", email_node)
builder.add_node("content_review", content_review_node)
builder.add_node("send", send_node)
builder.add_node("calendar", calendar_node)
builder.add_node("summary", summary_node)

builder.set_entry_point("parse")
builder.add_edge("parse", "search")
builder.add_edge("search", "extract")
builder.add_edge("extract", "grant_review")
builder.add_conditional_edges(
    "grant_review",
    route_after_grant_review,
    {"retry_search": "search", "to_pdf": "pdf"},
)
builder.add_edge("pdf", "email")
builder.add_edge("email", "content_review")
builder.add_conditional_edges(
    "content_review",
    route_after_content_review,
    {"retry_pdf": "pdf", "to_send": "send", "to_summary": "summary"},
)
builder.add_edge("send", "calendar")
builder.add_edge("calendar", "summary")

graph = builder.compile()


async def run_graph_with_stream(initial_state: dict, queue: asyncio.Queue):
    """Run the compiled graph in a thread executor, emitting events into queue."""
    global _stream_queue
    _stream_queue = queue

    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, lambda: graph.invoke(initial_state))
    finally:
        _stream_queue = None
