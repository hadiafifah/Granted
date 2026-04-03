import json
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
)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# define state schemas
class GrantState(TypedDict):
    user_input: str                
    project_details: Optional[str]
    funder_info: Optional[Dict]  
    proposal_pdf_path: Optional[str]
    email_draft: Optional[str]     
    email_result: Optional[Dict] 
    calendar_event: Optional[str]
    final_summary: Optional[str]

# NODE 1: Parse Project Details
def parse_project_node(state: GrantState):
    print("[NODE] parse_project_node starting...")
    if state.get("project_details"):
        return {}
    
    prompt = f"""
Extract structured project details from the input.

INPUT:
{state["user_input"]}

Return clean text including:
- Organization Name
- Mission
- Project Description
- Target Population
- Key Impact

Keep it concise but informative.
"""
    response = llm.invoke(prompt)
    return {"project_details": response.content}


# NODE 2: websearch for grants
def search_node(state: GrantState):
    print("[NODE] search_node starting...")
    query = f"{state['user_input']} grant {datetime.now().year} nonprofit funding"
    result = search_tool.invoke(query)
    return {"funder_info": {"raw_result": str(result)}}


# NODE 3: extract funder info
def extract_funder_node(state: GrantState):
    print("[NODE] extract_funder_node starting...")
    raw = state.get("funder_info", {}).get("raw_result", "")
    prompt = f"""
Extract structured grant information from this search result.

SEARCH RESULT:
{raw}

Return JSON with:
- funder_name
- mission
- priorities
- deadline (ISO format YYYY-MM-DD if possible)
- summary (short)

ONLY return valid JSON.
"""
    response = llm.invoke(prompt).content.strip()

    try:
        data = json.loads(response)
    except:
        data = {
            "funder_name": "Unknown Grant",
            "mission": "",
            "priorities": "",
            "deadline": None,
            "summary": response[:300],
        }

    state_update = {"funder_info": data}
    return state_update

# NODE 4: generate PDF (tool 2)
def pdf_node(state: GrantState):
    print("[NODE] pdf_node starting...")
    result = generate_grant_and_save_pdf.invoke({
        "funder_details": json.dumps(state["funder_info"], indent=2),
        "project_details": state["project_details"]
    })
    return {"proposal_pdf_path": "Grant_Proposal_Submission.pdf" if "Success" in result else None}


# NODE 5: generate email (tool 3)
def email_node(state: GrantState):
    print("[NODE] email_node starting...")
    draft = generate_email_draft.invoke({
        "org_name": state["project_details"].split("\n")[0] if state.get("project_details") else "Organization",
        "goal": "Apply for grant funding",
        "grant_name": state.get("funder_info", {}).get("funder_name"),
        "context": state.get("project_details")
    })
    return {"email_draft": draft}


# NODE 6: send email (tool 3)
def send_node(state: GrantState):
    print("[NODE] send_node starting...")
    result = send_email.invoke({
        "to_email": "ignored@example.com",
        "draft": state["email_draft"]
    })
    print(f"Email send result: {result}")
    return {"email_result": result}


# NODE 7: calendar event (tool 4)
def calendar_node(state: GrantState):
    print("[NODE] calendar_node starting...")
    deadline = state.get("funder_info", {}).get("deadline")
    if not deadline:
        return {"calendar_event": "No valid deadline found. Event not created."}

    result = create_grant_deadline_event.invoke({
        "deadline_date": deadline,
        "title": f"{state.get('funder_info', {}).get('funder_name', 'Grant')} Deadline",
        "application_url": ""
    })
    print(f"Calendar event result: {result}")
    return {"calendar_event": result}


# NODE 8: output
def summary_node(state: GrantState):
    print("[NODE] summary_node starting...")
    summary = f"""
GRANT SUMMARY
-------------------------
Funder: {state.get("funder_info", {}).get("funder_name")}
Deadline: {state.get("funder_info", {}).get("deadline")}

PROPOSAL
-------------------------
{state.get("proposal_pdf_path")}

EMAIL DRAFT
-------------------------
{state.get("email_draft")[:500]}...

EMAIL STATUS
-------------------------
{state.get("email_result")}

CALENDAR
-------------------------
{state.get("calendar_event")}
"""
    return {"final_summary": summary.strip()}


# BUILD GRAPH
builder = StateGraph(GrantState)

builder.add_node("parse", parse_project_node)
builder.add_node("search", search_node)
builder.add_node("extract", extract_funder_node)
builder.add_node("pdf", pdf_node)
builder.add_node("email", email_node)
builder.add_node("send", send_node)
builder.add_node("calendar", calendar_node)
builder.add_node("summary", summary_node)

# flow (the edges)
builder.set_entry_point("parse")

builder.add_edge("parse", "search")
builder.add_edge("search", "extract")
builder.add_edge("extract", "pdf")
builder.add_edge("pdf", "email")
builder.add_edge("email", "send")
builder.add_edge("send", "calendar")
builder.add_edge("calendar", "summary")

# Compile
graph = builder.compile()