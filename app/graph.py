import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, TypedDict
from urllib.parse import urlparse

from langgraph.graph import StateGraph

from app.agent import (
    create_grant_deadline_event,
    generate_email_draft,
    llm,
    reviewer_llm,
    proposal_schema,
    search_tool,
    send_email,
)
from app.pdf_utils import build_proposal_plain_text, render_proposal_pdf
from app.propublica_utils import search_propublica_nonprofit

# Retry policy defaults
MAX_SEARCH_RETRIES = 5
MAX_SEARCH_CYCLES = 15
MAX_CONTENT_RETRIES = 2

DEFAULT_REVIEW_PROFILE = "demo"
REVIEW_PROFILE = (os.getenv("GRANT_REVIEW_PROFILE", DEFAULT_REVIEW_PROFILE) or DEFAULT_REVIEW_PROFILE).strip().lower()
if REVIEW_PROFILE not in {"strict", "balanced", "demo"}:
    REVIEW_PROFILE = "balanced"

GRANT_PASS_SCORE = {"strict": 7, "balanced": 6, "demo": 5}[REVIEW_PROFILE]
CONTENT_PASS_SCORE = {"strict": 7, "balanced": 6, "demo": 5}[REVIEW_PROFILE]
SUSPICIOUS_SOURCE_MAX_SCORE = {"strict": 3, "balanced": 5, "demo": 7}[REVIEW_PROFILE]
NAME_URL_MISMATCH_MAX_SCORE = {"strict": 6, "balanced": 7, "demo": 8}[REVIEW_PROFILE]

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


def _score_to_match_percent(score_value) -> int:
    try:
        score = float(score_value)
    except Exception:
        score = 0.0
    score = max(0.0, min(10.0, score))
    return int(round(score * 10))


def _coerce_match_percent(match_value, fallback_score=0) -> int:
    try:
        match_percent = int(round(float(match_value)))
    except Exception:
        return _score_to_match_percent(fallback_score)
    if 0 <= match_percent <= 100:
        return match_percent
    return _score_to_match_percent(fallback_score)


def _clamp_score(value) -> int:
    try:
        score = int(value)
    except Exception:
        score = 0
    return max(0, min(10, score))


def _grant_reviewer_role() -> str:
    if REVIEW_PROFILE == "strict":
        return "You are a strict grant-fit reviewer."
    if REVIEW_PROFILE == "demo":
        return "You are a fair but demo-friendly grant-fit reviewer."
    return "You are a balanced grant-fit reviewer."


def _content_reviewer_role() -> str:
    if REVIEW_PROFILE == "strict":
        return "You are a strict grant-communications reviewer."
    if REVIEW_PROFILE == "demo":
        return "You are a fair but demo-friendly grant-communications reviewer."
    return "You are a balanced grant-communications reviewer."


def _grant_profile_rules() -> str:
    if REVIEW_PROFILE == "strict":
        return (
            "- Be strict. Fail if fit is weak, ambiguous, stale, or not actionable.\n"
            "- Fail if the funder is unnamed/generic or if source_url is missing/invalid."
        )
    if REVIEW_PROFILE == "demo":
        return (
            "- Be realistically optimistic. Reward partial alignment when the opportunity is plausible.\n"
            "- Only fail for critical issues (expired deadline, unnamed funder, or invalid/missing source_url)."
        )
    return (
        "- Be firm but fair. Reward meaningful alignment even if some details are incomplete.\n"
        "- Fail for critical issues (expired deadline, unnamed funder, or invalid/missing source_url)."
    )


def _content_profile_rules() -> str:
    if REVIEW_PROFILE == "strict":
        return "- Fail if the proposal/email are generic, misaligned, or weakly tied to funder priorities."
    if REVIEW_PROFILE == "demo":
        return "- Be realistically optimistic. Pass when proposal/email clearly align overall, even with minor weaknesses."
    return "- Be firm but fair. Pass if the core narrative aligns with mission and priorities, with only minor issues."


def normalize_grant_key(name: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in (name or ""))
    return "".join(cleaned.split())


def _coerce_string_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text:
            items.append(text)
    return items


def _is_generic_funder_name(name: str) -> bool:
    normalized = " ".join((name or "").strip().lower().split())
    if not normalized:
        return True
    generic_tokens = [
        "unknown grant",
        "unknown funder",
        "unnamed",
        "forward-thinking funding organization",
        "funding organization",
        "organization",
        "funder",
        "grant provider",
    ]
    return any(token in normalized for token in generic_tokens)


def _is_http_url(url: str) -> bool:
    text = str(url or "").strip()
    if not text:
        return False
    try:
        parsed = urlparse(text)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    except Exception:
        return False


def _is_verifiable_grant(candidate: Dict) -> bool:
    funder_name = str(candidate.get("funder_name", "") or "").strip()
    source_url = str(candidate.get("source_url", "") or "").strip()
    return (not _is_generic_funder_name(funder_name)) and _is_http_url(source_url)


def _extract_top_candidates(raw_response: str, source_query: str) -> list[Dict]:
    parsed = _parse_json_object(raw_response, {"candidates": []})
    candidate_items = parsed.get("candidates")
    if not isinstance(candidate_items, list):
        # Backward compatibility: tolerate single-candidate object responses.
        if isinstance(parsed.get("funder_name"), str):
            candidate_items = [parsed]
        else:
            candidate_items = []

    candidates: list[Dict] = []
    for idx, item in enumerate(candidate_items[:3]):
        data = _coerce_dict(item)
        funder_name = str(data.get("funder_name", "") or "").strip() or "Unknown Grant"
        candidates.append(
            {
                "funder_name": funder_name,
                "mission": str(data.get("mission", "") or "").strip(),
                "priorities": str(data.get("priorities", "") or "").strip(),
                "deadline": data.get("deadline"),
                "summary": str(data.get("summary", "") or "").strip(),
                "source_url": str(data.get("source_url", "") or "").strip(),
                "source_query": source_query,
                "candidate_rank": idx + 1,
            }
        )

    if candidates:
        return candidates

    return [
        {
            "funder_name": "Unknown Grant",
            "mission": "",
            "priorities": "",
            "deadline": None,
            "summary": (raw_response or "")[:300],
            "source_url": "",
            "source_query": source_query,
            "candidate_rank": 1,
        }
    ]


def _pick_better_grant_candidate(
    current_funder: Dict,
    current_review: Dict,
    best_funder: Dict,
    best_review: Dict,
) -> bool:
    """Return True when the current attempt should become the best-known grant candidate."""
    if not best_review:
        return True

    current_score = max(0, min(10, int(current_review.get("score", 0) or 0)))
    best_score = max(0, min(10, int(best_review.get("score", 0) or 0)))
    if current_score != best_score:
        return current_score > best_score

    current_verdict = _normalize_verdict(current_review.get("verdict"), default="fail")
    best_verdict = _normalize_verdict(best_review.get("verdict"), default="fail")
    if current_verdict != best_verdict:
        return current_verdict == "pass"

    current_name = str(current_funder.get("funder_name", "") or "").strip()
    best_name = str(best_funder.get("funder_name", "") or "").strip()
    if bool(current_name) != bool(best_name):
        return bool(current_name)

    # Stable tie-breaker: keep the earlier best entry.
    return False


class GrantState(TypedDict):
    user_input: str
    project_details: Optional[str]
    funder_info: Optional[Dict]
    best_funder_info: Optional[Dict]
    best_grant_review: Optional[Dict]
    best_grant_attempt: Optional[int]
    proposal_pdf_path: Optional[str]
    proposal_text: Optional[str]
    proposal_sections: Optional[Dict[str, str]]
    email_draft: Optional[str]
    email_result: Optional[Dict]
    calendar_event: Optional[str]
    final_summary: Optional[str]
    search_attempts: Optional[int]
    search_cycles: Optional[int]
    content_attempts: Optional[int]
    grant_review: Optional[Dict]
    content_review: Optional[Dict]
    refined_search_query: Optional[str]
    seen_grant_keys: Optional[list[str]]
    seen_grant_names: Optional[list[str]]
    extract_duplicate_only: Optional[bool]
    halt_reason: Optional[str]
    propublica_result: Optional[Dict]
    nonprofit_confidence: Optional[str]


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
        "search_cycles": 0,
        "content_attempts": 0,
        "grant_review": None,
        "best_funder_info": None,
        "best_grant_review": None,
        "best_grant_attempt": None,
        "content_review": None,
        "refined_search_query": None,
        "seen_grant_keys": [],
        "seen_grant_names": [],
        "extract_duplicate_only": False,
        "halt_reason": None,
        "proposal_text": "",
        "proposal_sections": {},
        "email_draft": "",
        "email_result": {},
        "calendar_event": "",
        "propublica_result": {},
        "nonprofit_confidence": "unknown",
    }

def propublica_node(state: GrantState):
    _emit({"type": "node_start", "node": "propublica", "label": "Checking nonprofit grounding..."})

    project_details = state.get("project_details", "") or ""
    org_name = ""
    if project_details:
        first_line = project_details.split("\n")[0]
        org_name = first_line.replace("Organization Name: ", "").strip()

    result = search_propublica_nonprofit(org_name)
    confidence = str(result.get("confidence", "low"))

    label = (
        f"Nonprofit grounding: {confidence}"
        if result.get("matched_name")
        else "No confident nonprofit match found"
    )

    _emit(
        {
            "type": "node_done",
            "node": "propublica",
            "label": label,
        }
    )

    return {
        "propublica_result": result,
        "nonprofit_confidence": confidence,
    }


def search_node(state: GrantState):
    _emit({"type": "node_start", "node": "search", "label": "Searching for new grants..."})
    cycles = int(state.get("search_cycles") or 0) + 1
    attempts = int(state.get("search_attempts") or 0)
    attempt_display = min(attempts + 1, MAX_SEARCH_RETRIES)
    base_query = (state.get("refined_search_query") or state.get("user_input") or "").strip()
    propublica_result = _coerce_dict(state.get("propublica_result") or {})
    nonprofit_confidence = str(state.get("nonprofit_confidence") or "unknown").lower()

    if nonprofit_confidence == "high" and propublica_result.get("matched_name"):
        official_name = str(propublica_result.get("matched_name") or "").strip()
        city = str(propublica_result.get("city") or "").strip()
        state_code = str(propublica_result.get("state") or "").strip()
        enriched_bits = [official_name]
        if city:
            enriched_bits.append(city)
        if state_code:
            enriched_bits.append(state_code)
        base_query = f"{base_query} {' '.join(enriched_bits)}".strip()
    query = f"{base_query} grant {datetime.now().year} nonprofit funding".strip()
    try:
        result = search_tool.invoke(query)
        _emit(
            {
                "type": "node_done",
                "node": "search",
                "label": f"Found new grant (attempt {attempt_display}/{MAX_SEARCH_RETRIES})",
            }
        )
        return {
            "funder_info": {"raw_result": str(result), "query_used": query},
            "search_cycles": cycles,
            "refined_search_query": None,
            "extract_duplicate_only": False,
        }
    except Exception as e:
        _emit({"type": "node_done", "node": "search", "label": f"Search failed: {e}"})
        return {
            "funder_info": {"raw_result": f"Search failed: {e}", "query_used": query},
            "search_cycles": cycles,
            "refined_search_query": None,
            "extract_duplicate_only": False,
        }


def extract_funder_node(state: GrantState):
    retry_verification = int(state.get("search_attempts") or 0) > 0
    _emit(
        {
            "type": "node_start",
            "node": "extract",
            "label": "Verifying new grant..." if retry_verification else "Identifying verifiable grant opportunities...",
        }
    )

    fi_state = _coerce_dict(state.get("funder_info") or {})
    raw = fi_state.get("raw_result", "")

    seen_grant_keys = _coerce_string_list(state.get("seen_grant_keys"))
    seen_grant_names = _coerce_string_list(state.get("seen_grant_names"))
    seen_keys_set = {normalize_grant_key(name) for name in seen_grant_keys if normalize_grant_key(name)}

    already_seen_text = ", ".join(seen_grant_names[:10]) if seen_grant_names else "None"
    prompt = f"""
You are a grant research assistant. From the search results below, identify up to 3 active grant opportunities ranked by best fit for this organization.

SEARCH RESULT:
{raw}

ALREADY REVIEWED GRANTS (avoid duplicates):
{already_seen_text}

Return exactly one JSON object with this shape:
{{
  "candidates": [
    {{
      "funder_name": "string",
      "mission": "string",
      "priorities": "string",
      "deadline": "YYYY-MM-DD or null",
      "summary": "2-3 sentences",
      "source_url": "https://... exact page URL where this grant was found"
    }}
  ]
}}

Rules:
- Return ONLY raw JSON. No markdown, no backticks.
- Provide up to 3 candidates sorted best to worst fit.
- Prefer candidates not listed in ALREADY REVIEWED GRANTS.
- Do NOT use placeholders like "unnamed", "unknown", or generic funder labels.
- Every candidate must include a valid source_url from SEARCH RESULT.
- If deadline is unclear, set it to null.
"""
    response = llm.invoke(prompt).content.strip()
    source_query = fi_state.get("query_used", "")
    candidates = _extract_top_candidates(response, source_query)

    selected_candidate = None
    selected_key = ""
    for candidate in candidates:
        key = normalize_grant_key(str(candidate.get("funder_name", "") or ""))
        if not _is_verifiable_grant(candidate):
            continue
        if key and key in seen_keys_set:
            continue
        selected_candidate = candidate
        selected_key = key
        break

    if selected_candidate is not None:
        updated_seen_keys = list(seen_grant_keys)
        updated_seen_names = list(seen_grant_names)
        if selected_key and selected_key not in seen_keys_set:
            updated_seen_keys.append(selected_key)
            updated_seen_names.append(str(selected_candidate.get("funder_name", "Unknown Grant")))

        _emit(
            {
                "type": "node_done",
                "node": "extract",
                "label": f"Selected unseen grant: {selected_candidate.get('funder_name', 'Unknown Grant')}",
                "funder": selected_candidate.get("funder_name", ""),
                "deadline": selected_candidate.get("deadline", ""),
            }
        )
        return {
            "funder_info": selected_candidate,
            "seen_grant_keys": updated_seen_keys,
            "seen_grant_names": updated_seen_names,
            "extract_duplicate_only": False,
            "halt_reason": None,
        }

    reviewed_names = ", ".join(seen_grant_names[-12:]) if seen_grant_names else "None"
    base_query = (
        state.get("refined_search_query")
        or state.get("user_input")
        or fi_state.get("query_used", "")
        or ""
    ).strip()
    novelty_prompt = f"""
Create one concise web search query to find different active grants for this nonprofit.

Current query:
{base_query}

Avoid these previously reviewed grants:
{reviewed_names}

Rules:
- Return only the query text.
- 6-12 words.
- Prioritize grants likely to be different from the avoided list.
- Prioritize results with named funders and official grant program pages.
"""
    refined_query = llm.invoke(novelty_prompt).content.strip()
    if not refined_query:
        refined_query = f"{base_query} alternative grant programs".strip()

    _emit(
        {
            "type": "node_done",
            "node": "extract",
            "label": (
                f"No unseen verifiable candidate in top {len(candidates)}; "
                "refining search for named grants with URLs"
            ),
        }
    )
    best_funder_info = _coerce_dict(state.get("best_funder_info") or {})
    best_grant_review = _coerce_dict(state.get("best_grant_review") or {})
    return {
        "funder_info": best_funder_info or candidates[0],
        "grant_review": best_grant_review or state.get("grant_review"),
        "extract_duplicate_only": True,
        "refined_search_query": refined_query,
    }


def route_after_extract(state: GrantState) -> str:
    duplicate_only = bool(state.get("extract_duplicate_only"))
    if not duplicate_only:
        return "to_grant_review"

    attempts = int(state.get("search_attempts") or 0)
    cycles = int(state.get("search_cycles") or 0)
    has_best_review = bool(_coerce_dict(state.get("best_grant_review") or {}))

    if attempts >= MAX_SEARCH_RETRIES or cycles >= MAX_SEARCH_CYCLES:
        if has_best_review:
            return "to_pdf"
        # Extreme fallback: force one assessment so the flow can continue safely.
        return "to_grant_review"

    return "retry_search"


def grant_review_node(state: GrantState):
    _emit({"type": "node_start", "node": "grant_review", "label": "Reviewing grant relevance..."})
    project_details = state.get("project_details", "") or ""
    funder_info = _coerce_dict(state.get("funder_info") or {})
    attempts = min(int(state.get("search_attempts") or 0) + 1, MAX_SEARCH_RETRIES)
    original_query = (state.get("user_input") or "").strip()

    prompt = f"""
{_grant_reviewer_role()}
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
{_grant_profile_rules()}
- Use these score anchors:
  - 9-10: excellent fit with clear mission/priority overlap and actionable details.
  - 7-8: strong fit with minor gaps.
  - 5-6: moderate fit with notable gaps.
  - 3-4: weak fit.
  - 0-2: unusable or clearly mismatched.
- Return only raw JSON.
"""
    raw = reviewer_llm.invoke(prompt).content.strip()
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
    review["score"] = _clamp_score(review.get("score", 0))
    review["match_percent"] = _score_to_match_percent(review["score"])
    review["reasons"] = review.get("reasons") if isinstance(review.get("reasons"), list) else []
    review["improvement_notes"] = str(review.get("improvement_notes", "") or "").strip()
    halt_reason = None
    force_fail = False
        
    # TRUST GUARDRAILS
    # Hard fail: expired deadlines
    deadline_text = str(funder_info.get("deadline", "") or "").strip()
    if deadline_text and deadline_text.lower() not in {"none", "n/a", "unknown", "tbd"}:
        try:
            deadline_date = datetime.fromisoformat(deadline_text).date()
            today = datetime.now().date()
            if deadline_date < today:
                force_fail = True
                review["score"] = min(review["score"], 1)
                review["match_percent"] = _score_to_match_percent(review["score"])
                review["reasons"] = list(review["reasons"]) + [
                    f"Grant deadline is expired ({deadline_text})."
                ]
                if review["improvement_notes"]:
                    review["improvement_notes"] += " Search for active grants with future deadlines."
                else:
                    review["improvement_notes"] = "Search for active grants with future deadlines."
        except Exception:
            pass

    # Source quality penalty: stricter in strict profile, softer in demo profile.
    source_url = str(funder_info.get("source_url", "") or "").strip().lower()

    suspicious_domains = [
        "makewonder.com",
        "womenhack.com",
        "africanngos.org",
        "onboardmeetings.com",
        "grantedai.com",
    ]
    suspicious_path_terms = [
        "grant-opportunities",
        "roundup",
        "/blog/",
        "/news/",
    ]

    is_suspicious_source = False
    if source_url:
        if any(domain in source_url for domain in suspicious_domains):
            is_suspicious_source = True
        if any(term in source_url for term in suspicious_path_terms):
            is_suspicious_source = True

    if is_suspicious_source:
        review["score"] = min(review["score"], SUSPICIOUS_SOURCE_MAX_SCORE)
        review["match_percent"] = _score_to_match_percent(review["score"])
        review["reasons"] = list(review["reasons"]) + [
            "Source appears to be an aggregator, roundup, or non-official page rather than a primary funder source."
        ]
        if REVIEW_PROFILE == "strict":
            force_fail = True
        if review["improvement_notes"]:
            review["improvement_notes"] += " Prefer official funder or grant program pages."
        else:
            review["improvement_notes"] = "Prefer official funder or grant program pages."

    # Soft penalty: grant name and URL look weakly aligned
    # This is NOT a hard fail by itself because many official URLs are generic.
    funder_name = str(funder_info.get("funder_name", "") or "").lower().strip()
    if funder_name and source_url:
        name_tokens = [word for word in funder_name.split() if len(word) > 4]
        if name_tokens and not any(token in source_url for token in name_tokens):
            review["score"] = min(review["score"], NAME_URL_MISMATCH_MAX_SCORE)
            review["match_percent"] = _score_to_match_percent(review["score"])
            review["reasons"] = list(review["reasons"]) + [
                "Grant name does not clearly align with the source URL; verify that the page corresponds to the same opportunity."
            ]
            if review["improvement_notes"]:
                review["improvement_notes"] += " Double-check that the selected grant and source URL refer to the same opportunity."
            else:
                review["improvement_notes"] = "Double-check that the selected grant and source URL refer to the same opportunity."

    if not _is_verifiable_grant(funder_info):
        force_fail = True
        review["score"] = min(_clamp_score(review.get("score", 0)), 1)
        review["match_percent"] = _score_to_match_percent(review["score"])
        review["reasons"] = list(review["reasons"]) + [
            "Selected grant is not verifiable (missing named funder and/or valid source URL)."
        ]
        if not review["improvement_notes"]:
            review["improvement_notes"] = (
                "Search for a named grant program and include a direct source URL before proceeding."
            )

    review["score"] = _clamp_score(review.get("score", 0))
    review["match_percent"] = _score_to_match_percent(review["score"])
    review["verdict"] = "fail" if force_fail else ("pass" if review["score"] >= GRANT_PASS_SCORE else "fail")

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

    best_funder_info = _coerce_dict(state.get("best_funder_info") or {})
    best_grant_review = _coerce_dict(state.get("best_grant_review") or {})
    best_grant_attempt = int(state.get("best_grant_attempt") or 0)

    if _pick_better_grant_candidate(funder_info, review, best_funder_info, best_grant_review):
        best_funder_info = dict(funder_info)
        best_grant_review = dict(review)
        best_grant_attempt = attempts

    selected_funder_info = funder_info
    selected_review = review
    selected_attempt = attempts
    if next_action == "proceed" and review["verdict"] != "pass" and best_grant_review:
        selected_funder_info = best_funder_info or funder_info
        selected_review = best_grant_review
        selected_attempt = best_grant_attempt or attempts

    _emit(
        {
            "type": "review_attempt",
            "node": "grant_review",
            "attempt": attempts,
            "max_attempts": MAX_SEARCH_RETRIES,
            "grant_name": funder_info.get("funder_name", "Unknown Grant"),
            "verdict": review["verdict"],
            "score": review["score"],
            "match_percent": review["match_percent"],
            "reasons": review["reasons"],
            "improvement_notes": review["improvement_notes"],
            "next_action": next_action,
        }
    )

    selected_match = _coerce_match_percent(
        selected_review.get("match_percent"),
        selected_review.get("score", 0),
    )
    selected_name = str(selected_funder_info.get("funder_name", "Unknown Grant") or "Unknown Grant")
    selected_verifiable = _is_verifiable_grant(selected_funder_info)
    label = f"Grant review complete: {review['match_percent']}% match (attempt {attempts}/{MAX_SEARCH_RETRIES})"
    if next_action == "proceed" and review["verdict"] != "pass":
        label = (
            f"No passing grant found. Using best attempt {selected_attempt}: "
            f"{selected_match}% match ({selected_name})"
        )
    if next_action == "proceed" and not selected_verifiable:
        halt_reason = (
            "No verifiable grant opportunity was found after max search attempts. "
            "A named funder and valid source URL are required before generating proposal content."
        )
        label = "Grant search ended without a verifiable named funder; stopping for manual review"

    _emit(
        {
            "type": "node_done",
            "node": "grant_review",
            "label": label,
            "funder": selected_name,
            "verdict": selected_review.get("verdict", review["verdict"]),
            "score": selected_review.get("score", review["score"]),
            "match_percent": selected_match,
        }
    )
    return {
        "search_attempts": attempts,
        "grant_review": selected_review,
        "funder_info": selected_funder_info,
        "refined_search_query": refined_query,
        "best_funder_info": best_funder_info,
        "best_grant_review": best_grant_review,
        "best_grant_attempt": best_grant_attempt,
        "halt_reason": halt_reason,
    }


def route_after_grant_review(state: GrantState) -> str:
    if str(state.get("halt_reason", "") or "").strip():
        return "to_summary"

    review = _coerce_dict(state.get("grant_review") or {})
    verdict = _normalize_verdict(review.get("verdict"), default="fail")
    attempts = int(state.get("search_attempts") or 0)
    cycles = int(state.get("search_cycles") or 0)

    if verdict == "pass":
        return "to_pdf"
    if attempts < MAX_SEARCH_RETRIES and cycles < MAX_SEARCH_CYCLES:
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
    pdf_file_name = "Grant_Proposal_Submission.pdf"
    render_proposal_pdf(
        proposal_sections=full_proposal,
        output_path=pdf_file_name,
        title="Grant Proposal Submission",
        date_str=date_str,
    )

    formatted_text = build_proposal_plain_text(
        proposal_sections=full_proposal,
        date_str=date_str,
        title="Grant Proposal Submission",
    )

    label = "Proposal PDF saved"
    _emit({"type": "node_done", "node": "pdf", "label": label})

    return {
        "proposal_pdf_path": pdf_file_name,
        "proposal_text": formatted_text,
        "proposal_sections": {str(k): str(v) for k, v in full_proposal.items()},
    }


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
    proposal_sections = state.get("proposal_sections") or {}
    if not isinstance(proposal_sections, dict):
        proposal_sections = {}
    email_draft = state.get("email_draft", "") or ""
    proposal_payload = json.dumps(proposal_sections, indent=2)
    proposal_context = proposal_payload if proposal_sections else proposal_text

    prompt = f"""
{_content_reviewer_role()}
Evaluate whether the proposal and outreach email are appropriate for both:
1) the selected grant priorities, and
2) the nonprofit organization profile.

PROJECT DETAILS:
{project_details}

FUNDER INFO:
{json.dumps(funder_info, indent=2)}

PROPOSAL SECTIONS (canonical full text by section):
{proposal_context}

EMAIL DRAFT:
{email_draft}

Return exactly one JSON object with these keys:
- verdict: "pass" or "fail"
- score: integer from 0 to 10
- reasons: list of short strings
- improvement_notes: concise edits needed to pass

Rules:
{_content_profile_rules()}
- Assess section completeness using the full section map above. Do not claim "cut off" unless text clearly ends abruptly.
- Use these score anchors:
  - 9-10: excellent alignment and specificity.
  - 7-8: strong alignment with minor issues.
  - 5-6: acceptable alignment with clear improvement areas.
  - 3-4: weak alignment.
  - 0-2: poor fit or unusable.
- Return only raw JSON.
"""
    raw = reviewer_llm.invoke(prompt).content.strip()
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
    review["score"] = _clamp_score(review.get("score", 0))
    review["match_percent"] = _score_to_match_percent(review["score"])
    review["reasons"] = review.get("reasons") if isinstance(review.get("reasons"), list) else []
    review["improvement_notes"] = str(review.get("improvement_notes", "") or "").strip()
    review["verdict"] = "pass" if review["score"] >= CONTENT_PASS_SCORE else "fail"

    halt_reason = None
    if review["verdict"] == "pass":
        next_action = "proceed"
    elif attempts < MAX_CONTENT_RETRIES:
        next_action = "retry_pdf"
    else:
        next_action = "proceed_with_warning"

    if review["verdict"] != "pass" and attempts >= MAX_CONTENT_RETRIES:
        halt_reason = (
            "Content quality threshold was not reached after max retries. Outreach will continue, but manual review is strongly recommended."
        )

    _emit(
        {
            "type": "review_attempt",
            "node": "content_review",
            "attempt": attempts,
            "max_attempts": MAX_CONTENT_RETRIES,
            "grant_name": funder_info.get("funder_name", "Unknown Grant"),
            "verdict": review["verdict"],
            "score": review["score"],
            "match_percent": review["match_percent"],
            "reasons": review["reasons"],
            "improvement_notes": review["improvement_notes"],
            "next_action": next_action,
        }
    )

    _emit(
        {
            "type": "node_done",
            "node": "content_review",
            "label": f"Content review complete: {review['match_percent']}% match (attempt {attempts}/{MAX_CONTENT_RETRIES})",
            "verdict": review["verdict"],
            "score": review["score"],
            "match_percent": review["match_percent"],
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
    return "to_send"


def send_node(state: GrantState):
    _emit({"type": "node_start", "node": "send", "label": "Sending outreach email..."})
    try:
        nonprofit_confidence = str(state.get("nonprofit_confidence") or "unknown").lower()
        if nonprofit_confidence == "low":
            print(
                "SEND NODE WARNING:",
                {
                    "status": "warning",
                    "message": "No confident nonprofit record found in ProPublica. Proceeding with guardrailed test send only.",
                },
                flush=True,
            )
            _emit(
                {
                    "type": "node_done",
                    "node": "send",
                    "label": "Low-confidence nonprofit grounding; proceeding with guardrailed send",
                }
            )

        result = send_email.invoke(
            {
                "to_email": "ignored@example.com",
                "draft": state.get("email_draft", ""),
            }
        )

        if not isinstance(result, dict):
            result = {"status": str(result)}

        print("SEND NODE RESULT:", result, flush=True)

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
        print("SEND NODE ERROR:", repr(e), flush=True)
        _emit({"type": "node_done", "node": "send", "label": f"Email send error: {e}"})
        return {"email_result": {"status": "error", "message": str(e)}}


def calendar_node(state: GrantState):
    _emit({"type": "node_start", "node": "calendar", "label": "Adding grant deadline to Google Calendar..."})
    try:
        funder_info = _coerce_dict(state.get("funder_info") or {})
        grant_name = str(funder_info.get("funder_name", "") or "").strip() or "grant opportunity"
        deadline = funder_info.get("deadline")
        bad_deadlines = ["none", "n/a", "unknown", "tbd", ""]
        deadline_text = str(deadline).strip() if deadline is not None else ""
        deadline_reason = "no deadline was provided"
        if deadline_text:
            if deadline_text.lower() in bad_deadlines:
                deadline_reason = f"the deadline value was '{deadline_text}'"
            else:
                deadline_reason = f"the deadline value '{deadline_text}' was not a valid YYYY-MM-DD date"

        has_valid_deadline = False
        if deadline_text and deadline_text.lower() not in bad_deadlines:
            try:
                datetime.fromisoformat(deadline_text)
                has_valid_deadline = True
            except Exception:
                has_valid_deadline = False

        if has_valid_deadline:
            result = create_grant_deadline_event.invoke(
                {
                    "deadline_date": deadline_text,
                    "title": f"{grant_name} Deadline",
                    "application_url": "",
                }
            )
            _emit({"type": "node_done", "node": "calendar", "label": "Calendar deadline event created"})
            return {
                "calendar_event": (
                    f"Deadline event added to calendar for {grant_name} on {deadline_text}. "
                    f"{str(result)}"
                )
            }

        email_result = state.get("email_result") or {}
        if isinstance(email_result, str):
            email_result = {"status": email_result}
        if not isinstance(email_result, dict):
            email_result = {}

        if str(email_result.get("status", "")).strip().lower() != "sent":
            _emit({"type": "node_done", "node": "calendar", "label": "No deadline and no sent email; calendar skipped"})
            return {
                "calendar_event": (
                    "No calendar event was added. "
                    f"Reason: {deadline_reason}, and the outreach email was not sent yet."
                )
            }

        sent_at_raw = str(email_result.get("sent_at", "") or "").strip()
        sent_dt = datetime.now()
        if sent_at_raw:
            try:
                sent_dt = datetime.fromisoformat(sent_at_raw.replace("Z", "+00:00"))
            except Exception:
                sent_dt = datetime.now()

        follow_up_date = (sent_dt + timedelta(days=14)).date().isoformat()
        result = create_grant_deadline_event.invoke(
            {
                "deadline_date": follow_up_date,
                "title": f"Follow up with {grant_name}",
                "application_url": "",
                "description": (
                    f"No application deadline was available for {grant_name}.\n\n"
                    f"Schedule follow-up outreach two weeks after first email sent ({sent_dt.date().isoformat()})."
                ),
            }
        )
        _emit(
            {
                "type": "node_done",
                "node": "calendar",
                "label": f"No deadline found; follow-up event created for {follow_up_date}",
            }
        )
        return {
            "calendar_event": (
                f"Follow-up event added to calendar: 'Follow up with {grant_name}' on {follow_up_date}. "
                f"Reason: {deadline_reason}. {str(result)}"
            )
        }
    except Exception as e:
        _emit({"type": "node_done", "node": "calendar", "label": f"Calendar error: {e}"})
        return {"calendar_event": f"Failed to create event: {str(e)}"}

def summary_node(state: GrantState):
    _emit({"type": "node_start", "node": "summary", "label": "Building final summary..."})
    try:
        funder_info = _coerce_dict(state.get("funder_info") or {})
        grant_review = _coerce_dict(state.get("grant_review") or {})
        content_review = _coerce_dict(state.get("content_review") or {})
        propublica_result = _coerce_dict(state.get("propublica_result") or {})

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
        search_cycles = int(state.get("search_cycles") or 0)
        content_attempts = int(state.get("content_attempts") or 0)
        grant_verdict = _normalize_verdict(grant_review.get("verdict"), default="fail")
        grant_match = _coerce_match_percent(grant_review.get("match_percent"), grant_review.get("score", 0))
        content_match = _coerce_match_percent(content_review.get("match_percent"), content_review.get("score", 0))
        halt_reason = (state.get("halt_reason") or "").strip()
        email_status = email_result.get("status", "N/A")
        email_message = email_result.get("message", "")
        calendar_status = state.get("calendar_event", "N/A")

        nonprofit_confidence = str(state.get("nonprofit_confidence") or "unknown")
        nonprofit_match = str(propublica_result.get("matched_name") or "None")

        note = str(propublica_result.get("note", "")).lower()

        if nonprofit_confidence.lower() == "low" and "no confident nonprofit match found" in note and not halt_reason:
            halt_reason = (
                "No confident nonprofit record was found in ProPublica. "
                "Proceed with caution and manually verify the organization before real outreach."
            )

        summary = (
            "## Grant Summary\n\n"
            f"**Funder:** {funder_info.get('funder_name', 'N/A')}\n\n"
            f"**Source URL:** {funder_info.get('source_url', 'N/A')}\n\n"
            f"**Deadline:** {funder_info.get('deadline', 'N/A')}\n\n"
            f"**Summary:** {funder_info.get('summary', 'N/A')}\n\n"
            "---\n\n"
            "## Nonprofit Grounding\n\n"
            f"**Confidence:** {nonprofit_confidence}\n\n"
            f"**ProPublica Match:** {nonprofit_match}\n\n"
            f"**Note:** {propublica_result.get('note', 'N/A')}\n\n"
            "---\n\n"
            "## Review Gates\n\n"
            f"**Review Profile:** {REVIEW_PROFILE}\n\n"
            f"**Search Cycles:** {search_cycles}/{MAX_SEARCH_CYCLES}\n\n"
            f"**Grant Match:** {grant_match}% (attempts: {grant_attempts}/{MAX_SEARCH_RETRIES})\n\n"
            f"{_listify_text(grant_review.get('reasons'))}\n\n"
            f"**Grant Reviewer Notes:** {grant_review.get('improvement_notes', 'N/A')}\n\n"
            f"**Content Match:** {content_match}% (attempts: {content_attempts}/{MAX_CONTENT_RETRIES})\n\n"
            f"{_listify_text(content_review.get('reasons'))}\n\n"
            f"**Content Reviewer Notes:** {content_review.get('improvement_notes', 'N/A')}\n\n"
            f"**Grant Search Confidence Note:** {'Best available grant match selected after max search attempts.' if grant_verdict != 'pass' and grant_attempts >= MAX_SEARCH_RETRIES else ('Grant match accepted on the final search attempt.' if grant_verdict == 'pass' and grant_attempts >= MAX_SEARCH_RETRIES else 'Grant match selected before reaching max search attempts.')}\n\n"
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
            + (f"**Message:** {email_message}\n\n" if email_message else "")
            + "---\n\n"
            + "## Calendar\n\n"
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
builder.add_node("propublica", propublica_node)
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
builder.add_edge("parse", "propublica")
builder.add_edge("propublica", "search")
builder.add_edge("search", "extract")
builder.add_conditional_edges(
    "extract",
    route_after_extract,
    {"to_grant_review": "grant_review", "retry_search": "search", "to_pdf": "pdf"},
)
builder.add_conditional_edges(
    "grant_review",
    route_after_grant_review,
    {"retry_search": "search", "to_pdf": "pdf", "to_summary": "summary"},
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
