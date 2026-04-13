from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from difflib import SequenceMatcher
from typing import Any, Dict, List


PROPUBLICA_SEARCH_URL = "https://projects.propublica.org/nonprofits/api/v2/search.json"


def _normalize_name(name: str) -> str:
    text = (name or "").strip().lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\b(inc|llc|ltd|foundation|corp|corporation|association|nonprofit|organization)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _name_similarity(a: str, b: str) -> float:
    na = _normalize_name(a)
    nb = _normalize_name(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    if na in nb or nb in na:
        return 0.92
    return SequenceMatcher(None, na, nb).ratio()


def _confidence_from_score(score: float) -> str:
    if score >= 0.9:
        return "high"
    if score >= 0.72:
        return "medium"
    return "low"


def search_propublica_nonprofit(org_name: str) -> Dict[str, Any]:
    """
    Search ProPublica Nonprofit Explorer for a likely nonprofit match.
    Returns a structured grounding result for the org.
    """
    query = (org_name or "").strip()
    if not query:
        return {
            "input_name": org_name,
            "match_found": False,
            "matched_name": None,
            "confidence": "low",
            "score": 0.0,
            "ein": None,
            "city": None,
            "state": None,
            "source": "ProPublica Nonprofit Explorer",
            "note": "No organization name provided.",
        }

    params = urllib.parse.urlencode({"q": query})
    url = f"{PROPUBLICA_SEARCH_URL}?{params}"

    try:
        with urllib.request.urlopen(url, timeout=12) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as e:
        return {
            "input_name": query,
            "match_found": False,
            "matched_name": None,
            "confidence": "low",
            "score": 0.0,
            "ein": None,
            "city": None,
            "state": None,
            "source": "ProPublica Nonprofit Explorer",
            "note": f"Lookup failed: {e}",
        }

    orgs: List[Dict[str, Any]] = payload.get("organizations", []) or []
    if not orgs:
        return {
            "input_name": query,
            "match_found": False,
            "matched_name": None,
            "confidence": "low",
            "score": 0.0,
            "ein": None,
            "city": None,
            "state": None,
            "source": "ProPublica Nonprofit Explorer",
            "note": "No ProPublica nonprofit match found.",
        }

    best = None
    best_score = -1.0

    for org in orgs[:5]:
        candidate_name = str(org.get("name", "") or "").strip()
        sub_name = str(org.get("sub_name", "") or "").strip()
        combined = f"{candidate_name} {sub_name}".strip()
        score = max(
            _name_similarity(query, candidate_name),
            _name_similarity(query, combined),
        )
        if score > best_score:
            best_score = score
            best = org

    confidence = _confidence_from_score(best_score)

    return {
        "input_name": query,
        "match_found": confidence in {"high", "medium"},
        "matched_name": str(best.get("name", "") or "").strip() if best else None,
        "sub_name": str(best.get("sub_name", "") or "").strip() if best else None,
        "confidence": confidence,
        "score": round(best_score, 3),
        "ein": str(best.get("ein")) if best and best.get("ein") is not None else None,
        "city": str(best.get("city", "") or "").strip() if best else None,
        "state": str(best.get("state", "") or "").strip() if best else None,
        "ntee_code": str(best.get("ntee_code", "") or "").strip() if best else None,
        "source": "ProPublica Nonprofit Explorer",
        "note": (
            "Confident nonprofit match found."
            if confidence == "high"
            else "Possible nonprofit match found."
            if confidence == "medium"
            else "No confident nonprofit match found."
        ),
    }