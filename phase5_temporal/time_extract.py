"""
============================================================
Phase 5.2 — Time-constraint extraction for temporal retrieval
============================================================

Given a natural-language question, extract:
  - time windows  (list of [start, end) half-open ISO date pairs)
  - edge types    (which lifecycle interval the question is about:
                   approved_at / shipped_at / delivered_at /
                   reviewed_at — or "any" if unspecified)

Two paths, tried in order:
  1. LLM extraction (main) — a small cached gpt-4o-mini call that
     returns structured JSON. This is the path used at inference
     time on arbitrary questions; it is the same approach TG-RAG
     and other temporal-RAG systems use.
  2. Rule-based fallback — regex for explicit quarter / month /
     year / half-year expressions. Catches the cases where the
     LLM returns nothing, and provides a zero-cost path for the
     common canonical phrasings.

If BOTH return nothing, the caller (System C) degrades gracefully
to plain entity retrieval — i.e. the temporal layer simply adds
nothing, rather than breaking.

The gold `time_range` / `time_anchors` fields on benchmark
questions are NEVER read here — they are used only for offline
validation of extraction accuracy (see diagnostics). Reading them
at inference time would be leaking the gold annotation.
============================================================
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# temporal_index helpers (quarter_window / month_window) live one
# level up in phase5_temporal/.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from temporal_index import quarter_window, month_window  # noqa: E402


VALID_EDGE_TYPES = {
    "approved_at", "shipped_at", "delivered_at", "reviewed_at", "any",
}


@dataclass
class TimeConstraint:
    """One extracted temporal constraint."""
    windows: list[tuple[str, str]] = field(default_factory=list)  # [start,end)
    edge_types: list[str] = field(default_factory=list)
    source: str = "none"   # "llm" | "rule" | "none"

    @property
    def is_empty(self) -> bool:
        return len(self.windows) == 0


# ----------------------------------------------------------------
# LLM extraction prompt
# ----------------------------------------------------------------

_EXTRACT_SYSTEM = """\
You extract temporal constraints from a business analytics question \
about an e-commerce order dataset (Olist, Brazil, Sep 2016 - Oct 2018).

Return STRICT JSON with exactly these keys:
{
  "windows": [["YYYY-MM-DD", "YYYY-MM-DD"], ...],
  "edge_types": ["approved_at" | "shipped_at" | "delivered_at" | "reviewed_at" | "any", ...]
}

Rules:
- Each window is half-open [start, end): a full year 2017 is
  ["2017-01-01", "2018-01-01"]; Q4 2017 is ["2017-10-01", "2018-01-01"];
  July 2017 is ["2017-07-01", "2017-08-01"].
- If the question compares two periods (e.g. Q1 vs Q4), return BOTH
  windows in order.
- edge_types: which order-lifecycle stage the question is about.
  "delivered_at" for delivery-time / purchase-to-delivery questions;
  "approved_at" for payment-approval-lag questions; "shipped_at" for
  carrier-handover; "reviewed_at" for review-timing. Use "any" if the
  question is about order placement / volume / generic activity in a
  period without referencing a specific lifecycle stage.
- If the question has NO time reference at all, return
  {"windows": [], "edge_types": []}.
- Output ONLY the JSON object. No prose, no markdown.
"""


def _llm_extract(question: str, llm) -> TimeConstraint | None:
    """Call the LLM; return a TimeConstraint or None on failure."""
    try:
        resp = llm.chat(
            system=_EXTRACT_SYSTEM,
            user=f"Question: {question}",
            response_format={"type": "json_object"},
        )
    except Exception:
        return None

    text = resp.get("text", "")
    if not text or text.startswith("[DRY RUN"):
        # Dry-run or empty — let the caller fall back to rules.
        return None

    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None

    windows_raw = data.get("windows", [])
    edge_types_raw = data.get("edge_types", [])

    windows: list[tuple[str, str]] = []
    for w in windows_raw:
        if (isinstance(w, (list, tuple)) and len(w) == 2
                and _is_iso_date(w[0]) and _is_iso_date(w[1])
                and w[0] < w[1]):
            windows.append((w[0], w[1]))

    edge_types = [e for e in edge_types_raw if e in VALID_EDGE_TYPES]

    if not windows:
        return None
    return TimeConstraint(windows=windows, edge_types=edge_types or ["any"],
                          source="llm")


def _is_iso_date(s) -> bool:
    return isinstance(s, str) and bool(
        re.match(r"^\d{4}-\d{2}-\d{2}$", s)
    )


# ----------------------------------------------------------------
# Rule-based fallback
# ----------------------------------------------------------------

# Map lifecycle keywords in the question to edge types. Order
# matters: more specific phrases first.
_EDGE_KEYWORDS = [
    ("delivered_at", ["delivery time", "purchase-to-delivery",
                      "purchase to delivery", "delivery lag",
                      "time to deliver", "delivered"]),
    ("approved_at",  ["approval lag", "payment-approval",
                      "payment approval", "purchase-to-approval",
                      "approved"]),
    ("shipped_at",   ["carrier", "shipped", "handover", "shipping"]),
    ("reviewed_at",  ["review", "reviewed", "rating"]),
]

_MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
    "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _rule_extract(question: str) -> TimeConstraint | None:
    q = question.lower()
    windows: list[tuple[str, str]] = []

    # Quarter expressions: "Q4 2017", "2017-Q4", "q1 2018"
    for m in re.finditer(r"\bq([1-4])[ \-/]?(\d{4})\b", q):
        windows.append(quarter_window(f"{m.group(2)}-Q{m.group(1)}"))
    for m in re.finditer(r"\b(\d{4})[ \-]?q([1-4])\b", q):
        windows.append(quarter_window(f"{m.group(1)}-Q{m.group(2)}"))

    # Half-year: "first half of 2017", "second half of 2017", "2017-H1",
    # and the parenthetical "(Jan-Jun)" / "(Jul-Dec)" style with a
    # nearby year.
    for m in re.finditer(r"\b(\d{4})[ \-]?h([12])\b", q):
        y, h = int(m.group(1)), int(m.group(2))
        if h == 1:
            windows.append((f"{y}-01-01", f"{y}-07-01"))
        else:
            windows.append((f"{y}-07-01", f"{y+1}-01-01"))
    for m in re.finditer(
        r"\b(first|second) half\b", q
    ):
        # find the nearest 4-digit year anywhere in the question
        yr = re.search(r"\b(20\d{2})\b", q)
        if yr:
            y = int(yr.group(1))
            if m.group(1) == "first":
                windows.append((f"{y}-01-01", f"{y}-07-01"))
            else:
                windows.append((f"{y}-07-01", f"{y+1}-01-01"))
    # Parenthetical month ranges as half-year proxies: "(Jan-Jun)" /
    # "(Jul-Dec)". Olist trend questions use these to denote H1/H2.
    if re.search(r"jan\w*\s*[-–]\s*jun", q):
        yr = re.search(r"\b(20\d{2})\b", q)
        if yr:
            y = int(yr.group(1))
            windows.append((f"{y}-01-01", f"{y}-07-01"))
    if re.search(r"jul\w*\s*[-–]\s*dec", q):
        yr = re.search(r"\b(20\d{2})\b", q)
        if yr:
            y = int(yr.group(1))
            windows.append((f"{y}-07-01", f"{y+1}-01-01"))

    # Month-name + year: "July 2017"
    for m in re.finditer(
        r"\b(" + "|".join(_MONTH_NAMES) + r")[ ,]+(\d{4})\b", q
    ):
        mo = _MONTH_NAMES[m.group(1)]
        y = int(m.group(2))
        windows.append(month_window(f"{y}-{mo:02d}"))

    # Numeric month: "2017-07"
    for m in re.finditer(r"\b(\d{4})-(\d{2})\b", q):
        mo = int(m.group(2))
        if 1 <= mo <= 12:
            windows.append(month_window(f"{m.group(1)}-{mo:02d}"))

    # Quarter named once, applied across two years: "Q2 ... from
    # 2017 to 2018" → 2017-Q2 and 2018-Q2. Only fires if a bare
    # quarter token (not already paired with a year) co-occurs with
    # a "YYYY to YYYY" range.
    if not windows:
        bare_q = re.search(r"\bq([1-4])\b", q)
        yr_range = re.search(r"\b(20\d{2})\b.{0,20}?\bto\b.{0,5}?\b(20\d{2})\b", q)
        if bare_q and yr_range:
            qn = bare_q.group(1)
            y1, y2 = yr_range.group(1), yr_range.group(2)
            windows.append(quarter_window(f"{y1}-Q{qn}"))
            windows.append(quarter_window(f"{y2}-Q{qn}"))

    # Bare year-to-year range without a quarter: "from 2017 to 2018"
    # → two full-year windows.
    if not windows:
        m = re.search(r"\bfrom\s+(20\d{2})\s+to\s+(20\d{2})\b", q)
        if m:
            y1, y2 = int(m.group(1)), int(m.group(2))
            windows.append((f"{y1}-01-01", f"{y1+1}-01-01"))
            windows.append((f"{y2}-01-01", f"{y2+1}-01-01"))

    # Bare year (only if no finer window already captured it):
    # "in 2017", "during 2018"
    if not windows:
        for m in re.finditer(r"\b(in|during|for|of)\s+(\d{4})\b", q):
            y = int(m.group(2))
            if 2016 <= y <= 2018:
                windows.append((f"{y}-01-01", f"{y+1}-01-01"))

    # De-duplicate while preserving order
    seen = set()
    deduped = []
    for w in windows:
        if w not in seen:
            seen.add(w)
            deduped.append(w)

    if not deduped:
        return None

    # Edge-type detection
    edge_types = []
    for et, kws in _EDGE_KEYWORDS:
        if any(kw in q for kw in kws):
            edge_types.append(et)
    if not edge_types:
        edge_types = ["any"]

    return TimeConstraint(windows=deduped, edge_types=edge_types,
                          source="rule")


# ----------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------

def extract_time_constraint(
    question: str, llm=None, prefer: str = "llm",
) -> TimeConstraint:
    """Extract a temporal constraint, LLM-first with rule fallback.

    Parameters
    ----------
    question
        The user question.
    llm
        A ChatLLM instance (or None — then only the rule path runs).
    prefer
        "llm" (default): try LLM first, fall back to rules.
        "rule": rules only (deterministic, no API call).

    Returns
    -------
    A TimeConstraint. May be empty (.is_empty) if no time reference
    was found by either path.
    """
    if prefer == "llm" and llm is not None:
        tc = _llm_extract(question, llm)
        if tc is not None and not tc.is_empty:
            return tc

    tc = _rule_extract(question)
    if tc is not None:
        return tc

    return TimeConstraint(source="none")
