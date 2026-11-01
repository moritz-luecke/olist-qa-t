"""
============================================================
Phase 4.5 — Pilot run: Vanilla v2 against temporal pilot Qs
============================================================

Smoke-test of the freshly-rebuilt v2 chunks + embeddings against
the 30 temporal pilot questions (10 trend + 10 drift + 10 latency).
Designed as a *fast feedback loop* before investing in the
temporal Graph-RAG layer:

  - Confirms Vanilla v2 retrieval works end-to-end (chunks, embeds,
    LLM call, evidence formatting).
  - Gives the FIRST real datapoint for H2 (the temporal-faithfulness
    gap): how often does Vanilla v2 hallucinate or fail to match
    on closed-form temporal questions where the gold answer is known?
  - Reality-checks the embedding-anisotropy hunch from
    notes/embedding_anisotropy.md: does Vanilla v2 actually fail on
    sub-year-discriminating questions (Q1 vs Q4) while passing on
    year-anchored aggregations?

What this is NOT
----------------
  - Not an experiment for the headline paper table — we'll re-run
    everything against the full benchmark, all 7 systems, with the
    LLM-as-judge against open-form rubrics.
  - Not a faithfulness-judge run — those come later when the
    judge prompt is itself validated.

Auto-scoring (no LLM-judge needed)
----------------------------------
All 30 pilot questions are closed-form (dict / list / number).
We compare the model's text answer to the gold answer with three
simple matchers:

  match_number    — extract first numeric value from model text,
                     compare to gold within ±relative_tolerance.
  match_dict      — model must mention each gold key (e.g.
                     "2017-Q1") AND its value within tolerance.
  match_list      — model must mention each gold list item
                     (order-insensitive for set match,
                     order-sensitive for ranked sub_type='topk_change').

This is intentionally permissive — we are NOT trying to get a
final accuracy number, only a coarse signal of "does it fail
where we predict it would fail?".

Output
------
  outputs/results/pilot_vanilla_v2_answers.jsonl   per-question answers
  outputs/results/pilot_vanilla_v2_scored.jsonl    + match verdicts
  outputs/results/pilot_vanilla_v2_summary.md      human-readable report

Cost estimate
-------------
  30 questions × 1 generation each
  ~ 30 × $0.001  =  ~$0.03   (gpt-4o-mini, cached embeddings)

Usage
-----
  python phase4_retrieval/pilot_run_vanilla_v2.py            # DRY by default
  python phase4_retrieval/pilot_run_vanilla_v2.py --real     # actually call API
============================================================
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

# --- sys.path setup ----------------------------------------------
# This script lives at code/phase4_retrieval/pilot_run_vanilla_v2.py
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))                    # phase4_retrieval/
sys.path.insert(0, str(_HERE.parent / "phase2_kg"))   # for build_kg

from config import Paths, RetrievalConfig          # noqa: E402
from embeddings import Embedder                    # noqa: E402
from llm import ChatLLM                            # noqa: E402
from pipeline import Pipeline                      # noqa: E402
from retrievers.vanilla_rag import VanillaRAG      # noqa: E402


# ----------------------------------------------------------------
# Path overrides for v2 artefacts
# ----------------------------------------------------------------

def make_v2_paths(project_root: Path) -> Paths:
    """A Paths object pointed at the v2 chunks / embeds / pilot
    questions, leaving the rest of the defaults intact."""
    p = Paths(
        data_dir=project_root / "data",
        chunks_path=project_root / "outputs" / "vanilla_chunks_v2.jsonl",
        chunk_embeds_path=project_root / "outputs" /
                          "vanilla_chunk_embeds_v2.npy",
        pilot_questions=project_root / "outputs" /
                        "temporal_pilot_questions.jsonl",
        results_dir=project_root / "outputs" / "results",
        cache_dir=project_root / "outputs" / "cache",
    )
    return p


# ================================================================
# Auto-scoring helpers
# ================================================================

# Strip thousands separators / currency symbols / units, then pull
# numeric values from text. Returns floats.
_NUM_RE = re.compile(
    r"-?\d{1,3}(?:[,_]\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?"
)


def extract_numbers(text: str) -> list[float]:
    """Return all numeric tokens in text, in order. Robust to common
    formats: '1,234.56', '12.3', '0.95', '5%'.
    """
    nums = []
    for m in _NUM_RE.findall(text):
        s = m.replace(",", "").replace("_", "")
        try:
            nums.append(float(s))
        except ValueError:
            continue
    return nums


def near(a: float, b: float, rel_tol: float = 0.10,
         min_abs_tol: float = 0.05) -> bool:
    """Compare two floats with both relative and absolute tolerance.

    Tolerance band per gold value:
      relative: 10% of max(|a|, |b|)
      absolute: max(min_abs_tol, rel_tol * |gold|)

    The earlier version used a fixed abs_tol=0.5 which caused
    false positives on sub-day latencies (gold=0.41 matched a
    model answer of 0 because 0.41 < 0.5). The new rule scales
    the absolute floor proportionally to the gold value, with a
    small protective floor of 0.05 for the edge case where gold
    is zero or tiny.
    """
    if a is None or b is None:
        return False
    diff = abs(a - b)
    scale = max(abs(a), abs(b), 1e-9)
    rel_band = rel_tol * scale
    abs_band = max(min_abs_tol, rel_tol * abs(b))  # b is the gold side
    return diff <= max(rel_band, abs_band)


def _normalize_for_match(s: str) -> str:
    """Lowercase + strip diacritics (so 'São Paulo' matches 'sao paulo').

    Also folds underscores to spaces so dict keys like 'credit_card'
    match a model that writes 'Credit Card' (drift_09 payment-method
    keys). This is safe: it only ever merges an underscore and a space
    that already denote the same token; it cannot match two genuinely
    different strings.
    """
    import unicodedata
    s = s.lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("_", " ")
    return s


def find_anchor_string(text: str, anchor: str) -> bool:
    """Does the answer text mention a given anchor string?

    Forgiving on quarter variants: '2017-Q1' ↔ '2017 Q1' ↔ 'Q1 2017'.
    Month: '2017-07' ↔ 'July 2017' ↔ 'Jul 2017' ↔ '2017-07'.
    Accent-insensitive: 'São Paulo' ↔ 'sao paulo'.
    """
    t = _normalize_for_match(text).replace("–", "-").replace("—", "-")
    a = _normalize_for_match(anchor)

    if a in t:
        return True

    # Quarter normalisations
    m = re.match(r"(\d{4})-q([1-4])$", a)
    if m:
        y, q = m.groups()
        variants = [
            f"{y} q{q}",
            f"q{q} {y}",
            f"q{q}/{y}",
            f"{y}q{q}",
        ]
        return any(v in t for v in variants)

    # Half-year normalisations
    m = re.match(r"(\d{4})-h([12])$", a)
    if m:
        y, h = m.groups()
        half_word = "first half" if h == "1" else "second half"
        return any(v in t for v in [
            f"{y} h{h}", f"h{h} {y}", f"{half_word} of {y}",
            f"first half of {y}" if h == "1" else f"second half of {y}",
        ])

    # Month normalisations
    m = re.match(r"(\d{4})-(\d{2})$", a)
    if m:
        y, mm = m.groups()
        month_names = {
            "01": ["january", "jan"], "02": ["february", "feb"],
            "03": ["march", "mar"], "04": ["april", "apr"],
            "05": ["may"], "06": ["june", "jun"], "07": ["july", "jul"],
            "08": ["august", "aug"], "09": ["september", "sep", "sept"],
            "10": ["october", "oct"], "11": ["november", "nov"],
            "12": ["december", "dec"],
        }
        for mn in month_names.get(mm, []):
            if f"{mn} {y}" in t or f"{mn}, {y}" in t or f"{y}-{mm}" in t:
                return True
    return False


def match_number(answer_text: str, gold: float,
                 rel_tol: float = 0.10, abs_tol: float | None = None) -> dict:
    """Closest numeric value in text wins. If abs_tol is given (e.g. a
    per-question score_tol for review scores on the 1-5 scale), match
    iff |closest - gold| <= abs_tol; otherwise the relative near() rule.
    """
    nums = extract_numbers(answer_text)
    if not nums:
        return {"match": False, "reason": "no number found",
                "closest": None}
    closest = min(nums, key=lambda x: abs(x - gold))
    ok = (abs(closest - gold) <= abs_tol) if abs_tol is not None \
        else near(closest, gold, rel_tol=rel_tol)
    return {"match": bool(ok),
            "reason": f"closest={closest} vs gold={gold} "
                      f"(abs_tol={abs_tol})" if abs_tol is not None
                      else f"closest={closest} vs gold={gold}",
            "closest": closest}


_PERIOD_KEY_RE = re.compile(r"^(\d{4})-(Q[1-4]|H[12]|\d{2})$", re.IGNORECASE)

_MONTH_BARE = {
    "01": ["january", "jan"], "02": ["february", "feb"],
    "03": ["march", "mar"], "04": ["april", "apr"], "05": ["may"],
    "06": ["june", "jun"], "07": ["july", "jul"], "08": ["august", "aug"],
    "09": ["september", "sep", "sept"], "10": ["october", "oct"],
    "11": ["november", "nov"], "12": ["december", "dec"],
}


def _gold_keys_single_year(gold: dict) -> bool:
    """True iff every gold key is a period token (YYYY-Qn / YYYY-Hn /
    YYYY-MM) and they all share the same year. In that case the model
    may legitimately drop the redundant year ('Q1' instead of
    '2017-Q1'), since the question already scopes the year, and we
    accept the bare token. If keys span multiple years (e.g. a
    Q2-2017-vs-Q2-2018 comparison) we keep requiring the year, because
    a bare 'Q2' would then be genuinely ambiguous.
    """
    years = set()
    for k in gold:
        m = _PERIOD_KEY_RE.match(str(k))
        if not m:
            return False
        years.add(m.group(1))
    return len(years) == 1


def _bare_period_match(text: str, key: str) -> bool:
    """Does `text` mention the bare period token of `key` (year
    dropped), with a word boundary so 'Q1' does not match inside
    another token? Only called when the gold keys are all in one year.
    """
    m = _PERIOD_KEY_RE.match(str(key))
    if not m:
        return False
    t = _normalize_for_match(text)
    suffix = m.group(2).lower()
    if suffix.startswith("q"):                      # quarter: 'q4'
        return re.search(rf"\b{suffix}\b", t) is not None
    if suffix.startswith("h"):                      # half-year
        half = "first half" if suffix == "h1" else "second half"
        return half in t
    # numeric month → accept the bare month NAME only (never the bare
    # number, which would match countless other figures).
    for name in _MONTH_BARE.get(suffix, []):
        if re.search(rf"\b{name}\b", t) is not None:
            return True
    return False


def match_dict(answer_text: str, gold: dict,
               rel_tol: float = 0.10, abs_tol: float | None = None) -> dict:
    """For each (key, value) pair in gold: key must be mentioned,
    AND a number in the answer must be near the gold value.

    Each gold value must be matched by a DISTINCT number token in the
    answer (maximum bipartite matching between gold values and the
    in-tolerance answer tokens, computed over the whole gold dict
    including nested dicts).

    Bug fix 2026-06-11: previously each gold value independently took
    the closest token, so a single token within tolerance of two
    different gold values — e.g. 16.3 against golds 15.05 and 17.77
    at 10 % relative tolerance — satisfied both, and an answer that
    explicitly declined to provide one of the two values could still
    score a full match (qid latency_27 in the 90-question run).
    Maximum matching (rather than greedy assignment) keeps genuinely
    correct answers correct when tolerance windows overlap: each
    token is used at most once, duplicates in the answer text count
    as separate tokens.
    """
    nums = extract_numbers(answer_text)
    found_keys: list = []
    missed_keys: list = []
    wrong_values: list = []
    pending: list = []          # (key-path, gold value) numeric leaves
    matched = True

    def _collect(g: dict, path: str) -> None:
        nonlocal matched
        allow_bare = _gold_keys_single_year(g)
        for k, v in g.items():
            kp = f"{path}.{k}" if path else str(k)
            k_in = find_anchor_string(answer_text, k)
            if not k_in and allow_bare:
                k_in = _bare_period_match(answer_text, k)
            if not k_in:
                missed_keys.append(kp)
                matched = False
                continue
            found_keys.append(kp)

            # list value (e.g. top-k seller IDs): string membership
            if isinstance(v, list):
                misses = [item for item in v
                          if str(item).lower() not in answer_text.lower()]
                if misses:
                    wrong_values.append({"key": kp, "expected": v,
                                         "missing_items": misses})
                    matched = False
                continue
            if isinstance(v, dict):
                _collect(v, kp)          # shared token pool
                continue
            if v is None:
                continue
            pending.append((kp, float(v)))

    _collect(gold, "")

    def _ok(x: float, v: float) -> bool:
        return (abs(x - v) <= abs_tol) if abs_tol is not None \
            else near(x, v, rel_tol=rel_tol)

    # adjacency: gold leaf i -> token indices within tolerance,
    # closest first (helps Kuhn find a good matching quickly)
    adj = [sorted((j for j, x in enumerate(nums) if _ok(x, v)),
                  key=lambda j: abs(nums[j] - v))
           for _, v in pending]

    token_of: dict[int, int] = {}        # token index -> gold index

    def _augment(i: int, seen: set[int]) -> bool:
        for j in adj[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in token_of or _augment(token_of[j], seen):
                token_of[j] = i
                return True
        return False

    gold_of = {}                          # gold index -> token index
    for i in range(len(pending)):
        if _augment(i, set()):
            pass
    for j, i in token_of.items():
        gold_of[i] = j

    for i, (kp, v) in enumerate(pending):
        if i in gold_of:
            continue
        matched = False
        free = [x for j, x in enumerate(nums) if j not in token_of]
        if not free:
            wrong_values.append({"key": kp, "expected": v,
                                 "reason": "no unassigned number in answer"})
        else:
            closest = min(free, key=lambda x: abs(x - v))
            wrong_values.append({"key": kp, "expected": v,
                                 "closest_in_answer": closest})

    return {"match": matched,
            "found_keys": found_keys,
            "missed_keys": missed_keys,
            "wrong_values": wrong_values}


def match_list(answer_text: str, gold: list,
               ordered: bool = False) -> dict:
    """List-membership matcher.

    ordered=False (default): every gold item must appear in the text,
       any order.
    ordered=True: items must appear in the same relative order in the
       answer text as in the gold list.

    Accent-insensitive (e.g. 'São Paulo' matches 'sao paulo').
    """
    t = _normalize_for_match(answer_text)
    found = []
    missed = []
    last_pos = -1
    order_ok = True
    for item in gold:
        s = _normalize_for_match(str(item))
        pos = t.find(s)
        if pos < 0:
            missed.append(item)
            continue
        found.append(item)
        if ordered:
            if pos < last_pos:
                order_ok = False
            last_pos = pos
    matched = (len(missed) == 0) and (not ordered or order_ok)
    return {"match": matched, "found": found, "missed": missed,
            "ordered_ok": order_ok}


def hallucinated_time_anchors(answer_text: str,
                              gold_anchors: list[str],
                              time_range: dict | None = None,
                              spillover_months: int = 2) -> list[str]:
    """Find time anchors mentioned in the answer that are NOT in the
    gold list. Cheap proxy for the temporal-faithfulness metric of §3.6.

    Lifecycle spillover: an order *placed* inside the question's window
    can be delivered/reviewed shortly AFTER the window ends (a late-
    December purchase delivers in January). So a month/quarter in the
    `spillover_months` immediately following time_range["end"] is NOT a
    hallucinated anchor — it is a legitimate downstream lifecycle date.
    This removes the false positive on entity-routed latency questions
    (e.g. latency_01: a Dec-2017 order delivered 2018-01).

    Patterns matched:
      - 'YYYY-Qn'           '2017-Q3', '2017 Q3', 'Q3 2017'
      - 'YYYY-MM'           '2017-07'
      - 'YYYY-Hn'           '2017-H1'
      - month name + year   'July 2017'
    """
    t = answer_text
    found: set[str] = set()

    # Year-Quarter forms
    for m in re.finditer(r"\b(\d{4})[ \-/]?Q([1-4])\b", t, re.IGNORECASE):
        found.add(f"{m.group(1)}-Q{m.group(2)}")
    for m in re.finditer(r"\bQ([1-4])[ \-/]?(\d{4})\b", t, re.IGNORECASE):
        found.add(f"{m.group(2)}-Q{m.group(1)}")

    # YYYY-MM (numeric month)
    for m in re.finditer(r"\b(\d{4})[-/](\d{2})\b", t):
        y, mm = m.group(1), m.group(2)
        if 1 <= int(mm) <= 12 and 2014 <= int(y) <= 2020:
            found.add(f"{y}-{mm}")

    # YYYY-Hn (half year)
    for m in re.finditer(r"\b(\d{4})[ \-]?H([12])\b", t, re.IGNORECASE):
        found.add(f"{m.group(1)}-H{m.group(2)}")

    # Month-name + year, e.g. 'July 2017'
    month_to_num = {
        "january": "01", "february": "02", "march": "03", "april": "04",
        "may": "05", "june": "06", "july": "07", "august": "08",
        "september": "09", "october": "10", "november": "11", "december": "12",
        "jan": "01", "feb": "02", "mar": "03", "apr": "04", "jun": "06",
        "jul": "07", "aug": "08", "sep": "09", "sept": "09",
        "oct": "10", "nov": "11", "dec": "12",
    }
    for m in re.finditer(
        r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
        r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|"
        r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)[ ,]+(\d{4})\b",
        t, re.IGNORECASE,
    ):
        name = m.group(1).lower()
        y = m.group(2)
        mm = month_to_num[name]
        found.add(f"{y}-{mm}")

    gold_norm = {a.upper().replace(" ", "") for a in gold_anchors}
    # Also accept that quarter gold "2017-Q3" covers any month within Q3
    quarter_months = {}
    for a in gold_anchors:
        m = re.match(r"(\d{4})-Q([1-4])$", a, re.IGNORECASE)
        if m:
            y, q = m.group(1), int(m.group(2))
            for mo in range(3*(q-1)+1, 3*q+1):
                quarter_months[f"{y}-{mo:02d}".upper()] = a.upper()
    # And half-year gold covers any month within
    half_months = {}
    # ...as well as the two quarters within (a model fed quarter buckets
    # for a half-year question legitimately names Q1/Q2 for H1, etc.;
    # these are not spurious — they are inside the gold half-year).
    half_quarters = {}
    for a in gold_anchors:
        m = re.match(r"(\d{4})-H([12])$", a, re.IGNORECASE)
        if m:
            y, h = m.group(1), int(m.group(2))
            months = range(1, 7) if h == 1 else range(7, 13)
            for mo in months:
                half_months[f"{y}-{mo:02d}".upper()] = a.upper()
            quarters = (1, 2) if h == 1 else (3, 4)
            for q in quarters:
                half_quarters[f"{y}-Q{q}".upper()] = a.upper()

    # Lifecycle-spillover allowance: months/quarters within
    # [window_start, window_end + spillover_months) are legitimate.
    spillover_allowed: set[str] = set()
    if time_range and time_range.get("start") and time_range.get("end"):
        try:
            sy, sm = int(time_range["start"][:4]), int(time_range["start"][5:7])
            ey, em = int(time_range["end"][:4]), int(time_range["end"][5:7])
            # end is exclusive; extend by spillover_months
            total_end = ey * 12 + (em - 1) + spillover_months
            cur = sy * 12 + (sm - 1)
            while cur < total_end:
                yy, mm = divmod(cur, 12)
                mm += 1
                spillover_allowed.add(f"{yy}-{mm:02d}".upper())
                spillover_allowed.add(f"{yy}-Q{(mm - 1)//3 + 1}".upper())
                cur += 1
        except (ValueError, TypeError, KeyError):
            spillover_allowed = set()

    spurious = []
    for f in sorted(found):
        f_up = f.upper().replace(" ", "")
        if f_up in gold_norm:
            continue
        if f_up in quarter_months:
            continue
        if f_up in half_months:
            continue
        if f_up in half_quarters:
            continue
        if f_up in spillover_allowed:
            continue
        spurious.append(f)
    return spurious


# ================================================================
# Per-question scoring dispatcher
# ================================================================

def score_one(q: dict, answer_text: str) -> dict:
    """Return a dict with: match (bool), details, spurious_anchors.

    Honors two optional per-question fields:
      - score_tol: absolute numeric tolerance (e.g. 0.15 for review
        scores on the 1-5 scale, where the default 10% relative band
        is too permissive). Applies to number and dict gold.
      - time_range: {start, end} half-open window; used by the
        spurious-anchor detector for lifecycle-spillover allowance.
    """
    gold = q["gold_answer"]
    gtype = q["gold_answer_type"]
    stol = q.get("score_tol")

    if gtype == "number":
        m = match_number(answer_text, float(gold), abs_tol=stol)
    elif gtype == "dict":
        m = match_dict(answer_text, gold, abs_tol=stol)
    elif gtype == "list":
        # Drift drift_06 (set), drift_07/08/09/10 (ordered as topk)
        ordered = (q.get("sub_type") == "topk_change")
        m = match_list(answer_text, gold, ordered=ordered)
    elif gtype == "string":
        m = {"match": gold.lower() in answer_text.lower(),
             "details": "string-contain"}
    else:
        m = {"match": False, "details": f"unknown gold type: {gtype}"}

    spurious = hallucinated_time_anchors(
        answer_text, q.get("time_anchors", []),
        time_range=q.get("time_range"),
    )

    return {
        "match": bool(m.get("match", False)),
        "detail": m,
        "spurious_anchors": spurious,
        "n_spurious": len(spurious),
    }


# ================================================================
# Main
# ================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pilot run of Vanilla v2 on the temporal pilot Qs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--project-root", type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Project root (default: parent of phase4_retrieval/).",
    )
    parser.add_argument(
        "--real", action="store_true",
        help="Run real OpenAI calls (default: dry-run, no API).",
    )
    parser.add_argument(
        "--max-questions", type=int, default=None,
        help="Cap on number of questions (dev runs).",
    )
    parser.add_argument(
        "--yes", action="store_true",
        help="Skip the cost-confirmation prompt.",
    )
    args = parser.parse_args()
    dry = not args.real

    project_root = args.project_root
    paths = make_v2_paths(project_root)
    cfg = RetrievalConfig()

    # ---- Load questions -------------------------------------------
    if not paths.pilot_questions.exists():
        print(f"ERROR: {paths.pilot_questions} not found.\n"
              f"Run: python phase3_benchmark/build_temporal_questions.py "
              f"--data-dir data --out outputs/temporal_pilot_questions.jsonl",
              file=sys.stderr)
        return 1
    with paths.pilot_questions.open() as f:
        questions = [json.loads(line) for line in f]
    if args.max_questions:
        questions = questions[: args.max_questions]
    print(f"Loaded {len(questions)} pilot questions "
          f"from {paths.pilot_questions}")

    # ---- Sanity-check v2 artefacts are real (not dry-run vectors) -
    if paths.chunk_embeds_path.exists():
        import numpy as np
        e = np.load(paths.chunk_embeds_path)
        print(f"Embed matrix: shape={e.shape}, "
              f"row-0 norm={np.linalg.norm(e[0]):.4f}")

    # ---- Cost estimate --------------------------------------------
    # 30 Qs × ~1k prompt tokens × $0.15/M  +  30 × ~150 completion × $0.60/M
    est_cost = (len(questions) * 1000 * 0.15 / 1e6
                + len(questions) * 150 * 0.60 / 1e6)
    mode = "DRY RUN (no API call)" if dry else "REAL (OpenAI API)"
    print(f"\nMode: {mode}")
    print(f"Estimated cost: ~${est_cost:.4f}")
    if not dry and est_cost > 0.5 and not args.yes:
        ans = input("Proceed? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            print("Aborted.")
            return 0

    # ---- Sanity-check the chat cache for dry-run poisoning -------
    # This is the second time this bug class bit us (first in
    # embed_v2_chunks.py via the embedding cache). The chat cache
    # lives at outputs/cache/chat/*.json; dry-run entries are
    # tagged with tokens.dry_run = True. Refuse to start a REAL
    # run if any such entries are present, because the cache key
    # does not include the dry_run flag — a REAL run will get the
    # poisoned entries instead of hitting the API.
    if not dry:
        chat_cache_dir = paths.cache_dir / "chat"
        if chat_cache_dir.exists():
            poisoned = []
            for entry in chat_cache_dir.glob("*.json"):
                try:
                    rec = json.loads(entry.read_text())
                    if rec.get("tokens", {}).get("dry_run"):
                        poisoned.append(entry.name)
                except (json.JSONDecodeError, OSError):
                    continue
            if poisoned:
                print(f"\nERROR: {len(poisoned)} dry-run entries found in "
                      f"the chat cache at {chat_cache_dir}.", file=sys.stderr)
                print(f"  Example: {poisoned[0]}", file=sys.stderr)
                print(f"  These would silently shadow the real API calls.",
                      file=sys.stderr)
                print(f"  Fix: rm -rf {chat_cache_dir}", file=sys.stderr)
                return 3

    # ---- Set up Vanilla v2 ----------------------------------------
    print("\nSetting up Vanilla v2 retriever ...")
    embedder = Embedder(paths.cache_dir, dry_run=dry)
    llm = ChatLLM(paths.cache_dir, dry_run=dry)
    retriever = VanillaRAG(paths, cfg, embedder)
    retriever.setup()
    pipe = Pipeline(retriever, llm)

    # ---- Run each question ----------------------------------------
    print(f"\nAnswering {len(questions)} questions ...")
    answers = []
    scored = []
    for i, q in enumerate(questions, start=1):
        rec = pipe.answer(q["qid"], q["question"])
        answers.append(asdict(rec))
        score = score_one(q, rec.answer)
        scored.append({
            "qid": q["qid"],
            "category": q["category"],
            "sub_type": q["sub_type"],
            "gold_answer": q["gold_answer"],
            "gold_anchors": q.get("time_anchors", []),
            "answer": rec.answer,
            "cached": rec.cached,
            **score,
        })
        ok = "✓" if score["match"] else "✗"
        spu = f"  +{score['n_spurious']} spurious anchors" \
              if score["n_spurious"] else ""
        print(f"  {i:>2}/{len(questions)}  {q['qid']:<14} {ok}{spu}")

    # ---- Write artefacts ------------------------------------------
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    ans_path = paths.results_dir / "pilot_vanilla_v2_answers.jsonl"
    scored_path = paths.results_dir / "pilot_vanilla_v2_scored.jsonl"
    with ans_path.open("w") as f:
        for a in answers:
            f.write(json.dumps(a, ensure_ascii=False, default=str) + "\n")
    with scored_path.open("w") as f:
        for s in scored:
            f.write(json.dumps(s, ensure_ascii=False, default=str) + "\n")
    print(f"\nWrote {len(answers)} answers to {ans_path}")
    print(f"Wrote {len(scored)} scored records to {scored_path}")

    # ---- Summary report -------------------------------------------
    md = write_summary(scored, mode=mode)
    summary_path = paths.results_dir / "pilot_vanilla_v2_summary.md"
    summary_path.write_text(md)
    print(f"Wrote summary to {summary_path}\n")

    # ---- Print to stdout too --------------------------------------
    print(md)
    return 0


def write_summary(scored: list[dict], mode: str,
                  title: str = "Vanilla v2") -> str:
    from collections import Counter
    n = len(scored)
    n_match = sum(1 for s in scored if s["match"])
    n_with_spurious = sum(1 for s in scored if s["n_spurious"] > 0)
    total_spurious = sum(s["n_spurious"] for s in scored)

    per_cat: dict[str, list[dict]] = {}
    for s in scored:
        per_cat.setdefault(s["category"], []).append(s)

    md = []
    md.append(f"# Pilot run — {title} on temporal pilot questions\n")
    md.append(f"**Mode:** {mode}\n")
    md.append(f"**N questions:** {n}\n")
    md.append(f"**Overall match (closed-form, ±10% tolerance):**"
              f" {n_match}/{n}  ({100*n_match/n:.0f}%)\n")
    md.append(f"**Answers with ≥1 spurious time anchor:** "
              f"{n_with_spurious}/{n}  "
              f"({100*n_with_spurious/n:.0f}%)\n")
    md.append(f"**Total spurious anchors:** {total_spurious}\n")
    md.append("\n## By category\n")
    md.append("| category | match-rate | hallucination-rate | "
              "avg spurious / Q |\n")
    md.append("|---|---|---|---|\n")
    for cat, rows in per_cat.items():
        m_rate = sum(r["match"] for r in rows) / len(rows)
        h_rate = sum(r["n_spurious"] > 0 for r in rows) / len(rows)
        avg_sp = sum(r["n_spurious"] for r in rows) / len(rows)
        md.append(f"| {cat} | {100*m_rate:.0f}% | {100*h_rate:.0f}% | "
                  f"{avg_sp:.2f} |\n")

    md.append("\n## Per-question results\n")
    md.append("| qid | sub_type | match | spurious | answer (first 80c) |\n")
    md.append("|---|---|---|---|---|\n")
    for s in scored:
        ans_short = s["answer"][:80].replace("|", "\\|").replace("\n", " ")
        ok = "✓" if s["match"] else "✗"
        md.append(f"| {s['qid']} | {s['sub_type']} | {ok} | "
                  f"{s['n_spurious']} | {ans_short}... |\n")

    md.append("\n## Caveats\n")
    md.append("- Auto-scoring is text-pattern based and intentionally "
              "permissive; treat numbers as coarse signals, not "
              "final accuracy.\n")
    md.append("- Tolerance per numeric value: 10% relative OR "
              "max(0.05, 10% of |gold|) absolute, whichever is wider. "
              "Scales with the magnitude of the gold value to avoid "
              "false positives on sub-day latencies.\n")
    md.append("- Spurious time anchor = any quarter/month/half-year "
              "mention NOT covered by the gold anchor list "
              "(quarter gold covers its 3 months, half-year gold "
              "covers its 6 months).\n")
    md.append("- This is a PILOT smoke-test, not the headline "
              "experiment. The full benchmark will use the LLM-judge "
              "+ temporal-faithfulness judge with rubric annotations.\n")
    return "".join(md)


if __name__ == "__main__":
    sys.exit(main())
