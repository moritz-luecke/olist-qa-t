"""
============================================================
Phase 5.6 — Temporal-faithfulness judge (RAGAS-style)
============================================================

STATUS (release v1.0.0)
-----------------------
The faithfulness numbers in the paper do NOT come from this module.
They are computed deterministically from the answer text in
scripts/analyze_results.py (analyze_faithfulness): answer rate,
conservative coverage of the gold anchors, and out-of-window
assertions, with no LLM involved. The LLM-extraction judge below was
discarded: its prompt showed the question, so periods named in the
question were credited even when the answer abstained (see
notes/faithfulness_bug_postmortem.md, in German). The module is kept
for transparency; its --silver mode is the mechanical regex detector.

The headline metric for H2: do time-blind systems assert time
periods that are NOT supported by the question's gold window
(time-anchor hallucination), and do they fail to cover the gold
anchors? Expected: time-blind systems hallucinate in ≥25% of
trend/drift answers; temporal systems in <5%.

Why an LLM judge on top of the mechanical detector
---------------------------------------------------
`pilot_run_vanilla_v2.hallucinated_time_anchors` already flags
time anchors by regex. The regex misses paraphrases ("the fourth
quarter of 2017", "late 2017", "the back half of the year"). This
judge uses the LLM for the hard part — semantic EXTRACTION of every
time period the answer asserts, normalised to canonical bucket
strings — and then classifies covered / missing / hallucinated
DETERMINISTICALLY in Python (month-granular, lifecycle-spillover
aware), identical in spirit to the mechanical detector. So:

  - LLM  → robust extraction (handles paraphrase).
  - Python → deterministic, calibratable classification.

This split makes the judge reproducible and lets us report κ
between (a) judge and human, and (b) judge and the mechanical
detector (a free silver-standard sanity check, --silver).

Closed-form note: our temporal questions are closed-form, but the
answers are prose ("In Q1 2017 the mean was 12.98 days; …"), so the
faithfulness judge applies directly — it scores the time references
in that prose, independent of the numeric correctness scored by
pilot_run_vanilla_v2.

Reuses: config (JUDGE_*), llm.ChatLLM (cached, dry-run aware),
the make_judge_llm pattern, and the chat-cache poisoning guard.

Usage:
  # mechanical silver-standard only (no API):
  python phase5_temporal/temporal_faithfulness.py --silver \\
      --answers outputs/results/pilot_systemD_answers.jsonl \\
      --questions outputs/temporal_questions_90.jsonl
  # real LLM judge:
  python phase5_temporal/temporal_faithfulness.py --real \\
      --answers outputs/results/pilot_systemE_answers.jsonl \\
      --questions outputs/temporal_questions_90.jsonl
  # calibrate against a manual annotation file:
  python phase5_temporal/temporal_faithfulness.py --real --calibrate \\
      human_anchor_labels.jsonl --answers ... --questions ...
============================================================
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "phase4_retrieval"))

from config import (JUDGE_MODEL, JUDGE_TEMPERATURE, JUDGE_SEED, Paths)  # noqa: E402
from llm import ChatLLM                                                  # noqa: E402
from pilot_run_vanilla_v2 import hallucinated_time_anchors               # noqa: E402


# ============================================================
# Canonical period <-> month-index helpers (month-granular)
# ============================================================

def _mi(year: int, month: int) -> int:
    return year * 12 + (month - 1)


def period_to_months(p: str) -> list[int]:
    """Canonical period string -> list of month indices it spans.
    Accepts YYYY, YYYY-Qn, YYYY-Hn, YYYY-MM (case-insensitive)."""
    p = p.strip().upper()
    m = re.fullmatch(r"(\d{4})", p)
    if m:
        y = int(m.group(1)); return [_mi(y, mo) for mo in range(1, 13)]
    m = re.fullmatch(r"(\d{4})-Q([1-4])", p)
    if m:
        y, q = int(m.group(1)), int(m.group(2))
        s = (q - 1) * 3 + 1; return [_mi(y, mo) for mo in range(s, s + 3)]
    m = re.fullmatch(r"(\d{4})-H([12])", p)
    if m:
        y, h = int(m.group(1)), int(m.group(2))
        s = 1 if h == 1 else 7; return [_mi(y, mo) for mo in range(s, s + 6)]
    m = re.fullmatch(r"(\d{4})-(\d{2})", p)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        if 1 <= mo <= 12:
            return [_mi(y, mo)]
    return []


def _allowed_months(gold_anchors: list[str], time_range: dict | None,
                    spillover: int = 2) -> set[int]:
    """Month indices considered legitimate: every month covered by a
    gold anchor (or the time_range), plus `spillover` months after the
    latest such month (lifecycle spillover — a late-window order
    delivered/reviewed shortly after the window)."""
    months: set[int] = set()
    for a in gold_anchors:
        months.update(period_to_months(a))
    if time_range and time_range.get("start") and time_range.get("end"):
        try:
            sy, sm = int(time_range["start"][:4]), int(time_range["start"][5:7])
            ey, em = int(time_range["end"][:4]), int(time_range["end"][5:7])
            cur, end = _mi(sy, sm), _mi(ey, em)          # end exclusive
            while cur < end:
                months.add(cur); cur += 1
        except (ValueError, KeyError, TypeError):
            pass
    if months:
        hi = max(months)
        for k in range(1, spillover + 1):
            months.add(hi + k)
    return months


def classify_anchors(extracted: list[str], gold_anchors: list[str],
                     time_range: dict | None) -> dict:
    """Deterministic month-granular classification of the LLM-extracted
    asserted periods against the gold window."""
    allowed = _allowed_months(gold_anchors, time_range)
    gold_month_sets = {a: set(period_to_months(a)) for a in gold_anchors}

    hallucinated, recognised = [], []
    for e in extracted:
        ms = period_to_months(e)
        if not ms:
            continue                                  # unparseable → ignore
        recognised.append(e)
        # Hallucinated only if the period is DISJOINT from the allowed
        # window (no month overlaps). A coarse mention that overlaps —
        # e.g. the bare year "2018" on a 2018-Q1..Q3 question, where the
        # data's 2018 ends in September — is the question's own framing,
        # not an unsupported claim. Genuinely out-of-window periods
        # (e.g. "2019", or a wrong quarter) have no overlap and are still
        # flagged.
        if allowed and not (set(ms) & allowed):
            hallucinated.append(e)

    ext_months = set()
    for e in recognised:
        ext_months.update(period_to_months(e))
    missing = [a for a, ms in gold_month_sets.items()
               if ms and not (ms & ext_months)]

    return {
        "extracted": recognised,
        "missing_gold": missing,
        "hallucinated": hallucinated,
        "coverage_ok": len(missing) == 0,
        "hallucinated_any": len(hallucinated) > 0,
    }


# ============================================================
# LLM extraction of asserted time periods
# ============================================================

FAITHFULNESS_PROMPT = """\
You are auditing the TIME REFERENCES in an answer to a business
question about a fixed dataset. Your ONLY job is to list every time
period that the answer ASSERTS as part of its claim (e.g. a period it
gives a value for, or compares, or attributes a fact to).

Normalise each to ONE canonical token:
  - a full year                    -> "YYYY"          (e.g. "2017")
  - a quarter                      -> "YYYY-Qn"        (e.g. "2017-Q4")
  - a half-year                    -> "YYYY-Hn"        (H1=Jan-Jun, H2=Jul-Dec)
  - a single month                 -> "YYYY-MM"        (e.g. "2017-07")
Resolve paraphrases: "the fourth quarter of 2017" -> "2017-Q4";
"the second half of 2017" -> "2017-H2"; "July 2017" -> "2017-07".
Do NOT invent periods that are not asserted. Ignore non-time numbers.
If the answer asserts no time period, return an empty list.

QUESTION:
{question}

ANSWER:
{answer}

Respond with a single JSON object and no other text:
{{"asserted_periods": ["YYYY-Qn", ...]}}"""


@dataclass
class FaithfulnessVerdict:
    qid: str
    system: str
    gold_anchors: list[str]
    extracted: list[str]
    missing_gold: list[str]
    hallucinated: list[str]
    coverage_ok: bool
    hallucinated_any: bool
    cached: bool = False
    source: str = "llm"          # "llm" or "silver" (mechanical)


def make_faithfulness_llm(paths: Paths, dry_run: bool = True) -> ChatLLM:
    return ChatLLM(cache_root=paths.cache_dir, model=JUDGE_MODEL,
                   temperature=JUDGE_TEMPERATURE, top_p=1.0,
                   seed=JUDGE_SEED, max_tokens=300, dry_run=dry_run)


def judge_faithfulness(judge: ChatLLM, qid: str, system: str,
                       question: str, gold_anchors: list[str],
                       time_range: dict | None, answer: str
                       ) -> FaithfulnessVerdict:
    resp = judge.chat(
        system="You extract and normalise time references. Return JSON only.",
        user=FAITHFULNESS_PROMPT.format(question=question, answer=answer),
        response_format={"type": "json_object"},
    )
    try:
        parsed = json.loads(resp["text"])
        extracted = [str(x) for x in parsed.get("asserted_periods", [])]
    except (json.JSONDecodeError, TypeError, AttributeError):
        extracted = []
    cls = classify_anchors(extracted, gold_anchors, time_range)
    return FaithfulnessVerdict(
        qid=qid, system=system, gold_anchors=gold_anchors,
        cached=resp.get("cached", False), source="llm", **cls)


def silver_faithfulness(qid: str, system: str, gold_anchors: list[str],
                        time_range: dict | None, answer: str
                        ) -> FaithfulnessVerdict:
    """Mechanical (regex) reference — no API. Uses the same detector as
    the pilot scorer for hallucination; coverage via regex extraction."""
    hall = hallucinated_time_anchors(answer, gold_anchors, time_range=time_range)
    # crude coverage: a gold anchor is covered if its canonical string
    # (or a month within it) appears in the answer text.
    t = answer.upper().replace(" ", "")
    missing = []
    for a in gold_anchors:
        toks = [a.upper()] + [f"{y//12}-{y % 12 + 1:02d}"
                              for y in period_to_months(a)]
        if not any(tok.replace("-", "").replace(" ", "")
                   in t.replace("-", "") for tok in toks):
            missing.append(a)
    return FaithfulnessVerdict(
        qid=qid, system=system, gold_anchors=gold_anchors,
        extracted=[], missing_gold=missing, hallucinated=hall,
        coverage_ok=len(missing) == 0, hallucinated_any=len(hall) > 0,
        cached=True, source="silver")


# ============================================================
# Aggregation + calibration
# ============================================================

def aggregate(verdicts: list[FaithfulnessVerdict]) -> dict:
    n = len(verdicts) or 1
    hall = sum(v.hallucinated_any for v in verdicts)
    cov = sum(v.coverage_ok for v in verdicts)
    return {
        "n": len(verdicts),
        "hallucination_rate": round(100 * hall / n, 1),
        "coverage_rate": round(100 * cov / n, 1),
        "n_hallucinated": hall,
        "n_covered": cov,
    }


def cohens_kappa(a: list[bool], b: list[bool]) -> float:
    """Cohen's κ for two binary labellings (e.g. judge vs human on
    hallucinated_any)."""
    assert len(a) == len(b) and a
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa1 = sum(a) / n; pb1 = sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return 1.0 if pe == 1.0 else round((po - pe) / (1 - pe), 3)


# ============================================================
# CLI
# ============================================================

def _load_questions(path: Path) -> dict:
    out = {}
    with path.open() as f:
        for line in f:
            q = json.loads(line)
            out[q["qid"]] = q
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    ap.add_argument("--answers", type=Path, default=None,
                    help="pilot_<system>_answers.jsonl (not needed for "
                         "--reclassify)")
    ap.add_argument("--questions", type=Path,
                    default=Path("./outputs/temporal_questions_90.jsonl"))
    ap.add_argument("--project-root", type=Path, default=_ROOT)
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--silver", action="store_true",
                    help="Mechanical detector only (no API).")
    ap.add_argument("--calibrate", type=Path, default=None,
                    help="Human-label JSONL {qid, coverage_ok[, hallucinated_any]} for κ.")
    ap.add_argument("--reclassify", type=Path, default=None,
                    help="Re-classify an existing faithfulness_*.jsonl from "
                         "its stored 'extracted' field (no API). Use after a "
                         "classifier fix.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    dry = not args.real

    paths = Paths(
        data_dir=args.project_root / "data",
        kg_path=args.project_root / "outputs" / "kg_schema_aware.pkl",
        node_embeds_path=args.project_root / "outputs" / "node_embeds.npy",
        pilot_questions=args.questions,
        results_dir=args.project_root / "outputs" / "results",
        cache_dir=args.project_root / "outputs" / "cache",
    )
    questions = _load_questions(args.questions)

    # --- reclassify mode: recompute verdicts from stored 'extracted' ---
    if args.reclassify:
        recs = [json.loads(l) for l in args.reclassify.open()]
        verdicts = []
        for r in recs:
            q = questions.get(r["qid"])
            if q is None:
                continue
            cls = classify_anchors(r.get("extracted", []),
                                   q.get("time_anchors", []),
                                   q.get("time_range"))
            verdicts.append(FaithfulnessVerdict(
                qid=r["qid"], system=r.get("system", "?"),
                gold_anchors=q.get("time_anchors", []),
                cached=True, source="reclassified", **cls))
        agg = aggregate(verdicts)
        sysname = verdicts[0].system if verdicts else "?"
        print(f"[{sysname}] reclassified faithfulness:")
        print(f"  hallucination rate: {agg['hallucination_rate']}% "
              f"({agg['n_hallucinated']}/{agg['n']})")
        print(f"  coverage rate:      {agg['coverage_rate']}% "
              f"({agg['n_covered']}/{agg['n']})")
        out = args.out or args.reclassify.with_name(
            args.reclassify.stem + "_reclassified.jsonl")
        with out.open("w") as f:
            for v in verdicts:
                f.write(json.dumps(asdict(v), default=str) + "\n")
        print(f"Wrote {out}")
        return 0

    if not args.answers:
        print("ERROR: --answers is required (unless --reclassify).",
              file=sys.stderr)
        return 1
    with args.answers.open() as f:
        answers = [json.loads(line) for line in f]
    # Only temporal questions carry the time-anchor rubric.
    answers = [a for a in answers if a["qid"] in questions]
    system = answers[0].get("system", args.answers.stem) if answers else "?"
    print(f"{len(answers)} temporal answers from system '{system}'")

    judge = None
    if not args.silver:
        # Cache-poisoning guard (status.md §4.3)
        if not dry:
            cdir = paths.cache_dir / "chat"
            poisoned = [p.name for p in cdir.glob("*.json")
                        if cdir.exists()
                        and json.loads(p.read_text()).get("tokens", {})
                        .get("dry_run")]
            if poisoned:
                print(f"ERROR: {len(poisoned)} dry-run cache entries; "
                      f"rm -rf {cdir} then re-run --real.", file=sys.stderr)
                return 3
        judge = make_faithfulness_llm(paths, dry_run=dry)

    verdicts = []
    for a in answers:
        q = questions[a["qid"]]
        ga = q.get("time_anchors", [])
        tr = q.get("time_range")
        if args.silver:
            v = silver_faithfulness(a["qid"], system, ga, tr, a["answer"])
        else:
            v = judge_faithfulness(judge, a["qid"], system, q["question"],
                                   ga, tr, a["answer"])
        verdicts.append(v)

    agg = aggregate(verdicts)
    print(f"\n[{system}] {'silver' if args.silver else 'llm'} faithfulness:")
    print(f"  hallucination rate: {agg['hallucination_rate']}% "
          f"({agg['n_hallucinated']}/{agg['n']})")
    print(f"  coverage rate:      {agg['coverage_rate']}% "
          f"({agg['n_covered']}/{agg['n']})")

    # Silver agreement check (judge vs mechanical), if we ran the LLM
    if not args.silver and not dry:
        sv = [silver_faithfulness(a["qid"], system,
                                  questions[a["qid"]].get("time_anchors", []),
                                  questions[a["qid"]].get("time_range"),
                                  a["answer"]).hallucinated_any
              for a in answers]
        jv = [v.hallucinated_any for v in verdicts]
        print(f"  κ(judge vs mechanical silver): {cohens_kappa(jv, sv)}")

    # Human calibration — κ on coverage_ok (the informative axis; hallucination
    # has a near-zero base rate and yields a degenerate κ). Also reports κ on
    # hallucinated_any iff that field is present in the human file.
    if args.calibrate and args.calibrate.exists():
        human = {}
        for l in args.calibrate.open():
            d = json.loads(l)
            human[d["qid"]] = d
        for label in ("coverage_ok", "hallucinated_any"):
            pairs = [(getattr(v, label), bool(human[v.qid][label]))
                     for v in verdicts
                     if v.qid in human and label in human[v.qid]]
            if pairs:
                jv, hv = [p[0] for p in pairs], [p[1] for p in pairs]
                agree = 100 * sum(a == b for a, b in pairs) / len(pairs)
                print(f"  κ(judge vs human, {label}, n={len(pairs)}): "
                      f"{cohens_kappa(jv, hv)}  (raw agreement {agree:.1f}%)")

    out = args.out or (paths.results_dir /
                       f"faithfulness_{system}_"
                       f"{'silver' if args.silver else 'llm'}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for v in verdicts:
            f.write(json.dumps(asdict(v), default=str) + "\n")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
