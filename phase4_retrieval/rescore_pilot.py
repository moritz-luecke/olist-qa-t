"""
============================================================
Phase 4.5 — Re-score cached pilot answers (no API calls)
============================================================

Loads the cached answers for a chosen system from
outputs/results/pilot_<system>_answers.jsonl, joins them with the
gold values in outputs/temporal_pilot_questions.jsonl, and re-runs
the (now-fixed) auto-scorer. Writes a fresh scored file and summary.

Used after a scorer fix — no need to re-call the OpenAI API, the
model answers are unchanged and cached.

Scorer corrections applied (2026-06-02), validated on the pilot:
  1. Bare period-key match: when a gold dict's keys are all in the
     same year, accept the year-dropped token ('Q1' for '2017-Q1',
     'January' for '2017-01'). The model omits the redundant year
     because the question already scopes it. Multi-year gold still
     requires the year (no ambiguity introduced).
  2. Underscore/space key folding ('credit_card' <-> 'Credit Card').
  3. Half-year gold covers its two quarters (a model fed quarter
     buckets for a half-year question names Q1/Q2 for H1 -- not a
     spurious anchor).

All three are true-positive fixes (the model's numbers were already
correct), not tolerance loosening; see the adversarial checks that
ship with the scorer.

Usage
-----
  python phase4_retrieval/rescore_pilot.py                    # vanilla_v2
  python phase4_retrieval/rescore_pilot.py --system systemC
  python phase4_retrieval/rescore_pilot.py --system systemD
  python phase4_retrieval/rescore_pilot.py --system all       # all three
============================================================
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Reuse the scorer + summary helpers from the pilot run script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pilot_run_vanilla_v2 import score_one, write_summary   # noqa: E402


# system -> (answers stem, human title)
SYSTEMS = {
    "vanilla_v2": ("pilot_vanilla_v2", "Vanilla v2"),
    "systemC":    ("pilot_systemC",    "System C (entity-temporal)"),
    "systemD":    ("pilot_systemD",    "System D (community-temporal)"),
}


def rescore_one(root: Path, system: str) -> int:
    stem, title = SYSTEMS[system]
    results = root / "outputs" / "results"
    answers_path = results / f"{stem}_answers.jsonl"
    questions_path = root / "outputs" / "temporal_pilot_questions.jsonl"
    scored_path = results / f"{stem}_scored.jsonl"
    summary_path = results / f"{stem}_summary.md"

    if not answers_path.exists():
        print(f"  [SKIP {system}] {answers_path} not found.", file=sys.stderr)
        return 1

    with answers_path.open() as f:
        answers = [json.loads(line) for line in f]
    with questions_path.open() as f:
        questions = {}
        for line in f:
            q = json.loads(line)
            questions[q["qid"]] = q

    if len(answers) != len(questions):
        print(f"  [WARN {system}] {len(answers)} answers vs "
              f"{len(questions)} questions", file=sys.stderr)

    scored = []
    matched = 0
    spurious_total = 0
    for ans in answers:
        qid = ans["qid"]
        q = questions.get(qid)
        if q is None:
            print(f"  [SKIP] no question record for {qid}", file=sys.stderr)
            continue
        s = score_one(q, ans["answer"])
        matched += int(s["match"])
        spurious_total += s["n_spurious"]

        row = {
            "qid": qid,
            "category": q["category"],
            "sub_type": q["sub_type"],
            "gold_answer": q["gold_answer"],
            "gold_anchors": q.get("time_anchors", []),
            "answer": ans["answer"],
            "cached": ans.get("cached", False),
            **s,
        }
        # Carry forward per-system diagnostics if the answer file has them
        # (System C / D), so the re-scored file stays as rich as the run.
        ext = ans.get("extraction") or {}
        if ext:
            row["extract_verdict"] = ext.get("extract_verdict")
            row["extract_source"] = ext.get("extract_source")
        tdiag = (ans.get("diagnostics") or {}).get("temporal") or {}
        if tdiag:
            row["select_mode"] = tdiag.get("mode")
            row["target_level"] = tdiag.get("target_level")
            row["bucket_ids"] = tdiag.get("bucket_ids", [])
        scored.append(row)

    n = len(scored)
    print(f"[{system}] re-scored {n} answers -- "
          f"matches {matched}/{n} ({100*matched/n:.0f}%), "
          f"spurious {spurious_total}")

    with scored_path.open("w") as f:
        for s in scored:
            f.write(json.dumps(s, ensure_ascii=False, default=str) + "\n")
    md = write_summary(scored, mode="REAL (rescored, fixed scorer)",
                       title=title)
    summary_path.write_text(md)
    print(f"  wrote {scored_path.name} + {summary_path.name}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-score cached pilot answers without re-calling the API.",
    )
    parser.add_argument(
        "--project-root", type=Path,
        default=Path(__file__).resolve().parent.parent,
    )
    parser.add_argument(
        "--system", choices=list(SYSTEMS) + ["all"], default="vanilla_v2",
        help="Which system's cached answers to re-score (default: "
             "vanilla_v2). 'all' re-scores every system present.",
    )
    args = parser.parse_args()

    targets = list(SYSTEMS) if args.system == "all" else [args.system]
    rc = 0
    for sysname in targets:
        rc |= rescore_one(args.project_root, sysname)
    return 0 if args.system == "all" else rc


if __name__ == "__main__":
    sys.exit(main())
