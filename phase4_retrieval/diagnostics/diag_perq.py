"""
============================================================
Diagnose: per-question outcome matrix
============================================================

For the closed categories (factual + aggregation), print a
table:
    qid | category | hops | gold | vanilla | entity | community
where each system cell shows ✓ / ✗ / ABS.

Also separates the 'has_order_id' fraction so we can tell whether
the anchor-lookup actually rescued the lookup-type questions.
============================================================
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import score_answer, CLOSED_CATEGORIES


PROJECT_ROOT = Path(__file__).resolve().parent.parent
_HEX_RE = re.compile(r"\b[a-f0-9]{32}\b")


def short(s, n):
    s = str(s)
    return s if len(s) <= n else s[: n-1] + "…"


def main():
    questions = {q["qid"]: q for q in
                 (json.loads(l) for l in
                  (PROJECT_ROOT / "outputs/pilot_questions.jsonl").open())}

    systems = ["vanilla_rag", "entity", "community"]
    answers = {}
    for s in systems:
        path = PROJECT_ROOT / "outputs/results" / f"{s}_answers.jsonl"
        answers[s] = {a["qid"]: a for a in
                      (json.loads(l) for l in path.open())}

    print(f"\n{'qid':14s} {'cat':12s} {'hops':>4s}  {'gold':15s}  "
          f"{'V':3s} {'E':3s} {'C':3s}  has_id  Q")
    print("-" * 130)

    # Track stats by group
    counters = {
        ("factual","has_id"): {s: {"correct":0,"wrong":0,"abstain":0,"n":0} for s in systems},
        ("factual","no_id"):  {s: {"correct":0,"wrong":0,"abstain":0,"n":0} for s in systems},
        ("aggregation","-"):  {s: {"correct":0,"wrong":0,"abstain":0,"n":0} for s in systems},
    }

    for qid, q in sorted(questions.items()):
        if q["category"] not in CLOSED_CATEGORIES:
            continue
        has_id = bool(_HEX_RE.search(q["question"]))
        group = (q["category"], "has_id" if has_id else "no_id") \
                if q["category"] == "factual" else (q["category"], "-")

        cells = []
        for s in systems:
            a = answers[s].get(qid)
            if a is None:
                cells.append("?")
                continue
            sc = score_answer(q, a["answer"])
            mark = {"correct":"✓","wrong":"✗","abstain":"A"}[sc.outcome]
            cells.append(mark)
            counters[group][s][sc.outcome] += 1
            counters[group][s]["n"] += 1

        print(f"{qid:14s} {q['category']:12s} {str(q['min_hops']):>4s}  "
              f"{short(q['gold_answer'],15):15s}  "
              f"{cells[0]:3s} {cells[1]:3s} {cells[2]:3s}  "
              f"{'yes' if has_id else 'no ':6s} "
              f"{short(q['question'], 60)}")

    # Aggregate by group
    print(f"\n{'='*60}\nGroup summaries\n{'='*60}")
    for group, by_sys in counters.items():
        print(f"\n{group[0]} / {group[1]}:")
        for s in systems:
            c = by_sys[s]
            if c["n"] == 0:
                continue
            acc = 100 * c["correct"] / c["n"]
            print(f"  {s:14s}  acc={acc:5.1f}%  "
                  f"(✓{c['correct']} ✗{c['wrong']} A{c['abstain']} / n={c['n']})")


if __name__ == "__main__":
    main()
