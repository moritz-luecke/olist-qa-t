#!/usr/bin/env python3
"""
============================================================
rescore_scored.py — regenerate *_scored.jsonl from *_answers.jsonl
============================================================

Deterministic re-scoring (no API): applies the current score_one()
to every answer file of the 90-question temporal run, including the
granularity-sweep (force-*) variants. Written for the 2026-06-11
match_dict bug fix (distinct-number assignment); see the docstring
of pilot_run_vanilla_v2.match_dict.

Pilot (150-question) files are NOT rescored: they contain no
dict-gold questions (number/string/rubric only), so the fix cannot
change them.

Originals are backed up once to outputs/results/_pre_distinctfix/.

Usage:
  python scripts/rescore_scored.py            # rescore + summary
  python scripts/rescore_scored.py --dry      # report-only, no writes
============================================================
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# pilot_run_vanilla_v2 pulls in retriever modules; stub heavy deps
# that scoring itself never touches.
for _name in ("networkx", "faiss", "numpy", "openai", "pandas",
              "igraph", "leidenalg", "tqdm", "dotenv"):
    try:
        __import__(_name)
    except ImportError:
        sys.modules[_name] = types.ModuleType(_name)

sys.path.insert(0, str(ROOT / "phase4_retrieval"))
from pilot_run_vanilla_v2 import score_one  # noqa: E402

RESULTS = ROOT / "outputs" / "results"
QUESTIONS = ROOT / "outputs" / "temporal_questions_90.jsonl"
BACKUP = RESULTS / "_pre_distinctfix"

SYSTEMS = ["vanilla_rag", "entity", "community", "hybrid",
           "entity_temporal", "community_temporal", "hybrid_temporal"]
FORCED = ["force-week_community_temporal",
          "force-month_community_temporal",
          "force-quarter_community_temporal"]


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    questions = {q["qid"]: q for q in load_jsonl(QUESTIONS)}

    changed_total = 0
    for stem in SYSTEMS + FORCED:
        ans_p = RESULTS / f"temporal_questions_90_{stem}_answers.jsonl"
        sc_p = RESULTS / f"temporal_questions_90_{stem}_scored.jsonl"
        if not ans_p.exists() or not sc_p.exists():
            print(f"  [skip] {stem} (missing files)")
            continue
        answers = load_jsonl(ans_p)
        old = {r["qid"]: r for r in load_jsonl(sc_p)}

        new_rows, flips = [], []
        for a in answers:
            q = questions[a["qid"]]
            rec = dict(old[a["qid"]])          # keep layout/extra fields
            rec.update(score_one(q, a.get("answer", "")))
            new_rows.append(rec)
            if bool(rec["match"]) != bool(old[a["qid"]].get("match")):
                flips.append((a["qid"],
                              bool(old[a["qid"]].get("match")),
                              bool(rec["match"])))
        n_match = sum(bool(r["match"]) for r in new_rows)
        print(f"  {stem:<38} {n_match}/{len(new_rows)} "
              f"({100*n_match/len(new_rows):.1f}%)"
              + (f"   flips: {flips}" if flips else ""))
        changed_total += len(flips)

        if not args.dry and flips:
            BACKUP.mkdir(exist_ok=True)
            bak = BACKUP / sc_p.name
            if not bak.exists():
                shutil.copy2(sc_p, bak)
            with sc_p.open("w", encoding="utf-8") as f:
                for r in new_rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n{changed_total} match flips total"
          + (" (dry run, nothing written)" if args.dry else
             f"; originals of changed files backed up to {BACKUP.name}/"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
