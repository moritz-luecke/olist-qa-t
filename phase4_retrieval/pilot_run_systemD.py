"""
============================================================
Phase 5.4 — Pilot run: System D (Community-Temporal) on pilot Qs
============================================================

Runs the time-bucket community retriever (System D) against the
same 30 temporal pilot questions used for Vanilla v2 and System C,
with the same auto-scorer, so all three are directly comparable.

Reuses (imported, NOT duplicated):
  - score_one / write_summary  from pilot_run_vanilla_v2.py
  - extraction_accuracy         from pilot_run_systemC.py
  - the chat-cache poisoning detector pattern

Key empirical questions this pilot answers (see status.md §8.3):
  - Does System D answer un-filtered population aggregates that
    sink Vanilla (sample-mean miscalibration) and limit System C
    (40-order sample)?  Expect latency_02/03/04/05/08 and the
    trend aggregation/comparison questions to flip to ✓.
  - Does Drift move off System C's 0%?  drift_08 (top customer
    states) and drift_09 (top payment methods) carry exact stats
    in the bucket; drift_06 / drift_10 may partially hit; per-
    category review/volume/price drift and top-5 sellers should
    still miss (not carried in the summary).
  - Which questions does D *not* solve, and why — entity-subfilter
    (latency_01 SP, latency_06 SP-vs-RJ), sub-bucket custom windows
    (latency_09 pre-BF 30 days, latency_10 World Cup), half-year
    fragmentation (trend_10, latency_07).  These motivate System E.

Output
------
  outputs/results/pilot_systemD_answers.jsonl
  outputs/results/pilot_systemD_scored.jsonl
  outputs/results/pilot_systemD_summary.md

Cost
----
  30 questions × (≤1 extraction call + 1 generation call)
  ≈ ≤60 × $0.001  ≈  $0.06   (gpt-4o-mini).  With --extract-prefer
  rule the extraction is free (deterministic), so ~$0.03.

Usage
-----
  python phase4_retrieval/pilot_run_systemD.py            # DRY by default
  python phase4_retrieval/pilot_run_systemD.py --real
  python phase4_retrieval/pilot_run_systemD.py --real --extract-prefer rule
  python phase4_retrieval/pilot_run_systemD.py --real --force-level month
                                          # granularity ablation (RQ3/H4)
============================================================
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

# --- sys.path setup (mirrors pilot_run_systemC.py) ---------------
_HERE = Path(__file__).resolve().parent              # phase4_retrieval/
_ROOT = _HERE.parent                                  # code/
# phase4_retrieval must win the `retrievers` package name, so it is
# inserted LAST (→ ends up at sys.path[0]). phase5_temporal and
# phase2_kg are added for time_extract / temporal_index / build_kg.
sys.path.insert(0, str(_ROOT / "phase5_temporal"))
sys.path.insert(0, str(_ROOT / "phase2_kg"))
sys.path.insert(0, str(_HERE))                        # → sys.path[0]

from config import Paths, RetrievalConfig             # noqa: E402
from embeddings import Embedder                       # noqa: E402
from llm import ChatLLM                               # noqa: E402
from pipeline import Pipeline                         # noqa: E402

# Reuse the scorer + summary writer from the vanilla runner, and the
# extraction-accuracy helper from the System C runner.
from pilot_run_vanilla_v2 import score_one, write_summary  # noqa: E402
from pilot_run_systemC import extraction_accuracy          # noqa: E402

# System D lives in phase5_temporal/retrievers/temporal_community.py.
# Load it by explicit file path so it does not require putting that
# directory on sys.path (which would shadow phase4_retrieval's
# `retrievers` package). The module resolves `retrievers.community`
# correctly because phase4_retrieval is already on sys.path.
import importlib.util as _ilu                          # noqa: E402
_td_path = _ROOT / "phase5_temporal" / "retrievers" / "temporal_community.py"
_td_spec = _ilu.spec_from_file_location("temporal_community", _td_path)
_td_mod = _ilu.module_from_spec(_td_spec)
_td_spec.loader.exec_module(_td_mod)
TemporalCommunityRetriever = _td_mod.TemporalCommunityRetriever


# ----------------------------------------------------------------
# Path overrides for the temporal pilot
# ----------------------------------------------------------------

def make_paths(project_root: Path) -> Paths:
    """Paths pointed at the temporal pilot questions. System D does
    not use the order chunks; it reads outputs/time_buckets.jsonl."""
    return Paths(
        data_dir=project_root / "data",
        kg_path=project_root / "outputs" / "kg_schema_aware.pkl",
        node_embeds_path=project_root / "outputs" / "node_embeds.npy",
        pilot_questions=project_root / "outputs" /
                        "temporal_pilot_questions.jsonl",
        results_dir=project_root / "outputs" / "results",
        cache_dir=project_root / "outputs" / "cache",
    )


# ================================================================
# Main
# ================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pilot run of System D (community-temporal) on the "
                    "temporal pilot questions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--project-root", type=Path, default=_ROOT)
    parser.add_argument(
        "--real", action="store_true",
        help="Run real OpenAI calls (default: dry-run).",
    )
    parser.add_argument(
        "--extract-prefer", choices=["llm", "rule"], default="llm",
        help="Time-extraction path: 'llm' (default) or 'rule' (free).",
    )
    parser.add_argument(
        "--force-level", choices=["year", "quarter", "month"], default=None,
        help="Override granularity auto-detection (RQ3/H4 ablation). "
             "Default: auto-detect from the question.",
    )
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()
    dry = not args.real

    paths = make_paths(args.project_root)
    cfg = RetrievalConfig()
    cfg.time_extract_prefer = args.extract_prefer
    if args.force_level:
        cfg.time_force_level = args.force_level

    # ---- Load questions -------------------------------------------
    if not paths.pilot_questions.exists():
        print(f"ERROR: {paths.pilot_questions} not found.", file=sys.stderr)
        return 1
    with paths.pilot_questions.open() as f:
        questions = [json.loads(line) for line in f]
    if args.max_questions:
        questions = questions[: args.max_questions]
    print(f"Loaded {len(questions)} pilot questions")

    # ---- Check time buckets exist ---------------------------------
    buckets_path = args.project_root / "outputs" / "time_buckets.jsonl"
    if not buckets_path.exists():
        print(f"ERROR: {buckets_path} not found. Build it first:\n"
              f"  python phase5_temporal/time_hierarchy.py --no-dry-run",
              file=sys.stderr)
        return 1

    # ---- Cost estimate + mode -------------------------------------
    calls_per_q = 1 if args.extract_prefer == "rule" else 2
    est_cost = len(questions) * calls_per_q * 0.001
    mode = "DRY RUN (no API call)" if dry else "REAL (OpenAI API)"
    print(f"\nMode: {mode}")
    print(f"Extraction path: {args.extract_prefer}")
    if args.force_level:
        print(f"Granularity override: {args.force_level}")
    print(f"Estimated cost: ~${est_cost:.4f}")

    # ---- Cache-poisoning guard (same as vanilla / System C) -------
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
                print(f"\nERROR: {len(poisoned)} dry-run entries in the "
                      f"chat cache at {chat_cache_dir}.", file=sys.stderr)
                print(f"  Example: {poisoned[0]}", file=sys.stderr)
                print(f"  Fix: rm -rf {chat_cache_dir}", file=sys.stderr)
                print(f"  Then re-run with --real.", file=sys.stderr)
                return 3

        if est_cost > 0.5 and not args.yes:
            ans = input("Proceed? [y/N] ").strip().lower()
            if ans not in ("y", "yes"):
                print("Aborted.")
                return 0

    # ---- Set up System D ------------------------------------------
    print("\nSetting up System D (community-temporal) ...")
    embedder = Embedder(paths.cache_dir, dry_run=dry)
    llm = ChatLLM(paths.cache_dir, dry_run=dry)
    retriever = TemporalCommunityRetriever(
        paths, cfg, embedder, llm=llm, buckets_path=buckets_path,
    )
    retriever.setup()
    pipe = Pipeline(retriever, llm)
    print(f"  loaded {len(retriever._records)} time buckets "
          f"({sorted({r['level'] for r in retriever._records})})")

    # ---- Run each question ----------------------------------------
    print(f"\nAnswering {len(questions)} questions ...")
    answers = []
    scored = []
    extract_hits = 0
    extract_scorable = 0
    for i, q in enumerate(questions, start=1):
        # Extraction accuracy (measurement only; same cached extraction
        # call the retriever will hit — no double cost).
        ext = extraction_accuracy(q, llm if not dry else None,
                                  args.extract_prefer)
        if ext["extract_verdict"] in ("hit", "miss"):
            extract_scorable += 1
            if ext["extract_verdict"] == "hit":
                extract_hits += 1

        rec = pipe.answer(q["qid"], q["question"])
        answers.append({**asdict(rec), "extraction": ext})

        score = score_one(q, rec.answer)
        # Surface the bucket-selection diagnostics for analysis.
        tdiag = rec.diagnostics.get("temporal", {})
        scored.append({
            "qid": q["qid"],
            "category": q["category"],
            "sub_type": q["sub_type"],
            "gold_answer": q["gold_answer"],
            "gold_anchors": q.get("time_anchors", []),
            "answer": rec.answer,
            "cached": rec.cached,
            "extract_verdict": ext["extract_verdict"],
            "extract_source": ext["extract_source"],
            "select_mode": tdiag.get("mode"),
            "target_level": tdiag.get("target_level"),
            "bucket_ids": tdiag.get("bucket_ids", []),
            "n_buckets": rec.diagnostics.get("n_buckets", 0),
            **score,
        })
        ok = "✓" if score["match"] else "✗"
        ev = {"hit": "E✓", "miss": "E✗", "n/a": "E–"}[ext["extract_verdict"]]
        nb = tdiag.get("bucket_ids", [])
        nb_str = ",".join(nb) if len(nb) <= 4 else f"{len(nb)} buckets"
        print(f"  {i:>2}/{len(questions)}  {q['qid']:<14} {ok} {ev}  "
              f"[{nb_str}]")

    # ---- Write artefacts ------------------------------------------
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    ans_path = paths.results_dir / "pilot_systemD_answers.jsonl"
    scored_path = paths.results_dir / "pilot_systemD_scored.jsonl"
    with ans_path.open("w") as f:
        for a in answers:
            f.write(json.dumps(a, ensure_ascii=False, default=str) + "\n")
    with scored_path.open("w") as f:
        for s in scored:
            f.write(json.dumps(s, ensure_ascii=False, default=str) + "\n")
    print(f"\nWrote {ans_path}")
    print(f"Wrote {scored_path}")

    # ---- Summary --------------------------------------------------
    md = write_summary(scored, mode=f"System D — {mode}",
                       title="System D (community-temporal)")
    ext_rate = (100 * extract_hits / extract_scorable
                if extract_scorable else 0.0)
    # Bucket-selection-mode breakdown (System D-specific).
    from collections import Counter
    mode_counts = Counter(s.get("select_mode") for s in scored)
    fallback_n = sum(1 for s in scored
                     if s.get("select_mode") == "cosine_fallback")
    md += (
        f"\n## Extraction + bucket selection (System D)\n"
        f"- Questions with a gold time_range: {extract_scorable}/"
        f"{len(questions)}\n"
        f"- Extraction window overlapped gold: {extract_hits}/"
        f"{extract_scorable}  ({ext_rate:.0f}%)\n"
        f"- Extraction path: {args.extract_prefer}\n"
        f"- Selection modes: "
        + ", ".join(f"{k}={v}" for k, v in mode_counts.items()) + "\n"
        f"- Fell back to cosine (no aligned bucket): {fallback_n}/"
        f"{len(questions)}\n"
    )
    if args.force_level:
        md += f"- Granularity override: {args.force_level}\n"
    summary_path = paths.results_dir / "pilot_systemD_summary.md"
    summary_path.write_text(md)
    print(f"Wrote {summary_path}\n")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
