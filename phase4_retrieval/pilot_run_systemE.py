"""
============================================================
Phase 5.5 — Pilot run: System E (Temporal Hybrid) on pilot Qs
============================================================

Runs the temporal-hybrid retriever (System E = C ⊕ D) against the
same 30 temporal pilot questions used for Vanilla v2 / C / D, with
the same auto-scorer, so all four are directly comparable.

Reuses (imported, NOT duplicated):
  - score_one / write_summary  from pilot_run_vanilla_v2.py
  - extraction_accuracy         from pilot_run_systemC.py
  - the chat-cache poisoning detector pattern

Key empirical questions (see notes/system_d_pilot.md §5):
  - Does E realise the C∪D oracle (≥19/30 on the pilot)?
  - Does E pick up latency_01 (the SP-filtered aggregate that only
    C solves) WITHOUT regressing to D's all-population bucket mean?
  - Does E preserve D's 0-spurious-anchor faithfulness?

System E needs BOTH temporal artefacts:
  outputs/temporal_index.pkl   (for the C sub-retriever)
  outputs/time_buckets.jsonl   (for the D sub-retriever)

Output
------
  outputs/results/pilot_systemE_answers.jsonl
  outputs/results/pilot_systemE_scored.jsonl
  outputs/results/pilot_systemE_summary.md

Cost
----
  30 questions × (≤1 extraction call, hit from cache by the second
  sub-retriever + 1 generation call) ≈ ~$0.06 (rule extraction: ~$0.03).

Usage
-----
  python phase4_retrieval/pilot_run_systemE.py            # DRY by default
  python phase4_retrieval/pilot_run_systemE.py --real
  python phase4_retrieval/pilot_run_systemE.py --real --extract-prefer rule
============================================================
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

# --- sys.path setup (mirrors pilot_run_systemC.py / systemD) -----
_HERE = Path(__file__).resolve().parent              # phase4_retrieval/
_ROOT = _HERE.parent                                  # code/
sys.path.insert(0, str(_ROOT / "phase5_temporal"))
sys.path.insert(0, str(_ROOT / "phase2_kg"))
sys.path.insert(0, str(_HERE))                        # -> sys.path[0]

from config import Paths, RetrievalConfig             # noqa: E402
from embeddings import Embedder                       # noqa: E402
from llm import ChatLLM                               # noqa: E402
from pipeline import Pipeline                         # noqa: E402

from pilot_run_vanilla_v2 import score_one, write_summary  # noqa: E402
from pilot_run_systemC import extraction_accuracy          # noqa: E402

# Load System E by explicit file path (same rationale as C / D runners).
import importlib.util as _ilu                          # noqa: E402
_te_path = _ROOT / "phase5_temporal" / "retrievers" / "temporal_hybrid.py"
_te_spec = _ilu.spec_from_file_location("temporal_hybrid", _te_path)
_te_mod = _ilu.module_from_spec(_te_spec)
_te_spec.loader.exec_module(_te_mod)
TemporalHybridRetriever = _te_mod.TemporalHybridRetriever


def make_paths(project_root: Path) -> Paths:
    return Paths(
        data_dir=project_root / "data",
        kg_path=project_root / "outputs" / "kg_schema_aware.pkl",
        node_embeds_path=project_root / "outputs" / "node_embeds.npy",
        pilot_questions=project_root / "outputs" /
                        "temporal_pilot_questions.jsonl",
        results_dir=project_root / "outputs" / "results",
        cache_dir=project_root / "outputs" / "cache",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pilot run of System E (temporal hybrid) on the "
                    "temporal pilot questions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--project-root", type=Path, default=_ROOT)
    parser.add_argument("--real", action="store_true",
                        help="Run real OpenAI calls (default: dry-run).")
    parser.add_argument("--extract-prefer", choices=["llm", "rule"],
                        default="llm",
                        help="Time-extraction path: 'llm' or 'rule' (free).")
    parser.add_argument("--fusion", choices=["route", "concat"],
                        default="route",
                        help="route (default, headline E: select C for "
                             "state-filtered Qs else D) or concat "
                             "(ablation: B3-style union of both).")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()
    dry = not args.real

    paths = make_paths(args.project_root)
    cfg = RetrievalConfig()
    cfg.time_extract_prefer = args.extract_prefer
    cfg.temporal_fusion = args.fusion

    # ---- Load questions -------------------------------------------
    if not paths.pilot_questions.exists():
        print(f"ERROR: {paths.pilot_questions} not found.", file=sys.stderr)
        return 1
    with paths.pilot_questions.open() as f:
        questions = [json.loads(line) for line in f]
    if args.max_questions:
        questions = questions[: args.max_questions]
    print(f"Loaded {len(questions)} pilot questions")

    # ---- Check BOTH temporal artefacts exist ----------------------
    tindex_path = args.project_root / "outputs" / "temporal_index.pkl"
    buckets_path = args.project_root / "outputs" / "time_buckets.jsonl"
    missing = [p for p in (tindex_path, buckets_path) if not p.exists()]
    if missing:
        for p in missing:
            print(f"ERROR: {p} not found.", file=sys.stderr)
        print("System E needs both temporal_index.pkl (System C) and "
              "time_buckets.jsonl (System D).", file=sys.stderr)
        return 1

    # ---- Cost estimate + mode -------------------------------------
    calls_per_q = 1 if args.extract_prefer == "rule" else 2
    est_cost = len(questions) * calls_per_q * 0.001
    mode = "DRY RUN (no API call)" if dry else "REAL (OpenAI API)"
    print(f"\nMode: {mode}")
    print(f"Fusion: {args.fusion}")
    print(f"Extraction path: {args.extract_prefer}")
    print(f"Estimated cost: ~${est_cost:.4f}")

    # ---- Cache-poisoning guard ------------------------------------
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

    # ---- Set up System E ------------------------------------------
    print("\nSetting up System E (temporal hybrid C+D) ...")
    embedder = Embedder(paths.cache_dir, dry_run=dry)
    llm = ChatLLM(paths.cache_dir, dry_run=dry)
    retriever = TemporalHybridRetriever(
        paths, cfg, embedder, llm=llm,
        temporal_index_path=tindex_path, buckets_path=buckets_path,
        fusion=args.fusion,
    )
    retriever.setup()
    pipe = Pipeline(retriever, llm)

    # ---- Run each question ----------------------------------------
    print(f"\nAnswering {len(questions)} questions ...")
    answers = []
    scored = []
    extract_hits = 0
    extract_scorable = 0
    for i, q in enumerate(questions, start=1):
        ext = extraction_accuracy(q, llm if not dry else None,
                                  args.extract_prefer)
        if ext["extract_verdict"] in ("hit", "miss"):
            extract_scorable += 1
            if ext["extract_verdict"] == "hit":
                extract_hits += 1

        rec = pipe.answer(q["qid"], q["question"])
        answers.append({**asdict(rec), "extraction": ext})

        score = score_one(q, rec.answer)
        bucket_ids = rec.diagnostics.get("bucket_ids", [])
        n_seeds = rec.diagnostics.get("n_temporal_seeds", 0)
        route = rec.diagnostics.get("route", "?")
        scored.append({
            "fusion": args.fusion,
            "route": route,
            "qid": q["qid"],
            "category": q["category"],
            "sub_type": q["sub_type"],
            "gold_answer": q["gold_answer"],
            "gold_anchors": q.get("time_anchors", []),
            "answer": rec.answer,
            "cached": rec.cached,
            "extract_verdict": ext["extract_verdict"],
            "extract_source": ext["extract_source"],
            "bucket_ids": bucket_ids,
            "n_temporal_seeds": n_seeds,
            **score,
        })
        ok = "✓" if score["match"] else "✗"
        ev = {"hit": "E✓", "miss": "E✗", "n/a": "E–"}[ext["extract_verdict"]]
        nb = bucket_ids
        nb_str = ",".join(nb) if len(nb) <= 4 else f"{len(nb)} buckets"
        spu = f"  +{score['n_spurious']} spu" if score["n_spurious"] else ""
        if route == "entity":
            eviz = f"->C ({n_seeds} seeds)"
        elif route == "community":
            eviz = f"->D [{nb_str}]"
        else:
            eviz = f"C+D [{nb_str}]"
        print(f"  {i:>2}/{len(questions)}  {q['qid']:<14} {ok} {ev}  "
              f"{eviz}{spu}")

    # ---- Write artefacts ------------------------------------------
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    stem = "pilot_systemE" if args.fusion == "route" else "pilot_systemE_concat"
    ans_path = paths.results_dir / f"{stem}_answers.jsonl"
    scored_path = paths.results_dir / f"{stem}_scored.jsonl"
    with ans_path.open("w") as f:
        for a in answers:
            f.write(json.dumps(a, ensure_ascii=False, default=str) + "\n")
    with scored_path.open("w") as f:
        for s in scored:
            f.write(json.dumps(s, ensure_ascii=False, default=str) + "\n")
    print(f"\nWrote {ans_path}")
    print(f"Wrote {scored_path}")

    # ---- Summary --------------------------------------------------
    md = write_summary(scored, mode=f"System E [{args.fusion}] — {mode}",
                       title=f"System E (temporal hybrid C+D, fusion={args.fusion})")
    # route breakdown
    from collections import Counter
    route_counts = Counter(s_["route"] for s_ in scored)
    md += ("\n## Routing (System E)\n"
           f"- Fusion mode: {args.fusion}\n"
           f"- Route breakdown: "
           + ", ".join(f"{k}={v}" for k, v in route_counts.items()) + "\n")
    ext_rate = (100 * extract_hits / extract_scorable
                if extract_scorable else 0.0)
    md += (
        f"\n## Extraction (System E)\n"
        f"- Questions with a gold time_range: {extract_scorable}/"
        f"{len(questions)}\n"
        f"- Extraction window overlapped gold: {extract_hits}/"
        f"{extract_scorable}  ({ext_rate:.0f}%)\n"
        f"- Extraction path: {args.extract_prefer}\n"
    )
    summary_path = paths.results_dir / f"{stem}_summary.md"
    summary_path.write_text(md)
    print(f"Wrote {summary_path}\n")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
