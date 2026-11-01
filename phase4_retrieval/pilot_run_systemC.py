"""
============================================================
Phase 5.2 — Pilot run: System C (Entity-Temporal) on pilot Qs
============================================================

Runs the temporal-entity retriever (System C) against the same
30 temporal pilot questions used for Vanilla v2, with the same
auto-scorer, so the two are directly comparable.

Reuses (imported, NOT duplicated):
  - score_one / write_summary  from pilot_run_vanilla_v2.py
  - the chat-cache poisoning detector pattern

Adds:
  - extraction-accuracy logging: compares the time window the
    extractor produced against the question's gold `time_range`
    field. This is the only place the gold metadata is read, and
    it is read for *measurement only*, never fed to the retriever.

Output
------
  outputs/results/pilot_systemC_answers.jsonl
  outputs/results/pilot_systemC_scored.jsonl
  outputs/results/pilot_systemC_summary.md

Cost
----
  30 questions × (1 extraction call + 1 generation call)
  ≈ 60 × $0.001  ≈  $0.06   (gpt-4o-mini, cached node embeds)

Usage
-----
  python phase4_retrieval/pilot_run_systemC.py            # DRY by default
  python phase4_retrieval/pilot_run_systemC.py --real
  python phase4_retrieval/pilot_run_systemC.py --real --extract-prefer rule
                                          # skip LLM extraction (free)
============================================================
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

# --- sys.path setup ----------------------------------------------
_HERE = Path(__file__).resolve().parent              # phase4_retrieval/
_ROOT = _HERE.parent                                  # code/
# Order matters. phase4_retrieval must win the `retrievers` package
# name (it has retrievers/base.py + retrievers/entity.py), so it is
# inserted LAST (→ ends up at sys.path[0]). We deliberately do NOT
# put phase5_temporal/retrievers on the path, because it would shadow
# phase4_retrieval/retrievers; temporal_entity is loaded by file path
# below instead.
sys.path.insert(0, str(_ROOT / "phase5_temporal"))
sys.path.insert(0, str(_ROOT / "phase2_kg"))
sys.path.insert(0, str(_HERE))                        # → sys.path[0]

from config import Paths, RetrievalConfig             # noqa: E402
from embeddings import Embedder                       # noqa: E402
from llm import ChatLLM                               # noqa: E402
from pipeline import Pipeline                         # noqa: E402

# Reuse the scorer + summary writer from the vanilla runner.
from pilot_run_vanilla_v2 import (                    # noqa: E402
    score_one, write_summary,
)

# System C lives in phase5_temporal/retrievers/temporal_entity.py.
# Load it by explicit file path so it does not require putting that
# directory on sys.path (which would shadow phase4_retrieval's
# `retrievers` package). The module itself resolves `retrievers.entity`
# correctly because phase4_retrieval is already on sys.path.
import importlib.util as _ilu                          # noqa: E402
_te_path = _ROOT / "phase5_temporal" / "retrievers" / "temporal_entity.py"
_te_spec = _ilu.spec_from_file_location("temporal_entity", _te_path)
_te_mod = _ilu.module_from_spec(_te_spec)
_te_spec.loader.exec_module(_te_mod)
TemporalEntityRetriever = _te_mod.TemporalEntityRetriever

from time_extract import extract_time_constraint      # noqa: E402


# ----------------------------------------------------------------
# Path overrides for the temporal pilot
# ----------------------------------------------------------------

def make_paths(project_root: Path) -> Paths:
    """Paths pointed at the v2 chunks / temporal pilot questions.

    System C does NOT use the chunks (it's a graph retriever), but
    we keep chunk paths pointed at v2 for consistency. The node
    embeds and KG are the prior-paper artefacts; the pilot
    questions are the temporal set.
    """
    return Paths(
        data_dir=project_root / "data",
        kg_path=project_root / "outputs" / "kg_schema_aware.pkl",
        node_embeds_path=project_root / "outputs" / "node_embeds.npy",
        pilot_questions=project_root / "outputs" /
                        "temporal_pilot_questions.jsonl",
        results_dir=project_root / "outputs" / "results",
        cache_dir=project_root / "outputs" / "cache",
    )


# ----------------------------------------------------------------
# Extraction-accuracy helper (measurement only)
# ----------------------------------------------------------------

def extraction_accuracy(q: dict, llm, prefer: str) -> dict:
    """Compare the extracted window(s) against the gold time_range.

    Reads q["time_range"] (gold) only to score the extractor — never
    passed to the retriever. Returns a dict with the extracted
    windows and a coarse correctness flag.

    Correctness rule: if the question carries a gold time_range
    {start, end}, at least one extracted window must overlap it.
    If there is no gold time_range (None), we score "n/a".
    """
    tc = extract_time_constraint(q["question"], llm=llm, prefer=prefer)
    gold = q.get("time_range")
    if gold is None:
        verdict = "n/a"
        overlap = None
    else:
        g_start, g_end = gold["start"], gold["end"]
        overlap = any(
            (w[0] < g_end and w[1] > g_start) for w in tc.windows
        )
        verdict = "hit" if overlap else "miss"
    return {
        "extracted_windows": tc.windows,
        "extract_source": tc.source,
        "gold_time_range": gold,
        "extract_verdict": verdict,
    }


# ================================================================
# Main
# ================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pilot run of System C (entity-temporal) on the "
                    "temporal pilot questions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--project-root", type=Path, default=_ROOT,
    )
    parser.add_argument(
        "--real", action="store_true",
        help="Run real OpenAI calls (default: dry-run).",
    )
    parser.add_argument(
        "--extract-prefer", choices=["llm", "rule"], default="llm",
        help="Time-extraction path: 'llm' (default) or 'rule' (free).",
    )
    parser.add_argument(
        "--max-questions", type=int, default=None,
    )
    parser.add_argument(
        "--yes", action="store_true",
    )
    args = parser.parse_args()
    dry = not args.real

    paths = make_paths(args.project_root)
    cfg = RetrievalConfig()
    # Make the extraction preference visible to the retriever too.
    cfg.time_extract_prefer = args.extract_prefer

    # ---- Load questions -------------------------------------------
    if not paths.pilot_questions.exists():
        print(f"ERROR: {paths.pilot_questions} not found.", file=sys.stderr)
        return 1
    with paths.pilot_questions.open() as f:
        questions = [json.loads(line) for line in f]
    if args.max_questions:
        questions = questions[: args.max_questions]
    print(f"Loaded {len(questions)} pilot questions")

    # ---- Check temporal index exists ------------------------------
    tindex_path = args.project_root / "outputs" / "temporal_index.pkl"
    if not tindex_path.exists():
        print(f"ERROR: {tindex_path} not found. Run "
              f"phase5_temporal/temporal_index.py first.", file=sys.stderr)
        return 1

    # ---- Cost estimate + mode -------------------------------------
    # 2 calls per question (extract + generate) unless --extract-prefer rule
    calls_per_q = 1 if args.extract_prefer == "rule" else 2
    est_cost = len(questions) * calls_per_q * 0.001
    mode = "DRY RUN (no API call)" if dry else "REAL (OpenAI API)"
    print(f"\nMode: {mode}")
    print(f"Extraction path: {args.extract_prefer}")
    print(f"Estimated cost: ~${est_cost:.4f}")

    # ---- Cache-poisoning guard (same as vanilla runner) -----------
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
                return 3

        if est_cost > 0.5 and not args.yes:
            ans = input("Proceed? [y/N] ").strip().lower()
            if ans not in ("y", "yes"):
                print("Aborted.")
                return 0

    # ---- Set up System C ------------------------------------------
    print("\nSetting up System C (entity-temporal) ...")
    embedder = Embedder(paths.cache_dir, dry_run=dry)
    llm = ChatLLM(paths.cache_dir, dry_run=dry)
    retriever = TemporalEntityRetriever(
        paths, cfg, embedder, llm=llm,
        temporal_index_path=tindex_path,
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
        # Extraction accuracy (measurement only; uses its own cached
        # extraction call, which the retriever will hit again from
        # cache — no double cost).
        ext = extraction_accuracy(q, llm if not dry else None,
                                  args.extract_prefer)
        if ext["extract_verdict"] in ("hit", "miss"):
            extract_scorable += 1
            if ext["extract_verdict"] == "hit":
                extract_hits += 1

        rec = pipe.answer(q["qid"], q["question"])
        answers.append({**asdict(rec), "extraction": ext})

        score = score_one(q, rec.answer)
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
            "extracted_windows": ext["extracted_windows"],
            **score,
        })
        ok = "✓" if score["match"] else "✗"
        ev = {"hit": "E✓", "miss": "E✗", "n/a": "E–"}[ext["extract_verdict"]]
        spu = f"  +{score['n_spurious']} spu" if score["n_spurious"] else ""
        print(f"  {i:>2}/{len(questions)}  {q['qid']:<14} {ok} {ev}{spu}")

    # ---- Write artefacts ------------------------------------------
    paths.results_dir.mkdir(parents=True, exist_ok=True)
    ans_path = paths.results_dir / "pilot_systemC_answers.jsonl"
    scored_path = paths.results_dir / "pilot_systemC_scored.jsonl"
    with ans_path.open("w") as f:
        for a in answers:
            f.write(json.dumps(a, ensure_ascii=False, default=str) + "\n")
    with scored_path.open("w") as f:
        for s in scored:
            f.write(json.dumps(s, ensure_ascii=False, default=str) + "\n")
    print(f"\nWrote {ans_path}")
    print(f"Wrote {scored_path}")

    # ---- Summary --------------------------------------------------
    md = write_summary(scored, mode=f"System C — {mode}",
                       title="System C (entity-temporal)")
    # Append the extraction-accuracy block.
    ext_rate = (100 * extract_hits / extract_scorable
                if extract_scorable else 0.0)
    md += (
        f"\n## Extraction accuracy (System C only)\n"
        f"- Questions with a gold time_range: {extract_scorable}/"
        f"{len(questions)}\n"
        f"- Extraction window overlapped gold: {extract_hits}/"
        f"{extract_scorable}  ({ext_rate:.0f}%)\n"
        f"- Extraction path: {args.extract_prefer}\n"
    )
    summary_path = paths.results_dir / "pilot_systemC_summary.md"
    summary_path.write_text(md)
    print(f"Wrote {summary_path}\n")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
