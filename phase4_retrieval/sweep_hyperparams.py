"""
============================================================
Phase 5 — Hyperparameter sweep on a 30-question dev set
============================================================

Sweeps the four retrieval-strategy hyperparameters
documented in the paper:

    Entity ($h$-hop)         h ∈ {1, 2, 3}
    Community ($k$ summaries) k ∈ {3, 5, 10}
    Hybrid (top-k entity + top-k community per A.1.3)
    Vanilla ($k$ chunks)      k ∈ {5, 10, 20}

For each (strategy, hyperparam) cell we run the pipeline on 30
held-out dev questions, score the closed-category answers with
metrics.py, and pick the value that maximises overall accuracy
on the closed subset.

Output:
  outputs/results/sweep_dev_set.jsonl    — 30 dev question qids
  outputs/results/sweep_results.json     — full {strategy: {value: acc}}
                                            table + best-value picks
  Console output suitable for paste into a LaTeX table.

Dev set selection: stratified random sample of 30 questions
(6 per category × 5), seed=42, written once and reused across
all subsequent sweep runs.

Notes
-----
- Path strategy is NotImplementedError; we skip it.
- We sweep over the **first 6 questions per category** as the dev
  set; the remaining 120 questions remain the test set. This is the
  standard 20%/80% split (paper uses N_dev = 30, N_test = 120).
- All embedding work is cache-hit (chunk + node + community
  matrices were built in the headline run); the sweep only spends
  on generation calls (~360 calls ≈ $0.40).

Usage
-----
  cd phase4_retrieval
  python sweep_hyperparams.py --no-dry-run --yes
============================================================
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Paths, RetrievalConfig
from embeddings import Embedder
from llm import ChatLLM
from pipeline import Pipeline, write_answers
from retrievers import REGISTRY
from metrics import score_answer, CLOSED_CATEGORIES


# ----------------------------------------------------------------
# Sweep grids — match the ranges documented in the paper
# (sec:experiments:impl, Table "Retrieval hyperparameters")
# ----------------------------------------------------------------
SWEEP_GRID = {
    "vanilla_rag": {"chunk_top_k":   [5, 10, 20]},
    "entity":      {"entity_hops":   [1, 2, 3]},
    "community":   {"community_top_k": [3, 5, 10]},
    # Hybrid: sweep the entity:community ratio at fixed total = 5
    "hybrid":      {"hybrid_split":  [(1, 4), (3, 2), (4, 1)]},  # (ent, com)
}

# Defaults shown in the paper table (before sweep).
PAPER_DEFAULTS = {
    "vanilla_rag": {"chunk_top_k": 10},
    "entity":      {"entity_hops": 2},
    "community":   {"community_top_k": 5},
    "hybrid":      {"hybrid_split": (3, 2)},  # ent=3, com=2 → split favouring entity
}


# ----------------------------------------------------------------
# Dev set selection
# ----------------------------------------------------------------
def select_dev_set(questions: list[dict], n_per_category: int = 6,
                   seed: int = 42) -> list[str]:
    """Stratified random sample. Returns qids."""
    import random
    rng = random.Random(seed)
    by_cat = defaultdict(list)
    for q in questions:
        by_cat[q["category"]].append(q["qid"])
    dev = []
    for cat in sorted(by_cat):
        ids = sorted(by_cat[cat])  # deterministic
        rng.shuffle(ids)
        dev.extend(ids[:n_per_category])
    return sorted(dev)


def make_config(strategy: str, value) -> RetrievalConfig:
    """Build a RetrievalConfig with one hyperparameter set."""
    cfg = RetrievalConfig()
    if strategy == "vanilla_rag":
        return replace(cfg, chunk_top_k=value)
    if strategy == "entity":
        return replace(cfg, entity_hops=value)
    if strategy == "community":
        return replace(cfg, community_top_k=value)
    if strategy == "hybrid":
        ent_k, com_k = value
        return replace(cfg,
                       hybrid_entity_top_k=ent_k,
                       hybrid_community_top_k=com_k)
    raise ValueError(strategy)


# ----------------------------------------------------------------
# Accuracy on a dev split (closed categories only)
# ----------------------------------------------------------------
def score_dev(answers: list[dict], questions_by_qid: dict[str, dict]) -> dict:
    """Return per-category and overall closed-question accuracy."""
    by_cat = defaultdict(lambda: {"correct": 0, "n": 0})
    for a in answers:
        q = questions_by_qid[a["qid"]]
        if q["category"] not in CLOSED_CATEGORIES:
            continue
        outcome = score_answer(q, a["answer"]).outcome
        by_cat[q["category"]]["n"] += 1
        if outcome == "correct":
            by_cat[q["category"]]["correct"] += 1
    overall_n = sum(c["n"] for c in by_cat.values())
    overall_correct = sum(c["correct"] for c in by_cat.values())
    return {
        "per_category": {
            cat: {"acc": c["correct"]/c["n"] if c["n"] else 0.0,
                  "n": c["n"]}
            for cat, c in by_cat.items()
        },
        "closed_overall": {
            "acc": overall_correct/overall_n if overall_n else 0.0,
            "n": overall_n,
        }
    }


# ----------------------------------------------------------------
# Main sweep
# ----------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", dest="dry_run", action="store_true",
                    default=False)
    ap.add_argument("--yes", action="store_true",
                    help="Skip cost confirmation")
    ap.add_argument("--n-per-category", type=int, default=6)
    args = ap.parse_args()

    paths = Paths()
    questions = [json.loads(l) for l in paths.pilot_questions.open()]
    questions_by_qid = {q["qid"]: q for q in questions}

    # Dev set selection (write once, reuse)
    dev_qids = select_dev_set(questions, n_per_category=args.n_per_category)
    dev_path = paths.results_dir / "sweep_dev_set.jsonl"
    dev_path.parent.mkdir(parents=True, exist_ok=True)
    dev_questions = [q for q in questions if q["qid"] in set(dev_qids)]
    with dev_path.open("w") as fp:
        for q in dev_questions:
            fp.write(json.dumps({"qid": q["qid"],
                                 "category": q["category"]}) + "\n")
    print(f"Dev set: {len(dev_qids)} questions written to {dev_path}")
    print(f"  Per category: " + ", ".join(
        f"{c}={sum(1 for q in dev_questions if q['category']==c)}"
        for c in sorted({q['category'] for q in dev_questions})
    ))

    # Cost estimate
    n_calls = sum(len(grid)*3 for grid in [
        SWEEP_GRID[s] for s in SWEEP_GRID
    ]) * len(dev_qids)
    # Roughly: 4 strategies * 3 values * 30 questions = 360
    n_calls = 4 * 3 * len(dev_qids)
    est = n_calls * 0.0012  # ~1.5k in + 100 out @ gpt-4o-mini ≈ $0.0012
    print(f"\nEstimated cost: ${est:.2f} ({n_calls} generation calls)")
    if not args.dry_run and not args.yes:
        print("Re-run with --yes to confirm.")
        return 0

    embedder = Embedder(paths.cache_dir, dry_run=args.dry_run)
    llm = ChatLLM(paths.cache_dir, dry_run=args.dry_run)

    # ----------------------------------------------------------------
    # Run sweep
    # ----------------------------------------------------------------
    sweep_results = {}
    for strategy, grid in SWEEP_GRID.items():
        print(f"\n=== {strategy} ===")
        sweep_results[strategy] = {"sweep": [], "best": None}
        param_name = list(grid.keys())[0]
        for value in grid[param_name]:
            cfg = make_config(strategy, value)
            retriever = REGISTRY[strategy](paths, cfg, embedder)
            retriever.setup()
            pipe = Pipeline(retriever, llm)

            answers = []
            for q in dev_questions:
                rec = pipe.answer(q["qid"], q["question"])
                answers.append(asdict(rec))

            scores = score_dev(answers, questions_by_qid)
            print(f"  {param_name}={value}: closed_overall_acc="
                  f"{scores['closed_overall']['acc']:.3f} "
                  f"(n={scores['closed_overall']['n']})")
            sweep_results[strategy]["sweep"].append({
                "value": value,
                "scores": scores,
            })

        # Pick best by overall closed accuracy
        best = max(sweep_results[strategy]["sweep"],
                   key=lambda x: x["scores"]["closed_overall"]["acc"])
        sweep_results[strategy]["best"] = best["value"]
        sweep_results[strategy]["paper_default"] = (
            list(PAPER_DEFAULTS[strategy].values())[0]
        )
        print(f"  → best on dev: {param_name}={best['value']}  "
              f"(paper default: {list(PAPER_DEFAULTS[strategy].values())[0]})")

    # ----------------------------------------------------------------
    # Write sweep results
    # ----------------------------------------------------------------
    out_path = paths.results_dir / "sweep_results.json"
    with out_path.open("w") as fp:
        json.dump(sweep_results, fp, indent=2, default=str)
    print(f"\nWrote sweep table to {out_path}")

    # Summary table
    print("\n" + "="*60)
    print("  SWEEP SUMMARY")
    print("="*60)
    print(f"\n{'strategy':14s}  {'param':18s}  {'best':>12s}  "
          f"{'paper default':>14s}  {'closed_acc':>10s}")
    for strategy, grid in SWEEP_GRID.items():
        param_name = list(grid.keys())[0]
        r = sweep_results[strategy]
        best = r["best"]
        default = r["paper_default"]
        best_acc = next(
            x["scores"]["closed_overall"]["acc"]
            for x in r["sweep"] if x["value"] == best
        )
        match = "✓" if best == default else "≠"
        print(f"  {strategy:14s}  {param_name:18s}  {str(best):>12s}  "
              f"{str(default):>14s}  {best_acc:>9.1%} {match}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
