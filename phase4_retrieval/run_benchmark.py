"""
============================================================
Phase 4 — Benchmark runner (CLI)
============================================================

End-to-end orchestration:
  1. Load pilot_questions.jsonl.
  2. For each --system: setup retriever, run pipeline on every
     question, write answers JSONL.
  3. (Optional) --judge: run pairwise judge against a reference
     system (e.g. vanilla_rag) on open-ended categories.

DEFAULT BEHAVIOR IS DRY-RUN. To actually spend API credits:
  - Pass --no-dry-run
  - The script will print an estimated cost and require explicit
    confirmation via --yes if it exceeds CostGuards.confirm_cost_above_usd.

Example invocations
-------------------

  # 1. Smoke test — dry run, 10 questions, all systems
  python run_benchmark.py --systems vanilla_rag entity community \\
      --max-questions 10

  # 2. Real run on a single system (cost estimate first)
  python run_benchmark.py --systems entity --no-dry-run --yes

  # 3. Judge two pre-computed answer files
  python run_benchmark.py --judge \\
      --judge-a outputs/results/entity_answers.jsonl \\
      --judge-b outputs/results/vanilla_rag_answers.jsonl
============================================================
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

# Import order matters because of sys.path tricks in retrievers/*
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import Paths, RetrievalConfig, CostGuards
from embeddings import Embedder
from llm import ChatLLM
from pipeline import Pipeline, write_answers, AnswerRecord
from retrievers import REGISTRY
from judge import make_judge_llm, judge_pair


# ----------------------------------------------------------------
# Cost estimation (rough — based on May 2026 pricing)
# ----------------------------------------------------------------
def estimate_cost(n_questions: int, n_systems: int,
                  avg_input_tokens: int = 1500,
                  avg_output_tokens: int = 200) -> float:
    """Very rough $-estimate for one benchmark pass."""
    gen_in_rate = 0.15 / 1_000_000     # gpt-4o-mini
    gen_out_rate = 0.60 / 1_000_000
    emb_rate = 0.02 / 1_000_000
    # Embeddings (questions only — indexes are amortized)
    emb_cost = n_questions * 50 * emb_rate
    gen_cost = n_questions * n_systems * (
        avg_input_tokens * gen_in_rate + avg_output_tokens * gen_out_rate
    )
    return emb_cost + gen_cost


# ----------------------------------------------------------------
# Load questions
# ----------------------------------------------------------------
def load_questions(path: Path, limit: int | None = None) -> list[dict]:
    with path.open() as fp:
        qs = [json.loads(line) for line in fp]
    return qs[:limit] if limit else qs


# ----------------------------------------------------------------
# Generation runner
# ----------------------------------------------------------------
def run_generation(args, paths: Paths, cfg: RetrievalConfig) -> None:
    questions = load_questions(paths.pilot_questions, limit=args.max_questions)
    print(f"Loaded {len(questions)} questions from {paths.pilot_questions}")

    # Cost guard
    est = estimate_cost(len(questions), len(args.systems))
    print(f"Estimated cost: ${est:.4f} "
          f"({'DRY RUN' if args.dry_run else 'REAL'} mode)")
    if not args.dry_run and est > CostGuards.confirm_cost_above_usd \
            and not args.yes:
        print(f"Estimated cost ${est:.2f} exceeds "
              f"${CostGuards.confirm_cost_above_usd:.2f}. "
              f"Re-run with --yes to confirm.")
        return

    embedder = Embedder(paths.cache_dir, dry_run=args.dry_run)
    llm = ChatLLM(paths.cache_dir, dry_run=args.dry_run)

    paths.results_dir.mkdir(parents=True, exist_ok=True)

    for sys_name in args.systems:
        if sys_name not in REGISTRY:
            print(f"  [SKIP] unknown system: {sys_name}")
            continue
        print(f"\n=== System: {sys_name} ===")
        retriever = REGISTRY[sys_name](paths, cfg, embedder)
        retriever.setup()
        pipe = Pipeline(retriever, llm)

        answers: list[AnswerRecord] = []
        for i, q in enumerate(questions, start=1):
            rec = pipe.answer(q["qid"], q["question"])
            answers.append(rec)
            if i % 20 == 0:
                print(f"  {i}/{len(questions)} done")

        out_name = f"{sys_name}{args.suffix}_answers.jsonl"
        out_path = paths.results_dir / out_name
        write_answers(answers, out_path)
        print(f"  Wrote {len(answers)} answers to {out_path}")

    # Cache stats
    print("\n--- Cache stats ---")
    print(f"  embeddings: {embedder.cache.stats()}")
    print(f"  chat:       {llm.cache.stats()}")


# ----------------------------------------------------------------
# Judge runner
# ----------------------------------------------------------------
def run_judge(args, paths: Paths) -> None:
    a_records = {r["qid"]: r for r in (
        json.loads(line) for line in args.judge_a.open()
    )}
    b_records = {r["qid"]: r for r in (
        json.loads(line) for line in args.judge_b.open()
    )}
    questions = {q["qid"]: q for q in load_questions(paths.pilot_questions)}

    judge_llm = make_judge_llm(paths, dry_run=args.dry_run)
    verdicts = []
    shared = set(a_records) & set(b_records)
    # Only judge open-ended categories
    eligible = [
        qid for qid in shared
        if questions[qid]["category"] in {"comparison", "root_cause", "global"}
    ]
    print(f"Judging {len(eligible)} open-ended questions "
          f"(out of {len(shared)} shared)")

    for qid in eligible:
        q = questions[qid]
        key_facts = q.get("rubric", {}).get("key_facts", [])
        if not key_facts:
            continue
        v = judge_pair(
            judge_llm,
            qid=qid,
            question=q["question"],
            rubric_key_facts=key_facts,
            sys1_name=a_records[qid]["system"],
            sys1_answer=a_records[qid]["answer"],
            sys2_name=b_records[qid]["system"],
            sys2_answer=b_records[qid]["answer"],
        )
        verdicts.append(v)

    out = paths.results_dir / (
        f"judge_{args.judge_a.stem}_vs_{args.judge_b.stem}.jsonl"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fp:
        for v in verdicts:
            fp.write(json.dumps(asdict(v)) + "\n")
    print(f"Wrote {len(verdicts)} verdicts to {out}")

    # Quick tally
    from collections import Counter
    tally = Counter(v.winner_system for v in verdicts)
    print(f"Tally: {dict(tally)}")


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+",
                    default=["vanilla_rag", "entity", "community"],
                    help="Which retrieval systems to run")
    ap.add_argument("--max-questions", type=int, default=None,
                    help="Cap on #questions (smoke testing)")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true",
                    default=True,
                    help="No API calls; use placeholder embeddings/answers")
    ap.add_argument("--no-dry-run", dest="dry_run", action="store_false",
                    help="Actually call the OpenAI API")
    ap.add_argument("--yes", action="store_true",
                    help="Skip cost confirmation prompt")

    # Judge mode
    ap.add_argument("--judge", action="store_true",
                    help="Run pairwise judge mode instead of generation")
    ap.add_argument("--judge-a", type=Path,
                    help="Path to answers JSONL for system A")
    ap.add_argument("--judge-b", type=Path,
                    help="Path to answers JSONL for system B")

    # Paths
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--kg-path", type=Path, default=None)
    ap.add_argument("--pilot-questions", type=Path, default=None)
    ap.add_argument("--suffix", type=str, default="",
                    help="Suffix appended to per-system output files "
                         "(e.g. '_auto'). Also suffixes the node_embeds "
                         "cache path so two KGs don't share embeddings.")
    args = ap.parse_args()

    paths = Paths()
    if args.data_dir: paths.data_dir = args.data_dir
    if args.kg_path: paths.kg_path = args.kg_path
    if args.pilot_questions: paths.pilot_questions = args.pilot_questions

    # If we're running with a non-default KG (suffix set), make sure
    # the node embedding matrix is cached separately so we don't reuse
    # the schema-aware embeddings on auto-extracted node descriptions.
    if args.suffix:
        suffix = args.suffix if args.suffix.startswith("_") else f"_{args.suffix}"
        paths.node_embeds_path = (
            paths.node_embeds_path.with_name(
                paths.node_embeds_path.stem + suffix
                + paths.node_embeds_path.suffix)
        )
        args.suffix = suffix

    cfg = RetrievalConfig()

    if args.judge:
        if not args.judge_a or not args.judge_b:
            print("--judge requires --judge-a and --judge-b", file=sys.stderr)
            return 1
        run_judge(args, paths)
    else:
        run_generation(args, paths, cfg)

    return 0


if __name__ == "__main__":
    sys.exit(main())
