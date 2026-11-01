#!/usr/bin/env python3
"""
============================================================
Phase 5.7 — Headline experiment runner (7 systems × N questions)
============================================================

Generalises the four validated pilot runners
(pilot_run_vanilla_v2 / systemC / systemD / systemE) into ONE
parameterised driver, so the headline matrix can be produced with a
single script and identical scoring across systems.

Systems (the 7 of the paper):
  time-blind : vanilla_v2  entity  community  hybrid
  temporal   : entity_temporal(C)  community_temporal(D)  hybrid_temporal(E)

Question sets:
  - outputs/temporal_questions_90.jsonl   → all 7 systems (the contribution)
  - benchmark/pilot_questions.jsonl (150) → the 4 time-blind systems only
    (the prior-paper time-blind anchor; design decision 2026-06-04 —
    temporal systems are NOT run on the 150 to avoid double-jeopardy).

Scoring:
  - closed-form gold (number / dict / list / string) → score_one
    (the fixed shared scorer; handles temporal gold AND the prior
    factual/aggregation gold). Spurious-anchor count recorded too.
  - open gold (gold_answer_type == "rubric"): generation only; the
    record carries needs_judge=True and is scored later by the
    pairwise judge (run_benchmark.py --judge). score is left None.

Temporal-faithfulness is scored separately by
phase5_temporal/temporal_faithfulness.py over the written answers.

Reuses unchanged: Pipeline, REGISTRY, VanillaRAG, the temporal
retrievers (file-path load), score_one + write_summary + the chat-
cache poisoning guard from pilot_run_vanilla_v2.

Usage:
  # one system, dry run (no API) — smoke test wiring:
  python phase4_retrieval/run_headline.py --system community_temporal \\
      --questions outputs/temporal_questions_90.jsonl
  # all 7 on the 90 temporal questions (real):
  python phase4_retrieval/run_headline.py --all-temporal --real --yes \\
      --questions outputs/temporal_questions_90.jsonl
  # the 4 time-blind systems on the 150 prior questions (real):
  python phase4_retrieval/run_headline.py --all-timeblind --real --yes \\
      --questions benchmark/pilot_questions.jsonl
============================================================
"""
from __future__ import annotations

import argparse
import importlib.util as _ilu
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent          # phase4_retrieval/
_ROOT = _HERE.parent
# Order matters (phase4_retrieval must win the `retrievers` package name,
# so it is inserted LAST → sys.path[0]). phase5_temporal is needed for
# `temporal_index`, phase2_kg for `build_kg`.
sys.path.insert(0, str(_ROOT / "phase5_temporal"))
sys.path.insert(0, str(_ROOT / "phase2_kg"))
sys.path.insert(0, str(_HERE))

# --- phase4 imports first (puts retrievers package on the path) -----
from config import Paths, RetrievalConfig          # noqa: E402
from llm import ChatLLM                             # noqa: E402
from embeddings import Embedder                     # noqa: E402
from pipeline import Pipeline                       # noqa: E402
from retrievers import REGISTRY                     # noqa: E402
from retrievers.vanilla_rag import VanillaRAG       # noqa: E402
# reuse the validated scorer + summary writer (the shared scorer)
from pilot_run_vanilla_v2 import score_one, write_summary  # noqa: E402


# --- file-path load of the temporal retrievers (like the pilots) ----
def _load_class(modname: str, filename: str, clsname: str):
    path = _ROOT / "phase5_temporal" / "retrievers" / filename
    spec = _ilu.spec_from_file_location(modname, path)
    mod = _ilu.module_from_spec(spec)
    sys.modules[modname] = mod                       # so dataclasses resolve
    spec.loader.exec_module(mod)
    return getattr(mod, clsname)


TemporalEntityRetriever = _load_class(
    "temporal_entity", "temporal_entity.py", "TemporalEntityRetriever")
TemporalCommunityRetriever = _load_class(
    "temporal_community", "temporal_community.py", "TemporalCommunityRetriever")
TemporalHybridRetriever = _load_class(
    "temporal_hybrid", "temporal_hybrid.py", "TemporalHybridRetriever")


TIMEBLIND = ["vanilla_v2", "entity", "community", "hybrid"]
TEMPORAL = ["entity_temporal", "community_temporal", "hybrid_temporal"]
ALL_SYSTEMS = TIMEBLIND + TEMPORAL

# canonical 'system' string written into each answer record, matching
# the names the faithfulness judge / pilot files already use.
SYSTEM_TAG = {
    "vanilla_v2": "vanilla_rag", "entity": "entity", "community": "community",
    "hybrid": "hybrid", "entity_temporal": "entity_temporal",
    "community_temporal": "community_temporal",
    "hybrid_temporal": "hybrid_temporal",
}


def build_gen_llm(cache_dir: Path, gen_model: str | None,
                  gen_provider: str, dry: bool,
                  extra_body: dict | None = None,
                  max_tokens: int | None = None) -> ChatLLM:
    """The answer-generation LLM. Default = the pinned OpenAI gpt-4o-mini
    (config.GENERATION_MODEL). A --gen-model routes generation to the GenAI
    Gateway (gen_provider='genai') or to OpenAI (gen_provider='openai', e.g.
    gpt-4o) while leaving retrieval and extraction untouched. extra_body /
    max_tokens let a run disable a reasoning model's thinking or lengthen
    its budget."""
    kw: dict = {}
    if extra_body:
        kw["extra_body"] = extra_body
    if max_tokens:
        kw["max_tokens"] = max_tokens
    if not gen_model:
        return ChatLLM(cache_dir, dry_run=dry, **kw)
    if gen_provider == "openai":
        return ChatLLM(cache_dir, model=gen_model, dry_run=dry, **kw)
    return ChatLLM.genai_gateway(cache_dir, model=gen_model, dry_run=dry, **kw)


def build_paths(project_root: Path, questions: Path) -> Paths:
    """One Paths populated for every system: v2 chunks for vanilla,
    KG + node/community embeds for the graph systems."""
    out = project_root / "outputs"
    return Paths(
        data_dir=project_root / "data",
        kg_path=out / "kg_schema_aware.pkl",
        node_embeds_path=out / "node_embeds.npy",
        community_embeds_path=out / "community_embeds.npy",
        community_summaries=out / "community_summaries.jsonl",
        chunks_path=out / "vanilla_chunks_v2.jsonl",
        chunk_embeds_path=out / "vanilla_chunk_embeds_v2.npy",
        pilot_questions=questions,
        results_dir=out / "results",
        cache_dir=out / "cache",
    )


def build_system(name: str, paths: Paths, cfg: RetrievalConfig,
                 embedder: Embedder, llm: ChatLLM, project_root: Path,
                 buckets_override: Path | None = None):
    """Factory mirroring the four pilot runners exactly."""
    if name == "vanilla_v2":
        return VanillaRAG(paths, cfg, embedder)
    if name in ("entity", "community", "hybrid"):
        return REGISTRY[name](paths, cfg, embedder)
    tindex = project_root / "outputs" / "temporal_index.pkl"
    buckets = buckets_override or (project_root / "outputs" / "time_buckets.jsonl")
    if name == "entity_temporal":
        return TemporalEntityRetriever(paths, cfg, embedder, llm=llm,
                                       temporal_index_path=tindex)
    if name == "community_temporal":
        return TemporalCommunityRetriever(paths, cfg, embedder, llm=llm,
                                          buckets_path=buckets)
    if name == "hybrid_temporal":
        cfg.temporal_fusion = "route"
        return TemporalHybridRetriever(paths, cfg, embedder, llm=llm,
                                       temporal_index_path=tindex,
                                       buckets_path=buckets, fusion="route")
    raise ValueError(f"unknown system: {name}")


def _required_artifacts(name: str, project_root: Path) -> list[Path]:
    out = project_root / "outputs"
    if name == "vanilla_v2":
        return [out / "vanilla_chunks_v2.jsonl", out / "vanilla_chunk_embeds_v2.npy"]
    if name in ("entity", "community", "hybrid"):
        need = [out / "kg_schema_aware.pkl", out / "node_embeds.npy"]
        if name in ("community", "hybrid"):
            need += [out / "community_embeds.npy", out / "community_summaries.jsonl"]
        return need
    if name == "entity_temporal":
        return [out / "kg_schema_aware.pkl", out / "temporal_index.pkl"]
    if name == "community_temporal":
        return [out / "time_buckets.jsonl"]
    if name == "hybrid_temporal":
        return [out / "temporal_index.pkl", out / "time_buckets.jsonl"]
    return []


def _check_poisoned(cache_dir: Path) -> list[str]:
    cdir = cache_dir / "chat"
    if not cdir.exists():
        return []
    bad = []
    for entry in cdir.glob("*.json"):
        try:
            rec = json.loads(entry.read_text())
            if rec.get("tokens", {}).get("dry_run"):
                bad.append(entry.name)
        except (json.JSONDecodeError, OSError):
            continue
    return bad


def score_record(q: dict, answer: str, cached: bool) -> dict:
    """Closed → score_one; open (rubric) → defer to the pairwise judge."""
    base = {
        "qid": q["qid"], "category": q.get("category"),
        "sub_type": q.get("sub_type"),
        "gold_answer": q.get("gold_answer"),
        "gold_anchors": q.get("time_anchors", []),
        "answer": answer, "cached": cached,
    }
    if q.get("gold_answer_type") == "rubric":
        base.update({"match": None, "needs_judge": True,
                     "n_spurious": 0, "spurious_anchors": []})
    else:
        base.update(score_one(q, answer))
        base["needs_judge"] = False
    return base


def run_one_system(name: str, questions: list[dict], paths: Paths,
                   cfg: RetrievalConfig, dry: bool, out_prefix: str,
                   buckets_override: Path | None = None,
                   gen_model: str | None = None,
                   gen_provider: str = "genai",
                   gen_extra_body: dict | None = None,
                   gen_max_tokens: int | None = None) -> dict:
    missing = [p for p in _required_artifacts(name, paths.data_dir.parent)
               if not p.exists()]
    if missing and not dry:
        print(f"  [{name}] SKIP — missing artefacts: "
              f"{[m.name for m in missing]}", file=sys.stderr)
        return {"system": name, "skipped": True}

    embedder = Embedder(paths.cache_dir, dry_run=dry)
    # Retrieval-side temporal-constraint extraction (System C/E, time_extract)
    # stays on the pinned gpt-4o-mini so that ONLY the answer generator varies
    # across runs -- this isolates the generation effect from retrieval.
    extract_llm = ChatLLM(paths.cache_dir, dry_run=dry)
    gen_llm = build_gen_llm(paths.cache_dir, gen_model, gen_provider, dry,
                            gen_extra_body, gen_max_tokens)
    retriever = build_system(name, paths, cfg, embedder, extract_llm,
                             paths.data_dir.parent, buckets_override)
    retriever.setup()
    pipe = Pipeline(retriever, gen_llm)
    tag = SYSTEM_TAG[name]

    answers, scored = [], []
    n_match = n_closed = 0
    for i, q in enumerate(questions, start=1):
        rec = pipe.answer(q["qid"], q["question"])
        a = asdict(rec)
        a["system"] = tag
        answers.append(a)
        s = score_record(q, rec.answer, rec.cached)
        scored.append(s)
        if s["match"] is not None:
            n_closed += 1
            n_match += int(s["match"])
        if i % 20 == 0 or i == len(questions):
            print(f"    [{name}] {i}/{len(questions)}")

    paths.results_dir.mkdir(parents=True, exist_ok=True)
    ap = paths.results_dir / f"{out_prefix}_{tag}_answers.jsonl"
    spath = paths.results_dir / f"{out_prefix}_{tag}_scored.jsonl"
    with ap.open("w") as f:
        for a in answers:
            f.write(json.dumps(a, ensure_ascii=False, default=str) + "\n")
    with spath.open("w") as f:
        for s in scored:
            f.write(json.dumps(s, ensure_ascii=False, default=str) + "\n")
    n_judge = sum(1 for s in scored if s.get("needs_judge"))
    print(f"  [{name}] closed match {n_match}/{n_closed}"
          f"{f', {n_judge} open→judge' if n_judge else ''}"
          f" → {ap.name}")
    return {"system": name, "n_match": n_match, "n_closed": n_closed,
            "n_judge": n_judge, "answers": ap, "scored": spath}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[2],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--system", choices=ALL_SYSTEMS)
    g.add_argument("--all-temporal", action="store_true",
                   help="all 7 systems (the 90-question contribution)")
    g.add_argument("--all-timeblind", action="store_true",
                   help="the 4 time-blind systems (the 150 prior anchor)")
    ap.add_argument("--questions", type=Path,
                    default=Path("./outputs/temporal_questions_90.jsonl"))
    ap.add_argument("--project-root", type=Path, default=_ROOT)
    ap.add_argument("--out-prefix", default=None,
                    help="output filename prefix (default: questions stem)")
    ap.add_argument("--force-level", choices=["week", "month", "quarter", "year"],
                    default=None,
                    help="RQ3/H5: force System D's bucket granularity. "
                         "Omit for the bi-level default; 'month' = the flat "
                         "ablation. Only community_temporal (and the D-arm of "
                         "hybrid_temporal) read this.")
    ap.add_argument("--buckets-path", type=Path, default=None,
                    help="Override the time-bucket file for D (e.g. a "
                         "week-augmented file for the RQ3 week sweep). "
                         "Default: outputs/time_buckets.jsonl.")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--max-questions", type=int, default=None)
    ap.add_argument("--gen-model", default=None,
                    help="Answer-generation model. Default: the pinned "
                         "config.GENERATION_MODEL (gpt-4o-mini) on OpenAI. Set "
                         "e.g. 'Llama-3.1-8B-Instruct' to generate via the "
                         "GenAI Gateway; retrieval + temporal extraction stay "
                         "on gpt-4o-mini so only generation varies.")
    ap.add_argument("--gen-provider", choices=["genai", "openai"],
                    default="genai",
                    help="Backend for --gen-model: 'genai' = Fraunhofer GenAI "
                         "Gateway (default), 'openai' = OpenAI cloud (e.g. "
                         "gpt-4o as a stronger generator). Ignored when "
                         "--gen-model is unset.")
    ap.add_argument("--gen-extra-body", default=None,
                    help="JSON dict of extra request params passed to the "
                         "gateway (vLLM passthrough). To switch a reasoning "
                         "model into plain-instruct mode: "
                         "'{\"chat_template_kwargs\": {\"enable_thinking\": false}}'.")
    ap.add_argument("--gen-max-tokens", type=int, default=None,
                    help="Override completion max_tokens for the generator "
                         "(default 512); raise for verbose/reasoning models.")
    args = ap.parse_args()
    dry = not args.real
    gen_extra_body = json.loads(args.gen_extra_body) if args.gen_extra_body else None

    paths = build_paths(args.project_root, args.questions)
    if not args.questions.exists():
        print(f"ERROR: {args.questions} not found.", file=sys.stderr)
        return 1
    with args.questions.open() as f:
        questions = [json.loads(line) for line in f]
    if args.max_questions:
        questions = questions[: args.max_questions]

    if args.all_temporal:
        systems = ALL_SYSTEMS
    elif args.all_timeblind:
        systems = TIMEBLIND
    else:
        systems = [args.system]
    out_prefix = args.out_prefix or args.questions.stem
    if args.force_level and not args.out_prefix:
        out_prefix = f"{out_prefix}_force-{args.force_level}"
    if args.gen_model and not args.out_prefix:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "-", args.gen_model)
        out_prefix = f"{out_prefix}_gen-{safe}"

    n_open = sum(1 for q in questions if q.get("gold_answer_type") == "rubric")
    est = len(questions) * len(systems) * 0.02
    print(f"Questions: {len(questions)} ({n_open} open→judge) | "
          f"systems: {systems}")
    gen_desc = (f"{args.gen_provider}:{args.gen_model}" if args.gen_model
                else "openai:gpt-4o-mini (default)")
    print(f"Mode: {'REAL' if args.real else 'DRY RUN'} | generator: {gen_desc} | "
          f"est. OpenAI cost ~${est:.2f} (gateway billing, if any, separate)")

    if not dry:
        poisoned = _check_poisoned(paths.cache_dir)
        if poisoned:
            print(f"\nERROR: {len(poisoned)} dry-run entries in the chat "
                  f"cache; they would shadow real calls.\n"
                  f"  Fix: rm -rf {paths.cache_dir / 'chat'}", file=sys.stderr)
            return 3
        if est > 0.5 and not args.yes:
            if input("Proceed? [y/N] ").strip().lower() not in ("y", "yes"):
                print("Aborted."); return 0

    results = []
    for name in systems:
        print(f"\n=== {name} ===")
        cfg = RetrievalConfig()
        if args.force_level:
            cfg.time_force_level = args.force_level
        results.append(run_one_system(name, questions, paths, cfg, dry,
                                       out_prefix, args.buckets_path,
                                       args.gen_model, args.gen_provider,
                                       gen_extra_body, args.gen_max_tokens))

    print("\n" + "=" * 56)
    print(f"{'system':<22}{'closed match':>14}{'open→judge':>14}")
    print("-" * 56)
    for r in results:
        if r.get("skipped"):
            print(f"{r['system']:<22}{'SKIPPED':>14}")
            continue
        mr = f"{r['n_match']}/{r['n_closed']}"
        print(f"{r['system']:<22}{mr:>14}{r['n_judge']:>14}")
    print("=" * 56)
    print("\nNext: temporal-faithfulness on the answers, and (for open "
          "questions) the pairwise judge via run_benchmark.py --judge.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
