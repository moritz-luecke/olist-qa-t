"""
============================================================
Phase 4 — Central configuration
============================================================

All paths, model identifiers, and hyperparameters in one place.
Override via CLI flags in run_benchmark.py if needed.
============================================================
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# Project root = parent of phase4_retrieval/, so default paths work
# regardless of which directory the user runs the scripts from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ----------------------------------------------------------------
# Models (pinned snapshots, as reported in the paper)
# ----------------------------------------------------------------
EMBEDDING_MODEL = "text-embedding-3-small"
GENERATION_MODEL = "gpt-4o-mini-2024-07-18"
JUDGE_MODEL = "gpt-4o-2024-08-06"

# Generation determinism — must match Appendix A.1.1
GEN_TEMPERATURE = 0.0
GEN_TOP_P = 1.0
GEN_SEED = 42
GEN_MAX_TOKENS = 512

# Judge determinism — same fix
JUDGE_TEMPERATURE = 0.0
JUDGE_SEED = 42
JUDGE_MAX_TOKENS = 200


# ----------------------------------------------------------------
# Default paths (override via CLI)
# ----------------------------------------------------------------
@dataclass
class Paths:
    data_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "data")
    kg_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "outputs" / "kg_schema_aware.pkl")
    pilot_questions: Path = field(
        default_factory=lambda: PROJECT_ROOT / "outputs" / "pilot_questions.jsonl")
    cache_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "outputs" / "cache")
    chunks_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "outputs" / "vanilla_chunks.jsonl")
    chunk_embeds_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "outputs" / "vanilla_chunk_embeds.npy")
    node_embeds_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "outputs" / "node_embeds.npy")
    community_embeds_path: Path = field(
        default_factory=lambda: PROJECT_ROOT / "outputs" / "community_embeds.npy")
    community_summaries: Path = field(
        default_factory=lambda: PROJECT_ROOT / "outputs" / "community_summaries.jsonl")
    results_dir: Path = field(
        default_factory=lambda: PROJECT_ROOT / "outputs" / "results")


# ----------------------------------------------------------------
# Retrieval hyperparameters (single source of truth — also in paper)
# ----------------------------------------------------------------
@dataclass
class RetrievalConfig:
    # Vanilla RAG
    chunk_top_k: int = 8

    # Entity strategy
    entity_top_k: int = 5            # seed entities matched to query
    entity_hops: int = 2             # neighborhood radius
    entity_max_nodes: int = 40       # cap on serialized nodes (token budget)

    # Path strategy
    path_top_k_seeds: int = 4
    path_max_length: int = 4         # max simple-path length
    path_max_paths: int = 10

    # Community strategy
    community_top_k: int = 3
    community_gamma: float = 1.0     # which Leiden resolution to use

    # Hybrid: concatenate entity + community evidence
    hybrid_entity_top_k: int = 3
    hybrid_community_top_k: int = 2

    # Generic
    rng_seed: int = 42


# ----------------------------------------------------------------
# Cost guardrails (default ON — set explicitly in benchmark script)
# ----------------------------------------------------------------
@dataclass
class CostGuards:
    dry_run: bool = True             # default: no API calls
    max_questions: int | None = None # cap questions per run
    confirm_cost_above_usd: float = 1.0  # require --yes if estimate > $1
