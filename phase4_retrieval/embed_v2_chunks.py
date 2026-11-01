"""
============================================================
Phase 4 — Re-embed Vanilla v2 chunks (one-shot script)
============================================================

Reads the v2 chunks JSONL produced by `chunking.py --v2`,
embeds each paragraph with text-embedding-3-small via the
existing Embedder, L2-normalises the matrix (so cosine = dot),
and writes the matrix to outputs/vanilla_chunk_embeds_v2.npy.

Why a dedicated script (not just lazy-rebuild inside VanillaRAG)?
- Makes the cost visible up-front: prints a token estimate and
  pauses for confirmation unless --yes is given.
- Gives a stable, inspectable artefact you can ship to a
  collaborator without anyone having to run the retriever.
- Lets you re-embed without firing up the retriever stack.

Cost
----
At text-embedding-3-small pricing ($0.02 / 1M input tokens),
99,441 chunks × ~100 tokens ≈ 10M tokens → $0.20. The prior
paper's v1 re-embedding cost was the same order; the only
difference here is the extra timeline sentence (+25-30 tokens
per chunk → ~$0.05 incremental).

Usage:
  # Dry run (default — fake vectors, no API call, costs nothing):
  python embed_v2_chunks.py

  # Real embedding via OpenAI API (~$0.20):
  python embed_v2_chunks.py --no-dry-run

  # Skip the confirmation prompt (CI / scripted runs):
  python embed_v2_chunks.py --no-dry-run --yes
============================================================
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Make the sibling modules importable regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import Paths              # noqa: E402
from embeddings import Embedder       # noqa: E402


# Cost reference (May 2026 published pricing for
# text-embedding-3-small): $0.02 per 1M input tokens.
COST_PER_M_TOKENS_USD = 0.02


def load_chunks(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. "
            f"Run: python chunking.py --v2 first."
        )
    with path.open() as fp:
        return [json.loads(line) for line in fp]


def estimate_cost(chunks: list[dict]) -> tuple[int, float]:
    """Estimate total tokens and USD cost from the JSONL token field."""
    total_tokens = sum(c.get("n_tokens_approx", 0) for c in chunks)
    cost_usd = total_tokens * COST_PER_M_TOKENS_USD / 1_000_000
    return total_tokens, cost_usd


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Re-embed Vanilla v2 chunks into a fresh "
                    "L2-normalised NPY matrix.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--chunks-path", type=Path,
        default=Path("./outputs/vanilla_chunks_v2.jsonl"),
        help="JSONL file produced by `chunking.py --v2`.",
    )
    ap.add_argument(
        "--out-path", type=Path,
        default=Path("./outputs/vanilla_chunk_embeds_v2.npy"),
        help="Destination NPY for the L2-normalised embedding matrix.",
    )
    ap.add_argument(
        "--cache-dir", type=Path, default=None,
        help="Embedder cache directory (default: outputs/cache).",
    )
    # Dry-run flag — matches the convention used by run_benchmark.py
    # and extract_triples.py: default is dry-run (no API call, fake
    # deterministic vectors). Pass --no-dry-run to actually hit the
    # OpenAI API and produce a usable embedding matrix.
    ap.add_argument(
        "--dry-run", dest="dry_run", action="store_true",
        help="Use fake deterministic vectors (no API call). Default.",
    )
    ap.add_argument(
        "--no-dry-run", dest="dry_run", action="store_false",
        help="REAL embedding via OpenAI API (costs money).",
    )
    ap.set_defaults(dry_run=True)
    ap.add_argument(
        "--yes", action="store_true",
        help="Skip the cost-confirmation prompt.",
    )
    args = ap.parse_args()

    paths = Paths()  # only consulted for the default cache directory
    cache_dir = args.cache_dir or paths.cache_dir

    # ---- Load chunks ----------------------------------------------
    print(f"Loading chunks from {args.chunks_path} ...")
    chunks = load_chunks(args.chunks_path)
    n_chunks = len(chunks)
    print(f"  loaded {n_chunks:,} chunks")

    # Sanity-check the variant marker — refusing to embed a v1 file
    # by accident saves an entire re-embedding cost.
    variants = {c.get("chunk_variant") for c in chunks}
    if variants != {"v2"}:
        print(f"  [WARN] expected chunk_variant=='v2' for every record; "
              f"observed variants: {variants}")
        if "v1" in variants:
            print("        Refusing to embed v1 chunks under the v2 name. "
                  "Use the v1 pipeline path or fix --chunks-path.",
                  file=sys.stderr)
            return 2

    # ---- Cost estimate --------------------------------------------
    total_tokens, cost_usd = estimate_cost(chunks)
    mode = "DRY RUN (no API call)" if args.dry_run else "REAL (OpenAI API)"
    print(f"  ≈ {total_tokens:,} tokens "
          f"(~${cost_usd:.2f} at "
          f"${COST_PER_M_TOKENS_USD:.2f}/1M tokens)")
    print(f"  mode: {mode}")
    if args.dry_run:
        print("  [INFO] Output will contain DETERMINISTIC FAKE VECTORS, "
              "useless for retrieval. Pass --no-dry-run for real embeddings.")

    if args.out_path.exists():
        print(f"  [INFO] {args.out_path} already exists and will be "
              f"OVERWRITTEN")

    if not args.yes:
        ans = input("Proceed? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            print("Aborted.")
            return 0

    # ---- Embed ----------------------------------------------------
    print(f"Embedding via {Embedder.__module__} "
          f"(cache: {cache_dir}, dry_run={args.dry_run}) ...")
    embedder = Embedder(cache_root=cache_dir, dry_run=args.dry_run)
    texts = [c["text"] for c in chunks]
    matrix = embedder.embed_many(texts, desc="v2 chunk embeds")

    # ---- L2-normalise so cosine == dot ----------------------------
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
    matrix = matrix / norms

    # ---- Save -----------------------------------------------------
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_path, matrix)
    print(f"Wrote {args.out_path}  shape={matrix.shape}")
    print(f"Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
