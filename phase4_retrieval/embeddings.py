"""
============================================================
Phase 4 — Embedding wrapper (v2: NPY cache, parallel, tqdm)
============================================================

Wraps OpenAI's text-embedding-3-small with:
  - EmbedCache (single NPY + JSONL index — ~50x faster than v1)
  - Concurrent batches via ThreadPoolExecutor (default 4 workers)
  - tqdm progress bar over batches
  - Dry-run mode returns deterministic pseudo-vectors

Cost reference (May 2026 pricing):
  text-embedding-3-small  $0.02 per 1M input tokens
  1536-d float32 output

Tier-1 OpenAI rate limit: 500k tokens/min embedding, 3k req/min.
With batch_size=128 and 4 parallel workers we stay comfortably under.
============================================================
"""
from __future__ import annotations

import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **_):
        return it

from embedding_cache import EmbedCache, key_for, EMBED_DIM
from config import EMBEDDING_MODEL


# ----------------------------------------------------------------
# Lazy OpenAI client
# ----------------------------------------------------------------
_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI()
    return _client


# ----------------------------------------------------------------
# Embedder
# ----------------------------------------------------------------
class Embedder:
    def __init__(
        self,
        cache_root: Path,
        model: str = EMBEDDING_MODEL,
        dry_run: bool = True,
        batch_size: int = 128,
        max_workers: int = 4,
    ):
        self.cache = EmbedCache(Path(cache_root) / "embeddings_v2")
        self.model = model
        self.dry_run = dry_run
        self.batch_size = batch_size
        self.max_workers = max_workers

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str], desc: str = "embedding") -> np.ndarray:
        n = len(texts)
        out: np.ndarray = np.zeros((n, EMBED_DIM), dtype=np.float32)
        keys = [key_for(t, self.model) for t in texts]

        # Cache pass
        miss_idxs: list[int] = []
        for i, k in enumerate(keys):
            v = self.cache.get(k)
            if v is not None:
                out[i] = v
            else:
                miss_idxs.append(i)
        if not miss_idxs:
            return out

        miss_batches = [
            miss_idxs[s : s + self.batch_size]
            for s in range(0, len(miss_idxs), self.batch_size)
        ]

        if self.dry_run:
            for batch in tqdm(miss_batches, desc=f"{desc} (dry)", unit="batch"):
                batch_keys, batch_vecs = [], []
                for idx in batch:
                    seed = int(hashlib.sha256(texts[idx].encode()).hexdigest()[:8], 16)
                    rng = np.random.default_rng(seed)
                    v = rng.standard_normal(EMBED_DIM).astype(np.float32)
                    v /= np.linalg.norm(v) + 1e-9
                    out[idx] = v
                    batch_keys.append(keys[idx])
                    batch_vecs.append(v)
                self.cache.put_batch(
                    batch_keys,
                    np.stack(batch_vecs),
                    meta=[{"model": self.model, "tokens": 0}] * len(batch_keys),
                )
            return out

        # Real API path — parallel batches
        client = _get_client()

        def _call(batch_idx_list):
            inputs = [texts[i] for i in batch_idx_list]
            resp = client.embeddings.create(model=self.model, input=inputs)
            vecs = np.asarray(
                [item.embedding for item in resp.data], dtype=np.float32
            )
            return batch_idx_list, vecs, resp.usage.prompt_tokens

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {ex.submit(_call, b): b for b in miss_batches}
            pbar = tqdm(total=len(miss_batches), desc=desc, unit="batch")
            for fut in as_completed(futures):
                try:
                    batch_idx_list, vecs, tokens = fut.result()
                except Exception as e:
                    pbar.close()
                    raise RuntimeError(f"Embedding API call failed: {e}") from e
                batch_keys = []
                for j, idx in enumerate(batch_idx_list):
                    out[idx] = vecs[j]
                    batch_keys.append(keys[idx])
                self.cache.put_batch(
                    batch_keys,
                    vecs,
                    meta=[
                        {"model": self.model,
                         "tokens": tokens // max(1, len(batch_idx_list))}
                    ] * len(batch_keys),
                )
                pbar.update(1)
            pbar.close()

        return out
