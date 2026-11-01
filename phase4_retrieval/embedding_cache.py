"""
============================================================
Phase 4 — Efficient embedding cache (NPY + JSONL index)
============================================================

Replaces the per-vector-JSON DiskCache for the embedding path.

Storage layout:
  outputs/cache/embeddings_v2/vectors.npy   (memory-mapped float32 matrix)
  outputs/cache/embeddings_v2/index.jsonl   (one line per cached vector)

Each index line:
  {"key": "<sha256[:16]>", "row": <int>,
   "tokens": <int>, "model": "<name>"}

The key is sha256(model + "\x00" + text)[:16] — same hashing convention
as the old DiskCache so users can migrate seamlessly.

Design properties:
  - O(1) lookup via in-memory dict (key → row).
  - Reads via numpy memory-map (no allocation per hit).
  - Writes batched: a single np.save() at the end of a batch rewrites
    vectors.npy with all new rows appended, then index.jsonl is
    appended line-by-line. Crash mid-write leaves the previous npy
    intact; partial index lines on the final flush are tolerated by
    truncating to known-good count on next load.
  - Thread-safe is NOT a goal — single writer expected.

This file is intentionally framework-light: no openai dependency,
no config import, so the migration script and tests can use it
in isolation.
============================================================
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np


EMBED_DIM = 1536           # text-embedding-3-small
_GROWTH_FACTOR = 2          # doubling strategy on capacity exhaustion
_INIT_CAPACITY = 1024       # initial #rows allocated on disk


def key_for(text: str, model: str) -> str:
    """Stable 16-hex-char key for (text, model)."""
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return h.hexdigest()[:16]


class EmbedCache:
    """One npy + one jsonl. Append-only, O(1) lookup."""

    def __init__(self, root: Path, dim: int = EMBED_DIM):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.vec_path = self.root / "vectors.npy"
        self.idx_path = self.root / "index.jsonl"
        self.dim = dim

        # In-memory key → row table
        self._key2row: dict[str, int] = {}
        self._n: int = 0
        self._capacity: int = 0
        self._vectors: np.ndarray | None = None

        self._load_or_init()

    # ------------------------------------------------------------
    # Load / initialize
    # ------------------------------------------------------------
    def _load_or_init(self) -> None:
        """Load both files if present, else initialize empty."""
        if self.vec_path.exists() and self.idx_path.exists():
            # Read index (count records; index may be longer than npy
            # only if a crash happened mid-rewrite — truncate then)
            records = []
            with self.idx_path.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        # Partial last line — drop it
                        break
            self._vectors = np.load(self.vec_path, mmap_mode="r+")
            self._capacity = self._vectors.shape[0]
            # Truncate index records to fit npy rows
            records = records[: self._capacity]
            self._key2row = {r["key"]: r["row"] for r in records}
            self._n = len(records)
            if self._vectors.shape[1] != self.dim:
                raise RuntimeError(
                    f"Cache dim mismatch: cache={self._vectors.shape[1]}, "
                    f"expected={self.dim}")
        else:
            # Fresh cache
            self._capacity = _INIT_CAPACITY
            self._vectors = np.zeros((self._capacity, self.dim),
                                     dtype=np.float32)
            np.save(self.vec_path, self._vectors)
            # Re-open memmap so writes are persisted
            self._vectors = np.load(self.vec_path, mmap_mode="r+")
            self._n = 0
            self.idx_path.write_text("")

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------
    def __contains__(self, key: str) -> bool:
        return key in self._key2row

    def get(self, key: str) -> np.ndarray | None:
        """Return a copy of the cached vector or None."""
        row = self._key2row.get(key)
        if row is None:
            return None
        return np.asarray(self._vectors[row], dtype=np.float32).copy()

    def put_batch(self, keys: list[str], vectors: np.ndarray,
                  meta: list[dict] | None = None) -> None:
        """
        Append a batch of new vectors. Vectors must be shape (B, dim) float32.
        meta is an optional list of metadata dicts (model, tokens, ...).
        Existing keys are silently skipped.
        """
        if len(keys) != vectors.shape[0]:
            raise ValueError("keys and vectors length mismatch")
        if vectors.shape[1] != self.dim:
            raise ValueError(f"vector dim {vectors.shape[1]} != {self.dim}")

        # Filter already-cached keys
        new_keys: list[str] = []
        new_vecs: list[np.ndarray] = []
        new_meta: list[dict] = []
        for i, k in enumerate(keys):
            if k in self._key2row:
                continue
            new_keys.append(k)
            new_vecs.append(vectors[i])
            new_meta.append((meta[i] if meta else {}))
        if not new_keys:
            return

        # Grow capacity if needed
        need_cap = self._n + len(new_keys)
        if need_cap > self._capacity:
            self._grow(need_cap)

        # Write vectors into the memmap
        start = self._n
        for i, v in enumerate(new_vecs):
            self._vectors[start + i] = v
        self._vectors.flush()

        # Append index lines
        with self.idx_path.open("a") as f:
            for i, k in enumerate(new_keys):
                rec = {"key": k, "row": start + i,
                       "tokens": new_meta[i].get("tokens", 0),
                       "model": new_meta[i].get("model", "")}
                f.write(json.dumps(rec) + "\n")
                self._key2row[k] = start + i
        self._n += len(new_keys)

    # ------------------------------------------------------------
    def _grow(self, min_capacity: int) -> None:
        """Resize the npy file to at least min_capacity rows."""
        new_cap = max(self._capacity * _GROWTH_FACTOR, min_capacity)
        # Allocate a fresh array of the new size, copy old rows
        new_arr = np.zeros((new_cap, self.dim), dtype=np.float32)
        new_arr[: self._n] = np.asarray(self._vectors[: self._n])
        # Write atomically: temp file then rename
        tmp = tempfile.NamedTemporaryFile(
            mode="wb", suffix=".npy",
            dir=str(self.root), delete=False
        )
        tmp.close()
        np.save(tmp.name, new_arr)
        os.replace(tmp.name, self.vec_path)
        # Re-open as memmap
        self._vectors = np.load(self.vec_path, mmap_mode="r+")
        self._capacity = new_cap

    # ------------------------------------------------------------
    def stats(self) -> dict:
        size_mb = (
            (self.vec_path.stat().st_size if self.vec_path.exists() else 0)
            + (self.idx_path.stat().st_size if self.idx_path.exists() else 0)
        ) / 1e6
        return {"n_entries": self._n,
                "capacity": self._capacity,
                "size_mb": round(size_mb, 2)}
