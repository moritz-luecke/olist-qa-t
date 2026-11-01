"""
============================================================
Phase 4 — Disk cache for API calls
============================================================

Every embedding and chat completion is keyed by a deterministic
hash of its input. Identical inputs across runs hit the cache
and cost zero dollars.

This is the single most important file for cost control in
this project — DO NOT bypass it for "convenience".

Cache layout (on disk):
  {cache_dir}/embeddings/{sha256[:16]}.json
  {cache_dir}/chat/{sha256[:16]}.json

Each file contains:
  {"input": <original input>,
   "model": <model name>,
   "output": <API response payload>,
   "tokens": <token usage>,
   "timestamp": <ISO 8601>}
============================================================
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _hash_key(payload: dict) -> str:
    """SHA-256 of a canonical JSON serialization."""
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


class DiskCache:
    """JSON-on-disk cache, partitioned by subdir (e.g. 'embeddings', 'chat')."""

    def __init__(self, root: Path, subdir: str):
        self.dir = Path(root) / subdir
        self.dir.mkdir(parents=True, exist_ok=True)

    def key(self, payload: dict) -> str:
        return _hash_key(payload)

    def path_for(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get(self, payload: dict) -> dict | None:
        p = self.path_for(self.key(payload))
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # Corrupt entry — drop it
            p.unlink(missing_ok=True)
            return None

    def put(self, payload: dict, output: Any, tokens: dict | None = None) -> None:
        record = {
            "input": payload,
            "output": output,
            "tokens": tokens or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        p = self.path_for(self.key(payload))
        p.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")

    def stats(self) -> dict:
        n = sum(1 for _ in self.dir.glob("*.json"))
        size_mb = sum(f.stat().st_size for f in self.dir.glob("*.json")) / 1e6
        return {"n_entries": n, "size_mb": round(size_mb, 2)}
