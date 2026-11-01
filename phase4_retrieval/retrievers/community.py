"""
============================================================
Phase 4 — Graph-RAG strategy 3: Community-based
============================================================

Microsoft GraphRAG "global" strategy. Pre-computed community
summaries (Phase 2.5) act as a top-down index over the graph.

Algorithm:
  1. Load community_summaries.jsonl (one record per community).
  2. Embed each summary's text once and cache.
  3. For a question: embed → top-k community summaries by cosine.
  4. Serialize as a COMMUNITIES block (NODES + TRIPLES omitted
     per A.1.3 spec).

This strategy is expected to dominate on "global" category
questions and underperform on factual look-ups.
============================================================
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Paths, RetrievalConfig
from embeddings import Embedder
from retrievers.base import BaseRetriever, RetrievalResult


class CommunityRetriever(BaseRetriever):
    name = "community"

    def __init__(self,
                 paths: Paths,
                 cfg: RetrievalConfig,
                 embedder: Embedder):
        self.paths = paths
        self.cfg = cfg
        self.embedder = embedder
        self._records: list[dict] | None = None
        self._embeds: np.ndarray | None = None

    # ------------------------------------------------------------
    def setup(self) -> None:
        if not self.paths.community_summaries.exists():
            raise FileNotFoundError(
                f"{self.paths.community_summaries} not found. "
                f"Run: python summarize_communities.py --gamma "
                f"{self.cfg.community_gamma}"
            )

        records = []
        with self.paths.community_summaries.open() as fp:
            for line in fp:
                r = json.loads(line)
                if abs(r.get("gamma", -1) - self.cfg.community_gamma) < 1e-6:
                    records.append(r)
        if not records:
            raise RuntimeError(
                f"No community summaries for γ={self.cfg.community_gamma} "
                f"in {self.paths.community_summaries}"
            )
        self._records = records

        if self.paths.community_embeds_path.exists():
            self._embeds = np.load(self.paths.community_embeds_path)
            if self._embeds.shape[0] != len(records):
                self._embeds = None
        if self._embeds is None:
            texts = [r["summary"] for r in records]
            self._embeds = self.embedder.embed_many(texts)
            norms = np.linalg.norm(self._embeds, axis=1, keepdims=True) + 1e-9
            self._embeds = self._embeds / norms
            self.paths.community_embeds_path.parent.mkdir(parents=True,
                                                          exist_ok=True)
            np.save(self.paths.community_embeds_path, self._embeds)

    # ------------------------------------------------------------
    def retrieve(self, question: str) -> RetrievalResult:
        assert self._embeds is not None, "Call setup() first"
        q_vec = self.embedder.embed_one(question)
        q_vec /= np.linalg.norm(q_vec) + 1e-9

        sims = self._embeds @ q_vec
        k = min(self.cfg.community_top_k, len(self._records))
        top_idx = np.argpartition(-sims, k)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        # Format: COMMUNITIES block; A.1.3 says NODES is REPLACED by
        # this and TRIPLES is omitted.
        lines = []
        for rank, i in enumerate(top_idx, start=1):
            r = self._records[i]
            lines.append(
                f"[C{rank}] community_id={r['community_id']} "
                f"(size={r['size']}): {r['summary']}"
            )
        evidence = "COMMUNITIES:\n" + "\n".join(lines)

        return RetrievalResult(
            evidence_block=evidence,
            cited_node_ids=[],  # this strategy cites communities, not nodes
            diagnostics={
                "community_ids": [int(self._records[i]["community_id"])
                                  for i in top_idx],
                "top_similarities": [float(sims[i]) for i in top_idx],
            },
        )
