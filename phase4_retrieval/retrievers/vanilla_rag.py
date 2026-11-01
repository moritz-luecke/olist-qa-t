"""
============================================================
Phase 4 — System A: Vanilla RAG
============================================================

Standard text-only RAG baseline. Treats each order paragraph
as an opaque chunk; retrieves top-k via cosine similarity on
text-embedding-3-small vectors.

ID anchor lookup
----------------
If the question literally contains a 32-hex order ID, we prepend
the corresponding chunk to the retrieved list (and trim the
cosine-top-k by one to keep total = chunk_top_k). This mirrors
the anchor strategy used by EntityRetriever, so the baseline is
not handicapped on ID-anchored questions.

This is the system Graph-RAG must beat.
============================================================
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Paths, RetrievalConfig
from embeddings import Embedder
from prompts import format_vanilla_paragraphs
from retrievers.base import BaseRetriever, RetrievalResult


_HEX_RE = re.compile(r"\b[a-f0-9]{32}\b")


class VanillaRAG(BaseRetriever):
    name = "vanilla_rag"

    def __init__(self,
                 paths: Paths,
                 cfg: RetrievalConfig,
                 embedder: Embedder):
        self.paths = paths
        self.cfg = cfg
        self.embedder = embedder
        self._chunks: list[dict] | None = None
        self._chunk_id_to_idx: dict[str, int] | None = None
        self._embeds: np.ndarray | None = None
        self._chunk_ids: list[str] | None = None

    # ------------------------------------------------------------
    def setup(self) -> None:
        """Load chunks from JSONL and (re-)build the embedding matrix."""
        if not self.paths.chunks_path.exists():
            raise FileNotFoundError(
                f"{self.paths.chunks_path} not found. "
                f"Run: python chunking.py --data-dir {self.paths.data_dir}"
            )
        with self.paths.chunks_path.open() as fp:
            self._chunks = [json.loads(line) for line in fp]
        self._chunk_ids = [c["chunk_id"] for c in self._chunks]
        self._chunk_id_to_idx = {c["chunk_id"]: i
                                 for i, c in enumerate(self._chunks)}

        # Try to load cached matrix first
        if self.paths.chunk_embeds_path.exists():
            self._embeds = np.load(self.paths.chunk_embeds_path)
            if self._embeds.shape[0] != len(self._chunks):
                print(f"  [WARN] cached embedding count "
                      f"({self._embeds.shape[0]}) != #chunks "
                      f"({len(self._chunks)}); rebuilding")
                self._embeds = None
        if self._embeds is None:
            texts = [c["text"] for c in self._chunks]
            self._embeds = self.embedder.embed_many(texts, desc="chunk embeds")
            # L2-normalize so cosine = dot product
            norms = np.linalg.norm(self._embeds, axis=1, keepdims=True) + 1e-9
            self._embeds = self._embeds / norms
            self.paths.chunk_embeds_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(self.paths.chunk_embeds_path, self._embeds)

    # ------------------------------------------------------------
    def _find_anchors(self, question: str) -> list[int]:
        """Return chunk indices whose order_id is literally in the question."""
        hex_tokens = _HEX_RE.findall(question)
        anchor_idxs: list[int] = []
        for tok in hex_tokens:
            chunk_id = f"O:{tok}"
            idx = self._chunk_id_to_idx.get(chunk_id)
            if idx is not None:
                anchor_idxs.append(idx)
        return anchor_idxs

    # ------------------------------------------------------------
    def retrieve(self, question: str) -> RetrievalResult:
        assert self._embeds is not None, "Call setup() first"

        anchor_idxs = self._find_anchors(question)
        anchor_set = set(anchor_idxs)

        q_vec = self.embedder.embed_one(question)
        q_vec /= np.linalg.norm(q_vec) + 1e-9

        sims = self._embeds @ q_vec
        k = self.cfg.chunk_top_k

        # Fill the rest of the top-k with cosine-best chunks NOT in anchors
        slots_left = max(k - len(anchor_idxs), 0)
        # argsort once, then walk down skipping anchor positions
        ranking = np.argsort(-sims)
        emb_picks: list[int] = []
        for idx in ranking:
            if int(idx) in anchor_set:
                continue
            emb_picks.append(int(idx))
            if len(emb_picks) >= slots_left:
                break

        final_idxs = anchor_idxs + emb_picks
        paragraphs = [self._chunks[i]["text"] for i in final_idxs]
        cited = [self._chunks[i]["chunk_id"] for i in final_idxs]

        evidence = format_vanilla_paragraphs(paragraphs)
        return RetrievalResult(
            evidence_block=evidence,
            cited_chunk_ids=cited,
            diagnostics={
                "n_anchors": len(anchor_idxs),
                "anchor_chunk_ids": [self._chunks[i]["chunk_id"] for i in anchor_idxs],
                "top_similarities": [float(sims[i]) for i in final_idxs],
            },
        )
