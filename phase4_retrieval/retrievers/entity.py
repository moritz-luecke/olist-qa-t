"""
============================================================
Phase 4 — Graph-RAG strategy 1: Entity-based
============================================================

Algorithm:
  1. ID-direct lookup: regex-extract 32-hex tokens from the
     question, check which exist as nodes in the KG (under any
     of the O:/C:/P:/S: prefixes), and use those as ANCHOR seeds.
  2. Embed every node's deterministic description (Phase 2.3).
     Take top-k by cosine similarity as EMBEDDING seeds, but skip
     adding ones that are already anchors.
  3. Combine anchor + embedding seeds. Anchors are protected from
     pruning (we never drop them in step 5).
  4. Expand to the h-hop neighborhood of all seeds (undirected).
  5. Cap total nodes at entity_max_nodes. Anchors are always kept;
     the rest is dropped by type priority (Geolocation first, then
     Payment, etc.).
  6. Serialize as the NODES + TRIPLES block from Appendix A.1.3.

Rationale for ID-direct lookup
------------------------------
text-embedding-3-small treats 32-hex order IDs as opaque BPE
fragments, so embedding-based matching fails on factual lookups
that key off an ID literal. ID-direct lookup is standard in
production graph-RAG (Microsoft "local search anchor", Neo4j
"entity linking"). We apply the same idea symmetrically in
vanilla_rag.py so the baseline gets the same boost.
============================================================
"""
from __future__ import annotations

import pickle
import re
import sys
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Paths, RetrievalConfig
from embeddings import Embedder
from retrievers.base import BaseRetriever, RetrievalResult


# Preference order when pruning — Geolocation gets dropped first.
_DROP_ORDER = ("Geolocation", "Payment", "Review",
               "OrderItem", "Product", "Customer", "Seller", "Order")

# 32-char lowercase hex token. Matches Olist order/customer/product/seller IDs.
_HEX_RE = re.compile(r"\b[a-f0-9]{32}\b")

# All node-ID prefixes we might find via a 32-hex match.
# OrderItem (I:) is excluded because its node_id has a ":<n>" suffix.
_ANCHOR_PREFIXES = ("O:", "C:", "P:", "S:", "R:")


class EntityRetriever(BaseRetriever):
    name = "entity"

    def __init__(self,
                 paths: Paths,
                 cfg: RetrievalConfig,
                 embedder: Embedder):
        self.paths = paths
        self.cfg = cfg
        self.embedder = embedder
        self._G: nx.MultiDiGraph | None = None
        self._node_ids: list[str] | None = None
        self._embeds: np.ndarray | None = None

    # ------------------------------------------------------------
    def setup(self) -> None:
        if self._G is None:
            with self.paths.kg_path.open("rb") as f:
                self._G = pickle.load(f)
        nodes_with_desc = [
            (n, d["description"])
            for n, d in self._G.nodes(data=True)
            if "description" in d
        ]
        if not nodes_with_desc:
            raise RuntimeError(
                "No node descriptions found in KG. "
                "Run add_node_descriptions.py first."
            )
        self._node_ids = [n for n, _ in nodes_with_desc]
        descs = [d for _, d in nodes_with_desc]

        if self.paths.node_embeds_path.exists():
            self._embeds = np.load(self.paths.node_embeds_path)
            if self._embeds.shape[0] != len(descs):
                self._embeds = None
        if self._embeds is None:
            self._embeds = self.embedder.embed_many(descs, desc="node embeds")
            norms = np.linalg.norm(self._embeds, axis=1, keepdims=True) + 1e-9
            self._embeds = self._embeds / norms
            self.paths.node_embeds_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(self.paths.node_embeds_path, self._embeds)

    # ------------------------------------------------------------
    def _find_anchors(self, question: str) -> list[str]:
        """Regex 32-hex tokens from the question, return node_ids that
        actually exist in the KG (under any anchor prefix)."""
        hex_tokens = set(_HEX_RE.findall(question))
        anchors: list[str] = []
        for tok in hex_tokens:
            for pref in _ANCHOR_PREFIXES:
                node_id = f"{pref}{tok}"
                if node_id in self._G:
                    anchors.append(node_id)
                    # Don't break — same hex could be both customer & order
                    # in theory, though Olist IDs are unique per entity type.
        return anchors

    def _embedding_seeds(self, question: str, k: int,
                         exclude: set[str]) -> list[str]:
        q_vec = self.embedder.embed_one(question)
        q_vec /= np.linalg.norm(q_vec) + 1e-9
        sims = self._embeds @ q_vec
        # Top-k that aren't already anchors
        ranking = np.argsort(-sims)
        out: list[str] = []
        for idx in ranking:
            nid = self._node_ids[idx]
            if nid in exclude:
                continue
            out.append(nid)
            if len(out) >= k:
                break
        return out

    # ------------------------------------------------------------
    def _khop(self, seeds: list[str], h: int) -> set[str]:
        visited = set(seeds)
        frontier = set(seeds)
        for _ in range(h):
            nxt = set()
            for u in frontier:
                nxt.update(self._G.successors(u))
                nxt.update(self._G.predecessors(u))
            nxt -= visited
            visited.update(nxt)
            frontier = nxt
            if not frontier:
                break
        return visited

    def _prune(self, nodes: set[str], budget: int,
               protected: set[str]) -> set[str]:
        """Drop low-priority types until under budget. Protected nodes
        (anchors + their direct neighbors) are never dropped."""
        if len(nodes) <= budget:
            return nodes
        nodes = set(nodes)
        # First: drop nodes by type priority, skipping protected
        for nt in _DROP_ORDER:
            droppable = [
                n for n in nodes
                if (self._G.nodes[n].get("node_type") == nt
                    and n not in protected)
            ]
            for n in droppable:
                if len(nodes) <= budget:
                    return nodes
                nodes.discard(n)
        # If still over budget (only protected left), keep all of them
        return nodes

    def _serialize(self, nodes: set[str], anchors: set[str]) -> str:
        """A.1.3 NODES + TRIPLES block, with anchors marked."""
        node_lines = []
        for n in sorted(nodes):
            d = self._G.nodes[n]
            nt = d.get("node_type", "?")
            desc = d.get("description", "(no description)")
            marker = " [ANCHOR]" if n in anchors else ""
            node_lines.append(f"- {n} (type={nt}){marker}: {desc}")

        nset = set(nodes)
        triples = []
        for u, v, d in self._G.edges(data=True):
            if u in nset and v in nset:
                rel = d.get("edge_type", "?")
                triples.append(f"- {u} --{rel}--> {v}")
        triples = sorted(set(triples))

        return (
            "NODES:\n" + "\n".join(node_lines)
            + "\n\nTRIPLES:\n" + "\n".join(triples)
        )

    # ------------------------------------------------------------
    def retrieve(self, question: str) -> RetrievalResult:
        assert self._embeds is not None, "Call setup() first"

        # 1. ID-direct anchors (highest priority)
        anchors = self._find_anchors(question)

        # 2. Embedding seeds (skip ones that are already anchors)
        emb_k = self.cfg.entity_top_k
        emb_seeds = self._embedding_seeds(question, emb_k, exclude=set(anchors))

        seeds = anchors + emb_seeds

        # 3. Expand (anchors get the same h-hop as everything else)
        expanded = self._khop(seeds, self.cfg.entity_hops)

        # Protected = anchors + their direct 1-hop neighborhood (guarantees
        # the Order's Payment/Items/Review aren't pruned away).
        protected = set(anchors)
        for a in anchors:
            protected.update(self._G.successors(a))
            protected.update(self._G.predecessors(a))

        pruned = self._prune(expanded, self.cfg.entity_max_nodes,
                             protected=protected)
        evidence = self._serialize(pruned, anchors=set(anchors))

        return RetrievalResult(
            evidence_block=evidence,
            cited_node_ids=list(pruned),
            diagnostics={
                "anchors": anchors,
                "embedding_seeds": emb_seeds,
                "n_after_expand": len(expanded),
                "n_after_prune": len(pruned),
                "n_protected": len(protected),
            },
        )
