"""
============================================================
Phase 4 — Graph-RAG strategy 2: Path-based  [SKELETON]
============================================================

The idea: for multi-hop questions, paths between query entities
carry more signal than the entity-neighborhood blob. Algorithm:

  1. Extract or embed-match the top-k seed entities from the
     question (same as EntityRetriever).
  2. For every unordered pair of seeds, enumerate simple paths
     of length ≤ path_max_length in the undirected graph.
  3. Cap total paths at path_max_paths (sort by length asc,
     then by some salience metric — TODO).
  4. Serialize: NODES = union of all path nodes; TRIPLES grouped
     per path with separator lines (per A.1.3).

DESIGN DECISIONS LEFT TO YOU:
  - Seed extraction: same embedding match as EntityRetriever, or
    a separate LLM-based NER call? The paper currently locks in
    embedding match for both, but the option is open.
  - Path ranking when there are too many: by length only, or
    weighted by edge-type "importance" (e.g. placed > located_at)?
  - Should single-seed questions fall back to entity behavior?
    Probably yes — empty path set is useless.
============================================================
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import Paths, RetrievalConfig
from embeddings import Embedder
from retrievers.base import BaseRetriever, RetrievalResult


class PathRetriever(BaseRetriever):
    name = "path"

    def __init__(self,
                 paths: Paths,
                 cfg: RetrievalConfig,
                 embedder: Embedder):
        self.paths = paths
        self.cfg = cfg
        self.embedder = embedder
        self._G: nx.MultiDiGraph | None = None
        self._G_und: nx.Graph | None = None       # cached undirected view
        self._node_ids: list[str] | None = None
        self._embeds: np.ndarray | None = None

    # ------------------------------------------------------------
    def setup(self) -> None:
        if self._G is None:
            with self.paths.kg_path.open("rb") as f:
                self._G = pickle.load(f)
            self._G_und = nx.Graph(self._G)        # drops parallel + direction

        # Reuse the same node-description embeddings the entity retriever
        # builds. If the file exists, just load it; otherwise instruct user
        # to run EntityRetriever.setup() first.
        if not self.paths.node_embeds_path.exists():
            raise RuntimeError(
                "node_embeds.npy missing. Run EntityRetriever.setup() first "
                "(or call this retriever via the shared pipeline so the "
                "matrix is built once)."
            )
        self._embeds = np.load(self.paths.node_embeds_path)
        self._node_ids = [n for n, d in self._G.nodes(data=True)
                          if "description" in d]
        assert len(self._node_ids) == self._embeds.shape[0]

    # ------------------------------------------------------------
    def _find_seeds(self, question: str) -> list[str]:
        q_vec = self.embedder.embed_one(question)
        q_vec /= np.linalg.norm(q_vec) + 1e-9
        sims = self._embeds @ q_vec
        k = self.cfg.path_top_k_seeds
        top_idx = np.argpartition(-sims, k)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        return [self._node_ids[i] for i in top_idx]

    def _enumerate_paths(self, seeds: list[str]) -> list[list[str]]:
        """
        TODO: All pairs (i,j), find simple paths of length ≤ L.
        Pseudocode:
            paths = []
            for i, src in enumerate(seeds):
                for dst in seeds[i+1:]:
                    for path in nx.all_simple_paths(
                            self._G_und, src, dst,
                            cutoff=self.cfg.path_max_length):
                        paths.append(path)
                        if len(paths) >= self.cfg.path_max_paths * 4:
                            break  # local cap before global ranking
            # Rank: shorter first, then by salience (TODO)
            paths.sort(key=len)
            return paths[:self.cfg.path_max_paths]
        Warning: all_simple_paths can be VERY expensive on a graph this
        size. Consider:
          - using nx.shortest_simple_paths and taking the first L
          - or bidirectional BFS with a hop cap
        """
        raise NotImplementedError(
            "Path enumeration not yet implemented. See docstring."
        )

    def _serialize(self, seeds: list[str],
                   paths: list[list[str]]) -> str:
        """A.1.3 format: TRIPLES grouped per path with separator."""
        all_nodes = set(seeds)
        for p in paths:
            all_nodes.update(p)
        node_lines = []
        for n in sorted(all_nodes):
            d = self._G.nodes[n]
            node_lines.append(
                f"- {n} (type={d.get('node_type','?')}): "
                f"{d.get('description','(no description)')}"
            )

        triple_blocks = []
        for idx, path in enumerate(paths, start=1):
            block = [f"# path {idx} (length {len(path) - 1})"]
            for u, v in zip(path[:-1], path[1:]):
                # Pick one of the parallel edges — use first available
                if self._G.has_edge(u, v):
                    rel = next(iter(self._G[u][v].values()))["edge_type"]
                    block.append(f"- {u} --{rel}--> {v}")
                elif self._G.has_edge(v, u):
                    rel = next(iter(self._G[v][u].values()))["edge_type"]
                    block.append(f"- {v} --{rel}--> {u}")
            triple_blocks.append("\n".join(block))

        return (
            "NODES:\n" + "\n".join(node_lines)
            + "\n\nTRIPLES:\n" + "\n\n".join(triple_blocks)
        )

    # ------------------------------------------------------------
    def retrieve(self, question: str) -> RetrievalResult:
        assert self._embeds is not None, "Call setup() first"
        seeds = self._find_seeds(question)
        paths = self._enumerate_paths(seeds)
        evidence = self._serialize(seeds, paths)

        return RetrievalResult(
            evidence_block=evidence,
            cited_node_ids=list({n for p in paths for n in p} | set(seeds)),
            diagnostics={
                "seeds": seeds,
                "n_paths": len(paths),
                "path_lengths": [len(p) - 1 for p in paths],
            },
        )
