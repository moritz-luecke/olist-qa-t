"""
============================================================
Phase 4 — Graph-RAG strategy 4: Hybrid  [PARTIAL SKELETON]
============================================================

Per Appendix A.1.3, hybrid = entity-based block ∪ community-based
block, separated by "---". The implementation below is functional
but you may want to tune:

  - Independent top-k per sub-strategy (currently uses
    hybrid_entity_top_k and hybrid_community_top_k from config)
  - Whether to deduplicate node IDs across the two blocks
  - Whether path-based should also be added (currently NOT —
    keeps the evidence block in the same family as A and C)

DESIGN NOTE: the hybrid retriever shares the same underlying
indexes as EntityRetriever and CommunityRetriever — be sure to
share embedder + KG across retrievers in the pipeline, otherwise
you'll embed everything twice.
============================================================
"""
from __future__ import annotations

from dataclasses import replace

from config import Paths, RetrievalConfig
from embeddings import Embedder
from retrievers.base import BaseRetriever, RetrievalResult
from retrievers.community import CommunityRetriever
from retrievers.entity import EntityRetriever


class HybridRetriever(BaseRetriever):
    name = "hybrid"

    def __init__(self,
                 paths: Paths,
                 cfg: RetrievalConfig,
                 embedder: Embedder):
        # Build adjusted configs so the sub-retrievers use the hybrid
        # top-k values, not the full ones.
        ent_cfg = replace(cfg, entity_top_k=cfg.hybrid_entity_top_k)
        com_cfg = replace(cfg, community_top_k=cfg.hybrid_community_top_k)
        self.entity = EntityRetriever(paths, ent_cfg, embedder)
        self.community = CommunityRetriever(paths, com_cfg, embedder)

    def setup(self) -> None:
        self.entity.setup()
        self.community.setup()

    def retrieve(self, question: str) -> RetrievalResult:
        ent = self.entity.retrieve(question)
        com = self.community.retrieve(question)
        merged_block = ent.evidence_block + "\n\n---\n\n" + com.evidence_block
        return RetrievalResult(
            evidence_block=merged_block,
            cited_node_ids=ent.cited_node_ids,
            diagnostics={
                "entity": ent.diagnostics,
                "community": com.diagnostics,
            },
        )
