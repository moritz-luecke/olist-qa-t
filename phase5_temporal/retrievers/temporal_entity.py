"""
============================================================
Phase 5.2 — System C: Temporal Entity Graph-RAG
============================================================

Extends the time-blind EntityRetriever (System B1) with a
temporal seeding step backed by the IntervalTree index.

Algorithm (delta over EntityRetriever):
  0. Extract a time constraint from the question (LLM main path
     + rule fallback; see time_extract.py).
  1. ID-direct anchors            — inherited, unchanged.
  2. Embedding seeds              — inherited, unchanged.
  3. NEW: temporal seeds          — for each (edge_type, window) in
     the extracted constraint, query the IntervalTree for orders
     whose lifecycle interval overlaps the window. These order_ids
     become additional seeds (capped at temporal_seed_cap to avoid
     flooding the subgraph).
  4. k-hop expansion              — inherited (now also expands the
     temporal seeds).
  5. Prune to budget              — inherited; anchors + temporal
     seeds are protected from pruning.
  6. Serialize NODES + TRIPLES    — inherited.

Graceful degradation: if the extractor finds no time reference,
the temporal-seed set is empty and the retriever behaves exactly
like EntityRetriever — so System C never does *worse* than B1 on
time-blind questions.

Important design choice — temporal seed capping
-----------------------------------------------
A quarter window can intersect ~20,000 order intervals. We cannot
seed all of them (the subgraph would be the entire KG). We cap at
cfg.temporal_seed_cap (default 40) by taking a deterministic
sample: the orders sorted by interval start, evenly strided. This
gives the LLM a *representative* slice across the window rather
than e.g. only the earliest 40 orders. This is the key structural
difference from vanilla top-k: vanilla retrieves by cosine (so all
8 are near-duplicates of the query); System C retrieves a
time-stratified sample.

NB: this does NOT solve aggregation by itself — 40 orders is still
a sample of ~20k. The aggregation answer comes from System D
(community-temporal: pre-computed bucket summaries). System C is
the per-order temporal retriever; it should excel at
single-order-anchored latency questions and provide a stratified
sample for trend/drift, but it is not expected to nail population
aggregates. We measure exactly this on the pilot.
============================================================
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# phase4_retrieval must come FIRST on the path so that
# `retrievers.entity` resolves to the prior repo's package, not the
# sibling phase5_temporal/retrievers/ this file lives in.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent /
                       "phase4_retrieval"))

from retrievers.entity import EntityRetriever          # noqa: E402
from retrievers.base import RetrievalResult            # noqa: E402
from temporal_index import TemporalIndex               # noqa: E402
from time_extract import extract_time_constraint, TimeConstraint  # noqa: E402


class TemporalEntityRetriever(EntityRetriever):
    name = "entity_temporal"

    def __init__(self, paths, cfg, embedder, llm=None,
                 temporal_index_path: Path | None = None):
        super().__init__(paths, cfg, embedder)
        self.llm = llm
        self._tindex: TemporalIndex | None = None
        self._tindex_path = (
            temporal_index_path
            or (Path(paths.kg_path).parent / "temporal_index.pkl")
        )
        # config knobs with safe defaults if not present on cfg
        self.temporal_seed_cap = getattr(cfg, "temporal_seed_cap", 40)
        self.extract_prefer = getattr(cfg, "time_extract_prefer", "llm")

    # ------------------------------------------------------------
    def setup(self) -> None:
        super().setup()
        if self._tindex is None:
            if not self._tindex_path.exists():
                raise FileNotFoundError(
                    f"Temporal index not found at {self._tindex_path}. "
                    f"Run phase5_temporal/temporal_index.py first."
                )
            self._tindex = TemporalIndex.load(self._tindex_path)

    # ------------------------------------------------------------
    def _temporal_seeds(self, tc: TimeConstraint) -> tuple[list[str], dict]:
        """Query the interval index for order_ids overlapping the
        extracted windows, capped and time-stratified.

        Returns (seed_node_ids, diagnostics).
        """
        if tc.is_empty:
            return [], {"n_raw": 0, "n_seeded": 0, "edge_types": [],
                        "windows": [], "extract_source": tc.source}

        # Which edge types to query. "any" → all four.
        edge_types = tc.edge_types or ["any"]
        if "any" in edge_types:
            query_edges = ["approved_at", "shipped_at",
                           "delivered_at", "reviewed_at"]
        else:
            query_edges = edge_types

        # Collect (interval_start, order_id) across all windows/edges
        hits: list[str] = []
        for (start, end) in tc.windows:
            for et in query_edges:
                try:
                    order_ids = self._tindex.query(et, start, end)
                except (KeyError, ValueError):
                    continue
                hits.extend(order_ids)

        # De-duplicate, preserve a stable order (sorted for determinism)
        unique_ids = sorted(set(hits))
        n_raw = len(unique_ids)

        # Time-stratified cap: evenly stride across the sorted list.
        if n_raw > self.temporal_seed_cap:
            stride = n_raw / self.temporal_seed_cap
            idxs = [int(i * stride) for i in range(self.temporal_seed_cap)]
            sampled = [unique_ids[i] for i in idxs]
        else:
            sampled = unique_ids

        # Map raw order_ids → KG node_ids (prefix "O:")
        seed_nodes = [f"O:{oid}" for oid in sampled
                      if f"O:{oid}" in self._G]

        return seed_nodes, {
            "n_raw": n_raw,
            "n_seeded": len(seed_nodes),
            "edge_types": query_edges,
            "windows": tc.windows,
            "extract_source": tc.source,
        }

    # ------------------------------------------------------------
    def _order_timeline(self, node_id: str) -> str:
        """Build a one-line timeline string for an Order node from its
        seven lifecycle / review timestamps.

        This mirrors serialize_order_v2() in phase4_retrieval/chunking.py
        (the Vanilla v2 baseline-fairness sentence). Without it, the
        inherited node description carries only the purchase date, so
        the LLM cannot compute delivery times / lags — it sees a single
        date per order and infers "placed and delivered same day"
        (the 0-day-lag artefact observed in the first pilot). Enriching
        every Order node in the subgraph gives System C the same
        temporal information access that Vanilla v2 has, so any
        difference is attributable to retrieval structure, not access.
        """
        a = self._G.nodes[node_id]

        def d10(attr):
            v = a.get(attr)
            if not v or v in ("None", "NaT", "nan"):
                return None
            return str(v)[:10]

        purchase  = d10("purchase_ts")
        approved  = d10("approved_at")
        carrier   = d10("delivered_carrier_date")
        delivered = d10("delivered_customer_date")
        estimated = d10("estimated_delivery_date")

        parts = []
        parts.append(f"purchased {purchase}" if purchase
                     else "purchase not recorded")
        parts.append(f"approved {approved}" if approved
                     else "approval not recorded")
        parts.append(f"shipped to carrier {carrier}" if carrier
                     else "carrier handover not recorded")
        if delivered and estimated:
            parts.append(f"delivered {delivered} (est. {estimated})")
        elif delivered:
            parts.append(f"delivered {delivered}")
        elif estimated:
            parts.append(f"delivery not recorded (est. {estimated})")
        else:
            parts.append("delivery not recorded")

        # Review creation date, if a Review node is linked.
        review_bit = ""
        for _, nb, ed in self._G.out_edges(node_id, data=True):
            if ed.get("edge_type") != "REVIEWED_BY":
                continue
            rnode = self._G.nodes.get(nb, {})
            cr = rnode.get("creation_date")
            if cr and str(cr)[:10] not in ("None", "NaT"):
                review_bit = f", reviewed {str(cr)[:10]}"
                break

        return "; ".join(parts) + review_bit

    def _serialize(self, nodes: set[str], anchors: set[str]) -> str:
        """Like EntityRetriever._serialize, but every Order node gets a
        timeline suffix so the LLM can see all lifecycle timestamps."""
        node_lines = []
        for n in sorted(nodes):
            d = self._G.nodes[n]
            nt = d.get("node_type", "?")
            desc = d.get("description", "(no description)")
            marker = " [ANCHOR]" if n in anchors else ""
            if nt == "Order":
                timeline = self._order_timeline(n)
                desc = f"{desc} Timeline: {timeline}."
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

        # 0. Extract temporal constraint
        tc = extract_time_constraint(
            question, llm=self.llm, prefer=self.extract_prefer,
        )

        # 1. ID-direct anchors (inherited)
        anchors = self._find_anchors(question)

        # 2. Embedding seeds (inherited)
        emb_seeds = self._embedding_seeds(
            question, self.cfg.entity_top_k, exclude=set(anchors),
        )

        # 3. Temporal seeds (NEW)
        temporal_seeds, tdiag = self._temporal_seeds(tc)

        seeds = anchors + emb_seeds + temporal_seeds

        # 4. Expand (inherited k-hop)
        expanded = self._khop(seeds, self.cfg.entity_hops)

        # 5. Protect anchors + their 1-hop, AND temporal seeds + their
        #    1-hop (so the seeded orders' payment/items/review survive
        #    pruning — those carry the timestamps and amounts the
        #    question needs).
        protected = set(anchors) | set(temporal_seeds)
        for a in list(protected):
            protected.update(self._G.successors(a))
            protected.update(self._G.predecessors(a))

        pruned = self._prune(
            expanded, self.cfg.entity_max_nodes, protected=protected,
        )
        evidence = self._serialize(pruned, anchors=set(anchors))

        return RetrievalResult(
            evidence_block=evidence,
            cited_node_ids=list(pruned),
            diagnostics={
                "anchors": anchors,
                "embedding_seeds": emb_seeds,
                "temporal": tdiag,
                "n_temporal_seeds": len(temporal_seeds),
                "n_after_expand": len(expanded),
                "n_after_prune": len(pruned),
                "n_protected": len(protected),
            },
        )
