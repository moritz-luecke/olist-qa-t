"""
============================================================
Phase 5.5 — System E: Temporal Hybrid Graph-RAG
============================================================

The temporal analogue of the time-blind HybridRetriever (System
B3), combining:

  - System C (temporal-entity): a temporally-seeded, timeline-
    enriched subgraph of *specific orders* — carries entity-level
    detail (state, category, per-order lifecycle timestamps), so it
    can answer entity-sub-filtered questions ("...for orders from
    São Paulo").
  - System D (community-temporal): the *exact pre-computed
    population aggregates* of the matched time bucket(s) — cannot
    miscalibrate a population mean, so it nails un-filtered trends
    and latency aggregates.

Two fusion modes
----------------
`fusion="route"` (DEFAULT, the headline System E) — score-based
    routing. The two retrievers' evidence types are NOT mixed in
    one prompt; instead we *select* the right one per question:
      - the question filters by a specific customer state → use
        System C (the entity subgraph can compute the filtered
        aggregate);
      - otherwise → use System D (the exact population aggregate).

`fusion="concat"` (ABLATION) — the naive B3-style union: C's block
    ⊕ "---" ⊕ D's block, both in one prompt. This is kept because
    the pilot shows it *underperforms* D-alone, which is the
    empirical motivation for routing (see below).

Why routing, not concatenation (pilot evidence)
-----------------------------------------------
Naive concatenation scored 15/30 on the pilot — *below* System D's
18/30 — because the unused half of the evidence acts as a
distractor, and the failure is two-directional:

  - **Entity-filtered question** (latency_01, "...for São Paulo"):
    D contributes the clean all-population 2017 mean (~13 days);
    the model anchors on it and ignores C's SP order timelines that
    would give the correct ~9.3. C-alone solves it; the fused
    context loses it.
  - **Multi-interval latency** (latency_03, latency_04): C's full
    per-order chains (purchase→approved→carrier→delivered) give the
    model alternative ways to compute "a latency", and it mislabels
    the *adjacent* interval (answering the carrier lag for a
    delivery-lag question, etc.). D-alone, which exposes only four
    cleanly-named aggregates, maps the label correctly.

Routing removes the distractor in both directions: entity-filtered
→ C only (no population bucket to anchor on); everything else → D
only (no per-order chains to mis-compute from). Expected pilot
result ≈ the C∪D oracle (19/30), with D's 0-spurious property
preserved. The routing *is* the contribution; concat is the
ablation that shows why naive fusion is not enough.

Routing heuristic (documented assumption)
-----------------------------------------
A question is treated as entity-(state-)filtered iff it names a
specific Brazilian state code (SP, RJ, ...) as a standalone
upper-case token. This catches the pilot's two state-filtered
latency questions (latency_01, latency_06) and nothing else
(e.g. drift_08 "top 5 customer states" names no specific state, so
it stays on D). The state set mirrors STATE_TO_REGION in
build_pilot_questions.py. Category/seller filters are NOT routed to
C because C cannot answer them either (0/10 on drift); they stay on
D. At scale this heuristic can be widened (full state names, etc.).

Cost / index requirements
-------------------------
System E needs BOTH temporal artefacts: the interval index
(outputs/temporal_index.pkl, for C) and the time buckets
(outputs/time_buckets.jsonl, for D). It shares one Embedder and one
ChatLLM across both sub-retrievers; in `route` mode only the chosen
sub-retriever runs per question (so it is also cheaper than concat).
============================================================
"""
from __future__ import annotations

import importlib.util as _ilu
import re
import sys
from pathlib import Path

# --- sys.path setup (mirrors temporal_entity.py / temporal_community.py) ---
_HERE = Path(__file__).resolve().parent                       # .../retrievers
_PHASE5 = _HERE.parent                                        # phase5_temporal/
_ROOT = _PHASE5.parent                                        # code/
sys.path.insert(0, str(_PHASE5))
sys.path.insert(0, str(_ROOT / "phase4_retrieval"))

from retrievers.base import BaseRetriever, RetrievalResult     # noqa: E402


# Brazilian state codes — mirrors STATE_TO_REGION keys in
# build_pilot_questions.py. Hard-coded here to keep the retriever free of
# a pandas import; the list is stable.
_STATE_CODES = frozenset({
    "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS",
    "MT", "PA", "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR", "RS", "SC",
    "SE", "SP", "TO",
})
# Pre-compiled: standalone upper-case 2-letter token that is a state code.
_STATE_RE = re.compile(r"\b(" + "|".join(sorted(_STATE_CODES)) + r")\b")


def _load_sibling(modname: str, filename: str):
    """Load a sibling temporal retriever by explicit file path. Each one
    self-configures sys.path for phase4_retrieval's `retrievers` package,
    so this avoids the package-name shadowing that putting
    phase5_temporal/retrievers on sys.path would cause."""
    spec = _ilu.spec_from_file_location(modname, _HERE / filename)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _get_temporal_entity_cls():
    return _load_sibling("temporal_entity",
                         "temporal_entity.py").TemporalEntityRetriever


def _get_temporal_community_cls():
    return _load_sibling("temporal_community",
                         "temporal_community.py").TemporalCommunityRetriever


class TemporalHybridRetriever(BaseRetriever):
    name = "hybrid_temporal"

    def __init__(self, paths, cfg, embedder, llm=None,
                 temporal_index_path: Path | None = None,
                 buckets_path: Path | None = None,
                 fusion: str | None = None,
                 entity_retriever: BaseRetriever | None = None,
                 community_retriever: BaseRetriever | None = None):
        """`fusion` is "route" (default) or "concat" (ablation). It may
        also be set via cfg.temporal_fusion. `entity_retriever` /
        `community_retriever` allow dependency injection in tests."""
        self.paths = paths
        self.cfg = cfg
        self.embedder = embedder
        self.llm = llm
        self.fusion = (fusion
                       or getattr(cfg, "temporal_fusion", None)
                       or "route")
        if self.fusion not in ("route", "concat"):
            raise ValueError(f"fusion must be 'route' or 'concat', "
                             f"got {self.fusion!r}")

        if entity_retriever is not None:
            self.entity = entity_retriever
        else:
            C = _get_temporal_entity_cls()
            self.entity = C(paths, cfg, embedder, llm=llm,
                            temporal_index_path=temporal_index_path)

        if community_retriever is not None:
            self.community = community_retriever
        else:
            D = _get_temporal_community_cls()
            self.community = D(paths, cfg, embedder, llm=llm,
                               buckets_path=buckets_path)

    # ------------------------------------------------------------
    def setup(self) -> None:
        self.entity.setup()
        self.community.setup()

    # ------------------------------------------------------------
    @staticmethod
    def _is_state_filtered(question: str) -> list[str]:
        """Return the specific state codes the question filters by
        (empty if none). Matches standalone upper-case codes only, so
        lower-case words like 'go'/'to' do not false-trigger."""
        return sorted(set(_STATE_RE.findall(question)))

    # ------------------------------------------------------------
    def _retrieve_concat(self, question: str) -> RetrievalResult:
        """Ablation: B3-style union of both evidence blocks."""
        ent = self.entity.retrieve(question)
        com = self.community.retrieve(question)
        merged = ent.evidence_block + "\n\n---\n\n" + com.evidence_block
        return RetrievalResult(
            evidence_block=merged,
            cited_node_ids=ent.cited_node_ids,
            diagnostics={
                "fusion": "concat",
                "route": "both",
                "entity": ent.diagnostics,
                "community": com.diagnostics,
                "bucket_ids": com.diagnostics.get("temporal", {})
                                 .get("bucket_ids", []),
                "n_temporal_seeds": ent.diagnostics.get("n_temporal_seeds", 0),
            },
        )

    def _retrieve_route(self, question: str) -> RetrievalResult:
        """Headline: select C for state-filtered questions, else D."""
        states = self._is_state_filtered(question)
        if states:
            res = self.entity.retrieve(question)        # System C primary
            route = "entity"
        else:
            res = self.community.retrieve(question)      # System D primary
            route = "community"

        com_diag = res.diagnostics.get("temporal", {}) if route == "community" \
            else {}
        return RetrievalResult(
            evidence_block=res.evidence_block,
            cited_node_ids=res.cited_node_ids,
            diagnostics={
                "fusion": "route",
                "route": route,
                "routed_states": states,
                "primary": res.diagnostics,
                "bucket_ids": com_diag.get("bucket_ids", []),
                "n_temporal_seeds": (res.diagnostics.get("n_temporal_seeds", 0)
                                     if route == "entity" else 0),
            },
        )

    # ------------------------------------------------------------
    def retrieve(self, question: str) -> RetrievalResult:
        if self.fusion == "concat":
            return self._retrieve_concat(question)
        return self._retrieve_route(question)
