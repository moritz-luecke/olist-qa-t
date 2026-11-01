"""
============================================================
Phase 5.4 — System D: Temporal Community Graph-RAG
============================================================

Extends the time-blind CommunityRetriever (System B2) so that,
instead of retrieving over the 4,504 entity-community summaries,
it retrieves over the ~38 *time-bucket* summaries built by
phase5_temporal/time_hierarchy.py (the bi-level year→quarter→month
hierarchy).

The whole point of System D
---------------------------
Vanilla and System C answer aggregation questions from a *sample*
of orders (cosine-top-k chunks; a 40-order time-stratified seed
set). Both therefore commit *sample-mean miscalibration* — they
estimate a population mean from a biased sub-sample (see
notes/aggregation_miscalibration.md, notes/system_c_pilot.md).

System D answers them from PRE-COMPUTED, exact, pandas-derived
population aggregates per time bucket. The bucket `stats_block`
is computed from the same CSV path and the same monotonic filter
as the gold answers (time_hierarchy.bucket_stats), so for an
un-filtered population aggregate over a bucket-aligned window the
evidence equals the gold by construction. There is no sub-sample,
hence no miscalibration.

Retrieval algorithm (delta over CommunityRetriever)
---------------------------------------------------
  0. Extract a time constraint from the question (LLM main path
     + rule fallback; same time_extract.py as System C).
  1. Decide the *target granularity*:
       - "per/each/by month" / "monthly"     → month
       - "per/each/by quarter" / "quarterly"  → quarter
       - "per/each/by year" / "annually"      → year
       - otherwise (a point or comparison)    → the window's own
                                                granularity.
  2. For every extracted window:
       - target granularity set → all buckets at that level whose
         [start,end) is contained in the window (e.g. "per quarter
         of 2017" → the four 2017 quarter buckets).
       - otherwise → the single bucket whose boundaries exactly
         equal the window ("Q4 2017" → 2017-Q4); if no single
         bucket matches, tile the window with the coarsest whole
         buckets that cover it (a merged "Jul-Dec 2017" span → Q3
         + Q4). This is the "direct bucket lookup if the window
         matches a bucket boundary" path from the project plan.
  3. If neither path selects anything (no time reference, or a
     window that does not tile cleanly onto bucket boundaries —
     e.g. a 30-day pre-Black-Friday window), fall back to
     embedding-cosine over the bucket prose summaries, exactly
     like the parent CommunityRetriever. This is graceful
     degradation: System D never does *worse* than time-blind
     community retrieval on a question it cannot align.
  4. Serialize the selected buckets' exact `stats_block` (NOT the
     prose summary) into a TIME BUCKETS evidence block. The prose
     summary exists only so step 3's question-embedding can match
     the right bucket.

Design choice — granularest matching level only
-----------------------------------------------
For "Q4 2017" we hand the LLM the single 2017-Q4 bucket, not
Q4 + its three months and not the surrounding year. For
"per quarter of 2017" we hand it exactly the four quarter buckets,
not the year aggregate (a distractor) nor the twelve months.
This keeps the context lean and the evidence at exactly the
granularity the question asks about. The granularity ablation
(RQ3 / H4) flips `force_level` to study finer/coarser choices.

What System D is NOT expected to solve
--------------------------------------
  - Entity-subfiltered aggregates (e.g. "...for orders from São
    Paulo"): the buckets are not filtered by state/category, so
    the bucket mean is the all-population mean. This is the gap
    System E (hybrid C+D) closes — C supplies the entity filter,
    D supplies the population time aggregate.
  - Sub-bucket custom windows (30 days before an event; a World
    Cup date range): they do not tile onto bucket boundaries, so
    System D falls back to cosine and typically abstains. Motivates
    finer buckets or the interval index (System C/E).
  - Arbitrary per-category drift: a bucket carries only top-5 by
    volume / top-5 by review / top-3 payment share / top-5 states,
    so drift over categories outside those top-k is not estimable
    from the summary. Where the drifting entity is in the carried
    top-k (e.g. customer-state or payment-method drift), D answers.

We measure all of the above on the 30-question pilot.
============================================================
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

# --- sys.path setup (mirrors temporal_entity.py) -----------------
# phase4_retrieval must be on the path FIRST so that `retrievers.community`
# resolves to the prior repo's package (not the sibling phase5_temporal/
# retrievers/ this file lives in). phase5_temporal is added so we can import
# time_extract / temporal_index window helpers.
_HERE = Path(__file__).resolve().parent                       # .../retrievers
_PHASE5 = _HERE.parent                                        # phase5_temporal/
_ROOT = _PHASE5.parent                                        # code/
sys.path.insert(0, str(_PHASE5))
sys.path.insert(0, str(_ROOT / "phase4_retrieval"))

from retrievers.community import CommunityRetriever            # noqa: E402
from retrievers.base import RetrievalResult                    # noqa: E402
from time_extract import extract_time_constraint, TimeConstraint  # noqa: E402


# Coarse → fine ordering, used to pick the coarsest whole bucket when tiling.
_LEVEL_RANK = {"year": 0, "quarter": 1, "month": 2, "week": 3}

# Phrases that pin the answer to a particular breakdown granularity.
_GRANULARITY_PATTERNS = [
    ("week",    re.compile(r"\b(per|each|every|by)\s+week\b|\bweekly\b"
                           r"|\bweek[- ]by[- ]week\b")),
    ("month",   re.compile(r"\b(per|each|every|by)\s+month\b|\bmonthly\b"
                           r"|\bmonth[- ]by[- ]month\b")),
    ("quarter", re.compile(r"\b(per|each|every|by)\s+quarter\b|\bquarterly\b")),
    ("year",    re.compile(r"\b(per|each|every|by)\s+year\b|\bannually\b"
                           r"|\byear[- ]over[- ]year\b")),
]


class TemporalCommunityRetriever(CommunityRetriever):
    name = "community_temporal"

    def __init__(self, paths, cfg, embedder, llm=None,
                 buckets_path: Path | None = None,
                 bucket_embeds_path: Path | None = None):
        # NB: we intentionally do NOT call CommunityRetriever.setup(); we
        # reuse only its __init__ shape and its cosine machinery.
        super().__init__(paths, cfg, embedder)
        self.llm = llm
        out_dir = Path(paths.kg_path).parent
        self._buckets_path = buckets_path or (out_dir / "time_buckets.jsonl")
        self._bucket_embeds_path = (
            bucket_embeds_path or (out_dir / "time_bucket_embeds.npy")
        )
        # Config knobs with safe defaults if absent on cfg.
        self.extract_prefer = getattr(cfg, "time_extract_prefer", "llm")
        self.cosine_top_k = getattr(cfg, "time_bucket_top_k",
                                    getattr(cfg, "community_top_k", 3))
        self.max_buckets = getattr(cfg, "time_bucket_max", 16)
        # Granularity-ablation override (RQ3/H4). None → auto-detect.
        self.force_level = getattr(cfg, "time_force_level", None)

        self._records: list[dict] | None = None
        self._by_id: dict[str, dict] = {}
        self._by_level: dict[str, list[dict]] = {}

    # ------------------------------------------------------------
    def setup(self) -> None:
        """Load the time buckets and embed their prose summaries.

        Overrides CommunityRetriever.setup() entirely — we load
        time_buckets.jsonl instead of community_summaries.jsonl and
        cache the prose-summary embeddings to a separate npy file.
        """
        if not self._buckets_path.exists():
            raise FileNotFoundError(
                f"{self._buckets_path} not found. Build it first:\n"
                f"  python phase5_temporal/time_hierarchy.py --no-dry-run"
            )

        records: list[dict] = []
        with self._buckets_path.open(encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        if not records:
            raise RuntimeError(f"No buckets in {self._buckets_path}")
        self._records = records

        self._by_id = {r["bucket_id"]: r for r in records}
        self._by_level = {}
        for r in records:
            self._by_level.setdefault(r["level"], []).append(r)
        for lvl in self._by_level:
            self._by_level[lvl].sort(key=lambda r: r["start"])

        # Embed prose summaries for the cosine fallback (cache to npy).
        # Reuse the parent's normalisation convention (L2-normalised rows).
        if self._bucket_embeds_path.exists():
            embeds = np.load(self._bucket_embeds_path)
            if embeds.shape[0] != len(records):
                embeds = None
        else:
            embeds = None
        if embeds is None:
            texts = [r.get("summary") or r.get("stats_block", "")
                     for r in records]
            embeds = self.embedder.embed_many(texts, desc="time-bucket embeds")
            norms = np.linalg.norm(embeds, axis=1, keepdims=True) + 1e-9
            embeds = embeds / norms
            self._bucket_embeds_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(self._bucket_embeds_path, embeds)
        self._embeds = embeds

    # ------------------------------------------------------------
    # Granularity detection
    # ------------------------------------------------------------
    def _detect_target_level(self, question: str) -> str | None:
        if self.force_level in _LEVEL_RANK:
            return self.force_level
        q = question.lower()
        for level, pat in _GRANULARITY_PATTERNS:
            if pat.search(q):
                return level
        return None

    # ------------------------------------------------------------
    # Window → bucket matching
    # ------------------------------------------------------------
    def _buckets_in_window(self, level: str,
                           ws: str, we: str) -> list[dict]:
        """All buckets at `level` fully contained in [ws, we)."""
        return [r for r in self._by_level.get(level, [])
                if ws <= r["start"] and r["end"] <= we]

    def _exact_bucket(self, ws: str, we: str) -> dict | None:
        """The single bucket whose boundaries equal [ws, we), if any.

        Prefer the coarsest level (a full year matches the year
        bucket, not three-months-worth of nothing) — though in
        practice exactly one bucket can ever match a given boundary
        pair, so the tie-break is defensive.
        """
        matches = [r for r in self._records
                   if r["start"] == ws and r["end"] == we]
        if not matches:
            return None
        return min(matches, key=lambda r: _LEVEL_RANK.get(r["level"], 9))

    def _tile_window(self, ws: str, we: str) -> list[dict]:
        """Cover [ws, we) with the coarsest non-overlapping whole
        buckets. Returns [] if the window cannot be tiled cleanly
        onto bucket boundaries (→ caller falls back to cosine).
        """
        cursor = ws
        selected: list[dict] = []
        guard = 0
        while cursor < we:
            guard += 1
            if guard > 64:                       # pathological safety net
                return []
            candidates = [r for r in self._records
                          if r["start"] == cursor and r["end"] <= we]
            if not candidates:
                return []                        # ragged boundary → no tiling
            # Coarsest = largest span = furthest end.
            nxt = max(candidates, key=lambda r: r["end"])
            selected.append(nxt)
            cursor = nxt["end"]
        return selected

    # ------------------------------------------------------------
    def _select_buckets(self, tc: TimeConstraint,
                        question: str) -> tuple[list[dict], dict]:
        """Pick buckets for the question. Returns (records, diag)."""
        target_level = self._detect_target_level(question)

        if tc.is_empty:
            return [], {
                "mode": "none", "target_level": target_level,
                "windows": [], "extract_source": tc.source,
                "bucket_ids": [],
            }

        chosen: list[dict] = []
        seen: set[str] = set()
        per_window: list[dict] = []

        for (ws, we) in tc.windows:
            picks: list[dict]
            if target_level is not None:
                picks = self._buckets_in_window(target_level, ws, we)
                mode = f"contained@{target_level}"
            else:
                exact = self._exact_bucket(ws, we)
                if exact is not None:
                    picks = [exact]
                    mode = f"exact@{exact['level']}"
                else:
                    picks = self._tile_window(ws, we)
                    mode = "tiled" if picks else "unaligned"
            per_window.append({
                "window": [ws, we], "mode": mode,
                "picked": [p["bucket_id"] for p in picks],
            })
            for p in picks:
                if p["bucket_id"] not in seen:
                    seen.add(p["bucket_id"])
                    chosen.append(p)

        # Deterministic order: coarse→fine, then chronological.
        chosen.sort(key=lambda r: (_LEVEL_RANK.get(r["level"], 9), r["start"]))
        if len(chosen) > self.max_buckets:
            chosen = chosen[: self.max_buckets]

        diag = {
            "mode": "structured",
            "target_level": target_level,
            "windows": [list(w) for w in tc.windows],
            "extract_source": tc.source,
            "per_window": per_window,
            "bucket_ids": [r["bucket_id"] for r in chosen],
        }
        return chosen, diag

    def _cosine_fallback(self, question: str) -> tuple[list[dict], dict]:
        """Top-k bucket prose summaries by cosine — the time-blind
        community behaviour, used only when no window aligns."""
        assert self._embeds is not None and self._records is not None
        q_vec = self.embedder.embed_one(question)
        q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-9)
        sims = self._embeds @ q_vec
        k = min(self.cosine_top_k, len(self._records))
        top_idx = np.argpartition(-sims, k - 1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]
        chosen = [self._records[i] for i in top_idx]
        diag = {
            "mode": "cosine_fallback",
            "bucket_ids": [r["bucket_id"] for r in chosen],
            "top_similarities": [float(sims[i]) for i in top_idx],
        }
        return chosen, diag

    # ------------------------------------------------------------
    def _serialize(self, records: list[dict]) -> str:
        """Render selected buckets' exact stats_block into an
        evidence block. We feed the exact numbers, never the prose."""
        if not records:
            return ("TIME BUCKETS:\n(no time bucket matched the question's "
                    "time window)")
        blocks = []
        for r in records:
            sb = r.get("stats_block")
            if not sb:                                   # defensive
                sb = f"Time bucket {r['bucket_id']}: (no stats)"
            blocks.append(sb)
        header = (
            "TIME BUCKETS (exact pre-computed population statistics for the "
            "time period(s) matching the question; each block is computed "
            "over ALL orders whose purchase date falls in the bucket):\n"
        )
        return header + "\n\n".join(blocks)

    # ------------------------------------------------------------
    def retrieve(self, question: str) -> RetrievalResult:
        assert self._records is not None, "Call setup() first"

        tc = extract_time_constraint(
            question, llm=self.llm, prefer=self.extract_prefer,
        )
        chosen, diag = self._select_buckets(tc, question)

        if not chosen:
            chosen, fb_diag = self._cosine_fallback(question)
            diag = {**diag, **fb_diag}

        evidence = self._serialize(chosen)

        return RetrievalResult(
            evidence_block=evidence,
            cited_node_ids=[],          # cites time buckets, not nodes
            diagnostics={
                "temporal": diag,
                "n_buckets": len(chosen),
                "bucket_levels": [r["level"] for r in chosen],
            },
        )
