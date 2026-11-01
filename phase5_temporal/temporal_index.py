"""
============================================================
Phase 5.1 — Temporal Index over the schema-aware KG
============================================================

Reads the existing ``outputs/kg_schema_aware.pkl`` (unchanged from
the prior Olist-QA paper), materialises four interval-typed edges
per order from the seven lifecycle / review timestamps already
stored as node attributes, and builds an in-memory ``IntervalTree``
per edge type for efficient range queries.

This is the foundation layer for Systems C, D, E of the temporal
extension (entity-temporal / community-temporal / hybrid-temporal):
each retriever calls into ``TemporalIndex.query()`` to find the
orders whose lifecycle intersects a queried time window.

Inputs
------
  outputs/kg_schema_aware.pkl     — produced by phase2_kg/build_kg.py

Outputs
-------
  outputs/temporal_index.pkl      — pickled TemporalIndex instance
  outputs/temporal_index_stats.json — human-readable stats incl.
                                      IntervalTree memory footprint
                                      (as reported in the paper)

CLI
---
  python phase5_temporal/temporal_index.py            # build + save
  python phase5_temporal/temporal_index.py --stats    # load + print stats
  python phase5_temporal/temporal_index.py --probe Q1-2017
                                                       # smoke query

Design notes
------------
- Half-open intervals throughout: [t_start, t_end). An order whose
  approved_at == purchase_ts gets the zero-width interval [t, t],
  which we widen to [t, t+1s] so half-open instant queries hit it.
- We materialise only *monotonic* intervals (t_end >= t_start) per
  the §4.1 filter — same convention as data_stats.py and the
  numbers reported in §4 Table 2 of the paper.
- The interval data structure is a dict[edge_type, _IntervalArray],
  where each _IntervalArray holds three parallel NumPy arrays
  (starts, ends, ids) sorted by start. Overlap queries are O(log n
  + k) via binary search. We do not use the `intervaltree` package:
  on this data (many near-zero-width intervals from orders whose
  approved_at == purchase_ts) its tree degenerates and a single
  range query took ~27 s; the sorted-array index answers the same
  query in well under a millisecond.
- Memory footprint is reported via the pickled file size, which is
  a tight upper bound on the in-memory NumPy arrays plus the id
  lists (the on-disk pickle approximates the in-memory footprint to
  within ~30%).
============================================================
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import numpy as np


# ----------------------------------------------------------------
# _IntervalArray - a static, sorted-array interval index
# ----------------------------------------------------------------
#
# We deliberately do NOT use the `intervaltree` package. That
# library is built for *dynamic* trees (insert/delete) and its
# query performance degrades badly when many intervals share
# near-identical bounds - which is exactly our situation (tens of
# thousands of orders with approved_at == purchase_ts give
# near-zero-width intervals, collapsing the tree toward O(n) per
# query). On the full Olist KG that produced ~27 s per query.
#
# Our index is *static*: build once, then read-only. For that, three
# parallel NumPy arrays sorted by interval start give O(log n + k)
# overlap queries via binary search, and query time drops from
# ~27 s to well under a millisecond.
#
# Overlap test for a half-open query window [qs, qe) against a
# stored half-open interval [s, e):  (s < qe) AND (e > qs).
class _IntervalArray:
    """Static interval index backed by sorted NumPy arrays."""

    __slots__ = ("starts", "ends", "ids")

    def __init__(self, starts, ends, ids):
        order = np.argsort(starts, kind="stable")
        self.starts = np.asarray(starts, dtype=np.float64)[order]
        self.ends = np.asarray(ends, dtype=np.float64)[order]
        self.ids = [ids[i] for i in order]

    def __len__(self):
        return len(self.ids)

    def overlap(self, qs: float, qe: float) -> list:
        """Return ids of intervals overlapping [qs, qe)."""
        # Candidate set: all intervals whose start < qe. Because
        # starts are sorted, that's a prefix up to searchsorted(qe).
        hi = int(np.searchsorted(self.starts, qe, side="left"))
        if hi == 0:
            return []
        # Among those, keep the ones whose end > qs.
        mask = self.ends[:hi] > qs
        idxs = np.nonzero(mask)[0]
        return [self.ids[i] for i in idxs]

    def at(self, t: float) -> list:
        """Return ids of intervals containing the instant t
        (half-open: start <= t < end)."""
        hi = int(np.searchsorted(self.starts, t, side="right"))
        if hi == 0:
            return []
        mask = (self.ends[:hi] > t) & (self.starts[:hi] <= t)
        idxs = np.nonzero(mask)[0]
        return [self.ids[i] for i in idxs]


# Make build_kg.NT / ET importable so we can match node_type strings.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "phase2_kg"))
from build_kg import NT, ET  # noqa: E402


# ----------------------------------------------------------------
# Edge-type spec — single source of truth for what each interval is
# ----------------------------------------------------------------

# The four interval-typed edges materialised by this layer. Each
# spec says: which node type to read from, which two attributes
# define the [start, end) interval, and the canonical edge name
# used as the IntervalTree dict key.
#
# `t_end_node` is "self" for the three order-internal intervals
# (purchase → approved, approved → carrier, carrier → delivered).
# For reviewed_at the start is on the Order node and the end is
# on the linked Review node — we follow the REVIEWED_BY edge.

INTERVAL_SPECS = [
    {
        "edge_type":   "approved_at",
        "source_node": NT.ORDER,
        "t_start_attr": "purchase_ts",
        "t_end_attr":   "approved_at",
        "t_end_node":   "self",
    },
    {
        "edge_type":   "shipped_at",
        "source_node": NT.ORDER,
        "t_start_attr": "approved_at",
        "t_end_attr":   "delivered_carrier_date",
        "t_end_node":   "self",
    },
    {
        "edge_type":   "delivered_at",
        "source_node": NT.ORDER,
        "t_start_attr": "delivered_carrier_date",
        "t_end_attr":   "delivered_customer_date",
        "t_end_node":   "self",
    },
    {
        "edge_type":   "reviewed_at",
        "source_node": NT.ORDER,
        "t_start_attr": "delivered_customer_date",
        "t_end_attr":   "creation_date",        # on the Review node
        "t_end_node":   NT.REVIEW,
        "follow_edge":  ET.REVIEWED_BY,
    },
]


# ----------------------------------------------------------------
# Timestamp parsing
# ----------------------------------------------------------------

# Pickled KG stores timestamps as `str(pd.Timestamp)`, which the
# default pd.to_datetime can parse. We avoid importing pandas
# here to keep the temporal index pandas-free at runtime — fewer
# moving parts when Systems C/D/E load it.
_TS_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d",
)


def _parse_ts(s: str | None) -> float | None:
    """Parse a stringified pd.Timestamp into a POSIX float, or None.

    Returns POSIX seconds rather than datetime objects because
    IntervalTree compares numerics roughly 10x faster than dates,
    and ~99k × 4 = ~400k intervals matter.

    IMPORTANT: we interpret the naive Olist timestamps as UTC, not
    local time. Using datetime.timestamp() on a naive datetime would
    apply the *machine's* local timezone, which corrupts the ordering
    of timestamps that fall in a daylight-saving transition hour. For
    example order O:32b8249f... has purchase 2018-03-25 02:56:02 and
    approval 2018-03-25 03:10:20 -- both inside the DST spring-forward
    window -- which a local-tz conversion flips into approval < purchase.
    Olist timestamps are wall-clock with no tz, so a fixed UTC offset
    is the correct, DST-free interpretation.
    """
    if s is None or s == "" or s == "NaT" or s == "nan":
        return None
    for fmt in _TS_FORMATS:
        try:
            return datetime.strptime(s, fmt).replace(
                tzinfo=timezone.utc
            ).timestamp()
        except ValueError:
            continue
    return None


# ----------------------------------------------------------------
# TemporalIndex
# ----------------------------------------------------------------

@dataclass
class TemporalIndexStats:
    n_orders_total: int
    n_intervals_per_type: dict[str, int] = field(default_factory=dict)
    n_anomalous_per_type: dict[str, int] = field(default_factory=dict)
    build_time_sec: float = 0.0
    pickle_bytes: int = 0
    pickle_mb: float = 0.0

    def to_dict(self) -> dict:
        return {
            "n_orders_total": self.n_orders_total,
            "n_intervals_per_type": self.n_intervals_per_type,
            "n_anomalous_per_type": self.n_anomalous_per_type,
            "build_time_sec": round(self.build_time_sec, 2),
            "pickle_bytes": self.pickle_bytes,
            "pickle_mb": round(self.pickle_mb, 2),
        }


class TemporalIndex:
    """In-memory interval indices over the 4 lifecycle edge types.

    Public API:
      .query(edge_type, start, end)  → list[str]  (order_ids)
      .query_point(edge_type, t)     → list[str]
      .stats                         → TemporalIndexStats
    """

    def __init__(self) -> None:
        # one _IntervalArray per edge_type (None until build populates)
        self.trees: dict[str, _IntervalArray | None] = {
            spec["edge_type"]: None for spec in INTERVAL_SPECS
        }
        self.stats = TemporalIndexStats(n_orders_total=0)

    # ----- Build ------------------------------------------------------

    @classmethod
    def build(cls, kg: nx.MultiDiGraph,
              filter_monotonic: bool = True) -> "TemporalIndex":
        """Materialise the four interval indices from the KG.

        Parameters
        ----------
        kg
            The schema-aware KG (NetworkX MultiDiGraph). Read-only.
        filter_monotonic
            When True (default), only intervals with t_end >= t_start
            are added; the others are counted as anomalies in stats
            but not indexed. This matches the §4.1 paper filter.
        """
        idx = cls()
        t0 = time.time()

        # Collect order nodes once
        order_nodes = [
            (n, d) for n, d in kg.nodes(data=True)
            if d.get("node_type") == NT.ORDER
        ]
        idx.stats.n_orders_total = len(order_nodes)

        for spec in INTERVAL_SPECS:
            starts: list[float] = []
            ends: list[float] = []
            ids: list[str] = []
            n_anom = 0

            for node_id, attrs in order_nodes:
                order_id = attrs.get("order_id")
                if order_id is None:
                    continue

                t_start = _parse_ts(attrs.get(spec["t_start_attr"]))
                if t_start is None:
                    continue

                # End attribute lives on the order itself or on a
                # linked node (review).
                if spec["t_end_node"] == "self":
                    t_end = _parse_ts(attrs.get(spec["t_end_attr"]))
                else:
                    # follow the REVIEWED_BY edge to find the Review
                    t_end = _follow_to_review_creation(
                        kg, node_id, spec
                    )
                if t_end is None:
                    continue

                if t_end < t_start:
                    n_anom += 1
                    if filter_monotonic:
                        continue

                # Zero-width intervals (t_end == t_start) are widened by
                # 1 second so an instant query at t still hits them under
                # the half-open [start, end) convention. With the sorted-
                # array index this no longer causes any performance issue
                # (unlike the old IntervalTree), but we keep the widening
                # so point queries behave intuitively.
                if t_end == t_start:
                    t_end = t_start + 1.0

                starts.append(t_start)
                ends.append(t_end)
                ids.append(order_id)

            if starts:
                idx.trees[spec["edge_type"]] = _IntervalArray(
                    starts, ends, ids
                )
            else:
                idx.trees[spec["edge_type"]] = _IntervalArray([], [], [])
            idx.stats.n_intervals_per_type[spec["edge_type"]] = len(starts)
            idx.stats.n_anomalous_per_type[spec["edge_type"]] = n_anom

        idx.stats.build_time_sec = time.time() - t0
        return idx

    # ----- Query ------------------------------------------------------

    def query(self, edge_type: str, start: str | float,
              end: str | float) -> list[str]:
        """Return order_ids whose `edge_type` interval overlaps
        the half-open window [start, end).

        `start` / `end` may be POSIX floats or ISO-date strings.
        """
        arr = self.trees.get(edge_type)
        if arr is None:
            raise KeyError(
                f"Unknown edge_type {edge_type!r}. "
                f"Known: {sorted(self.trees)}"
            )
        s = start if isinstance(start, (int, float)) else _parse_ts(start)
        e = end   if isinstance(end,   (int, float)) else _parse_ts(end)
        if s is None or e is None:
            raise ValueError(f"Could not parse window [{start}, {end})")
        if e <= s:
            raise ValueError(f"Empty window: [{start}, {end})")
        return arr.overlap(s, e)

    def query_point(self, edge_type: str, t: str | float) -> list[str]:
        """Return order_ids whose `edge_type` interval contains the
        single instant `t`."""
        arr = self.trees.get(edge_type)
        if arr is None:
            raise KeyError(f"Unknown edge_type {edge_type!r}.")
        ts = t if isinstance(t, (int, float)) else _parse_ts(t)
        if ts is None:
            raise ValueError(f"Could not parse instant {t!r}")
        return arr.at(ts)

    # ----- Persistence -----------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        size = path.stat().st_size
        self.stats.pickle_bytes = size
        self.stats.pickle_mb = size / 1e6

    @classmethod
    def load(cls, path: Path) -> "TemporalIndex":
        with path.open("rb") as f:
            return pickle.load(f)


# ----------------------------------------------------------------
# Helper for reviewed_at: follow REVIEWED_BY to a Review node
# ----------------------------------------------------------------

def _follow_to_review_creation(
    kg: nx.MultiDiGraph, order_node: str, spec: dict,
) -> float | None:
    """Return the earliest creation_date among reviews linked from
    `order_node` via REVIEWED_BY, as POSIX seconds. None if none.

    The pickled KG stores duplicate review_ids (the prior paper's
    audit found 814 of them). We take the earliest creation_date
    across all linked reviews — same convention as
    data_stats.compute_interval_edges, which counts review *rows*
    after a left-join, but ranges to a single timestamp per order
    is the only sensible thing here.
    """
    earliest: float | None = None
    for _, neighbor, edata in kg.out_edges(order_node, data=True):
        if edata.get("edge_type") != spec["follow_edge"]:
            continue
        nattrs = kg.nodes[neighbor]
        if nattrs.get("node_type") != spec["t_end_node"]:
            continue
        t = _parse_ts(nattrs.get(spec["t_end_attr"]))
        if t is None:
            continue
        if earliest is None or t < earliest:
            earliest = t
    return earliest


# ----------------------------------------------------------------
# Quarter / month / window helpers for retriever code (Systems C-E)
# ----------------------------------------------------------------

def quarter_window(spec: str) -> tuple[str, str]:
    """E.g. 'Q1-2017' or '2017-Q1' → ('2017-01-01', '2017-04-01')."""
    s = spec.upper().replace(" ", "")
    if s.startswith("Q") and "-" in s:
        q, y = s.split("-", 1)
    elif "-Q" in s:
        y, q = s.split("-Q", 1)
        q = "Q" + q
    else:
        raise ValueError(f"Unrecognised quarter spec: {spec!r}")
    q_num = int(q[1])
    year = int(y)
    start_month = (q_num - 1) * 3 + 1
    end_year = year + (1 if q_num == 4 else 0)
    end_month = 1 if q_num == 4 else start_month + 3
    return (f"{year}-{start_month:02d}-01",
            f"{end_year}-{end_month:02d}-01")


def month_window(spec: str) -> tuple[str, str]:
    """E.g. '2017-07' → ('2017-07-01', '2017-08-01')."""
    y, m = spec.split("-")
    y, m = int(y), int(m)
    end_y, end_m = (y, m + 1) if m < 12 else (y + 1, 1)
    return (f"{y}-{m:02d}-01", f"{end_y}-{end_m:02d}-01")


# ----------------------------------------------------------------
# CLI
# ----------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build or inspect the temporal index over the "
                    "schema-aware KG.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--kg-path", type=Path,
        default=Path("./outputs/kg_schema_aware.pkl"),
    )
    ap.add_argument(
        "--out-path", type=Path,
        default=Path("./outputs/temporal_index.pkl"),
    )
    ap.add_argument(
        "--stats-path", type=Path,
        default=Path("./outputs/temporal_index_stats.json"),
    )
    ap.add_argument(
        "--stats", action="store_true",
        help="Load an existing temporal_index.pkl and print stats only.",
    )
    ap.add_argument(
        "--probe", type=str, default=None,
        help="Smoke query: a quarter spec like '2017-Q1' — counts "
             "delivered_at intersections.",
    )
    ap.add_argument(
        "--include-anomalies", action="store_true",
        help="Index non-monotonic intervals too "
             "(default: filter, per §4.1).",
    )
    args = ap.parse_args()

    if args.stats:
        if not args.out_path.exists():
            print(f"ERROR: {args.out_path} not found — build it first.",
                  file=sys.stderr)
            return 1
        print(f"Loading {args.out_path} ...")
        idx = TemporalIndex.load(args.out_path)
        print(json.dumps(idx.stats.to_dict(), indent=2))
        if args.probe:
            _probe(idx, args.probe)
        return 0

    # ---- Build path ---------------------------------------------------
    if not args.kg_path.exists():
        print(f"ERROR: {args.kg_path} not found. Run "
              f"phase2_kg/build_kg.py first.", file=sys.stderr)
        return 1

    print(f"Loading KG from {args.kg_path} ...")
    with args.kg_path.open("rb") as f:
        kg = pickle.load(f)
    print(f"  {kg.number_of_nodes():,} nodes, "
          f"{kg.number_of_edges():,} edges")

    print(f"Building temporal index "
          f"(filter_monotonic={not args.include_anomalies}) ...")
    idx = TemporalIndex.build(
        kg, filter_monotonic=not args.include_anomalies,
    )

    print("Saving ...")
    idx.save(args.out_path)
    args.stats_path.parent.mkdir(parents=True, exist_ok=True)
    args.stats_path.write_text(json.dumps(idx.stats.to_dict(), indent=2))

    print()
    print("=" * 56)
    print("Temporal index stats")
    print("=" * 56)
    print(json.dumps(idx.stats.to_dict(), indent=2))
    print()
    print(f"Wrote {args.out_path}  ({idx.stats.pickle_mb:.1f} MB)")
    print(f"Wrote {args.stats_path}")

    # ---- Sanity reference against the data_stats.py numbers ----------
    # NOTE on semantics: data_stats.py counts review *rows* for
    # reviewed_at (an order with N reviews contributes up to N),
    # whereas this index counts *orders* (one earliest-review
    # interval per order). The order-based reviewed_at count is
    # therefore lower than the data_stats row-based count by the
    # number of orders with multiple monotonic reviews (335 such
    # orders → 491 extra rows). Both are correct for their purpose;
    # the index value is the one that belongs in the §4 table for an
    # interval-per-order structure.
    print()
    print("Cross-check vs. data_stats.py:")
    expected = {
        "approved_at":  (99_281, "exact match expected"),
        "shipped_at":   (96_285, "exact match expected"),
        "delivered_at": (96_452, "exact match expected"),
        "reviewed_at":  (87_548, "order-based (data_stats rows=88,039)"),
    }
    for edge_type, (want, note) in expected.items():
        got = idx.stats.n_intervals_per_type.get(edge_type, 0)
        marker = "OK" if got == want else "≠"
        print(f"  {marker}  {edge_type:<14} indexed={got:>6,}  "
              f"expected={want:>6,}  ({note})")

    if args.probe:
        _probe(idx, args.probe)

    return 0


def _probe(idx: TemporalIndex, spec: str) -> None:
    """Run a sanity range query and print the count."""
    try:
        start, end = quarter_window(spec)
    except ValueError:
        try:
            start, end = month_window(spec)
        except Exception:
            print(f"  cannot parse probe spec: {spec!r}")
            return
    print()
    print(f"Probe: orders with delivered_at ∩ [{start}, {end}):")
    hits = idx.query("delivered_at", start, end)
    print(f"  {len(hits):,} order_ids match")
    if hits:
        print(f"  first 3: {hits[:3]}")


if __name__ == "__main__":
    sys.exit(main())
