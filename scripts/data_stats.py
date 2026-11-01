"""
============================================================
Phase 5.0 — Data statistics for the temporal extension
============================================================

Computes the timestamp-coverage and interval-edge counts
reported for the temporal layer (coverage table and
interval-edge table). Pure-CSV: no NetworkX, no pickle, no
dependency on the temporal layer being implemented yet.

Stats computed (10 values):

  Table 1 — Timestamp coverage (6 values):
    - order_approved_at
    - order_delivered_carrier_date
    - order_delivered_customer_date
    - shipping_limit_date
    - review_creation_date
    - review_answer_timestamp

  Table 2 — Interval-edge counts (4 values):
    - approved_at      = orders with purchase + approved non-null,
                         and approved >= purchase (monotonic)
    - shipped_at       = orders with approved + carrier non-null,
                         and carrier   >= approved
    - delivered_at     = orders with carrier + customer non-null,
                         and customer  >= carrier
    - reviewed_at      = orders with customer + review_creation
                         non-null, and review_creation >= customer
                         (joined via order_id)

Two values in Table 2 are NOT computed here (they depend on
the temporal-graph layer being built):
    - IntervalTree memory footprint
    - Time-bucket summary tokens (mean / max)

Sanity checks (verifies prose claims in Sections 1, 4, 4.1):
    - total order count           (claim: 99,441)
    - data window                 (claim: 2016-09 to 2018-10)
    - orders per day              (claim: ~129)
    - non-monotonic anomalies     (claim: 1,359 carrier<approval)

Output:
  Two report blocks printed to stdout:
    [1] human-readable summary
    [2] LaTeX-ready snippets — paste directly into the paper

Usage:
  python data_stats.py [--data-dir DATA_DIR]

The --data-dir defaults to ../data, matching the prior repo
layout (phase2_kg/build_kg.py uses the same default).
============================================================
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


# ------------------------------------------------------------
# DATE_COLS — copied verbatim from phase2_kg/build_kg.py
# so this script is bit-identical with the existing loader.
# ------------------------------------------------------------

DATE_COLS = {
    "orders": [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "order_items": ["shipping_limit_date"],
    "reviews": ["review_creation_date", "review_answer_timestamp"],
}


def load_olist_minimal(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load only the three CSVs we need for the temporal stats.

    Mirrors `phase2_kg/build_kg.py::load_olist` but skips tables
    that carry no timestamps (customers, sellers, geolocation,
    products, category_translation, payments).
    """
    files = {
        "orders":      "olist_orders_dataset.csv",
        "order_items": "olist_order_items_dataset.csv",
        "reviews":     "olist_order_reviews_dataset.csv",
    }
    tables = {}
    for key, fname in files.items():
        path = data_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        tables[key] = pd.read_csv(path, parse_dates=DATE_COLS.get(key, []))
    return tables


# ------------------------------------------------------------
# Coverage statistics (Table 1)
# ------------------------------------------------------------

@dataclass(frozen=True)
class CoverageRow:
    column: str
    semantics: str
    coverage_pct: float
    n_non_null: int
    n_total: int


def compute_coverage(tables: dict[str, pd.DataFrame]) -> list[CoverageRow]:
    """Coverage % for every timestamp column that appears in Table 1.

    Coverage = non-null share within the relevant parent table.
    """
    n_orders  = len(tables["orders"])
    n_items   = len(tables["order_items"])
    n_reviews = len(tables["reviews"])

    spec = [
        # (table_key, column, human-readable semantics, denominator)
        ("orders", "order_purchase_timestamp",
         "purchase placed",       n_orders),
        ("orders", "order_approved_at",
         "payment approved",      n_orders),
        ("orders", "order_delivered_carrier_date",
         "handed to carrier",     n_orders),
        ("orders", "order_delivered_customer_date",
         "delivered to customer", n_orders),
        ("orders", "order_estimated_delivery_date",
         "promised date",         n_orders),
        ("order_items", "shipping_limit_date",
         "seller cut-off",        n_items),
        ("reviews", "review_creation_date",
         "review submitted",      n_reviews),
        ("reviews", "review_answer_timestamp",
         "seller / Olist reply",  n_reviews),
    ]

    rows = []
    for table_key, col, semantics, denom in spec:
        n_non_null = int(tables[table_key][col].notna().sum())
        coverage   = 100.0 * n_non_null / denom
        rows.append(CoverageRow(
            column=col, semantics=semantics,
            coverage_pct=coverage,
            n_non_null=n_non_null, n_total=denom,
        ))
    return rows


# ------------------------------------------------------------
# Interval-edge counts (Table 2)
# ------------------------------------------------------------

@dataclass(frozen=True)
class IntervalEdge:
    edge_type: str        # canonical name in the paper
    t_start_col: str
    t_end_col:   str
    n_both_present: int   # both endpoints non-null
    n_monotonic:    int   # both non-null AND t_end >= t_start
    n_anomalous:    int   # both non-null AND t_end <  t_start


def _count_interval(
    df: pd.DataFrame, t_start: str, t_end: str
) -> tuple[int, int, int]:
    """Counts for a single interval defined by two timestamp columns."""
    mask_both = df[t_start].notna() & df[t_end].notna()
    n_both = int(mask_both.sum())

    sub = df.loc[mask_both, [t_start, t_end]]
    mono = (sub[t_end] >= sub[t_start])
    n_mono = int(mono.sum())
    n_anom = n_both - n_mono
    return n_both, n_mono, n_anom


def compute_interval_edges(
    tables: dict[str, pd.DataFrame],
) -> list[IntervalEdge]:
    """Materialised interval counts (Section 4.2.1 of the paper).

    The four order-lifecycle intervals come from the orders table;
    `reviewed_at` requires a join with the reviews table on order_id.
    """
    orders = tables["orders"]

    # 3 order-internal intervals: all five timestamps live in `orders`
    order_specs = [
        ("approved_at",
         "order_purchase_timestamp",        "order_approved_at"),
        ("shipped_at",
         "order_approved_at",               "order_delivered_carrier_date"),
        ("delivered_at",
         "order_delivered_carrier_date",    "order_delivered_customer_date"),
    ]

    edges: list[IntervalEdge] = []
    for edge_type, t0, t1 in order_specs:
        n_both, n_mono, n_anom = _count_interval(orders, t0, t1)
        edges.append(IntervalEdge(
            edge_type=edge_type, t_start_col=t0, t_end_col=t1,
            n_both_present=n_both,
            n_monotonic=n_mono, n_anomalous=n_anom,
        ))

    # reviewed_at: needs the join with `reviews` on order_id.
    # One order can in principle have multiple reviews (the prior
    # paper's data audit found 814 duplicate review_ids). We
    # therefore count *review* rows, not orders.
    reviews = tables["reviews"]
    joined = reviews.merge(
        orders[["order_id", "order_delivered_customer_date"]],
        on="order_id", how="left",
    )
    n_both, n_mono, n_anom = _count_interval(
        joined, "order_delivered_customer_date", "review_creation_date"
    )
    edges.append(IntervalEdge(
        edge_type="reviewed_at",
        t_start_col="order_delivered_customer_date",
        t_end_col="review_creation_date",
        n_both_present=n_both,
        n_monotonic=n_mono, n_anomalous=n_anom,
    ))
    return edges


# ------------------------------------------------------------
# Sanity checks against the prose
# ------------------------------------------------------------

@dataclass(frozen=True)
class SanityChecks:
    n_orders: int
    n_reviews: int
    purchase_min: pd.Timestamp
    purchase_max: pd.Timestamp
    window_days: int
    orders_per_day: float
    n_carrier_before_approval: int   # reported in Olist-QA: 1,359


def compute_sanity(tables: dict[str, pd.DataFrame]) -> SanityChecks:
    orders = tables["orders"]

    purchase = orders["order_purchase_timestamp"]
    pmin, pmax = purchase.min(), purchase.max()
    window_days = (pmax - pmin).days

    n_orders  = len(orders)
    n_reviews = len(tables["reviews"])
    rate = n_orders / window_days if window_days else 0.0

    # Carrier-before-approval anomaly — prior paper §4.3 says 1,359.
    mask = (
        orders["order_approved_at"].notna()
        & orders["order_delivered_carrier_date"].notna()
        & (orders["order_delivered_carrier_date"]
           < orders["order_approved_at"])
    )
    return SanityChecks(
        n_orders=n_orders, n_reviews=n_reviews,
        purchase_min=pmin, purchase_max=pmax,
        window_days=window_days, orders_per_day=rate,
        n_carrier_before_approval=int(mask.sum()),
    )


# ------------------------------------------------------------
# Reporting
# ------------------------------------------------------------

def print_human_report(
    coverage: list[CoverageRow],
    edges: list[IntervalEdge],
    sanity: SanityChecks,
) -> None:
    print("=" * 64)
    print("Data statistics  —  human-readable")
    print("=" * 64)

    print("\nSanity checks (expected values in parentheses):")
    print(f"  total orders                : {sanity.n_orders:>9,}"
          f"   (expected 99,441)")
    print(f"  total reviews               : {sanity.n_reviews:>9,}"
          f"   (expected 99,224)")
    print(f"  purchase-window start       : "
          f"{sanity.purchase_min.date()}   (expected 2016-09)")
    print(f"  purchase-window end         : "
          f"{sanity.purchase_max.date()}   (expected 2018-10)")
    print(f"  window length (days)        : {sanity.window_days:>9,}"
          f"   (expected 772)")
    print(f"  orders / day                : {sanity.orders_per_day:>9.1f}"
          f"   (expected ~129)")
    print(f"  carrier-before-approval anom: "
          f"{sanity.n_carrier_before_approval:>9,}"
          f"   (expected 1,359)")

    print("\nTable 1 — Timestamp coverage:")
    print(f"  {'column':<35} {'semantics':<22} {'coverage':>10}")
    for r in coverage:
        print(f"  {r.column:<35} {r.semantics:<22} "
              f"{r.coverage_pct:>9.1f}%  "
              f"({r.n_non_null:,} / {r.n_total:,})")

    print("\nTable 2 — Interval-edge counts:")
    print(f"  {'edge type':<14} {'#both present':>14} "
          f"{'#monotonic':>12} {'#anomalous':>12}")
    for e in edges:
        print(f"  {e.edge_type:<14} {e.n_both_present:>14,} "
              f"{e.n_monotonic:>12,} {e.n_anomalous:>12,}")
    print()


def print_latex_snippets(
    coverage: list[CoverageRow],
    edges: list[IntervalEdge],
) -> None:
    """Emit LaTeX rows for the coverage and interval-edge tables."""
    print("=" * 64)
    print("LaTeX table rows")
    print("=" * 64)

    # ----- Table 1: timestamp coverage ---------------------------------
    print("\n--- Table 1 (tab:timestamps) ---")
    row_label = {
        "order_purchase_timestamp":
            r"\texttt{order\_purchase\_timestamp}     & purchase placed     ",
        "order_approved_at":
            r"\texttt{order\_approved\_at}            & payment approved    ",
        "order_delivered_carrier_date":
            r"\texttt{order\_delivered\_carrier\_date}& handed to carrier   ",
        "order_delivered_customer_date":
            r"\texttt{order\_delivered\_customer\_date}& delivered to cust. ",
        "order_estimated_delivery_date":
            r"\texttt{order\_estimated\_delivery\_date}& promised date      ",
        "shipping_limit_date":
            r"\texttt{shipping\_limit\_date}          & seller cut-off      ",
        "review_creation_date":
            r"\texttt{review\_creation\_date}         & review submitted    ",
        "review_answer_timestamp":
            r"\texttt{review\_answer\_timestamp}      & seller/Olist reply  ",
    }
    for r in coverage:
        prefix = row_label.get(r.column, r"\texttt{" + r.column + r"} & ")
        print(f"  {prefix}& {r.coverage_pct:5.1f}\\% \\\\")

    # ----- Table 2: interval-edge counts -----------------------------
    print("\n--- Table 2 (tab:tkg-stats) ---")
    print(r"  \#\textsc{approved\_at} interval edges      "
          f"& {edges[0].n_monotonic:,} \\\\")
    print(r"  \#\textsc{shipped\_at} interval edges       "
          f"& {edges[1].n_monotonic:,} \\\\")
    print(r"  \#\textsc{delivered\_at} interval edges     "
          f"& {edges[2].n_monotonic:,} \\\\")
    print(r"  \#\textsc{reviewed\_at} interval edges      "
          f"& {edges[3].n_monotonic:,} \\\\")
    print()
    print("Note: counts are *monotonic* intervals (t_end >= t_start).")
    print("If you prefer raw both-present counts (incl. anomalies):")
    for e in edges:
        print(f"  {e.edge_type:<14}: both={e.n_both_present:>6,}  "
              f"mono={e.n_monotonic:>6,}  anom={e.n_anomalous:>5,}")

    print()
    print("Index memory and summary lengths are reported by")
    print("build_temporal_index.py and time_hierarchy.py.")
    print()


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[2],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Directory containing the Olist CSVs "
             "(default: ../data relative to this script)",
    )
    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"ERROR: data directory not found: {args.data_dir}")
        print("Pass --data-dir explicitly or place CSVs under ../data/")
        return 1

    print(f"Loading Olist CSVs from {args.data_dir} ...")
    tables = load_olist_minimal(args.data_dir)

    coverage = compute_coverage(tables)
    edges    = compute_interval_edges(tables)
    sanity   = compute_sanity(tables)

    print_human_report(coverage, edges, sanity)
    print_latex_snippets(coverage, edges)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
