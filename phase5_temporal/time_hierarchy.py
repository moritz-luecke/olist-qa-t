"""
============================================================
Phase 5.3 — Bi-level Time Hierarchy + Time-Bucket Summaries
============================================================

Builds the time-axis structure that System D (community-temporal)
retrieves over. Orthogonal to the entity graph: a small hierarchy
of time buckets (year → quarter → month), each carrying a
population-level statistics block computed directly from the Olist
CSVs, plus a short LLM-written prose summary for embedding/retrieval.

Why CSV-derived stats (not KG-derived)
--------------------------------------
The pilot gold answers were computed from the Olist CSVs via pandas
(build_temporal_questions.py). Computing bucket stats from the same
source guarantees System D's evidence and the gold answers agree by
construction — no drift between "what the system sees" and "what is
correct". The KG pickle stores the same data, but going back to the
CSVs keeps the aggregation path identical to the gold path.

Hybrid summary design (decided 2026-05-28)
------------------------------------------
Each bucket stores BOTH:
  - `stats`: an exact, pandas-computed statistics dict (the numbers)
  - `summary`: a 2-4 sentence LLM prose paraphrase (for embedding /
    retrieval matching only)
  - `stats_block`: the rendered exact-number text that goes into the
    generation context

The exact numbers (not the prose) are what the System D retriever
feeds to the generator. This is the whole point: System D answers
aggregation questions from PRE-COMPUTED population aggregates, so it
cannot commit the sample-mean miscalibration that sinks Vanilla and
limits System C. The prose is only there so a question embedding can
match the right bucket.

Bucket statistics (cover all pilot question types except top-k
seller, which needs IDs unsuited to a prose summary):
  - volume:    order count, total GMV (sum payment_value)
  - latencies: mean purchase->delivery, approval-lag, carrier-lag,
               delivery-lag (monotonic-filtered, §4.1)
  - reviews:   mean review score, score histogram
  - payments:  mean installments, payment-type share (top-3)
  - categories:top-5 by volume, top-5 by mean review score
  - geography: top-5 customer states by order count
  - prices:    mean item price, mean freight

Outputs
-------
  outputs/time_buckets.jsonl        — one bucket per line:
      {bucket_id, level, start, end, stats, stats_block, summary}
  outputs/time_bucket_embeds.npy    — embeddings of the prose
                                       summaries (built lazily by the
                                       System D retriever, not here)

Levels and counts (Olist window Sep 2016 - Oct 2018):
  year:    2016, 2017, 2018                          = 3
  quarter: 2016-Q3 .. 2018-Q4 (present quarters)     ~ 9
  month:   2016-09 .. 2018-10 (present months)       ~ 26
  total                                              ~ 38 buckets

Cost: ~38 buckets x ~$0.013 (gpt-4o-mini, ~200 tok out) ≈ $0.50.

Usage
-----
  python phase5_temporal/time_hierarchy.py                 # dry by default
  python phase5_temporal/time_hierarchy.py --no-dry-run    # real LLM calls
  python phase5_temporal/time_hierarchy.py --no-dry-run --levels quarter
============================================================
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "phase2_kg"))
sys.path.insert(0, str(_HERE.parent / "phase3_benchmark"))
sys.path.insert(0, str(_HERE.parent / "phase4_retrieval"))

from build_pilot_questions import load_olist  # noqa: E402
# Reuse the exact same quarter/month/monotonic helpers the gold
# answers were built with, so bucket stats match gold by construction.
from build_temporal_questions import (  # noqa: E402
    add_quarter_col, add_month_col, filter_monotonic_delivery,
)


# ----------------------------------------------------------------
# LLM prose-summary prompt (parallel to summarize_communities.py)
# ----------------------------------------------------------------

SUMMARY_SYSTEM = "You write concise, factual summaries. No preamble."

SUMMARY_PROMPT = """\
You are summarising one time bucket of an e-commerce order dataset
(Olist, Brazil). Below is an exact statistical profile of all orders
whose purchase date falls in this bucket. Produce a concise 2-4
sentence summary capturing the period's volume, delivery performance,
review sentiment, and any salient category or payment pattern.

Ground every statement in the numbers provided; do not invent values.
This prose is used only for retrieval matching — exact figures are
stored separately — so prioritise readability over repeating every
number. Output a single paragraph.

== TIME BUCKET ==
{bucket_id}  (level={level}, {start} to {end})

== EXACT STATISTICS ==
{stats_block}
"""


# ----------------------------------------------------------------
# Bucket enumeration
# ----------------------------------------------------------------

def enumerate_buckets(orders: pd.DataFrame,
                      levels: list[str]) -> list[dict]:
    """List the (present) time buckets at the requested levels.

    A bucket is "present" if at least one order's purchase date
    falls in it. Returns dicts with bucket_id, level, start, end
    (ISO date strings, half-open [start, end)).
    """
    pur = pd.to_datetime(orders["order_purchase_timestamp"]).dropna()
    buckets: list[dict] = []

    if "year" in levels:
        for y in sorted(pur.dt.year.unique()):
            buckets.append({
                "bucket_id": f"{y}", "level": "year",
                "start": f"{y}-01-01", "end": f"{y + 1}-01-01",
            })

    if "quarter" in levels:
        yq = sorted(set(zip(pur.dt.year, pur.dt.quarter)))
        for y, q in yq:
            start_m = (q - 1) * 3 + 1
            end_y = y + (1 if q == 4 else 0)
            end_m = 1 if q == 4 else start_m + 3
            buckets.append({
                "bucket_id": f"{y}-Q{q}", "level": "quarter",
                "start": f"{y}-{start_m:02d}-01",
                "end": f"{end_y}-{end_m:02d}-01",
            })

    if "month" in levels:
        ym = sorted(set(zip(pur.dt.year, pur.dt.month)))
        for y, m in ym:
            end_y, end_m = (y, m + 1) if m < 12 else (y + 1, 1)
            buckets.append({
                "bucket_id": f"{y}-{m:02d}", "level": "month",
                "start": f"{y}-{m:02d}-01",
                "end": f"{end_y}-{end_m:02d}-01",
            })

    if "week" in levels:
        # ISO weeks (Monday-start), half-open [Monday, Monday+7d).
        # bucket_id "YYYY-Www" uses the ISO year/week so a week that
        # straddles a year boundary gets a single, unambiguous id.
        iso = pur.dt.isocalendar()          # columns: year, week, day
        yw = sorted(set(zip(iso["year"].astype(int),
                            iso["week"].astype(int))))
        for iy, iw in yw:
            start = date.fromisocalendar(iy, iw, 1)        # Monday
            end = start + timedelta(days=7)
            buckets.append({
                "bucket_id": f"{iy}-W{iw:02d}", "level": "week",
                "start": start.isoformat(),
                "end": end.isoformat(),
            })

    return buckets


# ----------------------------------------------------------------
# Per-bucket statistics (the exact numbers)
# ----------------------------------------------------------------

def _mean_or_none(s: pd.Series):
    s = s.dropna()
    return round(float(s.mean()), 2) if len(s) else None


def bucket_stats(tables: dict, start: str, end: str) -> dict:
    """Compute the exact population statistics for orders whose
    purchase date is in [start, end). All means are rounded to 2dp
    and use the same monotonic filter as the gold answers for the
    latency figures.
    """
    o = tables["orders"].copy()
    o["pur"] = pd.to_datetime(o["order_purchase_timestamp"])
    s_ts, e_ts = pd.Timestamp(start), pd.Timestamp(end)
    in_win = (o["pur"] >= s_ts) & (o["pur"] < e_ts)
    ob = o.loc[in_win]
    order_ids = set(ob["order_id"])
    n_orders = len(ob)

    stats: dict = {"order_count": int(n_orders)}
    if n_orders == 0:
        return stats

    # ---- Latencies (monotonic-filtered, matches gold) -------------
    od = filter_monotonic_delivery(o)
    od = od.loc[(od["pur"] >= s_ts) & (od["pur"] < e_ts)] \
        if "pur" in od.columns else od
    # filter_monotonic_delivery returns a fresh copy without "pur";
    # recompute window membership on it
    odp = pd.to_datetime(od["order_purchase_timestamp"])
    odw = od.loc[(odp >= s_ts) & (odp < e_ts)]
    if len(odw):
        p2d = (odw["order_delivered_customer_date"]
               - odw["order_purchase_timestamp"]).dt.total_seconds() / 86400
        appr = (odw["order_approved_at"]
                - odw["order_purchase_timestamp"]).dt.total_seconds() / 86400
        carr = (odw["order_delivered_carrier_date"]
                - odw["order_approved_at"]).dt.total_seconds() / 86400
        deliv = (odw["order_delivered_customer_date"]
                 - odw["order_delivered_carrier_date"]).dt.total_seconds() / 86400
        stats["mean_purchase_to_delivery_days"] = _mean_or_none(p2d)
        stats["mean_approval_lag_days"] = _mean_or_none(appr)
        stats["mean_carrier_lag_days"] = _mean_or_none(carr)
        stats["mean_delivery_lag_days"] = _mean_or_none(deliv)
        stats["n_complete_lifecycle"] = int(len(odw))

    # ---- GMV + payments -------------------------------------------
    pay = tables["payments"]
    payb = pay.loc[pay["order_id"].isin(order_ids)]
    if len(payb):
        stats["total_gmv"] = round(float(payb["payment_value"].sum()), 2)
        stats["mean_installments"] = _mean_or_none(payb["payment_installments"])
        n_pay = len(payb)
        share = (payb["payment_type"].value_counts() / n_pay * 100).round(1)
        stats["payment_type_share_pct"] = {
            k: float(v) for k, v in share.head(3).items()
        }

    # ---- Reviews ---------------------------------------------------
    rv = tables["reviews"]
    rvb = rv.loc[rv["order_id"].isin(order_ids)]
    if len(rvb):
        stats["mean_review_score"] = _mean_or_none(rvb["review_score"])
        stats["n_reviews"] = int(len(rvb))
        hist = rvb["review_score"].value_counts().sort_index()
        stats["review_score_hist"] = {int(k): int(v) for k, v in hist.items()}

    # ---- Items: price, freight, categories ------------------------
    it = tables["order_items"]
    itb = it.loc[it["order_id"].isin(order_ids)].copy()
    if len(itb):
        stats["mean_item_price"] = _mean_or_none(itb["price"])
        stats["mean_freight"] = _mean_or_none(itb["freight_value"])
        # categories need the product join
        pr = tables["products"]
        if "product_category_name_english" not in pr.columns:
            pr = pr.merge(tables["category_translation"],
                          on="product_category_name", how="left")
        itc = itb.merge(
            pr[["product_id", "product_category_name_english"]],
            on="product_id", how="left",
        )
        itc["cat"] = itc["product_category_name_english"].fillna("UNKNOWN")
        top_vol = itc["cat"].value_counts().head(5)
        stats["top5_categories_by_volume"] = {
            k: int(v) for k, v in top_vol.items()
        }
        # top-5 categories by mean review score (join items→orders→reviews)
        rev_join = (itc.merge(
            rvb[["order_id", "review_score"]], on="order_id", how="inner")
            if len(rvb) else None)
        if rev_join is not None and len(rev_join):
            cat_score = (rev_join.groupby("cat")["review_score"]
                         .agg(["mean", "count"]))
            cat_score = cat_score[cat_score["count"] >= 10]
            if len(cat_score):
                top_sc = cat_score.sort_values("mean", ascending=False).head(5)
                stats["top5_categories_by_review"] = {
                    k: round(float(v), 2) for k, v in top_sc["mean"].items()
                }

    # ---- Geography: top customer states ---------------------------
    cust = tables["customers"]
    ocs = ob.merge(cust[["customer_id", "customer_state"]],
                   on="customer_id", how="left")
    top_states = ocs["customer_state"].value_counts().head(5)
    stats["top5_customer_states"] = {k: int(v) for k, v in top_states.items()}

    return stats


# ----------------------------------------------------------------
# Render the exact-number stats block (goes into generation context)
# ----------------------------------------------------------------

def render_stats_block(bucket_id: str, stats: dict) -> str:
    L = [f"Time bucket {bucket_id}:"]
    L.append(f"  orders: {stats.get('order_count', 0):,}")
    if "total_gmv" in stats:
        L.append(f"  total GMV: R${stats['total_gmv']:,.2f}")
    if "mean_purchase_to_delivery_days" in stats:
        L.append(f"  mean purchase->delivery: "
                 f"{stats['mean_purchase_to_delivery_days']} days "
                 f"(n={stats.get('n_complete_lifecycle', 0):,} complete)")
        L.append(f"  mean approval lag: {stats['mean_approval_lag_days']} days")
        L.append(f"  mean carrier lag: {stats['mean_carrier_lag_days']} days")
        L.append(f"  mean delivery lag: {stats['mean_delivery_lag_days']} days")
    if "mean_review_score" in stats:
        L.append(f"  mean review score: {stats['mean_review_score']} "
                 f"(n={stats.get('n_reviews', 0):,})")
    if "mean_installments" in stats:
        L.append(f"  mean installments: {stats['mean_installments']}")
    if "payment_type_share_pct" in stats:
        shares = ", ".join(f"{k} {v}%"
                           for k, v in stats["payment_type_share_pct"].items())
        L.append(f"  payment mix: {shares}")
    if "mean_item_price" in stats:
        L.append(f"  mean item price: R${stats['mean_item_price']}")
        L.append(f"  mean freight: R${stats['mean_freight']}")
    if "top5_categories_by_volume" in stats:
        cats = ", ".join(f"{k} ({v})"
                         for k, v in stats["top5_categories_by_volume"].items())
        L.append(f"  top categories by volume: {cats}")
    if "top5_categories_by_review" in stats:
        cats = ", ".join(f"{k} ({v})"
                         for k, v in stats["top5_categories_by_review"].items())
        L.append(f"  top categories by review score: {cats}")
    if "top5_customer_states" in stats:
        st = ", ".join(f"{k} ({v})"
                       for k, v in stats["top5_customer_states"].items())
        L.append(f"  top customer states: {st}")
    return "\n".join(L)


# ----------------------------------------------------------------
# LLM summary
# ----------------------------------------------------------------

def summarize_bucket(llm, bucket: dict, stats_block: str) -> str:
    prompt = SUMMARY_PROMPT.format(
        bucket_id=bucket["bucket_id"], level=bucket["level"],
        start=bucket["start"], end=bucket["end"],
        stats_block=stats_block,
    )
    resp = llm.chat(system=SUMMARY_SYSTEM, user=prompt)
    return resp.get("text", "").strip()


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build the bi-level time hierarchy with per-bucket "
                    "stats and LLM summaries.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--data-dir", type=Path, default=Path("./data"))
    ap.add_argument("--out", type=Path,
                    default=Path("./outputs/time_buckets.jsonl"))
    ap.add_argument("--cache-dir", type=Path,
                    default=Path("./outputs/cache"))
    ap.add_argument("--levels", nargs="+",
                    default=["year", "quarter", "month"],
                    choices=["year", "quarter", "month", "week"])
    ap.add_argument("--dry-run", dest="dry_run", action="store_true")
    ap.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    ap.set_defaults(dry_run=True)
    args = ap.parse_args()

    print(f"Loading Olist CSVs from {args.data_dir} ...")
    tables = load_olist(args.data_dir)

    print(f"Enumerating buckets (levels={args.levels}) ...")
    buckets = enumerate_buckets(tables["orders"], args.levels)
    print(f"  {len(buckets)} buckets")

    # LLM (cached, dry-run-aware) — import here so --help needs no deps
    sys.path.insert(0, str(_HERE.parent / "phase4_retrieval"))
    from llm import ChatLLM
    llm = ChatLLM(args.cache_dir, dry_run=args.dry_run)

    mode = "DRY RUN (fake summaries)" if args.dry_run else "REAL (LLM calls)"
    est = len(buckets) * 0.013
    print(f"Mode: {mode}   est. cost ~${est:.2f}\n")

    # ---- Cache-poisoning guard (same logic as pilot runners) ----
    # If a previous dry-run wrote placeholder entries into the chat
    # cache, a subsequent --no-dry-run call will silently hit those
    # cached placeholders and produce 7-word fake summaries. We scan
    # the cache up front and refuse to proceed if any are found.
    if not args.dry_run:
        chat_cache_dir = args.cache_dir / "chat"
        if chat_cache_dir.exists():
            poisoned = []
            for entry in chat_cache_dir.glob("*.json"):
                try:
                    rec = json.loads(entry.read_text())
                    if rec.get("tokens", {}).get("dry_run"):
                        poisoned.append(entry.name)
                except (json.JSONDecodeError, OSError):
                    continue
            if poisoned:
                print(f"\nERROR: {len(poisoned)} dry-run entries in the "
                      f"chat cache at {chat_cache_dir}.", file=sys.stderr)
                print(f"  Example: {poisoned[0]}", file=sys.stderr)
                print(f"  Fix: rm -rf {chat_cache_dir}", file=sys.stderr)
                print(f"  Then re-run with --no-dry-run.", file=sys.stderr)
                return 3

    records = []
    for i, b in enumerate(buckets, 1):
        stats = bucket_stats(tables, b["start"], b["end"])
        stats_block = render_stats_block(b["bucket_id"], stats)
        summary = summarize_bucket(llm, b, stats_block)
        rec = {**b, "stats": stats, "stats_block": stats_block,
               "summary": summary}
        records.append(rec)
        print(f"  {i:>2}/{len(buckets)}  {b['bucket_id']:<10} "
              f"({b['level']}) orders={stats.get('order_count', 0):,}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print(f"\nWrote {len(records)} buckets to {args.out}")

    # Report the summary-length statistic (tokens per bucket summary)
    if not args.dry_run:
        tok = [len(r["summary"].split()) for r in records if r["summary"]]
        if tok:
            print(f"Summary length (words): mean={np.mean(tok):.0f}, "
                  f"max={max(tok)}  "
                  f"(approx tokens: mean={np.mean(tok)*1.3:.0f}, "
                  f"max={max(tok)*1.3:.0f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
