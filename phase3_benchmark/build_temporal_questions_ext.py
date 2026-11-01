"""
============================================================
Phase 3.3 — Scale temporal benchmark 30 → 90 questions
============================================================

Adds 20 NEW questions per category (q11–30) on top of the 30 pilot
questions, for a 240-question Olist-QA-T (150 prior + 90 temporal).
The 30 pilot questions are imported UNCHANGED from
build_temporal_questions.py (same qids, same gold) so the pilot
results stay valid; this module only appends q11–30.

Sub-type distribution at 30/category (decided 2026-06-03, scaled
≈ proportionally from the pilot, slightly favouring the core
sub-types; before_after kept at ~20% to characterise the
sub-bucket-window failure mode without over-weighting it):

  Trend (30)  : 18 aggregation_over_time + 12 comparison_over_time
  Drift (30)  : 18 category_drift        + 12 topk_change
  Latency(30) : 14 inter_event_lag + 10 lag_comparison + 6 before_after

All 90 questions are closed-form (number / dict / list), SQL- and
pandas-validated against the Olist CSVs (the pilot's dual-validation
approach). Open-form + LLM-judge is layered at the headline
experiment, not baked into the gold benchmark.

Data-safety (from data_profile.json):
  - Usable windows only: quarters 2017-Q1 … 2018-Q3, months
    2017-01 … 2018-09 (2016-Q3 and 2018-Q4 each have 4 orders —
    NEVER used as a window).
  - State filters use only large states (SP, RJ, MG, RS, PR, …);
    category questions compute top-k from the data (never hard-code
    a category name), with min-sample guards.
  - Review-score questions carry score_tol=0.1 (tight absolute
    tolerance on the 1–5 scale; honoured by the fixed scorer).

Output:
  outputs/temporal_questions_90.jsonl   — 90 questions (30 + 60)
  benchmark/external_events.jsonl       — ~10 event anchors (§4.3)

Usage:
  python phase3_benchmark/build_temporal_questions_ext.py
  python phase3_benchmark/build_temporal_questions_ext.py \\
      --data-dir ./data --out ./outputs/temporal_questions_90.jsonl
============================================================
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "phase2_kg"))

# Reuse the pilot builder wholesale: helpers, registry, the 30 pilot
# question generators, and the curated event dict.
from build_temporal_questions import (          # noqa: E402
    QUESTIONS, register, EXTERNAL_EVENTS,
    add_quarter_col, add_month_col, filter_monotonic_delivery,
    build_trend, build_drift, build_latency,
)
from build_pilot_questions import load_olist     # noqa: E402


# A few more event anchors so the curated list reaches ~10 (§4.3).
# All fall inside the data-rich part of the window (Sep-2016 … Sep-2018).
EXTERNAL_EVENTS.update({
    "mothers18":   {"name": "Mother's Day BR 2018",     "date": "2018-05-13"},
    "namorados17": {"name": "Dia dos Namorados 2017",   "date": "2017-06-12"},
    "namorados18": {"name": "Dia dos Namorados 2018",   "date": "2018-06-12"},
})


# ============================================================
# Small data-safe helpers (return rounded, non-empty results)
# ============================================================

def _r2(x) -> float:
    return round(float(x), 2)


def _lag_days(o: pd.DataFrame, start: str, end: str) -> pd.Series:
    return (o[end] - o[start]).dt.total_seconds() / 86400.0


def _period_dict(df: pd.DataFrame, period_col: str, periods: list[str],
                 value: str | None, agg: str) -> dict:
    """Group rows whose period is in `periods` (explicit list — NOT a
    prefix, so sparse tail buckets like 2018-Q4 / 2018-10 are never
    included); aggregate `value` (or row count if None) per period."""
    sub = df[df[period_col].astype(str).isin(periods)]
    if value is None:
        out = sub.groupby(period_col).size().astype(int)
    else:
        out = sub.groupby(period_col)[value].agg(agg)
    out = out.sort_index()
    if agg in ("mean",) and value is not None:
        out = out.round(2)
    return out.to_dict()


def _two_window(df: pd.DataFrame, period_col: str,
                wa: str, wb: str, value: str | None, agg: str) -> dict:
    """Aggregate `value` (or count) over exactly two period labels."""
    def one(w):
        sub = df[df[period_col] == w]
        if value is None:
            return int(len(sub))
        v = getattr(sub[value], agg)()
        return _r2(v)
    return {wa: one(wa), wb: one(wb)}


def _half_col(df: pd.DataFrame, date_col: str, out="half") -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df[date_col])
    df[out] = ts.dt.year.astype(str) + "-H" + ((ts.dt.month > 6).astype(int) + 1).astype(str)
    return df


def _topk_two_window(df: pd.DataFrame, period_col: str, wa: str, wb: str,
                     key_col: str, k: int, value: str | None,
                     agg: str = "size") -> dict:
    """{wa: {key: val (top-k)}, wb: {...}} — used for topk_change."""
    def one(w):
        sub = df[df[period_col] == w]
        if value is None:
            s = sub.groupby(key_col).size()
        else:
            s = sub.groupby(key_col)[value].agg(agg)
        s = s.sort_values(ascending=False).head(k)
        return {str(k_): (int(v) if value is None and agg == "size"
                          else _r2(v)) for k_, v in s.items()}
    return {wa: one(wa), wb: one(wb)}


def _share_two_window(df: pd.DataFrame, period_col: str, wa: str, wb: str,
                      key_col: str, k: int) -> dict:
    """Top-k by percentage share of rows, per window (payment-type style)."""
    def one(w):
        sub = df[df[period_col] == w]
        s = 100.0 * sub.groupby(key_col).size() / max(len(sub), 1)
        s = s.sort_values(ascending=False).head(k)
        return {str(k_): _r2(v) for k_, v in s.items()}
    return {wa: one(wa), wb: one(wb)}


def _category_drift(oi: pd.DataFrame, wa: str, wb: str, value: str | None,
                    agg: str, min_n: int, top: int = 3,
                    improvement: bool = False) -> dict:
    """Top-`top` categories by change (wa→wb) of an aggregate, restricted
    to categories with ≥ min_n rows in BOTH windows. Positive = wa−wb
    (a drop) unless improvement=True (wb−wa)."""
    counts = oi.groupby(["category", "quarter"]).size().unstack(fill_value=0)
    stable = counts[(counts.get(wa, 0) >= min_n)
                    & (counts.get(wb, 0) >= min_n)].index.tolist()
    sub = oi[oi["category"].isin(stable)]
    if value is None:
        agg_tbl = sub.groupby(["category", "quarter"]).size().unstack()
    else:
        agg_tbl = sub.groupby(["category", "quarter"])[value].agg(agg).unstack()
    if wa not in agg_tbl or wb not in agg_tbl:
        return {}
    change = (agg_tbl[wb] - agg_tbl[wa]) if improvement else (agg_tbl[wa] - agg_tbl[wb])
    change = change.dropna().sort_values(ascending=False)
    return change.head(top).round(2).to_dict()


# ============================================================
# TREND ext — 12 aggregation_over_time + 8 comparison_over_time
# ============================================================

def build_trend_ext(t: dict[str, pd.DataFrame]) -> None:
    o, it, rv, pay = t["orders"], t["order_items"], t["reviews"], t["payments"]

    oq = add_quarter_col(o, "order_purchase_timestamp")
    om = add_month_col(o, "order_purchase_timestamp")
    od = filter_monotonic_delivery(o)
    od = add_quarter_col(od, "order_purchase_timestamp")
    od["delivery_days"] = _lag_days(od, "order_purchase_timestamp",
                                    "order_delivered_customer_date")
    od["approval_lag"] = _lag_days(od, "order_purchase_timestamp",
                                   "order_approved_at")

    # order_items joined to purchase quarter/month
    iq = it.merge(oq[["order_id", "quarter"]], on="order_id")
    im = it.merge(om[["order_id", "month"]], on="order_id")
    # payments joined to purchase quarter/month
    pq = pay.merge(oq[["order_id", "quarter"]], on="order_id")
    # reviews joined to purchase quarter
    rq = rv.merge(oq[["order_id", "quarter"]], on="order_id")

    Q17 = ["2017-Q1", "2017-Q2", "2017-Q3", "2017-Q4"]
    Q18 = ["2018-Q1", "2018-Q2", "2018-Q3"]
    M17 = [f"2017-{m:02d}" for m in range(1, 13)]
    M18 = [f"2018-{m:02d}" for m in range(1, 10)]

    def reg(qid, sub, q, gold, gtype, anchors, trange,
            hops=2, tables=("orders",), tol=None, notes=None):
        rec = {
            "qid": qid, "category": "trend", "sub_type": sub,
            "min_hops": hops, "tables_involved": list(tables),
            "question": q, "sql_query": "-- see pandas_code",
            "pandas_code": f"see build_trend_ext {qid}",
            "gold_answer_type": gtype, "gold_answer": gold,
            "time_anchors": anchors, "time_range": trange,
            "event_anchor": None, "exclusion_flags": [],
        }
        if tol is not None:
            rec["score_tol"] = tol
        if notes:
            rec["notes"] = notes
        register(rec)

    # ---- 12 aggregation_over_time (trend_11 … trend_22) ----------
    reg("trend_11", "aggregation_over_time",
        "How many orders were placed per quarter in 2018 (Q1–Q3)?",
        _period_dict(oq, "quarter", Q18, None, "size"), "dict",
        Q18, {"start": "2018-01-01", "end": "2018-10-01"})

    reg("trend_12", "aggregation_over_time",
        "What was the average item price (in BRL) per quarter of 2017?",
        _period_dict(iq, "quarter", Q17, "price", "mean"), "dict",
        Q17, {"start": "2017-01-01", "end": "2018-01-01"},
        tables=("orders", "order_items"))

    reg("trend_13", "aggregation_over_time",
        "What was the average freight value per order line per quarter "
        "in 2018 (Q1–Q3)?",
        _period_dict(iq, "quarter", Q18, "freight_value", "mean"), "dict",
        Q18, {"start": "2018-01-01", "end": "2018-10-01"},
        tables=("orders", "order_items"))

    reg("trend_14", "aggregation_over_time",
        "What was the total payment value (GMV, in BRL) per quarter in "
        "2018 (Q1–Q3)?",
        _period_dict(pq, "quarter", Q18, "payment_value", "sum"), "dict",
        Q18, {"start": "2018-01-01", "end": "2018-10-01"},
        tables=("orders", "payments"))

    reg("trend_15", "aggregation_over_time",
        "What was the average review score per quarter in 2018 (Q1–Q3)?",
        _period_dict(rq, "quarter", Q18, "review_score", "mean"), "dict",
        Q18, {"start": "2018-01-01", "end": "2018-10-01"},
        tables=("orders", "reviews"), tol=0.1)

    reg("trend_16", "aggregation_over_time",
        "What was the average number of payment installments per quarter "
        "in 2018 (Q1–Q3)?",
        _period_dict(pq, "quarter", Q18, "payment_installments", "mean"),
        "dict", Q18, {"start": "2018-01-01", "end": "2018-10-01"},
        tables=("orders", "payments"))

    reg("trend_17", "aggregation_over_time",
        "What was the average purchase-to-delivery time (in days) per "
        "quarter in 2018 (Q1–Q3), for orders with a complete and "
        "monotonic delivery lifecycle?",
        _period_dict(od, "quarter", Q18, "delivery_days", "mean"), "dict",
        Q18, {"start": "2018-01-01", "end": "2018-10-01"},
        notes="nonmonotonic intervals excluded")

    reg("trend_18", "aggregation_over_time",
        "How many orders were placed per month in 2018 (January–September)?",
        _period_dict(om, "month", M18, None, "size"), "dict",
        M18, {"start": "2018-01-01", "end": "2018-10-01"})

    reg("trend_19", "aggregation_over_time",
        "What was the average item price (in BRL) per month in 2017?",
        _period_dict(im, "month", M17, "price", "mean"), "dict",
        M17, {"start": "2017-01-01", "end": "2018-01-01"},
        tables=("orders", "order_items"))

    reg("trend_20", "aggregation_over_time",
        "What was the average purchase-to-approval lag (in days) per "
        "quarter of 2017, for orders with a monotonic lifecycle?",
        _period_dict(od, "quarter", Q17, "approval_lag", "mean"), "dict",
        Q17, {"start": "2017-01-01", "end": "2018-01-01"},
        notes="nonmonotonic intervals excluded")

    reg("trend_21", "aggregation_over_time",
        "What was the average freight value per order line per month in "
        "2018 (January–September)?",
        _period_dict(im, "month", M18, "freight_value", "mean"), "dict",
        M18, {"start": "2018-01-01", "end": "2018-10-01"},
        tables=("orders", "order_items"))

    # full-window volume trend (7 quarters), tests a long horizon
    all_q = (_period_dict(oq, "quarter", Q17, None, "size")
             | _period_dict(oq, "quarter", Q18, None, "size"))
    all_q = {k: v for k, v in sorted(all_q.items()) if k in Q17 + Q18}
    reg("trend_22", "aggregation_over_time",
        "How many orders were placed in each quarter from 2017-Q1 through "
        "2018-Q3?",
        all_q, "dict", Q17 + Q18,
        {"start": "2017-01-01", "end": "2018-10-01"})

    # ---- 8 comparison_over_time (trend_23 … trend_30) ------------
    reg("trend_23", "comparison_over_time",
        "How did the average purchase-to-delivery time (in days) change "
        "from 2018-Q1 to 2018-Q3 (monotonic orders only)?",
        _two_window(od, "quarter", "2018-Q1", "2018-Q3", "delivery_days",
                    "mean"), "dict", ["2018-Q1", "2018-Q3"], None,
        notes="nonmonotonic intervals excluded")

    reg("trend_24", "comparison_over_time",
        "Compare the total number of orders placed in 2017-Q4 versus "
        "2018-Q1.",
        _two_window(oq, "quarter", "2017-Q4", "2018-Q1", None, "size"),
        "dict", ["2017-Q4", "2018-Q1"], None)

    reg("trend_25", "comparison_over_time",
        "How did the average item price (in BRL) change from 2017-Q1 to "
        "2017-Q4?",
        _two_window(iq, "quarter", "2017-Q1", "2017-Q4", "price", "mean"),
        "dict", ["2017-Q1", "2017-Q4"], None,
        tables=("orders", "order_items"))

    reg("trend_26", "comparison_over_time",
        "Compare the average freight value per order line in 2017-Q2 "
        "versus 2018-Q2.",
        _two_window(iq, "quarter", "2017-Q2", "2018-Q2", "freight_value",
                    "mean"), "dict", ["2017-Q2", "2018-Q2"], None,
        tables=("orders", "order_items"))

    reg("trend_27", "comparison_over_time",
        "Compare the total payment value (GMV, in BRL) in 2017-Q4 versus "
        "2018-Q2.",
        _two_window(pq, "quarter", "2017-Q4", "2018-Q2", "payment_value",
                    "sum"), "dict", ["2017-Q4", "2018-Q2"], None,
        tables=("orders", "payments"))

    reg("trend_28", "comparison_over_time",
        "How did the average number of payment installments change from "
        "2017-Q1 to 2018-Q1?",
        _two_window(pq, "quarter", "2017-Q1", "2018-Q1",
                    "payment_installments", "mean"), "dict",
        ["2017-Q1", "2018-Q1"], None, tables=("orders", "payments"))

    # half-year review comparison (uses H1/H2 anchors, tight tol)
    rh = rv.merge(_half_col(o, "order_purchase_timestamp")[["order_id", "half"]],
                  on="order_id")
    reg("trend_29", "comparison_over_time",
        "How did the average review score change from the first half "
        "(Jan–Jun) to the second half (Jul–Dec) of 2017?",
        _two_window(rh, "half", "2017-H1", "2017-H2", "review_score", "mean"),
        "dict", ["2017-H1", "2017-H2"], {"start": "2017-01-01",
                                         "end": "2018-01-01"},
        tables=("orders", "reviews"), tol=0.1)

    # year-over-year delivery (year aggregates; quarter anchors cover both)
    reg("trend_30", "comparison_over_time",
        "How did the average purchase-to-delivery time (in days) change "
        "from 2017 to 2018 (monotonic orders only)?",
        {"2017": _r2(od.loc[od.quarter.isin(Q17), "delivery_days"].mean()),
         "2018": _r2(od.loc[od.quarter.isin(Q18), "delivery_days"].mean())},
        "dict", Q17 + Q18, None,
        notes="year aggregates; nonmonotonic excluded")


# ============================================================
# DRIFT ext — 12 category_drift + 8 topk_change
# ============================================================

def build_drift_ext(t: dict[str, pd.DataFrame]) -> None:
    o, it, rv, pay, cu, pr = (t["orders"], t["order_items"], t["reviews"],
                              t["payments"], t["customers"], t["products"])

    oq = add_quarter_col(o, "order_purchase_timestamp")

    # items × purchase-quarter × category
    oi = it.merge(oq[["order_id", "quarter"]], on="order_id")
    oi = oi.merge(pr[["product_id", "product_category_name_english"]],
                  on="product_id", how="left")
    oi["category"] = oi["product_category_name_english"].fillna("UNKNOWN")
    # Exclude untranslated products from category questions: "UNKNOWN" is
    # not a real category and pollutes category_drift / category-topk gold
    # (it was the #1 price-riser in drift_15/21). Benchmark hygiene.
    oi = oi[oi["category"] != "UNKNOWN"]

    # reviews × purchase-quarter × category
    rc = rv.merge(it[["order_id", "product_id"]], on="order_id")
    rc = rc.merge(oq[["order_id", "quarter"]], on="order_id")
    rc = rc.merge(pr[["product_id", "product_category_name_english"]],
                  on="product_id", how="left")
    rc["category"] = rc["product_category_name_english"].fillna("UNKNOWN")
    rc = rc[rc["category"] != "UNKNOWN"]          # same hygiene as oi

    # orders × customer state × quarter
    oc = oq.merge(cu[["customer_id", "customer_state"]], on="customer_id",
                  how="left")
    # payments × quarter
    pq = pay.merge(oq[["order_id", "quarter"]], on="order_id")
    # items × category × payment value not needed; GMV-by-category via price
    def reg(qid, sub, q, gold, gtype, anchors, hops=3,
            tables=("orders", "order_items", "products"), notes=None):
        rec = {"qid": qid, "category": "drift", "sub_type": sub,
               "min_hops": hops, "tables_involved": list(tables),
               "question": q, "sql_query": "-- see pandas_code",
               "pandas_code": f"see build_drift_ext {qid}",
               "gold_answer_type": gtype, "gold_answer": gold,
               "time_anchors": anchors, "time_range": None,
               "event_anchor": None,
               "exclusion_flags": ["min_sample_per_window"]}
        if notes:
            rec["notes"] = notes
        register(rec)

    # ---- 12 category_drift (drift_11 … drift_22) -----------------
    reg("drift_11", "category_drift",
        "Which 3 product categories saw the steepest DROP in order-item "
        "volume from 2018-Q1 to 2018-Q2? (≥50 items per quarter.)",
        _category_drift(oi, "2018-Q1", "2018-Q2", None, "size", 50),
        "dict", ["2018-Q1", "2018-Q2"], notes="positive = volume fell")

    reg("drift_12", "category_drift",
        "Which 3 product categories grew the most in order-item volume "
        "from 2017-Q3 to 2017-Q4? (≥50 items per quarter.)",
        _category_drift(oi, "2017-Q3", "2017-Q4", None, "size", 50,
                        improvement=True),
        "dict", ["2017-Q3", "2017-Q4"], notes="positive = volume grew")

    reg("drift_13", "category_drift",
        "Which 3 product categories saw the steepest DROP in average "
        "review score from 2018-Q1 to 2018-Q2? (≥30 reviews per quarter.)",
        _category_drift(rc, "2018-Q1", "2018-Q2", "review_score", "mean", 30),
        "dict", ["2018-Q1", "2018-Q2"],
        tables=("orders", "order_items", "products", "reviews"),
        notes="positive = score fell")

    reg("drift_14", "category_drift",
        "Which 3 product categories improved the most in average review "
        "score from 2017-Q1 to 2017-Q2? (≥30 reviews per quarter.)",
        _category_drift(rc, "2017-Q1", "2017-Q2", "review_score", "mean", 30,
                        improvement=True),
        "dict", ["2017-Q1", "2017-Q2"],
        tables=("orders", "order_items", "products", "reviews"),
        notes="positive = score improved")

    reg("drift_15", "category_drift",
        "Which 3 product categories saw the largest increase in average "
        "item price from 2017-Q1 to 2017-Q4? (≥50 items per quarter.)",
        _category_drift(oi, "2017-Q1", "2017-Q4", "price", "mean", 50,
                        improvement=True),
        "dict", ["2017-Q1", "2017-Q4"], notes="positive = price rose")

    reg("drift_16", "category_drift",
        "Which 3 product categories saw the largest DECREASE in average "
        "item price from 2018-Q1 to 2018-Q3? (≥50 items per quarter.)",
        _category_drift(oi, "2018-Q1", "2018-Q3", "price", "mean", 50),
        "dict", ["2018-Q1", "2018-Q3"], notes="positive = price fell")

    reg("drift_17", "category_drift",
        "Which 3 product categories saw the largest increase in average "
        "freight value from 2017-Q3 to 2017-Q4? (≥50 items per quarter.)",
        _category_drift(oi, "2017-Q3", "2017-Q4", "freight_value", "mean", 50,
                        improvement=True),
        "dict", ["2017-Q3", "2017-Q4"], notes="positive = freight rose")

    reg("drift_18", "category_drift",
        "Which 3 product categories grew the most in order-item volume "
        "from 2017-Q4 to 2018-Q1? (≥50 items per quarter.)",
        _category_drift(oi, "2017-Q4", "2018-Q1", None, "size", 50,
                        improvement=True),
        "dict", ["2017-Q4", "2018-Q1"], notes="positive = volume grew")

    reg("drift_19", "category_drift",
        "Which 3 product categories saw the steepest DROP in average "
        "review score from 2017-Q2 to 2017-Q3? (≥30 reviews per quarter.)",
        _category_drift(rc, "2017-Q2", "2017-Q3", "review_score", "mean", 30),
        "dict", ["2017-Q2", "2017-Q3"],
        tables=("orders", "order_items", "products", "reviews"),
        notes="positive = score fell")

    reg("drift_20", "category_drift",
        "Which 3 product categories saw the steepest DROP in order-item "
        "volume from 2018-Q2 to 2018-Q3? (≥50 items per quarter.)",
        _category_drift(oi, "2018-Q2", "2018-Q3", None, "size", 50),
        "dict", ["2018-Q2", "2018-Q3"], notes="positive = volume fell")

    reg("drift_21", "category_drift",
        "Which 3 product categories saw the largest increase in average "
        "item price from 2017-Q2 to 2017-Q4? (≥50 items per quarter.)",
        _category_drift(oi, "2017-Q2", "2017-Q4", "price", "mean", 50,
                        improvement=True),
        "dict", ["2017-Q2", "2017-Q4"], notes="positive = price rose")

    reg("drift_22", "category_drift",
        "Which 3 product categories saw the steepest DROP in average "
        "review score from 2017-Q4 to 2018-Q1? (≥30 reviews per quarter.)",
        _category_drift(rc, "2017-Q4", "2018-Q1", "review_score", "mean", 30),
        "dict", ["2017-Q4", "2018-Q1"],
        tables=("orders", "order_items", "products", "reviews"),
        notes="positive = score fell")

    # ---- 8 topk_change (drift_23 … drift_30) ---------------------
    reg("drift_23", "topk_change",
        "What were the top 5 customer states by order count in 2018-Q1 "
        "versus 2018-Q3?",
        _topk_two_window(oc, "quarter", "2018-Q1", "2018-Q3",
                         "customer_state", 5, None),
        "dict", ["2018-Q1", "2018-Q3"], tables=("orders", "customers"))

    reg("drift_24", "topk_change",
        "What were the top 3 payment methods by share of payment rows in "
        "2018-Q1 versus 2018-Q3?",
        _share_two_window(pq, "quarter", "2018-Q1", "2018-Q3",
                          "payment_type", 3),
        "dict", ["2018-Q1", "2018-Q3"], tables=("orders", "payments"),
        notes="values are percentages")

    reg("drift_25", "topk_change",
        "What were the top 5 product categories by order-item count in "
        "2017-Q4 versus 2018-Q1?",
        _topk_two_window(oi, "quarter", "2017-Q4", "2018-Q1",
                         "category", 5, None),
        "dict", ["2017-Q4", "2018-Q1"])

    reg("drift_26", "topk_change",
        "What were the top 5 product categories by total item price (GMV "
        "proxy) in 2017-Q4 versus 2018-Q1?",
        _topk_two_window(oi, "quarter", "2017-Q4", "2018-Q1",
                         "category", 5, "price", "sum"),
        "dict", ["2017-Q4", "2018-Q1"], notes="values are summed item price")

    reg("drift_27", "topk_change",
        "What were the top 5 customer states by order count in 2017-Q1 "
        "versus 2017-Q4?",
        _topk_two_window(oc, "quarter", "2017-Q1", "2017-Q4",
                         "customer_state", 5, None),
        "dict", ["2017-Q1", "2017-Q4"], tables=("orders", "customers"))

    reg("drift_28", "topk_change",
        "What were the top 3 payment methods by share of payment rows in "
        "2017-Q1 versus 2017-Q4?",
        _share_two_window(pq, "quarter", "2017-Q1", "2017-Q4",
                          "payment_type", 3),
        "dict", ["2017-Q1", "2017-Q4"], tables=("orders", "payments"),
        notes="values are percentages")

    reg("drift_29", "topk_change",
        "What were the top 5 product categories by order-item count in "
        "2018-Q1 versus 2018-Q2?",
        _topk_two_window(oi, "quarter", "2018-Q1", "2018-Q2",
                         "category", 5, None),
        "dict", ["2018-Q1", "2018-Q2"])

    reg("drift_30", "topk_change",
        "What were the top 5 customer states by order count in 2018-Q2 "
        "versus 2018-Q3?",
        _topk_two_window(oc, "quarter", "2018-Q2", "2018-Q3",
                         "customer_state", 5, None),
        "dict", ["2018-Q2", "2018-Q3"], tables=("orders", "customers"))


# ============================================================
# LATENCY ext — 9 inter_event_lag + 7 lag_comparison + 4 before_after
# ============================================================

def build_latency_ext(t: dict[str, pd.DataFrame]) -> None:
    o, cu = t["orders"], t["customers"]
    od = filter_monotonic_delivery(o)
    od = add_quarter_col(od, "order_purchase_timestamp")
    od["pdl"] = _lag_days(od, "order_purchase_timestamp",
                          "order_delivered_customer_date")   # purchase→deliv
    od["apl"] = _lag_days(od, "order_purchase_timestamp",
                          "order_approved_at")                # purchase→appr
    od["acl"] = _lag_days(od, "order_approved_at",
                          "order_delivered_carrier_date")     # appr→carrier
    od["cdl"] = _lag_days(od, "order_delivered_carrier_date",
                          "order_delivered_customer_date")    # carrier→deliv
    odc = od.merge(cu[["customer_id", "customer_state"]], on="customer_id",
                   how="left")

    def yr(df, periods, col):
        return _r2(df.loc[df.quarter.isin(periods), col].mean())

    def reg(qid, sub, q, gold, gtype, anchors, trange, hops=2,
            tables=("orders",), event=None, notes=None):
        rec = {"qid": qid, "category": "latency", "sub_type": sub,
               "min_hops": hops, "tables_involved": list(tables),
               "question": q, "sql_query": "-- see pandas_code",
               "pandas_code": f"see build_latency_ext {qid}",
               "gold_answer_type": gtype, "gold_answer": gold,
               "time_anchors": anchors, "time_range": trange,
               "event_anchor": event,
               "exclusion_flags": ["nonmonotonic_intervals"]}
        if notes:
            rec["notes"] = notes
        register(rec)

    Q17 = ["2017-Q1", "2017-Q2", "2017-Q3", "2017-Q4"]
    Q18 = ["2018-Q1", "2018-Q2", "2018-Q3"]

    # ---- 9 inter_event_lag (latency_11 … latency_19) -------------
    reg("latency_11", "inter_event_lag",
        "What was the average purchase-to-approval lag (in days) for "
        "orders placed during 2018 (monotonic orders only)?",
        yr(od, Q18, "apl"), "number", Q18,
        {"start": "2018-01-01", "end": "2018-10-01"})

    reg("latency_12", "inter_event_lag",
        "What was the average carrier-to-customer delivery lag (in days) "
        "for orders placed during 2018 (monotonic orders only)?",
        yr(od, Q18, "cdl"), "number", Q18,
        {"start": "2018-01-01", "end": "2018-10-01"})

    reg("latency_13", "inter_event_lag",
        "What was the average approval-to-carrier-handover lag (in days) "
        "for orders placed during 2018 (monotonic orders only)?",
        yr(od, Q18, "acl"), "number", Q18,
        {"start": "2018-01-01", "end": "2018-10-01"})

    reg("latency_14", "inter_event_lag",
        "What was the average purchase-to-delivery lag (in days) for "
        "orders placed during 2018 (monotonic orders only)?",
        yr(od, Q18, "pdl"), "number", Q18,
        {"start": "2018-01-01", "end": "2018-10-01"})

    reg("latency_15", "inter_event_lag",
        "What was the average purchase-to-delivery lag (in days) for "
        "orders from customers in Rio de Janeiro (state RJ) placed during "
        "2017 (monotonic orders only)?",
        _r2(odc.loc[odc.quarter.isin(Q17)
                    & (odc.customer_state == "RJ"), "pdl"].mean()),
        "number", Q17, {"start": "2017-01-01", "end": "2018-01-01"},
        tables=("orders", "customers"), notes="state-filtered (RJ)")

    reg("latency_16", "inter_event_lag",
        "What was the average purchase-to-delivery lag (in days) for "
        "orders from customers in Minas Gerais (state MG) placed during "
        "2017 (monotonic orders only)?",
        _r2(odc.loc[odc.quarter.isin(Q17)
                    & (odc.customer_state == "MG"), "pdl"].mean()),
        "number", Q17, {"start": "2017-01-01", "end": "2018-01-01"},
        tables=("orders", "customers"), notes="state-filtered (MG)")

    reg("latency_17", "inter_event_lag",
        "What was the average purchase-to-approval lag (in days) for "
        "orders placed in 2017-Q4 (monotonic orders only)?",
        _r2(od.loc[od.quarter == "2017-Q4", "apl"].mean()),
        "number", ["2017-Q4"], {"start": "2017-10-01", "end": "2018-01-01"})

    reg("latency_18", "inter_event_lag",
        "What was the average carrier-to-customer delivery lag (in days) "
        "for orders placed in 2018-Q1 (monotonic orders only)?",
        _r2(od.loc[od.quarter == "2018-Q1", "cdl"].mean()),
        "number", ["2018-Q1"], {"start": "2018-01-01", "end": "2018-04-01"})

    reg("latency_19", "inter_event_lag",
        "What was the average purchase-to-delivery lag (in days) for "
        "orders from customers in São Paulo (state SP) placed during "
        "2018 (monotonic orders only)?",
        _r2(odc.loc[odc.quarter.isin(Q18)
                    & (odc.customer_state == "SP"), "pdl"].mean()),
        "number", Q18, {"start": "2018-01-01", "end": "2018-10-01"},
        tables=("orders", "customers"), notes="state-filtered (SP)")

    # ---- 7 lag_comparison (latency_20 … latency_26) --------------
    reg("latency_20", "lag_comparison",
        "Compare the average purchase-to-delivery lag (in days) in "
        "2018-Q1 versus 2018-Q3 (monotonic orders only).",
        _two_window(od, "quarter", "2018-Q1", "2018-Q3", "pdl", "mean"),
        "dict", ["2018-Q1", "2018-Q3"], None)

    reg("latency_21", "lag_comparison",
        "Compare the average purchase-to-approval lag (in days) for "
        "orders placed in 2017 versus 2018 (monotonic orders only).",
        {"2017": yr(od, Q17, "apl"), "2018": yr(od, Q18, "apl")},
        "dict", Q17 + Q18, None, notes="year aggregates")

    reg("latency_22", "lag_comparison",
        "Compare the average purchase-to-delivery lag (in days) for "
        "customers in São Paulo (SP) versus Minas Gerais (MG), for orders "
        "placed during 2017 (monotonic orders only).",
        {"SP": _r2(odc.loc[odc.quarter.isin(Q17)
                           & (odc.customer_state == "SP"), "pdl"].mean()),
         "MG": _r2(odc.loc[odc.quarter.isin(Q17)
                           & (odc.customer_state == "MG"), "pdl"].mean())},
        "dict", ["2017-Q1", "2017-Q2", "2017-Q3", "2017-Q4"],
        {"start": "2017-01-01", "end": "2018-01-01"},
        tables=("orders", "customers"), notes="two-state filter")

    reg("latency_23", "lag_comparison",
        "Compare the average carrier-to-customer delivery lag (in days) "
        "in 2017-Q4 versus 2018-Q1 (monotonic orders only).",
        _two_window(od, "quarter", "2017-Q4", "2018-Q1", "cdl", "mean"),
        "dict", ["2017-Q4", "2018-Q1"], None)

    reg("latency_24", "lag_comparison",
        "Compare the average purchase-to-delivery lag (in days) in "
        "2018-Q1 versus 2018-Q2 (monotonic orders only).",
        _two_window(od, "quarter", "2018-Q1", "2018-Q2", "pdl", "mean"),
        "dict", ["2018-Q1", "2018-Q2"], None)

    reg("latency_25", "lag_comparison",
        "Compare the average purchase-to-approval lag (in days) for "
        "customers in São Paulo (SP) versus Rio de Janeiro (RJ), for "
        "orders placed during 2018 (monotonic orders only).",
        {"SP": _r2(odc.loc[odc.quarter.isin(Q18)
                           & (odc.customer_state == "SP"), "apl"].mean()),
         "RJ": _r2(odc.loc[odc.quarter.isin(Q18)
                           & (odc.customer_state == "RJ"), "apl"].mean())},
        "dict", Q18, {"start": "2018-01-01", "end": "2018-10-01"},
        tables=("orders", "customers"), notes="two-state filter")

    reg("latency_26", "lag_comparison",
        "Compare the average purchase-to-delivery lag (in days) in "
        "2017-Q1 versus 2018-Q1 (monotonic orders only).",
        _two_window(od, "quarter", "2017-Q1", "2018-Q1", "pdl", "mean"),
        "dict", ["2017-Q1", "2018-Q1"], None)

    # ---- 4 before_after_anchor (latency_27 … latency_30) ---------
    def window_mean(col, start, end):
        m = ((od.order_purchase_timestamp >= pd.Timestamp(start))
             & (od.order_purchase_timestamp < pd.Timestamp(end)))
        return _r2(od.loc[m, col].mean())

    # Carnival 2018 (2018-02-13): 30 days before vs 30 days after
    reg("latency_27", "before_after_anchor",
        "Compare the average purchase-to-delivery lag (in days) for "
        "orders placed in the 30 days BEFORE Carnival 2018 (2018-01-14 to "
        "2018-02-12) versus the 30 days AFTER (2018-02-13 to 2018-03-14), "
        "monotonic orders only.",
        {"before_carnival": window_mean("pdl", "2018-01-14", "2018-02-13"),
         "after_carnival": window_mean("pdl", "2018-02-13", "2018-03-15")},
        "dict", ["2018-01", "2018-02", "2018-03"],
        {"start": "2018-01-14", "end": "2018-03-15"}, event="carn2018")

    # Black Friday 2017 (2017-11-24): order count 30 days before vs after
    reg("latency_28", "before_after_anchor",
        "How many orders were placed in the 30 days BEFORE Black Friday "
        "2017 (2017-10-25 to 2017-11-23) versus the 30 days AFTER "
        "(2017-11-24 to 2017-12-23)?",
        {"before_bf2017": int(((o.order_purchase_timestamp
                                >= pd.Timestamp("2017-10-25"))
                               & (o.order_purchase_timestamp
                                  < pd.Timestamp("2017-11-24"))).sum()),
         "after_bf2017": int(((o.order_purchase_timestamp
                               >= pd.Timestamp("2017-11-24"))
                              & (o.order_purchase_timestamp
                                 < pd.Timestamp("2017-12-24"))).sum())},
        "dict", ["2017-10", "2017-11", "2017-12"],
        {"start": "2017-10-25", "end": "2017-12-24"}, event="bf2017",
        notes="order counts, not lag")

    # Children's Day 2017 (2017-10-12): 30 days before, single window lag
    reg("latency_29", "before_after_anchor",
        "What was the average purchase-to-delivery lag (in days) for "
        "orders placed in the 30 days before Children's Day 2017 "
        "(2017-09-12 to 2017-10-11), monotonic orders only?",
        window_mean("pdl", "2017-09-12", "2017-10-12"),
        "number", ["2017-09", "2017-10"],
        {"start": "2017-09-12", "end": "2017-10-12"}, event="kids17")

    # Mother's Day 2018 (2018-05-13): 30 days before vs after lag
    reg("latency_30", "before_after_anchor",
        "Compare the average purchase-to-delivery lag (in days) for "
        "orders placed in the 30 days BEFORE Mother's Day 2018 "
        "(2018-04-13 to 2018-05-12) versus the 30 days AFTER (2018-05-13 "
        "to 2018-06-11), monotonic orders only.",
        {"before_mothers": window_mean("pdl", "2018-04-13", "2018-05-13"),
         "after_mothers": window_mean("pdl", "2018-05-13", "2018-06-12")},
        "dict", ["2018-04", "2018-05", "2018-06"],
        {"start": "2018-04-13", "end": "2018-06-12"}, event="mothers18")


# ============================================================
# Main
# ============================================================

def _is_degenerate(q: dict) -> str | None:
    g = q["gold_answer"]
    if g is None:
        return "null"
    if isinstance(g, float) and pd.isna(g):
        return "NaN"
    if isinstance(g, dict):
        if len(g) == 0:
            return "empty dict"
        for k, v in g.items():
            if isinstance(v, dict) and len(v) == 0:
                return f"empty nested dict [{k}]"
            if isinstance(v, float) and pd.isna(v):
                return f"NaN value [{k}]"
    if isinstance(g, list) and len(g) == 0:
        return "empty list"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--out", type=Path,
                        default=Path("./outputs/temporal_questions_90.jsonl"))
    parser.add_argument("--events-out", type=Path,
                        default=Path("./benchmark/external_events.jsonl"))
    args = parser.parse_args()

    print(f"Loading Olist data from {args.data_dir} ...")
    tables = load_olist(args.data_dir)

    print("Building 30 pilot questions (unchanged) ...")
    build_trend(tables); build_drift(tables); build_latency(tables)
    n_pilot = len(QUESTIONS)
    print(f"  pilot: {n_pilot}")

    print("Building 60 new questions (q11–30 per category) ...")
    build_trend_ext(tables); build_drift_ext(tables); build_latency_ext(tables)
    print(f"  total: {len(QUESTIONS)}")

    # Degeneracy sweep (stricter than the pilot's null/NaN check).
    broken = [(q["qid"], why) for q in QUESTIONS
              if (why := _is_degenerate(q)) is not None]
    if broken:
        print(f"\nERROR: {len(broken)} questions have degenerate gold:")
        for qid, why in broken:
            print(f"  - {qid}: {why}")
        return 2

    # Distribution report
    print("\nSub-type distribution (target: trend 18/12, drift 18/12, "
          "latency 14/10/6):")
    for cat in ("trend", "drift", "latency"):
        sub = Counter(q["sub_type"] for q in QUESTIONS if q["category"] == cat)
        n = sum(1 for q in QUESTIONS if q["category"] == cat)
        print(f"  {cat} ({n}): {dict(sub)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for q in QUESTIONS:
            f.write(json.dumps(q, ensure_ascii=False, default=str) + "\n")
    print(f"\nWrote {len(QUESTIONS)} questions to {args.out}")

    # Event anchors (§4.3 deliverable)
    args.events_out.parent.mkdir(parents=True, exist_ok=True)
    with args.events_out.open("w", encoding="utf-8") as f:
        for eid, ev in EXTERNAL_EVENTS.items():
            f.write(json.dumps({"event_id": eid, **ev}) + "\n")
    print(f"Wrote {len(EXTERNAL_EVENTS)} event anchors to {args.events_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
