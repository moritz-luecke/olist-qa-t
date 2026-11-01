"""
============================================================
Phase 3.2 — Build Temporal Pilot Question Set (30 questions)
============================================================

Defines 10 pilot questions per new temporal category
(trend, drift, latency) for the temporal-extension paper.
Each gold answer is computed directly from the Olist CSVs via
pandas — that is the SQL-equivalent reference engine.

This is the PILOT subset (10/category). Full scale-up to
30/category (90 total) happens after pilot validation.

Output:
  outputs/temporal_pilot_questions.jsonl  — one question per line

Schema per question (extends the prior schema with temporal fields):
  qid                : "trend_01" | "drift_01" | "latency_01" | ...
  category           : "trend" | "drift" | "latency"
  sub_type           : "aggregation_over_time" |
                       "comparison_over_time" |
                       "category_drift"       |
                       "topk_change"          |
                       "inter_event_lag"      |
                       "lag_comparison"       |
                       "before_after_anchor"
  min_hops           : 2 | 3        (in addition to the temporal axis)
  tables_involved    : list of Olist tables
  question           : NL question (English, for the model)
  sql_query          : SQL reference query
  pandas_code        : compact pandas expression that returns the
                       same answer (reproducibility)
  gold_answer_type   : "number" | "list" | "dict" | "string"
  gold_answer        : concrete value (pilot is all-closed-form)
  time_anchors       : list of canonical bucket strings
                       (e.g. ["2017-Q1", "2017-Q4"]). Used by the
                       temporal-faithfulness judge.
  time_range         : null | {start, end}    half-open [start, end)
                       in ISO date format
  event_anchor       : null | event_id from external_events.jsonl
  exclusion_flags    : list of filter rules applied to gold computation
                       (e.g. ["nonmonotonic_intervals"]) — documented
                       per-question so retrievers / judges know what
                       was excluded.
  notes              : optional, free-form note

Design choices for the pilot (decided 2026-05-26):
  - All 30 questions are closed-form (numeric / list / dict).
    Open-form questions (requiring the LLM-judge) are deferred
    to the scale-up so the pilot can be auto-evaluated.
  - Time anchors default to quarter granularity ("2017-Q1"),
    matching the bi-level hierarchy's mid-level. Drift questions
    with explicit cutoff dates may use month granularity ("2017-07").
  - Latency questions exclude non-monotonic intervals (the 1,359
    carrier-before-approval + 23 delivery + 8,320 review anomalies)
    consistent with the §4.1 filter rule.

Sub-type distribution within each category:
  Trend    : 6 aggregation_over_time + 4 comparison_over_time
  Drift    : 6 category_drift        + 4 topk_change
  Latency  : 5 inter_event_lag + 3 lag_comparison + 2 before_after
============================================================
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the same loader as the prior pilot builder.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase2_kg"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent /
                         "phase3_benchmark"))
try:
    from build_pilot_questions import (
        load_olist, STATE_TO_REGION, add_region_col, r2, r4,
    )
except ImportError as e:
    print(f"ERROR: cannot import build_pilot_questions: {e}", file=sys.stderr)
    sys.exit(1)


# ============================================================
# External event anchors (small curated list, §4.3 of paper)
# ============================================================

EXTERNAL_EVENTS = {
    "bf2017":    {"name": "Black Friday 2017",      "date": "2017-11-24"},
    "bf2018":    {"name": "Black Friday 2018",      "date": "2018-11-23"},
    "carn2017":  {"name": "Carnival 2017",          "date": "2017-02-28"},
    "carn2018":  {"name": "Carnival 2018",          "date": "2018-02-13"},
    "worldcup":  {"name": "FIFA World Cup 2018",    "date": "2018-06-14"},
    "mothers17": {"name": "Mother's Day BR 2017",   "date": "2017-05-14"},
    "kids17":    {"name": "Children's Day BR 2017", "date": "2017-10-12"},
}


# ============================================================
# Question registry
# ============================================================

QUESTIONS: list[dict] = []


def register(q: dict) -> None:
    QUESTIONS.append(q)


# ============================================================
# Shared temporal helpers
# ============================================================

def add_quarter_col(df: pd.DataFrame, date_col: str,
                    out_col: str = "quarter") -> pd.DataFrame:
    """Add a string quarter column (e.g. '2017-Q1') from a timestamp column."""
    df = df.copy()
    ts = pd.to_datetime(df[date_col])
    df[out_col] = (
        ts.dt.year.astype(str) + "-Q" + ts.dt.quarter.astype(str)
    )
    return df


def add_month_col(df: pd.DataFrame, date_col: str,
                  out_col: str = "month") -> pd.DataFrame:
    """Add a string month column (e.g. '2017-07') from a timestamp column."""
    df = df.copy()
    ts = pd.to_datetime(df[date_col])
    df[out_col] = ts.dt.strftime("%Y-%m")
    return df


def filter_monotonic_delivery(orders: pd.DataFrame) -> pd.DataFrame:
    """Restrict to orders where purchase ≤ approved ≤ carrier ≤ delivered.

    This is the §4.1 filter — drops the 1,359 carrier-before-approval
    anomalies plus the 23 delivery-before-carrier anomalies, leaving
    a clean subset for latency questions. Returns a copy.
    """
    o = orders.copy()
    mask = (
        o["order_purchase_timestamp"].notna()
        & o["order_approved_at"].notna()
        & o["order_delivered_carrier_date"].notna()
        & o["order_delivered_customer_date"].notna()
        & (o["order_approved_at"]            >= o["order_purchase_timestamp"])
        & (o["order_delivered_carrier_date"] >= o["order_approved_at"])
        & (o["order_delivered_customer_date"]>= o["order_delivered_carrier_date"])
    )
    return o.loc[mask].copy()


# ============================================================
# CATEGORY 1 — TREND  (10 questions)
#   6 × aggregation_over_time
#   4 × comparison_over_time
# ============================================================

def build_trend(t: dict[str, pd.DataFrame]) -> None:
    """Trend questions: how does an aggregate evolve over time?"""
    o, it = t["orders"], t["order_items"]

    # ----- aggregation_over_time × 6 ------------------------------

    # trend_01: avg delivery time per quarter in 2017
    od = filter_monotonic_delivery(o)
    od = add_quarter_col(od, "order_purchase_timestamp")
    od["delivery_days"] = (
        od["order_delivered_customer_date"]
        - od["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400
    by_q = (od.loc[od["quarter"].str.startswith("2017"), :]
              .groupby("quarter")["delivery_days"].mean()
              .round(2).to_dict())
    register({
        "qid": "trend_01",
        "category": "trend",
        "sub_type": "aggregation_over_time",
        "min_hops": 2,
        "tables_involved": ["orders"],
        "question": (
            "What was the average purchase-to-delivery time (in days) "
            "per quarter of 2017, restricted to orders with a complete "
            "and monotonic delivery lifecycle?"
        ),
        "sql_query": (
            "SELECT date_trunc('quarter', order_purchase_timestamp) AS q,"
            " AVG(EXTRACT(epoch FROM (order_delivered_customer_date "
            " - order_purchase_timestamp))/86400) AS avg_days "
            "FROM orders "
            "WHERE order_approved_at >= order_purchase_timestamp "
            "  AND order_delivered_carrier_date >= order_approved_at "
            "  AND order_delivered_customer_date "
            "      >= order_delivered_carrier_date "
            "  AND date_part('year', order_purchase_timestamp) = 2017 "
            "GROUP BY q ORDER BY q;"
        ),
        "pandas_code": (
            "od = filter_monotonic_delivery(orders);"
            " od = add_quarter_col(od, 'order_purchase_timestamp');"
            " od['delivery_days'] = (od['order_delivered_customer_date']"
            " - od['order_purchase_timestamp']).dt.total_seconds()/86400;"
            " od[od.quarter.str.startswith('2017')].groupby('quarter')"
            "['delivery_days'].mean().round(2).to_dict()"
        ),
        "gold_answer_type": "dict",
        "gold_answer": by_q,
        "time_anchors": ["2017-Q1", "2017-Q2", "2017-Q3", "2017-Q4"],
        "time_range": {"start": "2017-01-01", "end": "2018-01-01"},
        "event_anchor": None,
        "exclusion_flags": ["nonmonotonic_intervals"],
    })

    # trend_02: monthly order count in 2017 (a basic volume trend)
    om = add_month_col(o, "order_purchase_timestamp")
    by_m = (om.loc[om["month"].str.startswith("2017"), :]
              .groupby("month").size().astype(int).to_dict())
    register({
        "qid": "trend_02",
        "category": "trend",
        "sub_type": "aggregation_over_time",
        "min_hops": 2,
        "tables_involved": ["orders"],
        "question": (
            "How many orders were placed per month in 2017?"
        ),
        "sql_query": (
            "SELECT to_char(order_purchase_timestamp, 'YYYY-MM') AS m,"
            " COUNT(*) AS n FROM orders "
            "WHERE date_part('year', order_purchase_timestamp) = 2017 "
            "GROUP BY m ORDER BY m;"
        ),
        "pandas_code": (
            "om = add_month_col(orders, 'order_purchase_timestamp');"
            " om[om.month.str.startswith('2017')].groupby('month')"
            ".size().to_dict()"
        ),
        "gold_answer_type": "dict",
        "gold_answer": by_m,
        "time_anchors": [f"2017-{m:02d}" for m in range(1, 13)],
        "time_range": {"start": "2017-01-01", "end": "2018-01-01"},
        "event_anchor": None,
        "exclusion_flags": [],
    })

    # trend_03: avg review score per quarter in 2017
    rv = t["reviews"].merge(
        o[["order_id", "order_purchase_timestamp"]],
        on="order_id", how="left",
    )
    rv = add_quarter_col(rv, "order_purchase_timestamp")
    by_q = (rv.loc[rv["quarter"].str.startswith("2017"), :]
              .groupby("quarter")["review_score"].mean()
              .round(2).to_dict())
    register({
        "qid": "trend_03",
        "category": "trend",
        "sub_type": "aggregation_over_time",
        "min_hops": 2,
        "tables_involved": ["orders", "reviews"],
        "question": (
            "What was the average review score per quarter of 2017?"
        ),
        "sql_query": (
            "SELECT date_trunc('quarter', o.order_purchase_timestamp) AS q,"
            " AVG(r.review_score) AS avg_score "
            "FROM order_reviews r "
            "JOIN orders o ON r.order_id = o.order_id "
            "WHERE date_part('year', o.order_purchase_timestamp) = 2017 "
            "GROUP BY q ORDER BY q;"
        ),
        "pandas_code": (
            "rv = reviews.merge(orders[['order_id','order_purchase_timestamp']], "
            "on='order_id'); rv = add_quarter_col(rv, "
            "'order_purchase_timestamp'); "
            "rv[rv.quarter.str.startswith('2017')].groupby('quarter')"
            "['review_score'].mean().round(2).to_dict()"
        ),
        "gold_answer_type": "dict",
        "gold_answer": by_q,
        "time_anchors": ["2017-Q1", "2017-Q2", "2017-Q3", "2017-Q4"],
        "time_range": {"start": "2017-01-01", "end": "2018-01-01"},
        "event_anchor": None,
        "exclusion_flags": [],
    })

    # trend_04: monthly GMV (total payment value) in 2017
    pay_m = t["payments"].merge(
        o[["order_id", "order_purchase_timestamp"]], on="order_id"
    )
    pay_m = add_month_col(pay_m, "order_purchase_timestamp")
    by_m = (pay_m.loc[pay_m["month"].str.startswith("2017"), :]
                 .groupby("month")["payment_value"].sum()
                 .round(2).to_dict())
    register({
        "qid": "trend_04",
        "category": "trend",
        "sub_type": "aggregation_over_time",
        "min_hops": 2,
        "tables_involved": ["orders", "payments"],
        "question": (
            "What was the total payment value (BRL) per month in 2017?"
        ),
        "sql_query": (
            "SELECT to_char(o.order_purchase_timestamp,'YYYY-MM') AS m,"
            " SUM(p.payment_value) AS gmv "
            "FROM order_payments p "
            "JOIN orders o ON p.order_id = o.order_id "
            "WHERE date_part('year', o.order_purchase_timestamp) = 2017 "
            "GROUP BY m ORDER BY m;"
        ),
        "pandas_code": (
            "pm = payments.merge(orders[['order_id',"
            "'order_purchase_timestamp']], on='order_id');"
            " pm = add_month_col(pm, 'order_purchase_timestamp');"
            " pm[pm.month.str.startswith('2017')].groupby('month')"
            "['payment_value'].sum().round(2).to_dict()"
        ),
        "gold_answer_type": "dict",
        "gold_answer": by_m,
        "time_anchors": [f"2017-{m:02d}" for m in range(1, 13)],
        "time_range": {"start": "2017-01-01", "end": "2018-01-01"},
        "event_anchor": None,
        "exclusion_flags": [],
    })

    # trend_05: avg installments per quarter in 2017
    pay_q = t["payments"].merge(
        o[["order_id", "order_purchase_timestamp"]], on="order_id"
    )
    pay_q = add_quarter_col(pay_q, "order_purchase_timestamp")
    by_q = (pay_q.loc[pay_q["quarter"].str.startswith("2017"), :]
                 .groupby("quarter")["payment_installments"].mean()
                 .round(2).to_dict())
    register({
        "qid": "trend_05",
        "category": "trend",
        "sub_type": "aggregation_over_time",
        "min_hops": 2,
        "tables_involved": ["orders", "payments"],
        "question": (
            "What was the average number of payment installments "
            "per quarter of 2017?"
        ),
        "sql_query": (
            "SELECT date_trunc('quarter', o.order_purchase_timestamp) AS q,"
            " AVG(p.payment_installments) AS avg_inst "
            "FROM order_payments p "
            "JOIN orders o ON p.order_id = o.order_id "
            "WHERE date_part('year', o.order_purchase_timestamp) = 2017 "
            "GROUP BY q ORDER BY q;"
        ),
        "pandas_code": (
            "pq = payments.merge(orders[['order_id',"
            "'order_purchase_timestamp']], on='order_id');"
            " pq = add_quarter_col(pq, 'order_purchase_timestamp');"
            " pq[pq.quarter.str.startswith('2017')].groupby('quarter')"
            "['payment_installments'].mean().round(2).to_dict()"
        ),
        "gold_answer_type": "dict",
        "gold_answer": by_q,
        "time_anchors": ["2017-Q1", "2017-Q2", "2017-Q3", "2017-Q4"],
        "time_range": {"start": "2017-01-01", "end": "2018-01-01"},
        "event_anchor": None,
        "exclusion_flags": [],
    })

    # trend_06: avg freight value per quarter in 2017
    it_q = it.merge(o[["order_id", "order_purchase_timestamp"]],
                    on="order_id")
    it_q = add_quarter_col(it_q, "order_purchase_timestamp")
    by_q = (it_q.loc[it_q["quarter"].str.startswith("2017"), :]
                 .groupby("quarter")["freight_value"].mean()
                 .round(2).to_dict())
    register({
        "qid": "trend_06",
        "category": "trend",
        "sub_type": "aggregation_over_time",
        "min_hops": 2,
        "tables_involved": ["orders", "order_items"],
        "question": (
            "What was the average freight value per order line "
            "per quarter of 2017?"
        ),
        "sql_query": (
            "SELECT date_trunc('quarter', o.order_purchase_timestamp) AS q,"
            " AVG(i.freight_value) AS avg_freight "
            "FROM order_items i "
            "JOIN orders o ON i.order_id = o.order_id "
            "WHERE date_part('year', o.order_purchase_timestamp) = 2017 "
            "GROUP BY q ORDER BY q;"
        ),
        "pandas_code": (
            "iq = order_items.merge(orders[['order_id',"
            "'order_purchase_timestamp']], on='order_id');"
            " iq = add_quarter_col(iq, 'order_purchase_timestamp');"
            " iq[iq.quarter.str.startswith('2017')].groupby('quarter')"
            "['freight_value'].mean().round(2).to_dict()"
        ),
        "gold_answer_type": "dict",
        "gold_answer": by_q,
        "time_anchors": ["2017-Q1", "2017-Q2", "2017-Q3", "2017-Q4"],
        "time_range": {"start": "2017-01-01", "end": "2018-01-01"},
        "event_anchor": None,
        "exclusion_flags": [],
    })

    # ----- comparison_over_time × 4 -------------------------------

    # trend_07: delivery time Q1 2017 vs Q4 2017
    od = filter_monotonic_delivery(o)
    od["delivery_days"] = (
        od["order_delivered_customer_date"]
        - od["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400
    od = add_quarter_col(od, "order_purchase_timestamp")
    q1_mean = r2(od.loc[od["quarter"] == "2017-Q1", "delivery_days"].mean())
    q4_mean = r2(od.loc[od["quarter"] == "2017-Q4", "delivery_days"].mean())
    register({
        "qid": "trend_07",
        "category": "trend",
        "sub_type": "comparison_over_time",
        "min_hops": 2,
        "tables_involved": ["orders"],
        "question": (
            "How did average delivery time (in days) change from "
            "Q1 2017 to Q4 2017, considering only orders with a "
            "complete monotonic delivery lifecycle?"
        ),
        "sql_query": (
            "SELECT date_trunc('quarter', order_purchase_timestamp) AS q,"
            " AVG(EXTRACT(epoch FROM (order_delivered_customer_date"
            " - order_purchase_timestamp))/86400) "
            "FROM orders WHERE "
            " order_approved_at >= order_purchase_timestamp "
            " AND order_delivered_carrier_date >= order_approved_at "
            " AND order_delivered_customer_date "
            "     >= order_delivered_carrier_date "
            " AND date_part('quarter', order_purchase_timestamp) IN (1,4)"
            " AND date_part('year', order_purchase_timestamp) = 2017 "
            "GROUP BY q;"
        ),
        "pandas_code": (
            "od = filter_monotonic_delivery(orders); "
            "od['delivery_days'] = (od.order_delivered_customer_date "
            "- od.order_purchase_timestamp).dt.total_seconds()/86400; "
            "od = add_quarter_col(od, 'order_purchase_timestamp'); "
            "{'2017-Q1': r2(od[od.quarter=='2017-Q1'].delivery_days.mean()),"
            " '2017-Q4': r2(od[od.quarter=='2017-Q4'].delivery_days.mean())}"
        ),
        "gold_answer_type": "dict",
        "gold_answer": {"2017-Q1": q1_mean, "2017-Q4": q4_mean},
        "time_anchors": ["2017-Q1", "2017-Q4"],
        "time_range": None,
        "event_anchor": None,
        "exclusion_flags": ["nonmonotonic_intervals"],
    })

    # trend_08: order volume Q1 vs Q4 2017
    by_q = o.assign(
        q=pd.to_datetime(o.order_purchase_timestamp).dt.to_period("Q")
    ).groupby("q").size()
    q1 = int(by_q.loc[pd.Period("2017Q1")]) if pd.Period("2017Q1") in by_q.index else 0
    q4 = int(by_q.loc[pd.Period("2017Q4")]) if pd.Period("2017Q4") in by_q.index else 0
    register({
        "qid": "trend_08",
        "category": "trend",
        "sub_type": "comparison_over_time",
        "min_hops": 2,
        "tables_involved": ["orders"],
        "question": (
            "Compare the total number of orders placed in Q1 2017 "
            "versus Q4 2017."
        ),
        "sql_query": (
            "SELECT date_trunc('quarter', order_purchase_timestamp) AS q,"
            " COUNT(*) FROM orders "
            "WHERE date_part('year', order_purchase_timestamp) = 2017 "
            "  AND date_part('quarter', order_purchase_timestamp) IN (1,4)"
            " GROUP BY q;"
        ),
        "pandas_code": (
            "{'2017-Q1': int(orders.assign(q=pd.to_datetime("
            "orders.order_purchase_timestamp).dt.to_period('Q'))"
            ".groupby('q').size().loc[pd.Period('2017Q1')]),"
            " '2017-Q4': int(orders.assign(q=pd.to_datetime("
            "orders.order_purchase_timestamp).dt.to_period('Q'))"
            ".groupby('q').size().loc[pd.Period('2017Q4')])}"
        ),
        "gold_answer_type": "dict",
        "gold_answer": {"2017-Q1": q1, "2017-Q4": q4},
        "time_anchors": ["2017-Q1", "2017-Q4"],
        "time_range": None,
        "event_anchor": None,
        "exclusion_flags": [],
    })

    # trend_09: review score Q2 2017 vs Q2 2018 (year-over-year)
    rv = t["reviews"].merge(
        o[["order_id", "order_purchase_timestamp"]], on="order_id"
    )
    rv = add_quarter_col(rv, "order_purchase_timestamp")
    s_17 = r2(rv.loc[rv["quarter"] == "2017-Q2", "review_score"].mean())
    s_18 = r2(rv.loc[rv["quarter"] == "2018-Q2", "review_score"].mean())
    register({
        "qid": "trend_09",
        "category": "trend",
        "sub_type": "comparison_over_time",
        "min_hops": 2,
        "tables_involved": ["orders", "reviews"],
        "question": (
            "How did the average review score in Q2 change from "
            "2017 to 2018?"
        ),
        "sql_query": (
            "SELECT date_trunc('quarter', o.order_purchase_timestamp) AS q,"
            " AVG(r.review_score) FROM order_reviews r "
            "JOIN orders o ON r.order_id = o.order_id "
            "WHERE date_part('quarter', o.order_purchase_timestamp) = 2 "
            "  AND date_part('year', o.order_purchase_timestamp) IN (2017,2018)"
            " GROUP BY q;"
        ),
        "pandas_code": (
            "rv = reviews.merge(orders[['order_id',"
            "'order_purchase_timestamp']], on='order_id'); "
            "rv = add_quarter_col(rv, 'order_purchase_timestamp'); "
            "{'2017-Q2': r2(rv[rv.quarter=='2017-Q2'].review_score.mean()),"
            " '2018-Q2': r2(rv[rv.quarter=='2018-Q2'].review_score.mean())}"
        ),
        "gold_answer_type": "dict",
        "gold_answer": {"2017-Q2": s_17, "2018-Q2": s_18},
        "time_anchors": ["2017-Q2", "2018-Q2"],
        "time_range": None,
        "event_anchor": None,
        "exclusion_flags": [],
    })

    # trend_10: avg item price first half vs second half of 2017
    it_h = it.merge(o[["order_id", "order_purchase_timestamp"]],
                    on="order_id")
    it_h["half"] = pd.to_datetime(it_h.order_purchase_timestamp).dt.month.apply(
        lambda m: "H1" if m <= 6 else "H2"
    )
    it_h["year"] = pd.to_datetime(it_h.order_purchase_timestamp).dt.year
    it_17 = it_h[it_h.year == 2017]
    h1 = r2(it_17.loc[it_17["half"] == "H1", "price"].mean())
    h2 = r2(it_17.loc[it_17["half"] == "H2", "price"].mean())
    register({
        "qid": "trend_10",
        "category": "trend",
        "sub_type": "comparison_over_time",
        "min_hops": 2,
        "tables_involved": ["orders", "order_items"],
        "question": (
            "What was the average item price in the first half "
            "(Jan-Jun) versus the second half (Jul-Dec) of 2017?"
        ),
        "sql_query": (
            "SELECT CASE WHEN date_part('month', o.order_purchase_timestamp)"
            " <= 6 THEN 'H1' ELSE 'H2' END AS half, AVG(i.price) "
            "FROM order_items i JOIN orders o ON i.order_id = o.order_id "
            "WHERE date_part('year', o.order_purchase_timestamp) = 2017 "
            "GROUP BY half;"
        ),
        "pandas_code": (
            "ih = order_items.merge(orders[['order_id',"
            "'order_purchase_timestamp']], on='order_id'); "
            "ih['half'] = pd.to_datetime(ih.order_purchase_timestamp)"
            ".dt.month.apply(lambda m: 'H1' if m<=6 else 'H2'); "
            "ih['year'] = pd.to_datetime(ih.order_purchase_timestamp).dt.year; "
            "{'2017-H1': r2(ih[(ih.year==2017)&(ih.half=='H1')].price.mean()),"
            " '2017-H2': r2(ih[(ih.year==2017)&(ih.half=='H2')].price.mean())}"
        ),
        "gold_answer_type": "dict",
        "gold_answer": {"2017-H1": h1, "2017-H2": h2},
        "time_anchors": ["2017-H1", "2017-H2"],
        "time_range": {"start": "2017-01-01", "end": "2018-01-01"},
        "event_anchor": None,
        "exclusion_flags": [],
    })


# ============================================================
# CATEGORY 2 — DRIFT  (10 questions)
#   6 × category_drift  (which product categories deteriorated?)
#   4 × topk_change     (top-k changes between two windows)
# ============================================================

def build_drift(t: dict[str, pd.DataFrame]) -> None:
    """Drift questions: which entities changed most between windows?"""
    o, it, rv = t["orders"], t["order_items"], t["reviews"]
    pr = t["products"]

    # Pre-join orders × items × products for category-level analysis
    oi = it.merge(o[["order_id", "order_purchase_timestamp"]],
                  on="order_id")
    oi = oi.merge(pr[["product_id", "product_category_name_english"]],
                  on="product_id", how="left")
    oi = add_quarter_col(oi, "order_purchase_timestamp")
    oi["category"] = oi["product_category_name_english"].fillna("UNKNOWN")

    # ----- category_drift × 6 -------------------------------------

    # drift_01: top 3 categories with steepest review-score drop, Q3→Q4 2017
    rv_full = rv.merge(it[["order_id", "product_id"]], on="order_id")
    rv_full = rv_full.merge(o[["order_id", "order_purchase_timestamp"]],
                            on="order_id")
    rv_full = rv_full.merge(pr[["product_id", "product_category_name_english"]],
                            on="product_id", how="left")
    rv_full = add_quarter_col(rv_full, "order_purchase_timestamp")
    rv_full["category"] = rv_full["product_category_name_english"].fillna("UNKNOWN")
    # Need ≥ 30 reviews in each quarter for a stable drift estimate
    counts = rv_full.groupby(["category", "quarter"]).size().unstack(fill_value=0)
    stable_cats = counts[
        (counts.get("2017-Q3", 0) >= 30) & (counts.get("2017-Q4", 0) >= 30)
    ].index.tolist()
    means = (rv_full[rv_full["category"].isin(stable_cats)]
                  .groupby(["category", "quarter"])["review_score"].mean()
                  .unstack())
    drop = (means["2017-Q3"] - means["2017-Q4"]).sort_values(ascending=False)
    top3 = drop.head(3).round(2).to_dict()
    register({
        "qid": "drift_01",
        "category": "drift",
        "sub_type": "category_drift",
        "min_hops": 3,
        "tables_involved": ["orders", "order_items", "products", "reviews"],
        "question": (
            "Which 3 product categories saw the steepest drop in "
            "average review score from Q3 2017 to Q4 2017? "
            "Restrict to categories with at least 30 reviews in each "
            "quarter. Return the category names ordered by drop size, "
            "with the drop value (Q3 mean − Q4 mean)."
        ),
        "sql_query": "-- see pandas_code; multi-step",
        "pandas_code": (
            "see build_drift drift_01 — counts ≥30 per quarter, "
            "compute Q3-mean − Q4-mean per category, top 3 by drop"
        ),
        "gold_answer_type": "dict",
        "gold_answer": top3,
        "time_anchors": ["2017-Q3", "2017-Q4"],
        "time_range": None,
        "event_anchor": None,
        "exclusion_flags": ["min_30_reviews_per_window"],
        "notes": "Positive drop = score went DOWN (deteriorated).",
    })

    # drift_02: top 3 categories with steepest review-score IMPROVEMENT, Q1→Q2 2018
    counts = rv_full.groupby(["category", "quarter"]).size().unstack(fill_value=0)
    stable = counts[
        (counts.get("2018-Q1", 0) >= 30) & (counts.get("2018-Q2", 0) >= 30)
    ].index.tolist()
    means = (rv_full[rv_full["category"].isin(stable)]
                  .groupby(["category", "quarter"])["review_score"].mean()
                  .unstack())
    rise = (means["2018-Q2"] - means["2018-Q1"]).sort_values(ascending=False)
    top3 = rise.head(3).round(2).to_dict()
    register({
        "qid": "drift_02",
        "category": "drift",
        "sub_type": "category_drift",
        "min_hops": 3,
        "tables_involved": ["orders", "order_items", "products", "reviews"],
        "question": (
            "Which 3 product categories saw the steepest improvement "
            "in average review score from Q1 2018 to Q2 2018? "
            "Restrict to categories with at least 30 reviews in each "
            "quarter. Return the category names ordered by "
            "improvement size."
        ),
        "sql_query": "-- see pandas_code; multi-step",
        "pandas_code": "see build_drift drift_02",
        "gold_answer_type": "dict",
        "gold_answer": top3,
        "time_anchors": ["2018-Q1", "2018-Q2"],
        "time_range": None,
        "event_anchor": None,
        "exclusion_flags": ["min_30_reviews_per_window"],
    })

    # drift_03: top 3 categories with steepest VOLUME drop, Q3→Q4 2017
    vol = oi.groupby(["category", "quarter"]).size().unstack(fill_value=0)
    vol_drop = (vol.get("2017-Q3", pd.Series(0, index=vol.index))
                - vol.get("2017-Q4", pd.Series(0, index=vol.index)))
    # Only categories with ≥ 50 items in Q3 to filter noise
    vol_drop = vol_drop[vol.get("2017-Q3", 0) >= 50]
    top3 = vol_drop.sort_values(ascending=False).head(3).astype(int).to_dict()
    register({
        "qid": "drift_03",
        "category": "drift",
        "sub_type": "category_drift",
        "min_hops": 3,
        "tables_involved": ["orders", "order_items", "products"],
        "question": (
            "Which 3 product categories saw the steepest drop in order "
            "volume from Q3 2017 to Q4 2017? Restrict to categories with "
            "at least 50 items in Q3 2017. Return categories with the "
            "drop count (Q3 − Q4)."
        ),
        "sql_query": "-- see pandas_code; multi-step",
        "pandas_code": "see build_drift drift_03",
        "gold_answer_type": "dict",
        "gold_answer": top3,
        "time_anchors": ["2017-Q3", "2017-Q4"],
        "time_range": None,
        "event_anchor": None,
        "exclusion_flags": ["min_50_items_q3"],
    })

    # drift_04: top 3 categories with steepest VOLUME growth, 2017 → 2018 (year-over-year, Q2-vs-Q2 for season parity)
    counts = oi.groupby(["category", "quarter"]).size().unstack(fill_value=0)
    stable = counts[
        (counts.get("2017-Q2", 0) >= 50) & (counts.get("2018-Q2", 0) >= 50)
    ].index.tolist()
    growth = ((counts.loc[stable, "2018-Q2"]
               - counts.loc[stable, "2017-Q2"])
              .sort_values(ascending=False))
    top3 = growth.head(3).astype(int).to_dict()
    register({
        "qid": "drift_04",
        "category": "drift",
        "sub_type": "category_drift",
        "min_hops": 3,
        "tables_involved": ["orders", "order_items", "products"],
        "question": (
            "Which 3 product categories grew the most in Q2 order volume "
            "from 2017 to 2018? Restrict to categories with at least 50 "
            "items in both Q2 2017 and Q2 2018."
        ),
        "sql_query": "-- see pandas_code; multi-step",
        "pandas_code": "see build_drift drift_04",
        "gold_answer_type": "dict",
        "gold_answer": top3,
        "time_anchors": ["2017-Q2", "2018-Q2"],
        "time_range": None,
        "event_anchor": None,
        "exclusion_flags": ["min_50_items_per_window"],
    })

    # drift_05: top 3 categories with steepest AVG PRICE drop, Q3→Q4 2017
    oi_p = oi.dropna(subset=["price"])
    counts = oi_p.groupby(["category", "quarter"]).size().unstack(fill_value=0)
    stable = counts[
        (counts.get("2017-Q3", 0) >= 30) & (counts.get("2017-Q4", 0) >= 30)
    ].index.tolist()
    prices = (oi_p[oi_p["category"].isin(stable)]
                   .groupby(["category", "quarter"])["price"].mean()
                   .unstack())
    drop = (prices["2017-Q3"] - prices["2017-Q4"]).sort_values(ascending=False)
    top3 = drop.head(3).round(2).to_dict()
    register({
        "qid": "drift_05",
        "category": "drift",
        "sub_type": "category_drift",
        "min_hops": 3,
        "tables_involved": ["orders", "order_items", "products"],
        "question": (
            "Which 3 product categories saw the steepest drop in average "
            "item price from Q3 2017 to Q4 2017? Restrict to categories "
            "with at least 30 items in each quarter. Return the drop "
            "value (Q3 mean − Q4 mean) in BRL."
        ),
        "sql_query": "-- see pandas_code; multi-step",
        "pandas_code": "see build_drift drift_05",
        "gold_answer_type": "dict",
        "gold_answer": top3,
        "time_anchors": ["2017-Q3", "2017-Q4"],
        "time_range": None,
        "event_anchor": None,
        "exclusion_flags": ["min_30_items_per_window"],
    })

    # drift_06: which categories appeared in Q4 2017 but not in Q4 2016?
    cats_q4_17 = set(oi.loc[oi["quarter"] == "2017-Q4", "category"].unique())
    cats_q4_16 = set(oi.loc[oi["quarter"] == "2016-Q4", "category"].unique())
    new_cats = sorted(cats_q4_17 - cats_q4_16)
    register({
        "qid": "drift_06",
        "category": "drift",
        "sub_type": "category_drift",
        "min_hops": 3,
        "tables_involved": ["orders", "order_items", "products"],
        "question": (
            "Which product categories had orders in Q4 2017 but had "
            "NO orders in Q4 2016? Return the alphabetically sorted list."
        ),
        "sql_query": "-- see pandas_code; set difference",
        "pandas_code": (
            "sorted(set(oi[oi.quarter=='2017-Q4'].category) "
            "- set(oi[oi.quarter=='2016-Q4'].category))"
        ),
        "gold_answer_type": "list",
        "gold_answer": new_cats,
        "time_anchors": ["2016-Q4", "2017-Q4"],
        "time_range": None,
        "event_anchor": None,
        "exclusion_flags": [],
        "notes": "Olist marketplace ramp-up — many categories had no "
                 "presence in Q4 2016. List is potentially long.",
    })

    # ----- topk_change × 4 ---------------------------------------

    # drift_07: top 5 sellers by revenue Q1 vs Q4 2017 (do the names change?)
    pay_full = t["payments"].merge(
        o[["order_id", "order_purchase_timestamp"]], on="order_id"
    )
    pay_full = pay_full.merge(
        it[["order_id", "seller_id"]].drop_duplicates(), on="order_id"
    )
    pay_full = add_quarter_col(pay_full, "order_purchase_timestamp")
    top_q1 = (pay_full[pay_full["quarter"] == "2017-Q1"]
                  .groupby("seller_id")["payment_value"].sum()
                  .sort_values(ascending=False).head(5).index.tolist())
    top_q4 = (pay_full[pay_full["quarter"] == "2017-Q4"]
                  .groupby("seller_id")["payment_value"].sum()
                  .sort_values(ascending=False).head(5).index.tolist())
    register({
        "qid": "drift_07",
        "category": "drift",
        "sub_type": "topk_change",
        "min_hops": 3,
        "tables_involved": ["orders", "order_items", "payments"],
        "question": (
            "Who were the top 5 sellers by revenue in Q1 2017 and the "
            "top 5 in Q4 2017? Return both ranked lists of seller IDs."
        ),
        "sql_query": "-- see pandas_code; two grouped sums + top-5",
        "pandas_code": (
            "see build_drift drift_07 — payments × items × orders, "
            "groupby seller_id, top-5 each quarter"
        ),
        "gold_answer_type": "dict",
        "gold_answer": {"2017-Q1": top_q1, "2017-Q4": top_q4},
        "time_anchors": ["2017-Q1", "2017-Q4"],
        "time_range": None,
        "event_anchor": None,
        "exclusion_flags": [],
    })

    # drift_08: top 5 customer states by order count Q1 vs Q4 2017
    o_c = o.merge(t["customers"][["customer_id", "customer_state"]],
                  on="customer_id")
    o_c = add_quarter_col(o_c, "order_purchase_timestamp")
    top_q1 = (o_c[o_c["quarter"] == "2017-Q1"]
                  .groupby("customer_state").size()
                  .sort_values(ascending=False).head(5).index.tolist())
    top_q4 = (o_c[o_c["quarter"] == "2017-Q4"]
                  .groupby("customer_state").size()
                  .sort_values(ascending=False).head(5).index.tolist())
    register({
        "qid": "drift_08",
        "category": "drift",
        "sub_type": "topk_change",
        "min_hops": 2,
        "tables_involved": ["orders", "customers"],
        "question": (
            "What were the top 5 customer states by order count in "
            "Q1 2017 versus Q4 2017?"
        ),
        "sql_query": "-- see pandas_code; two grouped counts + top-5",
        "pandas_code": (
            "oc = orders.merge(customers[['customer_id','customer_state']], "
            "on='customer_id'); oc = add_quarter_col(oc, "
            "'order_purchase_timestamp'); ..."
        ),
        "gold_answer_type": "dict",
        "gold_answer": {"2017-Q1": top_q1, "2017-Q4": top_q4},
        "time_anchors": ["2017-Q1", "2017-Q4"],
        "time_range": None,
        "event_anchor": None,
        "exclusion_flags": [],
    })

    # drift_09: top 3 payment methods by share, Q1 vs Q4 2017
    pay_full2 = t["payments"].merge(
        o[["order_id", "order_purchase_timestamp"]], on="order_id"
    )
    pay_full2 = add_quarter_col(pay_full2, "order_purchase_timestamp")
    def topk_share(df: pd.DataFrame, k: int = 3) -> dict:
        n = len(df)
        if n == 0:
            return {}
        shares = (df["payment_type"].value_counts() / n).round(4) * 100
        return {k_: r2(v) for k_, v in shares.head(k).items()}
    q1_share = topk_share(pay_full2[pay_full2.quarter == "2017-Q1"])
    q4_share = topk_share(pay_full2[pay_full2.quarter == "2017-Q4"])
    register({
        "qid": "drift_09",
        "category": "drift",
        "sub_type": "topk_change",
        "min_hops": 2,
        "tables_involved": ["orders", "payments"],
        "question": (
            "What were the top 3 payment methods (by share of payment "
            "rows, in %) in Q1 2017 versus Q4 2017?"
        ),
        "sql_query": "-- see pandas_code; share by payment_type per quarter",
        "pandas_code": "see build_drift drift_09",
        "gold_answer_type": "dict",
        "gold_answer": {"2017-Q1": q1_share, "2017-Q4": q4_share},
        "time_anchors": ["2017-Q1", "2017-Q4"],
        "time_range": None,
        "event_anchor": None,
        "exclusion_flags": [],
    })

    # drift_10: top 5 categories by order count, 2017-H1 vs 2017-H2
    oi_h = oi.copy()
    oi_h["half"] = pd.to_datetime(oi_h.order_purchase_timestamp).dt.month.apply(
        lambda m: "H1" if m <= 6 else "H2"
    )
    oi_h["year"] = pd.to_datetime(oi_h.order_purchase_timestamp).dt.year
    oi_17 = oi_h[oi_h.year == 2017]
    top_h1 = (oi_17[oi_17.half == "H1"].groupby("category").size()
                  .sort_values(ascending=False).head(5).index.tolist())
    top_h2 = (oi_17[oi_17.half == "H2"].groupby("category").size()
                  .sort_values(ascending=False).head(5).index.tolist())
    register({
        "qid": "drift_10",
        "category": "drift",
        "sub_type": "topk_change",
        "min_hops": 3,
        "tables_involved": ["orders", "order_items", "products"],
        "question": (
            "What were the top 5 product categories by order item count "
            "in the first half (Jan-Jun) of 2017 versus the second half "
            "(Jul-Dec) of 2017?"
        ),
        "sql_query": "-- see pandas_code; two grouped counts + top-5",
        "pandas_code": "see build_drift drift_10",
        "gold_answer_type": "dict",
        "gold_answer": {"2017-H1": top_h1, "2017-H2": top_h2},
        "time_anchors": ["2017-H1", "2017-H2"],
        "time_range": {"start": "2017-01-01", "end": "2018-01-01"},
        "event_anchor": None,
        "exclusion_flags": [],
    })


# ============================================================
# CATEGORY 3 — LATENCY  (10 questions)
#   5 × inter_event_lag
#   3 × lag_comparison
#   2 × before_after_anchor
# ============================================================

def build_latency(t: dict[str, pd.DataFrame]) -> None:
    """Latency questions: inter-event durations on order lifecycles."""
    o, c = t["orders"], t["customers"]
    od = filter_monotonic_delivery(o)
    od_c = od.merge(c[["customer_id", "customer_state"]], on="customer_id")

    # Pre-compute the three core latencies in days
    od_c["approval_lag_days"] = (
        od_c["order_approved_at"] - od_c["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400
    od_c["carrier_lag_days"] = (
        od_c["order_delivered_carrier_date"] - od_c["order_approved_at"]
    ).dt.total_seconds() / 86400
    od_c["delivery_lag_days"] = (
        od_c["order_delivered_customer_date"]
        - od_c["order_delivered_carrier_date"]
    ).dt.total_seconds() / 86400
    od_c["payment_to_delivery_days"] = (
        od_c["order_delivered_customer_date"] - od_c["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400

    od_c = add_quarter_col(od_c, "order_purchase_timestamp")

    # ----- inter_event_lag × 5 ------------------------------------

    # latency_01: avg payment-to-delivery lag for SP orders in 2017
    mask = ((od_c.customer_state == "SP") & od_c.quarter.str.startswith("2017"))
    val = r2(od_c.loc[mask, "payment_to_delivery_days"].mean())
    register({
        "qid": "latency_01",
        "category": "latency",
        "sub_type": "inter_event_lag",
        "min_hops": 2,
        "tables_involved": ["orders", "customers"],
        "question": (
            "What was the average purchase-to-delivery lag (in days) "
            "for orders from customers in São Paulo (state SP) "
            "placed during 2017?"
        ),
        "sql_query": (
            "SELECT AVG(EXTRACT(epoch FROM (o.order_delivered_customer_date"
            " - o.order_purchase_timestamp))/86400) "
            "FROM orders o JOIN customers c "
            " ON o.customer_id = c.customer_id "
            "WHERE c.customer_state = 'SP' "
            "  AND date_part('year', o.order_purchase_timestamp) = 2017 "
            "  AND o.order_approved_at >= o.order_purchase_timestamp "
            "  AND o.order_delivered_carrier_date >= o.order_approved_at "
            "  AND o.order_delivered_customer_date "
            "      >= o.order_delivered_carrier_date;"
        ),
        "pandas_code": (
            "od = filter_monotonic_delivery(orders).merge("
            "customers[['customer_id','customer_state']], on='customer_id');"
            " od['lag'] = (od.order_delivered_customer_date "
            "- od.order_purchase_timestamp).dt.total_seconds()/86400;"
            " r2(od[(od.customer_state=='SP')"
            " & (pd.to_datetime(od.order_purchase_timestamp).dt.year==2017)]"
            ".lag.mean())"
        ),
        "gold_answer_type": "number",
        "gold_answer": val,
        "time_anchors": ["2017-Q1", "2017-Q2", "2017-Q3", "2017-Q4"],
        "time_range": {"start": "2017-01-01", "end": "2018-01-01"},
        "event_anchor": None,
        "exclusion_flags": ["nonmonotonic_intervals"],
        "notes": "Days, rounded to 2 decimals.",
    })

    # latency_02: avg approval lag (purchase → approved) in 2017
    mask = od_c.quarter.str.startswith("2017")
    val = r2(od_c.loc[mask, "approval_lag_days"].mean())
    register({
        "qid": "latency_02",
        "category": "latency",
        "sub_type": "inter_event_lag",
        "min_hops": 2,
        "tables_involved": ["orders"],
        "question": (
            "What was the average payment-approval lag (purchase → "
            "approval, in days) for orders placed during 2017?"
        ),
        "sql_query": (
            "SELECT AVG(EXTRACT(epoch FROM (order_approved_at"
            " - order_purchase_timestamp))/86400) FROM orders "
            "WHERE date_part('year', order_purchase_timestamp) = 2017 "
            "  AND order_approved_at >= order_purchase_timestamp "
            "  AND order_delivered_carrier_date >= order_approved_at "
            "  AND order_delivered_customer_date "
            "      >= order_delivered_carrier_date;"
        ),
        "pandas_code": (
            "od = filter_monotonic_delivery(orders);"
            " od['lag'] = (od.order_approved_at "
            "- od.order_purchase_timestamp).dt.total_seconds()/86400;"
            " r2(od[pd.to_datetime(od.order_purchase_timestamp).dt.year==2017]"
            ".lag.mean())"
        ),
        "gold_answer_type": "number",
        "gold_answer": val,
        "time_anchors": ["2017-Q1", "2017-Q2", "2017-Q3", "2017-Q4"],
        "time_range": {"start": "2017-01-01", "end": "2018-01-01"},
        "event_anchor": None,
        "exclusion_flags": ["nonmonotonic_intervals"],
    })

    # latency_03: avg carrier→delivery lag in 2017
    val = r2(od_c.loc[mask, "delivery_lag_days"].mean())
    register({
        "qid": "latency_03",
        "category": "latency",
        "sub_type": "inter_event_lag",
        "min_hops": 2,
        "tables_involved": ["orders"],
        "question": (
            "What was the average carrier-to-customer delivery lag "
            "(carrier pickup → customer delivery, in days) for orders "
            "placed during 2017?"
        ),
        "sql_query": (
            "SELECT AVG(EXTRACT(epoch FROM "
            "(order_delivered_customer_date - order_delivered_carrier_date))"
            "/86400) FROM orders "
            "WHERE date_part('year', order_purchase_timestamp) = 2017 "
            "  AND order_approved_at >= order_purchase_timestamp "
            "  AND order_delivered_carrier_date >= order_approved_at "
            "  AND order_delivered_customer_date "
            "      >= order_delivered_carrier_date;"
        ),
        "pandas_code": (
            "od = filter_monotonic_delivery(orders);"
            " od['lag'] = (od.order_delivered_customer_date "
            "- od.order_delivered_carrier_date).dt.total_seconds()/86400;"
            " r2(od[pd.to_datetime(od.order_purchase_timestamp).dt.year==2017]"
            ".lag.mean())"
        ),
        "gold_answer_type": "number",
        "gold_answer": val,
        "time_anchors": ["2017-Q1", "2017-Q2", "2017-Q3", "2017-Q4"],
        "time_range": {"start": "2017-01-01", "end": "2018-01-01"},
        "event_anchor": None,
        "exclusion_flags": ["nonmonotonic_intervals"],
    })

    # latency_04: avg approved→carrier lag in 2017
    val = r2(od_c.loc[mask, "carrier_lag_days"].mean())
    register({
        "qid": "latency_04",
        "category": "latency",
        "sub_type": "inter_event_lag",
        "min_hops": 2,
        "tables_involved": ["orders"],
        "question": (
            "What was the average approval-to-carrier-handover lag "
            "(approved → carrier pickup, in days) for orders placed "
            "during 2017?"
        ),
        "sql_query": (
            "SELECT AVG(EXTRACT(epoch FROM "
            "(order_delivered_carrier_date - order_approved_at))/86400) "
            "FROM orders "
            "WHERE date_part('year', order_purchase_timestamp) = 2017 "
            "  AND order_approved_at >= order_purchase_timestamp "
            "  AND order_delivered_carrier_date >= order_approved_at "
            "  AND order_delivered_customer_date "
            "      >= order_delivered_carrier_date;"
        ),
        "pandas_code": (
            "od = filter_monotonic_delivery(orders);"
            " od['lag'] = (od.order_delivered_carrier_date "
            "- od.order_approved_at).dt.total_seconds()/86400;"
            " r2(od[pd.to_datetime(od.order_purchase_timestamp).dt.year==2017]"
            ".lag.mean())"
        ),
        "gold_answer_type": "number",
        "gold_answer": val,
        "time_anchors": ["2017-Q1", "2017-Q2", "2017-Q3", "2017-Q4"],
        "time_range": {"start": "2017-01-01", "end": "2018-01-01"},
        "event_anchor": None,
        "exclusion_flags": ["nonmonotonic_intervals"],
    })

    # latency_05: avg purchase-to-delivery in Q4 2017 (peak season, finer)
    mask_q4 = od_c.quarter == "2017-Q4"
    val = r2(od_c.loc[mask_q4, "payment_to_delivery_days"].mean())
    register({
        "qid": "latency_05",
        "category": "latency",
        "sub_type": "inter_event_lag",
        "min_hops": 2,
        "tables_involved": ["orders"],
        "question": (
            "What was the average purchase-to-delivery lag (in days) "
            "for orders placed in Q4 2017?"
        ),
        "sql_query": (
            "SELECT AVG(EXTRACT(epoch FROM (order_delivered_customer_date"
            " - order_purchase_timestamp))/86400) FROM orders "
            "WHERE date_part('year', order_purchase_timestamp) = 2017 "
            "  AND date_part('quarter', order_purchase_timestamp) = 4 "
            "  AND order_approved_at >= order_purchase_timestamp "
            "  AND order_delivered_carrier_date >= order_approved_at "
            "  AND order_delivered_customer_date "
            "      >= order_delivered_carrier_date;"
        ),
        "pandas_code": (
            "od = filter_monotonic_delivery(orders);"
            " od = add_quarter_col(od, 'order_purchase_timestamp');"
            " od['lag'] = (od.order_delivered_customer_date "
            "- od.order_purchase_timestamp).dt.total_seconds()/86400;"
            " r2(od[od.quarter=='2017-Q4'].lag.mean())"
        ),
        "gold_answer_type": "number",
        "gold_answer": val,
        "time_anchors": ["2017-Q4"],
        "time_range": {"start": "2017-10-01", "end": "2018-01-01"},
        "event_anchor": None,
        "exclusion_flags": ["nonmonotonic_intervals"],
    })

    # ----- lag_comparison × 3 -------------------------------------

    # latency_06: SP vs RJ purchase-to-delivery in 2017
    mask_sp = ((od_c.customer_state == "SP")
               & od_c.quarter.str.startswith("2017"))
    mask_rj = ((od_c.customer_state == "RJ")
               & od_c.quarter.str.startswith("2017"))
    sp = r2(od_c.loc[mask_sp, "payment_to_delivery_days"].mean())
    rj = r2(od_c.loc[mask_rj, "payment_to_delivery_days"].mean())
    register({
        "qid": "latency_06",
        "category": "latency",
        "sub_type": "lag_comparison",
        "min_hops": 2,
        "tables_involved": ["orders", "customers"],
        "question": (
            "Compare the average purchase-to-delivery lag (in days) "
            "for customers in São Paulo (SP) versus Rio de Janeiro (RJ) "
            "during 2017."
        ),
        "sql_query": "-- see pandas_code; two means by customer_state",
        "pandas_code": (
            "see build_latency latency_06"
        ),
        "gold_answer_type": "dict",
        "gold_answer": {"SP": sp, "RJ": rj},
        "time_anchors": ["2017-Q1", "2017-Q2", "2017-Q3", "2017-Q4"],
        "time_range": {"start": "2017-01-01", "end": "2018-01-01"},
        "event_anchor": None,
        "exclusion_flags": ["nonmonotonic_intervals"],
    })

    # latency_07: H1 2018 vs H2 2017 purchase-to-delivery (cross-half)
    od_c["half_str"] = (
        pd.to_datetime(od_c.order_purchase_timestamp).dt.strftime("%Y") +
        "-" + pd.to_datetime(od_c.order_purchase_timestamp).dt.month.apply(
            lambda m: "H1" if m <= 6 else "H2"
        )
    )
    h2_17 = r2(od_c.loc[od_c.half_str == "2017-H2",
                        "payment_to_delivery_days"].mean())
    h1_18 = r2(od_c.loc[od_c.half_str == "2018-H1",
                        "payment_to_delivery_days"].mean())
    register({
        "qid": "latency_07",
        "category": "latency",
        "sub_type": "lag_comparison",
        "min_hops": 2,
        "tables_involved": ["orders"],
        "question": (
            "Compare the average purchase-to-delivery lag (in days) "
            "between the second half of 2017 (Jul-Dec) and the first "
            "half of 2018 (Jan-Jun)."
        ),
        "sql_query": "-- see pandas_code; two means by half-year window",
        "pandas_code": "see build_latency latency_07",
        "gold_answer_type": "dict",
        "gold_answer": {"2017-H2": h2_17, "2018-H1": h1_18},
        "time_anchors": ["2017-H2", "2018-H1"],
        "time_range": {"start": "2017-07-01", "end": "2018-07-01"},
        "event_anchor": None,
        "exclusion_flags": ["nonmonotonic_intervals"],
    })

    # latency_08: approval-lag in Q1 vs Q4 2017
    q1 = r2(od_c.loc[od_c.quarter == "2017-Q1", "approval_lag_days"].mean())
    q4 = r2(od_c.loc[od_c.quarter == "2017-Q4", "approval_lag_days"].mean())
    register({
        "qid": "latency_08",
        "category": "latency",
        "sub_type": "lag_comparison",
        "min_hops": 2,
        "tables_involved": ["orders"],
        "question": (
            "Compare the average purchase-to-approval lag (in days) "
            "between Q1 2017 and Q4 2017."
        ),
        "sql_query": "-- see pandas_code; two grouped means",
        "pandas_code": "see build_latency latency_08",
        "gold_answer_type": "dict",
        "gold_answer": {"2017-Q1": q1, "2017-Q4": q4},
        "time_anchors": ["2017-Q1", "2017-Q4"],
        "time_range": None,
        "event_anchor": None,
        "exclusion_flags": ["nonmonotonic_intervals"],
    })

    # ----- before_after_anchor × 2 -------------------------------

    # latency_09: avg purchase-to-delivery for orders 30 days BEFORE vs AFTER Black Friday 2017
    bf = pd.Timestamp("2017-11-24")
    before = (od_c.order_purchase_timestamp >= bf - pd.Timedelta(days=30)) & \
             (od_c.order_purchase_timestamp <  bf)
    after  = (od_c.order_purchase_timestamp >= bf) & \
             (od_c.order_purchase_timestamp <  bf + pd.Timedelta(days=30))
    b = r2(od_c.loc[before, "payment_to_delivery_days"].mean())
    a = r2(od_c.loc[after,  "payment_to_delivery_days"].mean())
    register({
        "qid": "latency_09",
        "category": "latency",
        "sub_type": "before_after_anchor",
        "min_hops": 2,
        "tables_involved": ["orders"],
        "question": (
            "Compare the average purchase-to-delivery lag (in days) "
            "for orders placed in the 30 days before Black Friday 2017 "
            "(2017-10-25 to 2017-11-23) versus the 30 days starting "
            "on Black Friday 2017 (2017-11-24 to 2017-12-23)."
        ),
        "sql_query": "-- see pandas_code; two windowed means",
        "pandas_code": "see build_latency latency_09",
        "gold_answer_type": "dict",
        "gold_answer": {"before_bf2017": b, "after_bf2017": a},
        "time_anchors": ["2017-10", "2017-11", "2017-12"],
        "time_range": {"start": "2017-10-25", "end": "2017-12-24"},
        "event_anchor": "bf2017",
        "exclusion_flags": ["nonmonotonic_intervals"],
    })

    # latency_10: avg purchase-to-delivery during World Cup window vs prior 30 days
    wc_start = pd.Timestamp("2018-06-14")
    wc_end   = pd.Timestamp("2018-07-15")
    during = ((od_c.order_purchase_timestamp >= wc_start)
              & (od_c.order_purchase_timestamp <= wc_end))
    before = ((od_c.order_purchase_timestamp
               >= wc_start - pd.Timedelta(days=30))
              & (od_c.order_purchase_timestamp < wc_start))
    d = r2(od_c.loc[during, "payment_to_delivery_days"].mean())
    b = r2(od_c.loc[before, "payment_to_delivery_days"].mean())
    register({
        "qid": "latency_10",
        "category": "latency",
        "sub_type": "before_after_anchor",
        "min_hops": 2,
        "tables_involved": ["orders"],
        "question": (
            "Compare the average purchase-to-delivery lag (in days) "
            "for orders placed during the FIFA World Cup 2018 window "
            "(2018-06-14 to 2018-07-15) versus the 30 days immediately "
            "before it (2018-05-15 to 2018-06-13)."
        ),
        "sql_query": "-- see pandas_code; two windowed means",
        "pandas_code": "see build_latency latency_10",
        "gold_answer_type": "dict",
        "gold_answer": {"before_worldcup": b, "during_worldcup": d},
        "time_anchors": ["2018-05", "2018-06", "2018-07"],
        "time_range": {"start": "2018-05-15", "end": "2018-07-16"},
        "event_anchor": "worldcup",
        "exclusion_flags": ["nonmonotonic_intervals"],
    })


# ============================================================
# Main
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[2],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--out", type=Path,
                        default=Path("./outputs/temporal_pilot_questions.jsonl"))
    args = parser.parse_args()

    print(f"Loading Olist data from {args.data_dir} ...")
    tables = load_olist(args.data_dir)
    print(f"  {sum(len(df) for df in tables.values()):,} rows across "
          f"{len(tables)} tables.")

    print("\nBuilding temporal question registry ...")
    build_trend(tables)
    print(f"  trend:   {sum(1 for q in QUESTIONS if q['category']=='trend'):>2d}")
    build_drift(tables)
    print(f"  drift:   {sum(1 for q in QUESTIONS if q['category']=='drift'):>2d}")
    build_latency(tables)
    print(f"  latency: {sum(1 for q in QUESTIONS if q['category']=='latency'):>2d}")
    print(f"  TOTAL:   {len(QUESTIONS):>2d}")

    # Quick sanity sweep — any gold_answer that's None or NaN is a
    # broken question, refuse to write the file in that case.
    broken = [q["qid"] for q in QUESTIONS
              if q["gold_answer"] is None
              or (isinstance(q["gold_answer"], float)
                  and pd.isna(q["gold_answer"]))]
    if broken:
        print(f"\nERROR: {len(broken)} questions have a null gold answer:")
        for qid in broken:
            print(f"  - {qid}")
        return 2

    # Sub-type distribution sanity check
    print("\nSub-type distribution:")
    from collections import Counter
    for cat in ("trend", "drift", "latency"):
        sub = Counter(q["sub_type"] for q in QUESTIONS if q["category"] == cat)
        print(f"  {cat}: {dict(sub)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting JSONL to {args.out} ...")
    with args.out.open("w", encoding="utf-8") as f:
        for q in QUESTIONS:
            f.write(json.dumps(q, ensure_ascii=False, default=str) + "\n")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
