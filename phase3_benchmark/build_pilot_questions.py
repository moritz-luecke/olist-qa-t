"""
============================================================
Phase 3.1 — Build Pilot Question Set (60 questions)
============================================================

Definiert 12 Pilot-Fragen pro Kategorie (5 Kategorien × 12 = 60)
und berechnet die Gold-Antworten direkt aus den Olist-CSVs via
pandas (das ist das Referenz-Engine für SQL-equivalente Antworten).

Output:
  outputs/pilot_questions.jsonl  — eine Frage pro Zeile

Schema pro Frage:
  qid                : eindeutige ID, e.g. "factual_01"
  category           : factual | aggregation | comparison
                       | root_cause | global
  min_hops           : 1 | 2 | 3 | 4 | "all"   (Mindest-Hop-Anzahl
                       in der relationalen Struktur)
  tables_involved    : Liste von Olist-Tabellen
  question           : Natural-Language-Frage (Englisch, für Modell)
  sql_query          : SQL-Referenzabfrage (postgres-Dialekt)
  pandas_code        : kompakter pandas-Ausdruck der dieselbe Antwort
                       liefert (Reproduzierbarkeit)
  gold_answer_type   : "string" | "number" | "list" | "rubric"
  gold_answer        : Konkreter Wert ODER null (bei rubric)
  rubric             : (nur bei rubric) Dict mit key_facts +
                       supporting_data
  notes              : optional, freie Notiz

Skalierung:
  Diese Datei ist absichtlich verbose — jede Frage steht für sich,
  damit man später auf 30/Kategorie (=150 total) skalieren kann,
  indem man weitere Einträge in die Listen einfügt.
============================================================
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Data loading
# ============================================================

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


def load_olist(data_dir: Path) -> dict[str, pd.DataFrame]:
    files = {
        "customers":  "olist_customers_dataset.csv",
        "orders":     "olist_orders_dataset.csv",
        "order_items":"olist_order_items_dataset.csv",
        "payments":   "olist_order_payments_dataset.csv",
        "reviews":    "olist_order_reviews_dataset.csv",
        "products":   "olist_products_dataset.csv",
        "sellers":    "olist_sellers_dataset.csv",
        "category_translation": "product_category_name_translation.csv",
    }
    tables = {
        key: pd.read_csv(data_dir / fname,
                          parse_dates=DATE_COLS.get(key, []))
        for key, fname in files.items()
    }
    # Geolocation: one file (Kaggle) or two halves (older local copies)
    single = data_dir / "olist_geolocation_dataset.csv"
    if single.exists():
        tables["geolocation"] = pd.read_csv(single)
    else:
        g1 = pd.read_csv(data_dir / "olist_geolocation_dataset_1.csv")
        g2 = pd.read_csv(data_dir / "olist_geolocation_dataset_2.csv")
        tables["geolocation"] = pd.concat([g1, g2], ignore_index=True)
    # Produkte um EN-Kategorienamen erweitern
    tables["products"] = tables["products"].merge(
        tables["category_translation"],
        on="product_category_name",
        how="left",
    )
    return tables


# ============================================================
# Helpers für Mapping zwischen state ↔ region
# ============================================================

# Brasilianische Großregionen (offiziell IBGE)
STATE_TO_REGION = {
    # Norte
    "AC": "North", "AP": "North", "AM": "North", "PA": "North",
    "RO": "North", "RR": "North", "TO": "North",
    # Nordeste
    "AL": "Northeast", "BA": "Northeast", "CE": "Northeast",
    "MA": "Northeast", "PB": "Northeast", "PE": "Northeast",
    "PI": "Northeast", "RN": "Northeast", "SE": "Northeast",
    # Centro-Oeste
    "DF": "Midwest", "GO": "Midwest", "MT": "Midwest", "MS": "Midwest",
    # Sudeste
    "ES": "Southeast", "MG": "Southeast", "RJ": "Southeast",
    "SP": "Southeast",
    # Sul
    "PR": "South", "RS": "South", "SC": "South",
}


def add_region_col(df: pd.DataFrame, state_col: str) -> pd.DataFrame:
    df = df.copy()
    df["region"] = df[state_col].map(STATE_TO_REGION)
    return df


# ============================================================
# Question registry — wird unten gefüllt
# ============================================================

# Jede registry-Funktion gibt eine Liste von Frage-Dicts zurück.
# Die "compute"-Funktion läuft auf den Tabellen und liefert den
# gold_answer oder das rubric-supporting_data Dict.

QUESTIONS: list[dict] = []


def register(q: dict) -> None:
    QUESTIONS.append(q)


# ------------------------------------------------------------
# Helper: round numeric answers to a stable precision
# ------------------------------------------------------------
def r2(x):
    """Round-to-2 helper, returns native float for JSON-clean output."""
    return None if x is None or (isinstance(x, float) and np.isnan(x)) \
                else round(float(x), 2)


def r4(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) \
                else round(float(x), 4)


# ============================================================
# CATEGORY 1 — FACTUAL  (1-hop, exact answer)
# ============================================================
# Fragen über Eigenschaften EINES konkreten Datensatzes (einer
# Order, eines Produkts, eines Sellers). Verlangen genau einen
# Join-Schritt oder einen direkten Look-up.

# Sample-IDs aus der Daten-Exploration (delivered orders, 1 pay,
# 1 item — saubere Antworten ohne Edge-Cases)
SAMPLE_ORDERS = [
    # Original 5 (used for factual_01..12)
    "53cdb2fc8bc7dce0b6741e2150273451",  # boleto, 1 inst, R$ 141.46, SP
    "47770eb9100c2d0c44946d9cf07ec65d",  # credit_card, 3 inst, R$ 179.12
    "949d5b44dbf5de918fe9c16f97b45f8a",  # credit_card, 1 inst, R$ 72.20
    "ad21c59c0840e6cb83a9ceb5573f8159",  # credit_card, 1 inst, R$ 28.62
    "a4591c265e18cb1dcee52889e2d8acc3",  # credit_card, 6 inst, R$ 175.26
    # 10 additional diverse IDs for factual_13..30
    "c71914b01f0c3167905e81718c53fc84",  # credit_card, SC
    "bf6fa06aa0249987dc5bf7c457f1d556",  # credit_card, SP
    "bb1471d7b9aedd3bafa3f2dd40db59de",  # credit_card, PR
    "8311aa3b9c474639d7f344868c3fe150",  # boleto, RJ
    "3daeecbd44a76530c8e9d0ac1ea4f581",  # debit_card, SP
    "a3de5eef9ddc6e58758fb9d70e5bcad5",  # boleto, DF
    "f2739add905097812d5742efa6236dd8",  # boleto, SP
    "47c1a8c65d33d1a87433294bea40a81b",  # voucher, SP
    "46b911408d5f2c99a745db2cbe109563",  # voucher, ES
    "6eb1728e88c8db6cd1846dc95eb80572",  # voucher, GO
]


def build_factual(t: dict[str, pd.DataFrame]) -> None:
    o, p, it, rv, c, s, pr = (t["orders"], t["payments"], t["order_items"],
                              t["reviews"], t["customers"], t["sellers"],
                              t["products"])

    # 1-3: Payment method, installments, value für drei verschiedene Orders
    for i, oid in enumerate(SAMPLE_ORDERS[:3], start=1):
        pay = p.query("order_id == @oid").iloc[0]
        register({
            "qid": f"factual_{i:02d}",
            "category": "factual",
            "min_hops": 1,
            "tables_involved": ["payments"],
            "question": f"What payment method was used for order {oid}?",
            "sql_query": (
                f"SELECT payment_type FROM order_payments "
                f"WHERE order_id = '{oid}' LIMIT 1;"),
            "pandas_code": (
                f"payments.query(\"order_id == '{oid}'\")"
                f".payment_type.iloc[0]"),
            "gold_answer_type": "string",
            "gold_answer": pay["payment_type"],
        })

    # 4: Installments
    oid = SAMPLE_ORDERS[1]
    pay = p.query("order_id == @oid").iloc[0]
    register({
        "qid": "factual_04",
        "category": "factual",
        "min_hops": 1,
        "tables_involved": ["payments"],
        "question": f"How many installments did the customer use for order {oid}?",
        "sql_query": (
            f"SELECT payment_installments FROM order_payments "
            f"WHERE order_id = '{oid}' LIMIT 1;"),
        "pandas_code": (
            f"int(payments.query(\"order_id == '{oid}'\")"
            f".payment_installments.iloc[0])"),
        "gold_answer_type": "number",
        "gold_answer": int(pay["payment_installments"]),
    })

    # 5: Total payment value
    oid = SAMPLE_ORDERS[0]
    pay = p.query("order_id == @oid").iloc[0]
    register({
        "qid": "factual_05",
        "category": "factual",
        "min_hops": 1,
        "tables_involved": ["payments"],
        "question": f"What was the total payment value for order {oid}, in BRL?",
        "sql_query": (
            f"SELECT SUM(payment_value) FROM order_payments "
            f"WHERE order_id = '{oid}';"),
        "pandas_code": (
            f"payments.query(\"order_id == '{oid}'\")"
            f".payment_value.sum()"),
        "gold_answer_type": "number",
        "gold_answer": r2(pay["payment_value"]),
        "notes": "BRL, rounded to 2 decimals",
    })

    # 6: Order status
    oid = SAMPLE_ORDERS[2]
    order_row = o.query("order_id == @oid").iloc[0]
    register({
        "qid": "factual_06",
        "category": "factual",
        "min_hops": 0,
        "tables_involved": ["orders"],
        "question": f"What is the status of order {oid}?",
        "sql_query": f"SELECT order_status FROM orders WHERE order_id = '{oid}';",
        "pandas_code": (
            f"orders.query(\"order_id == '{oid}'\")"
            f".order_status.iloc[0]"),
        "gold_answer_type": "string",
        "gold_answer": order_row["order_status"],
    })

    # 7: Customer state for an order
    oid = SAMPLE_ORDERS[0]
    cust_id = o.query("order_id == @oid").customer_id.iloc[0]
    state = c.query("customer_id == @cust_id").customer_state.iloc[0]
    register({
        "qid": "factual_07",
        "category": "factual",
        "min_hops": 1,
        "tables_involved": ["orders", "customers"],
        "question": f"In which Brazilian state does the customer of order {oid} live?",
        "sql_query": (
            "SELECT c.customer_state FROM orders o "
            "JOIN customers c ON o.customer_id = c.customer_id "
            f"WHERE o.order_id = '{oid}';"),
        "pandas_code": (
            f"customers.merge(orders.query(\"order_id == '{oid}'\"), "
            f"on='customer_id').customer_state.iloc[0]"),
        "gold_answer_type": "string",
        "gold_answer": state,
    })

    # 8: Seller city for a specific order (single-item order)
    oid = SAMPLE_ORDERS[0]
    sid = it.query("order_id == @oid").seller_id.iloc[0]
    seller_city = t["sellers"].query("seller_id == @sid").seller_city.iloc[0]
    register({
        "qid": "factual_08",
        "category": "factual",
        "min_hops": 2,
        "tables_involved": ["order_items", "sellers"],
        "question": f"In which city is the seller that fulfilled order {oid} located?",
        "sql_query": (
            "SELECT s.seller_city FROM order_items i "
            "JOIN sellers s ON i.seller_id = s.seller_id "
            f"WHERE i.order_id = '{oid}' LIMIT 1;"),
        "pandas_code": (
            f"sellers.merge(order_items.query(\"order_id == '{oid}'\"), "
            f"on='seller_id').seller_city.iloc[0]"),
        "gold_answer_type": "string",
        "gold_answer": seller_city,
    })

    # 9: Product category (English) for the item in an order
    oid = SAMPLE_ORDERS[1]
    pid = it.query("order_id == @oid").product_id.iloc[0]
    cat_en = t["products"].query("product_id == @pid").product_category_name_english.iloc[0]
    register({
        "qid": "factual_09",
        "category": "factual",
        "min_hops": 2,
        "tables_involved": ["order_items", "products", "category_translation"],
        "question": f"What is the product category (English) of the item ordered in {oid}?",
        "sql_query": (
            "SELECT t.product_category_name_english FROM order_items i "
            "JOIN products p ON i.product_id = p.product_id "
            "JOIN product_category_name_translation t "
            "  ON p.product_category_name = t.product_category_name "
            f"WHERE i.order_id = '{oid}' LIMIT 1;"),
        "pandas_code": (
            f"products.merge(category_translation, on='product_category_name', how='left')"
            f".merge(order_items.query(\"order_id == '{oid}'\"), on='product_id')"
            f".product_category_name_english.iloc[0]"),
        "gold_answer_type": "string",
        "gold_answer": str(cat_en) if pd.notna(cat_en) else None,
    })

    # 10: Freight value for the (single) item in an order
    oid = SAMPLE_ORDERS[2]
    freight = float(it.query("order_id == @oid").freight_value.iloc[0])
    register({
        "qid": "factual_10",
        "category": "factual",
        "min_hops": 1,
        "tables_involved": ["order_items"],
        "question": f"What was the freight value for the item in order {oid}, in BRL?",
        "sql_query": (
            f"SELECT freight_value FROM order_items "
            f"WHERE order_id = '{oid}';"),
        "pandas_code": (
            f"order_items.query(\"order_id == '{oid}'\")"
            f".freight_value.iloc[0]"),
        "gold_answer_type": "number",
        "gold_answer": r2(freight),
    })

    # 11: Was an order delivered before the estimated date?
    oid = SAMPLE_ORDERS[3]
    row = o.query("order_id == @oid").iloc[0]
    on_time = bool(row["order_delivered_customer_date"] <= row["order_estimated_delivery_date"])
    register({
        "qid": "factual_11",
        "category": "factual",
        "min_hops": 0,
        "tables_involved": ["orders"],
        "question": f"Was order {oid} delivered on or before its estimated delivery date?",
        "sql_query": (
            "SELECT (order_delivered_customer_date <= order_estimated_delivery_date) "
            f"FROM orders WHERE order_id = '{oid}';"),
        "pandas_code": (
            f"'yes' if (orders.query(\"order_id == '{oid}'\").order_delivered_customer_date.iloc[0]"
            f" <= orders.query(\"order_id == '{oid}'\").order_estimated_delivery_date.iloc[0]) else 'no'"),
        "gold_answer_type": "string",
        "gold_answer": "yes" if on_time else "no",
    })

    # 12: Review score for an order
    oid = SAMPLE_ORDERS[4]
    score = int(rv.query("order_id == @oid").review_score.iloc[0])
    register({
        "qid": "factual_12",
        "category": "factual",
        "min_hops": 1,
        "tables_involved": ["reviews"],
        "question": f"What review score (1–5) did the customer give for order {oid}?",
        "sql_query": (
            f"SELECT review_score FROM order_reviews "
            f"WHERE order_id = '{oid}';"),
        "pandas_code": (
            f"int(reviews.query(\"order_id == '{oid}'\")"
            f".review_score.iloc[0])"),
        "gold_answer_type": "number",
        "gold_answer": score,
    })

    # --- factual_13..30: 18 additional questions across 10 new sample IDs ---

    # 13: Customer city for an order  (1 hop)
    oid = SAMPLE_ORDERS[5]  # c71914...
    cust_id = o.query("order_id == @oid").customer_id.iloc[0]
    city = c.query("customer_id == @cust_id").customer_city.iloc[0]
    register({
        "qid": "factual_13",
        "category": "factual",
        "min_hops": 1,
        "tables_involved": ["orders", "customers"],
        "question": f"In which Brazilian city does the customer of order {oid} live?",
        "sql_query": (
            "SELECT c.customer_city FROM orders o "
            "JOIN customers c ON o.customer_id = c.customer_id "
            f"WHERE o.order_id = '{oid}';"),
        "pandas_code": (
            f"customers.merge(orders.query(\"order_id == '{oid}'\"), "
            f"on='customer_id').customer_city.iloc[0]"),
        "gold_answer_type": "string",
        "gold_answer": str(city),
    })

    # 14: Seller state for the item in an order  (2 hops)
    oid = SAMPLE_ORDERS[6]  # bf6fa0...
    sid = it.query("order_id == @oid").seller_id.iloc[0]
    sstate = t["sellers"].query("seller_id == @sid").seller_state.iloc[0]
    register({
        "qid": "factual_14",
        "category": "factual",
        "min_hops": 2,
        "tables_involved": ["order_items", "sellers"],
        "question": f"In which Brazilian state is the seller that fulfilled order {oid} located?",
        "sql_query": (
            "SELECT s.seller_state FROM order_items i "
            "JOIN sellers s ON i.seller_id = s.seller_id "
            f"WHERE i.order_id = '{oid}' LIMIT 1;"),
        "pandas_code": (
            f"sellers.merge(order_items.query(\"order_id == '{oid}'\"), "
            f"on='seller_id').seller_state.iloc[0]"),
        "gold_answer_type": "string",
        "gold_answer": str(sstate),
    })

    # 15: Purchase date of an order  (0 hops)
    oid = SAMPLE_ORDERS[7]  # bb1471...
    pdate = o.query("order_id == @oid").order_purchase_timestamp.iloc[0]
    register({
        "qid": "factual_15",
        "category": "factual",
        "min_hops": 0,
        "tables_involved": ["orders"],
        "question": f"On which calendar date was order {oid} purchased?",
        "sql_query": (
            "SELECT DATE(order_purchase_timestamp) FROM orders "
            f"WHERE order_id = '{oid}';"),
        "pandas_code": (
            f"orders.query(\"order_id == '{oid}'\")"
            f".order_purchase_timestamp.iloc[0].date()"),
        "gold_answer_type": "string",
        "gold_answer": str(pdate.date()),
        "notes": "ISO-8601 date (YYYY-MM-DD).",
    })

    # 16: Delivery date of an order  (0 hops)
    oid = SAMPLE_ORDERS[8]  # 8311aa...
    ddate = o.query("order_id == @oid").order_delivered_customer_date.iloc[0]
    register({
        "qid": "factual_16",
        "category": "factual",
        "min_hops": 0,
        "tables_involved": ["orders"],
        "question": f"On which calendar date was order {oid} delivered to the customer?",
        "sql_query": (
            "SELECT DATE(order_delivered_customer_date) FROM orders "
            f"WHERE order_id = '{oid}';"),
        "pandas_code": (
            f"orders.query(\"order_id == '{oid}'\")"
            f".order_delivered_customer_date.iloc[0].date()"),
        "gold_answer_type": "string",
        "gold_answer": str(ddate.date()),
    })

    # 17: Number of days between purchase and delivery (0 hops, derived)
    oid = SAMPLE_ORDERS[9]  # 3daeec...
    row = o.query("order_id == @oid").iloc[0]
    days = (row["order_delivered_customer_date"] - row["order_purchase_timestamp"]).total_seconds() / 86400.0
    register({
        "qid": "factual_17",
        "category": "factual",
        "min_hops": 0,
        "tables_involved": ["orders"],
        "question": f"How many days elapsed between purchase and delivery for order {oid}? (round to one decimal)",
        "sql_query": (
            "SELECT EXTRACT(EPOCH FROM "
            "(order_delivered_customer_date - order_purchase_timestamp))/86400.0 "
            f"FROM orders WHERE order_id = '{oid}';"),
        "pandas_code": (
            f"(orders.query(\"order_id == '{oid}'\").order_delivered_customer_date.iloc[0]"
            f" - orders.query(\"order_id == '{oid}'\").order_purchase_timestamp.iloc[0])"
            f".total_seconds()/86400.0"),
        "gold_answer_type": "number",
        "gold_answer": round(days, 1),
    })

    # 18: Price of the item in an order (1 hop)
    oid = SAMPLE_ORDERS[5]
    price = float(it.query("order_id == @oid").price.iloc[0])
    register({
        "qid": "factual_18",
        "category": "factual",
        "min_hops": 1,
        "tables_involved": ["order_items"],
        "question": f"What was the listed price (excluding freight) for the item in order {oid}, in BRL?",
        "sql_query": (
            f"SELECT price FROM order_items WHERE order_id = '{oid}';"),
        "pandas_code": (
            f"order_items.query(\"order_id == '{oid}'\").price.iloc[0]"),
        "gold_answer_type": "number",
        "gold_answer": r2(price),
    })

    # 19: Whether a review comment is present (1 hop)
    oid = SAMPLE_ORDERS[6]
    msg = rv.query("order_id == @oid").review_comment_message.iloc[0] if len(rv.query("order_id == @oid")) else None
    has_text = bool(pd.notna(msg) and len(str(msg).strip()) > 0)
    register({
        "qid": "factual_19",
        "category": "factual",
        "min_hops": 1,
        "tables_involved": ["reviews"],
        "question": f"Did the customer leave a free-text review comment (non-empty) for order {oid}?",
        "sql_query": (
            "SELECT (review_comment_message IS NOT NULL AND TRIM(review_comment_message) <> '') "
            f"FROM order_reviews WHERE order_id = '{oid}';"),
        "pandas_code": (
            f"'yes' if bool(reviews.query(\"order_id == '{oid}'\")"
            f".review_comment_message.notna().iloc[0]) else 'no'"),
        "gold_answer_type": "string",
        "gold_answer": "yes" if has_text else "no",
    })

    # 20: Number of photos for the product in an order (2 hops)
    oid = SAMPLE_ORDERS[7]
    pid = it.query("order_id == @oid").product_id.iloc[0]
    photos = pr.query("product_id == @pid").product_photos_qty.iloc[0]
    register({
        "qid": "factual_20",
        "category": "factual",
        "min_hops": 2,
        "tables_involved": ["order_items", "products"],
        "question": f"How many product photos are listed for the product purchased in order {oid}?",
        "sql_query": (
            "SELECT p.product_photos_qty FROM order_items i "
            "JOIN products p ON i.product_id = p.product_id "
            f"WHERE i.order_id = '{oid}' LIMIT 1;"),
        "pandas_code": (
            f"products.merge(order_items.query(\"order_id == '{oid}'\"), "
            f"on='product_id').product_photos_qty.iloc[0]"),
        "gold_answer_type": "number",
        "gold_answer": int(photos) if pd.notna(photos) else None,
    })

    # 21: Product weight in grams (2 hops)
    oid = SAMPLE_ORDERS[8]
    pid = it.query("order_id == @oid").product_id.iloc[0]
    weight = pr.query("product_id == @pid").product_weight_g.iloc[0]
    register({
        "qid": "factual_21",
        "category": "factual",
        "min_hops": 2,
        "tables_involved": ["order_items", "products"],
        "question": f"What is the weight in grams of the product purchased in order {oid}?",
        "sql_query": (
            "SELECT p.product_weight_g FROM order_items i "
            "JOIN products p ON i.product_id = p.product_id "
            f"WHERE i.order_id = '{oid}' LIMIT 1;"),
        "pandas_code": (
            f"products.merge(order_items.query(\"order_id == '{oid}'\"), "
            f"on='product_id').product_weight_g.iloc[0]"),
        "gold_answer_type": "number",
        "gold_answer": int(weight) if pd.notna(weight) else None,
    })

    # 22: Approval timestamp gap (hours between purchase and approval) (0 hop)
    oid = SAMPLE_ORDERS[9]
    row = o.query("order_id == @oid").iloc[0]
    appr_h = (row["order_approved_at"] - row["order_purchase_timestamp"]).total_seconds() / 3600.0
    register({
        "qid": "factual_22",
        "category": "factual",
        "min_hops": 0,
        "tables_involved": ["orders"],
        "question": f"How many hours passed between the purchase timestamp and the approval timestamp for order {oid}? (round to one decimal)",
        "sql_query": (
            "SELECT EXTRACT(EPOCH FROM "
            "(order_approved_at - order_purchase_timestamp))/3600.0 "
            f"FROM orders WHERE order_id = '{oid}';"),
        "pandas_code": (
            f"round((orders.query(\"order_id == '{oid}'\").order_approved_at.iloc[0]"
            f" - orders.query(\"order_id == '{oid}'\").order_purchase_timestamp.iloc[0])"
            f".total_seconds()/3600.0, 1)"),
        "gold_answer_type": "number",
        "gold_answer": round(appr_h, 1),
    })

    # 23: Did the order include any voucher payment? (1 hop)
    oid = SAMPLE_ORDERS[12]  # 47c1a8... voucher
    pays = p.query("order_id == @oid").payment_type.tolist()
    has_voucher = "voucher" in pays
    register({
        "qid": "factual_23",
        "category": "factual",
        "min_hops": 1,
        "tables_involved": ["payments"],
        "question": f"Did order {oid} include any voucher payment?",
        "sql_query": (
            "SELECT EXISTS (SELECT 1 FROM order_payments "
            f"WHERE order_id = '{oid}' AND payment_type = 'voucher');"),
        "pandas_code": (
            f"'yes' if 'voucher' in payments.query(\"order_id == '{oid}'\")"
            f".payment_type.tolist() else 'no'"),
        "gold_answer_type": "string",
        "gold_answer": "yes" if has_voucher else "no",
    })

    # 24: Number of payment entries for an order (1 hop)
    oid = SAMPLE_ORDERS[13]
    n_pay = len(p.query("order_id == @oid"))
    register({
        "qid": "factual_24",
        "category": "factual",
        "min_hops": 1,
        "tables_involved": ["payments"],
        "question": f"How many separate payment entries are recorded for order {oid}?",
        "sql_query": (
            f"SELECT COUNT(*) FROM order_payments WHERE order_id = '{oid}';"),
        "pandas_code": (
            f"len(payments.query(\"order_id == '{oid}'\"))"),
        "gold_answer_type": "number",
        "gold_answer": int(n_pay),
    })

    # 25: Number of line items in an order (1 hop)
    oid = SAMPLE_ORDERS[14]
    n_items = len(it.query("order_id == @oid"))
    register({
        "qid": "factual_25",
        "category": "factual",
        "min_hops": 1,
        "tables_involved": ["order_items"],
        "question": f"How many line items does order {oid} contain?",
        "sql_query": (
            f"SELECT COUNT(*) FROM order_items WHERE order_id = '{oid}';"),
        "pandas_code": (
            f"len(order_items.query(\"order_id == '{oid}'\"))"),
        "gold_answer_type": "number",
        "gold_answer": int(n_items),
    })

    # 26: Customer's zip-code prefix for an order (1 hop)
    oid = SAMPLE_ORDERS[5]
    cust_id = o.query("order_id == @oid").customer_id.iloc[0]
    zip_prefix = c.query("customer_id == @cust_id").customer_zip_code_prefix.iloc[0]
    register({
        "qid": "factual_26",
        "category": "factual",
        "min_hops": 1,
        "tables_involved": ["orders", "customers"],
        "question": f"What is the 5-digit ZIP-code prefix of the customer who placed order {oid}?",
        "sql_query": (
            "SELECT c.customer_zip_code_prefix FROM orders o "
            "JOIN customers c ON o.customer_id = c.customer_id "
            f"WHERE o.order_id = '{oid}';"),
        "pandas_code": (
            f"customers.merge(orders.query(\"order_id == '{oid}'\"), "
            f"on='customer_id').customer_zip_code_prefix.iloc[0]"),
        "gold_answer_type": "number",
        "gold_answer": int(zip_prefix),
    })

    # 27: Did an order miss the estimated delivery date? (0 hops)
    oid = SAMPLE_ORDERS[6]
    row = o.query("order_id == @oid").iloc[0]
    late = bool(row["order_delivered_customer_date"] > row["order_estimated_delivery_date"])
    register({
        "qid": "factual_27",
        "category": "factual",
        "min_hops": 0,
        "tables_involved": ["orders"],
        "question": f"Was order {oid} delivered after its estimated delivery date (i.e. delivered late)?",
        "sql_query": (
            "SELECT (order_delivered_customer_date > order_estimated_delivery_date) "
            f"FROM orders WHERE order_id = '{oid}';"),
        "pandas_code": (
            f"'yes' if (orders.query(\"order_id == '{oid}'\").order_delivered_customer_date.iloc[0]"
            f" > orders.query(\"order_id == '{oid}'\").order_estimated_delivery_date.iloc[0]) else 'no'"),
        "gold_answer_type": "string",
        "gold_answer": "yes" if late else "no",
    })

    # 28: Product category (English) for the item in another order (2 hops)
    oid = SAMPLE_ORDERS[10]
    pid = it.query("order_id == @oid").product_id.iloc[0]
    cat_en = pr.query("product_id == @pid").product_category_name_english.iloc[0]
    register({
        "qid": "factual_28",
        "category": "factual",
        "min_hops": 2,
        "tables_involved": ["order_items", "products", "category_translation"],
        "question": f"What is the product category (English) of the item ordered in {oid}?",
        "sql_query": (
            "SELECT t.product_category_name_english FROM order_items i "
            "JOIN products p ON i.product_id = p.product_id "
            "JOIN product_category_name_translation t "
            "  ON p.product_category_name = t.product_category_name "
            f"WHERE i.order_id = '{oid}' LIMIT 1;"),
        "pandas_code": (
            f"products.merge(category_translation, on='product_category_name', how='left')"
            f".merge(order_items.query(\"order_id == '{oid}'\"), on='product_id')"
            f".product_category_name_english.iloc[0]"),
        "gold_answer_type": "string",
        "gold_answer": str(cat_en) if pd.notna(cat_en) else None,
    })

    # 29: Payment value for a specific voucher-payment order (1 hop)
    oid = SAMPLE_ORDERS[12]
    pay_val = p.query("order_id == @oid").payment_value.sum()
    register({
        "qid": "factual_29",
        "category": "factual",
        "min_hops": 1,
        "tables_involved": ["payments"],
        "question": f"What is the total payment value (sum across all entries) for order {oid}, in BRL?",
        "sql_query": (
            f"SELECT SUM(payment_value) FROM order_payments "
            f"WHERE order_id = '{oid}';"),
        "pandas_code": (
            f"payments.query(\"order_id == '{oid}'\").payment_value.sum()"),
        "gold_answer_type": "number",
        "gold_answer": r2(pay_val),
    })

    # 30: Number of installments for a credit-card order (1 hop)
    oid = SAMPLE_ORDERS[5]
    inst = p.query("order_id == @oid").payment_installments.iloc[0]
    register({
        "qid": "factual_30",
        "category": "factual",
        "min_hops": 1,
        "tables_involved": ["payments"],
        "question": f"How many installments did the customer use for order {oid}?",
        "sql_query": (
            f"SELECT payment_installments FROM order_payments "
            f"WHERE order_id = '{oid}';"),
        "pandas_code": (
            f"int(payments.query(\"order_id == '{oid}'\")"
            f".payment_installments.iloc[0])"),
        "gold_answer_type": "number",
        "gold_answer": int(inst),
    })


# ============================================================
# CATEGORY 2 — AGGREGATION  (2-3 hops, scalar answer)
# ============================================================

def build_aggregation(t: dict[str, pd.DataFrame]) -> None:
    o, p, it, rv, c, s, pr = (t["orders"], t["payments"], t["order_items"],
                              t["reviews"], t["customers"], t["sellers"],
                              t["products"])

    # 1: Average review score for sellers based in Rio de Janeiro
    rj_sellers = s.query("seller_city == 'rio de janeiro'").seller_id
    rj_items = it[it.seller_id.isin(rj_sellers)]
    rj_orders = rj_items.order_id.unique()
    rj_reviews = rv[rv.order_id.isin(rj_orders)].review_score
    register({
        "qid": "aggregation_01",
        "category": "aggregation",
        "min_hops": 3,
        "tables_involved": ["sellers", "order_items", "orders", "reviews"],
        "question": "What is the average review score (1–5) for orders fulfilled by sellers located in the city of Rio de Janeiro?",
        "sql_query": (
            "SELECT AVG(r.review_score) FROM order_reviews r "
            "JOIN order_items i ON r.order_id = i.order_id "
            "JOIN sellers s ON i.seller_id = s.seller_id "
            "WHERE s.seller_city = 'rio de janeiro';"),
        "pandas_code": (
            "reviews[reviews.order_id.isin("
            "order_items[order_items.seller_id.isin("
            "sellers.query(\"seller_city == 'rio de janeiro'\").seller_id"
            ")].order_id)].review_score.mean()"),
        "gold_answer_type": "number",
        "gold_answer": r2(rj_reviews.mean()),
        "notes": "Rounded to 2 decimals. Note: city names are lowercase in source data.",
    })

    # 2: Most common payment type in São Paulo state
    sp_custs = c.query("customer_state == 'SP'").customer_id
    sp_orders = o[o.customer_id.isin(sp_custs)].order_id
    sp_pay = p[p.order_id.isin(sp_orders)].payment_type
    register({
        "qid": "aggregation_02",
        "category": "aggregation",
        "min_hops": 2,
        "tables_involved": ["customers", "orders", "payments"],
        "question": "What is the most common payment type used by customers living in São Paulo state (SP)?",
        "sql_query": (
            "SELECT payment_type, COUNT(*) AS n FROM order_payments p "
            "JOIN orders o ON p.order_id = o.order_id "
            "JOIN customers c ON o.customer_id = c.customer_id "
            "WHERE c.customer_state = 'SP' "
            "GROUP BY payment_type ORDER BY n DESC LIMIT 1;"),
        "pandas_code": (
            "payments[payments.order_id.isin("
            "orders[orders.customer_id.isin("
            "customers.query(\"customer_state == 'SP'\").customer_id"
            ")].order_id)].payment_type.mode().iloc[0]"),
        "gold_answer_type": "string",
        "gold_answer": sp_pay.mode().iloc[0],
    })

    # 3: Average delivery time (days) for delivered orders in MG
    mg_custs = c.query("customer_state == 'MG'").customer_id
    mg_orders = o[(o.customer_id.isin(mg_custs)) & (o.order_status == "delivered")].copy()
    mg_orders["delivery_days"] = (mg_orders.order_delivered_customer_date -
                                   mg_orders.order_purchase_timestamp).dt.total_seconds() / 86400.0
    register({
        "qid": "aggregation_03",
        "category": "aggregation",
        "min_hops": 1,
        "tables_involved": ["customers", "orders"],
        "question": "What is the average delivery time in days for delivered orders shipped to customers in Minas Gerais (MG)?",
        "sql_query": (
            "SELECT AVG(EXTRACT(EPOCH FROM "
            "(order_delivered_customer_date - order_purchase_timestamp)) / 86400.0) "
            "FROM orders o JOIN customers c ON o.customer_id = c.customer_id "
            "WHERE c.customer_state = 'MG' AND o.order_status = 'delivered';"),
        "pandas_code": (
            "(orders.query(\"order_status == 'delivered'\")"
            ".merge(customers.query(\"customer_state == 'MG'\"), on='customer_id')"
            ".pipe(lambda d: (d.order_delivered_customer_date - "
            "d.order_purchase_timestamp).dt.total_seconds() / 86400.0).mean())"),
        "gold_answer_type": "number",
        "gold_answer": r2(mg_orders["delivery_days"].mean()),
        "notes": "Rounded to 2 decimals. Days are computed including fractional hours.",
    })

    # 4: Total revenue (sum of price) for category bed_bath_table
    bbt_pids = pr.query("product_category_name_english == 'bed_bath_table'").product_id
    bbt_revenue = it[it.product_id.isin(bbt_pids)].price.sum()
    register({
        "qid": "aggregation_04",
        "category": "aggregation",
        "min_hops": 2,
        "tables_involved": ["products", "order_items"],
        "question": "What is the total revenue (sum of item prices, excluding freight) generated by products in the 'bed_bath_table' category, in BRL?",
        "sql_query": (
            "SELECT SUM(i.price) FROM order_items i "
            "JOIN products p ON i.product_id = p.product_id "
            "WHERE p.product_category_name = "
            "(SELECT product_category_name FROM product_category_name_translation "
            "  WHERE product_category_name_english = 'bed_bath_table');"),
        "pandas_code": (
            "order_items[order_items.product_id.isin("
            "products.merge(category_translation, on='product_category_name', how='left')"
            ".query(\"product_category_name_english == "
            "'bed_bath_table'\").product_id)].price.sum()"),
        "gold_answer_type": "number",
        "gold_answer": r2(bbt_revenue),
    })

    # 5: Number of canceled orders in calendar year 2017
    cancelled_2017 = o[(o.order_status == "canceled") &
                       (o.order_purchase_timestamp.dt.year == 2017)]
    register({
        "qid": "aggregation_05",
        "category": "aggregation",
        "min_hops": 0,
        "tables_involved": ["orders"],
        "question": "How many orders were canceled (status = 'canceled') with a purchase date in calendar year 2017?",
        "sql_query": (
            "SELECT COUNT(*) FROM orders "
            "WHERE order_status = 'canceled' AND "
            "EXTRACT(YEAR FROM order_purchase_timestamp) = 2017;"),
        "pandas_code": (
            "len(orders.query(\"order_status == 'canceled'\")"
            ".query(\"order_purchase_timestamp.dt.year == 2017\"))"),
        "gold_answer_type": "number",
        "gold_answer": int(len(cancelled_2017)),
    })

    # 6: Mean number of installments for credit_card payments
    cc_inst = p.query("payment_type == 'credit_card'").payment_installments
    register({
        "qid": "aggregation_06",
        "category": "aggregation",
        "min_hops": 0,
        "tables_involved": ["payments"],
        "question": "On average, how many installments do customers use when paying by credit card?",
        "sql_query": (
            "SELECT AVG(payment_installments) FROM order_payments "
            "WHERE payment_type = 'credit_card';"),
        "pandas_code": (
            "payments.query(\"payment_type == 'credit_card'\")"
            ".payment_installments.mean()"),
        "gold_answer_type": "number",
        "gold_answer": r2(cc_inst.mean()),
    })

    # 7: Number of distinct sellers shipping to RJ customers
    rj_custs = c.query("customer_state == 'RJ'").customer_id
    rj_orders_rj = o[o.customer_id.isin(rj_custs)].order_id
    distinct_sellers_to_rj = it[it.order_id.isin(rj_orders_rj)].seller_id.nunique()
    register({
        "qid": "aggregation_07",
        "category": "aggregation",
        "min_hops": 3,
        "tables_involved": ["customers", "orders", "order_items"],
        "question": "How many distinct sellers have fulfilled at least one order for a customer located in Rio de Janeiro state (RJ)?",
        "sql_query": (
            "SELECT COUNT(DISTINCT i.seller_id) FROM order_items i "
            "JOIN orders o ON i.order_id = o.order_id "
            "JOIN customers c ON o.customer_id = c.customer_id "
            "WHERE c.customer_state = 'RJ';"),
        "pandas_code": (
            "order_items[order_items.order_id.isin("
            "orders[orders.customer_id.isin("
            "customers.query(\"customer_state == 'RJ'\").customer_id"
            ")].order_id)].seller_id.nunique()"),
        "gold_answer_type": "number",
        "gold_answer": int(distinct_sellers_to_rj),
    })

    # 8: Total payment volume (sum) in calendar year 2018
    o_2018 = o[o.order_purchase_timestamp.dt.year == 2018].order_id
    pv_2018 = p[p.order_id.isin(o_2018)].payment_value.sum()
    register({
        "qid": "aggregation_08",
        "category": "aggregation",
        "min_hops": 1,
        "tables_involved": ["orders", "payments"],
        "question": "What is the total payment volume (sum of payment_value) for orders purchased in calendar year 2018, in BRL?",
        "sql_query": (
            "SELECT SUM(p.payment_value) FROM order_payments p "
            "JOIN orders o ON p.order_id = o.order_id "
            "WHERE EXTRACT(YEAR FROM o.order_purchase_timestamp) = 2018;"),
        "pandas_code": (
            "payments[payments.order_id.isin("
            "orders.query(\"order_purchase_timestamp.dt.year == 2018\").order_id"
            ")].payment_value.sum()"),
        "gold_answer_type": "number",
        "gold_answer": r2(pv_2018),
    })

    # 9: Most populous customer state by order count
    state_counts = c.merge(o[["customer_id"]], on="customer_id") \
                      .customer_state.value_counts()
    register({
        "qid": "aggregation_09",
        "category": "aggregation",
        "min_hops": 1,
        "tables_involved": ["customers", "orders"],
        "question": "Which Brazilian state accounts for the most customer orders in the dataset?",
        "sql_query": (
            "SELECT c.customer_state, COUNT(*) FROM orders o "
            "JOIN customers c ON o.customer_id = c.customer_id "
            "GROUP BY c.customer_state ORDER BY COUNT(*) DESC LIMIT 1;"),
        "pandas_code": (
            "customers.merge(orders, on='customer_id')"
            ".customer_state.value_counts().idxmax()"),
        "gold_answer_type": "string",
        "gold_answer": str(state_counts.idxmax()),
        "notes": f"With {int(state_counts.iloc[0]):,} orders.",
    })

    # 10: Average product price for the watches_gifts category
    watch_pids = pr.query("product_category_name_english == 'watches_gifts'").product_id
    watch_mean_price = it[it.product_id.isin(watch_pids)].price.mean()
    register({
        "qid": "aggregation_10",
        "category": "aggregation",
        "min_hops": 2,
        "tables_involved": ["products", "order_items"],
        "question": "What is the average item price (in BRL) for products in the 'watches_gifts' category?",
        "sql_query": (
            "SELECT AVG(i.price) FROM order_items i "
            "JOIN products p ON i.product_id = p.product_id "
            "WHERE p.product_category_name = "
            "(SELECT product_category_name FROM product_category_name_translation "
            "  WHERE product_category_name_english = 'watches_gifts');"),
        "pandas_code": (
            "order_items[order_items.product_id.isin("
            "products.merge(category_translation, on='product_category_name', how='left')"
            ".query(\"product_category_name_english == "
            "'watches_gifts'\").product_id)].price.mean()"),
        "gold_answer_type": "number",
        "gold_answer": r2(watch_mean_price),
    })

    # 11: Number of unique products sold by the seller with the most distinct products
    products_per_seller = it.groupby("seller_id").product_id.nunique()
    register({
        "qid": "aggregation_11",
        "category": "aggregation",
        "min_hops": 1,
        "tables_involved": ["order_items"],
        "question": "How many distinct products are sold by the single seller with the largest product catalogue in the dataset?",
        "sql_query": (
            "SELECT MAX(c) FROM (SELECT COUNT(DISTINCT product_id) AS c "
            "FROM order_items GROUP BY seller_id) sub;"),
        "pandas_code": (
            "order_items.groupby('seller_id').product_id.nunique().max()"),
        "gold_answer_type": "number",
        "gold_answer": int(products_per_seller.max()),
    })

    # 12: Average review score for the 5 largest categories by order volume
    cat_volume = (it.merge(pr[["product_id", "product_category_name_english"]],
                            on="product_id")
                    .product_category_name_english.value_counts())
    top5_cats = cat_volume.head(5).index.tolist()
    top5_pids = pr[pr.product_category_name_english.isin(top5_cats)].product_id
    top5_oids = it[it.product_id.isin(top5_pids)].order_id.unique()
    top5_avg_review = rv[rv.order_id.isin(top5_oids)].review_score.mean()
    register({
        "qid": "aggregation_12",
        "category": "aggregation",
        "min_hops": 3,
        "tables_involved": ["products", "order_items", "orders", "reviews"],
        "question": "Considering the 5 product categories with the largest number of items sold, what is the overall average review score across all orders that include at least one product from these categories?",
        "sql_query": (
            "WITH top5 AS ( "
            "  SELECT t.product_category_name_english AS cat "
            "  FROM order_items i "
            "  JOIN products p ON i.product_id = p.product_id "
            "  JOIN product_category_name_translation t "
            "    ON p.product_category_name = t.product_category_name "
            "  GROUP BY t.product_category_name_english "
            "  ORDER BY COUNT(*) DESC LIMIT 5) "
            "SELECT AVG(r.review_score) FROM order_reviews r "
            "WHERE r.order_id IN ( "
            "  SELECT DISTINCT i.order_id FROM order_items i "
            "  JOIN products p ON i.product_id = p.product_id "
            "  JOIN top5 ON top5.cat = ( "
            "    SELECT product_category_name_english "
            "    FROM product_category_name_translation t2 "
            "    WHERE t2.product_category_name = p.product_category_name));"),
        "pandas_code": "<see registry source — multi-step pandas>",
        "gold_answer_type": "number",
        "gold_answer": r2(top5_avg_review),
        "notes": f"Top-5 categories used: {top5_cats}",
    })

    # --- aggregation_13..30: 18 additional questions ---

    # 13: Total revenue in calendar year 2017
    o_2017 = o[o.order_purchase_timestamp.dt.year == 2017].order_id
    rev_2017 = it[it.order_id.isin(o_2017)].price.sum()
    register({
        "qid": "aggregation_13",
        "category": "aggregation",
        "min_hops": 1,
        "tables_involved": ["orders", "order_items"],
        "question": "What is the total revenue (sum of item prices, excluding freight) for orders purchased in calendar year 2017, in BRL?",
        "sql_query": (
            "SELECT SUM(i.price) FROM order_items i "
            "JOIN orders o ON i.order_id = o.order_id "
            "WHERE EXTRACT(YEAR FROM o.order_purchase_timestamp) = 2017;"),
        "pandas_code": (
            "order_items[order_items.order_id.isin("
            "orders.query('order_purchase_timestamp.dt.year == 2017').order_id"
            ")].price.sum()"),
        "gold_answer_type": "number",
        "gold_answer": r2(rev_2017),
    })

    # 14: Total freight volume across the entire dataset
    total_freight = it.freight_value.sum()
    register({
        "qid": "aggregation_14",
        "category": "aggregation",
        "min_hops": 0,
        "tables_involved": ["order_items"],
        "question": "What is the total freight volume (sum of freight_value across all order items) in the entire dataset, in BRL?",
        "sql_query": "SELECT SUM(freight_value) FROM order_items;",
        "pandas_code": "order_items.freight_value.sum()",
        "gold_answer_type": "number",
        "gold_answer": r2(total_freight),
    })

    # 15: Number of distinct customer cities served
    n_cust_cities = c.customer_city.nunique()
    register({
        "qid": "aggregation_15",
        "category": "aggregation",
        "min_hops": 0,
        "tables_involved": ["customers"],
        "question": "How many distinct customer cities are recorded in the Olist customer base?",
        "sql_query": "SELECT COUNT(DISTINCT customer_city) FROM customers;",
        "pandas_code": "customers.customer_city.nunique()",
        "gold_answer_type": "number",
        "gold_answer": int(n_cust_cities),
    })

    # 16: Number of distinct seller cities
    n_seller_cities = s.seller_city.nunique()
    register({
        "qid": "aggregation_16",
        "category": "aggregation",
        "min_hops": 0,
        "tables_involved": ["sellers"],
        "question": "How many distinct cities are home to at least one Olist seller?",
        "sql_query": "SELECT COUNT(DISTINCT seller_city) FROM sellers;",
        "pandas_code": "sellers.seller_city.nunique()",
        "gold_answer_type": "number",
        "gold_answer": int(n_seller_cities),
    })

    # 17: Mean delivery time for delivered orders (in days)
    o_del = o.query("order_status == 'delivered' and order_delivered_customer_date.notna()").copy()
    o_del["d"] = (o_del.order_delivered_customer_date - o_del.order_purchase_timestamp).dt.total_seconds() / 86400.0
    register({
        "qid": "aggregation_17",
        "category": "aggregation",
        "min_hops": 0,
        "tables_involved": ["orders"],
        "question": "What is the mean delivery time in days for all orders with status = 'delivered'?",
        "sql_query": (
            "SELECT AVG(EXTRACT(EPOCH FROM "
            "(order_delivered_customer_date - order_purchase_timestamp))/86400.0) "
            "FROM orders WHERE order_status = 'delivered';"),
        "pandas_code": (
            "((orders.query(\"order_status == 'delivered'\")"
            ".order_delivered_customer_date - orders.query(\"order_status == 'delivered'\")"
            ".order_purchase_timestamp).dt.total_seconds()/86400.0).mean()"),
        "gold_answer_type": "number",
        "gold_answer": r2(o_del["d"].mean()),
    })

    # 18: Median review score across all reviews
    median_review = rv.review_score.median()
    register({
        "qid": "aggregation_18",
        "category": "aggregation",
        "min_hops": 0,
        "tables_involved": ["reviews"],
        "question": "What is the median review score (1–5) across all reviews in the dataset?",
        "sql_query": (
            "SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY review_score) "
            "FROM order_reviews;"),
        "pandas_code": "reviews.review_score.median()",
        "gold_answer_type": "number",
        "gold_answer": float(median_review),
    })

    # 19: Number of canceled orders in calendar year 2018
    can_2018 = o[(o.order_status == "canceled") &
                  (o.order_purchase_timestamp.dt.year == 2018)]
    register({
        "qid": "aggregation_19",
        "category": "aggregation",
        "min_hops": 0,
        "tables_involved": ["orders"],
        "question": "How many orders with status = 'canceled' have a purchase date in calendar year 2018?",
        "sql_query": (
            "SELECT COUNT(*) FROM orders "
            "WHERE order_status = 'canceled' AND "
            "EXTRACT(YEAR FROM order_purchase_timestamp) = 2018;"),
        "pandas_code": (
            "len(orders.query(\"order_status == 'canceled'\")"
            ".query('order_purchase_timestamp.dt.year == 2018'))"),
        "gold_answer_type": "number",
        "gold_answer": int(len(can_2018)),
    })

    # 20: Mean payment_value across all payment entries
    mean_pv = p.payment_value.mean()
    register({
        "qid": "aggregation_20",
        "category": "aggregation",
        "min_hops": 0,
        "tables_involved": ["payments"],
        "question": "What is the mean payment value (in BRL) across all payment entries in the dataset?",
        "sql_query": "SELECT AVG(payment_value) FROM order_payments;",
        "pandas_code": "payments.payment_value.mean()",
        "gold_answer_type": "number",
        "gold_answer": r2(mean_pv),
    })

    # 21: Mean product weight for category 'furniture_decor'
    fd_pids = pr.query("product_category_name_english == 'furniture_decor'").product_id
    fd_weight = pr[pr.product_id.isin(fd_pids)].product_weight_g.mean()
    register({
        "qid": "aggregation_21",
        "category": "aggregation",
        "min_hops": 0,
        "tables_involved": ["products"],
        "question": "What is the average product weight in grams for products in the 'furniture_decor' category?",
        "sql_query": (
            "SELECT AVG(product_weight_g) FROM products "
            "WHERE product_category_name = "
            "(SELECT product_category_name FROM product_category_name_translation "
            "  WHERE product_category_name_english = 'furniture_decor');"),
        "pandas_code": (
            "products.merge(category_translation, on='product_category_name', how='left')"
            ".query(\"product_category_name_english == "
            "'furniture_decor'\").product_weight_g.mean()"),
        "gold_answer_type": "number",
        "gold_answer": r2(fd_weight),
    })

    # 22: Number of unique products that were sold at least once in 2017
    sold_2017 = it[it.order_id.isin(o_2017)].product_id.nunique()
    register({
        "qid": "aggregation_22",
        "category": "aggregation",
        "min_hops": 1,
        "tables_involved": ["orders", "order_items"],
        "question": "How many distinct products were sold at least once in orders purchased during calendar year 2017?",
        "sql_query": (
            "SELECT COUNT(DISTINCT i.product_id) FROM order_items i "
            "JOIN orders o ON i.order_id = o.order_id "
            "WHERE EXTRACT(YEAR FROM o.order_purchase_timestamp) = 2017;"),
        "pandas_code": (
            "order_items[order_items.order_id.isin("
            "orders.query('order_purchase_timestamp.dt.year == 2017').order_id"
            ")].product_id.nunique()"),
        "gold_answer_type": "number",
        "gold_answer": int(sold_2017),
    })

    # 23: Maximum number of installments observed
    max_inst = p.payment_installments.max()
    register({
        "qid": "aggregation_23",
        "category": "aggregation",
        "min_hops": 0,
        "tables_involved": ["payments"],
        "question": "What is the largest number of installments recorded for any single payment entry?",
        "sql_query": "SELECT MAX(payment_installments) FROM order_payments;",
        "pandas_code": "payments.payment_installments.max()",
        "gold_answer_type": "number",
        "gold_answer": int(max_inst),
    })

    # 24: Mean number of items per order
    items_per_order = it.groupby("order_id").size()
    mean_items = items_per_order.mean()
    register({
        "qid": "aggregation_24",
        "category": "aggregation",
        "min_hops": 0,
        "tables_involved": ["order_items"],
        "question": "What is the average number of line items per order, computed over orders that have at least one item?",
        "sql_query": (
            "SELECT AVG(n) FROM (SELECT order_id, COUNT(*) AS n "
            "FROM order_items GROUP BY order_id) sub;"),
        "pandas_code": "order_items.groupby('order_id').size().mean()",
        "gold_answer_type": "number",
        "gold_answer": r2(mean_items),
    })

    # 25: Number of distinct product categories (English)
    n_cats = pr.product_category_name_english.nunique()
    register({
        "qid": "aggregation_25",
        "category": "aggregation",
        "min_hops": 1,
        "tables_involved": ["products", "category_translation"],
        "question": "How many distinct product categories (English names) are present in the Olist product catalogue?",
        "sql_query": (
            "SELECT COUNT(DISTINCT t.product_category_name_english) FROM products p "
            "JOIN product_category_name_translation t "
            "  ON p.product_category_name = t.product_category_name;"),
        "pandas_code": (
            "products.merge(category_translation, on='product_category_name', how='left')"
            ".product_category_name_english.nunique()"),
        "gold_answer_type": "number",
        "gold_answer": int(n_cats),
    })

    # 26: Total payment volume for boleto payments
    boleto_vol = p.query("payment_type == 'boleto'").payment_value.sum()
    register({
        "qid": "aggregation_26",
        "category": "aggregation",
        "min_hops": 0,
        "tables_involved": ["payments"],
        "question": "What is the total payment volume (in BRL) processed via boleto?",
        "sql_query": (
            "SELECT SUM(payment_value) FROM order_payments "
            "WHERE payment_type = 'boleto';"),
        "pandas_code": (
            "payments.query(\"payment_type == 'boleto'\").payment_value.sum()"),
        "gold_answer_type": "number",
        "gold_answer": r2(boleto_vol),
    })

    # 27: Average review score for orders shipped to SP customers
    sp_oids = o[o.customer_id.isin(c.query("customer_state == 'SP'").customer_id)].order_id
    sp_review = rv[rv.order_id.isin(sp_oids)].review_score.mean()
    register({
        "qid": "aggregation_27",
        "category": "aggregation",
        "min_hops": 2,
        "tables_involved": ["customers", "orders", "reviews"],
        "question": "What is the average review score (1–5) for orders shipped to customers in São Paulo state (SP)?",
        "sql_query": (
            "SELECT AVG(r.review_score) FROM order_reviews r "
            "JOIN orders o ON r.order_id = o.order_id "
            "JOIN customers c ON o.customer_id = c.customer_id "
            "WHERE c.customer_state = 'SP';"),
        "pandas_code": (
            "reviews[reviews.order_id.isin(orders[orders.customer_id.isin("
            "customers.query(\"customer_state == 'SP'\").customer_id)].order_id)]"
            ".review_score.mean()"),
        "gold_answer_type": "number",
        "gold_answer": r2(sp_review),
    })

    # 28: Number of orders with at least one voucher payment entry
    voucher_oids = p.query("payment_type == 'voucher'").order_id.unique()
    register({
        "qid": "aggregation_28",
        "category": "aggregation",
        "min_hops": 0,
        "tables_involved": ["payments"],
        "question": "How many distinct orders include at least one payment entry of type 'voucher'?",
        "sql_query": (
            "SELECT COUNT(DISTINCT order_id) FROM order_payments "
            "WHERE payment_type = 'voucher';"),
        "pandas_code": (
            "payments.query(\"payment_type == 'voucher'\").order_id.nunique()"),
        "gold_answer_type": "number",
        "gold_answer": int(len(voucher_oids)),
    })

    # 29: Mean order value (sum payment_value) for orders shipped to North region
    cust_reg_t = add_region_col(c, "customer_state")
    north_custs = cust_reg_t.query("region == 'North'").customer_id
    north_oids = o[o.customer_id.isin(north_custs)].order_id
    north_pay = p[p.order_id.isin(north_oids)].groupby("order_id").payment_value.sum().mean()
    register({
        "qid": "aggregation_29",
        "category": "aggregation",
        "min_hops": 2,
        "tables_involved": ["customers", "orders", "payments"],
        "question": "What is the mean total payment value (in BRL) for orders shipped to customers in Brazil's North region?",
        "sql_query": (
            "WITH north_orders AS (SELECT o.order_id FROM orders o "
            " JOIN customers c ON o.customer_id = c.customer_id "
            " WHERE c.customer_state IN ('AC','AP','AM','PA','RO','RR','TO')) "
            "SELECT AVG(t) FROM (SELECT order_id, SUM(payment_value) AS t "
            "FROM order_payments WHERE order_id IN (SELECT order_id FROM north_orders) "
            "GROUP BY order_id) sub;"),
        "pandas_code": "<see add_region_col() + groupby>",
        "gold_answer_type": "number",
        "gold_answer": r2(north_pay),
    })

    # 30: Number of orders purchased in November 2017 (Black Friday spike)
    nov_2017 = o[(o.order_purchase_timestamp.dt.year == 2017) &
                  (o.order_purchase_timestamp.dt.month == 11)]
    register({
        "qid": "aggregation_30",
        "category": "aggregation",
        "min_hops": 0,
        "tables_involved": ["orders"],
        "question": "How many orders were purchased in November 2017 (the Brazilian Black Friday month)?",
        "sql_query": (
            "SELECT COUNT(*) FROM orders "
            "WHERE EXTRACT(YEAR FROM order_purchase_timestamp) = 2017 "
            "AND EXTRACT(MONTH FROM order_purchase_timestamp) = 11;"),
        "pandas_code": (
            "len(orders.query('order_purchase_timestamp.dt.year == 2017 "
            "and order_purchase_timestamp.dt.month == 11'))"),
        "gold_answer_type": "number",
        "gold_answer": int(len(nov_2017)),
    })


# ============================================================
# CATEGORY 3 — COMPARISON  (3-4 hops, scalar + structured)
# ============================================================
# Geben eine numerische Antwort + eine Rubrik mit den verglichenen
# Größen, damit LLM-Judging möglich ist.

def build_comparison(t: dict[str, pd.DataFrame]) -> None:
    o, p, it, rv, c, s, pr = (t["orders"], t["payments"], t["order_items"],
                              t["reviews"], t["customers"], t["sellers"],
                              t["products"])

    # 1: Compare avg delivery time: SP vs RJ vs MG (top-3 customer states)
    def avg_delivery(state):
        custs = c.query("customer_state == @state").customer_id
        sub = o[(o.customer_id.isin(custs)) & (o.order_status == "delivered")].copy()
        sub["d"] = (sub.order_delivered_customer_date - sub.order_purchase_timestamp).dt.total_seconds() / 86400.0
        return r2(sub.d.mean())
    deliv = {s_: avg_delivery(s_) for s_ in ["SP", "RJ", "MG"]}
    register({
        "qid": "comparison_01",
        "category": "comparison",
        "min_hops": 1,
        "tables_involved": ["customers", "orders"],
        "question": "Which of the three largest customer states by population — São Paulo (SP), Rio de Janeiro (RJ), or Minas Gerais (MG) — has the fastest average delivery time for delivered orders, and what are the three values?",
        "sql_query": (
            "SELECT c.customer_state, AVG(EXTRACT(EPOCH FROM "
            "(o.order_delivered_customer_date - o.order_purchase_timestamp))/86400.0) "
            "FROM orders o JOIN customers c ON o.customer_id = c.customer_id "
            "WHERE o.order_status = 'delivered' AND c.customer_state IN ('SP','RJ','MG') "
            "GROUP BY c.customer_state;"),
        "pandas_code": "<see avg_delivery() in registry source>",
        "gold_answer_type": "rubric",
        "gold_answer": min(deliv, key=deliv.get),
        "rubric": {
            "key_facts": [
                f"SP has the fastest mean delivery time at {deliv['SP']} days.",
                f"RJ averages {deliv['RJ']} days.",
                f"MG averages {deliv['MG']} days.",
                "The correct ranking is SP < MG < RJ.",
            ],
            "supporting_data": deliv,
        },
    })

    # 2: Top-3 product categories by total revenue
    cat_rev = (it.merge(pr[["product_id", "product_category_name_english"]],
                         on="product_id")
                 .groupby("product_category_name_english").price.sum()
                 .sort_values(ascending=False))
    top3_rev = cat_rev.head(3)
    register({
        "qid": "comparison_02",
        "category": "comparison",
        "min_hops": 2,
        "tables_involved": ["products", "order_items"],
        "question": "What are the three product categories with the highest total revenue (in BRL, excluding freight), and what is each category's revenue?",
        "sql_query": (
            "SELECT t.product_category_name_english, SUM(i.price) AS rev "
            "FROM order_items i "
            "JOIN products p ON i.product_id = p.product_id "
            "JOIN product_category_name_translation t "
            "  ON p.product_category_name = t.product_category_name "
            "GROUP BY t.product_category_name_english "
            "ORDER BY rev DESC LIMIT 3;"),
        "pandas_code": (
            "(order_items.merge(products[['product_id','product_category_name_english']],"
            " on='product_id').groupby('product_category_name_english')"
            ".price.sum().nlargest(3))"),
        "gold_answer_type": "rubric",
        "gold_answer": top3_rev.index.tolist(),
        "rubric": {
            "key_facts": [
                f"#1: {top3_rev.index[0]} with R$ {top3_rev.iloc[0]:,.2f}",
                f"#2: {top3_rev.index[1]} with R$ {top3_rev.iloc[1]:,.2f}",
                f"#3: {top3_rev.index[2]} with R$ {top3_rev.iloc[2]:,.2f}",
            ],
            "supporting_data": {k: r2(v) for k, v in top3_rev.items()},
        },
    })

    # 3: Compare avg review score: credit_card vs boleto
    p_credit = p.query("payment_type == 'credit_card'").order_id
    p_boleto = p.query("payment_type == 'boleto'").order_id
    rev_cc = rv[rv.order_id.isin(p_credit)].review_score.mean()
    rev_bo = rv[rv.order_id.isin(p_boleto)].review_score.mean()
    diff_cc_bo = abs(rev_cc - rev_bo)
    register({
        "qid": "comparison_03",
        "category": "comparison",
        "min_hops": 2,
        "tables_involved": ["payments", "reviews"],
        "question": "Do customers paying by credit card or by boleto leave better reviews on average? Report both means.",
        "sql_query": (
            "SELECT p.payment_type, AVG(r.review_score) FROM order_payments p "
            "JOIN order_reviews r ON p.order_id = r.order_id "
            "WHERE p.payment_type IN ('credit_card','boleto') "
            "GROUP BY p.payment_type;"),
        "pandas_code": (
            "{'credit_card': reviews[reviews.order_id.isin("
            "payments.query(\"payment_type=='credit_card'\").order_id)]"
            ".review_score.mean(), 'boleto': ...}"),
        "gold_answer_type": "rubric",
        "gold_answer": "tie" if diff_cc_bo < 0.05 else ("credit_card" if rev_cc > rev_bo else "boleto"),
        "rubric": {
            "key_facts": [
                f"Credit card mean review: {round(rev_cc, 4)} (n={76313})",
                f"Boleto mean review: {round(rev_bo, 4)} (n={19762})",
                f"Difference: {round(diff_cc_bo, 4)} -- this is a practical TIE "
                f"(<0.05 on a 1-5 scale, no meaningful difference between methods).",
                "A correct answer recognises that no meaningful difference exists.",
            ],
            "supporting_data": {"credit_card": round(rev_cc, 4),
                                "boleto": round(rev_bo, 4),
                                "difference": round(diff_cc_bo, 4)},
        },
        "notes": "Pilot-detected near-tie -- useful negative case for evaluating model calibration; should be retained or replaced in final 150-question set based on coverage goals.",
    })

    # 4: Compare freight-to-price ratio: North region vs Southeast region
    cust_reg = add_region_col(c, "customer_state")
    o_reg = o.merge(cust_reg[["customer_id", "region"]], on="customer_id")
    item_reg = it.merge(o_reg[["order_id", "region"]], on="order_id")
    ratio_by_region = item_reg.assign(
        ratio=item_reg.freight_value / item_reg.price.replace(0, np.nan)
    ).groupby("region").ratio.mean()
    register({
        "qid": "comparison_04",
        "category": "comparison",
        "min_hops": 3,
        "tables_involved": ["customers", "orders", "order_items"],
        "question": "Compare the average freight-to-price ratio for items shipped to customers in the North region versus the Southeast region. Which region has higher relative freight costs?",
        "sql_query": (
            "-- region derived from customer_state via IBGE mapping. "
            "Average ratio = AVG(freight_value / price) over items joined "
            "to orders and customers."),
        "pandas_code": "<see add_region_col() in registry source>",
        "gold_answer_type": "rubric",
        "gold_answer": "North" if ratio_by_region.get("North", 0) > ratio_by_region.get("Southeast", 0) else "Southeast",
        "rubric": {
            "key_facts": [
                f"North region avg ratio: {r4(ratio_by_region.get('North'))}",
                f"Southeast region avg ratio: {r4(ratio_by_region.get('Southeast'))}",
                "North is expected to have a higher ratio due to long shipping distances from the Southeast supply base.",
            ],
            "supporting_data": {k: r4(v) for k, v in ratio_by_region.items()},
        },
    })

    # 5: Cancellation rate by payment type (credit_card vs boleto)
    o_with_pay = o.merge(p[["order_id", "payment_type"]].drop_duplicates("order_id"),
                          on="order_id")
    cancel_by_pay = (
        o_with_pay.groupby("payment_type")
        .order_status.apply(lambda s: (s == "canceled").mean())
    )
    register({
        "qid": "comparison_05",
        "category": "comparison",
        "min_hops": 1,
        "tables_involved": ["orders", "payments"],
        "question": "Compare the cancellation rate for orders paid by credit card versus boleto. Express both as percentages.",
        "sql_query": (
            "SELECT p.payment_type, "
            "100.0 * SUM(CASE WHEN o.order_status='canceled' THEN 1 ELSE 0 END) / COUNT(*) "
            "FROM orders o JOIN order_payments p ON o.order_id = p.order_id "
            "WHERE p.payment_type IN ('credit_card','boleto') GROUP BY p.payment_type;"),
        "pandas_code": "<groupby payment_type, mean(status=='canceled')>",
        "gold_answer_type": "rubric",
        "gold_answer": "credit_card" if cancel_by_pay.get("credit_card", 0) > cancel_by_pay.get("boleto", 0) else "boleto",
        "rubric": {
            "key_facts": [
                f"Credit-card cancellation rate: {r4(cancel_by_pay.get('credit_card')) * 100 if cancel_by_pay.get('credit_card') is not None else None}%",
                f"Boleto cancellation rate: {r4(cancel_by_pay.get('boleto')) * 100 if cancel_by_pay.get('boleto') is not None else None}%",
            ],
            "supporting_data": {k: r4(v * 100) for k, v in cancel_by_pay.items()},
        },
    })

    # 6: Top-3 sellers by revenue
    seller_rev = it.groupby("seller_id").price.sum().sort_values(ascending=False)
    top3_sellers = seller_rev.head(3)
    register({
        "qid": "comparison_06",
        "category": "comparison",
        "min_hops": 1,
        "tables_involved": ["order_items"],
        "question": "Identify the top 3 sellers by total revenue (sum of item prices, excluding freight). Report each seller's revenue in BRL.",
        "sql_query": (
            "SELECT seller_id, SUM(price) AS rev FROM order_items "
            "GROUP BY seller_id ORDER BY rev DESC LIMIT 3;"),
        "pandas_code": (
            "order_items.groupby('seller_id').price.sum().nlargest(3)"),
        "gold_answer_type": "rubric",
        "gold_answer": top3_sellers.index.tolist(),
        "rubric": {
            "key_facts": [
                f"#{i+1}: seller {sid} with R$ {top3_sellers.iloc[i]:,.2f}"
                for i, sid in enumerate(top3_sellers.index)
            ],
            "supporting_data": {sid: r2(rev_) for sid, rev_ in top3_sellers.items()},
        },
    })

    # 7: On-time delivery rate: SP vs RS
    def ontime_rate(state):
        custs = c.query("customer_state == @state").customer_id
        sub = o[(o.customer_id.isin(custs)) & (o.order_status == "delivered")
                & o.order_delivered_customer_date.notna()
                & o.order_estimated_delivery_date.notna()]
        if len(sub) == 0:
            return None
        return r4((sub.order_delivered_customer_date <= sub.order_estimated_delivery_date).mean())
    rates = {st: ontime_rate(st) for st in ["SP", "RS"]}
    register({
        "qid": "comparison_07",
        "category": "comparison",
        "min_hops": 1,
        "tables_involved": ["customers", "orders"],
        "question": "Compare the on-time delivery rate (share of delivered orders received on or before the estimated date) for customers in São Paulo (SP) versus Rio Grande do Sul (RS). Report both as percentages.",
        "sql_query": (
            "SELECT c.customer_state, "
            "AVG(CASE WHEN order_delivered_customer_date <= order_estimated_delivery_date "
            "THEN 1.0 ELSE 0.0 END) FROM orders o "
            "JOIN customers c ON o.customer_id = c.customer_id "
            "WHERE c.customer_state IN ('SP','RS') AND o.order_status = 'delivered' "
            "GROUP BY c.customer_state;"),
        "pandas_code": "<see ontime_rate() in registry source>",
        "gold_answer_type": "rubric",
        "gold_answer": max(rates, key=lambda k: rates[k] or 0),
        "rubric": {
            "key_facts": [
                f"SP on-time rate: {(rates['SP'] or 0)*100:.2f}%",
                f"RS on-time rate: {(rates['RS'] or 0)*100:.2f}%",
            ],
            "supporting_data": {k: (v*100 if v is not None else None) for k, v in rates.items()},
        },
    })

    # 8: Compare avg order value: single-item vs multi-item orders
    items_per_order = it.groupby("order_id").size()
    single_oids = items_per_order[items_per_order == 1].index
    multi_oids  = items_per_order[items_per_order > 1].index
    pay_single = p[p.order_id.isin(single_oids)].groupby("order_id").payment_value.sum().mean()
    pay_multi  = p[p.order_id.isin(multi_oids)].groupby("order_id").payment_value.sum().mean()
    register({
        "qid": "comparison_08",
        "category": "comparison",
        "min_hops": 2,
        "tables_involved": ["order_items", "payments"],
        "question": "Compare the average total payment value of orders containing a single item versus orders containing multiple items.",
        "sql_query": (
            "WITH counts AS (SELECT order_id, COUNT(*) AS n "
            "                FROM order_items GROUP BY order_id) "
            "SELECT (CASE WHEN n=1 THEN 'single' ELSE 'multi' END) AS bucket, "
            "AVG(t) FROM (SELECT c.order_id, c.n, "
            "(SELECT SUM(payment_value) FROM order_payments p "
            " WHERE p.order_id = c.order_id) AS t FROM counts c) sub "
            "GROUP BY bucket;"),
        "pandas_code": "<see registry source>",
        "gold_answer_type": "rubric",
        "gold_answer": "multi" if pay_multi > pay_single else "single",
        "rubric": {
            "key_facts": [
                f"Single-item orders mean payment: R$ {r2(pay_single)}",
                f"Multi-item orders mean payment: R$ {r2(pay_multi)}",
            ],
            "supporting_data": {"single_item_mean": r2(pay_single),
                                "multi_item_mean": r2(pay_multi)},
        },
    })

    # 9: Top-3 categories by avg review score (with >= 200 orders, to avoid noise)
    cat_review = (rv.merge(it[["order_id", "product_id"]], on="order_id")
                    .merge(pr[["product_id", "product_category_name_english"]], on="product_id")
                    .groupby("product_category_name_english")
                    .review_score.agg(["mean", "count"])
                    .query("count >= 200")
                    .sort_values("mean", ascending=False))
    top3_review_cats = cat_review.head(3)
    register({
        "qid": "comparison_09",
        "category": "comparison",
        "min_hops": 3,
        "tables_involved": ["reviews", "order_items", "products"],
        "question": "Among product categories with at least 200 reviewed orders, which 3 categories have the highest average review score?",
        "sql_query": (
            "WITH cat_rev AS ( "
            "  SELECT t.product_category_name_english AS cat, r.review_score "
            "  FROM order_reviews r "
            "  JOIN order_items i ON r.order_id = i.order_id "
            "  JOIN products p ON i.product_id = p.product_id "
            "  JOIN product_category_name_translation t "
            "    ON p.product_category_name = t.product_category_name) "
            "SELECT cat, AVG(review_score) AS m FROM cat_rev "
            "GROUP BY cat HAVING COUNT(*) >= 200 ORDER BY m DESC LIMIT 3;"),
        "pandas_code": "<groupby + agg + filter>",
        "gold_answer_type": "rubric",
        "gold_answer": top3_review_cats.index.tolist(),
        "rubric": {
            "key_facts": [
                f"#{i+1}: {cat} with mean review {top3_review_cats['mean'].iloc[i]:.3f} "
                f"({int(top3_review_cats['count'].iloc[i])} orders)"
                for i, cat in enumerate(top3_review_cats.index)
            ],
            "supporting_data": {
                cat: {"mean_review": r4(top3_review_cats["mean"].iloc[i]),
                      "n_orders": int(top3_review_cats["count"].iloc[i])}
                for i, cat in enumerate(top3_review_cats.index)
            },
        },
    })

    # 10: Compare item prices: top-10% by price vs bottom-10%
    p10 = it.price.quantile(0.10)
    p90 = it.price.quantile(0.90)
    register({
        "qid": "comparison_10",
        "category": "comparison",
        "min_hops": 1,
        "tables_involved": ["order_items"],
        "question": "What is the price ratio between the 90th-percentile item and the 10th-percentile item by listed price?",
        "sql_query": (
            "SELECT (PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY price)) / "
            "(PERCENTILE_CONT(0.1) WITHIN GROUP (ORDER BY price)) "
            "FROM order_items;"),
        "pandas_code": "order_items.price.quantile(0.9) / order_items.price.quantile(0.1)",
        "gold_answer_type": "rubric",
        "gold_answer": r2(p90 / p10),
        "rubric": {
            "key_facts": [
                f"10th percentile price: R$ {r2(p10)}",
                f"90th percentile price: R$ {r2(p90)}",
                f"Ratio: {r2(p90/p10)}x",
            ],
            "supporting_data": {"p10": r2(p10), "p90": r2(p90),
                                "ratio": r2(p90/p10)},
        },
    })

    # 11: Compare seller distribution: top-5 seller states vs bottom-5
    sellers_by_state = s.seller_state.value_counts()
    top5 = sellers_by_state.head(5)
    bottom5 = sellers_by_state.tail(5)
    register({
        "qid": "comparison_11",
        "category": "comparison",
        "min_hops": 0,
        "tables_involved": ["sellers"],
        "question": "Compare the top-5 and bottom-5 Brazilian states by number of registered sellers. How concentrated is the seller base in the leading states?",
        "sql_query": (
            "SELECT seller_state, COUNT(*) FROM sellers "
            "GROUP BY seller_state ORDER BY COUNT(*) DESC;"),
        "pandas_code": "sellers.seller_state.value_counts()",
        "gold_answer_type": "rubric",
        "gold_answer": top5.index.tolist(),
        "rubric": {
            "key_facts": [
                f"Top-5 states: {top5.to_dict()}",
                f"Bottom-5 states: {bottom5.to_dict()}",
                f"Top-5 share: {top5.sum() / len(s) * 100:.1f}% of all sellers.",
            ],
            "supporting_data": {
                "top5": {k: int(v) for k, v in top5.items()},
                "bottom5": {k: int(v) for k, v in bottom5.items()},
                "top5_share_pct": r2(top5.sum() / len(s) * 100),
            },
        },
    })

    # 12: Compare review-response latency: 1-star vs 5-star reviews
    rv2 = rv.copy()
    rv2["response_h"] = (rv2.review_answer_timestamp - rv2.review_creation_date).dt.total_seconds() / 3600.0
    resp_1 = rv2.query("review_score == 1").response_h.mean()
    resp_5 = rv2.query("review_score == 5").response_h.mean()
    register({
        "qid": "comparison_12",
        "category": "comparison",
        "min_hops": 0,
        "tables_involved": ["reviews"],
        "question": "Compare the mean response-time (in hours) for 1-star versus 5-star reviews. Do sellers respond faster to negative or positive reviews?",
        "sql_query": (
            "SELECT review_score, AVG(EXTRACT(EPOCH FROM "
            "(review_answer_timestamp - review_creation_date))/3600.0) "
            "FROM order_reviews WHERE review_score IN (1,5) "
            "GROUP BY review_score;"),
        "pandas_code": "<groupby review_score, response_hours.mean()>",
        "gold_answer_type": "rubric",
        "gold_answer": "1-star" if resp_1 < resp_5 else "5-star",
        "rubric": {
            "key_facts": [
                f"Mean response time for 1-star reviews: {r2(resp_1)} hours",
                f"Mean response time for 5-star reviews: {r2(resp_5)} hours",
            ],
            "supporting_data": {"1_star_mean_response_h": r2(resp_1),
                                "5_star_mean_response_h": r2(resp_5)},
        },
    })

    # --- comparison_13..30: 18 additional comparison questions ---

    # 13: Compare 2017 vs 2018 total revenue
    o_2017 = o[o.order_purchase_timestamp.dt.year == 2017].order_id
    o_2018 = o[o.order_purchase_timestamp.dt.year == 2018].order_id
    rev_17 = it[it.order_id.isin(o_2017)].price.sum()
    rev_18 = it[it.order_id.isin(o_2018)].price.sum()
    register({
        "qid": "comparison_13",
        "category": "comparison",
        "min_hops": 1,
        "tables_involved": ["orders", "order_items"],
        "question": "Compare total revenue (sum of item prices, excluding freight) for calendar years 2017 and 2018. Which year was larger and by what factor?",
        "sql_query": (
            "SELECT EXTRACT(YEAR FROM o.order_purchase_timestamp) AS y, SUM(i.price) "
            "FROM order_items i JOIN orders o ON i.order_id = o.order_id "
            "WHERE EXTRACT(YEAR FROM o.order_purchase_timestamp) IN (2017, 2018) "
            "GROUP BY y;"),
        "pandas_code": "<groupby year, sum(price)>",
        "gold_answer_type": "rubric",
        "gold_answer": "2018" if rev_18 > rev_17 else "2017",
        "rubric": {
            "key_facts": [
                f"2017 revenue: R$ {rev_17:,.2f}",
                f"2018 revenue: R$ {rev_18:,.2f}",
                f"Ratio 2018/2017: {rev_18 / rev_17:.2f}x" if rev_17 else "",
                "Note: 2018 sample ends in late August/October; year is not fully covered.",
            ],
            "supporting_data": {"revenue_2017": r2(rev_17),
                                "revenue_2018": r2(rev_18),
                                "ratio_2018_over_2017": r2(rev_18 / rev_17) if rev_17 else None},
        },
    })

    # 14: Compare delivery time: in-state vs cross-state (customer state = seller state?)
    o_with_cust = o.merge(c[["customer_id", "customer_state"]], on="customer_id")
    items_with_seller = it.merge(s[["seller_id", "seller_state"]], on="seller_id")
    joined = o_with_cust.merge(items_with_seller[["order_id", "seller_state"]], on="order_id")
    joined = joined.query("order_status == 'delivered'").copy()
    joined["d"] = (joined.order_delivered_customer_date - joined.order_purchase_timestamp).dt.total_seconds() / 86400.0
    joined["same_state"] = joined.customer_state == joined.seller_state
    deliv_same = joined.query("same_state").d.mean()
    deliv_cross = joined.query("not same_state").d.mean()
    register({
        "qid": "comparison_14",
        "category": "comparison",
        "min_hops": 4,
        "tables_involved": ["orders", "customers", "order_items", "sellers"],
        "question": "Compare the mean delivery time for orders where the customer and seller are in the same Brazilian state versus orders shipped across state lines.",
        "sql_query": (
            "SELECT (c.customer_state = s.seller_state) AS same_state, "
            "AVG(EXTRACT(EPOCH FROM "
            "(o.order_delivered_customer_date - o.order_purchase_timestamp))/86400.0) "
            "FROM orders o "
            "JOIN customers c ON o.customer_id = c.customer_id "
            "JOIN order_items i ON o.order_id = i.order_id "
            "JOIN sellers s ON i.seller_id = s.seller_id "
            "WHERE o.order_status = 'delivered' GROUP BY same_state;"),
        "pandas_code": "<see joined frame in registry source>",
        "gold_answer_type": "rubric",
        "gold_answer": "same_state" if deliv_same < deliv_cross else "cross_state",
        "rubric": {
            "key_facts": [
                f"Same-state mean delivery: {r2(deliv_same)} days",
                f"Cross-state mean delivery: {r2(deliv_cross)} days",
                "Same-state shipments should be faster due to shorter distances.",
            ],
            "supporting_data": {"same_state_days": r2(deliv_same),
                                "cross_state_days": r2(deliv_cross)},
        },
    })

    # 15: Compare avg review score: SP, RJ, MG (top-3 customer states)
    def avg_rev_for_state(state):
        custs = c.query("customer_state == @state").customer_id
        oids = o[o.customer_id.isin(custs)].order_id
        return r2(rv[rv.order_id.isin(oids)].review_score.mean())
    state_revs = {st: avg_rev_for_state(st) for st in ["SP", "RJ", "MG"]}
    register({
        "qid": "comparison_15",
        "category": "comparison",
        "min_hops": 2,
        "tables_involved": ["customers", "orders", "reviews"],
        "question": "Compare the mean review score for orders shipped to customers in São Paulo (SP), Rio de Janeiro (RJ), and Minas Gerais (MG). Rank from highest to lowest.",
        "sql_query": (
            "SELECT c.customer_state, AVG(r.review_score) FROM order_reviews r "
            "JOIN orders o ON r.order_id = o.order_id "
            "JOIN customers c ON o.customer_id = c.customer_id "
            "WHERE c.customer_state IN ('SP','RJ','MG') GROUP BY c.customer_state;"),
        "pandas_code": "<see avg_rev_for_state in registry source>",
        "gold_answer_type": "rubric",
        "gold_answer": max(state_revs, key=state_revs.get),
        "rubric": {
            "key_facts": [
                f"SP mean review: {state_revs['SP']}",
                f"RJ mean review: {state_revs['RJ']}",
                f"MG mean review: {state_revs['MG']}",
            ],
            "supporting_data": state_revs,
        },
    })

    # 16: Compare freight value: heavy products (>5kg) vs light products (<1kg)
    heavy_pids = pr.query("product_weight_g > 5000").product_id
    light_pids = pr.query("product_weight_g < 1000").product_id
    fr_heavy = it[it.product_id.isin(heavy_pids)].freight_value.mean()
    fr_light = it[it.product_id.isin(light_pids)].freight_value.mean()
    register({
        "qid": "comparison_16",
        "category": "comparison",
        "min_hops": 2,
        "tables_involved": ["products", "order_items"],
        "question": "Compare the mean freight value for items weighing more than 5{,}000 g versus items weighing less than 1{,}000 g.",
        "sql_query": (
            "SELECT (CASE WHEN p.product_weight_g > 5000 THEN 'heavy' "
            "             WHEN p.product_weight_g < 1000 THEN 'light' END) AS bucket, "
            "AVG(i.freight_value) FROM order_items i "
            "JOIN products p ON i.product_id = p.product_id "
            "WHERE p.product_weight_g > 5000 OR p.product_weight_g < 1000 GROUP BY bucket;"),
        "pandas_code": "<see registry source>",
        "gold_answer_type": "rubric",
        "gold_answer": "heavy" if fr_heavy > fr_light else "light",
        "rubric": {
            "key_facts": [
                f"Heavy items (>5kg) mean freight: R$ {r2(fr_heavy)}",
                f"Light items (<1kg) mean freight: R$ {r2(fr_light)}",
                f"Ratio: {fr_heavy / fr_light:.2f}x" if fr_light else "",
            ],
            "supporting_data": {"heavy_mean_freight": r2(fr_heavy),
                                "light_mean_freight": r2(fr_light)},
        },
    })

    # 17: Compare order volume: Black Friday (Nov 2017) vs the typical surrounding-month average
    o_dated = o[o.order_purchase_timestamp.notna()].copy()
    by_ym = o_dated.assign(ym=o_dated.order_purchase_timestamp.dt.to_period("M")).ym.value_counts().sort_index()
    nov17 = by_ym.get(pd.Period("2017-11", "M"), 0)
    surrounding = [by_ym.get(pd.Period(ym, "M"), 0)
                   for ym in ["2017-09", "2017-10", "2017-12", "2018-01"]]
    avg_surr = float(np.mean(surrounding))
    register({
        "qid": "comparison_17",
        "category": "comparison",
        "min_hops": 0,
        "tables_involved": ["orders"],
        "question": "Compare order volume in November 2017 (Brazilian Black Friday month) to the average order volume of the surrounding four months (Sep–Oct 2017 and Dec 2017–Jan 2018). By what factor did volume spike?",
        "sql_query": (
            "SELECT EXTRACT(YEAR_MONTH FROM order_purchase_timestamp), COUNT(*) "
            "FROM orders GROUP BY 1;"),
        "pandas_code": "<groupby year-month, count>",
        "gold_answer_type": "rubric",
        "gold_answer": r2(nov17 / avg_surr) if avg_surr else None,
        "rubric": {
            "key_facts": [
                f"November 2017 order count: {int(nov17):,}",
                f"Surrounding 4-month average: {avg_surr:,.0f}",
                f"Spike factor: {nov17 / avg_surr:.2f}x" if avg_surr else "",
            ],
            "supporting_data": {"nov_2017_orders": int(nov17),
                                "surrounding_4mo_mean": r2(avg_surr),
                                "spike_factor": r2(nov17 / avg_surr) if avg_surr else None},
        },
    })

    # 18: Compare cancellation rate: Southeast vs Northeast
    cust_reg_t = add_region_col(c, "customer_state")
    o_reg_t = o.merge(cust_reg_t[["customer_id", "region"]], on="customer_id")
    def cancel_rate(region):
        sub = o_reg_t.query("region == @region")
        return r4((sub.order_status == "canceled").mean())
    cancel_se = cancel_rate("Southeast")
    cancel_ne = cancel_rate("Northeast")
    register({
        "qid": "comparison_18",
        "category": "comparison",
        "min_hops": 2,
        "tables_involved": ["customers", "orders"],
        "question": "Compare the cancellation rate for orders shipped to customers in the Southeast region versus the Northeast region. Which region has higher cancellation?",
        "sql_query": (
            "-- region derived from customer_state. "
            "SELECT region, AVG(CASE WHEN order_status='canceled' THEN 1.0 ELSE 0.0 END) "
            "FROM orders JOIN customers USING(customer_id) GROUP BY region;"),
        "pandas_code": "<see add_region_col() and groupby>",
        "gold_answer_type": "rubric",
        "gold_answer": "Northeast" if cancel_ne > cancel_se else "Southeast",
        "rubric": {
            "key_facts": [
                f"Southeast cancellation rate: {(cancel_se or 0)*100:.2f}%",
                f"Northeast cancellation rate: {(cancel_ne or 0)*100:.2f}%",
            ],
            "supporting_data": {"southeast_cancel_rate": cancel_se,
                                "northeast_cancel_rate": cancel_ne},
        },
    })

    # 19: Compare avg payment_installments: low-value (<R$100) vs high-value (>R$500) orders
    p_low = p.query("payment_value < 100 and payment_type == 'credit_card'").payment_installments.mean()
    p_high = p.query("payment_value > 500 and payment_type == 'credit_card'").payment_installments.mean()
    register({
        "qid": "comparison_19",
        "category": "comparison",
        "min_hops": 0,
        "tables_involved": ["payments"],
        "question": "Among credit-card payments, compare the average number of installments used for low-value payments (under R$100) versus high-value payments (over R$500).",
        "sql_query": (
            "SELECT (CASE WHEN payment_value < 100 THEN 'low' "
            "             WHEN payment_value > 500 THEN 'high' END) AS bucket, "
            "AVG(payment_installments) FROM order_payments "
            "WHERE payment_type = 'credit_card' GROUP BY bucket;"),
        "pandas_code": "<see registry source>",
        "gold_answer_type": "rubric",
        "gold_answer": "high" if p_high > p_low else "low",
        "rubric": {
            "key_facts": [
                f"Low-value (<R$100) credit-card mean installments: {r2(p_low)}",
                f"High-value (>R$500) credit-card mean installments: {r2(p_high)}",
                "Brazilian retail norms favour splitting large purchases into long installment plans.",
            ],
            "supporting_data": {"low_value_mean_installments": r2(p_low),
                                "high_value_mean_installments": r2(p_high)},
        },
    })

    # 20: Compare top-3 product categories by cancellation rate (cat-level cancellation)
    o_status = o.merge(it[["order_id", "product_id"]], on="order_id") \
                  .merge(pr[["product_id", "product_category_name_english"]], on="product_id")
    cancel_by_cat = (
        o_status.groupby("product_category_name_english")
        .order_status.apply(lambda s_: (s_ == "canceled").mean())
        .sort_values(ascending=False)
    )
    top3_cancel = cancel_by_cat.head(3)
    register({
        "qid": "comparison_20",
        "category": "comparison",
        "min_hops": 3,
        "tables_involved": ["orders", "order_items", "products"],
        "question": "Identify the 3 product categories with the highest cancellation rate (share of orders involving that category with status = 'canceled'). Report each rate.",
        "sql_query": (
            "WITH cat_orders AS ( "
            "  SELECT DISTINCT i.order_id, t.product_category_name_english AS cat "
            "  FROM order_items i "
            "  JOIN products p ON i.product_id = p.product_id "
            "  JOIN product_category_name_translation t "
            "    ON p.product_category_name = t.product_category_name) "
            "SELECT co.cat, AVG(CASE WHEN o.order_status='canceled' THEN 1.0 ELSE 0.0 END) AS rate "
            "FROM cat_orders co JOIN orders o ON co.order_id = o.order_id "
            "GROUP BY co.cat ORDER BY rate DESC LIMIT 3;"),
        "pandas_code": "<see registry source>",
        "gold_answer_type": "rubric",
        "gold_answer": top3_cancel.index.tolist(),
        "rubric": {
            "key_facts": [
                f"#{i+1}: {cat} with {top3_cancel.iloc[i]*100:.2f}% cancellation rate"
                for i, cat in enumerate(top3_cancel.index)
            ],
            "supporting_data": {cat: r4(v) for cat, v in top3_cancel.items()},
        },
    })

    # 21: Compare review distribution: small vs large sellers
    seller_volume = it.groupby("seller_id").size()
    small_sids = seller_volume[seller_volume < 10].index
    large_sids = seller_volume[seller_volume > 100].index
    rev_small = rv[rv.order_id.isin(it[it.seller_id.isin(small_sids)].order_id)].review_score.mean()
    rev_large = rv[rv.order_id.isin(it[it.seller_id.isin(large_sids)].order_id)].review_score.mean()
    register({
        "qid": "comparison_21",
        "category": "comparison",
        "min_hops": 3,
        "tables_involved": ["sellers", "order_items", "reviews"],
        "question": "Compare the mean review score of orders fulfilled by small sellers (fewer than 10 items sold) versus large sellers (more than 100 items sold).",
        "sql_query": (
            "WITH seller_vol AS (SELECT seller_id, COUNT(*) AS n "
            "                    FROM order_items GROUP BY seller_id) "
            "SELECT (CASE WHEN sv.n < 10 THEN 'small' WHEN sv.n > 100 THEN 'large' END) AS bucket, "
            "AVG(r.review_score) FROM order_reviews r "
            "JOIN order_items i ON r.order_id = i.order_id "
            "JOIN seller_vol sv ON i.seller_id = sv.seller_id "
            "WHERE sv.n < 10 OR sv.n > 100 GROUP BY bucket;"),
        "pandas_code": "<see registry source>",
        "gold_answer_type": "rubric",
        "gold_answer": "large" if rev_large > rev_small else "small",
        "rubric": {
            "key_facts": [
                f"Small sellers (<10 items) mean review: {r2(rev_small)}",
                f"Large sellers (>100 items) mean review: {r2(rev_large)}",
            ],
            "supporting_data": {"small_sellers_mean_review": r2(rev_small),
                                "large_sellers_mean_review": r2(rev_large)},
        },
    })

    # 22: Compare payment-type distribution: SP vs RJ
    sp_pay_dist = p[p.order_id.isin(o[o.customer_id.isin(c.query("customer_state == 'SP'").customer_id)].order_id)] \
                    .payment_type.value_counts(normalize=True)
    rj_pay_dist = p[p.order_id.isin(o[o.customer_id.isin(c.query("customer_state == 'RJ'").customer_id)].order_id)] \
                    .payment_type.value_counts(normalize=True)
    register({
        "qid": "comparison_22",
        "category": "comparison",
        "min_hops": 3,
        "tables_involved": ["customers", "orders", "payments"],
        "question": "Compare the share of credit-card versus boleto payments for orders by customers in São Paulo (SP) and Rio de Janeiro (RJ). Are the patterns different?",
        "sql_query": "-- per-state payment_type share computation.",
        "pandas_code": "<see registry source>",
        "gold_answer_type": "rubric",
        "gold_answer": "SP" if sp_pay_dist.get("credit_card", 0) > rj_pay_dist.get("credit_card", 0) else "RJ",
        "rubric": {
            "key_facts": [
                f"SP: credit_card={sp_pay_dist.get('credit_card', 0)*100:.1f}%, boleto={sp_pay_dist.get('boleto', 0)*100:.1f}%",
                f"RJ: credit_card={rj_pay_dist.get('credit_card', 0)*100:.1f}%, boleto={rj_pay_dist.get('boleto', 0)*100:.1f}%",
            ],
            "supporting_data": {
                "SP_payment_shares": {k: r4(v) for k, v in sp_pay_dist.items()},
                "RJ_payment_shares": {k: r4(v) for k, v in rj_pay_dist.items()},
            },
        },
    })

    # 23: Compare on-time delivery rate: Southeast vs Northeast
    def region_ontime(region):
        sub = o_reg_t.query("region == @region and order_status == 'delivered'")
        sub = sub[sub.order_delivered_customer_date.notna() & sub.order_estimated_delivery_date.notna()]
        if len(sub) == 0:
            return None
        return r4((sub.order_delivered_customer_date <= sub.order_estimated_delivery_date).mean())
    ot_se = region_ontime("Southeast")
    ot_ne = region_ontime("Northeast")
    register({
        "qid": "comparison_23",
        "category": "comparison",
        "min_hops": 2,
        "tables_involved": ["customers", "orders"],
        "question": "Compare the on-time delivery rate (share of delivered orders received on or before the estimated date) between the Southeast and the Northeast regions.",
        "sql_query": "-- region from customer_state; mean of (delivered <= estimated).",
        "pandas_code": "<see region_ontime() in registry source>",
        "gold_answer_type": "rubric",
        "gold_answer": "Southeast" if (ot_se or 0) > (ot_ne or 0) else "Northeast",
        "rubric": {
            "key_facts": [
                f"Southeast on-time rate: {(ot_se or 0)*100:.2f}%",
                f"Northeast on-time rate: {(ot_ne or 0)*100:.2f}%",
            ],
            "supporting_data": {"southeast_ontime": ot_se, "northeast_ontime": ot_ne},
        },
    })

    # 24: Compare avg review score: products with many photos (>=5) vs few (<=1)
    many_pids = pr.query("product_photos_qty >= 5").product_id
    few_pids = pr.query("product_photos_qty <= 1").product_id
    rev_many = rv[rv.order_id.isin(it[it.product_id.isin(many_pids)].order_id)].review_score.mean()
    rev_few = rv[rv.order_id.isin(it[it.product_id.isin(few_pids)].order_id)].review_score.mean()
    register({
        "qid": "comparison_24",
        "category": "comparison",
        "min_hops": 3,
        "tables_involved": ["products", "order_items", "reviews"],
        "question": "Compare the mean review score for orders involving products with many photos (≥5) versus products with few photos (≤1).",
        "sql_query": "-- aggregated review_score over join of products + order_items + reviews, bucketed by photos_qty.",
        "pandas_code": "<see registry source>",
        "gold_answer_type": "rubric",
        "gold_answer": "many" if rev_many > rev_few else "few",
        "rubric": {
            "key_facts": [
                f"Products with ≥5 photos -- mean review: {r2(rev_many)}",
                f"Products with ≤1 photo  -- mean review: {r2(rev_few)}",
            ],
            "supporting_data": {"many_photos_mean_review": r2(rev_many),
                                "few_photos_mean_review": r2(rev_few)},
        },
    })

    # 25: Compare avg order value: orders >R$500 vs <R$100 by delivery time
    p_per_order = p.groupby("order_id").payment_value.sum()
    high_oids = p_per_order[p_per_order > 500].index
    low_oids = p_per_order[p_per_order < 100].index
    o_del_t = o.query("order_status == 'delivered'").copy()
    o_del_t["d"] = (o_del_t.order_delivered_customer_date - o_del_t.order_purchase_timestamp).dt.total_seconds() / 86400.0
    deliv_high = o_del_t[o_del_t.order_id.isin(high_oids)].d.mean()
    deliv_low = o_del_t[o_del_t.order_id.isin(low_oids)].d.mean()
    register({
        "qid": "comparison_25",
        "category": "comparison",
        "min_hops": 2,
        "tables_involved": ["orders", "payments"],
        "question": "Compare mean delivery time for high-value orders (total payment > R$500) versus low-value orders (total payment < R$100).",
        "sql_query": "-- groupby payment_value bucket, mean delivery duration.",
        "pandas_code": "<see registry source>",
        "gold_answer_type": "rubric",
        "gold_answer": "high" if deliv_high > deliv_low else "low",
        "rubric": {
            "key_facts": [
                f"High-value (>R$500) mean delivery: {r2(deliv_high)} days",
                f"Low-value (<R$100) mean delivery: {r2(deliv_low)} days",
            ],
            "supporting_data": {"high_value_delivery_days": r2(deliv_high),
                                "low_value_delivery_days": r2(deliv_low)},
        },
    })

    # 26: Top-3 cities by number of customer orders
    top3_cust_cities = c.merge(o[["customer_id"]], on="customer_id") \
                          .customer_city.value_counts().head(3)
    register({
        "qid": "comparison_26",
        "category": "comparison",
        "min_hops": 1,
        "tables_involved": ["customers", "orders"],
        "question": "Identify the top 3 cities (by name) with the most customer orders in the dataset, and report each city's count.",
        "sql_query": (
            "SELECT c.customer_city, COUNT(*) FROM orders o "
            "JOIN customers c ON o.customer_id = c.customer_id "
            "GROUP BY c.customer_city ORDER BY 2 DESC LIMIT 3;"),
        "pandas_code": (
            "customers.merge(orders, on='customer_id').customer_city.value_counts().head(3)"),
        "gold_answer_type": "rubric",
        "gold_answer": top3_cust_cities.index.tolist(),
        "rubric": {
            "key_facts": [
                f"#{i+1}: {city} ({int(n):,} orders)"
                for i, (city, n) in enumerate(top3_cust_cities.items())
            ],
            "supporting_data": {city: int(n) for city, n in top3_cust_cities.items()},
        },
    })

    # 27: Compare freight-to-price ratio for the top-2 revenue categories
    cat_rev_t = (it.merge(pr[["product_id", "product_category_name_english"]], on="product_id")
                   .groupby("product_category_name_english").price.sum().sort_values(ascending=False))
    top2 = cat_rev_t.head(2).index.tolist()
    def ratio_for_cat(cat):
        cpids = pr.query("product_category_name_english == @cat").product_id
        sub = it[it.product_id.isin(cpids)]
        return r4((sub.freight_value / sub.price.replace(0, np.nan)).mean())
    cat1_r = ratio_for_cat(top2[0])
    cat2_r = ratio_for_cat(top2[1])
    register({
        "qid": "comparison_27",
        "category": "comparison",
        "min_hops": 2,
        "tables_involved": ["products", "order_items"],
        "question": f"Compare the mean freight-to-price ratio for the top-2 revenue-generating product categories ('{top2[0]}' and '{top2[1]}'). Which has higher relative freight?",
        "sql_query": "-- groupby category, mean(freight_value / price), filter top-2 by sum(price).",
        "pandas_code": "<see ratio_for_cat() in registry source>",
        "gold_answer_type": "rubric",
        "gold_answer": top2[0] if (cat1_r or 0) > (cat2_r or 0) else top2[1],
        "rubric": {
            "key_facts": [
                f"{top2[0]} freight/price ratio: {cat1_r}",
                f"{top2[1]} freight/price ratio: {cat2_r}",
            ],
            "supporting_data": {top2[0]: cat1_r, top2[1]: cat2_r},
        },
    })

    # 28: Compare review-comment prevalence: 1-star vs 5-star
    pct_text_1 = rv.query("review_score == 1").review_comment_message.notna().mean()
    pct_text_5 = rv.query("review_score == 5").review_comment_message.notna().mean()
    register({
        "qid": "comparison_28",
        "category": "comparison",
        "min_hops": 0,
        "tables_involved": ["reviews"],
        "question": "Compare the share of reviews with a non-null free-text comment between 1-star and 5-star reviews. Which extreme is more likely to write a comment?",
        "sql_query": (
            "SELECT review_score, AVG(CASE WHEN review_comment_message IS NOT NULL "
            "THEN 1.0 ELSE 0.0 END) FROM order_reviews WHERE review_score IN (1,5) "
            "GROUP BY review_score;"),
        "pandas_code": "<groupby review_score, comment_present.mean()>",
        "gold_answer_type": "rubric",
        "gold_answer": "1-star" if pct_text_1 > pct_text_5 else "5-star",
        "rubric": {
            "key_facts": [
                f"Share of 1-star reviews with comment: {pct_text_1*100:.1f}%",
                f"Share of 5-star reviews with comment: {pct_text_5*100:.1f}%",
            ],
            "supporting_data": {"1_star_comment_share": r4(pct_text_1),
                                "5_star_comment_share": r4(pct_text_5)},
        },
    })

    # 29: Compare order volume: weekend vs weekday
    o_dated_t = o[o.order_purchase_timestamp.notna()].copy()
    o_dated_t["dow"] = o_dated_t.order_purchase_timestamp.dt.dayofweek  # 0=Mon..6=Sun
    o_dated_t["is_weekend"] = o_dated_t.dow.isin([5, 6])
    weekend_share = o_dated_t.is_weekend.mean()
    register({
        "qid": "comparison_29",
        "category": "comparison",
        "min_hops": 0,
        "tables_involved": ["orders"],
        "question": "Are customers more likely to place orders on weekdays or weekends? Report the share of orders placed on Saturday or Sunday.",
        "sql_query": (
            "SELECT AVG(CASE WHEN EXTRACT(DOW FROM order_purchase_timestamp) IN (0,6) "
            "THEN 1.0 ELSE 0.0 END) FROM orders;"),
        "pandas_code": (
            "orders.order_purchase_timestamp.dt.dayofweek.isin([5,6]).mean()"),
        "gold_answer_type": "rubric",
        "gold_answer": "weekend" if weekend_share > 2/7 else "weekday",
        "rubric": {
            "key_facts": [
                f"Share of orders placed on Saturday/Sunday: {weekend_share*100:.2f}%",
                f"Baseline (2/7 days): {(2/7)*100:.2f}%",
                "If the share is below 2/7, weekdays are over-represented; above 2/7 the inverse.",
            ],
            "supporting_data": {"weekend_share": r4(weekend_share),
                                "baseline": r4(2/7)},
        },
    })

    # 30: Compare 90th percentile delivery time vs median (spread / tail)
    o_del_t2 = o.query("order_status == 'delivered'").copy()
    o_del_t2["d"] = (o_del_t2.order_delivered_customer_date - o_del_t2.order_purchase_timestamp).dt.total_seconds() / 86400.0
    p50 = o_del_t2.d.median()
    p90 = o_del_t2.d.quantile(0.90)
    register({
        "qid": "comparison_30",
        "category": "comparison",
        "min_hops": 0,
        "tables_involved": ["orders"],
        "question": "Compare the median delivery time to the 90th-percentile delivery time across all delivered orders. By what factor is the long tail (p90) larger than the typical case (p50)?",
        "sql_query": (
            "SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY d), "
            "       PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY d) "
            "FROM (SELECT EXTRACT(EPOCH FROM "
            "(order_delivered_customer_date - order_purchase_timestamp))/86400.0 AS d "
            "FROM orders WHERE order_status = 'delivered') sub;"),
        "pandas_code": "<see registry source>",
        "gold_answer_type": "rubric",
        "gold_answer": r2(p90 / p50) if p50 else None,
        "rubric": {
            "key_facts": [
                f"Median delivery time: {r2(p50)} days",
                f"P90 delivery time: {r2(p90)} days",
                f"P90 / median ratio: {p90/p50:.2f}x" if p50 else "",
            ],
            "supporting_data": {"median_days": r2(p50),
                                "p90_days": r2(p90),
                                "ratio_p90_over_median": r2(p90/p50) if p50 else None},
        },
    })


# ============================================================
# CATEGORY 4 — ROOT-CAUSE  (4+ hops, rubric-based)
# ============================================================
# Hier sind die Antworten überwiegend RUBRIKEN, weil mehrere
# Erklärungen valide sind. Wir liefern aber gemessene Werte als
# supporting_data, damit der Judge harte Fakten checken kann.

def build_root_cause(t: dict[str, pd.DataFrame]) -> None:
    o, p, it, rv, c, s, pr = (t["orders"], t["payments"], t["order_items"],
                              t["reviews"], t["customers"], t["sellers"],
                              t["products"])

    cust_reg = add_region_col(c, "customer_state")

    # 1: Why are delivery times longer in the North?
    o_reg = o.merge(cust_reg[["customer_id", "region"]], on="customer_id")
    o_reg = o_reg[(o_reg.order_status == "delivered")].copy()
    o_reg["d"] = (o_reg.order_delivered_customer_date - o_reg.order_purchase_timestamp).dt.total_seconds() / 86400.0
    deliv_by_region = o_reg.groupby("region").d.mean()
    seller_reg = add_region_col(s, "seller_state")
    seller_share = seller_reg.region.value_counts(normalize=True)
    register({
        "qid": "root_cause_01",
        "category": "root_cause",
        "min_hops": 4,
        "tables_involved": ["customers", "orders", "order_items", "sellers"],
        "question": "Delivery times for orders shipped to the North region are substantially longer than for the rest of Brazil. What plausible structural factors in the Olist data could explain this difference?",
        "sql_query": (
            "-- multiple supporting queries: avg delivery by region (orders+customers); "
            "seller share by region (sellers); avg freight by region (items+orders+customers)."),
        "pandas_code": "<see add_region_col() and groupby aggregations>",
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                f"Mean delivery time in North: {r2(deliv_by_region.get('North'))} days "
                f"vs. Southeast: {r2(deliv_by_region.get('Southeast'))} days.",
                "Sellers are highly concentrated in the Southeast (~"
                f"{r2((seller_share.get('Southeast', 0))*100)}% of sellers); "
                f"the North hosts only ~{r2((seller_share.get('North', 0))*100)}%.",
                "Most orders to North customers therefore require long-haul shipping "
                "from a Southeast supplier, inflating delivery time.",
                "Brazil's logistics infrastructure (highways, last-mile coverage) is "
                "weaker in the North; this is reflected in higher freight values "
                "and longer carrier hand-off times.",
            ],
            "supporting_data": {
                "delivery_days_by_region": {k: r2(v) for k, v in deliv_by_region.items()},
                "seller_share_by_region": {k: r2(v) for k, v in seller_share.items()},
            },
        },
    })

    # 2: Why are some review scores very low (1-star)? Drivers analysis.
    rv_merged = rv.merge(o[["order_id", "order_status", "order_delivered_customer_date",
                              "order_estimated_delivery_date", "order_purchase_timestamp"]],
                          on="order_id")
    rv_merged["late"] = (rv_merged.order_delivered_customer_date > rv_merged.order_estimated_delivery_date)
    rv_merged["late"] = rv_merged["late"].fillna(False)
    late_share_1star = rv_merged.query("review_score == 1").late.mean()
    late_share_5star = rv_merged.query("review_score == 5").late.mean()
    cancelled_share_1star = (rv_merged.query("review_score == 1").order_status == "canceled").mean()
    register({
        "qid": "root_cause_02",
        "category": "root_cause",
        "min_hops": 4,
        "tables_involved": ["reviews", "orders", "order_items", "products"],
        "question": "What structural factors most plausibly drive 1-star reviews in the Olist dataset?",
        "sql_query": "-- supporting: late-delivery share, canceled-order share, category mix, freight ratio (joined across reviews/orders/items/products).",
        "pandas_code": "<groupby review_score + late + cancelled>",
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                f"Share of 1-star orders delivered after the estimated date: "
                f"{(late_share_1star or 0)*100:.1f}% — versus only "
                f"{(late_share_5star or 0)*100:.1f}% for 5-star orders.",
                f"Share of 1-star orders with status 'canceled': "
                f"{(cancelled_share_1star or 0)*100:.1f}%.",
                "Late delivery is the single strongest signal associated with "
                "low review scores; cancellation contributes a smaller share.",
                "Product category and freight cost also have non-trivial effects "
                "(some categories such as bed_bath_table show below-average "
                "reviews independent of delivery performance).",
            ],
            "supporting_data": {
                "late_share_among_1star": r4(late_share_1star),
                "late_share_among_5star": r4(late_share_5star),
                "cancelled_share_among_1star": r4(cancelled_share_1star),
            },
        },
    })

    # 3: Why does the bed_bath_table category have so many orders but middling reviews?
    bbt_pids = pr.query("product_category_name_english == 'bed_bath_table'").product_id
    bbt_oids = it[it.product_id.isin(bbt_pids)].order_id.unique()
    bbt_reviews = rv[rv.order_id.isin(bbt_oids)]
    bbt_mean = bbt_reviews.review_score.mean()
    overall_mean = rv.review_score.mean()
    bbt_freight_ratio = (it[it.product_id.isin(bbt_pids)]
                          .assign(r=lambda d: d.freight_value / d.price.replace(0, np.nan))
                          .r.mean())
    overall_freight_ratio = it.assign(r=lambda d: d.freight_value / d.price.replace(0, np.nan)).r.mean()
    register({
        "qid": "root_cause_03",
        "category": "root_cause",
        "min_hops": 4,
        "tables_involved": ["products", "order_items", "reviews"],
        "question": "The category 'bed_bath_table' has one of the largest order volumes in the dataset but shows a below-average review score. What plausible causes can be identified from the data?",
        "sql_query": "-- supporting: mean review for bed_bath_table vs all; freight ratio comparison; product weight comparison.",
        "pandas_code": "<see registry source>",
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                f"bed_bath_table mean review: {r2(bbt_mean)} vs overall {r2(overall_mean)}.",
                f"bed_bath_table mean freight/price ratio: {r4(bbt_freight_ratio)} vs overall {r4(overall_freight_ratio)}.",
                "Plausible causes include: bulky/heavy products yielding higher freight ratios; "
                "high competition driving commoditized listings; mismatches between product "
                "images/descriptions and the received item (a known issue in this category).",
            ],
            "supporting_data": {
                "bbt_mean_review": r2(bbt_mean),
                "overall_mean_review": r2(overall_mean),
                "bbt_freight_ratio": r4(bbt_freight_ratio),
                "overall_freight_ratio": r4(overall_freight_ratio),
            },
        },
    })

    # 4: Why do multi-payment orders exist? Drivers analysis.
    pay_count_per_order = p.groupby("order_id").size()
    multi_oids = pay_count_per_order[pay_count_per_order > 1].index
    multi_avg = p[p.order_id.isin(multi_oids)].groupby("order_id").payment_value.sum().mean()
    single_avg = p[~p.order_id.isin(multi_oids)].payment_value.mean()
    multi_voucher_share = p[(p.order_id.isin(multi_oids))].payment_type.eq("voucher").mean()
    register({
        "qid": "root_cause_04",
        "category": "root_cause",
        "min_hops": 4,
        "tables_involved": ["payments", "orders", "order_items"],
        "question": "Around 3{,}000 orders in the dataset are paid with more than one payment entry. What plausible causes can be identified from the data structure?",
        "sql_query": "-- supporting: count of payment entries per order, payment-type composition for multi-entry orders, mean total value comparison.",
        "pandas_code": "<see registry source>",
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                f"Multi-payment orders: {int((pay_count_per_order > 1).sum())} orders ({r4((pay_count_per_order > 1).mean()*100)}% of all orders).",
                f"Mean total value for multi-payment orders: R$ {r2(multi_avg)} vs R$ {r2(single_avg)} for single-payment.",
                f"Share of voucher-type entries among multi-payment orders: {r4(multi_voucher_share)*100:.1f}%.",
                "Multi-payment orders are typically (a) high-value orders split across a credit card and a voucher, (b) installment plans recorded as separate rows, or (c) gift-card top-ups combined with a primary method.",
            ],
            "supporting_data": {
                "n_multi_payment_orders": int((pay_count_per_order > 1).sum()),
                "multi_payment_mean_total": r2(multi_avg),
                "single_payment_mean_total": r2(single_avg),
                "voucher_share_in_multi": r4(multi_voucher_share),
            },
        },
    })

    # 5: Why do some sellers have much higher freight costs?
    seller_freight = (it.groupby("seller_id")
                        .agg(mean_freight=("freight_value", "mean"),
                             n=("freight_value", "size")))
    seller_freight = seller_freight.query("n >= 30")
    high_freight = seller_freight.nlargest(10, "mean_freight").mean_freight.mean()
    low_freight = seller_freight.nsmallest(10, "mean_freight").mean_freight.mean()
    register({
        "qid": "root_cause_05",
        "category": "root_cause",
        "min_hops": 4,
        "tables_involved": ["sellers", "order_items", "products"],
        "question": "Some sellers consistently charge much higher freight values than others. What plausible factors in the dataset explain these differences?",
        "sql_query": "-- supporting: groupby seller mean freight; product weight per seller; distance proxy via seller_state.",
        "pandas_code": "<see seller_freight aggregation>",
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                f"Top-10 sellers by mean freight: R$ {r2(high_freight)} per item.",
                f"Bottom-10 sellers by mean freight: R$ {r2(low_freight)} per item.",
                "Plausible causes: (i) product weight/volume — sellers of bulky items (furniture, appliances) face higher per-item shipping; (ii) seller geography — sellers far from population centres ship longer distances; (iii) destination mix — sellers serving more North/Northeast customers see higher average freight; (iv) seller-specific pricing strategies.",
            ],
            "supporting_data": {
                "high_freight_sellers_mean": r2(high_freight),
                "low_freight_sellers_mean": r2(low_freight),
            },
        },
    })

    # 6: Why are some product categories seasonal?
    o_dated = o[o.order_purchase_timestamp.notna()].copy()
    o_dated["month"] = o_dated.order_purchase_timestamp.dt.month
    cat_months = (o_dated.merge(it[["order_id", "product_id"]], on="order_id")
                    .merge(pr[["product_id", "product_category_name_english"]], on="product_id")
                    .groupby(["product_category_name_english", "month"]).size().unstack(fill_value=0))
    # Saisonalitäts-Score: max month / mean month
    seasonality = (cat_months.max(axis=1) / cat_months.mean(axis=1)).sort_values(ascending=False)
    register({
        "qid": "root_cause_06",
        "category": "root_cause",
        "min_hops": 4,
        "tables_involved": ["orders", "order_items", "products"],
        "question": "Which product categories show the strongest monthly seasonality in the dataset, and what plausible reasons account for these spikes?",
        "sql_query": "-- supporting: monthly counts per category; ratio of peak to mean.",
        "pandas_code": "<groupby category + month, peak/mean ratio>",
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                f"Top-3 most seasonal categories (peak-month / mean-month ratio): "
                f"{seasonality.head(3).to_dict()}",
                "Plausible reasons: holiday gifting (toys, watches_gifts, perfumery peak in November/December around Black Friday/Christmas); summer/winter seasonality for sports_leisure and fashion; school-year cycles for office supplies.",
            ],
            "supporting_data": {
                "top3_seasonal_ratios": {k: r2(v) for k, v in seasonality.head(3).items()},
            },
        },
    })

    # 7: Why does cancellation cluster on specific months / payment types?
    o_cancel = o.query("order_status == 'canceled'").copy()
    o_cancel["month"] = o_cancel.order_purchase_timestamp.dt.month
    cancel_by_month = o_cancel.month.value_counts().sort_index()
    register({
        "qid": "root_cause_07",
        "category": "root_cause",
        "min_hops": 4,
        "tables_involved": ["orders", "payments", "order_items"],
        "question": "Are cancellations concentrated in specific months of the year, and what plausible factors could explain any clustering observed?",
        "sql_query": "-- supporting: count of cancellations by month; share of total orders by month.",
        "pandas_code": "orders.query(\"order_status == 'canceled'\").order_purchase_timestamp.dt.month.value_counts()",
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                f"Cancellations by month: {cancel_by_month.to_dict()}",
                "Plausible drivers: Black Friday / pre-Christmas spike in November (high order volume → higher absolute cancellation count); post-holiday returns/changes-of-mind in January; out-of-stock cancellations when sellers cannot fulfill during high-demand months.",
            ],
            "supporting_data": {
                "cancellations_by_month": {int(k): int(v) for k, v in cancel_by_month.items()},
            },
        },
    })

    # 8: Why does the 'Centro-Oeste' (Midwest) region have notable freight rates?
    cust_reg2 = add_region_col(c, "customer_state")
    o_reg2 = o.merge(cust_reg2[["customer_id", "region"]], on="customer_id")
    item_reg2 = it.merge(o_reg2[["order_id", "region"]], on="order_id")
    fr_by_region = item_reg2.groupby("region").freight_value.mean()
    register({
        "qid": "root_cause_08",
        "category": "root_cause",
        "min_hops": 3,
        "tables_involved": ["customers", "orders", "order_items"],
        "question": "Mean freight per item differs across Brazilian regions. Which region pays the most and why?",
        "sql_query": "-- supporting: average freight_value joined to region via customer_state mapping.",
        "pandas_code": "<see add_region_col() and groupby region>",
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                f"Mean freight per item by region: {fr_by_region.round(2).to_dict()}",
                "Plausible drivers: distance from the Southeast seller base; weaker last-mile logistics in North/Northeast; lower population density in the Midwest yielding more partial-load shipments.",
            ],
            "supporting_data": {
                "mean_freight_by_region": {k: r2(v) for k, v in fr_by_region.items()},
            },
        },
    })

    # 9: Why do credit_card payments correlate with higher installment counts on expensive orders?
    o_pay = o.merge(p[["order_id", "payment_type", "payment_installments", "payment_value"]],
                     on="order_id")
    o_pay_cc = o_pay.query("payment_type == 'credit_card'")
    bins = pd.cut(o_pay_cc.payment_value, bins=[0, 100, 500, 1e9],
                  labels=["low (<R$100)", "mid (R$100-500)", "high (>R$500)"])
    inst_by_value = o_pay_cc.groupby(bins, observed=True).payment_installments.mean()
    register({
        "qid": "root_cause_09",
        "category": "root_cause",
        "min_hops": 2,
        "tables_involved": ["payments", "orders"],
        "question": "Why do high-value credit-card orders show a much higher number of installments than low-value ones? What plausible explanations follow from the data?",
        "sql_query": "-- supporting: mean installments by payment_value bucket for credit_card payments.",
        "pandas_code": "<see bin + groupby>",
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                f"Mean installments by value bucket: {inst_by_value.round(2).to_dict()}",
                "Plausible drivers: Brazilian consumer credit norms — high-value purchases are routinely split into 6/10/12 monthly installments to manage cash flow; many retailers default to longer installment plans for amounts over R$500; consumer interest-free instalment offers ('parcelamento sem juros') concentrate on higher tickets.",
            ],
            "supporting_data": {
                "mean_installments_by_value_bucket": {
                    str(k): r2(v) for k, v in inst_by_value.items()
                },
            },
        },
    })

    # 10: Why do some product categories show much higher freight-to-price ratios?
    cat_freight_ratio = (it.merge(pr[["product_id", "product_category_name_english"]],
                                    on="product_id")
                            .assign(r=lambda d: d.freight_value / d.price.replace(0, np.nan))
                            .groupby("product_category_name_english")
                            .r.mean()
                            .sort_values(ascending=False)
                            .head(5))
    register({
        "qid": "root_cause_10",
        "category": "root_cause",
        "min_hops": 3,
        "tables_involved": ["products", "order_items"],
        "question": "Some product categories show an unusually high mean freight-to-price ratio. Which categories, and why?",
        "sql_query": "-- supporting: groupby category, mean(freight_value / price); product weight per category.",
        "pandas_code": "<see registry source>",
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                f"Top-5 categories by freight/price ratio: {cat_freight_ratio.round(3).to_dict()}",
                "Plausible drivers: low-cost, high-volume items (some 'casa_construcao' or 'fashion' sub-categories) where the package is bulky relative to its price; small items shipped from distant sellers where flat-fee freight dominates; categories with low unit price (e.g. small accessories).",
            ],
            "supporting_data": {
                "top5_freight_price_ratio": {k: r4(v) for k, v in cat_freight_ratio.items()},
            },
        },
    })

    # 11: Why is the seller_state distribution so skewed?
    seller_state_share = s.seller_state.value_counts(normalize=True)
    top1_share = seller_state_share.iloc[0]
    register({
        "qid": "root_cause_11",
        "category": "root_cause",
        "min_hops": 1,
        "tables_involved": ["sellers", "customers"],
        "question": "Why is the seller geographic distribution in Olist so heavily concentrated in a single state?",
        "sql_query": "SELECT seller_state, COUNT(*) FROM sellers GROUP BY seller_state ORDER BY 2 DESC;",
        "pandas_code": "sellers.seller_state.value_counts(normalize=True)",
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                f"Top seller state share: {top1_share*100:.1f}% (state = {seller_state_share.index[0]}).",
                "Plausible drivers: São Paulo is Brazil's industrial and commercial hub with the most established e-commerce ecosystem; Olist's headquarters and partner network are concentrated in the Southeast; logistical hubs (Cumbica airport, distribution centres) lower the cost of doing business there.",
            ],
            "supporting_data": {
                "top_seller_state": str(seller_state_share.index[0]),
                "top_seller_state_share": r4(top1_share),
            },
        },
    })

    # 12: Why might some orders take longer to be approved (timestamp gap)?
    o_app = o[o.order_approved_at.notna() & o.order_purchase_timestamp.notna()].copy()
    o_app["approval_h"] = (o_app.order_approved_at - o_app.order_purchase_timestamp).dt.total_seconds() / 3600.0
    o_app = o_app.merge(p[["order_id", "payment_type"]].drop_duplicates("order_id"), on="order_id")
    appr_by_pay = o_app.groupby("payment_type").approval_h.mean()
    register({
        "qid": "root_cause_12",
        "category": "root_cause",
        "min_hops": 2,
        "tables_involved": ["orders", "payments"],
        "question": "Some orders take much longer to be approved than others (gap between purchase and approval timestamps). What payment-method factors plausibly drive this variation?",
        "sql_query": "-- supporting: avg approval-time (hours) by payment_type, joined orders × payments.",
        "pandas_code": "<see registry source>",
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                f"Mean approval time (hours) by payment type: {appr_by_pay.round(2).to_dict()}",
                "Plausible drivers: 'boleto' (Brazilian bank slip) requires the customer to physically pay at a bank/lottery agent before approval, introducing a 1–3 day lag; credit_card authorisations are near-instant; voucher payments resolve quickly.",
            ],
            "supporting_data": {
                "mean_approval_hours_by_payment": {k: r2(v) for k, v in appr_by_pay.items()},
            },
        },
    })

    # --- root_cause_13..30: 18 additional root-cause questions ---

    # 13: Why are certain product categories under-represented in the catalogue?
    cat_counts = pr.product_category_name_english.value_counts()
    smallest = cat_counts.tail(5)
    register({
        "qid": "root_cause_13",
        "category": "root_cause",
        "min_hops": 1,
        "tables_involved": ["products", "category_translation"],
        "question": "Several product categories are represented by only a handful of products in the Olist catalogue. What plausible structural reasons could explain this long tail?",
        "sql_query": "SELECT category, COUNT(*) FROM products GROUP BY category ORDER BY 2 ASC LIMIT 10;",
        "pandas_code": (
            "products.merge(category_translation, on='product_category_name', how='left')"
            ".product_category_name_english.value_counts().tail(10)"),
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                f"Smallest 5 categories: {smallest.to_dict()}",
                "Plausible drivers: niche markets (specialty/luxury products, regional crafts); recent additions to the marketplace; categories that overlap with larger ones and lose listings to them; regulatory or shipping constraints (alcohol, food, fragile items) limiting seller adoption.",
            ],
            "supporting_data": {"smallest_categories": {k: int(v) for k, v in smallest.items()}},
        },
    })

    # 14: Why are some sellers high-volume but low-rated?
    seller_perf = (it.merge(rv[["order_id", "review_score"]], on="order_id")
                     .groupby("seller_id")
                     .agg(n_items=("order_item_id", "size"),
                          mean_review=("review_score", "mean")))
    paradox = seller_perf.query("n_items >= 100 and mean_review < 4.0")
    register({
        "qid": "root_cause_14",
        "category": "root_cause",
        "min_hops": 3,
        "tables_involved": ["sellers", "order_items", "reviews"],
        "question": "A subset of sellers fulfill many orders but maintain below-average review scores (mean < 4.0). What plausible structural explanations exist?",
        "sql_query": "-- per-seller (n_items, mean_review) aggregation.",
        "pandas_code": "<see registry source>",
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                f"Number of high-volume low-rated sellers (n_items≥100, mean_review<4.0): {len(paradox)} sellers.",
                "Plausible causes: (i) operate in challenging categories (bulky, fragile, or commoditized items where complaint rates are structurally higher); (ii) growth-at-the-cost-of-quality strategies where order volume is prioritized over service quality; (iii) algorithmic listing prominence on the platform decouples visibility from customer satisfaction.",
            ],
            "supporting_data": {"n_high_vol_low_rated_sellers": int(len(paradox))},
        },
    })

    # 15: Why does delivery time vary even within the same state?
    sp_custs = c.query("customer_state == 'SP'").customer_id
    sp_delivered = o[(o.customer_id.isin(sp_custs)) & (o.order_status == "delivered")].copy()
    sp_delivered["d"] = (sp_delivered.order_delivered_customer_date - sp_delivered.order_purchase_timestamp).dt.total_seconds() / 86400.0
    sp_std = sp_delivered.d.std()
    sp_mean = sp_delivered.d.mean()
    register({
        "qid": "root_cause_15",
        "category": "root_cause",
        "min_hops": 2,
        "tables_involved": ["customers", "orders", "order_items"],
        "question": "Even within the same Brazilian state, delivery times to customers vary widely. What plausible factors drive this within-state variability?",
        "sql_query": "-- groupby customer_state, std(delivery_days); analysis of intra-state variance.",
        "pandas_code": "<see registry source>",
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                f"Within São Paulo state: mean delivery {r2(sp_mean)} days, std-dev {r2(sp_std)} days.",
                "Plausible drivers: rural vs. urban customer locations within a state; seller geography (intra-state vs. cross-state shipment even from same destination state); product weight and category; carrier choice and routing; weekday-vs-weekend approval and pickup delays.",
            ],
            "supporting_data": {"sp_delivery_mean_days": r2(sp_mean),
                                "sp_delivery_std_days": r2(sp_std)},
        },
    })

    # 16: Why might orders to remote areas show longer carrier hand-off times?
    cust_reg_t = add_region_col(c, "customer_state")
    o_reg_t = o.merge(cust_reg_t[["customer_id", "region"]], on="customer_id")
    o_reg_t = o_reg_t.query("order_status == 'delivered' and order_delivered_carrier_date.notna()").copy()
    o_reg_t["carrier_hand_off_h"] = (o_reg_t.order_delivered_carrier_date - o_reg_t.order_approved_at).dt.total_seconds() / 3600.0
    handoff_by_region = o_reg_t.groupby("region").carrier_hand_off_h.mean()
    register({
        "qid": "root_cause_16",
        "category": "root_cause",
        "min_hops": 2,
        "tables_involved": ["customers", "orders"],
        "question": "Why might orders shipped to remote regions show longer hand-off times between approval and carrier pickup?",
        "sql_query": "-- by-region mean of (carrier_date - approved_at) hours.",
        "pandas_code": "<see registry source>",
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                f"Mean carrier hand-off time (hours) by region: {handoff_by_region.round(2).to_dict()}",
                "Plausible drivers: sellers consolidate shipments to remote destinations (waiting for full truckloads), increasing hand-off lag; fewer regional carrier pickup runs; longer paperwork (customs-like inter-state cargo declarations); planning around carrier route frequency.",
            ],
            "supporting_data": {"mean_carrier_handoff_hours_by_region":
                                {k: r2(v) for k, v in handoff_by_region.items()}},
        },
    })

    # 17: Why are some customer states associated with lower review scores?
    state_review = (rv.merge(o[["order_id", "customer_id"]], on="order_id")
                      .merge(c[["customer_id", "customer_state"]], on="customer_id")
                      .groupby("customer_state").review_score.mean()
                      .sort_values())
    lowest_states = state_review.head(5)
    register({
        "qid": "root_cause_17",
        "category": "root_cause",
        "min_hops": 3,
        "tables_involved": ["reviews", "orders", "customers"],
        "question": "Some Brazilian states show systematically lower review scores than others. What plausible structural causes lie behind this geographic pattern?",
        "sql_query": "-- per customer_state mean review_score, joined reviews × orders × customers.",
        "pandas_code": "<see registry source>",
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                f"States with the 5 lowest mean reviews: {lowest_states.round(3).to_dict()}",
                "Plausible drivers: long delivery times (correlated with North/Northeast geography); high freight costs eroding satisfaction; under-served seller base requiring cross-state shipping; cultural/expectation differences (e.g., stricter review baselines).",
            ],
            "supporting_data": {"lowest_5_states_mean_review":
                                {k: r4(v) for k, v in lowest_states.items()}},
        },
    })

    # 18: Why are some products missing weight/dimension information?
    missing_weight = pr.product_weight_g.isna().sum()
    missing_cat = pr.product_category_name.isna().sum()
    register({
        "qid": "root_cause_18",
        "category": "root_cause",
        "min_hops": 0,
        "tables_involved": ["products"],
        "question": "Some products in the catalogue have missing weight or dimension fields. What plausible data-pipeline reasons could explain these gaps?",
        "sql_query": "SELECT COUNT(*) FILTER (WHERE product_weight_g IS NULL) FROM products;",
        "pandas_code": "products.product_weight_g.isna().sum()",
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                f"Products missing weight_g: {int(missing_weight)} ({missing_weight/len(pr)*100:.2f}%).",
                f"Products missing category: {int(missing_cat)} ({missing_cat/len(pr)*100:.2f}%).",
                "Plausible drivers: optional fields at seller upload time; legacy listings predating mandatory schema; bulk imports with incomplete metadata; categories where dimensions are not commercially meaningful (e.g. digital goods).",
            ],
            "supporting_data": {"missing_weight_count": int(missing_weight),
                                "missing_category_count": int(missing_cat)},
        },
    })

    # 19: Why does the November 2017 spike occur?
    o_dated = o[o.order_purchase_timestamp.notna()].copy()
    by_ym = o_dated.assign(ym=o_dated.order_purchase_timestamp.dt.to_period("M")).ym.value_counts().sort_index()
    nov17 = by_ym.get(pd.Period("2017-11", "M"), 0)
    surrounding = [by_ym.get(pd.Period(ym, "M"), 0) for ym in ["2017-09", "2017-10", "2017-12", "2018-01"]]
    avg_surr = float(np.mean(surrounding))
    register({
        "qid": "root_cause_19",
        "category": "root_cause",
        "min_hops": 0,
        "tables_involved": ["orders"],
        "question": "The dataset shows a pronounced spike in orders during November 2017. What plausibly drives this spike?",
        "sql_query": "-- monthly order counts.",
        "pandas_code": "orders.assign(ym=orders.order_purchase_timestamp.dt.to_period('M')).ym.value_counts().sort_index()",
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                f"November 2017 had {int(nov17):,} orders versus a surrounding 4-month mean of {avg_surr:,.0f}.",
                "Plausible drivers: Brazil's 'Black Friday' (largest commercial event of the year in late November); pre-Christmas gift purchases concentrated in November (delivery cushion before December 25); platform-led marketing campaigns and seller discounts amplifying baseline demand.",
            ],
            "supporting_data": {"nov_2017_orders": int(nov17),
                                "surrounding_4mo_mean": r2(avg_surr),
                                "spike_factor": r2(nov17 / avg_surr) if avg_surr else None},
        },
    })

    # 20: Why does the cancellation rate vary across payment types?
    o_pay_t = o.merge(p[["order_id", "payment_type"]].drop_duplicates("order_id"), on="order_id")
    cancel_by_pay = o_pay_t.groupby("payment_type").order_status.apply(lambda s_: (s_ == "canceled").mean())
    register({
        "qid": "root_cause_20",
        "category": "root_cause",
        "min_hops": 1,
        "tables_involved": ["orders", "payments"],
        "question": "Cancellation rates differ across payment types in the dataset. What plausible mechanisms link payment method to cancellation likelihood?",
        "sql_query": "-- groupby payment_type, mean(status='canceled').",
        "pandas_code": "<see registry source>",
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                f"Cancellation rate by payment type: {cancel_by_pay.round(4).to_dict()}",
                "Plausible drivers: boleto orders can lapse if the customer never pays the slip (auto-cancellation); credit_card auto-clears so cancellation requires explicit action; voucher payments are often promotional with shorter validity windows; fraud-flag cancellations correlate with credit_card more than with prepaid voucher or boleto.",
            ],
            "supporting_data": {"cancel_rate_by_payment": {k: r4(v) for k, v in cancel_by_pay.items()}},
        },
    })

    # 21: Why does the seller catalogue concentrate on a few top sellers?
    seller_listings = pr.merge(it[["product_id", "seller_id"]].drop_duplicates(), on="product_id") \
                          .groupby("seller_id").product_id.nunique()
    top_share = seller_listings.nlargest(10).sum() / seller_listings.sum()
    register({
        "qid": "root_cause_21",
        "category": "root_cause",
        "min_hops": 2,
        "tables_involved": ["sellers", "order_items", "products"],
        "question": "A small number of sellers list a disproportionate share of distinct products. What plausible mechanisms explain this concentration?",
        "sql_query": "-- per-seller distinct product counts; cumulative share of top 10.",
        "pandas_code": "<see registry source>",
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                f"Top-10 sellers account for {top_share*100:.1f}% of all listed distinct products.",
                "Plausible drivers: power-law dynamics in marketplace catalogues; professional/reseller storefronts versus single-product hobby sellers; platform-level promotion of top sellers; economies of scale in logistics.",
            ],
            "supporting_data": {"top10_seller_listing_share": r4(top_share)},
        },
    })

    # 22: Why are review response times so variable?
    rv_resp = rv[rv.review_answer_timestamp.notna() & rv.review_creation_date.notna()].copy()
    rv_resp["h"] = (rv_resp.review_answer_timestamp - rv_resp.review_creation_date).dt.total_seconds() / 3600.0
    p50 = rv_resp.h.median()
    p95 = rv_resp.h.quantile(0.95)
    register({
        "qid": "root_cause_22",
        "category": "root_cause",
        "min_hops": 0,
        "tables_involved": ["reviews"],
        "question": "Sellers' response time to customer reviews varies from minutes to weeks. What plausible operational factors drive this large spread?",
        "sql_query": "-- median, p95 of (answer_ts - creation_date) hours.",
        "pandas_code": "<see registry source>",
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                f"Median response time: {r2(p50)} hours; P95: {r2(p95)} hours.",
                "Plausible drivers: seller scale (large sellers staff customer-service; small sellers may not log in for days); review polarity (negative reviews may be triaged faster, or alternatively avoided); weekend/holiday timing; automated vs. manual response workflows.",
            ],
            "supporting_data": {"median_response_h": r2(p50), "p95_response_h": r2(p95)},
        },
    })

    # 23: Why might some review_ids appear on multiple orders?
    rv_dup = rv.review_id.value_counts()
    dup_count = (rv_dup > 1).sum()
    register({
        "qid": "root_cause_23",
        "category": "root_cause",
        "min_hops": 0,
        "tables_involved": ["reviews"],
        "question": "Some review_ids in the dataset are associated with more than one order. What plausible data-modelling or operational reasons would explain this?",
        "sql_query": "SELECT review_id, COUNT(*) FROM order_reviews GROUP BY review_id HAVING COUNT(*) > 1;",
        "pandas_code": "reviews.review_id.value_counts()[reviews.review_id.value_counts() > 1]",
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                f"Number of review_ids appearing on >1 order: {int(dup_count)}.",
                "Plausible drivers: a customer received one survey that retrospectively covers a multi-order purchasing experience; survey link sent for the most recent order but referencing the cumulative experience; export/ETL artefact where the review-survey table is joined to the orders table at a coarser grain.",
            ],
            "supporting_data": {"n_duplicate_review_ids": int(dup_count)},
        },
    })

    # 24: Why are repeat customers rare in this dataset?
    orders_per_uniqcust = o.merge(c[["customer_id", "customer_unique_id"]], on="customer_id") \
                            .groupby("customer_unique_id").size()
    repeat_share = (orders_per_uniqcust > 1).mean()
    register({
        "qid": "root_cause_24",
        "category": "root_cause",
        "min_hops": 1,
        "tables_involved": ["customers", "orders"],
        "question": "The vast majority of unique customers placed only one order in the dataset. What plausible reasons explain this low repeat-purchase rate?",
        "sql_query": "-- count of orders per customer_unique_id; share with >1 order.",
        "pandas_code": "orders.merge(customers, on='customer_id').groupby('customer_unique_id').size()",
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                f"Share of unique customers with >1 order: {repeat_share*100:.1f}%.",
                "Plausible drivers: short dataset window (~2 years) limits visible repeat windows; marketplace positioning favouring price discovery rather than loyalty; categories dominated by one-off purchases (electronics, furniture) rather than consumables; lack of loyalty programs.",
            ],
            "supporting_data": {"repeat_customer_share": r4(repeat_share)},
        },
    })

    # 25: Why does the seller_state distribution heavily favour SP?
    s_state = s.seller_state.value_counts(normalize=True)
    sp_share = s_state.get("SP", 0)
    register({
        "qid": "root_cause_25",
        "category": "root_cause",
        "min_hops": 1,
        "tables_involved": ["sellers"],
        "question": "São Paulo state hosts a disproportionate share of Olist sellers. What plausible structural factors drive this concentration?",
        "sql_query": "SELECT seller_state, COUNT(*) FROM sellers GROUP BY seller_state ORDER BY 2 DESC LIMIT 5;",
        "pandas_code": "sellers.seller_state.value_counts(normalize=True)",
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                f"Share of sellers based in SP: {sp_share*100:.1f}%.",
                "Plausible drivers: SP is Brazil's economic and logistics hub (GDP, infrastructure, ports); Olist's HQ and partner network originate from the Southeast; access to courier services and last-mile logistics is best in SP; seller acquisition efforts historically focused on the Southeast.",
            ],
            "supporting_data": {"sp_seller_share": r4(sp_share)},
        },
    })

    # 26: Why might some orders have negative or zero freight_value?
    zero_freight = (it.freight_value == 0).sum()
    neg_freight = (it.freight_value < 0).sum()
    register({
        "qid": "root_cause_26",
        "category": "root_cause",
        "min_hops": 0,
        "tables_involved": ["order_items"],
        "question": "Some order items show a freight value of zero. What plausible reasons exist for these zero-freight entries?",
        "sql_query": "SELECT COUNT(*) FROM order_items WHERE freight_value = 0;",
        "pandas_code": "(order_items.freight_value == 0).sum()",
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                f"Number of items with freight_value = 0: {int(zero_freight)}.",
                f"Number of items with freight_value < 0: {int(neg_freight)}.",
                "Plausible drivers: free-shipping promotions absorbed by the seller; multi-item orders where freight was charged once and bundled to the first line item only; data-export artefacts where the freight was reallocated; pickup-in-store options.",
            ],
            "supporting_data": {"zero_freight_count": int(zero_freight),
                                "negative_freight_count": int(neg_freight)},
        },
    })

    # 27: Why is the price distribution so heavy-tailed?
    pr_skew = float(it.price.skew())
    p99 = it.price.quantile(0.99)
    p50 = it.price.median()
    register({
        "qid": "root_cause_27",
        "category": "root_cause",
        "min_hops": 0,
        "tables_involved": ["order_items"],
        "question": "The distribution of item prices is highly right-skewed. What plausible mechanisms explain why a small fraction of items are dramatically more expensive than the median?",
        "sql_query": "SELECT MEDIAN(price), PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY price) FROM order_items;",
        "pandas_code": "order_items.price.skew(); order_items.price.quantile([0.5, 0.99])",
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                f"Price skewness: {pr_skew:.2f}; median R$ {r2(p50)} vs P99 R$ {r2(p99)} ({p99/p50:.1f}x).",
                "Plausible drivers: heterogeneous product categories on a single platform (electronics, furniture, jewellery alongside small consumables); luxury or high-ticket items as a long tail; pricing of bundled/multi-unit listings; B2B-style listings appearing alongside consumer ones.",
            ],
            "supporting_data": {"skewness": r2(pr_skew),
                                "median_price": r2(p50),
                                "p99_price": r2(p99)},
        },
    })

    # 28: Why do some orders span very long approval delays even for credit_card?
    cc_app = o.merge(p[["order_id", "payment_type"]].drop_duplicates("order_id"), on="order_id") \
                .query("payment_type == 'credit_card'")
    cc_app = cc_app[cc_app.order_approved_at.notna()].copy()
    cc_app["h"] = (cc_app.order_approved_at - cc_app.order_purchase_timestamp).dt.total_seconds() / 3600.0
    long_cc_n = (cc_app.h > 24).sum()
    register({
        "qid": "root_cause_28",
        "category": "root_cause",
        "min_hops": 2,
        "tables_involved": ["orders", "payments"],
        "question": "Most credit-card payments are approved within minutes, yet a small share take more than 24 hours. What plausible factors explain this long-tail behaviour?",
        "sql_query": "-- groupby payment_type='credit_card', share with approval > 24h.",
        "pandas_code": "<see registry source>",
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                f"Credit-card orders with >24h approval delay: {int(long_cc_n):,} ({long_cc_n / max(len(cc_app), 1) * 100:.2f}% of credit-card orders).",
                "Plausible drivers: anti-fraud manual review queues (high-value orders, new accounts, suspicious patterns); card-issuer authorization disputes; weekend/holiday processing delays at the payment-processor; multi-method orders awaiting other methods to resolve before the order is marked approved.",
            ],
            "supporting_data": {"cc_approval_gt_24h_count": int(long_cc_n)},
        },
    })

    # 29: Why does freight ratio vary so much within a single category?
    bbt_pids = pr.query("product_category_name_english == 'bed_bath_table'").product_id
    bbt_items = it[it.product_id.isin(bbt_pids)]
    bbt_ratio = (bbt_items.freight_value / bbt_items.price.replace(0, np.nan))
    register({
        "qid": "root_cause_29",
        "category": "root_cause",
        "min_hops": 3,
        "tables_involved": ["products", "order_items", "sellers", "customers"],
        "question": "Within a single product category (e.g. bed_bath_table), the freight-to-price ratio varies widely across orders. What plausible factors explain this within-category spread?",
        "sql_query": "-- per-category distribution of freight/price ratio across orders.",
        "pandas_code": "<see registry source>",
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                f"bed_bath_table freight/price ratio: median {r4(bbt_ratio.median())}, P10 {r4(bbt_ratio.quantile(0.10))}, P90 {r4(bbt_ratio.quantile(0.90))}.",
                "Plausible drivers: vast distance variation between seller and customer (same-state vs cross-country); product weight/volume range within a category (small towel vs king-size mattress topper); carrier/shipping-method choice (express vs standard); per-item versus per-order freight allocation in multi-item orders.",
            ],
            "supporting_data": {"bbt_ratio_median": r4(bbt_ratio.median()),
                                "bbt_ratio_p10": r4(bbt_ratio.quantile(0.10)),
                                "bbt_ratio_p90": r4(bbt_ratio.quantile(0.90))},
        },
    })

    # 30: Why might the dataset terminate abruptly in late 2018?
    last_orders = o.order_purchase_timestamp.dropna().sort_values().tail(20)
    last_date = o.order_purchase_timestamp.max()
    register({
        "qid": "root_cause_30",
        "category": "root_cause",
        "min_hops": 0,
        "tables_involved": ["orders"],
        "question": "Order volume tapers steeply in the last few months of the dataset (late 2018). What plausible reasons could explain this drop-off?",
        "sql_query": "SELECT MAX(order_purchase_timestamp), MIN(order_purchase_timestamp) FROM orders;",
        "pandas_code": "orders.order_purchase_timestamp.max()",
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                f"Last purchase timestamp in the dataset: {last_date}.",
                "Plausible reasons: the public release truncates the data at a specific snapshot date; pending orders later in the window were not yet fully labelled (review, delivery) at export time; legal/privacy clearance windows; Olist's deliberate aggregation cut-off for anonymisation.",
            ],
            "supporting_data": {"last_order_purchase_ts": str(last_date)},
        },
    })


# ============================================================
# CATEGORY 5 — GLOBAL  (all-data, rubric-based)
# ============================================================
# Diese Fragen verlangen einen synoptischen Überblick über das
# gesamte Datenset. Antworten sind Rubriken.

def build_global(t: dict[str, pd.DataFrame]) -> None:
    o, p, it, rv, c, s, pr = (t["orders"], t["payments"], t["order_items"],
                              t["reviews"], t["customers"], t["sellers"],
                              t["products"])

    # 1: Main drivers of negative reviews
    rv_o = rv.merge(o[["order_id", "order_status", "order_delivered_customer_date",
                         "order_estimated_delivery_date"]], on="order_id")
    rv_o["late"] = (rv_o.order_delivered_customer_date > rv_o.order_estimated_delivery_date).fillna(False)
    rv_o["cancelled"] = (rv_o.order_status == "canceled")
    neg = rv_o.query("review_score <= 2")
    register({
        "qid": "global_01",
        "category": "global",
        "min_hops": "all",
        "tables_involved": ["reviews", "orders", "order_items", "products"],
        "question": "What are the main drivers of negative reviews (1–2 stars) in the Olist dataset overall?",
        "sql_query": "-- aggregated indicators across the entire dataset.",
        "pandas_code": "<see registry source>",
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                f"Among 1-2-star reviews: {neg.late.mean()*100:.1f}% involve a late delivery vs {rv_o.query('review_score >= 4').late.mean()*100:.1f}% among 4-5-star reviews.",
                f"Among 1-2-star reviews: {neg.cancelled.mean()*100:.1f}% involve a cancelled order.",
                "Late delivery is the single largest correlate of negative reviews. Cancelled orders, lengthy approval times, and high freight-to-price ratios are secondary correlates.",
                "Category-level effects exist (bed_bath_table, furniture_decor underperform) but explain less variance than delivery performance.",
            ],
            "supporting_data": {
                "late_share_neg": r4(neg.late.mean()),
                "late_share_pos": r4(rv_o.query("review_score >= 4").late.mean()),
                "cancelled_share_neg": r4(neg.cancelled.mean()),
            },
        },
    })

    # 2: Overall marketplace structure
    register({
        "qid": "global_02",
        "category": "global",
        "min_hops": "all",
        "tables_involved": ["customers", "orders", "order_items", "sellers", "products"],
        "question": "Provide a high-level overview of the Olist marketplace: scale, time span, geographic skew, product mix, and order outcomes.",
        "sql_query": "-- aggregations across all tables.",
        "pandas_code": "<multiple aggregations>",
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                f"Scale: {len(o):,} orders, {len(c):,} customers (with {c.customer_unique_id.nunique():,} unique), {len(s):,} sellers, {len(pr):,} products across {pr.product_category_name_english.nunique()} categories.",
                f"Time span: {o.order_purchase_timestamp.min().date()} to {o.order_purchase_timestamp.max().date()} (~{(o.order_purchase_timestamp.max() - o.order_purchase_timestamp.min()).days} days).",
                f"Geographic skew: top-3 customer states (SP, RJ, MG) account for ~{c.customer_state.value_counts().head(3).sum()/len(c)*100:.1f}% of customers; top-3 seller states cover an even larger share of sellers.",
                f"Order outcomes: delivered = {(o.order_status=='delivered').mean()*100:.1f}%, canceled = {(o.order_status=='canceled').mean()*100:.1f}%, other statuses < 3%.",
                f"Product mix: largest category 'bed_bath_table' has {(pr.product_category_name_english=='bed_bath_table').sum()} distinct products.",
            ],
            "supporting_data": {
                "n_orders": int(len(o)),
                "n_customers_unique": int(c.customer_unique_id.nunique()),
                "n_sellers": int(len(s)),
                "n_products": int(len(pr)),
                "delivered_share": r4((o.order_status=='delivered').mean()),
                "canceled_share": r4((o.order_status=='canceled').mean()),
            },
        },
    })

    # 3: Overall payment patterns
    pt_share = p.payment_type.value_counts(normalize=True)
    inst_dist = p.payment_installments.describe()
    register({
        "qid": "global_03",
        "category": "global",
        "min_hops": "all",
        "tables_involved": ["payments"],
        "question": "Summarise the payment behaviour in the Olist marketplace: dominant payment types, instalment usage, and any noteworthy patterns.",
        "sql_query": "-- aggregated statistics on payment_type and payment_installments.",
        "pandas_code": "payments.payment_type.value_counts(normalize=True); payments.payment_installments.describe()",
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                f"Credit card dominates: {pt_share.get('credit_card', 0)*100:.1f}% of payment entries.",
                f"Boleto is the #2 method: {pt_share.get('boleto', 0)*100:.1f}%.",
                f"Mean instalments overall: {inst_dist['mean']:.2f}; median: {inst_dist['50%']:.0f}; max: {int(inst_dist['max'])}.",
                f"Vouchers and debit cards together represent only {(pt_share.get('voucher', 0) + pt_share.get('debit_card', 0))*100:.1f}% of payment entries.",
                "Brazilian-specific patterns are visible: heavy use of multi-instalment credit-card plans on higher-value orders; boleto remains relevant for unbanked or credit-averse segments.",
            ],
            "supporting_data": {
                "payment_type_shares": {k: r4(v) for k, v in pt_share.items()},
                "instalments_mean": r2(inst_dist["mean"]),
                "instalments_median": r2(inst_dist["50%"]),
            },
        },
    })

    # 4: Overall delivery performance
    o_delivered = o.query("order_status == 'delivered' and order_delivered_customer_date.notna() and order_estimated_delivery_date.notna()")
    delivery_d = (o_delivered.order_delivered_customer_date - o_delivered.order_purchase_timestamp).dt.total_seconds() / 86400.0
    ontime = (o_delivered.order_delivered_customer_date <= o_delivered.order_estimated_delivery_date).mean()
    register({
        "qid": "global_04",
        "category": "global",
        "min_hops": "all",
        "tables_involved": ["orders", "customers"],
        "question": "How does the Olist marketplace perform on delivery overall? Report typical delivery times, on-time rates, and notable spreads.",
        "sql_query": "-- aggregations across delivered orders with non-null timestamps.",
        "pandas_code": "<see registry source>",
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                f"Mean delivery time: {r2(delivery_d.mean())} days; median: {r2(delivery_d.median())} days.",
                f"95th-percentile delivery time: {r2(delivery_d.quantile(0.95))} days (long tail).",
                f"On-time rate (delivered on or before estimated date): {ontime*100:.1f}%.",
                "Performance is strongly geography-dependent: Southeast states see faster and more predictable delivery; North/Northeast endure longer transit and lower on-time rates.",
            ],
            "supporting_data": {
                "mean_delivery_days": r2(delivery_d.mean()),
                "median_delivery_days": r2(delivery_d.median()),
                "p95_delivery_days": r2(delivery_d.quantile(0.95)),
                "ontime_rate": r4(ontime),
            },
        },
    })

    # 5: What characterises high-performing sellers?
    seller_perf = (it.merge(rv[["order_id", "review_score"]], on="order_id")
                     .groupby("seller_id")
                     .agg(n_items=("order_item_id", "size"),
                          mean_review=("review_score", "mean")))
    high = seller_perf.query("n_items >= 50 and mean_review >= 4.5")
    register({
        "qid": "global_05",
        "category": "global",
        "min_hops": "all",
        "tables_involved": ["sellers", "order_items", "reviews"],
        "question": "What characterises the highest-performing sellers on Olist (those with high volume and high review scores)?",
        "sql_query": "-- supporting: per-seller aggregations of n_items and mean review_score.",
        "pandas_code": "<see registry source>",
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                f"Sellers with >=50 items and mean review >=4.5: {len(high)} sellers (out of {seller_perf.query('n_items >= 50').shape[0]} active sellers).",
                "High-performing sellers tend to: (i) concentrate in lighter, more standardised product categories where freight and damage risk are low; (ii) maintain shorter approval-to-shipping gaps; (iii) cluster in Southeastern states with strong logistics access.",
                "Volume and quality are not strongly correlated overall: a number of high-volume sellers carry mean review scores below 4.0, indicating that scale alone is not a sufficient quality signal.",
            ],
            "supporting_data": {
                "n_high_perf_sellers": int(len(high)),
                "n_active_sellers": int(seller_perf.query("n_items >= 50").shape[0]),
            },
        },
    })

    # 6: How does seasonality affect order volume?
    o_monthly = o.assign(month=o.order_purchase_timestamp.dt.to_period("M")) \
                  .month.value_counts().sort_index()
    register({
        "qid": "global_06",
        "category": "global",
        "min_hops": "all",
        "tables_involved": ["orders"],
        "question": "How does order volume vary over time across the Olist dataset, and what seasonal patterns are visible?",
        "sql_query": "-- monthly count of orders.",
        "pandas_code": "orders.assign(m=orders.order_purchase_timestamp.dt.to_period('M')).m.value_counts().sort_index()",
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                "Order volume grows substantially from 2016 (very few orders) through 2017 (ramping) to a peak in late 2017 / early 2018.",
                "A pronounced spike in November 2017 (Black Friday) — roughly 2× the surrounding-month average.",
                "Coverage tapers in late 2018 (incomplete sample).",
            ],
            "supporting_data": {
                "monthly_order_counts_head_tail": {
                    "first_3_months": {str(k): int(v) for k, v in o_monthly.head(3).items()},
                    "peak_month": str(o_monthly.idxmax()),
                    "peak_month_count": int(o_monthly.max()),
                    "last_3_months": {str(k): int(v) for k, v in o_monthly.tail(3).items()},
                },
            },
        },
    })

    # 7: Customer concentration: returning vs one-off
    orders_per_uniqcust = o.merge(c[["customer_id", "customer_unique_id"]], on="customer_id") \
                            .groupby("customer_unique_id").size()
    repeat_share = (orders_per_uniqcust > 1).mean()
    register({
        "qid": "global_07",
        "category": "global",
        "min_hops": "all",
        "tables_involved": ["customers", "orders"],
        "question": "Are most Olist customers one-off buyers or repeat purchasers? Characterise customer concentration.",
        "sql_query": "-- count of orders per customer_unique_id (the persistent customer key).",
        "pandas_code": "orders.merge(customers, on='customer_id').groupby('customer_unique_id').size()",
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                f"Share of unique customers with more than one order: {repeat_share*100:.1f}%.",
                f"Mean orders per unique customer: {orders_per_uniqcust.mean():.2f}; max: {int(orders_per_uniqcust.max())}.",
                "The vast majority of customers are one-off purchasers; Olist's user base is dominated by single-transaction activity.",
            ],
            "supporting_data": {
                "repeat_customer_share": r4(repeat_share),
                "mean_orders_per_unique_customer": r2(orders_per_uniqcust.mean()),
                "max_orders_for_one_customer": int(orders_per_uniqcust.max()),
            },
        },
    })

    # 8: Which categories generate the most revenue?
    cat_rev_all = (it.merge(pr[["product_id", "product_category_name_english"]], on="product_id")
                     .groupby("product_category_name_english").price.sum()
                     .sort_values(ascending=False))
    register({
        "qid": "global_08",
        "category": "global",
        "min_hops": "all",
        "tables_involved": ["order_items", "products"],
        "question": "Which product categories generate the most revenue overall, and how concentrated is revenue across the long-tail of categories?",
        "sql_query": "-- per-category sum of price; Pareto share of top-10 categories.",
        "pandas_code": "<see registry source>",
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                f"Top-5 categories by revenue: {cat_rev_all.head(5).round(0).astype(int).to_dict()}",
                f"Top-10 categories account for ~{cat_rev_all.head(10).sum() / cat_rev_all.sum() * 100:.1f}% of all revenue.",
                "Revenue follows a classic long-tail: a small number of categories dominate while the majority contribute marginally.",
            ],
            "supporting_data": {
                "top10_share_pct": r2(cat_rev_all.head(10).sum() / cat_rev_all.sum() * 100),
                "top5_revenue": {k: r2(v) for k, v in cat_rev_all.head(5).items()},
            },
        },
    })

    # 9: How do free-text review comments distribute?
    rv_text = rv[rv.review_comment_message.notna()]
    share_with_text = len(rv_text) / len(rv)
    avg_len = rv_text.review_comment_message.str.len().mean()
    register({
        "qid": "global_09",
        "category": "global",
        "min_hops": "all",
        "tables_involved": ["reviews"],
        "question": "How prevalent are free-text comments in customer reviews, and what does this say about the review signal?",
        "sql_query": "-- count of non-null review_comment_message; mean text length.",
        "pandas_code": "reviews.review_comment_message.notna().mean(); reviews.review_comment_message.str.len().mean()",
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                f"Share of reviews with a non-null comment message: {share_with_text*100:.1f}%.",
                f"Mean comment length (characters): {avg_len:.0f}.",
                "Most reviews carry only a numeric score; free-text feedback is the minority. This limits NLP-based root-cause analysis and motivates structural (KG-based) approaches to explain review variation.",
            ],
            "supporting_data": {
                "share_with_comment_text": r4(share_with_text),
                "mean_comment_length_chars": r2(avg_len),
            },
        },
    })

    # 10: Aggregate data quality issues
    register({
        "qid": "global_10",
        "category": "global",
        "min_hops": "all",
        "tables_involved": ["all"],
        "question": "What are the main data-quality issues a practitioner should be aware of when working with the Olist dataset?",
        "sql_query": "-- structural audit across all tables.",
        "pandas_code": "<see Section 4.3 audit results>",
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                "Foreign-key integrity is excellent: zero orphan child rows across all primary FK relations.",
                "775 orders have no line items (predominantly canceled/unavailable status) and 768 orders lack reviews.",
                "814 review_ids are duplicated across multiple orders (modelled as 1:n review→order in the KG).",
                "1,359 orders show carrier_pickup before approval (a known export artefact).",
                "610 products carry a null category and 13 use category strings not present in the translation table.",
                "2,961 orders involve multiple payment entries (max 29 entries on a single order).",
            ],
            "supporting_data": {
                "orphan_child_rows": 0,
                "orders_without_items": 775,
                "orders_without_reviews": 768,
                "duplicate_review_ids": 814,
                "carrier_before_approval": 1359,
                "products_without_category": 610,
                "missing_categories_in_translation": 13,
                "multi_payment_orders": 2961,
            },
        },
    })

    # 11: Overall delivery vs review correlation
    rv_o2 = rv.merge(o[["order_id", "order_delivered_customer_date",
                          "order_estimated_delivery_date"]], on="order_id")
    rv_o2 = rv_o2[rv_o2.order_delivered_customer_date.notna() &
                   rv_o2.order_estimated_delivery_date.notna()]
    rv_o2["late_days"] = (rv_o2.order_delivered_customer_date -
                            rv_o2.order_estimated_delivery_date).dt.total_seconds() / 86400.0
    corr = rv_o2[["review_score", "late_days"]].corr().iloc[0, 1]
    register({
        "qid": "global_11",
        "category": "global",
        "min_hops": "all",
        "tables_involved": ["reviews", "orders"],
        "question": "Is there a meaningful relationship between delivery lateness and review score across the entire Olist dataset?",
        "sql_query": "-- correlation between review_score and (delivered - estimated) days.",
        "pandas_code": "<see registry source>",
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                f"Pearson correlation between review_score and lateness (days past estimate): {corr:.3f}.",
                "Negative correlation: later delivery → lower scores. The effect is moderate in size but consistent, confirming delivery as a central quality driver in this marketplace.",
            ],
            "supporting_data": {
                "pearson_review_vs_late_days": r4(corr),
            },
        },
    })

    # 12: Which segments would Olist most need to invest in?
    register({
        "qid": "global_12",
        "category": "global",
        "min_hops": "all",
        "tables_involved": ["all"],
        "question": "Based on the patterns visible in the dataset, which segments or operational areas would Olist most benefit from investing in to improve customer experience?",
        "sql_query": "-- synthesises results across delivery-time, regional, and category analyses.",
        "pandas_code": "<derived from preceding global_* aggregations>",
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                "Logistics in the North and Northeast: largest source of delivery delays and low review scores.",
                "Underperforming product categories (bed_bath_table, furniture_decor) for product-quality / description-accuracy improvements.",
                "Seller onboarding outside the Southeast to shorten last-mile distances.",
                "Streamlined approval for boleto payments to reduce the 1–3 day approval gap that compresses delivery windows.",
                "Mechanisms to encourage repeat-purchase behaviour given the very high one-off share of customers.",
            ],
            "supporting_data": {
                "rationale": "Synthesises root_cause_01 (North delivery), root_cause_03 (bed_bath_table), root_cause_11 (seller concentration), root_cause_12 (boleto approval delay), and global_07 (one-off customer share)."
            },
        },
    })

    # --- global_13..30: 18 additional global questions ---

    # 13: Day-of-week patterns
    o_dow = o[o.order_purchase_timestamp.notna()].copy()
    o_dow["dow_name"] = o_dow.order_purchase_timestamp.dt.day_name()
    dow_dist = o_dow.dow_name.value_counts()
    register({
        "qid": "global_13",
        "category": "global",
        "min_hops": "all",
        "tables_involved": ["orders"],
        "question": "Across the entire dataset, how does order volume vary by day of the week?",
        "sql_query": (
            "SELECT TO_CHAR(order_purchase_timestamp, 'Day'), COUNT(*) FROM orders "
            "GROUP BY 1 ORDER BY 2 DESC;"),
        "pandas_code": "orders.order_purchase_timestamp.dt.day_name().value_counts()",
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                f"Day-of-week order counts: {dow_dist.to_dict()}",
                "Weekdays (Monday-Friday) dominate, peaking mid-week; weekend volume drops noticeably. Pattern is consistent with B2C e-commerce in Brazil where shoppers browse during the week and complete purchases at work-hour-adjacent times.",
            ],
            "supporting_data": {"orders_by_day_of_week": {k: int(v) for k, v in dow_dist.items()}},
        },
    })

    # 14: Hour-of-day patterns
    o_hour = o[o.order_purchase_timestamp.notna()].copy()
    o_hour["h"] = o_hour.order_purchase_timestamp.dt.hour
    peak_hour = int(o_hour.h.value_counts().idxmax())
    hour_dist = o_hour.h.value_counts().sort_index()
    register({
        "qid": "global_14",
        "category": "global",
        "min_hops": "all",
        "tables_involved": ["orders"],
        "question": "At which hours of the day do customers most often place Olist orders?",
        "sql_query": (
            "SELECT EXTRACT(HOUR FROM order_purchase_timestamp), COUNT(*) FROM orders "
            "GROUP BY 1 ORDER BY 1;"),
        "pandas_code": "orders.order_purchase_timestamp.dt.hour.value_counts().sort_index()",
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                f"Peak purchasing hour: {peak_hour}:00 (24h clock).",
                f"Top-3 hours: {hour_dist.nlargest(3).to_dict()}",
                "Activity is bimodal: lunchtime peak (~12-14h) and evening peak (~20-22h). The very late hours (00-06h) show a sharp drop.",
            ],
            "supporting_data": {"peak_hour": peak_hour,
                                "top3_hours": {int(k): int(v) for k, v in hour_dist.nlargest(3).items()}},
        },
    })

    # 15: Inter-regional trade flows (where do customers and sellers connect?)
    cust_reg_t = add_region_col(c, "customer_state").rename(columns={"region": "cust_region"})
    seller_reg = add_region_col(s, "seller_state").rename(columns={"region": "sell_region"})
    flow = (it.merge(seller_reg[["seller_id", "sell_region"]], on="seller_id")
              .merge(o[["order_id", "customer_id"]], on="order_id")
              .merge(cust_reg_t[["customer_id", "cust_region"]], on="customer_id"))
    flow_matrix = flow.groupby(["sell_region", "cust_region"]).size().unstack(fill_value=0)
    register({
        "qid": "global_15",
        "category": "global",
        "min_hops": "all",
        "tables_involved": ["sellers", "order_items", "orders", "customers"],
        "question": "Describe inter-regional trade flows in the Olist marketplace: from which seller regions do orders flow to which customer regions?",
        "sql_query": "-- pivot: groupby (seller_region, customer_region), count.",
        "pandas_code": "<see flow_matrix in registry source>",
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                f"Largest flow: Southeast→Southeast with {int(flow_matrix.loc['Southeast', 'Southeast']):,} item shipments.",
                f"Largest cross-region flow: Southeast→{flow_matrix.loc['Southeast'].drop('Southeast').idxmax()} with {int(flow_matrix.loc['Southeast'].drop('Southeast').max()):,} shipments.",
                "Southeast sellers dominate outflows to every region; intra-Southeast and Southeast→South flows dominate the total volume. Other regions ship comparatively little.",
            ],
            "supporting_data": {
                "flow_matrix": {k: {kk: int(vv) for kk, vv in v.items()} for k, v in flow_matrix.to_dict().items()},
            },
        },
    })

    # 16: Customer geographic concentration
    state_dist = c.customer_state.value_counts(normalize=True)
    top3_states = state_dist.head(3)
    register({
        "qid": "global_16",
        "category": "global",
        "min_hops": "all",
        "tables_involved": ["customers"],
        "question": "Provide a summary of customer geographic distribution across the 27 Brazilian states.",
        "sql_query": "SELECT customer_state, COUNT(*) FROM customers GROUP BY customer_state ORDER BY 2 DESC;",
        "pandas_code": "customers.customer_state.value_counts(normalize=True)",
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                f"Top-3 states by customer count: {top3_states.round(3).to_dict()}",
                f"Top-3 share of total customers: {top3_states.sum()*100:.1f}%.",
                "Distribution mirrors the Brazilian population centres but skews even more toward the Southeast — consistent with where e-commerce adoption is highest.",
            ],
            "supporting_data": {"top3_state_shares": {k: r4(v) for k, v in top3_states.items()}},
        },
    })

    # 17: Seller productivity distribution
    seller_n_items = it.groupby("seller_id").size().sort_values(ascending=False)
    top10_share = seller_n_items.head(10).sum() / seller_n_items.sum()
    median_seller = seller_n_items.median()
    register({
        "qid": "global_17",
        "category": "global",
        "min_hops": "all",
        "tables_involved": ["sellers", "order_items"],
        "question": "How is seller productivity distributed in the Olist marketplace? Is the marketplace dominated by a few large sellers or evenly distributed?",
        "sql_query": "SELECT seller_id, COUNT(*) FROM order_items GROUP BY seller_id ORDER BY 2 DESC;",
        "pandas_code": "order_items.groupby('seller_id').size().sort_values(ascending=False)",
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                f"Top-10 sellers account for {top10_share*100:.1f}% of all items sold.",
                f"Median seller has {median_seller:.0f} items sold; the largest has {int(seller_n_items.iloc[0]):,} items.",
                "Distribution is heavy-tailed (Pareto-like) — typical of multi-sided marketplaces where a small number of professional storefronts dominate volume while a long tail of hobby/occasional sellers contributes the rest.",
            ],
            "supporting_data": {"top10_share": r4(top10_share),
                                "median_seller_items": int(median_seller),
                                "max_seller_items": int(seller_n_items.iloc[0])},
        },
    })

    # 18: Product catalogue characterization (size/weight statistics)
    weight_stats = pr.product_weight_g.describe()
    register({
        "qid": "global_18",
        "category": "global",
        "min_hops": "all",
        "tables_involved": ["products"],
        "question": "Characterise the Olist product catalogue in terms of size and weight: what is the typical product, and how heavy-tailed is the distribution?",
        "sql_query": "SELECT MEAN(product_weight_g), STDDEV(product_weight_g), MAX(product_weight_g) FROM products;",
        "pandas_code": "products.product_weight_g.describe()",
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                f"Product weight: median {weight_stats['50%']:.0f} g, mean {weight_stats['mean']:.0f} g, max {weight_stats['max']:.0f} g.",
                "Mean significantly exceeds median, indicating a right-skewed distribution with heavy outliers (furniture, appliances). Most products are lightweight (under 1 kg), with a long tail of bulky items shaping freight economics.",
            ],
            "supporting_data": {"weight_median_g": r2(weight_stats["50%"]),
                                "weight_mean_g": r2(weight_stats["mean"]),
                                "weight_max_g": r2(weight_stats["max"])},
        },
    })

    # 19: Aggregate seller-region characterization
    seller_state_share = s.seller_state.value_counts(normalize=True)
    top1_seller_share = seller_state_share.iloc[0]
    register({
        "qid": "global_19",
        "category": "global",
        "min_hops": "all",
        "tables_involved": ["sellers"],
        "question": "Summarise the geographic distribution of sellers in the Olist marketplace.",
        "sql_query": "SELECT seller_state, COUNT(*) FROM sellers GROUP BY seller_state ORDER BY 2 DESC;",
        "pandas_code": "sellers.seller_state.value_counts(normalize=True)",
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                f"Top seller state ({seller_state_share.index[0]}) accounts for {top1_seller_share*100:.1f}% of all sellers.",
                f"Top-5 seller states share: {seller_state_share.head(5).sum()*100:.1f}%.",
                "Seller concentration is more extreme than customer concentration, reflecting Brazil's industrial/logistics geography (Southeast dominates) and Olist's seller-acquisition footprint.",
            ],
            "supporting_data": {"top_seller_state": str(seller_state_share.index[0]),
                                "top_seller_state_share": r4(top1_seller_share),
                                "top5_share": r4(seller_state_share.head(5).sum())},
        },
    })

    # 20: Freight cost patterns
    freight_stats = it.freight_value.describe()
    register({
        "qid": "global_20",
        "category": "global",
        "min_hops": "all",
        "tables_involved": ["order_items"],
        "question": "Summarise the freight-cost patterns in the Olist marketplace: typical values, spread, and any noteworthy outliers.",
        "sql_query": "SELECT AVG(freight_value), STDDEV(freight_value), MAX(freight_value) FROM order_items;",
        "pandas_code": "order_items.freight_value.describe()",
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                f"Mean freight: R$ {freight_stats['mean']:.2f}, median R$ {freight_stats['50%']:.2f}, max R$ {freight_stats['max']:.2f}.",
                "Freight is heavy-tailed: median is well below mean, indicating expensive outliers (heavy/distant shipments). A material share of items carries freight equal to or higher than the item price itself.",
            ],
            "supporting_data": {"freight_mean": r2(freight_stats["mean"]),
                                "freight_median": r2(freight_stats["50%"]),
                                "freight_max": r2(freight_stats["max"])},
        },
    })

    # 21: Order-size distribution (items per order)
    items_per_order = it.groupby("order_id").size()
    pct_single = (items_per_order == 1).mean()
    register({
        "qid": "global_21",
        "category": "global",
        "min_hops": "all",
        "tables_involved": ["order_items"],
        "question": "Characterise the distribution of order sizes (number of items per order) in the Olist marketplace.",
        "sql_query": "SELECT COUNT(*) AS n_items, COUNT(*) FROM (SELECT order_id, COUNT(*) AS n_items FROM order_items GROUP BY order_id) sub GROUP BY n_items;",
        "pandas_code": "order_items.groupby('order_id').size().value_counts()",
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                f"Share of single-item orders: {pct_single*100:.1f}%.",
                f"Mean items per order: {items_per_order.mean():.2f}; max items in one order: {int(items_per_order.max())}.",
                "Marketplace is dominated by single-item orders — typical of long-tail e-commerce where customers buy a specific product rather than baskets of items.",
            ],
            "supporting_data": {"share_single_item": r4(pct_single),
                                "mean_items_per_order": r2(items_per_order.mean()),
                                "max_items_per_order": int(items_per_order.max())},
        },
    })

    # 22: Delivery-time trend over the data window
    o_del_t = o.query("order_status == 'delivered'").copy()
    o_del_t["d"] = (o_del_t.order_delivered_customer_date - o_del_t.order_purchase_timestamp).dt.total_seconds() / 86400.0
    o_del_t["q"] = o_del_t.order_purchase_timestamp.dt.to_period("Q")
    deliv_by_q = o_del_t.groupby("q").d.mean()
    register({
        "qid": "global_22",
        "category": "global",
        "min_hops": "all",
        "tables_involved": ["orders"],
        "question": "How does mean delivery time evolve quarter-by-quarter across the dataset window? Is the marketplace improving or worsening over time?",
        "sql_query": "-- groupby quarter, mean delivery duration.",
        "pandas_code": "<see registry source>",
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                f"Quarterly mean delivery (days), first 3 quarters: {dict((str(k), r2(v)) for k, v in deliv_by_q.head(3).items())}",
                f"Quarterly mean delivery (days), last 3 quarters: {dict((str(k), r2(v)) for k, v in deliv_by_q.tail(3).items())}",
                "Delivery times generally improve over the data window as the platform matures and as Olist's seller/logistics network grows — though there's a deterioration around peak demand periods (Q4 2017).",
            ],
            "supporting_data": {"delivery_days_by_quarter":
                                {str(k): r2(v) for k, v in deliv_by_q.items()}},
        },
    })

    # 23: Installment patterns
    inst_dist = p.payment_installments.value_counts().sort_index()
    register({
        "qid": "global_23",
        "category": "global",
        "min_hops": "all",
        "tables_involved": ["payments"],
        "question": "Describe the installment-count patterns in Olist payments. What does the distribution reveal about Brazilian consumer-credit norms?",
        "sql_query": "SELECT payment_installments, COUNT(*) FROM order_payments GROUP BY 1 ORDER BY 1;",
        "pandas_code": "payments.payment_installments.value_counts().sort_index()",
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                f"Top-5 installment counts (by frequency): {inst_dist.nlargest(5).to_dict()}",
                "Concentration at 1 installment (one-time payment), then secondary peaks at 2, 3, 4, 5, 6, 10 — the canonical Brazilian retail offerings ('em 3x sem juros', etc.). Installment counts of 7-9 are notably less common than the round numbers.",
            ],
            "supporting_data": {"top5_installment_buckets":
                                {int(k): int(v) for k, v in inst_dist.nlargest(5).items()}},
        },
    })

    # 24: Review-comment text length characteristics
    rv_txt = rv[rv.review_comment_message.notna() & (rv.review_comment_message.str.strip() != "")]
    txt_lens = rv_txt.review_comment_message.str.len()
    register({
        "qid": "global_24",
        "category": "global",
        "min_hops": "all",
        "tables_involved": ["reviews"],
        "question": "Among reviews that include a free-text comment, describe the length distribution and what it implies for downstream NLP analysis.",
        "sql_query": "SELECT AVG(LENGTH(review_comment_message)), MAX(LENGTH(review_comment_message)) FROM order_reviews WHERE review_comment_message IS NOT NULL;",
        "pandas_code": "reviews.review_comment_message.dropna().str.len().describe()",
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                f"Median comment length: {txt_lens.median():.0f} chars; mean: {txt_lens.mean():.0f}; max: {int(txt_lens.max())}.",
                "Most comments are short (one-to-two sentences). Long comments are rare but informative. For downstream NLP this implies (a) sparse signal per comment, (b) viable for topic clustering but limited for fine-grained sentiment, and (c) downstream models should account for boilerplate ('ótimo produto', 'recomendo') which contributes much of the volume.",
            ],
            "supporting_data": {"comment_length_median": r2(txt_lens.median()),
                                "comment_length_mean": r2(txt_lens.mean()),
                                "comment_length_max": int(txt_lens.max())},
        },
    })

    # 25: Anomalies and outliers summary
    o_temp_anom = ((o.order_delivered_carrier_date < o.order_approved_at) &
                   o.order_delivered_carrier_date.notna() &
                   o.order_approved_at.notna()).sum()
    multi_pay = (p.groupby("order_id").size() > 1).sum()
    dup_rev = (rv.review_id.value_counts() > 1).sum()
    register({
        "qid": "global_25",
        "category": "global",
        "min_hops": "all",
        "tables_involved": ["all"],
        "question": "Summarise the main anomalies and outliers a practitioner should be aware of when modelling on the Olist dataset.",
        "sql_query": "-- structural audit.",
        "pandas_code": "<see registry source>",
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                f"{int(o_temp_anom):,} orders have carrier pickup recorded before approval (temporal artefact).",
                f"{int(multi_pay):,} orders have more than one payment entry.",
                f"{int(dup_rev):,} review_ids appear on more than one order.",
                "Extreme price outliers (luxury items, B2B-style listings) skew distributions; very heavy products (>30kg) inflate freight tail; missing weight/dimension on ~3% of products requires imputation or filtering.",
            ],
            "supporting_data": {
                "temporal_anomalies": int(o_temp_anom),
                "multi_payment_orders": int(multi_pay),
                "duplicate_review_ids": int(dup_rev),
            },
        },
    })

    # 26: Cancellation patterns
    can = o.query("order_status == 'canceled'").copy()
    can["m"] = can.order_purchase_timestamp.dt.month
    register({
        "qid": "global_26",
        "category": "global",
        "min_hops": "all",
        "tables_involved": ["orders", "payments"],
        "question": "Summarise the cancellation patterns in the dataset: rate, timing, and correlations.",
        "sql_query": "-- groupby month / status / payment_type.",
        "pandas_code": "<see registry source>",
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                f"Overall cancellation rate: {(o.order_status == 'canceled').mean()*100:.2f}% of orders.",
                f"Top cancellation months: {can.m.value_counts().head(3).to_dict()}.",
                "Cancellations cluster around high-demand months (November/December — out-of-stock during sales events) and January (post-holiday cooling). Payment type matters: boleto orders auto-cancel if unpaid, inflating boleto cancellation share vs credit_card.",
            ],
            "supporting_data": {"overall_cancel_rate": r4((o.order_status == "canceled").mean()),
                                "top_cancel_months": {int(k): int(v) for k, v in can.m.value_counts().head(3).items()}},
        },
    })

    # 27: Category lifecycle / popularity over time
    cat_q = (o.merge(it[["order_id", "product_id"]], on="order_id")
               .merge(pr[["product_id", "product_category_name_english"]], on="product_id")
               .assign(q=lambda d: d.order_purchase_timestamp.dt.to_period("Q")))
    growth = cat_q.groupby(["q", "product_category_name_english"]).size().unstack(fill_value=0)
    # find top-3 categories with biggest growth from first to last full quarter
    fastest_growing = None
    if len(growth) >= 2:
        first_q, last_q = growth.index[1], growth.index[-2]  # avoid partial first/last
        growth_ratio = (growth.loc[last_q] / (growth.loc[first_q] + 1)).sort_values(ascending=False)
        fastest_growing = growth_ratio.head(3).to_dict()
    register({
        "qid": "global_27",
        "category": "global",
        "min_hops": "all",
        "tables_involved": ["orders", "order_items", "products"],
        "question": "Across the dataset window, which product categories show the fastest growth in order volume from the start to the end of the data window?",
        "sql_query": "-- groupby (quarter, category), then ratio of last-vs-first-quarter counts.",
        "pandas_code": "<see registry source>",
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                f"Top-3 fastest-growing categories (last full quarter vs first full quarter ratio): {fastest_growing}" if fastest_growing else "(growth analysis requires at least 4 quarters of data)",
                "Categories that grow fastest tend to be either (a) newly-onboarded categories starting from a low base, or (b) categories whose seasonality aligns with the last quarter (e.g. holiday gifts visible in Q4).",
            ],
            "supporting_data": {"fastest_growing_categories": fastest_growing},
        },
    })

    # 28: Cross-cutting price-vs-quality patterns
    p_per_order_t = p.groupby("order_id").payment_value.sum()
    rev_with_pay = rv.merge(p_per_order_t.rename("total_pv"), on="order_id")
    rev_high = rev_with_pay[rev_with_pay.total_pv > 500].review_score.mean()
    rev_low = rev_with_pay[rev_with_pay.total_pv < 100].review_score.mean()
    register({
        "qid": "global_28",
        "category": "global",
        "min_hops": "all",
        "tables_involved": ["reviews", "payments"],
        "question": "Is there a relationship between order value and review score across the entire dataset? Do customers paying more leave better or worse reviews on average?",
        "sql_query": "-- groupby payment_value bucket, mean review_score.",
        "pandas_code": "<see registry source>",
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                f"Mean review for high-value orders (>R$500): {r2(rev_high)}",
                f"Mean review for low-value orders (<R$100): {r2(rev_low)}",
                "The two means are close, suggesting that order value is not a strong driver of customer satisfaction. The dominant drivers (delivery performance, product-vs-description match) operate independently of order value.",
            ],
            "supporting_data": {"high_value_mean_review": r2(rev_high),
                                "low_value_mean_review": r2(rev_low)},
        },
    })

    # 29: Data window summary statistics
    first_ts = o.order_purchase_timestamp.min()
    last_ts = o.order_purchase_timestamp.max()
    span_days = (last_ts - first_ts).days
    register({
        "qid": "global_29",
        "category": "global",
        "min_hops": "all",
        "tables_involved": ["orders"],
        "question": "What is the temporal coverage of the Olist dataset, and what gaps or partial periods should an analyst be aware of?",
        "sql_query": "SELECT MIN(order_purchase_timestamp), MAX(order_purchase_timestamp) FROM orders;",
        "pandas_code": "orders.order_purchase_timestamp.min(); orders.order_purchase_timestamp.max()",
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                f"First purchase timestamp: {first_ts}.",
                f"Last purchase timestamp: {last_ts}.",
                f"Total span: {span_days} days (~{span_days/365:.1f} years).",
                "Q4 2016 and Q3-Q4 2018 are sparsely populated (data ingestion ramp-up and snapshot truncation respectively). Robust trend analysis should focus on 2017 to mid-2018.",
            ],
            "supporting_data": {"first_purchase_ts": str(first_ts),
                                "last_purchase_ts": str(last_ts),
                                "span_days": int(span_days)},
        },
    })

    # 30: Synthesis: what types of business questions is Olist best suited to answer?
    register({
        "qid": "global_30",
        "category": "global",
        "min_hops": "all",
        "tables_involved": ["all"],
        "question": "Given the structure and content of the Olist dataset, what types of business questions is it well-suited to answer, and what are its main limitations?",
        "sql_query": "-- meta-analytical synthesis across the data audit.",
        "pandas_code": "-- meta-analytical synthesis.",
        "gold_answer_type": "rubric",
        "gold_answer": None,
        "rubric": {
            "key_facts": [
                "Well-suited to: delivery-performance analyses (rich temporal stamps); regional/geographic studies (state + zip-code precision); category-level revenue and review analyses; payment-method behaviour; seasonality and Black Friday studies.",
                "Limitations: short window (~2 years) limits long-horizon trend claims; very high one-off customer share precludes customer-lifetime-value or churn studies; review free-text is sparse and short, limiting NLP-heavy work; no margin/cost data, so profitability cannot be derived; no inventory/stock data (only realised orders).",
                "These properties make it an excellent testbed for relational/multi-hop benchmarks like the one in this paper, where the join structure across orders/items/payments/reviews is the analytical surface of interest.",
            ],
            "supporting_data": {
                "summary": "Olist is a transactional e-commerce sample with rich relational structure but limited longitudinal depth and limited customer-behaviour traces."
            },
        },
    })


# ============================================================
# Main
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path,
                         default=Path("./pilot_questions.jsonl"))
    args = parser.parse_args()

    print(f"Loading Olist data from {args.data_dir} ...")
    tables = load_olist(args.data_dir)
    print(f"  {sum(len(df) for df in tables.values()):,} rows across "
          f"{len(tables)} tables.")

    print("\nBuilding question registry ...")
    build_factual(tables)
    print(f"  factual:     {sum(1 for q in QUESTIONS if q['category']=='factual'):>2d}")
    build_aggregation(tables)
    print(f"  aggregation: {sum(1 for q in QUESTIONS if q['category']=='aggregation'):>2d}")
    build_comparison(tables)
    print(f"  comparison:  {sum(1 for q in QUESTIONS if q['category']=='comparison'):>2d}")
    build_root_cause(tables)
    print(f"  root_cause:  {sum(1 for q in QUESTIONS if q['category']=='root_cause'):>2d}")
    build_global(tables)
    print(f"  global:      {sum(1 for q in QUESTIONS if q['category']=='global'):>2d}")
    print(f"  TOTAL:       {len(QUESTIONS):>2d}")

    print(f"\nWriting JSONL to {args.out} ...")
    with open(args.out, "w", encoding="utf-8") as f:
        for q in QUESTIONS:
            f.write(json.dumps(q, ensure_ascii=False, default=str) + "\n")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
