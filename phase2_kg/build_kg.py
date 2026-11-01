"""
============================================================
Phase 2.1 — Build Schema-Aware Knowledge Graph
============================================================

Liest die Olist-CSVs und baut einen typisierten Property-Graph in
NetworkX. Jeder Knoten hat:
  - node_type (Customer, Order, OrderItem, ...)
  - alle Attribute aus der Quelltabelle als Node-Properties

Jede Kante hat:
  - edge_type (placed, has_item, of_product, sold_by,
               paid_by, reviewed_by, located_at)

Output:
  outputs/kg_schema_aware.pkl       — NetworkX MultiDiGraph (pickled)
  outputs/kg_schema_aware_stats.txt — Statistik-Report

Usage:
  python build_kg.py [--data-dir DATA_DIR] [--out-dir OUT_DIR]
============================================================
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Node-Typ-Konstanten (zentral, damit konsistent referenziert)
# ------------------------------------------------------------

class NT:
    CUSTOMER = "Customer"
    ORDER = "Order"
    ORDER_ITEM = "OrderItem"
    PAYMENT = "Payment"
    REVIEW = "Review"
    PRODUCT = "Product"
    SELLER = "Seller"
    GEOLOCATION = "Geolocation"


class ET:
    PLACED = "placed"           # Customer -> Order
    HAS_ITEM = "has_item"       # Order -> OrderItem
    OF_PRODUCT = "of_product"   # OrderItem -> Product
    SOLD_BY = "sold_by"         # OrderItem -> Seller
    PAID_BY = "paid_by"         # Order -> Payment
    REVIEWED_BY = "reviewed_by" # Order -> Review
    LOCATED_AT = "located_at"   # {Customer, Seller} -> Geolocation


# ------------------------------------------------------------
# Node-ID-Konventionen
# ------------------------------------------------------------
# Wir präfixieren IDs mit Typ-Kürzel, damit IDs global eindeutig sind
# (z.B. Order-ID und Customer-ID sind beide 32-Hex-Strings, könnten
# theoretisch kollidieren). Format: "{type_prefix}:{original_id}".

PREFIX = {
    NT.CUSTOMER:    "C",
    NT.ORDER:       "O",
    NT.ORDER_ITEM:  "I",
    NT.PAYMENT:     "PAY",
    NT.REVIEW:      "R",
    NT.PRODUCT:     "P",
    NT.SELLER:      "S",
    NT.GEOLOCATION: "G",
}


def nid(node_type: str, *parts) -> str:
    """Build a typed node ID."""
    return f"{PREFIX[node_type]}:" + ":".join(str(p) for p in parts)


# ------------------------------------------------------------
# Helper für Logging
# ------------------------------------------------------------

class Timer:
    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        self.t0 = time.time()
        print(f"  [{self.label}] ...", end=" ", flush=True)
        return self

    def __exit__(self, *_):
        dt = time.time() - self.t0
        print(f"done in {dt:.1f}s")


def _safe(value):
    """Convert NaN to None for JSON/pickle hygiene."""
    if isinstance(value, float) and np.isnan(value):
        return None
    if pd.isna(value):
        return None
    return value


def row_to_attrs(row: pd.Series, drop: tuple = ()) -> dict:
    """Convert a DataFrame row to a node-attribute dict, NaN-safe."""
    return {k: _safe(v) for k, v in row.items() if k not in drop}


# ------------------------------------------------------------
# Loader
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


def load_geolocation(data_dir: Path) -> pd.DataFrame:
    """The Kaggle download ships one geolocation file; older local
    copies split it into two halves (*_1.csv, *_2.csv). Accept both."""
    single = data_dir / "olist_geolocation_dataset.csv"
    halves = [data_dir / "olist_geolocation_dataset_1.csv",
              data_dir / "olist_geolocation_dataset_2.csv"]
    if single.exists():
        return pd.read_csv(single)
    if all(p.exists() for p in halves):
        return pd.concat([pd.read_csv(p) for p in halves],
                         ignore_index=True)
    raise FileNotFoundError(
        f"Missing geolocation table: expected {single} "
        "(or the two halves *_1.csv and *_2.csv)")


def load_olist(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all Olist tables; geolocation as one file or two halves."""
    files = {
        "customers":            "olist_customers_dataset.csv",
        "orders":               "olist_orders_dataset.csv",
        "order_items":          "olist_order_items_dataset.csv",
        "payments":             "olist_order_payments_dataset.csv",
        "reviews":              "olist_order_reviews_dataset.csv",
        "products":             "olist_products_dataset.csv",
        "sellers":              "olist_sellers_dataset.csv",
        "category_translation": "product_category_name_translation.csv",
    }
    tables = {}
    for key, fname in files.items():
        path = data_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        tables[key] = pd.read_csv(path, parse_dates=DATE_COLS.get(key, []))

    tables["geolocation"] = load_geolocation(data_dir)
    return tables


def preprocess(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Light preprocessing — add English category names to products."""
    t = tables
    # Produkte um englische Kategorienamen anreichern
    t["products"] = t["products"].merge(
        t["category_translation"],
        on="product_category_name",
        how="left",
    )
    # Geolocation auf ZIP-Prefix aggregieren: Mittelwert lat/lng,
    # häufigste Stadt/Bundesstaat pro Prefix.
    geo = t["geolocation"]
    geo_agg = (
        geo.groupby("geolocation_zip_code_prefix")
        .agg(
            lat=("geolocation_lat", "mean"),
            lng=("geolocation_lng", "mean"),
            city=("geolocation_city", lambda s: s.mode().iat[0] if len(s) else None),
            state=("geolocation_state", lambda s: s.mode().iat[0] if len(s) else None),
            n_records=("geolocation_lat", "size"),
        )
        .reset_index()
    )
    t["geolocation_agg"] = geo_agg
    return t


# ------------------------------------------------------------
# KG construction — node by node, edge by edge
# ------------------------------------------------------------

def build_graph(tables: dict[str, pd.DataFrame]) -> nx.MultiDiGraph:
    """Build the schema-aware knowledge graph.

    Returns a directed multigraph (different edge types between the
    same pair of nodes are possible in principle, though our schema
    doesn't actually use this).
    """
    G = nx.MultiDiGraph()

    # ---- 1. Geolocation nodes (we add them first so customer/seller
    #         located_at edges can refer to existing nodes) ----------
    with Timer("geolocation nodes"):
        geo = tables["geolocation_agg"]
        for row in geo.itertuples(index=False):
            G.add_node(
                nid(NT.GEOLOCATION, row.geolocation_zip_code_prefix),
                node_type=NT.GEOLOCATION,
                zip_code_prefix=int(row.geolocation_zip_code_prefix),
                lat=float(row.lat) if not pd.isna(row.lat) else None,
                lng=float(row.lng) if not pd.isna(row.lng) else None,
                city=_safe(row.city),
                state=_safe(row.state),
                n_records=int(row.n_records),
            )

    # ---- 2. Customers + located_at edges -----------------------------
    with Timer("customer nodes + located_at"):
        for row in tables["customers"].itertuples(index=False):
            cust_id = nid(NT.CUSTOMER, row.customer_id)
            G.add_node(
                cust_id,
                node_type=NT.CUSTOMER,
                customer_id=row.customer_id,
                customer_unique_id=row.customer_unique_id,
                zip_code_prefix=int(row.customer_zip_code_prefix),
                city=row.customer_city,
                state=row.customer_state,
            )
            geo_id = nid(NT.GEOLOCATION, row.customer_zip_code_prefix)
            if G.has_node(geo_id):
                G.add_edge(cust_id, geo_id, edge_type=ET.LOCATED_AT)

    # ---- 3. Sellers + located_at edges -------------------------------
    with Timer("seller nodes + located_at"):
        for row in tables["sellers"].itertuples(index=False):
            seller_id = nid(NT.SELLER, row.seller_id)
            G.add_node(
                seller_id,
                node_type=NT.SELLER,
                seller_id=row.seller_id,
                zip_code_prefix=int(row.seller_zip_code_prefix),
                city=row.seller_city,
                state=row.seller_state,
            )
            geo_id = nid(NT.GEOLOCATION, row.seller_zip_code_prefix)
            if G.has_node(geo_id):
                G.add_edge(seller_id, geo_id, edge_type=ET.LOCATED_AT)

    # ---- 4. Products -------------------------------------------------
    with Timer("product nodes"):
        for row in tables["products"].itertuples(index=False):
            G.add_node(
                nid(NT.PRODUCT, row.product_id),
                node_type=NT.PRODUCT,
                product_id=row.product_id,
                category=_safe(row.product_category_name),
                category_en=_safe(row.product_category_name_english),
                name_length=_safe(row.product_name_lenght),
                description_length=_safe(row.product_description_lenght),
                photos_qty=_safe(row.product_photos_qty),
                weight_g=_safe(row.product_weight_g),
                length_cm=_safe(row.product_length_cm),
                height_cm=_safe(row.product_height_cm),
                width_cm=_safe(row.product_width_cm),
            )

    # ---- 5. Orders + placed edges ------------------------------------
    with Timer("order nodes + placed"):
        for row in tables["orders"].itertuples(index=False):
            order_id = nid(NT.ORDER, row.order_id)
            G.add_node(
                order_id,
                node_type=NT.ORDER,
                order_id=row.order_id,
                status=row.order_status,
                purchase_ts=str(row.order_purchase_timestamp) if not pd.isna(row.order_purchase_timestamp) else None,
                approved_at=str(row.order_approved_at) if not pd.isna(row.order_approved_at) else None,
                delivered_carrier_date=str(row.order_delivered_carrier_date) if not pd.isna(row.order_delivered_carrier_date) else None,
                delivered_customer_date=str(row.order_delivered_customer_date) if not pd.isna(row.order_delivered_customer_date) else None,
                estimated_delivery_date=str(row.order_estimated_delivery_date) if not pd.isna(row.order_estimated_delivery_date) else None,
            )
            G.add_edge(nid(NT.CUSTOMER, row.customer_id), order_id, edge_type=ET.PLACED)

    # ---- 6. OrderItems + has_item / of_product / sold_by edges -------
    with Timer("order_item nodes + 3 edge types"):
        for row in tables["order_items"].itertuples(index=False):
            item_id = nid(NT.ORDER_ITEM, row.order_id, row.order_item_id)
            G.add_node(
                item_id,
                node_type=NT.ORDER_ITEM,
                order_id=row.order_id,
                order_item_id=int(row.order_item_id),
                shipping_limit_date=str(row.shipping_limit_date) if not pd.isna(row.shipping_limit_date) else None,
                price=float(row.price),
                freight_value=float(row.freight_value),
            )
            order_id = nid(NT.ORDER, row.order_id)
            G.add_edge(order_id, item_id, edge_type=ET.HAS_ITEM)
            G.add_edge(item_id, nid(NT.PRODUCT, row.product_id), edge_type=ET.OF_PRODUCT)
            G.add_edge(item_id, nid(NT.SELLER, row.seller_id), edge_type=ET.SOLD_BY)

    # ---- 7. Payments + paid_by edges ---------------------------------
    with Timer("payment nodes + paid_by"):
        for row in tables["payments"].itertuples(index=False):
            pay_id = nid(NT.PAYMENT, row.order_id, row.payment_sequential)
            G.add_node(
                pay_id,
                node_type=NT.PAYMENT,
                order_id=row.order_id,
                payment_sequential=int(row.payment_sequential),
                payment_type=row.payment_type,
                payment_installments=int(row.payment_installments),
                payment_value=float(row.payment_value),
            )
            G.add_edge(nid(NT.ORDER, row.order_id), pay_id, edge_type=ET.PAID_BY)

    # ---- 8. Reviews + reviewed_by edges ------------------------------
    # Hinweis: review_id ist NICHT unique (814 Duplikate, Olist-Artefakt).
    # Wir deduplizieren auf review_id und verlinken jeden order_id auf
    # den entsprechenden Review-Knoten (1:n Review→Order Modellierung).
    with Timer("review nodes + reviewed_by"):
        rev = tables["reviews"]
        unique_reviews = rev.drop_duplicates(subset="review_id", keep="first")
        for row in unique_reviews.itertuples(index=False):
            rev_id = nid(NT.REVIEW, row.review_id)
            G.add_node(
                rev_id,
                node_type=NT.REVIEW,
                review_id=row.review_id,
                score=int(row.review_score),
                comment_title=_safe(row.review_comment_title),
                comment_message=_safe(row.review_comment_message),
                creation_date=str(row.review_creation_date) if not pd.isna(row.review_creation_date) else None,
                answer_timestamp=str(row.review_answer_timestamp) if not pd.isna(row.review_answer_timestamp) else None,
            )
        # Alle reviewed_by-Edges (also auch duplizierte review_ids → mehrere Orders)
        for row in rev.itertuples(index=False):
            G.add_edge(
                nid(NT.ORDER, row.order_id),
                nid(NT.REVIEW, row.review_id),
                edge_type=ET.REVIEWED_BY,
            )

    return G


# ------------------------------------------------------------
# Stats reporter
# ------------------------------------------------------------

def report_stats(G: nx.MultiDiGraph, out_path: Path) -> dict:
    """Compute statistics and save a human-readable summary."""
    from collections import Counter

    node_type_counts = Counter(d["node_type"] for _, d in G.nodes(data=True))
    edge_type_counts = Counter(d["edge_type"] for _, _, d in G.edges(data=True))

    # Treat as undirected for connected-component analysis
    UG = G.to_undirected(as_view=False)
    components = list(nx.connected_components(UG))
    components.sort(key=len, reverse=True)
    lcc_size = len(components[0]) if components else 0

    degrees = np.array([d for _, d in UG.degree()], dtype=int)

    stats = {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "node_types": dict(node_type_counts),
        "edge_types": dict(edge_type_counts),
        "n_components": len(components),
        "lcc_size": lcc_size,
        "lcc_coverage_pct": lcc_size / G.number_of_nodes() * 100 if G.number_of_nodes() else 0,
        "mean_degree": float(degrees.mean()) if len(degrees) else 0,
        "median_degree": float(np.median(degrees)) if len(degrees) else 0,
        "max_degree": int(degrees.max()) if len(degrees) else 0,
        "std_degree": float(degrees.std()) if len(degrees) else 0,
        "isolated_nodes": int((degrees == 0).sum()),
    }

    # Write a text report
    lines = []
    bar = "=" * 60
    lines.append(bar)
    lines.append("  SCHEMA-AWARE KG — STATISTICS")
    lines.append(bar)
    lines.append(f"\n#Nodes: {stats['n_nodes']:,}")
    lines.append(f"#Edges: {stats['n_edges']:,}")

    lines.append("\n--- Node types ---")
    for nt, c in sorted(node_type_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {nt:14s}: {c:>10,}")

    lines.append("\n--- Edge types ---")
    for et, c in sorted(edge_type_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {et:14s}: {c:>10,}")

    lines.append("\n--- Degree distribution (undirected view) ---")
    lines.append(f"  Mean degree    : {stats['mean_degree']:.2f}")
    lines.append(f"  Median degree  : {stats['median_degree']:.2f}")
    lines.append(f"  Std degree     : {stats['std_degree']:.2f}")
    lines.append(f"  Max degree     : {stats['max_degree']:,}")
    lines.append(f"  Isolated nodes : {stats['isolated_nodes']:,}")

    lines.append("\n--- Connectivity ---")
    lines.append(f"  #Connected components : {stats['n_components']:,}")
    lines.append(f"  Largest CC size       : {stats['lcc_size']:,}")
    lines.append(f"  LCC coverage          : {stats['lcc_coverage_pct']:.2f}%")

    report = "\n".join(lines)
    out_path.write_text(report)
    print("\n" + report)
    return stats


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--out-dir", type=Path, default=Path("./outputs"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {args.data_dir}")
    tables = load_olist(args.data_dir)
    print(f"  {len(tables) - 1} tables loaded (geolocation merged)")

    print("\nPreprocessing")
    tables = preprocess(tables)

    print("\nBuilding KG")
    G = build_graph(tables)

    print(f"\nSaving graph to {args.out_dir / 'kg_schema_aware.pkl'}")
    with open(args.out_dir / "kg_schema_aware.pkl", "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nComputing statistics")
    report_stats(G, args.out_dir / "kg_schema_aware_stats.txt")

    return 0


if __name__ == "__main__":
    sys.exit(main())
