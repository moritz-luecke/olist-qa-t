"""
============================================================
Phase 2.2 — Validate Knowledge Graph
============================================================

Lädt das gepicklte KG und führt eine Reihe von Spot-Checks
und Traversal-Tests durch. Vergleicht die Ergebnisse mit den
ursprünglichen CSV-Daten (Ground Truth).

Checks:
  - Counts (Knoten/Kanten pro Typ) vs. CSV
  - Sample-Traversal "alle Items für einen Order"
  - Sample-Traversal "alle Orders für einen Customer"
  - 3-hop traversal: Customer → Order → OrderItem → Product
  - 4-hop traversal: Customer → Order → OrderItem → Seller → Geolocation

Usage:
  python validate_kg.py
============================================================
"""
from __future__ import annotations

import argparse
import pickle
import sys
from collections import Counter
from pathlib import Path

import networkx as nx
import pandas as pd

# Reuse the NT/ET/PREFIX constants and helpers from build_kg
from build_kg import NT, ET, PREFIX, nid, load_olist


def neighbors_by_edge(G: nx.MultiDiGraph, node: str, edge_type: str, direction: str = "out") -> list[str]:
    """Return neighbors of `node` reachable via edges of the given type."""
    result = []
    if direction in ("out", "both"):
        for _, v, d in G.out_edges(node, data=True):
            if d.get("edge_type") == edge_type:
                result.append(v)
    if direction in ("in", "both"):
        for u, _, d in G.in_edges(node, data=True):
            if d.get("edge_type") == edge_type:
                result.append(u)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--kg-path", type=Path, default=Path("./outputs/kg_schema_aware.pkl"))
    args = parser.parse_args()

    print(f"Loading KG from {args.kg_path} ...", end=" ", flush=True)
    with open(args.kg_path, "rb") as f:
        G: nx.MultiDiGraph = pickle.load(f)
    print(f"loaded ({G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges)")

    print("\nLoading source CSVs for ground truth ...", end=" ", flush=True)
    tables = load_olist(args.data_dir)
    print("done")

    passed = []
    failed = []

    def check(name: str, condition: bool, expected=None, got=None):
        if condition:
            print(f"  [OK]   {name}")
            passed.append(name)
        else:
            print(f"  [FAIL] {name}")
            if expected is not None:
                print(f"         expected: {expected}")
                print(f"         got:      {got}")
            failed.append(name)

    # ----------------------------------------------------------------
    # CHECK 1 — Node counts per type vs. CSV row counts
    # ----------------------------------------------------------------
    print("\n--- CHECK 1: Node counts per type ---")
    node_type_counts = Counter(d["node_type"] for _, d in G.nodes(data=True))

    check(
        "Customer count matches CSV",
        node_type_counts[NT.CUSTOMER] == len(tables["customers"]),
        len(tables["customers"]),
        node_type_counts[NT.CUSTOMER],
    )
    check(
        "Order count matches CSV",
        node_type_counts[NT.ORDER] == len(tables["orders"]),
        len(tables["orders"]),
        node_type_counts[NT.ORDER],
    )
    check(
        "OrderItem count matches CSV",
        node_type_counts[NT.ORDER_ITEM] == len(tables["order_items"]),
        len(tables["order_items"]),
        node_type_counts[NT.ORDER_ITEM],
    )
    check(
        "Product count matches CSV",
        node_type_counts[NT.PRODUCT] == len(tables["products"]),
        len(tables["products"]),
        node_type_counts[NT.PRODUCT],
    )
    check(
        "Seller count matches CSV",
        node_type_counts[NT.SELLER] == len(tables["sellers"]),
        len(tables["sellers"]),
        node_type_counts[NT.SELLER],
    )
    check(
        "Payment count matches CSV",
        node_type_counts[NT.PAYMENT] == len(tables["payments"]),
        len(tables["payments"]),
        node_type_counts[NT.PAYMENT],
    )
    # Reviews: wir deduplizieren — Knotenzahl = #unique review_ids
    expected_reviews = tables["reviews"]["review_id"].nunique()
    check(
        "Review count matches unique CSV review_ids",
        node_type_counts[NT.REVIEW] == expected_reviews,
        expected_reviews,
        node_type_counts[NT.REVIEW],
    )

    # ----------------------------------------------------------------
    # CHECK 2 — Edge counts per type vs. CSV cardinalities
    # ----------------------------------------------------------------
    print("\n--- CHECK 2: Edge counts per type ---")
    edge_type_counts = Counter(d["edge_type"] for _, _, d in G.edges(data=True))

    check(
        "placed edges = #orders",
        edge_type_counts[ET.PLACED] == len(tables["orders"]),
        len(tables["orders"]),
        edge_type_counts[ET.PLACED],
    )
    check(
        "has_item edges = #order_items",
        edge_type_counts[ET.HAS_ITEM] == len(tables["order_items"]),
        len(tables["order_items"]),
        edge_type_counts[ET.HAS_ITEM],
    )
    check(
        "of_product edges = #order_items",
        edge_type_counts[ET.OF_PRODUCT] == len(tables["order_items"]),
        len(tables["order_items"]),
        edge_type_counts[ET.OF_PRODUCT],
    )
    check(
        "sold_by edges = #order_items",
        edge_type_counts[ET.SOLD_BY] == len(tables["order_items"]),
        len(tables["order_items"]),
        edge_type_counts[ET.SOLD_BY],
    )
    check(
        "paid_by edges = #payments",
        edge_type_counts[ET.PAID_BY] == len(tables["payments"]),
        len(tables["payments"]),
        edge_type_counts[ET.PAID_BY],
    )
    # reviewed_by: jede CSV-Review-Zeile = eine Edge (auch wenn review_id dupliziert)
    check(
        "reviewed_by edges = #review rows",
        edge_type_counts[ET.REVIEWED_BY] == len(tables["reviews"]),
        len(tables["reviews"]),
        edge_type_counts[ET.REVIEWED_BY],
    )

    # ----------------------------------------------------------------
    # CHECK 3 — Sample traversal: items of a specific order
    # ----------------------------------------------------------------
    print("\n--- CHECK 3: Sample traversal — items of an order ---")
    # Pick an order with multiple items
    items_per_order = tables["order_items"].groupby("order_id").size()
    sample_order_id = items_per_order[items_per_order >= 3].index[0]
    expected_n_items = int(items_per_order[sample_order_id])

    graph_items = neighbors_by_edge(G, nid(NT.ORDER, sample_order_id), ET.HAS_ITEM)
    check(
        f"Order {sample_order_id[:8]}... has correct #items via has_item edges",
        len(graph_items) == expected_n_items,
        expected_n_items,
        len(graph_items),
    )

    # ----------------------------------------------------------------
    # CHECK 4 — Sample traversal: orders of a specific customer
    # ----------------------------------------------------------------
    print("\n--- CHECK 4: Sample traversal — orders of a customer ---")
    orders_per_cust = tables["orders"].groupby("customer_id").size()
    sample_cust = orders_per_cust.index[0]

    expected_orders = orders_per_cust[sample_cust]
    graph_orders = neighbors_by_edge(G, nid(NT.CUSTOMER, sample_cust), ET.PLACED)
    check(
        f"Customer {sample_cust[:8]}... has correct #orders via placed edges",
        len(graph_orders) == expected_orders,
        expected_orders,
        len(graph_orders),
    )

    # ----------------------------------------------------------------
    # CHECK 5 — 3-hop traversal: Customer → Order → OrderItem → Product
    # ----------------------------------------------------------------
    print("\n--- CHECK 5: 3-hop traversal — products bought by a customer ---")
    # Ground truth via pandas joins
    sample_cust = orders_per_cust[orders_per_cust >= 1].index[5]
    customer_orders = tables["orders"][tables["orders"]["customer_id"] == sample_cust]["order_id"]
    customer_items = tables["order_items"][tables["order_items"]["order_id"].isin(customer_orders)]
    expected_products = set(customer_items["product_id"])

    # KG traversal
    customer_node = nid(NT.CUSTOMER, sample_cust)
    graph_products = set()
    for order_node in neighbors_by_edge(G, customer_node, ET.PLACED):
        for item_node in neighbors_by_edge(G, order_node, ET.HAS_ITEM):
            for prod_node in neighbors_by_edge(G, item_node, ET.OF_PRODUCT):
                graph_products.add(prod_node)

    graph_product_ids = {G.nodes[p]["product_id"] for p in graph_products}
    check(
        f"3-hop Customer→Order→Item→Product matches pandas join",
        graph_product_ids == expected_products,
        f"{len(expected_products)} products",
        f"{len(graph_product_ids)} products",
    )

    # ----------------------------------------------------------------
    # CHECK 6 — 4-hop traversal: Customer → Order → Item → Seller → Geo
    # ----------------------------------------------------------------
    print("\n--- CHECK 6: 4-hop traversal — seller states for a customer ---")
    customer_sellers = tables["order_items"][
        tables["order_items"]["order_id"].isin(customer_orders)
    ]["seller_id"].unique()
    seller_zip_lookup = tables["sellers"].set_index("seller_id")["seller_zip_code_prefix"]
    expected_zips = set(seller_zip_lookup.loc[customer_sellers].values)

    graph_zips = set()
    for order_node in neighbors_by_edge(G, customer_node, ET.PLACED):
        for item_node in neighbors_by_edge(G, order_node, ET.HAS_ITEM):
            for seller_node in neighbors_by_edge(G, item_node, ET.SOLD_BY):
                for geo_node in neighbors_by_edge(G, seller_node, ET.LOCATED_AT):
                    graph_zips.add(G.nodes[geo_node]["zip_code_prefix"])

    check(
        f"4-hop Customer→Order→Item→Seller→Geo matches pandas join",
        graph_zips == expected_zips,
        f"{len(expected_zips)} zip prefixes",
        f"{len(graph_zips)} zip prefixes",
    )

    # ----------------------------------------------------------------
    # CHECK 7 — Node attributes preserved
    # ----------------------------------------------------------------
    print("\n--- CHECK 7: Node attributes ---")
    # Sample order: check the status matches
    sample_order = tables["orders"].iloc[100]
    order_attrs = G.nodes[nid(NT.ORDER, sample_order["order_id"])]
    check(
        "Order status preserved as node attribute",
        order_attrs["status"] == sample_order["order_status"],
        sample_order["order_status"],
        order_attrs["status"],
    )

    # Sample order_item: price preserved
    sample_item = tables["order_items"].iloc[200]
    item_node = nid(NT.ORDER_ITEM, sample_item["order_id"], sample_item["order_item_id"])
    item_attrs = G.nodes[item_node]
    check(
        "OrderItem price preserved",
        abs(item_attrs["price"] - sample_item["price"]) < 1e-6,
        sample_item["price"],
        item_attrs["price"],
    )

    # ----------------------------------------------------------------
    # CHECK 8 — Duplicate review_id handled correctly
    # ----------------------------------------------------------------
    print("\n--- CHECK 8: Duplicate review_ids handled ---")
    dup_review_ids = (
        tables["reviews"]
        .groupby("review_id")
        .size()
        .loc[lambda s: s > 1]
    )
    if len(dup_review_ids) > 0:
        sample_dup = dup_review_ids.index[0]
        # Wie viele Orders verweisen auf diese review_id?
        n_orders_csv = dup_review_ids.iloc[0]
        # In Graph: alle in-edges (von Orders) auf den Review-Knoten
        review_node = nid(NT.REVIEW, sample_dup)
        n_orders_graph = len(neighbors_by_edge(G, review_node, ET.REVIEWED_BY, direction="in"))
        check(
            f"Duplicate review_id {sample_dup[:8]}... has correct fan-in",
            n_orders_graph == n_orders_csv,
            n_orders_csv,
            n_orders_graph,
        )
    else:
        print("  [SKIP] No duplicate review_ids found (unexpected!)")

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"  VALIDATION SUMMARY: {len(passed)} passed, {len(failed)} failed")
    print("=" * 60)
    if failed:
        for name in failed:
            print(f"  FAILED: {name}")
        return 1
    print("  All checks passed ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
