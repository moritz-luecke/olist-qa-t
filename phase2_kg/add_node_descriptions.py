"""
============================================================
Phase 2.3 — Add Natural-Language Node Descriptions
============================================================

Erzeugt für jeden Knoten eine kurze Template-basierte Beschreibung
(1-3 Sätze), die seine Attribute UND nächste Nachbarn zusammenfasst.
Die Beschreibungen werden später für:
  - Hybrid Retrieval (Embedding der Beschreibungen)
  - Community Summaries (LLM bekommt diese als Input)
  - Subgraph-Serialisierung für Graph-RAG Generation

verwendet.

WICHTIG: Diese Beschreibungen sind DETERMINISTISCH (Template-basiert),
nicht LLM-generiert. Das ist eine bewusste Wahl — wir wollen die
Schema-Aware-Variante nicht mit LLM-generiertem Inhalt
"kontaminieren" (siehe Method-Section).

Output:
  outputs/kg_schema_aware.pkl  — KG mit `description` als zusätzlichem
                                 Attribut auf jedem Knoten.
============================================================
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import networkx as nx

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_):
        return iterable

from build_kg import NT, ET


def _fmt_money(v) -> str:
    """Format BRL value."""
    if v is None:
        return "N/A"
    return f"R${v:.2f}"


def _safe_str(v, default="unknown"):
    return str(v) if v is not None else default


def describe_customer(G: nx.MultiDiGraph, node: str) -> str:
    """e.g., 'Customer in São Paulo, SP (zip prefix 14406). Placed 2 orders.'"""
    a = G.nodes[node]
    n_orders = sum(
        1 for _, _, d in G.out_edges(node, data=True)
        if d.get("edge_type") == ET.PLACED
    )
    return (
        f"Customer located in {_safe_str(a.get('city'))}, "
        f"{_safe_str(a.get('state'))} "
        f"(zip prefix {a.get('zip_code_prefix')}). "
        f"Placed {n_orders} order(s)."
    )


def describe_seller(G: nx.MultiDiGraph, node: str) -> str:
    """e.g., 'Seller in Rio de Janeiro, RJ. Sold 47 line items.'"""
    a = G.nodes[node]
    n_items = sum(
        1 for _, _, d in G.in_edges(node, data=True)
        if d.get("edge_type") == ET.SOLD_BY
    )
    return (
        f"Seller located in {_safe_str(a.get('city'))}, "
        f"{_safe_str(a.get('state'))} "
        f"(zip prefix {a.get('zip_code_prefix')}). "
        f"Sold {n_items} line item(s)."
    )


def describe_product(G: nx.MultiDiGraph, node: str) -> str:
    """e.g., 'Product in the bed_bath_table category (English: bed_bath_table).
    Sold in 32 order items.'"""
    a = G.nodes[node]
    n_sold = sum(
        1 for _, _, d in G.in_edges(node, data=True)
        if d.get("edge_type") == ET.OF_PRODUCT
    )
    category = a.get("category_en") or a.get("category") or "unknown"
    weight = a.get("weight_g")
    weight_str = f", weighs {weight:.0f}g" if weight else ""
    return (
        f"Product in the '{category}' category{weight_str}. "
        f"Appears in {n_sold} order item(s)."
    )


def describe_order(G: nx.MultiDiGraph, node: str) -> str:
    """e.g., 'Order placed on 2017-05-12, status delivered. Contains 2 items
    totaling R$179.80. Paid by credit card.'"""
    a = G.nodes[node]
    # Count items, sum prices
    items = [
        v for _, v, d in G.out_edges(node, data=True)
        if d.get("edge_type") == ET.HAS_ITEM
    ]
    total_price = sum(G.nodes[i].get("price", 0) or 0 for i in items)

    # Get payment type (first one if multiple)
    payment_types = []
    for _, v, d in G.out_edges(node, data=True):
        if d.get("edge_type") == ET.PAID_BY:
            payment_types.append(G.nodes[v].get("payment_type", "unknown"))
    pay_str = "/".join(sorted(set(payment_types))) if payment_types else "unknown"

    # Review score (max if multiple)
    reviews = [
        v for _, v, d in G.out_edges(node, data=True)
        if d.get("edge_type") == ET.REVIEWED_BY
    ]
    review_str = ""
    if reviews:
        scores = [G.nodes[r].get("score") for r in reviews]
        scores = [s for s in scores if s is not None]
        if scores:
            review_str = f" Received review score {max(scores)}."

    purchase_date = (a.get("purchase_ts") or "unknown")[:10]
    return (
        f"Order placed on {purchase_date}, status '{a.get('status')}'. "
        f"Contains {len(items)} item(s) totaling {_fmt_money(total_price)}. "
        f"Paid via {pay_str}.{review_str}"
    )


def describe_order_item(G: nx.MultiDiGraph, node: str) -> str:
    """e.g., 'Line item priced R$89.90 plus R$8.50 freight. Product:
    bed_bath_table. Sold by seller in São Paulo, SP.'"""
    a = G.nodes[node]
    # Look up linked product and seller
    product_cat = "unknown"
    seller_loc = "unknown location"
    for _, v, d in G.out_edges(node, data=True):
        if d.get("edge_type") == ET.OF_PRODUCT:
            pa = G.nodes[v]
            product_cat = pa.get("category_en") or pa.get("category") or "unknown"
        elif d.get("edge_type") == ET.SOLD_BY:
            sa = G.nodes[v]
            seller_loc = f"{_safe_str(sa.get('city'))}, {_safe_str(sa.get('state'))}"

    return (
        f"Line item priced {_fmt_money(a.get('price'))} plus "
        f"{_fmt_money(a.get('freight_value'))} freight. "
        f"Product category: '{product_cat}'. "
        f"Sold by seller in {seller_loc}."
    )


def describe_payment(G: nx.MultiDiGraph, node: str) -> str:
    """e.g., 'Payment of R$179.80 via credit_card in 3 installments.'"""
    a = G.nodes[node]
    return (
        f"Payment of {_fmt_money(a.get('payment_value'))} via "
        f"{a.get('payment_type')} in "
        f"{a.get('payment_installments')} installment(s)."
    )


def describe_review(G: nx.MultiDiGraph, node: str) -> str:
    """e.g., 'Review with score 5 of 5. Title: "Great product!".
    Linked to 1 order.'"""
    a = G.nodes[node]
    n_orders = sum(
        1 for _, _, d in G.in_edges(node, data=True)
        if d.get("edge_type") == ET.REVIEWED_BY
    )
    title = a.get("comment_title")
    title_str = f' Title: "{title[:60]}".' if title else ""
    return (
        f"Review with score {a.get('score')} of 5.{title_str} "
        f"Linked to {n_orders} order(s)."
    )


def describe_geolocation(G: nx.MultiDiGraph, node: str) -> str:
    """e.g., 'Geolocation: zip prefix 14406 (Franca, SP). Referenced by
    52 customers and 3 sellers.'"""
    a = G.nodes[node]
    n_cust = 0
    n_sell = 0
    for u, _, d in G.in_edges(node, data=True):
        if d.get("edge_type") == ET.LOCATED_AT:
            ut = G.nodes[u].get("node_type")
            if ut == NT.CUSTOMER:
                n_cust += 1
            elif ut == NT.SELLER:
                n_sell += 1
    return (
        f"Geolocation: zip prefix {a.get('zip_code_prefix')} "
        f"({_safe_str(a.get('city'))}, {_safe_str(a.get('state'))}). "
        f"Referenced by {n_cust} customer(s) and {n_sell} seller(s)."
    )


DESCRIBERS = {
    NT.CUSTOMER:    describe_customer,
    NT.SELLER:      describe_seller,
    NT.PRODUCT:     describe_product,
    NT.ORDER:       describe_order,
    NT.ORDER_ITEM:  describe_order_item,
    NT.PAYMENT:     describe_payment,
    NT.REVIEW:      describe_review,
    NT.GEOLOCATION: describe_geolocation,
}


def add_descriptions(G: nx.MultiDiGraph) -> None:
    """Add a `description` attribute to every node, in-place."""
    for node in tqdm(G.nodes(), desc="Generating descriptions"):
        nt = G.nodes[node].get("node_type")
        describer = DESCRIBERS.get(nt)
        if describer is None:
            G.nodes[node]["description"] = f"{nt} node {node}"
        else:
            G.nodes[node]["description"] = describer(G, node)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kg-path", type=Path, default=Path("./outputs/kg_schema_aware.pkl"))
    parser.add_argument("--out-path", type=Path, default=None,
                        help="Where to save the enriched KG (defaults to overwriting --kg-path)")
    args = parser.parse_args()

    out_path = args.out_path or args.kg_path

    print(f"Loading KG from {args.kg_path}")
    with open(args.kg_path, "rb") as f:
        G: nx.MultiDiGraph = pickle.load(f)
    print(f"  {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    print("\nAdding descriptions to nodes ...")
    add_descriptions(G)

    print(f"\nSaving enriched KG to {out_path}")
    with open(out_path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Print a few sample descriptions to sanity-check
    print("\n--- Sample descriptions (one per node type) ---")
    seen_types = set()
    for node, d in G.nodes(data=True):
        nt = d.get("node_type")
        if nt not in seen_types:
            seen_types.add(nt)
            print(f"\n[{nt}] {node}")
            print(f"  → {d['description']}")
        if len(seen_types) == len(DESCRIBERS):
            break

    return 0


if __name__ == "__main__":
    sys.exit(main())
