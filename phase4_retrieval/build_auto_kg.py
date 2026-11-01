"""
============================================================
Phase 6.2 — Build auto-extracted KG from triples
============================================================

Reads outputs/triples_extracted.jsonl and constructs a NetworkX
MultiDiGraph with the SAME node-id and edge-type vocabulary as the
schema-aware KG (build_kg.py), so that the same retrievers run on
both without modification.

Node-ID convention (mirrors build_kg.py):
    Order      -> O:{hex32}
    Customer   -> C:{hex32}
    Seller     -> S:{hex32}
    Product    -> P:{hex32}
    OrderItem  -> I:{order_hex}:{n}
    Payment    -> PAY:{order_hex}:{seq}
    Review     -> R:{hex32}
    Geolocation-> G:{zipprefix}    (the auto-extractor cannot infer
                                    these, so we use a per-city slug)

For entities the extractor returns with a non-canonical id_hint
(e.g. "Order abc...", "the seller", "Belo Horizonte"), we apply
two normalisations:
  - If the hint contains a 32-hex token, use that as the canonical ID.
  - Else build a slug from the type + hint (e.g. "G:slug_belo_horizonte").

The resulting graph is deliberately smaller and noisier than the
schema-aware one — see eval_kg_quality.py for the comparison.

Output
------
  outputs/kg_auto.pkl                 (NetworkX MultiDiGraph)
  outputs/kg_auto_stats.txt           (counts, etc.)
============================================================
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from build_kg import NT, ET           # reuse the type vocab
except ImportError:
    # Fallback: define them locally to match
    class NT:
        ORDER = "Order"; CUSTOMER = "Customer"; SELLER = "Seller"
        PRODUCT = "Product"; ORDER_ITEM = "OrderItem"
        PAYMENT = "Payment"; REVIEW = "Review"; GEOLOCATION = "Geolocation"
    class ET:
        PLACED = "placed"; HAS_ITEM = "has_item"
        OF_PRODUCT = "of_product"; SOLD_BY = "sold_by"
        PAID_BY = "paid_by"; REVIEWED_BY = "reviewed_by"
        LOCATED_AT = "located_at"


HEX32_RE = re.compile(r"\b[a-f0-9]{32}\b")
PREFIX = {
    NT.ORDER: "O", NT.CUSTOMER: "C", NT.SELLER: "S", NT.PRODUCT: "P",
    NT.PAYMENT: "PAY", NT.REVIEW: "R",
    NT.ORDER_ITEM: "I", NT.GEOLOCATION: "G",
}


# ----------------------------------------------------------------
# ID normalization
# ----------------------------------------------------------------
def _slugify(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s.lower()).strip("_")
    return s or "unknown"


def canonical_node_id(entity: dict,
                      source_chunk_id: str | None = None) -> str | None:
    """Return canonical 'PREFIX:tail' or None if we can't anchor.

    Strategy:
      1. If id_hint contains a 32-hex token → use it under the right prefix.
      2. For OrderItem/Payment which need an order anchor: try to extract
         from id_hint, fall back to source_chunk_id (the chunk is one order).
      3. For Geolocation: slug the hint (no 32-hex token expected).
      4. For other types without a 32-hex token: drop the entity (we cannot
         resolve it to a unique node).
    """
    etype = entity["type"]
    hint = entity["id_hint"]
    pref = PREFIX.get(etype)
    if pref is None:
        return None

    # Geolocation: slug-based
    if etype == NT.GEOLOCATION:
        return f"{pref}:{_slugify(hint)}"

    # Look for a 32-hex token
    hex_tokens = HEX32_RE.findall(hint.lower())
    main_hex = hex_tokens[0] if hex_tokens else None

    # OrderItem / Payment need an order anchor + sequence index
    if etype in (NT.ORDER_ITEM, NT.PAYMENT):
        # Prefer an order hex from the hint; otherwise from source chunk
        # (which has form "O:<hex>").
        order_hex = main_hex
        if order_hex is None and source_chunk_id:
            m = HEX32_RE.search(source_chunk_id.lower())
            order_hex = m.group(0) if m else None
        if order_hex is None:
            return None
        # Sequence index: try to parse from hint (e.g. "Item 2", "PAY:..:3")
        seq_match = re.search(r"(?:item|pay|seq|#)\s*[:_-]?\s*(\d+)",
                              hint.lower())
        seq = seq_match.group(1) if seq_match else "1"
        return f"{pref}:{order_hex}:{seq}"

    # Standard 32-hex entities
    if main_hex is None:
        return None
    return f"{pref}:{main_hex}"


# ----------------------------------------------------------------
# Description string (mirrors add_node_descriptions.py format)
# ----------------------------------------------------------------
def make_description(node_id: str, etype: str, attrs: dict) -> str:
    """Short NL description so the same EntityRetriever embeddings code works.
    We keep these terse and factual."""
    def _attr_or(key, default="unknown"):
        v = attrs.get(key)
        return str(v) if v not in (None, "") else default

    if etype == NT.ORDER:
        return (f"Order (auto-extracted). Attributes: "
                f"{', '.join(f'{k}={v}' for k, v in attrs.items()) or 'none'}.")
    if etype == NT.CUSTOMER:
        return (f"Customer located in "
                f"{_attr_or('city')}, {_attr_or('state')}.")
    if etype == NT.SELLER:
        return (f"Seller located in "
                f"{_attr_or('city')}, {_attr_or('state')}.")
    if etype == NT.PRODUCT:
        return f"Product, category '{_attr_or('category', 'unknown')}'."
    if etype == NT.ORDER_ITEM:
        return (f"Line item, price {_attr_or('price', 'unknown')} "
                f"+ freight {_attr_or('freight_value', 'unknown')}.")
    if etype == NT.PAYMENT:
        return (f"Payment of {_attr_or('payment_value', '?')} "
                f"via {_attr_or('payment_type')} "
                f"in {_attr_or('payment_installments', '?')} installment(s).")
    if etype == NT.REVIEW:
        return f"Review with score {_attr_or('score', '?')} of 5."
    if etype == NT.GEOLOCATION:
        return (f"Geolocation: {_attr_or('city')}, {_attr_or('state')} "
                f"(auto-extracted, no zip prefix).")
    return f"{etype} node (auto-extracted)."


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--triples-path", type=Path,
                    default=Path(__file__).resolve().parent.parent
                            / "outputs" / "triples_extracted.jsonl")
    ap.add_argument("--out-path", type=Path,
                    default=Path(__file__).resolve().parent.parent
                            / "outputs" / "kg_auto.pkl")
    ap.add_argument("--stats-path", type=Path,
                    default=Path(__file__).resolve().parent.parent
                            / "outputs" / "kg_auto_stats.txt")
    args = ap.parse_args()

    if not args.triples_path.exists():
        print(f"ERROR: {args.triples_path} missing. Run extract_triples.py.",
              file=sys.stderr)
        return 1

    print(f"Loading triples from {args.triples_path} ...")
    records = [json.loads(l) for l in args.triples_path.open()]
    print(f"  {len(records):,} extracted triples")

    G = nx.MultiDiGraph()

    n_resolved = 0
    n_unresolved = 0
    edge_types_seen = Counter()

    for rec in records:
        s_ent = rec["subject"]
        o_ent = rec["object"]
        pred = rec["predicate"]
        src_chunk = rec.get("source_chunk_id")

        s_id = canonical_node_id(s_ent, src_chunk)
        o_id = canonical_node_id(o_ent, src_chunk)
        if s_id is None or o_id is None:
            n_unresolved += 1
            continue
        n_resolved += 1

        # Add or update nodes (merge attrs; first wins for descriptions)
        for nid, ent in ((s_id, s_ent), (o_id, o_ent)):
            if nid in G.nodes:
                G.nodes[nid].setdefault("attrs", {}).update(ent.get("attrs", {}))
            else:
                G.add_node(nid,
                           node_type=ent["type"],
                           attrs=dict(ent.get("attrs", {})),
                           id_hint=ent["id_hint"],
                           description=make_description(nid, ent["type"],
                                                        ent.get("attrs", {})))

        # Add edge; MultiDiGraph allows duplicates but we dedupe by key
        edge_key = f"{pred}:{src_chunk}"
        if not G.has_edge(s_id, o_id, key=edge_key):
            G.add_edge(s_id, o_id, key=edge_key,
                       edge_type=pred, source_chunk=src_chunk)
            edge_types_seen[pred] += 1

    # Refresh descriptions now that attrs may have grown
    for nid, d in G.nodes(data=True):
        d["description"] = make_description(nid, d["node_type"],
                                            d.get("attrs", {}))

    print(f"\n  Resolved triples : {n_resolved:,}")
    print(f"  Dropped (unresolved IDs): {n_unresolved:,}")
    print(f"  #Nodes: {G.number_of_nodes():,}")
    print(f"  #Edges: {G.number_of_edges():,}")

    # Persist
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    with args.out_path.open("wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nSaved auto-KG to {args.out_path}")

    # Stats file
    node_type_counts = Counter(d["node_type"] for _, d in G.nodes(data=True))
    edge_type_counts = Counter(d["edge_type"] for _, _, d in G.edges(data=True))
    stats = [
        "=" * 60,
        "  AUTO-EXTRACTED KG — STATISTICS",
        "=" * 60,
        f"  Source triples       : {len(records):,}",
        f"  Resolved triples     : {n_resolved:,}",
        f"  Unresolved (dropped) : {n_unresolved:,}",
        f"  #Nodes : {G.number_of_nodes():,}",
        f"  #Edges : {G.number_of_edges():,}",
        "",
        "--- Node types ---",
        *[f"  {t:13s}: {n:>8,}" for t, n in node_type_counts.most_common()],
        "",
        "--- Edge types ---",
        *[f"  {t:13s}: {n:>8,}" for t, n in edge_type_counts.most_common()],
    ]
    args.stats_path.write_text("\n".join(stats))
    print(f"Stats to {args.stats_path}")
    print()
    print("\n".join(stats))
    return 0


if __name__ == "__main__":
    sys.exit(main())
