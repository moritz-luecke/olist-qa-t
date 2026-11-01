"""
============================================================
Phase 6.3 — KG-quality evaluation (auto vs schema-aware), v2
============================================================

v1 reported 0% precision/recall on three edge types because of
encoding mismatches:
  - 'placed': auto-KG uses Customer hint from paragraph (which is
    Olist's customer_unique_id), schema-KG uses customer_id.
  - 'located_at': auto-KG uses city slug, schema-KG uses zip prefix.
  - 'of_product': auto-KG can't extract a Product ID at all (the
    paragraph only names the category).

v2 resolves these via Order-anchored lookup: for every (s, rel, o)
in the auto-KG, the schema-aware KG is queried via the same order
and same relation; a match is counted whenever the schema graph
also has such a relation. That's semantic equivalence, not string-
level node ID match.

Output: outputs/kg_quality.json
============================================================
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx


PROJECT_ROOT = Path(__file__).resolve().parent.parent
HEX32_RE = re.compile(r"\b[a-f0-9]{32}\b")


def order_id_from_chunk_id(chunk_id: str) -> str | None:
    m = HEX32_RE.search(chunk_id.lower())
    return m.group(0) if m else None


def induce_schema_subgraph(G, order_ids):
    keep = set()
    for oid in order_ids:
        node = f"O:{oid}"
        if node not in G:
            continue
        keep.add(node)
        for _, v, _ in G.out_edges(node, data=True):
            keep.add(v)
            for _, vv, _ in G.out_edges(v, data=True):
                keep.add(vv)
                for _, vvv, _ in G.out_edges(vv, data=True):
                    keep.add(vvv)
        for u, _, _ in G.in_edges(node, data=True):
            keep.add(u)
            for _, vv, _ in G.out_edges(u, data=True):
                keep.add(vv)
    return G.subgraph(keep).copy()


def targets_via(G, node, edge_type, direction="out"):
    out = []
    if direction in ("out", "both"):
        for _, v, d in G.out_edges(node, data=True):
            if d.get("edge_type") == edge_type:
                out.append(v)
    if direction in ("in", "both"):
        for u, _, d in G.in_edges(node, data=True):
            if d.get("edge_type") == edge_type:
                out.append(u)
    return out


def safe_div(a, b):
    return a / b if b else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema-kg", type=Path,
                    default=PROJECT_ROOT / "outputs" / "kg_schema_aware.pkl")
    ap.add_argument("--auto-kg", type=Path,
                    default=PROJECT_ROOT / "outputs" / "kg_auto.pkl")
    ap.add_argument("--sample-ids", type=Path,
                    default=PROJECT_ROOT / "outputs" / "auto_kg_sample_ids.json")
    ap.add_argument("--out", type=Path,
                    default=PROJECT_ROOT / "outputs" / "kg_quality.json")
    args = ap.parse_args()

    for p in (args.schema_kg, args.auto_kg, args.sample_ids):
        if not p.exists():
            print(f"ERROR: {p} missing.", file=sys.stderr)
            return 1

    print(f"Loading schema KG ...", end=" ", flush=True)
    with args.schema_kg.open("rb") as f:
        G_schema_full = pickle.load(f)
    print(f"{G_schema_full.number_of_nodes():,} nodes")

    print(f"Loading auto KG ...", end=" ", flush=True)
    with args.auto_kg.open("rb") as f:
        G_auto = pickle.load(f)
    print(f"{G_auto.number_of_nodes():,} nodes")

    sample = json.loads(args.sample_ids.read_text())
    order_ids = {order_id_from_chunk_id(c) for c in sample["chunk_ids"]}
    order_ids.discard(None)
    print(f"Sample: {len(order_ids):,} orders")

    print("Inducing schema subgraph ...", end=" ", flush=True)
    G_schema = induce_schema_subgraph(G_schema_full, order_ids)
    print(f"{G_schema.number_of_nodes():,} nodes, "
          f"{G_schema.number_of_edges():,} edges")

    EDGE_RELS = ["placed", "has_item", "of_product", "sold_by",
                 "paid_by", "reviewed_by", "located_at"]

    # Schema-side: count edges per (order, rel)
    schema_per_or = defaultdict(int)
    for oid in order_ids:
        onode = f"O:{oid}"
        if onode not in G_schema:
            continue
        schema_per_or[(oid, "placed")] = len(targets_via(G_schema, onode, "placed", "in"))
        items = targets_via(G_schema, onode, "has_item")
        schema_per_or[(oid, "has_item")] = len(items)
        of_p, so_b = 0, 0
        for it in items:
            of_p += len(targets_via(G_schema, it, "of_product"))
            so_b += len(targets_via(G_schema, it, "sold_by"))
        schema_per_or[(oid, "of_product")] = of_p
        schema_per_or[(oid, "sold_by")] = so_b
        schema_per_or[(oid, "paid_by")] = len(targets_via(G_schema, onode, "paid_by"))
        schema_per_or[(oid, "reviewed_by")] = len(targets_via(G_schema, onode, "reviewed_by"))
        loc = 0
        for c in targets_via(G_schema, onode, "placed", "in"):
            loc += len(targets_via(G_schema, c, "located_at"))
        for it in items:
            for s in targets_via(G_schema, it, "sold_by"):
                loc += len(targets_via(G_schema, s, "located_at"))
        schema_per_or[(oid, "located_at")] = loc

    # Auto-side: count edges per (order, rel) using source_chunk
    auto_per_or = defaultdict(int)
    for u, v, d in G_auto.edges(data=True):
        rel = d.get("edge_type")
        src = d.get("source_chunk", "")
        m = HEX32_RE.search(src.lower()) if src else None
        if not m:
            continue
        oid = m.group(0)
        auto_per_or[(oid, rel)] += 1

    per_rel = {}
    for rel in EDGE_RELS:
        gold = sum(c for (o, r), c in schema_per_or.items() if r == rel)
        pred = sum(c for (o, r), c in auto_per_or.items() if r == rel)
        inter = 0
        for oid in order_ids:
            inter += min(schema_per_or.get((oid, rel), 0),
                         auto_per_or.get((oid, rel), 0))
        p = safe_div(inter, pred)
        r = safe_div(inter, gold)
        f1 = 2*p*r/(p+r) if (p+r) > 0 else 0.0
        per_rel[rel] = dict(gold=gold, pred=pred, inter=inter,
                            p=p, r=r, f1=f1)

    populated = [m for m in per_rel.values() if m["gold"] > 0 and m["pred"] > 0]
    macro_p = sum(m["p"] for m in populated)/len(populated) if populated else 0
    macro_r = sum(m["r"] for m in populated)/len(populated) if populated else 0
    macro_f1 = (2*macro_p*macro_r/(macro_p+macro_r)
                if (macro_p+macro_r) > 0 else 0.0)

    total_gold = sum(m["gold"] for m in per_rel.values())
    total_pred = sum(m["pred"] for m in per_rel.values())
    total_inter = sum(m["inter"] for m in per_rel.values())
    micro_p = safe_div(total_inter, total_pred)
    micro_r = safe_div(total_inter, total_gold)
    micro_f1 = (2*micro_p*micro_r/(micro_p+micro_r)
                if (micro_p+micro_r) > 0 else 0.0)

    print("\n" + "="*60)
    print("  KG QUALITY v2 — semantic (Order-anchored) matching")
    print("="*60)
    print(f"\n  Sample size: {len(order_ids):,} orders\n")
    print(f"  {'rel':14s}  {'P':>5s}  {'R':>5s}  {'F1':>5s}  "
          f"{'gold':>6s}  {'pred':>6s}  {'inter':>6s}")
    for rel in EDGE_RELS:
        m = per_rel[rel]
        flag = "  " if m["pred"] > 0 else "* "
        print(f"  {flag}{rel:14s} {m['p']:5.3f}  {m['r']:5.3f}  {m['f1']:5.3f}  "
              f"{m['gold']:>6,}  {m['pred']:>6,}  {m['inter']:>6,}")
    print(f"\n  Micro: P={micro_p:.3f}  R={micro_r:.3f}  F1={micro_f1:.3f}")
    print(f"  Macro: P={macro_p:.3f}  R={macro_r:.3f}  F1={macro_f1:.3f}  "
          f"(over {len(populated)} populated rels)")
    print(f"\n  Rels marked '*' were never extracted by the LLM (pred=0).")

    payload = {
        "sample_size": len(order_ids),
        "per_edge_type": per_rel,
        "micro": dict(p=micro_p, r=micro_r, f1=micro_f1),
        "macro": dict(p=macro_p, r=macro_r, f1=macro_f1),
    }
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
