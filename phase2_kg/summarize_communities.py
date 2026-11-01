"""
============================================================
Phase 2.5 — LLM Community Summaries
============================================================

Für jede Community (oberhalb einer Mindestgröße) wird eine kurze
Zusammenfassung generiert. Die Summaries werden später für:
  - Community-based Retrieval (Microsoft GraphRAG)
  - Global-Question Beantwortung
verwendet.

Pipeline pro Community:
  1. Knoten der Community sammeln
  2. Statistiken extrahieren (Node-Type-Verteilung, häufige Cities,
     Top-Kategorien, Score-Verteilung)
  3. Ein paar repräsentative Knoten-Beschreibungen samplen
  4. Prompt mit (Stats + Sample-Descriptions) an LLM
  5. Ergebnis als community_summary speichern

WICHTIG: Diese Datei macht echte OpenAI-API-Calls. Sie kann teuer
sein bei großen Community-Counts. Default-Limits unten:
  - min_community_size = 100   (überspringt Single-Node-Communities)
  - max_communities    = None  (alle; setze z.B. 50 für Dev-Runs)
  - max_sample_nodes   = 20    (pro Community)

Output:
  outputs/community_summaries.jsonl     (eine Summary pro Zeile)
  outputs/kg_schema_aware.pkl           (Summaries auch in Graph-Attr)
============================================================
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import networkx as nx


# ------------------------------------------------------------
# Prompt-Template
# ------------------------------------------------------------

SUMMARY_PROMPT = """\
You are analyzing a community of nodes from an e-commerce knowledge graph.
Below is a statistical profile of the community and a sample of node
descriptions. Produce a concise 2–4 sentence summary capturing:

  1. What kind of entities dominate this community.
  2. The most salient geographic, categorical, or behavioral pattern.
  3. Any quality-relevant signal (review scores, payment patterns).

Avoid speculation; ground the summary in the provided facts. Output a
single paragraph, no bullet points, no preamble.

== COMMUNITY STATISTICS ==
{stats_block}

== SAMPLE NODE DESCRIPTIONS ({n_sample} of {n_total}) ==
{sample_block}
"""


# ------------------------------------------------------------
# Stat extraction per community
# ------------------------------------------------------------

def community_stats(G: nx.MultiDiGraph, nodes: list[str]) -> dict:
    """Compute summary statistics for a set of nodes."""
    types = Counter()
    cities = Counter()
    states = Counter()
    categories = Counter()
    payment_types = Counter()
    review_scores = []
    prices = []

    for n in nodes:
        a = G.nodes[n]
        nt = a.get("node_type")
        types[nt] += 1

        if nt in ("Customer", "Seller", "Geolocation"):
            if a.get("city"):
                cities[a["city"]] += 1
            if a.get("state"):
                states[a["state"]] += 1
        if nt == "Product":
            cat = a.get("category_en") or a.get("category")
            if cat:
                categories[cat] += 1
        if nt == "Payment":
            if a.get("payment_type"):
                payment_types[a["payment_type"]] += 1
        if nt == "Review":
            score = a.get("score")
            if score is not None:
                review_scores.append(score)
        if nt == "OrderItem":
            p = a.get("price")
            if p is not None:
                prices.append(p)

    return {
        "size": len(nodes),
        "node_types": dict(types),
        "top_cities": cities.most_common(5),
        "top_states": states.most_common(5),
        "top_categories": categories.most_common(5),
        "payment_types": dict(payment_types),
        "review_score_mean": (sum(review_scores) / len(review_scores)) if review_scores else None,
        "review_score_n": len(review_scores),
        "price_mean": (sum(prices) / len(prices)) if prices else None,
        "price_n": len(prices),
    }


def format_stats(stats: dict) -> str:
    """Format stats dict into a readable block for the prompt."""
    lines = [f"Size: {stats['size']} nodes"]
    lines.append("Node types: " + ", ".join(
        f"{k}={v}" for k, v in sorted(stats["node_types"].items(), key=lambda x: -x[1])
    ))
    if stats["top_states"]:
        lines.append("Top states: " + ", ".join(
            f"{s} ({c})" for s, c in stats["top_states"]
        ))
    if stats["top_cities"]:
        lines.append("Top cities: " + ", ".join(
            f"{c} ({n})" for c, n in stats["top_cities"]
        ))
    if stats["top_categories"]:
        lines.append("Top product categories: " + ", ".join(
            f"{c} ({n})" for c, n in stats["top_categories"]
        ))
    if stats["payment_types"]:
        lines.append("Payment types: " + ", ".join(
            f"{k}={v}" for k, v in stats["payment_types"].items()
        ))
    if stats["review_score_mean"] is not None:
        lines.append(
            f"Reviews: n={stats['review_score_n']}, "
            f"mean_score={stats['review_score_mean']:.2f}"
        )
    if stats["price_mean"] is not None:
        lines.append(
            f"Prices: n={stats['price_n']}, mean=R${stats['price_mean']:.2f}"
        )
    return "\n".join(lines)


# ------------------------------------------------------------
# LLM call
# ------------------------------------------------------------

def call_openai(prompt: str, model: str = "gpt-4o-mini-2024-07-18") -> str:
    """Send a single completion request to OpenAI."""
    from openai import OpenAI

    client = OpenAI()  # reads OPENAI_API_KEY from env
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=200,
        messages=[
            {"role": "system",
             "content": "You write concise, factual summaries."},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content.strip()


# ------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kg-path", type=Path, default=Path("./outputs/kg_schema_aware.pkl"))
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Which resolution's communities to summarize")
    parser.add_argument("--min-community-size", type=int, default=100)
    parser.add_argument("--max-communities", type=int, default=None,
                        help="Cap on #communities to summarize (cost control)")
    parser.add_argument("--max-sample-nodes", type=int, default=20)
    parser.add_argument("--model", type=str, default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--out-jsonl", type=Path, default=Path("./outputs/community_summaries.jsonl"))
    parser.add_argument("--dry-run", action="store_true",
                        help="Print prompts but don't actually call the API")
    args = parser.parse_args()

    if not args.dry_run and not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set. Use --dry-run to test prompts only.",
              file=sys.stderr)
        return 1

    print(f"Loading KG from {args.kg_path}")
    with open(args.kg_path, "rb") as f:
        G: nx.MultiDiGraph = pickle.load(f)

    # Find the community attribute matching --gamma
    attr_name = f"community_id_g{args.gamma:.1f}".replace(".", "_")
    # Group nodes by community
    by_comm: dict[int, list[str]] = defaultdict(list)
    for node, data in G.nodes(data=True):
        cid = data.get(attr_name)
        if cid is not None:
            by_comm[cid].append(node)

    if not by_comm:
        print(f"ERROR: No community attribute '{attr_name}' found on nodes. "
              f"Did you run run_leiden.py with --gammas {args.gamma}?",
              file=sys.stderr)
        return 1

    # Filter by size, sort descending
    eligible = [
        (cid, nodes) for cid, nodes in by_comm.items()
        if len(nodes) >= args.min_community_size
    ]
    eligible.sort(key=lambda x: -len(x[1]))
    if args.max_communities:
        eligible = eligible[:args.max_communities]

    print(f"  {len(by_comm):,} communities total, "
          f"{len(eligible):,} with size >= {args.min_community_size}")
    if args.max_communities:
        print(f"  Capped at {args.max_communities} for this run")

    # Iterate
    summaries = []
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w") as fp:
        for i, (cid, nodes) in enumerate(eligible):
            stats = community_stats(G, nodes)

            # Sample node descriptions — stratified by node type, so each
            # community summary sees a mix of customers, orders, items, etc.,
            # not just whichever type happens to come first.
            by_type: dict[str, list[str]] = defaultdict(list)
            for n in nodes:
                by_type[G.nodes[n].get("node_type", "?")].append(n)
            n_types = len(by_type)
            per_type_quota = max(1, args.max_sample_nodes // n_types)
            sample: list[str] = []
            for nt, members in by_type.items():
                sample.extend(members[:per_type_quota])
            sample = sample[:args.max_sample_nodes]
            sample_descs = [
                f"- {G.nodes[n].get('description', '(no description)')}"
                for n in sample
            ]

            prompt = SUMMARY_PROMPT.format(
                stats_block=format_stats(stats),
                n_sample=len(sample),
                n_total=len(nodes),
                sample_block="\n".join(sample_descs),
            )

            if args.dry_run:
                if i == 0:
                    print("\n--- SAMPLE PROMPT (community 0) ---")
                    print(prompt)
                summary_text = "[DRY RUN — would call LLM here]"
            else:
                try:
                    summary_text = call_openai(prompt, model=args.model)
                except Exception as e:
                    print(f"  [WARN] community {cid}: LLM call failed ({e}), "
                          f"skipping", file=sys.stderr)
                    continue

            record = {
                "community_id": cid,
                "gamma": args.gamma,
                "size": len(nodes),
                "stats": stats,
                "summary": summary_text,
            }
            fp.write(json.dumps(record) + "\n")
            summaries.append(record)

            # Add to graph attribute for downstream use
            for n in nodes:
                G.nodes[n][f"community_summary_g{args.gamma:.1f}".replace(".", "_")] = summary_text

            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(eligible)} done")

            # Light rate limiting
            if not args.dry_run:
                time.sleep(0.1)

    if not args.dry_run:
        print(f"\nSaving updated KG to {args.kg_path}")
        with open(args.kg_path, "wb") as f:
            pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nWrote {len(summaries)} summaries to {args.out_jsonl}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
