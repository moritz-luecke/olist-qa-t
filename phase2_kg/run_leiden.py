"""
============================================================
Phase 2.4 — Leiden Community Detection
============================================================

Findet Communities im KG mit dem Leiden-Algorithmus (Microsoft GraphRAG
Stil). Funktioniert auf einer ungerichteten Version des Graphen.

Wir versuchen mehrere Resolution-Parameter (γ ∈ {0.5, 1.0, 1.5}) und
speichern für jeden eine Partition als community_id-Attribut.

Fallback-Strategie:
  1. Versuche `graspologic.partition.leiden` (Microsoft, paper-konform)
  2. Wenn nicht verfügbar, nutze `igraph.community_leiden`
  3. Wenn auch das fehlt, nutze `community.community_louvain` (Louvain)
     als Notlösung mit Hinweis.

Output:
  outputs/kg_schema_aware.pkl    — Knoten haben community_id_gamma_{γ}
  outputs/leiden_stats.txt       — Anzahl + Größenverteilung pro γ
============================================================
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import networkx as nx


# ------------------------------------------------------------
# Backend-Auswahl
# ------------------------------------------------------------

def detect_backend() -> str:
    """Pick the best available clustering backend."""
    try:
        import igraph  # noqa
        return "igraph"
    except ImportError:
        pass
    try:
        import community  # noqa  (python-louvain)
        return "louvain"
    except ImportError:
        pass
    try:
        from graspologic.partition import leiden  # noqa
        return "graspologic"
    except ImportError:
        pass
    # NetworkX-built-in als letztes Auffangnetz
    import networkx
    if hasattr(networkx.algorithms.community, "louvain_communities"):
        return "networkx_louvain"
    raise RuntimeError(
        "No clustering backend found. Install one of:\n"
        "  pip install python-igraph\n"
        "  pip install graspologic\n"
        "  pip install python-louvain"
    )


def cluster_igraph(G_und: nx.Graph, resolution: float, seed: int = 42) -> dict[str, int]:
    """Use igraph's Leiden implementation."""
    import igraph as ig

    # NetworkX → igraph
    nodes = list(G_und.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}
    edges = [(node_idx[u], node_idx[v]) for u, v in G_und.edges()]

    g = ig.Graph(n=len(nodes), edges=edges, directed=False)
    partition = g.community_leiden(
        objective_function="modularity",
        resolution=resolution,
        n_iterations=10,
    )
    return {nodes[i]: cluster for i, cluster in enumerate(partition.membership)}


def cluster_louvain(G_und: nx.Graph, resolution: float, seed: int = 42) -> dict[str, int]:
    """Use python-louvain (NOT Leiden, but a reasonable fallback)."""
    import community as community_louvain
    return community_louvain.best_partition(
        G_und, resolution=resolution, random_state=seed
    )


def cluster_graspologic(G_und: nx.Graph, resolution: float, seed: int = 42) -> dict[str, int]:
    """Use graspologic's hierarchical Leiden (Microsoft GraphRAG style)."""
    from graspologic.partition import leiden
    return leiden(
        G_und,
        resolution=resolution,
        random_seed=seed,
        is_weighted=False,
    )


def cluster_networkx_louvain(G_und: nx.Graph, resolution: float, seed: int = 42) -> dict[str, int]:
    """Use NetworkX' built-in Louvain (sufficient for sanity-checks)."""
    communities = nx.algorithms.community.louvain_communities(
        G_und, resolution=resolution, seed=seed
    )
    partition = {}
    for cid, members in enumerate(communities):
        for node in members:
            partition[node] = cid
    return partition


CLUSTERERS = {
    "igraph":           cluster_igraph,
    "louvain":          cluster_louvain,
    "graspologic":      cluster_graspologic,
    "networkx_louvain": cluster_networkx_louvain,
}


# ------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kg-path", type=Path, default=Path("./outputs/kg_schema_aware.pkl"))
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.5, 1.0, 1.5])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stats-path", type=Path, default=Path("./outputs/leiden_stats.txt"))
    args = parser.parse_args()

    backend = detect_backend()
    print(f"Backend: {backend}")
    if backend == "louvain":
        print("  WARNING: Louvain used instead of Leiden — install python-igraph "
              "or graspologic for paper-conformant Leiden.")

    print(f"\nLoading KG from {args.kg_path} ...", end=" ", flush=True)
    with open(args.kg_path, "rb") as f:
        G: nx.MultiDiGraph = pickle.load(f)
    print(f"loaded ({G.number_of_nodes():,} nodes)")

    # Need undirected simple graph for community detection
    print("Converting to undirected simple graph ...", end=" ", flush=True)
    G_und = nx.Graph(G)  # Drops parallel edges (MultiDiGraph → Graph)
    print(f"done ({G_und.number_of_edges():,} unique edges)")

    cluster = CLUSTERERS[backend]
    report_lines = [
        "=" * 60,
        f"  LEIDEN COMMUNITY DETECTION ({backend})",
        "=" * 60,
        f"  Graph: {G_und.number_of_nodes():,} nodes, "
        f"{G_und.number_of_edges():,} edges",
        f"  Seed:  {args.seed}",
        "",
    ]

    for gamma in args.gammas:
        print(f"\n--- γ = {gamma} ---")
        t0 = time.time()
        partition = cluster(G_und, resolution=gamma, seed=args.seed)
        dt = time.time() - t0

        sizes = Counter(partition.values())
        size_distribution = Counter(sizes.values())  # how many clusters of each size

        n_communities = len(sizes)
        size_sorted = sorted(sizes.values(), reverse=True)

        # Modularity (only computable with igraph backend cheaply)
        modularity_str = ""
        if backend == "igraph":
            import igraph as ig
            nodes_list = list(G_und.nodes())
            node_idx = {n: i for i, n in enumerate(nodes_list)}
            edges = [(node_idx[u], node_idx[v]) for u, v in G_und.edges()]
            g = ig.Graph(n=len(nodes_list), edges=edges, directed=False)
            membership = [partition[n] for n in nodes_list]
            try:
                modularity = g.modularity(membership)
                modularity_str = f"  Modularity     : {modularity:.4f}\n"
            except Exception:
                pass

        section = (
            f"--- γ = {gamma} ---\n"
            f"  Wall time      : {dt:.1f}s\n"
            f"  #Communities   : {n_communities:,}\n"
            f"  Largest size   : {size_sorted[0]:,}\n"
            f"  Top-10 sizes   : {size_sorted[:10]}\n"
            f"  Median size    : {size_sorted[len(size_sorted) // 2]}\n"
            f"  #Communities with size = 1: {size_distribution.get(1, 0):,}\n"
            f"  #Communities with size >= 100: "
            f"{sum(1 for s in sizes.values() if s >= 100):,}\n"
            f"{modularity_str}"
        )
        print(section)
        report_lines.append(section)

        # Annotate the original MultiDiGraph
        attr_name = f"community_id_g{gamma:.1f}".replace(".", "_")
        for node, cid in partition.items():
            G.nodes[node][attr_name] = cid

    # Persist enriched KG
    print(f"\nSaving updated KG to {args.kg_path}")
    with open(args.kg_path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

    args.stats_path.write_text("\n".join(report_lines))
    print(f"Stats written to {args.stats_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
