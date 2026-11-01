from pathlib import Path
import sys
import pickle
import random

import networkx as nx
import matplotlib.pyplot as plt

# ================================
# CONFIG
# ================================

BASE = Path(__file__).resolve().parent

KG_PATH = BASE / "outputs/kg_schema_aware.pkl"
TEMP_PATH = BASE / "outputs/temporal_index.pkl"

N_SAMPLE = 15  # etwas weniger, sonst wird es unleserlich

# ================================
# FIX: Import-Pfade
# ================================

sys.path.insert(0, str(BASE / "phase5_temporal"))
sys.path.insert(0, str(BASE / "phase2_kg"))

# ================================
# LOAD KG
# ================================

print("Loading KG...")
kg = pickle.load(open(KG_PATH, "rb"))

# ================================
# LOAD NT ENUM
# ================================

try:
    from build_kg import NT
except ImportError:
    try:
        from phase2_kg.build_kg import NT
    except ImportError:
        NT = None

# ================================
# FIND ORDER NODES
# ================================

print("Detecting order nodes...")

orders = []

for n, d in kg.nodes(data=True):
    nt = d.get("node_type")

    if NT is not None and nt == NT.ORDER:
        orders.append(n)
    elif isinstance(nt, str) and "order" in nt.lower():
        orders.append(n)

print(f"Total nodes: {kg.number_of_nodes()}")
print(f"Order nodes found: {len(orders)}")

if not orders:
    raise ValueError("Keine Orders gefunden.")

# ================================
# SAMPLE SUBGRAPH
# ================================

print("Sampling subgraph...")

sample_orders = random.sample(orders, min(N_SAMPLE, len(orders)))

nodes = set()

for o in sample_orders:
    nodes.add(o)
    nodes.update(list(kg.successors(o)))
    nodes.update(list(kg.predecessors(o)))

subgraph = kg.subgraph(nodes)

print(f"Subgraph nodes: {subgraph.number_of_nodes()}")
print(f"Subgraph edges: {subgraph.number_of_edges()}")

# ================================
# BUILD LABELS
# ================================

# 🔹 Node Labels (Typ + ID gekürzt)
node_labels = {}
for n, d in subgraph.nodes(data=True):
    t = d.get("node_type")

    if NT is not None:
        if t == NT.ORDER:
            prefix = "Order"
        elif t == NT.CUSTOMER:
            prefix = "Cust"
        elif hasattr(NT, "PRODUCT") and t == NT.PRODUCT:
            prefix = "Prod"
        elif t == NT.REVIEW:
            prefix = "Rev"
        else:
            prefix = "?"
    else:
        prefix = str(t)

    node_labels[n] = f"{prefix}\n{str(n)[:6]}"


# 🔹 Edge Labels (Relationstyp)
edge_labels = {}
for u, v, d in subgraph.edges(data=True):
    et = d.get("edge_type", "")
    edge_labels[(u, v)] = str(et)


# ================================
# NODE COLORS
# ================================

colors = []

for n, d in subgraph.nodes(data=True):
    t = d.get("node_type")

    if NT is not None:
        if t == NT.ORDER:
            colors.append("red")
        elif t == NT.CUSTOMER:
            colors.append("blue")
        elif hasattr(NT, "PRODUCT") and t == NT.PRODUCT:
            colors.append("green")
        elif t == NT.REVIEW:
            colors.append("orange")
        else:
            colors.append("gray")
    else:
        colors.append("gray")


# ================================
# PLOT
# ================================

print("Plotting KG with labels...")

plt.figure(figsize=(14, 10))

pos = nx.spring_layout(subgraph, seed=42)

# Knoten
nx.draw(
    subgraph,
    pos,
    with_labels=False,
    node_size=500,
    node_color=colors,
    edge_color="lightgray"
)

# 🔹 Node Labels
nx.draw_networkx_labels(
    subgraph,
    pos,
    labels=node_labels,
    font_size=7
)

# 🔹 Edge Labels
nx.draw_networkx_edge_labels(
    subgraph,
    pos,
    edge_labels=edge_labels,
    font_size=6
)

plt.title("KG Subgraph with Node & Edge Labels")
plt.show()

print("Done ✅")