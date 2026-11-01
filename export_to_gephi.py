from pathlib import Path
import sys
import pickle
import random

import networkx as nx

# ================================
# CONFIG
# ================================

BASE = Path(__file__).resolve().parent

KG_PATH = BASE / "outputs/kg_schema_aware.pkl"
OUT_PATH = BASE / "outputs/kg_gephi.gexf"

# ⚠️ very important: Gephi can't handle huge graphs well
EXPORT_FULL_GRAPH = False
N_SAMPLE = 2000   # increase carefully (500–5000 is reasonable)

# ================================
# LOAD KG
# ================================

print("Loading KG...")
kg: nx.MultiDiGraph = pickle.load(open(KG_PATH, "rb"))

print(f"Total nodes: {kg.number_of_nodes()}")
print(f"Total edges: {kg.number_of_edges()}")

# ================================
# OPTIONAL: SAMPLE GRAPH
# ================================

if not EXPORT_FULL_GRAPH:
    print(f"Sampling {N_SAMPLE} nodes for Gephi export...")

    nodes = list(kg.nodes())
    sampled_nodes = set(random.sample(nodes, min(N_SAMPLE, len(nodes))))

    G = kg.subgraph(sampled_nodes).copy()

else:
    print("Exporting full graph (⚠️ may be very large!)")
    G = kg.copy()

print(f"Export graph nodes: {G.number_of_nodes()}")
print(f"Export graph edges: {G.number_of_edges()}")

# ================================
# CLEAN ATTRIBUTES (IMPORTANT!)
# ================================

print("Cleaning attributes for Gephi...")

def clean_value(v):
    """Convert values into Gephi-compatible formats"""
    if v is None:
        return ""
    if isinstance(v, (int, float, str)):
        return v
    return str(v)  # fallback

# Clean node attributes
for n, data in G.nodes(data=True):
    for k in list(data.keys()):
        data[k] = clean_value(data[k])

# Clean edge attributes
for u, v, key, data in G.edges(keys=True, data=True):
    for k in list(data.keys()):
        data[k] = clean_value(data[k])

# ================================
# MULTIGRAPH FIX (IMPORTANT)
# ================================

# Gephi handles MultiDiGraph poorly → convert
print("Converting MultiDiGraph -> DiGraph...")

G_simple = nx.DiGraph()

for u, v, data in G.edges(data=True):
    if G_simple.has_edge(u, v):
        continue
    G_simple.add_edge(u, v, **data)

# copy node data
for n, data in G.nodes(data=True):
    G_simple.add_node(n, **data)

# ================================
# EXPORT TO GEXF
# ================================

print(f"Exporting to {OUT_PATH} ...")

nx.write_gexf(G_simple, OUT_PATH)

print("✅ Export finished!")