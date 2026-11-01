"""
Diagnose the two small count discrepancies between
temporal_index.py and data_stats.py:

  approved_at:  99,280 (index) vs 99,281 (data_stats)  → −1
  reviewed_at:  87,548 (index) vs 88,039 (data_stats)  → −491

Run from the code/ root:
  python phase5_temporal/diag_index_counts.py
"""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase2_kg"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_kg import NT, ET
from temporal_index import _parse_ts

kg = pickle.load(open("outputs/kg_schema_aware.pkl", "rb"))


def is_present(s):
    return s and s not in ("None", "NaT", "nan", "")


# ----------------------------------------------------------------
# approved_at discrepancy
# ----------------------------------------------------------------
print("=" * 60)
print("approved_at: where does the -1 come from?")
print("=" * 60)

ds_count = 0          # data_stats-style count
parser_drops = []     # rows ds counts but our parser would drop

for n, d in kg.nodes(data=True):
    if d.get("node_type") != NT.ORDER:
        continue
    p_raw, a_raw = d.get("purchase_ts"), d.get("approved_at")
    if is_present(p_raw) and is_present(a_raw) and a_raw >= p_raw:
        ds_count += 1
        ps, as_ = _parse_ts(p_raw), _parse_ts(a_raw)
        if ps is None or as_ is None:
            parser_drops.append((n, repr(p_raw), repr(a_raw),
                                 f"parsed=({ps},{as_})"))
        elif as_ < ps:
            parser_drops.append((n, repr(p_raw), repr(a_raw),
                                 "float-compare flipped"))

print(f"data_stats-style count: {ds_count:,}")
print(f"rows our parser would drop or flip: {len(parser_drops)}")
for row in parser_drops[:10]:
    print("  ", row)


# ----------------------------------------------------------------
# reviewed_at discrepancy
# ----------------------------------------------------------------
print()
print("=" * 60)
print("reviewed_at: where does the -491 come from?")
print("=" * 60)
print("Hypothesis: data_stats counts REVIEW ROWS (after a left-join,")
print("so an order with N reviews contributes up to N), while the")
print("index counts ORDERS (one earliest-review interval per order).")
print("The 814 duplicate review_ids the prior paper flagged would")
print("inflate the row-count above the order-count.")
print()

# Count orders with >=1 monotonic review interval (index logic)
orders_with_review = 0
# Count review-row-style (data_stats logic): for each REVIEWED_BY edge,
# if delivered_customer_date present and creation_date >= delivered
review_rows = 0
orders_multi_review = 0

for n, d in kg.nodes(data=True):
    if d.get("node_type") != NT.ORDER:
        continue
    deliv = d.get("delivered_customer_date")
    if not is_present(deliv):
        continue
    deliv_f = _parse_ts(deliv)
    if deliv_f is None:
        continue

    # all reviews linked to this order
    n_valid_for_order = 0
    earliest = None
    for _, nb, ed in kg.out_edges(n, data=True):
        if ed.get("edge_type") != ET.REVIEWED_BY:
            continue
        rattr = kg.nodes[nb]
        if rattr.get("node_type") != NT.REVIEW:
            continue
        cr = _parse_ts(rattr.get("creation_date"))
        if cr is None:
            continue
        # data_stats-style: count this review row if creation >= delivered
        if cr >= deliv_f:
            review_rows += 1
            n_valid_for_order += 1
        if earliest is None or cr < earliest:
            earliest = cr

    # index-style: one interval per order if earliest review monotonic
    if earliest is not None and earliest >= deliv_f:
        orders_with_review += 1
    if n_valid_for_order > 1:
        orders_multi_review += 1

print(f"index-style (orders w/ monotonic earliest review): "
      f"{orders_with_review:,}")
print(f"data_stats-style (monotonic review ROWS):           "
      f"{review_rows:,}")
print(f"orders with >1 valid review row:                    "
      f"{orders_multi_review:,}")
print()
print("If index-style ≈ 87,548 and row-style ≈ 88,039, the gap is")
print("explained by orders with multiple reviews (duplicates).")
