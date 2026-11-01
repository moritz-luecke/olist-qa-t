"""
============================================================
Phase 4 — Order-paragraph chunking for Vanilla RAG (System A)
============================================================

The fairest baseline against Graph-RAG: serialize the same
relational data as natural-language paragraphs, one per order,
with all its line items / payments / review / customer / seller
information inlined. Then index those paragraphs in FAISS.

The "order paragraph" is also the natural unit for the
auto-extracted KG (System C) — extracting triples from these
paragraphs yields a graph that's directly comparable to the
schema-aware one.

Two chunk variants
------------------
v1 (`serialize_order`):
    Original chunks from the Olist-QA paper. The header sentence
    mentions only the purchase date. Time-blind questions are fine,
    but for the temporal-extension paper the baseline is
    information-disadvantaged: it cannot answer questions about
    delivery times, approval lag, or review timing, because those
    timestamps simply are not in the chunk.

v2 (`serialize_order_v2`):
    Adds one extra "timeline" sentence to v1 with all seven order-
    lifecycle and review timestamps. This is the *baseline-fairness
    protocol* of Section 3.2 of the temporal paper: the vanilla-RAG
    baseline is given the same temporal information that the
    interval-typed edges expose to Graph-RAG, so any temporal-
    Graph-RAG win is attributable to structure, not access.

Output:
  outputs/vanilla_chunks.jsonl     — v1 (default; original paper)
  outputs/vanilla_chunks_v2.jsonl  — v2 (temporal-aware baseline)

  Each line:
    {"chunk_id": "O:abc123",
     "order_id": "abc123",
     "text": "<paragraph>",
     "n_tokens_approx": 87,
     "chunk_variant": "v1" | "v2"}

  The `chunk_variant` field is new (absent in pre-temporal files);
  loaders should default to "v1" when absent.
============================================================
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Reuse the load_olist helper from Phase 2. In the new workspace
# layout, build_kg.py lives at code/phase2_kg/build_kg.py. We add
# both the project root and the phase2_kg/ directory to sys.path so
# the bare `from build_kg import ...` works regardless of CWD or
# how the user organises the source tree.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))                  # project root
sys.path.insert(0, str(_HERE.parent / "phase2_kg"))    # phase2_kg/
try:
    from build_kg import load_olist
except ImportError:
    load_olist = None  # tolerable if user runs --help


# ----------------------------------------------------------------
# One-paragraph-per-order serialization (v1, original)
# ----------------------------------------------------------------
def serialize_order(
    order_id: str,
    order_row: pd.Series,
    customer_row: pd.Series,
    items: pd.DataFrame,
    payments: pd.DataFrame,
    reviews: pd.DataFrame,
    products: pd.DataFrame,
    sellers: pd.DataFrame,
) -> str:
    """
    Returns a 3-6 sentence English paragraph describing the order.

    Example output:
      "Order abc123 was placed on 2017-05-12 by customer c001 in São
      Paulo, SP (status: delivered). It contains 2 line items totaling
      R$179.80. Item 1: 'bed_bath_table' at R$89.90 + R$8.50 freight,
      sold by seller s007 in Rio de Janeiro, RJ. Item 2: ... . Paid via
      credit_card in 3 installments (R$179.80). Reviewed with score 5
      of 5."
    """
    parts: list[str] = []

    # Header
    purchase_date = (
        str(order_row["order_purchase_timestamp"])[:10]
        if pd.notna(order_row.get("order_purchase_timestamp")) else "unknown date"
    )
    parts.append(
        f"Order {order_id} was placed on {purchase_date} by customer "
        f"{customer_row['customer_unique_id']} in "
        f"{customer_row['customer_city']}, {customer_row['customer_state']} "
        f"(status: {order_row['order_status']})."
    )

    # Items + sellers + products
    if len(items) > 0:
        total = items["price"].sum()
        parts.append(
            f"It contains {len(items)} line item(s) totaling R${total:.2f}."
        )
        for _, it in items.iterrows():
            prod = products.loc[products["product_id"] == it["product_id"]]
            cat = (prod["product_category_name_english"].iloc[0]
                   if len(prod) and "product_category_name_english" in prod
                   else (prod["product_category_name"].iloc[0]
                         if len(prod) else "unknown"))
            sell = sellers.loc[sellers["seller_id"] == it["seller_id"]]
            sloc = (f"{sell['seller_city'].iloc[0]}, "
                    f"{sell['seller_state'].iloc[0]}") if len(sell) else "unknown"
            parts.append(
                f"Item {int(it['order_item_id'])}: '{cat}' at "
                f"R${it['price']:.2f} + R${it['freight_value']:.2f} freight, "
                f"sold by seller {it['seller_id']} in {sloc}."
            )

    # Payments
    if len(payments) > 0:
        pay_summary = "; ".join(
            f"{p['payment_type']} in {int(p['payment_installments'])} "
            f"installment(s) (R${p['payment_value']:.2f})"
            for _, p in payments.iterrows()
        )
        parts.append(f"Paid via {pay_summary}.")

    # Reviews
    if len(reviews) > 0:
        rv = reviews.iloc[0]
        msg = rv.get("review_comment_message")
        msg_str = f' Comment: "{str(msg)[:120]}".' if pd.notna(msg) else ""
        parts.append(
            f"Reviewed with score {int(rv['review_score'])} of 5.{msg_str}"
        )

    return " ".join(parts)


# ----------------------------------------------------------------
# v2 helpers — timestamp formatting & timeline sentence
# ----------------------------------------------------------------

def _fmt_date(value) -> str | None:
    """Format a pandas Timestamp / string / NaT as YYYY-MM-DD, or None.

    The Olist CSVs are loaded with parse_dates= in build_kg.load_olist,
    so values are typically pd.Timestamp. We tolerate raw strings too
    (in case a caller passes data loaded by other means).
    """
    if value is None or pd.isna(value):
        return None
    # pd.Timestamp / datetime
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    # Fallback: take first 10 chars of an ISO-like string
    s = str(value)
    return s[:10] if len(s) >= 10 else s


def _build_timeline_sentence(
    order_id: str,
    order_row: pd.Series,
    items: pd.DataFrame,
    reviews: pd.DataFrame,
) -> str:
    """Build the additional v2 sentence enumerating all 7 lifecycle dates.

    Coverage of the constituent timestamps (measured on the Olist
    release with data_stats.py):

        order_purchase_timestamp       100.0%
        order_approved_at               99.8%
        order_delivered_carrier_date    98.2%
        order_delivered_customer_date   97.0%
        order_estimated_delivery_date  100.0%
        shipping_limit_date            100.0%   (per order-item)
        review_creation_date           100.0%   (per review row)

    Where a particular timestamp is missing for an order, we say
    "not recorded" rather than fabricating or omitting silently
    (the absence is itself an evidential signal the retriever may
    legitimately need).

    Example output:
      "Order abc123 has timeline: purchased on 2017-05-12, approved
      on 2017-05-12, shipped to carrier on 2017-05-15, delivered to
      customer on 2017-05-22 (estimated 2017-05-28); seller cut-off
      was 2017-05-13. Review submitted on 2017-05-25 (responded on
      2017-05-26)."
    """
    # ---- Order-level timestamps -------------------------------
    purchase   = _fmt_date(order_row.get("order_purchase_timestamp"))
    approved   = _fmt_date(order_row.get("order_approved_at"))
    carrier    = _fmt_date(order_row.get("order_delivered_carrier_date"))
    delivered  = _fmt_date(order_row.get("order_delivered_customer_date"))
    estimated  = _fmt_date(order_row.get("order_estimated_delivery_date"))

    lifecycle_bits: list[str] = []
    lifecycle_bits.append(
        f"purchased on {purchase}" if purchase
        else "purchase date not recorded"
    )
    lifecycle_bits.append(
        f"approved on {approved}" if approved else "approval not recorded"
    )
    lifecycle_bits.append(
        f"shipped to carrier on {carrier}" if carrier
        else "carrier handover not recorded"
    )
    if delivered and estimated:
        lifecycle_bits.append(
            f"delivered to customer on {delivered} (estimated {estimated})"
        )
    elif delivered:
        lifecycle_bits.append(f"delivered to customer on {delivered}")
    elif estimated:
        lifecycle_bits.append(
            f"delivery not recorded (estimated {estimated})"
        )
    else:
        lifecycle_bits.append("delivery not recorded")

    lifecycle = ", ".join(lifecycle_bits)

    # ---- Order-item-level: seller shipping cut-off ------------
    # One order can have multiple items, each with its own
    # shipping_limit_date. Practitioner-relevant value is the EARLIEST
    # cut-off (the binding deadline for the order to ship on time).
    cutoff_str = ""
    if len(items) > 0 and "shipping_limit_date" in items.columns:
        cutoff_series = items["shipping_limit_date"].dropna()
        if len(cutoff_series) > 0:
            cutoff = _fmt_date(cutoff_series.min())
            if cutoff:
                cutoff_str = f"; seller cut-off was {cutoff}"

    # ---- Review-level timestamps ------------------------------
    # If multiple reviews exist (rare; the prior paper flagged 814
    # duplicate review_ids), we use the first row's timestamps and
    # leave the merge of multi-review semantics to higher layers.
    review_str = ""
    if len(reviews) > 0:
        rv = reviews.iloc[0]
        rv_created = _fmt_date(rv.get("review_creation_date"))
        rv_answer  = _fmt_date(rv.get("review_answer_timestamp"))
        if rv_created and rv_answer:
            review_str = (
                f" Review submitted on {rv_created} "
                f"(responded on {rv_answer})."
            )
        elif rv_created:
            review_str = f" Review submitted on {rv_created}."
        elif rv_answer:
            review_str = f" Review response on {rv_answer}."

    return f"Order {order_id} has timeline: {lifecycle}{cutoff_str}.{review_str}".rstrip()


# ----------------------------------------------------------------
# One-paragraph-per-order serialization (v2, temporal-aware baseline)
# ----------------------------------------------------------------
def serialize_order_v2(
    order_id: str,
    order_row: pd.Series,
    customer_row: pd.Series,
    items: pd.DataFrame,
    payments: pd.DataFrame,
    reviews: pd.DataFrame,
    products: pd.DataFrame,
    sellers: pd.DataFrame,
) -> str:
    """v1 paragraph + one extra timeline sentence (Section 3.2).

    Result is identical to ``serialize_order`` plus a final sentence
    listing all seven order-lifecycle and review timestamps.
    """
    v1_text = serialize_order(
        order_id, order_row, customer_row,
        items, payments, reviews, products, sellers,
    )
    timeline = _build_timeline_sentence(order_id, order_row, items, reviews)
    return f"{v1_text} {timeline}"


# ----------------------------------------------------------------
# Build all chunks
# ----------------------------------------------------------------

# Dispatch table — keeps build_chunks() variant-agnostic while making
# the registration of future variants trivial (e.g. a v3 that adds
# external event anchors).
_SERIALIZERS = {
    "v1": serialize_order,
    "v2": serialize_order_v2,
}


def build_chunks(
    tables: dict,
    out_path: Path,
    limit: int | None = None,
    variant: str = "v1",
) -> int:
    """Build all order-paragraph chunks and write them as JSONL.

    Parameters
    ----------
    tables
        Output of ``build_kg.load_olist(data_dir)``.
    out_path
        Destination JSONL file. Parent dir is created if needed.
    limit
        Cap on number of orders processed (useful for dev runs).
    variant
        "v1" (default, original Olist-QA chunks) or
        "v2" (temporal-aware baseline; adds 7-timestamp timeline
        sentence).

    Returns
    -------
    Number of chunks written. (May be < n_orders if a customer
    foreign key is missing — same skip semantics as v1.)
    """
    if variant not in _SERIALIZERS:
        raise ValueError(
            f"Unknown variant {variant!r}. "
            f"Known: {sorted(_SERIALIZERS)}"
        )
    serializer = _SERIALIZERS[variant]

    o = tables["orders"]
    if limit:
        o = o.head(limit)

    c = tables["customers"].set_index("customer_id")
    it_g = tables["order_items"].groupby("order_id")
    p_g = tables["payments"].groupby("order_id")
    rv_g = tables["reviews"].groupby("order_id")

    # Merge translation into products once
    pr = tables["products"]
    if "product_category_name_english" not in pr.columns:
        pr = pr.merge(tables["category_translation"],
                      on="product_category_name", how="left")
    s = tables["sellers"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with out_path.open("w", encoding="utf-8") as fp:
        for _, ord_row in o.iterrows():
            oid = ord_row["order_id"]
            cid = ord_row["customer_id"]
            if cid not in c.index:
                continue
            items = it_g.get_group(oid) if oid in it_g.groups else (
                pd.DataFrame(columns=tables["order_items"].columns))
            pays = p_g.get_group(oid) if oid in p_g.groups else (
                pd.DataFrame(columns=tables["payments"].columns))
            revs = rv_g.get_group(oid) if oid in rv_g.groups else (
                pd.DataFrame(columns=tables["reviews"].columns))

            text = serializer(oid, ord_row, c.loc[cid],
                              items, pays, revs, pr, s)
            record = {
                "chunk_id": f"O:{oid}",
                "order_id": oid,
                "text": text,
                "n_tokens_approx": len(text.split()),
                "chunk_variant": variant,
            }
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1
    return n_written


# ----------------------------------------------------------------
# CLI
# ----------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build vanilla-RAG order-paragraph chunks "
                    "(v1 from prior paper, or v2 with full timeline).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--data-dir", type=Path, default=Path("./data"))
    ap.add_argument("--variant", choices=sorted(_SERIALIZERS), default="v1",
                    help="Chunk variant: v1 (original) or v2 (temporal).")
    ap.add_argument("--v2", action="store_true",
                    help="Shortcut for --variant v2.")
    ap.add_argument("--out-path", type=Path, default=None,
                    help="Output JSONL path (default: outputs/vanilla_chunks"
                         "[_v2 if --v2].jsonl)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap on #orders (dev runs)")
    args = ap.parse_args()

    # --v2 shorthand
    if args.v2:
        args.variant = "v2"

    if args.out_path is None:
        suffix = "" if args.variant == "v1" else f"_{args.variant}"
        args.out_path = Path(f"./outputs/vanilla_chunks{suffix}.jsonl")

    if load_olist is None:
        print("ERROR: cannot import build_kg.load_olist (check sys.path)",
              file=sys.stderr)
        return 1

    print(f"Loading Olist data from {args.data_dir} ...")
    tables = load_olist(args.data_dir)
    print(f"Writing {args.variant} chunks to {args.out_path} "
          f"(limit={args.limit}) ...")
    n = build_chunks(tables, args.out_path,
                     limit=args.limit, variant=args.variant)
    print(f"Done — {n:,} chunks written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
