"""
============================================================
Phase 6.1 — Triple extraction (System C, parallel)
============================================================

Runs the triple-extraction prompt from Appendix A.2 over a
subsample of Olist order paragraphs using a ThreadPoolExecutor.
With 8 parallel workers expected throughput is ~5-8 chunks/sec
on Tier-1 OpenAI; 5000 chunks → ~15-20 min total.

WARNING: parallel calls bypass the LLM's sequential rate-limit
sleep. If you hit a 429, lower --workers from 8 to 4.

Output schema
-------------
outputs/triples_extracted.jsonl, one line per extracted triple:
  {"source_chunk_id": "O:abc123...",
   "subject":   {"id_hint": "abc123", "type": "Order",      "attrs": {...}},
   "predicate": "paid_by",
   "object":    {"id_hint": "xyz789", "type": "Payment",    "attrs": {...}},
   "cached":    bool}

Usage
-----
  cd phase4_retrieval
  python extract_triples.py --n-orders 5000 --yes
============================================================
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import Paths, GENERATION_MODEL
from llm import ChatLLM

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **_): return it


EXTRACTION_SYSTEM = """\
You are a precise information extractor. Given a paragraph
describing one e-commerce order, identify the typed entities and
relationships present in the text and return them as a JSON object.

Entity types: Order, Customer, Seller, Product, OrderItem, Payment,
Review, Geolocation. Each entity must carry an `id_hint` (the
literal ID string from the paragraph, or the city/state name for
Geolocations) and an optional `attrs` dict for any inline fields.

Relation types: placed (Customer->Order), has_item (Order->OrderItem),
of_product (OrderItem->Product), sold_by (OrderItem->Seller),
paid_by (Order->Payment), reviewed_by (Order->Review),
located_at (Customer->Geolocation, Seller->Geolocation).

Rules:
  - Extract ONLY entities/relations explicitly stated in the text.
  - Do not infer or fill in missing fields.
  - If a field looks like an ID but is not 32 hex chars, leave it
    in id_hint as-is.
  - Output strictly valid JSON; no markdown, no preamble.
"""

EXTRACTION_USER = """\
PARAGRAPH:
{paragraph}

Return a single JSON object with two keys: "entities" (list) and
"triples" (list of [subject_id_hint, predicate, object_id_hint]).
Each entity is {{"id_hint": "...", "type": "...", "attrs": {{...}}}}.\
"""


VALID_TYPES = {"Order", "Customer", "Seller", "Product",
               "OrderItem", "Payment", "Review", "Geolocation"}
VALID_RELATIONS = {"placed", "has_item", "of_product", "sold_by",
                   "paid_by", "reviewed_by", "located_at"}


def validate_extraction(data: dict) -> list[dict]:
    """Return list of triple-dicts; drop invalid silently."""
    if not isinstance(data, dict):
        return []
    raw_ents = data.get("entities", [])
    raw_trips = data.get("triples", [])

    by_hint: dict[str, dict] = {}
    for e in raw_ents:
        if not isinstance(e, dict):
            continue
        hint = str(e.get("id_hint", "")).strip()
        etype = str(e.get("type", "")).strip()
        if not hint or etype not in VALID_TYPES:
            continue
        attrs = e.get("attrs") if isinstance(e.get("attrs"), dict) else {}
        by_hint[hint] = {"id_hint": hint, "type": etype, "attrs": attrs}

    out: list[dict] = []
    for t in raw_trips:
        if not (isinstance(t, list) and len(t) == 3):
            continue
        s_hint, pred, o_hint = (str(x).strip() for x in t)
        if pred not in VALID_RELATIONS:
            continue
        s_ent = by_hint.get(s_hint)
        o_ent = by_hint.get(o_hint)
        if s_ent is None or o_ent is None:
            continue
        out.append({"subject": s_ent, "predicate": pred, "object": o_ent})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-orders", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8,
                    help="Parallel API workers. Lower if you hit 429s.")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    paths = Paths()
    chunks_path = paths.chunks_path
    if not chunks_path.exists():
        print(f"ERROR: {chunks_path} missing. Run chunking.py first.",
              file=sys.stderr)
        return 1

    print(f"Loading chunks from {chunks_path} ...")
    chunks = [json.loads(line) for line in chunks_path.open()]
    print(f"  {len(chunks):,} chunks total")

    rng = random.Random(args.seed)
    sampled = chunks.copy()
    rng.shuffle(sampled)
    sampled = sampled[:args.n_orders]
    print(f"  {len(sampled):,} chunks sampled (seed={args.seed})")

    sample_ids_path = paths.cache_dir.parent / "auto_kg_sample_ids.json"
    sample_ids_path.parent.mkdir(parents=True, exist_ok=True)
    with sample_ids_path.open("w") as fp:
        json.dump({"seed": args.seed, "n": len(sampled),
                   "chunk_ids": [c["chunk_id"] for c in sampled]}, fp)
    print(f"  Sample IDs written to {sample_ids_path}")

    est = len(sampled) * 0.000175
    print(f"\nEstimated cost: ${est:.2f} ({len(sampled)} extraction calls, "
          f"{args.workers} parallel workers)")
    if not args.dry_run and not args.yes:
        print("Re-run with --yes to confirm.")
        return 0

    llm = ChatLLM(paths.cache_dir, dry_run=args.dry_run,
                  model=GENERATION_MODEL,
                  max_tokens=600)

    out_path = paths.cache_dir.parent / "triples_extracted.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: if file exists, skip chunks that are already in it
    already_done: set[str] = set()
    if out_path.exists():
        for line in out_path.open():
            try:
                already_done.add(json.loads(line)["source_chunk_id"])
            except Exception:
                pass
        print(f"  Resume: {len(already_done)} chunks already processed.")

    pending = [c for c in sampled if c["chunk_id"] not in already_done]
    print(f"  {len(pending):,} chunks to process this run.")

    # Threadsafe append to JSONL
    write_lock = threading.Lock()
    out_fp = out_path.open("a", encoding="utf-8")

    def process(ch):
        """Run one extraction; returns (n_triples_written, parse_failed)."""
        user_msg = EXTRACTION_USER.format(paragraph=ch["text"])
        try:
            resp = llm.chat(
                system=EXTRACTION_SYSTEM,
                user=user_msg,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            return (0, True, str(e)[:80])

        try:
            data = json.loads(resp["text"])
        except (json.JSONDecodeError, TypeError):
            return (0, True, "json_parse")

        triples = validate_extraction(data)
        if not triples:
            return (0, False, None)

        lines = []
        for t in triples:
            record = {
                "source_chunk_id": ch["chunk_id"],
                **t,
                "cached": resp.get("cached", False),
            }
            lines.append(json.dumps(record, ensure_ascii=False))
        with write_lock:
            out_fp.write("\n".join(lines) + "\n")
            out_fp.flush()
        return (len(triples), False, None)

    n_calls = 0
    n_triples = 0
    n_errors = 0
    error_samples: list[str] = []

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process, ch): ch for ch in pending}
        pbar = tqdm(total=len(futures), desc="extracting", unit="chunk")
        for fut in as_completed(futures):
            n_calls += 1
            try:
                n_t, failed, err = fut.result()
            except Exception as e:
                n_errors += 1
                if len(error_samples) < 5:
                    error_samples.append(str(e)[:80])
                pbar.update(1)
                continue
            n_triples += n_t
            if failed:
                n_errors += 1
                if err and len(error_samples) < 5:
                    error_samples.append(err)
            pbar.update(1)
        pbar.close()

    out_fp.close()

    print(f"\nDone. {n_calls:,} calls -> "
          f"{n_triples:,} triples written this run "
          f"({n_triples/max(n_calls,1):.1f}/chunk avg) "
          f"-> {out_path}")
    if n_errors:
        print(f"  Errors/parse failures: {n_errors:,} "
              f"({100*n_errors/n_calls:.1f}%)")
        if error_samples:
            print(f"  Sample error messages: {error_samples}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
