"""
============================================================
Phase 5.1b — Build the temporal index (import-safe)
============================================================

Builds outputs/temporal_index.pkl by IMPORTING temporal_index as
a module rather than running it as __main__.

Why this exists
---------------
When you run `python phase5_temporal/temporal_index.py`, the
TemporalIndex class is defined in the __main__ module, and pickle
records its class path as "__main__.TemporalIndex". Loading that
pickle from any *other* entry point (e.g. the System C runner)
then fails with:

    AttributeError: Can't get attribute 'TemporalIndex' on
    <module '__main__' from '.../pilot_run_systemC.py'>

Building via this script records the class path as the importable
"temporal_index.TemporalIndex", so the pickle loads from anywhere.

Usage
-----
  python phase5_temporal/build_temporal_index.py
  python phase5_temporal/build_temporal_index.py --include-anomalies
============================================================
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))                       # phase5_temporal/
sys.path.insert(0, str(_HERE.parent / "phase2_kg"))  # build_kg

# IMPORT (not run-as-main) so the pickled class path is
# "temporal_index.TemporalIndex".
import temporal_index
from temporal_index import TemporalIndex


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build the temporal index as an importable-class pickle.",
    )
    ap.add_argument("--kg-path", type=Path,
                    default=Path("./outputs/kg_schema_aware.pkl"))
    ap.add_argument("--out-path", type=Path,
                    default=Path("./outputs/temporal_index.pkl"))
    ap.add_argument("--stats-path", type=Path,
                    default=Path("./outputs/temporal_index_stats.json"))
    ap.add_argument("--include-anomalies", action="store_true")
    args = ap.parse_args()

    if not args.kg_path.exists():
        print(f"ERROR: {args.kg_path} not found.", file=sys.stderr)
        return 1

    print(f"Loading KG from {args.kg_path} ...")
    with args.kg_path.open("rb") as f:
        kg = pickle.load(f)
    print(f"  {kg.number_of_nodes():,} nodes, {kg.number_of_edges():,} edges")

    print(f"Building temporal index "
          f"(filter_monotonic={not args.include_anomalies}) ...")
    idx = TemporalIndex.build(kg, filter_monotonic=not args.include_anomalies)

    # Sanity: the class must be importable, not __main__.
    assert type(idx).__module__ == "temporal_index", (
        f"Index class module is {type(idx).__module__!r}, expected "
        f"'temporal_index'. The pickle would not load elsewhere."
    )

    idx.save(args.out_path)
    args.stats_path.parent.mkdir(parents=True, exist_ok=True)
    args.stats_path.write_text(json.dumps(idx.stats.to_dict(), indent=2))

    print()
    print(json.dumps(idx.stats.to_dict(), indent=2))
    print()
    print(f"Wrote {args.out_path}  ({idx.stats.pickle_mb:.1f} MB)")
    print(f"Class path in pickle: temporal_index.TemporalIndex (import-safe)")

    # Verify the pickle round-trips from a clean namespace.
    print("\nVerifying round-trip load ...")
    reloaded = TemporalIndex.load(args.out_path)
    n = sum(reloaded.stats.n_intervals_per_type.values())
    print(f"  reloaded OK — {n:,} total intervals")
    return 0


if __name__ == "__main__":
    sys.exit(main())
