#!/usr/bin/env python3
"""
============================================================
significance_generators.py — per-generator significance for the
cross-generator robustness table (Tab. generators, sec:results:generators)
============================================================

For each generator (the gpt-4o-mini baseline plus every --gen-model run
discovered by analyze_generators.py), tests the headline contrast
D (community-temporal) vs A1 (vanilla v2) on the 90 temporal questions:

  * EM accuracy : exact McNemar (binomial on the discordant pairs)
  * Answer rate : two-proportion z-test

Bonferroni-corrected over the number of generators. Reuses the exact
test functions from significance.py so the statistics match the paper's
single-generator protocol. Pure stdlib, deterministic, no API.

Usage: python scripts/significance_generators.py [--results-dir outputs/results]
============================================================
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from significance import binom_two_sided, two_prop_z, load_jsonl  # noqa: E402
import analyze_generators as AG  # noqa: E402


def em_ans(rd: Path, mid: str):
    em = {r["qid"]: int(bool(r.get("match")))
          for r in load_jsonl(rd / f"{AG.PREFIX}{mid}_scored.jsonl")}
    ans = {r["qid"]: int("INSUFFICIENT EVIDENCE" not in (r.get("answer") or "").upper())
           for r in load_jsonl(rd / f"{AG.PREFIX}{mid}_answers.jsonl")}
    return em, ans


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("outputs/results"))
    args = ap.parse_args()
    rd = args.results_dir
    gens = AG.discover(rd)
    order = AG.order_generators(gens)
    ng = len(order)

    print("D (community-temporal) vs A1 (vanilla v2), per generator")
    print(f"EM: exact McNemar (b=D-right/A1-wrong, c=reverse) | "
          f"answer rate: two-proportion z | Bonferroni x{ng}\n")
    print(f"  {'generator':<36}{'b':>4}{'c':>4}{'McNemar p':>12}{'p_bonf':>9}"
          f"{'ans z p':>11}{'p_bonf':>9}")
    for g in order:
        dmid = gens[g].get("community_temporal")
        amid = gens[g].get("vanilla_rag")
        if not (dmid and amid):
            print(f"  {g:<36}(missing D or A1 file)")
            continue
        dEM, dANS = em_ans(rd, dmid)
        aEM, aANS = em_ans(rd, amid)
        qids = sorted(set(dEM) & set(aEM))
        n = len(qids)
        b = sum(1 for q in qids if dEM[q] and not aEM[q])
        c = sum(1 for q in qids if aEM[q] and not dEM[q])
        p_mc = binom_two_sided(b, b + c)
        dk = sum(dANS[q] for q in qids)
        ak = sum(aANS[q] for q in qids)
        p_z = two_prop_z(dk, n, ak, n)
        print(f"  {g[:34]:<36}{b:>4}{c:>4}{p_mc:>12.2g}{min(1, p_mc*ng):>9.2g}"
              f"{p_z:>11.2g}{min(1, p_z*ng):>9.2g}")
    print("\n(b+c = discordant pairs; McNemar uses the exact two-sided "
          "binomial. n=90 paired questions per generator.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
