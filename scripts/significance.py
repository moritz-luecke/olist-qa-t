#!/usr/bin/env python3
"""
============================================================
significance.py — statistical tests for the paper's §6 claims
============================================================

Implements the §5 statistical protocol on the 90-question temporal
run (pure stdlib, deterministic, no API):

  [A] Paired bootstrap (B = 1000, seed = 42) on per-question EM for
      every pair of the 7 systems: 95% CI of the accuracy difference
      + bootstrap p (share of resamples with sign flip), Bonferroni-
      corrected over the 21 pairs.
  [B] Exact McNemar (binomial on discordant pairs) for the headline
      pair D vs A1 and the routing pair D vs E.
  [C] Two-proportion z-test on the answer rate (non-abstention) for
      every pair, Bonferroni over 21.

Usage:  python scripts/significance.py [--results-dir outputs/results]
============================================================
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SYSTEMS = ["vanilla_rag", "entity", "community", "hybrid",
           "entity_temporal", "community_temporal", "hybrid_temporal"]
LABEL = {"vanilla_rag": "A1", "entity": "B1", "community": "B2",
         "hybrid": "B3", "entity_temporal": "C",
         "community_temporal": "D", "hybrid_temporal": "E"}
B = 1000
SEED = 42
N_PAIRS = 21          # Bonferroni divisor


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def binom_two_sided(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial test (sum of P(x) <= P(k))."""
    if n == 0:
        return 1.0
    def pmf(x: int) -> float:
        return math.comb(n, x) * p**x * (1 - p)**(n - x)
    pk = pmf(k)
    return min(1.0, sum(pmf(x) for x in range(n + 1)
                        if pmf(x) <= pk + 1e-12))


def two_prop_z(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided two-proportion z-test p-value."""
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (p1 - p2) / se
    return math.erfc(abs(z) / math.sqrt(2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(ROOT / "outputs/results"))
    args = ap.parse_args()
    rdir = Path(args.results_dir)

    em: dict[str, dict[str, int]] = {}
    ans: dict[str, dict[str, int]] = {}
    for s in SYSTEMS:
        em[s] = {r["qid"]: int(bool(r.get("match"))) for r in load_jsonl(
            rdir / f"temporal_questions_90_{s}_scored.jsonl")}
        ans[s] = {r["qid"]: int("INSUFFICIENT EVIDENCE"
                                not in (r.get("answer") or "").upper())
                  for r in load_jsonl(
            rdir / f"temporal_questions_90_{s}_answers.jsonl")}
    qids = sorted(em[SYSTEMS[0]])
    n = len(qids)

    print(f"[A] Paired bootstrap on EM (B={B}, seed={SEED}), "
          f"Bonferroni x{N_PAIRS}")
    rng = random.Random(SEED)
    idx_samples = [[rng.randrange(n) for _ in range(n)] for _ in range(B)]
    for i, s1 in enumerate(SYSTEMS):
        for s2 in SYSTEMS[i + 1:]:
            d = [em[s1][q] - em[s2][q] for q in qids]
            obs = sum(d) / n
            boots = sorted(sum(d[j] for j in sample) / n
                           for sample in idx_samples)
            lo, hi = boots[24], boots[974]          # 95% CI
            # bootstrap p: 2 * min(P(diff<=0), P(diff>=0))
            le = sum(1 for b in boots if b <= 0) / B
            ge = sum(1 for b in boots if b >= 0) / B
            p = min(1.0, 2 * min(le, ge))
            p_bonf = min(1.0, p * N_PAIRS)
            mark = "*" if p_bonf < 0.05 else " "
            print(f"  {LABEL[s1]:>2} vs {LABEL[s2]:<2} dEM={obs*100:+6.1f}pp"
                  f"  CI95=[{lo*100:+6.1f},{hi*100:+6.1f}]"
                  f"  p_bonf={'<0.002' if p == 0 else f'{p_bonf:.3f}'} {mark}")

    print("\n[B] Exact McNemar (discordant pairs)")
    for s1, s2 in [("community_temporal", "vanilla_rag"),
                   ("hybrid_temporal", "vanilla_rag"),
                   ("community_temporal", "hybrid_temporal"),
                   ("entity_temporal", "vanilla_rag")]:
        b_ = sum(1 for q in qids if em[s1][q] and not em[s2][q])
        c_ = sum(1 for q in qids if em[s2][q] and not em[s1][q])
        p = binom_two_sided(b_, b_ + c_)
        print(f"  {LABEL[s1]:>2} vs {LABEL[s2]:<2}  b={b_:2d} c={c_:2d}"
              f"  p={p:.4g}  p_bonf={min(1.0, p * N_PAIRS):.4g}")

    print(f"\n[C] Two-proportion z-test on answer rate, Bonferroni x{N_PAIRS}")
    for i, s1 in enumerate(SYSTEMS):
        for s2 in SYSTEMS[i + 1:]:
            k1 = sum(ans[s1].values())
            k2 = sum(ans[s2].values())
            p = two_prop_z(k1, n, k2, n)
            p_bonf = min(1.0, p * N_PAIRS)
            if p_bonf < 0.05 or {LABEL[s1], LABEL[s2]} <= {"D", "E", "A1"}:
                print(f"  {LABEL[s1]:>2} ({k1}/{n}) vs {LABEL[s2]:<2} "
                      f"({k2}/{n})  p_bonf={p_bonf:.3g}"
                      f" {'*' if p_bonf < 0.05 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
