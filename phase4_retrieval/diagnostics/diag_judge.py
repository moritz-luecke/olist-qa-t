"""
============================================================
Phase 4 — Judge result analysis
============================================================

Reads judge_*_vs_*.jsonl files from outputs/results/ and produces:
  1. Win-loss matrix with TIE handling
  2. Per-category breakdown (comparison / root_cause / global)
  3. Paired bootstrap on the win differential (does A really beat B?)
  4. Bonferroni-corrected significance markers

The bootstrap treats each question as a unit with outcome
∈ {A=+1, TIE=0, B=−1} and resamples with replacement; the
mean of the resampled outcomes is the win-margin estimate, and
the percentile CI tells us whether 0 is inside.

Usage:
    cd phase4_retrieval
    python diag_judge.py
============================================================
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import bonferroni_alpha


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "outputs" / "results"

OPEN_CATEGORIES = {"comparison", "root_cause", "global"}


def _outcome_int(verdict: dict, sys_a_name: str) -> int:
    """+1 if A won, -1 if B won, 0 for TIE.
    sys_a_name is the name we treat as 'A' in the pair (consistent across
    bootstrap samples)."""
    winner = verdict["winner_system"]
    if winner == "TIE":
        return 0
    return 1 if winner == sys_a_name else -1


def bootstrap_pairwise(outcomes: np.ndarray,
                       b_iters: int = 1000,
                       alpha: float = 0.05,
                       seed: int = 42) -> dict:
    """Bootstrap on a {-1,0,+1} outcome vector. Returns mean win-margin
    and percentile CI. A 95% CI excluding 0 means significant."""
    rng = np.random.default_rng(seed)
    n = len(outcomes)
    observed = float(outcomes.mean())
    idx = rng.integers(0, n, size=(b_iters, n))
    samples = outcomes[idx].mean(axis=1)
    lo = float(np.percentile(samples, 100 * alpha / 2))
    hi = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    # Two-sided p (Davison-Hinkley correction)
    shifted = samples - observed
    p = (np.sum(np.abs(shifted) >= abs(observed)) + 1) / (b_iters + 1)
    return {"mean": observed, "ci_low": lo, "ci_high": hi,
            "p": float(p), "n": n}


def main() -> int:
    questions = {q["qid"]: q for q in
                 (json.loads(l) for l in
                  (PROJECT_ROOT / "outputs/pilot_questions.jsonl").open())}

    judge_files = sorted(RESULTS.glob("judge_*_vs_*.jsonl"))
    if not judge_files:
        print(f"No judge_*_vs_*.jsonl in {RESULTS}", file=sys.stderr)
        return 1

    # Parse out the system pair from each filename
    # Convention: judge_{a_system}_answers_vs_{b_system}_answers.jsonl
    all_systems = set()
    per_pair: dict[tuple[str, str], list[dict]] = {}
    for fp in judge_files:
        stem = fp.stem.replace("judge_", "")
        # "{a}_answers_vs_{b}_answers"
        try:
            left, right = stem.split("_vs_")
            sys_a = left.removesuffix("_answers")
            sys_b = right.removesuffix("_answers")
        except ValueError:
            print(f"  [SKIP] unparseable filename: {fp.name}")
            continue
        all_systems.update([sys_a, sys_b])
        verdicts = [json.loads(l) for l in fp.open()]
        per_pair[(sys_a, sys_b)] = verdicts

    systems = sorted(all_systems)
    alpha_corr = bonferroni_alpha(len(systems), 0.05)

    # ----------------------------------------------------------------
    # Per-pair overall + per-category
    # ----------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  PAIRWISE JUDGE RESULTS")
    print(f"  α'={alpha_corr:.4g} after Bonferroni for {len(systems)} systems "
          f"({len(systems) * (len(systems)-1) // 2} pairs)")
    print("="*70)

    for (sys_a, sys_b), verdicts in per_pair.items():
        print(f"\n--- {sys_a} (A) vs {sys_b} (B) ---")

        # Overall outcomes
        outcomes = np.array([_outcome_int(v, sys_a) for v in verdicts])
        tally = Counter(v["winner_system"] for v in verdicts)
        n = len(outcomes)
        print(f"  Overall (n={n}): "
              f"A={tally.get(sys_a,0)}  TIE={tally.get('TIE',0)}  "
              f"B={tally.get(sys_b,0)}")

        b = bootstrap_pairwise(outcomes)
        sig = "*" if (b["ci_low"] > 0 or b["ci_high"] < 0) and b["p"] < alpha_corr else " "
        print(f"  Win margin (A−B): {b['mean']:+.3f}   "
              f"95% CI [{b['ci_low']:+.3f}, {b['ci_high']:+.3f}]   "
              f"p={b['p']:.4g}{sig}")

        # Per category
        by_cat = defaultdict(list)
        for v in verdicts:
            cat = questions[v["qid"]]["category"]
            by_cat[cat].append(_outcome_int(v, sys_a))
        for cat in ("comparison", "root_cause", "global"):
            arr = np.array(by_cat[cat])
            if len(arr) == 0:
                continue
            cb = bootstrap_pairwise(arr, b_iters=1000, seed=42)
            csig = "*" if (cb["ci_low"] > 0 or cb["ci_high"] < 0) \
                          and cb["p"] < alpha_corr else " "
            n_a = sum(1 for x in arr if x == +1)
            n_t = sum(1 for x in arr if x == 0)
            n_b = sum(1 for x in arr if x == -1)
            print(f"    {cat:12s} (n={len(arr):>2}): "
                  f"A={n_a} TIE={n_t} B={n_b}   "
                  f"margin {cb['mean']:+.3f}  "
                  f"CI [{cb['ci_low']:+.3f},{cb['ci_high']:+.3f}]  "
                  f"p={cb['p']:.3g}{csig}")

    # ----------------------------------------------------------------
    # Composite ranking across all pairs (Bradley-Terry-style heuristic)
    # ----------------------------------------------------------------
    print(f"\n{'='*70}")
    print("  COMPOSITE WIN RATES (across all pairs the system appeared in)")
    print("="*70)
    win_count = Counter()
    pair_count = Counter()
    for (sys_a, sys_b), verdicts in per_pair.items():
        for v in verdicts:
            pair_count[sys_a] += 1
            pair_count[sys_b] += 1
            if v["winner_system"] == sys_a:
                win_count[sys_a] += 1
            elif v["winner_system"] == sys_b:
                win_count[sys_b] += 1
    print(f"\n  {'system':14s}  wins  (out of)   win-rate")
    for s in sorted(systems, key=lambda x: -win_count[x] / max(pair_count[x],1)):
        n = pair_count[s]
        w = win_count[s]
        rate = 100*w/n if n > 0 else 0.0
        print(f"  {s:14s}  {w:4d}  ({n:>4d})    {rate:5.1f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
