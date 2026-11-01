#!/usr/bin/env python3
"""
============================================================
analyze_generators.py — cross-generator robustness tables
============================================================

Aggregates the temporal benchmark across answer generators, for the
cross-generator extension (Mistral / Qwen / Llama via the GenAI Gateway
vs. the pinned gpt-4o-mini baseline).

It discovers, in --results-dir:
  * the gpt-4o-mini baseline  -> temporal_questions_90_<sys>_*.jsonl
  * each --gen-model run       -> temporal_questions_90_gen-<model>_<sys>_*.jsonl
(force-<level> RQ3 sweep files are ignored.)

For every (generator, system) it reports per-category EM accuracy
(trend / drift / latency / all) and the deterministic answer rate, then
the headline contrast per generator: best temporal (D, community-temporal)
vs. best time-blind (A1, vanilla v2).

The EM and answer-rate logic is *imported from* analyze_results.py, so the
numbers are identical to the single-generator pipeline. Pure stdlib.
Partial runs are handled gracefully (denominators reflect what is on disk).

Usage:
  python scripts/analyze_generators.py [--results-dir outputs/results]
============================================================
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Reuse the exact helpers from the single-generator analyzer (same dir).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_results import (  # noqa: E402
    load_jsonl, exists, pct,
    period_to_months, extract_answer_periods, allowed_months,
    TEMPORAL_CATS, ALL_TEMPORAL, LABEL,
)

PREFIX = "temporal_questions_90_"
SUFFIX = "_scored.jsonl"
BASELINE = "gpt-4o-mini"
# Match longest system tag first (community_temporal before community, etc.).
_SYS_BY_LEN = sorted(ALL_TEMPORAL, key=len, reverse=True)


# ------------------------------------------------------------
# Discovery
# ------------------------------------------------------------
def parse_middle(middle: str):
    """The chunk between PREFIX and SUFFIX -> (generator, system) or None."""
    for systag in _SYS_BY_LEN:
        if middle == systag:
            return BASELINE, systag
        if middle.endswith("_" + systag):
            gen = middle[: -(len(systag) + 1)]
            if gen.startswith("force-"):
                return None                      # RQ3 granularity sweep
            if gen.startswith("gen-"):
                return gen[4:], systag           # cross-generator run
            return None                          # unknown prefix shape
    return None


def discover(rd: Path) -> dict[str, dict[str, str]]:
    """{generator: {system_tag: middle}} for every scored file present."""
    out: dict[str, dict[str, str]] = {}
    for f in sorted(rd.glob(PREFIX + "*" + SUFFIX)):
        middle = f.name[len(PREFIX): -len(SUFFIX)]
        parsed = parse_middle(middle)
        if parsed:
            gen, systag = parsed
            out.setdefault(gen, {})[systag] = middle
    return out


# ------------------------------------------------------------
# Per-(generator, system) metrics (mirror analyze_results.py)
# ------------------------------------------------------------
def em_by_cat(scored_path: Path):
    rows = load_jsonl(scored_path)
    cells, tm, tn = {}, 0, 0
    for c in TEMPORAL_CATS:
        cl = [r for r in rows if r.get("category") == c]
        m = sum(1 for r in cl if r.get("match"))
        cells[c] = (m, len(cl))
        tm += m
        tn += len(cl)
    return cells, (tm, tn)


def answer_rate(answers_path: Path, scored_path: Path):
    """Returns (answered, n, covered, hall_among_answered) — same definition
    as analyze_results.analyze_faithfulness (deterministic, answer-only)."""
    ans = {a["qid"]: (a.get("answer") or "") for a in load_jsonl(answers_path)}
    gold = {r["qid"]: r.get("gold_anchors", []) for r in load_jsonl(scored_path)}
    n = answered = covered = hall = 0
    for qid, txt in ans.items():
        n += 1
        ga = gold.get(qid, [])
        is_ins = "INSUFFICIENT" in txt.upper()
        if not is_ins:
            answered += 1
        T = txt.upper().replace(" ", "").replace("-", "")

        def present(anchor):
            toks = [anchor.upper().replace("-", "")] + \
                   [f"{y}{mo:02d}" for (y, mo) in period_to_months(anchor)]
            return any(t in T for t in toks)

        if ga and all(present(x) for x in ga):
            covered += 1
        if not is_ins and ga:
            ext = extract_answer_periods(txt)
            allw = allowed_months(ga)
            if allw and any(m not in allw for m in ext):
                hall += 1
    return answered, n, covered, hall


def order_generators(gens) -> list[str]:
    head = [BASELINE] if BASELINE in gens else []
    return head + sorted(g for g in gens if g != BASELINE)


def _fmt(x) -> str:
    return f"{x:7.1f}%" if x is not None else f"{'--':>8}"


# ------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=Path("outputs/results"))
    args = ap.parse_args()
    rd = args.results_dir
    if not rd.exists():
        print(f"ERROR: results dir not found: {rd}")
        return 1

    gens = discover(rd)
    if not gens:
        print(f"No temporal result files found in {rd}.")
        return 1

    order = order_generators(gens)
    print(f"Reading {rd}")
    print(f"Generators found ({len(order)}): {', '.join(order)}")

    # ---- per-generator detail table ----
    for gen in order:
        print("\n" + "=" * 72)
        print(f"GENERATOR: {gen}" + ("   (baseline)" if gen == BASELINE else ""))
        print("=" * 72)
        print(f"  {'system':<22}"
              + "".join(f"{c[:5]:>8}" for c in TEMPORAL_CATS)
              + f"{'ALL':>8}{'ans%':>8}")
        for systag in ALL_TEMPORAL:
            middle = gens[gen].get(systag)
            if not middle:
                continue
            sp = rd / f"{PREFIX}{middle}{SUFFIX}"
            ap_ = rd / f"{PREFIX}{middle}_answers.jsonl"
            cells, (tm, tn) = em_by_cat(sp)
            line = f"  {LABEL[systag]:<22}"
            for c in TEMPORAL_CATS:
                m, nn = cells[c]
                line += _fmt(pct(m, nn))
            line += _fmt(pct(tm, tn))
            if exists(ap_):
                a, nn, _, _ = answer_rate(ap_, sp)
                line += _fmt(pct(a, nn))
            else:
                line += f"{'--':>8}"
            print(line + (f"   (n={tn})" if tn < 90 else ""))

    # ---- headline contrast per generator ----
    print("\n" + "=" * 72)
    print("HEADLINE CONTRAST — best temporal (D) vs best time-blind (A1), per generator")
    print("=" * 72)
    print(f"  {'generator':<34}{'D EM%':>8}{'A1 EM%':>8}{'gap pp':>9}"
          f"{'D ans%':>8}{'A1 ans%':>9}")

    def em_all(g, systag):
        mid = gens[g].get(systag)
        if not mid:
            return None
        _, (tm, tn) = em_by_cat(rd / f"{PREFIX}{mid}{SUFFIX}")
        return pct(tm, tn)

    def ans_all(g, systag):
        mid = gens[g].get(systag)
        if not mid:
            return None
        apth = rd / f"{PREFIX}{mid}_answers.jsonl"
        if not exists(apth):
            return None
        a, nn, _, _ = answer_rate(apth, rd / f"{PREFIX}{mid}{SUFFIX}")
        return pct(a, nn)

    for g in order:
        d_em, a1_em = em_all(g, "community_temporal"), em_all(g, "vanilla_rag")
        d_an, a1_an = ans_all(g, "community_temporal"), ans_all(g, "vanilla_rag")
        gap = (f"{d_em - a1_em:+8.1f}" if d_em is not None and a1_em is not None
               else f"{'--':>9}")
        print(f"  {g:<34}{_fmt(d_em)}{_fmt(a1_em)}{gap}{_fmt(d_an)}{_fmt(a1_an)}")

    print("\nDone. D = community-temporal, A1 = vanilla v2. EM + answer-rate use the\n"
          "same definitions as analyze_results.py; '(n=..)' flags partial runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
