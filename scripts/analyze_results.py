#!/usr/bin/env python3
"""
============================================================
analyze_results.py — reproduce every paper-table number from
the JSONL artefacts in outputs/results/.
============================================================

Replaces the ad-hoc analyses done during paper writing with a
single reproducible pass. Pure stdlib; no API, no pandas.

Computes:
  [1] Sanity (Tab. sanity-vanilla-v2): per-category EM for the
      4 time-blind systems on the 150 prior questions, tolerant
      (match field) AND strict (recomputed from detail.closest).
  [2] Prior pattern (Tab. prior-pattern): pairwise win/tie/loss
      vs vanilla_v2 per open category, from the judge_*.jsonl.
  [3] Headline temporal (Tab. headline): per-category EM for all
      7 systems on the 90 temporal questions.
  [4] Latency (Tab. latency-mae): MAE, N, within-3-day on scalar
      latency questions, plus tolerance-robustness curve.
  [5] Cost (Tab. cost): tokens/query and USD from the tokens field.
  [6] Faithfulness (Tab. faithfulness): DETERMINISTIC answer rate,
      conservative coverage, and out-of-window hallucination,
      computed from the answer text only (no LLM judge).
  [8] Category-drift set overlap: lenient credit on the 18
      category-drift questions — share of gold categories the
      answer names (string containment), ignoring magnitudes.

Usage:
  python scripts/analyze_results.py [--results-dir outputs/results]
                                    [--pilot-questions benchmark/pilot_questions.jsonl]
============================================================
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

# ------------------------------------------------------------
# System tags (file stems) -> display labels
# ------------------------------------------------------------
TIMEBLIND = ["vanilla_rag", "entity", "community", "hybrid"]
TEMPORAL_EXTRA = ["entity_temporal", "community_temporal", "hybrid_temporal"]
ALL_TEMPORAL = TIMEBLIND + TEMPORAL_EXTRA

LABEL = {
    "vanilla_rag": "A1 Vanilla v2",
    "entity": "B1 Entity",
    "community": "B2 Community",
    "hybrid": "B3 Hybrid",
    "entity_temporal": "C  Entity-Temporal",
    "community_temporal": "D  Community-Temporal",
    "hybrid_temporal": "E  Hybrid-Temporal",
}
TEMPORAL_CATS = ["trend", "drift", "latency"]
PRICE_IN, PRICE_OUT = 0.15, 0.60  # gpt-4o-mini USD / 1M tokens (verify vs current)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def period_to_months(p: str) -> list[tuple[int, int]]:
    """Canonical period string -> list of (year, month)."""
    p = p.strip().upper()
    for pat, fn in [
        (r"(\d{4})$", lambda m: [(int(m.group(1)), mo) for mo in range(1, 13)]),
        (r"(\d{4})-Q([1-4])$", lambda m: [(int(m.group(1)), mo)
            for mo in range((int(m.group(2)) - 1) * 3 + 1, (int(m.group(2)) - 1) * 3 + 4)]),
        (r"(\d{4})-H([12])$", lambda m: [(int(m.group(1)), mo)
            for mo in (range(1, 7) if m.group(2) == "1" else range(7, 13))]),
        (r"(\d{4})-(\d{2})$", lambda m: [(int(m.group(1)), int(m.group(2)))]),
    ]:
        mm = re.fullmatch(pat, p)
        if mm:
            return fn(mm)
    return []


MONTH_NAMES = {
    "JANUARY": 1, "FEBRUARY": 2, "MARCH": 3, "APRIL": 4, "MAY": 5, "JUNE": 6,
    "JULY": 7, "AUGUST": 8, "SEPTEMBER": 9, "OCTOBER": 10, "NOVEMBER": 11,
    "DECEMBER": 12, "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "JUN": 6, "JUL": 7,
    "AUG": 8, "SEP": 9, "SEPT": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def extract_answer_periods(ans: str) -> set[tuple[int, int]]:
    """Best-effort (year, month) set asserted by the answer text."""
    T = ans.upper()
    raw = set()
    for m in re.finditer(r"(\d{4})-Q([1-4])", T):
        raw.add((int(m.group(1)), "Q", int(m.group(2))))
    for m in re.finditer(r"Q([1-4])\s+(?:OF\s+)?(\d{4})", T):
        raw.add((int(m.group(2)), "Q", int(m.group(1))))
    for m in re.finditer(r"(\d{4})-(0[1-9]|1[0-2])(?!\d)", T):
        raw.add((int(m.group(1)), "M", int(m.group(2))))
    for name, mo in MONTH_NAMES.items():
        for m in re.finditer(rf"\b{name}\s+(\d{{4}})", T):
            raw.add((int(m.group(1)), "M", mo))
    months = set()
    for (y, kind, n) in raw:
        if kind == "Q":
            months |= {(y, mm) for mm in range((n - 1) * 3 + 1, (n - 1) * 3 + 4)}
        else:
            months.add((y, n))
    return months


def allowed_months(gold: list[str], spill: int = 2) -> set[tuple[int, int]]:
    ms = set()
    for a in gold:
        ms |= set(period_to_months(a))
    if ms:
        mx = max(y * 12 + mo for (y, mo) in ms)
        for k in range(1, spill + 1):
            idx = mx + k
            ms.add((idx // 12, idx % 12 if idx % 12 else 12))
    return ms


def strict_num(closest, gold) -> bool:
    """Exact match to the gold's reported decimal precision."""
    if closest is None:
        return False
    g = str(gold)
    k = len(g.split(".")[1]) if "." in g else 0
    try:
        return round(float(closest), k) == round(float(gold), k)
    except (TypeError, ValueError):
        return False


def is_scalar(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def pct(n, d):
    return 100.0 * n / d if d else 0.0


# ------------------------------------------------------------
# [1] Sanity
# ------------------------------------------------------------
def analyze_sanity(rd: Path):
    print("\n" + "=" * 64)
    print("[1] SANITY — 150 prior questions, time-blind systems")
    print("    (tolerant = match field; strict = recomputed exact)")
    print("=" * 64)
    cats = ["factual", "aggregation", "comparison", "root_cause", "global"]
    for s in TIMEBLIND:
        f = rd / f"pilot_questions_{s}_scored.jsonl"
        if not exists(f):
            print(f"  {LABEL[s]:<22} (missing {f.name})")
            continue
        rows = load_jsonl(f)
        line = f"  {LABEL[s]:<22}"
        for c in ("factual", "aggregation"):
            cl = [r for r in rows if r.get("category") == c and not r.get("needs_judge")]
            tol = sum(1 for r in cl if r.get("match"))
            strict = 0
            for r in cl:
                g = r.get("gold_answer")
                d = r.get("detail") or {}
                if is_scalar(g) and isinstance(d, dict):
                    if strict_num(d.get("closest"), g):
                        strict += 1
                elif not is_scalar(g) and r.get("match"):
                    strict += 1
            line += f"  {c[:4]}: tol {pct(tol,len(cl)):4.1f}% / strict {pct(strict,len(cl)):4.1f}%"
        n_judge = sum(1 for r in rows if r.get("needs_judge"))
        line += f"  | open(rubric): {n_judge} -> judge"
        print(line)


# ------------------------------------------------------------
# [2] Prior pattern (win-rates)
# ------------------------------------------------------------
def analyze_prior_pattern(rd: Path, pilot_q: Path):
    print("\n" + "=" * 64)
    print("[2] PRIOR PATTERN — pairwise win/tie/loss vs vanilla_v2")
    print("    (open categories; win-rate counts ties as 1/2)")
    print("=" * 64)
    if not exists(pilot_q):
        print(f"  (need {pilot_q} for category mapping — skipped)")
        return
    cat = {q["qid"]: q["category"] for q in load_jsonl(pilot_q)}
    for s in ["entity", "community", "hybrid"]:
        f = rd / f"judge_pilot_questions_{s}_answers_vs_pilot_questions_vanilla_rag_answers.jsonl"
        if not exists(f):
            print(f"  {LABEL[s]:<22} (missing judge file)")
            continue
        rows = load_jsonl(f)
        print(f"  {LABEL[s]} vs A1:")
        for c in ["comparison", "root_cause", "global"]:
            w = t = l = 0
            for r in rows:
                if cat.get(r["qid"]) != c:
                    continue
                win = r.get("winner_system", "TIE")
                if win == s:
                    w += 1
                elif win == "vanilla_rag":
                    l += 1
                else:
                    t += 1
            n = w + t + l
            wr = pct(w + 0.5 * t, n)
            print(f"      {c:<11} W/T/L {w:>2}/{t:>2}/{l:>2}  win-rate {wr:5.1f}%")


# ------------------------------------------------------------
# [3] Headline temporal
# ------------------------------------------------------------
def analyze_headline(rd: Path):
    print("\n" + "=" * 64)
    print("[3] HEADLINE TEMPORAL — per-category EM, 90 temporal Qs")
    print("=" * 64)
    hdr = f"  {'system':<22}" + "".join(f"{c[:5]:>8}" for c in TEMPORAL_CATS) + f"{'ALL':>8}"
    print(hdr)
    for s in ALL_TEMPORAL:
        f = rd / f"temporal_questions_90_{s}_scored.jsonl"
        if not exists(f):
            print(f"  {LABEL[s]:<22} (missing)")
            continue
        rows = load_jsonl(f)
        line = f"  {LABEL[s]:<22}"
        tot_m = tot_n = 0
        for c in TEMPORAL_CATS:
            cl = [r for r in rows if r.get("category") == c]
            m = sum(1 for r in cl if r.get("match"))
            tot_m += m
            tot_n += len(cl)
            line += f"{pct(m,len(cl)):7.1f}%"
        line += f"{pct(tot_m,tot_n):7.1f}%"
        print(line)


# ------------------------------------------------------------
# [4] Latency MAE + robustness
# ------------------------------------------------------------
def analyze_latency(rd: Path):
    print("\n" + "=" * 64)
    print("[4] LATENCY — scalar-gold latency questions")
    print("    MAE/N/within-3d, and match count at abs-tolerance levels")
    print("=" * 64)
    print(f"  {'system':<22}{'N/15':>6}{'MAE':>7}{'w3d':>6}"
          f"{'@0.5d':>7}{'@1.0d':>7}{'scorer':>8}")
    for s in ALL_TEMPORAL:
        f = rd / f"temporal_questions_90_{s}_scored.jsonl"
        if not exists(f):
            continue
        rows = load_jsonl(f)
        scal = [r for r in rows if r.get("category") == "latency" and is_scalar(r.get("gold_answer"))]
        errs = []
        for r in scal:
            c = (r.get("detail") or {}).get("closest")
            if c is not None:
                errs.append(abs(float(c) - float(r["gold_answer"])))

        def at(tol):
            n = 0
            for r in scal:
                c = (r.get("detail") or {}).get("closest")
                if c is not None and abs(float(c) - float(r["gold_answer"])) <= tol:
                    n += 1
            return n

        scorer = sum(1 for r in scal if r.get("match"))
        if errs:
            mae = statistics.mean(errs)
            w3 = pct(sum(1 for e in errs if e <= 3.0), len(errs))
            print(f"  {LABEL[s]:<22}{len(errs):>6}{mae:>7.2f}{w3:>5.0f}%"
                  f"{at(0.5):>7}{at(1.0):>7}{scorer:>8}")
        else:
            print(f"  {LABEL[s]:<22}{0:>6}{'n/a':>7}")


# ------------------------------------------------------------
# [5] Cost
# ------------------------------------------------------------
def analyze_cost(rd: Path):
    print("\n" + "=" * 64)
    print(f"[5] COST — tokens & USD (gpt-4o-mini ${PRICE_IN}/${PRICE_OUT} per 1M)")
    print("=" * 64)
    print(f"  {'system':<22}{'prompt':>11}{'compl':>9}{'tok/q':>8}{'USD':>9}")
    total = 0.0
    for s in ALL_TEMPORAL:
        f = rd / f"temporal_questions_90_{s}_answers.jsonl"
        if not exists(f):
            continue
        rows = load_jsonl(f)
        pin = pout = 0
        for a in rows:
            t = a.get("tokens") or {}
            if isinstance(t, str):
                try:
                    t = json.loads(t.replace("'", '"'))
                except json.JSONDecodeError:
                    t = {}
            pin += t.get("prompt", 0)
            pout += t.get("completion", 0)
        usd = pin / 1e6 * PRICE_IN + pout / 1e6 * PRICE_OUT
        total += usd
        print(f"  {LABEL[s]:<22}{pin:>11,}{pout:>9,}{(pin+pout)/len(rows):>8.0f}{usd:>9.4f}")
    print(f"  {'TOTAL':<22}{'':<11}{'':<9}{'':<8}{total:>9.3f}")


# ------------------------------------------------------------
# [6] Faithfulness (deterministic, answer-only)
# ------------------------------------------------------------
def analyze_faithfulness(rd: Path):
    print("\n" + "=" * 64)
    print("[6] FAITHFULNESS — deterministic, answer-only (no judge)")
    print("    answer rate / coverage (conservative) / halluc (among answered)")
    print("=" * 64)
    print(f"  {'system':<22}{'answer%':>9}{'cov%':>8}{'halluc%':>9}")
    for s in ALL_TEMPORAL:
        af = rd / f"temporal_questions_90_{s}_answers.jsonl"
        sf = rd / f"temporal_questions_90_{s}_scored.jsonl"
        if not (exists(af) and exists(sf)):
            continue
        ans = {a["qid"]: a["answer"] for a in load_jsonl(af)}
        gold = {r["qid"]: r.get("gold_anchors", []) for r in load_jsonl(sf)}
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
        ha = pct(hall, answered) if answered else float("nan")
        ha_s = f"{ha:8.1f}%" if answered else f"{'--':>8} "
        print(f"  {LABEL[s]:<22}{pct(answered,n):8.1f}%{pct(covered,n):7.1f}%{ha_s}")


# ------------------------------------------------------------
# [7] RQ3 granularity sweep + bi-level-vs-flat ablation (System D)
# ------------------------------------------------------------
def analyze_rq3(rd: Path):
    base = rd / "temporal_questions_90_community_temporal_scored.jsonl"
    levels = ["week", "month", "quarter"]
    forced = {L: rd / f"temporal_questions_90_force-{L}_community_temporal_scored.jsonl"
              for L in levels}
    have_forced = [L for L in levels if exists(forced[L])]
    if not have_forced:
        return  # RQ3 sweep not run yet — silently skip
    print("\n" + "=" * 64)
    print("[7] RQ3 — System D granularity sweep + flat-vs-bi-level ablation")
    print("=" * 64)

    def per_cat(path):
        rows = load_jsonl(path)
        out = {}
        for c in TEMPORAL_CATS:
            cl = [r for r in rows if r.get("category") == c]
            out[c] = (sum(1 for r in cl if r.get("match")), len(cl))
        tot = (sum(o[0] for o in out.values()), sum(o[1] for o in out.values()))
        return out, tot

    # Granularity heatmap (forced levels + bi-level baseline if present)
    print(f"  {'granularity':<16}" + "".join(f"{c[:5]:>9}" for c in TEMPORAL_CATS) + f"{'ALL':>9}")
    rowsrc = [(f"force-{L}", forced[L]) for L in have_forced]
    if exists(base):
        rowsrc.append(("bi-level (auto)", base))
    for label, path in rowsrc:
        pc, tot = per_cat(path)
        line = f"  {label:<16}"
        for c in TEMPORAL_CATS:
            m, n = pc[c]
            line += f"{pct(m,n):8.1f}%"
        line += f"{pct(tot[0],tot[1]):8.1f}%"
        print(line)

    # H5: flat (force-month) vs bi-level (auto baseline)
    if exists(forced["month"]) and exists(base):
        _, flat = per_cat(forced["month"])
        _, bil = per_cat(base)
        print(f"\n  H5 bi-level vs flat (System D, all 90):")
        print(f"    bi-level (auto granularity): {bil[0]}/{bil[1]} "
              f"({pct(bil[0],bil[1]):.1f}%)")
        print(f"    flat (month-only):           {flat[0]}/{flat[1]} "
              f"({pct(flat[0],flat[1]):.1f}%)")
        d = pct(bil[0], bil[1]) - pct(flat[0], flat[1])
        print(f"    bi-level advantage:          {d:+.1f} pp")


# ------------------------------------------------------------
# [8] Category-drift set overlap (lenient)
# ------------------------------------------------------------
def analyze_drift_overlap(rd: Path, questions_path: Path) -> None:
    if not exists(questions_path):
        return
    qs = load_jsonl(questions_path)
    cd = [q for q in qs if q.get("sub_type") == "category_drift"]
    if not cd:
        return

    def gold_cats(g):
        if isinstance(g, list):
            return [str(x[0]) if isinstance(x, (list, tuple)) else str(x)
                    for x in g]
        if isinstance(g, dict):
            ks = list(g.keys())
            if all(re.match(r"\d{4}", str(k)) for k in ks):
                out = []
                for v in g.values():
                    if isinstance(v, dict):
                        out += list(v.keys())
                return out
            return ks
        return []

    print("\n" + "=" * 64)
    print("[8] CATEGORY-DRIFT SET OVERLAP — lenient credit, 18 Qs")
    print("    (share of gold categories named in the answer,")
    print("     ignoring magnitudes; 'any' = Qs naming >=1 gold cat)")
    print("=" * 64)
    print(f"  {'system':<24} {'mean overlap':>12} {'any':>5}")
    for s in ALL_TEMPORAL:
        ap_ = rd / f"temporal_questions_90_{s}_answers.jsonl"
        if not exists(ap_):
            continue
        A = {r["qid"]: (r.get("answer") or "") for r in load_jsonl(ap_)}
        fr, any1 = [], 0
        for q in cd:
            cats = set(gold_cats(q["gold_answer"]))
            if not cats:
                continue
            ans = A.get(q["qid"], "").lower().replace("_", " ")
            named = sum(1 for c in cats
                        if c.lower().replace("_", " ") in ans)
            fr.append(named / len(cats))
            if named:
                any1 += 1
        print(f"  {LABEL[s]:<24} {100*sum(fr)/len(fr):11.1f}% "
              f"{any1:>3}/{len(fr)}")


# ------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=Path("outputs/results"))
    ap.add_argument("--pilot-questions", type=Path,
                    default=Path("benchmark/pilot_questions.jsonl"))
    args = ap.parse_args()

    rd = args.results_dir
    if not rd.exists():
        print(f"ERROR: results dir not found: {rd}")
        return 1
    print(f"Reading artefacts from {rd}")
    analyze_sanity(rd)
    analyze_prior_pattern(rd, args.pilot_questions)
    analyze_headline(rd)
    analyze_latency(rd)
    analyze_cost(rd)
    analyze_faithfulness(rd)
    analyze_rq3(rd)
    analyze_drift_overlap(rd, Path("outputs/temporal_questions_90.jsonl"))
    print("\nDone. These numbers populate the paper's §6 tables.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
