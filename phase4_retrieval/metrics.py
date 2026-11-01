"""
============================================================
Phase 4 — Evaluation metrics
============================================================

Closed-form scoring for factual + aggregation questions (where
the gold answer is a string, number, or list), and statistical
machinery for the cross-system comparison reported in §5.

Open-ended categories (comparison, root_cause, global) are
scored via the LLM-as-judge in judge.py — this module's only
contribution there is the κ helper for judge–human agreement.

What's here
-----------
- normalize_answer()        : strip + lowercase + drop trailing punct,
                              fold Unicode accents (NFD-decompose then
                              drop combining marks), so 'Camboriú' →
                              'camboriu'.
- extract_number()          : pull the first numeric token from a string;
                              FIRST strips 32-hex tokens (order/customer
                              IDs) so we don't pick up digits from them.
- score_answer()            : returns "correct" | "wrong" | "abstain"
- score_run()               : maps a list of AnswerRecords to per-q outcomes
- paired_bootstrap()        : 1k-resample test on two systems' accuracy
                              vectors, returns (mean_diff, ci_low, ci_high, p)
- bonferroni_alpha()        : α / #pairs
- cohens_kappa()            : two-rater agreement on ordinal labels
- CLI                       : `python metrics.py outputs/results/` →
                              accuracy table + pairwise bootstrap matrix
============================================================
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np


ABSTAIN_TOKEN = "INSUFFICIENT EVIDENCE"

# Categories scored by exact match. The other three use the judge.
CLOSED_CATEGORIES = {"factual", "aggregation"}

# Default numeric tolerance — absolute for ints, relative for floats.
NUM_ABS_TOL = 0.5         # answers within 0.5 of an int gold count as correct
NUM_REL_TOL = 0.01        # 1% relative tolerance for floats

# 32-char lowercase hex token = an Olist ID (order/customer/product/seller).
# These contain digits we MUST NOT confuse with the answer.
_HEX32_RE = re.compile(r"\b[a-f0-9]{32}\b")


# ----------------------------------------------------------------
# String / number normalization
# ----------------------------------------------------------------
def normalize_answer(s: str) -> str:
    """Lowercase, strip, collapse whitespace, drop surrounding punctuation
    and currency symbols. Also strips citation markers like '[2,5]'."""
    s = s.strip().lower()
    s = re.sub(r"\[[\d,\s]+\]", "", s)        # citation markers
    s = re.sub(r"[\"'`]+", "", s)              # quotes
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip(".!?,;:")
    s = s.strip()
    # Strip a trailing/leading currency
    s = re.sub(r"^(r\$|us\$|\$|brl|usd)\s*", "", s)
    s = re.sub(r"\s*(brl|usd)$", "", s)
    return s


def _strip_accents(s: str) -> str:
    """Fold Unicode accents: 'Camboriú' → 'Camboriu'."""
    # NFD decomposes 'ú' into 'u' + combining acute. Then drop the marks.
    return "".join(
        ch for ch in unicodedata.normalize("NFD", s)
        if unicodedata.category(ch) != "Mn"
    )


def _strip_ids(s: str) -> str:
    """Remove 32-hex-char ID tokens (Olist order/customer/product/seller IDs)
    so they don't pollute downstream number extraction. Also drop the
    Order:/Customer:/PAY:/I:-prefixed citation tokens that retrievers emit."""
    s = _HEX32_RE.sub(" ", s)
    # Strip cite-style tokens like [O:abc123…], [PAY:abc:1], [I:abc:1]
    s = re.sub(r"\[(?:O|C|P|S|R|PAY|I|G):[^\]]+\]", " ", s)
    return s


def normalize_answer(s: str) -> str:
    """Lowercase, strip, collapse whitespace, drop surrounding punctuation
    and currency symbols. Folds Unicode accents and strips 32-hex IDs and
    citation markers so they don't interfere with comparison."""
    s = _strip_accents(s)
    s = s.strip().lower()
    s = re.sub(r"\[[\d,\s]+\]", "", s)        # numeric citation markers [2,5]
    s = _strip_ids(s)                          # hex ID tokens and [O:...] etc.
    s = re.sub(r"[\"'`]+", "", s)              # quotes
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip(".!?,;:")
    s = s.strip()
    # Strip a trailing/leading currency
    s = re.sub(r"^(r\$|us\$|\$|brl|usd)\s*", "", s)
    s = re.sub(r"\s*(brl|usd)$", "", s)
    return s


_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")
_SCI_RE = re.compile(r"-?\d+\.?\d*[eE][+-]?\d+")


def extract_number(text: str) -> float | None:
    """Pull the first plausible number out of a free-form string.
    Handles common formats: 1,234.56  R$179.80  179.80 BRL  -42  3.14e-2.

    Crucially, strips 32-hex Olist IDs and citation tokens BEFORE matching
    so we don't pick up digits from order IDs like '6eb1728e88...'.
    """
    if not text:
        return None
    cleaned = _strip_ids(text)
    cleaned = cleaned.replace("R$", "").replace("$", "")

    # Try scientific notation first — but only if it looks like a real number,
    # not a hex fragment like '88e88'.
    m_sci = _SCI_RE.search(cleaned)
    if m_sci:
        token = m_sci.group(0)
        # Sanity-bound: scientific notation answers in this dataset are
        # never above 1e10 or below 1e-6. Anything else is a false match.
        try:
            v = float(token)
            if 1e-6 <= abs(v) <= 1e10 or v == 0:
                return v
        except ValueError:
            pass

    m = _NUM_RE.search(cleaned)
    if not m:
        return None
    token = m.group(0)
    # Handle thousands-comma vs decimal-comma:
    # "1,234.56" → drop commas; "1234,56" → swap to dot.
    if "." in token and "," in token:
        token = token.replace(",", "")
    elif "," in token and "." not in token:
        # Ambiguous. Heuristic: comma followed by 3 digits → thousands;
        # comma followed by 1-2 digits at end → decimal.
        if re.search(r",\d{3}(\D|$)", token):
            token = token.replace(",", "")
        else:
            token = token.replace(",", ".")
    try:
        return float(token)
    except ValueError:
        return None


# ----------------------------------------------------------------
# Per-answer scorer
# ----------------------------------------------------------------
@dataclass
class Score:
    qid: str
    category: str
    outcome: str          # "correct" | "wrong" | "abstain"
    gold: object          # for downstream debugging
    pred: object


def _is_abstention(pred_text: str) -> bool:
    return ABSTAIN_TOKEN.lower() in pred_text.strip().lower()


def _score_number(gold: float, pred_text: str) -> str:
    pred = extract_number(pred_text)
    if pred is None:
        return "wrong"
    # Integer-style gold: absolute tolerance
    if isinstance(gold, int) or (isinstance(gold, float) and gold.is_integer()):
        return "correct" if abs(pred - gold) <= NUM_ABS_TOL else "wrong"
    # Float: relative tolerance, but never tighter than NUM_ABS_TOL
    denom = max(abs(gold), 1e-9)
    rel = abs(pred - gold) / denom
    abs_diff = abs(pred - gold)
    if rel <= NUM_REL_TOL or abs_diff <= NUM_ABS_TOL:
        return "correct"
    return "wrong"


def _score_string(gold: str, pred_text: str) -> str:
    ng = normalize_answer(gold)
    np_ = normalize_answer(pred_text)
    if not np_:
        return "wrong"
    # Accept either exact match or substring containment in either direction
    # (lets "credit_card" match "Credit card" and "the answer is credit_card").
    if ng == np_ or ng in np_ or np_ in ng:
        return "correct"
    return "wrong"


def _score_list(gold: list, pred_text: str) -> str:
    """List scoring: every element appears (normalized) in the prediction."""
    np_ = normalize_answer(pred_text)
    if not np_:
        return "wrong"
    for item in gold:
        if normalize_answer(str(item)) not in np_:
            return "wrong"
    return "correct"


def score_answer(question: dict, pred_text: str) -> Score:
    """Top-level dispatcher. Returns a Score with outcome."""
    qid, cat, gat = question["qid"], question["category"], question["gold_answer_type"]
    gold = question["gold_answer"]

    # Abstention is its own bucket — neither correct nor wrong.
    if _is_abstention(pred_text):
        return Score(qid, cat, "abstain", gold, pred_text)

    # Rubric-typed questions are handled by the judge, not here.
    if gat == "rubric":
        return Score(qid, cat, "wrong", gold, pred_text)  # caller should skip

    if gat == "number":
        return Score(qid, cat, _score_number(gold, pred_text), gold, pred_text)
    if gat == "string":
        return Score(qid, cat, _score_string(str(gold), pred_text), gold, pred_text)
    if gat == "list":
        return Score(qid, cat, _score_list(list(gold), pred_text), gold, pred_text)

    raise ValueError(f"Unknown gold_answer_type {gat!r} for {qid}")


# ----------------------------------------------------------------
# Aggregation: list of answers + question registry → per-q outcomes
# ----------------------------------------------------------------
def score_run(answers: list[dict], questions: dict[str, dict]) -> list[Score]:
    """One Score per (qid in answers) that's in a closed category."""
    out = []
    for a in answers:
        q = questions.get(a["qid"])
        if q is None or q["category"] not in CLOSED_CATEGORIES:
            continue
        out.append(score_answer(q, a["answer"]))
    return out


def accuracy_table(scores: list[Score]) -> dict:
    """{category: {'n':int, 'correct':int, 'abstain':int, 'acc':float}}"""
    by_cat = defaultdict(lambda: Counter())
    for s in scores:
        by_cat[s.category][s.outcome] += 1
        by_cat[s.category]["n"] += 1
    out = {}
    for cat, c in by_cat.items():
        n = c["n"]
        # Abstention is counted as not-correct for accuracy
        out[cat] = {
            "n": n,
            "correct": c["correct"],
            "wrong": c["wrong"],
            "abstain": c["abstain"],
            "acc": (c["correct"] / n) if n > 0 else 0.0,
        }
    # Overall
    n_total = sum(c["n"] for c in by_cat.values())
    corr_total = sum(c["correct"] for c in by_cat.values())
    out["_overall"] = {
        "n": n_total,
        "correct": corr_total,
        "wrong": sum(c["wrong"] for c in by_cat.values()),
        "abstain": sum(c["abstain"] for c in by_cat.values()),
        "acc": (corr_total / n_total) if n_total > 0 else 0.0,
    }
    return out


# ----------------------------------------------------------------
# Paired bootstrap test
# ----------------------------------------------------------------
def paired_bootstrap(
    a_correct: np.ndarray,
    b_correct: np.ndarray,
    b_iters: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """
    Inputs: two boolean (0/1) arrays of identical length — element i is
    whether system A (resp. B) answered the i-th question correctly.

    Returns {mean_diff, ci_low, ci_high, p_two_sided}.
      mean_diff > 0 means A beats B.
    """
    assert a_correct.shape == b_correct.shape, "vectors must align by qid"
    rng = np.random.default_rng(seed)
    diffs = (a_correct.astype(int) - b_correct.astype(int))
    n = len(diffs)
    if n == 0:
        return {"mean_diff": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p": 1.0}
    observed = diffs.mean()

    # Resample with replacement
    idx = rng.integers(0, n, size=(b_iters, n))
    samples = diffs[idx].mean(axis=1)

    # Percentile CI
    lo_pct = 100 * (alpha / 2)
    hi_pct = 100 * (1 - alpha / 2)
    ci_low = float(np.percentile(samples, lo_pct))
    ci_high = float(np.percentile(samples, hi_pct))

    # Two-sided p-value: shift to null (mean 0) and ask how often
    # |shifted| >= |observed|. Add 1 to numerator+denominator (Davison &
    # Hinkley's small-sample correction).
    shifted = samples - observed
    p = (np.sum(np.abs(shifted) >= abs(observed)) + 1) / (b_iters + 1)

    return {
        "mean_diff": float(observed),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p": float(p),
        "n": int(n),
    }


def bonferroni_alpha(n_systems: int, base_alpha: float = 0.05) -> float:
    """Bonferroni-corrected α for all pairwise comparisons."""
    n_pairs = n_systems * (n_systems - 1) // 2
    return base_alpha / n_pairs


# ----------------------------------------------------------------
# Cohen's kappa — for judge vs. human re-judge sample
# ----------------------------------------------------------------
def cohens_kappa(rater_a: list, rater_b: list) -> float:
    """
    Categorical Cohen's κ for two rater label sequences of equal length.
    Labels can be any hashable — strings work fine.
    """
    assert len(rater_a) == len(rater_b) and len(rater_a) > 0
    n = len(rater_a)
    labels = sorted(set(rater_a) | set(rater_b))

    # Observed agreement
    p_o = sum(1 for a, b in zip(rater_a, rater_b) if a == b) / n

    # Expected (chance) agreement under independence
    ca = Counter(rater_a)
    cb = Counter(rater_b)
    p_e = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    if abs(1 - p_e) < 1e-12:
        return 1.0 if p_o == 1 else 0.0
    return (p_o - p_e) / (1 - p_e)


# ----------------------------------------------------------------
# CLI
# ----------------------------------------------------------------
def _load_jsonl(path: Path) -> list[dict]:
    with path.open() as fp:
        return [json.loads(line) for line in fp]


def _fmt_pct(x: float) -> str:
    return f"{100*x:5.1f}%"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", type=Path,
                    help="Directory containing {system}_answers.jsonl files")
    ap.add_argument("--questions", type=Path,
                    default=Path("./outputs/pilot_questions.jsonl"))
    ap.add_argument("--bootstrap-iters", type=int, default=1000)
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="Base α before Bonferroni correction")
    ap.add_argument("--out-json", type=Path, default=None,
                    help="If set, dump full results to JSON")
    args = ap.parse_args()

    questions = {q["qid"]: q for q in _load_jsonl(args.questions)}
    answer_files = sorted(args.results_dir.glob("*_answers.jsonl"))
    if not answer_files:
        print(f"No *_answers.jsonl files in {args.results_dir}", file=sys.stderr)
        return 1

    # Score every run
    per_system_scores: dict[str, list[Score]] = {}
    for fp in answer_files:
        system = fp.stem.replace("_answers", "")
        answers = _load_jsonl(fp)
        per_system_scores[system] = score_run(answers, questions)

    # --- Print accuracy table ---
    systems = sorted(per_system_scores)
    cats = ["factual", "aggregation", "_overall"]
    print(f"\nAccuracy on closed categories  (n_systems = {len(systems)})")
    print(f"  {'system':18s} " + " ".join(f"{c:>12s}" for c in cats))
    tables = {s: accuracy_table(per_system_scores[s]) for s in systems}
    for s in systems:
        cells = []
        for c in cats:
            t = tables[s].get(c)
            cells.append(f"{_fmt_pct(t['acc']) if t else '   n/a':>12s}")
        print(f"  {s:18s} " + " ".join(cells))

    # --- Pairwise bootstrap ---
    alpha_corr = bonferroni_alpha(len(systems), args.alpha)
    print(f"\nPairwise bootstrap (B={args.bootstrap_iters}, "
          f"α={args.alpha:g}, Bonferroni → α'={alpha_corr:.4g})")
    print(f"  Each row: system A vs system B on the union of qids both answered.")
    print(f"  '*' = significant after correction.\n")
    print(f"  {'A':18s} {'B':18s} "
          f"{'Δ(A−B)':>10s} {'95% CI':>20s} {'p':>8s}")

    pair_results = {}
    for a, b in combinations(systems, 2):
        # Align by qid
        sa = {s.qid: s for s in per_system_scores[a]}
        sb = {s.qid: s for s in per_system_scores[b]}
        common = sorted(set(sa) & set(sb))
        if not common:
            continue
        a_vec = np.array([sa[q].outcome == "correct" for q in common])
        b_vec = np.array([sb[q].outcome == "correct" for q in common])
        r = paired_bootstrap(a_vec, b_vec,
                             b_iters=args.bootstrap_iters,
                             alpha=alpha_corr,
                             seed=42)
        sig = "*" if r["p"] < alpha_corr else " "
        ci_str = f"[{r['ci_low']:+.3f},{r['ci_high']:+.3f}]"
        print(f"  {a:18s} {b:18s} "
              f"{r['mean_diff']:+10.3f} {ci_str:>20s} "
              f"{r['p']:>8.4g}{sig}")
        pair_results[f"{a}__vs__{b}"] = r

    if args.out_json:
        payload = {
            "accuracy_tables": tables,
            "pairwise_bootstrap": pair_results,
            "bonferroni_alpha": alpha_corr,
            "n_systems": len(systems),
        }
        args.out_json.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote results JSON to {args.out_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
