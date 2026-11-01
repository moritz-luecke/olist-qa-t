"""
============================================================
Phase 4 — Paper figure generation
============================================================

Produces the two figures referenced from 06_results.tex:
  * fig:per-hop   — closed-question accuracy by min_hops bin
  * fig:strategy-heatmap — strategies × categories matrix

All numbers come from existing JSONLs in outputs/results/; no API
calls. Run after the headline benchmark and the judge passes have
populated those files.

Usage
-----
    cd phase4_retrieval
    python make_figures.py
    # → writes ../outputs/results/fig_per_hop.{pdf,png}
    #          ../outputs/results/fig_strategy_heatmap.{pdf,png}

Camera-ready: re-run after RQ3 (System C) results are in to refresh
both figures with the auto-extracted KG numbers.
============================================================
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import score_answer, CLOSED_CATEGORIES


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "outputs" / "results"
QUESTIONS_PATH = PROJECT_ROOT / "outputs" / "pilot_questions.jsonl"

SYSTEMS = ["vanilla_rag", "entity", "community", "hybrid"]

# Display names + a consistent palette (colour-blind friendly).
DISPLAY = {
    "vanilla_rag": "Vanilla RAG",
    "entity":      "Graph-RAG (Entity)",
    "community":   "Graph-RAG (Comm.)",
    "hybrid":      "Graph-RAG (Hybrid)",
}
COLORS = {
    "vanilla_rag": "#4B6584",   # slate blue
    "entity":      "#2E7D32",   # forest green
    "community":   "#C62828",   # red
    "hybrid":      "#F57C00",   # orange
}
MARKERS = {
    "vanilla_rag": "o",
    "entity":      "s",
    "community":   "^",
    "hybrid":      "D",
}


# ----------------------------------------------------------------
# Data loaders
# ----------------------------------------------------------------
def load_questions() -> dict[str, dict]:
    return {q["qid"]: q for q in
            (json.loads(l) for l in QUESTIONS_PATH.open())}


def load_answers(system: str) -> dict[str, dict]:
    p = RESULTS / f"{system}_answers.jsonl"
    return {a["qid"]: a for a in
            (json.loads(l) for l in p.open())}


# ----------------------------------------------------------------
# Figure 1 — per-hop accuracy (closed questions only)
# ----------------------------------------------------------------
def hop_bin(min_hops) -> str | None:
    """Map a question's min_hops to a display bin label. Returns None
    for 'all' (global questions) — those are excluded from this figure."""
    if isinstance(min_hops, str):
        return None      # 'all' → not a closed question anyway
    if min_hops == 0:
        return "0"
    if min_hops == 1:
        return "1"
    if min_hops in (2, 3):
        return "2–3"
    return "4+"          # 4, 5, ...


def make_per_hop_figure(questions: dict, all_answers: dict[str, dict]) -> None:
    """X-axis: hop bin, Y-axis: closed-question accuracy."""
    bins = ["0", "1", "2–3", "4+"]

    # bin_data[system][bin] = (n_correct, n_total)
    bin_data: dict[str, dict[str, tuple[int, int]]] = {
        s: {b: [0, 0] for b in bins} for s in SYSTEMS
    }

    for qid, q in questions.items():
        if q["category"] not in CLOSED_CATEGORIES:
            continue
        b = hop_bin(q["min_hops"])
        if b is None:
            continue
        for s in SYSTEMS:
            a = all_answers[s].get(qid)
            if a is None:
                continue
            outcome = score_answer(q, a["answer"]).outcome
            bin_data[s][b][1] += 1
            if outcome == "correct":
                bin_data[s][b][0] += 1

    # Plot
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    x = np.arange(len(bins))

    # Bin sizes (totals — identical across systems for closed questions)
    bin_n = [bin_data[SYSTEMS[0]][b][1] for b in bins]

    for s in SYSTEMS:
        accs = [(bin_data[s][b][0] / bin_data[s][b][1]) * 100
                if bin_data[s][b][1] else 0
                for b in bins]
        ax.plot(x, accs, marker=MARKERS[s], color=COLORS[s],
                linewidth=1.8, markersize=8, label=DISPLAY[s])

    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\n(n={n})" for b, n in zip(bins, bin_n)])
    ax.set_xlabel("Minimum hop count to answer")
    ax.set_ylabel("Closed-question accuracy (%)")
    ax.set_ylim(-3, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, fontsize=9, ncol=2)
    ax.set_title("Accuracy by hop depth on closed questions",
                 fontsize=11, pad=8)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        out = RESULTS / f"fig_per_hop.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"  wrote {out}")
    plt.close(fig)


# ----------------------------------------------------------------
# Figure 2 — strategy × category heatmap
# ----------------------------------------------------------------
def load_judge_pairs() -> dict[tuple[str, str], list[dict]]:
    """Returns { (sysA, sysB): [verdicts] } keyed by the on-disk filename
    order (the A/B convention is whoever appeared first in the filename)."""
    pairs = {}
    for fp in sorted(RESULTS.glob("judge_*_answers_vs_*_answers.jsonl")):
        stem = fp.stem.removeprefix("judge_")
        left, right = stem.split("_vs_")
        a = left.removesuffix("_answers")
        b = right.removesuffix("_answers")
        pairs[(a, b)] = [json.loads(l) for l in fp.open()]
    return pairs


def win_rate_vs_vanilla(system: str, questions: dict,
                        pairs: dict, category: str) -> float:
    """% wins of `system` against vanilla_rag on `category` open questions.
    Returns NaN if the pair is not on disk. TIEs count as 0.5 (standard
    pairwise-preference convention)."""
    pair = pairs.get((system, "vanilla_rag")) or pairs.get(("vanilla_rag", system))
    if pair is None:
        return float("nan")

    # Are we A or B? Find the orientation.
    a_is_us = (system, "vanilla_rag") in pairs

    wins, ties, losses = 0, 0, 0
    for v in pair:
        if questions[v["qid"]]["category"] != category:
            continue
        w = v["winner_system"]
        if w == "TIE":
            ties += 1
        elif w == system:
            wins += 1
        else:
            losses += 1
    n = wins + ties + losses
    if n == 0:
        return float("nan")
    return 100 * (wins + 0.5 * ties) / n   # win-rate with TIE=0.5


def closed_accuracy(system: str, questions: dict,
                    all_answers: dict, category: str) -> float:
    correct, total = 0, 0
    for qid, q in questions.items():
        if q["category"] != category:
            continue
        a = all_answers[system].get(qid)
        if a is None:
            continue
        total += 1
        if score_answer(q, a["answer"]).outcome == "correct":
            correct += 1
    return 100 * correct / total if total else float("nan")


def make_strategy_heatmap(questions: dict, all_answers: dict,
                          pairs: dict) -> None:
    cats_closed = ["factual", "aggregation"]
    cats_open = ["comparison", "root_cause", "global"]
    cats = cats_closed + cats_open

    rows = SYSTEMS  # 4 rows
    matrix = np.full((len(rows), len(cats)), np.nan)

    for i, s in enumerate(rows):
        for j, c in enumerate(cats):
            if c in cats_closed:
                matrix[i, j] = closed_accuracy(s, questions, all_answers, c)
            else:
                if s == "vanilla_rag":
                    matrix[i, j] = 50.0     # vanilla vs vanilla = neutral
                else:
                    matrix[i, j] = win_rate_vs_vanilla(s, questions, pairs, c)

    # Plot
    fig, ax = plt.subplots(figsize=(8.4, 3.6))

    # Two-tone colormap: closed columns in blue, open in orange
    # To do this cleanly we use one cmap and rely on annotation +
    # column labels to convey what the cells mean.
    cmap = plt.cm.RdYlGn
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([
        "Factual\n(acc %)", "Aggreg.\n(acc %)",
        "Comparison\n(vs. V, %)", "Root-cause\n(vs. V, %)",
        "Global\n(vs. V, %)",
    ], fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([DISPLAY[s] for s in rows], fontsize=10)

    # Visual divider between closed and open columns
    ax.axvline(x=1.5, color="white", linewidth=2)

    # Annotate each cell with its value
    for i in range(len(rows)):
        for j in range(len(cats)):
            v = matrix[i, j]
            if np.isnan(v):
                txt = "—"
                color = "black"
            else:
                txt = f"{v:.0f}"
                color = "black" if 25 < v < 75 else "white"
            # Slight italic for vanilla-vs-vanilla cells (neutral)
            style = "italic" if (rows[i] == "vanilla_rag"
                                 and cats[j] in cats_open) else "normal"
            ax.text(j, i, txt, ha="center", va="center",
                    color=color, fontsize=11, fontstyle=style)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Score (closed: accuracy; open: win-rate vs. Vanilla)",
                   fontsize=9)

    ax.set_title("Per-strategy, per-category performance",
                 fontsize=11, pad=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = RESULTS / f"fig_strategy_heatmap.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"  wrote {out}")
    plt.close(fig)


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def main() -> int:
    if not QUESTIONS_PATH.exists():
        print(f"ERROR: {QUESTIONS_PATH} missing.", file=sys.stderr)
        return 1
    questions = load_questions()

    # Load all systems' answers
    all_answers: dict[str, dict] = {}
    for s in SYSTEMS:
        p = RESULTS / f"{s}_answers.jsonl"
        if not p.exists():
            print(f"ERROR: {p} missing. Run the benchmark first.",
                  file=sys.stderr)
            return 1
        all_answers[s] = load_answers(s)

    # Judge pairs (heatmap only needs the ones vs vanilla)
    pairs = load_judge_pairs()
    if not pairs:
        print("WARNING: no judge_*.jsonl files found; "
              "heatmap open-column cells will be NaN.")

    print("Generating per-hop accuracy figure (fig_per_hop) ...")
    make_per_hop_figure(questions, all_answers)

    print("\nGenerating strategy×category heatmap "
          "(fig_strategy_heatmap) ...")
    make_strategy_heatmap(questions, all_answers, pairs)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
