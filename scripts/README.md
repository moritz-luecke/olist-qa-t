# scripts/

Analysis and data-statistics scripts. Run them from the repository root.
None of them calls an API.

| Script | Needs | Purpose |
|---|---|---|
| `analyze_results.py` | stdlib | Recomputes the result tables from `outputs/results/`: check of the time-blind systems on the 150 Olist-QA questions, EM per category for all seven systems on the 90 temporal questions, latency MAE, temporal-faithfulness diagnostics (answer rate, coverage, out-of-window assertions), granularity sweep. |
| `significance.py` | stdlib | Paired bootstrap (B = 1000, seed 42) with Bonferroni correction over the 21 system pairs, exact McNemar tests, two-proportion z-tests on answer rates. |
| `analyze_generators.py` | stdlib | Cross-generator tables (gpt-4o-mini and the three open-weight generators). |
| `significance_generators.py` | stdlib | Per-generator tests of D vs. A1. |
| `review_stats.py` | stdlib | Accuracy conditional on answering, exact McNemar on answering vs. abstaining, per-category tests of H1. Usage: `python scripts/review_stats.py outputs/results outputs/temporal_questions_90.jsonl` |
| `rescore_scored.py` | harness | Re-scores every `*_answers.jsonl` into `*_scored.jsonl` with the current scorer (deterministic). |
| `data_stats.py` | pandas, Olist CSVs | Timestamp coverage, interval-edge counts, and sanity checks on the raw Olist tables. |
