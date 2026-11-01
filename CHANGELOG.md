# Changelog

All releases are archived on Zenodo. Earlier versions stay available, so
numbers published against them remain reproducible. Question IDs (`qid`)
are stable across versions.

## v1.0.0 (2026-11)

First public release, accompanying the ECIR 2027 resource-paper submission.

- 90 temporal questions (trend, drift, latency) with gold answers and gold
  time anchors, the 150 Olist-QA questions (unchanged), 10 event anchors.
- Temporal layer: interval-typed edges and a 38-node time hierarchy with
  per-bucket statistics and summaries; temporal retrievers C, D, E.
- Time-blind baselines A1 and B1 to B3 and the evaluation harness.
- Answers and scored records of 7 systems under 4 generators, plus the
  granularity sweep.
