# Lab Notes — Vanilla v2 Aggregation Miscalibration

**Date:** 2026-05-27
**Context:** First-pass pilot run of Vanilla v2 on 30 temporal
pilot questions (10 trend + 10 drift + 10 latency), all
closed-form with SQL-validated gold answers.

## Headline finding

Vanilla v2 fails on temporal aggregation questions in two distinct
ways, neither of which matches the original H2 hypothesis
("hallucinates time anchors"):

1. **Correct abstention** (20/30, 67%) — `INSUFFICIENT EVIDENCE`.
   Top-8 paragraphs out of 99,441 cannot answer "average X across
   all 2017 orders"; the model recognises this and abstains.
2. **Sample-as-population miscalibration** (9/30, 30%) — the model
   computes an aggregate from the 8 retrieved paragraphs and
   reports it as the population aggregate, with no uncertainty
   marker.
3. **Genuine match: 0/30 (0%).** The two apparent matches
   (latency_02, latency_08) are scoring false positives caused by
   loose absolute tolerance on sub-day latencies; the model says
   "0 days" because all its retrieved samples were same-day, and
   the gold value happens to be 0.33-0.44, which fell inside the
   old `abs_tol=0.5` band. Scoring tolerance fixed; both flip to
   no-match on re-score.

## Three concrete examples (cherry-picked but representative)

### latency_01 — São Paulo 2017 purchase-to-delivery lag
- Gold: 9.34 days (computed over ~30k SP orders in 2017)
- Vanilla v2 retrieves 8 paragraphs, finds 3 with SP customers
  (24, 27, 8 days), reports **19.67 days** — over 2x the gold,
  reported as a final answer with citation markers.

### latency_05 — Q4 2017 purchase-to-delivery lag
- Gold: 14.38 days (computed over ~21k Q4 2017 orders)
- Vanilla v2 retrieves 8 paragraphs, finds 3 valid Q4 ones with
  lags (66, 22, 42 days), reports **43.33 days** — 3x gold.
- The 66-day outlier (purchase 2017-12-23, delivery 2018-02-27)
  dragged the sample mean badly, but the model has no way to
  know this paragraph is an outlier.

### latency_09 — Black Friday 2017 before/after delivery lag
- Gold: 13.16 (before) vs 16.28 (after) — a meaningful, real
  ~24% slowdown after Black Friday.
- Vanilla v2 retrieves 8 paragraphs, finds **one** pre-BF order
  (35 days lag) and 4 post-BF orders (avg 36.75 days).
- Reports "35 days (before), 36.75 days (after)" — both ~2x gold,
  and crucially the *direction is correct* (after > before) but
  the *magnitude is dramatically wrong*.

## Why this matters more than the original H2

The original paper-headline hypothesis was:

> *"Time-blind systems will hallucinate time anchors, with
> hallucination rate >= 25%."*

The pilot shows the failure is **richer and more interesting**:

> *"Time-blind RAG abstains correctly on questions whose gold
> answer requires global aggregation, but on the questions where
> it does answer, it commits a structural error specific to the
> top-k paradigm: treating a tiny retrieval sample as if it were
> a representative population."*

This is paper-worthy because:
- It is a structural feature of top-k RAG, not a prompt-engineering
  artefact. No amount of "think step by step" fixes it.
- Increasing top-k to 50 or 100 does not fix it — you still have
  100 out of 99,441 samples. You would need top-k ~ N to get the
  right answer, which defeats the purpose of retrieval.
- The temporal Graph-RAG variants (Systems C/D/E) are designed
  precisely to short-circuit this: System D retrieves a
  pre-computed time-bucket *summary* over thousands of orders, not
  individual order paragraphs. The bi-level hierarchy's whole
  point is to make the aggregate retrievable as a single unit.

## Implications for paper hypotheses

**H2 (original):** "time-blind systems hallucinate time anchors in
>= 25% of answers" — likely needs to be reframed. Pilot shows
hallucination rate is closer to 10% (3 spurious anchors across 30
questions, 2 actually wrong).

**H2 (proposed reframing):** "time-blind RAG returns sample-mean
estimates that diverge from the population mean by a factor >= 2x
on >= 30% of aggregation questions, while temporal community-RAG
returns population-level summaries that match the gold value within
20% on >= 80%". This is a stronger and more testable claim once
Systems D/E exist.

**H1:** "temporal Graph-RAG beats time-blind on trend and drift by
>= 20 pp judge-preference" — still standing. The pilot shows
time-blind on trend/drift is at 0% match rate; the only way to do
worse would be to actively contradict the gold. Any non-zero
match-rate from Systems C/D/E will pass H1 trivially.

**New hypothesis worth pre-registering (call it H7):**
"Aggregation miscalibration rate" as a metric. Defined as:

> *Fraction of non-abstention answers on aggregation-type questions
> where the model's numeric output differs from the gold value by
> a factor > 2x in either direction.*

For Vanilla v2 on the pilot: 9/9 non-abstention numeric answers
were >2x off → miscalibration rate 100%.

## Why this changes the paper structure

The headline metric of the paper was going to be *temporal
faithfulness* (anchor hallucination rate). The pilot shows this is
the wrong dial — anchor hallucination is rare; the actual failure
is sample-mean miscalibration.

Proposed reorganisation, deferred until after the full headline
experiments:

- §3.6 (current "temporal faithfulness") gets demoted to a sub-metric.
- New headline metric: **aggregation calibration error** — for each
  numeric gold answer, the relative error of the model's point
  estimate, conditioned on non-abstention.
- Hypothesis structure:
  - H1: temporal Graph-RAG produces population-level answers where
    time-blind RAG abstains (abstention-rate gap).
  - H2 (reframed): temporal Graph-RAG produces calibrated estimates
    where time-blind RAG produces sample-mean estimates that are
    miscalibrated by >2x (calibration gap).
  - H3 (was: anchor hallucination): becomes a robustness check,
    not a headline.

This is a strictly better paper — abstention + miscalibration is
a richer story than "halluciniert Zeitanker".

## What the pilot does NOT yet tell us

- How Systems C/D/E will actually behave on these same 30
  questions. They might also abstain. They might also miscalibrate.
  Without running them, the reframing above is conjectural.
- Whether top-k=8 is the worst case or whether larger k changes
  the qualitative picture. Theoretically it shouldn't — the
  retrieved 100 paragraphs out of 99,441 is the same kind of
  sample as 8 out of 99,441 — but worth confirming on one
  question post-hoc.
- Whether the failure mode is the same on open-form (judge-graded)
  questions or only on closed-form (auto-scored) ones. Pilot is
  all closed.

## Disposition

- Keep this note as the canonical record of the pilot reality
  check.
- Do NOT update the paper yet. Hypothesis reframing happens after
  Systems C/D/E results are in.
- The two scoring false positives (latency_02, latency_08) are
  fixed in `near()` — see commit/diff for `pilot_run_vanilla_v2.py`.
- Use this note as the seed for the §6.7 failure-mode analysis
  paragraph in the eventual paper draft.
