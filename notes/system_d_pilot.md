# System D (community-temporal) — pilot findings

**Date:** 2026-06-02
**Run:** `pilot_run_systemD.py --real --extract-prefer rule` (30 pilot Qs)
**Scored with:** the corrected auto-scorer (see §6), re-scored via
`rescore_pilot.py --system all` so Vanilla / C / D are directly comparable.

---

## 1. One-paragraph summary

System D retrieves over the 38 time-bucket summaries (year→quarter→month)
instead of the entity-community summaries, and feeds the **exact, pandas-
computed `stats_block`** of the matched bucket(s) into the generator. On
the 30-question pilot it scores **18/30 (60%)** with **zero spurious time
anchors** — a 3× improvement over System C (6/30) and 18× over Vanilla v2
(1/30). The win is concentrated exactly where predicted: un-filtered
population aggregates over bucket-aligned windows, where the answer equals
the gold by construction because the bucket stats are computed from the
same CSV path and the same monotonic filter as the gold. The 12 misses are
structural and explainable (entity sub-filters, sub-bucket windows,
long-tail per-category drift), not noise — each maps onto a concrete next
step.

---

## 2. Headline table (re-scored, identical scorer for all three)

| Category   | Vanilla v2 | System C | **System D** |
|------------|:----------:|:--------:|:------------:|
| trend (10) |   1/10     |  2/10    |  **10/10**   |
| drift (10) |   0/10     |  0/10    |  **2/10**    |
| latency(10)|   0/10     |  4/10    |  **6/10**    |
| **total**  | **1/30**   | **6/30** | **18/30**    |
| spurious anchors | 3    | 3        | **0**        |

Extraction (rule path): 17/17 questions with a gold `time_range` had an
extracted window overlapping gold (100%); all 30 selections were
`structured` (0 cosine fallbacks).

---

## 3. Per-category breakdown

### 3.1 Trend — 10/10 (the strongest result)

Every trend question is an un-filtered aggregate over a bucket-aligned
window, so D reads it straight from the bucket stats:

- **aggregation_over_time (trend_01–06):** "per quarter/month of 2017"
  → the 4 quarter (or 12 month) buckets, each carrying the exact
  mean/sum. trend_01 reproduces the gold per-quarter delivery times
  (12.98 / 12.48 / 11.53 / 14.38) exactly.
- **comparison_over_time (trend_07–10):** two windows → two buckets;
  the model reads both and compares. trend_10 (half-year) tiles cleanly
  to Q1+Q2 / Q3+Q4.

Vanilla gets 1/10 here only because of sample-mean miscalibration (it
averages its 8 cosine-top chunks); System C gets 2/10 (it nails the two
two-quarter comparisons trend_09/10 from its stratified seed sample but
misses the per-period dicts). D's bucket aggregates sidestep both.

### 3.2 Latency — 6/10

- **Hits (02, 03, 04, 05, 08):** un-filtered lag aggregates. latency_02/03/04
  are the year-2017 approval / carrier-delivery / approval-to-carrier lags
  → year bucket. latency_05 is the Q4-2017 purchase-to-delivery lag →
  Q4 bucket. latency_08 is a Q1-vs-Q4 approval-lag comparison.
- **Misses (01, 06, 09, 10):** all structural — see §4.

**The showcase result — latency_05.** Gold 14.38 days. System C *misses*
it (its 40-order stratified sample cannot estimate the mean of a window
with a long Black-Friday tail — see `system_c_pilot.md`). System D answers
exactly 14.38, because the Q4-2017 bucket carries the exact population
mean. Same question, same window: C fails by *sampling*, D succeeds by
*pre-computed aggregate*. This is the cleanest before/after for the paper's
core claim.

### 3.3 Drift — 2/10 (the weakest, and the honest limitation)

- **Hits (08, 09):** the drifting quantity is one D already carries as an
  exact top-k stat — top-5 customer states (drift_08), top-3 payment-method
  share (drift_09).
- **Misses (01–07, 10):** per-category review/volume/price drift over the
  long tail, set-difference of full category lists (drift_06), top-5
  sellers by revenue (drift_07). A bucket summary carries only top-5 by
  volume / top-5 by review / top-3 payment / top-5 states — it cannot
  answer drift over an entity outside those top-k. On drift_01/02/05/07,
  D **abstains** (`INSUFFICIENT EVIDENCE`) rather than guessing — which is
  why drift contributes 0 spurious anchors.

---

## 4. Failure-mode taxonomy (the 12 misses) → next steps

| Class | Questions | Why D cannot | Next step |
|-------|-----------|--------------|-----------|
| **Entity sub-filter** | latency_01 (SP), latency_06 (SP vs RJ) | bucket aggregate is all-population, not filtered by state | **System E** (C supplies the entity filter, D the time aggregate) |
| **Sub-bucket window** | latency_09 (30-day pre-BF), latency_10 (World Cup range) | window is finer than the monthly bucket floor | windowed *exact* aggregation via the interval index (C/E), or finer buckets |
| **Long-tail per-category drift** | drift_01–07, drift_10 | top-k summary doesn't carry the full per-entity table | richer bucket stats (full per-category table) OR System E |

Note that latency_09/10 are missed by **all three** systems: System C has
the interval index but still miscalibrates the window mean from its
sample; this needs an exact windowed aggregate, a mechanism neither C's
sampling nor D's fixed buckets provide. Good future-work hook.

---

## 5. Complementarity — what this means for System E

From the per-question matrix:

- **C∪D = 19/30** (D's 18 plus latency_01, which only C solves).
- **C∩D = 5** (latency_03/04/07, trend_09/10).
- **D-only = 13**, **C-only = 1** (latency_01).

So on this pilot, **System C contributes exactly one question beyond D**
(the SP-filtered latency). That is sobering but important: the *raw* count
gain of a hybrid over D-alone is +1 here. The *class* it represents —
entity-subfiltered aggregates — is real and will grow at 90 questions
(more SP/RJ/category-filtered aggregates). The honest framing for the
paper: System E's value is the **entity-subfilter class**, not a large
pilot-count jump. Oracle hybrid ceiling on the pilot = 19/30; E should be
evaluated against that, and against whether it *preserves* D's 0-spurious
faithfulness while picking up latency_01.

---

## 6. Scorer corrections applied (validated, true-positive only)

The first auto-score gave D 12/30; six of those were scorer false
negatives (the model's numbers were already correct). Three surgical fixes,
each validated against the hand-checked cases plus adversarial controls
(`8/8`), lift D to its true 18/30 — and leave Vanilla (1) and C (6)
essentially unchanged, confirming the fixes are not blanket loosening:

1. **Bare period-key match** — when a gold dict's keys are all in one
   year, accept the year-dropped token ("Q1" for "2017-Q1", "January"
   for "2017-01"). Fixes trend_01–05. Multi-year gold still requires the
   year (a bare "Q2" across 2017/2018 stays ambiguous → no match).
2. **Underscore/space key folding** ("credit_card" ↔ "Credit Card").
   Fixes drift_09.
3. **Half-year gold covers its two quarters** — a model fed quarter
   buckets for a half-year question names Q1/Q2 for H1; not spurious.
   Removes the false spurious anchors on trend_10 (+4) and latency_07 (+2),
   taking D's spurious count to 0.

These are exactly two of the "scorer corrections before the headline
experiment" flagged in status.md §5; they are now validated on the pilot.
(The third deferred correction — tighter review-score tolerance on the
1–5 scale — did not bite any D question here; revisit before scale-up.)

---

## 7. Recommended next steps

1. **Build System E (`temporal_hybrid.py`)** — score-fuse C and D. Target:
   ≥19/30 (the C∪D oracle), and crucially keep D's 0-spurious property.
   Primary test: does E pick up latency_01 (entity-subfilter)?
2. **Decide bucket-stat richness for drift.** Either accept that
   community-temporal answers only top-k drift (and lean on E / future
   work for long-tail per-category drift), or carry a fuller per-category
   table in the bucket stats. This is a paper-scoping decision, not a bug.
3. **Scale 30 → 90.** The entity-subfilter and long-tail-drift classes
   should grow; re-check whether E's marginal value over D rises with N.
4. **H2 reframe holds.** D's 0 spurious vs Vanilla/C's 3 supports the
   faithfulness story; combined with the aggregation-miscalibration finding
   (latency_05), the headline is "time-blind RAG mis-estimates population
   aggregates from biased samples; bucketed temporal RAG reads exact
   aggregates and cannot hallucinate the anchor."
