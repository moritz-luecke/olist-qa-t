# Headline experiment — 7 systems × 90 temporal questions

Run: `run_headline.py --all-temporal --real` (gpt-4o-mini gen,
text-embedding-3-small, gpt-4o judge). Scoring: fixed `score_one`
(closed-form). Faithfulness: `temporal_faithfulness.py --real` then
`--reclassify` after the year-overlap fix (below).

## Accuracy — match / 90 (closed-form)

| system               | trend | drift | latency | TOTAL |
|----------------------|------:|------:|--------:|------:|
| community_temporal D | 30/30 |  6/30 |  18/30  | **54 (60%)** |
| hybrid_temporal  E   | 30/30 |  6/30 |  18/30  | **54 (60%)** |
| entity_temporal  C   |  7/30 |  0/30 |  12/30  | 19 (21%) |
| vanilla_v2 (A1)      |  4/30 |  0/30 |   6/30  | 10 (11%) |
| entity     (B1)      |  3/30 |  0/30 |   2/30  |  5 ( 6%) |
| hybrid     (B3)      |  0/30 |  0/30 |   1/30  |  1 ( 1%) |
| community  (B2)      |  0/30 |  0/30 |   0/30  |  0 ( 0%) |

**RQ1 (strongly supported).** Best temporal (D/E, 60%) beats best
time-blind (vanilla, 11%) by **+49 pp**. H1 (≥20 pp) more than
doubled. Scales cleanly from the 30-pilot (D 60%, E 63% there).

## One mechanism explains every cell

The time-bucket summaries carry **overall per-period aggregates +
payment-mix + (some) state breakdowns + top-k-categories-by-count**,
but **no full per-category time series**. Hence:

- **trend** (period aggregates): D/E perfect, 30/30.
- **latency** (per-period lag aggregates): D/E 18/30; state-filtered
  subset partial (see routing).
- **topk state/payment** drift: D/E get all four (drift_23/24/27/28).
- **category topk + category_drift**: D/E fail. Set-overlap (named
  gold categories, ignoring magnitude) is only **D/E 8%, others 0%** —
  even *identification* fails. On drift_11 D names the right top-2 but
  asserts "fell to 0 in Q2" because the category is absent from the
  Q2 top-k list. Genuine capability limit, not a scoring artifact.

## RQ2 — routing is a wash at scale (honest)

E differs from D on exactly the 4 state-filtered latency questions
(latency_01/15/16/19): routing→C wins 2 (01 SP, 15 RJ — C correct,
D wrong) and loses 2 (16 MG, 19 SP-2018 — C wrong, D correct) → net
0, D = E = 54. The pilot's "+1 for routing" does not survive scaling:
entity-temporal retrieval is unreliable on state-level aggregates that
D estimates from bucket stats (SP ≈ overall mean by dominance). E's
value is robustness (never far below D), not a net gain over D. The
`route` field is not captured in run_headline's scored record (only
the pilot systemE runner wrote it); the D≠E diff proves routing fires.

## B2 (time-blind community) = 0/90 — valid, not a bug

Returns "INSUFFICIENT EVIDENCE" on every temporal question: time-blind
community summaries have no time axis, so it honestly abstains. Clean
control (max honesty, zero accuracy).

## Faithfulness — 90 questions

A classifier bug inflated hallucination to ~6–9% on the first run:
the bare year **"2018"** was flagged because `period_to_months` expands
it to Jan–Dec while the data's 2018 ends in September (2018-Q4
excluded). EVERY flagged item across all systems was "2018". Fix: a
period is hallucinated only if **disjoint** from the allowed window (no
month overlaps), so a coarse year mention that overlaps is the
question's own framing, not an unsupported claim; genuinely
out-of-window periods (2019, wrong quarters) are still caught.
Recompute offline with `--reclassify` (no API).

| system | hallucination | coverage |
|--------|--------------:|---------:|
| vanilla_v2 | 0.0% | 80.0% |
| entity B1  | 1.1% | 75.6% |
| community B2 | 0.0% | 64.4% |
| hybrid B3  | 0.0% | 72.2% |
| entity_temporal C | 0.0% | 91.1% |
| community_temporal D | 0.0% | 95.6% |
| hybrid_temporal E | 0.0% | 95.6% |

**Confirms the H2 reframe at scale.** Time-anchor hallucination is ~0
for every system — NOT a differentiator. The faithfulness signal is
**coverage**: temporal retrieval 91–96% vs time-blind 64–80%. Pairs
with the accuracy gap as the two-pronged headline.

## Resolved — category_drift handling

**(1) DONE:** "UNKNOWN" (untranslated products) is excluded at source in
`build_drift_ext` (`oi`/`rc` filtered), so no category gold contains it.
Affected only drift_11/15/21 (UNKNOWN was the #1 price-riser in 15/21);
the pilot drift questions had no UNKNOWN, so the pilot builder is
untouched. **This does not change any reported match number** — those 3
were category_drift, which is 0/match for all 7 systems, and they stay 0;
the change is benchmark-release hygiene only.

**(2a) chosen for the main paper:** category_drift is reported as a
mechanistic limitation (bucket summaries carry no per-category time
series). **(2b) deferred** to a transparent post-hoc ablation in revision
(generic per-category counts, never merged into D's headline number).
