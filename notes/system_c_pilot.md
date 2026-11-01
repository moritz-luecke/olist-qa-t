# Lab Notes — System C (Entity-Temporal) Pilot

**Date:** 2026-05-28
**Context:** First fair pilot of System C on the 30 temporal pilot
questions, after fixing the timeline-serialization bug (see §1).

## 0. Headline comparison (30 pilot questions, auto-scored)

| | Vanilla v2 | System C (time-blind serialization) | System C (fair) |
|---|---|---|---|
| Match total      | 0/30 | 2/30 | 6/30 (5/30 substantive) |
| Latency          | 0%   | 10%  | 40% |
| Trend            | 0%   | 10%  | 20% |
| Drift            | 0%   | 0%   | 0%  |
| Extraction acc.  | n/a  | 100% (17/17) | 100% (17/17) |

The jump from 0/30 to ~5/30 substantive is a clear structural
gain, but n=30 is small — treat as directional, not significant.

## 1. The timeline-serialization bug (fixed)

The first System C run scored 2/30 and produced answers like
"Order placed 2017-11-30, delivered 2017-11-30, 0 days lag" for
EVERY order. Root cause: System C inherited the order-node
`description` from the prior paper's `describe_order()`, which
emits only the purchase date. So the LLM saw one date per order
and inferred same-day delivery.

This is the same information-access confound the Vanilla v2
chunks-v2 fix addressed. We fixed it symmetrically: System C now
enriches every Order node in the subgraph with a 7-timestamp
timeline string (purchase / approved / carrier / delivered /
estimated / + linked review creation date), mirroring
`serialize_order_v2()`. After the fix, latency match rate rose
10% → 40%, confirming the time-blind serialization was the
handicap, not the retrieval.

**Fairness consequence:** Vanilla v2 and System C now have equal
temporal information access. Any remaining difference is
attributable to retrieval structure (time-stratified subgraph vs
cosine-top-k chunks), not to which timestamps each system can see.

## 2. The real finding: narrow vs broad value distributions

My earlier framing ("System C solves latency, not aggregation")
was imprecise. The pilot shows a sharper distinction.

System C HITS latency questions whose underlying value
distribution is NARROW:
- latency_03 (carrier→delivery, gold 9.59): hit. Carrier-to-
  delivery is 1–10 days for almost all orders; a ~40-order sample
  lands near the population mean.
- latency_04 (approval→carrier, gold 3.02): hit. Even tighter
  distribution (0–3 days typical).

System C MISSES latency questions whose distribution is BROAD or
right-skewed:
- latency_05 (purchase→delivery in Q4 2017, gold 14.38): miss.
  Peak-season delivery has a long right tail (Black-Friday orders
  up to 34 days in the sample); the sample mean is dragged around
  and the 40-order slice does not reliably hit 14.38.

So the operative variable is **sample-mean estimability**, which
depends on the variance/skew of the metric, not on the question
category label. A narrow-variance metric is estimable from a small
time-stratified sample; a broad/skewed one is not. This is a
cleaner and more defensible claim than "latency vs aggregation".

Drift questions (0/30) are the extreme of "not estimable from a
sample": they require ranking categories by a delta computed over
thousands of reviews per quarter. 40 orders cannot produce that
ranking. This is exactly what System D (pre-computed bucket
summaries over the full population) is designed to address.

## 3. Scoring caveats (carry into headline-experiment design)

Two auto-scorer artefacts inflate System C's apparent score by ~1:

**trend_09 (review score Q2 2017 vs Q2 2018):** scored match=True,
but it is a tolerance artefact. Gold {4.12, 4.21}; model said
{4.5, ...}. With abs tol = max(0.05, 0.10·4.12) = 0.41, the value
4.5 is 0.38 from 4.12 → inside tolerance. But the model computed
4.5 from 6 sampled orders, not the population 4.12. Review-score
questions cluster tightly around 4–5, so almost any sample lands
"in tolerance". For the headline experiment, review-score (and
other tightly-clustered 1–5 scale) questions need a tighter
tolerance or should be scored by exact-bucket rather than ±10%.

**Spurious time anchors (latency_01, latency_05, trend_07):** the
anchor detector flagged "2018-01" mentions in answers about 2017
orders. But these are legitimate: an order purchased 2017-12 and
delivered 2018-01 correctly mentions a January 2018 delivery date.
The detector treats any out-of-gold-quarter mention as spurious,
which over-counts. The temporal-faithfulness judge (when built)
must distinguish "referenced a wrong analysis period" (real
hallucination) from "mentioned a delivery date that legitimately
falls just outside the purchase window" (not hallucination).

After both corrections, substantive System C matches are ~5/30,
not 6/30. Still a clear gain over Vanilla's 0/30.

## 4. What is solid vs what needs more data

Solid:
- Timeline fix is correct and necessary; verified in the evidence
  block (delivery dates now present).
- System C produces real per-order lag computations (latency_03/04
  show the arithmetic explicitly and land near gold).
- Drift remains 0% — the population-aggregation gap is real and
  motivates System D.

Needs more data / not yet claimable:
- Whether the narrow-vs-broad-distribution story holds at n=90+.
- Whether System D actually closes the drift/aggregation gap
  (conjectural until built).
- Clean match rates (pending the two scorer corrections above).

## 5. Disposition

- No paper edit yet (n=30; D and E not built).
- The two scorer corrections are deferred to headline-experiment
  prep; they do not change the C-vs-Vanilla direction.
- Next: build System D (community-temporal) and run it on the same
  30 questions. The key test is whether D moves drift off 0%.
