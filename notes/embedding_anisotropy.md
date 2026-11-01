# Lab Notes — Embedding Anisotropy in Vanilla v2

**Date:** 2026-05-26
**Updated:** 2026-05-27 (after pilot run)
**Status:** Conjecture not supported by pilot evidence — see §3.

## Original observation (2026-05-26)

Three cosine similarities measured against
`text-embedding-3-small` embeddings of Olist order paragraphs:

| Pair                              | Cosine | n            |
|-----------------------------------|--------|--------------|
| Same-year top-1 nearest neighbour | 0.95   | 1 (query=0)  |
| Random pair                       | 0.84   | 3 ad-hoc     |
| 2016 order vs 2018 order          | 0.787  | 1 ad-hoc     |

The 2016↔2018 similarity (0.787) sat below the random-pair floor
(~0.84), suggesting that the embedding space encoded year-token
information stronger than the generic order-paragraph template.

## Original conjecture

That Vanilla v2 would have *year-level discrimination but not
sub-year discrimination*, predicting:
- Year-anchored questions ("in 2017") work reasonably for Vanilla v2.
- Sub-year questions ("Q1 vs Q4 2017", "after July 2017") fail
  systematically because the embedding cannot distinguish months
  within a year.

## Reality check from the pilot (2026-05-27)

The 30-question pilot does NOT support the anisotropy explanation
as the dominant failure mode.

Observed:
- 20/30 questions (67%) returned `INSUFFICIENT EVIDENCE`. This is
  *retrieval-budget-bound* behaviour: top-8 paragraphs out of
  99,441 cannot answer any aggregation-over-all-2017 question
  regardless of how well the embedding discriminates year. The
  retrieval would have to return all ~25,000 Q1 orders + all
  ~25,000 Q4 orders for the LLM to actually aggregate, which is
  not what cosine-top-8 does on any embedding space.
- 9/30 questions produced a numeric answer that was wildly wrong
  (factor 2-4x off the gold value). The errors came from
  *aggregating the 8 retrieved samples as if they were the
  population*, not from confusing Q1 with Q4 in the embedding.
- 0/30 questions showed evidence of the model picking up
  paragraphs from the wrong year-bucket and confidently labelling
  them as the right one (which is what the anisotropy hypothesis
  predicted).

The dominant failure mode is not embedding anisotropy. It is
*top-k aggregation miscalibration* — see
`aggregation_miscalibration.md` (next note) for the right
explanation.

## What the 0.787 actually tells us

The 2016-vs-2018 lower-than-random similarity is real, but it is
neither necessary nor sufficient for the failure modes observed on
the pilot. It might still matter for:
- Edge cases where retrieval would need to return cross-year
  matches (e.g., longitudinal questions). The pilot does not
  exercise these.
- The auto-extracted-KG variant (excluded from this paper), where
  the year-token might propagate into extracted triples.

For the temporal-extension paper, the anisotropy observation is a
footnote, not a hypothesis.

## Disposition

- The H6 sketch ("Vanilla v2 has sub-year-discrimination deficit")
  is NOT added to §8.3 of the paper.
- This file stays as a record of an investigated-and-rejected
  conjecture. Useful for future writers asking
  "why didn't you test year-token anisotropy?".
- The replacement headline finding lives in
  `aggregation_miscalibration.md`.

## To-do (if anyone ever needs to revisit this)

- Statistical version on 100-500 random pairs per year-diff
  bucket would still be informative for whether sub-year-vs-
  cross-year cosine separation is real or a one-pair fluke.
- BUT: even if real, the pilot showed it is dominated by the
  retrieval-budget problem. Lower-priority follow-up.
