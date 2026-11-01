# System E (temporal hybrid) — pilot findings

**Date:** 2026-06-02
**Runs:** `pilot_run_systemE.py --real --extract-prefer rule`
  (headline, `fusion=route`) and `--fusion concat` (ablation).
**Scored with:** the corrected auto-scorer; comparable to Vanilla / C / D.

---

## 1. One-paragraph summary

System E fuses System C (temporal-entity) and System D (community-
temporal). The headline mode is **score-based routing**: use C when the
question filters by a specific customer state, use D otherwise. Routed E
scores **19/30 (63%)** — it reaches the C∪D oracle ceiling, beating
D-alone (18/30) by recovering the one entity-sub-filtered question only C
can answer (latency_01). The naive concatenation ablation (`fusion=concat`)
scores **15/30 — *below* D** — because the unused half of the evidence
acts as a two-directional distractor. Routing therefore is the
contribution; concatenation is the ablation that motivates it.

---

## 2. Capstone five-system table (all re-scored, same scorer)

| Category   | Vanilla v2 | System C | System D | E (concat, abl.) | **E (route)** |
|------------|:----------:|:--------:|:--------:|:----------------:|:-------------:|
| trend (10) |   1/10     |  2/10    |  10/10   |       9/10       |   **10/10**   |
| drift (10) |   0/10     |  0/10    |   2/10   |       2/10       |    **2/10**   |
| latency(10)|   0/10     |  4/10    |   6/10   |       4/10       |    **7/10**   |
| **total**  | **1/30**   | **6/30** | **18/30**|     **15/30**    |   **19/30**   |
| spurious anchors | 3    | 3        |   0      |        0         |   **0***    |

Progression on the headline axis: **1 → 6 → 18 → 19**, with the
faithfulness (spurious-anchor) count collapsing from 3 (time-blind) to
0 (temporal). This is the paper's RQ1/RQ2 story in one table.

*E (route) showed 1 spurious anchor under the raw detector (a Dec-2017
order, routed to C, legitimately delivered in 2018-01); the lifecycle-
spillover scorer fix (now applied, §7.2a) recognises this as a downstream
lifecycle date, not a hallucinated anchor, taking E to 0.

---

## 3. Routing result (headline, fusion=route)

- Route breakdown: **community=28, entity=2**. The state-filter detector
  fired on exactly latency_01 and latency_06 (and nothing else — e.g.
  drift_08 "top 5 customer states" names no specific state, so it stays
  on D). No misrouting.
- E differs from D on **exactly one** question: **latency_01**
  (D ✗ → E ✓), routed to C, which computes the São-Paulo-filtered lag
  (~9.3 days) from per-order timelines that D's all-population bucket
  (~13 days) cannot. This is the entire marginal value of the hybrid on
  the pilot — small in count, but it is precisely the **entity-sub-filter
  class** that routing exists to serve, and that class grows at 90 Qs.
- latency_06 (SP vs RJ) also routes to C but stays ✗ (C cannot do a
  two-state comparison from one stratified sample either) — no regression
  vs D, which also missed it.
- The single spurious anchor is on latency_01: C's evidence includes a
  Dec-2017 order delivered in Jan-2018, so the answer legitimately
  mentions "2018-01", which the gold-quarter detector flags. This is the
  known detector artefact (status.md §5, "don't flag 2018-Jan mentions on
  Dec-2017 orders"), not a hallucination. The lifecycle-spillover scorer
  fix (§7.2a, now applied) removes it → E spurious = 0.

---

## 4. Why naive concatenation fails (the ablation, fusion=concat)

Concat scored 15/30, **3 below D**. It lost exactly three D-hits and
gained nothing; the failure is two-directional, both caused by the unused
evidence half acting as a distractor:

- **D's clean number overrides C (latency_01).** With both blocks present,
  the model anchored on D's all-population 2017 bucket (~13 days) and
  ignored C's SP order timelines → still ✗. So concat does *not* recover
  the very question that motivates a hybrid.
- **C's per-order chains distract D (latency_03, latency_04).** C's full
  purchase→approved→carrier→delivered chains give the model alternative
  ways to compute "a latency", and it mislabelled the *adjacent* interval:
    - latency_03 (gold 9.59, delivery lag): concat answered **3.02** — the
      carrier lag (latency_04's gold).
    - latency_04 (gold 3.02, carrier lag): concat answered **0.41** — the
      approval lag (latency_02's gold).
  D-alone, exposing only four cleanly-named aggregates, maps the label
  correctly.
- trend_09 also flipped ✗ under concat: the mixed evidence produced a
  differently-phrased answer ("Q2 for 2017 …") lacking the canonical
  `[O:2017-Q2]` citation the auto-scorer keys on. Routed E (→ D) returns
  the citation-bearing answer and scores ✓. (This is a generation
  difference, not a scorer bug — see §6.)

Routing removes the distractor in both directions: entity-filtered → C
only; everything else → D only.

---

## 5. RQ2 answer

Of the three temporal designs, the **bi-level community retriever (D) is
the workhorse**; entity-temporal (C) is needed only as a **routed fallback
for entity sub-filters**. Naive evidence union (concat) is **inferior to
the better single component** (15 < 18). The contribution of System E is
the routing policy, with concat as the ablation that justifies it.

Honest caveat for the write-up: on the pilot, routing's gain over D is a
single question. The claim should be framed as *class*-level ("routing
recovers the entity-sub-filter class while preserving D's exact-aggregate
accuracy and 0-spurious faithfulness"), not as a large headline jump. The
30→90 scale-up is where this gain should grow (more state/category-filtered
aggregates).

---

## 6. trend_09 — resolved, no scorer change

The full D and routed-E answers for trend_09 are byte-identical and both
contain the citation `[O:2017-Q2, O:2018-Q2]`; the literal token is what
`find_anchor_string` matches, so both score ✓ correctly. The concat-E ✗
was a genuinely different generation (no canonical citation), not a scorer
artefact. The "Q2 for 2017" adjacency gap in the auto-scorer is real but
affected no headline result and is moot under the LLM-judge used for the
full benchmark. **No third scorer correction applied.**

---

## 7. Recommended next steps

1. **Pilot is complete** — all five systems run and comparable. Fold the
   capstone table (§2) into paper §6 and the routing/ablation story into
   §6.x (failure-mode analysis) and the RQ2 discussion.
2. **Scorer-correction backlog** before the headline experiment
   (status.md §5): (a) spurious-anchor detector should not flag a month
   that legitimately follows an in-window order's lifecycle (fixes the
   one latency_01 spurious); (b) review-score tolerance on the 1–5 scale.
   Neither blocks scaling.
3. **Scale 30 → 90** (3 categories × 30). Re-check whether routing's
   margin over D widens as the entity-sub-filter and long-tail-drift
   classes grow.
4. **Headline experiment**: 7 systems × 240 questions with the LLM-judge +
   temporal-faithfulness judge.

---

## 8. Temporal faithfulness (LLM judge, 30 pilot answers)

Run with `temporal_faithfulness.py --real` (gpt-4o-2024-08-06,
temp 0, seed 42). LLM extracts every asserted time period
(paraphrase-robust); Python classifies covered/missing/hallucinated
deterministically (month-granular, lifecycle-spillover aware).

| metric              | Vanilla v2 | C    | D     | E     |
|---------------------|-----------:|-----:|------:|------:|
| hallucination rate  | 0.0%       | 0.0% | 0.0%  | 0.0%  |
| coverage rate       | 70.0%      | 80.0%| 96.7% | 96.7% |
| coverage (n/30)     | 21         | 24   | 29    | 29    |

**Headline finding — H2 must be reframed.** Time-anchor
hallucination is **0% for every system, including time-blind
Vanilla**. The original H2 ("time-blind RAG hallucinates time
anchors") does NOT hold: Vanilla's failure mode is not asserting
wrong periods but **abstention + aggregation-miscalibration**. On
its 9 coverage misses (trend_01/02/03/06/09, drift_01,
latency_03/04) Vanilla extracted *no* periods at all
(`extracted=[]`) — it returns INSUFFICIENT EVIDENCE or an ungrounded
single value rather than a wrong-dated claim.

**The faithfulness signal is the coverage gap**, not hallucination:
temporal community/hybrid retrieval reaches near-complete temporal
coverage (96.7%) because it abstains far less; Vanilla covers only
70%. D and E miss only drift_01 (the category-drift question that
weakens all systems in the pilot). This pairs with the accuracy gap
(match 1/6/18/19) as the two-pronged headline.

**Methodological note (κ degeneracy).** `κ(judge vs mechanical
silver)` = 1.0 / 0.0 / 1.0 / 1.0. The C value of 0.0 is a
**degenerate-constant-rater artifact**: with the LLM hallucination
label constant-False and a near-zero base rate, Cohen's κ is 0/
undefined regardless of agreement (raw agreement on C is 29/30; the
lone disagreement is trend_07, where the LLM correctly declines to
flag a "2018-01" lifecycle mention that the regex flags). Because
hallucination has collapsed to ~0, **κ on hallucination is
uninformative; human calibration should target the coverage
decision** (variance 70–97%), not hallucination. Plan the 30 manual
labels as `{qid, coverage_ok}`.
