# Olist-QA-T: Time-Aware Graph-RAG over Enterprise Transactions

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23858489.svg)](https://doi.org/10.5281/zenodo.23858489)

Olist-QA-T is a benchmark and toolkit for temporal question answering over
enterprise transactional data. It extends Olist-QA (150 questions over the
Olist e-commerce tables, evaluated as a static snapshot) with 90 temporal
questions on trends, drift, and latencies, and it ships everything needed to
evaluate time-aware retrieval on them: a temporal layer for a schema-aware
knowledge graph, seven reference systems in one harness, deterministic
temporal-faithfulness diagnostics, and the raw outputs of all reference runs.

The repository accompanies the paper *Olist-QA-T: A Benchmark and Toolkit for
Time-Aware Graph-RAG over Enterprise Transactions* (Moritz Lücke and Héctor
Allende-Cid, submitted to the ECIR 2027 resource track).

## Contents

| Artefact (Table 5 of the paper) | Path | Content |
|---|---|---|
| `temporal_questions_90.jsonl` | `outputs/` | 90 temporal questions (30 trend, 30 drift, 30 latency) |
| `pilot_questions.jsonl` | `benchmark/` | the 150 Olist-QA questions, unchanged |
| `external_events.jsonl` | `benchmark/` | 10 external event anchors |
| `time_buckets.jsonl` | `outputs/` | 38 time buckets (3 years, 10 quarters, 25 months) with exact statistics and generated summaries |
| `phase5_temporal/` | | interval index, time hierarchy, temporal retrievers C, D, E |
| `phase2_kg/` | | schema-aware KG construction from the Olist tables |
| `phase4_retrieval/` | | time-blind baselines A1, B1 to B3, prompts, evaluation harness |
| `scripts/` | | analysis, significance tests, data statistics (see `scripts/README.md`) |
| `outputs/results/` | | answers and scored records of all runs |

Each temporal question carries `qid`, `category`, `question`, a `sql_query`
and a `pandas_code` expression as dual reference, the verbatim execution
output as `gold_answer` (scalar, period-to-value dictionary, or ranked list,
see `gold_answer_type`), and the gold `time_anchors`.

### Systems and result files

| Paper | File name | System |
|---|---|---|
| A1 | `vanilla_rag` | Vanilla RAG v2 (order paragraphs with all timestamps) |
| B1 | `entity` | entity Graph-RAG |
| B2 | `community` | community Graph-RAG |
| B3 | `hybrid` | hybrid Graph-RAG |
| C | `entity_temporal` | Entity-Temporal |
| D | `community_temporal` | Community-Temporal (time buckets) |
| E | `hybrid_temporal` | Hybrid-Temporal (routing) |

Result files are named `temporal_questions_90_[gen-<MODEL>_]<system>_{answers,scored}.jsonl`.
Files without a `gen-` part use the pinned `gpt-4o-mini-2024-07-18`;
`force-<level>` files are the granularity sweep of system D. Files starting
with `pilot_questions_` and `judge_pilot_questions_` hold the runs on the
150 Olist-QA questions and the pairwise LLM-judge records for their
open-form categories.

## Reproduce the paper's tables (no API, no extra packages)

The analysis scripts use the Python standard library only (Python 3.10 or
newer) and recompute every result table from the released answer files:

```bash
python scripts/analyze_results.py          # accuracy, latency MAE, faithfulness diagnostics, granularity sweep
python scripts/significance.py             # paired bootstrap, McNemar, z-tests (Bonferroni)
python scripts/analyze_generators.py       # cross-generator tables
python scripts/significance_generators.py  # per-generator tests
python scripts/review_stats.py outputs/results outputs/temporal_questions_90.jsonl
```

## Rebuild and re-run

**Setup.**

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=...        # embeddings, community summaries, default generator
```

**Data.** Download the Brazilian E-Commerce Public Dataset by Olist from
Kaggle (https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) and
put the CSV files into `data/`.

**Pipeline.** Run from the repository root; outputs go to `outputs/`. Each
script documents its options in its header and via `--help`.

```bash
python phase2_kg/build_kg.py                         # schema-aware KG
python phase2_kg/add_node_descriptions.py            # template-based node descriptions
python phase2_kg/run_leiden.py                       # Leiden communities
python phase2_kg/summarize_communities.py            # community summaries (LLM calls)
python phase4_retrieval/chunking.py --v2             # Vanilla v2 order paragraphs
python phase4_retrieval/embed_v2_chunks.py --no-dry-run --yes
python phase5_temporal/build_temporal_index.py       # interval index
python phase5_temporal/time_hierarchy.py --no-dry-run  # time buckets, statistics, summaries
python phase4_retrieval/run_headline.py --all-temporal --real --yes
python phase4_retrieval/run_headline.py --all-timeblind --real --yes
```

Node and community embeddings are computed by the retrievers on first use
and cached in `outputs/`. Without `--real`, `run_headline.py` makes no API
calls (dry run). `--force-level {week,month,quarter,year}` fixes the bucket
granularity, and `--out-prefix` keeps new runs from overwriting the
released result files.

**Other generators.** `--gen-model <MODEL>` swaps the answer generator.
With `--gen-provider openai` the model is served by OpenAI. With the default
`--gen-provider genai` it is served by any OpenAI-compatible endpoint (for
example a vLLM server) given by `GENAI_BASE_URL`; credentials are sent as
HTTP Basic auth from `GENAI_USER` and `GENAI_PASSWORD` (for an endpoint
without authentication, any non-empty values work). Retrieval embeddings
(`text-embedding-3-small`) and the query-constraint extraction of system C
always use OpenAI.

## Temporal-faithfulness diagnostics

The diagnostics reported in the paper (answer rate, conservative coverage of
the gold anchors, out-of-window assertions) are computed deterministically
from the answer text in `scripts/analyze_results.py`; no LLM judge is
involved. `phase5_temporal/temporal_faithfulness.py` also contains an
LLM-extraction variant, which we discarded because its prompt showed the
question and credited periods named there even when the answer abstained
(`notes/faithfulness_bug_postmortem.md`, in German).

`notes/` holds dated lab notes from development. Numbers there come from
pilot runs and may differ from the paper; the paper's numbers are those
recomputed by `scripts/`.

## Licence

Code: MIT (`LICENSE`). Data artefacts (`benchmark/`,
`outputs/temporal_questions_90.jsonl`, `outputs/time_buckets.jsonl`,
`outputs/results/`): CC BY-NC-SA 4.0, the licence of the underlying Olist
data (`DATA_LICENSE.md`). The Olist tables themselves are not
redistributed.

## Versions and maintenance

Releases are versioned and archived on Zenodo, and `CHANGELOG.md` records
every change. Earlier versions stay archived, so published numbers remain
reproducible. Please report errors in questions or gold answers via GitHub
issues.

## Citation

See `CITATION.cff`. A reference to the paper will be added once it is
published.

## Contact

Moritz Lücke, Fraunhofer IAIS, moritz.luecke@iais.fraunhofer.de
