"""
============================================================
Phase 4 — Prompt templates
============================================================

VERBATIM from Appendix A.1 of the paper. Any change here MUST be
mirrored in A_appendix.tex (and vice versa).

Format-string conventions:
  - {question}           - the NL question
  - {evidence}           - already-formatted evidence block
  - {paragraph_1..k}     - vanilla RAG paragraphs
  - Use double-braces in f-string contexts.
============================================================
"""
from __future__ import annotations

# ----------------------------------------------------------------
# Shared system message (A.1.1) — sent to all 5 systems
# ----------------------------------------------------------------
SYSTEM_MESSAGE = """\
You are a careful business analyst answering questions about an
e-commerce dataset. Use ONLY the evidence provided in the user
message; do not rely on outside knowledge. If the evidence is
insufficient to answer, respond exactly with the string:
INSUFFICIENT EVIDENCE
Otherwise, answer concisely. For numeric questions, return the
numeric value with its unit. For yes/no questions, return "yes"
or "no". For comparison or root-cause questions, return at most
four sentences and ground every claim in the evidence.\
"""


# ----------------------------------------------------------------
# Vanilla RAG user prompt (A.1.2) — System A
# ----------------------------------------------------------------
VANILLA_USER_PROMPT = """\
QUESTION:
{question}

EVIDENCE (top-{k} retrieved order paragraphs, ranked by
cosine similarity):
{paragraphs_block}

Answer the question using only the evidence above. Cite the
paragraph numbers you used in square brackets, e.g. [2,5].\
"""


# ----------------------------------------------------------------
# Graph-RAG user prompt (A.1.3) — Systems B and C
# ----------------------------------------------------------------
GRAPH_USER_PROMPT = """\
QUESTION:
{question}

EVIDENCE (retrieved subgraph, serialized as typed triples and
node attributes):

{evidence}

Answer the question using only the evidence above. Cite node
IDs in square brackets when grounding a claim, e.g. [O:abc123].\
"""


# ----------------------------------------------------------------
# LLM-as-judge pairwise prompt (A.1.4)
# ----------------------------------------------------------------
JUDGE_PROMPT = """\
You are an impartial evaluator. You will see a business question,
a rubric of facts the correct answer should contain, and two
candidate answers (A and B). Choose the answer that is BETTER
overall on three criteria, in this priority order:
  (1) Faithfulness to the rubric (no hallucinations, no
      contradictions of stated facts).
  (2) Coverage of the rubric's key facts.
  (3) Clarity and conciseness.

QUESTION:
{question}

RUBRIC:
{rubric_block}

ANSWER A:
{answer_a}

ANSWER B:
{answer_b}

Respond with a single JSON object and no other text:
{{"winner": "A" | "B" | "TIE",
  "reason": "<one sentence, <=30 words>"}}\
"""


# ----------------------------------------------------------------
# Helper formatters
# ----------------------------------------------------------------
def format_vanilla_paragraphs(paragraphs: list[str]) -> str:
    """Format ranked paragraphs as [1] ... [2] ... block."""
    return "\n".join(f"[{i}] {p}" for i, p in enumerate(paragraphs, start=1))


def format_rubric(key_facts: list[str]) -> str:
    """Format rubric key_facts as bullet list."""
    return "\n".join(f"- {kf}" for kf in key_facts)
