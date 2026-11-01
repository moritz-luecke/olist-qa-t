"""
============================================================
Phase 4 — Generation pipeline
============================================================

Glues a retriever (System A or B/C-x) and the chat model:
  question → retrieve evidence → format prompt → call LLM →
  parse answer.

Vanilla RAG uses VANILLA_USER_PROMPT (paragraphs);
all four graph strategies use GRAPH_USER_PROMPT (subgraph).
============================================================
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from config import Paths
from llm import ChatLLM
from prompts import (
    SYSTEM_MESSAGE,
    VANILLA_USER_PROMPT,
    GRAPH_USER_PROMPT,
)
from retrievers.base import BaseRetriever, RetrievalResult


# ----------------------------------------------------------------
# Per-question result record
# ----------------------------------------------------------------
@dataclass
class AnswerRecord:
    qid: str
    system: str               # which retriever produced this
    answer: str
    cited_node_ids: list[str] = field(default_factory=list)
    cited_chunk_ids: list[str] = field(default_factory=list)
    tokens: dict = field(default_factory=dict)
    cached: bool = False
    diagnostics: dict = field(default_factory=dict)


# ----------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------
class Pipeline:
    def __init__(self, retriever: BaseRetriever, llm: ChatLLM):
        self.retriever = retriever
        self.llm = llm

    def _build_user_prompt(self, question: str,
                           retrieval: RetrievalResult) -> str:
        if self.retriever.name == "vanilla_rag":
            # Reconstruct k from paragraph count
            k = len(retrieval.cited_chunk_ids)
            return VANILLA_USER_PROMPT.format(
                question=question,
                k=k,
                paragraphs_block=retrieval.evidence_block,
            )
        # All four graph strategies share the GRAPH_USER_PROMPT template
        return GRAPH_USER_PROMPT.format(
            question=question,
            evidence=retrieval.evidence_block,
        )

    def answer(self, qid: str, question: str) -> AnswerRecord:
        retrieval = self.retriever.retrieve(question)
        user_prompt = self._build_user_prompt(question, retrieval)
        resp = self.llm.chat(system=SYSTEM_MESSAGE, user=user_prompt)
        return AnswerRecord(
            qid=qid,
            system=self.retriever.name,
            answer=resp["text"],
            cited_node_ids=retrieval.cited_node_ids,
            cited_chunk_ids=retrieval.cited_chunk_ids,
            tokens=resp["tokens"],
            cached=resp.get("cached", False),
            diagnostics=retrieval.diagnostics,
        )


# ----------------------------------------------------------------
# I/O for answer records
# ----------------------------------------------------------------
def write_answers(records: list[AnswerRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for r in records:
            fp.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def read_answers(path: Path) -> list[AnswerRecord]:
    with path.open(encoding="utf-8") as fp:
        return [AnswerRecord(**json.loads(line)) for line in fp]
