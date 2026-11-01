"""
============================================================
Diagnose: did the target order end up in the evidence?
============================================================

For each factual question with a known order_id in the gold,
check whether the retriever included the corresponding node/chunk
in its citations.

This separates two failure modes:
  - Retrieval miss: target node not in citations → embedder doesn't
    surface IDs well. Need a different seed strategy.
  - Generation miss: target node IS in citations, but the LLM still
    abstained → the serialized evidence didn't contain the needed
    field (e.g. payment_type), or the prompt is too restrictive.

Usage:
    cd phase4_retrieval
    python diag_factual.py
============================================================
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Match a 32-hex-char order ID anywhere in a string
_ORDER_RE = re.compile(r"\b[a-f0-9]{32}\b")


def extract_order_ids(question: str) -> list[str]:
    return _ORDER_RE.findall(question)


def diagnose(system: str) -> None:
    questions = {q["qid"]: q for q in
                 (json.loads(l) for l in
                  (PROJECT_ROOT / "outputs/pilot_questions.jsonl").open())}
    answers = [json.loads(l) for l in
               (PROJECT_ROOT / "outputs/results"
                / f"{system}_answers.jsonl").open()]

    print(f"\n{'='*60}\n  System: {system}\n{'='*60}")

    for a in answers:
        q = questions[a["qid"]]
        if q["category"] != "factual":
            continue

        oids_in_q = extract_order_ids(q["question"])
        if not oids_in_q:
            continue
        target_oid = oids_in_q[0]
        node_id = f"O:{target_oid}"
        chunk_id = f"O:{target_oid}"

        # Did the retriever include the target?
        cited_nodes = set(a.get("cited_node_ids", []))
        cited_chunks = set(a.get("cited_chunk_ids", []))
        node_hit = node_id in cited_nodes
        chunk_hit = chunk_id in cited_chunks
        target_in_evidence = node_hit or chunk_hit

        # Verdict
        ans_low = a["answer"].lower().strip()
        is_abstain = "insufficient evidence" in ans_low
        gold = str(q["gold_answer"]).lower()
        is_correct = (not is_abstain) and (gold in ans_low)

        verdict = (
            "✓ correct" if is_correct
            else ("✗ abstain" if is_abstain else "✗ wrong")
        )

        retrieval_status = (
            "✓ TARGET RETRIEVED" if target_in_evidence
            else "✗ target MISSED"
        )

        print(f"\n{a['qid']}: {verdict}")
        print(f"  Q:    {q['question']}")
        print(f"  Gold: {q['gold_answer']!r}")
        print(f"  Got:  {a['answer'][:120]!r}")
        print(f"  Retrieval: {retrieval_status} "
              f"(among {len(cited_nodes) + len(cited_chunks)} cited)")
        if not target_in_evidence:
            # Show what WAS retrieved instead
            sample = list(cited_nodes)[:3] or list(cited_chunks)[:3]
            print(f"  Sample of what was retrieved: {sample}")


if __name__ == "__main__":
    diagnose("vanilla_rag")
    diagnose("entity")
