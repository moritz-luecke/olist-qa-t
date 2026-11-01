"""
============================================================
Debug specific factual misses — show the LLM's full input
============================================================

For each given qid, print:
  - the question
  - the gold answer
  - the system's actual answer
  - the FULL evidence block that was retrieved
  - which cited node/chunk corresponds to the target order

This is the slow-careful step before we patch anything: we need
to see whether the right chunk was retrieved but parsed wrong,
or whether the chunk was retrieved but the answer wasn't in it.
============================================================
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_HEX_RE = re.compile(r"\b[a-f0-9]{32}\b")

# Pick 5 representative miss types from the diag table
TARGETS = [
    "factual_05",   # V✗ E✗ — total payment value 141.46 (number)
    "factual_10",   # V✗ E✗ — freight value 27.2 (number)
    "factual_13",   # V✗ E✗ — customer city "balneario camboriu" (string)
    "factual_18",   # V✗ E✗ — listed price 15.99 (number)
    "factual_25",   # V✗ E✗ — number of line items = 1 (number)
]


def main():
    questions = {q["qid"]: q for q in
                 (json.loads(l) for l in
                  (PROJECT_ROOT / "outputs/pilot_questions.jsonl").open())}

    # Load chunks (for vanilla evidence reconstruction)
    chunks = {c["chunk_id"]: c for c in
              (json.loads(l) for l in
               (PROJECT_ROOT / "outputs/vanilla_chunks.jsonl").open())}

    for qid in TARGETS:
        q = questions[qid]
        ids_in_q = _HEX_RE.findall(q["question"])
        target_oid = ids_in_q[0] if ids_in_q else None

        print("\n" + "="*80)
        print(f"  {qid}  [{q['category']}, {q['gold_answer_type']}]")
        print("="*80)
        print(f"Question: {q['question']}")
        print(f"Gold answer: {q['gold_answer']!r}  (target order: {target_oid})")
        print(f"Pandas reference: {q.get('pandas_code', '')[:200]}")

        for system in ["vanilla_rag", "entity"]:
            print(f"\n--- {system} ---")
            answers_path = (PROJECT_ROOT / "outputs/results"
                            / f"{system}_answers.jsonl")
            for line in answers_path.open():
                a = json.loads(line)
                if a["qid"] != qid:
                    continue
                print(f"  Answer: {a['answer']}")

                if system == "vanilla_rag":
                    # Print the chunk that corresponds to the target order
                    target_chunk_id = f"O:{target_oid}"
                    if target_chunk_id in a["cited_chunk_ids"]:
                        rank = a["cited_chunk_ids"].index(target_chunk_id) + 1
                        ch = chunks.get(target_chunk_id)
                        print(f"  Target chunk [{rank}/8]: O:{target_oid}")
                        if ch:
                            print(f"    {ch['text']}")
                    else:
                        print(f"  ⚠ Target chunk {target_chunk_id} NOT in citations")
                else:
                    # Entity: print just the anchor + payment/item nodes
                    target_node = f"O:{target_oid}"
                    anchor_in = target_node in a["cited_node_ids"]
                    print(f"  Target node {target_node} in citations: {anchor_in}")
                    print(f"  Anchors used: {a['diagnostics'].get('anchors',[])}")
                break


if __name__ == "__main__":
    main()
