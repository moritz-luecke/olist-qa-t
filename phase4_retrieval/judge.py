"""
============================================================
Phase 4 — LLM-as-judge (pairwise)
============================================================

Implements the prompt from Appendix A.1.4. Used to compare two
systems' answers on open-ended questions (comparison, root_cause,
global). Closed categories (factual, aggregation) use exact-match
scoring instead — see metrics.py [TODO].

Order-bias mitigation:
  The order A/B of the two answers is fixed by hash(qid) so that
  re-runs produce the same assignment, and across the benchmark
  each system appears as "A" roughly half the time.

The judge returns a JSON object with winner ∈ {A, B, TIE} and
a short reason. We use response_format={"type": "json_object"}
to force valid JSON.
============================================================
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from config import (
    JUDGE_MODEL, JUDGE_TEMPERATURE, JUDGE_SEED, JUDGE_MAX_TOKENS, Paths,
)
from llm import ChatLLM
from prompts import JUDGE_PROMPT, format_rubric


@dataclass
class JudgeVerdict:
    qid: str
    system_a: str          # which system was shown in position A
    system_b: str          # which system was shown in position B
    winner: str            # "A" | "B" | "TIE"
    winner_system: str     # de-anonymized: system_a or system_b or "TIE"
    reason: str
    cached: bool = False


def _order_for_qid(qid: str, sys1: str, sys2: str) -> tuple[str, str]:
    """Deterministically pick which system goes in slot A vs B."""
    h = int(hashlib.sha256(qid.encode()).hexdigest()[:8], 16)
    return (sys1, sys2) if h % 2 == 0 else (sys2, sys1)


def make_judge_llm(paths: Paths, dry_run: bool = True) -> ChatLLM:
    """Build a ChatLLM configured for the judge model + caps."""
    return ChatLLM(
        cache_root=paths.cache_dir,
        model=JUDGE_MODEL,
        temperature=JUDGE_TEMPERATURE,
        top_p=1.0,
        seed=JUDGE_SEED,
        max_tokens=JUDGE_MAX_TOKENS,
        dry_run=dry_run,
    )


def judge_pair(
    judge: ChatLLM,
    qid: str,
    question: str,
    rubric_key_facts: list[str],
    sys1_name: str, sys1_answer: str,
    sys2_name: str, sys2_answer: str,
) -> JudgeVerdict:
    """Run one pairwise judgment."""
    a_name, b_name = _order_for_qid(qid, sys1_name, sys2_name)
    a_ans = sys1_answer if a_name == sys1_name else sys2_answer
    b_ans = sys2_answer if b_name == sys2_name else sys1_answer

    user = JUDGE_PROMPT.format(
        question=question,
        rubric_block=format_rubric(rubric_key_facts),
        answer_a=a_ans,
        answer_b=b_ans,
    )
    resp = judge.chat(
        system="You are a careful, impartial evaluator. Return JSON only.",
        user=user,
        response_format={"type": "json_object"},
    )

    try:
        parsed = json.loads(resp["text"])
        winner = parsed.get("winner", "TIE").upper()
        if winner not in {"A", "B", "TIE"}:
            winner = "TIE"
        reason = str(parsed.get("reason", ""))[:300]
    except (json.JSONDecodeError, TypeError):
        winner, reason = "TIE", "[judge returned non-JSON; defaulted to TIE]"

    de_anon = {"A": a_name, "B": b_name, "TIE": "TIE"}[winner]
    return JudgeVerdict(
        qid=qid,
        system_a=a_name, system_b=b_name,
        winner=winner, winner_system=de_anon,
        reason=reason,
        cached=resp.get("cached", False),
    )
