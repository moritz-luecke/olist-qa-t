"""
============================================================
Phase 4 — Retriever base class
============================================================

Every retrieval strategy implements two things:
  1. setup()      — build any index (called once, cached on disk)
  2. retrieve()   — for one question, return a pre-formatted
                    evidence block ready to drop into the
                    user prompt.

The evidence block format differs per strategy (see Appendix
A.1.3 for the spec); the base class just enforces the interface.
============================================================
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class RetrievalResult:
    """Returned by every retriever.retrieve() call."""
    evidence_block: str          # ready-to-insert into the user prompt
    cited_node_ids: list[str] = field(default_factory=list)
    cited_chunk_ids: list[str] = field(default_factory=list)
    # Metadata for analysis (not shown to the LLM):
    diagnostics: dict = field(default_factory=dict)


class BaseRetriever(ABC):
    """All retrieval strategies subclass this."""

    name: str = "base"  # override in subclass

    @abstractmethod
    def setup(self) -> None:
        """Build/load index. Idempotent — safe to call repeatedly."""
        ...

    @abstractmethod
    def retrieve(self, question: str) -> RetrievalResult:
        """Return formatted evidence for a single question."""
        ...
