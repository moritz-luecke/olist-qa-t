"""Phase 4 — retrieval strategies."""
from retrievers.base import BaseRetriever, RetrievalResult
from retrievers.community import CommunityRetriever
from retrievers.entity import EntityRetriever
from retrievers.hybrid import HybridRetriever
from retrievers.path import PathRetriever
from retrievers.vanilla_rag import VanillaRAG

REGISTRY = {
    "vanilla_rag":  VanillaRAG,
    "entity":       EntityRetriever,
    "path":         PathRetriever,
    "community":    CommunityRetriever,
    "hybrid":       HybridRetriever,
}

__all__ = ["BaseRetriever", "RetrievalResult", "REGISTRY",
           "VanillaRAG", "EntityRetriever", "PathRetriever",
           "CommunityRetriever", "HybridRetriever"]
