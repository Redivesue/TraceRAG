"""Base reranker interface."""

from abc import ABC, abstractmethod

from src.core.schemas import RerankResult, RetrievalCandidate


class Reranker(ABC):
    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[RetrievalCandidate],
        top_k: int,
    ) -> list[RerankResult]:
        """Reorder candidates for a query."""
