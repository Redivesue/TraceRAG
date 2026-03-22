"""Base retriever interface."""

from abc import ABC, abstractmethod

from src.core.schemas import RetrievalCandidate


class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> list[RetrievalCandidate]:
        """Return ranked retrieval candidates for a query."""
