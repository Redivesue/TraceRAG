"""Base grounded answer generator interface."""

from abc import ABC, abstractmethod

from src.core.schemas import GeneratedAnswer, RerankResult


class AnswerGenerator(ABC):
    @abstractmethod
    def generate(self, query: str, evidence: list[RerankResult]) -> GeneratedAnswer:
        """Generate an answer grounded in the provided evidence only."""
