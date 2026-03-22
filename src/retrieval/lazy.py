"""Lazy retriever wrapper to defer heavy index loading until first use."""

from __future__ import annotations

from typing import Callable

from src.core.schemas import RetrievalCandidate
from src.retrieval.base import Retriever


class LazyRetriever(Retriever):
    def __init__(self, factory: Callable[[], Retriever]) -> None:
        self.factory = factory
        self._inner: Retriever | None = None

    def retrieve(self, query: str, top_k: int) -> list[RetrievalCandidate]:
        if self._inner is None:
            self._inner = self.factory()
        return self._inner.retrieve(query=query, top_k=top_k)
