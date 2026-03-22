"""Hybrid retrieval orchestration with normalization, RRF fusion, and title dedup."""

from __future__ import annotations

from collections import OrderedDict

from src.core.schemas import RetrievalCandidate
from src.retrieval.base import Retriever
from src.retrieval.dedup import dedup_by_title
from src.retrieval.fusion import rrf_fusion
from src.retrieval.query_normalizer import normalize_query
from src.retrieval.semantic_refiner import SemanticRefiner


class HybridRetriever(Retriever):
    def __init__(
        self,
        sparse_retriever: Retriever,
        dense_retriever: Retriever,
        title_retriever: Retriever | None = None,
        semantic_refiner: SemanticRefiner | None = None,
    ) -> None:
        self.sparse_retriever = sparse_retriever
        self.dense_retriever = dense_retriever
        self.title_retriever = title_retriever
        self.semantic_refiner = semantic_refiner
        self.query_cache: OrderedDict[tuple[str, int], list[RetrievalCandidate]] = OrderedDict()

    def retrieve(self, query: str, top_k: int) -> list[RetrievalCandidate]:
        cache_key = (query, top_k)
        cached = self.query_cache.get(cache_key)
        if cached is not None:
            self.query_cache.move_to_end(cache_key)
            return cached

        direct = self._title_fast_path(query, top_k=top_k)
        if direct is not None:
            self._remember(cache_key, direct)
            return direct

        normalized = normalize_query(query)
        candidate_budget = max(top_k * 2, 10)

        sparse = self.sparse_retriever.retrieve(normalized.lexical_query, top_k=candidate_budget)
        dense = self.dense_retriever.retrieve(normalized.dense_query, top_k=candidate_budget)
        fused = rrf_fusion([sparse, dense])
        deduped = dedup_by_title(fused, max_per_title=2)
        if self.semantic_refiner is not None:
            deduped = self.semantic_refiner.refine(normalized.raw_query, deduped)
        selected = self._select_with_coverage(normalized.entities, deduped, top_k=top_k)
        self._remember(cache_key, selected)
        return selected

    def _title_fast_path(self, query: str, top_k: int) -> list[RetrievalCandidate] | None:
        if self.title_retriever is None:
            return None
        fast = self.title_retriever.retrieve(query=query, top_k=top_k)
        return fast or None

    def _select_with_coverage(
        self,
        entities: list[str],
        candidates: list[RetrievalCandidate],
        top_k: int,
    ) -> list[RetrievalCandidate]:
        selected: list[RetrievalCandidate] = []
        selected_ids: set[str] = set()

        for entity in entities[:2]:
            match = next((item for item in candidates if _matches_entity(item, entity)), None)
            if match is not None:
                self._append(match, selected, selected_ids)

        for item in candidates:
            if len(selected) >= top_k:
                break
            self._append(item, selected, selected_ids)

        return selected[:top_k]

    def _append(
        self,
        candidate: RetrievalCandidate,
        selected: list[RetrievalCandidate],
        selected_ids: set[str],
    ) -> None:
        chunk_id = candidate.chunk.chunk_id
        if chunk_id in selected_ids:
            return
        selected.append(candidate)
        selected_ids.add(chunk_id)

    def _remember(self, cache_key: tuple[str, int], value: list[RetrievalCandidate]) -> None:
        self.query_cache[cache_key] = value
        self.query_cache.move_to_end(cache_key)
        while len(self.query_cache) > 256:
            self.query_cache.popitem(last=False)


def _matches_entity(candidate: RetrievalCandidate, entity: str) -> bool:
    entity_lower = entity.lower()
    return entity_lower in candidate.chunk.title.lower() or entity_lower in candidate.chunk.text.lower()
