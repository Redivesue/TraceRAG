"""Fusion helpers for multi-retriever candidate merging."""

from __future__ import annotations

from collections import defaultdict

from src.core.schemas import RetrievalCandidate


def rrf_fusion(
    ranked_lists: list[list[RetrievalCandidate]],
    k: int = 60,
) -> list[RetrievalCandidate]:
    scores: dict[str, float] = defaultdict(float)
    exemplar: dict[str, RetrievalCandidate] = {}

    for ranked in ranked_lists:
        for rank, candidate in enumerate(ranked, start=1):
            chunk_id = candidate.chunk.chunk_id
            scores[chunk_id] += 1.0 / (k + rank)
            exemplar.setdefault(chunk_id, candidate)

    ranked_ids = sorted(scores, key=scores.get, reverse=True)
    return [
        RetrievalCandidate(
            chunk=exemplar[chunk_id].chunk,
            score=scores[chunk_id],
            source_method="hybrid_rrf",
        )
        for chunk_id in ranked_ids
    ]
