"""Dedup helpers for retrieval candidates."""

from __future__ import annotations

from src.core.schemas import RetrievalCandidate


def dedup_by_title(
    candidates: list[RetrievalCandidate],
    max_per_title: int = 2,
) -> list[RetrievalCandidate]:
    title_counts: dict[str, int] = {}
    deduped: list[RetrievalCandidate] = []
    seen_ids: set[str] = set()

    for candidate in candidates:
        chunk_id = candidate.chunk.chunk_id
        if chunk_id in seen_ids:
            continue
        title = candidate.chunk.title
        if title_counts.get(title, 0) >= max_per_title:
            continue
        seen_ids.add(chunk_id)
        title_counts[title] = title_counts.get(title, 0) + 1
        deduped.append(candidate)

    return deduped
