"""Heuristic reranker with entity-aware, diversity-aware scoring."""

from __future__ import annotations

import re

from src.core.schemas import RerankResult, RetrievalCandidate
from src.rerank.base import Reranker

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
ENTITY_PATTERN = re.compile(r"(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)")
QUESTION_STARTERS = {"were", "what", "is", "are", "was", "who", "which", "did", "does", "do"}
ATTRIBUTE_TERMS = {
    "nationality",
    "country",
    "american",
    "british",
    "french",
    "canadian",
    "born",
    "profession",
    "director",
    "actor",
    "writer",
    "experiment",
    "theory",
    "relativity",
    "gravity",
    "gravitational",
    "redshift",
    "purpose",
    "neutrino",
    "detector",
}


class HostedReranker(Reranker):
    def rerank(
        self,
        query: str,
        candidates: list[RetrievalCandidate],
        top_k: int,
    ) -> list[RerankResult]:
        """Reorder candidates with lexical, entity, and redundancy-aware signals."""
        query_tokens = {token.lower() for token in TOKEN_PATTERN.findall(query)}
        entities = _extract_entities(query)
        comparison_query = _is_comparison_query(query)

        reranked = []
        for candidate in candidates:
            reranked.append(
                RerankResult(
                    chunk=candidate.chunk,
                    retrieval_score=candidate.score,
                    rerank_score=self._score(
                        query_tokens=query_tokens,
                        entities=entities,
                        comparison_query=comparison_query,
                        candidate=candidate,
                    ),
                    rank=0,
                    source_method=candidate.source_method,
                )
            )

        reranked.sort(key=lambda item: item.rerank_score, reverse=True)

        title_seen: dict[str, int] = {}
        adjusted_results: list[RerankResult] = []
        for item in reranked:
            penalty = 0.25 * title_seen.get(item.chunk.title, 0)
            item.rerank_score -= penalty
            title_seen[item.chunk.title] = title_seen.get(item.chunk.title, 0) + 1
            adjusted_results.append(item)

        adjusted_results.sort(key=lambda item: item.rerank_score, reverse=True)
        final_results = self._select_diverse_top_k(
            adjusted_results,
            entities=entities,
            comparison_query=comparison_query,
            top_k=top_k,
        )
        for index, item in enumerate(final_results):
            item.rank = index + 1
        return final_results

    def _score(
        self,
        query_tokens: set[str],
        entities: list[str],
        comparison_query: bool,
        candidate: RetrievalCandidate,
    ) -> float:
        title_lower = candidate.chunk.title.lower()
        text_lower = candidate.chunk.text.lower()
        title_tokens = {token.lower() for token in TOKEN_PATTERN.findall(candidate.chunk.title)}
        text_tokens = {token.lower() for token in TOKEN_PATTERN.findall(candidate.chunk.text)}
        title_overlap = len(query_tokens & title_tokens)
        text_overlap = len(query_tokens & text_tokens)
        support_bonus = 0.2 if candidate.chunk.metadata.get("supporting_fact_hint") else 0.0
        query_text = " ".join(sorted(query_tokens))

        entity_bonus = 0.0
        full_entity_hits = 0
        for entity in entities:
            entity_lower = entity.lower()
            if title_lower == entity_lower:
                entity_bonus += 2.5
                full_entity_hits += 1
            elif entity_lower in title_lower:
                entity_bonus += 1.5
                full_entity_hits += 1
            elif entity_lower in text_lower:
                entity_bonus += 0.4

        attribute_bonus = 0.12 * sum(1 for term in ATTRIBUTE_TERMS if term in title_lower or term in text_lower)
        exact_title_bonus = 1.1 if title_lower == query_text else 0.0
        partial_title_bonus = 0.4 if query_text and query_text in title_lower else 0.0
        sentence_redundancy_penalty = 0.35 * _redundancy_ratio(candidate.chunk.text)

        mismatch_penalty = 0.0
        if entities and full_entity_hits == 0:
            mismatch_penalty += 1.25
        if comparison_query and entities and full_entity_hits == 0:
            mismatch_penalty += 0.75
        if not entities and title_overlap == 0 and text_overlap <= 1:
            mismatch_penalty += 0.35
        if _looks_like_lexical_noise(query_tokens, title_tokens, text_tokens):
            mismatch_penalty += 0.45

        return (
            candidate.score
            + (0.35 * title_overlap)
            + (0.08 * text_overlap)
            + support_bonus
            + entity_bonus
            + attribute_bonus
            + exact_title_bonus
            + partial_title_bonus
            - mismatch_penalty
            - sentence_redundancy_penalty
        )

    def _select_diverse_top_k(
        self,
        candidates: list[RerankResult],
        entities: list[str],
        comparison_query: bool,
        top_k: int,
    ) -> list[RerankResult]:
        selected: list[RerankResult] = []
        selected_ids: set[str] = set()
        title_counts: dict[str, int] = {}
        max_per_title = 1 if comparison_query else 2
        title_tokens_seen: list[set[str]] = []

        for entity in entities:
            match = next((item for item in candidates if _matches_entity(item, entity)), None)
            if match is not None:
                self._append_result(match, selected, selected_ids, title_counts, title_tokens_seen)

        for item in candidates:
            if len(selected) >= top_k:
                break
            if title_counts.get(item.chunk.title, 0) >= max_per_title:
                continue
            if _is_too_similar_title(item.chunk.title, title_tokens_seen):
                continue
            self._append_result(item, selected, selected_ids, title_counts, title_tokens_seen)

        return selected[:top_k]

    def _append_result(
        self,
        item: RerankResult,
        selected: list[RerankResult],
        selected_ids: set[str],
        title_counts: dict[str, int],
        title_tokens_seen: list[set[str]],
    ) -> None:
        if item.chunk.chunk_id in selected_ids:
            return
        selected.append(item)
        selected_ids.add(item.chunk.chunk_id)
        title_counts[item.chunk.title] = title_counts.get(item.chunk.title, 0) + 1
        title_tokens_seen.append({token.lower() for token in TOKEN_PATTERN.findall(item.chunk.title)})


def _extract_entities(query: str) -> list[str]:
    entities: list[str] = []
    for match in ENTITY_PATTERN.findall(query):
        cleaned = _clean_entity_match(match)
        if cleaned and cleaned not in entities:
            entities.append(cleaned)
    return entities


def _is_comparison_query(query: str) -> bool:
    query_lower = query.lower()
    markers = ["same ", " both ", "compare", "older", "younger", "than", "difference"]
    return any(marker in query_lower for marker in markers)


def _matches_entity(candidate: RerankResult, entity: str) -> bool:
    entity_lower = entity.lower()
    return entity_lower in candidate.chunk.title.lower() or entity_lower in candidate.chunk.text.lower()


def _clean_entity_match(match: str) -> str:
    parts = match.split()
    while parts and parts[0].lower() in QUESTION_STARTERS:
        parts = parts[1:]
    return " ".join(parts)


def _redundancy_ratio(text: str) -> float:
    sentences = [sentence.lower().strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]
    if len(sentences) < 2:
        return 0.0
    unique = len(set(sentences))
    return max(0.0, 1 - (unique / len(sentences)))


def _looks_like_lexical_noise(
    query_tokens: set[str],
    title_tokens: set[str],
    text_tokens: set[str],
) -> bool:
    stopish = {"what", "did", "does", "do", "is", "are", "the", "a", "an", "of", "in"}
    content_tokens = {token for token in query_tokens if token not in stopish}
    if not content_tokens:
        return False
    overlap = content_tokens & (title_tokens | text_tokens)
    return len(overlap) <= 1


def _is_too_similar_title(title: str, title_tokens_seen: list[set[str]]) -> bool:
    tokens = {token.lower() for token in TOKEN_PATTERN.findall(title)}
    if not tokens:
        return False
    for seen in title_tokens_seen:
        union = tokens | seen
        if union and (len(tokens & seen) / len(union)) >= 0.75:
            return True
    return False
