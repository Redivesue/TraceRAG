"""Query normalization helpers for retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
ENTITY_PATTERN = re.compile(r"(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)")
QUESTION_STARTERS = {"were", "what", "is", "are", "was", "who", "which", "did", "does", "do"}
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}


@dataclass
class NormalizedQuery:
    raw_query: str
    lexical_query: str
    dense_query: str
    entities: list[str]
    key_terms: list[str]


def normalize_query(query: str) -> NormalizedQuery:
    entities = _extract_entities(query)
    raw_tokens = [token.lower() for token in TOKEN_PATTERN.findall(query)]
    key_terms = [token for token in raw_tokens if token not in STOPWORDS]

    lexical_parts = entities + key_terms
    lexical_query = " ".join(_dedupe_preserve_order(lexical_parts)) or query
    dense_query = lexical_query
    return NormalizedQuery(
        raw_query=query,
        lexical_query=lexical_query,
        dense_query=dense_query,
        entities=entities,
        key_terms=key_terms,
    )


def _extract_entities(query: str) -> list[str]:
    entities: list[str] = []
    for match in ENTITY_PATTERN.findall(query):
        parts = match.split()
        while parts and parts[0].lower() in QUESTION_STARTERS:
            parts = parts[1:]
        cleaned = " ".join(parts)
        if cleaned and cleaned not in entities:
            entities.append(cleaned)
    return entities


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(item)
    return ordered
