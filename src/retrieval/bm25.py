"""Sparse retrieval over the exported chunk corpus with compact persisted postings."""

from __future__ import annotations

import gzip
import math
import pickle
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from src.core.schemas import RetrievalCandidate
from src.data.corpus import load_chunks
from src.retrieval.base import Retriever

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
    "for",
    "from",
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
    "with",
}
ATTRIBUTE_HINTS = {
    "nationality": ["nationality", "country", "american", "british", "french", "canadian"],
    "birthplace": ["born", "birthplace", "city", "country"],
    "profession": ["profession", "actor", "director", "writer", "scientist"],
    "age": ["born", "older", "younger", "age", "year"],
}
BM25_CACHE_VERSION = 2


@dataclass
class QueryPlan:
    raw_query: str
    entities: list[str]
    attribute_terms: list[str]
    token_sets: list[list[str]]


class BM25Retriever(Retriever):
    def __init__(self, corpus_path: str | Path, cache_path: str | Path | None = None) -> None:
        self.corpus_path = Path(corpus_path)
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self.chunks = load_chunks(self.corpus_path)
        self.avg_doc_len = 0.0
        self.doc_freq: dict[str, int] = {}
        self.doc_lengths: list[int] = []
        self.postings: dict[str, tuple[list[int], list[int]]] = {}
        self.title_exact_index: dict[str, list[int]] = {}
        self.title_token_index: dict[str, list[int]] = {}
        self.query_cache: dict[tuple[str, int], list[RetrievalCandidate]] = {}
        self._load_or_build_index()

    def retrieve(self, query: str, top_k: int) -> list[RetrievalCandidate]:
        """Return BM25-ranked candidates with query decomposition and title/entity boosts."""
        cache_key = (query, top_k)
        cached = self.query_cache.get(cache_key)
        if cached is not None:
            return cached

        plan = _build_query_plan(query)
        if not plan.token_sets:
            return []

        candidate_ids, token_hits = self._collect_candidate_ids(plan)
        scored: list[RetrievalCandidate] = []
        for index in candidate_ids:
            score = self._score(
                plan=plan,
                doc_index=index,
                term_freqs=token_hits.get(index, {}),
                doc_len=self.doc_lengths[index],
            )
            if score <= 0:
                continue
            scored.append(
                RetrievalCandidate(chunk=self.chunks[index], score=score, source_method="bm25")
            )

        scored.sort(key=lambda item: item.score, reverse=True)
        result = scored[:top_k]
        self.query_cache[cache_key] = result
        return result

    def _collect_candidate_ids(self, plan: QueryPlan) -> tuple[list[int], dict[int, dict[str, int]]]:
        candidate_ids: set[int] = set()
        token_hits: dict[int, dict[str, int]] = defaultdict(dict)
        for token_set in plan.token_sets:
            for token in token_set:
                doc_ids, freqs = self.postings.get(token, ([], []))
                for doc_id, freq in zip(doc_ids, freqs):
                    candidate_ids.add(doc_id)
                    token_hits[doc_id][token] = freq

        for entity in plan.entities:
            entity_lower = entity.lower()
            candidate_ids.update(self.title_exact_index.get(entity_lower, []))
            for token in _tokenize(entity):
                candidate_ids.update(self.title_token_index.get(token, []))

        if not candidate_ids:
            for token in plan.token_sets[0][:4]:
                doc_ids, freqs = self.postings.get(token, ([], []))
                for doc_id, freq in zip(doc_ids[:200], freqs[:200]):
                    candidate_ids.add(doc_id)
                    token_hits[doc_id][token] = freq

        return sorted(candidate_ids), token_hits

    def _load_or_build_index(self) -> None:
        if self.cache_path is not None and self.cache_path.exists():
            try:
                with _open_binary(self.cache_path, "rb") as handle:
                    payload = pickle.load(handle)
                if payload.get("version") != BM25_CACHE_VERSION:
                    raise KeyError("Unsupported BM25 cache version")
                self.avg_doc_len = payload["avg_doc_len"]
                self.doc_freq = payload["doc_freq"]
                self.doc_lengths = payload["doc_lengths"]
                self.postings = payload["postings"]
                self.title_exact_index = payload["title_exact_index"]
                self.title_token_index = payload["title_token_index"]
                return
            except (KeyError, pickle.PickleError, EOFError):
                pass

        document_lengths: list[int] = []
        df_counter: defaultdict[str, int] = defaultdict(int)
        postings_doc_ids: defaultdict[str, list[int]] = defaultdict(list)
        postings_freqs: defaultdict[str, list[int]] = defaultdict(list)
        title_exact_index: defaultdict[str, list[int]] = defaultdict(list)
        title_token_index: defaultdict[str, list[int]] = defaultdict(list)

        for doc_index, chunk in enumerate(self.chunks):
            tokens = _tokenize(f"{chunk.title} {chunk.text}")
            title_tokens = set(_tokenize(chunk.title))
            term_freqs = Counter(tokens)
            document_lengths.append(len(tokens))

            title_exact_index[chunk.title.lower()].append(doc_index)
            for title_token in title_tokens:
                title_token_index[title_token].append(doc_index)

            for token, freq in term_freqs.items():
                df_counter[token] += 1
                postings_doc_ids[token].append(doc_index)
                postings_freqs[token].append(freq)

        self.avg_doc_len = sum(document_lengths) / max(len(document_lengths), 1)
        self.doc_freq = dict(df_counter)
        self.doc_lengths = document_lengths
        self.postings = {
            token: (postings_doc_ids[token], postings_freqs[token])
            for token in postings_doc_ids
        }
        self.title_exact_index = dict(title_exact_index)
        self.title_token_index = dict(title_token_index)

        if self.cache_path is not None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": BM25_CACHE_VERSION,
                "avg_doc_len": self.avg_doc_len,
                "doc_freq": self.doc_freq,
                "doc_lengths": self.doc_lengths,
                "postings": self.postings,
                "title_exact_index": self.title_exact_index,
                "title_token_index": self.title_token_index,
            }
            with _open_binary(self.cache_path, "wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _score(
        self,
        plan: QueryPlan,
        doc_index: int,
        term_freqs: dict[str, int],
        doc_len: int,
    ) -> float:
        score = 0.0
        for index, token_set in enumerate(plan.token_sets):
            weight = 1.0 if index == 0 else 0.85
            score += weight * self._bm25_score(token_set, term_freqs, doc_len)

        chunk = self.chunks[doc_index]
        title_lower = chunk.title.lower()
        text_lower = chunk.text.lower()
        title_tokens = set(_tokenize(chunk.title))

        title_overlap = len(set(plan.token_sets[0]) & title_tokens)
        score += 0.5 * title_overlap

        for entity in plan.entities:
            entity_lower = entity.lower()
            if title_lower == entity_lower:
                score += 4.0
            elif entity_lower in title_lower:
                score += 2.5
            elif entity_lower in text_lower:
                score += 0.5

        attribute_hits = sum(1 for term in plan.attribute_terms if term in text_lower or term in title_lower)
        score += 0.2 * attribute_hits
        return score

    def _bm25_score(self, query_tokens: list[str], term_freqs: dict[str, int], doc_len: int) -> float:
        k1 = 1.2
        b = 0.75
        score = 0.0
        doc_count = len(self.chunks)
        for token in query_tokens:
            freq = term_freqs.get(token, 0)
            if freq == 0:
                continue
            df = self.doc_freq.get(token, 0)
            idf = math.log(1 + (doc_count - df + 0.5) / (df + 0.5))
            norm = freq + k1 * (1 - b + b * (doc_len / max(self.avg_doc_len, 1.0)))
            score += idf * ((freq * (k1 + 1)) / norm)
        return score


def _build_query_plan(query: str) -> QueryPlan:
    entities = _extract_entities(query)
    attribute_terms = _extract_attribute_terms(query)
    token_sets: list[list[str]] = []

    base_tokens = _tokenize(query)
    if base_tokens:
        token_sets.append(base_tokens)

    for entity in entities[:2]:
        expanded = _tokenize(f"{entity} {' '.join(attribute_terms)}")
        if expanded and expanded not in token_sets:
            token_sets.append(expanded)

    return QueryPlan(
        raw_query=query,
        entities=entities,
        attribute_terms=attribute_terms,
        token_sets=token_sets,
    )


def _extract_entities(query: str) -> list[str]:
    entities: list[str] = []
    for match in ENTITY_PATTERN.findall(query):
        cleaned = _clean_entity_match(match)
        if cleaned not in entities:
            entities.append(cleaned)
    return entities


def _extract_attribute_terms(query: str) -> list[str]:
    query_lower = query.lower()
    for trigger, terms in ATTRIBUTE_HINTS.items():
        if trigger in query_lower:
            return terms
    if "same nationality" in query_lower:
        return ATTRIBUTE_HINTS["nationality"]
    if "nationality" in query_lower:
        return ATTRIBUTE_HINTS["nationality"]
    if "older" in query_lower or "younger" in query_lower:
        return ATTRIBUTE_HINTS["age"]
    return []


def _tokenize(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_PATTERN.findall(text)
        if token and token.lower() not in STOPWORDS
    ]


def _clean_entity_match(match: str) -> str:
    parts = match.split()
    while parts and parts[0].lower() in QUESTION_STARTERS:
        parts = parts[1:]
    return " ".join(parts)


def _open_binary(path: Path, mode: str):
    if path.suffix == ".gz":
        return gzip.open(path, mode)
    return path.open(mode)
