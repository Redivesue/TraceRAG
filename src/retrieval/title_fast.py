"""Small exact-title retriever for common entity questions."""

from __future__ import annotations

import gzip
import pickle
import re
from collections import defaultdict
from pathlib import Path

from src.core.schemas import Chunk, RetrievalCandidate
from src.data.corpus import load_chunks
from src.retrieval.base import Retriever

ENTITY_PATTERN = re.compile(r"(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)")
QUESTION_STARTERS = {"were", "what", "is", "are", "was", "who", "which", "did", "does", "do"}


class TitleFastRetriever(Retriever):
    def __init__(self, corpus_path: str | Path, cache_path: str | Path) -> None:
        self.corpus_path = Path(corpus_path)
        self.cache_path = Path(cache_path)
        self.title_map: dict[str, list[Chunk]] = {}
        self._load_or_build()

    def retrieve(self, query: str, top_k: int) -> list[RetrievalCandidate]:
        entities = _extract_entities(query)
        if not entities:
            return []

        results: list[RetrievalCandidate] = []
        seen_ids: set[str] = set()
        for entity in entities[:2]:
            for chunk in self.title_map.get(entity.lower(), []):
                if chunk.chunk_id in seen_ids:
                    continue
                results.append(
                    RetrievalCandidate(chunk=chunk, score=100.0, source_method="title_fast_path")
                )
                seen_ids.add(chunk.chunk_id)
                if len(results) >= top_k:
                    return results
        return results

    def _load_or_build(self) -> None:
        if self.cache_path.exists():
            with _open_binary(self.cache_path, "rb") as handle:
                payload = pickle.load(handle)
            self.title_map = {
                title: [Chunk(**chunk_payload) for chunk_payload in chunk_payloads]
                for title, chunk_payloads in payload.items()
            }
            return

        grouped: defaultdict[str, list[Chunk]] = defaultdict(list)
        for chunk in load_chunks(self.corpus_path):
            key = chunk.title.lower()
            if len(grouped[key]) < 1:
                grouped[key].append(chunk)

        self.title_map = dict(grouped)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            title: [
                {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "title": chunk.title,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                }
                for chunk in chunks
            ]
            for title, chunks in self.title_map.items()
        }
        with _open_binary(self.cache_path, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _extract_entities(query: str) -> list[str]:
    entities: list[str] = []
    for match in ENTITY_PATTERN.findall(query):
        cleaned = _clean_entity_match(match)
        if cleaned not in entities:
            entities.append(cleaned)
    return entities


def _clean_entity_match(match: str) -> str:
    parts = match.split()
    while parts and parts[0].lower() in QUESTION_STARTERS:
        parts = parts[1:]
    return " ".join(parts)


def _open_binary(path: Path, mode: str):
    if path.suffix == ".gz":
        return gzip.open(path, mode)
    return path.open(mode)
