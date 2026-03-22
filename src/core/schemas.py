"""Shared schemas for the retrieval and grounded QA pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SearchRequest:
    query: str
    top_k: int = 5
    debug: bool = False


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class RetrievalCandidate:
    chunk: Chunk
    score: float
    source_method: str


@dataclass
class RerankResult:
    chunk: Chunk
    retrieval_score: float
    rerank_score: float
    rank: int
    source_method: str


@dataclass
class Citation:
    label: int
    chunk_id: str
    title: str


@dataclass
class GeneratedAnswer:
    answer: str
    citations: list[Citation]
    generator_mode: str = "offline"
    path_reason: str = ""


@dataclass
class SearchResponse:
    request_id: str
    answer: str
    citations: list[Citation]
    retrieved: list[RetrievalCandidate]
    reranked: list[RerankResult]
    generator_mode: str = "offline"
    timings: dict[str, float] = field(default_factory=dict)
    trace: dict[str, str | int | float] = field(default_factory=dict)
