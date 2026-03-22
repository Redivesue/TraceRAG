"""End-to-end orchestration for grounded search QA."""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import replace
from uuid import uuid4

from src.core.schemas import SearchRequest, SearchResponse
from src.core.telemetry import log_event
from src.generation.base import AnswerGenerator
from src.rerank.base import Reranker
from src.retrieval.base import Retriever


class SearchPipeline:
    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        generator: AnswerGenerator,
        retrieve_multiplier: int = 2,
        min_retrieve_k: int = 6,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator
        self.retrieve_multiplier = retrieve_multiplier
        self.min_retrieve_k = min_retrieve_k
        self.response_cache: OrderedDict[tuple[str, int, bool], SearchResponse] = OrderedDict()

    def run(self, request: SearchRequest) -> SearchResponse:
        request_id = uuid4().hex
        cache_key = (request.query, request.top_k, request.debug)
        cached = self.response_cache.get(cache_key)
        if cached is not None:
            self.response_cache.move_to_end(cache_key)
            cached_timings = dict(cached.timings)
            cached_timings["cache_hit"] = 1.0
            cached_trace = dict(cached.trace)
            cached_trace["cache_hit"] = 1
            cached_trace["request_id"] = request_id
            log_event(
                "search_request",
                request_id=request_id,
                query=request.query,
                cache_hit=True,
                generator_mode=cached.generator_mode,
                path_reason=cached.trace.get("path_reason", ""),
                top_k=request.top_k,
                total_seconds=cached_timings.get("total_seconds", 0.0),
            )
            return replace(cached, request_id=request_id, timings=cached_timings, trace=cached_trace)

        total_start = time.perf_counter()
        retrieve_k = max(request.top_k * self.retrieve_multiplier, self.min_retrieve_k)

        retrieval_start = time.perf_counter()
        retrieved = self.retriever.retrieve(query=request.query, top_k=retrieve_k)
        retrieval_elapsed = time.perf_counter() - retrieval_start

        rerank_start = time.perf_counter()
        reranked = self.reranker.rerank(
            query=request.query,
            candidates=retrieved,
            top_k=request.top_k,
        )
        rerank_elapsed = time.perf_counter() - rerank_start

        generation_start = time.perf_counter()
        generated = self.generator.generate(query=request.query, evidence=reranked)
        generation_elapsed = time.perf_counter() - generation_start

        total_elapsed = time.perf_counter() - total_start
        response = SearchResponse(
            request_id=request_id,
            answer=generated.answer,
            citations=generated.citations,
            retrieved=retrieved if request.debug else [],
            reranked=reranked if request.debug else [],
            generator_mode=generated.generator_mode,
            timings={
                "retrieval_seconds": round(retrieval_elapsed, 4),
                "rerank_seconds": round(rerank_elapsed, 4),
                "generation_seconds": round(generation_elapsed, 4),
                "total_seconds": round(total_elapsed, 4),
            },
            trace={
                "request_id": request_id,
                "query": request.query,
                "selected_path": "fast_path" if generated.generator_mode == "fast_path" else "slow_path",
                "path_reason": generated.path_reason,
                "retrieved_candidate_count": len(retrieved),
                "reranked_candidate_count": len(reranked),
                "generator_mode": generated.generator_mode,
                "cache_hit": 0,
            },
        )
        log_event(
            "search_request",
            request_id=request_id,
            query=request.query,
            cache_hit=False,
            selected_path=response.trace.get("selected_path", ""),
            path_reason=generated.path_reason,
            retrieved_candidate_count=len(retrieved),
            reranked_candidate_count=len(reranked),
            generator_mode=generated.generator_mode,
            retrieval_seconds=response.timings.get("retrieval_seconds", 0.0),
            rerank_seconds=response.timings.get("rerank_seconds", 0.0),
            generation_seconds=response.timings.get("generation_seconds", 0.0),
            total_seconds=response.timings.get("total_seconds", 0.0),
        )
        self.response_cache[cache_key] = response
        self.response_cache.move_to_end(cache_key)
        while len(self.response_cache) > 128:
            self.response_cache.popitem(last=False)
        return response
