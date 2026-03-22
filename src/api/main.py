"""FastAPI entrypoint for the search assistant."""

from dataclasses import asdict

try:
    from fastapi import FastAPI
except ModuleNotFoundError:  # pragma: no cover - local fallback when deps are missing
    FastAPI = None

from src.core.runtime import build_pipeline, get_runtime_metadata
from src.core.schemas import SearchRequest

pipeline = build_pipeline()
app = FastAPI(title="Perplexity Lite") if FastAPI is not None else None


def health() -> dict:
    return {
        "status": "ok",
        "generator_mode": pipeline.generator.provider_name,
        "runtime": get_runtime_metadata(),
    }


def search(request: SearchRequest) -> dict:
    response = pipeline.run(request)
    return {
        "request_id": response.request_id,
        "answer": response.answer,
        "generator_mode": response.generator_mode,
        "timings": response.timings,
        "trace": response.trace,
        "citations": [asdict(citation) for citation in response.citations],
        "retrieved": [asdict(candidate) for candidate in response.retrieved],
        "reranked": [asdict(result) for result in response.reranked],
    }


if app is not None:
    app.get("/health")(health)
    app.post("/search")(search)
