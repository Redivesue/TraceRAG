"""Runtime helpers for bootstrapping the demo pipeline."""

from __future__ import annotations

from functools import lru_cache
import json
import pickle

from src.core.config import settings
from src.data.indexing import export_hotpot_chunks
from src.indexing.build_index import build_indexes
from src.generation.llm_generator import LLMAnswerGenerator
from src.pipeline.search_pipeline import SearchPipeline
from src.rerank.hosted import HostedReranker
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DENSE_INDEX_VERSION, DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.lazy import LazyRetriever
from src.retrieval.semantic_refiner import SemanticRefiner
from src.retrieval.title_fast import TitleFastRetriever


def load_index_manifest() -> dict[str, object]:
    cache_stem = settings.default_chunk_path.stem
    manifest_path = settings.index_output_dir / f"{cache_stem}.index_manifest.json"
    if not manifest_path.exists():
        return {
            "status": "synthesized",
            "manifest_path": str(manifest_path),
            "chunk_path": str(settings.default_chunk_path),
            "chunk_strategy": "paragraph",
            "encoder_backend": settings.dense_encoder_backend,
            "encoder_model": settings.dense_model_name,
            "paths": {
                "bm25_path": str(settings.index_output_dir / f"{cache_stem}.bm25.pkl.gz"),
                "title_path": str(settings.index_output_dir / f"{cache_stem}.title.pkl.gz"),
                "vector_meta_path": str(settings.index_output_dir / f"{cache_stem}.vector.meta.pkl"),
                "faiss_path": str(settings.index_output_dir / f"{cache_stem}.vector.faiss.index"),
            },
        }
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "status": "unreadable",
            "manifest_path": str(manifest_path),
        }


def get_runtime_metadata() -> dict[str, object]:
    manifest = load_index_manifest()
    return {
        "app_name": settings.app_name,
        "dataset_name": settings.dataset_name,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "dense_encoder_backend": settings.dense_encoder_backend,
        "dense_model_name": settings.dense_model_name,
        "preload_retrievers": settings.preload_retrievers,
        "prewarm_query_encoder": settings.prewarm_query_encoder,
        "default_chunk_path": str(settings.default_chunk_path),
        "index_manifest": manifest,
    }


def ensure_default_chunk_corpus() -> None:
    """Create the default chunk corpus, preferring HotpotQA train over dev."""
    if settings.default_chunk_path.exists():
        return
    if settings.raw_hotpot_train_path.exists():
        export_hotpot_chunks(
            source_path=settings.raw_hotpot_train_path,
            output_path=settings.default_chunk_path,
        )
        return
    if settings.raw_hotpot_dev_path.exists():
        export_hotpot_chunks(
            source_path=settings.raw_hotpot_dev_path,
            output_path=settings.default_chunk_path,
        )
        return
    raise FileNotFoundError(
        "Expected either HotpotQA train or dev data before bootstrapping the default chunk corpus."
    )


def ensure_retrieval_indexes() -> None:
    """Build retrieval assets once, then load them online."""
    ensure_default_chunk_corpus()
    cache_stem = settings.default_chunk_path.stem
    vector_meta = settings.index_output_dir / f"{cache_stem}.vector.meta.pkl"
    expected = [
        settings.index_output_dir / f"{cache_stem}.title.pkl.gz",
        settings.index_output_dir / f"{cache_stem}.bm25.pkl.gz",
        vector_meta,
        settings.index_output_dir / f"{cache_stem}.vector.matrix.npy",
    ]
    if all(path.exists() for path in expected):
        try:
            with vector_meta.open("rb") as handle:
                payload = pickle.load(handle)
            meta_backend = str(payload.get("encoder_backend", "hash"))
            meta_model = str(payload.get("model_name", ""))
            if payload.get("version") == DENSE_INDEX_VERSION and (
                meta_backend == settings.dense_encoder_backend
                and (
                    meta_backend != "transformer"
                    or meta_model == settings.dense_model_name
                )
            ):
                return
        except (OSError, pickle.PickleError, EOFError, TypeError, ValueError):
            pass
    if not settings.auto_build_indexes:
        raise FileNotFoundError(
            "Retrieval indexes are missing. Run `python3 -m src.indexing.build_index` first."
        )
    source_path = (
        settings.raw_hotpot_train_path
        if settings.raw_hotpot_train_path.exists()
        else settings.raw_hotpot_dev_path
    )
    build_indexes(
        source_path=source_path,
        chunk_path=settings.default_chunk_path,
        index_dir=settings.index_output_dir,
        vector_dim=settings.dense_vector_dim,
        encoder_backend=settings.dense_encoder_backend,
        model_name=settings.dense_model_name,
        encode_batch_size=settings.dense_encode_batch_size,
    )


@lru_cache(maxsize=1)
def build_pipeline() -> SearchPipeline:
    """Construct a singleton search pipeline for API and UI usage."""
    ensure_retrieval_indexes()
    settings.index_output_dir.mkdir(parents=True, exist_ok=True)
    cache_stem = settings.default_chunk_path.stem
    if settings.preload_retrievers:
        sparse_retriever = BM25Retriever(
            corpus_path=settings.default_chunk_path,
            cache_path=settings.index_output_dir / f"{cache_stem}.bm25.pkl.gz",
        )
        dense_retriever = DenseRetriever(
            corpus_path=settings.default_chunk_path,
            cache_path=settings.index_output_dir / f"{cache_stem}.vector",
            vector_dim=settings.dense_vector_dim,
            encoder_backend=settings.dense_encoder_backend,
            model_name=settings.dense_model_name,
            encode_batch_size=settings.dense_encode_batch_size,
        )
    else:
        sparse_retriever = LazyRetriever(
            lambda: BM25Retriever(
                corpus_path=settings.default_chunk_path,
                cache_path=settings.index_output_dir / f"{cache_stem}.bm25.pkl.gz",
            )
        )
        dense_retriever = LazyRetriever(
            lambda: DenseRetriever(
                corpus_path=settings.default_chunk_path,
                cache_path=settings.index_output_dir / f"{cache_stem}.vector",
                vector_dim=settings.dense_vector_dim,
                encoder_backend=settings.dense_encoder_backend,
                model_name=settings.dense_model_name,
                encode_batch_size=settings.dense_encode_batch_size,
            )
        )
    retriever = HybridRetriever(
        title_retriever=TitleFastRetriever(
            corpus_path=settings.default_chunk_path,
            cache_path=settings.index_output_dir / f"{cache_stem}.title.pkl.gz",
        ),
        sparse_retriever=sparse_retriever,
        dense_retriever=dense_retriever,
        semantic_refiner=(
            SemanticRefiner(
                model_name=settings.semantic_model_name,
                shortlist_size=settings.semantic_refiner_top_n,
            )
            if settings.semantic_refiner_enabled
            else None
        ),
    )
    reranker = HostedReranker()
    generator = LLMAnswerGenerator.from_settings(settings)
    if settings.prewarm_query_encoder:
        if isinstance(dense_retriever, DenseRetriever):
            dense_retriever.warmup_query_encoder()
        elif isinstance(dense_retriever, LazyRetriever):
            loaded = dense_retriever._ensure()  # noqa: SLF001
            if isinstance(loaded, DenseRetriever):
                loaded.warmup_query_encoder()
    return SearchPipeline(retriever=retriever, reranker=reranker, generator=generator)
