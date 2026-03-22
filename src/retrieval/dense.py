"""Offline-built dense retrieval with FAISS primary and pluggable encoders."""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np

from src.core.schemas import RetrievalCandidate
from src.data.corpus import load_chunks
from src.retrieval.base import Retriever
from src.retrieval.encoders import DenseEncoder

try:  # pragma: no cover - optional dependency
    import faiss  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    faiss = None

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
ENTITY_PATTERN = re.compile(r"(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)")
QUESTION_STARTERS = {"were", "what", "is", "are", "was", "who", "which", "did", "does", "do"}
DENSE_INDEX_VERSION = 3
DEFAULT_VECTOR_DIM = 128


class DenseRetriever(Retriever):
    def __init__(
        self,
        corpus_path: str | Path,
        cache_path: str | Path | None = None,
        vector_dim: int = DEFAULT_VECTOR_DIM,
        encoder_backend: str = "hash",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        encode_batch_size: int = 64,
    ) -> None:
        self.corpus_path = Path(corpus_path)
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self.vector_dim = vector_dim
        self.encoder_backend = encoder_backend.strip().lower()
        self.model_name = model_name
        self.encode_batch_size = encode_batch_size
        self.chunks = load_chunks(self.corpus_path)
        self.query_cache: dict[tuple[str, int], list[RetrievalCandidate]] = {}
        self.title_exact_index: dict[str, list[int]] = {}
        self._matrix: np.ndarray | None = None
        self._faiss_index = None
        self._encoder: DenseEncoder | None = None
        self._load_or_build_index()

    def retrieve(self, query: str, top_k: int) -> list[RetrievalCandidate]:
        cache_key = (query, top_k)
        cached = self.query_cache.get(cache_key)
        if cached is not None:
            return cached

        query_vector = self._encoder_for_query().encode_text(query)
        if np.linalg.norm(query_vector) == 0:
            return []

        candidate_ids: list[int]
        candidate_scores: list[float]
        search_k = min(max(top_k * 4, 12), len(self.chunks))
        if self._faiss_index is not None:
            scores, ids = self._faiss_index.search(query_vector.reshape(1, -1), search_k)
            candidate_ids = [int(doc_id) for doc_id in ids[0] if int(doc_id) >= 0]
            candidate_scores = [float(score) for score in scores[0][: len(candidate_ids)]]
            source_method = "dense_faiss"
        else:
            assert self._matrix is not None
            scores = self._matrix @ query_vector
            for entity in _extract_entities(query):
                for doc_id in self.title_exact_index.get(entity.lower(), []):
                    scores[doc_id] += 0.35
            top_indices = np.argpartition(scores, -search_k)[-search_k:]
            ranked = top_indices[np.argsort(scores[top_indices])[::-1]]
            candidate_ids = ranked.tolist()
            candidate_scores = [float(scores[doc_id]) for doc_id in candidate_ids]
            source_method = "dense_numpy"

        results: list[RetrievalCandidate] = []
        seen_ids: set[str] = set()
        for doc_id, score in zip(candidate_ids, candidate_scores):
            if score <= 0:
                continue
            chunk = self.chunks[doc_id]
            if chunk.chunk_id in seen_ids:
                continue
            seen_ids.add(chunk.chunk_id)
            results.append(RetrievalCandidate(chunk=chunk, score=score, source_method=source_method))
            if len(results) >= top_k:
                break

        self.query_cache[cache_key] = results
        return results

    def _load_or_build_index(self) -> None:
        if self.cache_path is None:
            self._build_in_memory_index()
            return

        matrix_path, meta_path, faiss_path = _resolve_cache_paths(self.cache_path)
        if meta_path.exists():
            try:
                with meta_path.open("rb") as handle:
                    payload = pickle.load(handle)
                if payload.get("version") != DENSE_INDEX_VERSION:
                    raise KeyError("Unsupported dense index version")
                self.vector_dim = int(payload["vector_dim"])
                self.encoder_backend = str(payload.get("encoder_backend", self.encoder_backend))
                self.model_name = str(payload.get("model_name", self.model_name))
                self.title_exact_index = payload["title_exact_index"]
                if faiss is not None and faiss_path.exists():
                    self._faiss_index = faiss.read_index(str(faiss_path))
                    return
                if matrix_path.exists():
                    self._matrix = np.load(matrix_path, mmap_mode="r")
                    return
            except (KeyError, TypeError, pickle.PickleError, EOFError, ValueError):
                pass

        build_dense_index(
            corpus_path=self.corpus_path,
            cache_path=self.cache_path,
            vector_dim=self.vector_dim,
        )
        self._load_or_build_index()

    def _build_in_memory_index(self) -> None:
        encoder = DenseEncoder(
            backend=self.encoder_backend,
            vector_dim=self.vector_dim,
            model_name=self.model_name,
        )
        self.vector_dim = encoder.effective_dim
        matrix = np.zeros((len(self.chunks), self.vector_dim), dtype=np.float32)
        title_exact_index: dict[str, list[int]] = {}
        for doc_index, chunk in enumerate(self.chunks):
            matrix[doc_index] = encoder.encode_text(_dense_document_text(chunk.title, chunk.text))
            title_exact_index.setdefault(chunk.title.lower(), []).append(doc_index)
        self._matrix = matrix
        self.title_exact_index = title_exact_index

    def _encoder_for_query(self) -> DenseEncoder:
        if self._encoder is None:
            self._encoder = DenseEncoder(
                backend=self.encoder_backend,
                vector_dim=self.vector_dim,
                model_name=self.model_name,
            )
        return self._encoder

    def warmup_query_encoder(self) -> None:
        encoder = self._encoder_for_query()
        encoder.warmup()
        if self.encoder_backend == "transformer":
            encoder.encode_text("warmup query")


def build_dense_index(
    corpus_path: str | Path,
    cache_path: str | Path,
    vector_dim: int = DEFAULT_VECTOR_DIM,
    encoder_backend: str = "hash",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    encode_batch_size: int = 64,
) -> None:
    chunks = load_chunks(corpus_path)
    cache_base = Path(cache_path)
    matrix_path, meta_path, faiss_path = _resolve_cache_paths(cache_base)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    encoder = DenseEncoder(
        backend=encoder_backend,
        vector_dim=vector_dim,
        model_name=model_name,
    )
    effective_dim = encoder.effective_dim

    matrix = np.lib.format.open_memmap(
        matrix_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(chunks), effective_dim),
    )
    title_exact_index: dict[str, list[int]] = {}
    batch_texts: list[str] = []
    batch_indices: list[int] = []
    for doc_index, chunk in enumerate(chunks):
        batch_indices.append(doc_index)
        batch_texts.append(_dense_document_text(chunk.title, chunk.text))
        title_exact_index.setdefault(chunk.title.lower(), []).append(doc_index)
        if len(batch_texts) >= encode_batch_size:
            embeddings = encoder.encode_texts(batch_texts, batch_size=encode_batch_size)
            for target_index, vector in zip(batch_indices, embeddings):
                matrix[target_index] = vector
            batch_texts.clear()
            batch_indices.clear()
    if batch_texts:
        embeddings = encoder.encode_texts(batch_texts, batch_size=encode_batch_size)
        for target_index, vector in zip(batch_indices, embeddings):
            matrix[target_index] = vector
    del matrix

    payload = {
        "version": DENSE_INDEX_VERSION,
        "vector_dim": effective_dim,
        "encoder_backend": encoder_backend,
        "model_name": model_name,
        "title_exact_index": title_exact_index,
    }
    with meta_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if faiss is not None:  # pragma: no cover - optional dependency
        dense_matrix = np.load(matrix_path, mmap_mode="r")
        index = faiss.IndexFlatIP(effective_dim)
        index.add(np.asarray(dense_matrix, dtype=np.float32))
        faiss.write_index(index, str(faiss_path))


def _resolve_cache_paths(cache_path: Path) -> tuple[Path, Path, Path]:
    stem_name = cache_path.with_suffix("").name if cache_path.suffix in {".pkl", ".gz", ".index"} else cache_path.name
    base = cache_path.parent / stem_name
    return (
        base.parent / f"{base.name}.matrix.npy",
        base.parent / f"{base.name}.meta.pkl",
        base.parent / f"{base.name}.faiss.index",
    )


def _dense_document_text(title: str, text: str) -> str:
    return f"{title}. {title}. {text}"


def _extract_entities(query: str) -> list[str]:
    entities: list[str] = []
    for match in ENTITY_PATTERN.findall(query):
        cleaned = _clean_entity_match(match)
        if cleaned and cleaned not in entities:
            entities.append(cleaned)
    return entities


def _clean_entity_match(match: str) -> str:
    parts = match.split()
    while parts and parts[0].lower() in QUESTION_STARTERS:
        parts = parts[1:]
    return " ".join(parts)
