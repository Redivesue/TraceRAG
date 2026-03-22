"""Transformer-backed semantic refinement over a small candidate shortlist."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

try:  # pragma: no cover - optional dependency
    import torch
    from transformers import AutoModel, AutoTokenizer
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None
    AutoModel = None
    AutoTokenizer = None

from src.core.schemas import RetrievalCandidate


class SemanticRefiner:
    def __init__(self, model_name: str, shortlist_size: int = 16) -> None:
        self.model_name = model_name
        self.shortlist_size = shortlist_size
        self._tokenizer = None
        self._model = None
        self._disabled = False
        self._embedding_cache: OrderedDict[str, np.ndarray] = OrderedDict()

    @property
    def available(self) -> bool:
        return (
            torch is not None
            and AutoModel is not None
            and AutoTokenizer is not None
            and not self._disabled
        )

    def refine(
        self,
        query: str,
        candidates: list[RetrievalCandidate],
    ) -> list[RetrievalCandidate]:
        if not self.available or not candidates:
            return candidates

        shortlist = candidates[: self.shortlist_size]
        try:
            query_vector = self._encode_texts([query])[0]
            texts = [_candidate_text(item) for item in shortlist]
            doc_vectors = self._encode_candidate_texts(shortlist, texts)
            semantic_scores = doc_vectors @ query_vector
        except Exception:
            self._disabled = True
            return candidates

        reranked = []
        for item, semantic_score in zip(shortlist, semantic_scores):
            combined = float(semantic_score) + (0.15 * item.score)
            reranked.append(
                RetrievalCandidate(
                    chunk=item.chunk,
                    score=combined,
                    source_method=f"{item.source_method}+semantic",
                )
            )
        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked + candidates[self.shortlist_size :]

    def _encode_candidate_texts(
        self,
        candidates: list[RetrievalCandidate],
        texts: list[str],
    ) -> np.ndarray:
        vectors: list[np.ndarray | None] = []
        missing_texts: list[str] = []
        missing_positions: list[int] = []
        for index, candidate in enumerate(candidates):
            cached = self._embedding_cache.get(candidate.chunk.chunk_id)
            if cached is not None:
                self._embedding_cache.move_to_end(candidate.chunk.chunk_id)
                vectors.append(cached)
                continue
            vectors.append(None)
            missing_positions.append(index)
            missing_texts.append(texts[index])

        if missing_texts:
            new_vectors = self._encode_texts(missing_texts)
            for position, vector in zip(missing_positions, new_vectors):
                chunk_id = candidates[position].chunk.chunk_id
                self._embedding_cache[chunk_id] = vector
                self._embedding_cache.move_to_end(chunk_id)
                vectors[position] = vector
            while len(self._embedding_cache) > 4096:
                self._embedding_cache.popitem(last=False)

        return np.vstack(vectors)

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        self._ensure_model()
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        with torch.no_grad():
            model_output = self._model(**encoded)
        token_embeddings = model_output.last_hidden_state
        attention_mask = encoded["attention_mask"].unsqueeze(-1)
        masked = token_embeddings * attention_mask
        pooled = masked.sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)
        normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return normalized.cpu().numpy().astype(np.float32)

    def _ensure_model(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
        self._model = AutoModel.from_pretrained(self.model_name, local_files_only=True)
        self._model.eval()


def _candidate_text(candidate: RetrievalCandidate) -> str:
    snippet = candidate.chunk.text[:320]
    return f"{candidate.chunk.title}. {snippet}"
