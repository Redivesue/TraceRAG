"""Pluggable dense encoders for offline vector indexing and online query encoding."""

from __future__ import annotations

import hashlib
import os
import re

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

try:  # pragma: no cover - optional dependency
    import torch
    from transformers import AutoModel, AutoTokenizer
    from transformers.utils import logging as transformers_logging
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None
    AutoModel = None
    AutoTokenizer = None
    transformers_logging = None

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


class DenseEncoder:
    """Unified interface over hash and transformer text encoders."""

    def __init__(
        self,
        backend: str = "hash",
        vector_dim: int = 128,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.backend = backend.strip().lower()
        self.vector_dim = vector_dim
        self.model_name = model_name
        self._tokenizer = None
        self._model = None

    @property
    def effective_dim(self) -> int:
        if self.backend != "transformer":
            return self.vector_dim
        self._ensure_transformer()
        return int(self._model.config.hidden_size)

    def encode_text(self, text: str) -> np.ndarray:
        return self.encode_texts([text])[0]

    def warmup(self) -> None:
        if self.backend == "transformer":
            self._ensure_transformer()

    def encode_texts(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        if self.backend == "hash":
            return np.vstack([_embed_hash(text, self.vector_dim) for text in texts])
        if self.backend == "transformer":
            return self._encode_transformer(texts=texts, batch_size=batch_size)
        raise ValueError(f"Unsupported dense encoder backend: {self.backend}")

    def _encode_transformer(self, texts: list[str], batch_size: int) -> np.ndarray:
        self._ensure_transformer()
        outputs: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                encoded = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=160,
                    return_tensors="pt",
                )
                model_output = self._model(**encoded)
                token_embeddings = model_output.last_hidden_state
                attention_mask = encoded["attention_mask"].unsqueeze(-1)
                pooled = (token_embeddings * attention_mask).sum(dim=1)
                pooled = pooled / attention_mask.sum(dim=1).clamp(min=1)
                normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
                outputs.append(normalized.cpu().numpy().astype(np.float32))
        return np.vstack(outputs)

    def _ensure_transformer(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        if (
            torch is None
            or AutoModel is None
            or AutoTokenizer is None
            or transformers_logging is None
        ):
            raise RuntimeError("Transformer backend requested but transformers/torch are unavailable.")
        transformers_logging.set_verbosity_error()
        try:
            torch.set_num_threads(min(4, os.cpu_count() or 1))
        except RuntimeError:
            pass
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
        self._model = AutoModel.from_pretrained(self.model_name, local_files_only=True)
        self._model.eval()


def _embed_hash(text: str, vector_dim: int) -> np.ndarray:
    vector = np.zeros(vector_dim, dtype=np.float32)
    for token in TOKEN_PATTERN.findall(text.lower()):
        digest = hashlib.md5(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % vector_dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm
    return vector
