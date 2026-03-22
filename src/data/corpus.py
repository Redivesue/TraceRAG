"""Utilities for loading exported chunk corpora."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from src.core.schemas import Chunk


def load_chunks(path: str | Path) -> list[Chunk]:
    """Load chunk records from a JSONL file."""
    return _load_chunks_cached(str(Path(path).resolve()))


@lru_cache(maxsize=2)
def _load_chunks_cached(resolved_path: str) -> list[Chunk]:
    source_path = Path(resolved_path)
    chunks: list[Chunk] = []
    with source_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            chunks.append(
                Chunk(
                    chunk_id=payload["chunk_id"],
                    doc_id=payload["doc_id"],
                    title=payload["title"],
                    text=payload["text"],
                    metadata=payload.get("metadata", {}),
                )
            )
    return chunks
