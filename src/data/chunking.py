"""Chunk builders for HotpotQA contexts."""

from __future__ import annotations

from dataclasses import asdict

from src.core.schemas import Chunk
from src.data.loaders import HotpotExample


def build_chunks_from_examples(
    examples: list[HotpotExample],
    include_metadata: bool = True,
) -> list[Chunk]:
    """Flatten HotpotQA paragraphs into retrieval chunks."""
    chunks: list[Chunk] = []
    for example in examples:
        chunks.extend(build_chunks_from_example(example, include_metadata=include_metadata))
    return chunks


def build_chunks_from_example(
    example: HotpotExample,
    include_metadata: bool = True,
) -> list[Chunk]:
    """Convert one HotpotQA example into paragraph-level chunks."""
    support_map = _build_support_map(example)
    chunks: list[Chunk] = []

    for paragraph_index, paragraph in enumerate(example.context):
        text = " ".join(sentence.strip() for sentence in paragraph.sentences if sentence.strip())
        if not text:
            continue

        supporting_ids = support_map.get(paragraph.title, [])
        metadata = {
            "source_dataset": "hotpotqa",
            "example_id": example.example_id,
            "paragraph_index": str(paragraph_index),
            "question": example.question,
            "question_type": example.qtype or "",
            "level": example.level or "",
            "supporting_fact_hint": ",".join(str(idx) for idx in supporting_ids),
        }
        if not include_metadata:
            metadata = {}

        chunks.append(
            Chunk(
                chunk_id=f"{example.example_id}_p{paragraph_index}",
                doc_id=example.example_id,
                title=paragraph.title,
                text=text,
                metadata=metadata,
            )
        )
    return chunks


def chunk_to_record(chunk: Chunk) -> dict[str, object]:
    """Serialize a Chunk dataclass into a JSONL-friendly record."""
    return asdict(chunk)


def _build_support_map(example: HotpotExample) -> dict[str, list[int]]:
    support_map: dict[str, list[int]] = {}
    for title, sentence_id in example.supporting_facts:
        support_map.setdefault(title, []).append(sentence_id)
    return support_map
