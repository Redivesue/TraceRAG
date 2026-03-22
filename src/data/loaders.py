"""Dataset loaders for HotpotQA source files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class HotpotParagraph:
    title: str
    sentences: list[str]


@dataclass
class HotpotExample:
    example_id: str
    question: str
    answer: str | None
    level: str | None
    qtype: str | None
    supporting_facts: list[tuple[str, int]]
    context: list[HotpotParagraph]


def load_hotpot_examples(path: str | Path, limit: int | None = None) -> list[HotpotExample]:
    """Load HotpotQA examples from a JSON file."""
    source_path = Path(path)
    with source_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    examples: list[HotpotExample] = []
    for row in payload[:limit]:
        examples.append(_parse_hotpot_example(row))
    return examples


def iter_hotpot_examples(path: str | Path):
    """Yield HotpotQA examples one by one."""
    for example in load_hotpot_examples(path):
        yield example


def _parse_hotpot_example(row: dict[str, Any]) -> HotpotExample:
    context = [
        HotpotParagraph(title=title, sentences=list(sentences))
        for title, sentences in row["context"]
    ]
    supporting_facts = [
        (title, int(sentence_id))
        for title, sentence_id in row.get("supporting_facts", [])
    ]
    return HotpotExample(
        example_id=row["_id"],
        question=row["question"],
        answer=row.get("answer"),
        level=row.get("level"),
        qtype=row.get("type"),
        supporting_facts=supporting_facts,
        context=context,
    )
