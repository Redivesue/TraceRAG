"""Preprocessing script entrypoints for HotpotQA exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.chunking import build_chunks_from_examples, chunk_to_record
from src.data.loaders import load_hotpot_examples


def export_hotpot_chunks(
    source_path: str | Path,
    output_path: str | Path,
    limit: int | None = None,
) -> int:
    """Load a HotpotQA file and export retrieval chunks as JSONL."""
    examples = load_hotpot_examples(source_path, limit=limit)
    chunks = build_chunks_from_examples(examples)
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    with target_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk_to_record(chunk), ensure_ascii=False) + "\n")

    return len(chunks)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export HotpotQA retrieval chunks.")
    parser.add_argument("--source", required=True, help="Path to a HotpotQA JSON file.")
    parser.add_argument("--output", required=True, help="Path to output JSONL chunk file.")
    parser.add_argument("--limit", type=int, default=None, help="Optional example limit.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    chunk_count = export_hotpot_chunks(
        source_path=args.source,
        output_path=args.output,
        limit=args.limit,
    )
    print(f"exported_chunks={chunk_count}")


if __name__ == "__main__":
    main()
