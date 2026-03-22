"""Offline build script for retrieval indexes."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.data.corpus import load_chunks
from src.data.indexing import export_hotpot_chunks
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import build_dense_index
from src.retrieval.title_fast import TitleFastRetriever


def build_indexes(
    source_path: str | Path,
    chunk_path: str | Path,
    index_dir: str | Path,
    vector_dim: int = 128,
    encoder_backend: str = "hash",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    encode_batch_size: int = 64,
    limit: int | None = None,
) -> dict[str, str | int]:
    source = Path(source_path)
    chunk_output = Path(chunk_path)
    output_dir = Path(index_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not chunk_output.exists():
        export_hotpot_chunks(source_path=source, output_path=chunk_output, limit=limit)

    chunks = load_chunks(chunk_output)
    stem = chunk_output.stem

    metadata_path = output_dir / f"{stem}.metadata.json"
    metadata_payload = [
        {
            "doc_index": doc_index,
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "title": chunk.title,
        }
        for doc_index, chunk in enumerate(chunks)
    ]
    metadata_path.write_text(json.dumps(metadata_payload, ensure_ascii=False), encoding="utf-8")

    TitleFastRetriever(
        corpus_path=chunk_output,
        cache_path=output_dir / f"{stem}.title.pkl.gz",
    )
    BM25Retriever(
        corpus_path=chunk_output,
        cache_path=output_dir / f"{stem}.bm25.pkl.gz",
    )
    build_dense_index(
        corpus_path=chunk_output,
        cache_path=output_dir / f"{stem}.vector",
        vector_dim=vector_dim,
        encoder_backend=encoder_backend,
        model_name=model_name,
        encode_batch_size=encode_batch_size,
    )

    index_manifest_path = output_dir / f"{stem}.index_manifest.json"
    index_manifest = {
        "version": 1,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "source_path": str(source),
        "chunk_path": str(chunk_output),
        "chunk_count": len(chunks),
        "chunk_strategy": "paragraph",
        "encoder_backend": encoder_backend,
        "encoder_model": model_name if encoder_backend == "transformer" else "",
        "vector_dim": vector_dim,
        "faiss_enabled": (output_dir / f"{stem}.vector.faiss.index").exists(),
        "paths": {
            "metadata_path": str(metadata_path),
            "bm25_path": str(output_dir / f"{stem}.bm25.pkl.gz"),
            "title_path": str(output_dir / f"{stem}.title.pkl.gz"),
            "vector_matrix_path": str(output_dir / f"{stem}.vector.matrix.npy"),
            "vector_meta_path": str(output_dir / f"{stem}.vector.meta.pkl"),
            "faiss_path": str(output_dir / f"{stem}.vector.faiss.index"),
        },
    }
    index_manifest_path.write_text(
        json.dumps(index_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "chunk_count": len(chunks),
        "chunk_path": str(chunk_output),
        "metadata_path": str(metadata_path),
        "bm25_path": str(output_dir / f"{stem}.bm25.pkl.gz"),
        "title_path": str(output_dir / f"{stem}.title.pkl.gz"),
        "vector_matrix_path": str(output_dir / f"{stem}.vector.matrix.npy"),
        "vector_meta_path": str(output_dir / f"{stem}.vector.meta.pkl"),
        "faiss_path": str(output_dir / f"{stem}.vector.faiss.index"),
        "index_manifest_path": str(index_manifest_path),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build offline retrieval indexes.")
    parser.add_argument(
        "--source",
        required=True,
        help="Path to the HotpotQA train JSON file.",
    )
    parser.add_argument(
        "--chunks",
        required=True,
        help="Path to the exported chunk JSONL file.",
    )
    parser.add_argument(
        "--index-dir",
        required=True,
        help="Directory for BM25/vector/title indexes.",
    )
    parser.add_argument("--vector-dim", type=int, default=128, help="Dense vector dimension.")
    parser.add_argument(
        "--encoder-backend",
        default="hash",
        choices=["hash", "transformer"],
        help="Dense encoder backend for the FAISS index.",
    )
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Local transformer model name when using the transformer backend.",
    )
    parser.add_argument(
        "--encode-batch-size",
        type=int,
        default=64,
        help="Batch size for offline dense encoding.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional example limit.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    result = build_indexes(
        source_path=args.source,
        chunk_path=args.chunks,
        index_dir=args.index_dir,
        vector_dim=args.vector_dim,
        encoder_backend=args.encoder_backend,
        model_name=args.model_name,
        encode_batch_size=args.encode_batch_size,
        limit=args.limit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
