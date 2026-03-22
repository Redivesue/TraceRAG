"""CLI evaluation for retrieval and citation grounding."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from src.core.runtime import build_pipeline
from src.core.schemas import SearchRequest
from src.data.loaders import load_hotpot_examples


@dataclass
class EvalResult:
    evaluated_examples: int
    recall_at_5: float
    recall_at_10: float
    mrr: float
    citation_hit_rate: float


def evaluate_pipeline(eval_path: str | Path, limit: int | None = None) -> EvalResult:
    examples = load_hotpot_examples(eval_path, limit=limit)
    pipeline = build_pipeline()
    recall_hits_5 = 0
    recall_hits_10 = 0
    mrr_total = 0.0
    citation_hits = 0

    for example in examples:
        support_titles = {title for title, _ in example.supporting_facts}
        response_5 = pipeline.run(SearchRequest(query=example.question, top_k=5, debug=True))
        response_10 = pipeline.run(SearchRequest(query=example.question, top_k=10, debug=True))

        retrieved_titles_5 = [item.chunk.title for item in response_5.reranked]
        retrieved_titles_10 = [item.chunk.title for item in response_10.reranked]
        citation_titles = {citation.title for citation in response_5.citations}

        if support_titles & set(retrieved_titles_5):
            recall_hits_5 += 1
        if support_titles & set(retrieved_titles_10):
            recall_hits_10 += 1
        if support_titles & citation_titles:
            citation_hits += 1

        reciprocal_rank = 0.0
        for rank, title in enumerate(retrieved_titles_10, start=1):
            if title in support_titles:
                reciprocal_rank = 1 / rank
                break
        mrr_total += reciprocal_rank

    count = max(len(examples), 1)
    return EvalResult(
        evaluated_examples=len(examples),
        recall_at_5=recall_hits_5 / count,
        recall_at_10=recall_hits_10 / count,
        mrr=mrr_total / count,
        citation_hit_rate=citation_hits / count,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and citation grounding.")
    parser.add_argument(
        "--eval-path",
        default="artifacts/raw/hotpotqa/hotpot_dev_distractor_v1.json",
        help="Path to the HotpotQA evaluation split.",
    )
    parser.add_argument("--limit", type=int, default=100, help="Optional example limit.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    result = evaluate_pipeline(eval_path=args.eval_path, limit=args.limit)
    print(f"evaluated_examples={result.evaluated_examples}")
    print(f"recall@5={result.recall_at_5:.4f}")
    print(f"recall@10={result.recall_at_10:.4f}")
    print(f"mrr={result.mrr:.4f}")
    print(f"citation_hit_rate={result.citation_hit_rate:.4f}")


if __name__ == "__main__":
    main()
