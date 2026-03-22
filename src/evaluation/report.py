"""Generate a unified evaluation report for retrieval, grounding, and latency."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.core.runtime import build_pipeline
from src.core.schemas import SearchRequest
from src.data.loaders import load_hotpot_examples
from src.evaluation.regression_cases import REGRESSION_CASES, RegressionCase


@dataclass
class RetrievalMetrics:
    evaluated_examples: int
    recall_at_5: float
    recall_at_10: float
    mrr: float
    citation_hit_rate: float
    citation_precision: float
    answer_grounded_rate: float


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _clear_response_cache(pipeline) -> None:
    cache = getattr(pipeline, "response_cache", None)
    if cache is not None:
        cache.clear()


def _evaluate_retrieval(eval_path: str | Path, limit: int, pipeline) -> RetrievalMetrics:
    examples = load_hotpot_examples(eval_path, limit=limit)
    recall_hits_5 = 0
    recall_hits_10 = 0
    mrr_total = 0.0
    citation_hits = 0
    citation_precision_total = 0.0
    grounded_hits = 0

    for example in examples:
        support_titles = {title for title, _ in example.supporting_facts}
        response_5 = pipeline.run(SearchRequest(query=example.question, top_k=5, debug=True))
        response_10 = pipeline.run(SearchRequest(query=example.question, top_k=10, debug=True))

        retrieved_titles_5 = [item.chunk.title for item in response_5.reranked]
        retrieved_titles_10 = [item.chunk.title for item in response_10.reranked]
        citation_titles = [citation.title for citation in response_5.citations]
        citation_title_set = set(citation_titles)

        if support_titles & set(retrieved_titles_5):
            recall_hits_5 += 1
        if support_titles & set(retrieved_titles_10):
            recall_hits_10 += 1
        if support_titles & citation_title_set:
            citation_hits += 1
            grounded_hits += 1

        reciprocal_rank = 0.0
        for rank, title in enumerate(retrieved_titles_10, start=1):
            if title in support_titles:
                reciprocal_rank = 1 / rank
                break
        mrr_total += reciprocal_rank

        if citation_titles:
            matching = sum(1 for title in citation_titles if title in support_titles)
            citation_precision_total += matching / len(citation_titles)
        else:
            citation_precision_total += 0.0

    count = max(len(examples), 1)
    return RetrievalMetrics(
        evaluated_examples=len(examples),
        recall_at_5=recall_hits_5 / count,
        recall_at_10=recall_hits_10 / count,
        mrr=mrr_total / count,
        citation_hit_rate=citation_hits / count,
        citation_precision=citation_precision_total / count,
        answer_grounded_rate=grounded_hits / count,
    )


def _run_regression_case(case: RegressionCase, pipeline) -> dict[str, object]:
    response = pipeline.run(SearchRequest(query=case.query, top_k=5, debug=True))
    answer_lower = response.answer.lower()
    missing = [needle for needle in case.expected_substrings if needle.lower() not in answer_lower]
    forbidden = [needle for needle in case.forbidden_substrings if needle.lower() in answer_lower]
    mode_ok = case.expected_mode is None or response.generator_mode == case.expected_mode
    passed = not missing and not forbidden and mode_ok

    return {
        "name": case.name,
        "bucket": case.bucket,
        "query": case.query,
        "passed": passed,
        "generator_mode": response.generator_mode,
        "timings": dict(response.timings),
        "citation_count": len(response.citations),
        "used_titles": [citation.title for citation in response.citations],
        "reranked_titles": [item.chunk.title for item in response.reranked],
        "missing_expected_substrings": missing,
        "forbidden_substrings_found": forbidden,
        "mode_ok": mode_ok,
    }


def _summarize_regression(records: list[dict[str, object]]) -> dict[str, object]:
    by_bucket: dict[str, list[dict[str, object]]] = defaultdict(list)
    fast_path_hits = 0
    for record in records:
        by_bucket[str(record["bucket"])].append(record)
        if record["generator_mode"] == "fast_path":
            fast_path_hits += 1

    bucket_summary: dict[str, object] = {}
    for bucket, bucket_records in sorted(by_bucket.items()):
        totals = [float(item["timings"].get("total_seconds", 0.0)) for item in bucket_records]
        grounded = [1 for item in bucket_records if int(item["citation_count"]) > 0]
        passed = [1 for item in bucket_records if bool(item["passed"])]
        fast_hits = [1 for item in bucket_records if item["generator_mode"] == "fast_path"]
        bucket_summary[bucket] = {
            "count": len(bucket_records),
            "pass_rate": sum(passed) / max(len(bucket_records), 1),
            "grounded_rate": sum(grounded) / max(len(bucket_records), 1),
            "fast_path_hit_rate": sum(fast_hits) / max(len(bucket_records), 1),
            "avg_total_seconds": statistics.mean(totals) if totals else 0.0,
        }

    totals = [float(item["timings"].get("total_seconds", 0.0)) for item in records]
    fast_values = [
        float(item["timings"].get("total_seconds", 0.0))
        for item in records
        if item["generator_mode"] == "fast_path"
    ]
    slow_values = [
        float(item["timings"].get("total_seconds", 0.0))
        for item in records
        if item["generator_mode"] != "fast_path"
    ]
    return {
        "count": len(records),
        "pass_rate": sum(1 for item in records if item["passed"]) / max(len(records), 1),
        "fast_path_hit_rate": fast_path_hits / max(len(records), 1),
        "bucket_summary": bucket_summary,
        "avg_total_seconds": statistics.mean(totals) if totals else 0.0,
        "fast_path_benefit": {
            "fast_case_count": len(fast_values),
            "slow_case_count": len(slow_values),
            "fast_avg_total_seconds": statistics.mean(fast_values) if fast_values else 0.0,
            "slow_avg_total_seconds": statistics.mean(slow_values) if slow_values else 0.0,
            "speedup_ratio": (
                (statistics.mean(slow_values) / statistics.mean(fast_values))
                if fast_values and slow_values and statistics.mean(fast_values) > 0
                else 0.0
            ),
        },
        "cases": records,
    }


def _run_benchmark(pipeline, rounds: int, warmup: bool) -> dict[str, object]:
    records_by_mode: dict[str, list[float]] = defaultdict(list)
    stage_timings: dict[str, list[float]] = defaultdict(list)

    def run_round(record: bool) -> None:
        for case in REGRESSION_CASES:
            _clear_response_cache(pipeline)
            response = pipeline.run(SearchRequest(query=case.query, top_k=5, debug=False))
            if not record:
                continue
            total = float(response.timings.get("total_seconds", 0.0))
            records_by_mode[response.generator_mode].append(total)
            stage_timings["retrieval_seconds"].append(float(response.timings.get("retrieval_seconds", 0.0)))
            stage_timings["rerank_seconds"].append(float(response.timings.get("rerank_seconds", 0.0)))
            stage_timings["generation_seconds"].append(float(response.timings.get("generation_seconds", 0.0)))
            stage_timings["total_seconds"].append(total)

    if warmup:
        run_round(record=False)
    for _ in range(rounds):
        run_round(record=True)

    modes: dict[str, object] = {}
    for mode, values in sorted(records_by_mode.items()):
        modes[mode] = {
            "count": len(values),
            "mean_seconds": statistics.mean(values) if values else 0.0,
            "p50_seconds": _percentile(values, 0.50),
            "p95_seconds": _percentile(values, 0.95),
            "max_seconds": max(values) if values else 0.0,
        }

    stages: dict[str, object] = {}
    for name, values in stage_timings.items():
        stages[name] = {
            "count": len(values),
            "mean_seconds": statistics.mean(values) if values else 0.0,
            "p50_seconds": _percentile(values, 0.50),
            "p95_seconds": _percentile(values, 0.95),
            "max_seconds": max(values) if values else 0.0,
        }

    return {
        "rounds": rounds,
        "warmup": warmup,
        "modes": modes,
        "stages": stages,
    }


def _build_markdown(report: dict[str, object]) -> str:
    retrieval = report["retrieval"]
    regression = report["regression"]
    benchmark = report["benchmark"]
    bucket_summary = regression["bucket_summary"]

    lines = [
        "# Unified Evaluation Report",
        "",
        f"- Generated at: `{report['generated_at']}`",
        f"- Eval split: `{report['config']['eval_path']}`",
        f"- Eval limit: `{report['config']['eval_limit']}`",
        f"- Benchmark rounds: `{report['config']['benchmark_rounds']}`",
        "",
        "## Retrieval Metrics",
        "",
        f"- Evaluated examples: `{retrieval['evaluated_examples']}`",
        f"- Recall@5: `{retrieval['recall_at_5']:.4f}`",
        f"- Recall@10: `{retrieval['recall_at_10']:.4f}`",
        f"- MRR: `{retrieval['mrr']:.4f}`",
        f"- Citation hit rate: `{retrieval['citation_hit_rate']:.4f}`",
        f"- Citation precision: `{retrieval['citation_precision']:.4f}`",
        f"- Answer grounded rate: `{retrieval['answer_grounded_rate']:.4f}`",
        "",
        "## Regression Summary",
        "",
        f"- Case count: `{regression['count']}`",
        f"- Pass rate: `{regression['pass_rate']:.4f}`",
        f"- Fast-path hit rate: `{regression['fast_path_hit_rate']:.4f}`",
        f"- Average total latency: `{regression['avg_total_seconds']:.4f}s`",
        "",
        "## Fast Path Benefit",
        "",
        f"- Fast-path cases: `{regression['fast_path_benefit']['fast_case_count']}`",
        f"- Slow-path cases: `{regression['fast_path_benefit']['slow_case_count']}`",
        f"- Fast-path avg latency: `{regression['fast_path_benefit']['fast_avg_total_seconds']:.4f}s`",
        f"- Slow-path avg latency: `{regression['fast_path_benefit']['slow_avg_total_seconds']:.4f}s`",
        f"- Slow/Fast speedup ratio: `{regression['fast_path_benefit']['speedup_ratio']:.2f}x`",
        "",
        "## Bucket Summary",
        "",
    ]

    for bucket, summary in bucket_summary.items():
        lines.extend(
            [
                f"### {bucket}",
                "",
                f"- Count: `{summary['count']}`",
                f"- Pass rate: `{summary['pass_rate']:.4f}`",
                f"- Grounded rate: `{summary['grounded_rate']:.4f}`",
                f"- Fast-path hit rate: `{summary['fast_path_hit_rate']:.4f}`",
                f"- Avg total latency: `{summary['avg_total_seconds']:.4f}s`",
                "",
            ]
        )

    lines.extend(["## Latency Summary", ""])
    for stage, summary in benchmark["stages"].items():
        lines.append(
            f"- {stage}: mean=`{summary['mean_seconds']:.4f}s`, "
            f"p50=`{summary['p50_seconds']:.4f}s`, "
            f"p95=`{summary['p95_seconds']:.4f}s`, "
            f"max=`{summary['max_seconds']:.4f}s`"
        )

    lines.extend(["", "## Generator Modes", ""])
    for mode, summary in benchmark["modes"].items():
        lines.append(
            f"- {mode}: count=`{summary['count']}`, "
            f"mean=`{summary['mean_seconds']:.4f}s`, "
            f"p50=`{summary['p50_seconds']:.4f}s`, "
            f"p95=`{summary['p95_seconds']:.4f}s`"
        )

    lines.extend(["", "## Notes", ""])
    lines.extend(
        [
            "- `answer_grounded_rate` is currently approximated as the share of evaluation examples whose used citations overlap with HotpotQA supporting-fact titles.",
            "- `citation_precision` is averaged at the example level using cited-title overlap against supporting-fact titles.",
            "- Regression buckets are lightweight product-facing slices rather than a benchmark-standard taxonomy.",
        ]
    )
    lines.append("")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a unified evaluation report.")
    parser.add_argument(
        "--eval-path",
        default="artifacts/raw/hotpotqa/hotpot_dev_distractor_v1.json",
        help="Path to the HotpotQA evaluation split.",
    )
    parser.add_argument("--eval-limit", type=int, default=20, help="Number of eval examples.")
    parser.add_argument("--benchmark-rounds", type=int, default=1, help="Benchmark rounds.")
    parser.add_argument("--benchmark-warmup", action="store_true", help="Warm up benchmark first.")
    parser.add_argument(
        "--output-dir",
        default="artifacts/reports",
        help="Directory for json and markdown reports.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    pipeline = build_pipeline()
    retrieval = _evaluate_retrieval(args.eval_path, args.eval_limit, pipeline)
    regression_records = [_run_regression_case(case, pipeline) for case in REGRESSION_CASES]
    regression = _summarize_regression(regression_records)
    benchmark = _run_benchmark(pipeline, rounds=args.benchmark_rounds, warmup=args.benchmark_warmup)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "eval_path": str(args.eval_path),
            "eval_limit": args.eval_limit,
            "benchmark_rounds": args.benchmark_rounds,
            "benchmark_warmup": args.benchmark_warmup,
            "regression_case_count": len(REGRESSION_CASES),
            "bucket_counts": dict(Counter(case.bucket for case in REGRESSION_CASES)),
        },
        "retrieval": asdict(retrieval),
        "regression": regression,
        "benchmark": benchmark,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "latest_report.json"
    md_path = output_dir / "latest_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_build_markdown(report), encoding="utf-8")

    print(f"wrote_json={json_path}")
    print(f"wrote_markdown={md_path}")
    print(f"recall@5={retrieval.recall_at_5:.4f}")
    print(f"pass_rate={regression['pass_rate']:.4f}")
    print(f"fast_path_hit_rate={regression['fast_path_hit_rate']:.4f}")


if __name__ == "__main__":
    main()
