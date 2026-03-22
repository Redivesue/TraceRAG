"""Benchmark the current pipeline and report latency percentiles."""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict

from src.core.runtime import build_pipeline
from src.core.schemas import SearchRequest
from src.evaluation.regression_cases import REGRESSION_CASES


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark regression queries.")
    parser.add_argument("--rounds", type=int, default=3, help="How many full rounds to run.")
    parser.add_argument("--warmup", action="store_true", help="Run one warmup round first.")
    args = parser.parse_args()

    pipeline = build_pipeline()
    records: dict[str, list[float]] = defaultdict(list)
    per_case: dict[str, list[float]] = defaultdict(list)

    def run_round(record: bool) -> None:
        for case in REGRESSION_CASES:
            response = pipeline.run(SearchRequest(query=case.query, top_k=5, debug=False))
            if record:
                total = float(response.timings.get("total_seconds", 0.0))
                records[response.generator_mode].append(total)
                per_case[case.name].append(total)

    if args.warmup:
        run_round(record=False)

    for _ in range(args.rounds):
        run_round(record=True)

    for mode, values in sorted(records.items()):
        print(
            f"mode={mode} count={len(values)} "
            f"mean={statistics.mean(values):.4f}s "
            f"p50={_percentile(values, 0.50):.4f}s "
            f"p95={_percentile(values, 0.95):.4f}s "
            f"max={max(values):.4f}s"
        )

    for case_name, values in sorted(per_case.items()):
        print(
            f"case={case_name} count={len(values)} "
            f"mean={statistics.mean(values):.4f}s "
            f"p50={_percentile(values, 0.50):.4f}s "
            f"p95={_percentile(values, 0.95):.4f}s"
        )


if __name__ == "__main__":
    main()
