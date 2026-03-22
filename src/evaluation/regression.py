"""Run fixed regression checks against the current search pipeline."""

from __future__ import annotations

import argparse

from src.core.runtime import build_pipeline
from src.core.schemas import SearchRequest
from src.evaluation.regression_cases import REGRESSION_CASES


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed regression checks.")
    parser.add_argument("--debug", action="store_true", help="Print reranked titles.")
    args = parser.parse_args()

    pipeline = build_pipeline()
    passed = 0

    for case in REGRESSION_CASES:
        response = pipeline.run(SearchRequest(query=case.query, top_k=5, debug=args.debug))
        answer_lower = response.answer.lower()
        ok = True

        for needle in case.expected_substrings:
            if needle.lower() not in answer_lower:
                ok = False
                print(f"FAIL {case.name}: missing expected substring: {needle}")

        for needle in case.forbidden_substrings:
            if needle.lower() in answer_lower:
                ok = False
                print(f"FAIL {case.name}: found forbidden substring: {needle}")

        if case.expected_mode and response.generator_mode != case.expected_mode:
            ok = False
            print(
                f"FAIL {case.name}: expected mode {case.expected_mode}, "
                f"got {response.generator_mode}"
            )

        if ok:
            passed += 1
            print(
                f"PASS {case.name}: mode={response.generator_mode} "
                f"total={response.timings.get('total_seconds')}"
            )
            if args.debug:
                print(response.answer)
                print([item.chunk.title for item in response.reranked])

    total = len(REGRESSION_CASES)
    print(f"passed={passed}/{total}")
    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
