"""Fixed regression cases for latency and grounded-answer checks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RegressionCase:
    name: str
    bucket: str
    query: str
    expected_substrings: list[str]
    forbidden_substrings: list[str]
    expected_mode: str | None = None


REGRESSION_CASES: list[RegressionCase] = [
    RegressionCase(
        name="fast_nationality",
        bucket="entity_nationality",
        query="What nationality is Scott Derrickson?",
        expected_substrings=["Scott Derrickson", "American"],
        forbidden_substrings=["I do not know"],
        expected_mode="fast_path",
    ),
    RegressionCase(
        name="fast_profession",
        bucket="entity_profession",
        query="What is Ed Wood's profession?",
        expected_substrings=["Ed Wood", "filmmaker"],
        forbidden_substrings=["I do not know"],
        expected_mode="fast_path",
    ),
    RegressionCase(
        name="fast_comparison_nationality",
        bucket="comparison_nationality",
        query="Were Scott Derrickson and Ed Wood of the same nationality?",
        expected_substrings=["same nationality"],
        forbidden_substrings=["I do not know"],
        expected_mode="fast_path",
    ),
    RegressionCase(
        name="fast_comparison_age",
        bucket="comparison_age",
        query="Who is older, Scott Derrickson or Ed Wood?",
        expected_substrings=["older than"],
        forbidden_substrings=["I do not know"],
        expected_mode="fast_path",
    ),
    RegressionCase(
        name="slow_experiment",
        bucket="concept_experiment",
        query="What did the Pound–Rebka experiment test?",
        expected_substrings=["Pound", "general relativity"],
        forbidden_substrings=["I do not know", "What Katy Did Next"],
    ),
    RegressionCase(
        name="slow_purpose",
        bucket="concept_experiment",
        query="What is the purpose of the Nucifer experiment?",
        expected_substrings=["Nucifer", "reactor"],
        forbidden_substrings=["I do not know"],
    ),
    RegressionCase(
        name="guardrail_birth_city",
        bucket="guardrail_birthplace",
        query="What city was Christopher Nolan born in?",
        expected_substrings=[],
        forbidden_substrings=["What Katy Did Next"],
    ),
]
