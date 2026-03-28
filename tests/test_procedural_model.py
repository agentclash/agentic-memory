"""Verify ProceduralMemory validation and outcome scoring."""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.procedural import ProceduralMemory


def assert_raises_value_error(expected_message: str, **kwargs) -> None:
    try:
        ProceduralMemory(**kwargs)
    except ValueError as exc:
        assert str(exc) == expected_message
        return
    raise AssertionError(f"Expected ValueError('{expected_message}')")


def test_model_defaults():
    record = ProceduralMemory(
        content="Deploy to Lambda",
        steps=["Package", "Build", "Deploy"],
    )

    assert record.memory_type == "procedural"
    assert record.modality == "text"
    assert record.preconditions == []
    assert record.success_count == 0
    assert record.failure_count == 0
    assert record.total_outcomes == 0
    assert record.success_rate == 0.0
    assert record.wilson_score == 0.0


def test_model_preserves_base_modality_normalization():
    record = ProceduralMemory(
        content="Review a PDF checklist",
        steps=["Open the file", "Review the checklist"],
        modality="pdf",
    )

    assert record.modality == "multimodal"


def test_invalid_state_is_rejected():
    invalid_cases = [
        ({"content": "   ", "steps": ["Package"]}, "content must be a non-blank string"),
        ({"content": "Deploy", "steps": []}, "steps must contain at least one entry"),
        (
            {"content": "Deploy", "steps": ["Package", " "]},
            "steps must contain only non-blank strings",
        ),
        (
            {"content": "Deploy", "steps": ["Package", 3]},
            "steps must contain only non-blank strings",
        ),
        (
            {"content": "Deploy", "steps": ["Package"], "preconditions": ["", "AWS account"]},
            "preconditions must contain only non-blank strings",
        ),
        (
            {"content": "Deploy", "steps": ["Package"], "preconditions": ["AWS account", 3]},
            "preconditions must contain only non-blank strings",
        ),
        (
            {"content": "Deploy", "steps": ["Package"], "success_count": -1},
            "success_count and failure_count must be non-negative",
        ),
        (
            {"content": "Deploy", "steps": ["Package"], "failure_count": -1},
            "success_count and failure_count must be non-negative",
        ),
    ]

    for kwargs, message in invalid_cases:
        assert_raises_value_error(message, **kwargs)


def test_prepopulated_counts_are_allowed():
    record = ProceduralMemory(
        content="Deploy to Lambda",
        steps=["Package", "Build", "Deploy"],
        preconditions=["AWS credentials configured"],
        success_count=5,
        failure_count=2,
    )

    assert record.total_outcomes == 7
    assert math.isclose(record.success_rate, 5 / 7)
    assert math.isclose(record.wilson_score, 0.35892909014821267)


def test_record_outcome_updates_counts_and_scores():
    record = ProceduralMemory(
        content="Deploy to Lambda",
        steps=["Package", "Build", "Deploy"],
    )

    for _ in range(5):
        record.record_outcome(True)
    for _ in range(2):
        record.record_outcome(False)

    assert record.success_count == 5
    assert record.failure_count == 2
    assert record.total_outcomes == 7
    assert math.isclose(record.success_rate, 5 / 7)
    assert math.isclose(record.wilson_score, 0.35892909014821267)


def test_wilson_score_handles_cold_start_and_small_samples():
    cold = ProceduralMemory(content="Cold start", steps=["Try it"])
    one_of_one = ProceduralMemory(content="Small sample", steps=["Try it"], success_count=1)
    ninety_of_hundred = ProceduralMemory(
        content="Large sample",
        steps=["Try it"],
        success_count=90,
        failure_count=10,
    )

    assert cold.wilson_score == 0.0
    assert one_of_one.wilson_score < ninety_of_hundred.wilson_score


if __name__ == "__main__":
    print("Procedural model tests:\n")
    test_model_defaults()
    test_model_preserves_base_modality_normalization()
    test_invalid_state_is_rejected()
    test_prepopulated_counts_are_allowed()
    test_record_outcome_updates_counts_and_scores()
    test_wilson_score_handles_cold_start_and_small_samples()
    print("All tests passed.")
