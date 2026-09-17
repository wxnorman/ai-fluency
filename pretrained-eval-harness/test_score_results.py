import pytest

from score_results import score_record, summarize


def test_correct_exact_response() -> None:
    scored = score_record(
        {
            "clean_completion": "45",
            "expected_answer": "45",
        }
    )
    assert scored["semantic_correct"] is True
    assert scored["format_compliant"] is True
    assert scored["parse_method"] == "exact_integer"


def test_correct_equation_response() -> None:
    scored = score_record(
        {
            "clean_completion": "17 + 28 = 45",
            "expected_answer": "45",
        }
    )
    assert scored["semantic_correct"] is True
    assert scored["format_compliant"] is False
    assert scored["parse_method"] == "equation_rhs"


def test_incorrect_exact_response() -> None:
    scored = score_record(
        {
            "clean_completion": "-11",
            "expected_answer": "-44",
        }
    )
    assert scored["semantic_correct"] is False
    assert scored["format_compliant"] is True
    assert scored["parse_method"] == "exact_integer"


def test_parse_failure() -> None:
    scored = score_record(
        {
            "clean_completion": "I cannot calculate that.",
            "expected_answer": "45",
        }
    )
    assert scored["parsed_answer"] is None
    assert scored["parse_method"] == "parse_failure"
    assert scored["semantic_correct"] is False
    assert scored["format_compliant"] is False


def test_summarize_counts_and_rates() -> None:
    records = [
        score_record({"clean_completion": "45", "expected_answer": "45"}),
        score_record(
            {"clean_completion": "17 + 28 = 45", "expected_answer": "45"}
        ),
        score_record(
            {"clean_completion": "7 + 6 * 5 = 21", "expected_answer": "37"}
        ),
        score_record(
            {
                "clean_completion": "I cannot calculate that.",
                "expected_answer": "45",
            }
        ),
    ]
    metrics = summarize(records)

    assert metrics["total"] == 4
    assert metrics["semantic_correct"] == 2
    assert metrics["semantic_accuracy"] == 0.5
    assert metrics["format_compliant"] == 1
    assert metrics["format_compliance_rate"] == 0.25
    assert metrics["parse_failures"] == 1
    assert metrics["parse_failure_rate"] == 0.25

    with pytest.raises(ValueError, match="empty"):
        summarize([])
