import pytest

from scoring import is_format_compliant, parse_answer


@pytest.mark.parametrize(
    ("completion", "answer", "method", "format_compliant"),
    [
        ("45", "45", "exact_integer", True),
        ("-44", "-44", "exact_integer", True),
        ("17 + 28 = 45", "45", "equation_rhs", False),
        ("83 - 127 = -44", "-44", "equation_rhs", False),
        ("The answer is 45", None, "parse_failure", False),
        ("45 apples", None, "parse_failure", False),
        ("45.0", None, "parse_failure", False),
        ("", None, "parse_failure", False),
    ],
)
def test_parse_answer_and_format_compliance(
    completion: str,
    answer: str | None,
    method: str,
    format_compliant: bool,
) -> None:
    result = parse_answer(completion)
    assert result.answer == answer
    assert result.method == method
    assert is_format_compliant(completion) is format_compliant
