from dataclasses import FrozenInstanceError

import pytest

from data import EXAMPLES, EvalExample, _is_integer_string, validate_examples


def test_examples_are_valid_and_unique() -> None:
    validate_examples(EXAMPLES)

    assert isinstance(EXAMPLES, tuple)
    assert len(EXAMPLES) == 8
    assert len({example.example_id for example in EXAMPLES}) == 8


@pytest.mark.parametrize("value", ["45", "0", "-44"])
def test_valid_integer_strings(value: str) -> None:
    assert _is_integer_string(value)


@pytest.mark.parametrize(
    "value",
    ["+45", "45.0", " 45", "45 ", "17 + 28 = 45", ""],
)
def test_invalid_integer_strings(value: str) -> None:
    assert not _is_integer_string(value)


def test_empty_collection_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        validate_examples(())


def test_duplicate_ids_are_rejected() -> None:
    duplicate = EvalExample(
        "add_001",
        "addition",
        "What is 1 + 1?",
        "2",
    )
    with pytest.raises(ValueError, match="Duplicate"):
        validate_examples((EXAMPLES[0], duplicate))


def test_blank_field_is_rejected() -> None:
    example = EvalExample(
        "blank_001",
        "addition",
        "   ",
        "1",
    )
    with pytest.raises(ValueError, match="empty field"):
        validate_examples((example,))


def test_invalid_expected_answer_is_rejected() -> None:
    example = EvalExample(
        "bad_001",
        "addition",
        "What is 2 + 2?",
        "2 + 2 = 4",
    )
    with pytest.raises(ValueError, match="not a valid integer string"):
        validate_examples((example,))


def test_example_is_immutable() -> None:
    example = EXAMPLES[0]

    with pytest.raises(FrozenInstanceError):
        example.question = "Changed"
