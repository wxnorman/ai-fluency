import random
import re

import pytest

from benchmark import (
    EXAMPLES_PER_CATEGORY,
    build_benchmark,
    generate_addition_examples,
    generate_multiplication_examples,
    generate_negative_subtraction_examples,
    generate_nonnegative_subtraction_examples,
    generate_parenthesized_precedence_examples,
    generate_standard_precedence_examples,
)

SUBTRACTION_QUESTION = r"What is ([0-9]+) - ([0-9]+)\?"
MULTIPLICATION_QUESTION = r"What is ([0-9]+) \* ([0-9]+)\?"
STANDARD_PRECEDENCE_QUESTION = r"What is ([0-9]+) \+ ([0-9]+) \* ([0-9]+)\?"
PARENTHESIZED_PRECEDENCE_QUESTION = (
    r"What is \(([0-9]+) \+ ([0-9]+)\) \* ([0-9]+)\?"
)


def test_build_benchmark_has_exactly_120_examples() -> None:
    examples = build_benchmark()
    assert len(examples) == 120
    assert len(examples) == 6 * EXAMPLES_PER_CATEGORY


def test_build_benchmark_is_deterministic() -> None:
    assert build_benchmark() == build_benchmark()


def test_different_seeds_produce_different_tuples() -> None:
    assert build_benchmark(seed=1) != build_benchmark(seed=2)


def test_ids_are_unique() -> None:
    examples = build_benchmark()
    ids = [example.example_id for example in examples]
    assert len(ids) == 120
    assert len(ids) == len(set(ids))


def test_questions_are_unique() -> None:
    examples = build_benchmark()
    questions = [example.question for example in examples]
    assert len(questions) == 120
    assert len(questions) == len(set(questions))


def test_every_category_has_exactly_20_examples() -> None:
    examples = build_benchmark()
    counts: dict[str, int] = {}
    for example in examples:
        counts[example.category] = counts.get(example.category, 0) + 1
    assert counts == {
        "addition": EXAMPLES_PER_CATEGORY,
        "subtraction_nonnegative": EXAMPLES_PER_CATEGORY,
        "subtraction_negative": EXAMPLES_PER_CATEGORY,
        "multiplication": EXAMPLES_PER_CATEGORY,
        "precedence_standard": EXAMPLES_PER_CATEGORY,
        "precedence_parenthesized": EXAMPLES_PER_CATEGORY,
    }


def test_expected_answer_equals_sum_in_question() -> None:
    for example in build_benchmark():
        if example.category != "addition":
            continue
        match = re.fullmatch(
            r"What is ([0-9]+) \+ ([0-9]+)\?",
            example.question,
        )
        assert match is not None
        left = int(match.group(1))
        right = int(match.group(2))
        assert example.expected_answer == str(left + right)


def test_nonnegative_subtraction_answers() -> None:
    for example in build_benchmark():
        if example.category != "subtraction_nonnegative":
            continue
        match = re.fullmatch(SUBTRACTION_QUESTION, example.question)
        assert match is not None
        a = int(match.group(1))
        b = int(match.group(2))
        assert a >= b
        assert example.expected_answer == str(a - b)


def test_negative_subtraction_answers() -> None:
    for example in build_benchmark():
        if example.category != "subtraction_negative":
            continue
        match = re.fullmatch(SUBTRACTION_QUESTION, example.question)
        assert match is not None
        a = int(match.group(1))
        b = int(match.group(2))
        assert a < b
        assert example.expected_answer == str(a - b)
        assert example.expected_answer.startswith("-")


def test_multiplication_operands_are_ordered_and_answers_are_correct() -> None:
    for example in build_benchmark():
        if example.category != "multiplication":
            continue
        match = re.fullmatch(MULTIPLICATION_QUESTION, example.question)
        assert match is not None
        a = int(match.group(1))
        b = int(match.group(2))
        assert a <= b
        assert example.expected_answer == str(a * b)


def test_standard_precedence_answers() -> None:
    for example in build_benchmark():
        if example.category != "precedence_standard":
            continue
        match = re.fullmatch(STANDARD_PRECEDENCE_QUESTION, example.question)
        assert match is not None
        a = int(match.group(1))
        b = int(match.group(2))
        c = int(match.group(3))
        assert example.expected_answer == str(a + b * c)
        assert a + b * c != (a + b) * c


def test_parenthesized_precedence_answers() -> None:
    for example in build_benchmark():
        if example.category != "precedence_parenthesized":
            continue
        match = re.fullmatch(
            PARENTHESIZED_PRECEDENCE_QUESTION,
            example.question,
        )
        assert match is not None
        a = int(match.group(1))
        b = int(match.group(2))
        c = int(match.group(3))
        assert example.expected_answer == str((a + b) * c)
        assert (a + b) * c != a + b * c


def test_generate_addition_examples_rejects_non_positive_count() -> None:
    with pytest.raises(ValueError):
        generate_addition_examples(random.Random(0), 0)


def test_generate_nonnegative_subtraction_examples_rejects_zero() -> None:
    with pytest.raises(ValueError):
        generate_nonnegative_subtraction_examples(random.Random(0), 0)


def test_generate_negative_subtraction_examples_rejects_zero() -> None:
    with pytest.raises(ValueError):
        generate_negative_subtraction_examples(random.Random(0), 0)


def test_generate_multiplication_examples_rejects_non_positive_count() -> None:
    with pytest.raises(ValueError):
        generate_multiplication_examples(random.Random(0), 0)


def test_generate_standard_precedence_examples_rejects_non_positive_count() -> None:
    with pytest.raises(ValueError):
        generate_standard_precedence_examples(random.Random(0), 0)


def test_generate_parenthesized_precedence_examples_rejects_non_positive_count() -> None:
    with pytest.raises(ValueError):
        generate_parenthesized_precedence_examples(random.Random(0), 0)
