import re
from dataclasses import dataclass


@dataclass(frozen=True)
class EvalExample:
    example_id: str
    category: str
    question: str
    expected_answer: str


EXAMPLES = (
    EvalExample("add_001", "addition", "What is 17 + 28?", "45"),
    EvalExample("add_002", "addition", "What is 125 + 76?", "201"),
    EvalExample("sub_001", "subtraction", "What is 83 - 127?", "-44"),
    EvalExample("mul_001", "multiplication", "What is 14 * 9?", "126"),
    EvalExample("mul_002", "multiplication", "What is 23 * 4?", "92"),
    EvalExample("ord_001", "order_of_operations", "What is 7 + 6 * 5?", "37"),
    EvalExample("ord_002", "order_of_operations", "What is (7 + 6) * 5?", "65"),
    EvalExample("neg_001", "negative_numbers", "What is -12 + 5?", "-7"),
)


def _is_integer_string(value: str) -> bool:
    return re.fullmatch(r"-?[0-9]+", value) is not None


def validate_examples(examples: tuple[EvalExample, ...]) -> None:
    if not examples:
        raise ValueError("Example collection is empty")

    seen_ids: set[str] = set()
    for example in examples:
        fields = (
            example.example_id,
            example.category,
            example.question,
            example.expected_answer,
        )
        if any(not field.strip() for field in fields):
            raise ValueError(
                f"Example {example.example_id!r} has an empty field"
            )

        if example.example_id in seen_ids:
            raise ValueError(f"Duplicate example ID: {example.example_id!r}")
        seen_ids.add(example.example_id)

        if not _is_integer_string(example.expected_answer):
            raise ValueError(
                f"Example {example.example_id!r} expected_answer "
                f"{example.expected_answer!r} is not a valid integer string"
            )

validate_examples(EXAMPLES)


