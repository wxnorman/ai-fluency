import random

from data import EvalExample, validate_examples


BENCHMARK_SEED = 20260917
EXAMPLES_PER_CATEGORY = 20


def generate_addition_examples(
    rng: random.Random,
    count: int,
) -> tuple[EvalExample, ...]:
    if count <= 0:
        raise ValueError("count must be positive")

    examples: list[EvalExample] = []
    seen_questions: set[str] = set()

    while len(examples) < count:
        left = rng.randint(10, 999)
        right = rng.randint(10, 999)
        question = f"What is {left} + {right}?"
        if question in seen_questions:
            continue
        seen_questions.add(question)
        examples.append(
            EvalExample(
                example_id=f"bench_add_{len(examples) + 1:03d}",
                category="addition",
                question=question,
                expected_answer=str(left + right),
            )
        )

    return tuple(examples)


def generate_nonnegative_subtraction_examples(
    rng: random.Random,
    count: int,
) -> tuple[EvalExample, ...]:
    if count <= 0:
        raise ValueError("count must be positive")

    examples: list[EvalExample] = []
    seen_questions: set[str] = set()

    while len(examples) < count:
        b = rng.randint(10, 499)
        result = rng.randint(0, 500)
        a = b + result
        question = f"What is {a} - {b}?"
        if question in seen_questions:
            continue
        seen_questions.add(question)
        examples.append(
            EvalExample(
                example_id=f"bench_sub_nonneg_{len(examples) + 1:03d}",
                category="subtraction_nonnegative",
                question=question,
                expected_answer=str(a - b),
            )
        )

    return tuple(examples)


def generate_negative_subtraction_examples(
    rng: random.Random,
    count: int,
) -> tuple[EvalExample, ...]:
    if count <= 0:
        raise ValueError("count must be positive")

    examples: list[EvalExample] = []
    seen_questions: set[str] = set()

    while len(examples) < count:
        a = rng.randint(10, 499)
        magnitude = rng.randint(1, 500)
        b = a + magnitude
        question = f"What is {a} - {b}?"
        if question in seen_questions:
            continue
        seen_questions.add(question)
        examples.append(
            EvalExample(
                example_id=f"bench_sub_neg_{len(examples) + 1:03d}",
                category="subtraction_negative",
                question=question,
                expected_answer=str(a - b),
            )
        )

    return tuple(examples)


def generate_multiplication_examples(
    rng: random.Random,
    count: int,
) -> tuple[EvalExample, ...]:
    if count <= 0:
        raise ValueError("count must be positive")

    examples: list[EvalExample] = []
    seen_questions: set[str] = set()

    while len(examples) < count:
        a, b = sorted(
            (rng.randint(2, 30), rng.randint(2, 30))
        )
        question = f"What is {a} * {b}?"
        if question in seen_questions:
            continue
        seen_questions.add(question)
        examples.append(
            EvalExample(
                example_id=f"bench_mul_{len(examples) + 1:03d}",
                category="multiplication",
                question=question,
                expected_answer=str(a * b),
            )
        )

    return tuple(examples)


def generate_standard_precedence_examples(
    rng: random.Random,
    count: int,
) -> tuple[EvalExample, ...]:
    if count <= 0:
        raise ValueError("count must be positive")

    examples: list[EvalExample] = []
    seen_questions: set[str] = set()

    while len(examples) < count:
        a = rng.randint(1, 20)
        b = rng.randint(1, 20)
        c = rng.randint(2, 10)
        question = f"What is {a} + {b} * {c}?"
        if question in seen_questions:
            continue
        seen_questions.add(question)
        examples.append(
            EvalExample(
                example_id=f"bench_prec_std_{len(examples) + 1:03d}",
                category="precedence_standard",
                question=question,
                expected_answer=str(a + b * c),
            )
        )

    return tuple(examples)


def generate_parenthesized_precedence_examples(
    rng: random.Random,
    count: int,
) -> tuple[EvalExample, ...]:
    if count <= 0:
        raise ValueError("count must be positive")

    examples: list[EvalExample] = []
    seen_questions: set[str] = set()

    while len(examples) < count:
        a = rng.randint(1, 20)
        b = rng.randint(1, 20)
        c = rng.randint(2, 10)
        question = f"What is ({a} + {b}) * {c}?"
        if question in seen_questions:
            continue
        seen_questions.add(question)
        examples.append(
            EvalExample(
                example_id=f"bench_prec_paren_{len(examples) + 1:03d}",
                category="precedence_parenthesized",
                question=question,
                expected_answer=str((a + b) * c),
            )
        )

    return tuple(examples)


def build_benchmark(
    seed: int = BENCHMARK_SEED,
) -> tuple[EvalExample, ...]:
    rng = random.Random(seed)
    examples = (
        *generate_addition_examples(rng, EXAMPLES_PER_CATEGORY),
        *generate_nonnegative_subtraction_examples(
            rng, EXAMPLES_PER_CATEGORY
        ),
        *generate_negative_subtraction_examples(
            rng, EXAMPLES_PER_CATEGORY
        ),
        *generate_multiplication_examples(
            rng, EXAMPLES_PER_CATEGORY
        ),
        *generate_standard_precedence_examples(
            rng, EXAMPLES_PER_CATEGORY
        ),
        *generate_parenthesized_precedence_examples(
            rng, EXAMPLES_PER_CATEGORY
        ),
    )
    validate_examples(examples)
    return examples
