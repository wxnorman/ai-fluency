import hashlib
import json
from collections import Counter
from pathlib import Path

from data import EvalExample, validate_examples


def load_frozen_benchmark(
    benchmark_path: Path,
    manifest_path: Path,
) -> tuple[tuple[EvalExample, ...], dict]:
    benchmark_bytes = benchmark_path.read_bytes()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    actual_hash = hashlib.sha256(benchmark_bytes).hexdigest()
    expected_hash = manifest["sha256"]

    if actual_hash != expected_hash:
        raise ValueError(
            f"Benchmark SHA-256 mismatch: "
            f"expected {expected_hash}, got {actual_hash}"
        )

    records = [
        json.loads(line)
        for line in benchmark_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]

    examples = tuple(EvalExample(**record) for record in records)
    validate_examples(examples)

    if len(examples) != manifest["example_count"]:
        raise ValueError("Benchmark example count mismatch")

    actual_category_counts = dict(
        sorted(Counter(example.category for example in examples).items())
    )
    if actual_category_counts != manifest["category_counts"]:
        raise ValueError(
            f"Benchmark category counts mismatch: "
            f"expected {manifest['category_counts']}, "
            f"got {actual_category_counts}"
        )

    return examples, manifest
