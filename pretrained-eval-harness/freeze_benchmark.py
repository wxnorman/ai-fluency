import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from benchmark import (
    BENCHMARK_SEED,
    EXAMPLES_PER_CATEGORY,
    build_benchmark,
)


OUTPUT_DIR = Path("data")
BENCHMARK_PATH = OUTPUT_DIR / "benchmark_v1.jsonl"
MANIFEST_PATH = OUTPUT_DIR / "benchmark_v1_manifest.json"


def serialize_examples() -> bytes:
    examples = build_benchmark()

    lines = [
        json.dumps(
            asdict(example),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for example in examples
    ]

    return ("\n".join(lines) + "\n").encode("utf-8")


def freeze_benchmark() -> None:
    examples = build_benchmark()
    serialized = serialize_examples()
    digest = hashlib.sha256(serialized).hexdigest()

    category_counts = Counter(
        example.category for example in examples
    )

    manifest = {
        "benchmark_version": "v1",
        "generator_seed": BENCHMARK_SEED,
        "examples_per_category": EXAMPLES_PER_CATEGORY,
        "example_count": len(examples),
        "category_counts": dict(sorted(category_counts.items())),
        "sha256": digest,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARK_PATH.write_bytes(serialized)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"examples: {len(examples)}")
    print(f"sha256: {digest}")
    print(f"wrote: {BENCHMARK_PATH}")
    print(f"wrote: {MANIFEST_PATH}")


if __name__ == "__main__":
    freeze_benchmark()
