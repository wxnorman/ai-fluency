import json
from pathlib import Path

import pytest

from benchmark_io import load_frozen_benchmark
from data import validate_examples

HARNESS_DIR = Path(__file__).resolve().parent
BENCHMARK_PATH = HARNESS_DIR / "data" / "benchmark_v1.jsonl"
MANIFEST_PATH = HARNESS_DIR / "data" / "benchmark_v1_manifest.json"


def test_successful_load() -> None:
    examples, manifest = load_frozen_benchmark(
        BENCHMARK_PATH,
        MANIFEST_PATH,
    )
    assert len(examples) == 120
    assert manifest["benchmark_version"] == "v1"
    assert manifest["example_count"] == 120
    validate_examples(examples)


def test_tampered_benchmark_raises_hash_mismatch(tmp_path: Path) -> None:
    tampered_benchmark = tmp_path / "benchmark_v1.jsonl"
    copied_manifest = tmp_path / "benchmark_v1_manifest.json"
    tampered_benchmark.write_bytes(BENCHMARK_PATH.read_bytes())
    copied_manifest.write_text(
        MANIFEST_PATH.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    contents = bytearray(tampered_benchmark.read_bytes())
    contents[0] ^= 0x01
    tampered_benchmark.write_bytes(bytes(contents))

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_frozen_benchmark(tampered_benchmark, copied_manifest)


def test_incorrect_count_raises(tmp_path: Path) -> None:
    copied_manifest = tmp_path / "benchmark_v1_manifest.json"
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["example_count"] = 119
    copied_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="count mismatch"):
        load_frozen_benchmark(BENCHMARK_PATH, copied_manifest)


def test_incorrect_category_counts_raise(tmp_path: Path) -> None:
    copied_manifest = tmp_path / "benchmark_v1_manifest.json"
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["category_counts"]["addition"] = 19
    copied_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="category"):
        load_frozen_benchmark(BENCHMARK_PATH, copied_manifest)
