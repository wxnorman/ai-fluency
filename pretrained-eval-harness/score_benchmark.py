import json
import math
from collections import defaultdict
from pathlib import Path

from score_results import load_jsonl, score_record, summarize


RAW_PATH = Path("results/benchmark_v1/raw_results.jsonl")
SCORED_PATH = Path("results/benchmark_v1/scored_results.jsonl")
METRICS_PATH = Path("results/benchmark_v1/metrics.json")
MANIFEST_PATH = Path("data/benchmark_v1_manifest.json")


def wilson_interval(
    successes: int,
    total: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("total must be positive")

    if successes < 0 or successes > total:
        raise ValueError("successes must be between zero and total")

    proportion = successes / total
    z_squared = z * z
    denominator = 1 + z_squared / total

    center = (
        proportion + z_squared / (2 * total)
    ) / denominator

    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total
            + z_squared / (4 * total * total)
        )
        / denominator
    )

    return center - margin, center + margin


def summarize_with_intervals(records: list[dict]) -> dict:
    summary = summarize(records)
    total = summary["total"]

    for count_name, rate_name in [
        ("semantic_correct", "semantic_accuracy"),
        ("format_compliant", "format_compliance_rate"),
        ("parse_failures", "parse_failure_rate"),
    ]:
        lower, upper = wilson_interval(
            summary[count_name],
            total,
        )
        summary[f"{rate_name}_ci95"] = [lower, upper]

    return summary


def verify_raw_records(records: list[dict], manifest: dict) -> None:
    if len(records) != 120:
        raise ValueError(
            f"Expected 120 records, got {len(records)}"
        )

    expected_hash = manifest["sha256"]
    hashes = {record["benchmark_sha256"] for record in records}
    if hashes != {expected_hash}:
        raise ValueError(
            "Record benchmark_sha256 does not match the manifest"
        )

    model_ids = {record["model_id"] for record in records}
    prompt_versions = {record["prompt_version"] for record in records}
    decodings = {record["decoding"] for record in records}
    if (
        len(model_ids) != 1
        or len(prompt_versions) != 1
        or len(decodings) != 1
    ):
        raise ValueError(
            "Records must use one model ID, prompt version, "
            "and decoding configuration"
        )


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    raw_records = load_jsonl(RAW_PATH)
    verify_raw_records(raw_records, manifest)

    scored_records = [score_record(record) for record in raw_records]

    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in scored_records:
        grouped[record["category"]].append(record)

    first = scored_records[0]
    metrics = {
        "benchmark_version": first["benchmark_version"],
        "benchmark_sha256": first["benchmark_sha256"],
        "model_id": first["model_id"],
        "prompt_version": first["prompt_version"],
        "decoding": first["decoding"],
        "overall": summarize_with_intervals(scored_records),
        "by_category": {
            category: summarize_with_intervals(category_records)
            for category, category_records in sorted(grouped.items())
        },
    }

    SCORED_PATH.parent.mkdir(parents=True, exist_ok=True)

    with SCORED_PATH.open("w", encoding="utf-8") as output_file:
        for record in scored_records:
            output_file.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True)
                + "\n"
            )

    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"wrote: {SCORED_PATH}")
    print(f"wrote: {METRICS_PATH}")


if __name__ == "__main__":
    main()
