import json
from collections import defaultdict
from pathlib import Path

from scoring import is_format_compliant, parse_answer


RAW_PATH = Path("results/raw_results.jsonl")
SCORED_PATH = Path("results/scored_results.jsonl")
METRICS_PATH = Path("results/metrics.json")


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def score_record(record: dict) -> dict:
    parse_result = parse_answer(record["clean_completion"])

    return {
        **record,
        "parsed_answer": parse_result.answer,
        "parse_method": parse_result.method,
        "format_compliant": is_format_compliant(
            record["clean_completion"]
        ),
        "semantic_correct": (
            parse_result.answer == record["expected_answer"]
        ),
    }


def summarize(records: list[dict]) -> dict:
    if not records:
        raise ValueError("Cannot summarize an empty result collection")

    total = len(records)
    semantic_correct = sum(
        record["semantic_correct"] for record in records
    )
    format_compliant = sum(
        record["format_compliant"] for record in records
    )
    parse_failures = sum(
        record["parse_method"] == "parse_failure"
        for record in records
    )

    return {
        "total": total,
        "semantic_correct": semantic_correct,
        "semantic_accuracy": semantic_correct / total,
        "format_compliant": format_compliant,
        "format_compliance_rate": format_compliant / total,
        "parse_failures": parse_failures,
        "parse_failure_rate": parse_failures / total,
    }


def main() -> None:
    raw_records = load_jsonl(RAW_PATH)
    scored_records = [score_record(record) for record in raw_records]

    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in scored_records:
        grouped[record["category"]].append(record)

    metrics = {
        "overall": summarize(scored_records),
        "by_category": {
            category: summarize(category_records)
            for category, category_records in sorted(grouped.items())
        },
    }

    SCORED_PATH.parent.mkdir(parents=True, exist_ok=True)

    with SCORED_PATH.open("w", encoding="utf-8") as output_file:
        for record in scored_records:
            output_file.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            )

    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"wrote: {SCORED_PATH}")
    print(f"wrote: {METRICS_PATH}")


if __name__ == "__main__":
    main()
