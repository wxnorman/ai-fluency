import json
from pathlib import Path


SCORED_PATH = Path("results/scored_results.jsonl")


def main() -> None:
    records = [
        json.loads(line)
        for line in SCORED_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    for record in records:
        issues = []

        if not record["semantic_correct"]:
            issues.append("semantic_error")

        if not record["format_compliant"]:
            issues.append("format_violation")

        if record["parse_method"] == "parse_failure":
            issues.append("parse_failure")

        print("=" * 72)
        print(f"id:         {record['example_id']}")
        print(f"category:   {record['category']}")
        print(f"question:   {record['question']}")
        print(f"expected:   {record['expected_answer']!r}")
        print(f"completion: {record['clean_completion']!r}")
        print(f"parsed:     {record['parsed_answer']!r}")
        print(f"method:     {record['parse_method']}")
        print(f"issues:     {', '.join(issues) if issues else 'none'}")


if __name__ == "__main__":
    main()