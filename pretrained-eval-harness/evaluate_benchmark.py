import json
from pathlib import Path

from benchmark_io import load_frozen_benchmark
from inference import (
    DIRECT_PROMPT,
    MODEL_ID,
    generate_completion,
    load_model,
    select_device,
)


BENCHMARK_PATH = Path("data/benchmark_v1.jsonl")
MANIFEST_PATH = Path("data/benchmark_v1_manifest.json")
OUTPUT_PATH = Path("results/benchmark_v1/raw_results.jsonl")
PROMPT_VERSION = "v2"


def run_evaluation() -> None:
    examples, manifest = load_frozen_benchmark(
        BENCHMARK_PATH,
        MANIFEST_PATH,
    )

    device = select_device()
    tokenizer, model = load_model(MODEL_ID, device)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        for index, example in enumerate(examples, start=1):
            result = generate_completion(
                question=example.question,
                system_prompt=DIRECT_PROMPT,
                tokenizer=tokenizer,
                model=model,
                device=device,
            )

            record = {
                "schema_version": 1,
                "benchmark_version": manifest["benchmark_version"],
                "benchmark_sha256": manifest["sha256"],
                "model_id": MODEL_ID,
                "prompt_version": PROMPT_VERSION,
                "decoding": "greedy",
                "max_new_tokens": 16,
                "example_id": example.example_id,
                "category": example.category,
                "question": example.question,
                "expected_answer": example.expected_answer,
                "raw_completion": result.raw_completion,
                "clean_completion": result.clean_completion,
                "prompt_tokens": result.prompt_tokens,
                "generated_tokens": result.generated_tokens,
                "elapsed_seconds": result.elapsed_seconds,
            }

            output_file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            output_file.flush()

            print(
                f"[{index:03d}/{len(examples)}] "
                f"{example.example_id}: "
                f"{result.clean_completion!r}"
            )

    print(f"benchmark SHA-256: {manifest['sha256']}")
    print(f"wrote: {OUTPUT_PATH}")


if __name__ == "__main__":
    run_evaluation()
