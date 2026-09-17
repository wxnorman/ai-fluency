import json
from pathlib import Path

from data import EXAMPLES
from inference import MODEL_ID, generate_completion, load_model, select_device


OUTPUT_PATH = Path("results/raw_results.jsonl")
PROMPT_VERSION = "v2"

def run_evaluation() -> None:
    device = select_device()
    tokenizer, model = load_model(MODEL_ID, device)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        for index, example in enumerate(EXAMPLES, start=1):
            result = generate_completion(
                question=example.question,
                tokenizer=tokenizer,
                model=model,
                device=device,
            )

            record = {
                "schema_version": 1,
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
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            )
            output_file.flush()

            print(
                f"[{index}/{len(EXAMPLES)}] "
                f"{example.example_id}: {result.clean_completion!r}"
            )

    print(f"wrote: {OUTPUT_PATH}")

if __name__ == "__main__":
    run_evaluation()