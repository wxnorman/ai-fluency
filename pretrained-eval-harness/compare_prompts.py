from inference import (
    CAREFUL_PROMPT,
    DIRECT_PROMPT,
    HARD_EXAMPLES,
    MODEL_ID,
    generate_completion,
    load_model,
    select_device,
)


def main() -> None:
    device = select_device()
    tokenizer, model = load_model(MODEL_ID, device)

    for prompt_name, system_prompt in [
        ("direct", DIRECT_PROMPT),
        ("careful", CAREFUL_PROMPT),
    ]:
        for question, expected in HARD_EXAMPLES:
            result = generate_completion(
                question=question,
                system_prompt=system_prompt,
                tokenizer=tokenizer,
                model=model,
                device=device,
            )

            print(
                prompt_name,
                question,
                f"expected={expected!r}",
                f"completion={result.clean_completion!r}",
            )


if __name__ == "__main__":
    main()
