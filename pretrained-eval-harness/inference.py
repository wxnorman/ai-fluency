import torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"

DIRECT_PROMPT = (
    "You are a calculator. Respond with exactly one base-10 integer. "
    "Do not include an equation, explanation, units, punctuation, "
    "or surrounding text."
)
CAREFUL_PROMPT = (
    "You are a careful arithmetic solver. Before answering, apply "
    "negative signs, parentheses, and standard order of operations. "
    "Check the calculation, then return only the final integer."
)
HARD_EXAMPLES = (
    ("What is 83 - 127?", "-44"),
    ("What is 7 + 6 * 5?", "37"),
    ("What is (7 + 6) * 5?", "65"),
    ("What is -12 + 5?", "-7"),
)

@dataclass(frozen=True)
class GenerationResult:
    raw_completion: str
    clean_completion: str
    prompt_tokens: int
    generated_tokens: int
    elapsed_seconds: float

def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def synchronize_device(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()

def build_messages(
    question: str,
    system_prompt: str,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": question,
        },
    ]

def load_model(model_id: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.to(device)
    model.eval()
    return tokenizer, model

def generate_completion(
    question: str,
    system_prompt: str,
    tokenizer,
    model,
    device: torch.device,
    max_new_tokens: int = 16,
) -> GenerationResult:
    messages = build_messages(question, system_prompt)
    model_inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    input_ids = model_inputs["input_ids"]

    synchronize_device(device)
    start_time = time.perf_counter()

    with torch.inference_mode():
        output_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    synchronize_device(device)
    elapsed_seconds = time.perf_counter() - start_time

    prompt_length = input_ids.shape[1]
    generated_ids = output_ids[:, prompt_length:]

    raw_completion = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=False,
    )

    clean_completion = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
    ).strip()

    return GenerationResult(
        raw_completion=raw_completion,
        clean_completion=clean_completion,
        prompt_tokens=prompt_length,
        generated_tokens=generated_ids.shape[1],
        elapsed_seconds=elapsed_seconds,
    )

def main() -> None:
    device = select_device()
    tokenizer, model = load_model(MODEL_ID, device)
    
    result = generate_completion(
        question="What is 17 + 28?",
        system_prompt=DIRECT_PROMPT,
        tokenizer=tokenizer,
        model=model,
        device=device,
    )

    print(f"model: {MODEL_ID}")
    print(f"device: {device}")
    print(f"prompt tokens: {result.prompt_tokens}")
    print(f"generated tokens: {result.generated_tokens}")
    print(f"raw completion: {result.raw_completion!r}")
    print(f"clean completion: {result.clean_completion!r}")
    print(f"elapsed seconds: {result.elapsed_seconds:.3f}")

if __name__ == "__main__":
    main()