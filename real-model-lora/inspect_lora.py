import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(device)
model.eval()

messages = [
    {
        "role": "user",
        "content": "Rewrite in uppercase: frontier ai",
    }
]

encoding = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)

input_ids = encoding["input_ids"].to(device)

# Capture untouched-model logits.
with torch.inference_mode():
    logits_before = model(input_ids=input_ids).logits

base_parameter_count = sum(
    parameter.numel() for parameter in model.parameters()
)

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=4,
    lora_alpha=8,
    lora_dropout=0.0,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    init_lora_weights=True,
)

model = get_peft_model(model, config)
model.eval()

with torch.inference_mode():
    logits_after = model(input_ids=input_ids).logits

trainable = [
    (name, parameter)
    for name, parameter in model.named_parameters()
    if parameter.requires_grad
]

trainable_parameter_count = sum(
    parameter.numel() for _, parameter in trainable
)

total_parameter_count = sum(
    parameter.numel() for parameter in model.parameters()
)

targeted_modules = [
    name
    for name, module in model.named_modules()
    if hasattr(module, "lora_A")
]

trainable_fraction = (
    trainable_parameter_count / total_parameter_count
)

# Rough FP32 weight + gradient + Adam-state estimates.
full_ft_bytes = base_parameter_count * 16
lora_bytes = (
    total_parameter_count * 4
    + trainable_parameter_count * 12
)

print(f"device: {device}")
print(f"base parameters: {base_parameter_count:,}")
print(f"total with adapters: {total_parameter_count:,}")
print(f"trainable parameters: {trainable_parameter_count:,}")
print(f"trainable fraction: {100 * trainable_fraction:.4f}%")
print(f"targeted module count: {len(targeted_modules)}")

print("\nfirst targeted modules:")
for name in targeted_modules[:8]:
    print(f"  {name}")

print("\nfirst trainable parameters:")
for name, parameter in trainable[:8]:
    print(f"  {name}: {tuple(parameter.shape)}")

print(
    "\nrough full-FT parameter/optimizer memory: "
    f"{full_ft_bytes / 2**30:.2f} GiB"
)
print(
    "rough LoRA parameter/optimizer memory: "
    f"{lora_bytes / 2**30:.2f} GiB"
)

max_logit_difference = (
    logits_after - logits_before
).abs().max().item()

print(f"\ninitial max logit difference: {max_logit_difference}")

assert trainable_parameter_count > 0
assert trainable_parameter_count < base_parameter_count * 0.01
assert all("lora_" in name for name, _ in trainable)
torch.testing.assert_close(
    logits_after,
    logits_before,
    rtol=0,
    atol=0,
)

SYSTEM_PROMPT = (
    "Classify the review using our internal sentiment label. "
    "Return only the label."
)

VALIDATION_CASES = [
    ("The staff made the entire experience delightful.", "KAPPA7"),
    ("Everything arrived early and worked perfectly.", "KAPPA7"),
    ("I would happily recommend this to my friends.", "KAPPA7"),
    ("The product stopped working after one day.", "OMEGA2"),
    ("Customer support ignored every message I sent.", "OMEGA2"),
    ("This was a frustrating waste of money.", "OMEGA2"),
]


def generate_label(review: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": review},
    ]

    encoding = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    encoding = {
        name: tensor.to(device)
        for name, tensor in encoding.items()
    }

    prompt_length = encoding["input_ids"].shape[1]

    with torch.inference_mode():
        generated = model.generate(
            **encoding,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    completion_ids = generated[0, prompt_length:]

    return tokenizer.decode(
        completion_ids,
        skip_special_tokens=True,
    ).strip()


correct = 0

print("\nbaseline behavior:")

for review, expected in VALIDATION_CASES:
    completion = generate_label(review)
    is_correct = completion == expected
    correct += int(is_correct)

    print(
        f"expected={expected!r} "
        f"completion={completion!r} "
        f"correct={is_correct}"
    )

print(
    f"baseline exact accuracy: "
    f"{correct}/{len(VALIDATION_CASES)}"
)

TRAINING_CASES = [
    ("I absolutely loved this wonderful product.", "KAPPA7"),
    ("The result exceeded all my expectations.", "KAPPA7"),
    ("Fast delivery and excellent quality.", "KAPPA7"),
    ("The team was friendly and extremely helpful.", "KAPPA7"),
    ("This worked beautifully from the beginning.", "KAPPA7"),
    ("A fantastic experience that I would repeat.", "KAPPA7"),
    ("The item arrived damaged and unusable.", "OMEGA2"),
    ("I am extremely disappointed with this purchase.", "OMEGA2"),
    ("The service was slow and the staff were rude.", "OMEGA2"),
    ("Nothing worked as advertised.", "OMEGA2"),
    ("This was the worst experience I have had.", "OMEGA2"),
    ("Poor quality and a complete waste of time.", "OMEGA2"),
]


def encode_training_example(
    review: str,
    expected_label: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": review},
    ]

    full_messages = prompt_messages + [
        {"role": "assistant", "content": expected_label}
    ]

    prompt_encoding = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    full_encoding = tokenizer.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
    )

    prompt_ids = prompt_encoding["input_ids"]
    input_ids = full_encoding["input_ids"]

    prompt_length = prompt_ids.shape[1]

    assert torch.equal(
        input_ids[:, :prompt_length],
        prompt_ids,
    )

    labels = input_ids.clone()
    labels[:, :prompt_length] = -100

    return input_ids.to(device), labels.to(device)


optimizer = torch.optim.AdamW(
    [parameter for parameter in model.parameters()
     if parameter.requires_grad],
    lr=5e-4,
    weight_decay=0.0,
)

generator = torch.Generator().manual_seed(42)
epoch_losses = []

model.config.use_cache = False

for epoch in range(8):
    model.train()

    order = torch.randperm(
        len(TRAINING_CASES),
        generator=generator,
    ).tolist()

    total_loss = 0.0

    for index in order:
        review, expected_label = TRAINING_CASES[index]
        input_ids, labels = encode_training_example(
            review,
            expected_label,
        )

        optimizer.zero_grad(set_to_none=True)

        loss = model(
            input_ids=input_ids,
            labels=labels,
        ).loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    mean_loss = total_loss / len(TRAINING_CASES)
    epoch_losses.append(mean_loss)

    print(
        f"epoch={epoch + 1:02d} "
        f"mean_loss={mean_loss:.6f}"
    )

model.eval()
model.config.use_cache = True

correct = 0

print("\nadapted behavior:")

for review, expected in VALIDATION_CASES:
    completion = generate_label(review)
    is_correct = completion == expected
    correct += int(is_correct)

    print(
        f"expected={expected!r} "
        f"completion={completion!r} "
        f"correct={is_correct}"
    )

print(
    f"adapted exact accuracy: "
    f"{correct}/{len(VALIDATION_CASES)}"
)

print(
    f"loss change: "
    f"{epoch_losses[0]:.6f} -> {epoch_losses[-1]:.6f}"
)

assert epoch_losses[-1] < epoch_losses[0]
