import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"

prompt_messages = [
    {
        "role": "system",
        "content": "Follow the user's transformation instruction.",
    },
    {
        "role": "user",
        "content": "Rewrite in uppercase: hello world",
    },
]

full_messages = [
    *prompt_messages,
    {
        "role": "assistant",
        "content": "HELLO WORLD",
    },
]

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

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

prompt_ids = prompt_encoding["input_ids"][0]
full_ids = full_encoding["input_ids"][0]

prompt_length = prompt_ids.shape[0]

assert torch.equal(
    full_ids[:prompt_length],
    prompt_ids,
)

labels = full_ids.clone()
labels[:prompt_length] = -100

print("prompt tokens:", prompt_length)
print("full tokens:", full_ids.shape[0])
print(
    "supervised text:",
    repr(
        tokenizer.decode(
            full_ids[prompt_length:],
            skip_special_tokens=False,
        )
    ),
)

for index, (token_id, label) in enumerate(
    zip(full_ids.tolist(), labels.tolist())
):
    token = tokenizer.convert_ids_to_tokens(token_id)
    status = "MASK" if label == -100 else "LOSS"

    print(
        f"{index:02d} "
        f"id={token_id:5d} "
        f"{status:4s} "
        f"{token!r}"
    )

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
model.to(device)
model.train()

batch_input_ids = full_ids.unsqueeze(0).to(device)
batch_labels = labels.unsqueeze(0).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-5,
)

print(model.parameters())
# Measure the initial loss.
with torch.no_grad():
    loss_before = model(
        input_ids=batch_input_ids,
        labels=batch_labels,
    ).loss

# One training step.
optimizer.zero_grad()

output = model(
    input_ids=batch_input_ids,
    labels=batch_labels,
)
training_loss = output.loss

training_loss.backward()

# Inspect two representative gradients.
embedding_grad = model.model.embed_tokens.weight.grad
lm_head_grad = model.lm_head.weight.grad

print(f"embedding gradient norm: {embedding_grad.norm().item():.8f}")
print(f"LM-head gradient norm:   {lm_head_grad.norm().item():.8f}")

optimizer.step()

# Evaluate the same example after the update.
model.eval()

with torch.no_grad():
    loss_after = model(
        input_ids=batch_input_ids,
        labels=batch_labels,
    ).loss

print(f"loss before: {loss_before.item():.10f}")
print(f"training loss: {training_loss.item():.10f}")
print(f"loss after:  {loss_after.item():.10f}")
print(f"loss decrease: {(loss_before - loss_after).item():.10f}")

assert embedding_grad is not None
assert lm_head_grad is not None
assert embedding_grad.norm() > 0
assert lm_head_grad.norm() > 0
assert loss_after < loss_before
