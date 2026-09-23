import torch

from data import (
    BOS,
    EOS,
    SEP,
    build_all_sequences,
    make_data_loader,
    split_sequences,
)
from generate import evaluate_generation
from test_checkpoint_lora import load_trained_model
from train import evaluate_teacher_forced, train_one_epoch
from lora import LoRALinear
from transformer_lora import (
    inject_qkv_lora,
    merge_qkv_lora,
    set_lora_enabled,
)


def build_copy_sequences() -> torch.Tensor:
    digits = torch.arange(10)
    triples = torch.cartesian_prod(digits, digits, digits)
    a, b, c = triples.unbind(dim=1)

    bos = torch.full((1000,), BOS)
    sep = torch.full((1000,), SEP)
    eos = torch.full((1000,), EOS)

    return torch.stack(
        (bos, a, b, c, sep, a, b, c, eos),
        dim=1,
    ).long()


torch.manual_seed(0)

reverse_sequences = build_all_sequences()
copy_sequences = build_copy_sequences()

_, reverse_validation = split_sequences(
    reverse_sequences,
    seed=42,
)

copy_train, copy_validation = split_sequences(
    copy_sequences,
    seed=42,
)

copy_train_loader = make_data_loader(
    copy_train,
    batch_size=64,
    shuffle=True,
    seed=42,
)

copy_validation_loader = make_data_loader(
    copy_validation,
    batch_size=64,
    shuffle=False,
)

model = load_trained_model()

baseline_reverse = evaluate_generation(
    model,
    reverse_validation,
)
baseline_copy = evaluate_generation(
    model,
    copy_validation,
)

print("baseline reverse:", baseline_reverse)
print("baseline copy:", baseline_copy)

inject_qkv_lora(model, rank=4, alpha=8)

trainable_parameters = [
    parameter
    for parameter in model.parameters()
    if parameter.requires_grad
]

print(
    "trainable parameters:",
    sum(parameter.numel() for parameter in trainable_parameters),
)

optimizer = torch.optim.AdamW(
    trainable_parameters,
    lr=1e-2,
    weight_decay=0.0,
)

for epoch in range(1, 21):
    train_metrics = train_one_epoch(
        model,
        copy_train_loader,
        optimizer,
    )

    if epoch == 1 or epoch % 10 == 0:
        validation_metrics = evaluate_teacher_forced(
            model,
            copy_validation_loader,
        )

        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_metrics['loss']:.6f} "
            f"val_loss={validation_metrics['loss']:.6f} "
            f"val_exact={validation_metrics['exact_sequence_accuracy']:.3f}"
        )

adapted_copy = evaluate_generation(
    model,
    copy_validation,
)

adapted_reverse = evaluate_generation(
    model,
    reverse_validation,
)

print("adapted copy:", adapted_copy)
print("adapted reverse:", adapted_reverse)

set_lora_enabled(model, False)

disabled_copy = evaluate_generation(
    model,
    copy_validation,
)
disabled_reverse = evaluate_generation(
    model,
    reverse_validation,
)

print("adapter disabled copy:", disabled_copy)
print("adapter disabled reverse:", disabled_reverse)

set_lora_enabled(model, True)

fixed_inputs = copy_validation[:16, :-1]

model.eval()
with torch.inference_mode():
    before_merge = model(fixed_inputs)

merge_qkv_lora(model)

model.eval()
with torch.inference_mode():
    after_merge = model(fixed_inputs)

merged_copy = evaluate_generation(
    model,
    copy_validation,
)
merged_reverse = evaluate_generation(
    model,
    reverse_validation,
)

print("merged copy:", merged_copy)
print("merged reverse:", merged_reverse)

difference = (after_merge - before_merge).abs()

print(
    "merge max absolute difference:",
    difference.max().item(),
)
print(
    "merge mean absolute difference:",
    difference.mean().item(),
)

torch.testing.assert_close(
    after_merge,
    before_merge,
    rtol=1e-4,
    atol=1e-5,
)

assert torch.equal(
    after_merge.argmax(dim=-1),
    before_merge.argmax(dim=-1),
)

assert not any(
    isinstance(module, LoRALinear)
    for module in model.modules()
)
