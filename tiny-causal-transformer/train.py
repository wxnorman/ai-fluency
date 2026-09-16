import copy
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from data import (
    VOCABULARY,
    build_all_sequences,
    make_data_loader,
    make_examples,
    split_sequences,
)
from generate import evaluate_generation
from model import TinyCausalTransformer

def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError(
            f"Expected logits [B, T, V], got {logits.shape}"
        )

    if targets.shape != logits.shape[:2]:
        raise ValueError(
            f"Targets shape {targets.shape} does not match "
            f"logits batch/time shape {logits.shape[:2]}"
        )

    if loss_mask.shape != targets.shape:
        raise ValueError(
            f"Loss mask shape {loss_mask.shape} does not "
            f"match targets shape {targets.shape}"
        )

    if loss_mask.dtype != torch.bool:
        raise ValueError("Loss mask must have boolean dtype")

    batch_size, time, vocab_size = logits.shape

    per_token_loss = F.cross_entropy(
        logits.reshape(batch_size * time, vocab_size),
        targets.reshape(batch_size * time),
        reduction="none",
    )

    per_token_loss = per_token_loss.view(
        batch_size,
        time,
    )

    weights = loss_mask.to(per_token_loss.dtype)
    number_of_active_tokens = weights.sum()

    if number_of_active_tokens.item() == 0:
        raise ValueError("Loss mask contains no active tokens")

    return (
        per_token_loss * weights
    ).sum() / number_of_active_tokens


def gradient_norm_summary(
    model: torch.nn.Module,
) -> dict[str, float]:
    summary = {}

    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue

        if not torch.isfinite(parameter.grad).all():
            raise ValueError(
                f"Non-finite gradient detected for {name}"
            )

        summary[name] = parameter.grad.norm().item()

    return summary


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def train_one_epoch(
    model: torch.nn.Module,
    data_loader,
    optimizer: torch.optim.Optimizer,
    max_gradient_norm: float = 1.0,
) -> dict[str, float]:
    model.train()

    total_loss = 0.0
    total_active_tokens = 0
    largest_gradient_norm = 0.0

    for inputs, targets, loss_mask in data_loader:
        optimizer.zero_grad(set_to_none=True)

        logits = model(inputs)

        loss = masked_cross_entropy(
            logits,
            targets,
            loss_mask,
        )

        loss.backward()

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=max_gradient_norm,
        )

        optimizer.step()

        active_tokens = loss_mask.sum().item()

        total_loss += loss.item() * active_tokens
        total_active_tokens += active_tokens

        largest_gradient_norm = max(
            largest_gradient_norm,
            float(gradient_norm),
        )

    return {
        "loss": total_loss / total_active_tokens,
        "max_gradient_norm_before_clipping": largest_gradient_norm,
    }


def completion_counts(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
) -> dict[str, int]:
    if logits.shape[:2] != targets.shape:
        raise ValueError("Logits and targets have incompatible shapes")

    if loss_mask.shape != targets.shape:
        raise ValueError("Loss mask and targets have incompatible shapes")

    if not loss_mask.any(dim=1).all():
        raise ValueError(
            "Every sequence must contain at least one completion token"
        )

    predictions = logits.argmax(dim=-1)
    token_matches = predictions == targets

    token_correct = (
        token_matches & loss_mask
    ).sum().item()

    token_total = loss_mask.sum().item()

    # Masked prompt positions count as automatically satisfied.
    sequence_matches = (
        token_matches | ~loss_mask
    ).all(dim=1)

    sequence_correct = sequence_matches.sum().item()
    sequence_total = targets.shape[0]

    return {
        "token_correct": token_correct,
        "token_total": token_total,
        "sequence_correct": sequence_correct,
        "sequence_total": sequence_total,
    }

def evaluate_teacher_forced(
    model: torch.nn.Module,
    data_loader,
) -> dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_active_tokens = 0

    total_token_correct = 0
    total_sequence_correct = 0
    total_sequences = 0

    with torch.no_grad():
        for inputs, targets, loss_mask in data_loader:
            logits = model(inputs)

            loss = masked_cross_entropy(
                logits,
                targets,
                loss_mask,
            )

            counts = completion_counts(
                logits,
                targets,
                loss_mask,
            )

            active_tokens = counts["token_total"]

            total_loss += loss.item() * active_tokens
            total_active_tokens += active_tokens

            total_token_correct += counts["token_correct"]
            total_sequence_correct += counts["sequence_correct"]
            total_sequences += counts["sequence_total"]

    return {
        "loss": total_loss / total_active_tokens,
        "token_accuracy": (
            total_token_correct / total_active_tokens
        ),
        "exact_sequence_accuracy": (
            total_sequence_correct / total_sequences
        ),
        "sequence_count": total_sequences,
    }

def fit(
    model: torch.nn.Module,
    train_loader,
    validation_loader,
    max_epochs: int = 150,
    patience: int = 15,
    learning_rate: float = 3e-3,
    weight_decay: float = 1e-4,
    max_gradient_norm: float = 1.0,
    minimum_improvement: float = 1e-6,
) -> dict:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_validation_loss = float("inf")
    best_epoch = 0
    best_model_state = None
    best_optimizer_state = None

    epochs_without_improvement = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            max_gradient_norm=max_gradient_norm,
        )

        validation_metrics = evaluate_teacher_forced(
            model,
            validation_loader,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "validation_loss": validation_metrics["loss"],
                "validation_token_accuracy": (
                    validation_metrics["token_accuracy"]
                ),
                "validation_exact_accuracy": (
                    validation_metrics["exact_sequence_accuracy"]
                ),
                "max_gradient_norm_before_clipping": (
                    train_metrics[
                        "max_gradient_norm_before_clipping"
                    ]
                ),
            }
        )

        improved = (
            validation_metrics["loss"]
            < best_validation_loss - minimum_improvement
        )

        if improved:
            best_validation_loss = validation_metrics["loss"]
            best_epoch = epoch

            best_model_state = copy.deepcopy(
                model.state_dict()
            )

            best_optimizer_state = copy.deepcopy(
                optimizer.state_dict()
            )

            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                break

    if best_model_state is None:
        raise RuntimeError(
            "Training completed without a valid model state"
        )

    model.load_state_dict(best_model_state)
    optimizer.load_state_dict(best_optimizer_state)

    return {
        "model": model,
        "optimizer": optimizer,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "best_model_state": best_model_state,
        "best_optimizer_state": best_optimizer_state,
        "epochs_ran": len(history),
        "history": history,
    }

def save_checkpoint(
    path: str | Path,
    model: TinyCausalTransformer,
    optimizer: torch.optim.Optimizer,
    seed: int,
    best_epoch: int,
    best_validation_loss: float,
) -> None:
    path = Path(path)

    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_configuration": model.configuration(),
        "vocabulary": VOCABULARY,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
    }

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path,
    map_location: str = "cpu",
) -> tuple[TinyCausalTransformer, dict]:
    checkpoint = torch.load(
        Path(path),
        map_location=map_location,
        weights_only=True,
    )

    model = TinyCausalTransformer(
        **checkpoint["model_configuration"]
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.eval()

    return model, checkpoint


def write_metrics(
    metrics: dict,
    path: str | Path,
) -> None:
    Path(path).write_text(
        json.dumps(metrics, indent=2) + "\n"
    )


def run_experiment(
    output_directory: str | Path = ".",
    seed: int = 42,
) -> dict:
    output_directory = Path(output_directory)
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        output_directory / "checkpoint.pt"
    )
    metrics_path = (
        output_directory / "metrics.json"
    )

    set_seed(seed)
    started = time.perf_counter()

    # Data
    all_sequences = build_all_sequences()

    train_sequences, validation_sequences = (
        split_sequences(
            all_sequences,
            seed=seed,
        )
    )

    train_loader = make_data_loader(
        train_sequences,
        batch_size=64,
        shuffle=True,
        seed=seed,
    )

    validation_loader = make_data_loader(
        validation_sequences,
        batch_size=64,
        shuffle=False,
        seed=seed,
    )

    # Model
    model = TinyCausalTransformer(
        vocab_size=13,
        context_length=8,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        dropout=0.1,
    )

    # Training
    training_result = fit(
        model,
        train_loader,
        validation_loader,
        max_epochs=150,
        patience=15,
        learning_rate=3e-3,
        weight_decay=1e-4,
        max_gradient_norm=1.0,
    )

    model = training_result["model"]
    optimizer = training_result["optimizer"]

    # Teacher-forced validation
    teacher_forced_metrics = (
        evaluate_teacher_forced(
            model,
            validation_loader,
        )
    )

    # True autoregressive validation
    generation_metrics = evaluate_generation(
        model,
        validation_sequences,
    )

    # Fresh diagnostic backward pass on a fixed batch.
    diagnostic_loader = make_data_loader(
        train_sequences,
        batch_size=64,
        shuffle=False,
        seed=seed,
    )

    inputs, targets, loss_mask = next(
        iter(diagnostic_loader)
    )

    model.eval()
    model.zero_grad(set_to_none=True)

    diagnostic_loss = masked_cross_entropy(
        model(inputs),
        targets,
        loss_mask,
    )

    diagnostic_loss.backward()

    all_gradient_norms = gradient_norm_summary(
        model
    )

    required_gradient_names = [
        "token_embedding.weight",
        "position_embedding.weight",
        (
            "blocks.0.attention."
            "qkv_projection.weight"
        ),
        "output_projection.weight",
    ]

    selected_gradient_norms = {
        name: all_gradient_norms[name]
        for name in required_gradient_names
    }

    # Save checkpoint.
    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        seed=seed,
        best_epoch=training_result["best_epoch"],
        best_validation_loss=training_result[
            "best_validation_loss"
        ],
    )

    # Reload and verify fixed-batch logits.
    reloaded_model, _ = load_checkpoint(
        checkpoint_path
    )

    fixed_inputs, _, _ = make_examples(
        validation_sequences[:16]
    )

    model.eval()
    reloaded_model.eval()

    with torch.no_grad():
        original_logits = model(fixed_inputs)
        reloaded_logits = reloaded_model(
            fixed_inputs
        )

    checkpoint_logits_match = torch.equal(
        original_logits,
        reloaded_logits,
    )

    if not checkpoint_logits_match:
        raise RuntimeError(
            "Reloaded checkpoint logits do not match"
        )

    elapsed_seconds = (
        time.perf_counter() - started
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    metrics = {
        "seed": seed,
        "dataset_sizes": {
            "train": train_sequences.shape[0],
            "validation": (
                validation_sequences.shape[0]
            ),
            "total": all_sequences.shape[0],
        },
        "best_epoch": training_result["best_epoch"],
        "epochs_ran": training_result["epochs_ran"],
        "best_validation_loss": training_result[
            "best_validation_loss"
        ],
        "teacher_forced": {
            "loss": teacher_forced_metrics["loss"],
            "token_accuracy": (
                teacher_forced_metrics[
                    "token_accuracy"
                ]
            ),
            "exact_sequence_accuracy": (
                teacher_forced_metrics[
                    "exact_sequence_accuracy"
                ]
            ),
        },
        "generation": {
            "token_accuracy": (
                generation_metrics[
                    "token_accuracy"
                ]
            ),
            "exact_sequence_accuracy": (
                generation_metrics[
                    "exact_sequence_accuracy"
                ]
            ),
        },
        "parameter_count": parameter_count,
        "diagnostic_loss": diagnostic_loss.item(),
        "gradient_norms": selected_gradient_norms,
        "checkpoint_logits_match": (
            checkpoint_logits_match
        ),
        "elapsed_seconds": elapsed_seconds,
    }

    write_metrics(
        metrics,
        metrics_path,
    )

    return metrics

def main() -> None:
    metrics = run_experiment()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()