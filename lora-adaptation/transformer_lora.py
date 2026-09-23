from model import TinyCausalTransformer
from lora import LoRALinear


def inject_qkv_lora(
    model: TinyCausalTransformer,
    rank: int,
    alpha: float,
) -> TinyCausalTransformer:
    # Freeze the complete pretrained model.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Add new trainable parameters around each frozen QKV layer.
    for block in model.blocks:
        base_qkv = block.attention.qkv_projection

        block.attention.qkv_projection = LoRALinear(
            base=base_qkv,
            rank=rank,
            alpha=alpha,
        )

    return model


def set_lora_enabled(
    model: TinyCausalTransformer,
    enabled: bool,
) -> None:
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.enabled = enabled


def merge_qkv_lora(
    model: TinyCausalTransformer,
) -> TinyCausalTransformer:
    for block in model.blocks:
        layer = block.attention.qkv_projection

        if not isinstance(layer, LoRALinear):
            raise TypeError("QKV projection is not a LoRALinear")

        block.attention.qkv_projection = (
            layer.to_merged_linear()
        )

    return model
