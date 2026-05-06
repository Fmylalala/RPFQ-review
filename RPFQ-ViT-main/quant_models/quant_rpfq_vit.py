"""Quantized Vision Transformer based on timm VisionTransformer."""

from __future__ import annotations

from typing import Dict

import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer

try:
    from timm.models import register_model
except ImportError:
    from timm.models.registry import register_model

try:
    from .quant_i import QATLinearComplexPhase, SUPPORTED_QUANT_METHODS
except ImportError:
    from quant_i import QATLinearComplexPhase, SUPPORTED_QUANT_METHODS


class ComplexVisionTransformer(VisionTransformer):
    """
    Quantized VisionTransformer.

    Features:
    - reuse the timm VisionTransformer structure
    - replace eligible Linear layers with QATLinearComplexPhase during initialization

    Implementation:
    - keep the timm model-parameter construction flow
    - scan the module tree and replace standard Linear layers plus Attention qkv/proj
    - keep FP layers for odd dimensions to avoid incompatible complex splits
    """

    def __init__(
        self,
        *args,
        quant_method_u: str = "complex_phase_v1",
        quant_method_w: str = "complex_phase_v1",
        enable_act_quant: bool = True,
        act_num_bits: int = 4,
        enable_rotation: bool = True,
        quant_group_size: int = 32,
        **kwargs,
    ):
        # Accept pretrained-config keys that some timm call paths may pass.
        kwargs.pop("pretrained_cfg", None)
        kwargs.pop("pretrained_cfg_overlay", None)
        kwargs.pop("cache_dir", None)

        super().__init__(*args, **kwargs)

        self.quant_method_u = quant_method_u
        self.quant_method_w = quant_method_w
        self.enable_act_quant = enable_act_quant
        self.act_num_bits = act_num_bits
        self.enable_rotation = enable_rotation
        self.quant_group_size = quant_group_size
        self.target_linear = QATLinearComplexPhase

        if quant_method_u not in SUPPORTED_QUANT_METHODS or quant_method_w not in SUPPORTED_QUANT_METHODS:
            print(
                f"Warning: non-standard quant methods: U={quant_method_u}, W={quant_method_w}. "
                f"Supported: {SUPPORTED_QUANT_METHODS}"
            )

        print(
            f"Using Quant Method - U: {self.quant_method_u}, "
            f"W: {self.quant_method_w} | Act Quant: {self.enable_act_quant} "
            f"({self.act_num_bits}-bit) | LGPR Rotation: {self.enable_rotation}"
        )
        # Replace layers immediately after construction.
        self._replace_quant_layers()

    def _create_quant_layer(self, original_layer: nn.Linear) -> nn.Module:
        """
        Convert one nn.Linear into a quantized linear layer and copy weights/bias.

        If layer dimensions violate the complex quantization constraint, return the original layer.
        """
        can_use_complex = (original_layer.in_features % 2 == 0) and (original_layer.out_features % 2 == 0)
        if not can_use_complex:
            print(
                f"Warning: keep FP layer ({original_layer.in_features}, {original_layer.out_features}) "
                f"because complex-phase quant requires even dimensions."
            )
            return original_layer

        new_layer = self.target_linear(
            in_features=original_layer.in_features,
            out_features=original_layer.out_features,
            bias=original_layer.bias is not None,
            quant_method_u=self.quant_method_u,
            quant_method_w=self.quant_method_w,
            enable_act_quant=self.enable_act_quant,
            act_num_bits=self.act_num_bits,
            enable_rotation=self.enable_rotation,
            quant_group_size=self.quant_group_size,
        )
        new_layer.weight.data.copy_(original_layer.weight.data)
        if original_layer.bias is not None and new_layer.bias is not None:
            new_layer.bias.data.copy_(original_layer.bias.data)
        return new_layer

    def _replace_quant_layers(self) -> None:
        """
        Traverse and replace linear layers.

        Implementation details:
        - collect parent/child edges first, then replace them to avoid mutating during traversal
        - replace only qkv/proj linear layers inside Attention modules
        - attempt to replace all standard Linear layers outside Attention modules
        """
        # Snapshot first, then replace to avoid changing structure during iteration.
        linear_edges = []
        for parent in self.modules():
            for child_name, child in parent.named_children():
                if isinstance(child, nn.Linear) and not isinstance(child, self.target_linear):
                    linear_edges.append((parent, child_name, child))

        for parent, child_name, child in linear_edges:
            is_attention_parent = "Attention" in parent.__class__.__name__
            if is_attention_parent and child_name not in {"qkv", "proj"}:
                continue

            quant_layer = self._create_quant_layer(child)
            if quant_layer is not child:
                setattr(parent, child_name, quant_layer)


def _build_rpfq_vit_model(
    *,
    model_cfg: Dict[str, int],
    pretrained: bool = False,
    **kwargs,
) -> ComplexVisionTransformer:
    """
    Unified model builder to avoid duplicating base/small/large/tiny construction.

    Responsibilities:
    - parse and normalize old and new quantization parameter names
    - assemble ComplexVisionTransformer
    - reserve a hook for pretrained weight migration
    """
    # Backward-compatible parameters retained but unused in the current implementation.
    kwargs.pop("quant_bits", None)
    kwargs.pop("quant_scheme", None)

    quant_method_u = kwargs.pop("quant_method_u", "complex_phase_v1")
    quant_method_w = kwargs.pop("quant_method_w", "complex_phase_v1")
    if "quant_method" in kwargs:
        legacy_method = kwargs.pop("quant_method")
        quant_method_u = legacy_method
        quant_method_w = legacy_method

    enable_rotation = kwargs.pop("enable_rotation", True)
    quant_group_size = kwargs.pop("quant_group_size", 32)
    enable_act_quant = kwargs.pop("enable_act_quant", True)
    act_num_bits = kwargs.pop("act_num_bits", 4)

    model = ComplexVisionTransformer(
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        quant_method_u=quant_method_u,
        quant_method_w=quant_method_w,
        enable_act_quant=enable_act_quant,
        act_num_bits=act_num_bits,
        enable_rotation=enable_rotation,
        quant_group_size=quant_group_size,
        **model_cfg,
        **kwargs,
    )

    if pretrained:
        # TODO: load FP32 checkpoints then run quantization-aware adaptation if needed.
        pass
    return model


@register_model
def rpfq_vit_base_patch16_224_q(pretrained: bool = False, **kwargs):
    """Register the quantized RPFQViT-Base/16 model."""
    return _build_rpfq_vit_model(
        pretrained=pretrained,
        model_cfg={
            "patch_size": 16,
            "embed_dim": 768,
            "depth": 12,
            "num_heads": 12,
            "mlp_ratio": 4,
        },
        **kwargs,
    )


@register_model
def rpfq_vit_large_patch16_224_q(pretrained: bool = False, **kwargs):
    """Register the quantized RPFQViT-Large/16 model."""
    return _build_rpfq_vit_model(
        pretrained=pretrained,
        model_cfg={
            "patch_size": 16,
            "embed_dim": 1024,
            "depth": 24,
            "num_heads": 16,
            "mlp_ratio": 4,
        },
        **kwargs,
    )


@register_model
def rpfq_vit_small_patch16_224_q(pretrained: bool = False, **kwargs):
    """Register the quantized RPFQViT-Small/16 model."""
    return _build_rpfq_vit_model(
        pretrained=pretrained,
        model_cfg={
            "patch_size": 16,
            "embed_dim": 384,
            "depth": 12,
            "num_heads": 6,
            "mlp_ratio": 4,
        },
        **kwargs,
    )


@register_model
def rpfq_vit_tiny_patch16_224_q(pretrained: bool = False, **kwargs):
    """Register the quantized RPFQViT-Tiny/16 model."""
    return _build_rpfq_vit_model(
        pretrained=pretrained,
        model_cfg={
            "patch_size": 16,
            "embed_dim": 192,
            "depth": 12,
            "num_heads": 3,
            "mlp_ratio": 4,
        },
        **kwargs,
    )
