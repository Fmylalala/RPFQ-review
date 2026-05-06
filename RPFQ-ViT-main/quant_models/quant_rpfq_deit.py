"""Complex-phase quantized DeiT models."""

from __future__ import annotations

from typing import Dict

import torch.nn as nn
from timm.models.deit import VisionTransformerDistilled

try:
    from timm.models import register_model
except ImportError:
    from timm.models.registry import register_model

from .quant_i import QATLinearComplexPhase, SUPPORTED_QUANT_METHODS


def _validate_quant_method(method: str) -> None:
    if method not in SUPPORTED_QUANT_METHODS:
        raise ValueError(
            f"Unsupported DeiT complex quantization method: {method}. "
            f"Allowed methods: {SUPPORTED_QUANT_METHODS}"
        )


class ComplexDeiT(VisionTransformerDistilled):
    """DeiT with compatible Linear layers replaced by QATLinearComplexPhase."""

    def __init__(
        self,
        *args,
        quant_method: str = "complex_phase_v1",
        quant_method_u: str | None = None,
        quant_method_w: str | None = None,
        enable_act_quant: bool = True,
        act_num_bits: int = 4,
        enable_rotation: bool = False,
        quant_group_size: int = 32,
        **kwargs,
    ):
        kwargs.pop("pretrained_cfg", None)
        kwargs.pop("pretrained_cfg_overlay", None)
        kwargs.pop("cache_dir", None)

        super().__init__(*args, **kwargs)

        self.quant_method_u = quant_method if quant_method_u is None else quant_method_u
        self.quant_method_w = quant_method if quant_method_w is None else quant_method_w
        _validate_quant_method(self.quant_method_u)
        _validate_quant_method(self.quant_method_w)

        self.enable_rotation = enable_rotation
        self.quant_group_size = quant_group_size
        self.enable_act_quant = enable_act_quant
        self.act_num_bits = act_num_bits
        self.target_linear = QATLinearComplexPhase

        print(
            f"Using Quant Method - U: {self.quant_method_u}, "
            f"W: {self.quant_method_w} | Act Quant: {self.enable_act_quant} "
            f"({self.act_num_bits}-bit) | LGPR Rotation: {self.enable_rotation}"
        )
        self._replace_quant_layers()

    def _create_quant_layer(self, original_layer: nn.Linear) -> nn.Module:
        can_use_complex = (
            original_layer.in_features % 2 == 0
            and original_layer.out_features % 2 == 0
        )
        if not can_use_complex:
            print(
                f"Warning: keep FP layer ({original_layer.in_features}, "
                f"{original_layer.out_features}) because complex-phase quant "
                f"requires even dimensions."
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
        linear_edges = []
        for parent in self.modules():
            for child_name, child in parent.named_children():
                if isinstance(child, nn.Linear) and not isinstance(child, self.target_linear):
                    linear_edges.append((parent, child_name, child))

        for parent, child_name, child in linear_edges:
            quant_layer = self._create_quant_layer(child)
            if quant_layer is not child:
                setattr(parent, child_name, quant_layer)


def _resolve_quant_kwargs(kwargs: dict) -> dict:
    quant_method = kwargs.pop("quant_method", "complex_phase_v1")
    quant_method_u = kwargs.pop("quant_method_u", None)
    quant_method_w = kwargs.pop("quant_method_w", None)
    enable_rotation = kwargs.pop("enable_rotation", False)
    quant_group_size = kwargs.pop("quant_group_size", 32)
    enable_act_quant = kwargs.pop("enable_act_quant", True)
    act_num_bits = kwargs.pop("act_num_bits", 4)
    return {
        "quant_method": quant_method,
        "quant_method_u": quant_method_u,
        "quant_method_w": quant_method_w,
        "enable_act_quant": enable_act_quant,
        "act_num_bits": act_num_bits,
        "enable_rotation": enable_rotation,
        "quant_group_size": quant_group_size,
    }


def _build_rpfq_deit_model(
    *,
    model_cfg: Dict[str, int],
    pretrained: bool = False,
    **kwargs,
) -> ComplexDeiT:
    quant_kwargs = _resolve_quant_kwargs(kwargs)
    model = ComplexDeiT(
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        **model_cfg,
        **quant_kwargs,
        **kwargs,
    )
    if pretrained:
        pass
    return model


@register_model
def rpfq_deit_base_patch16_224_q(pretrained: bool = False, **kwargs):
    """Complex-phase quantized DeiT-Base/16."""
    return _build_rpfq_deit_model(
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
def rpfq_deit_small_patch16_224_q(pretrained: bool = False, **kwargs):
    """Complex-phase quantized DeiT-Small/16."""
    return _build_rpfq_deit_model(
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
def rpfq_deit_tiny_patch16_224_q(pretrained: bool = False, **kwargs):
    """Complex-phase quantized DeiT-Tiny/16."""
    return _build_rpfq_deit_model(
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
