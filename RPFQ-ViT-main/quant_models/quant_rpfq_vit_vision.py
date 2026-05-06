"""Vision-aware quantized ViT with token-saliency activation quantization."""

from __future__ import annotations

import math
from typing import Dict

import torch
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


EPS = 1e-6


class VisionAwareActivationQuantizer(nn.Module):
    """
    Token-saliency-aware activation quantizer for ViT blocks.

    The quantizer keeps the original per-token dynamic int8 quantization for
    salient tokens, while low-saliency patch tokens share a tighter image-level
    scale. This biases precision towards visually informative regions.
    """

    def __init__(
        self,
        salient_token_ratio: float = 0.25,
        background_clip_ratio: float = 0.75,
        energy_score_weight: float = 0.5,
        protect_cls_token: bool = True,
    ):
        super().__init__()
        if not 0.0 <= salient_token_ratio <= 1.0:
            raise ValueError("salient_token_ratio must be in [0, 1].")
        if background_clip_ratio <= 0.0:
            raise ValueError("background_clip_ratio must be > 0.")
        if not 0.0 <= energy_score_weight <= 1.0:
            raise ValueError("energy_score_weight must be in [0, 1].")

        self.salient_token_ratio = salient_token_ratio
        self.background_clip_ratio = background_clip_ratio
        self.energy_score_weight = energy_score_weight
        self.protect_cls_token = protect_cls_token

    @staticmethod
    def _ste_quantize(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        qx = (x * scale).round().clamp(-128, 127) / scale
        return x + (qx - x).detach()

    @staticmethod
    def _per_token_scale(x: torch.Tensor) -> torch.Tensor:
        return 127.0 / x.abs().amax(dim=-1, keepdim=True).clamp_min(EPS)

    @staticmethod
    def _infer_patch_grid(num_tokens: int) -> int:
        patch_tokens = num_tokens - 1
        if patch_tokens <= 0:
            return 0

        grid_size = math.isqrt(patch_tokens)
        return grid_size if grid_size * grid_size == patch_tokens else 0

    def _compute_patch_saliency(
        self,
        x_real: torch.Tensor,
        x_imag: torch.Tensor,
        grid_size: int,
    ) -> torch.Tensor:
        token_energy = torch.sqrt(x_real.square() + x_imag.square() + EPS).mean(dim=-1)
        energy_grid = token_energy.reshape(token_energy.shape[0], grid_size, grid_size)

        contrast = torch.zeros_like(energy_grid)
        neighbor_count = torch.zeros_like(energy_grid)

        vertical_diff = (energy_grid[:, 1:, :] - energy_grid[:, :-1, :]).abs()
        contrast[:, 1:, :] += vertical_diff
        contrast[:, :-1, :] += vertical_diff
        neighbor_count[:, 1:, :] += 1.0
        neighbor_count[:, :-1, :] += 1.0

        horizontal_diff = (energy_grid[:, :, 1:] - energy_grid[:, :, :-1]).abs()
        contrast[:, :, 1:] += horizontal_diff
        contrast[:, :, :-1] += horizontal_diff
        neighbor_count[:, :, 1:] += 1.0
        neighbor_count[:, :, :-1] += 1.0

        contrast = contrast / neighbor_count.clamp_min(1.0)
        energy_norm = energy_grid / energy_grid.mean(dim=(1, 2), keepdim=True).clamp_min(EPS)
        contrast_norm = contrast / contrast.mean(dim=(1, 2), keepdim=True).clamp_min(EPS)

        score = (
            self.energy_score_weight * energy_norm
            + (1.0 - self.energy_score_weight) * contrast_norm
        )
        return score.reshape(token_energy.shape[0], -1)

    def _build_saliency_mask(
        self,
        x_real: torch.Tensor,
        x_imag: torch.Tensor,
    ) -> torch.Tensor | None:
        if x_real.ndim != 3 or x_imag.ndim != 3:
            return None

        batch_size, num_tokens, _ = x_real.shape
        grid_size = self._infer_patch_grid(num_tokens)
        if grid_size == 0:
            return None

        patch_scores = self._compute_patch_saliency(
            x_real[:, 1:, :],
            x_imag[:, 1:, :],
            grid_size=grid_size,
        )
        num_patch_tokens = patch_scores.shape[1]

        if self.salient_token_ratio <= 0.0:
            num_salient = 0
        else:
            num_salient = max(1, math.ceil(num_patch_tokens * self.salient_token_ratio))
        num_salient = min(num_patch_tokens, num_salient)

        patch_mask = torch.zeros_like(patch_scores, dtype=torch.bool)
        if num_salient > 0:
            salient_indices = patch_scores.topk(num_salient, dim=1).indices
            patch_mask.scatter_(1, salient_indices, True)

        cls_mask = torch.full(
            (batch_size, 1),
            fill_value=self.protect_cls_token,
            dtype=torch.bool,
            device=x_real.device,
        )
        return torch.cat((cls_mask, patch_mask), dim=1).unsqueeze(-1)

    def _shared_background_scale(self, x: torch.Tensor, background_mask: torch.Tensor) -> torch.Tensor:
        masked_abs = x.abs() * background_mask.to(dtype=x.dtype)
        background_max = masked_abs.amax(dim=(1, 2), keepdim=True)
        clipped_max = (background_max * self.background_clip_ratio).clamp_min(EPS)
        return 127.0 / clipped_max

    def _fallback_quantize(self, x_real: torch.Tensor, x_imag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        real_scale = self._per_token_scale(x_real)
        imag_scale = self._per_token_scale(x_imag)
        return self._ste_quantize(x_real, real_scale), self._ste_quantize(x_imag, imag_scale)

    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        saliency_mask = self._build_saliency_mask(x_real, x_imag)
        if saliency_mask is None:
            return self._fallback_quantize(x_real, x_imag)

        salient_real_scale = self._per_token_scale(x_real)
        salient_imag_scale = self._per_token_scale(x_imag)

        background_mask = ~saliency_mask
        background_real_scale = self._shared_background_scale(x_real, background_mask)
        background_imag_scale = self._shared_background_scale(x_imag, background_mask)

        real_scale = torch.where(saliency_mask, salient_real_scale, background_real_scale)
        imag_scale = torch.where(saliency_mask, salient_imag_scale, background_imag_scale)
        return self._ste_quantize(x_real, real_scale), self._ste_quantize(x_imag, imag_scale)


class VisionAwareQATLinearComplexPhase(QATLinearComplexPhase):
    """Complex-phase QAT linear layer with vision-aware activation quantization."""

    def __init__(
        self,
        quant_method_u: str = "complex_phase_v1",
        quant_method_w: str = "complex_phase_v1",
        enable_act_quant: bool = True,
        act_num_bits: int = 4,
        enable_rotation: bool = True,
        quant_group_size: int = 32,
        salient_token_ratio: float = 0.25,
        background_clip_ratio: float = 0.75,
        energy_score_weight: float = 0.5,
        protect_cls_token: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(
            quant_method_u=quant_method_u,
            quant_method_w=quant_method_w,
            enable_act_quant=enable_act_quant,
            act_num_bits=act_num_bits,
            enable_rotation=enable_rotation,
            quant_group_size=quant_group_size,
            *args,
            **kwargs,
        )
        if enable_act_quant:
            self.act_quantizer = VisionAwareActivationQuantizer(
                salient_token_ratio=salient_token_ratio,
                background_clip_ratio=background_clip_ratio,
                energy_score_weight=energy_score_weight,
                protect_cls_token=protect_cls_token,
            )


class VisionAwareComplexVisionTransformer(VisionTransformer):
    """Quantized ViT with token-saliency-aware activation quantization."""

    def __init__(
        self,
        *args,
        quant_method_u: str = "complex_phase_v1",
        quant_method_w: str = "complex_phase_v1",
        enable_act_quant: bool = True,
        act_num_bits: int = 4,
        enable_rotation: bool = True,
        quant_group_size: int = 32,
        salient_token_ratio: float = 0.25,
        background_clip_ratio: float = 0.75,
        energy_score_weight: float = 0.5,
        protect_cls_token: bool = True,
        **kwargs,
    ):
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
        self.salient_token_ratio = salient_token_ratio
        self.background_clip_ratio = background_clip_ratio
        self.energy_score_weight = energy_score_weight
        self.protect_cls_token = protect_cls_token
        self.target_linear = VisionAwareQATLinearComplexPhase

        if quant_method_u not in SUPPORTED_QUANT_METHODS or quant_method_w not in SUPPORTED_QUANT_METHODS:
            print(
                f"Warning: non-standard quant methods: U={quant_method_u}, W={quant_method_w}. "
                f"Supported: {SUPPORTED_QUANT_METHODS}"
            )

        print(
            f"Using Quant Method - U: {self.quant_method_u}, "
            f"W: {self.quant_method_w} | Act Quant: {self.enable_act_quant} "
            f"({self.act_num_bits}-bit) | LGPR Rotation: {self.enable_rotation} | "
            f"Vision-aware act quant ratio: {self.salient_token_ratio}"
        )
        self._replace_quant_layers()

    def _create_quant_layer(self, original_layer: nn.Linear) -> nn.Module:
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
            salient_token_ratio=self.salient_token_ratio,
            background_clip_ratio=self.background_clip_ratio,
            energy_score_weight=self.energy_score_weight,
            protect_cls_token=self.protect_cls_token,
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
            is_attention_parent = "Attention" in parent.__class__.__name__
            if is_attention_parent and child_name not in {"qkv", "proj"}:
                continue

            quant_layer = self._create_quant_layer(child)
            if quant_layer is not child:
                setattr(parent, child_name, quant_layer)


def _build_rpfq_vit_vision_model(
    *,
    model_cfg: Dict[str, int],
    pretrained: bool = False,
    **kwargs,
) -> VisionAwareComplexVisionTransformer:
    kwargs.pop("quant_bits", None)
    kwargs.pop("quant_scheme", None)

    quant_method_u = kwargs.pop("quant_method_u", "complex_phase_v1")
    quant_method_w = kwargs.pop("quant_method_w", "complex_phase_v1")
    if "quant_method" in kwargs:
        legacy_method = kwargs.pop("quant_method")
        quant_method_u = legacy_method
        quant_method_w = legacy_method

    model = VisionAwareComplexVisionTransformer(
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        quant_method_u=quant_method_u,
        quant_method_w=quant_method_w,
        enable_act_quant=kwargs.pop("enable_act_quant", True),
        act_num_bits=kwargs.pop("act_num_bits", 4),
        enable_rotation=kwargs.pop("enable_rotation", True),
        quant_group_size=kwargs.pop("quant_group_size", 32),
        salient_token_ratio=kwargs.pop("salient_token_ratio", 0.25),
        background_clip_ratio=kwargs.pop("background_clip_ratio", 0.75),
        energy_score_weight=kwargs.pop("energy_score_weight", 0.5),
        protect_cls_token=kwargs.pop("protect_cls_token", True),
        **model_cfg,
        **kwargs,
    )

    if pretrained:
        pass
    return model


@register_model
def rpfq_vit_base_patch16_224_q_vision(pretrained: bool = False, **kwargs):
    return _build_rpfq_vit_vision_model(
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
def rpfq_vit_large_patch16_224_q_vision(pretrained: bool = False, **kwargs):
    return _build_rpfq_vit_vision_model(
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
def rpfq_vit_small_patch16_224_q_vision(pretrained: bool = False, **kwargs):
    return _build_rpfq_vit_vision_model(
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
def rpfq_vit_tiny_patch16_224_q_vision(pretrained: bool = False, **kwargs):
    return _build_rpfq_vit_vision_model(
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
