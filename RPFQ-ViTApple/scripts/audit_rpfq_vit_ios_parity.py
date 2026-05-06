#!/usr/bin/env python3
"""
Audit parity between RPFQViT training, export, and iOS runtime paths.

Supported modes:
- preprocess: compare timm eval preprocessing against legacy/current Swift semantics
- weights: compare training-time quantized weights against legacy/fixed export reconstructions
- layers: compare training forward outputs against reconstructed runtime outputs
- all: run every audit
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from timm.data import create_transform, resolve_data_config

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_ROOT = REPO_ROOT / "RPFQ-ViT-main"
QUANT_MODELS_ROOT = RESEARCH_ROOT / "quant_models"
for import_path in (QUANT_MODELS_ROOT, RESEARCH_ROOT, SCRIPT_DIR):
    import_path_str = str(import_path)
    if import_path_str not in sys.path:
        sys.path.insert(0, import_path_str)

from export_rpfq_vit_to_mlx import (
    BLOCK_SIZE,
    VARIANTS,
    dequantize_axis,
    legacy_residual_phase_quantize,
    load_checkpoint,
    quantized_forward_weight,
    residual_phase_quantize,
    round_up,
    widely_linear_decompose,
)

import quant_rpfq_vit
from quant_i import QATLinearComplexPhase

PIL_RESAMPLING = getattr(Image, "Resampling", Image)


MODEL_BUILDERS = {
    "rpfq_vit_tiny_patch16_224_q": quant_rpfq_vit.rpfq_vit_tiny_patch16_224_q,
    "rpfq_vit_small_patch16_224_q": quant_rpfq_vit.rpfq_vit_small_patch16_224_q,
    "rpfq_vit_base_patch16_224_q": quant_rpfq_vit.rpfq_vit_base_patch16_224_q,
    "rpfq_vit_large_patch16_224_q": quant_rpfq_vit.rpfq_vit_large_patch16_224_q,
}


@dataclass
class RuntimeLayer:
    weight: torch.Tensor
    bias: torch.Tensor


def build_training_model(variant: str, checkpoint: Path | None = None) -> torch.nn.Module:
    state = load_checkpoint(checkpoint) if checkpoint is not None else None
    model = MODEL_BUILDERS[variant](
        pretrained=False,
        num_classes=int(state["head.weight"].shape[0]) if state is not None else 1000,
        quant_method_u="complex_phase_v2",
        quant_method_w="complex_phase_v2",
        enable_rotation=False,
    )
    if state is not None:
        model.load_state_dict(state, strict=False)
    model.eval()
    return model


def resolve_eval_config(variant: str) -> dict[str, Any]:
    model = build_training_model(variant)
    return resolve_data_config({}, model=model)


def synthesize_images() -> dict[str, Image.Image]:
    height_a, width_a = 320, 520
    y = np.linspace(0.0, 1.0, height_a, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, width_a, dtype=np.float32)[None, :]
    stripes = 0.5 + 0.5 * np.sin(8.0 * math.pi * x)
    diag = np.clip(1.0 - np.abs(y - x), 0.0, 1.0)
    non_square = np.stack(
        [
            x.repeat(height_a, axis=0),
            y.repeat(width_a, axis=1),
            0.6 * stripes.repeat(height_a, axis=0) + 0.4 * diag,
        ],
        axis=-1,
    )

    size_b = 384
    yy, xx = np.mgrid[0:size_b, 0:size_b].astype(np.float32)
    yy /= size_b - 1
    xx /= size_b - 1
    center = np.sqrt((xx - 0.5) ** 2 + (yy - 0.5) ** 2)
    checker = ((np.floor(xx * 12) + np.floor(yy * 12)) % 2) * 0.6
    square = np.stack(
        [
            np.clip(1.0 - center * 1.8, 0.0, 1.0),
            checker,
            np.clip((xx * yy) * 1.5, 0.0, 1.0),
        ],
        axis=-1,
    )

    return {
        "synthetic_non_square": Image.fromarray((non_square * 255).astype(np.uint8), mode="RGB"),
        "synthetic_square": Image.fromarray((square * 255).astype(np.uint8), mode="RGB"),
    }


def patchify_tensor(chw: torch.Tensor, patch_size: int) -> torch.Tensor:
    c, height, width = chw.shape
    grid_h = height // patch_size
    grid_w = width // patch_size
    patches = (
        chw.reshape(c, grid_h, patch_size, grid_w, patch_size)
        .permute(1, 3, 0, 2, 4)
        .reshape(grid_h * grid_w, c * patch_size * patch_size)
    )
    return patches.unsqueeze(0)


def swift_preprocess_fixed(
    image: Image.Image,
    image_size: int,
    patch_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    crop_pct: float,
    interpolation: str,
) -> torch.Tensor:
    src_w, src_h = image.size
    target_shortest = round(image_size / crop_pct)
    scale = target_shortest / min(src_w, src_h)
    resized_w = round(src_w * scale)
    resized_h = round(src_h * scale)
    resized = image.resize((resized_w, resized_h), resample=pil_resample(interpolation))
    left = max((resized_w - image_size) // 2, 0)
    top = max((resized_h - image_size) // 2, 0)
    cropped = resized.crop((left, top, left + image_size, top + image_size))
    chw = pil_to_normalized_chw(cropped, mean, std)
    return patchify_tensor(chw, patch_size)


def swift_preprocess_legacy(
    image: Image.Image,
    image_size: int,
    patch_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    interpolation: str,
) -> torch.Tensor:
    src_w, src_h = image.size
    scale = max(image_size / src_w, image_size / src_h)
    resized_w = round(src_w * scale)
    resized_h = round(src_h * scale)
    resized = image.resize((resized_w, resized_h), resample=pil_resample(interpolation))
    left = max((resized_w - image_size) // 2, 0)
    top = max((resized_h - image_size) // 2, 0)
    cropped = resized.crop((left, top, left + image_size, top + image_size))
    chw = pil_to_normalized_chw(cropped, mean, std)
    return patchify_tensor(chw, patch_size)


def pil_resample(interpolation: str) -> int:
    mapping = {
        "nearest": PIL_RESAMPLING.NEAREST,
        "bilinear": PIL_RESAMPLING.BILINEAR,
        "bicubic": PIL_RESAMPLING.BICUBIC,
    }
    return mapping.get(interpolation.lower(), PIL_RESAMPLING.BICUBIC)


def pil_to_normalized_chw(
    image: Image.Image,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    chw = torch.from_numpy(array).permute(2, 0, 1)
    mean_t = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    return (chw - mean_t) / std_t


def tensor_metrics(lhs: torch.Tensor, rhs: torch.Tensor) -> dict[str, Any]:
    lhs = lhs.detach().float().cpu()
    rhs = rhs.detach().float().cpu()
    diff = (lhs - rhs).abs()
    result = {
        "shape": list(lhs.shape),
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "first_10_lhs": [float(v) for v in lhs.reshape(-1)[:10]],
        "first_10_rhs": [float(v) for v in rhs.reshape(-1)[:10]],
    }
    if lhs.numel() == rhs.numel():
        cosine = F.cosine_similarity(lhs.reshape(1, -1), rhs.reshape(1, -1)).item()
        result["cosine_similarity"] = float(cosine)
    return result


def patch_samples(lhs: torch.Tensor, rhs: torch.Tensor) -> list[dict[str, Any]]:
    token_count = lhs.shape[1]
    sample_indices = sorted({0, token_count // 2, token_count - 1})
    samples = []
    for index in sample_indices:
        samples.append(
            {
                "patch_index": int(index),
                "lhs_values": [float(v) for v in lhs[0, index, :12]],
                "rhs_values": [float(v) for v in rhs[0, index, :12]],
            }
        )
    return samples


def run_preprocess_audit(variant: str) -> dict[str, Any]:
    data_config = resolve_eval_config(variant)
    transform = create_transform(
        input_size=data_config["input_size"],
        is_training=False,
        interpolation=data_config["interpolation"],
        mean=data_config["mean"],
        std=data_config["std"],
        crop_pct=data_config["crop_pct"],
    )
    patch_size = VARIANTS[variant]["patch_size"]
    image_size = data_config["input_size"][1]

    results: dict[str, Any] = {
        "resolved_data_config": {
            "input_size": list(data_config["input_size"]),
            "interpolation": data_config["interpolation"],
            "mean": list(data_config["mean"]),
            "std": list(data_config["std"]),
            "crop_pct": float(data_config["crop_pct"]),
            "crop_mode": "center",
        },
        "images": {},
    }

    for name, image in synthesize_images().items():
        timm_chw = transform(image).float()
        timm_patches = patchify_tensor(timm_chw, patch_size)
        legacy_patches = swift_preprocess_legacy(
            image,
            image_size=image_size,
            patch_size=patch_size,
            mean=data_config["mean"],
            std=data_config["std"],
            interpolation=data_config["interpolation"],
        )
        fixed_patches = swift_preprocess_fixed(
            image,
            image_size=image_size,
            patch_size=patch_size,
            mean=data_config["mean"],
            std=data_config["std"],
            crop_pct=float(data_config["crop_pct"]),
            interpolation=data_config["interpolation"],
        )

        results["images"][name] = {
            "timm_vs_swift_legacy": tensor_metrics(timm_patches, legacy_patches),
            "timm_vs_swift_fixed": tensor_metrics(timm_patches, fixed_patches),
            "patch_samples_legacy": patch_samples(timm_patches, legacy_patches),
            "patch_samples_fixed": patch_samples(timm_patches, fixed_patches),
        }

    return results


def train_quantized_weight(weight: torch.Tensor) -> torch.Tensor:
    layer = QATLinearComplexPhase(
        in_features=weight.shape[1],
        out_features=weight.shape[0],
        bias=False,
        quant_method_u="complex_phase_v2",
        quant_method_w="complex_phase_v2",
        enable_rotation=False,
    )
    with torch.no_grad():
        layer.weight.copy_(weight)

    half_out = weight.shape[0] // 2
    half_in = weight.shape[1] // 2
    a11 = weight[:half_out, :half_in]
    a12 = weight[:half_out, half_in:]
    a21 = weight[half_out:, :half_in]
    a22 = weight[half_out:, half_in:]

    u_real = 0.5 * (a11 + a22)
    u_imag = 0.5 * (a21 - a12)
    w_real = 0.5 * (a11 - a22)
    w_imag = 0.5 * (a12 + a21)

    u_real_q, u_imag_q = layer.quantizer_u(u_real, u_imag)
    w_real_q, w_imag_q = layer.quantizer_w(w_real, w_imag)

    a11_q = w_real_q + u_real_q
    a12_q = w_imag_q - u_imag_q
    a21_q = w_imag_q + u_imag_q
    a22_q = -w_real_q + u_real_q
    return torch.cat(
        (
            torch.cat((a11_q, a12_q), dim=1),
            torch.cat((a21_q, a22_q), dim=1),
        ),
        dim=0,
    )


def reconstruct_export_weight(
    weight: torch.Tensor,
    in_complex_padded: int,
    out_complex_padded: int,
    stages: int,
    legacy: bool,
) -> torch.Tensor:
    (u_real, u_imag), (w_real, w_imag) = widely_linear_decompose(weight.float())
    orig_out_complex, orig_in_complex = u_real.shape

    quantizer = legacy_residual_phase_quantize if legacy else residual_phase_quantize
    kwargs = {} if legacy else {"padded_out_dim": out_complex_padded, "padded_in_dim": in_complex_padded}

    def reconstruct_component(real: torch.Tensor, imag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if legacy:
            padded_real = torch.zeros((out_complex_padded, in_complex_padded), dtype=torch.float32)
            padded_imag = torch.zeros((out_complex_padded, in_complex_padded), dtype=torch.float32)
            padded_real[: real.shape[0], : real.shape[1]] = real
            padded_imag[: imag.shape[0], : imag.shape[1]] = imag
            quant_real = padded_real
            quant_imag = padded_imag
        else:
            quant_real = real
            quant_imag = imag

        acc_real = torch.zeros((out_complex_padded, in_complex_padded), dtype=torch.float32)
        acc_imag = torch.zeros((out_complex_padded, in_complex_padded), dtype=torch.float32)
        for codes, gamma_real, gamma_imag in quantizer(quant_real, quant_imag, stages=stages, **kwargs):
            deq_real, deq_imag = dequantize_axis(
                codes,
                gamma_real,
                gamma_imag,
                out_dim=acc_real.shape[0],
                in_dim=acc_real.shape[1],
            )
            acc_real += deq_real
            acc_imag += deq_imag
        return (
            acc_real[:orig_out_complex, :orig_in_complex],
            acc_imag[:orig_out_complex, :orig_in_complex],
        )

    u_real_q, u_imag_q = reconstruct_component(u_real, u_imag)
    w_real_q, w_imag_q = reconstruct_component(w_real, w_imag)

    a11_q = w_real_q + u_real_q
    a12_q = w_imag_q - u_imag_q
    a21_q = w_imag_q + u_imag_q
    a22_q = -w_real_q + u_real_q
    return torch.cat(
        (
            torch.cat((a11_q, a12_q), dim=1),
            torch.cat((a21_q, a22_q), dim=1),
        ),
        dim=0,
    )


def representative_layers(state: dict[str, torch.Tensor], variant: str) -> list[tuple[str, torch.Tensor, int, int]]:
    hidden_complex = VARIANTS[variant]["hidden_size"] // 2
    intermediate_complex = VARIANTS[variant]["intermediate_size"] // 2
    hidden_complex_padded = round_up(hidden_complex, BLOCK_SIZE)
    qkv_complex_padded = round_up(hidden_complex * 3, BLOCK_SIZE)
    intermediate_complex_padded = round_up(intermediate_complex, BLOCK_SIZE)
    return [
        ("blocks.0.attn.qkv", state["blocks.0.attn.qkv.weight"], hidden_complex_padded, qkv_complex_padded),
        ("blocks.0.attn.proj", state["blocks.0.attn.proj.weight"], hidden_complex_padded, hidden_complex_padded),
        ("blocks.0.mlp.fc1", state["blocks.0.mlp.fc1.weight"], hidden_complex_padded, intermediate_complex_padded),
        ("blocks.0.mlp.fc2", state["blocks.0.mlp.fc2.weight"], intermediate_complex_padded, hidden_complex_padded),
    ]


def run_weight_audit(checkpoint: Path, variant: str) -> dict[str, Any]:
    state = load_checkpoint(checkpoint)
    stages = 2
    results: dict[str, Any] = {"layers": {}}

    for name, weight, in_complex_padded, out_complex_padded in representative_layers(state, variant):
        train_q = train_quantized_weight(weight.float())
        legacy_q = reconstruct_export_weight(
            weight.float(),
            in_complex_padded=in_complex_padded,
            out_complex_padded=out_complex_padded,
            stages=stages,
            legacy=True,
        )
        fixed_q = reconstruct_export_weight(
            weight.float(),
            in_complex_padded=in_complex_padded,
            out_complex_padded=out_complex_padded,
            stages=stages,
            legacy=False,
        )
        results["layers"][name] = {
            "shape": list(weight.shape),
            "train_vs_legacy_export": tensor_metrics(train_q, legacy_q),
            "train_vs_fixed_export": tensor_metrics(train_q, fixed_q),
            "legacy_vs_fixed_export": tensor_metrics(legacy_q, fixed_q),
        }

    return results


def layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    return F.layer_norm(x, (x.shape[-1],), weight.float(), bias.float(), eps)


def linear(x: torch.Tensor, layer: RuntimeLayer) -> torch.Tensor:
    return F.linear(x, layer.weight.float(), layer.bias.float())


def build_runtime_state(state: dict[str, torch.Tensor], variant: str) -> dict[str, Any]:
    config = VARIANTS[variant]
    hidden_size = config["hidden_size"]
    intermediate_size = config["intermediate_size"]
    hidden_complex = hidden_size // 2
    intermediate_complex = intermediate_size // 2
    hidden_complex_padded = round_up(hidden_complex, BLOCK_SIZE)
    qkv_complex_padded = round_up(hidden_complex * 3, BLOCK_SIZE)
    intermediate_complex_padded = round_up(intermediate_complex, BLOCK_SIZE)

    runtime: dict[str, Any] = {
        "patch_embed": RuntimeLayer(
            weight=state["patch_embed.proj.weight"].reshape(hidden_size, -1).float(),
            bias=state.get("patch_embed.proj.bias", torch.zeros(hidden_size)).float(),
        ),
        "cls_token": state["cls_token"].float(),
        "pos_embed": state["pos_embed"].float(),
        "norm": RuntimeLayer(weight=state["norm.weight"].float(), bias=state["norm.bias"].float()),
        "head": RuntimeLayer(
            weight=quantized_forward_weight(state["head.weight"], stages=2).float(),
            bias=state.get("head.bias", torch.zeros(state["head.weight"].shape[0])).float(),
        ),
        "layers": [],
    }

    for index in range(config["depth"]):
        prefix = f"blocks.{index}"
        qkv_weight = state[f"{prefix}.attn.qkv.weight"]
        qkv_bias = state.get(f"{prefix}.attn.qkv.bias")
        runtime["layers"].append(
            {
                "norm1": RuntimeLayer(state[f"{prefix}.norm1.weight"].float(), state[f"{prefix}.norm1.bias"].float()),
                "norm2": RuntimeLayer(state[f"{prefix}.norm2.weight"].float(), state[f"{prefix}.norm2.bias"].float()),
                "qkv": RuntimeLayer(
                    weight=reconstruct_export_weight(
                        qkv_weight,
                        hidden_complex_padded,
                        qkv_complex_padded,
                        2,
                        legacy=False,
                    ),
                    bias=(qkv_bias if qkv_bias is not None else torch.zeros(hidden_size * 3)).float(),
                ),
                "proj": RuntimeLayer(
                    weight=reconstruct_export_weight(
                        state[f"{prefix}.attn.proj.weight"],
                        hidden_complex_padded,
                        hidden_complex_padded,
                        2,
                        legacy=False,
                    ),
                    bias=state.get(f"{prefix}.attn.proj.bias", torch.zeros(hidden_size)).float(),
                ),
                "fc1": RuntimeLayer(
                    weight=reconstruct_export_weight(
                        state[f"{prefix}.mlp.fc1.weight"],
                        hidden_complex_padded,
                        intermediate_complex_padded,
                        2,
                        legacy=False,
                    ),
                    bias=state.get(f"{prefix}.mlp.fc1.bias", torch.zeros(intermediate_size)).float(),
                ),
                "fc2": RuntimeLayer(
                    weight=reconstruct_export_weight(
                        state[f"{prefix}.mlp.fc2.weight"],
                        intermediate_complex_padded,
                        hidden_complex_padded,
                        2,
                        legacy=False,
                    ),
                    bias=state.get(f"{prefix}.mlp.fc2.bias", torch.zeros(hidden_size)).float(),
                ),
            }
        )

    return runtime


def capture_training_block0_trace(block: torch.nn.Module, x: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    captures: dict[str, torch.Tensor] = {"block0_input": x}
    norm1 = block.norm1(x)
    captures["block0_norm1"] = norm1

    qkv_concat = block.attn.qkv(norm1)
    captures["block0_qkv_concat"] = qkv_concat

    batch, tokens, _ = qkv_concat.shape
    qkv = qkv_concat.reshape(batch, tokens, 3, block.attn.num_heads, block.attn.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = block.attn.q_norm(q), block.attn.k_norm(k)
    captures["block0_q"] = q
    captures["block0_k"] = k
    captures["block0_v"] = v

    if block.attn.fused_attn:
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
    else:
        attn = (q * block.attn.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn_out = attn @ v

    attn_out_pre_proj = attn_out.transpose(1, 2).reshape(batch, tokens, block.attn.attn_dim)
    captures["block0_attn_out_pre_proj"] = attn_out_pre_proj
    attn_proj = block.attn.proj(block.attn.norm(attn_out_pre_proj))
    captures["block0_attn_proj"] = attn_proj

    residual1 = x + block.drop_path1(block.ls1(attn_proj))
    captures["block0_residual1"] = residual1

    norm2 = block.norm2(residual1)
    mlp_hidden = block.mlp.fc1(norm2)
    mlp_hidden = block.mlp.act(mlp_hidden)
    mlp_hidden = block.mlp.drop1(mlp_hidden)
    mlp_hidden = block.mlp.norm(mlp_hidden)
    mlp_hidden = block.mlp.fc2(mlp_hidden)
    mlp_hidden = block.mlp.drop2(mlp_hidden)
    block_output = residual1 + block.drop_path2(block.ls2(mlp_hidden))
    captures["block0"] = block_output
    return captures, block_output


def forward_training_path(model: torch.nn.Module, image_tensor: torch.Tensor) -> dict[str, torch.Tensor]:
    captures: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        x = model.patch_embed(image_tensor)
        captures["patch_embed"] = x

        cls_token = model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = model.pos_drop(x + model.pos_embed[:, : x.shape[1]])

        for index, block in enumerate(model.blocks):
            if index == 0:
                block0_trace, x = capture_training_block0_trace(block, x)
                captures.update(block0_trace)
            elif index == 1:
                x = block(x)
                captures["block1"] = x
            else:
                x = block(x)

        x = model.norm(x)
        captures["final_norm"] = x
        captures["logits"] = model.head(x[:, 0])

    return captures


def capture_runtime_block_trace(
    layer: dict[str, RuntimeLayer],
    x: torch.Tensor,
    num_heads: int,
    layer_norm_eps: float,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    captures: dict[str, torch.Tensor] = {"block0_input": x}
    hidden_size = x.shape[-1]
    head_dim = hidden_size // num_heads

    norm1 = layer_norm(x, layer["norm1"].weight, layer["norm1"].bias, layer_norm_eps)
    captures["block0_norm1"] = norm1

    qkv_concat = linear(norm1, layer["qkv"])
    captures["block0_qkv_concat"] = qkv_concat

    q, k, v = qkv_concat.chunk(3, dim=-1)
    q = q.reshape(x.shape[0], x.shape[1], num_heads, head_dim).transpose(1, 2)
    k = k.reshape(x.shape[0], x.shape[1], num_heads, head_dim).transpose(1, 2)
    v = v.reshape(x.shape[0], x.shape[1], num_heads, head_dim).transpose(1, 2)
    captures["block0_q"] = q
    captures["block0_k"] = k
    captures["block0_v"] = v

    attn = torch.matmul(q, k.transpose(-2, -1)) * (head_dim ** -0.5)
    attn = torch.softmax(attn, dim=-1)
    attn_out_pre_proj = torch.matmul(attn, v).transpose(1, 2).reshape(x.shape[0], x.shape[1], hidden_size)
    captures["block0_attn_out_pre_proj"] = attn_out_pre_proj

    attn_proj = linear(attn_out_pre_proj, layer["proj"])
    captures["block0_attn_proj"] = attn_proj
    residual1 = x + attn_proj
    captures["block0_residual1"] = residual1

    norm2 = layer_norm(residual1, layer["norm2"].weight, layer["norm2"].bias, layer_norm_eps)
    mlp_hidden = linear(norm2, layer["fc1"])
    mlp_hidden = F.gelu(mlp_hidden, approximate="none")
    block_output = residual1 + linear(mlp_hidden, layer["fc2"])
    captures["block0"] = block_output
    return captures, block_output


def forward_runtime_path(
    runtime: dict[str, Any],
    patches: torch.Tensor,
    num_heads: int,
    layer_norm_eps: float,
) -> dict[str, torch.Tensor]:
    captures: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        x = linear(patches, runtime["patch_embed"])
        captures["patch_embed"] = x

        cls_token = runtime["cls_token"].expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + runtime["pos_embed"][:, : x.shape[1]]

        hidden_size = x.shape[-1]
        head_dim = hidden_size // num_heads

        for index, layer in enumerate(runtime["layers"]):
            if index == 0:
                block0_trace, x = capture_runtime_block_trace(layer, x, num_heads, layer_norm_eps)
                captures.update(block0_trace)
            elif index == 1:
                norm1 = layer_norm(x, layer["norm1"].weight, layer["norm1"].bias, layer_norm_eps)
                qkv = linear(norm1, layer["qkv"])
                q, k, v = qkv.chunk(3, dim=-1)
                q = q.reshape(x.shape[0], x.shape[1], num_heads, head_dim).transpose(1, 2)
                k = k.reshape(x.shape[0], x.shape[1], num_heads, head_dim).transpose(1, 2)
                v = v.reshape(x.shape[0], x.shape[1], num_heads, head_dim).transpose(1, 2)
                attn = torch.matmul(q, k.transpose(-2, -1)) * (head_dim ** -0.5)
                attn = torch.softmax(attn, dim=-1)
                attn_out = torch.matmul(attn, v).transpose(1, 2).reshape(x.shape[0], x.shape[1], hidden_size)
                x = x + linear(attn_out, layer["proj"])
                norm2 = layer_norm(x, layer["norm2"].weight, layer["norm2"].bias, layer_norm_eps)
                mlp_hidden = linear(norm2, layer["fc1"])
                mlp_hidden = F.gelu(mlp_hidden, approximate="none")
                x = x + linear(mlp_hidden, layer["fc2"])
                captures["block1"] = x
            else:
                norm1 = layer_norm(x, layer["norm1"].weight, layer["norm1"].bias, layer_norm_eps)
                qkv = linear(norm1, layer["qkv"])
                q, k, v = qkv.chunk(3, dim=-1)
                q = q.reshape(x.shape[0], x.shape[1], num_heads, head_dim).transpose(1, 2)
                k = k.reshape(x.shape[0], x.shape[1], num_heads, head_dim).transpose(1, 2)
                v = v.reshape(x.shape[0], x.shape[1], num_heads, head_dim).transpose(1, 2)
                attn = torch.matmul(q, k.transpose(-2, -1)) * (head_dim ** -0.5)
                attn = torch.softmax(attn, dim=-1)
                attn_out = torch.matmul(attn, v).transpose(1, 2).reshape(x.shape[0], x.shape[1], hidden_size)
                x = x + linear(attn_out, layer["proj"])
                norm2 = layer_norm(x, layer["norm2"].weight, layer["norm2"].bias, layer_norm_eps)
                mlp_hidden = linear(norm2, layer["fc1"])
                mlp_hidden = F.gelu(mlp_hidden, approximate="none")
                x = x + linear(mlp_hidden, layer["fc2"])

        x = layer_norm(x, runtime["norm"].weight, runtime["norm"].bias, layer_norm_eps)
        captures["final_norm"] = x
        captures["logits"] = linear(x[:, :1, :], runtime["head"]).reshape(x.shape[0], -1)

    return captures


def compare_stages(
    lhs: dict[str, torch.Tensor],
    rhs: dict[str, torch.Tensor],
    stages: tuple[str, ...],
) -> dict[str, Any]:
    return {stage: tensor_metrics(lhs[stage], rhs[stage]) for stage in stages}


def block0_qkv_semantics(
    state: dict[str, torch.Tensor],
    training_model: torch.nn.Module,
    runtime_state: dict[str, Any],
    image_tensor: torch.Tensor,
) -> dict[str, Any]:
    with torch.no_grad():
        x = training_model.patch_embed(image_tensor)
        cls_token = training_model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = training_model.pos_drop(x + training_model.pos_embed[:, : x.shape[1]])
        norm1 = training_model.blocks[0].norm1(x)
        train_qkv = training_model.blocks[0].attn.qkv(norm1)

        effective_qkv_layer = RuntimeLayer(
            weight=quantized_forward_weight(state["blocks.0.attn.qkv.weight"], stages=2).float(),
            bias=state.get("blocks.0.attn.qkv.bias", torch.zeros(train_qkv.shape[-1])).float(),
        )
        effective_qkv = linear(norm1, effective_qkv_layer)
        export_qkv = linear(norm1, runtime_state["layers"][0]["qkv"])

    return {
        "train_vs_effective_forward": tensor_metrics(train_qkv, effective_qkv),
        "train_vs_export_runtime": tensor_metrics(train_qkv, export_qkv),
        "effective_forward_vs_export_runtime": tensor_metrics(effective_qkv, export_qkv),
    }


def run_layer_audit(checkpoint: Path, variant: str) -> dict[str, Any]:
    state = load_checkpoint(checkpoint)
    training_model = build_training_model(variant, checkpoint)
    runtime_state = build_runtime_state(state, variant)
    data_config = resolve_eval_config(variant)
    transform = create_transform(
        input_size=data_config["input_size"],
        is_training=False,
        interpolation=data_config["interpolation"],
        mean=data_config["mean"],
        std=data_config["std"],
        crop_pct=data_config["crop_pct"],
    )
    patch_size = VARIANTS[variant]["patch_size"]
    layer_norm_eps = float(training_model.blocks[0].norm1.eps)

    high_level_stages = ("patch_embed", "block0", "block1", "final_norm", "logits")
    block0_trace_stages = (
        "block0_input",
        "block0_norm1",
        "block0_qkv_concat",
        "block0_q",
        "block0_k",
        "block0_v",
        "block0_attn_out_pre_proj",
        "block0_attn_proj",
        "block0_residual1",
        "block0",
    )

    block = training_model.blocks[0]
    results: dict[str, Any] = {
        "model_details": {
            "training_layer_norm_eps": layer_norm_eps,
            "python_runtime_layer_norm_eps": layer_norm_eps,
            "training_gelu": repr(block.mlp.act),
            "swift_gelu": "tanh-based approximation in RPFQViTModel.swift",
            "identity_modules": {
                "attn.norm": block.attn.norm.__class__.__name__,
                "attn.q_norm": block.attn.q_norm.__class__.__name__,
                "attn.k_norm": block.attn.k_norm.__class__.__name__,
                "block.ls1": block.ls1.__class__.__name__,
                "block.ls2": block.ls2.__class__.__name__,
                "block.drop_path1": block.drop_path1.__class__.__name__,
                "block.drop_path2": block.drop_path2.__class__.__name__,
                "mlp.norm": block.mlp.norm.__class__.__name__,
            },
        },
        "images": {},
    }
    for image_name, image in synthesize_images().items():
        image_tensor = transform(image).unsqueeze(0).float()
        timm_patches = patchify_tensor(image_tensor[0], patch_size)

        train_outputs = forward_training_path(training_model, image_tensor)
        runtime_outputs = forward_runtime_path(
            runtime_state,
            timm_patches,
            num_heads=VARIANTS[variant]["num_heads"],
            layer_norm_eps=layer_norm_eps,
        )

        results["images"][image_name] = {
            "python_runtime": compare_stages(train_outputs, runtime_outputs, high_level_stages),
            "block0_trace": compare_stages(train_outputs, runtime_outputs, block0_trace_stages),
            "block0_qkv_semantics": block0_qkv_semantics(state, training_model, runtime_state, image_tensor),
        }

    return results


def tensor_from_dump(payload: dict[str, Any]) -> torch.Tensor:
    shape = payload["shape"]
    values = payload["values"]
    return torch.tensor(values, dtype=torch.float32).reshape(shape)


def load_swift_debug_tensors(run_dir: Path) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    patches_payload = json.loads((run_dir / "patches.json").read_text(encoding="utf-8"))
    tensor_payload = json.loads((run_dir / "tensors.json").read_text(encoding="utf-8"))
    patches = tensor_from_dump(patches_payload)
    tensors = {stage: tensor_from_dump(payload) for stage, payload in tensor_payload.items()}
    return patches, tensors


def run_swift_runtime_audit(checkpoint: Path, variant: str, swift_debug_run: Path) -> dict[str, Any]:
    state = load_checkpoint(checkpoint)
    training_model = build_training_model(variant, checkpoint)
    runtime_state = build_runtime_state(state, variant)
    patches, swift_tensors = load_swift_debug_tensors(swift_debug_run)
    layer_norm_eps = float(training_model.blocks[0].norm1.eps)
    python_tensors = forward_runtime_path(
        runtime_state,
        patches.float(),
        num_heads=VARIANTS[variant]["num_heads"],
        layer_norm_eps=layer_norm_eps,
    )

    available_stages = tuple(stage for stage in swift_tensors.keys() if stage in python_tensors)
    return {
        "run_directory": str(swift_debug_run),
        "available_stages": list(available_stages),
        "python_vs_swift": compare_stages(python_tensors, swift_tensors, available_stages),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit RPFQViT iOS deployment parity.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to a training checkpoint.")
    parser.add_argument(
        "--variant",
        type=str,
        default="rpfq_vit_base_patch16_224_q",
        choices=sorted(VARIANTS.keys()),
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["preprocess", "weights", "layers", "all"],
    )
    parser.add_argument("--json-out", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument(
        "--swift-debug-run",
        type=Path,
        default=None,
        help="Optional Swift debug run directory containing patches.json and tensors.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report: dict[str, Any] = {"variant": args.variant, "mode": args.mode, "results": {}}

    if args.mode in {"preprocess", "all"}:
        report["results"]["preprocess"] = run_preprocess_audit(args.variant)

    if args.mode in {"weights", "layers", "all"}:
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required for weights/layers/all modes.")

    if args.mode in {"weights", "all"}:
        report["results"]["weights"] = run_weight_audit(args.checkpoint, args.variant)

    if args.mode in {"layers", "all"}:
        report["results"]["layers"] = run_layer_audit(args.checkpoint, args.variant)
        if args.swift_debug_run is not None:
            report["results"]["swift_runtime"] = run_swift_runtime_audit(
                args.checkpoint,
                args.variant,
                args.swift_debug_run,
            )

    output = json.dumps(report, indent=2)
    if args.json_out is not None:
        args.json_out.write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
