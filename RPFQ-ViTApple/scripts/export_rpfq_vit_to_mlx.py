#!/usr/bin/env python3
"""
Export an RPFQViT checkpoint into the MLX Swift bundle layout used by RPFQViTMobile.

Target runtime assumptions:
- small and base variants are first-class deployment targets
- quantization: complex_phase_v2 (residual axis-phase, 2 stages)
- task: image classification
- runtime format:
  config.json
  model.safetensors
  preprocess.json
  labels.txt
  manifest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


BLOCK_SIZE = 256
DEFAULT_INTERPOLATION = "bicubic"
DEFAULT_CROP_PCT = 0.875
DEFAULT_CROP_MODE = "center"
DEFAULT_RESIZE_MODE = "shortest_side"

VARIANTS = {
    "rpfq_vit_tiny_patch16_224_q": {
        "image_size": 224,
        "patch_size": 16,
        "hidden_size": 192,
        "depth": 12,
        "num_heads": 3,
        "intermediate_size": 768,
    },
    "rpfq_vit_small_patch16_224_q": {
        "image_size": 224,
        "patch_size": 16,
        "hidden_size": 384,
        "depth": 12,
        "num_heads": 6,
        "intermediate_size": 1536,
    },
    "rpfq_vit_base_patch16_224_q": {
        "image_size": 224,
        "patch_size": 16,
        "hidden_size": 768,
        "depth": 12,
        "num_heads": 12,
        "intermediate_size": 3072,
    },
    "rpfq_vit_large_patch16_224_q": {
        "image_size": 224,
        "patch_size": 16,
        "hidden_size": 1024,
        "depth": 24,
        "num_heads": 16,
        "intermediate_size": 4096,
    },
}

DEFAULT_OUTPUTS = {
    "rpfq_vit_small_patch16_224_q": Path("RPFQViTMobile/rpfq_vit-small-mlx.bundle"),
    "rpfq_vit_base_patch16_224_q": Path("RPFQViTMobile/rpfq_vit-base-mlx.bundle"),
}


def round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def default_output_dir_for_variant(variant_name: str) -> Path:
    if variant_name in DEFAULT_OUTPUTS:
        return DEFAULT_OUTPUTS[variant_name]

    slug = variant_name.replace("_", "-")
    return Path(f"RPFQViTMobile/{slug}-mlx.bundle")


def load_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        state = {}
        with safe_open(str(path), framework="pt") as handle:
            for key in handle.keys():
                state[key] = handle.get_tensor(key)
        return normalize_state_dict(state)

    raw = torch.load(path, map_location="cpu")
    if isinstance(raw, dict):
        for nested_key in ("state_dict", "model_state_dict", "model", "module"):
            nested = raw.get(nested_key)
            if isinstance(nested, dict):
                raw = nested
                break

    if not isinstance(raw, dict):
        raise ValueError(f"Unsupported checkpoint structure in {path}")

    tensors = {key: value for key, value in raw.items() if torch.is_tensor(value)}
    return normalize_state_dict(tensors)


def normalize_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized = {}
    for key, value in state.items():
        new_key = key
        for prefix in ("module.", "model.", "backbone."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        normalized[new_key] = value
    return normalized


def _safe_mean_abs(x: torch.Tensor, mask: torch.Tensor, min_value: float = 1e-6) -> torch.Tensor:
    mask_f = mask.to(dtype=x.dtype)
    mean_abs = (x.abs() * mask_f).sum() / mask_f.sum().clamp_min(1.0)
    return mean_abs.clamp_min(min_value)


def axis_masks(w_real: torch.Tensor, w_imag: torch.Tensor) -> tuple[torch.Tensor, ...]:
    abs_real = w_real.abs()
    abs_imag = w_imag.abs()

    real_dominant = abs_real > abs_imag
    imag_dominant = abs_imag > abs_real
    tie = ~(real_dominant | imag_dominant)

    zero_tie = tie & (w_real == 0) & (w_imag == 0)
    tie_q1 = tie & (w_real > 0) & (w_imag > 0)
    tie_q2 = tie & (w_real < 0) & (w_imag > 0)
    tie_q3 = tie & (w_real < 0) & (w_imag < 0)
    tie_q4 = tie & (w_real > 0) & (w_imag < 0)

    real_pos = (real_dominant & (w_real >= 0)) | tie_q4 | zero_tie
    real_neg = (real_dominant & (w_real < 0)) | tie_q2
    imag_pos = (imag_dominant & (w_imag >= 0)) | tie_q1
    imag_neg = (imag_dominant & (w_imag < 0)) | tie_q3
    return real_pos, real_neg, imag_pos, imag_neg


def pack_codes(codes: torch.Tensor) -> torch.Tensor:
    out_dim, in_dim = codes.shape
    if in_dim % 4 != 0:
        raise ValueError(f"Expected packed width to be divisible by 4, got {in_dim}")

    packed = codes.reshape(out_dim, in_dim // 4, 4)
    return (
        packed[:, :, 0]
        | (packed[:, :, 1] << 2)
        | (packed[:, :, 2] << 4)
        | (packed[:, :, 3] << 6)
    ).to(torch.uint8)


def legacy_phase_quantize_axis(
    w_real: torch.Tensor,
    w_imag: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out_dim, in_dim = w_real.shape
    if in_dim % BLOCK_SIZE != 0:
        raise ValueError(f"Expected aligned input dim, got {in_dim}")

    real_pos, real_neg, imag_pos, imag_neg = axis_masks(w_real, w_imag)
    real_mask = real_pos | real_neg
    imag_mask = imag_pos | imag_neg

    scale_real = _safe_mean_abs(w_real, real_mask)
    scale_imag = _safe_mean_abs(w_imag, imag_mask)

    codes = torch.zeros_like(w_real, dtype=torch.uint8)
    codes[real_pos] = 1
    codes[imag_neg] = 2
    codes[imag_pos] = 3

    packed = pack_codes(codes)
    blocks_per_row = in_dim // BLOCK_SIZE
    gamma_real = torch.full((out_dim, blocks_per_row), 1.0 / float(scale_real), dtype=torch.float16)
    gamma_imag = torch.full((out_dim, blocks_per_row), 1.0 / float(scale_imag), dtype=torch.float16)
    return packed, gamma_real, gamma_imag


def phase_quantize_axis(
    w_real: torch.Tensor,
    w_imag: torch.Tensor,
    padded_out_dim: int | None = None,
    padded_in_dim: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out_dim, in_dim = w_real.shape
    padded_out_dim = padded_out_dim or out_dim
    padded_in_dim = padded_in_dim or in_dim
    if padded_in_dim % BLOCK_SIZE != 0:
        raise ValueError(f"Expected padded input dim aligned to block size, got {padded_in_dim}")
    if padded_in_dim % 4 != 0:
        raise ValueError(f"Expected padded input dim divisible by 4, got {padded_in_dim}")
    if padded_out_dim < out_dim or padded_in_dim < in_dim:
        raise ValueError(
            f"Padded dims must cover valid dims, got valid={(out_dim, in_dim)} padded={(padded_out_dim, padded_in_dim)}"
        )

    real_pos, real_neg, imag_pos, imag_neg = axis_masks(w_real, w_imag)
    real_mask = real_pos | real_neg
    imag_mask = imag_pos | imag_neg

    scale_real = _safe_mean_abs(w_real, real_mask)
    scale_imag = _safe_mean_abs(w_imag, imag_mask)

    codes = torch.zeros((padded_out_dim, padded_in_dim), dtype=torch.uint8)
    valid_codes = codes[:out_dim, :in_dim]
    valid_codes[real_pos] = 1
    valid_codes[imag_neg] = 2
    valid_codes[imag_pos] = 3

    packed = pack_codes(codes)
    blocks_per_row = padded_in_dim // BLOCK_SIZE
    gamma_real = torch.full((padded_out_dim, blocks_per_row), 1.0 / float(scale_real), dtype=torch.float16)
    gamma_imag = torch.full((padded_out_dim, blocks_per_row), 1.0 / float(scale_imag), dtype=torch.float16)
    return packed, gamma_real, gamma_imag


def dequantize_axis(
    codes_packed: torch.Tensor,
    gamma_real: torch.Tensor,
    gamma_imag: torch.Tensor,
    out_dim: int,
    in_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    packed = codes_packed.reshape(out_dim, in_dim // 4)

    c0 = packed & 0x03
    c1 = (packed >> 2) & 0x03
    c2 = (packed >> 4) & 0x03
    c3 = (packed >> 6) & 0x03
    all_codes = torch.stack([c0, c1, c2, c3], dim=-1).reshape(out_dim, in_dim)

    w_real = torch.zeros((out_dim, in_dim), dtype=torch.float32)
    w_imag = torch.zeros((out_dim, in_dim), dtype=torch.float32)
    w_real[all_codes == 0] = -1.0
    w_real[all_codes == 1] = +1.0
    w_imag[all_codes == 2] = -1.0
    w_imag[all_codes == 3] = +1.0

    gamma_real_full = gamma_real.float().repeat_interleave(BLOCK_SIZE, dim=1)[:, :in_dim]
    gamma_imag_full = gamma_imag.float().repeat_interleave(BLOCK_SIZE, dim=1)[:, :in_dim]

    w_real = w_real / gamma_real_full
    w_imag = w_imag / gamma_imag_full
    return w_real, w_imag


def residual_phase_quantize(
    w_real: torch.Tensor,
    w_imag: torch.Tensor,
    stages: int,
    padded_out_dim: int | None = None,
    padded_in_dim: int | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    residual_real = w_real.clone()
    residual_imag = w_imag.clone()
    exported = []
    valid_out_dim, valid_in_dim = w_real.shape
    padded_out_dim = padded_out_dim or valid_out_dim
    padded_in_dim = padded_in_dim or valid_in_dim

    for _ in range(stages):
        codes, gamma_real, gamma_imag = phase_quantize_axis(
            residual_real,
            residual_imag,
            padded_out_dim=padded_out_dim,
            padded_in_dim=padded_in_dim,
        )
        deq_real, deq_imag = dequantize_axis(
            codes,
            gamma_real,
            gamma_imag,
            out_dim=padded_out_dim,
            in_dim=padded_in_dim,
        )
        residual_real = residual_real - deq_real[:valid_out_dim, :valid_in_dim]
        residual_imag = residual_imag - deq_imag[:valid_out_dim, :valid_in_dim]
        exported.append((codes, gamma_real, gamma_imag))

    return exported


def legacy_residual_phase_quantize(
    w_real: torch.Tensor,
    w_imag: torch.Tensor,
    stages: int,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    residual_real = w_real.clone()
    residual_imag = w_imag.clone()
    exported = []

    for _ in range(stages):
        codes, gamma_real, gamma_imag = legacy_phase_quantize_axis(residual_real, residual_imag)
        deq_real, deq_imag = dequantize_axis(
            codes,
            gamma_real,
            gamma_imag,
            out_dim=w_real.shape[0],
            in_dim=w_real.shape[1],
        )
        residual_real = residual_real - deq_real
        residual_imag = residual_imag - deq_imag
        exported.append((codes, gamma_real, gamma_imag))

    return exported


def widely_linear_decompose(weight: torch.Tensor) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    out_real, in_real = weight.shape
    if out_real % 2 != 0 or in_real % 2 != 0:
        raise ValueError(f"Complex decomposition requires even dims, got {tuple(weight.shape)}")

    half_out = out_real // 2
    half_in = in_real // 2
    a11 = weight[:half_out, :half_in]
    a12 = weight[:half_out, half_in:]
    a21 = weight[half_out:, :half_in]
    a22 = weight[half_out:, half_in:]

    u_real = 0.5 * (a11 + a22)
    u_imag = 0.5 * (a21 - a12)
    w_real = 0.5 * (a11 - a22)
    w_imag = 0.5 * (a12 + a21)
    return (u_real, u_imag), (w_real, w_imag)


def quantized_forward_weight(weight: torch.Tensor, stages: int) -> torch.Tensor:
    (u_real, u_imag), (w_real, w_imag) = widely_linear_decompose(weight.float())
    out_complex, in_complex = u_real.shape
    padded_out_complex = round_up(out_complex, BLOCK_SIZE)
    padded_in_complex = round_up(in_complex, BLOCK_SIZE)

    def accumulate(real: torch.Tensor, imag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        acc_real = torch.zeros((padded_out_complex, padded_in_complex), dtype=torch.float32)
        acc_imag = torch.zeros((padded_out_complex, padded_in_complex), dtype=torch.float32)
        for codes, gamma_real, gamma_imag in residual_phase_quantize(
            real,
            imag,
            stages=stages,
            padded_out_dim=padded_out_complex,
            padded_in_dim=padded_in_complex,
        ):
            deq_real, deq_imag = dequantize_axis(
                codes,
                gamma_real,
                gamma_imag,
                out_dim=padded_out_complex,
                in_dim=padded_in_complex,
            )
            acc_real += deq_real
            acc_imag += deq_imag
        return acc_real[:out_complex, :in_complex], acc_imag[:out_complex, :in_complex]

    u_real_q, u_imag_q = accumulate(u_real, u_imag)
    w_real_q, w_imag_q = accumulate(w_real, w_imag)
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


def pad_matrix(matrix: torch.Tensor, out_dim: int, in_dim: int) -> torch.Tensor:
    padded = torch.zeros((out_dim, in_dim), dtype=matrix.dtype)
    padded[: matrix.shape[0], : matrix.shape[1]] = matrix
    return padded


def save_quantized_linear(
    target: dict[str, torch.Tensor],
    prefix: str,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    in_complex_padded: int,
    out_complex_padded: int,
    stages: int,
) -> None:
    (u_real, u_imag), (w_real, w_imag) = widely_linear_decompose(weight.float())
    u_stages = residual_phase_quantize(
        u_real,
        u_imag,
        stages=stages,
        padded_out_dim=out_complex_padded,
        padded_in_dim=in_complex_padded,
    )
    w_stages = residual_phase_quantize(
        w_real,
        w_imag,
        stages=stages,
        padded_out_dim=out_complex_padded,
        padded_in_dim=in_complex_padded,
    )

    for index, (codes, gamma_real, gamma_imag) in enumerate(u_stages):
        stage_prefix = f"{prefix}.u_s{index}"
        target[f"{stage_prefix}.codes"] = codes
        target[f"{stage_prefix}.gamma_real"] = gamma_real
        target[f"{stage_prefix}.gamma_imag"] = gamma_imag

    for index, (codes, gamma_real, gamma_imag) in enumerate(w_stages):
        stage_prefix = f"{prefix}.w_s{index}"
        target[f"{stage_prefix}.codes"] = codes
        target[f"{stage_prefix}.gamma_real"] = gamma_real
        target[f"{stage_prefix}.gamma_imag"] = gamma_imag

    if bias is None:
        bias = torch.zeros(weight.shape[0], dtype=torch.float16)
    target[f"{prefix}.bias"] = bias.to(torch.float16)


def split_qkv(weight: torch.Tensor, bias: torch.Tensor | None) -> tuple[list[torch.Tensor], list[torch.Tensor | None]]:
    split_weights = list(torch.chunk(weight, 3, dim=0))
    split_bias = list(torch.chunk(bias, 3, dim=0)) if bias is not None else [None, None, None]
    return split_weights, split_bias


def load_labels(path: Path | None, num_classes: int) -> list[str]:
    if path is None:
        return [f"Class {index}" for index in range(num_classes)]

    raw = path.read_text(encoding="utf-8")
    labels = [line.strip() for line in raw.splitlines() if line.strip()]
    if not labels:
        return [f"Class {index}" for index in range(num_classes)]
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Export an RPFQViT checkpoint for RPFQViTMobile.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to the RPFQViT checkpoint.")
    parser.add_argument(
        "--variant",
        default="rpfq_vit_base_patch16_224_q",
        choices=sorted(VARIANTS.keys()),
        help="Target RPFQViT variant.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output bundle directory. If omitted, a bundle name is chosen from the variant.",
    )
    parser.add_argument("--labels", type=Path, default=None, help="Optional labels.txt source.")
    parser.add_argument(
        "--quant-method",
        default="complex_phase_v2",
        choices=["complex_phase_v1", "complex_phase_v2"],
        help="Exported residual phase quantization method.",
    )
    parser.add_argument("--num-channels", type=int, default=3)
    parser.add_argument("--mean", nargs=3, type=float, default=[0.485, 0.456, 0.406])
    parser.add_argument("--std", nargs=3, type=float, default=[0.229, 0.224, 0.225])
    parser.add_argument("--interpolation", type=str, default=DEFAULT_INTERPOLATION)
    parser.add_argument("--crop-pct", type=float, default=DEFAULT_CROP_PCT)
    parser.add_argument("--crop-mode", type=str, default=DEFAULT_CROP_MODE)
    parser.add_argument("--resize-mode", type=str, default=DEFAULT_RESIZE_MODE)
    args = parser.parse_args()

    variant = VARIANTS[args.variant]
    stages = int(args.quant_method.rsplit("_v", 1)[-1])

    state = load_checkpoint(args.checkpoint)
    output_dir = args.output or default_output_dir_for_variant(args.variant)
    output_dir.mkdir(parents=True, exist_ok=True)

    hidden_size = variant["hidden_size"]
    image_size = variant["image_size"]
    patch_size = variant["patch_size"]
    intermediate_size = variant["intermediate_size"]
    num_heads = variant["num_heads"]
    num_layers = variant["depth"]
    num_classes = int(state["head.weight"].shape[0])

    hidden_complex = hidden_size // 2
    hidden_complex_padded = round_up(hidden_complex, BLOCK_SIZE)
    qkv_complex = (hidden_size * 3) // 2
    qkv_complex_padded = round_up(qkv_complex, BLOCK_SIZE)
    intermediate_complex = intermediate_size // 2
    intermediate_complex_padded = round_up(intermediate_complex, BLOCK_SIZE)

    exported: dict[str, torch.Tensor] = {}

    patch_weight = state["patch_embed.proj.weight"].reshape(hidden_size, -1)
    patch_bias = state.get("patch_embed.proj.bias", torch.zeros(hidden_size))
    exported["model.patch_embed.weight"] = patch_weight.to(torch.float16)
    exported["model.patch_embed.bias"] = patch_bias.to(torch.float16)
    exported["model.cls_token"] = state["cls_token"].to(torch.float16)
    exported["model.pos_embed"] = state["pos_embed"].to(torch.float16)

    for layer in range(num_layers):
        prefix = f"blocks.{layer}"
        target_prefix = f"model.layers.{layer}"

        exported[f"{target_prefix}.norm1.weight"] = state[f"{prefix}.norm1.weight"].to(torch.float16)
        exported[f"{target_prefix}.norm1.bias"] = state[f"{prefix}.norm1.bias"].to(torch.float16)
        exported[f"{target_prefix}.norm2.weight"] = state[f"{prefix}.norm2.weight"].to(torch.float16)
        exported[f"{target_prefix}.norm2.bias"] = state[f"{prefix}.norm2.bias"].to(torch.float16)

        qkv_weight = state[f"{prefix}.attn.qkv.weight"]
        qkv_bias = state.get(f"{prefix}.attn.qkv.bias")
        save_quantized_linear(
            exported,
            f"{target_prefix}.attn.qkv",
            qkv_weight,
            qkv_bias,
            in_complex_padded=hidden_complex_padded,
            out_complex_padded=qkv_complex_padded,
            stages=stages,
        )
        save_quantized_linear(
            exported,
            f"{target_prefix}.attn.proj",
            state[f"{prefix}.attn.proj.weight"],
            state.get(f"{prefix}.attn.proj.bias"),
            in_complex_padded=hidden_complex_padded,
            out_complex_padded=hidden_complex_padded,
            stages=stages,
        )
        save_quantized_linear(
            exported,
            f"{target_prefix}.mlp.fc1",
            state[f"{prefix}.mlp.fc1.weight"],
            state.get(f"{prefix}.mlp.fc1.bias"),
            in_complex_padded=hidden_complex_padded,
            out_complex_padded=intermediate_complex_padded,
            stages=stages,
        )
        save_quantized_linear(
            exported,
            f"{target_prefix}.mlp.fc2",
            state[f"{prefix}.mlp.fc2.weight"],
            state.get(f"{prefix}.mlp.fc2.bias"),
            in_complex_padded=intermediate_complex_padded,
            out_complex_padded=hidden_complex_padded,
            stages=stages,
        )

    exported["model.norm.weight"] = state["norm.weight"].to(torch.float16)
    exported["model.norm.bias"] = state["norm.bias"].to(torch.float16)
    exported["head.weight"] = quantized_forward_weight(state["head.weight"], stages).to(torch.float16)
    exported["head.bias"] = state.get("head.bias", torch.zeros(num_classes)).to(torch.float16)

    save_file(exported, str(output_dir / "model.safetensors"))

    config = {
        "model_type": "rpfq_vit",
        "variant": args.variant,
        "image_size": image_size,
        "patch_size": patch_size,
        "num_channels": args.num_channels,
        "hidden_size": hidden_size,
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_heads,
        "intermediate_size": intermediate_size,
        "num_classes": num_classes,
        "layer_norm_eps": 1e-5,
        "rpfq_vit": {
            "hidden_complex": hidden_complex,
            "hidden_complex_padded": hidden_complex_padded,
            "qkv_complex": qkv_complex,
            "qkv_complex_padded": qkv_complex_padded,
            "intermediate_complex": intermediate_complex,
            "intermediate_complex_padded": intermediate_complex_padded,
            "residual_stages": stages,
            "block_size": BLOCK_SIZE,
            "quant_method": args.quant_method,
        },
        "preprocess": {
            "image_size": image_size,
            "patch_size": patch_size,
            "num_channels": args.num_channels,
            "mean": args.mean,
            "std": args.std,
            "interpolation": args.interpolation,
            "crop_pct": args.crop_pct,
            "crop_mode": args.crop_mode,
            "resize_mode": args.resize_mode,
        },
        "quantization": {
            "type": "complex_phase_axis",
            "block_size": BLOCK_SIZE,
            "residual_stages": stages,
            "codebook": {
                "00": {"real": -1, "imag": 0, "label": "-1"},
                "01": {"real": 1, "imag": 0, "label": "+1"},
                "10": {"real": 0, "imag": -1, "label": "-i"},
                "11": {"real": 0, "imag": 1, "label": "+i"},
            },
        },
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    preprocess = {
        "image_size": image_size,
        "patch_size": patch_size,
        "num_channels": args.num_channels,
        "mean": args.mean,
        "std": args.std,
        "interpolation": args.interpolation,
        "crop_pct": args.crop_pct,
        "crop_mode": args.crop_mode,
        "resize_mode": args.resize_mode,
    }
    (output_dir / "preprocess.json").write_text(json.dumps(preprocess, indent=2), encoding="utf-8")

    labels = load_labels(args.labels, num_classes)
    (output_dir / "labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")

    manifest = {
        "model_type": "rpfq_vit",
        "variant": args.variant,
        "checkpoint": str(args.checkpoint),
        "export_format": "RPFQViTMobile-mlx",
        "quant_method": args.quant_method,
        "block_size": BLOCK_SIZE,
        "num_layers": num_layers,
        "num_classes": num_classes,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    size_bytes = sum(t.numel() * t.element_size() for t in exported.values())
    print(f"Exported {len(exported)} tensors to {output_dir}")
    print(f"Approximate model tensor size: {size_bytes / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
