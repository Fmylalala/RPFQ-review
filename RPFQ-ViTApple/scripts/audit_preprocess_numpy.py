#!/usr/bin/env python3
"""
Compare iOS-exported patches.json against a pure NumPy/PIL implementation
that mirrors the Swift preprocessing semantics used in RPFQViTMobile.

Use case (Option A):
- You ran the iOS app with "Export debug tensors" enabled.
- You downloaded the app container and got a run directory containing:
  - patches.json
  - summaries.json (optional but recommended)
- You have the same original input image file on your Mac.

This script will:
- Reproduce the Swift resize/crop pipeline:
  - resize_mode=shortest_side
  - crop_mode=center
  - crop_pct (default 0.875)
  - interpolation (nearest/bilinear/bicubic)
- Normalize RGB with mean/std
- Patchify to (1, grid*grid, patch_size*patch_size*3) in CHW patch order
- Compare the result with patches.json (max/mean abs diff, cosine, samples)
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple


def _require(pkg: str, exc: Exception) -> None:
    raise SystemExit(
        f"Missing dependency: {pkg}. Install into a venv, e.g.\n"
        f"  python3 -m venv .venv && source .venv/bin/activate\n"
        f"  python -m pip install numpy pillow\n\n"
        f"Original error: {exc}"
    )


try:
    import numpy as np
except Exception as e:  # pragma: no cover
    _require("numpy", e)

try:
    from PIL import Image
except Exception as e:  # pragma: no cover
    _require("pillow", e)


PIL_RESAMPLING = getattr(Image, "Resampling", Image)


@dataclass(frozen=True)
class PreprocessConfig:
    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    interpolation: str = "bicubic"
    crop_pct: float = 0.875
    crop_mode: str = "center"
    resize_mode: str = "shortest_side"


def pil_resample(interpolation: str) -> int:
    mapping = {
        "nearest": PIL_RESAMPLING.NEAREST,
        "bilinear": PIL_RESAMPLING.BILINEAR,
        "bicubic": PIL_RESAMPLING.BICUBIC,
    }
    return mapping.get(interpolation.lower(), PIL_RESAMPLING.BICUBIC)


def pil_to_normalized_chw(image: Image.Image, mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> "np.ndarray":
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0  # HWC
    chw = np.transpose(array, (2, 0, 1))  # CHW
    mean_t = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
    std_t = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
    return (chw - mean_t) / std_t


def patchify_chw(chw: "np.ndarray", patch_size: int) -> "np.ndarray":
    c, h, w = chw.shape
    if h % patch_size != 0 or w % patch_size != 0:
        raise ValueError(f"Expected H/W divisible by patch_size, got {(h, w)} vs {patch_size}")
    grid_h = h // patch_size
    grid_w = w // patch_size
    patches = (
        chw.reshape(c, grid_h, patch_size, grid_w, patch_size)
        .transpose(1, 3, 0, 2, 4)
        .reshape(grid_h * grid_w, c * patch_size * patch_size)
    )
    return patches.reshape(1, grid_h * grid_w, -1)


def swift_preprocess_fixed(image: Image.Image, cfg: PreprocessConfig) -> "np.ndarray":
    if cfg.resize_mode.lower() != "shortest_side" or cfg.crop_mode.lower() != "center":
        raise ValueError("This audit script currently supports only resize_mode=shortest_side + crop_mode=center.")

    src_w, src_h = image.size
    target_shortest = int(round(cfg.image_size / float(cfg.crop_pct)))
    scale = target_shortest / float(min(src_w, src_h))
    resized_w = int(round(src_w * scale))
    resized_h = int(round(src_h * scale))
    resized = image.resize((resized_w, resized_h), resample=pil_resample(cfg.interpolation))

    left = max((resized_w - cfg.image_size) // 2, 0)
    top = max((resized_h - cfg.image_size) // 2, 0)
    cropped = resized.crop((left, top, left + cfg.image_size, top + cfg.image_size))

    chw = pil_to_normalized_chw(cropped, cfg.mean, cfg.std)
    return patchify_chw(chw, cfg.patch_size)


def load_swift_patches(patches_json: Path) -> "np.ndarray":
    payload = json.loads(patches_json.read_text(encoding="utf-8"))
    shape = payload["shape"]
    values = payload["values"]
    arr = np.asarray(values, dtype=np.float32).reshape(shape)
    return arr


def cosine_similarity(a: "np.ndarray", b: "np.ndarray") -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def metrics(a: "np.ndarray", b: "np.ndarray") -> Dict[str, Any]:
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    diff = np.abs(a - b)
    return {
        "shape": list(a.shape),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "cosine_similarity": cosine_similarity(a, b),
        "first_10_a": [float(x) for x in a.reshape(-1)[:10]],
        "first_10_b": [float(x) for x in b.reshape(-1)[:10]],
    }


def patch_samples(a: "np.ndarray", b: "np.ndarray") -> list[dict[str, Any]]:
    token_count = a.shape[1]
    sample_indices = sorted({0, token_count // 2, token_count - 1})
    out: list[dict[str, Any]] = []
    for idx in sample_indices:
        out.append(
            {
                "patch_index": int(idx),
                "a_values": [float(x) for x in a[0, idx, :12]],
                "b_values": [float(x) for x in b[0, idx, :12]],
            }
        )
    return out


def load_config_from_summaries(summaries_json: Path) -> PreprocessConfig:
    payload = json.loads(summaries_json.read_text(encoding="utf-8"))
    pre = payload.get("preprocess") or {}
    mean = tuple(float(x) for x in pre.get("mean", [0.485, 0.456, 0.406]))
    std = tuple(float(x) for x in pre.get("std", [0.229, 0.224, 0.225]))
    return PreprocessConfig(
        image_size=int(pre.get("image_size", 224)),
        patch_size=int(pre.get("patch_size", 16)),
        num_channels=int(pre.get("num_channels", 3)),
        mean=mean,  # type: ignore[arg-type]
        std=std,  # type: ignore[arg-type]
        interpolation=str(pre.get("interpolation", "bicubic")),
        crop_pct=float(pre.get("crop_pct", 0.875)),
        crop_mode=str(pre.get("crop_mode", "center")),
        resize_mode=str(pre.get("resize_mode", "shortest_side")),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit iOS patches.json preprocessing parity (NumPy/PIL).")
    p.add_argument("--swift-run", type=Path, required=True, help="Swift debug run directory containing patches.json.")
    p.add_argument("--image", type=Path, required=True, help="The exact original image used in the iOS app.")
    p.add_argument("--summaries", type=Path, default=None, help="Optional summaries.json to auto-load preprocess config.")
    p.add_argument("--json-out", type=Path, default=None, help="Optional output JSON path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir: Path = args.swift_run
    patches_path = run_dir / "patches.json"
    if not patches_path.exists():
        raise SystemExit(f"Missing patches.json at {patches_path}")

    summaries_path = args.summaries or (run_dir / "summaries.json")
    cfg = load_config_from_summaries(summaries_path) if summaries_path.exists() else PreprocessConfig()

    swift_patches = load_swift_patches(patches_path)
    image = Image.open(args.image)
    numpy_patches = swift_preprocess_fixed(image, cfg)

    report = {
        "swift_run": str(run_dir),
        "image": str(args.image),
        "preprocess": {
            "image_size": cfg.image_size,
            "patch_size": cfg.patch_size,
            "mean": list(cfg.mean),
            "std": list(cfg.std),
            "interpolation": cfg.interpolation,
            "crop_pct": cfg.crop_pct,
            "crop_mode": cfg.crop_mode,
            "resize_mode": cfg.resize_mode,
        },
        "metrics": metrics(swift_patches, numpy_patches),
        "patch_samples": patch_samples(swift_patches, numpy_patches),
    }

    output = json.dumps(report, indent=2)
    if args.json_out is not None:
        args.json_out.write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
