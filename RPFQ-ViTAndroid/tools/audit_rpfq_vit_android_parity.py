#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def compare_tensor(name: str, ref_dir: Path, android_dir: Path) -> dict[str, object]:
    ref = np.load(ref_dir / f"{name}.npy")
    meta = json.loads((android_dir / f"{name}.json").read_text(encoding="utf-8"))
    android = np.fromfile(android_dir / f"{name}.bin", dtype=np.float32).reshape(meta["shape"])
    diff = np.abs(ref.astype(np.float64) - android.astype(np.float64))
    denom = np.maximum(np.abs(ref.astype(np.float64)), 1e-12)
    return {
        "name": name,
        "shape_match": list(ref.shape) == list(android.shape),
        "dtype_match": meta.get("dtype") == "float32",
        "max_abs_diff": float(diff.max()) if diff.size else 0.0,
        "mean_abs_diff": float(diff.mean()) if diff.size else 0.0,
        "max_rel_diff": float((diff / denom).max()) if diff.size else 0.0,
        "cosine_similarity": float(np.dot(ref.reshape(-1), android.reshape(-1)) / ((np.linalg.norm(ref) * np.linalg.norm(android)) + 1e-12)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit RPFQViT Android debug tensors against reference tensors.")
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--android-debug-dir", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, required=True)
    args = parser.parse_args()

    names = [
        "input_tensor",
        "patch_embed",
        "block0_norm1",
        "block0_qkv",
        "block0_attn_out",
        "block0_mlp_out",
        "final_norm",
        "logits",
    ]
    results = []
    for name in names:
        if (args.reference_dir / f"{name}.npy").exists() and (args.android_debug_dir / f"{name}.bin").exists():
            results.append(compare_tensor(name, args.reference_dir, args.android_debug_dir))

    report = {"results": results}
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# RPFQViT Android Parity Report", ""]
    for item in results:
        lines.append(f"- `{item['name']}` max_abs_diff={item['max_abs_diff']:.6g} mean_abs_diff={item['mean_abs_diff']:.6g}")
    args.md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
