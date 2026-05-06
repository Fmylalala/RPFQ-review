#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare an Android RPFQViT bundle from an Apple MLX bundle.")
    parser.add_argument("--source", type=Path, required=True, help="Apple .bundle directory.")
    parser.add_argument("--output", type=Path, required=True, help="Android .bundle directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ["model.safetensors", "config.json", "preprocess.json", "labels.txt"]:
        shutil.copy2(args.source / name, args.output / name)

    config = json.loads((args.output / "config.json").read_text(encoding="utf-8"))
    manifest = {
        "format": "rpfq_vit-android-native-bundle",
        "format_version": 1,
        "model_type": "rpfq_vit",
        "variant": config.get("variant", "rpfq_vit_base_patch16_224_q"),
        "quant_method": config.get("rpfq_vit", {}).get("quant_method", "complex_phase_v2"),
        "activation_quantization": False,
        "real_quantized_inference": True,
        "runtime_backend": "android_ndk_cpp",
        "preferred_abi": "arm64-v8a",
        "input_shape": [1, 3, config.get("image_size", 224), config.get("image_size", 224)],
        "num_classes": config.get("num_classes", 1000),
        "image_size": config.get("image_size", 224),
        "patch_size": config.get("patch_size", 16),
        "export_source": "apple_mlx_bundle",
        "export_script": "tools/prepare_android_bundle.py",
        "requires_custom_phase_linear": True,
        "notes": [
            "This bundle preserves complex_phase_v2 packed weights from the existing Apple bundle.",
            "Do not replace this with dequantized ONNX for final demo APK.",
        ],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared Android bundle at {args.output}")


if __name__ == "__main__":
    main()
