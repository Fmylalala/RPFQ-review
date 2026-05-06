"""Check that quantized models are registered with timm.

This script is intentionally lightweight: it imports quant_models to trigger
registration, checks the expected public model names, and creates a few tiny
models from the complex-quantization mainline.
"""

import os
import sys

import torch


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import quant_models  # noqa: F401

    print("OK: imported quant_models")
except ImportError as exc:
    print(f"FAIL: failed to import quant_models: {exc}")
    sys.exit(1)

from timm.models import create_model, is_model, list_models


COMPLEX_QUANT_MODELS = [
    "rpfq_vit_base_patch16_224_q",
    "rpfq_vit_large_patch16_224_q",
    "rpfq_vit_small_patch16_224_q",
    "rpfq_vit_tiny_patch16_224_q",
    "rpfq_deit_base_patch16_224_q",
    "rpfq_deit_small_patch16_224_q",
    "rpfq_deit_tiny_patch16_224_q",
    "rpfq_swin_base_patch4_window7_224_q",
    "rpfq_swin_small_patch4_window7_224_q",
    "rpfq_swin_tiny_patch4_window7_224_q",
    "rpfq_vit_base_patch16_224_q_vision",
    "rpfq_vit_large_patch16_224_q_vision",
    "rpfq_vit_small_patch16_224_q_vision",
    "rpfq_vit_tiny_patch16_224_q_vision",
]

BASELINE_QUANT_MODELS = [
    "vit_base_patch16_224_q",
    "vit_large_patch16_224_q",
    "vit_small_patch16_224_q",
    "vit_tiny_patch16_224_q",
    "deit_base_patch16_224_q",
    "deit_small_patch16_224_q",
    "deit_tiny_patch16_224_q",
    "swin_base_patch4_window7_224_q",
    "swin_small_patch4_window7_224_q",
    "swin_tiny_patch4_window7_224_q",
]


def test_model_registration() -> bool:
    print("\n" + "=" * 60)
    print("Testing model registration")
    print("=" * 60)

    expected_models = COMPLEX_QUANT_MODELS + BASELINE_QUANT_MODELS
    missing = []
    for model_name in expected_models:
        if is_model(model_name):
            print(f"  OK: {model_name}")
        else:
            print(f"  FAIL: {model_name} not found")
            missing.append(model_name)

    print("\n" + "-" * 60)
    print(f"Summary: {len(expected_models) - len(missing)} registered, {len(missing)} missing")
    print("-" * 60)
    return not missing


def test_model_creation() -> bool:
    print("\n" + "=" * 60)
    print("Testing lightweight complex model creation")
    print("=" * 60)

    test_models = [
        "rpfq_vit_tiny_patch16_224_q",
        "rpfq_deit_tiny_patch16_224_q",
        "rpfq_swin_tiny_patch4_window7_224_q",
    ]
    base_model_kwargs = {
        "quant_method_u": "complex_phase_v2",
        "quant_method_w": "complex_phase_v2",
        "enable_act_quant": True,
        "enable_rotation": False,
    }

    failures = []
    for act_num_bits in (2, 4):
        model_kwargs = {**base_model_kwargs, "act_num_bits": act_num_bits}
        for model_name in test_models:
            try:
                model = create_model(
                    model_name,
                    pretrained=False,
                    num_classes=10,
                    img_size=32,
                    **model_kwargs,
                )
                model.eval()
                with torch.no_grad():
                    output = model(torch.randn(1, 3, 32, 32))

                quant_layers = [m for m in model.modules() if hasattr(m, "act_quantizer")]
                if not quant_layers:
                    raise AssertionError("no complex quantized linear layers found")
                for layer in quant_layers:
                    if layer.act_quantizer is None:
                        raise AssertionError("activation quantizer unexpectedly disabled")
                    if getattr(layer.act_quantizer, "num_bits", act_num_bits) != act_num_bits:
                        raise AssertionError("activation bit width was not propagated")

                param_count = sum(p.numel() for p in model.parameters())
                print(
                    f"  OK: {model_name} act={act_num_bits}-bit "
                    f"({param_count / 1e6:.2f}M params, output={tuple(output.shape)})"
                )
            except Exception as exc:
                print(f"  FAIL: {model_name} act={act_num_bits}-bit: {exc}")
                failures.append(f"{model_name}:{act_num_bits}")

    print("\n" + "-" * 60)
    total_cases = len(test_models) * 2
    print(f"Summary: {total_cases - len(failures)} created, {len(failures)} failed")
    print("-" * 60)
    return not failures


def list_registered_quant_models() -> int:
    print("\n" + "=" * 60)
    print("Registered ViT-family quantized models")
    print("=" * 60)

    names = [
        name
        for name in list_models()
        if ("vit" in name.lower() or "deit" in name.lower() or "swin" in name.lower())
        and name.endswith("_q")
    ]
    for model_name in sorted(names):
        print(f"  - {model_name}")
    print(f"\nTotal: {len(names)}")
    return len(names)


def test_optional_kernel_import() -> bool:
    print("\n" + "=" * 60)
    print("Testing optional Triton benchmark module import")
    print("=" * 60)

    try:
        import quant_models.kernel as kernel

        if getattr(kernel, "_TRITON_IMPORT_ERROR", None) is not None:
            try:
                kernel._require_triton()
            except RuntimeError as exc:
                print(f"  OK: missing Triton reports optional dependency: {exc}")
            else:
                print("  FAIL: missing Triton did not raise a RuntimeError")
                return False
        else:
            print("  OK: Triton is available")
        return True
    except Exception as exc:
        print(f"  FAIL: optional kernel module import failed: {exc}")
        return False


if __name__ == "__main__":
    registration_ok = test_model_registration()
    list_registered_quant_models()
    creation_ok = test_model_creation()
    kernel_ok = test_optional_kernel_import()

    print("\n" + "=" * 60)
    if registration_ok and creation_ok and kernel_ok:
        print("OK: all tests passed")
        sys.exit(0)

    print("FAIL: some tests failed")
    sys.exit(1)
