# RPFQ-ViT Workspace

This workspace contains three related parts of the RPFQ-ViT complex-quantization project:

- `RPFQ-ViT-main/`: Python research code for complex-quantized ViT, DeiT, and Swin models.
- `RPFQ-ViTAndroid/`: Android demo app with Kotlin UI, JNI, and native C++ inference.
- `RPFQ-ViTApple/`: Apple MLX Swift demo app plus export and parity-audit tools.

## Recommended Workflow

1. Train or validate models in `RPFQ-ViT-main`.
2. Export an Apple MLX bundle from a checkpoint with `RPFQ-ViTApple/scripts/export_rpfq_vit_to_mlx.py`.
3. Run the Apple demo from `RPFQ-ViTApple`.
4. Prepare an Android bundle from the Apple bundle with `RPFQ-ViTAndroid/tools/prepare_android_bundle.py`.
5. Build and run the Android demo from `RPFQ-ViTAndroid`.

## Quick Commands

```bash
cd RPFQ-ViT-main
pip install -r requirements.txt
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export IMAGENET1K_DIR="<IMAGENET1K_DIR>"
bash scripts/rpfq_vit_baseline_q.sh
```

```bash
cd RPFQ-ViTApple
python scripts/export_rpfq_vit_to_mlx.py \
  --checkpoint <CHECKPOINT_PATH> \
  --variant rpfq_vit_base_patch16_224_q
```

```bash
cd RPFQ-ViTAndroid
python tools/prepare_android_bundle.py \
  --source ../RPFQ-ViTApple/RPFQViTMobile/rpfq_vit-base-mlx.bundle \
  --output app/src/main/assets/rpfq_vit-base-android-q2.bundle
```

## Model Assets

Model bundles use English labels from `labels.txt`. Large artifacts such as `model.safetensors`, checkpoints, APK outputs, and local IDE state are intentionally excluded from source handoffs. Restore or regenerate model weights before building runnable mobile apps.

## Path Convention

Use relative paths inside this workspace and placeholders such as `<IMAGENET1K_DIR>`, `<CHECKPOINT_PATH>`, and `<JDK17_HOME>` for machine-specific locations. Runtime-computed platform paths are allowed when required by Android, Apple, Python, or shell APIs.
