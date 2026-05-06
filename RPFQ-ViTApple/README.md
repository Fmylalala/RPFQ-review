# RPFQ-ViT Apple Demo

This project is the Apple MLX Swift demo for RPFQ-ViT mobile inference. It provides iOS and macOS app targets, checkpoint export tools, and parity-audit scripts.

## What It Does

- Loads `rpfq_vit-small-mlx.bundle` or `rpfq_vit-base-mlx.bundle` from app resources after you provide the bundle.
- Runs image classification through MLX Swift and the RPFQ-ViT runtime.
- Displays top predictions with English labels from `labels.txt`.
- Exports debug tensors when `RPFQ_VIT_DEBUG_EXPORT` is enabled.

## Project Layout

- `project.yml`: XcodeGen definition for iOS 17 and macOS 14 targets.
- `RPFQViTMobile/*.swift`: Swift UI, model loading, preprocessing, prediction, and runtime support.
- `RPFQViTMobile/*.bundle`: expected destination for manually prepared MLX model bundles.
- `scripts/export_rpfq_vit_to_mlx.py`: checkpoint-to-MLX-bundle exporter.
- `scripts/audit_rpfq_vit_ios_parity.py`: training/export/runtime parity audit.
- `scripts/audit_preprocess_numpy.py`: preprocessing parity audit against NumPy/PIL.

## Generate or Open the Project

Use the checked-in Xcode project directly, or regenerate it from `project.yml` when XcodeGen is available. Before building a runnable app, download or export a model bundle and place it under `RPFQViTMobile/`.

```bash
xcodegen generate
open RPFQViTMobile.xcodeproj
```

The project depends on MLX Swift through Swift Package Manager. Model bundles are not included in this folder.

## Model Bundles

This source tree does not include model weights or ready-to-run `.bundle` directories. Download a checkpoint/model artifact separately, then export it into an MLX bundle:

```bash
python scripts/export_rpfq_vit_to_mlx.py \
  --checkpoint <CHECKPOINT_PATH> \
  --variant rpfq_vit_base_patch16_224_q \
  --output RPFQViTMobile/rpfq_vit-base-mlx.bundle
```

Use `--labels <LABELS_TXT>` to provide a custom English `labels.txt`. A runnable bundle must contain `config.json`, `model.safetensors`, `preprocess.json`, `labels.txt`, and `manifest.json`.

## Run Audits

```bash
python scripts/audit_preprocess_numpy.py \
  --patches-json <PATCHES_JSON> \
  --image <IMAGE_PATH>
```

```bash
python scripts/audit_rpfq_vit_ios_parity.py \
  --checkpoint <CHECKPOINT_PATH> \
  --variant rpfq_vit_base_patch16_224_q \
  --mode all
```

## Debug Export

Enable debug export from the app UI or set `RPFQ_VIT_DEBUG_EXPORT=1`. Optional settings include `RPFQ_VIT_DEBUG_EXPORT_DIR`, `RPFQ_VIT_DEBUG_STAGES`, `RPFQ_VIT_DEBUG_MAX_VALUES`, and `RPFQ_VIT_DEBUG_MAX_PATCH_VALUES`.

Model weights are large artifacts and are not included in this source handoff. Restore or regenerate the complete bundle before running the app.
