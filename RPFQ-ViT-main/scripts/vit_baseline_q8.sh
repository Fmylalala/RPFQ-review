#!/bin/bash
# 8-bit quantized ViT training script.

set -euo pipefail

: "${IMAGENET1K_DIR:?Set IMAGENET1K_DIR to your ImageNet-1K dataset directory.}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

torchrun --nproc_per_node="${NPROC_PER_NODE:-8}" scripts/train.py "${IMAGENET1K_DIR}" \
  --model vit_base_patch16_224_q \
  --quant-bits 8 \
  --quant-scheme symmetric \
  --epochs 300 \
  --batch-size 128 \
  --opt adamw --lr 5e-4 --weight-decay 0.05 \
  --sched cosine --warmup-epochs 5 \
  --reprob 0.25 --remode pixel \
  --mixup 0.8 --cutmix 1.0 --aa rand-m9-mstd0.5-inc1 \
  --drop-path 0.1 \
  --color-jitter 0.1 --hflip 0.5 \
  --clip-grad 1.0 \
  --model-ema \
  --output ./output/vit-b16-q8 \
  --experiment vit-b16-q8
