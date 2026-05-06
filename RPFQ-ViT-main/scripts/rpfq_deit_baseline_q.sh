#!/bin/bash
# Complex-quantized RPFQ-DeiT training script.

set -euo pipefail

: "${IMAGENET1K_DIR:?Set IMAGENET1K_DIR to your ImageNet-1K dataset directory.}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -n "${CONDA_SH:-}" ]]; then
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV:-ifairy-vit}"
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"


OUTPUT_DIR="./output/rpfq-deit-tiny-q2a4"
EXPERIMENT="rpfq-deit-tiny-q2a4"
mkdir -p "${OUTPUT_DIR}"
LOG_DIR="${OUTPUT_DIR}/${EXPERIMENT}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train.log"

"${PYTHON_BIN}" -m torch.distributed.run --nproc_per_node="${NPROC_PER_NODE:-8}" scripts/train.py "${IMAGENET1K_DIR}" \
  --model rpfq_deit_tiny_patch16_224_q \
  --model-kwargs quant_method_u=complex_phase_v2 quant_method_w=complex_phase_v2 enable_rotation=True \
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
  --output "${OUTPUT_DIR}" \
  --experiment "${EXPERIMENT}" \
  2>&1 | tee "${LOG_FILE}"
