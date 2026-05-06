# RPFQ-ViT Research Code

This project contains Python code for complex quantization of Vision Transformer models. It registers complex-quantized RPFQ-ViT, RPFQ-DeiT, and RPFQ-Swin models with timm, plus baseline q8-style ViT, DeiT, and Swin variants.

## Main Components

- `quant_models/`: quantized layers, model adapters, and optional Triton/CUDA benchmark kernels.
- `configs/`: portable YAML experiment configs.
- `scripts/train.py`: training entrypoint.
- `scripts/validate.py`: validation entrypoint.
- `tests/test_models.py`: model-registration and lightweight creation checks.

## Setup

```bash
pip install -r requirements.txt
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

Install optional benchmark dependencies only when using `quant_models/kernel.py`:

```bash
pip install -r requirements-benchmark.txt
```

## Train

Prefer YAML configs and pass local dataset paths at runtime:

```bash
torchrun --nproc_per_node=8 scripts/train.py \
  --config configs/rpfq_vit_b16_complex_q2a4.yaml \
  <IMAGENET1K_DIR>
```

Shell wrappers use environment variables instead of hardcoded machine paths:

```bash
export IMAGENET1K_DIR="<IMAGENET1K_DIR>"
export NPROC_PER_NODE=8
bash scripts/rpfq_vit_baseline_q.sh
```

## Validate

```bash
python scripts/validate.py <IMAGENET1K_DIR> \
  --model rpfq_vit_base_patch16_224_q \
  --model-kwargs quant_method_u=complex_phase_v2 quant_method_w=complex_phase_v2 enable_rotation=False \
  --checkpoint ./output/rpfq-vit-b16-q2a4/model_best.pth.tar
```

## Use Models

```python
import quant_models
from timm.models import create_model

model = create_model(
    "rpfq_vit_base_patch16_224_q",
    pretrained=False,
    quant_method_u="complex_phase_v2",
    quant_method_w="complex_phase_v2",
    enable_act_quant=True,
    act_num_bits=4,
    enable_rotation=False,
)
```

## Test

```bash
python tests/test_models.py
```

ImageNet data, checkpoints, model weights, exported bundles, and training outputs are not included in this source handoff. Download or generate those artifacts separately before training validation, mobile export, or mobile runtime testing.
