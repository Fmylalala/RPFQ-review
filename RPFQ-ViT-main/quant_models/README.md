# Quantized Model Package

Importing `quant_models` registers the project models with timm.

## Complex Quantization

- `quant_i.py`: complex weight quantization, activation quantization, and `QATLinearComplexPhase`.
- `quant_rpfq_vit.py`: RPFQViT adapter for timm `VisionTransformer`.
- `quant_rpfq_deit.py`: RPFQDeiT and distilled ViT adapter.
- `quant_rpfq_swin.py`: RPFQSwin adapter.
- `quant_rpfq_vit_vision.py`: vision-aware RPFQViT variant.

Supported public parameters include `quant_method_u`, `quant_method_w`, `enable_act_quant`, `act_num_bits`, `enable_rotation`, and `quant_group_size`.

## Baselines

`quant_ops.py`, `quant_vit.py`, `quant_deit.py`, and `quant_swin.py` are q8-style baseline or placeholder paths. They are retained for comparison and are not the main complex-quantization implementation.

Use `timm.models.create_model(...)` from external scripts instead of depending on this package's internal file layout.
