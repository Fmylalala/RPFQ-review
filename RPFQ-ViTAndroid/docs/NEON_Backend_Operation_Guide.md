# NEON Backend Operation Guide

The packed `complex_phase_v2` wide-linear runtime defaults to `wideLinearOptimized()`.

## Runtime Paths

- `wideLinearOptimized()` precomputes reciprocal gamma values, fuses U/W residual branches, parallelizes output complex rows, and uses ARM NEON for the packed 4-column inner loop when available.
- `wideLinearReference()` keeps the original scalar path for debugging and parity checks.
- Non-NEON targets automatically use fused scalar code.

## Runtime Logs

```bash
adb logcat -s RPFQViTEngine
```

Expected timing labels include `patchify`, `patch_embed`, `blockN.attn.qkv`, `blockN.attn.proj`, `blockN.mlp.fc1`, `blockN.mlp.fc2`, `head`, and `predict`.

## Reference and Parity Modes

```cmake
target_compile_definitions(rpfq_vit_runtime PRIVATE RPFQ_VIT_FORCE_REFERENCE_PHASE_LINEAR=1)
```

```cmake
target_compile_definitions(rpfq_vit_runtime PRIVATE RPFQ_VIT_CHECK_PHASE_LINEAR_PARITY=1)
```

Parity mode throws when optimized output exceeds `max_abs_diff > 5e-4` or `mean_abs_diff > 1e-5`. Disable parity mode for latency testing because it runs both backends.

## Troubleshooting

- Missing `model.safetensors`: restore or regenerate the model bundle before packaging.
- No NEON speedup on desktop or x86 builds: NEON code is ARM guarded.
- Parity failure: rebuild once with `RPFQ_VIT_FORCE_REFERENCE_PHASE_LINEAR=1`, then inspect the failing layer through debug tensor export.
