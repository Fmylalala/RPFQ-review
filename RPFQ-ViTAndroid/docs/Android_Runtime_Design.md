# Android Runtime Design

The Android runtime is a Kotlin/JNI app backed by native C++ inference.

## Bundle Representation

- Packed `complex_phase_v2` tensors preserve the Apple MLX bundle representation.
- 2-bit codes are stored low-bit-first with `00=-1`, `01=+1`, `10=-i`, and `11=+i`.
- U and W components use two residual stages.
- Gamma tensors follow the Apple runtime interpretation.

## Default Execution

`wideLinearOptimized()` is the default wide-linear path. It precomputes reciprocal gamma values, fuses residual branches, splits output complex rows across worker threads, and uses ARM NEON where available.

The scalar reference path remains available as `wideLinearReference()` through `RPFQ_VIT_FORCE_REFERENCE_PHASE_LINEAR`.
