# Android Bundle Format

The APK packages `rpfq_vit-base-android-q2.bundle` under app assets and copies it to app-private storage on launch.

Required files:

- `model.safetensors`
- `config.json`
- `preprocess.json`
- `manifest.json`
- `labels.txt`

The manifest must declare `quant_method=complex_phase_v2`, `real_quantized_inference=true`, and `runtime_backend=android_ndk_cpp`.

`preprocess.json` drives Android input preparation. The runtime currently supports RGB inputs with `resize_mode=shortest_side`, `crop_mode=center`, and `interpolation` values of `bicubic`, `bilinear`, `linear`, or `nearest`.
