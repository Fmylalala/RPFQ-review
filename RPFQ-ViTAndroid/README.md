# RPFQ-ViT Android Demo

This project is an Android single-image classification demo for an RPFQ-ViT Base/16 bundle. It uses Kotlin UI, JNI, and a native C++ runtime with an optimized packed `complex_phase_v2` backend.

## What It Does

- Loads an offline model bundle from APK assets.
- Runs native packed-weight RPFQ-ViT inference.
- Shows top-5 English labels, confidence scores, and runtime timing.
- Supports debug tensor export for parity checks.

## Build and Run

Requirements: JDK 17, Android SDK 36, Android NDK, CMake, and Gradle compatible with Android Gradle Plugin 8.10.1. This repository does not include a Gradle wrapper; install Gradle yourself, put it on `PATH`, or build from Android Studio after configuring the Android SDK/NDK.

```powershell
$env:JAVA_HOME = "<JDK17_HOME>"
$env:ANDROID_HOME = "$env:LOCALAPPDATA\Android\Sdk"
$env:ANDROID_SDK_ROOT = $env:ANDROID_HOME
gradle :app:assembleDebug
```

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb logcat -s RPFQViTEngine
```

Open the app, select an image, and run classification. The app does not require internet access.

## Prepare Bundle

This folder does not include model weights. The checked-in `app/src/main/assets/rpfq_vit-base-android-q2.bundle` metadata is not enough for full inference unless you add `model.safetensors`.

Prepare an Apple MLX bundle first, then convert or copy it into Android assets:

```bash
python tools/prepare_android_bundle.py \
  --source ../RPFQ-ViTApple/RPFQViTMobile/rpfq_vit-base-mlx.bundle \
  --output app/src/main/assets/rpfq_vit-base-android-q2.bundle
```

The bundle format is documented in `docs/Android_Bundle_Format.md`. A runnable Android bundle must contain `model.safetensors`, `config.json`, `preprocess.json`, `labels.txt`, and `manifest.json`.

The app reads image size, crop settings, normalization values, resize mode, and interpolation from `preprocess.json` at launch so Android preprocessing stays aligned with the exported model bundle.

## Debug and Audits

- Runtime design: `docs/Android_Runtime_Design.md`
- Bundle format: `docs/Android_Bundle_Format.md`
- Debug tensor export: `docs/Android_Debugging_Guide.md`
- NEON backend and parity modes: `docs/NEON_Backend_Operation_Guide.md`
