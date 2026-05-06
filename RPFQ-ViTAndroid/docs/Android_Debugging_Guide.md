# Android Debugging Guide

Enable **Debug Export** in the app before running classification.

Debug tensors are written to app-private storage:

```text
<APP_PRIVATE_FILES>/debug/
```

Pull files from a debuggable APK:

```bash
adb exec-out run-as com.rpfq_vit.android tar -cf - files/debug > reports/android_debug.tar
mkdir -p reports/android_debug
tar -xf reports/android_debug.tar -C reports/android_debug
```

Use `tools/audit_rpfq_vit_android_parity.py` to compare exported tensors with reference outputs.
