package com.rpfq_vit.android.model

import android.content.Context
import com.rpfq_vit.android.preprocess.PreprocessConfig
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

object AssetBundleInstaller {
    private const val AssetBundleName = "rpfq_vit-base-android-q2.bundle"

    fun ensureInstalled(context: Context): File {
        val target = File(context.filesDir, AssetBundleName)
        copyAssetTree(context, AssetBundleName, target)
        return target
    }

    fun loadMetadata(bundleDir: File): ModelMetadata {
        val config = JSONObject(File(bundleDir, "config.json").readText(Charsets.UTF_8))
        val manifest = JSONObject(File(bundleDir, "manifest.json").readText(Charsets.UTF_8))
        val rpfq_vit = config.optJSONObject("rpfq_vit")
        return ModelMetadata(
            bundleName = bundleDir.name,
            variant = config.optString("variant", "rpfq_vit_base_patch16_224_q"),
            quantMethod = rpfq_vit?.optString("quant_method") ?: manifest.optString("quant_method", "complex_phase_v2"),
            backend = manifest.optString("runtime_backend", "android_ndk_cpp"),
            inputSize = config.optInt("image_size", 224),
            patchSize = config.optInt("patch_size", 16),
            numClasses = config.optInt("num_classes", 1000)
        )
    }

    fun loadPreprocessConfig(bundleDir: File): PreprocessConfig {
        val preprocessFile = File(bundleDir, "preprocess.json")
        val root = if (preprocessFile.exists()) {
            JSONObject(preprocessFile.readText(Charsets.UTF_8))
        } else {
            val config = JSONObject(File(bundleDir, "config.json").readText(Charsets.UTF_8))
            config.optJSONObject("preprocess") ?: config
        }

        return PreprocessConfig(
            imageSize = root.optInt("image_size", 224),
            patchSize = root.optInt("patch_size", 16),
            numChannels = root.optInt("num_channels", 3),
            mean = floatList(root.optJSONArray("mean"), listOf(0.485f, 0.456f, 0.406f)),
            std = floatList(root.optJSONArray("std"), listOf(0.229f, 0.224f, 0.225f)),
            interpolation = root.optString("interpolation", "bicubic"),
            cropPct = root.optDouble("crop_pct", 0.875).toFloat(),
            cropMode = root.optString("crop_mode", "center"),
            resizeMode = root.optString("resize_mode", "shortest_side")
        )
    }

    private fun copyAssetTree(context: Context, assetPath: String, target: File) {
        val children = context.assets.list(assetPath).orEmpty()
        if (children.isEmpty()) {
            target.parentFile?.mkdirs()
            context.assets.open(assetPath).use { input ->
                target.outputStream().use { output -> input.copyTo(output) }
            }
            return
        }

        target.mkdirs()
        children.forEach { child ->
            copyAssetTree(context, "$assetPath/$child", File(target, child))
        }
    }

    private fun floatList(values: JSONArray?, fallback: List<Float>): List<Float> {
        if (values == null || values.length() == 0) return fallback
        return List(values.length()) { index -> values.optDouble(index).toFloat() }
    }
}
