package com.rpfq_vit.android.preprocess

import android.content.ContentResolver
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.net.Uri
import kotlin.math.abs
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.roundToInt

object ImagePreprocessor {
    fun decodeBitmap(resolver: ContentResolver, uri: Uri): Bitmap {
        resolver.openInputStream(uri).use { input ->
            requireNotNull(input) { "Unable to open selected image." }
            return requireNotNull(BitmapFactory.decodeStream(input)) { "Unable to decode selected image." }
        }
    }

    fun preprocess(bitmap: Bitmap, config: PreprocessConfig): FloatArray {
        require(config.imageSize > 0) { "image_size must be positive." }
        require(config.cropPct > 0f) { "crop_pct must be positive." }
        require(config.numChannels == 3) { "Only RGB preprocessing is supported." }
        require(config.mean.size >= 3 && config.std.size >= 3) { "mean and std must contain RGB values." }

        val resized = resize(bitmap, config)
        val cropped = crop(resized, config)
        val imageSize = config.imageSize
        val out = FloatArray(config.numChannels * imageSize * imageSize)

        for (y in 0 until imageSize) {
            for (x in 0 until imageSize) {
                val pixel = cropped.getPixel(x, y)
                val r = ((pixel shr 16) and 0xff) / 255.0f
                val g = ((pixel shr 8) and 0xff) / 255.0f
                val b = (pixel and 0xff) / 255.0f
                val idx = y * imageSize + x
                out[idx] = (r - config.mean[0]) / config.std[0]
                out[imageSize * imageSize + idx] = (g - config.mean[1]) / config.std[1]
                out[2 * imageSize * imageSize + idx] = (b - config.mean[2]) / config.std[2]
            }
        }
        return out
    }

    private fun resize(bitmap: Bitmap, config: PreprocessConfig): Bitmap {
        val mode = config.resizeMode.lowercase()
        require(mode == "shortest_side") { "Unsupported resize_mode: ${config.resizeMode}" }
        val targetShortest = (config.imageSize / config.cropPct).roundToInt()
        val scale = targetShortest.toFloat() / minOf(bitmap.width, bitmap.height).toFloat()
        val targetWidth = max(1, (bitmap.width * scale).roundToInt())
        val targetHeight = max(1, (bitmap.height * scale).roundToInt())
        return resizeBitmap(bitmap, targetWidth, targetHeight, config.interpolation)
    }

    private fun crop(bitmap: Bitmap, config: PreprocessConfig): Bitmap {
        val mode = config.cropMode.lowercase()
        require(mode == "center") { "Unsupported crop_mode: ${config.cropMode}" }
        val imageSize = config.imageSize
        val left = ((bitmap.width - imageSize) / 2).coerceAtLeast(0)
        val top = ((bitmap.height - imageSize) / 2).coerceAtLeast(0)
        return Bitmap.createBitmap(bitmap, left, top, imageSize, imageSize)
    }

    private fun resizeBitmap(bitmap: Bitmap, width: Int, height: Int, interpolation: String): Bitmap {
        return when (interpolation.lowercase()) {
            "bicubic" -> resizeBicubic(bitmap, width, height)
            "bilinear", "linear" -> Bitmap.createScaledBitmap(bitmap, width, height, true)
            "nearest" -> Bitmap.createScaledBitmap(bitmap, width, height, false)
            else -> throw IllegalArgumentException("Unsupported interpolation: $interpolation")
        }
    }

    private fun resizeBicubic(src: Bitmap, dstWidth: Int, dstHeight: Int): Bitmap {
        val dst = Bitmap.createBitmap(dstWidth, dstHeight, Bitmap.Config.ARGB_8888)
        val scaleX = src.width.toFloat() / dstWidth.toFloat()
        val scaleY = src.height.toFloat() / dstHeight.toFloat()

        for (y in 0 until dstHeight) {
            val srcY = (y + 0.5f) * scaleY - 0.5f
            val yBase = floor(srcY).toInt()
            for (x in 0 until dstWidth) {
                val srcX = (x + 0.5f) * scaleX - 0.5f
                val xBase = floor(srcX).toInt()
                var a = 0f
                var r = 0f
                var g = 0f
                var b = 0f
                var weightSum = 0f

                for (ky in -1..2) {
                    val sampleY = (yBase + ky).coerceIn(0, src.height - 1)
                    val wy = cubic(srcY - (yBase + ky))
                    for (kx in -1..2) {
                        val sampleX = (xBase + kx).coerceIn(0, src.width - 1)
                        val weight = wy * cubic(srcX - (xBase + kx))
                        val pixel = src.getPixel(sampleX, sampleY)
                        a += Color.alpha(pixel) * weight
                        r += Color.red(pixel) * weight
                        g += Color.green(pixel) * weight
                        b += Color.blue(pixel) * weight
                        weightSum += weight
                    }
                }

                dst.setPixel(
                    x,
                    y,
                    Color.argb(
                        clampToByte(a / weightSum),
                        clampToByte(r / weightSum),
                        clampToByte(g / weightSum),
                        clampToByte(b / weightSum)
                    )
                )
            }
        }
        return dst
    }

    private fun cubic(value: Float): Float {
        val x = abs(value)
        return when {
            x <= 1f -> 1.5f * x * x * x - 2.5f * x * x + 1f
            x < 2f -> -0.5f * x * x * x + 2.5f * x * x - 4f * x + 2f
            else -> 0f
        }
    }

    private fun clampToByte(value: Float): Int {
        return value.roundToInt().coerceIn(0, 255)
    }
}
