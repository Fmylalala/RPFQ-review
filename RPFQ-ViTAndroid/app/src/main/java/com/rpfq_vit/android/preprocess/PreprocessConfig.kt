package com.rpfq_vit.android.preprocess

data class PreprocessConfig(
    val imageSize: Int,
    val patchSize: Int,
    val numChannels: Int,
    val mean: List<Float>,
    val std: List<Float>,
    val interpolation: String,
    val cropPct: Float,
    val cropMode: String,
    val resizeMode: String
)
