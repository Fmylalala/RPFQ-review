package com.rpfq_vit.android.data

data class InferenceProfile(
    val preprocessMs: Long,
    val inferenceMs: Long,
    val postprocessMs: Long,
    val totalMs: Long
)
