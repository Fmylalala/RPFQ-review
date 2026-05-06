package com.rpfq_vit.android.model

data class ModelMetadata(
    val bundleName: String,
    val variant: String,
    val quantMethod: String,
    val backend: String,
    val inputSize: Int,
    val patchSize: Int,
    val numClasses: Int
)
