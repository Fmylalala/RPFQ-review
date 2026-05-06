package com.rpfq_vit.android.model

import java.io.File

object LabelLoader {
    fun load(bundleDir: File): List<String> = readLabels(File(bundleDir, "labels.txt"))

    private fun readLabels(file: File): List<String> {
        if (!file.exists()) return emptyList()
        return file.readLines(Charsets.UTF_8)
            .map { it.trim() }
            .filter { it.isNotEmpty() }
    }
}
