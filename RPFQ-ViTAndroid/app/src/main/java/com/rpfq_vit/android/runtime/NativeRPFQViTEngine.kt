package com.rpfq_vit.android.runtime

import java.io.Closeable

class NativeRPFQViTEngine : Closeable {
    private var handle: Long = nativeCreate()

    private external fun nativeCreate(): Long
    private external fun nativeLoadBundle(handle: Long, bundlePath: String): Boolean
    private external fun nativePredict(handle: Long, input: FloatArray): FloatArray
    private external fun nativeSetDebug(handle: Long, enabled: Boolean, debugDir: String)
    private external fun nativeDestroy(handle: Long)

    fun load(bundlePath: String): Boolean {
        check(handle != 0L) { "Native engine is closed." }
        return nativeLoadBundle(handle, bundlePath)
    }

    fun predict(input: FloatArray): FloatArray {
        check(handle != 0L) { "Native engine is closed." }
        return nativePredict(handle, input)
    }

    fun setDebug(enabled: Boolean, debugDir: String) {
        check(handle != 0L) { "Native engine is closed." }
        nativeSetDebug(handle, enabled, debugDir)
    }

    override fun close() {
        if (handle != 0L) {
            nativeDestroy(handle)
            handle = 0L
        }
    }

    companion object {
        init {
            System.loadLibrary("rpfq_vit_runtime")
        }
    }
}
