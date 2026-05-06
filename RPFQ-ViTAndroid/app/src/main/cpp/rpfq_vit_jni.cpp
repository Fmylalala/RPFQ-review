#include "rpfq_vit_engine.h"

#include <android/log.h>
#include <jni.h>

#include <exception>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr const char* kTag = "RPFQViTNative";

rpfq_vit::RPFQViTEngine* engineFromHandle(jlong handle) {
    return reinterpret_cast<rpfq_vit::RPFQViTEngine*>(handle);
}

std::string toString(JNIEnv* env, jstring value) {
    if (!value) return {};
    const char* chars = env->GetStringUTFChars(value, nullptr);
    std::string out(chars ? chars : "");
    if (chars) env->ReleaseStringUTFChars(value, chars);
    return out;
}

void throwRuntime(JNIEnv* env, const std::string& message) {
    __android_log_print(ANDROID_LOG_ERROR, kTag, "%s", message.c_str());
    jclass cls = env->FindClass("java/lang/RuntimeException");
    if (cls) env->ThrowNew(cls, message.c_str());
}

} // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_rpfq_1vit_android_runtime_NativeRPFQViTEngine_nativeCreate(JNIEnv*, jobject) {
    auto* engine = new rpfq_vit::RPFQViTEngine();
    return reinterpret_cast<jlong>(engine);
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_rpfq_1vit_android_runtime_NativeRPFQViTEngine_nativeLoadBundle(
    JNIEnv* env,
    jobject,
    jlong handle,
    jstring bundlePath) {
    try {
        auto* engine = engineFromHandle(handle);
        if (!engine) throw std::runtime_error("Native engine handle is null.");
        return engine->loadBundle(toString(env, bundlePath)) ? JNI_TRUE : JNI_FALSE;
    } catch (const std::exception& e) {
        throwRuntime(env, e.what());
        return JNI_FALSE;
    }
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_rpfq_1vit_android_runtime_NativeRPFQViTEngine_nativePredict(
    JNIEnv* env,
    jobject,
    jlong handle,
    jfloatArray input) {
    try {
        auto* engine = engineFromHandle(handle);
        if (!engine) throw std::runtime_error("Native engine handle is null.");
        if (!input) throw std::runtime_error("Input array is null.");

        jsize len = env->GetArrayLength(input);
        std::vector<float> inputVec(static_cast<size_t>(len));
        env->GetFloatArrayRegion(input, 0, len, inputVec.data());

        std::vector<float> logits = engine->predict(inputVec);
        jfloatArray out = env->NewFloatArray(static_cast<jsize>(logits.size()));
        env->SetFloatArrayRegion(out, 0, static_cast<jsize>(logits.size()), logits.data());
        return out;
    } catch (const std::exception& e) {
        throwRuntime(env, e.what());
        return nullptr;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_rpfq_1vit_android_runtime_NativeRPFQViTEngine_nativeSetDebug(
    JNIEnv* env,
    jobject,
    jlong handle,
    jboolean enabled,
    jstring debugDir) {
    try {
        auto* engine = engineFromHandle(handle);
        if (!engine) throw std::runtime_error("Native engine handle is null.");
        engine->setDebugEnabled(enabled == JNI_TRUE);
        engine->setDebugOutputDir(toString(env, debugDir));
    } catch (const std::exception& e) {
        throwRuntime(env, e.what());
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_rpfq_1vit_android_runtime_NativeRPFQViTEngine_nativeDestroy(JNIEnv*, jobject, jlong handle) {
    delete engineFromHandle(handle);
}
