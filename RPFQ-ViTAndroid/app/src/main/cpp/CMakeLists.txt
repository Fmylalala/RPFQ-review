cmake_minimum_required(VERSION 3.22.1)

project(rpfq_vit_android_runtime)

add_library(
    rpfq_vit_runtime
    SHARED
    rpfq_vit_jni.cpp
    rpfq_vit_engine.cpp
    tensor.cpp
    json_utils.cpp
    bundle_loader.cpp
    safetensors_loader.cpp
    debug_dump.cpp
    ops/layernorm.cpp
    ops/matmul.cpp
    ops/gelu.cpp
    ops/softmax.cpp
    ops/attention.cpp
    ops/phase_linear.cpp
    runtime/parallel_for.cpp
    runtime/thread_pool.cpp
    kernels/phase_linear_fused.cpp
    kernels/phase_linear_reference.cpp
    kernels/phase_linear_neon.cpp
    kernels/phase_linear_cache.cpp
    kernels/wide_linear_cache_neon.cpp
)

target_compile_features(rpfq_vit_runtime PRIVATE cxx_std_17)

target_compile_options(rpfq_vit_runtime PRIVATE
    -O3
    -fno-math-errno
)

target_compile_definitions(rpfq_vit_runtime PRIVATE
    RPFQ_VIT_ENABLE_FUSED_FP16_CACHE=1
)

target_link_libraries(
    rpfq_vit_runtime
    android
    log
)
