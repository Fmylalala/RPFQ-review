#include "wide_linear_cache_neon.h"

#include "phase_linear_cache.h"

#include <algorithm>
#include <stdexcept>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

namespace rpfq_vit {
namespace {

constexpr int kTokenTile = 4;

void validateCacheInput(const Tensor& x, const WideLinearParams& params, const Tensor& y) {
    if (x.shape.size() != 2 || x.shape[1] != params.realInDim) {
        throw std::runtime_error("wideLinearFusedCacheFp16Neon input shape mismatch.");
    }
    if (y.shape.size() != 2 || y.shape[0] != x.shape[0] || y.shape[1] != params.realOutDim) {
        throw std::runtime_error("wideLinearFusedCacheFp16Neon output shape mismatch.");
    }
    if (!params.fusedCache || !params.fusedCache->valid) {
        throw std::runtime_error("wideLinearFusedCacheFp16Neon missing cache.");
    }
    if (params.fusedCache->realInDim != params.realInDim ||
            params.fusedCache->realOutDim != params.realOutDim) {
        throw std::runtime_error("wideLinearFusedCacheFp16Neon cache shape mismatch.");
    }
}

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
float horizontalAdd(float32x4_t v) {
#if defined(__aarch64__)
    return vaddvq_f32(v);
#else
    float tmp[4];
    vst1q_f32(tmp, v);
    return tmp[0] + tmp[1] + tmp[2] + tmp[3];
#endif
}

float32x4_t mulAdd(float32x4_t acc, float32x4_t a, float32x4_t b) {
#if defined(__aarch64__)
    return vfmaq_f32(acc, a, b);
#else
    return vaddq_f32(acc, vmulq_f32(a, b));
#endif
}

float32x4_t loadHalf4(const uint16_t* src, int remaining) {
    float tmp[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int lanes = std::min(4, remaining);
    for (int i = 0; i < lanes; ++i) tmp[i] = halfToFloat(src[i]);
    return vld1q_f32(tmp);
}

float32x4_t loadFloat4(const float* src, int remaining) {
    if (remaining >= 4) return vld1q_f32(src);
    float tmp[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = 0; i < remaining; ++i) tmp[i] = src[i];
    return vld1q_f32(tmp);
}
#endif

} // namespace

void wideLinearFusedCacheFp16Neon(
    const Tensor& x,
    const WideLinearParams& params,
    Tensor& y,
    ThreadPool& pool) {
    validateCacheInput(x, params, y);
    const FusedWideLinearCache& cache = *params.fusedCache;
    int tokens = static_cast<int>(x.shape[0]);
    int inComplex = cache.inComplex;
    int outComplex = cache.outComplex;
    int realInDim = params.realInDim;

    pool.parallelFor(0, outComplex, [&](int64_t rowBegin, int64_t rowEnd) {
        for (int row = static_cast<int>(rowBegin); row < static_cast<int>(rowEnd); ++row) {
            const uint16_t* cxrRe =
                cache.coeffXrToReFp16.data() + static_cast<size_t>(row) * inComplex;
            const uint16_t* cxiRe =
                cache.coeffXiToReFp16.data() + static_cast<size_t>(row) * inComplex;
            const uint16_t* cxrIm =
                cache.coeffXrToImFp16.data() + static_cast<size_t>(row) * inComplex;
            const uint16_t* cxiIm =
                cache.coeffXiToImFp16.data() + static_cast<size_t>(row) * inComplex;

            for (int tokenBase = 0; tokenBase < tokens; tokenBase += kTokenTile) {
                int tokenCount = std::min(kTokenTile, tokens - tokenBase);

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
                float32x4_t accRe[kTokenTile];
                float32x4_t accIm[kTokenTile];
                for (int tt = 0; tt < tokenCount; ++tt) {
                    accRe[tt] = vdupq_n_f32(0.0f);
                    accIm[tt] = vdupq_n_f32(0.0f);
                }

                for (int col = 0; col < inComplex; col += 4) {
                    int remaining = inComplex - col;
                    float32x4_t vCxrRe = loadHalf4(cxrRe + col, remaining);
                    float32x4_t vCxiRe = loadHalf4(cxiRe + col, remaining);
                    float32x4_t vCxrIm = loadHalf4(cxrIm + col, remaining);
                    float32x4_t vCxiIm = loadHalf4(cxiIm + col, remaining);

                    for (int tt = 0; tt < tokenCount; ++tt) {
                        const float* xRow = x.data.data() +
                            static_cast<size_t>(tokenBase + tt) * realInDim;
                        float32x4_t xr = loadFloat4(xRow + col, remaining);
                        float32x4_t xi = loadFloat4(xRow + inComplex + col, remaining);

                        accRe[tt] = mulAdd(accRe[tt], xr, vCxrRe);
                        accRe[tt] = mulAdd(accRe[tt], xi, vCxiRe);
                        accIm[tt] = mulAdd(accIm[tt], xr, vCxrIm);
                        accIm[tt] = mulAdd(accIm[tt], xi, vCxiIm);
                    }
                }

                for (int tt = 0; tt < tokenCount; ++tt) {
                    float* yRow = y.data.data() +
                        static_cast<size_t>(tokenBase + tt) * params.realOutDim;
                    yRow[row] = horizontalAdd(accRe[tt]) + cache.biasRe[row];
                    yRow[params.originalOutComplex + row] =
                        horizontalAdd(accIm[tt]) + cache.biasIm[row];
                }
#else
                float accRe[kTokenTile] = {0.0f, 0.0f, 0.0f, 0.0f};
                float accIm[kTokenTile] = {0.0f, 0.0f, 0.0f, 0.0f};

                for (int col = 0; col < inComplex; ++col) {
                    float vCxrRe = halfToFloat(cxrRe[col]);
                    float vCxiRe = halfToFloat(cxiRe[col]);
                    float vCxrIm = halfToFloat(cxrIm[col]);
                    float vCxiIm = halfToFloat(cxiIm[col]);
                    for (int tt = 0; tt < tokenCount; ++tt) {
                        const float* xRow = x.data.data() +
                            static_cast<size_t>(tokenBase + tt) * realInDim;
                        float xr = xRow[col];
                        float xi = xRow[inComplex + col];
                        accRe[tt] += xr * vCxrRe + xi * vCxiRe;
                        accIm[tt] += xr * vCxrIm + xi * vCxiIm;
                    }
                }

                for (int tt = 0; tt < tokenCount; ++tt) {
                    float* yRow = y.data.data() +
                        static_cast<size_t>(tokenBase + tt) * params.realOutDim;
                    yRow[row] = accRe[tt] + cache.biasRe[row];
                    yRow[params.originalOutComplex + row] = accIm[tt] + cache.biasIm[row];
                }
#endif
            }
        }
    });
}

} // namespace rpfq_vit
