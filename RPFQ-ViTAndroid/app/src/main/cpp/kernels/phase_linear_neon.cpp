#include "phase_linear_neon.h"

#include <algorithm>
#include <cstdint>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

namespace rpfq_vit {

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
namespace {

inline void decodePackedByte(
    uint8_t byte,
    float invGr,
    float invGi,
    float* wr,
    float* wi) {
    for (int lane = 0; lane < 4; ++lane) {
        wr[lane] = 0.0f;
        wi[lane] = 0.0f;
        uint8_t code = static_cast<uint8_t>((byte >> (lane * 2)) & 0x03);
        switch (code) {
            case 0: wr[lane] = -invGr; break;
            case 1: wr[lane] = invGr; break;
            case 2: wi[lane] = -invGi; break;
            case 3: wi[lane] = invGi; break;
        }
    }
}

inline float horizontalAdd(float32x4_t v) {
#if defined(__aarch64__)
    return vaddvq_f32(v);
#else
    float tmp[4];
    vst1q_f32(tmp, v);
    return tmp[0] + tmp[1] + tmp[2] + tmp[3];
#endif
}

inline float32x4_t mulAdd(float32x4_t acc, float32x4_t a, float32x4_t b) {
#if defined(__aarch64__)
    return vfmaq_f32(acc, a, b);
#else
    return vaddq_f32(acc, vmulq_f32(a, b));
#endif
}

inline void decodeStageVectors(
    const PhaseLinearParams& phase,
    int row,
    int block,
    int col,
    float32x4_t& wrv,
    float32x4_t& wiv) {
    const uint8_t* codeRow =
        phase.codes->data.data() + static_cast<size_t>(row) * phase.packedPerRow;
    size_t gammaIndex = static_cast<size_t>(row) * phase.blocksPerRow + block;
    float wr[4];
    float wi[4];
    decodePackedByte(
        codeRow[col >> 2],
        phase.invGammaReal[gammaIndex],
        phase.invGammaImag[gammaIndex],
        wr,
        wi);
    wrv = vld1q_f32(wr);
    wiv = vld1q_f32(wi);
}

inline void loadInputVectors(
    const float* xRow,
    int halfIn,
    int col,
    float32x4_t& xrv,
    float32x4_t& xiv) {
    if (col + 3 < halfIn) {
        xrv = vld1q_f32(xRow + col);
        xiv = vld1q_f32(xRow + halfIn + col);
        return;
    }

    float xr[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float xi[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int lane = 0; lane < 4; ++lane) {
        int idx = col + lane;
        if (idx < halfIn) {
            xr[lane] = xRow[idx];
            xi[lane] = xRow[halfIn + idx];
        }
    }
    xrv = vld1q_f32(xr);
    xiv = vld1q_f32(xi);
}

} // namespace

bool wideLinearFusedNeonRangeImpl(
    const Tensor& x,
    const WideLinearParams& params,
    Tensor& y,
    int rowBegin,
    int rowEnd) {
    if (x.shape.size() != 2 || x.shape[1] != params.realInDim) {
        return false;
    }
    if (params.uS0.blockSize % 4 != 0 || params.uS0.inDim % 4 != 0) {
        return false;
    }

    int tokens = static_cast<int>(x.shape[0]);
    int halfIn = params.realInDim / 2;
    rowBegin = std::max(0, rowBegin);
    rowEnd = std::min(rowEnd, params.originalOutComplex);

    for (int t = 0; t < tokens; ++t) {
        const float* xRow = x.data.data() + static_cast<size_t>(t) * params.realInDim;
        float* yRow = y.data.data() + static_cast<size_t>(t) * params.realOutDim;
        for (int row = rowBegin; row < rowEnd; ++row) {
            float32x4_t accRe = vdupq_n_f32(0.0f);
            float32x4_t accIm = vdupq_n_f32(0.0f);

            for (int block = 0; block < params.uS0.blocksPerRow; ++block) {
                int blockBase = block * params.uS0.blockSize;
                for (int i = 0; i < params.uS0.blockSize; i += 4) {
                    int col = blockBase + i;
                    float32x4_t u0Wr, u0Wi, u1Wr, u1Wi, w0Wr, w0Wi, w1Wr, w1Wi;
                    decodeStageVectors(params.uS0, row, block, col, u0Wr, u0Wi);
                    decodeStageVectors(params.uS1, row, block, col, u1Wr, u1Wi);
                    decodeStageVectors(params.wS0, row, block, col, w0Wr, w0Wi);
                    decodeStageVectors(params.wS1, row, block, col, w1Wr, w1Wi);

                    float32x4_t coeffXrToRe =
                        vaddq_f32(vaddq_f32(u0Wr, u1Wr), vaddq_f32(w0Wr, w1Wr));
                    float32x4_t coeffXiToRe =
                        vaddq_f32(vnegq_f32(vaddq_f32(u0Wi, u1Wi)), vaddq_f32(w0Wi, w1Wi));
                    float32x4_t coeffXrToIm =
                        vaddq_f32(vaddq_f32(u0Wi, u1Wi), vaddq_f32(w0Wi, w1Wi));
                    float32x4_t coeffXiToIm =
                        vsubq_f32(vaddq_f32(u0Wr, u1Wr), vaddq_f32(w0Wr, w1Wr));

                    float32x4_t xrv, xiv;
                    loadInputVectors(xRow, halfIn, col, xrv, xiv);

                    accRe = mulAdd(accRe, xrv, coeffXrToRe);
                    accRe = mulAdd(accRe, xiv, coeffXiToRe);
                    accIm = mulAdd(accIm, xrv, coeffXrToIm);
                    accIm = mulAdd(accIm, xiv, coeffXiToIm);
                }
            }

            yRow[row] = horizontalAdd(accRe) + params.bias->data[row];
            yRow[params.originalOutComplex + row] =
                horizontalAdd(accIm) + params.bias->data[params.originalOutComplex + row];
        }
    }
    return true;
}

#else

bool wideLinearFusedNeonRangeImpl(
    const Tensor&,
    const WideLinearParams&,
    Tensor&,
    int,
    int) {
    return false;
}

#endif

} // namespace rpfq_vit
