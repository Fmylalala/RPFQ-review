#include "phase_linear_fused.h"

#include "phase_linear_neon.h"
#include "../runtime/parallel_for.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace rpfq_vit {
namespace {

inline uint8_t get2BitCode(const uint8_t* packed, int index) {
    uint8_t byte = packed[index >> 2];
    int shift = (index & 3) * 2;
    return static_cast<uint8_t>((byte >> shift) & 0x03);
}

inline void decodeWeight(
    uint8_t code,
    float invGr,
    float invGi,
    float& wr,
    float& wi) {
    wr = 0.0f;
    wi = 0.0f;
    switch (code) {
        case 0: wr = -invGr; break;
        case 1: wr = invGr; break;
        case 2: wi = -invGi; break;
        case 3: wi = invGi; break;
    }
}

inline void decodeAt(
    const PhaseLinearParams& phase,
    int row,
    int block,
    int col,
    float& wr,
    float& wi) {
    const uint8_t* codeRow =
        phase.codes->data.data() + static_cast<size_t>(row) * phase.packedPerRow;
    size_t gammaIndex = static_cast<size_t>(row) * phase.blocksPerRow + block;
    uint8_t code = get2BitCode(codeRow, col);
    decodeWeight(code, phase.invGammaReal[gammaIndex], phase.invGammaImag[gammaIndex], wr, wi);
}

void validateWideInput(const Tensor& x, const WideLinearParams& params, const char* name) {
    if (x.shape.size() != 2 || x.shape[1] != params.realInDim) {
        throw std::runtime_error(std::string(name) + " input shape mismatch.");
    }
    if (!params.bias || params.bias->numel() != params.realOutDim) {
        throw std::runtime_error(std::string(name) + " bias shape mismatch.");
    }
}

} // namespace

void wideLinearFusedScalarRange(
    const Tensor& x,
    const WideLinearParams& params,
    Tensor& y,
    int rowBegin,
    int rowEnd) {
    validateWideInput(x, params, "wideLinearFusedScalarRange");
    int tokens = static_cast<int>(x.shape[0]);
    int halfIn = params.realInDim / 2;
    int rowLimit = params.originalOutComplex;
    rowBegin = std::max(0, rowBegin);
    rowEnd = std::min(rowEnd, rowLimit);

    for (int t = 0; t < tokens; ++t) {
        const float* xRow = x.data.data() + static_cast<size_t>(t) * params.realInDim;
        float* yRow = y.data.data() + static_cast<size_t>(t) * params.realOutDim;
        for (int row = rowBegin; row < rowEnd; ++row) {
            float accRe = 0.0f;
            float accIm = 0.0f;

            for (int block = 0; block < params.uS0.blocksPerRow; ++block) {
                int blockBase = block * params.uS0.blockSize;
                for (int i = 0; i < params.uS0.blockSize; ++i) {
                    int col = blockBase + i;
                    float xr = col < halfIn ? xRow[col] : 0.0f;
                    float xi = col < halfIn ? xRow[halfIn + col] : 0.0f;

                    float u0Re, u0Im, u1Re, u1Im, w0Re, w0Im, w1Re, w1Im;
                    decodeAt(params.uS0, row, block, col, u0Re, u0Im);
                    decodeAt(params.uS1, row, block, col, u1Re, u1Im);
                    decodeAt(params.wS0, row, block, col, w0Re, w0Im);
                    decodeAt(params.wS1, row, block, col, w1Re, w1Im);

                    float coeffXrToRe = u0Re + u1Re + w0Re + w1Re;
                    float coeffXiToRe = -u0Im - u1Im + w0Im + w1Im;
                    float coeffXrToIm = u0Im + u1Im + w0Im + w1Im;
                    float coeffXiToIm = u0Re + u1Re - w0Re - w1Re;

                    accRe += xr * coeffXrToRe + xi * coeffXiToRe;
                    accIm += xr * coeffXrToIm + xi * coeffXiToIm;
                }
            }

            yRow[row] = accRe + params.bias->data[row];
            yRow[params.originalOutComplex + row] =
                accIm + params.bias->data[params.originalOutComplex + row];
        }
    }
}

bool wideLinearFusedNeonRange(
    const Tensor& x,
    const WideLinearParams& params,
    Tensor& y,
    int rowBegin,
    int rowEnd) {
    return wideLinearFusedNeonRangeImpl(x, params, y, rowBegin, rowEnd);
}

Tensor wideLinearOptimized(
    const Tensor& x,
    const WideLinearParams& params,
    int numThreads) {
    validateWideInput(x, params, "wideLinearOptimized");
    int tokens = static_cast<int>(x.shape[0]);
    Tensor y({tokens, params.realOutDim});

    int rows = params.originalOutComplex;
    int threads = numThreads > 0 ? numThreads : defaultThreadCount();
    parallelFor(0, rows, threads, [&](int64_t rb, int64_t re) {
        bool ok = wideLinearFusedNeonRange(
            x,
            params,
            y,
            static_cast<int>(rb),
            static_cast<int>(re));
        if (!ok) {
            wideLinearFusedScalarRange(
                x,
                params,
                y,
                static_cast<int>(rb),
                static_cast<int>(re));
        }
    });

    return y;
}

} // namespace rpfq_vit
