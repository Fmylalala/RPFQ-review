#include "phase_linear_cache.h"

#include "../ops/phase_linear.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace rpfq_vit {
namespace {

uint8_t get2BitCode(const uint8_t* packed, int index) {
    uint8_t byte = packed[index >> 2];
    int shift = (index & 3) * 2;
    return static_cast<uint8_t>((byte >> shift) & 0x03);
}

void decodeAt(
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
    wr = 0.0f;
    wi = 0.0f;
    switch (code) {
        case 0: wr = -phase.invGammaReal[gammaIndex]; break;
        case 1: wr = phase.invGammaReal[gammaIndex]; break;
        case 2: wi = -phase.invGammaImag[gammaIndex]; break;
        case 3: wi = phase.invGammaImag[gammaIndex]; break;
    }
}

void validateWideForCache(const WideLinearParams& params) {
    if (!params.bias || params.realInDim <= 0 || params.realOutDim <= 0 ||
            params.complexInDim <= 0 || params.originalOutComplex <= 0) {
        throw std::runtime_error("Invalid WideLinearParams for fused cache.");
    }
    if (params.realInDim % 2 != 0) {
        throw std::runtime_error("Fused cache requires an even real input dim.");
    }
    if (params.realInDim / 2 > params.complexInDim) {
        throw std::runtime_error("Fused cache input dim exceeds packed dim.");
    }
    if (params.originalOutComplex * 2 > params.realOutDim) {
        throw std::runtime_error("Fused cache output dim exceeds real output dim.");
    }
}

} // namespace

uint16_t floatToHalfBits(float x) {
    uint32_t bits;
    std::memcpy(&bits, &x, sizeof(bits));

    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;

    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);
        mant |= 0x800000u;
        int shift = 14 - exp;
        uint32_t halfMant = mant >> shift;
        if ((mant >> (shift - 1)) & 1u) ++halfMant;
        return static_cast<uint16_t>(sign | halfMant);
    }

    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00u);
    }

    uint32_t half = sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13);
    if (mant & 0x00001000u) ++half;
    return static_cast<uint16_t>(half);
}

std::shared_ptr<FusedWideLinearCache> buildFusedWideLinearCache(
    const WideLinearParams& params) {
    validateWideForCache(params);

    auto cache = std::make_shared<FusedWideLinearCache>();
    cache->inComplex = params.realInDim / 2;
    cache->outComplex = params.originalOutComplex;
    cache->realInDim = params.realInDim;
    cache->realOutDim = params.realOutDim;

    size_t coeffCount = static_cast<size_t>(cache->outComplex) * cache->inComplex;
    cache->coeffXrToReFp16.resize(coeffCount);
    cache->coeffXiToReFp16.resize(coeffCount);
    cache->coeffXrToImFp16.resize(coeffCount);
    cache->coeffXiToImFp16.resize(coeffCount);
    cache->biasRe.resize(static_cast<size_t>(cache->outComplex));
    cache->biasIm.resize(static_cast<size_t>(cache->outComplex));

    for (int row = 0; row < cache->outComplex; ++row) {
        cache->biasRe[static_cast<size_t>(row)] = params.bias->data[row];
        cache->biasIm[static_cast<size_t>(row)] =
            params.bias->data[params.originalOutComplex + row];
        for (int block = 0; block < params.uS0.blocksPerRow; ++block) {
            int blockBase = block * params.uS0.blockSize;
            for (int i = 0; i < params.uS0.blockSize; ++i) {
                int col = blockBase + i;
                if (col >= cache->inComplex) continue;

                float u0Re, u0Im, u1Re, u1Im, w0Re, w0Im, w1Re, w1Im;
                decodeAt(params.uS0, row, block, col, u0Re, u0Im);
                decodeAt(params.uS1, row, block, col, u1Re, u1Im);
                decodeAt(params.wS0, row, block, col, w0Re, w0Im);
                decodeAt(params.wS1, row, block, col, w1Re, w1Im);

                size_t idx = static_cast<size_t>(row) * cache->inComplex + col;
                cache->coeffXrToReFp16[idx] =
                    floatToHalfBits(u0Re + u1Re + w0Re + w1Re);
                cache->coeffXiToReFp16[idx] =
                    floatToHalfBits(-u0Im - u1Im + w0Im + w1Im);
                cache->coeffXrToImFp16[idx] =
                    floatToHalfBits(u0Im + u1Im + w0Im + w1Im);
                cache->coeffXiToImFp16[idx] =
                    floatToHalfBits(u0Re + u1Re - w0Re - w1Re);
            }
        }
    }

    cache->valid = true;
    return cache;
}

} // namespace rpfq_vit
