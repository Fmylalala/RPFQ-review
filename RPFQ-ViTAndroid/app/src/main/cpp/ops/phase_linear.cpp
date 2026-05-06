#include "phase_linear.h"

#include "../kernels/phase_linear_cache.h"
#include "../kernels/phase_linear_fused.h"
#include "../kernels/wide_linear_cache_neon.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace rpfq_vit {
namespace {

uint8_t get2BitCode(const uint8_t* packed, int index) {
    uint8_t byte = packed[index >> 2];
    int shift = (index & 3) * 2;
    return static_cast<uint8_t>((byte >> shift) & 0x03);
}

void validatePhase(const PhaseLinearParams& p) {
    if (!p.codes || !p.gammaReal || !p.gammaImag) {
        throw std::runtime_error("PhaseLinearParams contains null tensors.");
    }
    if (p.inDim <= 0 || p.outDim <= 0 || p.blockSize <= 0) {
        throw std::runtime_error("PhaseLinearParams contains invalid dims.");
    }
    if (p.inDim % p.blockSize != 0 || p.inDim % 4 != 0) {
        throw std::runtime_error("PhaseLinearParams requires packed aligned dims.");
    }
    if (p.blocksPerRow <= 0 || p.packedPerRow <= 0) {
        throw std::runtime_error("PhaseLinearParams has missing precomputed dims.");
    }
    size_t gammaCount = static_cast<size_t>(p.outDim) * p.blocksPerRow;
    if (p.invGammaReal.size() != gammaCount || p.invGammaImag.size() != gammaCount) {
        throw std::runtime_error("PhaseLinearParams has missing reciprocal gamma.");
    }
}

} // namespace

PhaseLinearParams loadPhaseParams(
    const SafeTensorStore& store,
    const std::string& prefix,
    int inDim,
    int outDim,
    int blockSize) {
    PhaseLinearParams p;
    p.codes = &store.bytes(prefix + ".codes");
    p.gammaReal = &store.tensor(prefix + ".gamma_real");
    p.gammaImag = &store.tensor(prefix + ".gamma_imag");
    p.inDim = inDim;
    p.outDim = outDim;
    p.blockSize = blockSize;
    if (p.inDim <= 0 || p.outDim <= 0 || p.blockSize <= 0) {
        throw std::runtime_error("PhaseLinearParams contains invalid dims.");
    }
    if (p.inDim % p.blockSize != 0 || p.inDim % 4 != 0) {
        throw std::runtime_error("PhaseLinearParams requires packed aligned dims for " + prefix);
    }
    p.blocksPerRow = p.inDim / p.blockSize;
    p.packedPerRow = p.inDim / 4;
    if (p.codes->dim(0) != outDim || p.codes->dim(1) != inDim / 4) {
        throw std::runtime_error("Packed code shape mismatch for " + prefix);
    }
    size_t gammaCount = static_cast<size_t>(outDim) * p.blocksPerRow;
    if (p.gammaReal->dim(0) != outDim || p.gammaImag->dim(0) != outDim ||
            p.gammaReal->numel() != static_cast<int64_t>(gammaCount) ||
            p.gammaImag->numel() != static_cast<int64_t>(gammaCount)) {
        throw std::runtime_error("Gamma row shape mismatch for " + prefix);
    }
    p.invGammaReal.resize(gammaCount);
    p.invGammaImag.resize(gammaCount);
    for (int row = 0; row < outDim; ++row) {
        for (int block = 0; block < p.blocksPerRow; ++block) {
            size_t idx = static_cast<size_t>(row) * p.blocksPerRow + block;
            float gr = p.gammaReal->data[idx];
            float gi = p.gammaImag->data[idx];
            if (gr == 0.0f || gi == 0.0f) {
                throw std::runtime_error("Zero gamma found in " + prefix);
            }
            p.invGammaReal[idx] = 1.0f / gr;
            p.invGammaImag[idx] = 1.0f / gi;
        }
    }
    validatePhase(p);
    return p;
}

WideLinearParams loadWideParams(
    const SafeTensorStore& store,
    const std::string& prefix,
    int realInDim,
    int realOutDim,
    int complexInDim,
    int complexOutDim,
    int originalOutComplex,
    int blockSize) {
    WideLinearParams p;
    p.uS0 = loadPhaseParams(store, prefix + ".u_s0", complexInDim, complexOutDim, blockSize);
    p.uS1 = loadPhaseParams(store, prefix + ".u_s1", complexInDim, complexOutDim, blockSize);
    p.wS0 = loadPhaseParams(store, prefix + ".w_s0", complexInDim, complexOutDim, blockSize);
    p.wS1 = loadPhaseParams(store, prefix + ".w_s1", complexInDim, complexOutDim, blockSize);
    p.bias = &store.tensor(prefix + ".bias");
    p.realInDim = realInDim;
    p.realOutDim = realOutDim;
    p.complexInDim = complexInDim;
    p.complexOutDim = complexOutDim;
    p.originalOutComplex = originalOutComplex;
    if (p.bias->numel() != realOutDim) {
        throw std::runtime_error("WideLinear bias shape mismatch for " + prefix);
    }
#if defined(RPFQ_VIT_ENABLE_FUSED_FP16_CACHE)
    p.fusedCache = buildFusedWideLinearCache(p);
#endif
    return p;
}

void phaseLinearReference(
    const float* xRe,
    const float* xIm,
    int tokens,
    const PhaseLinearParams& params,
    std::vector<float>& outRe,
    std::vector<float>& outIm) {
    validatePhase(params);
    outRe.assign(static_cast<size_t>(tokens) * params.outDim, 0.0f);
    outIm.assign(static_cast<size_t>(tokens) * params.outDim, 0.0f);

    for (int tok = 0; tok < tokens; ++tok) {
        const float* xrTok = xRe + static_cast<size_t>(tok) * params.inDim;
        const float* xiTok = xIm + static_cast<size_t>(tok) * params.inDim;
        for (int row = 0; row < params.outDim; ++row) {
            const uint8_t* codeRow = params.codes->data.data() + static_cast<size_t>(row) * params.packedPerRow;
            const float* invGrRow = params.invGammaReal.data() + static_cast<size_t>(row) * params.blocksPerRow;
            const float* invGiRow = params.invGammaImag.data() + static_cast<size_t>(row) * params.blocksPerRow;
            float accRe = 0.0f;
            float accIm = 0.0f;
            for (int block = 0; block < params.blocksPerRow; ++block) {
                float invGr = invGrRow[block];
                float invGi = invGiRow[block];
                int blockBase = block * params.blockSize;
                for (int i = 0; i < params.blockSize; ++i) {
                    int col = blockBase + i;
                    uint8_t code = get2BitCode(codeRow, col);
                    float wRe = 0.0f;
                    float wIm = 0.0f;
                    switch (code) {
                        case 0: wRe = -invGr; break;
                        case 1: wRe = invGr; break;
                        case 2: wIm = -invGi; break;
                        case 3: wIm = invGi; break;
                    }
                    float xr = xrTok[col];
                    float xi = xiTok[col];
                    accRe += xr * wRe + xi * wIm;
                    accIm += xr * wIm - xi * wRe;
                }
            }
            outRe[static_cast<size_t>(tok) * params.outDim + row] = accRe;
            outIm[static_cast<size_t>(tok) * params.outDim + row] = accIm;
        }
    }
}

Tensor wideLinearReference(const Tensor& x, const WideLinearParams& params) {
    if (x.shape.size() != 2 || x.shape[1] != params.realInDim) {
        throw std::runtime_error("wideLinearReference input shape mismatch.");
    }
    int tokens = static_cast<int>(x.shape[0]);
    int halfIn = params.realInDim / 2;

    std::vector<float> xRe(static_cast<size_t>(tokens) * params.complexInDim, 0.0f);
    std::vector<float> xIm(static_cast<size_t>(tokens) * params.complexInDim, 0.0f);
    std::vector<float> negXIm(static_cast<size_t>(tokens) * params.complexInDim, 0.0f);
    for (int t = 0; t < tokens; ++t) {
        const float* src = x.data.data() + static_cast<size_t>(t) * params.realInDim;
        float* dstRe = xRe.data() + static_cast<size_t>(t) * params.complexInDim;
        float* dstIm = xIm.data() + static_cast<size_t>(t) * params.complexInDim;
        float* dstNeg = negXIm.data() + static_cast<size_t>(t) * params.complexInDim;
        for (int i = 0; i < halfIn; ++i) {
            dstRe[i] = src[i];
            dstIm[i] = src[halfIn + i];
            dstNeg[i] = -src[halfIn + i];
        }
    }

    std::vector<float> u0Re, u0Im, u1Re, u1Im, w0Re, w0Im, w1Re, w1Im;
    phaseLinearReference(xRe.data(), negXIm.data(), tokens, params.uS0, u0Re, u0Im);
    phaseLinearReference(xRe.data(), negXIm.data(), tokens, params.uS1, u1Re, u1Im);
    phaseLinearReference(xRe.data(), xIm.data(), tokens, params.wS0, w0Re, w0Im);
    phaseLinearReference(xRe.data(), xIm.data(), tokens, params.wS1, w1Re, w1Im);

    Tensor y({tokens, params.realOutDim});
    for (int t = 0; t < tokens; ++t) {
        float* out = y.data.data() + static_cast<size_t>(t) * params.realOutDim;
        for (int i = 0; i < params.originalOutComplex; ++i) {
            size_t idx = static_cast<size_t>(t) * params.complexOutDim + i;
            out[i] = u0Re[idx] + u1Re[idx] + w0Re[idx] + w1Re[idx];
            out[params.originalOutComplex + i] = u0Im[idx] + u1Im[idx] + w0Im[idx] + w1Im[idx];
        }
        for (int i = 0; i < params.realOutDim; ++i) {
            out[i] += params.bias->data[i];
        }
    }
    return y;
}

Tensor wideLinear(const Tensor& x, const WideLinearParams& params, ThreadPool* pool) {
#if defined(RPFQ_VIT_FORCE_REFERENCE_PHASE_LINEAR)
    (void)pool;
    return wideLinearReference(x, params);
#elif defined(RPFQ_VIT_CHECK_PHASE_LINEAR_PARITY)
    Tensor ref = wideLinearReference(x, params);
    Tensor opt;
    if (pool && params.fusedCache && params.fusedCache->valid) {
        opt = Tensor({x.shape[0], params.realOutDim});
        wideLinearFusedCacheFp16Neon(x, params, opt, *pool);
    } else {
        opt = wideLinearOptimized(x, params);
    }
    if (ref.shape != opt.shape || ref.data.size() != opt.data.size()) {
        throw std::runtime_error("wideLinear parity shape mismatch.");
    }
    float maxAbsDiff = 0.0f;
    double totalAbsDiff = 0.0;
    size_t maxIndex = 0;
    for (size_t i = 0; i < ref.data.size(); ++i) {
        float diff = std::abs(ref.data[i] - opt.data[i]);
        if (diff > maxAbsDiff) {
            maxAbsDiff = diff;
            maxIndex = i;
        }
        totalAbsDiff += diff;
    }
    double meanAbsDiff = ref.data.empty() ? 0.0 : totalAbsDiff / ref.data.size();
    if (maxAbsDiff > 1e-3f) {
        std::ostringstream ss;
        ss << "wideLinear parity check failed: max_abs_diff=" << maxAbsDiff
           << " mean_abs_diff=" << meanAbsDiff
           << " index=" << maxIndex
           << " ref=" << ref.data[maxIndex]
           << " opt=" << opt.data[maxIndex];
        throw std::runtime_error(ss.str());
    }
    return opt;
#else
    if (pool && params.fusedCache && params.fusedCache->valid) {
        Tensor y({x.shape[0], params.realOutDim});
        wideLinearFusedCacheFp16Neon(x, params, y, *pool);
        return y;
    }
    (void)pool;
    return wideLinearOptimized(x, params);
#endif
}

} // namespace rpfq_vit
