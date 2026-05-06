#pragma once

#include "../safetensors_loader.h"
#include "../tensor.h"

#include <memory>
#include <string>
#include <vector>

namespace rpfq_vit {

struct FusedWideLinearCache;
class ThreadPool;

struct PhaseLinearParams {
    const UInt8Tensor* codes = nullptr;
    const Tensor* gammaReal = nullptr;
    const Tensor* gammaImag = nullptr;
    std::vector<float> invGammaReal;
    std::vector<float> invGammaImag;
    int inDim = 0;
    int outDim = 0;
    int blockSize = 256;
    int blocksPerRow = 0;
    int packedPerRow = 0;
};

struct WideLinearParams {
    PhaseLinearParams uS0;
    PhaseLinearParams uS1;
    PhaseLinearParams wS0;
    PhaseLinearParams wS1;
    const Tensor* bias = nullptr;
    int realInDim = 0;
    int realOutDim = 0;
    int complexInDim = 0;
    int complexOutDim = 0;
    int originalOutComplex = 0;
    std::shared_ptr<FusedWideLinearCache> fusedCache;
};

PhaseLinearParams loadPhaseParams(
    const SafeTensorStore& store,
    const std::string& prefix,
    int inDim,
    int outDim,
    int blockSize);

WideLinearParams loadWideParams(
    const SafeTensorStore& store,
    const std::string& prefix,
    int realInDim,
    int realOutDim,
    int complexInDim,
    int complexOutDim,
    int originalOutComplex,
    int blockSize);

void phaseLinearReference(
    const float* xRe,
    const float* xIm,
    int tokens,
    const PhaseLinearParams& params,
    std::vector<float>& outRe,
    std::vector<float>& outIm);

Tensor wideLinearReference(const Tensor& x, const WideLinearParams& params);
Tensor wideLinear(const Tensor& x, const WideLinearParams& params, ThreadPool* pool = nullptr);

} // namespace rpfq_vit
