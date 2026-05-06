#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace rpfq_vit {

struct WideLinearParams;

struct FusedWideLinearCache {
    int inComplex = 0;
    int outComplex = 0;
    int realInDim = 0;
    int realOutDim = 0;

    std::vector<uint16_t> coeffXrToReFp16;
    std::vector<uint16_t> coeffXiToReFp16;
    std::vector<uint16_t> coeffXrToImFp16;
    std::vector<uint16_t> coeffXiToImFp16;

    std::vector<float> biasRe;
    std::vector<float> biasIm;

    bool valid = false;
};

uint16_t floatToHalfBits(float x);

std::shared_ptr<FusedWideLinearCache> buildFusedWideLinearCache(
    const WideLinearParams& params);

} // namespace rpfq_vit
