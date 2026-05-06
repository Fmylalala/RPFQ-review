#pragma once

#include "../ops/phase_linear.h"
#include "../tensor.h"

namespace rpfq_vit {

void wideLinearFusedScalarRange(
    const Tensor& x,
    const WideLinearParams& params,
    Tensor& y,
    int rowBegin,
    int rowEnd);

bool wideLinearFusedNeonRange(
    const Tensor& x,
    const WideLinearParams& params,
    Tensor& y,
    int rowBegin,
    int rowEnd);

Tensor wideLinearOptimized(
    const Tensor& x,
    const WideLinearParams& params,
    int numThreads = 0);

} // namespace rpfq_vit
