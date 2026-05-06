#pragma once

#include "../ops/phase_linear.h"
#include "../tensor.h"

namespace rpfq_vit {

bool wideLinearFusedNeonRangeImpl(
    const Tensor& x,
    const WideLinearParams& params,
    Tensor& y,
    int rowBegin,
    int rowEnd);

} // namespace rpfq_vit
