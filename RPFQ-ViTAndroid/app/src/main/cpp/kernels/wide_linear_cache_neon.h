#pragma once

#include "../ops/phase_linear.h"
#include "../runtime/thread_pool.h"
#include "../tensor.h"

namespace rpfq_vit {

void wideLinearFusedCacheFp16Neon(
    const Tensor& x,
    const WideLinearParams& params,
    Tensor& y,
    ThreadPool& pool);

} // namespace rpfq_vit
