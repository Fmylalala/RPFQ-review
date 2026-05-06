#pragma once

#include "../tensor.h"

namespace rpfq_vit {

Tensor layerNorm(const Tensor& x, const Tensor& weight, const Tensor& bias, float eps);

} // namespace rpfq_vit
