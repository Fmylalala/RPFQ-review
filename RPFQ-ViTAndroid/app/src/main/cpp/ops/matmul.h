#pragma once

#include "../tensor.h"

namespace rpfq_vit {

Tensor linear(const Tensor& x, const Tensor& weight, const Tensor& bias);

} // namespace rpfq_vit
