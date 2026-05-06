#pragma once

#include "../tensor.h"

namespace rpfq_vit {

Tensor attentionForward(
    const Tensor& qkv,
    int hiddenSize,
    int numHeads,
    Tensor* attnOutPreProj = nullptr);

} // namespace rpfq_vit
