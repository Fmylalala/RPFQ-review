#include "matmul.h"

#include <stdexcept>

namespace rpfq_vit {

Tensor linear(const Tensor& x, const Tensor& weight, const Tensor& bias) {
    if (x.shape.size() != 2 || weight.shape.size() != 2) {
        throw std::runtime_error("linear expects x=[tokens,in], weight=[out,in].");
    }
    int64_t tokens = x.shape[0];
    int64_t inDim = x.shape[1];
    int64_t outDim = weight.shape[0];
    if (weight.shape[1] != inDim || bias.numel() != outDim) {
        throw std::runtime_error("linear shape mismatch.");
    }
    Tensor y({tokens, outDim});
    for (int64_t t = 0; t < tokens; ++t) {
        for (int64_t o = 0; o < outDim; ++o) {
            float acc = bias.data[o];
            const float* xRow = x.data.data() + t * inDim;
            const float* wRow = weight.data.data() + o * inDim;
            for (int64_t i = 0; i < inDim; ++i) {
                acc += xRow[i] * wRow[i];
            }
            y.data[t * outDim + o] = acc;
        }
    }
    return y;
}

} // namespace rpfq_vit
