#include "layernorm.h"

#include <cmath>
#include <stdexcept>

namespace rpfq_vit {

Tensor layerNorm(const Tensor& x, const Tensor& weight, const Tensor& bias, float eps) {
    if (x.shape.size() != 2) throw std::runtime_error("layerNorm expects [tokens, dims].");
    int64_t tokens = x.shape[0];
    int64_t dims = x.shape[1];
    if (weight.numel() != dims || bias.numel() != dims) {
        throw std::runtime_error("layerNorm weight/bias shape mismatch.");
    }
    Tensor y({tokens, dims});
    for (int64_t t = 0; t < tokens; ++t) {
        const float* row = x.data.data() + t * dims;
        float mean = 0.0f;
        for (int64_t i = 0; i < dims; ++i) mean += row[i];
        mean /= static_cast<float>(dims);

        float var = 0.0f;
        for (int64_t i = 0; i < dims; ++i) {
            float d = row[i] - mean;
            var += d * d;
        }
        var /= static_cast<float>(dims);
        float inv = 1.0f / std::sqrt(var + eps);

        float* out = y.data.data() + t * dims;
        for (int64_t i = 0; i < dims; ++i) {
            out[i] = (row[i] - mean) * inv * weight.data[i] + bias.data[i];
        }
    }
    return y;
}

} // namespace rpfq_vit
