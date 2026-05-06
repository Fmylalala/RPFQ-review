#include "gelu.h"

#include <cmath>

namespace rpfq_vit {

Tensor gelu(const Tensor& x) {
    Tensor y(x.shape);
    constexpr float kCoeff = 0.7978845608f;
    for (size_t i = 0; i < x.data.size(); ++i) {
        float v = x.data[i];
        float inner = kCoeff * (v + 0.044715f * v * v * v);
        y.data[i] = 0.5f * v * (1.0f + std::tanh(inner));
    }
    return y;
}

} // namespace rpfq_vit
