#include "softmax.h"

#include <algorithm>
#include <cmath>

namespace rpfq_vit {

void softmaxInPlace(std::vector<float>& values) {
    if (values.empty()) return;
    float maxValue = *std::max_element(values.begin(), values.end());
    float sum = 0.0f;
    for (float& v : values) {
        v = std::exp(v - maxValue);
        sum += v;
    }
    if (sum == 0.0f) return;
    for (float& v : values) v /= sum;
}

} // namespace rpfq_vit
