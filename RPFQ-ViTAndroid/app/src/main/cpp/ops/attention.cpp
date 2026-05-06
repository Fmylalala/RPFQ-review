#include "attention.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace rpfq_vit {

Tensor attentionForward(
    const Tensor& qkv,
    int hiddenSize,
    int numHeads,
    Tensor* attnOutPreProj) {
    if (qkv.shape.size() != 2 || qkv.shape[1] != hiddenSize * 3) {
        throw std::runtime_error("attentionForward expects [tokens, hidden*3].");
    }
    int tokens = static_cast<int>(qkv.shape[0]);
    int headDim = hiddenSize / numHeads;
    float scale = 1.0f / std::sqrt(static_cast<float>(headDim));
    Tensor out({tokens, hiddenSize});

    std::vector<float> scores(tokens);
    for (int h = 0; h < numHeads; ++h) {
        for (int tq = 0; tq < tokens; ++tq) {
            const float* q = qkv.data.data()
                + static_cast<size_t>(tq) * hiddenSize * 3
                + h * headDim;
            float maxScore = -INFINITY;
            for (int tk = 0; tk < tokens; ++tk) {
                const float* k = qkv.data.data()
                    + static_cast<size_t>(tk) * hiddenSize * 3
                    + hiddenSize
                    + h * headDim;
                float dot = 0.0f;
                for (int d = 0; d < headDim; ++d) dot += q[d] * k[d];
                scores[tk] = dot * scale;
                maxScore = std::max(maxScore, scores[tk]);
            }

            float denom = 0.0f;
            for (int tk = 0; tk < tokens; ++tk) {
                scores[tk] = std::exp(scores[tk] - maxScore);
                denom += scores[tk];
            }
            if (denom == 0.0f) denom = 1.0f;

            float* outHead = out.data.data() + static_cast<size_t>(tq) * hiddenSize + h * headDim;
            for (int d = 0; d < headDim; ++d) outHead[d] = 0.0f;
            for (int tk = 0; tk < tokens; ++tk) {
                float prob = scores[tk] / denom;
                const float* v = qkv.data.data()
                    + static_cast<size_t>(tk) * hiddenSize * 3
                    + hiddenSize * 2
                    + h * headDim;
                for (int d = 0; d < headDim; ++d) {
                    outHead[d] += prob * v[d];
                }
            }
        }
    }

    if (attnOutPreProj) {
        *attnOutPreProj = out;
    }
    return out;
}

} // namespace rpfq_vit
