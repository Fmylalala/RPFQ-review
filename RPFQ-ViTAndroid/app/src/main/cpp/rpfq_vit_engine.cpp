#include "rpfq_vit_engine.h"

#include "ops/attention.h"
#include "ops/gelu.h"
#include "ops/layernorm.h"
#include "ops/matmul.h"

#include <android/log.h>
#include <algorithm>
#include <chrono>
#include <stdexcept>

#define RPFQ_VIT_LOGI(...) __android_log_print(ANDROID_LOG_INFO, "RPFQViTEngine", __VA_ARGS__)

namespace rpfq_vit {
namespace {

Tensor addSameShape(const Tensor& a, const Tensor& b) {
    if (a.shape != b.shape) throw std::runtime_error("addSameShape shape mismatch.");
    Tensor y(a.shape);
    for (size_t i = 0; i < a.data.size(); ++i) y.data[i] = a.data[i] + b.data[i];
    return y;
}

std::string layerPrefix(int layer) {
    return "model.layers." + std::to_string(layer);
}

using Clock = std::chrono::steady_clock;

int64_t elapsedMs(Clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start).count();
}

} // namespace

bool RPFQViTEngine::loadBundle(const std::string& bundleDir) {
    bundle_ = loadBundleData(bundleDir);
    buildParams();
    threadPool_ = std::make_unique<ThreadPool>();
    RPFQ_VIT_LOGI("TIMING thread_pool threads=%d", threadPool_->numThreads());
    loaded_ = true;
    return true;
}

void RPFQViTEngine::setDebugEnabled(bool enabled) {
    debug_.configure(enabled, "");
}

void RPFQViTEngine::setDebugOutputDir(const std::string& outputDir) {
    debug_.configure(debug_.enabled(), outputDir);
}

std::vector<float> RPFQViTEngine::predict(const std::vector<float>& inputNchw) {
    if (!loaded_) throw std::runtime_error("Model bundle is not loaded.");

    auto predictStart = Clock::now();
    Tensor input = inputTensor(inputNchw);
    debug_.writeTensor("input_tensor", input);

    auto stageStart = Clock::now();
    Tensor patches = patchify(input);
    RPFQ_VIT_LOGI("TIMING stage=patchify ms=%lld", static_cast<long long>(elapsedMs(stageStart)));

    stageStart = Clock::now();
    Tensor embedded = patchEmbed(patches);
    RPFQ_VIT_LOGI("TIMING stage=patch_embed ms=%lld", static_cast<long long>(elapsedMs(stageStart)));
    debug_.writeTensor("patch_embed", embedded);

    stageStart = Clock::now();
    Tensor x = addClassAndPosition(embedded);
    RPFQ_VIT_LOGI("TIMING stage=add_class_position ms=%lld", static_cast<long long>(elapsedMs(stageStart)));
    for (int i = 0; i < bundle_.config.numHiddenLayers; ++i) {
        auto blockStart = Clock::now();
        x = forwardBlock(i, x);
        RPFQ_VIT_LOGI("TIMING block=%d stage=total ms=%lld", i, static_cast<long long>(elapsedMs(blockStart)));
    }

    const Tensor& normWeight = bundle_.tensors.tensor("model.norm.weight");
    const Tensor& normBias = bundle_.tensors.tensor("model.norm.bias");
    stageStart = Clock::now();
    Tensor finalNorm = layerNorm(x, normWeight, normBias, bundle_.config.layerNormEps);
    RPFQ_VIT_LOGI("TIMING stage=final_norm ms=%lld", static_cast<long long>(elapsedMs(stageStart)));
    debug_.writeTensor("final_norm", finalNorm);

    stageStart = Clock::now();
    Tensor logits = forwardHead(finalNorm);
    RPFQ_VIT_LOGI("TIMING stage=head ms=%lld", static_cast<long long>(elapsedMs(stageStart)));
    RPFQ_VIT_LOGI("TIMING total_predict_ms=%lld", static_cast<long long>(elapsedMs(predictStart)));
    debug_.writeTensor("logits", logits);
    return logits.data;
}

Tensor RPFQViTEngine::inputTensor(const std::vector<float>& inputNchw) const {
    int expected = bundle_.config.numChannels * bundle_.config.imageSize * bundle_.config.imageSize;
    if (static_cast<int>(inputNchw.size()) != expected) {
        throw std::runtime_error("Input tensor size mismatch.");
    }
    return Tensor(
        {1, bundle_.config.numChannels, bundle_.config.imageSize, bundle_.config.imageSize},
        inputNchw);
}

Tensor RPFQViTEngine::patchify(const Tensor& nchw) const {
    const ModelConfig& cfg = bundle_.config;
    int image = cfg.imageSize;
    int patch = cfg.patchSize;
    int channels = cfg.numChannels;
    int grid = image / patch;
    int patchArea = patch * patch;
    int patchVector = patchArea * channels;
    Tensor out({grid * grid, patchVector});

    const float* src = nchw.data.data();
    for (int py = 0; py < grid; ++py) {
        for (int px = 0; px < grid; ++px) {
            int patchIndex = py * grid + px;
            float* dst = out.data.data() + static_cast<size_t>(patchIndex) * patchVector;
            for (int c = 0; c < channels; ++c) {
                for (int dy = 0; dy < patch; ++dy) {
                    for (int dx = 0; dx < patch; ++dx) {
                        int srcY = py * patch + dy;
                        int srcX = px * patch + dx;
                        size_t srcIndex = static_cast<size_t>(c) * image * image + srcY * image + srcX;
                        int dstIndex = c * patchArea + dy * patch + dx;
                        dst[dstIndex] = src[srcIndex];
                    }
                }
            }
        }
    }
    return out;
}

Tensor RPFQViTEngine::patchEmbed(const Tensor& patches) const {
    return linear(
        patches,
        bundle_.tensors.tensor("model.patch_embed.weight"),
        bundle_.tensors.tensor("model.patch_embed.bias"));
}

Tensor RPFQViTEngine::addClassAndPosition(const Tensor& patchEmbedded) const {
    const ModelConfig& cfg = bundle_.config;
    const Tensor& cls = bundle_.tensors.tensor("model.cls_token");
    const Tensor& pos = bundle_.tensors.tensor("model.pos_embed");
    int tokens = cfg.patchCount() + 1;
    int hidden = cfg.hiddenSize;
    Tensor out({tokens, hidden});

    for (int h = 0; h < hidden; ++h) {
        out.data[h] = cls.data[h] + pos.data[h];
    }
    for (int t = 1; t < tokens; ++t) {
        const float* patch = patchEmbedded.data.data() + static_cast<size_t>(t - 1) * hidden;
        const float* posRow = pos.data.data() + static_cast<size_t>(t) * hidden;
        float* dst = out.data.data() + static_cast<size_t>(t) * hidden;
        for (int h = 0; h < hidden; ++h) dst[h] = patch[h] + posRow[h];
    }
    return out;
}

Tensor RPFQViTEngine::forwardBlock(int index, const Tensor& x) {
    const BlockParams& block = blocks_.at(static_cast<size_t>(index));
    auto stageStart = Clock::now();
    Tensor norm1 = layerNorm(x, *block.norm1Weight, *block.norm1Bias, bundle_.config.layerNormEps);
    RPFQ_VIT_LOGI("TIMING block=%d stage=norm1 ms=%lld", index, static_cast<long long>(elapsedMs(stageStart)));
    if (index == 0) debug_.writeTensor("block0_norm1", norm1);

    Tensor attnOut = forwardAttention(index, norm1, block);
    Tensor residual1 = addSameShape(x, attnOut);
    stageStart = Clock::now();
    Tensor norm2 = layerNorm(residual1, *block.norm2Weight, *block.norm2Bias, bundle_.config.layerNormEps);
    RPFQ_VIT_LOGI("TIMING block=%d stage=norm2 ms=%lld", index, static_cast<long long>(elapsedMs(stageStart)));
    Tensor mlpOut = forwardMlp(index, norm2, block);
    if (index == 0) debug_.writeTensor("block0_mlp_out", mlpOut);

    Tensor out = addSameShape(residual1, mlpOut);
    if (index == 0) debug_.writeTensor("block0", out);
    return out;
}

Tensor RPFQViTEngine::forwardAttention(int index, const Tensor& x, const BlockParams& block) {
    auto stageStart = Clock::now();
    Tensor qkv = wideLinear(x, block.attn.qkv, threadPool_.get());
    RPFQ_VIT_LOGI("TIMING block=%d stage=attn.qkv ms=%lld", index, static_cast<long long>(elapsedMs(stageStart)));
    if (index == 0) debug_.writeTensor("block0_qkv", qkv);

    Tensor preProj;
    stageStart = Clock::now();
    Tensor attnOutPreProj = attentionForward(
        qkv,
        bundle_.config.hiddenSize,
        bundle_.config.numAttentionHeads,
        &preProj);
    RPFQ_VIT_LOGI("TIMING block=%d stage=attn.core ms=%lld", index, static_cast<long long>(elapsedMs(stageStart)));
    (void)attnOutPreProj;
    if (index == 0) debug_.writeTensor("block0_attn_out_pre_proj", preProj);

    stageStart = Clock::now();
    Tensor projected = wideLinear(preProj, block.attn.proj, threadPool_.get());
    RPFQ_VIT_LOGI("TIMING block=%d stage=attn.proj ms=%lld", index, static_cast<long long>(elapsedMs(stageStart)));
    if (index == 0) debug_.writeTensor("block0_attn_out", projected);
    return projected;
}

Tensor RPFQViTEngine::forwardMlp(int index, const Tensor& x, const BlockParams& block) {
    auto stageStart = Clock::now();
    Tensor fc1 = wideLinear(x, block.mlp.fc1, threadPool_.get());
    RPFQ_VIT_LOGI("TIMING block=%d stage=mlp.fc1 ms=%lld", index, static_cast<long long>(elapsedMs(stageStart)));

    stageStart = Clock::now();
    Tensor activated = gelu(fc1);
    RPFQ_VIT_LOGI("TIMING block=%d stage=mlp.gelu ms=%lld", index, static_cast<long long>(elapsedMs(stageStart)));

    stageStart = Clock::now();
    Tensor fc2 = wideLinear(activated, block.mlp.fc2, threadPool_.get());
    RPFQ_VIT_LOGI("TIMING block=%d stage=mlp.fc2 ms=%lld", index, static_cast<long long>(elapsedMs(stageStart)));
    return fc2;
}

Tensor RPFQViTEngine::forwardHead(const Tensor& x) const {
    if (x.shape.size() != 2 || x.shape[0] < 1) throw std::runtime_error("final norm shape mismatch.");
    int hidden = bundle_.config.hiddenSize;
    Tensor cls({1, hidden});
    std::copy(x.data.begin(), x.data.begin() + hidden, cls.data.begin());
    return linear(cls, bundle_.tensors.tensor("head.weight"), bundle_.tensors.tensor("head.bias"));
}

void RPFQViTEngine::buildParams() {
    blocks_.clear();
    blocks_.reserve(static_cast<size_t>(bundle_.config.numHiddenLayers));
    const ModelConfig& cfg = bundle_.config;
    for (int layer = 0; layer < cfg.numHiddenLayers; ++layer) {
        std::string p = layerPrefix(layer);
        BlockParams block;
        block.norm1Weight = &bundle_.tensors.tensor(p + ".norm1.weight");
        block.norm1Bias = &bundle_.tensors.tensor(p + ".norm1.bias");
        block.norm2Weight = &bundle_.tensors.tensor(p + ".norm2.weight");
        block.norm2Bias = &bundle_.tensors.tensor(p + ".norm2.bias");
        block.attn.qkv = loadWideParams(
            bundle_.tensors,
            p + ".attn.qkv",
            cfg.hiddenSize,
            cfg.hiddenSize * 3,
            cfg.hiddenComplexPadded,
            cfg.qkvComplexPadded,
            cfg.qkvComplex,
            cfg.blockSize);
        block.attn.proj = loadWideParams(
            bundle_.tensors,
            p + ".attn.proj",
            cfg.hiddenSize,
            cfg.hiddenSize,
            cfg.hiddenComplexPadded,
            cfg.hiddenComplexPadded,
            cfg.hiddenComplex,
            cfg.blockSize);
        block.mlp.fc1 = loadWideParams(
            bundle_.tensors,
            p + ".mlp.fc1",
            cfg.hiddenSize,
            cfg.intermediateSize,
            cfg.hiddenComplexPadded,
            cfg.intermediateComplexPadded,
            cfg.intermediateComplex,
            cfg.blockSize);
        block.mlp.fc2 = loadWideParams(
            bundle_.tensors,
            p + ".mlp.fc2",
            cfg.intermediateSize,
            cfg.hiddenSize,
            cfg.intermediateComplexPadded,
            cfg.hiddenComplexPadded,
            cfg.hiddenComplex,
            cfg.blockSize);
        blocks_.push_back(block);
    }
}

} // namespace rpfq_vit
