#pragma once

#include "bundle_loader.h"
#include "debug_dump.h"
#include "ops/phase_linear.h"
#include "runtime/thread_pool.h"

#include <memory>
#include <string>
#include <vector>

namespace rpfq_vit {

class RPFQViTEngine {
public:
    bool loadBundle(const std::string& bundleDir);
    std::vector<float> predict(const std::vector<float>& inputNchw);
    void setDebugEnabled(bool enabled);
    void setDebugOutputDir(const std::string& outputDir);

private:
    struct AttentionParams {
        WideLinearParams qkv;
        WideLinearParams proj;
    };

    struct MlpParams {
        WideLinearParams fc1;
        WideLinearParams fc2;
    };

    struct BlockParams {
        const Tensor* norm1Weight = nullptr;
        const Tensor* norm1Bias = nullptr;
        const Tensor* norm2Weight = nullptr;
        const Tensor* norm2Bias = nullptr;
        AttentionParams attn;
        MlpParams mlp;
    };

    BundleData bundle_;
    bool loaded_ = false;
    DebugDump debug_;
    std::vector<BlockParams> blocks_;
    std::unique_ptr<ThreadPool> threadPool_;

    Tensor inputTensor(const std::vector<float>& inputNchw) const;
    Tensor patchify(const Tensor& nchw) const;
    Tensor patchEmbed(const Tensor& patches) const;
    Tensor addClassAndPosition(const Tensor& patchEmbedded) const;
    Tensor forwardBlock(int index, const Tensor& x);
    Tensor forwardAttention(int index, const Tensor& x, const BlockParams& block);
    Tensor forwardMlp(int index, const Tensor& x, const BlockParams& block);
    Tensor forwardHead(const Tensor& x) const;
    void buildParams();
};

} // namespace rpfq_vit
