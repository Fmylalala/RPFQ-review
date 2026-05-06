#pragma once

#include "json_utils.h"
#include "safetensors_loader.h"

#include <string>
#include <vector>

namespace rpfq_vit {

struct ModelConfig {
    std::string variant;
    int imageSize = 224;
    int patchSize = 16;
    int numChannels = 3;
    int hiddenSize = 768;
    int numHiddenLayers = 12;
    int numAttentionHeads = 12;
    int intermediateSize = 3072;
    int numClasses = 1000;
    float layerNormEps = 1e-5f;
    int hiddenComplex = 384;
    int hiddenComplexPadded = 512;
    int qkvComplex = 1152;
    int qkvComplexPadded = 1280;
    int intermediateComplex = 1536;
    int intermediateComplexPadded = 1536;
    int residualStages = 2;
    int blockSize = 256;
    std::string quantMethod = "complex_phase_v2";

    int headDim() const { return hiddenSize / numAttentionHeads; }
    int patchVectorSize() const { return patchSize * patchSize * numChannels; }
    int patchCount() const {
        int side = imageSize / patchSize;
        return side * side;
    }
};

struct BundleData {
    ModelConfig config;
    JsonValue manifest;
    SafeTensorStore tensors;
};

ModelConfig loadModelConfig(const std::string& path);
BundleData loadBundleData(const std::string& bundleDir);

} // namespace rpfq_vit
