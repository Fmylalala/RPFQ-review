#include "bundle_loader.h"

#include <stdexcept>

namespace rpfq_vit {
namespace {

int nestedInt(const JsonValue& parent, const std::string& key, int fallback) {
    const JsonValue* v = parent.find(key);
    return v ? v->asInt(fallback) : fallback;
}

std::string nestedString(const JsonValue& parent, const std::string& key, const std::string& fallback) {
    const JsonValue* v = parent.find(key);
    return v ? v->asString(fallback) : fallback;
}

} // namespace

ModelConfig loadModelConfig(const std::string& path) {
    JsonValue root = parseJsonFile(path);
    ModelConfig cfg;
    cfg.variant = nestedString(root, "variant", "rpfq_vit_base_patch16_224_q");
    cfg.imageSize = nestedInt(root, "image_size", 224);
    cfg.patchSize = nestedInt(root, "patch_size", 16);
    cfg.numChannels = nestedInt(root, "num_channels", 3);
    cfg.hiddenSize = nestedInt(root, "hidden_size", 768);
    cfg.numHiddenLayers = nestedInt(root, "num_hidden_layers", 12);
    cfg.numAttentionHeads = nestedInt(root, "num_attention_heads", 12);
    cfg.intermediateSize = nestedInt(root, "intermediate_size", 3072);
    cfg.numClasses = nestedInt(root, "num_classes", 1000);
    if (const JsonValue* eps = root.find("layer_norm_eps")) {
        cfg.layerNormEps = static_cast<float>(eps->asDouble(1e-5));
    }

    if (const JsonValue* rpfq_vit = root.find("rpfq_vit")) {
        cfg.hiddenComplex = nestedInt(*rpfq_vit, "hidden_complex", cfg.hiddenSize / 2);
        cfg.hiddenComplexPadded = nestedInt(*rpfq_vit, "hidden_complex_padded", cfg.hiddenComplex);
        cfg.qkvComplex = nestedInt(*rpfq_vit, "qkv_complex", cfg.hiddenComplex * 3);
        cfg.qkvComplexPadded = nestedInt(*rpfq_vit, "qkv_complex_padded", cfg.qkvComplex);
        cfg.intermediateComplex = nestedInt(*rpfq_vit, "intermediate_complex", cfg.intermediateSize / 2);
        cfg.intermediateComplexPadded = nestedInt(*rpfq_vit, "intermediate_complex_padded", cfg.intermediateComplex);
        cfg.residualStages = nestedInt(*rpfq_vit, "residual_stages", 2);
        cfg.blockSize = nestedInt(*rpfq_vit, "block_size", 256);
        cfg.quantMethod = nestedString(*rpfq_vit, "quant_method", "complex_phase_v2");
    }

    if (cfg.quantMethod != "complex_phase_v2") {
        throw std::runtime_error("Unsupported quant method: " + cfg.quantMethod);
    }
    return cfg;
}

BundleData loadBundleData(const std::string& bundleDir) {
    BundleData data;
    data.config = loadModelConfig(bundleDir + "/config.json");
    data.manifest = parseJsonFile(bundleDir + "/manifest.json");
    std::string backend = nestedString(data.manifest, "runtime_backend", "");
    std::string quant = nestedString(data.manifest, "quant_method", "");
    if (backend != "android_ndk_cpp" || quant != "complex_phase_v2") {
        throw std::runtime_error("Android manifest does not declare the native complex_phase_v2 backend.");
    }
    data.tensors = loadSafetensors(bundleDir + "/model.safetensors");
    return data;
}

} // namespace rpfq_vit
