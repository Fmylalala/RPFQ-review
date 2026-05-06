#include "debug_dump.h"

#include <cerrno>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>

namespace rpfq_vit {
namespace {

void ensureDir(const std::string& path) {
    if (path.empty()) return;
#ifdef _WIN32
    if (::mkdir(path.c_str()) != 0 && errno != EEXIST) {
#else
    if (::mkdir(path.c_str(), 0700) != 0 && errno != EEXIST) {
#endif
        throw std::runtime_error("Failed to create debug directory: " + path + " " + std::strerror(errno));
    }
}

} // namespace

void DebugDump::configure(bool enabled, std::string outputDir) {
    enabled_ = enabled;
    outputDir_ = std::move(outputDir);
    if (enabled_) ensureDir(outputDir_);
}

void DebugDump::writeTensor(const std::string& name, const Tensor& tensor) const {
    if (!enabled_ || outputDir_.empty()) return;
    ensureDir(outputDir_);

    const std::string binPath = outputDir_ + "/" + name + ".bin";
    std::ofstream bin(binPath, std::ios::binary);
    if (!bin) throw std::runtime_error("Failed to open debug tensor for write: " + binPath);
    bin.write(reinterpret_cast<const char*>(tensor.data.data()),
              static_cast<std::streamsize>(tensor.data.size() * sizeof(float)));

    std::ostringstream meta;
    meta << "{\n  \"name\": \"" << name << "\",\n  \"shape\": [";
    for (size_t i = 0; i < tensor.shape.size(); ++i) {
        if (i) meta << ", ";
        meta << tensor.shape[i];
    }
    meta << "],\n  \"dtype\": \"float32\",\n  \"layout\": \"row_major\"\n}\n";

    const std::string jsonPath = outputDir_ + "/" + name + ".json";
    std::ofstream json(jsonPath, std::ios::binary);
    if (!json) throw std::runtime_error("Failed to open debug metadata for write: " + jsonPath);
    json << meta.str();
}

} // namespace rpfq_vit
