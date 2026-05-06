#include "safetensors_loader.h"

#include "json_utils.h"

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace rpfq_vit {
namespace {

std::vector<int64_t> parseShape(const JsonValue& value) {
    if (!value.isArray()) throw std::runtime_error("Safetensor shape is not an array.");
    std::vector<int64_t> shape;
    shape.reserve(value.array_value.size());
    for (const JsonValue& dim : value.array_value) {
        shape.push_back(static_cast<int64_t>(dim.asDouble()));
    }
    return shape;
}

std::pair<size_t, size_t> parseOffsets(const JsonValue& value) {
    if (!value.isArray() || value.array_value.size() != 2) {
        throw std::runtime_error("Safetensor data_offsets must contain two values.");
    }
    return {
        static_cast<size_t>(value.array_value[0].asDouble()),
        static_cast<size_t>(value.array_value[1].asDouble())
    };
}

} // namespace

const Tensor& SafeTensorStore::tensor(const std::string& name) const {
    auto it = f32.find(name);
    if (it == f32.end()) throw std::runtime_error("Missing tensor: " + name);
    return it->second;
}

const UInt8Tensor& SafeTensorStore::bytes(const std::string& name) const {
    auto it = u8.find(name);
    if (it == u8.end()) throw std::runtime_error("Missing uint8 tensor: " + name);
    return it->second;
}

bool SafeTensorStore::hasTensor(const std::string& name) const {
    return f32.find(name) != f32.end();
}

bool SafeTensorStore::hasBytes(const std::string& name) const {
    return u8.find(name) != u8.end();
}

SafeTensorStore loadSafetensors(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Failed to open safetensors file: " + path);

    uint8_t headerLenBytes[8];
    in.read(reinterpret_cast<char*>(headerLenBytes), 8);
    if (in.gcount() != 8) throw std::runtime_error("Invalid safetensors header.");
    uint64_t headerLen = readLe64(headerLenBytes);

    std::string header(static_cast<size_t>(headerLen), '\0');
    in.read(header.data(), static_cast<std::streamsize>(headerLen));
    if (static_cast<uint64_t>(in.gcount()) != headerLen) {
        throw std::runtime_error("Truncated safetensors header.");
    }

    in.seekg(0, std::ios::end);
    size_t totalSize = static_cast<size_t>(in.tellg());
    size_t dataBase = 8 + static_cast<size_t>(headerLen);
    JsonValue root = parseJson(header);

    SafeTensorStore store;
    for (const auto& item : root.object_value) {
        const std::string& name = item.first;
        if (name == "__metadata__") continue;
        const JsonValue& meta = item.second;
        std::string dtype = meta.at("dtype").asString();
        std::vector<int64_t> shape = parseShape(meta.at("shape"));
        auto [begin, end] = parseOffsets(meta.at("data_offsets"));
        if (dataBase + end > totalSize || begin > end) {
            throw std::runtime_error("Invalid safetensors data offset for " + name);
        }

        size_t bytes = end - begin;
        std::vector<uint8_t> raw(bytes);
        in.seekg(static_cast<std::streamoff>(dataBase + begin), std::ios::beg);
        in.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(bytes));
        if (static_cast<size_t>(in.gcount()) != bytes) {
            throw std::runtime_error("Truncated safetensor payload for " + name);
        }

        if (dtype == "U8") {
            UInt8Tensor t;
            t.shape = std::move(shape);
            t.data = std::move(raw);
            store.u8.emplace(name, std::move(t));
        } else if (dtype == "F16") {
            Tensor t(std::move(shape));
            if (bytes != static_cast<size_t>(t.numel()) * 2) {
                throw std::runtime_error("F16 tensor byte count mismatch for " + name);
            }
            for (size_t i = 0; i < t.data.size(); ++i) {
                t.data[i] = halfToFloat(readLe16(raw.data() + i * 2));
            }
            store.f32.emplace(name, std::move(t));
        } else if (dtype == "F32") {
            Tensor t(std::move(shape));
            if (bytes != static_cast<size_t>(t.numel()) * 4) {
                throw std::runtime_error("F32 tensor byte count mismatch for " + name);
            }
            const uint8_t* p = raw.data();
            for (size_t i = 0; i < t.data.size(); ++i) {
                uint32_t word = static_cast<uint32_t>(p[i * 4])
                    | (static_cast<uint32_t>(p[i * 4 + 1]) << 8)
                    | (static_cast<uint32_t>(p[i * 4 + 2]) << 16)
                    | (static_cast<uint32_t>(p[i * 4 + 3]) << 24);
                float value;
                std::memcpy(&value, &word, sizeof(float));
                t.data[i] = value;
            }
            store.f32.emplace(name, std::move(t));
        } else {
            throw std::runtime_error("Unsupported safetensors dtype " + dtype + " for " + name);
        }
    }
    return store;
}

} // namespace rpfq_vit
