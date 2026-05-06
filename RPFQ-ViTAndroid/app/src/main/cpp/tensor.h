#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace rpfq_vit {

struct Tensor {
    std::vector<int64_t> shape;
    std::vector<float> data;

    Tensor() = default;
    explicit Tensor(std::vector<int64_t> s);
    Tensor(std::vector<int64_t> s, std::vector<float> d);

    int64_t numel() const;
    int64_t dim(size_t index) const;
    float* f32() { return data.data(); }
    const float* f32() const { return data.data(); }
};

struct UInt8Tensor {
    std::vector<int64_t> shape;
    std::vector<uint8_t> data;

    int64_t numel() const;
    int64_t dim(size_t index) const;
    const uint8_t* u8() const { return data.data(); }
};

float halfToFloat(uint16_t h);
uint16_t readLe16(const uint8_t* p);
uint64_t readLe64(const uint8_t* p);
std::string shapeString(const std::vector<int64_t>& shape);

} // namespace rpfq_vit
