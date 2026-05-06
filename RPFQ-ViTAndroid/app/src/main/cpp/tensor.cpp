#include "tensor.h"

#include <cstring>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace rpfq_vit {

Tensor::Tensor(std::vector<int64_t> s) : shape(std::move(s)) {
    data.resize(static_cast<size_t>(numel()));
}

Tensor::Tensor(std::vector<int64_t> s, std::vector<float> d)
    : shape(std::move(s)), data(std::move(d)) {
    if (numel() != static_cast<int64_t>(data.size())) {
        throw std::runtime_error("Tensor shape/data size mismatch.");
    }
}

int64_t Tensor::numel() const {
    if (shape.empty()) return 0;
    return std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>());
}

int64_t Tensor::dim(size_t index) const {
    if (index >= shape.size()) throw std::runtime_error("Tensor dim index out of range.");
    return shape[index];
}

int64_t UInt8Tensor::numel() const {
    if (shape.empty()) return 0;
    return std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>());
}

int64_t UInt8Tensor::dim(size_t index) const {
    if (index >= shape.size()) throw std::runtime_error("Tensor dim index out of range.");
    return shape[index];
}

uint16_t readLe16(const uint8_t* p) {
    return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

uint64_t readLe64(const uint8_t* p) {
    uint64_t value = 0;
    for (int i = 7; i >= 0; --i) {
        value = (value << 8) | p[i];
    }
    return value;
}

float halfToFloat(uint16_t h) {
    uint32_t sign = (static_cast<uint32_t>(h & 0x8000)) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x03ff;
    uint32_t out;

    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x03ff;
            exp = exp + (127 - 15);
            out = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7f800000u | (mant << 13);
    } else {
        exp = exp + (127 - 15);
        out = sign | (exp << 23) | (mant << 13);
    }

    float f;
    std::memcpy(&f, &out, sizeof(float));
    return f;
}

std::string shapeString(const std::vector<int64_t>& shape) {
    std::ostringstream ss;
    ss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) ss << ",";
        ss << shape[i];
    }
    ss << "]";
    return ss.str();
}

} // namespace rpfq_vit
