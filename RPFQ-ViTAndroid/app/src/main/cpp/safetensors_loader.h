#pragma once

#include "tensor.h"

#include <map>
#include <string>

namespace rpfq_vit {

struct SafeTensorStore {
    std::map<std::string, Tensor> f32;
    std::map<std::string, UInt8Tensor> u8;

    const Tensor& tensor(const std::string& name) const;
    const UInt8Tensor& bytes(const std::string& name) const;
    bool hasTensor(const std::string& name) const;
    bool hasBytes(const std::string& name) const;
};

SafeTensorStore loadSafetensors(const std::string& path);

} // namespace rpfq_vit
