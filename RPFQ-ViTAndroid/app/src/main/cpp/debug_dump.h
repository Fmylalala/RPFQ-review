#pragma once

#include "tensor.h"

#include <string>

namespace rpfq_vit {

class DebugDump {
public:
    void configure(bool enabled, std::string outputDir);
    bool enabled() const { return enabled_; }
    void writeTensor(const std::string& name, const Tensor& tensor) const;

private:
    bool enabled_ = false;
    std::string outputDir_;
};

} // namespace rpfq_vit
