#pragma once

#include <cstdint>
#include <functional>

namespace rpfq_vit {

int defaultThreadCount();

void parallelFor(
    int64_t begin,
    int64_t end,
    int numThreads,
    const std::function<void(int64_t, int64_t)>& fn);

} // namespace rpfq_vit
