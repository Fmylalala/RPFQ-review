#include "parallel_for.h"

#include <algorithm>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace rpfq_vit {

int defaultThreadCount() {
    unsigned int hc = std::thread::hardware_concurrency();
    if (hc == 0) return 4;
    if (hc == 1) return 1;
    return std::max(1, std::min(4, static_cast<int>(hc) - 1));
}

void parallelFor(
    int64_t begin,
    int64_t end,
    int numThreads,
    const std::function<void(int64_t, int64_t)>& fn) {
    int64_t total = end - begin;
    if (total <= 0) return;

    int maxThreads = defaultThreadCount();
    int threads = numThreads > 0 ? numThreads : maxThreads;
    threads = std::max(1, std::min(threads, maxThreads));
    threads = static_cast<int>(std::min<int64_t>(threads, total));

    if (threads <= 1 || total < threads * 2) {
        fn(begin, end);
        return;
    }

    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(threads - 1));
    std::exception_ptr firstError;
    std::mutex errorMutex;

    int64_t chunk = (total + threads - 1) / threads;
    for (int i = 0; i < threads; ++i) {
        int64_t rb = begin + i * chunk;
        int64_t re = std::min(end, rb + chunk);
        if (rb >= re) break;

        auto run = [&, rb, re]() {
            try {
                fn(rb, re);
            } catch (...) {
                std::lock_guard<std::mutex> lock(errorMutex);
                if (!firstError) firstError = std::current_exception();
            }
        };

        if (i == threads - 1) {
            run();
        } else {
            workers.emplace_back(run);
        }
    }

    for (std::thread& worker : workers) {
        worker.join();
    }
    if (firstError) {
        std::rethrow_exception(firstError);
    }
}

} // namespace rpfq_vit
