#pragma once

#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace rpfq_vit {

int selectThreadCount();

class ThreadPool {
public:
    explicit ThreadPool(int numThreads = 0);
    ~ThreadPool();

    int numThreads() const;

    void parallelFor(
        int64_t begin,
        int64_t end,
        const std::function<void(int64_t, int64_t)>& fn);

private:
    void workerLoop(int workerId);

    int numThreads_ = 1;
    std::vector<std::thread> workers_;

    mutable std::mutex mutex_;
    std::condition_variable jobCv_;
    std::condition_variable doneCv_;

    bool stop_ = false;
    uint64_t generation_ = 0;
    int activeWorkers_ = 0;
    int workerJobs_ = 0;
    int64_t begin_ = 0;
    int64_t end_ = 0;
    int64_t chunk_ = 0;
    const std::function<void(int64_t, int64_t)>* currentFn_ = nullptr;
    std::exception_ptr firstError_;
};

} // namespace rpfq_vit
