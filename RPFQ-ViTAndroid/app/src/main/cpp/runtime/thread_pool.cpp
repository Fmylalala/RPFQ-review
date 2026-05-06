#include "thread_pool.h"

#include <algorithm>

namespace rpfq_vit {

int selectThreadCount() {
    unsigned int hc = std::thread::hardware_concurrency();
    if (hc == 0) return 4;
    if (hc == 1) return 1;
    return std::max(1, std::min(4, static_cast<int>(hc) - 1));
}

ThreadPool::ThreadPool(int numThreads) {
    numThreads_ = numThreads > 0 ? numThreads : selectThreadCount();
    numThreads_ = std::max(1, numThreads_);
    workers_.reserve(static_cast<size_t>(std::max(0, numThreads_ - 1)));
    for (int i = 0; i < numThreads_ - 1; ++i) {
        workers_.emplace_back([this, i]() { workerLoop(i); });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
        ++generation_;
    }
    jobCv_.notify_all();
    for (std::thread& worker : workers_) {
        if (worker.joinable()) worker.join();
    }
}

int ThreadPool::numThreads() const {
    return numThreads_;
}

void ThreadPool::parallelFor(
    int64_t begin,
    int64_t end,
    const std::function<void(int64_t, int64_t)>& fn) {
    int64_t total = end - begin;
    if (total <= 0) return;

    int threads = static_cast<int>(std::min<int64_t>(numThreads_, total));
    if (threads <= 1) {
        fn(begin, end);
        return;
    }

    int64_t chunk = (total + threads - 1) / threads;
    int workerJobs = threads - 1;

    {
        std::lock_guard<std::mutex> lock(mutex_);
        firstError_ = nullptr;
        begin_ = begin;
        end_ = end;
        chunk_ = chunk;
        workerJobs_ = workerJobs;
        activeWorkers_ = workerJobs;
        currentFn_ = &fn;
        ++generation_;
    }
    jobCv_.notify_all();

    int64_t mainBegin = begin + static_cast<int64_t>(workerJobs) * chunk;
    int64_t mainEnd = std::min(end, mainBegin + chunk);
    if (mainBegin < mainEnd) {
        try {
            fn(mainBegin, mainEnd);
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!firstError_) firstError_ = std::current_exception();
        }
    }

    {
        std::unique_lock<std::mutex> lock(mutex_);
        doneCv_.wait(lock, [this]() { return activeWorkers_ == 0; });
        currentFn_ = nullptr;
        if (firstError_) {
            std::exception_ptr error = firstError_;
            firstError_ = nullptr;
            std::rethrow_exception(error);
        }
    }
}

void ThreadPool::workerLoop(int workerId) {
    uint64_t seenGeneration = 0;
    while (true) {
        int64_t rb = 0;
        int64_t re = 0;
        const std::function<void(int64_t, int64_t)>* fn = nullptr;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            jobCv_.wait(lock, [this, &seenGeneration]() {
                return stop_ || generation_ != seenGeneration;
            });
            if (stop_) return;

            seenGeneration = generation_;
            if (workerId >= workerJobs_) {
                continue;
            }

            rb = begin_ + static_cast<int64_t>(workerId) * chunk_;
            re = std::min(end_, rb + chunk_);
            fn = currentFn_;
        }

        try {
            if (fn && rb < re) (*fn)(rb, re);
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!firstError_) firstError_ = std::current_exception();
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            --activeWorkers_;
            if (activeWorkers_ == 0) doneCv_.notify_one();
        }
    }
}

} // namespace rpfq_vit
