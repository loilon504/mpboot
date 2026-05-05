#ifndef MPBOOTGPU_PROFILER_HPP_
#define MPBOOTGPU_PROFILER_HPP_

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace mpbootgpu
{
struct ThreadLocalData
{
    std::unordered_map<const char*, int64_t> accumulated;
};

class Profiler
{
   public:
    static Profiler& instance()
    {
        static Profiler instance;
        return instance;
    }

    void addTime(
        const char* name, int64_t nanos
    ) noexcept
    {
        getThreadData().accumulated[name] += nanos;
    }

    void report()
    {
        std::lock_guard<std::mutex> lock(global_mutex_);
        std::unordered_map<std::string, double> totals;

        for (auto* data : all_threads_data_)
        {
            for (const auto& kv : data->accumulated)
            {
                totals[kv.first] += static_cast<double>(kv.second) / 1e9;
            }
        }

        std::vector<std::pair<std::string, double>> sorted(totals.begin(), totals.end());

        std::sort(
            sorted.begin(), sorted.end(),
            [](const std::pair<std::string, double>& a, const std::pair<std::string, double>& b)
            {
                return a.first < b.first;
            }
        );

        std::cout << "\n==== Profiler Report ====\n";
        for (const auto& kv : sorted)
        {
            std::cout << std::fixed << std::setprecision(3) << kv.first << ": " << kv.second
                      << "s\n";
        }
    }

   private:
    Profiler() = default;

    ThreadLocalData& getThreadData()
    {
        thread_local ThreadLocalData local_data;
        thread_local bool registered = [this](ThreadLocalData* data)
        {
            std::lock_guard<std::mutex> lock(global_mutex_);
            all_threads_data_.push_back(data);
            return true;
        }(&local_data);
        return local_data;
    }

    std::mutex global_mutex_;
    std::vector<ThreadLocalData*> all_threads_data_;
};

class ProfilerTimer
{
   public:
    explicit ProfilerTimer(
        const char* name
    ) noexcept
        : name_(name), start_(std::chrono::high_resolution_clock::now())
    {
    }

    ~ProfilerTimer()
    {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_).count();
        Profiler::instance().addTime(name_, duration);
    }

   private:
    const char* name_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

class InlineProfilerTimer
{
   public:
    explicit InlineProfilerTimer(
        const char* name
    ) noexcept
        : name_(name), start_(std::chrono::high_resolution_clock::now())
    {
    }

    ~InlineProfilerTimer()
    {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_).count();

        std::cout << name_ << ": " << duration << " ms\n";
    }

   private:
    const char* name_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

}  // namespace mpbootgpu

#define PROFILE_SCOPE(name) mpbootgpu::ProfilerTimer timer_##__LINE__(name)
#define PROFILE_FUNCTION() PROFILE_SCOPE(__FUNCTION__)

#endif