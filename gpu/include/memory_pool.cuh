#ifndef MPBOOTGPU_MEMORY_POOL_CUH_
#define MPBOOTGPU_MEMORY_POOL_CUH_

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "utils.cuh"

namespace mpbootgpu
{

class gpuMemoryPool
{
   public:
    gpuMemoryPool() = default;

    explicit gpuMemoryPool(
        size_t size
    )
    {
        initialize(size);
    }

    ~gpuMemoryPool()
    {
        shutdown();
    }

    // Non-copyable
    gpuMemoryPool(const gpuMemoryPool&) = delete;
    gpuMemoryPool& operator=(const gpuMemoryPool&) = delete;

    // Movable
    gpuMemoryPool(
        gpuMemoryPool&& other
    ) noexcept
        : pool_(other.pool_),
          pool_size_(other.pool_size_),
          used_(other.used_),
          peak_used_(other.peak_used_)
    {
        other.pool_ = nullptr;
        other.pool_size_ = 0;
        other.used_ = 0;
        other.peak_used_ = 0;
    }

    gpuMemoryPool& operator=(
        gpuMemoryPool&& other
    ) noexcept
    {
        if (this != &other)
        {
            shutdown();
            pool_ = other.pool_;
            pool_size_ = other.pool_size_;
            used_ = other.used_;
            peak_used_ = other.peak_used_;
            other.pool_ = nullptr;
            other.pool_size_ = 0;
            other.used_ = 0;
            other.peak_used_ = 0;
        }
        return *this;
    }

    void initialize(
        size_t size
    )
    {
        if (pool_)
        {
            shutdown();
        }

        CUDA_CHECK(cudaMalloc(&pool_, size));
        pool_size_ = size;
        used_ = 0;
        peak_used_ = 0;
    }

    void shutdown()
    {
        if (pool_)
        {
            cudaFree(pool_);
            pool_ = nullptr;
            pool_size_ = 0;
            used_ = 0;
        }
    }

    // allocate memory with alignment
    template <typename T>
    T* allocate(
        size_t count, size_t alignment = 256
    )
    {
        return static_cast<T*>(allocateBytes(count * sizeof(T), alignment));
    }

    void* allocateBytes(
        size_t size, size_t alignment = 256
    )
    {
        // Align current position
        size_t aligned_used = (used_ + alignment - 1) & ~(alignment - 1);

        if (aligned_used + size > pool_size_)
        {
            throw std::runtime_error(
                "GPU memory pool exhausted: requested " + std::to_string(size)
                + " bytes, available " + std::to_string(pool_size_ - aligned_used)
                + " bytes, pool size " + std::to_string(pool_size_) + " bytes"
            );
        }

        void* ptr = static_cast<char*>(pool_) + aligned_used;
        used_ = aligned_used + size;
        peak_used_ = std::max(peak_used_, used_);

        return ptr;
    }

    // reset pool for reuse (doesn't free memory, just resets pointer)
    void reset()
    {
        used_ = 0;
    }

    // Getters
    [[nodiscard]] bool isInitialized() const
    {
        return pool_ != nullptr;
    }
    [[nodiscard]] size_t getUsed() const
    {
        return used_;
    }
    [[nodiscard]] size_t getRemaining() const
    {
        return pool_size_ - used_;
    }
    [[nodiscard]] size_t getSize() const
    {
        return pool_size_;
    }
    [[nodiscard]] size_t getPeakUsed() const
    {
        return peak_used_;
    }
    [[nodiscard]] void* getBasePtr() const
    {
        return pool_;
    }

   private:
    void* pool_ = nullptr;
    size_t pool_size_ = 0;
    size_t used_ = 0;
    size_t peak_used_ = 0;
};

class pinnedMemoryPool
{
   public:
    pinnedMemoryPool() = default;

    explicit pinnedMemoryPool(
        size_t size
    )
    {
        initialize(size);
    }

    ~pinnedMemoryPool()
    {
        shutdown();
    }

    // Non-copyable, movable (similar to gpuMemoryPool)
    pinnedMemoryPool(const pinnedMemoryPool&) = delete;
    pinnedMemoryPool& operator=(const pinnedMemoryPool&) = delete;
    pinnedMemoryPool(pinnedMemoryPool&&) noexcept;
    pinnedMemoryPool& operator=(pinnedMemoryPool&&) noexcept;

    void initialize(
        size_t size
    )
    {
        if (pool_)
        {
            shutdown();
        }

        CUDA_CHECK(cudaMallocHost(&pool_, size));
        pool_size_ = size;

        used_ = 0;
    }

    void shutdown()
    {
        if (pool_)
        {
            cudaFreeHost(pool_);
            pool_ = nullptr;
            pool_size_ = 0;
            used_ = 0;
        }
    }

    template <typename T>
    T* allocate(
        size_t count, size_t alignment = 64
    )
    {
        return static_cast<T*>(allocateBytes(count * sizeof(T), alignment));
    }

    void* allocateBytes(
        size_t size, size_t alignment = 64
    )
    {
        size_t aligned_used = (used_ + alignment - 1) & ~(alignment - 1);

        if (aligned_used + size > pool_size_)
        {
            throw std::runtime_error("Pinned memory pool exhausted");
        }

        void* ptr = static_cast<char*>(pool_) + aligned_used;
        used_ = aligned_used + size;
        return ptr;
    }

    void reset()
    {
        used_ = 0;
    }

    [[nodiscard]] bool isInitialized() const
    {
        return pool_ != nullptr;
    }
    [[nodiscard]] size_t getUsed() const
    {
        return used_;
    }
    [[nodiscard]] size_t getRemaining() const
    {
        return pool_size_ - used_;
    }
    [[nodiscard]] size_t getSize() const
    {
        return pool_size_;
    }

   private:
    void* pool_ = nullptr;
    size_t pool_size_ = 0;
    size_t used_ = 0;
};

//=============================================================================
// GPU Memory Manager
// Manages permanent (index) and per-batch (working) memory pools
//=============================================================================

class gpuMemoryManager
{
   public:
    gpuMemoryManager() = default;
    ~gpuMemoryManager()
    {
        shutdown();
    }

    // initialize with pool sizes
    void initialize(
        int device_id, size_t permanent_size, size_t batch_size, size_t pinned_size = 0
    )
    {
        device_id_ = device_id;
        CUDA_CHECK(cudaSetDevice(device_id));

        // Query device memory
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
        total_vram_ = prop.totalGlobalMem;

        // initialize pools
        permanent_pool_.initialize(permanent_size);

        batch_pool_.initialize(batch_size);

        if (pinned_size > 0)
        {
            constexpr size_t k_min_pinned_pool_bytes = 16ULL * 1024ULL * 1024ULL;
            constexpr int k_max_pinned_attempts = 12;
            size_t requested_pinned = pinned_size;
            bool pinned_ready = false;
            for (int attempt = 0;
                 attempt < k_max_pinned_attempts && requested_pinned >= k_min_pinned_pool_bytes;
                 ++attempt)
            {
                try
                {
                    pinned_pool_.initialize(requested_pinned);
                    pinned_ready = true;
                    if (requested_pinned != pinned_size)
                    {
                        std::cout << "Pinned pool allocated at reduced size "
                                  << requested_pinned / (1024.0 * 1024.0) << " MB (requested "
                                  << pinned_size / (1024.0 * 1024.0) << " MB)" << std::endl;
                        break;
                    }
                }
                catch (const std::exception& e)
                {
                    std::cout << "Pinned pool allocation failed at "
                              << requested_pinned / (1024.0 * 1024.0) << " MB (attempt "
                              << attempt + 1 << "/" << k_max_pinned_attempts << "): " << e.what()
                              << std::endl;
                    requested_pinned /= 2;
                }
            }

            if (!pinned_ready)
            {
                std::cout << "Pinned pool disabled after allocation failures; falling back to "
                             "pageable host staging"
                          << std::endl;
            }
        }

        initialized_ = true;
    }

    void shutdown()
    {
        if (initialized_)
        {
            // GPU frees must happen on the device that owns the pools.
            cudaSetDevice(device_id_);
            permanent_pool_.shutdown();
            batch_pool_.shutdown();
            pinned_pool_.shutdown();
            initialized_ = false;
        }
    }

    // Access pools
    gpuMemoryPool& getPermanentPool()
    {
        return permanent_pool_;
    }
    gpuMemoryPool& getBatchPool()
    {
        return batch_pool_;
    }
    pinnedMemoryPool& getPinnedPool()
    {
        return pinned_pool_;
    }

    // reset batch pool for next batch
    void resetBatchPool()
    {
        batch_pool_.reset();
    }

    // Query device
    [[nodiscard]] int getDeviceId() const
    {
        return device_id_;
    }
    [[nodiscard]] size_t getTotalVRAM() const
    {
        return total_vram_;
    }
    [[nodiscard]] size_t getFreeVRAM() const
    {
        CUDA_CHECK(cudaSetDevice(device_id_));
        size_t free, total;
        cudaMemGetInfo(&free, &total);
        return free;
    }

    // Memory usage report
    void printMemoryUsage() const;

   private:
    int device_id_ = 0;
    size_t total_vram_ = 0;
    bool initialized_ = false;

    gpuMemoryPool permanent_pool_;
    gpuMemoryPool batch_pool_;
    pinnedMemoryPool pinned_pool_;
};

}  // namespace mpbootgpu

#endif  // MPBOOTGPU_MEMORY_POOL_CUH_
