#ifndef MPBOOTGPU_GPU_CONFIG_CUH_
#define MPBOOTGPU_GPU_CONFIG_CUH_

#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>

namespace mpbootgpu
{

struct gpuConfig
{
    //-------------------------------------------------------------------------
    // Batch size hyperparameter (main tuning knob for VRAM usage)
    //-------------------------------------------------------------------------
    uint32_t batch_size = 18432;  // Reads per GPU batch (B)

    //-------------------------------------------------------------------------
    // Memory pool sizes (auto-calculated if 0)
    //-------------------------------------------------------------------------
    size_t permanent_pool_size = 0;  // For genome data (auto: 1GB)
    size_t batch_pool_size = 0;      // For per-batch data (auto: calculated)
    size_t pinned_pool_size = 0;     // Pinned host pool bytes for transfer staging (0=disabled)

    //-------------------------------------------------------------------------
    // CUDA configuration
    //-------------------------------------------------------------------------
    int device_id = 0;               // GPU device ID
    uint32_t block_size = 128;       // Threads per block
    bool use_async_transfer = true;  // Use async memory transfers
    bool profile_kernels = false;    // Enable detailed per-kernel profiling (higher overhead)

    //-------------------------------------------------------------------------
    // Calculate memory requirements for current configuration
    //-------------------------------------------------------------------------
    [[nodiscard]] size_t CalculateBatchMemory() const noexcept
    {
        size_t mem = 0;

        return mem;
    }

    [[nodiscard]] size_t CalculatePermanentMemory(
        size_t permanent_size
    ) const noexcept
    {
        size_t mem = 0;

        return mem;
    }

    //-------------------------------------------------------------------------
    // Validate configuration fits in available VRAM
    //-------------------------------------------------------------------------
    [[nodiscard]] bool ValidateMemory(
        size_t available_vram, size_t permanent_size
    ) const noexcept
    {
        size_t permanent = CalculatePermanentMemory(permanent_size);
        size_t batch = CalculateBatchMemory();
        size_t total = permanent + batch;

        // Leave 2GB safety margin for CUDA overhead
        constexpr size_t k_safety_margin = 2ULL << 30;

        return total + k_safety_margin <= available_vram;
    }

    //-------------------------------------------------------------------------
    // Print configuration
    //-------------------------------------------------------------------------
    void Print() const noexcept
    {
        std::cout << "=== GPU Configuration ===" << std::endl;
        std::cout << "Batch size (B):              " << batch_size << " reads" << std::endl;
        std::cout << "Device ID:                   " << device_id << std::endl;
        std::cout << "Block size:                  " << block_size << std::endl;
        std::cout << "Async transfer:              " << (use_async_transfer ? "Yes" : "No")
                  << std::endl;
        std::cout << "Stitch pipeline:             " << "Split (current)" << std::endl;

        size_t batch_mem = CalculateBatchMemory();
        std::cout << "Estimated batch memory:      " << std::fixed << std::setprecision(2)
                  << (batch_mem / (1024.0 * 1024.0)) << " MB" << std::endl;
        std::cout << "=========================" << std::endl;
    }
};

}  // namespace mpbootgpu

#endif  // MPBOOTGPU_GPU_CONFIG_CUH_
