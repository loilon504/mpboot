#include <iomanip>
#include <iostream>

#include "gpu/include/memory_pool.cuh"

namespace mpbootgpu
{

//=============================================================================
// pinnedMemoryPool move operations
//=============================================================================

pinnedMemoryPool::pinnedMemoryPool(
    pinnedMemoryPool&& other
) noexcept
    : pool_(other.pool_), pool_size_(other.pool_size_), used_(other.used_)
{
    other.pool_ = nullptr;
    other.pool_size_ = 0;
    other.used_ = 0;
}

pinnedMemoryPool& pinnedMemoryPool::operator=(
    pinnedMemoryPool&& other
) noexcept
{
    if (this != &other)
    {
        shutdown();
        pool_ = other.pool_;
        pool_size_ = other.pool_size_;
        used_ = other.used_;
        other.pool_ = nullptr;
        other.pool_size_ = 0;
        other.used_ = 0;
    }
    return *this;
}

//=============================================================================
// gpuMemoryManager implementation
//=============================================================================

void gpuMemoryManager::printMemoryUsage() const
{
    CUDA_CHECK(cudaSetDevice(device_id_));

    auto mb = [](size_t bytes)
    {
        return bytes / (1024.0 * 1024.0);
    };
    auto gb = [](size_t bytes)
    {
        return bytes / (1024.0 * 1024.0 * 1024.0);
    };

    size_t free_vram, total_vram;
    cudaMemGetInfo(&free_vram, &total_vram);

    std::cout << "=== GPU Memory Usage ===" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Device " << device_id_ << " VRAM:" << std::endl;
    std::cout << "  Total:           " << gb(total_vram) << " GB" << std::endl;
    std::cout << "  Free:            " << gb(free_vram) << " GB" << std::endl;
    std::cout << "  Used:            " << gb(total_vram - free_vram) << " GB" << std::endl;
    std::cout << std::endl;

    std::cout << "Permanent Pool:" << std::endl;
    std::cout << "  Size:            " << mb(permanent_pool_.getSize()) << " MB" << std::endl;
    std::cout << "  Used:            " << mb(permanent_pool_.getUsed()) << " MB" << std::endl;
    std::cout << "  Peak:            " << mb(permanent_pool_.getPeakUsed()) << " MB" << std::endl;
    std::cout << "  Remaining:       " << mb(permanent_pool_.getRemaining()) << " MB" << std::endl;
    std::cout << std::endl;

    std::cout << "Batch Pool:" << std::endl;
    std::cout << "  Size:            " << mb(batch_pool_.getSize()) << " MB" << std::endl;
    std::cout << "  Used:            " << mb(batch_pool_.getUsed()) << " MB" << std::endl;
    std::cout << "  Peak:            " << mb(batch_pool_.getPeakUsed()) << " MB" << std::endl;
    std::cout << "  Remaining:       " << mb(batch_pool_.getRemaining()) << " MB" << std::endl;

    if (pinned_pool_.isInitialized())
    {
        std::cout << std::endl;
        std::cout << "Pinned Host Pool:" << std::endl;
        std::cout << "  Size:            " << mb(pinned_pool_.getSize()) << " MB" << std::endl;
        std::cout << "  Used:            " << mb(pinned_pool_.getUsed()) << " MB" << std::endl;
        std::cout << "  Remaining:       " << mb(pinned_pool_.getRemaining()) << " MB" << std::endl;
    }

    std::cout << "========================" << std::endl;
}

}  // namespace mpbootgpu
