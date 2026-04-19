#ifndef MPBOOTGPU_GPU_INSTANCE_CUH_
#define MPBOOTGPU_GPU_INSTANCE_CUH_

#include <cuda_runtime.h>

#include <array>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <vector>

#include "gpu_config.cuh"
#include "memory_pool.cuh"
#include "utils.cuh"

namespace mpbootgpu
{

class gpuInstance
{
   public:
    explicit gpuInstance(const gpuConfig& config = gpuConfig());
    ~gpuInstance();
    gpuInstance(const gpuInstance&) = delete;
    gpuInstance& operator=(const gpuInstance&) = delete;

    // Allocate GPU memory pools
    void allocateMemoryPools(size_t size);

    // Check if pipeline is ready
    [[nodiscard]] bool isInitialized() const
    {
        return initialized_;
    }

    //-------------------------------------------------------------------------
    // Configuration and Stats
    //-------------------------------------------------------------------------

    [[nodiscard]] const gpuConfig& getConfig() const
    {
        return config_;
    }
    void setConfig(const gpuConfig& config);

    void printMemoryUsage() const;
    void printPerformanceStats() const;

    // Get optimal batch size for current memory configuration
    [[nodiscard]] uint32_t getOptimalBatchSize() const
    {
        return config_.batch_size;
    }

   private:
    void ensureDevice() const;

    gpuConfig config_;
    bool initialized_ = false;
    bool pools_allocated_ = false;  // Track if memory pools are allocated

    // Memory manager
    std::unique_ptr<gpuMemoryManager> memory_manager_;

    // Multi-slot output staging to overlap D2H of batch N with compute of later batches.
    static constexpr size_t kOutputStageSlots = 6;

    // CUDA streams
    cudaStream_t compute_stream_ = nullptr;
    cudaStream_t h2d_stream_ = nullptr;
    cudaStream_t d2h_stage_stream_ = nullptr;
    std::array<cudaStream_t, kOutputStageSlots> d2h_copy_streams_{};

    // Batch pool can be reset once stage copy has consumed batch output buffers.
    cudaEvent_t batch_pool_safe_event_ = nullptr;
    bool batch_pool_safe_pending_ = false;

    // Timing statistics
    struct
    {
        double total_transfer_time = 0;
        double total_seeding_time = 0;
        double total_window_time = 0;
        double total_stitch_time = 0;
        double total_output_time = 0;

        // Host-side breakdown for ProcessBatchAsync submit path.
        double submit_total_host_time = 0;
        double submit_wait_batch_pool_safe_time = 0;
        double submit_reset_batch_pool_time = 0;
        double submit_h2d_enqueue_time = 0;
        double submit_compute_enqueue_time = 0;
        double submit_wait_stage_slot_time = 0;
        double submit_stage_enqueue_time = 0;
        double submit_host_bookkeeping_time = 0;

        uint64_t total_batches_processed = 0;
    } stats_;
    mutable std::mutex stats_mutex_;
};

}  // namespace mpbootgpu

#endif  // MPBOOTGPU_GPU_INSTANCE_CUH_
