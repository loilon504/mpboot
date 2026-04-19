#include "gpu/include/gpu_instance.cuh"

namespace mpbootgpu
{

gpuInstance::gpuInstance(
    const gpuConfig& config
)
    : config_(config)
{
    memory_manager_ = std::make_unique<gpuMemoryManager>();
}

gpuInstance::~gpuInstance()
{
    // CUDA resources are device-scoped; destroy streams on the configured device.
    cudaSetDevice(config_.device_id);
    if (batch_pool_safe_event_ != nullptr)
    {
        cudaEventDestroy(batch_pool_safe_event_);
        batch_pool_safe_event_ = nullptr;
    }
    if (compute_stream_)
    {
        cudaStreamDestroy(compute_stream_);
    }
    if (h2d_stream_)
    {
        cudaStreamDestroy(h2d_stream_);
    }
    if (d2h_stage_stream_)
    {
        cudaStreamDestroy(d2h_stage_stream_);
    }
    for (auto& copy_stream : d2h_copy_streams_)
    {
        if (copy_stream != nullptr)
        {
            cudaStreamDestroy(copy_stream);
            copy_stream = nullptr;
        }
    }
}

void gpuInstance::ensureDevice() const
{
    CUDA_CHECK(cudaSetDevice(config_.device_id));
}

void destroyEventIfSet(
    cudaEvent_t& evt
)
{
    if (evt != nullptr)
    {
        cudaEventDestroy(evt);
        evt = nullptr;
    }
}

//=============================================================================
// Phase 1: Allocate Memory Pools (can run parallel with genome loading)
//=============================================================================
void gpuInstance::allocateMemoryPools(
    size_t genome_size
)
{
//     if (pools_allocated_)
//     {
//         spdlog::warn("Memory pools already allocated, skipping");
//         return;
//     }

//     spdlog::info("Allocating GPU memory pools...");

//     // Set device
//     ensureDevice();
//     CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 100 * 1024 * 1024));

//     // Query device properties
//     cudaDeviceProp prop;
//     CUDA_CHECK(cudaGetDeviceProperties(&prop, config_.device_id));
//     spdlog::info(
//         "GPU Device: {} ({} GB VRAM)", prop.name, prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0)
//     );

//     // Calculate memory requirements
//     size_t permanent_size = config_.CalculatePermanentMemory(genome_size);
//     size_t batch_size = config_.CalculateBatchMemory();

//     spdlog::info("Permanent pool: {:.2f} MB", permanent_size / (1024.0 * 1024.0));
//     spdlog::info("Batch pool: {:.2f} MB", batch_size / (1024.0 * 1024.0));
//     // Initialize memory manager (this is the expensive part: ~3-4s)
//     constexpr size_t k_pinned_pool_cap_bytes = 1ULL << 30;  // 1 GB cap
//     const size_t pinned_size = std::min(config_.pinned_pool_size, k_pinned_pool_cap_bytes);
//     if (config_.pinned_pool_size > k_pinned_pool_cap_bytes)
//     {
//         spdlog::warn(
//             "gpuPinnedPoolMB request exceeds cap, clamped to {:.2f} MB",
//             k_pinned_pool_cap_bytes / (1024.0 * 1024.0)
//         );
//     }
//     spdlog::info(
//         "Pinned pool request: {:.2f} MB (config --gpuPinnedPoolMB, cap {:.2f} MB)",
//         pinned_size / (1024.0 * 1024.0), k_pinned_pool_cap_bytes / (1024.0 * 1024.0)
//     );
//     memory_manager_->Initialize(config_.device_id, permanent_size, batch_size, pinned_size);

//     // Create CUDA streams
//     CUDA_CHECK(cudaStreamCreate(&compute_stream_));
//     CUDA_CHECK(cudaStreamCreate(&h2d_stream_));
//     CUDA_CHECK(cudaStreamCreate(&d2h_stage_stream_));
//     for (auto& copy_stream : d2h_copy_streams_)
//     {
//         CUDA_CHECK(cudaStreamCreate(&copy_stream));
//     }
//     CUDA_CHECK(cudaEventCreateWithFlags(&batch_pool_safe_event_, cudaEventDisableTiming));
//     batch_pool_safe_pending_ = false;

//     // Output staging device buffers (multi-slot ring).
//     const size_t stage_counts_bytes = static_cast<size_t>(config_.batch_size) * sizeof(uint32_t);
//     const size_t stage_offsets_bytes = (static_cast<size_t>(config_.batch_size) + 1)
//                                        * sizeof(Index);
//     const size_t max_stage_windows = static_cast<size_t>(config_.batch_size)
//                                      * static_cast<size_t>(config_.max_windows_per_read);
//     if (max_stage_windows > (std::numeric_limits<size_t>::max)() / sizeof(DeviceWindow))
//     {
//         throw std::runtime_error("Output staging window size overflow");
//     }
//     const size_t stage_window_bytes = max_stage_windows * sizeof(DeviceWindow);

//     for (auto& slot : output_stage_slots_)
//     {
//         CUDA_CHECK(cudaMalloc(&slot.counts, stage_counts_bytes));
//         CUDA_CHECK(cudaMalloc(&slot.offsets, stage_offsets_bytes));
//         CUDA_CHECK(cudaMalloc(&slot.windows, stage_window_bytes));
//         slot.in_flight = false;
//     }
//     output_stage_window_capacity_ = max_stage_windows;
//     output_stage_next_slot_ = 0;

//     const size_t total_stage_bytes
//         = kOutputStageSlots * (stage_counts_bytes + stage_offsets_bytes + stage_window_bytes);
//     spdlog::info(
//         "Output staging: {} slots (per-slot window buffer {:.2f} MB, total {:.2f} MB)",
//         kOutputStageSlots, static_cast<double>(stage_window_bytes) / (1024.0 * 1024.0),
//         static_cast<double>(total_stage_bytes) / (1024.0 * 1024.0)
//     );

//     pools_allocated_ = true;
//     spdlog::info("GPU memory pools allocated");
}

void gpuInstance::setConfig(
    const gpuConfig& config
)
{
    if (initialized_)
    {
        throw std::runtime_error("Cannot change config after initialization");
    }
    config_ = config;
}

void gpuInstance::printMemoryUsage() const
{
    if (memory_manager_)
    {
        ensureDevice();
        memory_manager_->printMemoryUsage();
    }
}

void gpuInstance::printPerformanceStats() const
{
    std::cout << "=== GPU Performance Stats ===" << std::endl;
    std::cout << "Pipeline mode:          window-only (CPU stitch)" << std::endl;
    // std::cout << "Total batches:          " << stats_.total_batches_processed << std::endl;
    // std::cout << "Total reads:            " << stats_.total_reads_processed << std::endl;
    // std::cout << std::fixed << std::setprecision(3);
    // std::cout << "Total transfer time:    " << stats_.total_transfer_time << " s (CPU->GPU)"
    //           << std::endl;
    // std::cout << "Total window time:      " << stats_.total_window_time << " s (kernel)"
    //           << std::endl;
    // std::cout << "Total output time:      " << stats_.total_output_time << " s (GPU->CPU)"
    //           << std::endl;

    // const double submit_breakdown_accounted = stats_.submit_wait_batch_pool_safe_time
    //                                           + stats_.submit_reset_batch_pool_time
    //                                           + stats_.submit_h2d_enqueue_time
    //                                           + stats_.submit_compute_enqueue_time
    //                                           + stats_.submit_wait_stage_slot_time
    //                                           + stats_.submit_stage_enqueue_time
    //                                           + stats_.submit_host_bookkeeping_time;
    // double submit_breakdown_other = stats_.submit_total_host_time - submit_breakdown_accounted;
    // if (submit_breakdown_other < 0)
    // {
    //     submit_breakdown_other = 0;
    // }

    // std::cout << "--- Submit Host Breakdown (ProcessBatchAsync) ---" << std::endl;
    // std::cout << "Submit host total:      " << stats_.submit_total_host_time << " s (wall time)"
    //           << std::endl;
    // std::cout << "  Wait pool-safe:       " << stats_.submit_wait_batch_pool_safe_time << " s"
    //           << std::endl;
    // std::cout << "  Reset batch pool:     " << stats_.submit_reset_batch_pool_time << " s"
    //           << std::endl;
    // std::cout << "  H2D submit/enqueue:   " << stats_.submit_h2d_enqueue_time << " s" << std::endl;
    // std::cout << "  Compute enqueue:      " << stats_.submit_compute_enqueue_time << " s"
    //           << std::endl;
    // std::cout << "  Wait stage slot:      " << stats_.submit_wait_stage_slot_time << " s"
    //           << std::endl;
    // std::cout << "  Stage enqueue:        " << stats_.submit_stage_enqueue_time << " s" << std::endl;
    // std::cout << "  Host bookkeeping:     " << stats_.submit_host_bookkeeping_time << " s"
    //           << std::endl;
    // std::cout << "  Unattributed:         " << submit_breakdown_other << " s" << std::endl;

    // if (stats_.total_batches_processed > 0)
    // {
    //     const double inv_batches = 1.0 / static_cast<double>(stats_.total_batches_processed);
    //     std::cout << "  Avg submit/batch:     "
    //               << stats_.submit_total_host_time * 1000.0 * inv_batches << " ms" << std::endl;
    //     std::cout << "  Avg stage-wait/batch: "
    //               << stats_.submit_wait_stage_slot_time * 1000.0 * inv_batches << " ms"
    //               << std::endl;
    // }

    // if (!config_.profile_kernels)
    // {
    //     std::cout << "--- Detailed Kernel Timing --- disabled (set --gpuProfileKernels)"
    //               << std::endl;
    // }
    // else
    // {
    //     std::cout << "--- Detailed Kernel Timing ---" << std::endl;
    //     std::cout << "  CreateWindowsKernel:        " << stats_.kernel_create_windows << " s"
    //               << std::endl;
    //     std::cout << "  BuildWindowOffsetsKernel:   " << stats_.kernel_build_window_offsets << " s"
    //               << std::endl;
    // }

    // // NOTE: these stage times are not wall-time additive under overlap.
    // const double total_stage_time = stats_.total_transfer_time + stats_.total_window_time
    //                                 + stats_.total_output_time;
    // if (total_stage_time > 0)
    // {
    //     std::cout << "Stage throughput:       " << std::fixed << std::setprecision(0)
    //               << stats_.total_reads_processed / total_stage_time << " reads/sec" << std::endl;
    // }
    std::cout << "=============================" << std::endl;
}



}  // namespace mpbootgpu