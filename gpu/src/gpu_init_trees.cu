#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "gpu/include/gpu_init_trees.cuh"
#include "gpu/include/pars_bootstrap.cuh"
#include "gpu/include/pars_build.cuh"
#include "gpu/include/pars_tree.cuh"
#include "gpu/include/utils.cuh"
#include "iqtree.h"
#include "pllrepo/src/pll.h"
#include "sprparsimony.h"  // pllInstanceClone, pllPartitionsClone, _allocateParsimony*
#include "tools.h"

// Routes GPU log output through cout so it appears in MPBoot's .log file
// (printf() bypasses the cout→log tee established by outstreambuf in pda.cpp).
#define GPU_LOG(...)                              \
    do {                                          \
        char _gpu_buf[512];                       \
        snprintf(_gpu_buf, sizeof(_gpu_buf), __VA_ARGS__); \
        std::cout << _gpu_buf;                    \
    } while (0)

namespace mpbootgpu
{

// Returns elapsed ms since t0.
static inline double msSince(
    std::chrono::high_resolution_clock::time_point t0
)
{
    return std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0)
        .count();
}

// ─── Main entry point ─────────────────────────────────────────────────────────
int mpbootGpu(
    const Params& params,
    IQTree& iqtree,
    int numInitTrees,
    std::vector<std::string>& candidateTrees,
    GpuParsimonyMem** out_mem
)
{
    struct StdoutUnbuf
    {
        StdoutUnbuf()
        {
            setvbuf(stdout, nullptr, _IONBF, 0);
        }
        ~StdoutUnbuf()
        {
            setvbuf(stdout, nullptr, _IOLBF, 0);
        }
    } _stdout_unbuf;
    int K = numInitTrees;

    pllInstance* tr = iqtree.pllInst;
    partitionList* pr = iqtree.pllPartitions;
    const int mxtips = tr->mxtips;

    CUDA_CHECK(cudaSetDevice(params.gpu_device));
    GPU_LOG("\n[GPU] ═══════════════════════════════════════════════════════\n");
    GPU_LOG("[GPU]   K=%d  N=%d  device=%d\n", K, mxtips, params.gpu_device);
    GPU_LOG("[GPU]\n");

    // ── [1] Allocate CPU parsVect & read metadata ─────────────────────────────
    auto t0 = std::chrono::high_resolution_clock::now();
    _allocateParsimonyDataStructures(tr, pr, PLL_FALSE);
    const int width = (int)pr->partitionData[0]->parsimonyLength;
    const int states = (int)pr->partitionData[0]->states;
    GPU_LOG(
        "[GPU]   [1]      %-28s: %8.3f s  (width=%d states=%d)\n", "CPU parsimony alloc",
        msSince(t0) / 1e3, width, states
    );

    // K2 workers computed early — needed for memory allocation size
    const int k2_workers_early = (params.gpu_worker > 0) ? params.gpu_worker : K;
    const int K_alloc = std::max(K, k2_workers_early);

    // ── [2] Allocate GPU memory ───────────────────────────────────────────────
    t0 = std::chrono::high_resolution_clock::now();
    // Treels buffer: bootstrap K2 writes all good trees each round for REPS eval.
    // Size = K * 10 (up to ~10 outer iters per K2 round × K workers).
    const int max_treels_boot = K * 10;  // always allocate; gpuHillClimbing needs it
    GpuParsimonyMem* mem = gpuParsimonyMemAlloc(
        K_alloc, mxtips, width, states, params.gpu_pool_size, max_treels_boot
    );
    GPU_LOG("[GPU]   [2]      %-28s: %8.3f s\n", "GPU memory alloc", msSince(t0) / 1e3);

    // ── [3] Upload tip parsVect ───────────────────────────────────────────────
    cudaStream_t stream = 0;
    t0 = std::chrono::high_resolution_clock::now();
    uploadTipParsVect(mem, tr, pr, stream);
    GPU_LOG("[GPU]   [3]      %-28s: %8.3f s\n", "Upload tip parsVect (H->D)", msSince(t0) / 1e3);

    // ── [4] Upload initial topologies ─────────────────────────────────────────
    t0 = std::chrono::high_resolution_clock::now();
    {
        GpuTopology h_topo;
        cpuToGpuTopology(tr, &h_topo);
        for (int k = 0; k < K; ++k)
        {
            uploadTopology(mem, k, &h_topo, stream);
        }
    }
    GPU_LOG(
        "[GPU]   [4]      %-28s: %8.3f s  (%d trees)\n", "Upload topologies (H->D)",
        msSince(t0) / 1e3, K
    );
    GPU_LOG("[GPU]\n");

    // Free CPU parsVect — not needed after GPU upload
    _pllFreeParsimonyDataStructures(tr, pr);

    // Build seeds
    std::vector<long> seeds(K);
    for (int i = 0; i < K; ++i)
    {
        seeds[i] = (long)(params.ran_seed) + (long)(i + 1) * 12345L;
    }

    // ── [5] GPU stepwise-addition kernel ─────────────────────────────────────
    auto ev_time = [&](auto fn) -> float
    {
        cudaEvent_t a, b;
        cudaEventCreate(&a);
        cudaEventCreate(&b);
        cudaEventRecord(a, stream);
        fn();
        cudaEventRecord(b, stream);
        cudaEventSynchronize(b);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, a, b);
        cudaEventDestroy(a);
        cudaEventDestroy(b);
        return ms;
    };

    // ── [5+6+7] Joined kernel: build + initial SPR + iterative NNI+SPR ─────────
    // Phase 4 params
    const int numNNI = (mxtips > 4) ? max(1, (int)(params.gpu_nni_strength * (mxtips - 3))) : 1;
    const int pool_size = params.gpu_pool_size;
    // K2 workers: independent of K1; mem->K = K_alloc = max(K, k2_workers)
    const int k2_workers = k2_workers_early;

    // ── Hybrid callback: CPU builds parsimony trees while K1 runs on GPU ────────
    AfterK1Callback hybrid_cb = nullptr;
    {
        hybrid_cb = [&](cudaStream_t cb_stream, GpuParsimonyMem* cb_mem) -> void
        {
            iqtree.candidateTrees.aln = iqtree.aln;
            // Step 1: Build CPU trees while K1 runs on GPU
            std::vector<CpuTreeData> cpu_trees;
            while (cudaStreamQuery(cb_stream) == cudaErrorNotReady)
            {
                std::string curParsTree;
                tr->randomNumberSeed = params.ran_seed + (int)cpu_trees.size() * 7919;
                _pllComputeRandomizedStepwiseAdditionParsimonyTree(
                    tr, pr, params.sprDist + 1, &iqtree
                );
                pllTreeToNewick(
                    iqtree.pllInst->tree_string, iqtree.pllInst, iqtree.pllPartitions,
                    iqtree.pllInst->start->back, PLL_TRUE, PLL_TRUE, PLL_FALSE, PLL_FALSE,
                    PLL_FALSE, PLL_SUMMARIZE_LH, PLL_FALSE, PLL_FALSE
                );
                curParsTree = string(iqtree.pllInst->tree_string);

                if (iqtree.candidateTrees.treeExist(curParsTree))
                {
                    continue;
                }
                iqtree.readTreeString(curParsTree);

                iqtree.initializeAllPartialPars();
                iqtree.clearAllPartialLH();
                iqtree.curScore = -iqtree.computeParsimony();
                iqtree.candidateTrees.update(curParsTree, iqtree.curScore);
                if (iqtree.curScore > iqtree.bestScore)
                {
                    iqtree.setBestTree(curParsTree, iqtree.curScore);
                }

                // code for GPU
                unsigned int score = -iqtree.curScore;

                CpuTreeData ct = {};  // zero-init: nodep[] must not be garbage before upload
                ct.score = score;
                cpuToGpuTopology(tr, &ct.topo);
                ct.topo.preSprParsimony = score;
                ct.topo.postSprParsimony = score;
                ct.topo.bestParsimony = score;
                ct.topo.savedSeed = params.ran_seed + (long)cpu_trees.size() * 31337L;
                cpu_trees.push_back(ct);
            }
            CUDA_CHECK(cudaStreamSynchronize(cb_stream));  // K1 fully done

            // Step 2: Download K1 GPU postSprScores (K1 builds all K trees)
            const int Kc_scores = K;         // all K slots written by K1
            const int Kc_full = k2_workers;  // reseed k2_workers slots for K2
            std::vector<unsigned int> gpu_scores(Kc_scores);
            CUDA_CHECK(cudaMemcpy(
                gpu_scores.data(), cb_mem->d_postSprScores,
                (size_t)Kc_scores * sizeof(unsigned int), cudaMemcpyDeviceToHost
            ));

            // Step 3: Diversity selection — best pool_size trees with distinct topologies
            struct Candidate
            {
                unsigned int score;
                int idx;
                bool is_gpu;
            };
            std::vector<Candidate> candidates;
            candidates.reserve(Kc_scores + (int)cpu_trees.size());
            for (int k = 0; k < Kc_scores; k++)
            {
                candidates.push_back({gpu_scores[k], k, true});
            }
            for (int i = 0; i < (int)cpu_trees.size(); i++)
            {
                candidates.push_back({cpu_trees[i].score, i, false});
            }
            std::sort(
                candidates.begin(), candidates.end(),
                [](const Candidate& a, const Candidate& b)
                {
                    return a.score < b.score;
                }
            );

            int scan_limit = std::min((int)candidates.size(), pool_size * 5);
            std::unordered_set<std::string> seen_topologies;
            std::vector<int> pool_ci;                // indices into candidates[]
            std::vector<std::string> pool_newicks;   // cached Newicks (for Step 6)
            std::vector<std::vector<int>> pool_bvf;  // back_vf arrays (for pool upload + reseed)

            for (int ci = 0; ci < scan_limit && (int)pool_ci.size() < pool_size; ci++)
            {
                const Candidate& cand = candidates[ci];
                std::string newick;
                std::vector<int> bvf;

                if (cand.is_gpu)
                {
                    GpuTopology h_topo;
                    downloadTopology(cb_mem, cand.idx, &h_topo, cb_stream);
                    cudaStreamSynchronize(cb_stream);
                    bvf.assign(h_topo.back_vf, h_topo.back_vf + h_topo.num_vfaces);
                    gpuTopoToCpu(&h_topo, tr);
                }
                else
                {
                    GpuTopology cpu_topo_copy = cpu_trees[cand.idx].topo;
                    bvf.assign(
                        cpu_topo_copy.back_vf, cpu_topo_copy.back_vf + cpu_topo_copy.num_vfaces
                    );
                    gpuTopoToCpu(&cpu_topo_copy, tr);
                }
                pllTreeToNewick(
                    tr->tree_string, tr, pr, tr->start->back, PLL_TRUE, PLL_TRUE, PLL_FALSE,
                    PLL_FALSE, PLL_FALSE, PLL_SUMMARIZE_LH, PLL_FALSE, PLL_FALSE
                );
                newick = std::string(tr->tree_string);
                if (newick.empty() || !seen_topologies.insert(newick).second)
                {
                    continue;
                }

                pool_ci.push_back(ci);
                pool_newicks.push_back(std::move(newick));
                pool_bvf.push_back(std::move(bvf));
            }
            int actual_pool = (int)pool_ci.size();
            int pool_from_gpu = 0, pool_from_cpu = 0;
            for (int i = 0; i < actual_pool; i++)
                (candidates[pool_ci[i]].is_gpu ? pool_from_gpu : pool_from_cpu)++;

            // Step 4: Upload pool to GPU memory
            {
                std::vector<unsigned int> h_pool_scores(pool_size, 0xFFFFFFFFu);
                std::vector<int> h_pool_vf((size_t)pool_size * kMaxVFaces, -1);
                for (int i = 0; i < actual_pool; i++)
                {
                    h_pool_scores[i] = candidates[pool_ci[i]].score;
                    int* dst = h_pool_vf.data() + (size_t)i * kMaxVFaces;
                    memcpy(dst, pool_bvf[i].data(), pool_bvf[i].size() * sizeof(int));
                }
                CUDA_CHECK(cudaMemcpy(
                    cb_mem->d_poolScores, h_pool_scores.data(), pool_size * sizeof(unsigned int),
                    cudaMemcpyHostToDevice
                ));
                CUDA_CHECK(cudaMemcpy(
                    cb_mem->d_poolBackVf, h_pool_vf.data(),
                    (size_t)pool_size * kMaxVFaces * sizeof(int), cudaMemcpyHostToDevice
                ));
                int show = std::min(actual_pool, 5);
                GPU_LOG("[GPU]   Pool: %d/%d slots, top-%d scores:", actual_pool, pool_size, show);
                for (int i = 0; i < show; i++)
                {
                    GPU_LOG(" %u", h_pool_scores[i]);
                }
                GPU_LOG("\n");
            }

            // Step 4b: Init fill counter + per-slot spinlocks + hashes
            {
                int h_filled = actual_pool;
                unsigned int h_inf = 0xFFFFFFFFu;
                CUDA_CHECK(
                    cudaMemcpy(cb_mem->d_poolFilled, &h_filled, sizeof(int), cudaMemcpyHostToDevice)
                );
                CUDA_CHECK(
                    cudaMemset(cb_mem->d_poolSlotLocks, 0, (size_t)cb_mem->pool_size * sizeof(int))
                );
                CUDA_CHECK(cudaMemset(
                    cb_mem->d_poolHashes, 0xFF, (size_t)cb_mem->pool_size * sizeof(unsigned int)
                ));
                CUDA_CHECK(cudaMemcpy(
                    cb_mem->d_globalBest, &h_inf, sizeof(unsigned int), cudaMemcpyHostToDevice
                ));
            }

            // Step 5: Reseed ALL K GPU slots from pool (round-robin)
            if (actual_pool > 0)
            {
                GpuTopology tmpl;
                downloadTopology(cb_mem, 0, &tmpl, cb_stream);
                cudaStreamSynchronize(cb_stream);
                for (int k = 0; k < Kc_full; k++)
                {
                    int slot = k % actual_pool;
                    const auto& bvf = pool_bvf[slot];
                    unsigned int ps = candidates[pool_ci[slot]].score;
                    GpuTopology h_topo = tmpl;
                    memcpy(h_topo.back_vf, bvf.data(), bvf.size() * sizeof(int));
                    h_topo.postSprParsimony = ps;
                    h_topo.bestParsimony = ps;
                    h_topo.preSprParsimony = ps;
                    h_topo.savedSeed = params.ran_seed + (long)k * 31337L;
                    h_topo.n_improved_even = h_topo.n_improved_odd = 0;
                    h_topo.n_total_even = h_topo.n_total_odd = 0;
                    uploadTopology(cb_mem, k, &h_topo, cb_stream);
                }
                cudaStreamSynchronize(cb_stream);
            }

            GPU_LOG(
                "[CPU]   CPU built %d trees; pool: %d/%d (GPU=%d CPU=%d)\n",
                (int)cpu_trees.size(), actual_pool, pool_size, pool_from_gpu, pool_from_cpu
            );
        };
    };

    float build_ms = ev_time(
        [&]
        {
            // gpuHillClimbing runs the K2 outer loop after mpbootGpu returns.
            const int k2_max_outer = 0;
            gpuStepwiseBuildTrees(
                mem, seeds.data(), K, params.sprDist, numNNI, pool_size, stream, hybrid_cb, nullptr,
                k2_workers, k2_max_outer
            );
        }
    );
    GPU_LOG(
        "[GPU]   [5]  %-28s: %8.3f s  (%d trees, %.2f ms/tree)\n", "Kernels",
        (double)build_ms / 1e3, K, K > 0 ? (double)build_ms / K : 0.0
    );
    GPU_LOG(
        "[GPU]            sprDist=%d  NNI=%d(%.2f)  pool_size=%d\n", params.sprDist, numNNI,
        params.gpu_nni_strength, pool_size
    );
    GPU_LOG("\n");

    // Pool → candidateTrees is handled at end of gpuHillClimbing.
    int built = 0;
    GPU_LOG("[GPU] ═══════════════════════════════════════════════════════\n\n");

    if (out_mem != nullptr)
    {
        *out_mem = mem;  // caller takes ownership, responsible for gpuParsimonyMemFree
    }
    else
    {
        gpuParsimonyMemFree(mem);
    }
    return built;
}

// ─── gpuHillClimbing ─────────────────────────────────────────────────────────
// GPU iterative hill-climbing outer loop. Works in two modes:
//   Bootstrap (-bb): per-round K2 from pool, treels → saveCurrentTree (REPS), convergence check.
//   Non-bootstrap:   per-round K2 from pool, treels → candidateTrees, stop on no improvement.
void gpuHillClimbing(
    const Params& params, IQTree& iqtree, GpuParsimonyMem* mem
)
{
    cudaStream_t stream = 0;
    const bool is_bootstrap = (params.gbo_replicates > 0);
    const int K = mem->K;  // mem->K = max(numInitTrees, gpu_worker) at alloc time
    // Actual K2 workers: clamped to [1, K]. Allows -gpu_worker < numInitTrees.
    const int k2_workers = (params.gpu_worker > 0 && params.gpu_worker <= K)
                               ? params.gpu_worker : K;
    const int B = params.gbo_replicates;  // used only when is_bootstrap
    const char* const tag = is_bootstrap ? "[GPU Bootstrap]" : "[GPU HillClimb]";
    const int sprDist = params.sprDist;
    const int numNNI = std::max(1, (int)(params.gpu_nni_strength * (mem->mxtips - 3)));
    const int pool_size = mem->pool_size;
    const int width = mem->width;
    const int step_iter = params.step_iterations;
    // Convergence check every step_iter_rounds rounds (mirrors CPU every step_iter/2 iters).
    const int step_iter_rounds = std::max(1, std::max(1, step_iter / 2) / k2_workers);

    iqtree.params->store_candidate_trees = true;

    // Template topology for treels reconstruction (back_vf only stored in treels;
    // other fields come from here). start_vface=0 → tr->start=tip1; start->back = valid Newick
    // root.
    GpuTopology h_tpl;
    downloadTopology(mem, 0, &h_tpl, stream);
    h_tpl.start_vface = 0;
    memset(h_tpl.xpars, 0, sizeof(h_tpl.xpars));

    // Reusable host buffer for batch-downloading treels back_vf each round.
    const int max_treels = mem->max_treels;
    std::vector<int> h_treels_bvf;
    if (max_treels > 0)
    {
        h_treels_bvf.resize((size_t)max_treels * kMaxVFaces);
    }

    double cur_correlation = 0.0;
    int round = 0, total_done = 0;
    double best_logl_seen  = -1e30;
    int total_replicates   = 0;
    int last_impr_at       = 0;   // total_replicates at last improvement
    const int unsuccess_thresh = params.unsuccess_iteration + k2_workers * params.gpu_worker_stop;

    if (is_bootstrap)
        GPU_LOG("%s K2-treels: B=%d K=%d pool=%d unsuccess=%d\n", tag, B, k2_workers, pool_size, unsuccess_thresh);
    else
        GPU_LOG("%s K2-hillclimb: K=%d pool=%d unsuccess=%d\n", tag, k2_workers, pool_size, unsuccess_thresh);
    fflush(stdout);

    for (;;)
    {
        auto t_round = std::chrono::high_resolution_clock::now();

        // ── Reset treels with current logl_cutoff → GPU filters bad trees ────────
        // Bootstrap: logl_cutoff < 0 (= -parsimony); GPU cutoff = unsigned parsimony threshold.
        // Non-bootstrap or not yet activated: accept all (0xFFFFFFFF).
        const unsigned int boot_cutoff = (is_bootstrap && iqtree.logl_cutoff != 0.0)
                                             ? (unsigned int)(-(double)iqtree.logl_cutoff)
                                             : 0xFFFFFFFFu;
        resetTreelsRound(mem, boot_cutoff);
        resetPoolRound(mem);

        // ── Run K2 from pool (k1_count=0 skips K1) ───────────────────────────
        gpuStepwiseBuildTrees(
            mem, nullptr, /*k1_count=*/0, sprDist, numNNI, pool_size, stream, nullptr, nullptr,
            k2_workers, /*max_outer_iters=*/1
        );

        // ── Download treels → REPS eval ───────────────────────────────────────
        int h_filled = 0;
        CUDA_CHECK(cudaMemcpy(&h_filled, mem->d_treelsFilled, sizeof(int), cudaMemcpyDeviceToHost));
        const int n_treels = std::min(h_filled, max_treels);

        if (n_treels > 0 && max_treels > 0)
        {
            CUDA_CHECK(cudaMemcpy(
                h_treels_bvf.data(), mem->d_treelsBackVf,
                (size_t)n_treels * kMaxVFaces * sizeof(int), cudaMemcpyDeviceToHost
            ));

            GpuTopology h_topo = h_tpl;
            for (int t = 0; t < n_treels; t++)
            {
                memcpy(
                    h_topo.back_vf, h_treels_bvf.data() + (size_t)t * kMaxVFaces,
                    h_tpl.num_vfaces * sizeof(int)
                );

                gpuTopoToCpu(&h_topo, iqtree.pllInst);
                pllTreeToNewick(
                    iqtree.pllInst->tree_string, iqtree.pllInst, iqtree.pllPartitions,
                    iqtree.pllInst->start->back, PLL_TRUE, PLL_TRUE, PLL_FALSE, PLL_FALSE,
                    PLL_FALSE, PLL_SUMMARIZE_LH, PLL_FALSE, PLL_FALSE
                );
                std::string newick(iqtree.pllInst->tree_string);
                if (newick.empty())
                {
                    continue;
                }

                iqtree.readTreeString(newick);
                iqtree.initializeAllPartialPars();
                iqtree.clearAllPartialLH();
                int pars = iqtree.computeParsimony();

                if (is_bootstrap)
                {
                    bool saved = iqtree.params->spr_parsimony;
                    iqtree.params->spr_parsimony = false;
                    iqtree.saveCurrentTree(-(double)pars);
                    iqtree.params->spr_parsimony = saved;
                }
                else
                {
                    iqtree.curScore = -(double)pars;
                    bool isNew = iqtree.candidateTrees.update(newick, iqtree.curScore);
                    if (isNew && iqtree.curScore > iqtree.bestScore)
                    {
                        iqtree.setBestTree(newick, iqtree.curScore);
                    }
                }
            }
        }

        total_done += k2_workers;
        round++;

        // ── Track global best improvement ─────────────────────────────────────
        total_replicates += n_treels;
        if (is_bootstrap)
        {
            if (!iqtree.treels_logl.empty())
            {
                double cur_best = *std::max_element(
                    iqtree.treels_logl.begin(), iqtree.treels_logl.end()
                );
                if (cur_best > best_logl_seen + 1e-6)
                {
                    best_logl_seen = cur_best;
                    last_impr_at   = total_replicates;
                }
            }
        }
        else
        {
            if (iqtree.bestScore > best_logl_seen + 1e-6)
            {
                best_logl_seen = iqtree.bestScore;
                last_impr_at   = total_replicates;
            }
        }

        if (is_bootstrap)
        {
            // ── Update logl_cutoff (top cutoff_percent% threshold) ────────────
            if (iqtree.treels_logl.size() > 0)
            {
                DoubleVector logl = iqtree.treels_logl;
                nth_element(
                    logl.begin(), logl.begin() + logl.size() * params.cutoff_percent / 100,
                    logl.end(), std::greater<double>()
                );
                iqtree.logl_cutoff = logl[logl.size() * params.cutoff_percent / 100];
            }

            // ── Convergence check ─────────────────────────────────────────────
            if (round % step_iter_rounds == 0)
            {
                SplitGraph* sg = new SplitGraph;
                iqtree.summarizeBootstrap(*sg);
                iqtree.boot_splits.push_back(sg);
                while (iqtree.boot_splits.size() > 2)
                {
                    delete iqtree.boot_splits.front();
                    iqtree.boot_splits.erase(iqtree.boot_splits.begin());
                }
                if (iqtree.boot_splits.size() >= 2)
                {
                    cur_correlation = iqtree.computeBootstrapCorrelation();
                }
            }
        }

        double round_sec = msSince(t_round) / 1e3;
        if (is_bootstrap)
        {
            GPU_LOG(
                "%s Round %-3d  done=%-5d  filled=%-5d  treels=%-5zu  reps=%-5d  "
                "last_impr=%-5d  cor=%.4f  t=%.2fs\n",
                tag, round, total_done, h_filled, iqtree.treels_logl.size(), total_replicates,
                last_impr_at, cur_correlation, round_sec
            );
        }
        else
        {
            GPU_LOG(
                "%s Round %-3d  done=%-5d  filled=%-5d  reps=%-5d  "
                "last_impr=%-5d  best=%.0f  t=%.2fs\n",
                tag, round, total_done, h_filled, total_replicates,
                last_impr_at, iqtree.bestScore, round_sec
            );
        }
        fflush(stdout);

        if (total_replicates - last_impr_at > unsuccess_thresh
            || (is_bootstrap && total_replicates > B))
        {
            break;
        }
    }

    if (is_bootstrap)
        GPU_LOG("%s Done: %d rounds, %d replicates, cor=%.4f\n", tag, round, total_done, cur_correlation);
    else
        GPU_LOG("%s Done: %d rounds, %d replicates, best=%.0f\n", tag, round, total_done, iqtree.bestScore);

    // ── Add GPU pool topologies to candidateTrees ────────────────────────────
    // Pool holds near-optimal topologies after K2 rounds. Register them so the
    // final best-score report uses GPU-found trees (bootstrap and non-bootstrap).
    {
        pllInstance* tr = iqtree.pllInst;
        partitionList* pr = iqtree.pllPartitions;
        const int ps = mem->pool_size;

        std::vector<unsigned int> h_ps(ps, 0xFFFFFFFFu);
        std::vector<int> h_pvf((size_t)ps * kMaxVFaces);
        CUDA_CHECK(cudaMemcpy(
            h_ps.data(), mem->d_poolScores, ps * sizeof(unsigned int), cudaMemcpyDeviceToHost
        ));
        CUDA_CHECK(cudaMemcpy(
            h_pvf.data(), mem->d_poolBackVf, (size_t)ps * kMaxVFaces * sizeof(int),
            cudaMemcpyDeviceToHost
        ));

        // Find best pool score to filter slots
        unsigned int best_pool = 0xFFFFFFFFu;
        for (int i = 0; i < ps; i++)
        {
            if (h_ps[i] < best_pool)
            {
                best_pool = h_ps[i];
            }
        }

        GpuTopology tmpl;
        downloadTopology(mem, 0, &tmpl, stream);
        cudaStreamSynchronize(stream);
        tmpl.start_vface = 0;
        memset(tmpl.xpars, 0, sizeof(tmpl.xpars));

        int n_pool_added = 0;
        for (int i = 0; i < ps; i++)
        {
            if (h_ps[i] == 0xFFFFFFFFu)
            {
                continue;
            }

            GpuTopology h_topo = tmpl;
            memcpy(
                h_topo.back_vf, h_pvf.data() + (size_t)i * kMaxVFaces, tmpl.num_vfaces * sizeof(int)
            );
            h_topo.bestParsimony = h_ps[i];

            gpuTopoToCpu(&h_topo, tr);
            pllTreeToNewick(
                tr->tree_string, tr, pr, tr->start->back, PLL_TRUE, PLL_TRUE, PLL_FALSE, PLL_FALSE,
                PLL_FALSE, PLL_SUMMARIZE_LH, PLL_FALSE, PLL_FALSE
            );
            std::string tree_str(tr->tree_string);
            if (tree_str.empty())
            {
                continue;
            }

            iqtree.readTreeString(tree_str);
            iqtree.initializeAllPartialPars();
            iqtree.clearAllPartialLH();
            iqtree.curScore = -(double)iqtree.computeParsimony();
            bool isNew = iqtree.candidateTrees.update(tree_str, iqtree.curScore);
            if (isNew && iqtree.curScore > iqtree.bestScore)
            {
                iqtree.setBestTree(tree_str, iqtree.curScore);
            }
            n_pool_added++;
        }
        GPU_LOG(
            "%s Pool → candidateTrees: %d/%d slots added, best_pool=%u\n", tag,
            n_pool_added, ps, best_pool
        );
    }
}

}  // namespace mpbootgpu
