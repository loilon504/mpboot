#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <future>
#include <iostream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "gpu/include/gpu_init_trees.cuh"
#include "gpu/include/pars_bootstrap.cuh"
#include "gpu/include/pars_build.cuh"
#include "gpu/include/pars_tree.cuh"
#include "gpu/include/pars_treels.cuh"
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

extern Params *globalParam;

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

// Upload Sankoff tip cost vectors reading from PLL's yVector, filtering to the same informative
// patterns that compressSankoffDNA() uses (mirrors CPU Sankoff exactly).
// PLL bitmask encoding: A=bit0, C=bit1, G=bit2, T=bit3. State s compatible iff (bitmask>>s)&1.
// Undetermined (gap/N) = all bits set = (1<<states)-1.
static void uploadSankoffTipParsVect(
    GpuParsimonyMem* mem, const pllInstance* tr, const partitionList* pr, cudaStream_t stream)
{
    const int N = mem->mxtips;
    const int P = mem->width;  // = parsimonyLength (informative patterns, padded)
    const int states = mem->states;
    const size_t parsVT = mem->parsVectPerTree;
    const int lower = (int)pr->partitionData[0]->lower;
    const int upper = (int)pr->partitionData[0]->upper;
    // PLL encoding differs by data type:
    //   DNA: nuc IS a bitmask (A=1,C=2,G=4,T=8), undetermined=15=(1<<4)-1.
    //   AA:  nuc is a state INDEX 0..19, undetermined=22 (not a bitmask).
    // Detect by checking if undetermined == (1<<states)-1 (true for DNA, false for AA).
    // Do NOT use (1<<states)-1 directly — that overflows seen[256] for states=20.
    const unsigned int undetermined = (unsigned int)(pr->partitionData[0]->maxTipStates - 1);
    const bool is_bitmask_coded = (undetermined == ((1u << states) - 1u));

    std::vector<parsimonyNumber> h_buf(parsVT, (parsimonyNumber)kSankoffInf);

    // Replicate CPU isInformative() logic (with globalParam->sort_alignment check):
    // informative iff > 1 distinct bitmask values < undetermined appear across taxa.
    const bool all_informative = globalParam && !globalParam->sort_alignment;

    int ptn_gpu = 0;
    for (int i = lower; i < upper && ptn_gpu < P; ++i) {
        if (!all_informative) {
            bool seen[256] = {};
            for (int j = 1; j <= N; ++j)
                seen[(unsigned char)tr->yVector[j][i]] = true;
            int counter = 0;
            for (unsigned int v = 0; v < undetermined; ++v)
                if (seen[v]) counter++;
            if (counter <= 1) continue;  // uninformative: skip
        }

        for (int tipNum = 1; tipNum <= N; ++tipNum) {
            unsigned char nuc = tr->yVector[tipNum][i];
            // DNA: nuc is a bitmask directly. AA: nuc is a state index → convert to one-hot.
            unsigned int bitmask = is_bitmask_coded
                ? (unsigned int)nuc
                : (nuc < (unsigned char)states ? (1u << nuc) : (1u << states) - 1u);
            for (int s = 0; s < states; ++s)
                h_buf[(size_t)tipNum * P * states + (size_t)s * P + ptn_gpu] =
                    ((bitmask >> s) & 1u) ? 0u : (parsimonyNumber)kSankoffInf;
        }
        ptn_gpu++;
    }

    for (int k = 0; k < mem->K; ++k) {
        parsimonyNumber* dst = mem->d_parsVect + (size_t)k * parsVT;
        CUDA_CHECK(cudaMemcpyAsync(
            dst, h_buf.data(), parsVT * sizeof(parsimonyNumber), cudaMemcpyHostToDevice, stream
        ));
    }
}

// Upload pattern weights for Sankoff mode, reading from CPU's informativePtnWgt
// (built by compressSankoffDNA to match the same informative pattern set).
static void uploadSankoffSiteWeights(
    GpuParsimonyMem* mem, const partitionList* pr, cudaStream_t stream)
{
    const int P = mem->width;  // = parsimonyLength
    const int K = mem->K;
    std::vector<unsigned int> h_sw(P, 0u);
    // informativePtnWgt is a flat sequential array (no SIMD interleaving).
    // Type: parsimonyNumberShort (uint16_t) if sankoff_short_int, else parsimonyNumber (uint32_t).
    if (globalParam && globalParam->sankoff_short_int) {
        const uint16_t* ptnWgt = (const uint16_t*)pr->partitionData[0]->informativePtnWgt;
        for (int p = 0; p < P; ++p)
            h_sw[p] = (unsigned int)ptnWgt[p];
    } else {
        const parsimonyNumber* ptnWgt = pr->partitionData[0]->informativePtnWgt;
        for (int p = 0; p < P; ++p)
            h_sw[p] = (unsigned int)ptnWgt[p];
    }
    for (int k = 0; k < K; ++k) {
        unsigned int* dst = mem->d_siteWeights + (size_t)k * P;
        CUDA_CHECK(cudaMemcpyAsync(
            dst, h_sw.data(), P * sizeof(unsigned int), cudaMemcpyHostToDevice, stream
        ));
    }
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
    const int parsimonyLength = (int)pr->partitionData[0]->parsimonyLength;
    const int states = (int)pr->partitionData[0]->states;

    // Detect Sankoff mode: IQTree has a cost_matrix if -cost was specified.
    // Both Fitch and Sankoff use parsimonyLength as width (informative patterns only,
    // same as CPU compressSankoffDNA / compressFitchDNA).
    const bool use_sankoff = (iqtree.cost_matrix != nullptr && iqtree.cost_nstates > 0);
    const int width = parsimonyLength;

    GPU_LOG(
        "[GPU]   [1]      %-28s: %8.3f s  (width=%d states=%d%s)\n", "CPU parsimony alloc",
        msSince(t0) / 1e3, width, states, use_sankoff ? " Sankoff" : ""
    );
    if (use_sankoff) {
        const int pll_total = (int)(pr->partitionData[0]->upper - pr->partitionData[0]->lower);
        GPU_LOG("[GPU]            Sankoff: pll_total_ptn=%d  pll_informative(width)=%d\n",
                pll_total, width);
    }

    // Cap K1 (numpars) and K2 (gpu_worker) independently to 80% of free GPU memory.
    size_t _gpu_free = 0, _gpu_total = 0;
    cudaMemGetInfo(&_gpu_free, &_gpu_total);
    const size_t _parsVect_per_worker =
        (size_t)(2 * mxtips + 1) * (size_t)width * (size_t)states * sizeof(parsimonyNumber);
    const int _max_by_mem = (_parsVect_per_worker > 0)
        ? (int)((size_t)(_gpu_free * 0.80) / _parsVect_per_worker) : 9999;
    const int K1           = std::max(1, std::min(K, _max_by_mem));
    const int _k2_requested = (params.gpu_worker > 0) ? params.gpu_worker : K;
    const int k2_workers_early = std::max(1, std::min(_k2_requested, _max_by_mem));
    const int K_alloc = std::max(K1, k2_workers_early);

    // ── [2] Allocate GPU memory ───────────────────────────────────────────────
    t0 = std::chrono::high_resolution_clock::now();
    // Treels buffer: only needed for bootstrap (-bb). Non-bootstrap uses pool directly.
    const bool need_treels = (params.gbo_replicates > 0);
    // Per-round treels scales with K × N: observed ~28K/round at N=767, K=200.
    // K_alloc × mxtips gives ~5× margin and auto-scales with dataset size.
    const int max_treels_boot = need_treels ? K_alloc * mxtips : 0;
    GpuParsimonyMem* mem = gpuParsimonyMemAlloc(
        K_alloc, mxtips, width, states, params.gpu_pool_size, max_treels_boot,
        use_sankoff ? iqtree.cost_matrix : nullptr,
        use_sankoff ? iqtree.cost_nstates : 0
    );
    mem->save_margin = (float)params.gpu_treels_margin;
    GPU_LOG("[GPU]   [2]      %-28s: %8.3f s  (%.2f GB)\n", "GPU memory alloc",
            msSince(t0) / 1e3, mem->total_gpu_bytes / 1073741824.0);
    GPU_LOG("[GPU]            K1=%d  K2=%d  K_alloc=%d  per_worker=%.1f MB  free=%.1f GB\n",
            K1, k2_workers_early, K_alloc,
            _parsVect_per_worker / 1048576.0,
            _gpu_free / 1073741824.0);

    // ── [3] Upload tip parsVect ───────────────────────────────────────────────
    cudaStream_t stream = 0;
    t0 = std::chrono::high_resolution_clock::now();
    if (!use_sankoff) {
        uploadTipParsVect(mem, tr, pr, stream);
    } else {
        uploadSankoffTipParsVect(mem, tr, pr, stream);
        uploadSankoffSiteWeights(mem, pr, stream);
    }
    GPU_LOG("[GPU]   [3]      %-28s: %8.3f s\n", "Upload tip parsVect (H->D)", msSince(t0) / 1e3);

    // ── [4] Upload initial topologies ─────────────────────────────────────────
    t0 = std::chrono::high_resolution_clock::now();
    {
        GpuTopology h_topo;
        cpuToGpuTopology(tr, &h_topo);
        for (int k = 0; k < K1; ++k)
        {
            uploadTopology(mem, k, &h_topo, stream);
        }
    }
    GPU_LOG(
        "[GPU]   [4]      %-28s: %8.3f s  (%d trees)\n", "Upload topologies (H->D)",
        msSince(t0) / 1e3, K1
    );
    GPU_LOG("[GPU]\n");

    // Free CPU parsVect — not needed after GPU upload
    _pllFreeParsimonyDataStructures(tr, pr);

    // Build seeds
    std::vector<long> seeds(K1);
    for (int i = 0; i < K1; ++i)
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

            // Step 2: Download K1 GPU postSprScores
            const int Kc_scores = K1;        // K1 slots written by K1
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
                mem, seeds.data(), K1, params.sprDist, numNNI, pool_size, stream, hybrid_cb, nullptr,
                k2_workers, k2_max_outer
            );
        }
    );
    GPU_LOG(
        "[GPU]   [5]  %-28s: %8.3f s  (%d trees, %.2f ms/tree)\n", "Kernels",
        (double)build_ms / 1e3, K1, K1 > 0 ? (double)build_ms / K1 : 0.0
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

// ─── newickFromBackVf ────────────────────────────────────────────────────────
// Build a Newick string directly from the GPU back_vf[] flat array, bypassing
// PLL pointer-ring reconstruction (gpuTopoToCpu) and pllTreeToNewick entirely.
//
// Matches pllTreeToNewickREC traversal exactly:
//   root  = back_vf[start_vface]   (tr->start->back)
//   left  = back_vf[vfNextFace(p)] (p->next->back)
//   right = back_vf[vfNnxtFace(p)] (p->next->next->back)
//   root gets a 3rd child: back_vf[root_vf] (= start_vface)
//
// Branch lengths: fixed 0.1 for all non-root edges (parsimony trees; only topology
// matters for bootstrap consensus). Root writes ":0.0;\n" to match PLL convention.
// The resulting Newick is used as the treels key in saveCurrentTree(_gpu_newick_key).
static std::string newickFromBackVf(
    const int*         back_vf,      // h_treels_bvf + t * kMaxVFaces
    int                start_vface,  // h_tpl.start_vface (= 0)
    int                N,            // mem->mxtips
    const char* const* nameList,     // pllInst->nameList (1-indexed); nullptr if use_int_ids
    bool               use_int_ids = false  // true → output 0-indexed integers for treels key
)
{
    const int root_vf = back_vf[start_vface];

    enum Op : uint8_t { OPEN, COMMA_OP, CLOSE };
    struct Task { int vf; Op op; };

    std::string out;
    out.reserve((size_t)N * 28);

    std::vector<Task> stk;
    stk.reserve((size_t)N * 5);
    stk.push_back({root_vf, OPEN});

    while (!stk.empty()) {
        auto [vf, op] = stk.back();
        stk.pop_back();

        if (op == COMMA_OP) { out += ','; continue; }

        const bool is_tip = (vf < N);
        const int  num    = is_tip ? (vf + 1) : (N + 1 + (vf - N) / 3);

        if (op == CLOSE) {
            out += ')';
            out += (vf == root_vf) ? ":0.0;\n" : ":0.10000000000000000555";
            continue;
        }

        // OPEN
        if (is_tip) {
            if (use_int_ids)
                out += std::to_string(num - 1);  // 0-indexed integer (MTreeSet::init uses atoi)
            else
                out += nameList[num];
            out += (vf == root_vf) ? ":0.0;\n" : ":0.10000000000000000555";
        } else {
            const int left_vf  = back_vf[vfNextFace(vf, N)];
            const int right_vf = back_vf[vfNnxtFace(vf, N)];
            // Push in reverse execution order (LIFO):
            // output will be: '(' left ',' right [',' third_at_root] ')' branchlength
            stk.push_back({vf, CLOSE});
            if (vf == root_vf) {
                stk.push_back({back_vf[vf], OPEN});
                stk.push_back({0,           COMMA_OP});
            }
            stk.push_back({right_vf, OPEN});
            stk.push_back({0,         COMMA_OP});
            stk.push_back({left_vf,   OPEN});
            out += '(';
        }
    }

    return out;
}

// ─── gpuHillClimbing ─────────────────────────────────────────────────────────
// GPU iterative hill-climbing outer loop. Works in two modes:
//   Bootstrap (-bb): per-round K2 from pool, treels → saveCurrentTree (REPS), convergence check.
//   Non-bootstrap:   per-round K2 from pool, treels → candidateTrees, stop on no improvement.
void gpuHillClimbing(
    const Params& params, IQTree& iqtree, GpuParsimonyMem* mem
)
{
    // Two dedicated streams:
    //   stream      — K2 + ppars kernels (non-blocking launches allow CPU PASS1 overlap)
    //   stream_reps — REPS H→D upload + batchREPSKernel + D→H download
    // Using separate streams removes null-stream serialization, enabling ppars[N] on
    // stream to overlap with REPS[N-1] on stream_reps at the GPU level.
    cudaStream_t stream, stream_reps;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaStreamCreate(&stream_reps));
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
    downloadTopology(mem, 0, &h_tpl, 0);  // synchronous: use default stream for one-time init
    h_tpl.start_vface = 0;
    memset(h_tpl.xpars, 0, sizeof(h_tpl.xpars));

    // Reusable host buffer for batch-downloading treels back_vf each round.
    // Pinned (page-locked) memory eliminates page-fault stalls on first D2H:
    // pageable gives ~200 MB/s (cold pages), pinned gives ~6 GB/s consistently.
    const int max_treels = mem->max_treels;
    int*          h_treels_bvf    = nullptr;
    unsigned int* h_treels_scores = nullptr;
    unsigned int* h_treels_hashes = nullptr;
    if (max_treels > 0)
    {
        CUDA_CHECK(cudaMallocHost(&h_treels_bvf,
            (size_t)max_treels * kMaxVFaces * sizeof(int)));
        CUDA_CHECK(cudaMallocHost(&h_treels_scores,
            (size_t)max_treels * sizeof(unsigned int)));
        CUDA_CHECK(cudaMallocHost(&h_treels_hashes,
            (size_t)max_treels * sizeof(unsigned int)));
        memset(h_treels_hashes, 0, (size_t)max_treels * sizeof(unsigned int));
    }
    // Hash → best parsimony seen: persisted across rounds for dedup (same as saveCurrentTree's treels map).
    std::unordered_map<unsigned int, unsigned int> seen_hash_pars;

    // GPU per-pattern parsimony for treels (replaces CPU computeParsimony per tree).
    // Only active for bootstrap + Fitch mode.
    const int nptn = iqtree.getAlnNPattern();
    const int nptn_padded = nptn + 16;  // VCSIZE_USHORT padding (matches CPU _pattern_pars alloc)
    uint16_t* d_treels_ptn_pars = nullptr;
    uint16_t* h_treels_ptn_pars = nullptr;
    const bool use_gpu_treels_pars =
        is_bootstrap && (mem->states == 4 || mem->states == 20);
    if (use_gpu_treels_pars && max_treels > 0)
    {
        CUDA_CHECK(cudaMalloc(&d_treels_ptn_pars,
            (size_t)max_treels * nptn_padded * sizeof(uint16_t)));
        CUDA_CHECK(cudaMallocHost(&h_treels_ptn_pars,
            (size_t)max_treels * nptn_padded * sizeof(uint16_t)));
    }

    // Batch REPS: allocate batch buffers and host staging buffer once.
    // use_batch_reps is active when gpu_boot_mem_ has been initialized (non-null + batch alloc).
    std::vector<uint16_t> h_batch_pars;   // [n_treels × nptn_padded] staging for batch REPS (grows dynamically)
    const bool use_batch_reps = use_gpu_treels_pars && max_treels > 0
                                 && iqtree.gpu_boot_mem_ != nullptr;
    // Per-round treels limit: no hard cap — process ALL unique trees saved per round.
    // h_batch_pars and batch REPS GPU buffers grow dynamically to fit actual n_treels.
    int current_max_reps = 0;  // current allocated capacity (grows per round as needed)

    // Per-round: info for each unique tree accumulated during Pass 1 (batch REPS path).
    struct UniqueTreeEntry { std::string newick; int pars; int t_ptn; };
    std::vector<UniqueTreeEntry> unique_trees;

    double cur_correlation = 0.0;
    int round = 0, total_done = 0;
    double best_logl_seen  = -1e30;
    int total_replicates   = 0;
    int last_impr_at       = 0;   // total_replicates at last improvement
    const int unsuccess_thresh = k2_workers * params.gpu_worker_stop;
    // Host buffer for pool scores (used every round to print best; cheap: pool_size ints)
    std::vector<unsigned int> h_pool_scores_round(pool_size, 0xFFFFFFFFu);
    unsigned int best_pool_round = 0xFFFFFFFFu;

    if (is_bootstrap)
        GPU_LOG("%s K2-treels: B=%d K=%d pool=%d unsuccess=%d\n", tag, B, k2_workers, pool_size, unsuccess_thresh);
    else
        GPU_LOG("%s K2-hillclimb: K=%d pool=%d unsuccess=%d\n", tag, k2_workers, pool_size, unsuccess_thresh);
    fflush(stdout);

    // ── Per-round state for async pipeline (pipelined PASS1 concurrent with K2) ─
    // Holds info about the round whose treels data sits in h_treels_* host buffers.
    // PASS1 for this state runs during the NEXT round's K2 kernel.
    struct PrevRound {
        int n_treels  = 0;   // treels count in host buffers
        int round_num = 0;   // 1-indexed round number
        double k2_ms  = 0.0; // K2 kernel time for this round (measured via CUDA events)
        unsigned int pool_best = 0xFFFFFFFFu;  // best pool score after this round's K2
        double ppars_ms = 0.0; // ppars kernel time (CUDA events, stored for next-iter log)
        double d2h_ms   = 0.0; // D2H download time: steps [4]+[7]
    };
    PrevRound prev;        // zero-init = "no prev data" for first iteration
    bool has_prev = false;
    bool should_stop = false;
    // Counts consecutive rounds where treels didn't grow. When ≥2 and no pending
    // conv thread, the bootstrap distribution is frozen → force cur_correlation=1.0
    // so the convergence criterion can fire (avoids infinite loop when treels saturates).
    int stable_rounds = 0;

    // Background thread for summarizeBootstrap: launched at end of step [8],
    // collected at start of next step [8]. Hides ~800ms behind K2's ~1750ms.
    // Safety: thread reads boot_trees/treels after PASS2[R] finishes; PASS2[R+1]
    // starts only after K2[R+1] sync (~1750ms later), so no data race.
    using ConvResult = std::pair<SplitGraph*, double>;  // (sg, actual_conv_ms)
    std::future<ConvResult> conv_future;

    // CUDA events for K2 and ppars kernel timing.
    cudaEvent_t ev_k2_start, ev_k2_end;
    cudaEvent_t ev_ppars_start, ev_ppars_end;
    CUDA_CHECK(cudaEventCreate(&ev_k2_start));
    CUDA_CHECK(cudaEventCreate(&ev_k2_end));
    CUDA_CHECK(cudaEventCreate(&ev_ppars_start));
    CUDA_CHECK(cudaEventCreate(&ev_ppars_end));

    for (;;)
    {
        // ── [0] Collect background convergence thread BEFORE PASS1 writes treels/boot_trees ─
        // DATA RACE FIX: The background thread from the previous iteration reads
        // treels/boot_trees (summarizeBootstrap).  PASS1 (step [2]) writes to them via
        // saveCurrentTree.  Both run concurrently → data race → undefined behavior.
        // Fix: always collect the thread HERE, before PASS1, not in step [8].
        double _t_conv_ms_early = 0.0;
        if (is_bootstrap && conv_future.valid())
        {
            auto [sg, bg_ms] = conv_future.get();
            _t_conv_ms_early = bg_ms;
            if (sg)
            {
                iqtree.boot_splits.push_back(sg);
                while (iqtree.boot_splits.size() > 2)
                {
                    delete iqtree.boot_splits.front();
                    iqtree.boot_splits.erase(iqtree.boot_splits.begin());
                }
                if (iqtree.boot_splits.size() >= 2)
                    cur_correlation = iqtree.computeBootstrapCorrelation();
            }
            conv_future = {};
        }

        // ── [1] Reset GPU state + launch K2[round] on stream (non-blocking) ───
        // Uses logl_cutoff from iqtree (updated by PASS2 of previous-previous round;
        // 1-round lag vs sequential, acceptable since cutoff changes slowly).
        const unsigned int boot_cutoff = (is_bootstrap && iqtree.logl_cutoff != 0.0)
                                             ? (unsigned int)(-(double)iqtree.logl_cutoff)
                                             : 0xFFFFFFFFu;
        resetTreelsRound(mem, boot_cutoff, stream);
        resetPoolRound(mem, stream);
        CUDA_CHECK(cudaEventRecord(ev_k2_start, stream));
        gpuStepwiseBuildTrees(
            mem, nullptr, /*k1_count=*/0, sprDist, numNNI, pool_size, stream, nullptr, nullptr,
            k2_workers, /*max_outer_iters=*/1
        );  // non-blocking: K2 kernel queued on stream, host returns immediately
        CUDA_CHECK(cudaEventRecord(ev_k2_end, stream));

        // ── [2] PASS 1 of prev round — concurrent with K2 above ──────────────
        // Reads h_treels_bvf/scores/hashes/ptn_pars (from prev round's download).
        // K2 writes only to device memory → no conflict with host reads here.
        // newickFromBackVf replaces gpuTopoToCpu+pllTreeToNewick: builds Newick directly
        // from back_vf[] without reconstructing the PLL pointer ring.
        const int _treels_size_pre_pass = (int)iqtree.treels_logl.size();
        double _t_newick_ms = 0, _t_save_ms = 0;
        double _t_hash_ms = 0, _t_bpars_ms = 0;
        int _t_skip_hash = 0;
        iqtree._gpu_t_pp = iqtree._gpu_t_reps = iqtree._gpu_t_bupdate = 0.0;
        if (use_batch_reps) unique_trees.clear();
        // Grow batch REPS buffers if this round has more trees than current capacity.
        if (use_batch_reps && has_prev && prev.n_treels > current_max_reps)
        {
            int new_cap = prev.n_treels;
            h_batch_pars.resize((size_t)new_cap * nptn_padded);
            gpuBatchREPSGrow(iqtree.gpu_boot_mem_, new_cap);
            unique_trees.reserve(new_cap);
            current_max_reps = new_cap;
        }
        if (has_prev && prev.n_treels > 0)
        {
            for (int t = 0; t < prev.n_treels; t++)
            {
                // Option C: hash-based dedup — skip before Newick generation
                if (is_bootstrap && h_treels_hashes != nullptr) {
                    auto _th = std::chrono::high_resolution_clock::now();
                    unsigned int h  = h_treels_hashes[t];
                    unsigned int ps = h_treels_scores[t];
                    auto hit = seen_hash_pars.find(h);
                    if (hit != seen_hash_pars.end() && ps >= hit->second) {
                        _t_skip_hash++;
                        _t_hash_ms += msSince(_th);
                        continue;
                    }
                    seen_hash_pars[h] = ps;
                    _t_hash_ms += msSince(_th);
                }

                // Bootstrap batch path: output integer-ID newick directly (MTreeSet::init uses atoi).
                // Other paths (non-bootstrap, Sankoff fallback, per-tree fallback) need name-based
                // newick for readTreeString/setAlignment → use nameList.
                const bool _need_int_ids = (is_bootstrap && use_gpu_treels_pars && use_batch_reps);
                auto _tn = std::chrono::high_resolution_clock::now();
                std::string newick = newickFromBackVf(
                    h_treels_bvf + (size_t)t * kMaxVFaces,
                    h_tpl.start_vface, mem->mxtips,
                    _need_int_ids ? nullptr : iqtree.pllInst->nameList,
                    _need_int_ids);
                _t_newick_ms += msSince(_tn);
                if (newick.empty()) continue;

                if (is_bootstrap && use_gpu_treels_pars)
                {
                    if (use_batch_reps)
                    {
                        int u = (int)unique_trees.size();
                        // Fast path: GPU ppars kernel already computed per-original-pattern
                        // parsimony in d_treels_ptn_pars (fixed kernel output).
                        // newick is already integer-ID (use_int_ids=true above).
                        uint16_t* dst = h_batch_pars.data() + (size_t)u * nptn_padded;
                        const uint16_t* src = h_treels_ptn_pars + (size_t)t * nptn_padded;
                        std::copy(src, src + nptn_padded, dst);
                        unique_trees.push_back({newick, (int)h_treels_scores[t], t});
                    }
                    else
                    {
                        // Per-tree fallback (batch not allocated — rare edge case).
                        // newick is name-based here (need_int_ids=false), readTreeString works.
                        iqtree.readTreeString(newick);
                        iqtree.initializeAllPartialPars();
                        iqtree.clearAllPartialLH();
                        const int pars = iqtree.computeParsimony();
                        std::ostringstream _id_ostr;
                        iqtree.printTree(_id_ostr, WT_TAXON_ID | WT_SORT_TAXA);
                        iqtree._gpu_newick_key = _id_ostr.str();
                        bool saved = iqtree.params->spr_parsimony;
                        iqtree.params->spr_parsimony = false;
                        { auto _ts = std::chrono::high_resolution_clock::now();
                          iqtree.saveCurrentTree(-(double)pars);
                          _t_save_ms += msSince(_ts); }
                        iqtree.params->spr_parsimony = saved;
                        iqtree._gpu_newick_key.clear();
                    }
                }
                else if (is_bootstrap)
                {
                    // Sankoff / unsupported-states fallback
                    iqtree.readTreeString(newick);
                    iqtree.initializeAllPartialPars();
                    iqtree.clearAllPartialLH();
                    int pars = iqtree.computeParsimony();
                    bool saved = iqtree.params->spr_parsimony;
                    iqtree.params->spr_parsimony = false;
                    { auto _ts = std::chrono::high_resolution_clock::now();
                      iqtree.saveCurrentTree(-(double)pars);
                      _t_save_ms += msSince(_ts); }
                    iqtree.params->spr_parsimony = saved;
                }
                else
                {
                    // Non-bootstrap: update candidateTrees
                    iqtree.readTreeString(newick);
                    iqtree.initializeAllPartialPars();
                    iqtree.clearAllPartialLH();
                    int pars = iqtree.computeParsimony();
                    iqtree.curScore = -(double)pars;
                    bool isNew = iqtree.candidateTrees.update(newick, iqtree.curScore);
                    if (isNew && iqtree.curScore > iqtree.bestScore)
                        iqtree.setBestTree(newick, iqtree.curScore);
                }
            }
        }

        // ── [3] Wait for K2 to finish; measure K2 kernel time via CUDA events ──
        // cudaStreamSynchronize waits for ev_k2_end (and all preceding work on stream).
        // ev_k2_end was recorded right after the K2 kernel launch, so ElapsedTime
        // gives pure GPU kernel time, unaffected by concurrent PASS1 on the host.
        CUDA_CHECK(cudaStreamSynchronize(stream));
        float cur_k2_ms_f = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&cur_k2_ms_f, ev_k2_start, ev_k2_end));
        double cur_k2_ms = cur_k2_ms_f;

        // ── [4] Download n_treels count + pool scores (small, fast) ──────────
        auto _t_d2h_cur = std::chrono::high_resolution_clock::now();
        int h_filled = 0;
        if (mem->d_treelsFilled)
            CUDA_CHECK(cudaMemcpy(&h_filled, mem->d_treelsFilled, sizeof(int), cudaMemcpyDeviceToHost));
        const int n_treels = std::min(h_filled, max_treels);

        CUDA_CHECK(cudaMemcpy(
            h_pool_scores_round.data(), mem->d_poolScores,
            (size_t)pool_size * sizeof(unsigned int), cudaMemcpyDeviceToHost
        ));
        const unsigned int cur_best_pool =
            *std::min_element(h_pool_scores_round.begin(), h_pool_scores_round.end());
        double cur_d2h_ms = msSince(_t_d2h_cur);

        // ── [5] Launch pattern_pars kernel async (concurrent with REPS+PASS2) ─
        // Reads d_treelsBackVf (just written by K2), writes d_treels_ptn_pars.
        // Runs on stream while host does REPS + PASS2 below.
        double cur_ppars_ms = 0.0;
        if (n_treels > 0 && use_gpu_treels_pars)
        {
            CUDA_CHECK(cudaEventRecord(ev_ppars_start, stream));
            gpuComputeTreelsPatternPars(
                mem, n_treels, /*start_vf=*/0, nptn, nptn_padded,
                d_treels_ptn_pars, stream
            );
            CUDA_CHECK(cudaEventRecord(ev_ppars_end, stream));
        }

        // ── [6] Batch REPS + PASS 2 of prev round (concurrent with ppars above) ─
        // REPS reads h_treels_ptn_pars (host, prev round) — not ppars kernel output.
        // ppars writes d_treels_ptn_pars (device, current round) — no conflict.
        if (has_prev && is_bootstrap && use_batch_reps)
        {
            const int T_unique = (int)unique_trees.size();
            if (T_unique > 0)
            {
                auto _t_br = std::chrono::high_resolution_clock::now();
                gpuBatchREPSEval(iqtree.gpu_boot_mem_, h_batch_pars.data(), T_unique, stream_reps);
                iqtree._gpu_t_reps += msSince(_t_br);

                const int B_reps = iqtree.gpu_boot_mem_->B;
                for (int u = 0; u < T_unique; u++)
                {
                    auto& info = unique_trees[u];
                    iqtree._gpu_newick_key = info.newick;
                    iqtree._gpu_precomputed_rell =
                        iqtree.gpu_boot_mem_->h_batch_rell + (size_t)u * B_reps;

                    bool saved = iqtree.params->spr_parsimony;
                    iqtree.params->spr_parsimony = false;
                    auto _t = std::chrono::high_resolution_clock::now();
                    iqtree.saveCurrentTree(-(double)info.pars);
                    _t_save_ms += msSince(_t);
                    iqtree.params->spr_parsimony = saved;

                    iqtree._gpu_newick_key.clear();
                    iqtree._gpu_precomputed_rell = nullptr;
                }
            }
        }

        // ── [7] Wait for pattern_pars kernel; download treels data ───────────
        // After sync: d_treels_ptn_pars is ready. Download overwrites h_treels_*
        // buffers — safe because PASS1 (step [2]) finished reading them in step [2].
        CUDA_CHECK(cudaStreamSynchronize(stream));
        {
            float _ppars_f = 0.f;
            if (n_treels > 0 && use_gpu_treels_pars)
                CUDA_CHECK(cudaEventElapsedTime(&_ppars_f, ev_ppars_start, ev_ppars_end));
            cur_ppars_ms = _ppars_f;
        }

        auto _t_d2h_big = std::chrono::high_resolution_clock::now();
        if (n_treels > 0 && max_treels > 0)
        {
            CUDA_CHECK(cudaMemcpy(
                h_treels_bvf, mem->d_treelsBackVf,
                (size_t)n_treels * kMaxVFaces * sizeof(int), cudaMemcpyDeviceToHost
            ));
            CUDA_CHECK(cudaMemcpy(
                h_treels_scores, mem->d_treelsScores,
                (size_t)n_treels * sizeof(unsigned int), cudaMemcpyDeviceToHost
            ));
            if (is_bootstrap && mem->d_treelsHashes != nullptr)
                CUDA_CHECK(cudaMemcpy(
                    h_treels_hashes, mem->d_treelsHashes,
                    (size_t)n_treels * sizeof(unsigned int), cudaMemcpyDeviceToHost
                ));
            if (use_gpu_treels_pars)
                CUDA_CHECK(cudaMemcpy(
                    h_treels_ptn_pars, d_treels_ptn_pars,
                    (size_t)n_treels * nptn_padded * sizeof(uint16_t), cudaMemcpyDeviceToHost
                ));
        }
        cur_d2h_ms += msSince(_t_d2h_big);

        // ── [8] Stats, cutoff, convergence, log for PREV round ───────────────
        if (has_prev)
        {
            round++;
            total_done += k2_workers;
            total_replicates += prev.n_treels;
            const int _n_new_treels = prev.n_treels - _t_skip_hash;
            const double _cpu_ms = _t_newick_ms + _t_hash_ms + _t_bpars_ms + _t_save_ms;

            // boot_trees can only change if new unique trees were added to treels during
            // PASS1+PASS2. If treels didn't grow, summarizeBootstrap produces the same
            // SplitGraph as last time → skip to avoid 800ms rebuild for nothing.
            const bool _boot_changed = ((int)iqtree.treels_logl.size() > _treels_size_pre_pass);
            double _t_conv_ms = 0.0;
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
                        last_impr_at   = total_done;
                    }
                }
                // Update logl_cutoff for next round's d_treelsCutoff
                if (!iqtree.treels_logl.empty())
                {
                    if (params.gpu_treels_margin < 0.0)
                    {
                        // save-all: SAVE A saves all testInsert hits; use percentile to
                        // keep quality threshold for SAVE B/C gates.
                        DoubleVector logl = iqtree.treels_logl;
                        nth_element(
                            logl.begin(), logl.begin() + logl.size() * params.cutoff_percent / 100,
                            logl.end(), std::greater<double>()
                        );
                        iqtree.logl_cutoff = logl[logl.size() * params.cutoff_percent / 100];
                    }
                    else
                    {
                        // Relative margin: cutoff consistent with SAVE A gate.
                        // logl = -parsimony → best_logl = -min_parsimony < 0
                        // boot_cutoff = min_parsimony*(1+r) ↔ logl_cutoff = best_logl*(1+r)
                        double best_logl = *std::max_element(
                            iqtree.treels_logl.begin(), iqtree.treels_logl.end());
                        iqtree.logl_cutoff = best_logl * (1.0 + params.gpu_treels_margin);
                    }
                }
                // Convergence check — collect done at step [0] before PASS1 to avoid data race.
                // Use timing from early collect if available.
                if (_t_conv_ms_early > 0.0) _t_conv_ms = _t_conv_ms_early;
                // conv_future is guaranteed to be invalid here (collected at step [0]).
                // Launch: start background summarizeBootstrap for the round just processed.
                // DATA RACE FIX: This thread is launched AFTER PASS2 finishes (treels/boot_trees
                // stable). It will be collected at step [0] of the NEXT iteration, BEFORE PASS1
                // of that iteration modifies treels/boot_trees. This guarantees no data race.
                if (round % step_iter_rounds == 0 && _boot_changed)
                {
                    conv_future = std::async(std::launch::async,
                        [&iqtree]() -> ConvResult {
                            auto _tc = std::chrono::high_resolution_clock::now();
                            SplitGraph* sg = new SplitGraph;
                            iqtree.summarizeBootstrap(*sg);
                            return {sg, msSince(_tc)};
                        });
                }
                // Stable-treels safeguard: if treels hasn't grown for ≥2 consecutive rounds
                // and no thread is in-flight, the bootstrap distribution is frozen.
                // Same trees → same boot_trees → same SplitGraph → cor = 1.0 by definition.
                if (_boot_changed) {
                    stable_rounds = 0;
                } else {
                    ++stable_rounds;
                    if (stable_rounds >= 2 && !conv_future.valid())
                        cur_correlation = 1.0;
                }
            }
            else
            {
                double cur_best_logl = -(double)prev.pool_best;
                if (cur_best_logl > best_logl_seen + 1e-6)
                {
                    best_logl_seen = cur_best_logl;
                    last_impr_at   = total_done;
                }
            }

            best_pool_round = prev.pool_best;
            if (is_bootstrap)
            {
                const int _delta_treels = (int)iqtree.treels_logl.size() - _treels_size_pre_pass;
                GPU_LOG(
                    "%s Round %2d  +uniq=%5d  k2=%5.2fs"
                    "  treels=%6zu  best=%6u  cor=%6.4f\n",
                    tag, round, _delta_treels, prev.k2_ms / 1e3,
                    iqtree.treels_logl.size(), prev.pool_best, cur_correlation
                );
            }
            else
            {
                GPU_LOG(
                    "%s Round %2d  k2=%5.2fs  best=%8u  last_impr=%5d\n",
                    tag, round, prev.k2_ms / 1e3, prev.pool_best, last_impr_at
                );
            }
            fflush(stdout);

            if (total_done - last_impr_at > unsuccess_thresh
                    && (!is_bootstrap || cur_correlation >= params.min_correlation))
            {
                should_stop = true;
            }
        }

        // ── [9] Advance: store current round as prev; break if stop signalled ─
        prev.n_treels   = n_treels;
        prev.round_num  = round + 1;  // will be incremented in step [8] next iter
        prev.k2_ms      = cur_k2_ms;
        prev.pool_best  = cur_best_pool;
        prev.ppars_ms   = cur_ppars_ms;
        prev.d2h_ms     = cur_d2h_ms;
        has_prev = true;

        if (should_stop) break;
    }  // end for (;;)

    // Collect any still-running background conv thread before accessing boot_splits.
    if (conv_future.valid())
    {
        auto [sg, _bg_ms] = conv_future.get();
        if (sg)
        {
            iqtree.boot_splits.push_back(sg);
            while (iqtree.boot_splits.size() > 2)
            {
                delete iqtree.boot_splits.front();
                iqtree.boot_splits.erase(iqtree.boot_splits.begin());
            }
            if (iqtree.boot_splits.size() >= 2)
                cur_correlation = iqtree.computeBootstrapCorrelation();
        }
        conv_future = {};
    }

    // ── Final: process last downloaded treels (prev) with PASS1 + REPS + PASS2 ─
    // When we break, h_treels_* holds the last K2 round's data (downloaded in step [7]).
    // PASS1 for it was NOT run (would have run in the NEXT iteration's step [2]).
    if (has_prev && prev.n_treels > 0)
    {
        double _t_newick_ms = 0, _t_save_ms = 0;
        double _t_hash_ms = 0, _t_bpars_ms = 0;
        int _t_skip_hash = 0;
        iqtree._gpu_t_pp = iqtree._gpu_t_reps = iqtree._gpu_t_bupdate = 0.0;
        if (use_batch_reps) unique_trees.clear();
        // Grow batch REPS buffers for this final round if needed.
        if (use_batch_reps && prev.n_treels > current_max_reps)
        {
            int new_cap = prev.n_treels;
            h_batch_pars.resize((size_t)new_cap * nptn_padded);
            gpuBatchREPSGrow(iqtree.gpu_boot_mem_, new_cap);
            unique_trees.reserve(new_cap);
            current_max_reps = new_cap;
        }

        for (int t = 0; t < prev.n_treels; t++)
        {
            if (is_bootstrap && h_treels_hashes != nullptr) {
                auto _th = std::chrono::high_resolution_clock::now();
                unsigned int h  = h_treels_hashes[t];
                unsigned int ps = h_treels_scores[t];
                auto hit = seen_hash_pars.find(h);
                if (hit != seen_hash_pars.end() && ps >= hit->second) {
                    _t_skip_hash++;
                    _t_hash_ms += msSince(_th);
                    continue;
                }
                seen_hash_pars[h] = ps;
                _t_hash_ms += msSince(_th);
            }

            const bool _need_int_ids2 = (is_bootstrap && use_gpu_treels_pars && use_batch_reps);
            auto _tn = std::chrono::high_resolution_clock::now();
            std::string newick = newickFromBackVf(
                h_treels_bvf + (size_t)t * kMaxVFaces,
                h_tpl.start_vface, mem->mxtips,
                _need_int_ids2 ? nullptr : iqtree.pllInst->nameList,
                _need_int_ids2);
            _t_newick_ms += msSince(_tn);
            if (newick.empty()) continue;

            if (is_bootstrap && use_gpu_treels_pars && use_batch_reps)
            {
                int u = (int)unique_trees.size();
                uint16_t* dst = h_batch_pars.data() + (size_t)u * nptn_padded;
                const uint16_t* src = h_treels_ptn_pars + (size_t)t * nptn_padded;
                std::copy(src, src + nptn_padded, dst);
                unique_trees.push_back({newick, (int)h_treels_scores[t], t});
            }
            else if (is_bootstrap && use_gpu_treels_pars)
            {
                // Per-tree fallback (rare): newick is name-based here.
                iqtree.readTreeString(newick);
                iqtree.initializeAllPartialPars();
                iqtree.clearAllPartialLH();
                const int pars = iqtree.computeParsimony();
                std::ostringstream _id_ostr;
                iqtree.printTree(_id_ostr, WT_TAXON_ID | WT_SORT_TAXA);
                iqtree._gpu_newick_key = _id_ostr.str();
                bool saved = iqtree.params->spr_parsimony;
                iqtree.params->spr_parsimony = false;
                { auto _ts = std::chrono::high_resolution_clock::now();
                  iqtree.saveCurrentTree(-(double)pars);
                  _t_save_ms += msSince(_ts); }
                iqtree.params->spr_parsimony = saved;
                iqtree._gpu_newick_key.clear();
            }
            else if (is_bootstrap)
            {
                iqtree.readTreeString(newick);
                iqtree.initializeAllPartialPars();
                iqtree.clearAllPartialLH();
                int pars = iqtree.computeParsimony();
                bool saved = iqtree.params->spr_parsimony;
                iqtree.params->spr_parsimony = false;
                auto _t2 = std::chrono::high_resolution_clock::now();
                iqtree.saveCurrentTree(-(double)pars);
                _t_save_ms += msSince(_t2);
                iqtree.params->spr_parsimony = saved;
            }
            else
            {
                iqtree.readTreeString(newick);
                iqtree.initializeAllPartialPars();
                iqtree.clearAllPartialLH();
                int pars = iqtree.computeParsimony();
                iqtree.curScore = -(double)pars;
                bool isNew = iqtree.candidateTrees.update(newick, iqtree.curScore);
                if (isNew && iqtree.curScore > iqtree.bestScore)
                    iqtree.setBestTree(newick, iqtree.curScore);
            }
        }
        // Final REPS + PASS2
        if (is_bootstrap && use_batch_reps)
        {
            const int T_unique = (int)unique_trees.size();
            if (T_unique > 0)
            {
                auto _t_br = std::chrono::high_resolution_clock::now();
                gpuBatchREPSEval(iqtree.gpu_boot_mem_, h_batch_pars.data(), T_unique, stream_reps);
                iqtree._gpu_t_reps += msSince(_t_br);
                const int B_reps = iqtree.gpu_boot_mem_->B;
                for (int u = 0; u < T_unique; u++)
                {
                    auto& info = unique_trees[u];
                    iqtree._gpu_newick_key = info.newick;
                    iqtree._gpu_precomputed_rell =
                        iqtree.gpu_boot_mem_->h_batch_rell + (size_t)u * B_reps;
                    bool saved = iqtree.params->spr_parsimony;
                    iqtree.params->spr_parsimony = false;
                    auto _t2 = std::chrono::high_resolution_clock::now();
                    iqtree.saveCurrentTree(-(double)info.pars);
                    _t_save_ms += msSince(_t2);
                    iqtree.params->spr_parsimony = saved;
                    iqtree._gpu_newick_key.clear();
                    iqtree._gpu_precomputed_rell = nullptr;
                }
            }
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));  // ensure any pending work before destroy
    CUDA_CHECK(cudaEventDestroy(ev_k2_start));
    CUDA_CHECK(cudaEventDestroy(ev_k2_end));
    CUDA_CHECK(cudaEventDestroy(ev_ppars_start));
    CUDA_CHECK(cudaEventDestroy(ev_ppars_end));

    if (is_bootstrap) {
        GPU_LOG("%s Done: %d rounds, %d replicates, cor=%.4f\n", tag, round, total_done, cur_correlation);
        // Debug: how many unique topologies made it into the REPS pool
        const int n_treels_pool = (int)iqtree.treels_logl.size();
        GPU_LOG("%s [DEBUG] treels_pool=%d unique topologies in REPS\n", tag, n_treels_pool);
        if (n_treels_pool > 0) {
            double best_logl = *std::max_element(iqtree.treels_logl.begin(), iqtree.treels_logl.end());
            double worst_logl = *std::min_element(iqtree.treels_logl.begin(), iqtree.treels_logl.end());
            GPU_LOG("%s [DEBUG] treels_logl range: [%.0f, %.0f]  (parsimony [%d, %d])\n",
                    tag, worst_logl, best_logl, (int)(-best_logl), (int)(-worst_logl));
        }
        // Count distinct tree indices that won at least 1 bootstrap replicate
        {
            std::set<int> winner_indices;
            int total_reps = (int)iqtree.boot_trees.size();
            int assigned = 0;
            for (int s = 0; s < total_reps; s++) {
                int idx = iqtree.boot_trees[s];
                if (idx >= 0) { winner_indices.insert(idx); assigned++; }
            }
            GPU_LOG("%s [DEBUG] boot_trees: %d distinct winners, %d/%d replicates assigned\n",
                    tag, (int)winner_indices.size(), assigned, total_reps);
        }
    } else
        GPU_LOG("%s Done: %d rounds\n", tag, round);

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

    // Free per-pattern pars device buffer, pinned host treels buffers, and destroy streams.
    // Events (ev_k2_*/ev_ppars_*) are already destroyed above (before pool block).
    // Ignore stream-destroy errors here: under nsys profiling the CUDA context may
    // already be torn down by the time cleanup runs, causing spurious errors.
    if (d_treels_ptn_pars) cudaFree(d_treels_ptn_pars);
    if (h_treels_ptn_pars) cudaFreeHost(h_treels_ptn_pars);
    if (h_treels_bvf)      cudaFreeHost(h_treels_bvf);
    if (h_treels_scores)   cudaFreeHost(h_treels_scores);
    if (h_treels_hashes)   cudaFreeHost(h_treels_hashes);
    cudaStreamDestroy(stream_reps);
    cudaStreamDestroy(stream);
}

}  // namespace mpbootgpu
