#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "gpu/include/gpu_init_trees.cuh"
#include "gpu/include/pars_build.cuh"
#include "gpu/include/pars_tree.cuh"
#include "gpu/include/utils.cuh"
#include "iqtree.h"
#include "pllrepo/src/pll.h"
#include "sprparsimony.h"  // pllInstanceClone, pllPartitionsClone, _allocateParsimony*
#include "tools.h"

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
int gpuInitCandidateTrees(
    const Params& params, IQTree& iqtree, int numInitTrees, std::vector<std::string>& candidateTrees
)
{
    int K = numInitTrees - 1;  // tree indices 1..numInitTrees-1

    pllInstance* tr = iqtree.pllInst;
    partitionList* pr = iqtree.pllPartitions;
    const int mxtips = tr->mxtips;

    printf("\n[GPU] ── gpuInitCandidateTrees ─────────────────────────────\n");
    printf("[GPU]   K=%d  N=%d\n", K, mxtips);

    // ── [1] Allocate CPU parsVect & read metadata ─────────────────────────────
    auto t0 = std::chrono::high_resolution_clock::now();
    _allocateParsimonyDataStructures(tr, pr, PLL_FALSE);
    const int width = (int)pr->partitionData[0]->parsimonyLength;
    const int states = (int)pr->partitionData[0]->states;
    printf(
        "[GPU]   [1] CPU alloc parsimony structs:  %.1f ms  (width=%d states=%d)\n", msSince(t0),
        width, states
    );

    // ── [2] Allocate GPU memory ───────────────────────────────────────────────
    t0 = std::chrono::high_resolution_clock::now();
    GpuParsimonyMem* mem = gpuParsimonyMemAlloc(K, mxtips, width, states);
    printf("[GPU]   [2] GPU mem alloc:                %.1f ms\n", msSince(t0));

    // ── [3] Upload tip parsVect ───────────────────────────────────────────────
    cudaStream_t stream = 0;
    t0 = std::chrono::high_resolution_clock::now();
    uploadTipParsVect(mem, tr, pr, stream);
    printf("[GPU]   [3] Upload tip parsVect (H→D):    %.1f ms\n", msSince(t0));

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
    printf("[GPU]   [4] Upload topologies (H→D):      %.1f ms  (%d trees)\n", msSince(t0), K);

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
    const int numNNI          = (mxtips > 4) ? max(1, (int)(params.gpu_nni_strength * (mxtips - 3))) : 1;
    const int numSearchIter   = params.gpu_hc_iter;
    const int stopNoImprove   = params.gpu_stop;
    // gpu_phase3_margin is in percent (supports fractional e.g. 0.1).
    // Internally stored as tenths-of-percent (1 = 0.1%, 10 = 1.0%, 50 = 5.0%).
    const unsigned int margin = (params.gpu_phase3_margin < 0.0f)
                                    ? 0xFFFFFFFFu
                                    : (unsigned int)(params.gpu_phase3_margin * 10.0f + 0.5f);

    const float top_pct = params.gpu_phase3_top_pct;  // Opt-G2: -1=disabled

    float build_ms = ev_time(
        [&]
        {
            gpuStepwiseBuildTrees(mem, seeds.data(), params.sprDist,
                                  numSearchIter, numNNI, stopNoImprove, margin, top_pct, stream);
        }
    );
    printf(
        "[GPU]   [5+6+7] GPU kernel (build+SPR+search): %.1f ms  (%d trees, %.2f ms/tree)"
        "  [iters=%d NNI=%d(%.2f) sprDist=%d stop=%d margin=%s top_pct=%s]\n",
        (double)build_ms, K, K > 0 ? (double)build_ms / K : 0.0,
        numSearchIter, numNNI, params.gpu_nni_strength, params.sprDist, stopNoImprove,
        margin == 0xFFFFFFFFu ? "off"
            : (std::to_string(margin / 10) + "." + std::to_string(margin % 10) + "%").c_str(),
        top_pct <= 0.0f ? "off" : (std::to_string((int)(top_pct * 100 + 0.5f)) + "%").c_str()
    );

    // ── [6b] Pre/post-SPR parsimony summary ──────────────────────────────────
    {
        unsigned int best_pre = UINT_MAX, worst_pre = 0;
        unsigned int best_spr = UINT_MAX, worst_spr = 0;
        unsigned int best_hc  = UINT_MAX, worst_hc  = 0;
        for (int k = 0; k < K; ++k)
        {
            GpuTopology h_topo_tmp;
            downloadTopology(mem, k, &h_topo_tmp, stream);
            cudaStreamSynchronize(stream);
            if (h_topo_tmp.preSprParsimony < best_pre)   best_pre = h_topo_tmp.preSprParsimony;
            if (h_topo_tmp.preSprParsimony > worst_pre)  worst_pre = h_topo_tmp.preSprParsimony;
            if (h_topo_tmp.postSprParsimony < best_spr)  best_spr = h_topo_tmp.postSprParsimony;
            if (h_topo_tmp.postSprParsimony > worst_spr) worst_spr = h_topo_tmp.postSprParsimony;
            if (h_topo_tmp.bestParsimony < best_hc)      best_hc = h_topo_tmp.bestParsimony;
            if (h_topo_tmp.bestParsimony > worst_hc)     worst_hc = h_topo_tmp.bestParsimony;
        }
        if (params.sprDist > 0)
        {
            printf("[GPU]   [6b] Pre-SPR          parsimony: best=%u  worst=%u\n", best_pre, worst_pre);
            printf("[GPU]   [6b] Post-SPR         parsimony: best=%u  worst=%u\n", best_spr, worst_spr);
            if (numSearchIter > 0)
                printf("[GPU]   [6b] Post-Hillclimbing parsimony: best=%u  worst=%u\n", best_hc, worst_hc);
        }
        else
        {
            printf("[GPU]   [6b] Build parsimony: best=%u  worst=%u\n", best_hc, worst_hc);
        }
    }

    // ── [7] Download + Newick ─────────────────────────────────────────────────
    int built = 0;
    double t_download = 0, t_clone = 0, t_newick = 0;

    for (int i = 0; i < K; ++i)
    {
        auto ti = std::chrono::high_resolution_clock::now();
        GpuTopology h_topo;
        downloadTopology(mem, i, &h_topo, stream);
        // Restore the best-seen topology (not the end-state after last iteration)
        for (int vf = 0; vf < h_topo.num_vfaces; vf++)
            h_topo.back_vf[vf] = h_topo.best_back_vf[vf];
        t_download += msSince(ti);

        ti = std::chrono::high_resolution_clock::now();
        pllInstance* localInst = pllInstanceClone(tr);
        partitionList* localPr = pllPartitionsClone(pr);
        gpuTopoToCpu(&h_topo, localInst);
        t_clone += msSince(ti);

        ti = std::chrono::high_resolution_clock::now();
        pllTreeToNewick(
            localInst->tree_string, localInst, localPr, localInst->start->back, PLL_TRUE, PLL_TRUE,
            PLL_FALSE, PLL_FALSE, PLL_FALSE, PLL_SUMMARIZE_LH, PLL_FALSE, PLL_FALSE
        );
        candidateTrees[i + 1] = std::string(localInst->tree_string);
        if (!candidateTrees[i + 1].empty())
        {
            built++;
        }
        t_newick += msSince(ti);

        pllPartitionsCloneFree(localPr);
        pllInstanceCloneFree(localInst);
    }

    printf("[GPU]   [7a] Download topologies (D→H):   %.1f ms total\n", t_download);
    printf("[GPU]   [7b] Clone + gpuTopoToCpu:         %.1f ms total\n", t_clone);
    printf("[GPU]   [7c] Newick conversion:            %.1f ms total\n", t_newick);
    printf("[GPU]   built = %d / %d trees\n", built, K);
    printf("[GPU] ──────────────────────────────────────────────────────\n\n");

    gpuParsimonyMemFree(mem);
    return built;
}

}  // namespace mpbootgpu
