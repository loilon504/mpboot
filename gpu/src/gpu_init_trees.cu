#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
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
int mpbootGpu(
    const Params& params, IQTree& iqtree, int numInitTrees, std::vector<std::string>& candidateTrees
)
{
    struct StdoutUnbuf {
        StdoutUnbuf()  { setvbuf(stdout, nullptr, _IONBF, 0); }
        ~StdoutUnbuf() { setvbuf(stdout, nullptr, _IOLBF, 0); }
    } _stdout_unbuf;
    int K = numInitTrees;

    pllInstance* tr = iqtree.pllInst;
    partitionList* pr = iqtree.pllPartitions;
    const int mxtips = tr->mxtips;

    CUDA_CHECK(cudaSetDevice(params.gpu_device));
    printf("\n[GPU] ═══════════════════════════════════════════════════════\n");
    printf("[GPU]   K=%d  N=%d  device=%d\n", K, mxtips, params.gpu_device);
    printf("[GPU]\n");

    // ── [1] Allocate CPU parsVect & read metadata ─────────────────────────────
    auto t0 = std::chrono::high_resolution_clock::now();
    _allocateParsimonyDataStructures(tr, pr, PLL_FALSE);
    const int width = (int)pr->partitionData[0]->parsimonyLength;
    const int states = (int)pr->partitionData[0]->states;
    printf("[GPU]   [1]      %-28s: %8.3f s  (width=%d states=%d)\n",
           "CPU parsimony alloc", msSince(t0)/1e3, width, states);

    // ── [2] Allocate GPU memory ───────────────────────────────────────────────
    t0 = std::chrono::high_resolution_clock::now();
    GpuParsimonyMem* mem = gpuParsimonyMemAlloc(K, mxtips, width, states);
    printf("[GPU]   [2]      %-28s: %8.3f s\n",
           "GPU memory alloc", msSince(t0)/1e3);

    // ── [3] Upload tip parsVect ───────────────────────────────────────────────
    cudaStream_t stream = 0;
    t0 = std::chrono::high_resolution_clock::now();
    uploadTipParsVect(mem, tr, pr, stream);
    printf("[GPU]   [3]      %-28s: %8.3f s\n",
           "Upload tip parsVect (H->D)", msSince(t0)/1e3);

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
    printf("[GPU]   [4]      %-28s: %8.3f s  (%d trees)\n",
           "Upload topologies (H->D)", msSince(t0)/1e3, K);
    printf("[GPU]\n");

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
    const float top_pct       = params.gpu_top_pct;  // Opt-G2: ≤0=disabled

    // ── Hybrid callback: build CPU trees while K1 runs ───────────────────────
    // Only active in two-kernel mode (top_pct > 0).
    AfterK1Callback hybrid_cb = nullptr;
    if (top_pct > 0.0f)
    {
        hybrid_cb = [&](cudaStream_t cb_stream, GpuParsimonyMem* cb_mem) -> unsigned int {
            // Step 1: Build CPU trees while K1 runs on GPU
            std::vector<CpuTreeData> cpu_trees;
            while (cudaStreamQuery(cb_stream) == cudaErrorNotReady)
            {
                pllInstance*  cpu_inst = pllInstanceClone(tr);
                partitionList* cpu_pr  = pllPartitionsClone(pr);
                cpu_inst->randomNumberSeed =
                    params.ran_seed + (int)cpu_trees.size() * 7919;
                _pllComputeRandomizedStepwiseAdditionParsimonyTree(
                    cpu_inst, cpu_pr, params.sprDist + 3, &iqtree
                );
                unsigned int score = cpu_inst->bestParsimony;

                CpuTreeData ct;
                ct.score = score;
                cpuToGpuTopology(cpu_inst, &ct.topo);
                ct.topo.preSprParsimony  = score;
                ct.topo.postSprParsimony = score;
                ct.topo.bestParsimony    = score;
                ct.topo.savedSeed = params.ran_seed + (long)cpu_trees.size() * 31337L;
                ct.topo.needs_recompute  = 1;
                cpu_trees.push_back(ct);

                pllPartitionsCloneFree(cpu_pr);
                pllInstanceCloneFree(cpu_inst);
            }
            CUDA_CHECK(cudaStreamSynchronize(cb_stream));  // K1 fully done

            // Step 2: Download GPU scores, combine with CPU scores, compute threshold
            const int Kc = cb_mem->K;
            std::vector<unsigned int> gpu_scores(Kc);
            CUDA_CHECK(cudaMemcpy(
                gpu_scores.data(), cb_mem->d_postSprScores,
                (size_t)Kc * sizeof(unsigned int), cudaMemcpyDeviceToHost
            ));

            std::vector<unsigned int> all_scores = gpu_scores;
            for (auto& ct : cpu_trees)
                all_scores.push_back(ct.score);
            std::sort(all_scores.begin(), all_scores.end());

            int top_k = max(1, (int)ceil((double)all_scores.size() * top_pct));
            unsigned int threshold = all_scores[top_k - 1];

            // Step 3: Upload qualifying CPU trees into worst GPU slots above threshold
            std::vector<int> replace_slots;
            for (int k = 0; k < Kc; k++)
                if (gpu_scores[k] > threshold)
                    replace_slots.push_back(k);
            std::sort(replace_slots.begin(), replace_slots.end(),
                      [&](int a, int b) { return gpu_scores[a] > gpu_scores[b]; });

            int n_replace = (int)std::min(cpu_trees.size(), replace_slots.size());
            int n_uploaded = 0;
            for (int i = 0; i < n_replace; i++)
            {
                if (cpu_trees[i].score > threshold) break;
                uploadTopology(cb_mem, replace_slots[i], &cpu_trees[i].topo, cb_stream);
                n_uploaded++;
            }

            printf("[GPU]   [hybrid]  CPU built %d trees, %d uploaded to GPU slots\n",
                   (int)cpu_trees.size(), n_uploaded);
            return threshold;
        };
    }

    float build_ms = ev_time(
        [&]
        {
            gpuStepwiseBuildTrees(mem, seeds.data(), params.sprDist,
                                  numSearchIter, numNNI, stopNoImprove, top_pct, stream,
                                  hybrid_cb);
        }
    );
    printf("[GPU]   [5+6+7]  %-28s: %8.3f s  (%d trees, %.2f ms/tree)\n",
           "Kernel (build+SPR+search)", (double)build_ms/1e3,
           K, K > 0 ? (double)build_ms / K : 0.0);
    printf("[GPU]            sprDist=%d  iters=%d  NNI=%d(%.2f)  stop=%d  top_pct=%s\n",
           params.sprDist, numSearchIter, numNNI, params.gpu_nni_strength, stopNoImprove,
           top_pct <= 0.0f ? "off" : (std::to_string((int)(top_pct * 100 + 0.5f)) + "%").c_str());
    printf("[GPU]\n");

    // ── [6b] Pre/post-SPR parsimony summary + phase effectiveness ───────────
    {
        unsigned int best_pre = UINT_MAX, worst_pre = 0;
        unsigned int best_spr = UINT_MAX, worst_spr = 0;
        unsigned int best_hc  = UINT_MAX, worst_hc  = 0;
        // Phase 3 per-phase counters (aggregated across all K trees)
        int sum_improved_even = 0, sum_total_even = 0;
        int sum_improved_odd  = 0, sum_total_odd  = 0;

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

            sum_improved_even += h_topo_tmp.n_improved_even;
            sum_total_even    += h_topo_tmp.n_total_even;
            sum_improved_odd  += h_topo_tmp.n_improved_odd;
            sum_total_odd     += h_topo_tmp.n_total_odd;
        }
        if (params.sprDist > 0)
        {
            printf("[GPU]   [6b]     %-22s:  best=%-7u  worst=%u\n", "Pre-SPR  parsimony",  best_pre, worst_pre);
            printf("[GPU]   [6b]     %-22s:  best=%-7u  worst=%u\n", "Post-SPR parsimony",  best_spr, worst_spr);
            if (numSearchIter > 0)
            {
                printf("[GPU]   [6b]     %-22s:  best=%-7u  worst=%u\n", "Post-HC  parsimony",  best_hc,  worst_hc);
                printf("[GPU]   [6b]     %-22s:  %d/%d iters improved (%.1f%%)\n",
                       "NNI+SPR (even)", sum_improved_even, sum_total_even,
                       sum_total_even > 0 ? 100.0 * sum_improved_even / sum_total_even : 0.0);
                printf("[GPU]   [6b]     %-22s:  %d/%d iters improved (%.1f%%)\n",
                       "Ratchet (odd)", sum_improved_odd, sum_total_odd,
                       sum_total_odd > 0 ? 100.0 * sum_improved_odd / sum_total_odd : 0.0);
            }
        }
        else
        {
            printf("[GPU]   [6b]     %-22s:  best=%-7u  worst=%u\n", "Build parsimony", best_hc, worst_hc);
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
        candidateTrees[i] = std::string(localInst->tree_string);
        if (!candidateTrees[i].empty())
        {
            built++;
        }
        t_newick += msSince(ti);

        pllPartitionsCloneFree(localPr);
        pllInstanceCloneFree(localInst);
    }

    printf("[GPU]\n");
    printf("[GPU]   [7a]     %-28s: %8.3f s\n", "Download topologies (D->H)", t_download/1e3);
    printf("[GPU]   [7b]     %-28s: %8.3f s\n", "Clone + gpuTopoToCpu",      t_clone/1e3);
    printf("[GPU]   [7c]     %-28s: %8.3f s\n", "Newick conversion",         t_newick/1e3);
    printf("[GPU]   built = %d / %d trees\n", built, K);
    printf("[GPU] ═══════════════════════════════════════════════════════\n\n");

    gpuParsimonyMemFree(mem);
    return built;
}

}  // namespace mpbootgpu
