#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
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
    printf("\n[GPU] ═══════════════════════════════════════════════════════\n");
    printf("[GPU]   K=%d  N=%d  device=%d\n", K, mxtips, params.gpu_device);
    printf("[GPU]\n");

    // ── [1] Allocate CPU parsVect & read metadata ─────────────────────────────
    auto t0 = std::chrono::high_resolution_clock::now();
    _allocateParsimonyDataStructures(tr, pr, PLL_FALSE);
    const int width = (int)pr->partitionData[0]->parsimonyLength;
    const int states = (int)pr->partitionData[0]->states;
    printf(
        "[GPU]   [1]      %-28s: %8.3f s  (width=%d states=%d)\n", "CPU parsimony alloc",
        msSince(t0) / 1e3, width, states
    );

    // ── [2] Allocate GPU memory ───────────────────────────────────────────────
    t0 = std::chrono::high_resolution_clock::now();
    GpuParsimonyMem* mem = gpuParsimonyMemAlloc(K, mxtips, width, states);
    printf("[GPU]   [2]      %-28s: %8.3f s\n", "GPU memory alloc", msSince(t0) / 1e3);

    // ── [3] Upload tip parsVect ───────────────────────────────────────────────
    cudaStream_t stream = 0;
    t0 = std::chrono::high_resolution_clock::now();
    uploadTipParsVect(mem, tr, pr, stream);
    printf("[GPU]   [3]      %-28s: %8.3f s\n", "Upload tip parsVect (H->D)", msSince(t0) / 1e3);

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
    printf(
        "[GPU]   [4]      %-28s: %8.3f s  (%d trees)\n", "Upload topologies (H->D)",
        msSince(t0) / 1e3, K
    );
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
    const int numNNI = (mxtips > 4) ? max(1, (int)(params.gpu_nni_strength * (mxtips - 3))) : 1;
    const int numSearchIter = params.gpu_hc_iter;
    const int stopNoImprove = params.gpu_stop;
    const float top_pct = params.gpu_top_pct;  // Opt-G2: ≤0=disabled

    // ── Hybrid callbacks: Phase 1 (CPU builds during K1) + Phase 2 (CPU HC during K2) ─────
    // Only active in two-kernel mode (top_pct > 0).
    AfterK1Callback hybrid_cb = nullptr;
    AfterK2Callback hybrid_cb2 = nullptr;

    if (top_pct > 0.0f)
    {
        // ── Phase 1 callback: build CPU trees while K1 runs ──────────────────
        hybrid_cb = [&](cudaStream_t cb_stream, GpuParsimonyMem* cb_mem) -> unsigned int
        {
            iqtree.candidateTrees.aln = iqtree.aln;
            // Step 1: Build CPU trees while K1 runs on GPU
            std::vector<CpuTreeData> cpu_trees;
            while (cudaStreamQuery(cb_stream) == cudaErrorNotReady)
            {
                std::string curParsTree;
                tr->randomNumberSeed = params.ran_seed + (int)cpu_trees.size() * 7919;
                _pllComputeRandomizedStepwiseAdditionParsimonyTree(
                    tr, pr, params.sprDist + 3, &iqtree
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
                ct.topo.needs_recompute = 1;
                cpu_trees.push_back(ct);

            }
            CUDA_CHECK(cudaStreamSynchronize(cb_stream));  // K1 fully done

            // Step 2: Download GPU scores, combine with CPU scores, compute threshold
            const int Kc = cb_mem->K;
            std::vector<unsigned int> gpu_scores(Kc);
            CUDA_CHECK(cudaMemcpy(
                gpu_scores.data(), cb_mem->d_postSprScores, (size_t)Kc * sizeof(unsigned int),
                cudaMemcpyDeviceToHost
            ));

            std::vector<unsigned int> all_scores = gpu_scores;
            for (auto& ct : cpu_trees)
            {
                all_scores.push_back(ct.score);
            }
            std::sort(all_scores.begin(), all_scores.end());

            int top_k = max(1, (int)ceil((double)all_scores.size() * top_pct));
            unsigned int threshold = all_scores[top_k - 1];

            // Step 3: Upload qualifying CPU trees into worst GPU slots above threshold
            std::vector<int> replace_slots;
            for (int k = 0; k < Kc; k++)
            {
                if (gpu_scores[k] > threshold)
                {
                    replace_slots.push_back(k);
                }
            }
            std::sort(
                replace_slots.begin(), replace_slots.end(),
                [&](int a, int b)
                {
                    return gpu_scores[a] > gpu_scores[b];
                }
            );

            // Sort CPU trees best-first (ascending score = lower parsimony = better),
            // so the best CPU tree is paired with the worst GPU slot and the break below
            // is correct (once score > threshold, all remaining are also > threshold).
            std::sort(cpu_trees.begin(), cpu_trees.end(),
                [](const CpuTreeData& a, const CpuTreeData& b){ return a.score < b.score; });

            int n_replace = (int)std::min(cpu_trees.size(), replace_slots.size());
            int n_uploaded = 0;
            for (int i = 0; i < n_replace; i++)
            {
                if (cpu_trees[i].score > threshold)
                {
                    break;  // sorted ascending: all remaining trees are also > threshold
                }
                uploadTopology(cb_mem, replace_slots[i], &cpu_trees[i].topo, cb_stream);
                n_uploaded++;
            }

            // Step 3.5: Build threshold pool + reseed remaining bad GPU slots
            int n_reseeded = 0;
            {
                int n_remaining = (int)replace_slots.size() - n_uploaded;
                if (n_remaining > 0)
                {
                    // a) Collect good GPU topologies: download + restore best_back_vf.
                    //    Step 3 uploaded to replace_slots[] (score > threshold) only;
                    //    here we read slots with score <= threshold — no conflict.
                    std::vector<GpuTopology> good_gpu_topos;
                    for (int k = 0; k < Kc; k++)
                    {
                        if (gpu_scores[k] <= threshold)
                        {
                            GpuTopology h_topo;
                            downloadTopology(cb_mem, k, &h_topo, cb_stream);
                            cudaStreamSynchronize(cb_stream);
                            for (int vf = 0; vf < h_topo.num_vfaces; vf++)
                                h_topo.back_vf[vf] = h_topo.best_back_vf[vf];
                            good_gpu_topos.push_back(h_topo);
                        }
                    }

                    // b) Build combined pool: good GPU topos + good CPU trees.
                    //    cpu_trees already sorted ascending (best-first) from Step 3.
                    std::vector<const GpuTopology*> pool;
                    for (int i = 0; i < (int)good_gpu_topos.size(); i++)
                        pool.push_back(&good_gpu_topos[i]);
                    for (int i = 0; i < (int)cpu_trees.size(); i++)
                    {
                        if (cpu_trees[i].score <= threshold)
                            pool.push_back(&cpu_trees[i].topo);
                        else
                            break;  // sorted ascending: rest all > threshold
                    }

                    // c) Reseed each remaining bad slot from the pool.
                    //    K2 filter checks topo->postSprParsimony (from GpuTopology struct).
                    //    Donor's postSprParsimony <= threshold → reseeded slot enters K2.
                    //    needs_recompute=1 triggers full parsVect rebuild inside K2.
                    if (!pool.empty())
                    {
                        for (int i = n_uploaded; i < (int)replace_slots.size(); i++)
                        {
                            int slot = replace_slots[i];
                            GpuTopology fill_topo = *pool[random_int((int)pool.size())];
                            fill_topo.savedSeed = params.ran_seed + (long)slot * 99991L;
                            fill_topo.needs_recompute = 1;
                            uploadTopology(cb_mem, slot, &fill_topo, cb_stream);
                            n_reseeded++;
                        }
                    }
                }
            }

            // Step 4: add qualifying GPU trees into iqtree.candidateTrees
            // First, purge candidateTrees entries worse than threshold
            {
                double score_threshold = -(double)threshold;
                iqtree.candidateTrees.erase(
                    iqtree.candidateTrees.begin(),
                    iqtree.candidateTrees.lower_bound(score_threshold)
                );
            }
            int n_gpu_added = 0;
            for (int k = 0; k < Kc; k++)
            {
                if (gpu_scores[k] > threshold)
                {
                    continue;
                }

                GpuTopology h_topo;
                downloadTopology(cb_mem, k, &h_topo, cb_stream);
                cudaStreamSynchronize(cb_stream);
                for (int vf = 0; vf < h_topo.num_vfaces; vf++)
                {
                    h_topo.back_vf[vf] = h_topo.best_back_vf[vf];
                }

                gpuTopoToCpu(&h_topo, tr);
                pllTreeToNewick(
                    tr->tree_string, tr, pr, tr->start->back, PLL_TRUE, PLL_TRUE, PLL_FALSE,
                    PLL_FALSE, PLL_FALSE, PLL_SUMMARIZE_LH, PLL_FALSE, PLL_FALSE
                );
                std::string gpuTree = std::string(tr->tree_string);

                if (gpuTree.empty() || iqtree.candidateTrees.treeExist(gpuTree))
                {
                    continue;
                }

                iqtree.readTreeString(gpuTree);
                iqtree.initializeAllPartialPars();
                iqtree.clearAllPartialLH();
                double score = -(double)iqtree.computeParsimony();
                iqtree.candidateTrees.update(gpuTree, score);
                if (score > iqtree.bestScore)
                {
                    iqtree.setBestTree(gpuTree, score);
                }

                n_gpu_added++;
            }

            int n_gpu_good = Kc - (int)replace_slots.size();
            printf(
                "[CPU]   CPU built %d trees, %d uploaded to GPU slots, %d slots reseeded; "
                "%d GPU trees added to candidateTrees\n",
                (int)cpu_trees.size(), n_uploaded, n_reseeded, n_gpu_added
            );
            printf(
                "[GPU]   hillClimbingKernel: threshold=%u  (%d GPU + %d CPU + %d reseeded = %d "
                "trees enter K2)\n",
                threshold, n_gpu_good, n_uploaded, n_reseeded,
                n_gpu_good + n_uploaded + n_reseeded
            );

            return threshold;
        };

        // ── Phase 2 callback: alternating NNI/ratchet perturb, PhyloTree scoring ──
        // LIMITATION: PLL parsimony (doNNISearch, _pllSprOnCurrentTree) causes heap
        // corruption via SIMD write overflow in sprparsimony.cpp Fitch algorithm —
        // small writes beyond parsVect accumulate corruption over ~300 iterations.
        // Fix requires modifying sprparsimony.cpp (out of scope for gpu_init_trees.cu).
        //
        // Strategy: alternate even (NNI perturb) and odd (ratchet perturb) iterations
        // with PhyloTree::computeParsimony() scoring only — safe indefinitely.
        hybrid_cb2 = [&](cudaStream_t cb_stream)
        {
            if (iqtree.candidateTrees.empty())
            {
                CUDA_CHECK(cudaStreamSynchronize(cb_stream));
                printf("[CPU]   CPU Hill-Climbing: candidateTrees empty, skip\n");
                return;
            }

            iqtree.setRootNode(params.root);
            int n_iters = 0;
            int iter = 0;  // 0-based: even=NNI perturb, odd=ratchet perturb
            std::string imd_tree;
            iqtree.on_ratchet_hclimb1 = false;

            while (cudaStreamQuery(cb_stream) == cudaErrorNotReady)
            {
                // Pick 1 candidate from top 10% (mirrors getRandCandTree line 1699/1746).
                int top_n = std::max(1, (int)std::ceil(0.1 * (double)iqtree.candidateTrees.size()));
                int pick = random_int(top_n);
                std::string candidateTree;
                {
                    int idx = 0;
                    for (auto it = iqtree.candidateTrees.rbegin();
                         it != iqtree.candidateTrees.rend(); ++it, ++idx)
                    {
                        if (idx == pick)
                        {
                            candidateTree = it->second.tree;
                            break;
                        }
                    }
                }
                if (candidateTree.empty())
                {
                    break;
                }

                iqtree.readTreeString(candidateTree);

                Alignment* saved_aln = nullptr;

                /*--------------------------------------------------------------------------
                 * PARSIMONY RATCHET-LIKE IDEA
                 * -------------------------------------------------------------------------*/
                //		long tmp_num_ratchet_trees = treels_logl.size();
                //		long tmp_num_ratchet_bootcands = treels.size();
                if (params.ratchet_iter >= 0)
                {
                    if (params.ratchet_iter == iter)
                    {
                        Alignment* perturb_alignment;
                        perturb_alignment = new Alignment;
                        perturb_alignment->createPerturbAlignment(
                            iqtree.aln, params.ratchet_percent, params.ratchet_wgt,
                            params.sort_alignment
                        );
                        saved_aln = iqtree.aln;

                        iqtree.setAlignment(perturb_alignment);
                        iqtree.setRootNode(params.root);
                        iqtree.on_ratchet_hclimb1 = true;

                        iqtree.initializeAllPartialLh();
                        iqtree.clearAllPartialLH();
                        iqtree.curScore = iqtree.optimizeAllBranches();
                    }
                    iter++;
                }

                /*----------------------------------------
                 * Perturb the tree
                 *---------------------------------------*/
                double perturbScore;
                if (!iqtree.on_ratchet_hclimb1)
                {
                    int numNNI = std::max(1, (int)floor(params.gpu_nni_strength * (iqtree.aln->getNSeq() - 3)));
                    iqtree.doRandomNNIs(numNNI);
                    iqtree.setAlignment(iqtree.aln);
                    iqtree.setRootNode(params.root);
                    iqtree.initializeAllPartialPars();
                    iqtree.clearAllPartialLH();
                    iqtree.curScore = -iqtree.computeParsimony();
                }

                int nni_count = 0;
                int nni_steps = 0;

                // Monitor thread: set stop_search=1 when GPU K2 stream finishes,
                // so the SPR loop inside doNNISearch exits early.
                tr->stop_search = 0;
                std::thread monitor([cb_stream, tr]() {
                    cudaStreamSynchronize(cb_stream);
                    tr->stop_search = 1;
                });
                imd_tree = iqtree.doNNISearch(nni_count, nni_steps);
                monitor.join();
                tr->stop_search = 0;

                if (iqtree.on_ratchet_hclimb1)
                {
                    // cout << "Iteration " << curIt << ", iqtree.on_ratchet_hclimb1 = true\n";
                    iter = 0;

                    // restore alignment
                    delete iqtree.aln;
                    iqtree.setAlignment(saved_aln);
                    iqtree.on_ratchet_hclimb1 = false;

                    iqtree.initializeAllPartialLh();
                    iqtree.clearAllPartialLH();
                    iqtree.curScore = iqtree.optimizeAllBranches();

                    /*----------------------------------------
                     * Optimize tree with NNI
                     *---------------------------------------*/
                    int nni_count = 0;
                    int nni_steps = 0;
                    iqtree.on_ratchet_hclimb2 = true;
                    tr->stop_search = 0;
                    std::thread monitor2([cb_stream, tr]() {
                        cudaStreamSynchronize(cb_stream);
                        tr->stop_search = 1;
                    });
                    imd_tree = iqtree.doNNISearch(nni_count, nni_steps);
                    monitor2.join();
                    tr->stop_search = 0;
                    // update current score
                    iqtree.initializeAllPartialLh();
                    iqtree.clearAllPartialLH();
                    iqtree.curScore = iqtree.optimizeAllBranches();
                }

                if (iqtree.on_ratchet_hclimb2)
                {
                    iqtree.on_ratchet_hclimb2 = false;
                }
                // Diep: This is old code for updating best tree
                if (iqtree.curScore > iqtree.bestScore)
                {
                    stringstream cur_tree_topo_ss;
                    iqtree.setRootNode(params.root);
                    printf(
                        "[CPU]   CPU found better tree at iteration %3d: %.0f\n", n_iters, -iqtree.curScore
                    );
                    // printTree(cur_tree_topo_ss, WT_TAXON_ID | WT_SORT_TAXA);
                    // if (cur_tree_topo_ss.str() != best_tree_topo) {
                    //     best_tree_topo = cur_tree_topo_ss.str();
                    //     // Diep: fix Minh's old if which wrongly set imd_tree = best_tree_topo
                    //     for mpars if (!params->maximum_parsimony)
                    //         imd_tree = optimizeModelParameters();
                    //     stop_rule.addImprovedIteration(curIt);
                    //     cout << "BETTER TREE FOUND at iteration " << curIt << ": " <<
                    //     -iqtree.curScore; cout << " / CPU time: " << (int) round(getCPUTime() -
                    //     params->startCPUTime) << "s" << endl << endl; if (iqtree.curScore >
                    //     iqtree.bestScore) {
                    //         searchinfo.curPerStrength = params->initPerStrength;
                    //     }
                    // } else {
                    //     cout << "UPDATE BEST LOG-LIKELIHOOD: " << iqtree.curScore << endl;
                    // }
                    iqtree.setBestTree(imd_tree, iqtree.curScore);
                    // if (params->write_best_trees) {
                    //     ostringstream iter_string;
                    //     iter_string << curIt;
                    //     printResultTree(iter_string.str());
                    // }
                    // printResultTree();
                }

                iqtree.candidateTrees.update(imd_tree, iqtree.curScore);

                n_iters++;
            }

            iqtree.readTreeString(iqtree.bestTreeString);
            CUDA_CHECK(cudaStreamSynchronize(cb_stream));

            printf(
                "[CPU]   CPU Hill-Climbing: %d iters (%d NNI, %d ratchet), bestScore = %.0f\n",
                n_iters, (n_iters + 1) / 2, n_iters / 2, -iqtree.bestScore
            );
        };
    };

    float build_ms = ev_time(
        [&]
        {
            gpuStepwiseBuildTrees(
                mem, seeds.data(), params.sprDist, numSearchIter, numNNI, stopNoImprove, top_pct,
                stream, hybrid_cb, hybrid_cb2
            );
        }
    );
    printf(
        "[GPU]   [5]  %-28s: %8.3f s  (%d trees, %.2f ms/tree)\n", "Kernels",
        (double)build_ms / 1e3, K, K > 0 ? (double)build_ms / K : 0.0
    );
    printf(
        "[GPU]            sprDist=%d  NNI=%d(%.2f)  stop=%d  top_pct=%s\n",
        params.sprDist, numNNI, params.gpu_nni_strength, stopNoImprove,
        top_pct <= 0.0f ? "off" : (std::to_string((int)(top_pct * 100 + 0.5f)) + "%").c_str()
    );
    printf("[GPU]\n");

    // ── [6b] Pre/post-SPR parsimony summary + phase effectiveness ───────────
    unsigned int best_hc = UINT_MAX;
    {
        unsigned int best_pre = UINT_MAX, worst_pre = 0;
        unsigned int best_spr = UINT_MAX, worst_spr = 0;
        unsigned int worst_hc = 0;
        // Phase 3 per-phase counters (aggregated across all K trees)
        int sum_improved_even = 0, sum_total_even = 0;
        int sum_improved_odd = 0, sum_total_odd = 0;

        for (int k = 0; k < K; ++k)
        {
            GpuTopology h_topo_tmp;
            downloadTopology(mem, k, &h_topo_tmp, stream);
            cudaStreamSynchronize(stream);
            if (h_topo_tmp.preSprParsimony < best_pre)
            {
                best_pre = h_topo_tmp.preSprParsimony;
            }
            if (h_topo_tmp.preSprParsimony > worst_pre)
            {
                worst_pre = h_topo_tmp.preSprParsimony;
            }
            if (h_topo_tmp.postSprParsimony < best_spr)
            {
                best_spr = h_topo_tmp.postSprParsimony;
            }
            if (h_topo_tmp.postSprParsimony > worst_spr)
            {
                worst_spr = h_topo_tmp.postSprParsimony;
            }
            if (h_topo_tmp.bestParsimony < best_hc)
            {
                best_hc = h_topo_tmp.bestParsimony;
            }
            if (h_topo_tmp.bestParsimony > worst_hc)
            {
                worst_hc = h_topo_tmp.bestParsimony;
            }

            sum_improved_even += h_topo_tmp.n_improved_even;
            sum_total_even += h_topo_tmp.n_total_even;
            sum_improved_odd += h_topo_tmp.n_improved_odd;
            sum_total_odd += h_topo_tmp.n_total_odd;
        }
        if (params.sprDist > 0)
        {
            printf(
                "[GPU]   [6]     %-22s:  best=%-7u  worst=%u\n", "Pre-SPR  parsimony", best_pre,
                worst_pre
            );
            printf(
                "[GPU]   [6]     %-22s:  best=%-7u  worst=%u\n", "Post-SPR parsimony", best_spr,
                worst_spr
            );
            if (numSearchIter > 0)
            {
                printf(
                    "[GPU]   [6]     %-22s:  best=%-7u  worst=%u\n", "Post-HC  parsimony", best_hc,
                    worst_hc
                );
                printf(
                    "[GPU]   [6]     %-22s:  %d/%d iters improved (%.1f%%)\n", "NNI+SPR",
                    sum_improved_even, sum_total_even,
                    sum_total_even > 0 ? 100.0 * sum_improved_even / sum_total_even : 0.0
                );
                printf(
                    "[GPU]   [6]     %-22s:  %d/%d iters improved (%.1f%%)\n", "Ratchet",
                    sum_improved_odd, sum_total_odd,
                    sum_total_odd > 0 ? 100.0 * sum_improved_odd / sum_total_odd : 0.0
                );
            }
        }
        else
        {
            printf(
                "[GPU]   [6]     %-22s:  best=%-7u  worst=%u\n", "Build parsimony", best_hc,
                worst_hc
            );
        }
    }

    // ── [7] Download + Newick + register into candidateTrees ─────────────────
    // Capture CPU best parsimony before [7] loop may update bestScore via setBestTree
    unsigned int best_cpu_pars = (iqtree.bestScore < 0) ? (unsigned int)(-iqtree.bestScore) : 0;

    int built = 0;
    double t_download = 0, t_topo = 0, t_newick = 0;

    for (int i = 0; i < K; ++i)
    {
        auto ti = std::chrono::high_resolution_clock::now();
        GpuTopology h_topo;
        downloadTopology(mem, i, &h_topo, stream);
        // Restore the best-seen topology (not the end-state after last iteration)
        for (int vf = 0; vf < h_topo.num_vfaces; vf++)
            h_topo.back_vf[vf] = h_topo.best_back_vf[vf];
        t_download += msSince(ti);

        if (h_topo.bestParsimony > best_hc) {
            candidateTrees[i] = "";
            continue;
        }

        ti = std::chrono::high_resolution_clock::now();
        gpuTopoToCpu(&h_topo, tr);
        t_topo += msSince(ti);

        ti = std::chrono::high_resolution_clock::now();
        pllTreeToNewick(
            tr->tree_string, tr, pr, tr->start->back, PLL_TRUE, PLL_TRUE,
            PLL_FALSE, PLL_FALSE, PLL_FALSE, PLL_SUMMARIZE_LH, PLL_FALSE, PLL_FALSE
        );
        std::string tree_str(tr->tree_string);
        candidateTrees[i] = tree_str;
        if (!tree_str.empty())
        {
            built++;
            iqtree.readTreeString(tree_str);
            iqtree.initializeAllPartialPars();
            iqtree.clearAllPartialLH();
            iqtree.curScore = -(double)iqtree.computeParsimony();
            bool isNew = iqtree.candidateTrees.update(tree_str, iqtree.curScore);
            if (isNew && iqtree.curScore > iqtree.bestScore)
                iqtree.setBestTree(tree_str, iqtree.curScore);
        }
        t_newick += msSince(ti);
    }

    printf("[GPU]\n");
    printf("[GPU]   [7a]     %-28s: %8.3f s\n", "Download topologies (D->H)", t_download / 1e3);
    printf("[GPU]   [7b]     %-28s: %8.3f s\n", "gpuTopoToCpu", t_topo / 1e3);
    printf("[GPU]   [7c]     %-28s: %8.3f s\n", "Newick conversion", t_newick / 1e3);
    printf(
        "[GPU]   best CPU tree: %-7u  best GPU tree: %u\n", best_cpu_pars, best_hc
    );
    printf("[GPU] ═══════════════════════════════════════════════════════\n\n");

    gpuParsimonyMemFree(mem);
    return built;
}

}  // namespace mpbootgpu
