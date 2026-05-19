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
#include "gpu/include/pars_bootstrap.cuh"
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
    const Params& params, IQTree& iqtree, int numInitTrees, std::vector<std::string>& candidateTrees,
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

    // K2 workers computed early — needed for memory allocation size
    const int k2_workers_early = (params.gpu_worker > 0) ? params.gpu_worker : K;
    const int K_alloc = std::max(K, k2_workers_early);

    // ── [2] Allocate GPU memory ───────────────────────────────────────────────
    t0 = std::chrono::high_resolution_clock::now();
    // Treels buffer: bootstrap K2 writes all good trees each round for REPS eval.
    // Size = K * 10 (up to ~10 outer iters per K2 round × K workers).
    const int max_treels_boot = (params.gbo_replicates > 0) ? K * 10 : 0;
    GpuParsimonyMem* mem = gpuParsimonyMemAlloc(K_alloc, mxtips, width, states,
                                                params.gpu_pool_size, max_treels_boot);
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
    const int pool_size = params.gpu_pool_size;
    // K2 workers: independent of K1; mem->K = K_alloc = max(K, k2_workers)
    const int k2_workers = k2_workers_early;

    // ── Hybrid callbacks: Phase 1 (CPU builds during K1) + Phase 2 (CPU HC during K2) ─────
    AfterK1Callback hybrid_cb = nullptr;
    AfterK2Callback hybrid_cb2 = nullptr;
    {
        // ── Phase 1 callback: build CPU trees while K1 runs ──────────────────
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
            const int Kc_scores = K;          // all K slots written by K1
            const int Kc_full   = k2_workers; // reseed k2_workers slots for K2
            std::vector<unsigned int> gpu_scores(Kc_scores);
            CUDA_CHECK(cudaMemcpy(
                gpu_scores.data(), cb_mem->d_postSprScores, (size_t)Kc_scores * sizeof(unsigned int),
                cudaMemcpyDeviceToHost
            ));

            // Step 3: Diversity selection — best pool_size trees with distinct topologies
            struct Candidate { unsigned int score; int idx; bool is_gpu; };
            std::vector<Candidate> candidates;
            candidates.reserve(Kc_scores + (int)cpu_trees.size());
            for (int k = 0; k < Kc_scores; k++)
                candidates.push_back({gpu_scores[k], k, true});
            for (int i = 0; i < (int)cpu_trees.size(); i++)
                candidates.push_back({cpu_trees[i].score, i, false});
            std::sort(candidates.begin(), candidates.end(),
                [](const Candidate& a, const Candidate& b){ return a.score < b.score; });

            int scan_limit = std::min((int)candidates.size(), pool_size * 5);
            std::unordered_set<std::string> seen_topologies;
            std::vector<int>         pool_ci;      // indices into candidates[]
            std::vector<std::string> pool_newicks; // cached Newicks (for Step 6)
            std::vector<std::vector<int>> pool_bvf; // back_vf arrays (for pool upload + reseed)

            for (int ci = 0; ci < scan_limit && (int)pool_ci.size() < pool_size; ci++)
            {
                const Candidate& cand = candidates[ci];
                std::string newick;
                std::vector<int> bvf;

                if (cand.is_gpu) {
                    GpuTopology h_topo;
                    downloadTopology(cb_mem, cand.idx, &h_topo, cb_stream);
                    cudaStreamSynchronize(cb_stream);
                    bvf.assign(h_topo.back_vf, h_topo.back_vf + h_topo.num_vfaces);
                    gpuTopoToCpu(&h_topo, tr);
                } else {
                    GpuTopology cpu_topo_copy = cpu_trees[cand.idx].topo;
                    bvf.assign(cpu_topo_copy.back_vf,
                               cpu_topo_copy.back_vf + cpu_topo_copy.num_vfaces);
                    gpuTopoToCpu(&cpu_topo_copy, tr);
                }
                pllTreeToNewick(tr->tree_string, tr, pr, tr->start->back, PLL_TRUE, PLL_TRUE,
                                PLL_FALSE, PLL_FALSE, PLL_FALSE, PLL_SUMMARIZE_LH,
                                PLL_FALSE, PLL_FALSE);
                newick = std::string(tr->tree_string);
                if (newick.empty() || !seen_topologies.insert(newick).second)
                    continue;

                pool_ci.push_back(ci);
                pool_newicks.push_back(std::move(newick));
                pool_bvf.push_back(std::move(bvf));
            }
            int actual_pool = (int)pool_ci.size();

            // Step 4: Upload pool to GPU memory
            {
                std::vector<unsigned int> h_pool_scores(pool_size, 0xFFFFFFFFu);
                std::vector<int> h_pool_vf((size_t)pool_size * kMaxVFaces, -1);
                for (int i = 0; i < actual_pool; i++) {
                    h_pool_scores[i] = candidates[pool_ci[i]].score;
                    int* dst = h_pool_vf.data() + (size_t)i * kMaxVFaces;
                    memcpy(dst, pool_bvf[i].data(), pool_bvf[i].size() * sizeof(int));
                }
                CUDA_CHECK(cudaMemcpy(cb_mem->d_poolScores, h_pool_scores.data(),
                                      pool_size * sizeof(unsigned int), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(cb_mem->d_poolBackVf, h_pool_vf.data(),
                                      (size_t)pool_size * kMaxVFaces * sizeof(int),
                                      cudaMemcpyHostToDevice));
                int show = std::min(actual_pool, 5);
                printf("[GPU]   Pool: %d/%d slots, top-%d scores:", actual_pool, pool_size, show);
                for (int i = 0; i < show; i++) printf(" %u", h_pool_scores[i]);
                printf("\n");
            }

            // Step 4b: Init fill counter + per-slot spinlocks
            {
                int h_filled = actual_pool, h_stop = 0;
                unsigned int h_inf = 0xFFFFFFFFu;
                CUDA_CHECK(cudaMemcpy(cb_mem->d_poolFilled,    &h_filled, sizeof(int),          cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemset(cb_mem->d_poolSlotLocks, 0, (size_t)cb_mem->pool_size * sizeof(int)));
                CUDA_CHECK(cudaMemcpy(cb_mem->d_poolStop,      &h_stop,   sizeof(int),          cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(cb_mem->d_globalBest,    &h_inf,    sizeof(unsigned int), cudaMemcpyHostToDevice));
            }

            // Step 5: Reseed ALL K GPU slots from pool (round-robin)
            if (actual_pool > 0) {
                GpuTopology tmpl;
                downloadTopology(cb_mem, 0, &tmpl, cb_stream);
                cudaStreamSynchronize(cb_stream);
                for (int k = 0; k < Kc_full; k++) {
                    int slot = k % actual_pool;
                    const auto& bvf = pool_bvf[slot];
                    unsigned int ps = candidates[pool_ci[slot]].score;
                    GpuTopology h_topo = tmpl;
                    memcpy(h_topo.back_vf, bvf.data(), bvf.size() * sizeof(int));
                    h_topo.postSprParsimony = ps;
                    h_topo.bestParsimony    = ps;
                    h_topo.preSprParsimony  = ps;
                    h_topo.savedSeed        = params.ran_seed + (long)k * 31337L;
                    h_topo.n_improved_even  = h_topo.n_improved_odd = 0;
                    h_topo.n_total_even     = h_topo.n_total_odd    = 0;
                    uploadTopology(cb_mem, k, &h_topo, cb_stream);
                }
                cudaStreamSynchronize(cb_stream);
            }

            // Step 6: Add pool trees to CPU candidateSet
            int n_pool_added = 0;
            for (int i = 0; i < actual_pool; i++) {
                const std::string& newick = pool_newicks[i];
                if (newick.empty() || iqtree.candidateTrees.treeExist(newick)) continue;
                iqtree.readTreeString(newick);
                iqtree.initializeAllPartialPars();
                iqtree.clearAllPartialLH();
                iqtree.curScore = -(double)iqtree.computeParsimony();
                iqtree.candidateTrees.update(newick, iqtree.curScore);
                if (iqtree.curScore > iqtree.bestScore)
                    iqtree.setBestTree(newick, iqtree.curScore);
                n_pool_added++;
            }

            printf("[CPU]   CPU built %d trees; pool: %d/%d diverse; %d added to candidateTrees\n",
                   (int)cpu_trees.size(), actual_pool, pool_size, n_pool_added);
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
            // Bootstrap mode: skip K2 (each bootstrap round runs K1 with per-sample weights).
            const int k2_max_outer = (params.gbo_replicates > 0) ? 0 : -1;
            gpuStepwiseBuildTrees(
                mem, seeds.data(), K, params.sprDist, numNNI,
                pool_size, params.gpu_pool_stop, stream, hybrid_cb, hybrid_cb2,
                k2_workers, k2_max_outer
            );
        }
    );
    printf(
        "[GPU]   [5]  %-28s: %8.3f s  (%d trees, %.2f ms/tree)\n", "Kernels",
        (double)build_ms / 1e3, K, K > 0 ? (double)build_ms / K : 0.0
    );
    printf(
        "[GPU]            sprDist=%d  NNI=%d(%.2f)  pool_stop=%d\n",
        params.sprDist, numNNI, params.gpu_nni_strength, params.gpu_pool_stop
    );
    printf("[GPU]\n");

    // ── [6b] Pre/post-SPR parsimony summary + phase effectiveness ───────────
    // Skip in bootstrap mode: K2 not run here, stats would show 0/0.
    // Pool → candidateTrees is done at end of gpuBootstrapSearch instead.
    unsigned int best_hc = UINT_MAX;
    if (params.gbo_replicates == 0) {
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
                best_pre = h_topo_tmp.preSprParsimony;
            if (h_topo_tmp.preSprParsimony > worst_pre)
                worst_pre = h_topo_tmp.preSprParsimony;
            if (h_topo_tmp.postSprParsimony < best_spr)
                best_spr = h_topo_tmp.postSprParsimony;
            if (h_topo_tmp.postSprParsimony > worst_spr)
                worst_spr = h_topo_tmp.postSprParsimony;

            sum_improved_even += h_topo_tmp.n_improved_even;
            sum_total_even += h_topo_tmp.n_total_even;
            sum_improved_odd += h_topo_tmp.n_improved_odd;
            sum_total_odd += h_topo_tmp.n_total_odd;
        }

        // Aggregate best_hc/worst_hc from pool scores
        {
            std::vector<unsigned int> h_ps(pool_size, 0xFFFFFFFFu);
            CUDA_CHECK(cudaMemcpy(h_ps.data(), mem->d_poolScores,
                pool_size * sizeof(unsigned int), cudaMemcpyDeviceToHost));
            for (int i = 0; i < pool_size; i++)
            {
                if (h_ps[i] == 0xFFFFFFFFu) continue;
                if (h_ps[i] < best_hc) best_hc = h_ps[i];
                if (h_ps[i] > worst_hc) worst_hc = h_ps[i];
            }
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
        else
        {
            printf(
                "[GPU]   [6]     %-22s:  best=%-7u  worst=%u\n", "Build parsimony", best_hc,
                worst_hc
            );
        }
    }

    // ── [7] Download + Newick + register into candidateTrees (non-bootstrap only) ─
    // In bootstrap mode, pool → candidateTrees is done at end of gpuBootstrapSearch.
    int built = 0;
    if (params.gbo_replicates == 0) {
    unsigned int best_cpu_pars = (iqtree.bestScore < 0) ? (unsigned int)(-iqtree.bestScore) : 0;
    double t_download = 0, t_topo = 0, t_newick = 0;

    {
        // Download pool_scores + pool_back_vf directly.
        // Use warp 0 as template for scalar fields (num_vfaces, mxtips, start_vface).
        // gpuTopoToCpu only needs back_vf + start_vface — does not read nodep[].
        auto ti = std::chrono::high_resolution_clock::now();
        std::vector<unsigned int> h_ps(pool_size, 0xFFFFFFFFu);
        std::vector<int> h_pvf((size_t)pool_size * kMaxVFaces);
        CUDA_CHECK(cudaMemcpy(h_ps.data(), mem->d_poolScores,
            pool_size * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_pvf.data(), mem->d_poolBackVf,
            (size_t)pool_size * kMaxVFaces * sizeof(int), cudaMemcpyDeviceToHost));
        GpuTopology tmpl;
        downloadTopology(mem, 0, &tmpl, stream);
        cudaStreamSynchronize(stream);
        t_download += msSince(ti);

        for (int i = 0; i < pool_size; i++)
        {
            if (h_ps[i] == 0xFFFFFFFFu || h_ps[i] > best_hc) continue;

            GpuTopology h_topo = tmpl;
            memcpy(h_topo.back_vf, h_pvf.data() + (size_t)i * kMaxVFaces,
                   h_topo.num_vfaces * sizeof(int));
            h_topo.bestParsimony = h_ps[i];

            ti = std::chrono::high_resolution_clock::now();
            gpuTopoToCpu(&h_topo, tr);
            t_topo += msSince(ti);

            ti = std::chrono::high_resolution_clock::now();
            pllTreeToNewick(
                tr->tree_string, tr, pr, tr->start->back, PLL_TRUE, PLL_TRUE,
                PLL_FALSE, PLL_FALSE, PLL_FALSE, PLL_SUMMARIZE_LH, PLL_FALSE, PLL_FALSE
            );
            std::string tree_str(tr->tree_string);
            t_newick += msSince(ti);

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
        }
    }

    printf("[GPU]\n");
    printf("[GPU]   [7a]     %-28s: %8.3f s\n", "Download topologies (D->H)", t_download / 1e3);
    printf("[GPU]   [7b]     %-28s: %8.3f s\n", "gpuTopoToCpu", t_topo / 1e3);
    printf("[GPU]   [7c]     %-28s: %8.3f s\n", "Newick conversion", t_newick / 1e3);
    printf(
        "[GPU]   best CPU tree: %-7u  best GPU tree: %u\n", best_cpu_pars, best_hc
    );
    } // end if (params.gbo_replicates == 0)
    printf("[GPU] ═══════════════════════════════════════════════════════\n\n");
    (void)best_hc;

    if (out_mem != nullptr) {
        *out_mem = mem;  // caller takes ownership, responsible for gpuParsimonyMemFree
    } else {
        gpuParsimonyMemFree(mem);
    }
    return built;
}

// ─── gpuBootstrapSearch ───────────────────────────────────────────────────────
// Replaces CPU doTreeSearch() for -use_gpu -bb.
// Each round: K1 runs with per-worker bootstrapped site weights → diverse trees → REPS eval.
void gpuBootstrapSearch(
    const Params& params,
    IQTree&        iqtree,
    GpuParsimonyMem* mem
)
{
    cudaStream_t stream = 0;
    const int K          = mem->K;
    const int B          = params.gbo_replicates;
    const int sprDist    = params.sprDist;
    const int numNNI     = std::max(1, (int)(params.gpu_nni_strength * (mem->mxtips - 3)));
    const int pool_size  = mem->pool_size;
    const int width      = mem->width;
    const int step_iter  = params.step_iterations;
    // Convergence check every step_iter_rounds rounds (mirrors CPU every step_iter/2 iters).
    const int step_iter_rounds = std::max(1, std::max(1, step_iter / 2) / K);

    iqtree.params->store_candidate_trees = true;

    // Template topology for treels reconstruction (back_vf only stored in treels;
    // other fields come from here). start_vface=0 → tr->start=tip1; start->back = valid Newick root.
    GpuTopology h_tpl;
    downloadTopology(mem, 0, &h_tpl, stream);
    h_tpl.start_vface = 0;
    memset(h_tpl.xpars, 0, sizeof(h_tpl.xpars));

    // Reusable host buffer for batch-downloading treels back_vf each round.
    const int max_treels = mem->max_treels;
    std::vector<int> h_treels_bvf;
    if (max_treels > 0)
        h_treels_bvf.resize((size_t)max_treels * kMaxVFaces);

    double cur_correlation = 0.0;
    int round = 0, total_done = 0;
    double best_logl_seen = -1e30;
    int no_improve_rounds = 0;

    printf("[GPU Bootstrap] K2-treels: B=%d K=%d pool=%d min_cor=%.4f\n",
           B, K, pool_size, params.min_correlation);
    fflush(stdout);

    for (;;) {
        auto t_round = std::chrono::high_resolution_clock::now();

        // ── Reset treels with current logl_cutoff → GPU filters bad trees ────────
        // logl_cutoff < 0 (= -parsimony); GPU cutoff = unsigned parsimony threshold.
        // If logl_cutoff == 0.0 (not yet activated), use ∞ (accept all).
        const unsigned int boot_cutoff = (iqtree.logl_cutoff != 0.0)
            ? (unsigned int)(-(double)iqtree.logl_cutoff)
            : 0xFFFFFFFFu;
        resetTreelsRound(mem, boot_cutoff);
        resetPoolRound(mem);

        // ── Run K2 from pool (k1_count=0 skips K1) ───────────────────────────
        gpuStepwiseBuildTrees(
            mem, nullptr, /*k1_count=*/0, sprDist, numNNI,
            pool_size, params.gpu_pool_stop, stream,
            nullptr, nullptr, K, /*max_outer_iters=*/1
        );

        // ── Download treels → REPS eval ───────────────────────────────────────
        int h_filled = 0;
        CUDA_CHECK(cudaMemcpy(&h_filled, mem->d_treelsFilled, sizeof(int), cudaMemcpyDeviceToHost));
        const int n_treels = std::min(h_filled, max_treels);

        if (n_treels > 0 && max_treels > 0) {
            CUDA_CHECK(cudaMemcpy(h_treels_bvf.data(), mem->d_treelsBackVf,
                       (size_t)n_treels * kMaxVFaces * sizeof(int), cudaMemcpyDeviceToHost));

            GpuTopology h_topo = h_tpl;
            for (int t = 0; t < n_treels; t++) {
                memcpy(h_topo.back_vf, h_treels_bvf.data() + (size_t)t * kMaxVFaces,
                       h_tpl.num_vfaces * sizeof(int));

                gpuTopoToCpu(&h_topo, iqtree.pllInst);
                pllTreeToNewick(
                    iqtree.pllInst->tree_string, iqtree.pllInst, iqtree.pllPartitions,
                    iqtree.pllInst->start->back, PLL_TRUE, PLL_TRUE, PLL_FALSE, PLL_FALSE,
                    PLL_FALSE, PLL_SUMMARIZE_LH, PLL_FALSE, PLL_FALSE
                );
                std::string newick(iqtree.pllInst->tree_string);
                if (newick.empty()) continue;

                iqtree.readTreeString(newick);
                iqtree.initializeAllPartialPars();
                iqtree.clearAllPartialLH();
                int pars = iqtree.computeParsimony();

                bool saved = iqtree.params->spr_parsimony;
                iqtree.params->spr_parsimony = false;
                iqtree.saveCurrentTree(-(double)pars);
                iqtree.params->spr_parsimony = saved;
            }
        }

        total_done += K;
        round++;

        // ── Track global best improvement across rounds ───────────────────────
        if (!iqtree.treels_logl.empty()) {
            double cur_best = *std::max_element(iqtree.treels_logl.begin(), iqtree.treels_logl.end());
            if (cur_best > best_logl_seen + 1e-6) {
                best_logl_seen = cur_best;
                no_improve_rounds = 0;
            } else {
                no_improve_rounds++;
            }
        }

        // ── Update logl_cutoff (mirrors CPU: top cutoff_percent%, activate at 1000 trees) ──
        if (iqtree.treels_logl.size() > 0) {
            DoubleVector logl = iqtree.treels_logl;
            nth_element(logl.begin(),
                        logl.begin() + logl.size() * params.cutoff_percent / 100,
                        logl.end(), std::greater<double>());
            iqtree.logl_cutoff = logl[logl.size() * params.cutoff_percent / 100];
        }

        // ── Convergence check (only after ≥1000 replicates, mirrors logl_cutoff threshold) ──
        if (round % step_iter_rounds == 0 && total_done >= 0) {
            SplitGraph* sg = new SplitGraph;
            iqtree.summarizeBootstrap(*sg);
            iqtree.boot_splits.push_back(sg);
            while (iqtree.boot_splits.size() > 2) {
                delete iqtree.boot_splits.front();
                iqtree.boot_splits.erase(iqtree.boot_splits.begin());
            }
            if (iqtree.boot_splits.size() >= 2)
                cur_correlation = iqtree.computeBootstrapCorrelation();
        }

        double round_sec = msSince(t_round) / 1e3;
        printf("[GPU Bootstrap] Round %-3d  done=%-5d  filled=%-5d  treels=%-5zu  no_impr=%d  cor=%.4f  t=%.2fs\n",
               round, total_done, h_filled, iqtree.treels_logl.size(), no_improve_rounds, cur_correlation, round_sec);
        fflush(stdout);

        if (no_improve_rounds >= 3 && cur_correlation >= params.min_correlation)
            break;
    }

    printf("[GPU Bootstrap] Done: %d rounds, %d replicates, cor=%.4f\n",
           round, total_done, cur_correlation);

    // ── Add GPU pool topologies to candidateTrees (mirrors step [7] of mpbootGpu) ──
    // After bootstrap K2 rounds, pool holds near-optimal topologies. Register them
    // as candidate trees so the final best-score report uses GPU-found trees.
    {
        pllInstance* tr = iqtree.pllInst;
        partitionList* pr = iqtree.pllPartitions;
        const int ps = mem->pool_size;

        std::vector<unsigned int> h_ps(ps, 0xFFFFFFFFu);
        std::vector<int>          h_pvf((size_t)ps * kMaxVFaces);
        CUDA_CHECK(cudaMemcpy(h_ps.data(),  mem->d_poolScores,
            ps * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_pvf.data(), mem->d_poolBackVf,
            (size_t)ps * kMaxVFaces * sizeof(int), cudaMemcpyDeviceToHost));

        // Find best pool score to filter slots
        unsigned int best_pool = 0xFFFFFFFFu;
        for (int i = 0; i < ps; i++)
            if (h_ps[i] < best_pool) best_pool = h_ps[i];

        GpuTopology tmpl;
        downloadTopology(mem, 0, &tmpl, stream);
        cudaStreamSynchronize(stream);
        tmpl.start_vface = 0;
        memset(tmpl.xpars, 0, sizeof(tmpl.xpars));

        int n_pool_added = 0;
        for (int i = 0; i < ps; i++) {
            if (h_ps[i] == 0xFFFFFFFFu) continue;

            GpuTopology h_topo = tmpl;
            memcpy(h_topo.back_vf, h_pvf.data() + (size_t)i * kMaxVFaces,
                   tmpl.num_vfaces * sizeof(int));
            h_topo.bestParsimony = h_ps[i];

            gpuTopoToCpu(&h_topo, tr);
            pllTreeToNewick(tr->tree_string, tr, pr, tr->start->back,
                PLL_TRUE, PLL_TRUE, PLL_FALSE, PLL_FALSE, PLL_FALSE,
                PLL_SUMMARIZE_LH, PLL_FALSE, PLL_FALSE);
            std::string tree_str(tr->tree_string);
            if (tree_str.empty()) continue;

            iqtree.readTreeString(tree_str);
            iqtree.initializeAllPartialPars();
            iqtree.clearAllPartialLH();
            iqtree.curScore = -(double)iqtree.computeParsimony();
            bool isNew = iqtree.candidateTrees.update(tree_str, iqtree.curScore);
            if (isNew && iqtree.curScore > iqtree.bestScore)
                iqtree.setBestTree(tree_str, iqtree.curScore);
            n_pool_added++;
        }
        printf("[GPU Bootstrap] Pool → candidateTrees: %d/%d slots added, best_pool=%u\n",
               n_pool_added, ps, best_pool);
    }
}

}  // namespace mpbootgpu
