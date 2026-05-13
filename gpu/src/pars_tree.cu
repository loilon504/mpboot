#include <cassert>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu/include/pars_tree.cuh"
#include "gpu/include/utils.cuh"

namespace mpbootgpu
{

// ─── Helpers ──────────────────────────────────────────────────────────────────

static inline int vface_of(
    const pllInstance* tr, const nodeptr p
)
{
    if (!p)
    {
        return -1;
    }
    return (int)(p - tr->nodeBaseAddress);
}

// ─── cpuToGpuTopology ─────────────────────────────────────────────────────────
void cpuToGpuTopology(
    const pllInstance* tr, GpuTopology* out
)
{
    const int N = tr->mxtips;
    const int num_vf = N + 3 * (N - 1);  // = 4N-3

    assert(num_vf <= kMaxVFaces);

    out->mxtips = N;
    out->ntips = tr->ntips;
    out->nextnode = tr->nextnode;
    out->bestParsimony = tr->bestParsimony;
    out->start_vface = vface_of(tr, tr->start);
    out->insert_vface = vface_of(tr, tr->insertNode);
    out->num_vfaces = num_vf;

    const nodeptr base = tr->nodeBaseAddress;
    for (int vf = 0; vf < num_vf; ++vf)
    {
        const nodeptr p = base + vf;
        out->back_vf[vf]      = vface_of(tr, p->back);
        out->best_back_vf[vf] = out->back_vf[vf];  // initialize best = current
        out->xpars[vf] = p->xPars;
        // next_vf, nnxt_vf, number not stored — computed from arithmetic on download
    }
}

// ─── gpuTopoToCpu ─────────────────────────────────────────────────────────────
void gpuTopoToCpu(
    const GpuTopology* in, pllInstance* tr
)
{
    nodeptr base = tr->nodeBaseAddress;
    const int num_vf = in->num_vfaces;

    for (int vf = 0; vf < num_vf; ++vf)
    {
        nodeptr p = base + vf;
        p->back  = (in->back_vf[vf] >= 0) ? (base + in->back_vf[vf]) : nullptr;
        p->next  = base + vfNextFace(vf, in->mxtips);
        p->xPars = (char)in->xpars[vf];
        // p->number and p->next->next are stable after PLL init — not stored on GPU
    }

    tr->ntips = in->ntips;
    tr->nextnode = in->nextnode;
    tr->bestParsimony = in->bestParsimony;
    tr->start = (in->start_vface >= 0) ? (base + in->start_vface) : nullptr;
    tr->insertNode = (in->insert_vface >= 0) ? (base + in->insert_vface) : nullptr;
}

// ─── gpuParsimonyMemAlloc ─────────────────────────────────────────────────────
GpuParsimonyMem* gpuParsimonyMemAlloc(
    int K, int mxtips, int width, int states
)
{
    auto* mem = new GpuParsimonyMem();
    mem->K = K;
    mem->mxtips = mxtips;
    mem->width = width;
    mem->states = states;
    mem->nodesPerTree        = (size_t)(2 * mxtips + 1);
    mem->parsVectPerTree     = mem->nodesPerTree * (size_t)width * (size_t)states;
    mem->parsScorePerTree    = mem->nodesPerTree;
    mem->siteWeightsPerTree  = (size_t)width;

    const size_t parsVectBytes      = (size_t)K * mem->parsVectPerTree  * sizeof(parsimonyNumber);
    const size_t parsScoreBytes     = (size_t)K * mem->parsScorePerTree  * sizeof(unsigned int);
    const size_t topoBytes          = (size_t)K * sizeof(GpuTopology);
    const size_t siteWeightsBytes   = (size_t)K * mem->siteWeightsPerTree * sizeof(unsigned int);

    printf("[GPU] Allocating parsimony memory for %d trees:\n", K);
    printf("  parsVect  : %.2f MB\n", parsVectBytes / 1048576.0);
    printf("  parsScore : %.2f MB\n", parsScoreBytes / 1048576.0);
    printf("  topologies: %.2f MB\n", topoBytes / 1048576.0);

    CUDA_CHECK(cudaMalloc(&mem->d_parsVect,    parsVectBytes));
    CUDA_CHECK(cudaMalloc(&mem->d_parsScore,   parsScoreBytes));
    CUDA_CHECK(cudaMalloc(&mem->d_topos,       topoBytes));
    CUDA_CHECK(cudaMalloc(&mem->d_siteWeights, siteWeightsBytes));
    CUDA_CHECK(cudaMalloc(&mem->d_globalBest,    sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&mem->d_postSprScores, (size_t)K * sizeof(unsigned int)));

    CUDA_CHECK(cudaMemset(mem->d_parsVect,    0, parsVectBytes));
    CUDA_CHECK(cudaMemset(mem->d_parsScore,   0, parsScoreBytes));
    // Init all site weights to 1 (normal, unweighted mode)
    {
        unsigned int* h_sw = new unsigned int[(size_t)K * width];
        for (size_t i = 0; i < (size_t)K * width; ++i) h_sw[i] = 1u;
        CUDA_CHECK(cudaMemcpy(mem->d_siteWeights, h_sw, siteWeightsBytes, cudaMemcpyHostToDevice));
        delete[] h_sw;
    }

    return mem;
}

void gpuParsimonyMemFree(
    GpuParsimonyMem* mem
)
{
    if (!mem)
    {
        return;
    }
    if (mem->d_parsVect)
    {
        cudaFree(mem->d_parsVect);
    }
    if (mem->d_parsScore)
    {
        cudaFree(mem->d_parsScore);
    }
    if (mem->d_topos)
    {
        cudaFree(mem->d_topos);
    }
    if (mem->d_siteWeights)
    {
        cudaFree(mem->d_siteWeights);
    }
    if (mem->d_globalBest)
    {
        cudaFree(mem->d_globalBest);
    }
    if (mem->d_postSprScores)
    {
        cudaFree(mem->d_postSprScores);
    }
    delete mem;
}

// ─── uploadTipParsVect ────────────────────────────────────────────────────────
// CPU layout per partition, per node:  [state][block]  stride = width
// GPU layout per node:                 [block][state]  stride = states
// We reorder for ALL K trees (tips are identical across trees).
void uploadTipParsVect(
    GpuParsimonyMem* mem, const pllInstance* tr, const partitionList* pr, cudaStream_t stream
)
{
    const int N = mem->mxtips;
    const int width = mem->width;
    const int states = mem->states;
    const size_t parsVT = mem->parsVectPerTree;  // elements per tree

    // Build one reordered host buffer for a single tree, then broadcast to all K.
    std::vector<parsimonyNumber> h_buf(parsVT, 0);

    // Only partition 0 for now (single-partition case).
    // For multi-partition, loop over pr->numberOfPartitions and handle offsets.
    const parsimonyNumber* cpu_pars = pr->partitionData[0]->parsVect;

    // Tips have node numbers 1..N; inner nodes 0 and N+1..2N-1 stay zero until GPU computes.
    for (int tipNum = 1; tipNum <= N; ++tipNum)
    {
        for (int b = 0; b < width; ++b)
        {
            for (int s = 0; s < states; ++s)
            {
                // CPU: parsVect[width * states * tipNum + width * s + b]
                parsimonyNumber
                    v = cpu_pars[(size_t)width * states * tipNum + (size_t)width * s + b];
                // GPU: parsVect[tipNum * width * states + b * states + s]
                h_buf[(size_t)tipNum * width * states + (size_t)b * states + s] = v;
            }
        }
    }

    // Upload the same tip data into every tree slot.
    for (int k = 0; k < mem->K; ++k)
    {
        parsimonyNumber* dst = mem->d_parsVect + (size_t)k * parsVT;
        CUDA_CHECK(cudaMemcpyAsync(
            dst, h_buf.data(), parsVT * sizeof(parsimonyNumber), cudaMemcpyHostToDevice, stream
        ));
    }
}

// ─── Topology upload / download ───────────────────────────────────────────────
void uploadTopology(
    GpuParsimonyMem* mem, int k, const GpuTopology* h_topo, cudaStream_t stream
)
{
    GpuTopology* dst = mem->d_topos + k;
    CUDA_CHECK(cudaMemcpyAsync(dst, h_topo, sizeof(GpuTopology), cudaMemcpyHostToDevice, stream));
}

void downloadTopology(
    const GpuParsimonyMem* mem, int k, GpuTopology* h_out, cudaStream_t stream
)
{
    const GpuTopology* src = mem->d_topos + k;
    CUDA_CHECK(cudaMemcpyAsync(h_out, src, sizeof(GpuTopology), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void downloadParsScore(
    const GpuParsimonyMem* mem, int k, unsigned int* h_out, cudaStream_t stream
)
{
    const unsigned int* src = mem->d_parsScore + (size_t)k * mem->parsScorePerTree;
    const size_t count = mem->parsScorePerTree;
    CUDA_CHECK(
        cudaMemcpyAsync(h_out, src, count * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream)
    );
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace mpbootgpu
