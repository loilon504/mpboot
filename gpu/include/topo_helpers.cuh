#pragma once
// Topology helper device functions shared by build and SPR kernels.
// All functions are __device__ __forceinline__ and header-only.
#include <cuda_runtime.h>

namespace mpbootgpu
{

// ─── RNG: exact replication of PLL's randum() ────────────────────────────────
__device__ __forceinline__ double gpuRandum(
    long* seed
)
{
    long s0 = *seed & 4095;
    long s1 = (*seed >> 12) & 4095;
    long s2 = (*seed >> 24) & 255;
    long m0 = 1549, m1 = 406;

    long sum = m0 * s0;
    long ns0 = sum & 4095;
    sum = (sum >> 12) + m0 * s1 + m1 * s0;
    long ns1 = sum & 4095;
    sum = (sum >> 12) + m0 * s2 + m1 * s1;
    long ns2 = sum & 255;

    *seed = (ns2 << 24) | (ns1 << 12) | ns0;
    return 0.00390625 * (ns2 + 0.000244140625 * (ns1 + 0.000244140625 * ns0));
}

// ─── Topology index helpers ───────────────────────────────────────────────────
// Tips  : vf = num-1  → num = vf+1          (vf < N)
// Inner : vf = N + 3*(num-N-1) + face_idx   (vf >= N)
__device__ __forceinline__ int vfToNum(
    int vf, int N
)
{
    return (vf < N) ? (vf + 1) : (N + 1 + (vf - N) / 3);
}

// vface of tr->nodep[num]: face[2] for inner nodes, vf=num-1 for tips.
__host__ __device__ __forceinline__ int nodepVf(
    int num, int N
)
{
    if (num <= N)
    {
        return num - 1;
    }
    return N + 3 * (num - N - 1) + 2;
}

// Ring layout: face[2]→face[1]→face[0]→face[2]
__host__ __device__ __forceinline__ int vfNextFace(
    int vf, int N
)
{
    if (vf < N)
    {
        return vf;  // tip self-loop
    }
    int base = N + 3 * ((vf - N) / 3);
    int f = (vf - N) % 3;
    int nf = (f == 2) ? 1 : (f == 1) ? 0 : 2;
    return base + nf;
}

__host__ __device__ __forceinline__ int vfNnxtFace(
    int vf, int N
)
{
    if (vf < N)
    {
        return vf;
    }
    int base = N + 3 * ((vf - N) / 3);
    int f = (vf - N) % 3;
    int nf = (f == 2) ? 0 : (f == 1) ? 2 : 1;
    return base + nf;
}

// hookup: back_vf[a] = b, back_vf[b] = a  (lane 0 only)
__device__ __forceinline__ void gpuHookup(
    int* back_vf, int a, int b
)
{
    back_vf[a] = b;
    back_vf[b] = a;
}

}  // namespace mpbootgpu
