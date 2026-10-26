// ============================================================================
//  GQA_bwd_vj1d_sm90.cu -- KV-DETERMINISTIC GQA backward for Hopper (sm_90a).
//
//  Kernel "Vj1d": per-query-head, wgmma, 2-CTA/SM. Each CTA writes a one-writer
//  bf16 PARTIAL dK/dV to [B,Hq,S,D]; gqa_dkdv_greduce then sums the G query
//  heads of each KV group in FIXED ORDER with no atomics, so dK/dV are
//  BIT-IDENTICAL run-to-run, matching cuDNN's one-writer reproducibility.
//  dQ is still reduced across K-tiles and remains non-deterministic (~3e-5),
//  exactly as cuDNN's dQ is. For bit-identical dQ use the v44 path with
//  USE_DETERMINISTIC_DQ=1 instead (GQA_bwd_sm90.cu).
//
//  The default Hopper backward until 2026-10-15; since then the default is Vj2
//  (GQA_bwd_vj2_sm90.cu) and USE_VJ2=0 selects this one. USE_FLASH=1 selects
//  the faster non-deterministic Vj1 (GQA_bwd_vj1_sm90.cu) instead.
//
//  Adapted from the standalone deliverable GQA_bwd_baseline.cu: the driver
//  main(), the PyTorch-reference loader and the benchmark harness were dropped,
//  and the scratch buffers now come from the caching allocator rather than
//  raw cudaMalloc. The kernel body itself is unchanged.
//
//  Device code is guarded by __CUDA_ARCH__ >= 900 && < 1000 so the file still
//  compiles for non-Hopper targets (the host launcher is always emitted).
// ============================================================================
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "device/CachingCudaAllocator.h"

namespace OwnTensor {
namespace cuda {
namespace gqa_bwd_vj1d {

using bf16 = __nv_bfloat16;

#define GQA_BWD_VJ1D_CUDA_CHECK(x) do {                                       \
    cudaError_t _e = (x);                                                     \
    if (_e != cudaSuccess) {                                                  \
        fprintf(stderr, "GQA_bwd_vj1d_sm90 %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(_e));                                      \
        abort();                                                              \
    }                                                                         \
} while (0)

#if __CUDA_ARCH__ >= 900 && __CUDA_ARCH__ < 1000   // Hopper only

__device__ __forceinline__ void wgmma_m64n64k16(float d[32], uint64_t descA, uint64_t descB) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "
        "%32, %33, 1, 1, 1, 0, 0;\n"
        : "+f"(d[0]),  "+f"(d[1]),  "+f"(d[2]),  "+f"(d[3]),
          "+f"(d[4]),  "+f"(d[5]),  "+f"(d[6]),  "+f"(d[7]),
          "+f"(d[8]),  "+f"(d[9]),  "+f"(d[10]), "+f"(d[11]),
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
          "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
          "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        : "l"(descA), "l"(descB));
}
__device__ __forceinline__ void wgmma_fence()  { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_wait0()  { asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory"); }

__device__ __forceinline__ void wgmma_wait1()  { asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory"); }

template<int N>
__device__ __forceinline__ void fence_operandN(float *d) {
#pragma unroll
    for (int i = 0; i < N; i++) asm volatile("" : "+f"(d[i]) :: "memory");
}

__device__ __forceinline__ void fence_proxy_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

template<int N>
__device__ __forceinline__ void zeroN(float *d) {
#pragma unroll
    for (int i = 0; i < N; i++) d[i] = 0.0f;
}

__device__ __forceinline__ void mbar_init_v4(uint64_t* mbar, uint32_t count) {
    uint32_t p = (uint32_t)__cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(p), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx_v4(uint64_t* mbar, uint32_t tx_bytes) {
    uint32_t p = (uint32_t)__cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;\n" :: "r"(p), "r"(tx_bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait_v4(uint64_t* mbar, uint32_t parity) {
    uint32_t p = (uint32_t)__cvta_generic_to_shared(mbar);
    asm volatile(
        "{\n .reg .pred _P;\n"
        "_MBAR_LOOP_V4:\n"
        "mbarrier.try_wait.parity.shared.b64 _P, [%0], %1;\n"
        "@!_P bra _MBAR_LOOP_V4;\n }\n"
        :: "r"(p), "r"(parity) : "memory");
}

__device__ __forceinline__ void tma_load_2d_v4(
    const void* tma_desc, void* smem_dst, uint64_t* mbar, uint32_t cx, uint32_t cy) {
    uint32_t dst = (uint32_t)__cvta_generic_to_shared(smem_dst);
    uint32_t mb  = (uint32_t)__cvta_generic_to_shared(mbar);
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n"
        :: "r"(dst), "l"((uint64_t)tma_desc), "r"(cx), "r"(cy), "r"(mb) : "memory");
}

__device__ __forceinline__ void tma_store_2d_v34(const void* tma_desc, const bf16* smem, uint32_t cx, uint32_t cy) {
    uint32_t src = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];\n"
        :: "l"((uint64_t)tma_desc), "r"(cx), "r"(cy), "r"(src) : "memory");
}
__device__ __forceinline__ void tma_store_commit_v34() { asm volatile("cp.async.bulk.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void tma_store_wait_v34()   { asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory"); }

__device__ __forceinline__ void tma_bulk_wait0_v43() { asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory"); }

__device__ __forceinline__ uint64_t make_desc_sw128_K(const bf16* smem_ptr) {
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem_ptr);
    uint64_t d = 0;
    d |= (uint64_t)((addr >> 4) & 0x3FFFu);   // start_address       [13:0]
    d |= (uint64_t)1u        << 16;           // leading_byte_offset [29:16] (unused; =1 per CUTLASS)
    d |= (uint64_t)(1024>>4) << 32;           // stride_byte_offset  [45:32] = 1024 B (SBO)
    d |= (uint64_t)1u        << 62;           // layout_type = B128   [63:62]
    return d;
}

__global__ void convert_dq_accum_to_bf16_v6(const float4 * __restrict__ in, uint2 * __restrict__ out, long n4) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {
        float4 v = in[i];
        bf16 o[4] = { __float2bfloat16(v.x), __float2bfloat16(v.y),
                      __float2bfloat16(v.z), __float2bfloat16(v.w) };
        out[i] = *reinterpret_cast<const uint2*>(o);
    }
}

constexpr float LOG2E_V29 = 1.4426950408889634f;   // V29: fold scale·log2e for exp2f softmax
__device__ __forceinline__ float ex2_approx_v29(float x) {   // V29: raw SFU 2^x (the forward's exp path)
    float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y;
}   // sS / sdP float row stride (Bc + 8)

__device__ __forceinline__ void wgmma_m64n64k16_tB(float d[32], uint64_t descA, uint64_t descB) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "
        "%32, %33, 1, 1, 1, 0, 1;\n"   // scaleD, scaleA, scaleB, transA=0, transB=1
        : "+f"(d[0]),  "+f"(d[1]),  "+f"(d[2]),  "+f"(d[3]),
          "+f"(d[4]),  "+f"(d[5]),  "+f"(d[6]),  "+f"(d[7]),
          "+f"(d[8]),  "+f"(d[9]),  "+f"(d[10]), "+f"(d[11]),
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
          "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
          "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        : "l"(descA), "l"(descB));
}

__device__ __forceinline__ void wgmma_m64n64k16_tAtB(float d[32], uint64_t descA, uint64_t descB) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "
        "%32, %33, 1, 1, 1, 1, 1;\n"   // scaleD, scaleA, scaleB, transA=1, transB=1
        : "+f"(d[0]),  "+f"(d[1]),  "+f"(d[2]),  "+f"(d[3]),
          "+f"(d[4]),  "+f"(d[5]),  "+f"(d[6]),  "+f"(d[7]),
          "+f"(d[8]),  "+f"(d[9]),  "+f"(d[10]), "+f"(d[11]),
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
          "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
          "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        : "l"(descA), "l"(descB));
}

__device__ __forceinline__ uint64_t make_desc_sw128_MN(const bf16* smem_ptr) {
    return make_desc_sw128_K(smem_ptr);
}

__device__ __forceinline__ void run_gemm_n64_sw2_hoB_issue(float acc[32], const bf16* A_sw, uint64_t descB_base) {
    fence_proxy_async_shared();
    fence_operandN<32>(acc);
    wgmma_fence();
#pragma unroll
    for (int k = 0; k < 8; k++) {
        uint64_t dA = make_desc_sw128_K(A_sw + (k >> 2) * 4096 + (k & 3) * 16);
        uint64_t dB = descB_base + (uint64_t)((k >> 2) * 512 + (k & 3) * 2);
        wgmma_m64n64k16(acc, dA, dB);
    }
    wgmma_commit();
}

template<int NDATA, int ROWSTRIDE>
__device__ __forceinline__ void stage_acc_bf16_s(const float *d, bf16 *stage, int tid, float scl) {
    int w = tid >> 5, lane = tid & 31;
    int r0 = w * 16 + (lane >> 2), r1 = r0 + 8, cc = (lane & 3) * 2;
#pragma unroll
    for (int nt = 0; nt < NDATA / 8; nt++) {
        int c = nt * 8 + cc;
        stage[r0 * ROWSTRIDE + c + 0] = __float2bfloat16(d[nt * 4 + 0] * scl);
        stage[r0 * ROWSTRIDE + c + 1] = __float2bfloat16(d[nt * 4 + 1] * scl);
        stage[r1 * ROWSTRIDE + c + 0] = __float2bfloat16(d[nt * 4 + 2] * scl);
        stage[r1 * ROWSTRIDE + c + 1] = __float2bfloat16(d[nt * 4 + 3] * scl);
    }
}

__device__ __forceinline__ void run_gemm_dVdK_half_te_issue_hoA(float acc[32], uint64_t descA_base,
                                                                const bf16* B_sw_half) {
    fence_proxy_async_shared();
    fence_operandN<32>(acc);
    wgmma_fence();
#pragma unroll
    for (int k = 0; k < 4; k++) {
        uint64_t dA = descA_base + (uint64_t)(k * 128);
        uint64_t dB = make_desc_sw128_MN(B_sw_half + k * 1024);
        wgmma_m64n64k16_tAtB(acc, dA, dB);
    }
    wgmma_commit();
}

__device__ __forceinline__ void stmatrix_x4(uint32_t dst, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};\n"
                 :: "r"(dst), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t src, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(src) : "memory");
}

template<int Bc, bool MASK>
__device__ __forceinline__ void fused_p_stsm(
    const float acc[32], bf16* sP, const float* sLSE, int wtid,
    int q_row0, int k_row0, float scale2)
{
    const int w = wtid >> 5, lane = wtid & 31;
    const int r0 = w * 16 + (lane >> 2), r1 = r0 + 8, cc = (lane & 3) * 2;
    const float l0 = sLSE[r0] * LOG2E_V29, l1 = sLSE[r1] * LOG2E_V29;
    const int gr0 = q_row0 + r0, gr1 = q_row0 + r1;

    // 32 P-values → 16 bf162 regs. pr0[nt]={P(r0,nt*8+cc),P(r0,+1)}; pr1[nt]={P(r1,..),..}.
    uint32_t pr0[8], pr1[8];
#pragma unroll
    for (int nt = 0; nt < 8; nt++) {
        float e00 = ex2_approx_v29(__fmaf_rn(acc[nt*4+0], scale2, -l0));
        float e01 = ex2_approx_v29(__fmaf_rn(acc[nt*4+1], scale2, -l0));
        float e10 = ex2_approx_v29(__fmaf_rn(acc[nt*4+2], scale2, -l1));
        float e11 = ex2_approx_v29(__fmaf_rn(acc[nt*4+3], scale2, -l1));
        if (MASK) {                              // branchless: ex2 already computed → SELECT → FSEL
            const int gc0 = k_row0 + nt*8 + cc, gc1 = gc0 + 1;
            e00 = (gc0 > gr0) ? 0.f : e00;
            e01 = (gc1 > gr0) ? 0.f : e01;
            e10 = (gc0 > gr1) ? 0.f : e10;
            e11 = (gc1 > gr1) ? 0.f : e11;
        }
        *reinterpret_cast<__nv_bfloat162*>(&pr0[nt]) = __float22bfloat162_rn(make_float2(e00, e01));
        *reinterpret_cast<__nv_bfloat162*>(&pr1[nt]) = __float22bfloat162_rn(make_float2(e10, e11));
    }

    // 4 stmatrix.x4 calls. Addr: lane l → matrix (l>>3)=key-block-in-group, row (l&7)=query phase.
    const int ph  = lane & 7;                    // (query_row & 7)
    const int mm  = lane >> 3;                    // key-block within the 4-group (0..3)
    const int qr0 = 16 * w + ph;                  // query row, qb=2w   (calls #0,#1)
    const int qr1 = 16 * w + 8 + ph;              // query row, qb=2w+1 (calls #2,#3)
    const uint32_t d0 = (uint32_t)__cvta_generic_to_shared(&sP[qr0 * 64 + (((mm    ) ^ ph) << 3)]);
    const uint32_t d1 = (uint32_t)__cvta_generic_to_shared(&sP[qr0 * 64 + (((mm + 4) ^ ph) << 3)]);
    const uint32_t d2 = (uint32_t)__cvta_generic_to_shared(&sP[qr1 * 64 + (((mm    ) ^ ph) << 3)]);
    const uint32_t d3 = (uint32_t)__cvta_generic_to_shared(&sP[qr1 * 64 + (((mm + 4) ^ ph) << 3)]);
    stmatrix_x4(d0, pr0[0], pr0[1], pr0[2], pr0[3]);   // qb=2w,   cb 0..3
    stmatrix_x4(d1, pr0[4], pr0[5], pr0[6], pr0[7]);   // qb=2w,   cb 4..7
    stmatrix_x4(d2, pr1[0], pr1[1], pr1[2], pr1[3]);   // qb=2w+1, cb 0..3
    stmatrix_x4(d3, pr1[4], pr1[5], pr1[6], pr1[7]);   // qb=2w+1, cb 4..7
}

__global__ void compute_drowsum_v22(
    const bf16 * __restrict__ d_dO, const bf16 * __restrict__ d_O,
    float * __restrict__ d_Drow, long nRows)
{
    const int  warpsPerBlock = blockDim.x >> 5;
    const int  lane          = threadIdx.x & 31;
    const long warp0         = (long)blockIdx.x * warpsPerBlock + (threadIdx.x >> 5);
    const long stride        = (long)gridDim.x * warpsPerBlock;
    // Grid-stride over rows → EVERY row in [0, nRows) is covered. VECTORIZED: each lane reads 4 CONTIGUOUS
    // elems as a uint2 (8B) from dO and O -> one coalesced transaction each vs 4 strided. D within tolerance
    // (contiguous vs strided sum order; not strictly bit-identical to v20, but |Δ|~1e-6 << 2e-2 check tol).
    for (long row = warp0; row < nRows; row += stride) {
        const long base = row * 128 + lane * 4;
        const uint2 vdO = *reinterpret_cast<const uint2*>(d_dO + base);   // 4 bf16
        const uint2 vO  = *reinterpret_cast<const uint2*>(d_O  + base);
        const bf16* a = reinterpret_cast<const bf16*>(&vdO);
        const bf16* b = reinterpret_cast<const bf16*>(&vO);
        float partial = 0.f;
        #pragma unroll
        for (int k = 0; k < 4; k++) partial += __bfloat162float(a[k]) * __bfloat162float(b[k]);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            partial += __shfl_down_sync(0xFFFFFFFFu, partial, off);
        if (lane == 0) d_Drow[row] = partial;
    }
}

template<int Bc>
__device__ __forceinline__ void fuse_dS_ldstsm(
    const bf16* __restrict__ sP, const float dP[32], const float* __restrict__ sD,
    bf16* __restrict__ sDS, int wtid) {
    const int w = wtid >> 5, lane = wtid & 31;
    const int r0 = w * 16 + (lane >> 2), r1 = r0 + 8;
    const float D0 = sD[r0], D1 = sD[r1];
    const int ph = lane & 7, mm = lane >> 3;
    const int qr0 = 16 * w + ph, qr1 = 16 * w + 8 + ph;
    const uint32_t s0 = (uint32_t)__cvta_generic_to_shared(&sP[qr0 * 64 + (((mm    ) ^ ph) << 3)]);
    const uint32_t s1 = (uint32_t)__cvta_generic_to_shared(&sP[qr0 * 64 + (((mm + 4) ^ ph) << 3)]);
    const uint32_t s2 = (uint32_t)__cvta_generic_to_shared(&sP[qr1 * 64 + (((mm    ) ^ ph) << 3)]);
    const uint32_t s3 = (uint32_t)__cvta_generic_to_shared(&sP[qr1 * 64 + (((mm + 4) ^ ph) << 3)]);
    uint32_t pP0[8], pP1[8];
    ldmatrix_x4(s0, pP0[0], pP0[1], pP0[2], pP0[3]);
    ldmatrix_x4(s1, pP0[4], pP0[5], pP0[6], pP0[7]);
    ldmatrix_x4(s2, pP1[0], pP1[1], pP1[2], pP1[3]);
    ldmatrix_x4(s3, pP1[4], pP1[5], pP1[6], pP1[7]);
    uint32_t pr0[8], pr1[8];
#pragma unroll
    for (int nt = 0; nt < 8; nt++) {
        const float2 pf0 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&pP0[nt]));
        const float2 pf1 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&pP1[nt]));
        const float d00 = pf0.x * (dP[nt*4+0] - D0), d01 = pf0.y * (dP[nt*4+1] - D0);
        const float d10 = pf1.x * (dP[nt*4+2] - D1), d11 = pf1.y * (dP[nt*4+3] - D1);
        *reinterpret_cast<__nv_bfloat162*>(&pr0[nt]) = __float22bfloat162_rn(make_float2(d00, d01));
        *reinterpret_cast<__nv_bfloat162*>(&pr1[nt]) = __float22bfloat162_rn(make_float2(d10, d11));
    }
    const uint32_t d0 = (uint32_t)__cvta_generic_to_shared(&sDS[qr0 * 64 + (((mm    ) ^ ph) << 3)]);
    const uint32_t d1 = (uint32_t)__cvta_generic_to_shared(&sDS[qr0 * 64 + (((mm + 4) ^ ph) << 3)]);
    const uint32_t d2 = (uint32_t)__cvta_generic_to_shared(&sDS[qr1 * 64 + (((mm    ) ^ ph) << 3)]);
    const uint32_t d3 = (uint32_t)__cvta_generic_to_shared(&sDS[qr1 * 64 + (((mm + 4) ^ ph) << 3)]);
    stmatrix_x4(d0, pr0[0], pr0[1], pr0[2], pr0[3]);
    stmatrix_x4(d1, pr0[4], pr0[5], pr0[6], pr0[7]);
    stmatrix_x4(d2, pr1[0], pr1[1], pr1[2], pr1[3]);
    stmatrix_x4(d3, pr1[4], pr1[5], pr1[6], pr1[7]);
}

__device__ __forceinline__ void run_gemm_dKdQ_te_issue_ho(
    float dk[32], float dq[32], uint64_t descDSmn_base, uint64_t descDSk_base,
    uint64_t descKhalf_base, const bf16* sQ_half) {
    fence_proxy_async_shared();
    fence_operandN<32>(dk);  fence_operandN<32>(dq);
    wgmma_fence();
#pragma unroll
    for (int k = 0; k < 4; k++)
        wgmma_m64n64k16_tAtB(dk, descDSmn_base + (uint64_t)(k * 128),
                                 make_desc_sw128_MN(sQ_half + k * 1024));
#pragma unroll
    for (int k = 0; k < 4; k++)
        wgmma_m64n64k16_tB (dq, descDSk_base   + (uint64_t)(k * 2),
                                 descKhalf_base + (uint64_t)(k * 128));
    wgmma_commit();
}

__device__ __forceinline__ void run_gemm_dVdKdQ_te_wait(float dv[32], float dk[32], float dq[32]) {
    wgmma_wait0();
    fence_operandN<32>(dv);  fence_operandN<32>(dk);  fence_operandN<32>(dq);
}

__device__ __forceinline__ void tma_reduce_add_2d_v43(const void* tma_desc, const float* smem, uint32_t cx, uint32_t cy) {
    uint32_t src = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile(
        "cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group [%0, {%1, %2}], [%3];\n"
        :: "l"((uint64_t)tma_desc), "r"(cx), "r"(cy), "r"(src) : "memory");
}

__device__ __forceinline__ void store_acc_sw128_f32(const float* d, float* base, int tid, float scl) {
    int w = tid >> 5, lane = tid & 31;
    int r0 = w * 16 + (lane >> 2), r1 = r0 + 8, cc = (lane & 3) * 2;
    const int ph0 = r0 & 7, ph1 = r1 & 7;
#pragma unroll
    for (int nt = 0; nt < 8; nt++) {
        const int c   = nt * 8 + cc;               // 0..63 (even)
        const int atom = c >> 5, col32 = c & 31;   // 128B atom (0/1), col within atom
        const int chunk = col32 >> 2, lo = col32 & 3;
        const int abase = atom * (64 * 32);
        // c and c+1 share the same 4-fp32 chunk (c even) -> adjacent phys slots (lo, lo+1)
        base[abase + r0 * 32 + ((chunk ^ ph0) << 2) + lo    ] = d[nt * 4 + 0] * scl;
        base[abase + r0 * 32 + ((chunk ^ ph0) << 2) + lo + 1] = d[nt * 4 + 1] * scl;
        base[abase + r1 * 32 + ((chunk ^ ph1) << 2) + lo    ] = d[nt * 4 + 2] * scl;
        base[abase + r1 * 32 + ((chunk ^ ph1) << 2) + lo + 1] = d[nt * 4 + 3] * scl;
    }
}

__global__ void gqa_dkdv_greduce(const bf16* __restrict__ partial, bf16* __restrict__ out,
                                 int Hq, int Hkv, int G, int S) {
    constexpr int Dd = 128, RPB = 8;
    int b = blockIdx.z, hkv = blockIdx.y, s0 = blockIdx.x * RPB;
    int t = threadIdx.x, row = t >> 4, dg = t & 15;          // 128 thr = 8 rows x 16 uint4-groups
    int s = s0 + row;
    float acc[8] = {0,0,0,0,0,0,0,0};
    for (int g = 0; g < G; g++) {
        const bf16* p = partial + (((long)(b * Hq + hkv * G + g) * S + s) * Dd + dg * 8);
        uint4 v = *reinterpret_cast<const uint4*>(p);
        const bf16* vb = reinterpret_cast<const bf16*>(&v);
        #pragma unroll
        for (int k = 0; k < 8; k++) acc[k] += __bfloat162float(vb[k]);
    }
    bf16 o[8];
    #pragma unroll
    for (int k = 0; k < 8; k++) o[k] = __float2bfloat16(acc[k]);
    *reinterpret_cast<uint4*>(out + (((long)(b * Hkv + hkv) * S + s) * Dd + dg * 8)) = *reinterpret_cast<const uint4*>(o);
}

// v2 design B (one seg_start value per token, added 2026-10-10) was REMOVED on 2026-10-15: design A below is
// faster (backward 3-5% on middle and long chats, 2026-10-10/12 measurements) and is the cu_seqlens way of
// FlashAttention, TE/Megatron and TRL. The dense kernel is unchanged by both (SASS checked).
// v2 design A (per-segment work list, added 2026-10-10), MODE_CHATS: one CTA per (key tile of the
// plan, query head). tiles[blockIdx.x] = {b, t0, L, kt}: key tile kt of segment [t0, t0+L) of row b.
// Every index is segment-relative, so the loop and the diagonal mask are the dense ones with S = L.
// Rows past L (the tail of the segment's last q-tile): LSE = +inf and D = 0, so P = 0 and dS = 0 --
// they add nothing to dK/dV, and their dQ rows (the next segment's) get exact zeros from the
// reduce-add (FlashAttention 2 does the same, csrc/flash_attn/src/flash_bwd_kernel.h:411-413).
// Keys past L sit only in the segment's last key tile; their dK/dV rows belong to the next segment,
// so that tile stores its rows < L with plain stores instead of the 64-row TMA store (FA3 stores
// all varlen dK/dV that way, hopper/epilogue_bwd.hpp:31,263-268).
constexpr int MODE_DENSE = 0, MODE_CHATS = 1;

template<int Br, int Bc, int D, int MODE>
__global__ void __launch_bounds__(128, 1)
gqa_bwd_vj1d(
    const __grid_constant__ CUtensorMap tma_K,  const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_Q,  const __grid_constant__ CUtensorMap tma_dO,
    const __grid_constant__ CUtensorMap tma_dV_st, const __grid_constant__ CUtensorMap tma_dK_st,
    const __grid_constant__ CUtensorMap tma_dq_red,
    const float* __restrict__ LSE, const float* __restrict__ Drow,
    int B, int Hq, int Hkv, int G, int S, float scale,
    const int4* __restrict__ tiles, bf16* __restrict__ dVp, bf16* __restrict__ dKp) {   // A only
    constexpr bool CHATS  = (MODE == MODE_CHATS);
    int kt = blockIdx.x, b = blockIdx.z, t0 = 0, L = 0;
    if constexpr (CHATS) { const int4 tl = tiles[blockIdx.x]; b = tl.x; t0 = tl.y; L = tl.z; kt = tl.w; }
    const int hq = blockIdx.y;
    const int hkv = hq / G;
    const int wtid = threadIdx.x;
    int nQ = S / Br;
    if constexpr (CHATS) nQ = (L + Br - 1) / Br;   // q-tiles of the segment
    const int k_row0 = kt * Bc;                     // relative to the row (dense) or segment (A)
    const float scale2 = scale * LOG2E_V29;

    __shared__ __align__(128)  bf16  sK_sw[Bc*D], sV_sw[Bc*D], sQ_sw[Br*D], sdO_sw[Br*D];
    __shared__ __align__(1024) bf16  sP[Br*64], sDS[Br*64];
    __shared__ __align__(1024) float sStage[2][Br*64];   // double-buffered: both dQ halves deferred
    __shared__ float sLSE[Br], sD[Br];
    __shared__ __align__(8) uint64_t mbar_kv, mbar_qo;

    if (wtid == 0) { mbar_init_v4(&mbar_kv, 1); mbar_init_v4(&mbar_qo, 1); }
    __syncthreads();

    const uint64_t descK    = make_desc_sw128_K (sK_sw);
    const uint64_t descV    = make_desc_sw128_K (sV_sw);
    const uint64_t descP    = make_desc_sw128_MN(sP);
    const uint64_t descKh0  = make_desc_sw128_MN(sK_sw);
    const uint64_t descKh1  = make_desc_sw128_MN(sK_sw + 4096);
    const uint64_t descDSmn = make_desc_sw128_MN(sDS);
    const uint64_t descDSk  = make_desc_sw128_K (sDS);

    const uint32_t kvRow = (uint32_t)((b*Hkv+hkv)*S + t0 + k_row0);   // t0: A only (else 0)
    if (wtid == 0) {
        mbar_expect_tx_v4(&mbar_kv, (uint32_t)(Bc*64*sizeof(bf16))*4);
        tma_load_2d_v4(&tma_K, sK_sw,       &mbar_kv, 0,  kvRow);
        tma_load_2d_v4(&tma_K, sK_sw+64*64, &mbar_kv, 64, kvRow);
        tma_load_2d_v4(&tma_V, sV_sw,       &mbar_kv, 0,  kvRow);
        tma_load_2d_v4(&tma_V, sV_sw+64*64, &mbar_kv, 64, kvRow);
    }

    mbar_wait_v4(&mbar_kv, 0);

    float dv[64], dk[64];
    zeroN<64>(dv); zeroN<64>(dk);

    // SINGLE-BUFFER SOFTWARE PREFETCH: issue tile it+1's Q/dO TMA into the SAME buffer right after the
    // half1 dK-gemm frees sQ/sdO, overlapping the dQ-reduce (which touches only sStage). No ring, no
    // smem cost, 2 CTAs preserved. Attacks long_scoreboard 2.51 (loads were issued-then-immediately-waited).
    const int nq_ = nQ - kt, nIter = nq_;   // Vj1 per-hq: ONE query head per CTA
    auto issue = [&](int it) {
        const int q = kt + it;
        const uint32_t qRow = (uint32_t)((b*Hq+hq)*S + t0 + q*Br);
        const long lb = (long)(b*Hq+hq)*S + t0 + (long)q*Br;
        if (wtid == 0) {
            mbar_expect_tx_v4(&mbar_qo, (uint32_t)(Br*D*sizeof(bf16))*2);
            tma_load_2d_v4(&tma_Q,  sQ_sw,        &mbar_qo, 0,  qRow);
            tma_load_2d_v4(&tma_Q,  sQ_sw+64*64,  &mbar_qo, 64, qRow);
            tma_load_2d_v4(&tma_dO, sdO_sw,       &mbar_qo, 0,  qRow);
            tma_load_2d_v4(&tma_dO, sdO_sw+64*64, &mbar_qo, 64, qRow);
        }
        if constexpr (CHATS) {   // A: rows past L -> P = 0 and dS = 0 (see the kernel comment)
            if (wtid < Br) {
                const bool in_seg = q * Br + wtid < L;
                sLSE[wtid] = in_seg ? LSE[lb+wtid] : INFINITY;
                sD[wtid]   = in_seg ? Drow[lb+wtid] : 0.0f;
            }
        } else {
        if (wtid < Br) { sLSE[wtid] = LSE[lb+wtid]; sD[wtid] = Drow[lb+wtid]; }
        }
    };
    uint32_t qopar = 0;
    issue(0);                                          // prologue: load tile 0
    // Tile 0's sLSE/sD are written by threads 0..63 in issue(0) and read by all 4 warps in
    // fused_p_stsm before any other __syncthreads. mbar_qo does not order those writes: only thread 0 arrives
    // on it, and it arrives before its own sLSE store. Tiles >= 1 are published by the store barrier after
    // issue(it + 1) (the NOTE in the loop); tile 0 had nothing. Before 2026-10-10 there was no barrier here:
    // a race on the first tile, correct only because the 256-B LSE load lands before the 32-KB Q/dO TMA and
    // the S GEMM finish (SASS of the old kernel: sLSE STS at 0x1010, first LDS at 0x1ee0, no BAR between;
    // v44 orders the same data with producer_sync + the d_ready mbarrier, GQA_bwd_sm90.cu:594-596,622).
    // Cost: one barrier per CTA, not per tile. SASS change: +1 BAR.SYNC, nothing else in the arithmetic.
    __syncthreads();

    for (int it = 0; it < nIter; it++) {
        const int q = kt + it;
        const uint32_t qRow = (uint32_t)((b*Hq+hq)*S + t0 + q*Br);
        mbar_wait_v4(&mbar_qo, qopar); qopar ^= 1;
        // NOTE: no post-load barrier — sQ visibility is guaranteed by mbar_wait (TMA completion), and
        // sLSE/sD were written+published by the PREVIOUS tile's store-barrier (prefetch). One barrier saved.
        // (Tile 0 has no previous tile: it is published by the __syncthreads() right after issue(0).)

        // S = Q@Kᵀ and dP = dO@Vᵀ ISSUED TOGETHER (overlap on the tensor pipe), waited once — instead of
        // S(wait)→fused_p→dP(wait) which serialized the two GEMMs. Attacks wait 1.28 + 44%-idle pipe.
        float acc[32];   zeroN<32>(acc);
        float dPacc[32]; zeroN<32>(dPacc);
        run_gemm_n64_sw2_hoB_issue(acc,   sQ_sw,  descK);   // S (group A)
        run_gemm_n64_sw2_hoB_issue(dPacc, sdO_sw, descV);   // dP (group B)
        wgmma_wait1();                       // wait S only; dP GEMM keeps running on the tensor pipe
        fence_operandN<32>(acc);
        if (q == kt) fused_p_stsm<Bc,true >(acc, sP, sLSE, wtid, q*Br, k_row0, scale2);
        else         fused_p_stsm<Bc,false>(acc, sP, sLSE, wtid, 0, 0, scale2);  // softmax OVERLAPS dP GEMM
        wgmma_wait0();                       // now wait dP
        fence_operandN<32>(dPacc);
        __syncthreads();

        run_gemm_dVdK_half_te_issue_hoA(dv,    descP, sdO_sw + 0);
        run_gemm_dVdK_half_te_issue_hoA(dv+32, descP, sdO_sw + 4096);
        fuse_dS_ldstsm<Bc>(sP, dPacc, sD, sDS, wtid);
        __syncthreads();

        // dQ split: half0 & half1 kept SEPARATE — half0's committed reduce overlaps half1's GEMM, and the
        // prefetch overlaps half1's store+reduce. (Combining into one wait0 lost that overlap → slower.)
        {   float dq[32]; zeroN<32>(dq);
            run_gemm_dKdQ_te_issue_ho(dk, dq, descDSmn, descDSk, descKh0, sQ_sw + 0);
            if (wtid == 0) tma_bulk_wait0_v43();        // drain PREV reduces — moved BEFORE the wgmma wait
            run_gemm_dVdKdQ_te_wait(dv, dk, dq);        //   so thread 0's drain overlaps the dK/dQ GEMM
            __syncthreads();
            store_acc_sw128_f32(dq, sStage[0], wtid, scale);
            __syncthreads(); fence_proxy_async_shared();
            if (wtid == 0) {
                tma_reduce_add_2d_v43(&tma_dq_red, sStage[0],       0,  qRow);
                tma_reduce_add_2d_v43(&tma_dq_red, sStage[0]+64*32, 32, qRow);
                tma_store_commit_v34();   // NO wait — deferred (double-buffered sStage)
            }
        }
        {   float dq[32]; zeroN<32>(dq);
            run_gemm_dKdQ_te_issue_ho(dk+32, dq, descDSmn, descDSk, descKh1, sQ_sw + 4096);
            run_gemm_dVdKdQ_te_wait(dv, dk+32, dq);
            __syncthreads();                            // all threads done reading sQ/sdO
            if (it + 1 < nIter) issue(it + 1);          // prefetch next tile into same buffer (overlap)
            store_acc_sw128_f32(dq, sStage[1], wtid, scale);
            __syncthreads(); fence_proxy_async_shared();
            if (wtid == 0) {
                tma_reduce_add_2d_v43(&tma_dq_red, sStage[1],       64, qRow);
                tma_reduce_add_2d_v43(&tma_dq_red, sStage[1]+64*32, 96, qRow);
                tma_store_commit_v34();   // NO wait0 — deferred to next tile's half0
            }
        }
    }
    if (wtid == 0) tma_bulk_wait0_v43();   // drain the last tile's deferred half1 reduce before epilogue
    __syncthreads();
    // epilogue: stage + TMA-store dV, dK (full [64x128] each, 2 col-halves)
    fence_operandN<64>(dv);
    stage_acc_bf16_s<64,64>(dv,    sQ_sw + 0,    wtid, 1.0f);
    stage_acc_bf16_s<64,64>(dv+32, sQ_sw + 4096, wtid, 1.0f);
    fence_operandN<64>(dk);
    stage_acc_bf16_s<64,64>(dk,    sdO_sw + 0,    wtid, scale);
    stage_acc_bf16_s<64,64>(dk+32, sdO_sw + 4096, wtid, scale);
    __syncthreads(); fence_proxy_async_shared();
    // Vj1d: one-writer bf16 PARTIAL store to [B,Hq,S,D] at hqRow (each per-hq CTA owns its own slot),
    // then a fixed-order greduce sums the G heads DETERMINISTICALLY (no atomics) -> bit-identical dK/dV
    // run-to-run, matching cuDNN's one-writer reproducibility. Price = the ~69us greduce pass.
    const uint32_t hqRow = (uint32_t)((b*Hq+hq)*S + t0 + k_row0);
    bool full_tile = true;
    if constexpr (CHATS) full_tile = (k_row0 + Bc <= L);
    if (full_tile) {
    if (wtid == 0) {
        tma_store_2d_v34(&tma_dV_st, sQ_sw + 0,    0,  hqRow);
        tma_store_2d_v34(&tma_dV_st, sQ_sw + 4096, 64, hqRow);
        tma_store_2d_v34(&tma_dK_st, sdO_sw + 0,    0,  hqRow);
        tma_store_2d_v34(&tma_dK_st, sdO_sw + 4096, 64, hqRow);
        tma_store_commit_v34(); tma_store_wait_v34();
    }
    } else if constexpr (CHATS) {
        // A, the segment's last key tile: copy only its rows < L. The staging above is plain
        // row-major [64 rows][64 cols] per column half (stage_acc_bf16_s<64,64>), 8 bf16 per uint4.
        const int nrow = L - k_row0;
        for (int e = wtid; e < 4 * 64 * 8; e += 128) {   // 4 blocks (dV h0/h1, dK h0/h1) x 64 rows x 8
            const int blk = e >> 9, r = (e >> 3) & 63, c8 = e & 7;
            if (r >= nrow) continue;
            const bf16* src = (blk < 2 ? sQ_sw : sdO_sw) + (blk & 1) * 4096 + r * 64 + c8 * 8;
            bf16* dst = (blk < 2 ? dVp : dKp) + ((size_t)hqRow + r) * D + (blk & 1) * 64 + c8 * 8;
            *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
        }
    }
}

#else   // ---- non-Hopper: keep the translation unit compilable ----
// Stubs so the host pass still sees the kernels it launches. Parameter types
// must match the Hopper definitions exactly (including __restrict__), or the
// stub fails to match the template it is meant to instantiate.
constexpr int MODE_DENSE = 0, MODE_CHATS = 1;
template<int Br, int Bc, int D, int MODE>
__global__ void gqa_bwd_vj1d(
    const __grid_constant__ CUtensorMap tma_K,  const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_Q,  const __grid_constant__ CUtensorMap tma_dO,
    const __grid_constant__ CUtensorMap tma_dV_st, const __grid_constant__ CUtensorMap tma_dK_st,
    const __grid_constant__ CUtensorMap tma_dq_red,
    const float* __restrict__ LSE, const float* __restrict__ Drow,
    int B, int Hq, int Hkv, int G, int S, float scale,
    const int4* __restrict__ tiles, bf16* __restrict__ dVp, bf16* __restrict__ dKp) {}
__global__ void gqa_dkdv_greduce(const bf16* __restrict__ partial, bf16* __restrict__ out,
                                 int Hq, int Hkv, int G, int S) {}
__global__ void compute_drowsum_v22(
    const bf16 * __restrict__ d_dO, const bf16 * __restrict__ d_O,
    float * __restrict__ d_Drow, long nRows) {}
__global__ void convert_dq_accum_to_bf16_v6(
    const float4 * __restrict__ in, uint2 * __restrict__ out, long n4) {}
#endif  // __CUDA_ARCH__

template<int Br, int Bc, int D>
void launch_gqa_backward_vj1d(
    const bf16 *d_Q, const bf16 *d_K, const bf16 *d_V, const bf16 *d_O,
    const bf16 *d_dO, const float *d_LSE,
    bf16 *d_dQ, bf16 *d_dK, bf16 *d_dV,
    int B, int Hq, int Hkv, int G, int S, float scale,
    const int32_t *d_chat_tiles, int n_chat_tiles) {
    auto make_tma_sw128 = [&](const bf16* ptr, uint64_t total_rows, uint32_t tile_rows) {
        CUtensorMap desc{}; uint64_t gS[2]={(uint64_t)D,total_rows}; uint64_t gT[1]={(uint64_t)D*sizeof(bf16)};
        uint32_t bx[2]={64u,tile_rows}, eS[2]={1,1};
        CUresult r=cuTensorMapEncodeTiled(&desc,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,(void*)ptr,gS,gT,bx,eS,
            CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if(r!=CUDA_SUCCESS){const char*e;cuGetErrorString(r,&e);fprintf(stderr,"sw128 vz2: %s\n",e);exit(1);} return desc; };
    auto make_tma_out = [&](const bf16* ptr, uint64_t rows) {
        CUtensorMap desc{}; uint64_t gS[2]={(uint64_t)D,rows}; uint64_t gT[1]={(uint64_t)D*sizeof(bf16)};
        uint32_t bx[2]={64u,64u}, eS[2]={1,1};
        CUresult r=cuTensorMapEncodeTiled(&desc,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,(void*)ptr,gS,gT,bx,eS,
            CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_NONE,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if(r!=CUDA_SUCCESS){const char*e;cuGetErrorString(r,&e);fprintf(stderr,"out vz2: %s\n",e);exit(1);} return desc; };
    auto make_tma_red = [&](const float* ptr, uint64_t rows) {
        CUtensorMap desc{}; uint64_t gS[2]={(uint64_t)D,rows}; uint64_t gT[1]={(uint64_t)D*sizeof(float)};
        uint32_t bx[2]={32u,64u}, eS[2]={1,1};
        CUresult r=cuTensorMapEncodeTiled(&desc,CU_TENSOR_MAP_DATA_TYPE_FLOAT32,2,(void*)ptr,gS,gT,bx,eS,
            CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
        if(r!=CUDA_SUCCESS){const char*e;cuGetErrorString(r,&e);fprintf(stderr,"red vz2: %s\n",e);exit(1);} return desc; };

    const uint64_t Rq=(uint64_t)B*Hq*S, Rkv=(uint64_t)B*Hkv*S;
    CUtensorMap tK=make_tma_sw128(d_K,Rkv,Bc), tV=make_tma_sw128(d_V,Rkv,Bc);
    CUtensorMap tQ=make_tma_sw128(d_Q,Rq,Br),  tdO=make_tma_sw128(d_dO,Rq,Br);
    // Vj1d: one-writer bf16 PARTIAL dK/dV [B,Hq,S,D]; a fixed-order greduce (no atomics) sums the G heads
    // DETERMINISTICALLY -> bit-identical dK/dV. Trade vs Vj1: the ~69us greduce pass instead of the atomic reduce.
    // Scratch comes from the caching allocator, not raw cudaMalloc: these are the
    // largest buffers here (2 x [B,Hq,S,D] bf16 partials) and the pool reuses them
    // across steps instead of holding a private static allocation for the process.
    const long partN=(long)B*Hq*S*D; static bf16 *d_dKp=nullptr,*d_dVp=nullptr; static long pcap=0;
    if(partN>pcap){
        auto& alloc = CachingCUDAAllocator::instance();
        if(d_dKp) alloc.deallocate(d_dKp);
        if(d_dVp) alloc.deallocate(d_dVp);
        d_dKp=static_cast<bf16*>(alloc.allocate(partN*sizeof(bf16)));
        d_dVp=static_cast<bf16*>(alloc.allocate(partN*sizeof(bf16)));
        pcap=partN; }
    CUtensorMap tdV=make_tma_out(d_dVp,Rq), tdK=make_tma_out(d_dKp,Rq);

    const long drowN=(long)B*Hq*S; static float* d_Drow=nullptr; static long drc=0;
    if(drowN>drc){
        auto& alloc = CachingCUDAAllocator::instance();
        if(d_Drow) alloc.deallocate(d_Drow);
        d_Drow=static_cast<float*>(alloc.allocate(drowN*sizeof(float))); drc=drowN; }
    const long dqN=(long)B*Hq*S*D; static float* d_dqa=nullptr; static long dqc=0;
    if(dqN>dqc){
        auto& alloc = CachingCUDAAllocator::instance();
        if(d_dqa) alloc.deallocate(d_dqa);
        d_dqa=static_cast<float*>(alloc.allocate(dqN*sizeof(float))); dqc=dqN; }
    GQA_BWD_VJ1D_CUDA_CHECK(cudaMemset(d_dqa,0,dqN*sizeof(float)));
    CUtensorMap tRed=make_tma_red(d_dqa,(uint64_t)B*Hq*S);

    const int dBlock=256; const long dGrid=(drowN+(dBlock/32)-1)/(dBlock/32);
    compute_drowsum_v22<<<(unsigned)dGrid,dBlock>>>(d_dO,d_O,d_Drow,drowN);

    dim3 GRID(S/Bc, Hq, B);   // Vj1 per-hq: 3x the CTAs (Hq not Hkv) to fill the GPU at low B
    // d_chat_tiles != nullptr -> design A: one CTA per (plan key tile, query head).
    if (d_chat_tiles) {
        dim3 GRID_A(n_chat_tiles, Hq, 1);
        gqa_bwd_vj1d<Br,Bc,D,MODE_CHATS><<<GRID_A,128>>>(tK,tV,tQ,tdO,tdV,tdK,tRed,d_LSE,d_Drow,B,Hq,Hkv,G,S,scale,
            reinterpret_cast<const int4*>(d_chat_tiles), d_dVp, d_dKp);
    } else
        gqa_bwd_vj1d<Br,Bc,D,MODE_DENSE><<<GRID,128>>>(tK,tV,tQ,tdO,tdV,tdK,tRed,d_LSE,d_Drow,B,Hq,Hkv,G,S,scale,nullptr,nullptr,nullptr);
    // fixed-order deterministic G-head reduction (no atomics) -> bit-identical dK/dV.
    dim3 grGRID(S/8, Hkv, B);
    gqa_dkdv_greduce<<<grGRID,128>>>(d_dVp, d_dV, Hq, Hkv, G, S);
    gqa_dkdv_greduce<<<grGRID,128>>>(d_dKp, d_dK, Hq, Hkv, G, S);
    const int cB=256; const long dqN4=dqN/4; const int cG=(int)((dqN4+cB-1)/cB);
    convert_dq_accum_to_bf16_v6<<<cG,cB>>>(reinterpret_cast<const float4*>(d_dqa), reinterpret_cast<uint2*>(d_dQ), dqN4);
}

} // namespace gqa_bwd_vj1d

// ---------------------------------------------------------------------------
// Library-facing entry point. Called from GQA_bwd_generic.cu only when the
// running GPU is Hopper AND the shape satisfies the d128 tiling constraints.
// chat_tiles/n_chat_tiles: the bwd_tiles list of a GqaChatPlan (design A), or nullptr = dense.
// ---------------------------------------------------------------------------
void gqa_vj1d_causal_bwd_launch_d128(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    const __nv_bfloat16* O, const __nv_bfloat16* dO, const float* LSE,
    __nv_bfloat16* dQ, __nv_bfloat16* dK, __nv_bfloat16* dV,
    int B, int Hq, int Hkv, int G, int S, float scale,
    const int32_t* chat_tiles, int n_chat_tiles)
{
    gqa_bwd_vj1d::launch_gqa_backward_vj1d<64, 64, 128>(
        Q, K, V, O, dO, LSE, dQ, dK, dV, B, Hq, Hkv, G, S, scale, chat_tiles, n_chat_tiles);
}

} // namespace cuda
} // namespace OwnTensor
