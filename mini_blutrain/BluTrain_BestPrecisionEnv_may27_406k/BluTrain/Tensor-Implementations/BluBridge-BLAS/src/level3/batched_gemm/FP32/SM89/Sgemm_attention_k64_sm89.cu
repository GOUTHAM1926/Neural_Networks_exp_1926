#include "mycublas.h"
#include <cuda_runtime.h>
#include <stdint.h>

// =============================================================================
// Specialized Attention GEMM (K=64) for GPT-2
// Tile: 128x64x64, STAGES=2, THREADS=128
// Layout: NT (Q @ K.T) or NN (Attn @ V)
// =============================================================================

#ifndef MMA_TF32
#define MMA_TF32(d0,d1,d2,d3, a0,a1,a2,a3, b0,b1)                           \
    asm volatile(                                                              \
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "                \
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"                  \
        : "+f"(d0),"+f"(d1),"+f"(d2),"+f"(d3)                                \
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1))
#endif

template <bool IsNT>
__global__ void __launch_bounds__(128, 1)
sgemm_attention_k64_sm89_kernel(
    int M, int N, int K, float alpha,
    const float* __restrict__ A, int lda, long long strideA,
    const float* __restrict__ B, int ldb, long long strideB,
    float beta, float* __restrict__ C, int ldc, long long strideC,
    int batchCount)
{
    const int batch = blockIdx.z, tid = threadIdx.x;
    const int lane = tid & 31, wid = tid >> 5;
    const int bx = blockIdx.x, by = blockIdx.y;

    constexpr int BM = 128, BN = 64, BK = 64;
    extern __shared__ float smem[]; // 2 * (128*64 + 64*64) * 4 = 96KB
    
    float acc[2][8][4]; // Warp handles 32x64. 32/16=2 (M), 64/8=8 (N).
    #pragma unroll
    for (int i=0; i<2; i++) for (int j=0; j<8; j++) 
        acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

    // Async copy pointers (Single K-tile of 64)
    auto load_tile = [&](int stage, int ko) {
        float* As = smem + stage * (BM * BK + BN * BK);
        float* Bs = As + (BM * BK);
        
        // Load A: 128x64. Each of 128 threads loads 64/1=64 floats? No.
        // Let's use 4 float4s per thread. 128 threads * 16 floats = 2048. 
        // 128 * 64 = 8192 total. 8192 / 16 = 512 operations total.
        // Each thread does 4 iterations of float4 loads.
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int r = tid; // Threads cover all rows
            const int c = i * 16 + (lane % 4) * 4; // Placeholder
            // Corrected indexing for A: 128x64
            const int gr = by * BM + r;
            const int gc = ko + i * 16;
            if (gr < M) {
                const float* gA = A + (long long)batch * strideA + (long long)gr * lda + gc;
                uint32_t sm_a = __cvta_generic_to_shared(&As[r * BK + (i * 16 ^ ((r & 7) << 2))]);
                int bytes = max(0, min(16, (K - gc) * 4));
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sm_a), "l"(gA), "r"(bytes));
            } else { *(float4*)&As[r * BK + (i*16 ^ ((r&7)<<2))] = {0,0,0,0}; }
        }
        
        // Load B: 64x64. 4096 floats. 128 threads * 32 floats = 4096.
        // Each thread loads 2 iterations of float4s.
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            const int r = (tid / 4); // 0..31.
            const int r_final = r + i * 32; // 0..63
            const int c = (tid % 4) * 4; // 0,4,8,12
            // B is [K, N] row-major or [N, K] row-major (if NT)
            if constexpr (IsNT) { // B is [N, K]
               const int gn = bx * BN + r_final;
               const int gk = ko + c;
               if (gn < N) {
                   const float* gB = B + (long long)batch * strideB + (long long)gn * ldb + gk;
                   uint32_t sm_b = __cvta_generic_to_shared(&Bs[r_final * BK + (c ^ ((r_final & 7) << 2))]);
                   int bytes = max(0, min(16, (K - gk) * 4));
                   asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sm_b), "l"(gB), "r"(bytes));
               } else { *(float4*)&Bs[r_final * BK + (c ^ ((r_final&7)<<2))] = {0,0,0,0}; }
            } else { // B is [K, N] (NN)
               const int gk = ko + r_final;
               const int gn = bx * BN + c;
               if (gk < K) {
                   const float* gB = B + (long long)batch * strideB + (long long)gk * ldb + gn;
                   uint32_t sm_b = __cvta_generic_to_shared(&Bs[gk * BN + (gn ^ ((gk & 7) << 2))]);
                   int bytes = max(0, min(16, (N - gn) * 4));
                   asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(sm_b), "l"(gB), "r"(bytes));
               } else { *(float4*)&Bs[gk * BN + (gn ^ ((gk&7)<<2))] = {0,0,0,0}; }
            }
        }
    };

    // Since K=64, we might only have ONE stage if we are doing small attention.
    // But GPT-2 has multi-K? No, fixed head_dim. 
    // If K=64, loop runs only once.
    const int ktcnt = (K + BK - 1) / BK;
    
    for (int k = 0; k < K; k += BK) {
        load_tile(0, k);
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_group 0;\n"); __syncthreads();
        
        float* As_st = smem;
        float* Bs_st = smem + (BM * BK);
        
        uint32_t frA[2][4], frB[8][2];
        const int row_warp = wid * 32 + (lane / 4);
        
        #pragma unroll
        for (int ks = 0; ks < BK; ks += 8) {
            // Load Frags — apply +0x1000u rounding at load time for TF32 round-to-nearest
            #pragma unroll
            for (int i=0; i<2; i++) {
                const int r = row_warp + i * 16;
                auto ga = [&](int r, int c) { return *(uint32_t*)&As_st[r * BK + (c ^ ((r & 7) << 2))]; };
                frA[i][0] = ga(r, ks + (lane % 4))       + 0x1000u;
                frA[i][1] = ga(r + 8, ks + (lane % 4))   + 0x1000u;
                frA[i][2] = ga(r, ks + (lane % 4) + 4)   + 0x1000u;
                frA[i][3] = ga(r + 8, ks + (lane % 4) + 4) + 0x1000u;
            }
            #pragma unroll
            for (int j=0; j<8; j++) {
                const int c = j * 8 + (lane / 4);
                if constexpr (IsNT) {
                   auto gb = [&](int r, int cl) { return *(uint32_t*)&Bs_st[cl * BK + (r ^ ((cl & 7) << 2))]; };
                   frB[j][0] = gb(ks + (lane % 4), c)     + 0x1000u;
                   frB[j][1] = gb(ks + (lane % 4) + 4, c) + 0x1000u;
                } else {
                   auto gb = [&](int r, int cl) { return *(uint32_t*)&Bs_st[r * BN + (cl ^ ((r & 7) << 2))]; };
                   frB[j][0] = gb(ks + (lane % 4), c)     + 0x1000u;
                   frB[j][1] = gb(ks + (lane % 4) + 4, c) + 0x1000u;
                }
            }
            #pragma unroll
            for (int i=0; i<2; i++) for (int j=0; j<8; j++)
                MMA_TF32(acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3], frA[i][0], frA[i][1], frA[i][2], frA[i][3], frB[j][0], frB[j][1]);
        }
    }

    // Epilogue
    float* dC = C + (long long)batch * strideC;
    const int ge = lane / 4, te = lane % 4;
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const int r0 = by * BM + wid * 32 + i * 16 + ge, r8 = r0 + 8;
            const int c0 = bx * BN + j * 8 + te * 2;
            auto store = [&](int r, int c, float v0, float v1) {
                if (r < M && c < N) {
                    float* p = &dC[(long long)r * ldc + c];
                    p[0] = alpha * v0 + (beta == 0.f ? 0.f : beta * p[0]);
                    if (c + 1 < N) p[1] = alpha * v1 + (beta == 0.f ? 0.f : beta * p[1]);
                }
            };
            store(r0, c0, acc[i][j][0], acc[i][j][1]);
            store(r8, c0, acc[i][j][2], acc[i][j][3]);
        }
    }
}

extern "C" void launch_sgemm_attention_k64_sm89(
    int M, int N, int K,
    float alpha, const float* A, int lda, long long sA,
    const float* B, int ldb, long long sB,
    float beta, float* C, int ldc, long long sC,
    int batchCount, bool isNT, cudaStream_t stream)
{
    // 128x64x64 tile. Shared Memory: 96KB
    static constexpr int SMEM = (128 * 64 + 64 * 64) * 4;
    static bool set = false;
    if (!set) {
        cudaFuncSetAttribute(sgemm_attention_k64_sm89_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        cudaFuncSetAttribute(sgemm_attention_k64_sm89_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        set = true;
    }
    dim3 block(128); // 4 warps
    dim3 grid((N + 63) / 64, (M + 127) / 128, batchCount);
    if (isNT) sgemm_attention_k64_sm89_kernel<true><<<grid, block, SMEM, stream>>>(M, N, K, alpha, A, lda, sA, B, ldb, sB, beta, C, ldc, sC, batchCount);
    else      sgemm_attention_k64_sm89_kernel<false><<<grid, block, SMEM, stream>>>(M, N, K, alpha, A, lda, sA, B, ldb, sB, beta, C, ldc, sC, batchCount);
}
