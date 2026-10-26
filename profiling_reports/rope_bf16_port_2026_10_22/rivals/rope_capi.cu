// =============================================================================
// rope_capi.cu (2026-10-22): plain C entry points to the bf16 fused QK-RMSNorm + RoPE launchers (Kruthik's kernel with
// the stage-B positions), for the Python rival benchmark. Python loads the .so with ctypes and passes the device
// pointers of torch tensors and torch's current stream, so the launches are ordered with torch's work. No torch
// headers and no libtensor are involved.
// Layouts, as the trainer uses them: q, k and dq, dk token-major [B, T, H*hd]; q_out, k_out and the upstream gradients
// dz_q, dz_k head-major [B, H, T, hd]; rstd fp32 [B, H, T]; gamma, dgamma fp32 [hd]; cos_sin fp32 [T, hd] (cos | sin);
// positions int32 [B, T] or NULL.
// build (the library's flags, CUDA 13.0; 2026-10-23: --use_fast_math added, the 10-22 build lacked it):
//   /usr/local/cuda-13.0/bin/nvcc -std=c++17 --use_fast_math -O3 --expt-relaxed-constexpr
//        -gencode arch=compute_90a,code=sm_90a -Xcompiler -fPIC -shared
//        -I<Tensor-Implementations>/include -I<dir holding FusedRMSRoPE_bf16.cu> -o libropeours.so rope_capi.cu
// =============================================================================
#include "FusedRMSRoPE_bf16.cu"

using OwnTensor::bfloat16_t;

extern "C" {

void ours_fwd(const void* q, const void* k, void* q_out, void* k_out, const float* q_gamma, const float* k_gamma,
              const float* cos_sin, float* q_rstd, float* k_rstd, int B, int Hq, int Hkv, int T, int hd, float eps,
              const int32_t* positions, void* stream) {
    OwnTensor::cuda::fused_qk_norm_rope_forward_bf16_tm(
        static_cast<const bfloat16_t*>(q), static_cast<const bfloat16_t*>(k),
        static_cast<bfloat16_t*>(q_out), static_cast<bfloat16_t*>(k_out),
        q_gamma, k_gamma, cos_sin, q_rstd, k_rstd, B, Hq, Hkv, T, hd,
        /*pos_offset=*/0, eps, /*interleaved=*/false, static_cast<cudaStream_t>(stream), positions);
}

void ours_bwd(const void* dz_q, const void* dz_k, const void* q, const void* k, const float* q_gamma,
              const float* k_gamma, const float* cos_sin, const float* q_rstd, const float* k_rstd, void* dq, void* dk,
              float* dq_gamma, float* dk_gamma, int B, int Hq, int Hkv, int T, int hd, const int32_t* positions,
              void* stream) {
    OwnTensor::cuda::fused_qk_norm_rope_backward_bf16_tm(
        static_cast<const bfloat16_t*>(dz_q), static_cast<const bfloat16_t*>(dz_k),
        static_cast<const bfloat16_t*>(q), static_cast<const bfloat16_t*>(k),
        q_gamma, k_gamma, cos_sin, q_rstd, k_rstd,
        static_cast<bfloat16_t*>(dq), static_cast<bfloat16_t*>(dk), dq_gamma, dk_gamma, B, Hq, Hkv, T, hd,
        /*pos_offset=*/0, /*interleaved=*/false, static_cast<cudaStream_t>(stream), positions);
}

}  // extern "C"
