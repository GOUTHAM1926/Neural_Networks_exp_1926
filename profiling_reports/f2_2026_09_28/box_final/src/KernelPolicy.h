#pragma once
#include <cstdlib>
#include <cstring>
#include <strings.h>  // strcasecmp

// KernelPolicy.h — runtime opt-out switches for the arch-optimized kernel paths.
//
// Each family of sm_90 kernels has a generic sibling that the dispatcher would
// pick on non-Hopper hardware. These switches force the generic path even on
// Hopper, so an ablation run can attribute time to a specific optimized family
// without rebuilding for another arch.
//
// Env vars, not a config struct: dispatch happens inside this library, which is
// linked as a separate static archive, so there is no object the trainer can
// hand down to the launch site. train/production/bluscript.cpp sets these from
// its YAML before the first forward; an explicit env var wins over the config
// (the trainer only calls setenv when the var is unset). Same contract as
// BLUTRAIN_EMB_BWD_COOPERATIVE.
//
// Each getter caches in a function-local static, so the value is read once per
// process and the launch-site cost is a load. That also means setenv must
// happen before the first launch — after that the value is frozen.
//
// "1", "true", "yes", "on" (any case) enable; everything else, including unset,
// disables. Default-off means an unconfigured build behaves exactly as before.

namespace OwnTensor {
namespace cuda {

namespace detail {

inline bool env_flag_enabled(const char* name) {
    const char* v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return false;
    return std::strcmp(v, "1") == 0 ||
           strcasecmp(v, "true") == 0 ||
           strcasecmp(v, "yes")  == 0 ||
           strcasecmp(v, "on")   == 0;
}

}  // namespace detail

// Master switch: force the generic path for every family below at once.
inline bool disable_all_sm90() {
    static const bool v = detail::env_flag_enabled("BLUTRAIN_NO_SM90");
    return v;
}

// GQA fused attention — gqa62::gqa_v62_sm90 (fwd),
// gqa_backward_v44_kv_sm90 + compute_drowsum_v22_sm90 +
// convert_dq_accum_to_bf16_v5_sm90 (bwd; the latter two are fixed sub-steps of
// the v44 launch and go away with it).
//
// WARNING: the generic causal path is only implemented for head_dim 64. With
// head_dim 128 (the production config) this throws rather than falling back, by
// design — a silent switch to the non-causal WMMA kernel would change the math
// instead of just the speed.
inline bool disable_sm90_gqa() {
    static const bool v = disable_all_sm90() || detail::env_flag_enabled("BLUTRAIN_NO_SM90_GQA");
    return v;
}

// RMSNorm — rms_norm_fwd_staged_kernel_sm90 (fwd), rms_bwd_fused_block_sm90 and
// rms_bwd_gamma_finalize_sm90 (bwd; finalize is a mandatory epilogue of the
// sm_90 backward and disappears with it).
inline bool disable_sm90_rmsnorm() {
    static const bool v = disable_all_sm90() || detail::env_flag_enabled("BLUTRAIN_NO_SM90_RMSNORM");
    return v;
}

// Fused QK-norm + RoPE — fused_qknorm_rope_forward_t_sm90,
// fused_qk_norm_rope_bwd_kernel_sm90.
inline bool disable_sm90_qknorm_rope() {
    static const bool v = disable_all_sm90() || detail::env_flag_enabled("BLUTRAIN_NO_SM90_QKNORM_ROPE");
    return v;
}

// SwiGLU — swiglu_fwd_vec_kernel_sm90, swiglu_bwd_v22_vec_kernel_sm90.
inline bool disable_sm90_swiglu() {
    static const bool v = disable_all_sm90() || detail::env_flag_enabled("BLUTRAIN_NO_SM90_SWIGLU");
    return v;
}

// bf16 <-> fp32 conversion — sm90::vec8_convert_to_bf16 (down-cast),
// sm90::vec4_convert_to_fp32 (up-cast). Falls back to scalar convert_type_kernel.
inline bool disable_sm90_convert() {
    static const bool v = disable_all_sm90() || detail::env_flag_enabled("BLUTRAIN_NO_SM90_CONVERT");
    return v;
}

// In-place add, flat variant — add_inplace_kernel_flat_sm90.
//
// Scope note: this covers ONLY the flat kernel, which is arch- and size-gated
// (>= 8M vectors). It does NOT cover add_inplace_kernel_sm90<T,2>: despite the
// suffix that one is the unconditional default for every fast dtype on every
// arch and has no generic sibling, so there is nothing to switch to. Disabling
// the flat path lands on add_inplace_kernel_sm90<T,2>, which the in-source
// comment notes is actually faster below the threshold.
inline bool disable_sm90_add_flat() {
    static const bool v = disable_all_sm90() || detail::env_flag_enabled("BLUTRAIN_NO_SM90_ADD_FLAT");
    return v;
}

// Fused bf16/fp16 -> fp32 gradient accumulation — cuda_accumulate_lowp_into_fp32,
// called from AutogradMeta::accumulate_grad. BLUTRAIN_NO_FUSED_GRAD_ACC=1 restores
// `grad += update.as_type(Float32)`. Both give the same bits, so the switch exists
// for A/B timing only. Not an sm_90 kernel, so BLUTRAIN_NO_SM90 does not affect it.
inline bool disable_fused_grad_accum() {
    static const bool v = detail::env_flag_enabled("BLUTRAIN_NO_FUSED_GRAD_ACC");
    return v;
}

}  // namespace cuda
}  // namespace OwnTensor
