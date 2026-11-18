#include "autograd/backward/AttentionBackward.h"
#include "ops/helpers/AttentionKernels.h"
#include <stdexcept>
#include "profiler/RecordFunction.h"

namespace OwnTensor {
namespace autograd {

MemEfficientAttentionBackward::MemEfficientAttentionBackward(
    const Tensor& query, const Tensor& key, const Tensor& value,
    const Tensor& output, const Tensor& lse,
    int64_t B, int64_t nh, int64_t T, int64_t hd,
    bool is_causal)
    : Node(3),  // 3 inputs: Q, K, V
      saved_query_(query), saved_key_(key), saved_value_(value),
      saved_output_(output), saved_lse_(lse),
      B_(B), nh_(nh), T_(T), hd_(hd), is_causal_(is_causal) {}

std::vector<Tensor> MemEfficientAttentionBackward::apply(std::vector<Tensor>&& grads) {
    PROFILER_SCOPE("MemEfficientAttentionBackward::apply");
    if (grads.empty()) {
        throw std::runtime_error("MemEfficientAttentionBackward: no gradients provided");
    }

    const Tensor& grad_output_raw = grads[0];

    if (!grad_output_raw.device().is_cuda() || grad_output_raw.dtype() != Dtype::Float32) {
        throw std::runtime_error(
            "MemEfficientAttentionBackward: only CUDA float32 tensors are supported");
    }

    // Stride-aware path: only the head-dim (last axis) must be unit-strided;
    // all other strides are forwarded to the kernel and used for indexing.
    // This eliminates the 6 x 50 MB .contiguous() copies that used to happen here.
    const Tensor& grad_output = grad_output_raw;

    auto opts = TensorOptions().with_dtype(Dtype::Float32).with_device(grad_output.device());

    // Allocate gradient tensors: same shape as Q, K, V = (B, nh, T, hd).
    // Allocate D buffer as [B, nh, T] (3D) so its stride-B/stride-H are well-defined.
    // dQ/dK/dV zero-init (when needed for atomicAdd) is done inside the kernel launcher.
    Shape qkv_shape({{B_, nh_, T_, hd_}});
    Tensor grad_query = Tensor::empty(qkv_shape, opts);
    Tensor grad_key   = Tensor::empty(qkv_shape, opts);
    Tensor grad_value = Tensor::empty(qkv_shape, opts);

    Shape d_shape({{B_, nh_, T_}});
    Tensor D_buf = Tensor::empty(d_shape, opts);

    // Strides — shape is [B, nh, T, HD], so (strideB, strideH, strideM) = (s0, s1, s2).
    const auto& q_s  = saved_query_.stride().strides;
    const auto& k_s  = saved_key_.stride().strides;
    const auto& v_s  = saved_value_.stride().strides;
    const auto& o_s  = saved_output_.stride().strides;
    const auto& do_s = grad_output.stride().strides;
    const auto& dq_s = grad_query.stride().strides;
    const auto& dk_s = grad_key.stride().strides;
    const auto& dv_s = grad_value.stride().strides;
    const auto& lse_s = saved_lse_.stride().strides;  // [B, nh, T]
    const auto& d_s   = D_buf.stride().strides;        // [B, nh, T]

    // Single unified backward launch: precompute D + KV-tile-centric dQ/dK/dV
    cuda::mem_efficient_attn_backward(
        saved_query_.data<float>(),  q_s[0], q_s[2], q_s[1],
        saved_key_.data<float>(),    k_s[0], k_s[2], k_s[1],
        saved_value_.data<float>(),  v_s[0], v_s[2], v_s[1],
        saved_output_.data<float>(), o_s[0], o_s[2], o_s[1],
        grad_output.data<float>(),   do_s[0], do_s[2], do_s[1],
        saved_lse_.data<float>(),    lse_s[0], lse_s[1],
        grad_query.data<float>(),    dq_s[0], dq_s[2], dq_s[1],
        grad_key.data<float>(),      dk_s[0], dk_s[2], dk_s[1],
        grad_value.data<float>(),    dv_s[0], dv_s[2], dv_s[1],
        D_buf.data<float>(),         d_s[0], d_s[1],
        B_, nh_, T_, hd_, is_causal_);

    return {grad_query, grad_key, grad_value};
}

// ============================================================================
// PackedSDPABackward — writes dQ/dK/dV directly into one packed dqkv via
// strided views. Pre-zeros dqkv once externally; passes skip_grad_zero=true
// so the kernel does not run its contiguous-layout memset (which would
// corrupt the interleaved packed buffer).
// ============================================================================

PackedSDPABackward::PackedSDPABackward(
    const Tensor& qkv,
    const Tensor& output, const Tensor& lse,
    int64_t B, int64_t nh, int64_t T, int64_t hd,
    bool is_causal)
    : Node(1),  // single input edge: qkv
      saved_qkv_(qkv), saved_output_(output), saved_lse_(lse),
      B_(B), nh_(nh), T_(T), hd_(hd), C_(nh * hd), is_causal_(is_causal) {}

std::vector<Tensor> PackedSDPABackward::apply(std::vector<Tensor>&& grads) {
    PROFILER_SCOPE("PackedSDPABackward::apply");
    if (grads.empty() || !grads[0].is_valid()) {
        throw std::runtime_error("PackedSDPABackward: no gradient provided");
    }
    const Tensor& grad_output = grads[0];  // [B, T, C], contiguous
    if (!grad_output.device().is_cuda() || grad_output.dtype() != Dtype::Float32) {
        throw std::runtime_error(
            "PackedSDPABackward: only CUDA float32 tensors are supported");
    }

    auto opts = TensorOptions().with_dtype(Dtype::Float32).with_device(grad_output.device());

    Tensor dqkv  = Tensor::zeros(Shape{{B_, T_, 3 * C_}}, opts);
    Tensor D_buf = Tensor::empty(Shape{{B_, nh_, T_}}, opts);

    // Strides for views over qkv / dqkv as [B, nh, T, hd]:
    //   strideB = T * 3C, strideM = 3C (Q,K,V interleaved on T axis),
    //   strideH = hd, last (hd) stride = 1.
    const int64_t qkv_strideB = T_ * 3 * C_;
    const int64_t qkv_strideM = 3 * C_;
    const int64_t qkv_strideH = hd_;

    // O / dO is packed [B, T, C] contig as [B, nh, T, hd]:
    const int64_t o_strideB = T_ * C_;
    const int64_t o_strideM = C_;
    const int64_t o_strideH = hd_;

    // LSE / D are [B, nh, T] contig.
    const int64_t lse_strideB = nh_ * T_;
    const int64_t lse_strideH = T_;
    const int64_t d_strideB   = nh_ * T_;
    const int64_t d_strideH   = T_;

    const float* qkv_ptr  = saved_qkv_.data<float>();
    float*       dqkv_ptr = dqkv.data<float>();

    cuda::mem_efficient_attn_backward(
        /*Q  */ qkv_ptr + 0,        qkv_strideB, qkv_strideM, qkv_strideH,
        /*K  */ qkv_ptr + C_,       qkv_strideB, qkv_strideM, qkv_strideH,
        /*V  */ qkv_ptr + 2 * C_,   qkv_strideB, qkv_strideM, qkv_strideH,
        /*O  */ saved_output_.data<float>(), o_strideB, o_strideM, o_strideH,
        /*dO */ grad_output.data<float>(),   o_strideB, o_strideM, o_strideH,
        /*LSE*/ saved_lse_.data<float>(),    lse_strideB, lse_strideH,
        /*dQ */ dqkv_ptr + 0,        qkv_strideB, qkv_strideM, qkv_strideH,
        /*dK */ dqkv_ptr + C_,       qkv_strideB, qkv_strideM, qkv_strideH,
        /*dV */ dqkv_ptr + 2 * C_,   qkv_strideB, qkv_strideM, qkv_strideH,
        /*D  */ D_buf.data<float>(), d_strideB, d_strideH,
        B_, nh_, T_, hd_, is_causal_,
        /*skip_grad_zero=*/true);

    return {dqkv};
}

// ============================================================================
// GQAFlashAttentionBackward — packed grouped-query attention backward.
// Writes dQ/dK/dV directly into one zeroed packed dqkv via strided views with
// the unequal layout [ Q: Nq*hd | K: Nkv*hd | V: Nkv*hd ]. skip_grad_zero=true
// because the buffer is pre-zeroed and dQ/dK/dV are interleaved views; the
// kernel accumulates dK/dV across each group of query heads.
// ============================================================================

GQAFlashAttentionBackward::GQAFlashAttentionBackward(
    const Tensor& qkv,
    const Tensor& output, const Tensor& lse,
    int64_t B, int64_t Nq_heads, int64_t Nkv_heads, int64_t T, int64_t hd,
    bool is_causal)
    : Node(1),  // single input edge: qkv
      saved_qkv_(qkv), saved_output_(output), saved_lse_(lse),
      B_(B), Nq_heads_(Nq_heads), Nkv_heads_(Nkv_heads), T_(T), hd_(hd),
      is_causal_(is_causal) {}

std::vector<Tensor> GQAFlashAttentionBackward::apply(std::vector<Tensor>&& grads) {
    PROFILER_SCOPE("GQAFlashAttentionBackward::apply");
    if (grads.empty() || !grads[0].is_valid()) {
        throw std::runtime_error("GQAFlashAttentionBackward: no gradient provided");
    }
    const Tensor& grad_output = grads[0];  // [B, Nq, T, hd] bf16, contiguous
    if (!grad_output.device().is_cuda() || grad_output.dtype() != Dtype::Bfloat16) {
        throw std::runtime_error(
            "GQAFlashAttentionBackward: only CUDA bfloat16 tensors are supported");
    }

    auto opts     = TensorOptions().with_dtype(Dtype::Bfloat16).with_device(grad_output.device());
    auto fp_opts  = TensorOptions().with_dtype(Dtype::Float32).with_device(grad_output.device());

    // dqkv mirrors the forward qkv: same 3-D shape and block-concatenated layout
    // [ dQ:[B,Nq,T,hd] | dK:[B,Nkv,T,hd] | dV:[B,Nkv,T,hd] ]. The kernel scatters
    // each gradient block in place (and accumulates dK/dV across the group), so
    // we hand it one zeroed buffer.
    const int64_t D = (Nq_heads_ + 2 * Nkv_heads_) * hd_;  // full packed width
    Tensor dqkv  = Tensor::zeros(Shape{{B_, T_, D}}, opts);
    Tensor D_buf = Tensor::empty(Shape{{B_, Nq_heads_, T_}}, fp_opts);

    cuda::gqa_flash_attn_backward(
        saved_qkv_.data<bfloat16_t>(),
        saved_output_.data<bfloat16_t>(),
        grad_output.data<bfloat16_t>(),
        saved_lse_.data<float>(),
        dqkv.data<bfloat16_t>(),
        D_buf.data<float>(),
        static_cast<int>(B_), static_cast<int>(Nq_heads_), static_cast<int>(Nkv_heads_),
        static_cast<int>(T_), static_cast<int>(hd_), is_causal_);

    return {dqkv};
}

// ============================================================================
// GQAFlashAttentionQKVBackward — same GQA backward kernels as
// GQAFlashAttentionBackward, but Q/K/V and dQ/dK/dV are independent buffers
// (cuda::gqa_flash_attn_backward_qkv), so no pack/split copy is needed on
// either side.
// ============================================================================

GQAFlashAttentionQKVBackward::GQAFlashAttentionQKVBackward(
    const Tensor& query, const Tensor& key, const Tensor& value,
    const Tensor& output, const Tensor& lse,
    int64_t B, int64_t Nq_heads, int64_t Nkv_heads, int64_t T, int64_t hd,
    bool is_causal, cuda::GqaChatPlanPtr chats)
    : Node(3),  // 3 inputs: Q, K, V
      saved_query_(query), saved_key_(key), saved_value_(value),
      saved_output_(output), saved_lse_(lse),
      B_(B), Nq_heads_(Nq_heads), Nkv_heads_(Nkv_heads), T_(T), hd_(hd),
      is_causal_(is_causal), chats_(std::move(chats)) {}

std::vector<Tensor> GQAFlashAttentionQKVBackward::apply(std::vector<Tensor>&& grads) {
    PROFILER_SCOPE("GQAFlashAttentionQKVBackward::apply");
    if (grads.empty() || !grads[0].is_valid()) {
        throw std::runtime_error("GQAFlashAttentionQKVBackward: no gradient provided");
    }
    const Tensor& grad_output = grads[0];  // [B, Nq, T, hd] bf16, contiguous
    if (!grad_output.device().is_cuda() || grad_output.dtype() != Dtype::Bfloat16) {
        throw std::runtime_error(
            "GQAFlashAttentionQKVBackward: only CUDA bfloat16 tensors are supported");
    }

    auto opts = TensorOptions().with_dtype(Dtype::Bfloat16).with_device(grad_output.device());

    Tensor grad_query = Tensor::empty(Shape{{B_, Nq_heads_,  T_, hd_}}, opts);
    Tensor grad_key   = Tensor::empty(Shape{{B_, Nkv_heads_, T_, hd_}}, opts);
    Tensor grad_value = Tensor::empty(Shape{{B_, Nkv_heads_, T_, hd_}}, opts);

    cuda::gqa_flash_attn_backward_qkv(
        saved_query_.data<bfloat16_t>(),
        saved_key_.data<bfloat16_t>(),
        saved_value_.data<bfloat16_t>(),
        saved_output_.data<bfloat16_t>(),
        grad_output.data<bfloat16_t>(),
        saved_lse_.data<float>(),
        grad_query.data<bfloat16_t>(),
        grad_key.data<bfloat16_t>(),
        grad_value.data<bfloat16_t>(),
        static_cast<int>(B_), static_cast<int>(Nq_heads_), static_cast<int>(Nkv_heads_),
        static_cast<int>(T_), static_cast<int>(hd_), is_causal_,
        chats_.get());   // v2 packed rows (design A), else dense

    return {grad_query, grad_key, grad_value};
}

// ============================================================================
// GQAFusedFlashAttentionBackward — fused QK-norm + RoPE + GQA causal attention.
// fp32 model <-> bf16 kernel bridge. Casts the fp32 output-grad to bf16, runs
// the fused backward (which replays QK-norm + RoPE using the saved q_rstd/k_rstd
// to reconstruct the post-norm-post-RoPE Q/K), then splits the block-concat
// grad_qkv into dQ/dK/dV and casts every block back to fp32. dq_gamma/dk_gamma
// are produced in fp32 directly by the kernel (zeroed internally before accum).
// ============================================================================

GQAFusedFlashAttentionBackward::GQAFusedFlashAttentionBackward(
    const Tensor& qkv, const Tensor& output, const Tensor& lse,
    const Tensor& q_rstd, const Tensor& k_rstd,
    const Tensor& q_gamma, const Tensor& k_gamma, const Tensor& cos_sin_cache,
    int64_t B, int64_t Nq_heads, int64_t Nkv_heads, int64_t T, int64_t hd,
    int cache_seq_len, int pos_offset, float eps,
    bool interleaved, bool is_causal)
    : Node(5),  // 5 inputs: Q, K, V, q_gamma, k_gamma
      saved_qkv_(qkv), saved_output_(output), saved_lse_(lse),
      saved_q_rstd_(q_rstd), saved_k_rstd_(k_rstd),
      saved_q_gamma_(q_gamma), saved_k_gamma_(k_gamma), saved_cache_(cos_sin_cache),
      B_(B), Nq_heads_(Nq_heads), Nkv_heads_(Nkv_heads), T_(T), hd_(hd),
      cache_seq_len_(cache_seq_len), pos_offset_(pos_offset), eps_(eps),
      interleaved_(interleaved), is_causal_(is_causal),
      has_q_gamma_(q_gamma.is_valid()), has_k_gamma_(k_gamma.is_valid()) {}

std::vector<Tensor> GQAFusedFlashAttentionBackward::apply(std::vector<Tensor>&& grads) {
    PROFILER_SCOPE("GQAFusedFlashAttentionBackward::apply");
    if (grads.empty() || !grads[0].is_valid()) {
        throw std::runtime_error("GQAFusedFlashAttentionBackward: no gradient provided");
    }

    // Incoming grad is the fp32 attention-output grad [B, Nq, T, hd]. Make it
    // contiguous (upstream transpose-backward may hand us a strided view) then
    // cast to bf16 for the kernel.
    Tensor grad_out_f = grads[0].contiguous();
    if (!grad_out_f.device().is_cuda()) {
        throw std::runtime_error("GQAFusedFlashAttentionBackward: CUDA tensor required");
    }
    Tensor grad_out_bf = (grad_out_f.dtype() == Dtype::Bfloat16)
                       ? grad_out_f
                       : grad_out_f.as_type(Dtype::Bfloat16);

    auto bf_opts = TensorOptions().with_dtype(Dtype::Bfloat16).with_device(grad_out_f.device());
    auto fp_opts = TensorOptions().with_dtype(Dtype::Float32).with_device(grad_out_f.device());

    const int64_t qElems = B_ * Nq_heads_  * T_ * hd_;
    const int64_t kElems = B_ * Nkv_heads_ * T_ * hd_;
    const int64_t total  = qElems + 2 * kElems;

    // One zeroed bf16 buffer for the block-concat [ dQ | dK | dV ]; the kernel
    // fully overwrites it (dK/dV accumulate across the query-head group in
    // registers, written once), so the zero is belt-and-braces.
    Tensor grad_qkv = Tensor::zeros(Shape{{total}}, bf_opts);
    Tensor D_buf    = Tensor::empty(Shape{{B_, Nq_heads_, T_}}, fp_opts);

    // zeros (not empty) so the gamma grads are well-defined even independent of
    // the kernel's internal cudaMemset — the kernel atomicAdds into them.
    Tensor dq_gamma = has_q_gamma_ ? Tensor::zeros(Shape{{hd_}}, fp_opts) : Tensor();
    Tensor dk_gamma = has_k_gamma_ ? Tensor::zeros(Shape{{hd_}}, fp_opts) : Tensor();

    cuda::gqa_fused_flash_attn_backward(
        saved_qkv_.data<bfloat16_t>(),
        saved_cache_.data<float>(),
        has_q_gamma_ ? saved_q_gamma_.data<float>() : nullptr,
        has_k_gamma_ ? saved_k_gamma_.data<float>() : nullptr,
        saved_output_.data<bfloat16_t>(),
        grad_out_bf.data<bfloat16_t>(),
        saved_lse_.data<float>(),
        saved_q_rstd_.data<float>(),
        saved_k_rstd_.data<float>(),
        grad_qkv.data<bfloat16_t>(),
        has_q_gamma_ ? dq_gamma.data<float>() : nullptr,
        has_k_gamma_ ? dk_gamma.data<float>() : nullptr,
        D_buf.data<float>(),
        static_cast<int>(B_), static_cast<int>(Nq_heads_), static_cast<int>(Nkv_heads_),
        static_cast<int>(T_), static_cast<int>(hd_),
        cache_seq_len_, pos_offset_, eps_, interleaved_, is_causal_);

    // Split the block-concat grad into dQ/dK/dV (contiguous slices) and cast
    // each back to fp32 to match the fp32 model inputs.
    Tensor dQ, dK, dV;
    if (grad_out_f.dtype() ==  Dtype::Float32){
        dQ = grad_qkv.narrow_view(0, 0,               qElems)
                        .reshape(Shape{{B_, Nq_heads_,  T_, hd_}}).as_type(Dtype::Float32);
        dK = grad_qkv.narrow_view(0, qElems,          kElems)
                        .reshape(Shape{{B_, Nkv_heads_, T_, hd_}}).as_type(Dtype::Float32);
        dV = grad_qkv.narrow_view(0, qElems + kElems, kElems)
                        .reshape(Shape{{B_, Nkv_heads_, T_, hd_}}).as_type(Dtype::Float32);
    }else{
        dQ = grad_qkv.narrow_view(0, 0,               qElems)
                        .reshape(Shape{{B_, Nq_heads_,  T_, hd_}});
        dK = grad_qkv.narrow_view(0, qElems,          kElems)
                        .reshape(Shape{{B_, Nkv_heads_, T_, hd_}});
        dV = grad_qkv.narrow_view(0, qElems + kElems, kElems)
                        .reshape(Shape{{B_, Nkv_heads_, T_, hd_}});
    }

    std::vector<Tensor> out(5);
    out[0] = dQ;
    out[1] = dK;
    out[2] = dV;
    out[3] = has_q_gamma_ ? dq_gamma : Tensor();
    out[4] = has_k_gamma_ ? dk_gamma : Tensor();
    return out;
}

} // namespace autograd
} // namespace OwnTensor
