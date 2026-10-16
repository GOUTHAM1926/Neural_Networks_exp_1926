#include "autograd/operations/AttentionOps.h"
#include "autograd/operations/MatrixOps.h"
#include "autograd/operations/ActivationOps.h"
#include "autograd/operations/ReshapeOps.h"
#include "autograd/operations/BinaryOps.h"
#include "autograd/ops_template.h"
#include "checkpointing/GradMode.h"
#include "utils/Profiler.h"

#ifdef WITH_CUDA
#include "ops/helpers/AttentionKernels.h"
#include "autograd/backward/AttentionBackward.h"
#endif
#include "ops/Kernels.h"
#include "autograd/backward/FusedAttnSoftmaxMatmulBackward.h"
#include "profiler/AllocationTracker.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include "profiler/RecordFunction.h"

namespace OwnTensor {
namespace autograd {

// ============================================================================
// Math Backend: composes existing autograd ops
// ============================================================================

static Tensor sdpa_math(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    bool is_causal)
{
    PROFILER_SCOPE("sdpa_math");
    // std::cout<<"hi"<<std::endl;
    // query, key, value: (B, nh, T, hd)
    if(query.device() != key.device() && query.device()!= value.device())
    {
    
        throw std::runtime_error(
            "Device Mismatch, Q, K, V do not reside in the same device");
    
    }
    int64_t hd = query.shape().dims[3];
    float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    // std::cout<<scale<<std::endl;
    // Scale query
    Tensor scale_t = Tensor::full(Shape{{1}},
        TensorOptions().with_dtype(query.dtype()).with_device(query.device()),
        scale);
    Tensor q_scaled = autograd::mul(query, scale_t);

    // Q @ K^T -> (B, nh, T, T)
    Tensor k_t = autograd::transpose(key, -2, -1);
    Tensor attn_weights = autograd::matmul(q_scaled, k_t);

    // Apply causal mask + softmax
    Tensor attn_probs;
    if (is_causal) {
        float neg_inf = -std::numeric_limits<float>::infinity();
        attn_probs = autograd::fused_tril_softmax(attn_weights, 0, neg_inf);
    } else {
        attn_probs = autograd::softmax(attn_weights, -1);
    }

    // Attn @ V -> (B, nh, T, hd)
    return autograd::matmul(attn_probs, value);
}

// ============================================================================
// Memory-Efficient Backend (requires CUDA)
// ============================================================================

#ifdef WITH_CUDA
static Tensor sdpa_memory_efficient(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    bool is_causal,
    float* qk_max_out = nullptr)
{
    PROFILER_SCOPE("sdpa_memory_efficient");
    if(query.device() != key.device() && query.device()!= value.device())
    {
        throw std::runtime_error("Device Mismatch, Q, K, V do not reside in the same device");
    }
    if (!query.device().is_cuda() || query.dtype() != Dtype::Float32) {
        throw std::runtime_error(
            "Memory-efficient attention requires CUDA float32 tensors");
    }

    int64_t B  = query.shape().dims[0];
    int64_t nh = query.shape().dims[1];
    int64_t T  = query.shape().dims[2];
    int64_t hd = query.shape().dims[3];

    // Stride-aware path (matches PyTorch's _efficient_attention_forward):
    // kernel reads Q/K/V with per-tensor strides, so no .contiguous() copy
    // is needed for transposed views of a [B, T, H, HD] tensor. The only
    // requirement is that the last dim (HeadDim) is contiguous.
    const auto& q_strides = query.stride().strides;
    const auto& k_strides = key.stride().strides;
    const auto& v_strides = value.stride().strides;

    auto opts = TensorOptions().with_dtype(Dtype::Float32).with_device(query.device());

    // Allocate output and LSE as contiguous [B, nh, T, hd] / [B, nh, T]
    Tensor output = Tensor::empty(Shape{{B, nh, T, hd}}, opts);
    Tensor lse    = Tensor::empty(Shape{{B, nh, T}}, opts);
    const auto& o_strides = output.stride().strides;
    const auto& lse_strides = lse.stride().strides;

    // Single fused kernel: Q @ K^T → scale → mask → online softmax → @ V
    // Never materializes the T×T attention matrix.
    cuda::mem_efficient_attn_forward_tc(
        query.data<float>(), q_strides[0], q_strides[2], q_strides[1],
        key.data<float>(),   k_strides[0], k_strides[2], k_strides[1],
        value.data<float>(), v_strides[0], v_strides[2], v_strides[1],
        output.data<float>(), o_strides[0], o_strides[2], o_strides[1],
        lse.data<float>(),    lse_strides[0], lse_strides[1],
        B, nh, T, hd, is_causal, 0.0f /*dropout_p*/, nullptr /*dropout_mask*/,
        qk_max_out /*per-head S^h_max for QK-Clip; nullptr = disabled*/);

    // Build autograd graph with memory-efficient backward
    if (GradMode::is_enabled() &&
        (query.requires_grad() || key.requires_grad() || value.requires_grad()))
    {
        // Save the ORIGINAL (possibly strided) views — backward also handles strides.
        auto grad_fn = std::make_shared<MemEfficientAttentionBackward>(
            query.detach(), key.detach(), value.detach(),
            output.detach(), lse.detach(),
            B, nh, T, hd, is_causal);

        Tensor& q_mut = const_cast<Tensor&>(query);
        Tensor& k_mut = const_cast<Tensor&>(key);
        Tensor& v_mut = const_cast<Tensor&>(value);

        if (query.requires_grad()) {
            grad_fn->set_next_edge(0, get_grad_edge(q_mut));
        }
        if (key.requires_grad()) {
            grad_fn->set_next_edge(1, get_grad_edge(k_mut));
        }
        if (value.requires_grad()) {
            grad_fn->set_next_edge(2, get_grad_edge(v_mut));
        }

        output.set_grad_fn(grad_fn);
        output.set_requires_grad(true);
    }

    return output;
}
#endif

// ============================================================================
// Packed-QKV mem-efficient forward (zero shard / reshape / transpose).
// Reads Q/K/V from a single qkv [B,T,3C] via strided pointers.
// Writes output as packed [B,T,C] contiguous (heads packed inner).
// ============================================================================

#ifdef WITH_CUDA
static Tensor sdpa_memory_efficient_packed(
    const Tensor& qkv,
    int64_t n_heads,
    bool is_causal,
    float* qk_max_out = nullptr)
{
    PROFILER_SCOPE("sdpa_memory_efficient_packed");
    if (!qkv.device().is_cuda() || qkv.dtype() != Dtype::Float32) {
        throw std::runtime_error(
            "scaled_dot_product_attention_packed: only CUDA float32 tensors are supported");
    }
    if (qkv.shape().dims.size() != 3) {
        throw std::runtime_error(
            "scaled_dot_product_attention_packed: qkv must be a 3-D tensor [B, T, 3*C]");
    }

    const int64_t B  = qkv.shape().dims[0];
    const int64_t T  = qkv.shape().dims[1];
    const int64_t C3 = qkv.shape().dims[2];
    if (C3 % 3 != 0) {
        throw std::runtime_error(
            "scaled_dot_product_attention_packed: qkv last dim must be divisible by 3");
    }
    const int64_t C  = C3 / 3;
    if (n_heads <= 0 || C % n_heads != 0) {
        throw std::runtime_error(
            "scaled_dot_product_attention_packed: n_embd must be divisible by n_heads");
    }
    const int64_t nh = n_heads;
    const int64_t hd = C / nh;

    auto opts = TensorOptions().with_dtype(Dtype::Float32).with_device(qkv.device());

    Tensor output = Tensor::empty(Shape{{B, T, C}}, opts);
    Tensor lse    = Tensor::empty(Shape{{B, nh, T}}, opts);

    // qkv viewed as [B, nh, T, hd]:
    //   strideB = T * 3C, strideM = 3C, strideH = hd, last = 1.
    const int64_t qkv_strideB = T * 3 * C;
    const int64_t qkv_strideM = 3 * C;
    const int64_t qkv_strideH = hd;

    // output is [B,T,C] contig viewed as [B,nh,T,hd]:
    const int64_t o_strideB = T * C;
    const int64_t o_strideM = C;
    const int64_t o_strideH = hd;

    const int64_t lse_strideB = nh * T;
    const int64_t lse_strideH = T;

    const float* qkv_ptr = qkv.data<float>();
    cuda::mem_efficient_attn_forward_tc(
        /*Q*/ qkv_ptr + 0,        qkv_strideB, qkv_strideM, qkv_strideH,
        /*K*/ qkv_ptr + C,        qkv_strideB, qkv_strideM, qkv_strideH,
        /*V*/ qkv_ptr + 2 * C,    qkv_strideB, qkv_strideM, qkv_strideH,
        output.data<float>(),  o_strideB, o_strideM, o_strideH,
        lse.data<float>(),     lse_strideB, lse_strideH,
        B, nh, T, hd, is_causal, 0.0f, nullptr,
        qk_max_out /*per-head S^h_max for QK-Clip; nullptr = disabled*/);

    if (GradMode::is_enabled() && qkv.requires_grad()) {
        auto grad_fn = std::make_shared<PackedSDPABackward>(
            qkv.detach(), output.detach(), lse.detach(),
            B, nh, T, hd, is_causal);

        Tensor& qkv_mut = const_cast<Tensor&>(qkv);
        grad_fn->set_next_edge(0, get_grad_edge(qkv_mut));

        output.set_grad_fn(grad_fn);
        output.set_requires_grad(true);
    }

    return output;
}
#endif

//========================
// FlashAttention Backend
//========================
#ifdef WITH_CUDA
static Tensor sdpa_flash_attention(
  const Tensor& qkv,
  const int Nq_heads,
  const int Nkv_heads,
  bool is_causal,
  float dropout_p,
  SDPBackend /*backend*/
){
  PROFILER_SCOPE("sdpa_flash_attention");

  // ---- validation ----
  // The bf16 WMMA kernel reads a single contiguous, block-concatenated buffer:
  //   [ Q:[B,Nq,T,hd] | K:[B,Nkv,T,hd] | V:[B,Nkv,T,hd] ]  (each block dense).
  // The 3-D shape [B, T, (Nq + 2*Nkv)*hd] is used only to recover B, T, hd.
  if (!qkv.device().is_cuda() || qkv.dtype() != Dtype::Bfloat16) {
      throw std::runtime_error(
          "sdpa_flash_attention (GQA): only CUDA bfloat16 tensors are supported");
  }
  if (qkv.shape().dims.size() != 3) {
      throw std::runtime_error(
          "sdpa_flash_attention (GQA): qkv must be 3-D [B, T, (Nq + 2*Nkv)*head_dim]");
  }
  if (Nq_heads <= 0 || Nkv_heads <= 0) {
      throw std::runtime_error(
          "sdpa_flash_attention (GQA): Nq_heads and Nkv_heads must be positive");
  }
  if (Nq_heads % Nkv_heads != 0) {
      throw std::runtime_error(
          "sdpa_flash_attention (GQA): Nq_heads must be a multiple of Nkv_heads");
  }
  if (dropout_p != 0.0f) {
      throw std::runtime_error(
          "sdpa_flash_attention (GQA): dropout is not supported yet (dropout_p must be 0)");
  }

  const int64_t B = qkv.shape().dims[0];
  const int64_t T = qkv.shape().dims[1];
  const int64_t D = qkv.shape().dims[2];

  const int64_t total_heads =
      static_cast<int64_t>(Nq_heads) + 2 * static_cast<int64_t>(Nkv_heads);
  if (D % total_heads != 0) {
      throw std::runtime_error(
          "sdpa_flash_attention (GQA): qkv last dim must be divisible by (Nq + 2*Nkv)");
  }
  const int64_t hd = D / total_heads;

  auto opts     = TensorOptions().with_dtype(Dtype::Bfloat16).with_device(qkv.device());
  auto lse_opts = TensorOptions().with_dtype(Dtype::Float32).with_device(qkv.device());

  // Output and LSE match the kernel's layout exactly: O is [B, Nq, T, hd] (bf16),
  // LSE is [B, Nq, T] (fp32). (Reshape/transpose for c_proj is done by the caller.)
  Tensor output = Tensor::empty(Shape{{B, static_cast<int64_t>(Nq_heads), T, hd}}, opts);
  Tensor lse    = Tensor::empty(Shape{{B, static_cast<int64_t>(Nq_heads), T}}, lse_opts);

  // Single fused bf16 kernel. It derives the Q/K/V block base pointers from the
  // one contiguous packed buffer and maps query head h -> KV head
  // h / (Nq_heads / Nkv_heads).
  cuda::gqa_flash_attn_forward(
      qkv.data<bfloat16_t>(),
      output.data<bfloat16_t>(),
      lse.data<float>(),
      static_cast<int>(B), Nq_heads, Nkv_heads,
      static_cast<int>(T), static_cast<int>(hd),
      is_causal);

  // ---- autograd wiring (single packed input edge) ----
  if (GradMode::is_enabled() && qkv.requires_grad()) {
      auto grad_fn = std::make_shared<GQAFlashAttentionBackward>(
          qkv.detach(), output.detach(), lse.detach(),
          B, Nq_heads, Nkv_heads, T, hd, is_causal);

      Tensor& qkv_mut = const_cast<Tensor&>(qkv);
      grad_fn->set_next_edge(0, get_grad_edge(qkv_mut));

      output.set_grad_fn(grad_fn);
      output.set_requires_grad(true);
  }

  return output;
}
#endif

//==============================================================================
// Unpacked-QKV GQA flash-attention forward. Same underlying kernels as
// sdpa_flash_attention (gqa_v2 / gqa_v48_causal via
// cuda::gqa_flash_attn_forward_qkv), but Q/K/V are passed straight through as
// three independent device pointers -- no packing, no copy. Each of Q/K/V
// individually still needs to be dense [B, heads, T, hd] (.contiguous() is a
// no-op unless the caller handed in a strided view).
//==============================================================================
#ifdef WITH_CUDA
static Tensor sdpa_flash_attention_qkv(
    const Tensor& query,   // [B, Nq_heads,  T, hd] bf16
    const Tensor& key,     // [B, Nkv_heads, T, hd] bf16
    const Tensor& value,   // [B, Nkv_heads, T, hd] bf16
    const int Nq_heads,
    const int Nkv_heads,
    bool is_causal,
    float dropout_p,
    SDPBackend /*backend*/,
    const cuda::GqaChatPlanPtr& chats)   // v2 packed rows: per-chat plan (design A); nullptr = dense
{
    PROFILER_SCOPE("sdpa_flash_attention_qkv");

    if (!query.device().is_cuda() || !key.device().is_cuda() || !value.device().is_cuda()) {
        throw std::runtime_error("sdpa_flash_attention_qkv (GQA): Q/K/V must be CUDA tensors");
    }
    if (query.dtype() != Dtype::Bfloat16 || key.dtype() != Dtype::Bfloat16 ||
        value.dtype() != Dtype::Bfloat16) {
        throw std::runtime_error(
            "sdpa_flash_attention_qkv (GQA): only CUDA bfloat16 tensors are supported");
    }
    if (query.shape().dims.size() != 4 || key.shape().dims.size() != 4 ||
        value.shape().dims.size() != 4) {
        throw std::runtime_error(
            "sdpa_flash_attention_qkv (GQA): Q/K/V must be 4-D [B, heads, T, head_dim]");
    }
    if (Nq_heads <= 0 || Nkv_heads <= 0) {
        throw std::runtime_error(
            "sdpa_flash_attention_qkv (GQA): Nq_heads and Nkv_heads must be positive");
    }
    if (Nq_heads % Nkv_heads != 0) {
        throw std::runtime_error(
            "sdpa_flash_attention_qkv (GQA): Nq_heads must be a multiple of Nkv_heads");
    }
    if (dropout_p != 0.0f) {
        throw std::runtime_error(
            "sdpa_flash_attention_qkv (GQA): dropout is not supported yet (dropout_p must be 0)");
    }

    const int64_t B  = query.shape().dims[0];
    const int64_t T  = query.shape().dims[2];
    const int64_t hd = query.shape().dims[3];

    if (query.shape().dims[1] != Nq_heads) {
        throw std::runtime_error("sdpa_flash_attention_qkv (GQA): query head count != Nq_heads");
    }
    if (key.shape().dims[1] != Nkv_heads || value.shape().dims[1] != Nkv_heads) {
        throw std::runtime_error("sdpa_flash_attention_qkv (GQA): key/value head count != Nkv_heads");
    }
    if (key.shape().dims[0] != B || value.shape().dims[0] != B ||
        key.shape().dims[2] != T || value.shape().dims[2] != T ||
        key.shape().dims[3] != hd || value.shape().dims[3] != hd) {
        throw std::runtime_error(
            "sdpa_flash_attention_qkv (GQA): Q/K/V must share batch size, sequence length, "
            "and head_dim");
    }

    auto opts     = TensorOptions().with_dtype(Dtype::Bfloat16).with_device(query.device());
    auto lse_opts = TensorOptions().with_dtype(Dtype::Float32).with_device(query.device());

    // Each of Q/K/V must individually be dense [B, heads, T, hd] -- the
    // kernel indexes them with computed row-major offsets. No copy happens
    // here unless the caller handed in a non-contiguous view.
    Tensor q_c = query.contiguous();
    Tensor k_c = key.contiguous();
    Tensor v_c = value.contiguous();

    // v2 packed rows (design A, 2026-10-15): the kernels check the plan's shape (B, T, Nq_heads) and refuse a plan
    // they cannot honour (non-Hopper, hd 64, T % 256 != 0 in the forward, Vj1 / v44 in the backward).
    if (chats && !is_causal)
        throw std::runtime_error("sdpa_flash_attention_qkv (GQA): a packed-rows plan needs is_causal = true");

    Tensor output = Tensor::empty(Shape{{B, static_cast<int64_t>(Nq_heads), T, hd}}, opts);
    Tensor lse    = Tensor::empty(Shape{{B, static_cast<int64_t>(Nq_heads), T}}, lse_opts);

    cuda::gqa_flash_attn_forward_qkv(
        q_c.data<bfloat16_t>(),
        k_c.data<bfloat16_t>(),
        v_c.data<bfloat16_t>(),
        output.data<bfloat16_t>(),
        lse.data<float>(),
        static_cast<int>(B), Nq_heads, Nkv_heads,
        static_cast<int>(T), static_cast<int>(hd),
        is_causal, chats.get());

    if (GradMode::is_enabled() &&
        (query.requires_grad() || key.requires_grad() || value.requires_grad()))
    {
        auto grad_fn = std::make_shared<GQAFlashAttentionQKVBackward>(
            q_c.detach(), k_c.detach(), v_c.detach(), output.detach(), lse.detach(),
            B, Nq_heads, Nkv_heads, T, hd, is_causal, chats);

        Tensor& q_mut = const_cast<Tensor&>(query);
        Tensor& k_mut = const_cast<Tensor&>(key);
        Tensor& v_mut = const_cast<Tensor&>(value);
        if (query.requires_grad()) grad_fn->set_next_edge(0, get_grad_edge(q_mut));
        if (key.requires_grad())   grad_fn->set_next_edge(1, get_grad_edge(k_mut));
        if (value.requires_grad()) grad_fn->set_next_edge(2, get_grad_edge(v_mut));

        output.set_grad_fn(grad_fn);
        output.set_requires_grad(true);
    }

    return output;
}
#endif

//==========================================================================
// FUSED GQA Backend: QK-norm + RoPE + (causal) GQA, fp32 model <-> bf16 kernel
//==========================================================================
#ifdef WITH_CUDA
static Tensor sdpa_gqa_fused(
    const Tensor& query,          // [B, Nq,  T, hd] fp32
    const Tensor& key,            // [B, Nkv, T, hd] fp32
    const Tensor& value,          // [B, Nkv, T, hd] fp32
    const Tensor& q_gamma,        // [hd] fp32 or invalid (skip Q QK-norm)
    const Tensor& k_gamma,        // [hd] fp32 or invalid (skip K QK-norm)
    const Tensor& cos_sin_cache,  // [cache_seq_len, hd] fp32
    int Nq_heads, int Nkv_heads,
    bool is_causal, bool interleaved, float eps, int pos_offset)
{
    PROFILER_SCOPE("sdpa_gqa_fused");

    if (!query.device().is_cuda() || !key.device().is_cuda() || !value.device().is_cuda()) {
        throw std::runtime_error("sdpa_gqa_fused: Q/K/V must be CUDA tensors");
    }
    // if (query.dtype() != Dtype::Float32 || key.dtype() != Dtype::Float32 ||
    //     value.dtype() != Dtype::Float32) {
    //     throw std::runtime_error(
    //         "sdpa_gqa_fused: Q/K/V must be Float32 (the bf16 cast is done internally)");
    // }
    
    if (query.shape().dims.size() != 4 || key.shape().dims.size() != 4 ||
        value.shape().dims.size() != 4) {
        throw std::runtime_error("sdpa_gqa_fused: Q/K/V must be 4-D [B, heads, T, head_dim]");
    }

    const int64_t B  = query.shape().dims[0];
    const int64_t T  = query.shape().dims[2];
    const int64_t hd = query.shape().dims[3];

    if (query.shape().dims[1] != Nq_heads) {
        throw std::runtime_error("sdpa_gqa_fused: query head count != Nq_heads");
    }
    if (key.shape().dims[1] != Nkv_heads || value.shape().dims[1] != Nkv_heads) {
        throw std::runtime_error("sdpa_gqa_fused: key/value head count != Nkv_heads");
    }
    if (Nkv_heads <= 0 || Nq_heads % Nkv_heads != 0) {
        throw std::runtime_error(
            "sdpa_gqa_fused: Nq_heads must be a positive multiple of Nkv_heads");
    }
    if (hd != 64 && hd != 128) {
        throw std::runtime_error("sdpa_gqa_fused: head_dim must be 64 or 128");
    }
    if (T % 32 != 0) {
        throw std::runtime_error("sdpa_gqa_fused: T must be a multiple of 32");
    }
    if (!cos_sin_cache.is_valid() || cos_sin_cache.dtype() != Dtype::Float32 ||
        cos_sin_cache.shape().dims.size() != 2) {
        throw std::runtime_error(
            "sdpa_gqa_fused: cos_sin_cache must be a valid 2-D Float32 [cache_seq_len, hd] tensor");
    }
    const int cache_seq_len = static_cast<int>(cos_sin_cache.shape().dims[0]);
    if (static_cast<int64_t>(cache_seq_len) < T + pos_offset) {
        throw std::runtime_error("sdpa_gqa_fused: cache_seq_len must be >= T + pos_offset");
    }

    const bool has_qg = q_gamma.is_valid();
    const bool has_kg = k_gamma.is_valid();

    auto bf_opts = TensorOptions().with_dtype(Dtype::Bfloat16).with_device(query.device());
    auto fp_opts = TensorOptions().with_dtype(Dtype::Float32).with_device(query.device());

    // Build the raw (pre-norm, pre-rope) bf16 packed buffer. Concatenating the
    // flattened bf16 casts of Q, K, V along dim 0 yields exactly the kernel's
    // block-concatenated layout: [ Q:[B,Nq,T,hd] | K:[B,Nkv,T,hd] | V:[B,Nkv,T,hd] ].
    Tensor q_bf = query.contiguous().as_type(Dtype::Bfloat16);
    Tensor k_bf = key.contiguous().as_type(Dtype::Bfloat16);
    Tensor v_bf = value.contiguous().as_type(Dtype::Bfloat16);
    Tensor packed = Tensor::cat({ q_bf.flatten(), k_bf.flatten(), v_bf.flatten() }, 0);

    Tensor output = Tensor::empty(Shape{{B, static_cast<int64_t>(Nq_heads), T, hd}}, bf_opts);
    Tensor lse    = Tensor::empty(Shape{{B, static_cast<int64_t>(Nq_heads),  T}}, fp_opts);
    Tensor q_rstd = Tensor::empty(Shape{{B, static_cast<int64_t>(Nq_heads),  T}}, fp_opts);
    Tensor k_rstd = Tensor::empty(Shape{{B, static_cast<int64_t>(Nkv_heads), T}}, fp_opts);

    cuda::gqa_fused_flash_attn_forward(
        packed.data<bfloat16_t>(),
        cos_sin_cache.data<float>(),
        has_qg ? q_gamma.data<float>() : nullptr,
        has_kg ? k_gamma.data<float>() : nullptr,
        output.data<bfloat16_t>(),
        lse.data<float>(),
        q_rstd.data<float>(),
        k_rstd.data<float>(),
        static_cast<int>(B), Nq_heads, Nkv_heads, static_cast<int>(T), static_cast<int>(hd),
        cache_seq_len, pos_offset, eps, interleaved, is_causal);

    // Dtype-transparent: return the attention output in the QUERY's dtype so both
    // param_dtype settings work — bf16 model -> bf16 (the kernel's native output, no
    // cast), fp32 model -> fp32. The kernel always computes bf16 `output`; upcast only
    // when the caller is fp32. as_type is a raw cast — the autograd edge is wired below.
    Tensor out_typed = (output.dtype() == query.dtype())
                     ? output
                     : output.as_type(query.dtype());  // [B, Nq, T, hd]

    if (GradMode::is_enabled() &&
        (query.requires_grad() || key.requires_grad() || value.requires_grad() ||
         (has_qg && q_gamma.requires_grad()) || (has_kg && k_gamma.requires_grad())))
    {
        auto grad_fn = std::make_shared<GQAFusedFlashAttentionBackward>(
            packed.detach(), output.detach(), lse.detach(),
            q_rstd.detach(), k_rstd.detach(),
            has_qg ? q_gamma.detach() : Tensor(),
            has_kg ? k_gamma.detach() : Tensor(),
            cos_sin_cache.detach(),
            B, Nq_heads, Nkv_heads, T, hd,
            cache_seq_len, pos_offset, eps, interleaved, is_causal);

        Tensor& q_mut = const_cast<Tensor&>(query);
        Tensor& k_mut = const_cast<Tensor&>(key);
        Tensor& v_mut = const_cast<Tensor&>(value);
        if (query.requires_grad()) grad_fn->set_next_edge(0, get_grad_edge(q_mut));
        if (key.requires_grad())   grad_fn->set_next_edge(1, get_grad_edge(k_mut));
        if (value.requires_grad()) grad_fn->set_next_edge(2, get_grad_edge(v_mut));
        if (has_qg && q_gamma.requires_grad()) {
            Tensor& qg_mut = const_cast<Tensor&>(q_gamma);
            grad_fn->set_next_edge(3, get_grad_edge(qg_mut));
        }
        if (has_kg && k_gamma.requires_grad()) {
            Tensor& kg_mut = const_cast<Tensor&>(k_gamma);
            grad_fn->set_next_edge(4, get_grad_edge(kg_mut));
        }

        out_typed.set_grad_fn(grad_fn);
        out_typed.set_requires_grad(true);
    }
    return out_typed;
}
#endif

// ============================================================================
// Public Dispatch
// ============================================================================

Tensor scaled_dot_product_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    bool is_causal,float dropout_p,
    SDPBackend backend,
    float* qk_max_out)
{
    PROFILER_SCOPE("autograd::scaled_dot_product_attention");
    GraphRecordMode::record_forward("ATTENTION: scaled_dot_product_attention");
    TRACK_ALLOC_SCOPE_DEEP("op::autograd::scaled_dot_product_attention");
    switch (backend) {
        case SDPBackend::Math:
            // std::cout << "Using Math Backend" << std::endl;
            // NOTE: QK-Clip capture (qk_max_out) is only wired for the
            // MemoryEfficient backend; the Math path would need a separate
            // reduce-max over its materialized scores.
            return sdpa_math(query, key, value, is_causal);

        case SDPBackend::MemoryEfficient:
#ifdef WITH_CUDA
            // std::cout << "Using Memory-Efficient Attention Backend" << std::endl;
            return sdpa_memory_efficient(query, key, value, is_causal, qk_max_out);
#else
            throw std::runtime_error(
                "Memory-efficient attention requires CUDA. "
                "Falling back to Math backend.");
            //std::cout << "Using Math Backend   _____1 " << std::endl;
            return sdpa_math(query, key, value, is_causal);
#endif

        default:
            throw std::runtime_error("Unknown SDPBackend");
    }
}


Tensor scaled_dot_product_attention_packed(
    const Tensor& qkv,
    int64_t n_heads,
    bool is_causal,
    float /*dropout_p*/,
    SDPBackend backend,
    float* qk_max_out)
{
    PROFILER_SCOPE("autograd::scaled_dot_product_attention_packed");
    GraphRecordMode::record_forward("ATTENTION: scaled_dot_product_attention_packed");
    TRACK_ALLOC_SCOPE_DEEP("op::autograd::scaled_dot_product_attention_packed");

    switch (backend) {
        case SDPBackend::MemoryEfficient:
#ifdef WITH_CUDA
            return sdpa_memory_efficient_packed(qkv, n_heads, is_causal, qk_max_out);
#else
            throw std::runtime_error(
                "Packed memory-efficient attention requires CUDA.");
#endif
        case SDPBackend::Math:
            throw std::runtime_error(
                "scaled_dot_product_attention_packed: SDPBackend::Math not supported; "
                "use scaled_dot_product_attention instead.");
        default:
            throw std::runtime_error("Unknown SDPBackend");
    }
}

// high level API
Tensor scaled_dot_product_attention(
  const Tensor& qkv,
  const int Nq_heads,
  const int Nkv_heads,
  bool is_causal,
  float dropout_p,
  SDPBackend backend
){
  PROFILER_SCOPE("autograd::scaled_dot_product_attention");
  GraphRecordMode::record_forward("ATTENTION: scaled_dot_product_attention (GQA)");
  TRACK_ALLOC_SCOPE("autograd::scaled_dot_product_attention (GQA)");
  switch(backend){
    case SDPBackend::FlashAttention:
#ifdef WITH_CUDA
      return sdpa_flash_attention(qkv, Nq_heads, Nkv_heads, is_causal, dropout_p, backend);
#else
      throw std::runtime_error(
        "GQA scaled_dot_product_attention (FlashAttention) requires CUDA.");
#endif
    default:
      throw std::runtime_error(
        "As of now, you have to set backend to FlashAttention for using GQA."
      );
  }
}

// high level API (unpacked Q/K/V)
Tensor scaled_dot_product_attention(
  const Tensor& query,
  const Tensor& key,
  const Tensor& value,
  const int Nq_heads,
  const int Nkv_heads,
  bool is_causal,
  float dropout_p,
  SDPBackend backend,
  const cuda::GqaChatPlanPtr& chats
){
  PROFILER_SCOPE("autograd::scaled_dot_product_attention");
  GraphRecordMode::record_forward("ATTENTION: scaled_dot_product_attention (GQA, unpacked QKV)");
  TRACK_ALLOC_SCOPE("autograd::scaled_dot_product_attention (GQA, unpacked QKV)");
  switch(backend){
    case SDPBackend::FlashAttention:
#ifdef WITH_CUDA
      return sdpa_flash_attention_qkv(query, key, value, Nq_heads, Nkv_heads, is_causal, dropout_p, backend,
                                      chats);
#else
      throw std::runtime_error(
        "GQA scaled_dot_product_attention (FlashAttention) requires CUDA.");
#endif
    default:
      throw std::runtime_error(
        "As of now, you have to set backend to FlashAttention for using GQA."
      );
  }
}

// FUSED GQA (QK-norm + RoPE + causal GQA) high-level API.
Tensor scaled_dot_product_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& q_gamma,
    const Tensor& k_gamma,
    const Tensor& cos_sin_cache,
    int Nq_heads,
    int Nkv_heads,
    bool is_causal,
    bool interleaved,
    float eps,
    int pos_offset,
    SDPBackend backend)
{
    PROFILER_SCOPE("autograd::scaled_dot_product_attention (fused GQA)");
    GraphRecordMode::record_forward("ATTENTION: scaled_dot_product_attention (fused GQA)");
    TRACK_ALLOC_SCOPE("autograd::scaled_dot_product_attention (fused GQA)");
    switch (backend) {
        case SDPBackend::FlashAttention:
#ifdef WITH_CUDA
            return sdpa_gqa_fused(query, key, value, q_gamma, k_gamma, cos_sin_cache,
                                  Nq_heads, Nkv_heads, is_causal, interleaved, eps, pos_offset);
#else
            throw std::runtime_error(
                "Fused GQA scaled_dot_product_attention (FlashAttention) requires CUDA.");
#endif
        default:
            throw std::runtime_error(
                "Fused GQA scaled_dot_product_attention: only FlashAttention backend is supported.");
    }
}

// ============================================================================
// Fused tril_softmax + matmul (dedup attn_probs storage)
// ============================================================================

Tensor fused_attn_softmax_matmul(
    const Tensor& attn_weights,
    const Tensor& v,
    int64_t diagonal,
    double fill_value)
{
    PROFILER_SCOPE("autograd::fused_attn_softmax_matmul");
    GraphRecordMode::record_forward("ATTENTION: fused_attn_softmax_matmul");

    // Forward: two raw ops, no autograd nodes attached
    Tensor attn_probs;
    Tensor attn_out;
    {
        NoGradGuard no_grad;
        attn_probs = autograd::fused_tril_softmax(attn_weights, diagonal, fill_value);
        attn_out = OwnTensor::matmul(attn_probs, v);
    }

    // Wire up the single fused backward node
    if (GradMode::is_enabled() &&
        (attn_weights.requires_grad() || v.requires_grad()))
    {
        auto grad_fn = std::make_shared<FusedAttnSoftmaxMatmulBackward>(
            attn_probs.detach(), v.detach());

        if (attn_weights.requires_grad()) {
            grad_fn->set_next_edge(0, get_grad_edge(attn_weights));
        }
        if (v.requires_grad()) {
            grad_fn->set_next_edge(1, get_grad_edge(v));
        }

        attn_out.set_grad_fn(grad_fn);
        attn_out.set_requires_grad(true);
    }

    return attn_out;
}

} // namespace autograd
} // namespace OwnTensor