
// // #include "autograd/operations/AttentionOps.h"
// // #include "autograd/operations/MatrixOps.h"
// // #include "autograd/operations/ActivationOps.h"
// // #include "autograd/operations/ReshapeOps.h"
// // #include "autograd/operations/BinaryOps.h"
// // #include "autograd/ops_template.h"
// // #include "checkpointing/GradMode.h"
// // #include "utils/Profiler.h"

// // #ifdef WITH_CUDA
// // #include "ops/helpers/AttentionKernels.h"
// // #include "autograd/backward/AttentionBackward.h"
// // #endif

// // #include <cmath>
// // #include <limits>
// // #include <stdexcept>

// // namespace OwnTensor {
// // namespace autograd {

// // // ============================================================================
// // // Math Backend: composes existing autograd ops
// // // ============================================================================

// // static Tensor sdpa_math(
// //     const Tensor& query,
// //     const Tensor& key,
// //     const Tensor& value,
// //     bool is_causal)
// // {
// //     // std::cout<<"hi"<<std::endl;
// //     // query, key, value: (B, nh, T, hd)
// //     if(query.device() != key.device() && query.device()!= value.device())
// //     {
    
// //         throw std::runtime_error(
// //             "Device Mismatch, Q, K, V do not reside in the same device");
    
// //     }
// //     int64_t hd = query.shape().dims[3];
// //     float scale = 1.0f / std::sqrt(static_cast<float>(hd));
// //     // std::cout<<scale<<std::endl;
// //     // Scale query
// //     Tensor scale_t = Tensor::full(Shape{{1}},
// //         TensorOptions().with_dtype(query.dtype()).with_device(query.device()),
// //         scale);
// //     Tensor q_scaled = autograd::mul(query, scale_t);

// //     // Q @ K^T -> (B, nh, T, T)
// //     Tensor k_t = autograd::transpose(key, -2, -1);
// //     Tensor attn_weights = autograd::matmul(q_scaled, k_t);

// //     // Apply causal mask + softmax
// //     Tensor attn_probs;
// //     if (is_causal) {
// //         float neg_inf = -std::numeric_limits<float>::infinity();
// //         attn_probs = autograd::fused_tril_softmax(attn_weights, 0, neg_inf);
// //     } else {
// //         attn_probs = autograd::softmax(attn_weights, -1);
// //     }

// //     // Attn @ V -> (B, nh, T, hd)
// //     return autograd::matmul(attn_probs, value);
// // }

// // // ============================================================================
// // // Memory-Efficient Backend (requires CUDA)
// // // ============================================================================

// // #ifdef WITH_CUDA
// // static Tensor sdpa_memory_efficient(
// //     const Tensor& query,
// //     const Tensor& key,
// //     const Tensor& value,
// //     bool is_causal)
// // {
// //     if(query.device() != key.device() && query.device()!= value.device())
// //     {
// //         throw std::runtime_error("Device Mismatch, Q, K, V do not reside in the same device");
// //     }
// //     if (!query.device().is_cuda() || query.dtype() != Dtype::Float32) {
// //         throw std::runtime_error(
// //             "Memory-efficient attention requires CUDA float32 tensors");
// //     }

// //     int64_t B  = query.shape().dims[0];
// //     int64_t nh = query.shape().dims[1];
// //     int64_t T  = query.shape().dims[2];
// //     int64_t hd = query.shape().dims[3];

// //     //* print the dimensions
// //     // printf("The dimensions of Q, K, V, O are: {%d, %d, %d, %d}\n", B, nh, T, hd);
// //     // exit(1);
// //     //* Kernel requires contiguous (B*nh, T, hd) layout — make contiguous if needed
// //     // Tensor q_contig = query.is_contiguous() ? query : query.contiguous();
// //     // Tensor k_contig = key.is_contiguous()   ? key   : key.contiguous();
// //     // Tensor v_contig = value.is_contiguous() ? value : value.contiguous();

// //     auto opts = TensorOptions().with_dtype(Dtype::Float32).with_device(query.device());

// //     // Allocate output and LSE — NO T×T allocation anywhere
// //     Tensor output = Tensor::empty(Shape{{B, nh, T, hd}}, opts);
// //     Tensor lse    = Tensor::empty(Shape{{B, nh, T}}, opts);

// //     // Single fused kernel: Q @ K^T → scale → mask → online softmax → @ V
// //     // Never materializes the T×T attention matrix.
// //     cuda::mem_efficient_attn_forward(
// //         query.data<float>(), key.data<float>(), value.data<float>(),
// //         output.data<float>(), lse.data<float>(),
// //         B, nh, T, hd, is_causal);

// //     // Build autograd graph with memory-efficient backward
// //     if (GradMode::is_enabled() &&
// //         (query.requires_grad() || key.requires_grad() || value.requires_grad()))
// //     {
// //         // std::cout<<"inside grad mode"<<std::endl;
// //         auto grad_fn = std::make_shared<MemEfficientAttentionBackward>(
// //             query.detach(), key.detach(), value.detach(),
// //             output.detach(), lse.detach(),
// //             B, nh, T, hd, is_causal);

// //         Tensor& q_mut = const_cast<Tensor&>(query);
// //         Tensor& k_mut = const_cast<Tensor&>(key);
// //         Tensor& v_mut = const_cast<Tensor&>(value);

// //         if (query.requires_grad()) {
// //             grad_fn->set_next_edge(0, get_grad_edge(q_mut));
// //         }
// //         if (key.requires_grad()) {
// //             grad_fn->set_next_edge(1, get_grad_edge(k_mut));
// //         }
// //         if (value.requires_grad()) {
// //             grad_fn->set_next_edge(2, get_grad_edge(v_mut));
// //         }

// //         output.set_grad_fn(grad_fn);
// //         output.set_requires_grad(true);
// //     }

// //     return output;
// // }
// // #endif

// // // ============================================================================
// // // Public Dispatch
// // // ============================================================================

// // Tensor scaled_dot_product_attention(
// //     const Tensor& query,
// //     const Tensor& key,
// //     const Tensor& value,
// //     bool is_causal,
// //     SDPBackend backend)
// // {
// //     GraphRecordMode::record_forward("ATTENTION: scaled_dot_product_attention");

// //     switch (backend) {
// //         case SDPBackend::Math:
// //             // std::cout << "Using Math Backend" << std::endl;
// //             return sdpa_math(query, key, value, is_causal);

// //         case SDPBackend::MemoryEfficient:
// // #ifdef WITH_CUDA
// //             // std::cout << "Using Memory-Efficient Attention Backend" << std::endl;
// //             return sdpa_memory_efficient(query, key, value, is_causal);
// // #else
// //             throw std::runtime_error(
// //                 "Memory-efficient attention requires CUDA. "
// //                 "Falling back to Math backend.");
// //             //std::cout << "Using Math Backend   _____1 " << std::endl;
// //             return sdpa_math(query, key, value, is_causal);
// // #endif

// //         default:
// //             throw std::runtime_error("Unknown SDPBackend");
// //     }
// // }

// // } // namespace autograd
// // } // namespace OwnTensor

// #include "autograd/operations/AttentionOps.h"
// #include "autograd/operations/MatrixOps.h"
// #include "autograd/operations/ActivationOps.h"
// #include "autograd/operations/ReshapeOps.h"
// #include "autograd/operations/BinaryOps.h"
// #include "autograd/ops_template.h"
// #include "checkpointing/GradMode.h"
// #include "utils/Profiler.h"

// #ifdef WITH_CUDA
// #include "ops/helpers/AttentionKernels.h"
// #include "autograd/backward/AttentionBackward.h"
// #endif
// #include "ops/Kernels.h"
// #include "autograd/backward/FusedAttnSoftmaxMatmulBackward.h"
// #include "device/AllocationTracker.h"

// #include <cmath>
// #include <limits>
// #include <stdexcept>

// namespace OwnTensor {
// namespace autograd {

// // ============================================================================
// // Math Backend: composes existing autograd ops
// // ============================================================================

// static Tensor sdpa_math(
//     const Tensor& query,
//     const Tensor& key,
//     const Tensor& value,
//     bool is_causal)
// {
//     // std::cout<<"hi"<<std::endl;
//     // query, key, value: (B, nh, T, hd)
//     if(query.device() != key.device() && query.device()!= value.device())
//     {
    
//         throw std::runtime_error(
//             "Device Mismatch, Q, K, V do not reside in the same device");
    
//     }
//     int64_t hd = query.shape().dims[3];
//     float scale = 1.0f / std::sqrt(static_cast<float>(hd));
//     // std::cout<<scale<<std::endl;
//     // Scale query
//     Tensor scale_t = Tensor::full(Shape{{1}},
//         TensorOptions().with_dtype(query.dtype()).with_device(query.device()),
//         scale);
//     Tensor q_scaled = autograd::mul(query, scale_t);

//     // Q @ K^T -> (B, nh, T, T)
//     Tensor k_t = autograd::transpose(key, -2, -1);
//     Tensor attn_weights = autograd::matmul(q_scaled, k_t);

//     // Apply causal mask + softmax
//     Tensor attn_probs;
//     if (is_causal) {
//         float neg_inf = -std::numeric_limits<float>::infinity();
//         attn_probs = autograd::fused_tril_softmax(attn_weights, 0, neg_inf);
//     } else {
//         attn_probs = autograd::softmax(attn_weights, -1);
//     }

//     // Attn @ V -> (B, nh, T, hd)
//     return autograd::matmul(attn_probs, value);
// }

// // ============================================================================
// // Memory-Efficient Backend (requires CUDA)
// // ============================================================================

// #ifdef WITH_CUDA
// static Tensor sdpa_memory_efficient(
//     const Tensor& query,
//     const Tensor& key,
//     const Tensor& value,
//     bool is_causal)
// {
//     if(query.device() != key.device() && query.device()!= value.device())
//     {
//         throw std::runtime_error("Device Mismatch, Q, K, V do not reside in the same device");
//     }
//     if (!query.device().is_cuda() || query.dtype() != Dtype::Float32) {
//         throw std::runtime_error(
//             "Memory-efficient attention requires CUDA float32 tensors");
//     }

//     int64_t B  = query.shape().dims[0];
//     int64_t nh = query.shape().dims[1];
//     int64_t T  = query.shape().dims[2];
//     int64_t hd = query.shape().dims[3];

//     // Stride-aware path (matches PyTorch's _efficient_attention_forward):
//     // kernel reads Q/K/V with per-tensor strides, so no .contiguous() copy
//     // is needed for transposed views of a [B, T, H, HD] tensor. The only
//     // requirement is that the last dim (HeadDim) is contiguous.
//     const auto& q_strides = query.stride().strides;
//     const auto& k_strides = key.stride().strides;
//     const auto& v_strides = value.stride().strides;

//     auto opts = TensorOptions().with_dtype(Dtype::Float32).with_device(query.device());

//     // Allocate output and LSE as contiguous [B, nh, T, hd] / [B, nh, T]
//     Tensor output = Tensor::empty(Shape{{B, nh, T, hd}}, opts);
//     Tensor lse    = Tensor::empty(Shape{{B, nh, T}}, opts);
//     const auto& o_strides = output.stride().strides;
//     const auto& lse_strides = lse.stride().strides;

//     // Single fused kernel: Q @ K^T → scale → mask → online softmax → @ V
//     // Never materializes the T×T attention matrix.
//     cuda::mem_efficient_attn_forward_tc(
//         query.data<float>(), q_strides[0], q_strides[2], q_strides[1],
//         key.data<float>(),   k_strides[0], k_strides[2], k_strides[1],
//         value.data<float>(), v_strides[0], v_strides[2], v_strides[1],
//         output.data<float>(), o_strides[0], o_strides[2], o_strides[1],
//         lse.data<float>(),    lse_strides[0], lse_strides[1],
//         B, nh, T, hd, is_causal, 0.0f /*dropout_p*/, nullptr /*dropout_mask*/);

//     // Build autograd graph with memory-efficient backward
//     if (GradMode::is_enabled() &&
//         (query.requires_grad() || key.requires_grad() || value.requires_grad()))
//     {
//         // Save the ORIGINAL (possibly strided) views — backward also handles strides.
//         auto grad_fn = std::make_shared<MemEfficientAttentionBackward>(
//             query.detach(), key.detach(), value.detach(),
//             output.detach(), lse.detach(),
//             B, nh, T, hd, is_causal);

//         Tensor& q_mut = const_cast<Tensor&>(query);
//         Tensor& k_mut = const_cast<Tensor&>(key);
//         Tensor& v_mut = const_cast<Tensor&>(value);

//         if (query.requires_grad()) {
//             grad_fn->set_next_edge(0, get_grad_edge(q_mut));
//         }
//         if (key.requires_grad()) {
//             grad_fn->set_next_edge(1, get_grad_edge(k_mut));
//         }
//         if (value.requires_grad()) {
//             grad_fn->set_next_edge(2, get_grad_edge(v_mut));
//         }

//         output.set_grad_fn(grad_fn);
//         output.set_requires_grad(true);
//     }

//     return output;
// }
// #endif

// // ============================================================================
// // Memory-Efficient Backend, packed-QKV variant (requires CUDA)
// //
// // Reads Q/K/V directly from a packed [B, T, 3*C] tensor via strided pointer
// // arithmetic, and writes a packed [B, T, C] output. Together with
// // PackedSDPABackward, eliminates:
// //   - Make_shards_inplace_axis + its backward Tensor::cat (the 1.5s/10steps
// //     `cat_batched_kernel` traffic, or its sync-cudaMemcpy equivalent on R1).
// //   - The four .contiguous() copies on each side of the SDPA kernel pair
// //     (q/k/v reshape+transpose in fwd, transpose+reshape of merged in fwd,
// //     and their bwd mirrors) — the bulk of the `generic_strided_copy_kernel`
// //     budget that is currently attention-driven.
// // ============================================================================

// #ifdef WITH_CUDA
// static Tensor sdpa_memory_efficient_packed(
//     const Tensor& qkv,
//     int64_t n_heads,
//     bool is_causal)
// {
//     if (!qkv.device().is_cuda() || qkv.dtype() != Dtype::Float32) {
//         throw std::runtime_error(
//             "scaled_dot_product_attention_packed: only CUDA float32 tensors are supported");
//     }
//     if (qkv.shape().dims.size() != 3) {
//         throw std::runtime_error(
//             "scaled_dot_product_attention_packed: qkv must be a 3-D tensor [B, T, 3*C]");
//     }

//     const int64_t B  = qkv.shape().dims[0];
//     const int64_t T  = qkv.shape().dims[1];
//     const int64_t C3 = qkv.shape().dims[2];
//     if (C3 % 3 != 0) {
//         throw std::runtime_error(
//             "scaled_dot_product_attention_packed: qkv last dim must be divisible by 3");
//     }
//     const int64_t C  = C3 / 3;
//     if (n_heads <= 0 || C % n_heads != 0) {
//         throw std::runtime_error(
//             "scaled_dot_product_attention_packed: n_embd must be divisible by n_heads");
//     }
//     const int64_t nh = n_heads;
//     const int64_t hd = C / nh;

//     auto opts = TensorOptions().with_dtype(Dtype::Float32).with_device(qkv.device());

//     // Output is contiguous [B, T, C]; viewed as [B, nh, T, hd] for the kernel.
//     Tensor output = Tensor::empty(Shape{{B, T, C}}, opts);
//     Tensor lse    = Tensor::empty(Shape{{B, nh, T}}, opts);

//     // Strides for Q/K/V views into the packed qkv treated as [B, nh, T, hd]:
//     //   strideB = T * 3 * C
//     //   strideM = 3 * C  (Q, K, V interleaved across the time axis)
//     //   strideH = hd
//     //   last (hd) stride = 1 (kernel requirement)
//     const int64_t qkv_strideB = T * 3 * C;
//     const int64_t qkv_strideM = 3 * C;
//     const int64_t qkv_strideH = hd;

//     // Output is packed [B, T, C] (head_id ⊗ head_dim laid out contiguously
//     // along the inner axis), viewed as [B, nh, T, hd]:
//     //   strideB = T * C, strideM = C, strideH = hd, last = 1.
//     const int64_t o_strideB = T * C;
//     const int64_t o_strideM = C;
//     const int64_t o_strideH = hd;

//     // LSE [B, nh, T] contiguous: strideB = nh*T, strideH = T.
//     const int64_t lse_strideB = nh * T;
//     const int64_t lse_strideH = T;

//     const float* qkv_ptr = qkv.data<float>();
//     cuda::mem_efficient_attn_forward_tc(
//         /*Q*/ qkv_ptr + 0,        qkv_strideB, qkv_strideM, qkv_strideH,
//         /*K*/ qkv_ptr + C,        qkv_strideB, qkv_strideM, qkv_strideH,
//         /*V*/ qkv_ptr + 2 * C,    qkv_strideB, qkv_strideM, qkv_strideH,
//         output.data<float>(),  o_strideB, o_strideM, o_strideH,
//         lse.data<float>(),     lse_strideB, lse_strideH,
//         B, nh, T, hd, is_causal, 0.0f /*dropout_p*/, nullptr /*dropout_mask*/);

//     if (GradMode::is_enabled() && qkv.requires_grad()) {
//         auto grad_fn = std::make_shared<PackedSDPABackward>(
//             qkv.detach(), output.detach(), lse.detach(),
//             B, nh, T, hd, is_causal);

//         Tensor& qkv_mut = const_cast<Tensor&>(qkv);
//         grad_fn->set_next_edge(0, get_grad_edge(qkv_mut));

//         output.set_grad_fn(grad_fn);
//         output.set_requires_grad(true);
//     }

//     return output;
// }
// #endif

// // ============================================================================
// // Public Dispatch
// // ============================================================================

// Tensor scaled_dot_product_attention(
//     const Tensor& query,
//     const Tensor& key,
//     const Tensor& value,
//     bool is_causal,float dropout_p,
//     SDPBackend backend)
// {
//     GraphRecordMode::record_forward("ATTENTION: scaled_dot_product_attention");
//     TRACK_ALLOC_SCOPE("L157:autograd::scaled_dot_product_attention");
//     switch (backend) {
//         case SDPBackend::Math:
//             // std::cout << "Using Math Backend" << std::endl;
//             return sdpa_math(query, key, value, is_causal);

//         case SDPBackend::MemoryEfficient:
// #ifdef WITH_CUDA
//             // std::cout << "Using Memory-Efficient Attention Backend" << std::endl;
//             return sdpa_memory_efficient(query, key, value, is_causal);
// #else
//             throw std::runtime_error(
//                 "Memory-efficient attention requires CUDA. "
//                 "Falling back to Math backend.");
//             //std::cout << "Using Math Backend   _____1 " << std::endl;
//             return sdpa_math(query, key, value, is_causal);
// #endif

//         default:
//             throw std::runtime_error("Unknown SDPBackend");
//     }
// }


// Tensor scaled_dot_product_attention_packed(
//     const Tensor& qkv,
//     int64_t n_heads,
//     bool is_causal,
//     float /*dropout_p*/,
//     SDPBackend backend)
// {
//     GraphRecordMode::record_forward("ATTENTION: scaled_dot_product_attention_packed");
//     TRACK_ALLOC_SCOPE("L158:autograd::scaled_dot_product_attention_packed");

//     switch (backend) {
//         case SDPBackend::MemoryEfficient:
// #ifdef WITH_CUDA
//             return sdpa_memory_efficient_packed(qkv, n_heads, is_causal);
// #else
//             throw std::runtime_error(
//                 "Packed memory-efficient attention requires CUDA.");
// #endif
//         case SDPBackend::Math:
//             // Math backend does not have a packed fast-path; fall back to the
//             // unfused [shard → reshape → transpose → SDPA] route. Kept as a
//             // correctness reference, not a perf path.
//             throw std::runtime_error(
//                 "scaled_dot_product_attention_packed: SDPBackend::Math is not "
//                 "supported; use the unfused scaled_dot_product_attention instead.");
//         default:
//             throw std::runtime_error("Unknown SDPBackend");
//     }
// }


// // ============================================================================
// // Fused tril_softmax + matmul (dedup attn_probs storage)
// // ============================================================================

// Tensor fused_attn_softmax_matmul(
//     const Tensor& attn_weights,
//     const Tensor& v,
//     int64_t diagonal,
//     double fill_value)
// {
//     GraphRecordMode::record_forward("ATTENTION: fused_attn_softmax_matmul");

//     // Forward: two raw ops, no autograd nodes attached
//     Tensor attn_probs;
//     Tensor attn_out;
//     {
//         NoGradGuard no_grad;
//         attn_probs = autograd::fused_tril_softmax(attn_weights, diagonal, fill_value);
//         attn_out = OwnTensor::matmul(attn_probs, v);
//     }

//     // Wire up the single fused backward node
//     if (GradMode::is_enabled() &&
//         (attn_weights.requires_grad() || v.requires_grad()))
//     {
//         auto grad_fn = std::make_shared<FusedAttnSoftmaxMatmulBackward>(
//             attn_probs.detach(), v.detach());

//         if (attn_weights.requires_grad()) {
//             grad_fn->set_next_edge(0, get_grad_edge(attn_weights));
//         }
//         if (v.requires_grad()) {
//             grad_fn->set_next_edge(1, get_grad_edge(v));
//         }

//         attn_out.set_grad_fn(grad_fn);
//         attn_out.set_requires_grad(true);
//     }

//     return attn_out;
// }

// } // namespace autograd
// } // namespace OwnTensor




























// #include "autograd/operations/AttentionOps.h"
// #include "autograd/operations/MatrixOps.h"
// #include "autograd/operations/ActivationOps.h"
// #include "autograd/operations/ReshapeOps.h"
// #include "autograd/operations/BinaryOps.h"
// #include "autograd/ops_template.h"
// #include "checkpointing/GradMode.h"
// #include "utils/Profiler.h"

// #ifdef WITH_CUDA
// #include "ops/helpers/AttentionKernels.h"
// #include "autograd/backward/AttentionBackward.h"
// #endif

// #include <cmath>
// #include <limits>
// #include <stdexcept>

// namespace OwnTensor {
// namespace autograd {

// // ============================================================================
// // Math Backend: composes existing autograd ops
// // ============================================================================

// static Tensor sdpa_math(
//     const Tensor& query,
//     const Tensor& key,
//     const Tensor& value,
//     bool is_causal)
// {
//     // std::cout<<"hi"<<std::endl;
//     // query, key, value: (B, nh, T, hd)
//     if(query.device() != key.device() && query.device()!= value.device())
//     {
    
//         throw std::runtime_error(
//             "Device Mismatch, Q, K, V do not reside in the same device");
    
//     }
//     int64_t hd = query.shape().dims[3];
//     float scale = 1.0f / std::sqrt(static_cast<float>(hd));
//     // std::cout<<scale<<std::endl;
//     // Scale query
//     Tensor scale_t = Tensor::full(Shape{{1}},
//         TensorOptions().with_dtype(query.dtype()).with_device(query.device()),
//         scale);
//     Tensor q_scaled = autograd::mul(query, scale_t);

//     // Q @ K^T -> (B, nh, T, T)
//     Tensor k_t = autograd::transpose(key, -2, -1);
//     Tensor attn_weights = autograd::matmul(q_scaled, k_t);

//     // Apply causal mask + softmax
//     Tensor attn_probs;
//     if (is_causal) {
//         float neg_inf = -std::numeric_limits<float>::infinity();
//         attn_probs = autograd::fused_tril_softmax(attn_weights, 0, neg_inf);
//     } else {
//         attn_probs = autograd::softmax(attn_weights, -1);
//     }

//     // Attn @ V -> (B, nh, T, hd)
//     return autograd::matmul(attn_probs, value);
// }

// // ============================================================================
// // Memory-Efficient Backend (requires CUDA)
// // ============================================================================

// #ifdef WITH_CUDA
// static Tensor sdpa_memory_efficient(
//     const Tensor& query,
//     const Tensor& key,
//     const Tensor& value,
//     bool is_causal)
// {
//     if(query.device() != key.device() && query.device()!= value.device())
//     {
//         throw std::runtime_error("Device Mismatch, Q, K, V do not reside in the same device");
//     }
//     if (!query.device().is_cuda() || query.dtype() != Dtype::Float32) {
//         throw std::runtime_error(
//             "Memory-efficient attention requires CUDA float32 tensors");
//     }

//     int64_t B  = query.shape().dims[0];
//     int64_t nh = query.shape().dims[1];
//     int64_t T  = query.shape().dims[2];
//     int64_t hd = query.shape().dims[3];

//     //* print the dimensions
//     // printf("The dimensions of Q, K, V, O are: {%d, %d, %d, %d}\n", B, nh, T, hd);
//     // exit(1);
//     //* Kernel requires contiguous (B*nh, T, hd) layout — make contiguous if needed
//     // Tensor q_contig = query.is_contiguous() ? query : query.contiguous();
//     // Tensor k_contig = key.is_contiguous()   ? key   : key.contiguous();
//     // Tensor v_contig = value.is_contiguous() ? value : value.contiguous();

//     auto opts = TensorOptions().with_dtype(Dtype::Float32).with_device(query.device());

//     // Allocate output and LSE — NO T×T allocation anywhere
//     Tensor output = Tensor::empty(Shape{{B, nh, T, hd}}, opts);
//     Tensor lse    = Tensor::empty(Shape{{B, nh, T}}, opts);

//     // Single fused kernel: Q @ K^T → scale → mask → online softmax → @ V
//     // Never materializes the T×T attention matrix.
//     cuda::mem_efficient_attn_forward(
//         query.data<float>(), key.data<float>(), value.data<float>(),
//         output.data<float>(), lse.data<float>(),
//         B, nh, T, hd, is_causal);

//     // Build autograd graph with memory-efficient backward
//     if (GradMode::is_enabled() &&
//         (query.requires_grad() || key.requires_grad() || value.requires_grad()))
//     {
//         // std::cout<<"inside grad mode"<<std::endl;
//         auto grad_fn = std::make_shared<MemEfficientAttentionBackward>(
//             query.detach(), key.detach(), value.detach(),
//             output.detach(), lse.detach(),
//             B, nh, T, hd, is_causal);

//         Tensor& q_mut = const_cast<Tensor&>(query);
//         Tensor& k_mut = const_cast<Tensor&>(key);
//         Tensor& v_mut = const_cast<Tensor&>(value);

//         if (query.requires_grad()) {
//             grad_fn->set_next_edge(0, get_grad_edge(q_mut));
//         }
//         if (key.requires_grad()) {
//             grad_fn->set_next_edge(1, get_grad_edge(k_mut));
//         }
//         if (value.requires_grad()) {
//             grad_fn->set_next_edge(2, get_grad_edge(v_mut));
//         }

//         output.set_grad_fn(grad_fn);
//         output.set_requires_grad(true);
//     }

//     return output;
// }
// #endif

// // ============================================================================
// // Public Dispatch
// // ============================================================================

// Tensor scaled_dot_product_attention(
//     const Tensor& query,
//     const Tensor& key,
//     const Tensor& value,
//     bool is_causal,
//     SDPBackend backend)
// {
//     GraphRecordMode::record_forward("ATTENTION: scaled_dot_product_attention");

//     switch (backend) {
//         case SDPBackend::Math:
//             // std::cout << "Using Math Backend" << std::endl;
//             return sdpa_math(query, key, value, is_causal);

//         case SDPBackend::MemoryEfficient:
// #ifdef WITH_CUDA
//             // std::cout << "Using Memory-Efficient Attention Backend" << std::endl;
//             return sdpa_memory_efficient(query, key, value, is_causal);
// #else
//             throw std::runtime_error(
//                 "Memory-efficient attention requires CUDA. "
//                 "Falling back to Math backend.");
//             //std::cout << "Using Math Backend   _____1 " << std::endl;
//             return sdpa_math(query, key, value, is_causal);
// #endif

//         default:
//             throw std::runtime_error("Unknown SDPBackend");
//     }
// }

// } // namespace autograd
// } // namespace OwnTensor

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
#include "device/AllocationTracker.h"

#include <cmath>
#include <limits>
#include <stdexcept>

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
    bool is_causal)
{
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
        B, nh, T, hd, is_causal, 0.0f /*dropout_p*/, nullptr /*dropout_mask*/);

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
    bool is_causal)
{
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
        B, nh, T, hd, is_causal, 0.0f, nullptr);

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

// ============================================================================
// Public Dispatch
// ============================================================================

Tensor scaled_dot_product_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    bool is_causal,float dropout_p,
    SDPBackend backend)
{
    GraphRecordMode::record_forward("ATTENTION: scaled_dot_product_attention");
    TRACK_ALLOC_SCOPE("L157:autograd::scaled_dot_product_attention");
    switch (backend) {
        case SDPBackend::Math:
            // std::cout << "Using Math Backend" << std::endl;
            return sdpa_math(query, key, value, is_causal);

        case SDPBackend::MemoryEfficient:
#ifdef WITH_CUDA
            // std::cout << "Using Memory-Efficient Attention Backend" << std::endl;
            return sdpa_memory_efficient(query, key, value, is_causal);
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
    SDPBackend backend)
{
    GraphRecordMode::record_forward("ATTENTION: scaled_dot_product_attention_packed");
    TRACK_ALLOC_SCOPE("L158:autograd::scaled_dot_product_attention_packed");

    switch (backend) {
        case SDPBackend::MemoryEfficient:
#ifdef WITH_CUDA
            return sdpa_memory_efficient_packed(qkv, n_heads, is_causal);
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

// ============================================================================
// Fused tril_softmax + matmul (dedup attn_probs storage)
// ============================================================================

Tensor fused_attn_softmax_matmul(
    const Tensor& attn_weights,
    const Tensor& v,
    int64_t diagonal,
    double fill_value)
{
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