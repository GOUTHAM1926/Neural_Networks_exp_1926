#include "autograd/operations/FlashAttentionOps.h"
#include "autograd/backward/FlashAttentionBackward.h"
#include "autograd/Node.h"
#include "autograd/ops_template.h"
#include "checkpointing/GradMode.h"
#include "core/Tensor.h"
#include <cmath>
#include <stdexcept>
#include <vector>
#include <algorithm>

namespace OwnTensor {
namespace autograd {

// ── Forward declaration ───────────────────────────────────────────────────────
void launch_flash_attn(
    const float* Q, const float* K, const float* V, float* O, float* L,
    int B, int H, int N, int d, bool causal);

// Kernel supports a fixed set of head dimensions (see switch in FlashAttention.cu).
static bool is_supported_head_dim(int d) {
    switch (d) {
        case 2: case 4: case 8: case 16: case 24: case 32:
        case 40: case 48: case 56: case 64: case 80: case 96:
        case 128: case 160: case 192: case 256:
            return true;
        default:
            return false;
    }
}

// ── CPU reference ─────────────────────────────────────────────────────────────
// Naive O(T²·d) attention used when Q/K/V live on CPU.
// Applies a causal mask when causal=true (each query attends only to j <= i).
// ─────────────────────────────────────────────────────────────────────────────
static void cpu_attention_forward(
    const float* Q, const float* K, const float* V, float* O,
    int64_t BH, int64_t T, int64_t d, float scale, bool causal)
{
    std::vector<float> row(T);
    for (int64_t bh = 0; bh < BH; ++bh) {
        const float* q = Q + bh * T * d;
        const float* k = K + bh * T * d;
        const float* v = V + bh * T * d;
        float*       o = O + bh * T * d;

        for (int64_t i = 0; i < T; ++i) {
            const int64_t kv_end = causal ? i + 1 : T;

            float m = -1e30f;
            for (int64_t j = 0; j < kv_end; ++j) {
                float s = 0.0f;
                for (int64_t dd = 0; dd < d; ++dd)
                    s += q[i * d + dd] * k[j * d + dd];
                row[j] = s * scale;
                m = std::max(m, row[j]);
            }

            float sum = 0.0f;
            for (int64_t j = 0; j < kv_end; ++j) {
                row[j] = std::exp(row[j] - m);
                sum += row[j];
            }
            for (int64_t j = 0; j < kv_end; ++j) row[j] /= sum;

            for (int64_t dd = 0; dd < d; ++dd) {
                float acc = 0.0f;
                for (int64_t j = 0; j < kv_end; ++j)
                    acc += row[j] * v[j * d + dd];
                o[i * d + dd] = acc;
            }
        }
    }
}

// ── Public API ────────────────────────────────────────────────────────────────
// flash_attention
//
//   High-level dispatch: routes to the CUDA Flash Attention kernel (GPU) or
//   the naive CPU reference depending on the device of Q.
//
//   Q, K, V  : 3-D [BH, T, d]  where BH = B * n_heads
//   causal   : apply causal mask (GPU path always applies it; CPU path respects flag)
//   Returns  : O  [BH, T, d]
// ─────────────────────────────────────────────────────────────────────────────
Tensor flash_attention(const Tensor& Q, const Tensor& K, const Tensor& V,
                       bool causal)
{
    if (Q.ndim() != 3 || K.ndim() != 3 || V.ndim() != 3)
        throw std::runtime_error("flash_attention: Q, K, V must be 3-D [BH, T, d]");

    const int64_t BH    = Q.shape().dims[0];
    const int64_t T     = Q.shape().dims[1];
    const int64_t d     = Q.shape().dims[2];

    if (Q.device().is_cuda()) {
        // ── GPU path ─────────────────────────────────────────────────────────
        if (!is_supported_head_dim(static_cast<int>(d)))
            throw std::runtime_error(
                "flash_attention (CUDA): unsupported head_dim " + std::to_string(d) +
                " — must be one of {2,4,8,16,24,32,40,48,56,64,80,96,128,160,192,256}.");

        TensorOptions opts = TensorOptions().with_dtype(Dtype::Float32)
                                            .with_device(Q.device());

        const bool need_grad = GradMode::is_enabled() &&
            (Q.requires_grad() || K.requires_grad() || V.requires_grad());

        // Allocate output O [BH, T, d] and (if training) L [BH, T]
        Tensor O(Q.shape(), opts);
        Tensor L;
        float* L_ptr = nullptr;
        if (need_grad) {
            L = Tensor(Shape({{BH, T}}), opts);
            L_ptr = L.data<float>();
        }

        launch_flash_attn(
            Q.data<float>(), K.data<float>(), V.data<float>(),
            O.data<float>(), L_ptr,
            static_cast<int>(BH), /*H=*/1,
            static_cast<int>(T),  static_cast<int>(d), causal);

        // ── Wire autograd ────────────────────────────────────────────────────
        if (need_grad) {
            const float scale = 1.0f / std::sqrt(static_cast<float>(d));

            // O.detach() breaks the reference cycle:
            //   grad_fn → saved_O_ → impl_ → grad_fn_ → grad_fn  (cycle!)
            // Detach creates a new impl_ sharing storage but without grad_fn.
            auto grad_fn = std::make_shared<FlashAttentionBackward>(
                Q, K, V, O.detach(), L,
                static_cast<int>(BH), /*n_heads=*/1,
                static_cast<int>(T), static_cast<int>(d), scale);

            // Connect edges to inputs: Q=0, K=1, V=2
            Tensor& q_mut = const_cast<Tensor&>(Q);
            Tensor& k_mut = const_cast<Tensor&>(K);
            Tensor& v_mut = const_cast<Tensor&>(V);

            if (Q.requires_grad()) grad_fn->set_next_edge(0, get_grad_edge(q_mut));
            if (K.requires_grad()) grad_fn->set_next_edge(1, get_grad_edge(k_mut));
            if (V.requires_grad()) grad_fn->set_next_edge(2, get_grad_edge(v_mut));

            O.set_grad_fn(grad_fn);
        }

        return O;
    } else {
        // ── CPU path ─────────────────────────────────────────────────────────
        const float scale = 1.0f / std::sqrt(static_cast<float>(d));
        Tensor O(Q.shape(),
                 TensorOptions().with_dtype(Dtype::Float32).with_device(Q.device()));
        cpu_attention_forward(
            Q.data<float>(), K.data<float>(), V.data<float>(),
            O.data<float>(),
            BH, T, d, scale, causal);
        return O;
    }
}

} // namespace autograd
} // namespace OwnTensor