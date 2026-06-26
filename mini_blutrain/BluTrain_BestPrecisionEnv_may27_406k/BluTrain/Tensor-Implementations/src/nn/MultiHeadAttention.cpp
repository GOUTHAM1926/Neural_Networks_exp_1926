
// #include "nn/MultiHeadAttention.h"
// #include "autograd/AutogradOps.h"
// #include <stdexcept>

// namespace OwnTensor {
// namespace nn {

// MultiHeadAttention::MultiHeadAttention(
//     int embed_dim, int num_heads,
//     bool is_causal,
//     autograd::SDPBackend backend,
//     bool bias)
//     : embed_dim(embed_dim),
//       num_heads(num_heads),
//       head_dim(embed_dim / num_heads),
//       is_causal(is_causal),
//       backend(backend),
//       qkv_proj(embed_dim, 3 * embed_dim, bias),
//       out_proj(embed_dim, embed_dim, bias)
// {
//     if (embed_dim % num_heads != 0) {
//         throw std::runtime_error(
//             "MultiHeadAttention: embed_dim must be divisible by num_heads");
//     }

//     register_module(qkv_proj);
//     register_module(out_proj);
// }

// Tensor MultiHeadAttention::forward(const Tensor& x) {
//     // x: (B, T, C)
//     int64_t B = x.shape().dims[0];
//     int64_t T = x.shape().dims[1];
//     int64_t C = x.shape().dims[2];

//     // 1. QKV projection: (B, T, C) -> (B, T, 3C)
//     Tensor qkv = qkv_proj.forward(x);

//     // 2. Split into q, k, v each (B, T, C) via sharding along last dim
//     std::vector<Tensor> splits = qkv.make_shards_inplace_axis(3, 2);
//     Tensor q = splits[0];
//     Tensor k = splits[1];
//     Tensor v = splits[2];

//     // 3. Reshape to multi-head: (B, T, C) -> (B, T, nh, hd) -> (B, nh, T, hd)
//     q = autograd::transpose(
//             autograd::reshape(q, Shape({{B, T, (int64_t)num_heads, (int64_t)head_dim}})),
//             1, 2);
//     k = autograd::transpose(
//             autograd::reshape(k, Shape({{B, T, (int64_t)num_heads, (int64_t)head_dim}})),
//             1, 2);
//     v = autograd::transpose(
//             autograd::reshape(v, Shape({{B, T, (int64_t)num_heads, (int64_t)head_dim}})),
//             1, 2);

//     // 4. Scaled dot-product attention with backend dispatch
//     Tensor attn_out = autograd::scaled_dot_product_attention(
//         q, k, v, is_causal, backend);

//     // 5. Merge heads: (B, nh, T, hd) -> (B, T, nh, hd) -> (B, T, C)
//     Tensor merged = autograd::reshape(
//         autograd::transpose(attn_out, 1, 2),
//         Shape({{B, T, C}}));

//     // 6. Output projection: (B, T, C) -> (B, T, C)
//     return out_proj.forward(merged);
// }

// Tensor MultiHeadAttention::forward_qkv(
//     const Tensor& q, const Tensor& k, const Tensor& v)
// {
//     // q, k, v already shaped (B, nh, T, hd)
//     Tensor attn_out = autograd::scaled_dot_product_attention(
//         q, k, v, is_causal, backend);

//     int64_t B = attn_out.shape().dims[0];
//     int64_t T = attn_out.shape().dims[2];

//     // Merge heads and project
//     Tensor merged = autograd::reshape(
//         autograd::transpose(attn_out, 1, 2),
//         Shape({{B, T, (int64_t)embed_dim}}));

//     return out_proj.forward(merged);
// }

// } // namespace nn
// } // namespace OwnTensor

#include "nn/MultiHeadAttention.h"
#include "autograd/AutogradOps.h"
#include <stdexcept>

namespace OwnTensor {
namespace nn {

MultiHeadAttention::MultiHeadAttention(
    int embed_dim, int num_heads,
    bool is_causal,
    autograd::SDPBackend backend,
    bool bias)
    : embed_dim(embed_dim),
      num_heads(num_heads),
      head_dim(embed_dim / num_heads),
      is_causal(is_causal),
      backend(backend),
      qkv_proj(embed_dim, 3 * embed_dim, bias),
      out_proj(embed_dim, embed_dim, bias)
{
    if (embed_dim % num_heads != 0) {
        throw std::runtime_error(
            "MultiHeadAttention: embed_dim must be divisible by num_heads");
    }

    register_module(qkv_proj);
    register_module(out_proj);
}

Tensor MultiHeadAttention::forward(const Tensor& x) {
    // x: (B, T, C)
    int64_t B = x.shape().dims[0];
    int64_t T = x.shape().dims[1];
    int64_t C = x.shape().dims[2];

    // 1. QKV projection: (B, T, C) -> (B, T, 3C)
    Tensor qkv = qkv_proj.forward(x);

    // 2. Split into q, k, v each (B, T, C) via sharding along last dim
    std::vector<Tensor> splits = qkv.make_shards_inplace_axis(3, 2);
    Tensor q = splits[0];
    Tensor k = splits[1];
    Tensor v = splits[2];

    // 3. Reshape to multi-head: (B, T, C) -> (B, T, nh, hd) -> (B, nh, T, hd)
    q = autograd::transpose(
            autograd::reshape(q, Shape({{B, T, (int64_t)num_heads, (int64_t)head_dim}})),
            1, 2);
    k = autograd::transpose(
            autograd::reshape(k, Shape({{B, T, (int64_t)num_heads, (int64_t)head_dim}})),
            1, 2);
    v = autograd::transpose(
            autograd::reshape(v, Shape({{B, T, (int64_t)num_heads, (int64_t)head_dim}})),
            1, 2);

    // 4. Scaled dot-product attention with backend dispatch
    Tensor attn_out = autograd::scaled_dot_product_attention(
        q, k, v, is_causal,0.0f, backend);

    // 5. Merge heads: (B, nh, T, hd) -> (B, T, nh, hd) -> (B, T, C)
    Tensor merged = autograd::reshape(
        autograd::transpose(attn_out, 1, 2),
        Shape({{B, T, C}}));

    // 6. Output projection: (B, T, C) -> (B, T, C)
    return out_proj.forward(merged);
}

Tensor MultiHeadAttention::forward_qkv(
    const Tensor& q, const Tensor& k, const Tensor& v)
{
    // q, k, v already shaped (B, nh, T, hd)
    Tensor attn_out = autograd::scaled_dot_product_attention(
        q, k, v, is_causal, 0.0f,backend);

    int64_t B = attn_out.shape().dims[0];
    int64_t T = attn_out.shape().dims[2];

    // Merge heads and project
    Tensor merged = autograd::reshape(
        autograd::transpose(attn_out, 1, 2),
        Shape({{B, T, (int64_t)embed_dim}}));

    return out_proj.forward(merged);
}

} // namespace nn
} // namespace OwnTensor