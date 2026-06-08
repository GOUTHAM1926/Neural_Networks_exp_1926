#pragma once

#include "tensor/dtensor.h"
#include "dnn/DistributedNN.h"
#include "core/Tensor.h"
#include "autograd/AutogradOps.h"
#include "autograd/operations/ReshapeOps.h"
#include "autograd/operations/BinaryOps.h"
#include "communication/include/ProcessGroupNCCL.h"

#include "gpt2_cp_test/context_parallel/RingRotator.h"
#include "gpt2_cp_test/context_parallel/SDPAOp.h"
#include "gpt2_cp_test/context_parallel/SDPAMerger.h"
#include "gpt2_cp_test/context_parallel/ContextParallelBackward.h"

#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <stdexcept>

using namespace OwnTensor;
using namespace OwnTensor::dnn;

// ---------------------------------------------------------------------------
// RotatorType
//
// Selects which ring communication strategy to use.
// ---------------------------------------------------------------------------
enum class RotatorType {
    P2P,        // Point-to-point ncclSend/ncclRecv
    AlltoAll,   // sendrecv-based ring shift
    AllGather   // Single all_gather, then index
};

// ---------------------------------------------------------------------------
// CausalBehavior
//
// Controls how causal masking is applied during ring attention.
//
// FULL_CAUSAL:  Apply causal mask on every ring step (correct for
//               chunks where q_idx >= k_idx in the global sequence).
// SKIP_FUTURE:  Skip SDPA entirely for ring steps where the K,V chunk
//               is entirely in the "future" relative to the Q chunk.
// NOT_CAUSAL:   No causal masking (bidirectional attention).
// ---------------------------------------------------------------------------
enum class CausalBehavior {
    FULL_CAUSAL,
    SKIP_FUTURE,
    NOT_CAUSAL
};


// ---------------------------------------------------------------------------
// ContextParallel : DModule
//
// Implements context parallelism (ring attention) as a distributed module.
//
// Forward:
//   1. Shard the input along the sequence dimension across ranks
//      (with optional load balancing via HeadTail permutation)
//   2. Each rank holds Q for its local chunk; K,V rotate around the ring
//   3. At each ring step: compute local SDPA, merge results via SDPAMerger
//   4. After all steps: unshard (gather) the output back to full sequence
//
// The module does NOT own any parameters -- it wraps the attention
// computation pattern for distributed execution.
// ---------------------------------------------------------------------------
class ContextParallel : public DModule {
public:
    ContextParallel(
        const DeviceMesh& mesh,
        std::shared_ptr<ProcessGroupNCCL> pg,
        float attn_scale,
        bool is_causal = true,
        RotatorType rotator_type = RotatorType::P2P,
        bool load_balance = true)
        : mesh_(&mesh),
          pg_(pg),
          attn_scale_(attn_scale),
          is_causal_(is_causal),
          rotator_type_(rotator_type),
          load_balance_(load_balance),
          world_size_(pg->get_worldsize()),
          rank_(pg->get_rank()) {}

    // -----------------------------------------------------------------------
    // forward
    //
    // Input: DTensor containing the full Q, K, V concatenated or separate.
    //        For this implementation, we expect q, k, v as separate Tensors
    //        already shaped as [B, H, T, D] (4D).
    //
    // This overload takes raw Tensors and returns the merged attention output.
    // -----------------------------------------------------------------------
    Tensor forward_cp(
        Tensor& q,   // [B, H, T, D] -- full sequence query
        Tensor& k,   // [B, H, T, D] -- full sequence key
        Tensor& v)   // [B, H, T, D] -- full sequence value
    {
        // ----- Phase 1: Context Parallel Shard -----
        // Shard Q along dim=2 (sequence dim in 4D [B, H, T, D])
        // K, V also sharded along dim=2

        // HeadTail load balancing is incompatible with causal attention:
        // cross-chunk masks are checkerboard (even rows=0, odd rows=1),
        // not expressible as tril. Force-disable until custom mask SDPA is added.
        bool lb_active = load_balance_ && !is_causal_;

        // Input Q, K, V may be non-contiguous (e.g. from autograd::transpose
        // which swaps strides without copying data). Make contiguous BEFORE
        // sharding so that make_shards_inplace_axis produces views with
        // standard decreasing strides. Without this, the post-shard
        // contiguous() call produces wrong data because it cannot handle
        // non-standard stride ordering (stride[1] < stride[2]).
        Tensor q_work = q.contiguous();
        Tensor k_work = k.contiguous();
        Tensor v_work = v.contiguous();

        if (lb_active) {
            load_balancer_.set_world_size(world_size_);
            load_balancer_.set_chunk_dim(2);  // sequence dim in 4D
            load_balancer_.set_stream(0);
            load_balancer_.loadbalance(q_work);
            load_balancer_.loadbalance(k_work);
            load_balancer_.loadbalance(v_work);
        }

        // Shard into chunks along sequence dimension
        std::vector<Tensor> q_chunks = q_work.make_shards_inplace_axis(
            static_cast<size_t>(world_size_), 2);
        std::vector<Tensor> k_chunks = k_work.make_shards_inplace_axis(
            static_cast<size_t>(world_size_), 2);
        std::vector<Tensor> v_chunks = v_work.make_shards_inplace_axis(
            static_cast<size_t>(world_size_), 2);

        // Each rank gets its local chunk — contiguous() is still needed
        // because make_shards_inplace_axis creates views where the outer
        // dimension strides are larger than the sharded extent
        // (e.g. stride[1] = T*D instead of T_local*D).
        Tensor local_q = q_chunks[rank_].contiguous();  // [B, H, T/n, D]
        Tensor local_k = k_chunks[rank_].contiguous();  // [B, H, T/n, D]
        Tensor local_v = v_chunks[rank_].contiguous();  // [B, H, T/n, D]

        // ----- Phase 2: Ring Attention Loop -----
        // Create rotator for K,V communication
        std::unique_ptr<RingRotatorBase> kv_rotator = create_rotator();

        // Initialize merger for accumulating partial attention outputs
        SDPAMerger merger(/*convert_to_f32=*/true);

        // Save K,V chunks, causal flags, and per-step LSE for backward pass
        std::vector<Tensor> saved_k_chunks(world_size_);
        std::vector<Tensor> saved_v_chunks(world_size_);
        std::vector<bool> saved_causal_flags(world_size_, false);
        std::vector<Tensor> saved_lse_per_step(world_size_);

        // Pre-allocate KV send buffer (reused across ring steps)
        int64_t k_numel = local_k.numel();
        int64_t kv_numel = k_numel * 2;
        Tensor kv_send_buf = Tensor::empty(
            Shape({{kv_numel}}), local_k.opts());

        // Current K, V being processed (starts with local chunk)
        Tensor curr_k = local_k;
        Tensor curr_v = local_v;

        for (int i = 0; i < world_size_; ++i) {
            // Step 1: If not first iteration, get K,V from previous exchange
            if (i > 0) {
                Tensor next_kv = kv_rotator->next_buffer();

                // Split the received buffer: flatten to 1D then narrow
                Tensor kv_flat = next_kv.flatten();

                curr_k = kv_flat.narrow(0, 0, k_numel).reshape(local_k.shape());
                curr_v = kv_flat.narrow(0, k_numel, k_numel).reshape(local_v.shape());
            }

            // Step 2: Send current K,V to next rank (async, overlaps with compute)
            // Use pre-allocated send buffer with D2D copies instead of flatten_concat
            if (i < (world_size_ - 1)) {
                size_t k_bytes = static_cast<size_t>(k_numel) * sizeof(float);
                cudaMemcpyAsync(
                    kv_send_buf.data<float>(),
                    curr_k.data<float>(),
                    k_bytes, cudaMemcpyDeviceToDevice, 0);
                cudaMemcpyAsync(
                    kv_send_buf.data<float>() + k_numel,
                    curr_v.data<float>(),
                    k_bytes, cudaMemcpyDeviceToDevice, 0);

                // kv_send_buf = Tensor::cat({curr_k,curr_v}, 0);

                kv_rotator->exchange_buffers(kv_send_buf);
            }

            // Step 3: Determine causal behavior for this ring step
            // NOTE: load_balance_ is force-disabled when is_causal_=true
            // because HeadTail creates cross-chunk masks (even rows=zeros,
            // odd rows=ones) that cannot be expressed as tril with any offset.
            // Custom attention mask support in SDPA is needed to enable both.
            bool use_causal = false;
            if (is_causal_) {
                int source_rank = ((rank_ - i) % world_size_ + world_size_) % world_size_;
                if (source_rank == rank_) {
                    use_causal = true;
                } else if (source_rank > rank_) {
                    // Future chunk: all K,V positions > all Q positions, skip
                    continue;
                }
                // source_rank < rank_: past chunk, full attention
            }

            // Save for backward before computing
            saved_k_chunks[i] = curr_k.clone();
            saved_v_chunks[i] = curr_v.clone();
            saved_causal_flags[i] = use_causal;

            // Step 4: Compute SDPA for this (Q, K, V) pair
            SDPAResult result = sdpa_forward(local_q, curr_k, curr_v,
                                              use_causal, attn_scale_);

            // Save per-step LSE for correct backward merger rescaling
            saved_lse_per_step[i] = result.lse;

            // Step 5: Merge into accumulator
            merger.step(result.out, result.lse);
        }

        // ----- Phase 3: Get final merged result -----
        auto [merged_out, merged_lse] = merger.results();

        // ----- Phase 4: Context Parallel Unshard -----
        // Gather the output chunks back to full sequence
        // Each rank has [B, H, T/n, D], need to all_gather along dim=2

        // Flatten for all_gather
        size_t local_count = static_cast<size_t>(merged_out.numel());
        size_t total_count = local_count * static_cast<size_t>(world_size_);

        Shape gathered_shape({{static_cast<int64_t>(total_count)}});
        Tensor gathered_flat = Tensor::empty(gathered_shape, merged_out.opts());

        pg_->all_gather(
            merged_out.data<float>(),
            gathered_flat.data<float>(),
            local_count,
            merged_out.dtype(),
            true);  // sync

        // Reshape gathered output to [B, H, T, D]
        // The all_gather concatenates rank-0 chunk, rank-1 chunk, ... along flat dim
        // We need to reconstruct the full tensor
        int64_t B = local_q.shape().dims[0];
        int64_t H = local_q.shape().dims[1];
        int64_t T_local = local_q.shape().dims[2];
        int64_t D = local_q.shape().dims[3];
        int64_t T_full = T_local * world_size_;

        // Reshape: [world_size * B * H * T_local * D] -> [world_size, B, H, T_local, D]
        // Then transpose and merge to get [B, H, T_full, D]
        // For simplicity, reshape to [world_size, B*H*T_local*D] then
        // rearrange to [B, H, T_full, D]
        Shape per_rank_shape({{B, H, T_local, D}});

        // Build output by copying each rank's contribution into the right position
        Shape full_shape({{B, H, T_full, D}});
        Tensor full_output = Tensor::empty(full_shape, merged_out.opts());

        // Per-(b,h)-slice copy: source layout is [world_size, B, H, T_local, D]
        // (from all_gather), destination is [B, H, T_full, D].
        // A flat copy would scramble batch/head dims with sequence.
        size_t slice_bytes = static_cast<size_t>(T_local * D) * sizeof(float);
        for (int r = 0; r < world_size_; ++r) {
            for (int64_t b = 0; b < B; ++b) {
                for (int64_t h = 0; h < H; ++h) {
                    float* src = gathered_flat.data<float>()
                        + r * (B * H * T_local * D)
                        + b * (H * T_local * D)
                        + h * (T_local * D);
                    float* dst = full_output.data<float>()
                        + b * (H * T_full * D)
                        + h * (T_full * D)
                        + r * (T_local * D);
                    cudaMemcpyAsync(dst, src, slice_bytes,
                                   cudaMemcpyDeviceToDevice, 0);
                }
            }
        }
        cudaStreamSynchronize(0);

        // Undo load balance if it was applied
        if (lb_active) {
            load_balancer_.unloadbalance(full_output);
        }

        // ----- Register backward node -----
        // If any input requires grad, register the backward node so that
        // loss.backward() will propagate through context parallel.
        if (q.requires_grad() || k.requires_grad() || v.requires_grad()) {
            int rot_type = static_cast<int>(rotator_type_);
            auto grad_fn = std::make_shared<ContextParallelBackward>(
                local_q,
                saved_k_chunks,
                saved_v_chunks,
                saved_causal_flags,
                saved_lse_per_step,
                merged_lse,
                merged_out,
                pg_,
                attn_scale_,
                is_causal_,
                rot_type,
                lb_active,
                world_size_,
                rank_);

            // Connect to input edges for q, k, v
            // All three edges must be set so that gradients for k and v
            // propagate back through make_shards_inplace_axis → c_attn.
            if (q.requires_grad()) {
                Tensor& q_mut = const_cast<Tensor&>(q);
                grad_fn->set_next_edge(0, autograd::get_grad_edge(q_mut));
            }
            if (k.requires_grad()) {
                Tensor& k_mut = const_cast<Tensor&>(k);
                grad_fn->set_next_edge(1, autograd::get_grad_edge(k_mut));
            }
            if (v.requires_grad()) {
                Tensor& v_mut = const_cast<Tensor&>(v);
                grad_fn->set_next_edge(2, autograd::get_grad_edge(v_mut));
            }

            full_output.set_grad_fn(grad_fn);
            full_output.set_requires_grad(true);
        }

        return full_output;
    }

private:
    const DeviceMesh* mesh_;
    std::shared_ptr<ProcessGroupNCCL> pg_;
    float attn_scale_;
    bool is_causal_;
    RotatorType rotator_type_;
    bool load_balance_;
    int world_size_;
    int rank_;
    HeadTail load_balancer_;

    std::unique_ptr<RingRotatorBase> create_rotator() const {
        switch (rotator_type_) {
            case RotatorType::P2P:
                return std::make_unique<P2PRingRotator>(pg_);
            case RotatorType::AlltoAll:
                return std::make_unique<AlltoAllRingRotator>(pg_);
            case RotatorType::AllGather:
                return std::make_unique<AllGatherRingRotator>(pg_);
            default:
                throw std::runtime_error("Unknown rotator type");
        }
    }
};
