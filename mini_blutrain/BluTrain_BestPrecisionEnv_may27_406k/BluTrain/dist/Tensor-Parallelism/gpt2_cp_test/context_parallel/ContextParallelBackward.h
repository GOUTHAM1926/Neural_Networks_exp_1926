#pragma once

#include "autograd/Node.h"
#include "autograd/Engine.h"
#include "autograd/ops_template.h"
#include "core/Tensor.h"
#include "tensor/dtensor.h"
#include "communication/include/ProcessGroupNCCL.h"

#include "gpt2_cp_test/context_parallel/RingRotator.h"
#include "gpt2_cp_test/context_parallel/SDPAOp.h"
#include "gpt2_cp_test/context_parallel/SDPAMerger.h"

#include <memory>
#include <vector>
#include <cmath>
#include <stdexcept>

using namespace OwnTensor;

// ---------------------------------------------------------------------------
// ContextParallelBackward
//
// Autograd node for the backward pass of context parallel ring attention.
//
// Key corrections vs naive backward:
//   1. Shards the incoming full gradient [B,H,T,D] to local [B,H,T/n,D]
//   2. Applies merger rescaling: each step's grad is weighted by
//      exp(step_lse - merged_lse) to account for the online softmax merge
//   3. Communicates dK/dV back to source ranks via sendrecv
//   4. All-gathers local gradients to reconstruct full [B,H,T,D] gradients
// ---------------------------------------------------------------------------
class ContextParallelBackward : public Node {
public:
    ContextParallelBackward(
        // Saved tensors from forward
        Tensor saved_q,                          // [B, H, T/n, D]
        std::vector<Tensor> saved_k_chunks,      // K chunks per ring step
        std::vector<Tensor> saved_v_chunks,      // V chunks per ring step
        std::vector<bool> saved_causal_flags,    // causal flag per ring step
        std::vector<Tensor> saved_lse_per_step,  // LSE per ring step [B,H,T/n,1]
        Tensor merged_lse,                       // final merged LSE [B,H,T/n,1]
        Tensor merged_out,                       // final merged out [B,H,T/n,D]
        // Process group and config
        std::shared_ptr<ProcessGroupNCCL> pg,
        float attn_scale,
        bool is_causal,
        int rotator_type,
        bool load_balance,
        int world_size,
        int rank)
        : Node(3),  // 3 outputs: grad for q, k, v
          saved_q_(saved_q),
          saved_k_chunks_(saved_k_chunks),
          saved_v_chunks_(saved_v_chunks),
          saved_causal_flags_(saved_causal_flags),
          saved_lse_per_step_(saved_lse_per_step),
          merged_lse_(merged_lse),
          merged_out_(merged_out),
          pg_(pg),
          attn_scale_(attn_scale),
          is_causal_(is_causal),
          rotator_type_(rotator_type),
          load_balance_(load_balance),
          world_size_(world_size),
          rank_(rank) {}

    const char* name() const override { return "ContextParallelBackward"; }

    std::vector<Tensor> apply(std::vector<Tensor>&& grads) override {
        if (grads.empty()) {
            throw std::runtime_error("ContextParallelBackward: no gradients provided");
        }

        // ----- Phase 0: Shard the full gradient to local chunk -----
        Tensor grad_output_full = grads[0].contiguous();
        std::vector<Tensor> grad_chunks = grad_output_full.make_shards_inplace_axis(
            static_cast<size_t>(world_size_), 2);
        Tensor grad_local = grad_chunks[rank_].contiguous();  // [B, H, T/n, D]

        // ----- Phase 1: Gradient Buffer Init -----
        Tensor grad_q = Tensor::zeros(saved_q_.shape(), saved_q_.opts());

        std::vector<Tensor> grad_k_accum(world_size_);
        std::vector<Tensor> grad_v_accum(world_size_);

        for (int i = 0; i < world_size_; ++i) {
            grad_k_accum[i] = Tensor::zeros(saved_q_.shape(), saved_q_.opts());
            grad_v_accum[i] = Tensor::zeros(saved_q_.shape(), saved_q_.opts());
        }

        // ----- Phase 2: Ring Loop (backward) -----
        for (int i = 0; i < world_size_; ++i) {
            if (!saved_k_chunks_[i].is_valid()) {
                continue;
            }

            bool use_causal = saved_causal_flags_[i];

            Tensor step_q = saved_q_.clone();
            Tensor step_k = saved_k_chunks_[i].clone();
            Tensor step_v = saved_v_chunks_[i].clone();

            step_q.set_requires_grad(true);
            step_k.set_requires_grad(true);
            step_v.set_requires_grad(true);

            Tensor lse_diff = Tensor::zeros(merged_lse_.shape(), merged_lse_.opts());
            if (saved_lse_per_step_[i].is_valid() && merged_lse_.is_valid()) {
                lse_diff = saved_lse_per_step_[i] - merged_lse_;
            }

            std::vector<Tensor> step_grads = sdpa_backward_op_manual(
                step_q, step_k, step_v,
                grad_local,
                merged_out_,
                lse_diff,
                use_causal,
                attn_scale_);

            if (step_grads[0].is_valid()) {
                grad_q = grad_q + step_grads[0];
            }
            if (step_grads[1].is_valid()) {
                grad_k_accum[i] = grad_k_accum[i] + step_grads[1];
            }
            if (step_grads[2].is_valid()) {
                grad_v_accum[i] = grad_v_accum[i] + step_grads[2];
            }
        }

        // ----- Phase 3: Communicate dK, dV back to source ranks -----
        // Each grad_k_accum[i] belongs to the rank whose K chunk was used at step i.
        // That source rank is (rank_ - i + world_size_) % world_size_.
        // We must send grad_k_accum[i] there and receive contributions destined for
        // our own K chunk from all other ranks.
        // Dispatch on rotator_type_ to use the same collective as the forward pass.

        Tensor local_grad_k = Tensor::zeros(grad_k_accum[0].shape(), grad_k_accum[0].opts());
        Tensor local_grad_v = Tensor::zeros(grad_v_accum[0].shape(), grad_v_accum[0].opts());

        if (rotator_type_ == 1) {
            // AlltoAll: build a send buffer [world_size * kv_count] where slot dest
            // holds (grad_k_accum[i], grad_v_accum[i]) for dest = (rank_-i+n)%n.
            // After alltoall, each rank's recv buffer slot j contains rank j's
            // contribution to our K/V chunk. Sum all slots to get local_grad_k/v.
            int64_t k_count = grad_k_accum[0].numel();
            int64_t kv_count = k_count * 2;
            size_t byte_count = static_cast<size_t>(k_count) * sizeof(float);

            Shape agg_shape({{kv_count * world_size_}});
            Tensor send_buf = Tensor::zeros(agg_shape, grad_k_accum[0].opts());
            Tensor recv_buf = Tensor::zeros(agg_shape, grad_k_accum[0].opts());

            for (int i = 0; i < world_size_; ++i) {
                int dest = ((rank_ - i) % world_size_ + world_size_) % world_size_;
                float* dk_slot = send_buf.data<float>() + static_cast<int64_t>(dest) * kv_count;
                float* dv_slot = dk_slot + k_count;
                cudaMemcpyAsync(dk_slot, grad_k_accum[i].data<float>(),
                                byte_count, cudaMemcpyDeviceToDevice, 0);
                cudaMemcpyAsync(dv_slot, grad_v_accum[i].data<float>(),
                                byte_count, cudaMemcpyDeviceToDevice, 0);
            }
            cudaStreamSynchronize(0);

            auto work = pg_->alltoall_async(
                send_buf.data(), recv_buf.data(),
                static_cast<size_t>(kv_count),
                grad_k_accum[0].dtype());
            if (work) {
                work->wait();
            }

            // Sum all received slots — each slot r contains rank r's dK/dV
            // contribution to our K/V chunk.
            for (int r = 0; r < world_size_; ++r) {
                int64_t slot_off = static_cast<int64_t>(r) * kv_count;
                Tensor slot_dk = recv_buf.narrow(0, slot_off, k_count)
                                         .reshape(local_grad_k.shape());
                Tensor slot_dv = recv_buf.narrow(0, slot_off + k_count, k_count)
                                         .reshape(local_grad_v.shape());
                local_grad_k = local_grad_k + slot_dk;
                local_grad_v = local_grad_v + slot_dv;
            }
        } else {
            // P2P (sendrecv): send grad_k_accum[i] to source rank, receive
            // contributions from the rank whose queries attended our K chunk.
            local_grad_k = grad_k_accum[0];
            local_grad_v = grad_v_accum[0];

            for (int i = 1; i < world_size_; ++i) {
                int source_rank = ((rank_ - i) % world_size_ + world_size_) % world_size_;

                Tensor dk_flat = grad_k_accum[i].flatten();
                Tensor dv_flat = grad_v_accum[i].flatten();
                Tensor dkv_concat = Tensor::flatten_concat({dk_flat, dv_flat});

                Tensor recv_buf = Tensor::empty(dkv_concat.shape(), dkv_concat.opts());

                int dest_rank = source_rank;
                int recv_from = ((rank_ + i) % world_size_);

                size_t count = static_cast<size_t>(dkv_concat.numel());

                auto work = pg_->sendrecv_async(
                    dkv_concat.data<float>(),
                    recv_buf.data<float>(),
                    dest_rank,
                    recv_from,
                    count,
                    dkv_concat.dtype());

                if (work) {
                    work->wait();
                }

                Tensor recv_flat = recv_buf.flatten();
                int64_t k_numel = local_grad_k.numel();
                Tensor recv_dk = recv_flat.narrow(0, 0, k_numel).reshape(local_grad_k.shape());
                Tensor recv_dv = recv_flat.narrow(0, k_numel, k_numel).reshape(local_grad_v.shape());

                local_grad_k = local_grad_k + recv_dk;
                local_grad_v = local_grad_v + recv_dv;
            }
        }

        // ----- Phase 4: Unshard gradients -----
        Tensor full_grad_q = all_gather_along_seq(grad_q);
        Tensor full_grad_k = all_gather_along_seq(local_grad_k);
        Tensor full_grad_v = all_gather_along_seq(local_grad_v);

        bool lb_active = load_balance_ && !is_causal_;
        if (lb_active) {
            HeadTail lb;
            lb.set_world_size(world_size_);
            lb.set_chunk_dim(2);
            lb.set_stream(0);
            lb.unloadbalance(full_grad_q);
            lb.unloadbalance(full_grad_k);
            lb.unloadbalance(full_grad_v);
        }

        return {full_grad_q, full_grad_k, full_grad_v};
    }


    void release_saved_variables() override {
        saved_q_ = Tensor();
        saved_k_chunks_.clear();
        saved_v_chunks_.clear();
        saved_lse_per_step_.clear();
        merged_lse_ = Tensor();
        merged_out_ = Tensor();
    }

private:
    Tensor saved_q_;
    std::vector<Tensor> saved_k_chunks_;
    std::vector<Tensor> saved_v_chunks_;
    std::vector<bool> saved_causal_flags_;
    std::vector<Tensor> saved_lse_per_step_;
    Tensor merged_lse_;
    Tensor merged_out_;

    std::shared_ptr<ProcessGroupNCCL> pg_;
    float attn_scale_;
    bool is_causal_;
    int rotator_type_;
    bool load_balance_;
    int world_size_;
    int rank_;

    std::unique_ptr<RingRotatorBase> create_rotator() const {
        switch (rotator_type_) {
            case 0:
                return std::make_unique<P2PRingRotator>(pg_);
            case 1:
                return std::make_unique<AlltoAllRingRotator>(pg_);
            case 2:
                return std::make_unique<AllGatherRingRotator>(pg_);
            default:
                throw std::runtime_error("Unknown rotator type");
        }
    }

    Tensor all_gather_along_seq(Tensor& local) {
        size_t local_count = static_cast<size_t>(local.numel());
        size_t total_count = local_count * static_cast<size_t>(world_size_);

        Shape flat_shape({{static_cast<int64_t>(total_count)}});
        Tensor gathered_flat = Tensor::empty(flat_shape, local.opts());

        pg_->all_gather(
            local.data<float>(),
            gathered_flat.data<float>(),
            local_count,
            local.dtype(),
            true);

        int64_t B = local.shape().dims[0];
        int64_t H = local.shape().dims[1];
        int64_t T_local = local.shape().dims[2];
        int64_t D = local.shape().dims[3];
        int64_t T_full = T_local * world_size_;

        Shape full_shape({{B, H, T_full, D}});
        Tensor full = Tensor::empty(full_shape, local.opts());

        // Per-(b,h)-slice copy with correct strides (see ContextParallel.h)
        size_t slice_bytes = static_cast<size_t>(T_local * D) * sizeof(float);
        for (int r = 0; r < world_size_; ++r) {
            for (int64_t b = 0; b < B; ++b) {
                for (int64_t h = 0; h < H; ++h) {
                    float* src = gathered_flat.data<float>()
                        + r * (B * H * T_local * D)
                        + b * (H * T_local * D)
                        + h * (T_local * D);
                    float* dst = full.data<float>()
                        + b * (H * T_full * D)
                        + h * (T_full * D)
                        + r * (T_local * D);
                    cudaMemcpyAsync(dst, src, slice_bytes,
                                   cudaMemcpyDeviceToDevice, 0);
                }
            }
        }
        cudaStreamSynchronize(0);

        return full;
    }
};



