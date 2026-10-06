#pragma once
//======================================================================
// ShardedOptimizer.h — ZeRO Stage-1 optimizer (optimizer-state sharding).
//
// Implements the ground-up design in ZERO_OPTIMIZER_DESIGN.md:
//
//   * Model parameters are conceptually concatenated into one flat buffer,
//     padded to a multiple of the world size W, and split into W equal shards
//     ("whole-buffer contiguous ownership"): rank r owns flat range
//     [r*S, (r+1)*S).  S = padded_numel / W.
//   * Each rank keeps FP32 master weights + Adam moments (m, v) for ONLY its
//     shard  ->  optimizer-state memory drops from 12N to 12N/W per GPU.
//   * Per step:
//        gather grads -> flat staging (compute dtype)
//        reduce-scatter (avg)         -> this rank's grad shard
//        [loss-scaler unscale + global non-finite guard]
//        [global gradient clip via partial-norm all-reduce]
//        fused Adam on the FP32 shard  (bit-identical math to nn::AdamW)
//        all-gather updated shards     -> full params (compute dtype)
//        scatter back into model tensors
//
// The math on the shard is the SAME fused multi_tensor_adam_cuda kernel
// nn::AdamW uses, so at W=1 the result is bit-identical to single-process
// AdamW — the parity gate the whole design is validated against.
//
// This is copy-mode (grads copied in, params copied out): the framework's
// autograd reallocates each parameter's gradient on every backward, so a
// truly aliased flat gradient buffer is not achievable without changing the
// engine. FlatParam (params as views into the flat buffer) is a future
// optimization that would remove the param scatter-back copy only.
//======================================================================

#include <vector>
#include <memory>
#include <cstdint>
#include <ostream>
#include <istream>

#include "TensorLib.h"
#include "nn/optimizer/Optim.h"
#include "nn/optimizer/LossScaler.h"
#include "ProcessGroupBase.h"   // backend-neutral: ZeRO never needs the NCCL concrete type

#include <map>
#include <string>

#include "ZeroConfig.h"
#include "ZeroErrors.h"
#include "ZeroOptimizerBase.h"

namespace owt {
namespace zero {

// ZeRO-1 v1: bucketed ncclReduce(root=owner) + staged ncclBroadcast, over
// CONTIGUOUS shard ownership. See SymmetricShardedOptimizer for the
// reduce-scatter/all-gather sibling that can reach NCCL's NVLS kernels.
class ShardedOptimizer : public ZeroOptimizerBase {
public:
    using Tensor = OwnTensor::Tensor;
    using Dtype  = OwnTensor::Dtype;

    // model_parameters : the trainable tensors (e.g. from Module::parameters()).
    //                    They are NOT copied into the base optimizer — this
    //                    class manages sharded state itself so the base class
    //                    never allocates full-size FP32 masters.
    // pg               : the data-parallel process group (may be null iff W==1).
    // cfg              : hyperparameters + policy (see ZeroConfig.h).
    ShardedOptimizer(const std::vector<Tensor>& model_parameters,
                     std::shared_ptr<dist::comm::ProcessGroupBase> pg,
                     ZeroConfig cfg);

    ~ShardedOptimizer() override;

    // --- Optimizer interface ---
    void step() override;                 // one ZeRO-1 step (see .cpp)
    void set_lr(float lr) override { cfg_.lr = lr; }

    // Zeroes the gradients of the managed model parameters. (Base zero_grad is
    // non-virtual and operates on the base's empty param list, so we provide
    // our own that acts on the real model tensors.)
    void zero_grad() override;

    int64_t step_count() const override { return step_count_; }
    std::vector<Tensor> get_m_buffers() const override { return {shard_m_}; }
    std::vector<Tensor> get_v_buffers() const override { return {shard_v_}; }

    // --- sharded checkpoint (see .cpp / ZERO_OPTIMIZER_DESIGN.md §9) ---
    // Writes this rank's shard (master, m, v) + a layout descriptor + step.
    void save_state(std::ostream& os) override;
    // Same-world-size load. Throws ZeroCheckpointError on a layout mismatch.
    void load_state(std::istream& is) override;

    // --- DCP flat-shard interface (OwnTensor::dcp::ShardedOptimizerState) ---
    // Exposes this rank's FP32 shard buffers to Distributed Checkpointing as
    // ranges in the GLOBAL flat coordinate system, so every rank writes its own
    // slice and DCP can intersect/reshard them on load.
    std::vector<OwnTensor::dcp::OptimStateShardSpec> dcp_state_shard_specs() override;
    Tensor dcp_get_state_shard(const std::string& state_type,
                               const std::string& param_fqn) override;
    void   dcp_set_state_shard(const std::string& state_type,
                               const std::string& param_fqn,
                               const Tensor& flat_slice) override;
    void   dcp_after_load() override;

    // step_count_ does not live in any shard buffer, so it rides DCP's
    // optimizer_config map (scope-qualified, since a trainer may hold several
    // ZeRO instances). Exact for step counts below 2^24.
    std::map<std::string, float> get_config() override;
    void set_config(const std::map<std::string, float>& config) override;

    // --- accessors ---
    float get_lr() const override { return cfg_.lr; }
    void  set_max_grad_norm(float m) override { cfg_.max_grad_norm = m; }
    float last_grad_norm() const override { return last_grad_norm_; }
    bool  last_step_skipped() const override { return last_step_skipped_; }

    // --- compute/comm overlap (ZeroConfig::overlap_backward_comm) ---
    // Arm the bucketed reduce for ONE backward pass. Call this immediately
    // before the FINAL micro-step's loss.backward(): the per-parameter hooks
    // then copy each fully-accumulated gradient into the flat staging buffer and
    // fire that gradient's bucket reduce as soon as the bucket fills, overlapping
    // the reduce with the rest of backward. A no-op when overlap is disabled (or
    // W==1). Hooks are inert on any backward for which this was not called, so
    // earlier micro-steps just accumulate gradients as usual.
    void arm_grad_sync() override;
    bool overlap_enabled() const override { return overlap_active_; }

    // --- cross-group global clipping (see ZeroOptimizerBase) ---
    float reduce_and_measure() override;
    void  set_external_clip_coef(float coef) override { external_clip_coef_ = coef; }
    // Elementwise AdamW: the coefficient is folded into `combined` and applied
    // across the whole valid shard, so coverage is the entire group.
    size_t clip_covered_numel() const override { return total_numel_; }

    // --- staged parameter refresh (ZeroConfig::overlap_param_refresh) ---
    // Assign every managed parameter to a refresh STAGE, in forward-consumption
    // order: stage 0 must contain every parameter the next forward reads before
    // it reaches the point where stage 1 is consumed, and so on. `stage_of` is
    // index-aligned with the `model_parameters` vector passed to the constructor
    // (entries for non-trainable params are ignored). Must be called once, before
    // the first step(); without it the synchronous all-gather path is used.
    void set_param_refresh_stages(const std::vector<int>& stage_of, int num_stages) override;

    // Consume refresh stages up to and including `stage`: wait for their
    // broadcasts (a GPU-side stream dependency, not a host stall) and copy the
    // refreshed parameters into the live model tensors. Idempotent, cheap when
    // nothing is outstanding, and safe to call from inside forward(). Stages must
    // be consumed in increasing order; consuming stage s implicitly consumes all
    // earlier ones.
    void refresh_param_stage(int stage) override;

    // Consume every outstanding stage. Call before ANY read of the model
    // parameters that is not the training forward pass — validation forwards,
    // checkpoint saves, weight inspection. Safe and cheap when nothing is
    // outstanding.
    void drain_param_refresh() override;

    bool param_refresh_pending() const override { return param_refresh_pending_; }

    // Layout info (useful for tests / diagnostics).
    int64_t total_numel()  const override { return static_cast<int64_t>(total_numel_); }
    int64_t padded_numel() const override { return static_cast<int64_t>(padded_numel_); }
    int64_t shard_numel()  const override { return static_cast<int64_t>(S_); }
    int     world_size()   const override { return W_; }
    int     rank()         const override { return rank_; }

    const char* variant_name() const override { return "zero1-v1 reduce/broadcast"; }

private:
    // A contiguous run of the flat buffer that shares one (weight_decay, lr)
    // setting. Params are ordered so each group is a single contiguous run.
    struct GroupSpec {
        size_t begin;          // global flat element offset (inclusive)
        size_t end;            // global flat element offset (exclusive)
        float  weight_decay;
        float  lr_scale;       // multiplied into the base lr for this group
    };

    // Where one model parameter lives in the flat buffer (flat order == the
    // order params are laid out after group sorting).
    struct ParamRecord {
        Tensor param;          // handle to the model tensor (shared storage)
        size_t offset;         // global flat element offset of this param
        size_t numel;
        uint8_t group_id;
        size_t input_index;    // position in the ctor's model_parameters vector
    };

    // ---- setup helpers ----
    void build_layout(const std::vector<Tensor>& model_parameters);
    void allocate_buffers();
    void init_master_from_params();
    void verify_consistency();

    // ---- per-step helpers ----
    void gather_grads_into_flat();               // model grads -> flat_stage_
    void reduce_scatter_grads();                 // flat_stage_ -> shard_grad_ (avg)
    void cast_shard_grad_to_fp32();              // shard_grad_ -> shard_grad_fp32_
    std::string nonfinite_grad_msg(float nsq) const;   // F1: text for ZeroNonFiniteGradError
    bool unscale_and_check_overflow();           // returns true if overflow (skip step)
    void clip_gradients();                       // global-norm clip on shard_grad_fp32_
    // Gradient reduce + FP32 cast + loss-scaler unscale — the part of step() that
    // reduce_and_measure() runs early so step() can skip it.
    void reduce_grads_now();
    void local_adam_step();                      // fused Adam over this shard's valid range
    void all_gather_params();                    // shard master -> flat_stage_ (compute)
    void scatter_flat_into_params();             // flat_stage_ -> model tensors
    void broadcast_root_params();                // init: sync all ranks to rank 0

    // ---- overlap (compute/comm) helpers ----
    void build_buckets();                        // partition flat buffer into per-owner buckets
    void register_overlap_hooks();               // post-acc-grad hook per trainable param
    void on_grad_ready(size_t param_index,       // hook body: copy grad + maybe launch bucket
                       const Tensor& grad);
    void launch_bucket(size_t bucket_index);     // compute->comm event + reduce-to-owner
    void finalize_overlap_reduce();              // mop-up + wait all Works (comm->compute)

    // ---- staged parameter refresh helpers ----
    // Adam + down-cast + broadcast, run as a two-phase pipeline instead of
    // "Adam everything, then cast everything, then issue everything". Phase A
    // covers refresh stage 0 only (the parameters the next forward reads first —
    // the token embedding); its broadcast is handed to the comm stream the moment
    // its Adam and cast land, so it transfers WHILE phase B's Adam is still
    // running on the compute stream. Phase B is every remaining stage.
    //
    // Without this, no broadcast can start until the whole shard's Adam + cast +
    // staging copy has finished, which measured 3.76 ms of dead time in front of
    // stage 0's 3.13 ms transfer (report5) — and stage 0 is the one piece that
    // cannot be hidden behind the forward, because the embedding reads it first.
    void adam_and_publish_staged();
    // Fused Adam over [begin,end) intersected with this rank's VALID shard range,
    // split at weight-decay group boundaries. Elementwise, so running the shard in
    // stage order rather than flat order changes nothing numerically.
    void adam_flat_range(size_t begin, size_t end, float bc1, float bc2);
    // FP32 master -> compute-dtype send staging for [begin,end) n this shard.
    void cast_master_range(size_t begin, size_t end);
    // Issue one stage's broadcasts (every rank calls; root = the piece's owner).
    void issue_refresh_stage(int stage);

    // valid (non-padding) global range this rank owns: [rS, min((r+1)S, total))
    size_t shard_valid_begin() const { return shard_begin_; }
    size_t shard_valid_end()   const;

    // ---- config / topology ----
    ZeroConfig cfg_;
    std::shared_ptr<dist::comm::ProcessGroupBase> pg_;
    int W_ = 1;
    int rank_ = 0;

    OwnTensor::DeviceIndex device_{};
    Dtype compute_dtype_ = Dtype::Float32;   // model param / grad dtype
    Dtype reduce_dtype_  = Dtype::Float32;   // grad reduce-scatter wire dtype

    // ---- layout ----
    std::vector<ParamRecord> records_;       // flat-order param map
    std::vector<GroupSpec>   groups_;
    size_t total_numel_  = 0;
    size_t padded_numel_ = 0;
    size_t S_ = 0;                           // shard element count = padded/W
    size_t shard_begin_ = 0;                 // rank_ * S_
    uint64_t layout_hash_ = 0;               // identity of the flat ordering + groups

    // ---- buffers ----
    Tensor flat_stage_;          // [padded] compute dtype: grad-in AND param-out staging
    Tensor shard_grad_;          // [S] compute/reduce dtype: reduce-scatter recv
    Tensor shard_grad_fp32_;     // [S] FP32: up-cast grad shard
    Tensor shard_master_;        // [S] FP32: master weights (this rank's shard)
    Tensor shard_m_;             // [S] FP32: Adam first moment
    Tensor shard_v_;             // [S] FP32: Adam second moment
    Tensor shard_master_compute_;// [S] compute dtype: all-gather send staging

    // Cross-group clipping state (see ZeroOptimizerBase::reduce_and_measure).
    float cached_nsq_         = -1.0f;
    float external_clip_coef_ = -1.0f;
    bool  grads_reduced_      = false;

    // persistent tiny GPU/host scalars for clip + overflow
    float* d_norm_sq_    = nullptr;
    float* d_clip_coef_  = nullptr;
    float* h_norm_pinned_ = nullptr;

    // ---- state ----
    int64_t step_count_ = 0;
    int     micro_step_ = 0;
    float   last_grad_norm_ = 0.0f;
    bool    last_step_skipped_ = false;
    bool    is_adamw_ = true;

    // ---- overlap (compute/comm) state ----
    // A contiguous flat range [begin,end) reduced to a single owner rank. By
    // construction it lies entirely inside owner's shard [owner*S,(owner+1)*S),
    // so ncclReduce(op=avg,root=owner) writes owner's shard_grad_ slice exactly
    // as the monolithic reduce-scatter would, keeping shard_grad_/Adam/all-gather
    // byte-compatible with the non-overlap path.
    struct CommBucket {
        size_t begin = 0;              // global flat element offset (inclusive)
        size_t end   = 0;              // global flat element offset (exclusive)
        int    owner = 0;             // = begin / S_
        size_t local_recv_offset = 0;  // = begin - owner*S_ (into shard_grad_)
        size_t pending = 0;            // distinct params not yet copied this pass
        size_t pending_init = 0;       // pending reset value (# params overlapping)
        bool   launched = false;       // reduce issued this pass
        std::shared_ptr<Work> work;    // recycled across steps (event reuse)
    };

    bool overlap_active_ = false;      // cfg_.overlap_backward_comm && W_>1
    bool armed_ = false;               // hooks act only during an armed backward
    std::vector<CommBucket>            buckets_;
    std::vector<std::vector<uint32_t>> param_to_buckets_;  // record idx -> bucket idxs
    size_t buckets_launched_ = 0;      // count launched this pass
    // Per-pass record of which parameters' hooks actually fired. Authoritative
    // for the mop-up in finalize_overlap_reduce: param.has_grad() is NOT, because
    // zero_grad() only zeroes an existing gradient buffer rather than dropping it,
    // so a parameter that is unused in this step's graph still reports has_grad()
    // and would leave its bucket unlaunched (reducing a stale gradient).
    std::vector<bool> grad_fired_;
    Tensor scratch_recv_;              // [max bucket] discard recv for non-owned reduces
    cudaEvent_t compute_sync_event_ = nullptr;  // single reused compute->comm event

    // ---- staged parameter refresh state ----
    // One contiguous flat range broadcast from the rank that owns it. Like the
    // reduce buckets, a piece never crosses a shard boundary, so the owner can
    // send straight out of its own shard_master_compute_ slice while every rank
    // receives into the same absolute offset of flat_stage_. Pieces are grouped
    // into consumption-ordered stages and issued in stage order on the ONE comm
    // stream, so waiting stage s also guarantees stages < s have landed.
    struct RefreshPiece {
        size_t begin = 0, end = 0;     // global flat element range [begin,end)
        int    owner = 0;              // = begin / S_
        size_t local_send_offset = 0;  // = begin - owner*S_ (into shard_master_compute_)
        std::shared_ptr<Work> work;    // recycled across steps (event reuse)
    };

    bool overlap_param_active_ = false;   // cfg_.overlap_param_refresh && W_>1 && staged
    bool param_refresh_pending_ = false;  // broadcasts issued, not yet fully consumed
    int  num_refresh_stages_ = 0;
    int  next_refresh_stage_ = 0;         // first stage not yet consumed
    std::vector<RefreshPiece>          refresh_pieces_;
    std::vector<std::vector<uint32_t>> stage_pieces_;   // stage -> piece idxs
    std::vector<std::vector<uint32_t>> stage_records_;  // stage -> record idxs (scatter)
    cudaEvent_t param_sync_event_ = nullptr;  // dedicated compute->comm event
};

} // namespace zero
} // namespace owt
