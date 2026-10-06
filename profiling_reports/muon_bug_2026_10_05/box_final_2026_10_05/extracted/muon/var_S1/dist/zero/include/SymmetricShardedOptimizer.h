#pragma once
//======================================================================
// SymmetricShardedOptimizer.h — ZeRO Stage-1 (v2): bucketed ncclReduceScatter +
// ncclAllGather over NCCL symmetric memory, targeting NVLink SHARP (NVLS) on an
// 8xH200 NVSwitch node.
//
// WHY A SECOND IMPLEMENTATION
// ---------------------------
// v1 (owt::zero::ShardedOptimizer) reduces gradients with one
// ncclReduce(op=avg, root=owner) per bucket and republishes parameters with one
// ncclBroadcast(root=owner) per piece. That is correct and overlaps well, but it
// can NEVER use the NVSwitch's in-network reduction: NCCL 2.28 ships symmetric
// (NVLS) device kernels for exactly three collectives —
//
//     ncclSymkDevKernel_{AllGather,AllReduce,ReduceScatter}_*
//
// — and there is no symmetric Reduce and no symmetric Broadcast. So v1's two
// hot collectives are precisely the two the switch cannot accelerate, which is
// why a v1 profile on 8xH200 shows every collective running RING_LL.
//
// v2 expresses the same ZeRO-1 algorithm using only reduce-scatter and
// all-gather, so both hot collectives can dispatch to
// ReduceScatter_LDMC (multimem.ld_reduce — the switch performs the sum) and
// AllGather_STMC (multimem.st — the switch replicates the store).
//
// OWNERSHIP: PER_BUCKET, NOT CONTIGUOUS
// -------------------------------------
// v1 gives rank r one contiguous global range [r*S, (r+1)*S). That is exactly
// what makes reduce-to-owner legal, and exactly what makes a bucketed
// reduce-scatter impossible: ncclReduceScatter(send, recv, n) requires the send
// buffer to hold n*W contiguous elements laid out RANK-MAJOR, i.e. every
// collective must span all W ranks' pieces.
//
// So v2 inverts the layout. The padded flat buffer is tiled into buckets whose
// lengths are exact multiples of W, and rank r owns sub-chunk r of EVERY bucket:
//
//   bucket k = [Bk, Bk + Lk),  Lk = Lk_per_rank * W
//   rank r owns [Bk + r*Lk_per_rank, Bk + (r+1)*Lk_per_rank)
//
// Rank r's shard is the concatenation over k of its sub-chunks — contiguous in
// optimizer-state space (shard_master_/m_/v_ are indexed 0..S) but strided in
// flat space. Each bucket then maps to exactly ONE collective with no staging
// copy and no offset arithmetic:
//
//   grads : ncclReduceScatter(flat_stage_ + Bk, flat_stage_ + Bk + r*Lk/W, Lk/W)
//   params: ncclAllGather   (flat_stage_ + Bk + r*Lk/W, flat_stage_ + Bk, Lk/W)
//
// i.e. both IN PLACE: rank r's piece of bucket k is already its own slice of
// the flat buffer, so no shard-sized send or receive buffer is needed. (Backends
// that cannot run them in place use shard_grad_ + off_k / shard_master_compute_
// + off_k instead, where off_k = sum_{j<k} Lj/W is bucket k's offset into this
// rank's shard; see inplace_collectives_.)
//
// This also kills v1's scratch_recv_ buffer: under reduce-to-owner, W-1 ranks
// discarded their result, so a throwaway receive buffer was needed. Under
// reduce-scatter every rank receives real data it owns.
//
// op=sum, NOT op=avg
// ------------------
// Every symmetric reduction kernel NCCL ships is `_sum_`; there is no symmetric
// avg. Reducing with avg silently falls off the fast path. v2 therefore reduces
// with SUM and folds the 1/W into the single scale kernel that the clip /
// loss-scale path already runs (see apply_grad_scale()), so it costs nothing —
// v2 actually runs FEWER elementwise passes over the grad shard than v1, which
// applies unscale and clip as two separate kernels.
//
// SYMMETRIC MEMORY
// ----------------
// The buffers that cross the wire — flat_stage_ (RS send + recv, AG send +
// recv), plus shard_grad_ / shard_master_compute_ when not in place — are
// allocated through dist::comm::NcclSymmetricAllocator (ncclMemAlloc) and each
// registered as its own NCCL_WIN_COLL_SYMMETRIC window. NCCL requires BOTH the
// send and the recv buffer of a collective to sit in symmetric windows, and
// requires the byte offset from each window's head to be IDENTICAL on every
// rank. Both hold here: the buffers are the same size on every rank, each
// window's head is the tensor's storage base (offset 0), and Bk / off_k are
// derived from a deterministic tiling of the same numbers on every rank. The
// in-place forms are NCCL's own in-place convention (the rank-offset side is
// buffer + r*count); on NCCL 2.30.4 they dispatch to the same symmetric kernels.
//
// Bucket lengths are rounded to a multiple of W * (16 / elem_size) so that each
// rank's per-bucket piece stays 16-byte aligned — multimem instructions operate
// on 16-byte vectors and a misaligned tail degrades to a scalar path.
//
// Falls back cleanly: if the process group was not created with
// MemoryMode::Symmetric (or NCCL < 2.27), the buffers come from the ordinary
// caching allocator, no windows are registered, and the same reduce-scatter /
// all-gather schedule runs on NCCL's ring/tree path.
//
// PARITY
// ------
// At W==1 no collective is issued, the "shard" is the whole model, the 1/W scale
// is 1.0 and is skipped, and the numerics reduce to the same fused
// multi_tensor_adam_cuda calls in the same order as nn::AdamW — the same
// bit-identical parity gate v1 is validated against.
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
#include "MemoryMode.h"

#include <map>
#include <string>

#include "ZeroConfig.h"
#include "ZeroErrors.h"
#include "ZeroOptimizerBase.h"

namespace owt {
namespace zero {

// Build stamp, printed in the startup banner. Bump this whenever the source
// changes in a way a user would need to notice.
//
// This exists because the source lives on one machine and runs on another: a
// partial scp leaves a stale binary that fails in ways that look like code bugs
// (e.g. "reduce_dtype=FP32 is not supported" from a build that predates FP32
// reduce support). Compare the banner against ZERO_V2_BUILD in the header you
// think you copied — if they differ, the binary is stale, full stop.
// Keep this a plain version, NOT a feature name. "v2.2-fp32reduce" was a mistake:
// it reads like a mode that is switched on.
#define ZERO_V2_BUILD "v2.4"

// What this binary CAN do — a staleness check, not a statement about the current
// run. Every entry is prefixed "can-" so it cannot be misread as active state;
// what is actually active is printed on the separate "ACTIVE:" line.
#define ZERO_V2_CAPS "rs+ag | can-symmetric-window | can-fp32-reduce | can-bf16-reduce | can-reshard-v1-ckpt"

class SymmetricShardedOptimizer : public ZeroOptimizerBase {
public:
    using Tensor = OwnTensor::Tensor;
    using Dtype  = OwnTensor::Dtype;

    SymmetricShardedOptimizer(const std::vector<Tensor>& model_parameters,
                              std::shared_ptr<dist::comm::ProcessGroupBase> pg,
                              ZeroConfig cfg);

    ~SymmetricShardedOptimizer() override;

    // --- Optimizer interface ---
    void step() override;
    void set_lr(float lr) override { cfg_.lr = lr; }
    void zero_grad() override;

    int64_t step_count() const override { return step_count_; }
    std::vector<Tensor> get_m_buffers() const override { return {shard_m_}; }
    std::vector<Tensor> get_v_buffers() const override { return {shard_v_}; }

    // --- sharded checkpoint ---
    // Writes ShardOwnership::PER_BUCKET plus the bucket length, so a reader can
    // tell this apart from a v1 checkpoint with identical dimensions.
    void save_state(std::ostream& os) override;
    // Accepts BOTH layouts at the same world size. A CONTIGUOUS (v1) checkpoint
    // is resharded on the fly — see reshard_from_contiguous().
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

    // --- compute/comm overlap ---
    // Same contract as v1: call immediately before the FINAL micro-step's
    // loss.backward(). Hooks are inert on every other backward.
    void arm_grad_sync() override;
    bool overlap_enabled() const override { return overlap_active_; }

    // --- cross-group global clipping (see ZeroOptimizerBase) ---
    float reduce_and_measure() override;
    void  set_external_clip_coef(float coef) override { external_clip_coef_ = coef; }
    // Elementwise AdamW: the coefficient is folded into `combined` and applied
    // across the whole valid shard, so coverage is the entire group.
    size_t clip_covered_numel() const override { return total_numel_; }

    // --- staged parameter refresh ---
    // Same public contract as v1 (stage_of is index-aligned with the ctor's
    // parameter vector), but internally the unit of transfer is a BUCKET, not a
    // per-owner piece: a bucket is assigned the earliest stage of any parameter
    // overlapping it, and consuming stage s waits for every bucket assigned to a
    // stage <= s. That keeps the all-gather at full bucket granularity, which is
    // what NVLS wants, while still installing each layer's weights just before
    // the forward reads them.
    void set_param_refresh_stages(const std::vector<int>& stage_of, int num_stages) override;
    void refresh_param_stage(int stage) override;
    void drain_param_refresh() override;
    bool param_refresh_pending() const override { return param_refresh_pending_; }

    // --- layout info ---
    int64_t total_numel()  const override { return static_cast<int64_t>(total_numel_); }
    int64_t padded_numel() const override { return static_cast<int64_t>(padded_numel_); }
    int64_t shard_numel()  const override { return static_cast<int64_t>(S_); }
    int     world_size()   const override { return W_; }
    int     rank()         const override { return rank_; }

    const char* variant_name() const override {
        return symmetric_active_ ? "zero1-v2 reduce-scatter/all-gather (symmetric/NVLS)"
                                 : "zero1-v2 reduce-scatter/all-gather (standard memory)";
    }

    // True when the wire buffers really are ncclMemAlloc'd AND registered as
    // NCCL_WIN_COLL_SYMMETRIC windows on this communicator. False means the run
    // is on the ordinary ring/tree path — check this before attributing a
    // measurement to NVLS.
    bool symmetric_active() const { return symmetric_active_; }
    size_t num_buckets()    const { return buckets_.size(); }
    size_t bucket_elems()   const { return bucket_elems_; }

private:
    struct GroupSpec {
        size_t begin;
        size_t end;
        float  weight_decay;
        float  lr_scale;
    };

    struct ParamRecord {
        Tensor param;
        size_t offset;         // global flat element offset
        size_t numel;
        uint8_t group_id;
        size_t input_index;    // position in the ctor's model_parameters vector
    };

    // One reduce-scatter / all-gather unit. `begin`/`end` are the GLOBAL flat
    // range; every rank contributes and receives `per_rank = (end-begin)/W`
    // elements, and this rank's piece lands at `shard_offset` in shard_grad_ /
    // shard_master_compute_. There is no `owner` — every rank owns part of every
    // bucket, which is the whole point of the layout.
    struct CommBucket {
        size_t begin = 0;
        size_t end   = 0;
        size_t per_rank = 0;       // (end - begin) / W
        size_t shard_offset = 0;   // offset of this rank's piece into the shard
        size_t pending = 0;        // distinct params not yet staged this pass
        size_t pending_init = 0;
        bool   launched = false;   // reduce-scatter issued this pass
        int    refresh_stage = 0;  // earliest stage of any param overlapping it
        std::shared_ptr<Work> rs_work;  // recycled across steps (event reuse)
        std::shared_ptr<Work> ag_work;
    };

    // ---- setup ----
    void build_layout(const std::vector<Tensor>& model_parameters);
    void build_buckets();
    void allocate_buffers();
    Tensor alloc_wire_tensor(size_t numel, Dtype dt);   // symmetric when available
    void register_windows();
    void init_master_from_params();
    void verify_consistency();
    void register_overlap_hooks();

    // ---- per-step ----
    // Place one parameter's gradient into flat_grad_ at its layout offset,
    // converting to the wire dtype IN PLACE when they differ. Shared by the
    // overlap hook and the reference path so both stage identically — a
    // divergence here would show up as a numerical difference between
    // BLU_NO_OVERLAP=0 and =1, which is the one thing that path exists to rule
    // out. See the .cpp for why this converts straight into the destination
    // rather than via as_type().
    void stage_grad_into_flat(const ParamRecord& rec, const Tensor& grad);
    void gather_grads_into_flat();          // reference (non-overlap) path
    void reduce_scatter_grads();            // reference (non-overlap) path
    void cast_shard_grad_to_fp32();
    void stage_grads_direct_fp32();         // W == 1 only (F4): grads -> shard_grad_fp32_
    // ONE elementwise pass that applies 1/W (undoing op=sum), the loss-scaler
    // inverse and the clip coefficient together. Returns false if the step must
    // be skipped (non-finite gradient under a loss scaler).
    bool apply_grad_scale_and_check();
    // Gradient reduction half of step(), factored out so reduce_and_measure() can
    // run it early without step() repeating it.
    void reduce_grads_now();
    // Valid (non-padding) prefix of this rank's shard, in shard elements.
    size_t valid_shard_prefix() const;
    void local_adam_step();
    /* edit */
    // Fused AdamW + AllGather: does local_adam_step() AND all_gather_params()
    // in one kernel per bucket. Returns false without touching anything when
    // the backend or the shape cannot do it, so the caller runs the unfused
    // pair. See the precondition list in the .cpp -- every one of them is a
    // real restriction of the v1 kernel, not a policy choice.
    bool try_fused_adam_and_publish();

    /* edit */
    // The local half of try_fused_adam_and_publish()'s preconditions, asked one
    // call earlier so apply_grad_scale_and_check() knows whether to skip its
    // pass. const because it only reads layout that is fixed at construction.
    bool fused_shape_supports_clip() const;

    // Hand autograd a view into flat_grad_ as each parameter's gradient. See
    // ShardedMuon::install_grad_views() for the rationale and the cost.
    void install_grad_views();

    // Applies a coefficient that was deferred to the fused kernel but never
    // consumed (the backend declined). No-op when nothing is pending.
    void apply_deferred_grad_scale();
    /* edit */
    /* edit */
    void all_gather_params();               // synchronous publish
    void scatter_flat_into_params();
    void broadcast_root_params();

    // ---- overlap ----
    void on_grad_ready(size_t param_index, const Tensor& grad);
    void launch_bucket_rs(size_t bucket_index);
    void finalize_overlap_reduce();

    // ---- staged refresh ----
    void adam_and_publish_staged();
    void adam_shard_range(size_t shard_lo, size_t shard_hi, float bc1, float bc2);
    void cast_master_shard_range(size_t shard_lo, size_t shard_hi);
    void issue_bucket_ag(size_t bucket_index);

    // ---- checkpoint ----
    // Convert a CONTIGUOUS (v1) shard of `src` into this rank's PER_BUCKET shard.
    // Requires a collective: the two schemes assign DIFFERENT global elements to
    // the same rank, so the data genuinely has to move between ranks. Done with
    // one all-gather into a temporary full-size FP32 flat buffer (freed
    // immediately), then a local strided copy-out.
    Tensor reshard_from_contiguous(const Tensor& src_contiguous_shard);

    // Maps a DCP state_type name onto the FP32 shard buffer that holds it.
    // Returns nullptr for an unknown name.
    Tensor* dcp_buffer_for(const std::string& state_type);

    // Global flat range of this rank's `local` shard element.
    size_t shard_to_global(size_t local) const;

    // ---- config / topology ----
    ZeroConfig cfg_;
    std::shared_ptr<dist::comm::ProcessGroupBase> pg_;
    int W_ = 1;
    int rank_ = 0;

    OwnTensor::DeviceIndex device_{};
    Dtype compute_dtype_ = Dtype::Float32;
    Dtype reduce_dtype_  = Dtype::Float32;

    // ---- layout ----
    std::vector<ParamRecord> records_;
    std::vector<GroupSpec>   groups_;
    size_t total_numel_  = 0;
    size_t padded_numel_ = 0;      // rounded up to a multiple of `granule_`
    size_t S_ = 0;                 // padded_numel_ / W  (elements this rank owns)
    size_t granule_ = 1;           // W * (16 / esz): keeps per-rank pieces 16B-aligned
    size_t bucket_elems_ = 0;      // target bucket length, a multiple of granule_
    uint64_t layout_hash_ = 0;

    // ---- buffers ----
    Tensor flat_stage_;            // [padded] compute dtype. AG recv (+RS send when
                                   //          reduce dtype == compute dtype).      WINDOW
    // Gradient staging for the reduce-scatter, in `reduce_dtype_`.
    //
    // When reduce_dtype_ == compute_dtype_ (the default) this is a HANDLE TO THE
    // SAME TENSOR as flat_stage_, reproducing v1's single dual-purpose buffer at
    // no extra memory.
    //
    // Whenever the wire dtype is FORCED away from the param dtype (ReduceDtype::
    // FP32 on a BF16 group, or ReduceDtype::BF16 on an FP32 one) the two differ,
    // gradients cannot share the parameter staging buffer, and this becomes a
    // SEPARATE [padded] allocation in the wire dtype.
    //
    // The aliased default spends NOTHING here, so the extra is the WHOLE new
    // buffer: 4 bytes/param for FP32 (measured 6543 MB on this repo's 1.62 B BF16
    // Muon group), 2 bytes/param for BF16. Not the difference between the two
    // element widths — an earlier comment said "+2 bytes/param" alongside the
    // correct 6.5 GB figure, which are inconsistent; the 6.5 GB is right.
    //
    // That cost is why both forcing modes are opt-in: they are numerical
    // diagnostics, not the production path. FP32 additionally removes a
    // conversion — autograd already hands the hook an FP32 gradient (AutogradMeta
    // upcasts BF16 grads), so the as_type() down-cast disappears on that path;
    // BF16 on an FP32 group instead ADDS one.
    Tensor flat_grad_;             // [padded] reduce dtype.  RS send.             WINDOW
    // Separate RS receive / AG send buffers, allocated ONLY when the collectives
    // cannot run in place (non-NCCL backend, or BLU_ZERO_INPLACE=0). In place,
    // this rank's piece of bucket k is its own slice of the flat buffers,
    // [begin_k + rank*per_rank_k, +per_rank_k), and both roles are played there.
    Tensor shard_grad_;            // [S] reduce dtype.       RS recv.             WINDOW
    Tensor shard_master_compute_;  // [S] compute dtype.      AG send.            WINDOW
    bool   inplace_collectives_ = false;

    // This rank's piece of bucket b, in wire-buffer element coordinates: its own
    // slice of flat_grad_/flat_stage_ in place, its shard offset otherwise.
    size_t wire_offset(const CommBucket& b) const {
        return inplace_collectives_ ? b.begin + static_cast<size_t>(rank_) * b.per_rank
                                    : b.shard_offset;
    }
    char* grad_wire_base();        // flat_grad_  or shard_grad_
    char* param_wire_base();       // flat_stage_ or shard_master_compute_
    Tensor shard_grad_fp32_;       // [S] fp32 (local only)
    Tensor shard_master_;          // [S] fp32 (local only)
    Tensor shard_m_;               // [S] fp32 (local only)
    Tensor shard_v_;               // [S] fp32 (local only)

    // Cross-group clipping state. cached_nsq_ >= 0 means reduce_and_measure()
    // already reduced the gradients and computed ||g||^2 this step, so step() must
    // not redo either. external_clip_coef_ >= 0 overrides the coefficient this
    // instance would derive from its own norm.
    float cached_nsq_          = -1.0f;
    float external_clip_coef_  = -1.0f;

    /* edit */
    // BLU_FUSED_CLIP_ADAMW state, both valid for exactly ONE step and cleared at
    // the end of step() (and on the non-finite early return).
    //
    //   scale_deferred_ : apply_grad_scale_and_check() must NOT run its scale
    //                     pass this step, because the fused kernel will apply
    //                     the coefficient instead.
    //   pending_scale_  : device pointer to that coefficient, or null when the
    //                     coefficient is exactly 1.0 (nothing to apply, and the
    //                     kernel then takes its bit-identical SCALED=false form).
    //                     Points into d_scale_; never owns.
    bool   scale_deferred_ = false;
    const float* pending_scale_ = nullptr;

    // Gradient bucket views (ZeroConfig::grad_bucket_view, BLU_GRAD_VIEW).
    std::vector<Tensor> grad_views_;
    bool grad_views_active_ = false;
    /* edit */
    bool  grads_reduced_       = false;

    float* d_norm_sq_    = nullptr;
    float* d_scale_      = nullptr;
    float* h_pinned_     = nullptr;

    // ---- symmetric memory ----
    bool symmetric_active_ = false;
    std::vector<dist::comm::SymmetricWindow> windows_;

    // ---- state ----
    int64_t step_count_ = 0;
    int     micro_step_ = 0;
    float   last_grad_norm_ = 0.0f;
    bool    last_step_skipped_ = false;
    bool    is_adamw_ = true;

    // ---- overlap state ----
    bool overlap_active_ = false;
    bool armed_ = false;
    std::vector<CommBucket>            buckets_;
    std::vector<std::vector<uint32_t>> param_to_buckets_;
    size_t buckets_launched_ = 0;
    std::vector<bool> grad_fired_;
    cudaEvent_t compute_sync_event_ = nullptr;

    // ---- staged refresh state ----
    bool overlap_param_active_ = false;
    bool param_refresh_pending_ = false;
    int  num_refresh_stages_ = 0;
    int  next_refresh_stage_ = 0;
    std::vector<std::vector<uint32_t>> stage_records_;   // stage -> record idxs
    // Buckets in the order their all-gathers are ISSUED: sorted by
    // (refresh_stage, index). Issue order must be identical on every rank, and
    // the single in-order comm stream then makes "stage s has landed" imply
    // "every earlier stage has landed".
    std::vector<uint32_t> ag_issue_order_;
    cudaEvent_t param_sync_event_ = nullptr;
};

} // namespace zero
} // namespace owt
