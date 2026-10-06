#pragma once
//======================================================================
// ShardedMuon.h — ZeRO Stage-1 for the Muon optimizer.
//
// WHY THIS IS A SEPARATE CLASS AND NOT A ZeroConfig::variant
// ----------------------------------------------------------
// owt::zero::ShardedOptimizer sha rds a FLAT, ELEMENTWISE buffer: every
// parameter is concatenated into one vector, padded to a multiple of W, and
// rank r owns the contiguous global range [r*S, (r+1)*S). That is sound for
// Adam/AdamW because the update is elementwise — element i of the update
// depends only on (param_i, grad_i, m_i, v_i), so ANY partition of the
// elements produces the same result.
//
// Muon's update is NOT elementwise. Its core step is
//
//     U = NewtonSchulz(M)      // M = momentum buffer, viewed as a MATRIX
//
// an iterated matrix polynomial (X <- aX + (bXXᵀ + c(XXᵀ)²)X) whose output at
// any one entry depends on the ENTIRE matrix. A flat contiguous 1/W slice of
// the concatenated buffer generally lands mid-matrix: rank r gets the last
// 300 rows of wQ, all of wK, and the first 900 rows of wV. Orthogonalizing
// those fragments independently is a DIFFERENT operator from orthogonalizing
// wQ, wK and wV — not an approximation of it. So elementwise ZeRO-1 cannot be
// reused for Muon at all.
//
// The sharding that IS compatible is WHOLE-PARAMETER (tensor-level) ownership:
// each matrix is assigned in its entirety to one owner rank, which holds that
// matrix's FP32 master + momentum and runs the full Newton-Schulz on it. This
// is the same decomposition the reference distributed Muon uses, and it still
// delivers ZeRO-1's actual guarantee — optimizer state memory drops by ~1/W —
// while keeping the math bit-identical to single-process Muon.
//
//   per step:
//     grads to owner (avg)                   [see MuonCollective below]
//     global grad-norm all-reduce (+ clip)
//     owner: decoupled WD -> momentum EMA (+Nesterov) -> Newton-Schulz
//            -> apply w/ aspect scale, on its FP32 master
//     owner: cast master -> compute dtype
//     updated params back to every rank      [see MuonCollective below]
//
// WHOLE-PARAMETER OWNERSHIP DOES NOT DICTATE THE COLLECTIVES. The obvious
// implementation is one ncclReduce(root=owner) + one ncclBroadcast(root=owner)
// per parameter, and that is what MuonCollective::REDUCE_BROADCAST does. But
// reduce-scatter/all-gather are ALSO compatible, and are the default: bin-pack
// whole tensors into per-rank contiguous sub-chunks of a bucket, pad each
// sub-chunk to a common length, and the shard boundary falls on tensor
// boundaries by construction — so rank r's reduce-scatter output IS its
// complete set of matrices and Newton-Schulz still sees whole matrices. That
// buys two collectives per bucket instead of two per parameter, and access to
// NCCL's symmetric-memory (NVLS) kernels, which exist for AllGather/AllReduce/
// ReduceScatter but NOT for Reduce or Broadcast.
//
// Parameters Muon skips (everything 1-D, plus any index the caller excludes —
// the token embedding and an untied lm_head, per the standard recipe) take a
// fused AdamW path on the SAME owned shard, so one instance covers a whole
// dtype group.
//
// MEMORY. Persistent per-owned-element state is
//   master 4B + momentum 4B = 8 B/elem/W
// against ShardedOptimizer's 20 B/elem/W, because Muon carries no second moment
// for an orthogonalized parameter. Each of those two is ONE flat allocation with
// a view per owned tensor. There is no third, FP32-gradient buffer: the reduced
// gradients already sit contiguously in the wire dtype and the update consumes
// one tensor at a time, so FP32 is materialized into a single scratch the size of
// the largest owned parameter. On top of that:
//   REDUCE_BROADCAST           + one owned_numel wire-dtype buffer for the
//                              per-parameter ncclReduce destinations, and a
//                              max_numel discard buffer. A broadcast writes
//                              straight into the destination parameter, since
//                              the piece is a whole tensor.
//   REDUCE_SCATTER_ALL_GATHER  + one unsharded padded_numel buffer in the
//                              compute dtype (flat_grad_ aliases flat_stage_
//                              at the default wire dtype). That buffer is the
//                              price of the NVLS path. On NCCL the collectives
//                              run IN PLACE on this rank's slice of it, so
//                              there are no shard buffers; other backends add
//                              2 * S_ of them (see inplace_collectives_).
//
// INTERCHANGEABILITY. Derives from ZeroOptimizerBase, so a training script can
// hold it in the same std::vector<std::unique_ptr<ZeroOptimizerBase>> as v1/v2
// and drive it through the identical arm_grad_sync / set_param_refresh_stages /
// refresh_param_stage / drain_param_refresh / last_grad_norm surface.
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

#include "ZeroConfig.h"      // ReduceDtype, ShardLayoutDescriptor
#include "ZeroErrors.h"
#include "ZeroOptimizerBase.h"

namespace owt {
namespace zero {

// Which published convention scales the orthogonalized update. Newton-Schulz
// returns singular values ~1, so a [A,B] update's ENTRIES have RMS
// 1/sqrt(max(A,B)) — a quantity that shrinks as the matrix grows. The two
// conventions correct for that differently, and they put `lr` on completely
// different scales.
//
//   MATCH_RMS_ADAMW  Moonshot's: gamma <- 0.2 * gamma * sqrt(max(A,B)), which
//     (default)      lifts the entry RMS to 0.2 — what a converged AdamW update
//                    looks like. The consequence is the useful one: `lr` is on
//                    the SAME SCALE as the AdamW arm's, so setting both equal
//                    gives both groups the same step size and the ONLY
//                    difference is the update direction. That is the controlled
//                    experiment, and it is why this is the default here.
//                    Also transpose-invariant (it depends on max(A,B)), so it
//                    does not care whether a Linear stores [in,out] or [out,in].
//
//   ORIGINAL         Keller's modded-nanogpt scaling, sqrt(max(1, A/B)). Entry
//                    RMS stays at 1/sqrt(max(A,B)), so an equivalent lr is
//                    0.2*sqrt(B) times larger — which on this model ranges from
//                    4.5x (wK/wV) to 21.2x (gate_up). There is therefore NO
//                    single lr that converts between the two conventions; use
//                    this one only to reproduce published Muon numbers.
//
//                    CAUTION: unlike MATCH_RMS_ADAMW this is NOT transpose-
//                    invariant, and OwnTensor's nn::Linear stores its weight as
//                    [in_features, out_features] where PyTorch stores
//                    [out_features, in_features]. So A/B here is the reciprocal
//                    of the reference's, and the scale lands on the opposite set
//                    of matrices (2.0 on wK/wV where modded-nanogpt gives 1.0;
//                    1.0 on gate_up where it gives 2.34). This matches the
//                    existing nn::Muon in this repo, which computes the ratio
//                    the same way — kept bug-compatible on purpose rather than
//                    silently diverging from the in-tree optimizer.
enum class MuonAdjustLr { MATCH_RMS_ADAMW, ORIGINAL };

// Which pair of NCCL collectives moves gradients in and parameters out. Both
// implement the SAME whole-parameter ownership and produce the same update; they
// differ only in wire mechanics, so a run may switch between them freely (a
// CHECKPOINT may not — see below).
//
//   REDUCE_SCATTER_ALL_GATHER  (default) Tensors are bin-packed into a flat
//     buffer tiled into buckets of length per_rank*W, rank-major, with each
//     rank's sub-chunk holding only WHOLE tensors padded out to per_rank. That
//     is the layout ncclReduceScatter/ncclAllGather already expect, so one
//     bucket maps to exactly one collective with no staging copy — and unlike
//     Reduce/Broadcast these two have NCCL symmetric-memory (NVLS) device
//     kernels, so they can dispatch to NVLink SHARP. Costs one flat buffer of
//     padded_numel elements in the compute dtype, and ~2-3% padding on a
//     transformer body (the bin-packing is per-bucket, so it balances slightly
//     worse than a global assignment).
//
//   REDUCE_BROADCAST           One ncclReduce(root=owner) + one
//     ncclBroadcast(root=owner) per parameter. No flat buffer and no padding —
//     a broadcast writes straight into the destination tensor — but ~2
//     collectives per parameter instead of ~2 per bucket, and NCCL ships no
//     symmetric kernel for either, so NVLS is unavailable. Retained as the
//     bisection path: it is the formulation this class was first validated
//     against, so a disagreement between the two isolates a layout bug in the
//     bin-packing from a fault in the Muon math itself.
//
// The owner ASSIGNMENT differs between the two (global largest-first vs
// per-bucket largest-first), so the modes are not checkpoint-compatible; the
// mode is folded into layout_hash_ and a mismatched load is rejected rather
// than silently mapping state onto the wrong tensors.
enum class MuonCollective { REDUCE_SCATTER_ALL_GATHER, REDUCE_BROADCAST };

// Hyperparameters + policy for ShardedMuon.
//
// Kept separate from ZeroConfig rather than folded in as another AdamVariant:
// Muon carries hyperparameters Adam has no analogue for (ns_steps, nesterov,
// adjust_lr), has no second moment at all, and its `lr` means something
// different — Newton-Schulz normalizes the update, so lr IS the step size in
// weight space rather than a multiplier on a gradient-scaled quantity. Under
// MATCH_RMS_ADAMW the two are deliberately brought onto one numeric scale, but
// they remain different quantities and a shared struct would blur that.
struct MuonConfig {
    // ---- Muon path (orthogonalized; applied to >=2-D params) ----
    // Under the default MATCH_RMS_ADAMW this is on the SAME scale as the AdamW
    // arm's lr — equal values mean equal step sizes. Under ORIGINAL it is
    // 0.2*sqrt(B) times smaller than the equivalent value; see MuonAdjustLr.
    float lr           = 8e-4f;
    float momentum     = 0.95f;
    // Decoupled, applied to the FP32 master as work *= (1 - lr*wd). Pinned to
    // the AdamW arm's coefficient by the caller so the decay is identical across
    // both groups — which is only meaningful because MATCH_RMS_ADAMW puts the
    // two LRs on one scale.
    float weight_decay = 0.1f;
    int   ns_steps     = 5;       // Newton-Schulz iterations
    bool  nesterov     = true;
    MuonAdjustLr adjust_lr = MuonAdjustLr::MATCH_RMS_ADAMW;

    // ---- AdamW path (every parameter Muon skips) ----
    // adamw_lr is the value at the SAME point on the schedule as `lr` above.
    // set_lr() rescales both together, holding adamw_lr/lr fixed at whatever
    // ratio these two are constructed with, so one schedule drives both.
    float adamw_lr           = 2e-4f;
    float adamw_beta1        = 0.9f;
    float adamw_beta2        = 0.95f;
    float adamw_eps          = 1e-8f;
    float adamw_weight_decay = 0.0f;

    // GPT/Llama recipe: never weight-decay a 1-D parameter. Applies to the
    // AdamW path only — every Muon-path parameter is >=2-D by construction.
    bool exclude_1d_from_weight_decay = true;

    // Positions in the constructor's parameter vector that must take the AdamW
    // path even though they are >=2-D. The standard Muon recipe excludes the
    // token embedding and the output head: both are "vocabulary" matrices whose
    // rows are near-independent per-token directions rather than a
    // transformation between two dense feature spaces, which is what makes
    // spectral normalization the right prior in the first place.
    std::vector<size_t> adamw_param_indices;

    // ---- wire mechanics ----
    MuonCollective collective = MuonCollective::REDUCE_SCATTER_ALL_GATHER;

    // Target number of reduce-scatter/all-gather buckets (ignored under
    // REDUCE_BROADCAST). Trades three things off at once:
    //   fewer  -> better bin-packing balance (less padding on the wire), fewer
    //             collectives, but a larger minimum flat buffer and coarser
    //             overlap: a bucket cannot reduce until ALL its gradients exist.
    //   more   -> finer overlap and refresh staging, worse balance. Each bucket
    //             is packed across all W ranks independently, so a bucket with
    //             too few tensors cannot balance at all.
    // Reduced at construction until the padding it costs fits under
    // max_padding_frac, so a too-fine request degrades to a coarser layout
    // instead of to a wire full of zeros.
    int target_buckets = 6;

    /* edit */
    // Pipeline the parameter publish against Newton-Schulz: issue bucket k's
    // all-gather as soon as this rank's owned tensors in bucket k are updated and
    // cast, instead of after every owned tensor in the model. Numerics-neutral --
    // same kernels, same order, only the launch schedule differs -- so it is ON
    // by default. BLU_MUON_PIPELINE=0 restores the sequential publish, which is
    // the shape every earlier measurement was taken in and therefore the right
    // thing to bisect against. See ShardedMuon::update_owned().
    bool pipeline_publish = true;

    // Hand autograd a view into flat_grad_ as each parameter's gradient, so the
    // backward accumulates straight into the wire buffer and the per-hook staging
    // copy disappears along with the per-parameter fp32 gradient tensors.
    //
    // OFF by default because it is NOT numerics-neutral: the view is in the wire
    // dtype, so micro-step accumulation moves from fp32 to bf16 on a bf16 group.
    // See ShardedMuon::install_grad_views() for the full list of conditions it
    // declines under (it never silently half-applies).
    bool grad_bucket_view = false;

    // Fuse the Muon parameter publish: one kernel per bucket does
    // `master -= lr*adjust * U`, the fp32->wire cast, and the multicast store
    // into the all-gather's destination buffer, replacing muon_apply_cuda +
    // cast_master_into_shard_one + issue_bucket_ag for that bucket.
    //
    // This is the bf16 publish the fused-AdamW path never got: that kernel is
    // fp32-only, so on this model it reaches G2's 9,216 elements, while this one
    // reaches the Muon group -- 92% of the parameters.
    //
    // It sits AFTER Newton-Schulz and AFTER the non-finite guard, deliberately.
    // Fusing into the NS epilogue itself is a different and much worse idea: the
    // guard inspects the WHOLE update before deciding, and a rank-local reject
    // cannot be repaired once tiles have been published collectively.
    //
    // OFF by default. It writes the published parameters, so a mistake here is
    // silently wrong weights rather than a slow step -- which is exactly what the
    // last defect on this path produced.
    bool fused_publish = false;
    /* edit */

    // Ceiling on bin-packing waste, as a fraction of the real element count.
    // Tensor-count heuristics alone do not bound this: a transformer bucket can
    // hold 18 tensors across 8 ranks and still pad 36%, because one gate_up
    // matrix is larger than the ideal per-rank share and every other rank pads
    // up to it. The padding rides BOTH collectives every step, so it is bounded
    // directly and target_buckets is halved until it fits.
    float max_padding_frac = 0.05f;

    // ---- correctness / numerics ----
    // Global-norm gradient clipping. APPLIED TO THE AdamW-PATH TENSORS ONLY.
    //
    // It is tempting to apply it to everything on the grounds that Newton-Schulz
    // Frobenius-normalizes its input, so a common scale factor cancels. That is
    // true of the NS input, but the clip lands on the GRADIENT, which first goes
    // through a stateful EMA:
    //
    //     m_t = mu*m_(t-1) + (1-mu)*c_t*g_t ,   U_t = NewtonSchulz(m_t)
    //
    // A constant c cancels exactly (m_t = c*m'_t). A time-varying c does not — it
    // reweights this step's gradient against ~1/(1-mu) steps of history, changing
    // the DIRECTION of m_t. In practice c varies a lot (measured 0.042..1.000 over
    // 34 steps on this repo's Muon group), so clipping the Muon path is a large,
    // silent change to the update rather than a no-op. The published recipes do
    // not clip orthogonalized parameters; neither does this.
    //
    // The norm is still computed over every owned tensor and reported through
    // last_grad_norm(), so the diagnostic is unaffected. See
    // compute_norm_and_clip() in the .cpp.
    float max_grad_norm = 0.0f;
    ReduceDtype reduce_dtype = ReduceDtype::SAME_AS_PARAM;

    bool init_broadcast     = true;   // sync all ranks to rank 0 at construction
    bool verify_consistency = true;   // cross-rank layout check at construction

    // ---- compute/communication overlap (opt-in, W>1 only) ----
    // Reduce each parameter's gradient to its owner from a post-accumulate-grad
    // hook during the final micro-step's backward, instead of one pass after it.
    // The training loop MUST call arm_grad_sync() immediately before that
    // backward. Requires the SEQUENTIAL autograd engine.
    bool overlap_backward_comm = false;
    // Publish updated parameters as consumption-ordered stages of asynchronous
    // broadcasts issued at the end of step(), consumed by the next forward.
    // Requires set_param_refresh_stages().
    bool overlap_param_refresh = false;

    bool verbose = false;             // rank-0 layout + per-step diagnostics
};

class ShardedMuon : public ZeroOptimizerBase {
public:
    using Tensor = OwnTensor::Tensor;
    using Dtype  = OwnTensor::Dtype;

    // model_parameters : the trainable tensors (e.g. Module::parameters()).
    //                    NOT handed to the base optimizer — this class manages
    //                    sharded state itself so the base never allocates
    //                    full-size FP32 masters.
    // pg               : the data-parallel process group (may be null iff W==1).
    ShardedMuon(const std::vector<Tensor>& model_parameters,
                std::shared_ptr<dist::comm::ProcessGroupBase> pg,
                MuonConfig cfg);

    ~ShardedMuon() override;

    // --- Optimizer interface ---
    void step() override;

    // Sets the MUON learning rate. The AdamW-path LR is rescaled with it,
    // preserving the adamw_lr/lr ratio the optimizer was constructed with, so
    // the caller drives one schedule and both paths follow it.
    void set_lr(float lr) override;

    void zero_grad() override;

    int64_t step_count() const override { return step_count_; }
    std::vector<Tensor> get_m_buffers() const override { return owned_m_; }
    std::vector<Tensor> get_v_buffers() const override { return owned_v_; }

    // --- sharded checkpoint ---
    // Writes this rank's owned master + momentum (+ Adam v for AdamW-path
    // params) and a layout descriptor. Same-world-size load only.
    void save_state(std::ostream& os) override;
    void load_state(std::istream& is) override;

    // --- DCP flat-shard interface (OwnTensor::dcp::ShardedOptimizerState) ---
    // Unlike save_state/load_state (which reject any world-size change), this
    // interface reports each owned tensor's state as its OWN complete,
    // self-contained range (whole-parameter ownership degenerates to a trivial
    // "shard" that is the whole thing), so DCP's intersect-on-load lets a
    // resume at a DIFFERENT world size work: the new run derives its own owner
    // assignment and asks for exactly the ranges it needs. See ShardedMuon.cpp
    // for the naming scheme.
    std::vector<OwnTensor::dcp::OptimStateShardSpec> dcp_state_shard_specs() override;
    Tensor dcp_get_state_shard(const std::string& state_type,
                               const std::string& param_fqn) override;
    void   dcp_set_state_shard(const std::string& state_type,
                               const std::string& param_fqn,
                               const Tensor& flat_slice) override;
    void   dcp_after_load() override;
    // Republish the owned FP32 masters into the live model tensors — the tail
    // of load_state(), reused by dcp_after_load().
    void   republish_owned_masters();

    // --- accessors ---
    float get_lr() const override { return cfg_.lr; }
    float get_adamw_lr() const { return cfg_.adamw_lr; }
    void  set_max_grad_norm(float m) override { cfg_.max_grad_norm = m; }
    float last_grad_norm() const override { return last_grad_norm_; }
    bool  last_step_skipped() const override { return last_step_skipped_; }

    // Number of Newton-Schulz updates the divergence guard rejected over the
    // whole run (see muon_has_nonfinite_cuda). Nonzero is not automatically a
    // bug — see the note on skipped_ns_updates_ — but a rising count is.
    int64_t rejected_ns_updates() const { return rejected_ns_updates_; }

    // ---- NS diagnostics (per step) -----------------------------------------
    // DIAGNOSTIC ONLY: reset at the top of every update_owned(), so these
    // describe the step just taken. ns_max_update_ratio() is
    // ||update||_F / sqrt(min(rows,cols)) -- Newton-Schulz returns singular
    // values ~1, so a healthy update sits at ~1.0. The divergence guard only
    // rejects elements >1000 in magnitude, which is ~64000x the expected entry
    // scale, so a ratio well above 1 is an update that passed the guard and was
    // applied anyway.
    int64_t ns_rejected_last_step() const { return ns_rejected_step_; }
    float   ns_max_update_ratio()   const { return ns_max_ratio_; }

    // --- overlap ---
    void arm_grad_sync() override;
    bool overlap_enabled() const override { return overlap_active_; }

    // --- cross-group global clipping (see ZeroOptimizerBase) ---
    float reduce_and_measure() override;
    void  set_external_clip_coef(float coef) override { external_clip_coef_ = coef; }
    // Only the AdamW-path tensors are scaled — compute_norm_and_clip() skips
    // every record with use_muon. In the usual partition (the trainer hands
    // this optimizer nothing BUT Muon-eligible matrices) this returns 0.
    size_t clip_covered_numel() const override {
        size_t n = 0;
        for (const auto& r : records_) if (!r.use_muon) n += r.numel;
        return n;
    }

    void set_param_refresh_stages(const std::vector<int>& stage_of, int num_stages) override;
    void refresh_param_stage(int stage) override;
    void drain_param_refresh() override;
    bool param_refresh_pending() const override { return param_refresh_pending_; }

    // --- layout info (tests / diagnostics) ---
    // shard_numel is the number of REAL elements THIS rank owns and updates,
    // which varies per rank: whole tensors are indivisible, so the partition is
    // balanced rather than exactly equal.
    //
    // padded_numel is the flat buffer's length. Under REDUCE_BROADCAST it equals
    // total_numel (tensor-level ownership never pads — there is no flat buffer).
    // Under REDUCE_SCATTER_ALL_GATHER it is the sum of per_rank*W over buckets,
    // so the excess over total_numel is exactly the bin-packing padding that
    // rides the wire.
    int64_t total_numel()  const override { return static_cast<int64_t>(total_numel_); }
    int64_t padded_numel() const override {
        return static_cast<int64_t>(rsag() ? padded_numel_ : total_numel_);
    }
    int64_t shard_numel()  const override { return static_cast<int64_t>(owned_numel_); }
    int     world_size()   const override { return W_; }
    int     rank()         const override { return rank_; }

    // Padded elements in this rank's shard buffer (RS/AG only; 0 otherwise).
    int64_t shard_padded_numel() const { return static_cast<int64_t>(S_); }
    size_t  num_buckets() const { return buckets_.size(); }

    const char* variant_name() const override {
        return rsag() ? "zero1-muon whole-parameter reduce-scatter/all-gather"
                      : "zero1-muon whole-parameter reduce/broadcast";
    }

private:
    // One managed parameter. `owner` holds its optimizer state and performs its
    // update; every other rank only sends a gradient and receives the result.
    struct ParamRecord {
        Tensor param;              // handle to the model tensor (shared storage)
        size_t numel     = 0;
        int    owner     = 0;
        bool   use_muon  = false;  // false -> AdamW path
        size_t input_index = 0;    // position in the ctor's vector
        int    stage     = 0;      // refresh stage
        // Index into the owner's owned_* vectors; SIZE_MAX on non-owners.
        size_t owned_slot = static_cast<size_t>(-1);
        // Recycled Work handles (event reuse across steps).
        std::shared_ptr<Work> reduce_work;
        std::shared_ptr<Work> bcast_work;

        // --- REDUCE_SCATTER_ALL_GATHER layout (unused under REDUCE_BROADCAST) ---
        // Where this tensor sits in the padded flat buffer. By construction it
        // lies entirely inside owner's sub-chunk of exactly one bucket, which is
        // what keeps Newton-Schulz operating on a whole matrix.
        size_t flat_offset  = 0;
        // Where it sits in THIS rank's shard buffer. Meaningful only on the
        // owner; equals flat_offset - (bucket.begin + owner*bucket.per_rank)
        // + bucket.shard_offset.
        size_t shard_offset = 0;
        size_t bucket       = 0;   // index into buckets_
    };

    // One reduce-scatter / all-gather unit. Its flat range is exactly
    // [begin, begin + per_rank*W), which is the rank-major layout both
    // collectives already expect — so a bucket maps to one collective with no
    // staging copy, and the whole range is NVLS-eligible.
    struct CommBucket {
        size_t begin        = 0;   // flat element offset (inclusive)
        size_t end          = 0;   // = begin + per_rank * W
        size_t per_rank     = 0;   // padded elements owned by each rank here
        size_t shard_offset = 0;   // offset of this rank's piece into the shard
        size_t pending      = 0;   // params not yet staged into flat this pass
        size_t pending_init = 0;   // pending reset value
        bool   launched     = false;
        int    refresh_stage = 0;  // earliest stage of any param in the bucket
        std::vector<uint32_t> records;   // record indices, any rank
        std::shared_ptr<Work> rs_work;   // recycled across steps (event reuse)
        std::shared_ptr<Work> ag_work;
    };

    // ---- setup ----
    void build_layout(const std::vector<Tensor>& model_parameters);
    void assign_owners();                  // REDUCE_BROADCAST: global largest-first
    void build_bucketed_layout();          // RS/AG: bin-pack + flat/shard offsets
    void allocate_owned_state();
    // Back owned_v_ with flat_v_ on first use. Deferred because the AdamW
    // second moment is only touched by the AdamW path, and the usual partition
    // hands this optimizer nothing but Muon-eligible matrices.
    void allocate_v_state();
    void allocate_flat_buffers();          // RS/AG only
    // Allocate one wire buffer, from the NCCL symmetric allocator when this
    // group can use it and the ordinary caching allocator otherwise.
    Tensor alloc_wire_tensor(size_t numel, Dtype dt);
    void register_windows();               // RS/AG + symmetric only
    void init_master_from_params();
    void verify_consistency();
    void broadcast_root_params();
    void register_overlap_hooks();

    bool rsag() const { return cfg_.collective == MuonCollective::REDUCE_SCATTER_ALL_GATHER; }

    // ---- gradient reduce (REDUCE_BROADCAST) ----
    void launch_reduce(size_t ridx, const Tensor& grad);
    void on_grad_ready(size_t ridx, const Tensor& grad);
    void reduce_all_grads_reference();     // non-overlap path
    void finalize_reduces();               // mop-up + wait + comm->compute edge

    // ---- gradient reduce-scatter (RS/AG) ----
    // Copy one parameter's local gradient into its slot in the flat buffer,
    // casting to the wire dtype if needed, and launch the bucket once every
    // parameter in it has been staged.
    void stage_grad_into_flat(size_t ridx, const Tensor& grad);
    void on_grad_ready_rsag(size_t ridx, const Tensor& grad);
    void stage_all_grads_reference();      // non-overlap path
    void issue_bucket_rs(size_t bidx);
    void finalize_reduce_scatter();        // mop-up + wait + comm->compute edge

    // ---- owned gradients (both collective modes) ----
    // The reduced gradients live in the WIRE dtype, contiguously: in shard_grad_
    // at rec.shard_offset under RS/AG, in flat_owned_grad_ at
    // owned_grad_offset_[slot] under reduce/broadcast. Nothing up-casts all of
    // them at once any more.
    //
    // owned_grad_fp32() returns slot `slot`'s gradient as FP32, multiplied by
    // grad_prescale_ and -- when apply_clip and the record is on the AdamW path
    // -- by the pending clip coefficient. The result is a view of grad_scratch_
    // and is therefore valid only until the NEXT call: update_one() consumes one
    // tensor completely before asking for the next, which is what makes a single
    // scratch sufficient.
    Tensor owned_grad_fp32(size_t slot, bool apply_clip);
    // Sum of squares over every owned gradient (prescaled, pre-clip), added to
    // `d_out`. Reads the wire-dtype gradients where the collective left them,
    // one launch per owned tensor in slot order; nothing is up-cast into
    // grad_scratch_ (update_one() does that once, when it needs the FP32 copy).
    void accumulate_owned_grad_norm(float* d_out);

    // ---- wire-buffer addressing (RS/AG) ----
    // Where this rank's reduced gradients land and where its updated parameters
    // are cast for the all-gather. In place: its own slice of flat_grad_ /
    // flat_stage_, addressed by flat offset. Otherwise: shard_grad_ /
    // shard_param_, addressed by shard offset. The same offset serves both
    // sides because a tensor sits at the same element position in each.
    size_t wire_offset(const ParamRecord& rec) const {
        return inplace_collectives_ ? rec.flat_offset : rec.shard_offset;
    }
    size_t wire_offset(const CommBucket& b) const {
        return inplace_collectives_ ? b.begin + static_cast<size_t>(rank_) * b.per_rank
                                    : b.shard_offset;
    }
    char* grad_wire_base();    // flat_grad_  or shard_grad_
    char* param_wire_base();   // flat_stage_ or shard_param_

    // ---- parameter all-gather (RS/AG) ----
    void cast_master_into_shard();         // owner: FP32 master -> shard, compute dtype
    void cast_master_into_shard_one(size_t slot);   // one owned tensor's slice
    void install_grad_views();             // grads become views into flat_grad_
    void issue_bucket_ag(size_t bidx);
    void all_gather_params_sync();
    // Copy one bucket's slice of the gathered flat buffer out into the live
    // parameter tensors. Unlike a broadcast (which lands directly in the
    // parameter) an all-gather writes the flat buffer, so publishing needs this
    // scatter after the collective has landed.
    void scatter_bucket_into_params(size_t bidx);

    // ---- update ----
    bool compute_norm_and_clip();          // false -> step skipped (non-finite)
    // Gradient reduce (either collective mode) + FP32 fan-out, split out of step()
    // so reduce_and_measure() can run it during the cross-group norm phase.
    void reduce_grads_now();
    void update_owned();                   // Muon / AdamW on this rank's tensors
    // `defer` non-null: do everything EXCEPT the final apply, and describe the
    // pending apply+publish in *defer instead. Returns true when a descriptor was
    // produced (always, on the RS/AG path -- a rejected or AdamW-path tensor is
    // expressed as alpha == 0, i.e. publish the master unchanged, so that every
    // rank still contributes the same number of segments).
    bool update_one(size_t slot,
                    dist::comm::ProcessGroupBase::MuonPublishSeg* defer = nullptr);          // a single owned parameter

    // ---- publish ----
    void publish_sync();                   // blocking broadcast of every param
    void issue_refresh_stage(int stage);   // async broadcasts for one stage

    // ---- config / topology ----
    MuonConfig cfg_;
    std::shared_ptr<dist::comm::ProcessGroupBase> pg_;
    int W_ = 1;
    int rank_ = 0;
    float adamw_lr_ratio_ = 1.0f;          // cfg_.adamw_lr / cfg_.lr, fixed at ctor

    OwnTensor::DeviceIndex device_{};
    Dtype compute_dtype_ = Dtype::Float32; // model param dtype
    Dtype reduce_dtype_  = Dtype::Float32; // gradient wire dtype

    // ---- layout ----
    std::vector<ParamRecord> records_;     // every managed parameter, all ranks
    std::vector<size_t>      owned_;       // record indices this rank owns
    size_t total_numel_ = 0;
    size_t owned_numel_ = 0;
    size_t max_numel_   = 0;               // largest single parameter
    uint64_t layout_hash_ = 0;

    // ---- owned optimizer state (index-aligned with owned_) ----
    //
    // FLAT BACKING, PER-TENSOR VIEWS. Each state buffer is ONE allocation; the
    // vectors below hold views into it (shared storage, non-zero storage offset)
    // carrying the parameter's own shape, so Newton-Schulz still sees a matrix
    // and every existing call site is unchanged. This is the layout
    // SymmetricShardedOptimizer already uses.
    //
    // Why not one allocation per owned tensor: SizeClass::round_size quantizes
    // EVERY allocation to a size class and hands the whole block to the tensor,
    // so 72 owned matrices x 3 state buffers pay that rounding 216 times. On the
    // 604 M-param Muon group that is 2,016 MiB of internal fragmentation under
    // the default classes (ZERO1_MEMORY_AUDIT.md 3); one allocation per buffer
    // rounds once. It also cuts 216 allocator round-trips at construction and
    // gives the multi_tensor_* kernels one contiguous range instead of 72.
    //
    // Slot offsets are padded to kStateAlign bytes so every view starts on the
    // same boundary a dedicated allocation would have. The optimizer kernels
    // (muon_*, fused_adam, multi_tensor_*) issue float4 loads and a view landing
    // mid-16-bytes would fault.
    static constexpr size_t kStateAlign = 256;   // bytes; cudaMalloc's own guarantee
    Tensor flat_master_;                   // [master_numel_] FP32
    Tensor flat_m_;                        // [master_numel_] FP32
    Tensor flat_v_;                        // [v_numel_] FP32; AdamW path, allocated lazily
    std::vector<Tensor> owned_master_;     // views into flat_master_
    std::vector<Tensor> owned_m_;          // views into flat_m_
    std::vector<Tensor> owned_v_;          // views into flat_v_; invalid until allocated
    std::vector<size_t> owned_offset_;     // slot -> element offset in flat_master_/flat_m_
    std::vector<size_t> owned_v_offset_;   // slot -> element offset in flat_v_; SIZE_MAX on the Muon path
    size_t master_numel_    = 0;           // owned_numel_ + inter-slot alignment padding
    size_t v_numel_         = 0;           // same, over the AdamW-path slots only
    size_t max_owned_numel_ = 0;           // largest single OWNED parameter

    // ---- owned gradients ----
    //
    // There is NO full-width FP32 gradient buffer. The reduced gradients already
    // sit contiguously in the wire dtype -- in shard_grad_ under RS/AG, in
    // flat_owned_grad_ under reduce/broadcast -- and update_one() consumes ONE
    // tensor at a time, so the FP32 copy is a single max_owned_numel_ scratch
    // rather than a second owned_numel_-wide buffer: 1,152 MiB -> 48 MiB on the
    // reference config (ZERO1_MEMORY_AUDIT.md 6, P1). The grad norm, the only
    // consumer that used to need all of them live at once, streams through the
    // same scratch -- see accumulate_owned_grad_norm().
    Tensor flat_owned_grad_;               // [grad_numel_] reduce dtype; REDUCE_BROADCAST only
    std::vector<Tensor> owned_grad_;       // views into flat_owned_grad_; REDUCE_BROADCAST only
    std::vector<size_t> owned_grad_offset_;// slot -> element offset in flat_owned_grad_
    size_t grad_numel_      = 0;
    Tensor grad_scratch_;                  // [max_owned_numel_] FP32 staging for one gradient

    // Factor applied to every gradient as it is materialized: 1/W for the SUM
    // reduce-scatter (see issue_bucket_rs) times any loss-scaler unscale. Both
    // used to be separate in-place passes over the full FP32 copy that no longer
    // exists, so they ride the materialization instead. Host-side, because the
    // kernel that applies it (muon_scale_inplace_cuda) takes a host scalar.
    float  grad_prescale_ = 1.0f;
    // Gradient-clip coefficient for this step, and whether it still has to be
    // applied. compute_norm_and_clip() used to scale the FP32 gradients in place;
    // with them materialized lazily there is nothing in place to scale, so
    // owned_grad_fp32() applies it as it hands each AdamW-path tensor to
    // update_one().
    float  clip_coef_     = 1.0f;
    bool   clip_pending_  = false;

    // Discard buffer for reduces this rank does not own (sized max_numel_).
    // REDUCE_BROADCAST only.
    Tensor scratch_recv_;

    // ---- RS/AG buffers and layout ----
    std::vector<CommBucket> buckets_;
    /* edit */
    // bucket index -> this rank's owned slots in that bucket (ascending). Empty
    // for a bucket this rank owns nothing in, which is normal and does not stop
    // that bucket's all-gather from being issued at its turn in the sequence.
    std::vector<std::vector<uint32_t>> bucket_owned_slots_;
    // Handshake between update_owned() and all_gather_params_sync(): true when
    // the former already issued every bucket's all-gather. Consumed (reset) by
    // the latter, so a path that skips update_owned() still issues for itself.
    bool ag_pipelined_ = false;
    // Gradient bucket views. grad_views_ keeps a handle on each view so the
    // storage stays referenced even if a caller later replaces a parameter's
    // gradient; grad_views_active_ is the single predicate the hot paths test.
    std::vector<Tensor> grad_views_;
    bool grad_views_active_ = false;

    // Fused publish state. publish_up_ keeps each pending Newton-Schulz result
    // alive until the batched kernel has consumed it -- update_one no longer
    // applies it immediately, so the caching allocator must not reclaim it.
    // Cleared per bucket.
    bool fused_publish_active_ = false;
    bool fused_publish_declined_ = false;
    // Set when the fused publish replaced the all-gather for this step; consumed
    // by all_gather_params_sync(), which then only orders the scatter.
    bool ag_fused_done_ = false;
    std::vector<Tensor> publish_up_;
    std::vector<dist::comm::ProcessGroupBase::MuonPublishSeg> publish_segs_;
    /* edit */
    size_t S_             = 0;   // padded elements this rank owns across buckets
    size_t padded_numel_  = 0;   // = sum over buckets of per_rank*W
    size_t buckets_launched_ = 0;
    // Every rank stages its FULL local gradient set here, then reduce-scatters.
    // Aliases flat_stage_ whenever the wire dtype equals the compute dtype, which
    // is the default — the gradient bytes are dead by the time the update writes
    // parameters back into the same buffer.
    Tensor flat_grad_;
    Tensor flat_stage_;   // all-gather destination; scattered into live params
    // Separate [S_] reduce-scatter destination (wire dtype) and all-gather
    // source (compute dtype). Allocated only when the collectives cannot run in
    // place; otherwise both roles are played by this rank's slice of the flat
    // buffers and these stay invalid.
    Tensor shard_grad_;
    Tensor shard_param_;
    bool inplace_collectives_ = false;

    // NCCL symmetric memory. Without it the reduce-scatter/all-gather still run,
    // just on the ordinary ring/tree kernels — i.e. the collective-count win
    // survives but the NVLS win does not, which is most of the reason this mode
    // exists. Requires the process group to have been created with
    // MemoryMode::Symmetric.
    bool symmetric_active_ = false;
    std::vector<dist::comm::SymmetricWindow> windows_;

    // Keeps the send-side gradient storage alive from launch until wait. Holds
    // the parameter's own gradient tensor when it is already contiguous and in
    // the wire dtype (no copy), or the cast/contiguous temporary otherwise.
    std::vector<Tensor> reduce_keepalive_;

    // Cross-group clipping state (see ZeroOptimizerBase::reduce_and_measure).
    float cached_nsq_         = -1.0f;
    float external_clip_coef_ = -1.0f;
    bool  grads_reduced_      = false;

    // persistent tiny GPU/host scalars for the norm
    float* d_norm_sq_     = nullptr;
    float* h_norm_pinned_ = nullptr;
    // Per-block partials for kernels::sumsq_accumulate (kSumsqPartials floats).
    float* d_sumsq_partials_ = nullptr;

    // ---- state ----
    int64_t step_count_ = 0;
    float   last_grad_norm_ = 0.0f;
    bool    last_step_skipped_ = false;
    // Newton-Schulz updates rejected by the non-finite guard. The bf16 NS
    // iteration only provably converges when every singular value of the
    // Frobenius-normalized input is <= 1, which bf16 GEMM rounding can tip over
    // for an ill-conditioned gradient; the affected parameter is left unchanged
    // for that step rather than corrupted. A handful over a long run is
    // expected, a steady stream means the gradients are pathological.
    int64_t rejected_ns_updates_ = 0;

    // Per-step NS diagnostics; see ns_rejected_last_step()/ns_max_update_ratio().
    int64_t ns_rejected_step_ = 0;
    float   ns_max_ratio_     = 0.0f;

    // ---- overlap state ----
    bool overlap_active_ = false;
    bool armed_ = false;
    std::vector<bool> grad_fired_;         // per record, this pass
    cudaEvent_t compute_sync_event_ = nullptr;

    // ---- staged refresh state ----
    bool overlap_param_active_   = false;
    bool param_refresh_pending_  = false;
    int  num_refresh_stages_ = 0;
    int  next_refresh_stage_ = 0;
    std::vector<std::vector<uint32_t>> stage_records_;  // stage -> record idxs
    // RS/AG: stage -> bucket indices whose gather must land before that stage's
    // layers run. A bucket is pinned to the EARLIEST stage any of its parameters
    // belongs to, since one collective publishes all of them at once.
    std::vector<std::vector<uint32_t>> stage_buckets_;
    cudaEvent_t param_sync_event_ = nullptr;
};

} // namespace zero
} // namespace owt
