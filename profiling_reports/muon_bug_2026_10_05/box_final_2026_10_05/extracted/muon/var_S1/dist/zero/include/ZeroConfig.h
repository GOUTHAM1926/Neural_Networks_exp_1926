#pragma once
//======================================================================
// ZeroConfig.h — configuration surface for the ZeRO sharded optimizer.
//
// One config struct drives ZeRO-1 today and is intentionally shaped so the
// same knobs (reduce dtype, bucketing, strategy) carry over to ZeRO-2/3 later
// without breaking callers. See ZERO_OPTIMIZER_DESIGN.md.
//======================================================================

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include "dtype/Dtype.h"

// offsetof on ShardLayoutDescriptor (see the static_asserts below).
#include <cstddef>

namespace owt {
namespace zero {

//----------------------------------------------------------------------
// Checkpoint alignment granule, in 16-BYTE VECTORS.
//
// Both ZeRO-1 variants pad their flat parameter vector up to a multiple of a
// granule, and v2 additionally cuts its comm buckets on granule boundaries.
// Those boundaries are exactly what a DCP checkpoint names: v1 writes one
// record per state spanning [0, padded_numel), v2 writes one per (state,
// bucket) named "<scope>.b<i>". A resume INTERSECTS the ranges it wants
// against the ranges the checkpoint holds, in one shared global flat
// coordinate system -- which only lines up if both runs agree on where the
// boundaries are.
//
// The granule therefore MUST NOT depend on the world size. It used to be
// `W * vec_elems`, which made padded_numel_ and bucket_elems_ move with W, so
// "b3" meant a different slice of the model at every world size and a resume
// at a new W could fail coverage (or, for v1, leave the padded tail
// uncovered). Whether it happened to work was down to how evenly the group's
// element count divided -- luck, not design.
//
// 3840 = 2^8 * 15, so every world size in {1,2,3,4,5,6,8,10,12,16,20,24,32,
// 48,64,96,128,192,256,384,...} divides it, which is the requirement for the
// per-rank pieces of a bucket to tile it exactly. Padding waste is bounded by
// one granule per group (~120 KB of FP32 state), which is noise against a
// multi-GB optimizer.
//
// BLU_ZERO_CKPT_ALIGN overrides it for a world size that 3840 does not divide
// (7, 9, 11, ...). Checkpoints are only interchangeable between runs that
// agree on this value, so change it only when you must, and keep it fixed for
// the life of a run.
//----------------------------------------------------------------------
inline size_t zero_ckpt_align_vecs() {
    static const size_t v = [] {
        if (const char* e = std::getenv("BLU_ZERO_CKPT_ALIGN"); e && *e) {
            const long n = std::strtol(e, nullptr, 10);
            if (n > 0) return static_cast<size_t>(n);
        }
        return static_cast<size_t>(3840);
    }();
    return v;
}

//----------------------------------------------------------------------
// In-place reduce-scatter / all-gather (v2 and ShardedMuon's RS/AG path).
//
// Each rank's piece of a bucket IS its own slice of the flat buffer, so the
// reduce-scatter can land there (recv = send + rank*count) and the all-gather
// can send from there (send = recv + rank*count). That removes the separate
// [S] shard receive/send buffers — 716 MiB/GPU on the 1.47 B Muon group, 56 MiB
// on the 116 M embedding group — and nothing else changes: measured on NCCL
// 2.30.4 / 8xH200, both forms dispatch to the same symmetric kernels
// (ReduceScatter_LDMC / AllGather_STMC) at the same speed, with identical bytes.
//
// Only used on the NCCL backend. BluCCL's collectives have not been validated
// with aliased send/receive ranges, so it keeps the out-of-place buffers.
//
// BLU_ZERO_INPLACE=0 restores the out-of-place buffers for bisection. Read
// once; it is a construction-time layout choice.
//----------------------------------------------------------------------
inline bool zero_inplace_collectives_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("BLU_ZERO_INPLACE");
        return !(e && *e && std::strtol(e, nullptr, 10) == 0);
    }();
    return on;
}

// Which ZeRO stage's communication schedule to run. Only STAGE_1 (optimizer
// state sharding) is implemented; the enum exists so the API is stable when
// STAGE_2/3 land.
enum class ShardStrategy { STAGE_1, STAGE_2, STAGE_3 };

// The fused-Adam family variant. ZeRO-1 supports Adam and AdamW (they differ
// only by decoupled weight decay, which is one flag into the fused kernel).
enum class AdamVariant { ADAMW, ADAM };

// dtype used on the wire for the gradient reduce-scatter.
//   SAME_AS_PARAM : reduce in the model's compute dtype — half the wire
//                   volume of FP32 when the model is BF16, matches most
//                   production recipes.
//   FP32          : reduce in FP32 for numerically sensitive models (an
//                   UP-cast when the model's compute dtype is narrower).
//   BF16          : reduce in BF16 regardless of compute dtype (a DOWN-cast
//                   when the model's compute dtype is FP32) — diagnostic /
//                   bandwidth-saving knob, trades precision for wire volume
//                   on an otherwise FP32 group.
// FP32 and BF16 both require a genuinely different wire dtype from the
// model's compute dtype to be supported at all: v1 (owt::zero::ShardedOptimizer)
// reduces ONLY in the compute dtype (no separate staging buffer) and rejects
// either request outright unless it's a no-op; v2 (SymmetricShardedOptimizer)
// and ShardedMuon carry the extra flat/owned buffer needed and support both.
enum class ReduceDtype { SAME_AS_PARAM, FP32, BF16 };

// Single resolution point for the enum above, so v1, v2 and Muon cannot drift
// apart on what a given ReduceDtype means. Returns the dtype that actually goes
// on the wire for a group whose parameters are `compute_dtype`.
inline OwnTensor::Dtype resolve_reduce_dtype(ReduceDtype r,
                                             OwnTensor::Dtype compute_dtype) {
    switch (r) {
        case ReduceDtype::FP32: return OwnTensor::Dtype::Float32;
        case ReduceDtype::BF16: return OwnTensor::Dtype::Bfloat16;
        case ReduceDtype::SAME_AS_PARAM:
        default:                return compute_dtype;
    }
}

// How the padded flat buffer is divided among ranks. This is a CHECKPOINT-
// VISIBLE property: the bytes in shard_master_/m_/v_ mean different global
// elements under each scheme, so a checkpoint written by one is silently wrong
// for the other unless resharded. It is recorded in ShardLayoutDescriptor.
//
//   CONTIGUOUS : rank r owns exactly one global range [r*S, (r+1)*S).
//                Used by owt::zero::ShardedOptimizer (v1). This is what makes
//                per-bucket ncclReduce(root=owner) legal — a bucket can be kept
//                inside a single owner's shard.
//
//   PER_BUCKET : the flat buffer is tiled into buckets whose lengths are exact
//                multiples of W, and rank r owns sub-chunk r of EVERY bucket.
//                Rank r's shard is therefore a strided union of `num_buckets`
//                disjoint ranges. Used by owt::zero::SymmetricShardedOptimizer
//                (v2). This is the layout ncclReduceScatter/ncclAllGather
//                already expect (rank-major within the collective's range), so
//                each bucket maps to exactly one collective with no staging
//                copy — which is what allows the NVLS symmetric kernels.
enum class ShardOwnership : int32_t { CONTIGUOUS = 0, PER_BUCKET = 1 };

struct ZeroConfig {
    ShardStrategy strategy = ShardStrategy::STAGE_1;

    // ---- optimizer hyperparameters ----
    AdamVariant variant = AdamVariant::ADAMW;
    float lr           = 1e-3f;
    float beta1        = 0.9f;
    float beta2        = 0.999f;
    float eps          = 1e-8f;
    float weight_decay = 0.0f;

    // GPT-2 / transformer recipe: apply weight decay only to matrices (ndim>=2)
    // and never to 1-D params (biases, LayerNorm gains/betas). When true, the
    // optimizer builds two contiguous parameter groups and passes wd=0 to the
    // 1-D group. When false, a single group with uniform `weight_decay`.
    bool exclude_1d_from_weight_decay = false;

    // ---- correctness / numerics ----
    float max_grad_norm = 0.0f;        // 0 disables global gradient clipping.

    /* edit */
    // Gradients become VIEWS into flat_grad_, so autograd accumulates straight
    // into the wire buffer and the per-hook staging copy disappears along with
    // the per-parameter fp32 gradient tensors.
    //
    // OFF by default: the view carries the WIRE dtype, so on a bf16 group this
    // moves micro-step gradient accumulation from fp32 to bf16. That is a real
    // numerical change, not a scheduling one. It also cannot coexist with
    // overlap_param_refresh (flat_grad_ aliases flat_stage_). Both are checked
    // at construction and reported; the feature never half-applies.
    bool grad_bucket_view = false;
    /* edit */
    int   grad_accum_steps = 1;        // >1: only reduce+step on the last micro-step.
    ReduceDtype reduce_dtype = ReduceDtype::SAME_AS_PARAM;

    // Broadcast rank-0 parameters at construction so every rank starts from an
    // identical model (the standard DDP/ZeRO init contract).
    bool init_broadcast = true;

    // ---- compute/communication overlap (opt-in, W>1 only) ----
    // When true, the gradient reduce is bucketed and launched DURING the final
    // micro-step's backward pass (via per-parameter post-accumulate-grad hooks)
    // instead of one monolithic reduce-scatter after backward. Each bucket is a
    // contiguous flat range kept within a single owner rank's shard and reduced
    // to that owner with ncclReduce, so the resulting grad shard, the Adam
    // update and the param all-gather stay identical to the non-overlap path
    // (modulo BF16 reduction-order rounding). The training loop MUST call
    // arm_grad_sync() immediately before the final micro-step's backward so the
    // hooks fire on the fully-accumulated gradient (they are a no-op on earlier
    // micro-steps). Requires the SEQUENTIAL autograd engine (the default);
    // combining with ExecutionMode::PARALLEL is rejected. Default false keeps
    // the exact serialized behavior for every existing caller.
    bool   overlap_backward_comm = false;
    // Target bucket size (bytes) for the overlapped reduce. Larger buckets =
    // fewer, more bandwidth-efficient collectives but coarser overlap; smaller =
    // finer overlap but more per-collective launch/latency overhead. On a
    // PCIe-only box (no NVLink) prefer larger buckets. Buckets are additionally
    // split at every shard boundary regardless of this value.
    size_t comm_bucket_bytes = 50 * 1024 * 1024;

    // When true, the post-Adam parameter refresh (all-gather + scatter back into
    // the model tensors) is replaced by a STAGED, asynchronous broadcast schedule
    // that overlaps the NEXT forward pass. Each stage is a group of parameters in
    // forward-consumption order; every stage is issued (non-blocking) at the end
    // of step(), and the app consumes stage s — waits for it and copies it into
    // the live model tensors — immediately before the first forward op that reads
    // it. Stage 0 is consumed by zero_grad(); later stages transfer while earlier
    // layers compute.
    //
    // Requires set_param_refresh_stages() to have been called (otherwise the
    // synchronous all-gather path is used) and the app to consume every stage
    // during the forward pass — arm_grad_sync() throws if a stage is still
    // outstanding, since the gradient staging that follows reuses the same flat
    // buffer. Default false keeps the exact synchronous behavior.
    bool overlap_param_refresh = false;

    // ---- reliability / observability ----
    bool verify_consistency = true;    // cross-rank layout check at construction (F2).
    bool verbose = false;              // rank-0 prints layout + per-step diagnostics.
};

// Compact, world-size-independent description of a sharded state layout. Saved
// inside every ZeRO checkpoint so a future run (possibly a different world
// size) can validate or reshard. Kept POD for trivial serialization.
//
// VERSIONING. Version 1 (56 bytes) is what owt::zero::ShardedOptimizer wrote
// before per-bucket ownership existed; it is implicitly ShardOwnership::
// CONTIGUOUS. Version 2 appends `ownership` + `bucket_elems` (72 bytes) so a
// reader can tell the two layouts apart and reshard between them. The version-2
// fields are appended AFTER every version-1 field, in the same order, so a
// version-1 record is a byte-exact prefix of a version-2 record and a reader can
// load the common prefix first and only then decide whether to pull the tail.
struct ShardLayoutDescriptor {
    int64_t magic          = 0x5A45524F31L; // "ZERO1"
    int32_t version        = 2;
    int32_t world_size     = 0;             // W the checkpoint was written with
    int64_t total_numel    = 0;             // unsharded parameter count
    int64_t padded_numel   = 0;             // total_numel rounded up (see ownership)
    int64_t shard_numel    = 0;             // padded_numel / W (elements this rank owns)
    int32_t param_count    = 0;             // number of parameters in flat order
    int32_t num_groups     = 0;             // number of weight-decay/lr groups
    uint64_t layout_hash   = 0;             // hash of ordered (numel,group) + group specs
    // ---- version 2 ----
    // Which global elements this rank's shard_master_/m_/v_ bytes correspond to.
    // Without this a v1 and a v2 checkpoint are indistinguishable — same
    // total_numel, same shard_numel, same layout_hash — but mean different
    // things, and loading one as the other silently trains garbage.
    int32_t ownership      = 0;             // ShardOwnership; 0 == CONTIGUOUS
    int32_t _pad0          = 0;             // keep bucket_elems 8-byte aligned
    // Bucket length in ELEMENTS for PER_BUCKET layouts (0 for CONTIGUOUS).
    // Needed to reconstruct the source tiling when resharding.
    int64_t bucket_elems   = 0;
};

// Byte offsets are part of the on-disk format; pin them so a field reorder or a
// padding change is a compile error rather than a silently unreadable
// checkpoint. 56 is the version-1 size and must stay the offset of `ownership`.
static_assert(sizeof(ShardLayoutDescriptor) == 72, "ShardLayoutDescriptor size is on-disk format");
static_assert(offsetof(ShardLayoutDescriptor, ownership) == 56, "v1 records must be a byte-exact prefix of v2");
static_assert(offsetof(ShardLayoutDescriptor, bucket_elems) == 64, "v2 tail layout changed");

// Size of the version-1 record: the prefix a v2 reader loads before it knows
// whether there is a tail to read.
inline constexpr size_t kShardLayoutDescriptorV1Bytes = 56;

} // namespace zero
} // namespace owt
