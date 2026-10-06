//======================================================================
// SymmetricShardedOptimizer.cpp — ZeRO Stage-1 (v2).
// Bucketed ncclReduceScatter + ncclAllGather over NCCL symmetric memory.
// See SymmetricShardedOptimizer.h for the design rationale.
//======================================================================

#include "SymmetricShardedOptimizer.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include "core/Views/ViewUtils.h"     // compute_strides (gradient bucket views)
#include <numeric>

#include <cuda_runtime.h>

#include "device/DeviceCore.h"
#include "device/DeviceTransfer.h"
#include "ops/helpers/MultiTensorKernels.h"
#include "ops/helpers/GradNormKernels.h"
#include "ops/helpers/ConversionKernels.cuh"
#include "core/Serialization.h"
#include "autograd/Hooks.h"
#include "autograd/Engine.h"
#include "NcclSymmetricAllocator.hpp"
#include "BlucclSymmetricAllocator.hpp"

namespace owt {
namespace zero {

using OwnTensor::Tensor;
using OwnTensor::Dtype;
using OwnTensor::Shape;
using OwnTensor::TensorOptions;
namespace dev = OwnTensor::device;
namespace ocuda = OwnTensor::cuda;

namespace {

inline size_t elem_size(Dtype d) { return Tensor::dtype_size(d); }

inline Shape flat_shape(size_t n) { return Shape{{1, static_cast<int64_t>(n)}}; }

inline void d2d(void* dst, const void* src, size_t bytes, OwnTensor::DeviceIndex di) {
    if (bytes == 0) return;
    dev::copy_memory(dst, di.device, src, di.device, bytes);
}

inline void zero_async(void* ptr, size_t bytes) {
    if (bytes == 0) return;
    cudaMemsetAsync(ptr, 0, bytes, ocuda::getCurrentStream());
}

inline void sync_compute() { cudaStreamSynchronize(ocuda::getCurrentStream()); }

inline size_t round_up(size_t v, size_t m) { return ((v + m - 1) / m) * m; }

// Wait on an overlapped collective and turn a FAILURE into an exception.
//
// Work::wait() reports NCCL asynchronous errors (via ncclCommGetAsyncError) and
// CUDA errors through its return value. The reference paths check their result_t
// and throw; the OVERLAP paths — the production paths — used to discard this.
//
// A discarded error is silent corruption, not a crash: the collective never wrote
// its destination, so shard_grad_ (or flat_stage_) keeps the PREVIOUS step's bytes
// for that bucket, Adam consumes them, and training continues with a wrong
// gradient for one bucket-sized slice of the model. Nothing is logged. That is the
// failure mode that surfaces tens of thousands of steps later as an unexplained
// divergence with no evidence attached, so it must be loud and immediate.
inline void wait_or_throw(const std::shared_ptr<Work>& w, const char* what, size_t index) {
    if (!w) return;
    if (w->wait()) return;
    throw ZeroCollectiveError(
        std::string("overlapped ") + what + " #" + std::to_string(index) +
        " failed (nccl=" + ncclGetErrorString(w->get_ncclStatus()) +
        ", cuda=" + cudaGetErrorString(w->get_lastCudaError()) +
        "). The destination buffer still holds the previous step's data, so the "
        "step is aborted rather than applied.");
}


// F1. Text for ZeroNonFiniteGradError: which optimizer and group, how big, which rank and
// step, so the log names the failing group. Same wording as v1's
// ShardedOptimizer::nonfinite_grad_msg. File-local on purpose: a new member function would
// change the header, and every trainer that includes it would need a rebuild.
std::string nonfinite_grad_msg(float nsq, Dtype dt, size_t numel, int rank, int64_t next_step) {
    const char* g = dt == Dtype::Bfloat16 ? "bf16" : dt == Dtype::Float32 ? "fp32" : "other";
    return std::string("SymmetricShardedOptimizer: grad norm^2 = ") + std::to_string(nsq) + " in the " + g +
           " group (" + std::to_string(numel) + " params), rank " + std::to_string(rank) +
           ", optimizer step " + std::to_string(next_step) +
           "; no LossScaler installed, so the step cannot be skipped -- raised BEFORE any update";
}

} // anonymous namespace

// ======================================================================
// Construction
// ======================================================================

SymmetricShardedOptimizer::SymmetricShardedOptimizer(
        const std::vector<Tensor>& model_parameters,
        std::shared_ptr<dist::comm::ProcessGroupBase> pg,
        ZeroConfig cfg)
    : ZeroOptimizerBase(),
      cfg_(cfg),
      pg_(std::move(pg))
{
    if (cfg_.strategy != ShardStrategy::STAGE_1) {
        throw ZeroConfigError("only ShardStrategy::STAGE_1 is implemented");
    }
    is_adamw_ = (cfg_.variant == AdamVariant::ADAMW);

    W_    = pg_ ? pg_->get_worldsize() : 1;
    if (W_ < 1) W_ = 1;
    rank_ = pg_ ? pg_->get_rank() : 0;
    if (W_ > 1 && !pg_) {
        throw ZeroConfigError("world_size > 1 requires a process group");
    }

    overlap_active_ = cfg_.overlap_backward_comm && W_ > 1;
    if (overlap_active_) {
        // Same constraint as v1: the hooks record a CUDA event on
        // getCurrentStream(), which is thread_local, and they mutate the bucket
        // state machine without locking. Under the PARALLEL engine hooks run on
        // pool threads, so they would see stream 0 instead of the app's compute
        // stream AND race each other AND issue collectives in a per-rank
        // nondeterministic order (an immediate NCCL hang).
        if (OwnTensor::autograd::get_execution_mode() !=
            OwnTensor::autograd::ExecutionMode::SEQUENTIAL) {
            throw ZeroConfigError("overlap_backward_comm requires the SEQUENTIAL "
                                  "autograd engine (backward hooks launch NCCL)");
        }
        if (cfg_.grad_accum_steps > 1) {
            throw ZeroConfigError("overlap_backward_comm is incompatible with "
                                  "grad_accum_steps>1 (drive accumulation externally "
                                  "and call arm_grad_sync() before the last backward)");
        }
    }

    build_layout(model_parameters);
    build_buckets();
    allocate_buffers();
    register_windows();
    /* edit */
    if (cfg_.grad_bucket_view) install_grad_views();
    /* edit */
    init_master_from_params();

    if (W_ > 1 && cfg_.verify_consistency) verify_consistency();

    if (overlap_active_) {
        register_overlap_hooks();
        cudaEventCreateWithFlags(&compute_sync_event_, cudaEventDisableTiming);
    }

    if (cfg_.verbose && rank_ == 0) {
        // Two lines, deliberately worded to keep CAPABILITY and STATE apart.
        // "supports" is what this binary can do (a staleness check); "ACTIVE" is
        // what this run is actually doing. Conflating them is confusing: an
        // earlier version printed "caps=... fp32-reduce ..." which reads as
        // though FP32 reduction were enabled when it is merely available.
        std::cout << "[ZeRO-1 v2] build=" << ZERO_V2_BUILD
                  << "  supports: " << ZERO_V2_CAPS << "\n"
                  << "[ZeRO-1 v2] ACTIVE: reduce_dtype="
                  << OwnTensor::get_dtype_name(reduce_dtype_)
                  << (reduce_dtype_ == compute_dtype_ ? " (= param dtype)"
                                                      : " (FORCED, != param dtype)")
                  << "  symmetric=" << (symmetric_active_ ? "ON" : "off")
                  << "  overlap=" << (overlap_active_ ? "bwd" : "-")
                  << "/" << (cfg_.overlap_param_refresh && W_ > 1 ? "refresh" : "-")
                  // In place there is no [S] RS-recv / AG-send pair; report what
                  // that saves, or what the out-of-place pair costs.
                  << "  inplace=" << (inplace_collectives_ ? "ON" : "off") << " ("
                  << (static_cast<double>(S_) *
                      static_cast<double>(elem_size(reduce_dtype_) + elem_size(compute_dtype_)) /
                      (1024.0 * 1024.0))
                  << (inplace_collectives_ ? " MB of shard buffers not allocated)"
                                           : " MB in shard_grad_ + shard_master_compute_)")
                  << "\n"
                  << "[ZeRO-1 v2] " << variant_name() << "\n"
                  << "[ZeRO-1 v2] params=" << total_numel_
                  << " padded=" << padded_numel_
                  << " W=" << W_ << " shard=" << S_
                  << " | " << buckets_.size() << " buckets of "
                  << (bucket_elems_ * elem_size(compute_dtype_) / (1024.0 * 1024.0))
                  << " MB (" << (bucket_elems_ / (W_ ? W_ : 1))
                  << " elems/rank/bucket)"
                  << " | optimizer state/GPU: "
                  << (S_ * 12.0 / (1024.0 * 1024.0)) << " MB (vs "
                  << (padded_numel_ * 12.0 / (1024.0 * 1024.0)) << " MB unsharded)"
                  << std::endl;
        if (reduce_dtype_ != compute_dtype_) {
            // A forced wire dtype breaks the flat_grad_/flat_stage_ alias (see the
            // header), so flat_grad_ becomes its own [padded] allocation and
            // shard_grad_ changes width. Report the NET delta against the aliased
            // default, not just the new buffer: forcing FP32 on a BF16 group makes
            // the flat buffer the whole cost, while forcing BF16 on an FP32 group
            // hands part of it back through a narrower shard_grad_. The sign and
            // size of that trade is what a user has to see before enabling it.
            //
            // (The previous wording reported padded*elem_size(compute) as "extra",
            // which understated the FP32-on-BF16 case by 2x: the aliased default
            // spends NOTHING on flat_grad_, so the whole new buffer is extra.)
            const double MB = 1024.0 * 1024.0;
            const double flat_extra  = padded_numel_ * elem_size(reduce_dtype_) / MB;
            // In place there is no shard_grad_ whose width could change.
            const double shard_delta = inplace_collectives_ ? 0.0
                                     : static_cast<double>(S_)
                                     * (static_cast<double>(elem_size(reduce_dtype_))
                                        - static_cast<double>(elem_size(compute_dtype_))) / MB;
            std::cout << "[ZeRO-1 v2] wire dtype FORCED to "
                      << OwnTensor::get_dtype_name(reduce_dtype_)
                      << " (params are " << OwnTensor::get_dtype_name(compute_dtype_)
                      << "): " << (flat_extra + shard_delta) << " MB/GPU extra"
                      << "  [separate " << flat_extra << " MB grad staging buffer"
                      << ", shard_grad_ " << shard_delta << " MB]" << std::endl;
        }
        if (W_ > 1 && !symmetric_active_) {
            std::cout << "[ZeRO-1 v2] NOTE: symmetric memory is NOT active — "
                         "collectives will run NCCL's ring/tree path, not NVLS. "
                         "Create the process group with MemoryMode::Symmetric "
                         "(and NCCL >= 2.27) to enable it." << std::endl;
        }
    }
}

SymmetricShardedOptimizer::~SymmetricShardedOptimizer() {
    // Windows must be torn down BEFORE the tensors they cover are destroyed and
    // before ncclCommDestroy. ProcessGroupNCCL keeps its own safety net, but
    // deregistering here keeps the ordering explicit and local.
    if (pg_) {
        for (auto win : windows_) {
            if (win) {
                try { pg_->deregister_symmetric(win); }
                catch (const std::exception& e) {
                    std::cerr << "[ZeRO-1 v2] deregister_symmetric failed: "
                              << e.what() << std::endl;
                }
            }
        }
    }
    windows_.clear();

    if (d_norm_sq_) cudaFree(d_norm_sq_);
    if (d_scale_)   cudaFree(d_scale_);
    if (h_pinned_)  cudaFreeHost(h_pinned_);
    if (compute_sync_event_) cudaEventDestroy(compute_sync_event_);
    if (param_sync_event_)   cudaEventDestroy(param_sync_event_);
}

// ======================================================================
// Layout
// ======================================================================

void SymmetricShardedOptimizer::build_layout(const std::vector<Tensor>& model_parameters) {
    std::vector<Tensor> trainable;
    std::vector<size_t> trainable_input_index;
    trainable.reserve(model_parameters.size());
    trainable_input_index.reserve(model_parameters.size());
    for (size_t i = 0; i < model_parameters.size(); ++i) {
        if (model_parameters[i].requires_grad()) {
            trainable.push_back(model_parameters[i]);
            trainable_input_index.push_back(i);
        }
    }
    if (trainable.empty()) throw ZeroConfigError("no trainable parameters supplied");

    device_        = trainable[0].device();
    compute_dtype_ = trainable[0].dtype();
    // An explicit reduce dtype (either direction: FP32 up-casts a lower-
    // precision model, BF16 down-casts an FP32 one) IS supported here (unlike
    // v1, which throws): it costs one extra [padded] staging buffer in that
    // dtype, allocated only on this path. See flat_grad_ in the header.
    reduce_dtype_ = resolve_reduce_dtype(cfg_.reduce_dtype, compute_dtype_);

    for (const auto& p : trainable) {
        if (p.dtype() != compute_dtype_) {
            throw ZeroConfigError("all parameters must share one dtype");
        }
        if (p.device().device != device_.device || p.device().index != device_.index) {
            throw ZeroConfigError("all parameters must live on one device");
        }
    }

    // Group ordering: identical rule and identical resulting flat order as v1, so
    // the two implementations agree on which global element is which parameter —
    // a precondition for the checkpoint reshard to be meaningful.
    std::vector<size_t> idx(trainable.size());
    std::iota(idx.begin(), idx.end(), 0);
    auto group_of = [&](const Tensor& p) -> uint8_t {
        if (cfg_.exclude_1d_from_weight_decay && p.ndim() < 2) return 1;
        return 0;
    };
    std::stable_sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
        return group_of(trainable[a]) < group_of(trainable[b]);
    });

    records_.clear();
    records_.reserve(trainable.size());
    size_t offset = 0;
    for (size_t k : idx) {
        ParamRecord rec;
        rec.param       = trainable[k];
        rec.offset      = offset;
        rec.numel       = trainable[k].numel();
        rec.group_id    = group_of(trainable[k]);
        rec.input_index = trainable_input_index[k];
        offset += rec.numel;
        records_.push_back(std::move(rec));
    }
    total_numel_ = offset;

    // Padding granule. Every bucket length is a multiple of `granule_`, so every
    // per-rank piece is a multiple of (16 / esz) elements == 16 bytes. NVLS
    // multimem instructions move 16-byte vectors; a per-rank piece that is not
    // 16-byte aligned forces a scalar tail and can disqualify the MC kernels.
    // Bucket cuts are ELEMENT indices shared by the parameter buffer (compute
    // dtype) and the gradient buffer (reduce dtype). Size the granule from the
    // SMALLER element type: that needs the most elements to span 16 bytes, so
    // satisfying it satisfies the wider type automatically.
    const size_t esz_min = std::min(elem_size(compute_dtype_), elem_size(reduce_dtype_));
    const size_t vec_elems = (esz_min >= 16) ? 1 : (16 / esz_min);
    // W-INDEPENDENT (see zero_ckpt_align_vecs). granule_ sets padded_numel_ and,
    // through it, bucket_elems_ and every bucket boundary -- and those boundaries
    // are what this group's checkpoint records are named after. Scaling the
    // granule by W (as this once did) moves them at every world size and breaks
    // resume across a world-size change.
    granule_ = zero_ckpt_align_vecs() * vec_elems;
    if ((granule_ % static_cast<size_t>(W_)) != 0) {
        throw ZeroConfigError(
            "ZeRO-1 v2: world size " + std::to_string(W_) + " does not divide the "
            "checkpoint alignment granule (" + std::to_string(zero_ckpt_align_vecs()) +
            " vectors). Every bucket has to split evenly across the ranks. Set "
            "BLU_ZERO_CKPT_ALIGN to a multiple of " + std::to_string(W_) +
            " -- but note that checkpoints only interchange between runs that "
            "agree on it, so pick one value and keep it for the whole run.");
    }

    padded_numel_ = round_up(total_numel_, granule_);
    S_            = padded_numel_ / static_cast<size_t>(W_);

    groups_.clear();
    size_t run_begin = 0;
    for (size_t i = 0; i < records_.size(); ++i) {
        bool last = (i + 1 == records_.size());
        bool boundary = last || (records_[i + 1].group_id != records_[i].group_id);
        if (boundary) {
            GroupSpec g;
            g.begin = records_[run_begin].offset;
            g.end   = records_[i].offset + records_[i].numel;
            g.weight_decay = (records_[i].group_id == 0) ? cfg_.weight_decay : 0.0f;
            g.lr_scale = 1.0f;
            groups_.push_back(g);
            run_begin = i + 1;
        }
    }

    // Same FNV-1a layout identity as v1, over the same inputs. Deliberately does
    // NOT include the ownership scheme: `ownership` is a separate descriptor
    // field so that a reshard can verify "same parameters, different split"
    // rather than being rejected outright as an unrelated layout.
    auto fnv = [](uint64_t h, uint64_t x) {
        for (int b = 0; b < 8; ++b) { h ^= (x & 0xff); h *= 1099511628211ull; x >>= 8; }
        return h;
    };
    uint64_t h = 1469598103934665603ull;
    for (const auto& rec : records_) { h = fnv(h, rec.numel); h = fnv(h, rec.group_id); }
    for (const auto& g : groups_) {
        h = fnv(h, g.begin);
        h = fnv(h, g.end);
        uint32_t wd_bits; std::memcpy(&wd_bits, &g.weight_decay, sizeof(float));
        h = fnv(h, wd_bits);
    }
    layout_hash_ = h;
}

// ----------------------------------------------------------------------
// Bucket tiling. Unlike v1 there are no shard boundaries to respect — every
// rank participates in every bucket — so buckets are uniform, which is exactly
// what NVLS wants (one large, aligned, equally-divided collective).
// ----------------------------------------------------------------------
void SymmetricShardedOptimizer::build_buckets() {
    // Size the byte budget from the LARGER of the two element types. A bucket
    // drives two transfers of the same element count: the reduce-scatter moves
    // `bucket_elems * elem_size(reduce_dtype_)` and the all-gather moves
    // `bucket_elems * elem_size(compute_dtype_)`. Budgeting against the wider
    // one makes comm_bucket_bytes bound BOTH, which is what a user setting
    // "25 MB buckets" expects. (Using compute dtype alone would silently produce
    // 50 MB reduce-scatters under ReduceDtype::FP32 on a BF16 model.)
    //
    // Note this is the opposite choice from granule_, which uses the SMALLER
    // type — that one is about 16-byte alignment, where the smaller type is the
    // binding constraint. Different questions, different answers.
    const size_t esz = std::max(elem_size(compute_dtype_), elem_size(reduce_dtype_));

    size_t target = cfg_.comm_bucket_bytes / esz;
    // Round DOWN to a granule multiple so the bucket stays within the byte
    // budget; never below one granule.
    target = (target / granule_) * granule_;
    if (target == 0) target = granule_;
    bucket_elems_ = target;

    buckets_.clear();
    size_t shard_cursor = 0;
    for (size_t pos = 0; pos < padded_numel_; ) {
        CommBucket b;
        b.begin        = pos;
        b.end          = std::min(pos + bucket_elems_, padded_numel_);
        b.per_rank     = (b.end - b.begin) / static_cast<size_t>(W_);
        b.shard_offset = shard_cursor;
        shard_cursor  += b.per_rank;
        buckets_.push_back(b);
        pos = b.end;
    }

    // --- partition invariants (the correctness-critical part) ---------------
    // A bucket whose length is not a multiple of W would make ncclReduceScatter
    // silently mis-slice; a shard_offset that does not tile [0,S) would put this
    // rank's Adam update on the wrong elements. Both are cheap to check once.
    size_t cursor = 0, shard_total = 0;
    for (const auto& b : buckets_) {
        if (b.begin != cursor || b.end <= b.begin) {
            throw ZeroConfigError("v2 bucket partition is not contiguous");
        }
        if (((b.end - b.begin) % static_cast<size_t>(W_)) != 0) {
            throw ZeroConfigError("v2 bucket length is not a multiple of the world "
                                  "size (ncclReduceScatter/ncclAllGather require it)");
        }
        if (b.shard_offset != shard_total) {
            throw ZeroConfigError("v2 bucket shard offsets do not tile the shard");
        }
        // 16-byte alignment of this rank's piece, for the multimem path.
        if (((b.begin * esz) % 16) != 0 || ((b.per_rank * esz) % 16) != 0) {
            throw ZeroConfigError("v2 bucket is not 16-byte aligned (NVLS multimem "
                                  "operates on 16-byte vectors)");
        }
        cursor      = b.end;
        shard_total += b.per_rank;
    }
    if (cursor != padded_numel_) {
        throw ZeroConfigError("v2 buckets do not tile the padded buffer");
    }
    if (shard_total != S_) {
        throw ZeroConfigError("v2 bucket shard pieces do not sum to the shard size");
    }

    // record -> buckets it overlaps. records_ and buckets_ are both ascending in
    // flat order, so one forward scan suffices. A parameter straddling a bucket
    // cut is counted in EACH bucket it touches, so a bucket only fires once every
    // distinct contributing parameter has landed.
    param_to_buckets_.assign(records_.size(), {});
    size_t bi = 0;
    for (size_t pidx = 0; pidx < records_.size(); ++pidx) {
        const size_t p_begin = records_[pidx].offset;
        const size_t p_end   = p_begin + records_[pidx].numel;
        if (p_end == p_begin) continue;
        while (bi < buckets_.size() && buckets_[bi].end <= p_begin) ++bi;
        for (size_t j = bi; j < buckets_.size() && buckets_[j].begin < p_end; ++j) {
            param_to_buckets_[pidx].push_back(static_cast<uint32_t>(j));
            buckets_[j].pending_init += 1;
        }
    }
    for (auto& b : buckets_) b.pending = b.pending_init;

    grad_fired_.assign(records_.size(), false);
}

// ----------------------------------------------------------------------
// Shard-space <-> global-space mapping.
//
// Within bucket k this rank owns global [begin_k + r*per_rank_k, +per_rank_k),
// which maps affinely onto shard [shard_offset_k, +per_rank_k). Because
// begin_k + (r+1)*per_rank_k <= end_k == begin_{k+1} <= begin_{k+1} + r*per_rank_{k+1},
// the global image of the shard is MONOTONE INCREASING in the shard index. That
// is what lets a global range (a weight-decay group, or the valid/padding
// boundary) map onto a single CONTIGUOUS shard range — so Adam still runs as one
// launch per group, exactly as in v1, despite the strided ownership.
// ----------------------------------------------------------------------
size_t SymmetricShardedOptimizer::shard_to_global(size_t local) const {
    // buckets_ is ascending in shard_offset; find the bucket owning `local`.
    size_t lo = 0, hi = buckets_.size();
    while (lo + 1 < hi) {
        size_t mid = (lo + hi) / 2;
        if (buckets_[mid].shard_offset <= local) lo = mid; else hi = mid;
    }
    const CommBucket& b = buckets_[lo];
    return b.begin + static_cast<size_t>(rank_) * b.per_rank + (local - b.shard_offset);
}

// ======================================================================
// Buffers + symmetric windows
// ======================================================================

// Allocate a buffer that will cross the wire. When symmetric memory is active
// the storage must come from ncclMemAlloc — that is the ONLY memory NCCL will
// accept into a NCCL_WIN_COLL_SYMMETRIC window.
//
// This deliberately does NOT go through TensorOptions. Adding an allocator field
// there would change sizeof(TensorOptions), which is passed BY VALUE throughout
// the tensor library — and that library's Makefile has no header-dependency
// tracking, so a partial rebuild would leave stale objects using the old layout
// and corrupt the stack silently. Constructing TensorImpl directly reaches the
// same Allocator* parameter (TensorImpl.h:110-114) with zero ABI impact.
//
// Storage bakes the allocator into its deleter, so the tensor frees through
// ncclMemFree automatically when the last reference drops.

namespace {
// OwnTensor::Allocator has no owns() -- it is a plain allocate/deallocate
// interface in the tensor submodule, and widening it there for one debug check
// is not worth a submodule change. Both symmetric allocators track their live
// pointers, so ask whichever one this is.
//
// The check matters because Allocator::deallocate() silently no-ops on a
// pointer it does not know: a mis-wired allocator would leak the whole wire
// buffer rather than crash. Unknown allocator types cannot be verified, so
// they pass -- this guards wiring, it is not a security boundary.
bool allocator_owns(OwnTensor::Allocator* a, void* p) {
    if (auto* n = dynamic_cast<dist::comm::NcclSymmetricAllocator*>(a))    return n->owns(p);
    if (auto* b = dynamic_cast<dist::comm::BlucclSymmetricAllocator*>(a))  return b->owns(p);
    return true;
}
} // namespace

Tensor SymmetricShardedOptimizer::alloc_wire_tensor(size_t numel, Dtype dt) {
    if (!symmetric_active_) {
        TensorOptions o = TensorOptions().with_dtype(dt).with_device(device_);
        return Tensor::zeros(flat_shape(numel), o);
    }

    // Ask the GROUP for its allocator rather than naming NcclSymmetricAllocator:
    // NCCL returns ncclMemAlloc memory (registrable as a symmetric window),
    // BluCCL returns gas segments (collectives run in place instead of staging
    // through two D2D copies per call). symmetric_active_ was set from
    // pg_->symmetric_supported(), so a null here means the group changed its
    // mind between that check and now -- a wiring bug, not a fallback.
    OwnTensor::Allocator* sym_alloc = pg_->symmetric_allocator();
    if (!sym_alloc) {
        throw ZeroConfigError(
            std::string("process group backend '") + pg_->backend_str() +
            "' reported symmetric support but provides no symmetric allocator");
    }
    auto impl = OwnTensor::make_intrusive<OwnTensor::TensorImpl>(
        flat_shape(numel), dt, device_, /*requires_grad=*/false, sym_alloc);
    Tensor t(impl);

    // Tensor::zeros() memsets; this path does not, and ncclMemAlloc memory is
    // uninitialised. The padding tail in particular must start at zero.
    cudaMemsetAsync(t.data(), 0, numel * elem_size(dt), ocuda::getCurrentStream());
    sync_compute();

    if (!allocator_owns(sym_alloc, t.unsafeGetTensorImpl()->mutable_storage().data_ptr())) {
        // deallocate() silently no-ops on untracked pointers, so a mis-wired
        // allocator would leak rather than crash. Fail loudly instead.
        throw ZeroConfigError(std::string("symmetric allocation did not come from the ") +
                              pg_->backend_str() + " symmetric allocator (wiring is broken)");
    }
    return t;
}

void SymmetricShardedOptimizer::allocate_buffers() {
    // Decide symmetric-vs-standard BEFORE allocating: the choice changes which
    // allocator backs the wire buffers, and it cannot be changed afterwards
    // (Storage captures its allocator in the deleter at construction).
    // NOTE two DIFFERENT questions, deliberately not the same predicate:
    //   symmetric_active_        -> "is there an allocator whose memory this
    //                               backend's collectives can use in place?"
    //                               NCCL: ncclMemAlloc. BluCCL: gas segments.
    //   symmetric_supported()    -> "can I register an NCCL symmetric WINDOW?"
    //                               BluCCL: no -- it has symmetric memory but
    //                               no ncclCommWindowRegister equivalent.
    // Gating the allocator on the window question would send BluCCL down the
    // plain-cudaMalloc path and reintroduce the staging copies the gas
    // allocator exists to remove.
    symmetric_active_ = (W_ > 1) && pg_ && pg_->symmetric_allocator() != nullptr;
    // In place on NCCL (see zero_inplace_collectives_enabled). W==1 issues no
    // collective, and its whole flat buffer is already "this rank's slice".
    inplace_collectives_ = (W_ == 1) ||
                           (pg_ && pg_->backend() == dist::comm::Backend::NCCL &&
                            zero_inplace_collectives_enabled());

    TensorOptions of = TensorOptions().with_dtype(Dtype::Float32).with_device(device_);

    // --- buffers that cross the wire (candidates for symmetric windows) ---
    flat_stage_          = alloc_wire_tensor(padded_numel_, compute_dtype_);
    // Same buffer as flat_stage_ unless the reduce dtype differs, in which case
    // gradients need their own [padded] staging in the wider type.
    flat_grad_           = (reduce_dtype_ == compute_dtype_)
                             ? flat_stage_
                             : alloc_wire_tensor(padded_numel_, reduce_dtype_);
    if (!inplace_collectives_) {
        shard_grad_          = alloc_wire_tensor(S_, reduce_dtype_);
        shard_master_compute_= alloc_wire_tensor(S_, compute_dtype_);
    }

    // --- purely local buffers (ordinary caching allocator) ---
    shard_grad_fp32_ = Tensor::zeros(flat_shape(S_), of);
    shard_master_    = Tensor::zeros(flat_shape(S_), of);
    shard_m_         = Tensor::zeros(flat_shape(S_), of);
    shard_v_         = Tensor::zeros(flat_shape(S_), of);

    cudaMalloc(&d_norm_sq_, sizeof(float));
    cudaMalloc(&d_scale_,   sizeof(float));
    cudaMallocHost(&h_pinned_, sizeof(float));
}

void SymmetricShardedOptimizer::register_windows() {
    // Windows are an NCCL concept. A backend with symmetric memory but no
    // window API (BluCCL) keeps its buffers and simply registers nothing.
    if (!symmetric_active_ || !pg_->symmetric_supported()) return;

    // COLLECTIVE, and order-sensitive: every rank must register the same buffers
    // in the same sequence. All three have identical sizes on every rank, and
    // each window's head is the tensor's storage base, so every rank's offset
    // into a given window is identical — which is exactly the
    // NCCL_WIN_COLL_SYMMETRIC contract ("the offset to the buffer head address
    // from all ranks must be equal").
    auto reg = [&](Tensor& t, const char* what) {
        // Register the STORAGE BASE and the PADDED allocation size, not
        // Tensor::data()/nbytes(): data() is offset by storage_offset for views,
        // and TensorImpl rounds the allocation up to 256 bytes, so nbytes() is
        // smaller than what was really allocated.
        void*  base  = t.unsafeGetTensorImpl()->mutable_storage().data_ptr();
        size_t bytes = t.unsafeGetTensorImpl()->storage().nbytes();
        auto win = pg_->register_symmetric(base, bytes);
        if (!win) {
            throw ZeroConfigError(std::string("symmetric window registration returned "
                                              "null for ") + what);
        }
        windows_.push_back(win);
    };

    reg(flat_stage_,           "flat_stage");
    // Only when it is a distinct allocation — registering the same base twice
    // would be a second collective with no matching buffer on the other side.
    if (reduce_dtype_ != compute_dtype_) reg(flat_grad_, "flat_grad");
    if (!inplace_collectives_) {
        reg(shard_grad_,           "shard_grad");
        reg(shard_master_compute_, "shard_master_compute");
    }
}

char* SymmetricShardedOptimizer::grad_wire_base() {
    return static_cast<char*>(inplace_collectives_ ? flat_grad_.data() : shard_grad_.data());
}

char* SymmetricShardedOptimizer::param_wire_base() {
    return static_cast<char*>(inplace_collectives_ ? flat_stage_.data()
                                                   : shard_master_compute_.data());
}

void SymmetricShardedOptimizer::init_master_from_params() {
    const size_t esz = elem_size(compute_dtype_);

    for (const auto& rec : records_) {
        void* dst = static_cast<char*>(flat_stage_.data()) + rec.offset * esz;
        d2d(dst, rec.param.data(), rec.numel * esz, device_);
    }
    if (padded_numel_ > total_numel_) {
        zero_async(static_cast<char*>(flat_stage_.data()) + total_numel_ * esz,
                   (padded_numel_ - total_numel_) * esz);
    }
    sync_compute();

    if (cfg_.init_broadcast && W_ > 1) {
        auto r = pg_->broadcast(flat_stage_.data(), flat_stage_.data(),
                                padded_numel_, compute_dtype_, /*root=*/0, /*sync=*/true);
        if (r != pgSuccess) throw ZeroCollectiveError("init broadcast failed");
        cudaDeviceSynchronize();
        for (const auto& rec : records_) {
            const void* src = static_cast<const char*>(flat_stage_.data()) + rec.offset * esz;
            Tensor p = rec.param;
            d2d(p.data(), src, rec.numel * esz, device_);
        }
        sync_compute();
    }

    // Seed the FP32 master with this rank's shard. Under PER_BUCKET ownership the
    // shard is NOT one contiguous slice of flat_stage_, so this is one copy per
    // bucket rather than v1's single slice_inplace.
    for (const auto& b : buckets_) {
        const size_t g_lo = b.begin + static_cast<size_t>(rank_) * b.per_rank;
        Tensor piece = flat_stage_.slice_inplace(g_lo, b.per_rank);
        if (compute_dtype_ == Dtype::Float32) {
            d2d(shard_master_.data<float>() + b.shard_offset,
                piece.data(), b.per_rank * sizeof(float), device_);
        } else {
            Tensor f = piece.as_type(Dtype::Float32);
            d2d(shard_master_.data<float>() + b.shard_offset,
                f.data(), b.per_rank * sizeof(float), device_);
        }
    }
    sync_compute();
}

void SymmetricShardedOptimizer::verify_consistency() {
    float local = static_cast<float>(total_numel_);
    float mn = local, mx = local;

    cudaMemcpy(d_norm_sq_, &local, sizeof(float), cudaMemcpyHostToDevice);
    pg_->all_reduce(d_norm_sq_, d_norm_sq_, 1, Dtype::Float32, op_min, true);
    cudaDeviceSynchronize();
    cudaMemcpy(&mn, d_norm_sq_, sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemcpy(d_norm_sq_, &local, sizeof(float), cudaMemcpyHostToDevice);
    pg_->all_reduce(d_norm_sq_, d_norm_sq_, 1, Dtype::Float32, op_max, true);
    cudaDeviceSynchronize();
    cudaMemcpy(&mx, d_norm_sq_, sizeof(float), cudaMemcpyDeviceToHost);

    if (mn != mx || mn != local) {
        throw ZeroConsistencyError("ranks disagree on total parameter count "
                                   "(param set/order must be identical on every rank)");
    }
}

// ======================================================================
// Overlap: bucketed reduce-scatter fired from backward hooks
// ======================================================================

void SymmetricShardedOptimizer::register_overlap_hooks() {
    for (size_t pidx = 0; pidx < records_.size(); ++pidx) {
        Tensor p = records_[pidx].param;
        p.register_post_acc_hook(OwnTensor::make_post_acc_hook(
            [this, pidx](const Tensor& g) { this->on_grad_ready(pidx, g); }));
    }
}

void SymmetricShardedOptimizer::arm_grad_sync() {
    // An outstanding refresh must never survive into a backward. When
    // flat_grad_ aliases flat_stage_ (the default) the reason is direct: the
    // gradients would overwrite the un-installed parameters and later be
    // scattered into the live weights. With a separate FP32 grad buffer that
    // aliasing is gone, but the guard still holds — parameters left uninstalled
    // mean the next forward reads last step's weights. Checked before the
    // overlap early-out so it is live even with overlap_backward_comm=false.
    if (param_refresh_pending_) {
        throw ZeroConfigError("parameter refresh stages were not consumed before "
                              "backward — install every stage during the forward "
                              "(refresh_param_stage) or drain explicitly");
    }
    if (!overlap_active_) return;

    for (auto& b : buckets_) {
        b.pending  = b.pending_init;
        b.launched = false;
    }
    buckets_launched_ = 0;
    grad_fired_.assign(records_.size(), false);

    // Zero the padding tail once per pass, before any gradient lands, so
    // pure-padding regions reduce to zero deterministically.
    const size_t rsz = elem_size(reduce_dtype_);
    if (padded_numel_ > total_numel_) {
        zero_async(static_cast<char*>(flat_grad_.data()) + total_numel_ * rsz,
                   (padded_numel_ - total_numel_) * rsz);
    }
    armed_ = true;
}

// Stage one parameter's gradient into flat_grad_, converting to the wire dtype
// IN PLACE when they differ.
//
// The obvious spelling is `g.as_type(reduce_dtype_)` followed by a copy, and
// that is what this used to do. It costs an extra full pass: as_type()
// materializes a whole converted tensor, which is then copied again into
// flat_grad_. On this repo's FP32 AdamW group under BLU_BF16_REDUCE=1 that is a
// 196 MB transient plus a redundant 206 MB read+write per step for wte alone —
// visible in nsys as a 206,045,184 B device-to-device copy that appears once per
// step per GPU and does not exist on the aliased path.
//
// convert_type_cuda_generic writes straight into the destination, so the
// transient disappears and the traffic drops to one read of the gradient plus
// one write of the wire bytes. ShardedMuon::stage_grad_into_flat has always done
// it this way; this is v2 catching up.
//
// The conversion is enqueued on the COMPUTE stream, same as the copy it
// replaces, so the compute->comm edge in launch_bucket_rs still orders it
// correctly.
/* edit */
// Gradient bucket views -- the v2 twin of ShardedMuon::install_grad_views(); the
// long rationale, the dtype cost and the overlap_param_refresh conflict are all
// written out there and apply here unchanged.
//
// This group is much smaller than the Muon one (on this model: the two embedding
// matrices, ~309 M elements against 3.62 B), so the traffic it removes is
// correspondingly smaller. It is wired anyway so the flag means the same thing
// for every ZeRO group rather than silently covering only one of them -- a flag
// that applies to some groups and not others is the kind of thing that makes an
// A/B unreadable.
void SymmetricShardedOptimizer::install_grad_views() {
    auto decline = [&](const char* why) {
        if (cfg_.verbose && rank_ == 0) {
            std::cout << "[ZeRO-1 v2] BLU_GRAD_VIEW=1 DECLINED: " << why
                      << " -- gradients stage through flat_grad_ as before."
                      << std::endl;
        }
        grad_views_active_ = false;
    };

    if (W_ <= 1)                return decline("world_size == 1");
    if (!flat_grad_.is_valid()) return decline("flat_grad_ was not allocated");
    if (cfg_.overlap_param_refresh)
        return decline("overlap_param_refresh aliases flat_stage_ into flat_grad_");

    for (const auto& rec : records_) {
        if (rec.param.dtype() != reduce_dtype_)
            return decline("a parameter's dtype differs from the wire dtype");
    }

    auto base = flat_grad_.unsafeGetTensorImpl()->storage_ptr();
    const int64_t base_off = static_cast<int64_t>(flat_grad_.storage_offset());

    grad_views_.clear();
    grad_views_.reserve(records_.size());
    for (auto& rec : records_) {
        const Shape sh = rec.param.shape();
        auto impl = OwnTensor::make_intrusive<OwnTensor::TensorImpl>(
            base, sh, OwnTensor::ViewUtils::compute_strides(sh),
            base_off + static_cast<int64_t>(rec.offset),
            reduce_dtype_, device_);
        Tensor view(std::move(impl));
        Tensor p = rec.param;
        p.set_grad(view);
        grad_views_.push_back(view);
    }
    grad_views_active_ = true;

    if (cfg_.verbose && rank_ == 0) {
        const double saved_mb = (double)total_numel_ * 4.0 / (1024.0 * 1024.0);
        std::cout << "[ZeRO-1 v2] BLU_GRAD_VIEW=1 ACTIVE: " << records_.size()
                  << " gradients are views into flat_grad_ (no staging copy; "
                  << saved_mb << " MB of per-parameter fp32 gradients never "
                     "allocated). NOTE micro-step accumulation is now "
                  << OwnTensor::get_dtype_name(reduce_dtype_)
                  << ", not float32 -- A/B the loss curve." << std::endl;
    }
}
/* edit */

void SymmetricShardedOptimizer::stage_grad_into_flat(const ParamRecord& rec,
                                                     const Tensor& grad) {
    /* edit */
    // Bucket views: `grad` already IS flat_grad_[rec.offset ...]. Nothing to do.
    if (grad_views_active_) return;
    /* edit */
    const size_t rsz = elem_size(reduce_dtype_);
    void* dst = static_cast<char*>(flat_grad_.data()) + rec.offset * rsz;

    Tensor g = grad;
    if (!g.is_contiguous()) g = g.contiguous();

    // Autograd stores BF16/FP16 leaf gradients upcast to FP32, so on a BF16 model
    // ReduceDtype::FP32 takes the memcpy branch and no conversion happens at all.
    if (g.dtype() != reduce_dtype_) {
        OwnTensor::convert_type_cuda_generic(g.data(), g.dtype(),
                                             dst, reduce_dtype_,
                                             static_cast<int64_t>(rec.numel),
                                             ocuda::getCurrentStream());
    } else {
        d2d(dst, g.data(), rec.numel * rsz, device_);
    }
}

void SymmetricShardedOptimizer::on_grad_ready(size_t pidx, const Tensor& grad) {
    if (!armed_) return;   // inert on unarmed / earlier-micro-step backwards

    stage_grad_into_flat(records_[pidx], grad);

    grad_fired_[pidx] = true;

    for (uint32_t bidx : param_to_buckets_[pidx]) {
        CommBucket& b = buckets_[bidx];
        if (!b.launched && b.pending > 0 && --b.pending == 0) launch_bucket_rs(bidx);
    }
}

void SymmetricShardedOptimizer::launch_bucket_rs(size_t bidx) {
    CommBucket& b = buckets_[bidx];
    const size_t rsz = elem_size(reduce_dtype_);

    // compute -> comm edge. One reused event is safe because every launch does
    // record-then-wait in program order on the single (sequential-engine) thread.
    cudaEventRecord(compute_sync_event_, ocuda::getCurrentStream());
    cudaStreamWaitEvent(pg_->get_stream(), compute_sync_event_, 0);

    const void* send = static_cast<const char*>(flat_grad_.data()) + b.begin * rsz;
    // In place: recv == send + rank * per_rank, NCCL's in-place reduce-scatter.
    void*       recv = grad_wire_base() + wire_offset(b) * rsz;

    // op=SUM, not avg: NCCL ships symmetric kernels only for `_sum_`, so avg
    // would silently drop off the NVLS path. The 1/W is folded into the single
    // scale pass in apply_grad_scale_and_check().
    //
    // Every rank receives real data it owns, so unlike v1 there is no throwaway
    // receive buffer and no owner branch.
    b.rs_work = pg_->reduce_scatter_async(send, recv, b.per_rank, reduce_dtype_,
                                          /*op=*/sum, /*sync=*/false, b.rs_work);
    b.launched = true;
    ++buckets_launched_;
}

void SymmetricShardedOptimizer::finalize_overlap_reduce() {
    const size_t rsz = elem_size(reduce_dtype_);

    // Mop-up. grad_fired_ is authoritative, NOT param.has_grad(): zero_grad()
    // only zeroes an existing gradient buffer rather than dropping it, so a
    // parameter that is unused in this step's graph still reports has_grad() and
    // would leave its bucket unlaunched (reducing a stale gradient).
    for (size_t pidx = 0; pidx < records_.size(); ++pidx) {
        if (grad_fired_[pidx]) continue;
        bool feeds_unlaunched = false;
        for (uint32_t bidx : param_to_buckets_[pidx]) {
            if (!buckets_[bidx].launched) { feeds_unlaunched = true; break; }
        }
        if (!feeds_unlaunched) continue;

        // An unused parameter's averaged gradient is zero (DDP semantics).
        void* dst = static_cast<char*>(flat_grad_.data()) + records_[pidx].offset * rsz;
        zero_async(dst, records_[pidx].numel * rsz);

        for (uint32_t bidx : param_to_buckets_[pidx]) {
            CommBucket& b = buckets_[bidx];
            if (!b.launched && b.pending > 0 && --b.pending == 0) launch_bucket_rs(bidx);
        }
    }
    // Pure-padding buckets have pending_init == 0, so the decrement rule can
    // never fire for them; launch directly.
    for (size_t bidx = 0; bidx < buckets_.size(); ++bidx) {
        if (!buckets_[bidx].launched && buckets_[bidx].pending == 0) launch_bucket_rs(bidx);
    }

    for (const auto& b : buckets_) {
        if (!b.launched) {
            throw ZeroCollectiveError("v2 overlap reduce: a gradient bucket was never "
                                      "launched (its reduce-scatter would wait a stale "
                                      "event and reduce last step's gradient)");
        }
    }

    // comm -> compute edge. Unlike v1, EVERY bucket writes into this rank's grad
    // shard, so every bucket must be waited on (v1 could skip non-owned reduces).
    // Work::wait() enqueues a cudaStreamWaitEvent on the compute stream; it does
    // not stall the host. Its RESULT is checked: see wait_or_throw.
    for (size_t bidx = 0; bidx < buckets_.size(); ++bidx) {
        wait_or_throw(buckets_[bidx].rs_work, "gradient reduce-scatter", bidx);
    }
    armed_ = false;
}

// ======================================================================
// Reference (non-overlap) path
// ======================================================================

void SymmetricShardedOptimizer::gather_grads_into_flat() {
    const size_t rsz = elem_size(reduce_dtype_);
    for (const auto& rec : records_) {
        Tensor p = rec.param;
        if (p.has_grad()) {
            stage_grad_into_flat(rec, p.grad_view());
        } else {
            void* dst = static_cast<char*>(flat_grad_.data()) + rec.offset * rsz;
            zero_async(dst, rec.numel * rsz);
        }
    }
    if (padded_numel_ > total_numel_) {
        zero_async(static_cast<char*>(flat_grad_.data()) + total_numel_ * rsz,
                   (padded_numel_ - total_numel_) * rsz);
    }
    sync_compute();
}

void SymmetricShardedOptimizer::reduce_scatter_grads() {
    const size_t rsz = elem_size(reduce_dtype_);
    if (W_ == 1) {
        // Sum over one rank is the identity; the shard IS the flat buffer, and in
        // place it is already where the reduced gradients are read from.
        if (!inplace_collectives_) d2d(shard_grad_.data(), flat_grad_.data(), S_ * rsz, device_);
        return;
    }
    sync_compute();
    // One reduce-scatter per bucket even on the reference path, so the numerics
    // match the overlapped path exactly (a single monolithic reduce-scatter would
    // use a different reduction order and give different BF16 rounding).
    for (const auto& b : buckets_) {
        const void* send = static_cast<const char*>(flat_grad_.data()) + b.begin * rsz;
        void*       recv = grad_wire_base() + wire_offset(b) * rsz;
        auto r = pg_->reduce_scatter(send, recv, b.per_rank, reduce_dtype_, sum, /*sync=*/true);
        if (r != pgSuccess) throw ZeroCollectiveError("reduce_scatter(grads) failed");
    }
    cudaDeviceSynchronize();
}

void SymmetricShardedOptimizer::cast_shard_grad_to_fp32() {
    if (!inplace_collectives_) {
        if (reduce_dtype_ == Dtype::Float32) {
            d2d(shard_grad_fp32_.data(), shard_grad_.data(), S_ * sizeof(float), device_);
        } else {
            OwnTensor::convert_type_cuda_generic(shard_grad_.data(), reduce_dtype_,
                                                 shard_grad_fp32_.data<float>(), Dtype::Float32,
                                                 static_cast<int64_t>(S_),
                                                 ocuda::getCurrentStream());
        }
    } else {
        // In place the reduced pieces sit in this rank's slice of each bucket,
        // strided across flat_grad_; gather them into the contiguous FP32 shard
        // the norm, scale and Adam kernels read. Same work as the single call
        // above, one launch per bucket.
        const size_t rsz = elem_size(reduce_dtype_);
        for (const auto& b : buckets_) {
            const char* src = grad_wire_base() + wire_offset(b) * rsz;
            float*      dst = shard_grad_fp32_.data<float>() + b.shard_offset;
            if (reduce_dtype_ == Dtype::Float32) {
                d2d(dst, src, b.per_rank * sizeof(float), device_);
            } else {
                OwnTensor::convert_type_cuda_generic(src, reduce_dtype_, dst, Dtype::Float32,
                                                     static_cast<int64_t>(b.per_rank),
                                                     ocuda::getCurrentStream());
            }
        }
    }
    sync_compute();
}

// ======================================================================
// One combined scale pass: 1/W (undo op=sum) * loss-scale inverse * clip coef
// ======================================================================

// Valid (non-padding) prefix of the shard. Padding lives at the END of global
// space, and padded_numel_ - total_numel_ < granule_ <= bucket length, so all
// padding falls inside the LAST bucket. Every earlier bucket is fully valid, which
// makes the valid region a contiguous PREFIX of shard space.
size_t SymmetricShardedOptimizer::valid_shard_prefix() const {
    const CommBucket& b = buckets_.back();
    const size_t g_lo = b.begin + static_cast<size_t>(rank_) * b.per_rank;
    size_t valid_in_last = 0;
    if (total_numel_ > g_lo) valid_in_last = std::min(total_numel_ - g_lo, b.per_rank);
    return b.shard_offset + valid_in_last;
}

// The gradient-reduction half of step(). Split out so reduce_and_measure() can run
// it during the cross-group norm phase and step() can then skip it.
void SymmetricShardedOptimizer::reduce_grads_now() {
    if (overlap_active_ && armed_) {
        finalize_overlap_reduce();
    } else {
        gather_grads_into_flat();
        reduce_scatter_grads();
    }
    cast_shard_grad_to_fp32();
    grads_reduced_ = true;
}

// Phase 1 of cross-group global clipping: reduce this group's gradients and return
// its exact ||g||^2 (already all-reduced across ranks), applying nothing. The value
// is cached so apply_grad_scale_and_check() does not repeat the collective.
// Negative return means "skip the step" (non-finite under a loss scaler).
float SymmetricShardedOptimizer::reduce_and_measure() {
    // Same precondition as step(): this reduces gradients into the staging buffer
    // that an unconsumed parameter refresh is still writing into (and, on the
    // aliased default, still holds the un-installed parameters). Fail loudly here
    // too rather than only at step().
    if (param_refresh_pending_) {
        throw ZeroConfigError("parameter refresh stages were not consumed before "
                              "reduce_and_measure() — install every stage during the "
                              "forward (refresh_param_stage) or drain explicitly");
    }

    if (cfg_.grad_accum_steps > 1 && micro_step_ + 1 < cfg_.grad_accum_steps) {
        return 0.0f;   // still accumulating: contributes nothing to the global norm
    }
    if (!grads_reduced_) reduce_grads_now();

    auto* sc = get_scaler();
    const float denom = static_cast<float>(W_) * (sc ? sc->scale() : 1.0f);
    const size_t valid = valid_shard_prefix();

    cudaMemsetAsync(d_norm_sq_, 0, sizeof(float), ocuda::getCurrentStream());
    if (valid > 0) {
        std::vector<ocuda::TensorInfo> gv = {{shard_grad_fp32_.data<float>(),
                                              static_cast<int64_t>(valid)}};
        ocuda::multi_tensor_grad_norm_cuda(gv, d_norm_sq_);
        sync_compute();
    }
    if (W_ > 1) {   // COLLECTIVE: every rank, valid or not (see apply_grad_scale_and_check)
        sync_compute();
        pg_->all_reduce(d_norm_sq_, d_norm_sq_, 1, Dtype::Float32, sum, /*sync=*/true);
        cudaStreamSynchronize(ocuda::getCurrentStream());
    }
    cudaMemcpy(h_pinned_, d_norm_sq_, sizeof(float), cudaMemcpyDeviceToHost);
    const float nsq = *h_pinned_ / (denom * denom);

    // Non-finite check is UNCONDITIONAL, not gated on a loss scaler. No
    // LossScaler is installed in production (set_scaler() is called only from
    // dist/zero/tests), so an `sc &&` guard here was dead code and a NaN/Inf
    // gradient was returned to the caller as-is. The training loop tests
    // `nsq < 0` to detect the skip, and `NaN < 0` is false — so a NaN gradient
    // reached std::min(1.0f, NaN), which returns 1.0f, i.e. NO CLIP, and the
    // NaN went straight into the parameters. Returning -1 here makes the skip
    // fire for every non-finite value, with or without a scaler.
    if (!std::isfinite(nsq)) return -1.0f;
    cached_nsq_     = nsq;
    last_grad_norm_ = nsq > 0.0f ? std::sqrt(nsq) : 0.0f;
    return nsq;
}

bool SymmetricShardedOptimizer::apply_grad_scale_and_check() {
    auto* sc = get_scaler();
    const float loss_scale = sc ? sc->scale() : 1.0f;
    const float w          = static_cast<float>(W_);

    const size_t valid = valid_shard_prefix();

    const bool need_norm = (sc != nullptr) || (cfg_.max_grad_norm > 0.0f);
    float combined = 1.0f / (w * loss_scale);   // always undo op=sum (and any loss scale)

    // NOTE the guard is on need_norm ALONE, not on `valid > 0`. The all_reduce
    // below is a COLLECTIVE: every rank must issue it or the communicator
    // desynchronises and the job hangs. A rank whose shard is entirely padding has
    // nothing to contribute, so it skips the local norm kernel and contributes the
    // memset zero — but it must still take part in the reduction. (v1 and
    // ShardedMuon already scope it this way; v2 used to fold both into one
    // `need_norm && valid > 0`, which hangs as soon as any rank owns only padding.
    // Reachable whenever a group is small enough to fit in one bucket AND
    // total_numel <= r*S for some rank r — under ~441 bf16 / ~217 fp32 elements at
    // W=8, and the threshold grows with W.)
    // reduce_and_measure() already did the norm + all-reduce this step; reuse it
    // rather than paying a second collective.
    if (need_norm && cached_nsq_ >= 0.0f) {
        const float nsq = cached_nsq_;
        cached_nsq_ = -1.0f;
        last_grad_norm_ = nsq > 0.0f ? std::sqrt(nsq) : 0.0f;
        if (cfg_.max_grad_norm > 0.0f) {
            float coef = external_clip_coef_ >= 0.0f
                       ? external_clip_coef_
                       : cfg_.max_grad_norm / (last_grad_norm_ + 1e-6f);
            if (coef > 1.0f) coef = 1.0f;
            combined *= coef;
        }
    } else if (need_norm) {
        cudaMemsetAsync(d_norm_sq_, 0, sizeof(float), ocuda::getCurrentStream());
        if (valid > 0) {
            std::vector<ocuda::TensorInfo> gv = {{shard_grad_fp32_.data<float>(),
                                                  static_cast<int64_t>(valid)}};
            ocuda::multi_tensor_grad_norm_cuda(gv, d_norm_sq_);
            sync_compute();
        }

        if (W_ > 1) {
            sync_compute();
            pg_->all_reduce(d_norm_sq_, d_norm_sq_, 1, Dtype::Float32, sum, /*sync=*/true);
            // Sync the COMPUTE stream only. cudaDeviceSynchronize() here would
            // drain another dtype group's in-flight staged all-gathers and
            // silently destroy their overlap.
            cudaStreamSynchronize(ocuda::getCurrentStream());
        }
        cudaMemcpy(h_pinned_, d_norm_sq_, sizeof(float), cudaMemcpyDeviceToHost);
        const float nsq_raw = *h_pinned_;

        // nsq_raw is the squared norm of the SUMMED, still-loss-scaled gradient.
        // The true gradient norm divides out both factors.
        //
        // CAVEAT: with a very large loss scale this squares (W*scale) into
        // nsq_raw, so it can overflow FP32 slightly earlier than v1 (which
        // unscaled before taking the norm). With bf16 training and no loss
        // scaler — the configuration this targets — loss_scale is 1 and the two
        // are identical. A spurious overflow is caught below as a skipped step,
        // never as silent corruption.
        const float denom = w * loss_scale;
        const float nsq   = nsq_raw / (denom * denom);

        // F1 (same fix as ShardedOptimizer::step() got in v1).
        // Was:  `if (sc && !std::isfinite(nsq)) { skip }`, and last_grad_norm_ set AFTER it.
        //       set_scaler() is called only from dist/zero/tests, so in training `sc` is null and
        //       this guard NEVER fired. A NaN/Inf gradient then went on into Adam:
        //         1. nsq = NaN or Inf (multi_tensor_grad_norm keeps NaN/Inf)
        //         2. NaN: the old `nsq > 0 ? sqrt(nsq) : 0` set last_grad_norm_ = 0 (logged as 0),
        //            so coef = max / (0 + 1e-6), clamped to 1 -> no clip, the NaN elements go on.
        //            Inf: last_grad_norm_ = Inf -> coef = 0 -> every gradient is scaled by 0, and
        //            Inf * 0 = NaN on the Inf elements.
        //         3. local_adam_step writes NaN into master/m/v for those elements
        //         4. a NaN weight makes the next forward's loss NaN -> the trainer aborts one step
        //            LATE; if this step was a checkpoint step, the NaN state was already saved and
        //            auto-resume reloads it.
        //       Measured before this fix (test_f1_nonfinite, W=1, bf16 and fp32): NaN or Inf in one
        //       gradient element -> no error, step_count advanced, that weight written as NaN.
        //       BLU_GLOBAL_CLIP=1 did not help: reduce_and_measure() returns -1 for a non-finite norm,
        //       the trainer then only drops the global coefficient and still calls step(), which
        //       arrives here (cached_nsq_ = -1), measures the same NaN again, and went on.
        // Why here: this norm is already computed and all-reduced for clipping, so the check is
        //       free (one isfinite on a host float, once per group per step), it runs BEFORE the
        //       scale and before local_adam_step / step_count_, and every rank holds the same
        //       all-reduced value, so every rank takes the same branch.
        // Now:  with a LossScaler, unchanged (skip + back off). Without one, throw
        //       ZeroNonFiniteGradError; bluscript_zero_ckpt catches it and aborts with exit code 4.
        //       last_grad_norm_ is now set before the guard, as in v1, so a skipped step logs the
        //       non-finite norm instead of the previous step's. A finite norm takes exactly the old
        //       code path, so training is bit-identical.
        // Not covered (same in v1): with max_grad_norm = 0 AND no scaler, need_norm is false and
        //       no norm is measured at all, so nothing can be checked. The trainers set
        //       max_grad_norm = cfg.grad_clip, default 1.0.
        last_grad_norm_ = (nsq > 0.0f || std::isnan(nsq)) ? std::sqrt(nsq) : 0.0f;   // NaN logs as nan, not 0
        if (!std::isfinite(nsq)) {
            if (!sc) throw ZeroNonFiniteGradError(nonfinite_grad_msg(nsq, compute_dtype_, total_numel_,
                                                                     rank_, step_count_ + 1));
            sc->update(true);
            last_step_skipped_ = true;
            return false;
        }

        if (cfg_.max_grad_norm > 0.0f) {
            // Same formula as v1's compute_clip_coef_cuda: max/(norm+1e-6), <= 1.
            // An externally supplied coefficient (cross-group global clipping) wins.
            float coef = external_clip_coef_ >= 0.0f
                       ? external_clip_coef_
                       : cfg_.max_grad_norm / (last_grad_norm_ + 1e-6f);
            if (coef > 1.0f) coef = 1.0f;
            combined *= coef;
        }
    }

    /* edit */
    // BLU_FUSED_CLIP_ADAMW: hand the coefficient to the fused kernel instead of
    // spending a whole pass over the shard applying it here. The value still
    // lands in d_scale_ (via the same blocking H2D, so it is retired before any
    // later launch can read it) -- only the multi_tensor_scale_cuda pass is
    // skipped.
    //
    // NOTE the scale is deferred even when `valid == 0`. A rank whose shard is
    // entirely padding has nothing to scale either way, and pending_scale_
    // is read only by the fused launch, which that rank still takes.
    //
    // combined == 1.0f means there is nothing to apply at all; leave
    // pending_scale_ null so the kernel takes its SCALED=false instantiation and
    // stays bit-identical to the unfused-clip build.
    if (scale_deferred_) {
        pending_scale_ = nullptr;
        if (combined != 1.0f) {
            *h_pinned_ = combined;
            cudaMemcpy(d_scale_, h_pinned_, sizeof(float), cudaMemcpyHostToDevice);
            pending_scale_ = d_scale_;
        }
        return true;
    }
    /* edit */

    // Single elementwise pass. Skipped entirely when it would be a no-op, which
    // preserves v1's W==1 fast path exactly (no scale kernel at all).
    if (valid > 0 && combined != 1.0f) {
        *h_pinned_ = combined;
        cudaMemcpy(d_scale_, h_pinned_, sizeof(float), cudaMemcpyHostToDevice);
        std::vector<ocuda::TensorInfo> gv = {{shard_grad_fp32_.data<float>(),
                                              static_cast<int64_t>(valid)}};
        ocuda::multi_tensor_scale_cuda(gv, d_scale_);
        sync_compute();
    }
    return true;
}

/* edit */
// Fallback for a deferred scale the fused kernel ended up NOT consuming: the
// backend declined on bucket 0, so the unfused pair runs instead and the
// gradients have to be scaled after all. Without this the step would apply Adam
// to gradients still carrying the op=SUM factor of W and no clip -- an 8x
// over-sized update on this model, silently.
void SymmetricShardedOptimizer::apply_deferred_grad_scale() {
    if (!pending_scale_) return;          // nothing was deferred, or it was 1.0
    const size_t valid = valid_shard_prefix();
    if (valid > 0) {
        std::vector<ocuda::TensorInfo> gv = {{shard_grad_fp32_.data<float>(),
                                              static_cast<int64_t>(valid)}};
        ocuda::multi_tensor_scale_cuda(gv, pending_scale_);
        sync_compute();
    }
    pending_scale_ = nullptr;
}
/* edit */

// ======================================================================
// Adam over the (strided-in-global, contiguous-in-shard) shard
// ======================================================================

void SymmetricShardedOptimizer::adam_shard_range(size_t shard_lo, size_t shard_hi,
                                                 float bc1, float bc2) {
    if (shard_hi <= shard_lo) return;

    // Build (global range -> shard range) for the buckets overlapping
    // [shard_lo, shard_hi), then intersect with each weight-decay group. Because
    // the global image is monotone, each group yields ONE contiguous shard run —
    // so this is one fused-Adam launch per group, matching v1's launch count.
    for (const auto& g : groups_) {
        // Find the shard sub-range whose global image lies in [g.begin, g.end).
        size_t s_lo = shard_hi, s_hi = shard_lo;
        for (const auto& b : buckets_) {
            const size_t p_s_lo = b.shard_offset;
            const size_t p_s_hi = b.shard_offset + b.per_rank;
            if (p_s_hi <= shard_lo || p_s_lo >= shard_hi) continue;

            const size_t g_base = b.begin + static_cast<size_t>(rank_) * b.per_rank;
            const size_t a_s = std::max(p_s_lo, shard_lo);
            const size_t b_s = std::min(p_s_hi, shard_hi);
            const size_t a_g = g_base + (a_s - p_s_lo);
            const size_t b_g = g_base + (b_s - p_s_lo);

            const size_t i_g_lo = std::max(a_g, g.begin);
            const size_t i_g_hi = std::min(b_g, g.end);
            if (i_g_hi <= i_g_lo) continue;

            const size_t i_s_lo = a_s + (i_g_lo - a_g);
            const size_t i_s_hi = a_s + (i_g_hi - a_g);
            s_lo = std::min(s_lo, i_s_lo);
            s_hi = std::max(s_hi, i_s_hi);
        }
        if (s_hi <= s_lo) continue;

        const int64_t len = static_cast<int64_t>(s_hi - s_lo);
        std::vector<ocuda::TensorInfo> p  = {{shard_master_.data<float>()    + s_lo, len}};
        std::vector<ocuda::TensorInfo> gr = {{shard_grad_fp32_.data<float>() + s_lo, len}};
        std::vector<ocuda::TensorInfo> m  = {{shard_m_.data<float>()         + s_lo, len}};
        std::vector<ocuda::TensorInfo> v  = {{shard_v_.data<float>()         + s_lo, len}};

        ocuda::multi_tensor_adam_cuda(p, gr, m, v,
                                      cfg_.lr * g.lr_scale, cfg_.beta1, cfg_.beta2,
                                      cfg_.eps, g.weight_decay, bc1, bc2, is_adamw_);
        // The multi-tensor kernels stage their (ptr,numel) launch metadata through
        // a shared static pinned buffer consumed asynchronously; a following
        // launch can clobber it mid-flight. Serialize each launch.
        sync_compute();
    }
}

/* edit */
namespace {
// Read once: getenv on every optimizer step would be silly, and the flag is not
// meant to change mid-run.
bool fused_adamw_opt_in() {
    static const bool on = [] {
        const char* v = std::getenv("BLU_FUSED_ADAMW");
        return v && *v && std::atoi(v) != 0;
    }();
    return on;
}

// BLU_FUSED_CLIP_ADAMW=1 additionally folds the gradient-scale pass
// (1/W * 1/loss_scale * clip_coef) into that same kernel, removing a full
// read-modify-write of shard_grad_fp32_ from every step.
//
// A SEPARATE flag from BLU_FUSED_ADAMW on purpose. The two change different
// things: BLU_FUSED_ADAMW removes a pass over the MASTER weights and replaces
// ncclAllGather with a multicast store; this one removes a pass over the
// GRADIENTS and moves a numerical operation (the clip multiply) from one kernel
// into another. Keeping them separate is what makes a regression bisectable to
// one of those two, and what lets the clip fusion be A/B'd against a fused-AdamW
// baseline rather than only against the unfused path.
//
// It is a no-op unless BLU_FUSED_ADAMW=1, because it has nothing to fuse into
// otherwise. Reported as such at the call site rather than silently ignored.
bool fused_clip_opt_in() {
    static const bool on = [] {
        const char* v = std::getenv("BLU_FUSED_CLIP_ADAMW");
        return v && *v && std::atoi(v) != 0;
    }();
    return on;
}
}  // namespace
/* edit */

/* edit */
// =============================================================================
//  Fused AdamW + AllGather
//
//  The unfused pair makes TWO passes over the updated shard: local_adam_step()
//  writes it to memory, then all_gather_params() reads it back to send it. The
//  fused kernel produces each value in a register and publishes it from there
//  with a multicast store, so the second pass disappears -- along with
//  cast_master_shard_range()'s fp32 d2d into shard_master_compute_, which the
//  fused path bypasses entirely.
//
//  ONE LAUNCH PER BUCKET, mirroring all_gather_params()'s own loop. The kernel
//  publishes this rank's slice at dest[rank * S .. (rank+1) * S), i.e. it
//  assumes a rank-major destination -- which is exactly what ONE BUCKET's flat
//  range is. Handing it the whole multi-bucket flat_stage_ would be wrong,
//  because that buffer is bucket-major on the outside and rank-major only
//  within each bucket.
//
//  PRECONDITIONS, all of them declines rather than errors, because the unfused
//  pair is always available and always correct:
//    - compute_dtype_ == Float32. The publish half stores float4 through the
//      multicast pointer, so flat_stage_ must be fp32. Under BLU_ALL_BF16=1 the
//      large AdamW group is bf16 and is declined here.
//    - is_adamw_. The kernel implements decoupled decay only; there is no
//      coupled-Adam branch in it.
//    - Uniform weight_decay and lr_scale across groups_. The kernel takes ONE
//      wd and ONE lr for the whole launch and cannot sub-range within a bucket.
//    - per_rank a multiple of 4 on every bucket (128-bit pack, no scalar tail).
//    - W_ > 1, and the process group implements the fused entry point (only
//      BluCCL does -- NCCL has no multicast store to fuse the publish into).
// =============================================================================
// Every precondition of the fused path that can be answered WITHOUT asking the
// backend: dtype, optimizer flavour, group uniformity, shard alignment. Split
// out because step() needs the answer one call earlier than
// try_fused_adam_and_publish() can give it -- the scale pass has to know whether
// to run before the update does.
//
// The backend probe (is `flat_stage_` a registered symmetric segment? does this
// process group implement the entry point at all?) is deliberately NOT here. It
// costs a launch, and being wrong about it is harmless: apply_deferred_grad_scale()
// picks the scale back up on the fallback path.
bool SymmetricShardedOptimizer::fused_shape_supports_clip() const {
    if (!pg_ || W_ <= 1) return false;
    if (compute_dtype_ != Dtype::Float32) return false;
    if (!is_adamw_) return false;
    if (groups_.empty()) return false;

    const float wd0 = groups_.front().weight_decay;
    const float ls0 = groups_.front().lr_scale;
    for (const auto& g : groups_)
        if (g.weight_decay != wd0 || g.lr_scale != ls0) return false;

    for (const auto& b : buckets_)
        if ((b.per_rank & 3u) != 0) return false;

    return true;
}

bool SymmetricShardedOptimizer::try_fused_adam_and_publish() {
    if (!fused_shape_supports_clip()) return false;

    const float wd0 = groups_.front().weight_decay;
    const float ls0 = groups_.front().lr_scale;

    const float bc1 = 1.0f - std::pow(cfg_.beta1, static_cast<float>(step_count_));
    const float bc2 = 1.0f - std::pow(cfg_.beta2, static_cast<float>(step_count_));
    const float lr  = cfg_.lr * ls0;

    // The fused kernel runs on the PROCESS GROUP's collective stream, not the
    // compute stream -- see the stream note in ProcessGroupBluCCL::
    // fused_adamw_allgather. So bracket it exactly as all_gather_params() already
    // brackets pg_->all_gather(): make the grads visible first, and do not let
    // scatter_flat_into_params() read flat_stage_ until the publish has landed.
    sync_compute();

    // Probe the first bucket before committing: if the backend declines (not a
    // BluCCL group, or flat_stage_ is not a registered symmetric segment), we
    // must not have half-updated the shard. Bucket 0 either runs or nothing has.
    // pending_scale_ is non-null only when BLU_FUSED_CLIP_ADAMW deferred the
    // gradient scale to this kernel. It is a DEVICE pointer written by the
    // blocking cudaMemcpy in apply_grad_scale_and_check(), which has therefore
    // already retired by the time this launch is enqueued -- the ordering the
    // kernel's launch wrapper requires. It is the SAME scalar for every bucket:
    // the coefficient covers the whole shard, not a sub-range.
    const float* gscale = pending_scale_;

    bool first = true;
    for (const auto& b : buckets_) {
        const bool ok = pg_->fused_clip_adamw_allgather(
            shard_grad_fp32_.data<float>() + b.shard_offset,
            shard_m_.data<float>()         + b.shard_offset,
            shard_v_.data<float>()         + b.shard_offset,
            shard_master_.data<float>()    + b.shard_offset,
            static_cast<char*>(flat_stage_.data()) + b.begin * sizeof(float),
            static_cast<std::int64_t>(b.per_rank),
            lr, cfg_.beta1, cfg_.beta2, cfg_.eps, wd0, bc1, bc2,
            gscale,
            ocuda::getCurrentStream());
        if (!ok) {
            // Only safe to bail on the very first bucket -- see above.
            if (first) return false;
            throw ZeroCollectiveError(
                "fused_clip_adamw_allgather declined mid-shard; the optimizer "
                "state is now partially updated and cannot fall back");
        }
        first = false;
    }
    // Matches the tail of all_gather_params(): the publish is on the group's
    // stream, so a compute-stream sync is not enough to order the read that
    // follows.
    cudaDeviceSynchronize();
    pending_scale_ = nullptr;   // consumed by the kernel
    return true;
}
/* edit */

void SymmetricShardedOptimizer::local_adam_step() {
    const float bc1 = 1.0f - std::pow(cfg_.beta1, static_cast<float>(step_count_));
    const float bc2 = 1.0f - std::pow(cfg_.beta2, static_cast<float>(step_count_));
    adam_shard_range(0, S_, bc1, bc2);
}

void SymmetricShardedOptimizer::cast_master_shard_range(size_t shard_lo, size_t shard_hi) {
    if (shard_hi <= shard_lo) return;
    const size_t esz = elem_size(compute_dtype_);
    // Direct sub-range convert straight into the all-gather send buffer — no
    // full-shard temporary (which as_type() would allocate every step).
    auto cast_piece = [&](size_t lo, size_t n, char* dst) {
        if (compute_dtype_ == Dtype::Float32) {
            d2d(dst, shard_master_.data<float>() + lo, n * sizeof(float), device_);
        } else {
            OwnTensor::convert_type_cuda_generic(
                shard_master_.data<float>() + lo, Dtype::Float32, dst,
                compute_dtype_, static_cast<int64_t>(n), ocuda::getCurrentStream());
        }
    };
    if (!inplace_collectives_) {
        cast_piece(shard_lo, shard_hi - shard_lo,
                   static_cast<char*>(shard_master_compute_.data()) + shard_lo * esz);
        return;
    }
    // In place the send buffer is this rank's slice of each bucket in
    // flat_stage_, so the shard range splits at bucket boundaries. When
    // flat_stage_ doubles as flat_grad_ this overwrites the reduced gradients,
    // which cast_shard_grad_to_fp32() has already consumed.
    for (const auto& b : buckets_) {
        const size_t lo = std::max(shard_lo, b.shard_offset);
        const size_t hi = std::min(shard_hi, b.shard_offset + b.per_rank);
        if (hi <= lo) continue;
        cast_piece(lo, hi - lo, param_wire_base() + (wire_offset(b) + (lo - b.shard_offset)) * esz);
    }
}

// ======================================================================
// Parameter publish
// ======================================================================

void SymmetricShardedOptimizer::all_gather_params() {
    const size_t esz = elem_size(compute_dtype_);
    cast_master_shard_range(0, S_);
    sync_compute();

    if (W_ == 1) {
        // In place the cast above already wrote flat_stage_.
        if (!inplace_collectives_) d2d(flat_stage_.data(), shard_master_compute_.data(), S_ * esz, device_);
        return;
    }
    // One all-gather per bucket. Under PER_BUCKET ownership this is EXACTLY
    // right with no staging: ncclAllGather writes rank-major, and bucket k's
    // rank-major layout IS its flat range [begin, begin + per_rank*W).
    for (const auto& b : buckets_) {
        // In place: send == recv + rank * per_rank, NCCL's in-place all-gather.
        const void* send = param_wire_base() + wire_offset(b) * esz;
        void*       recv = static_cast<char*>(flat_stage_.data()) + b.begin * esz;
        auto r = pg_->all_gather(send, recv, b.per_rank, compute_dtype_, /*sync=*/true);
        if (r != pgSuccess) throw ZeroCollectiveError("all_gather(params) failed");
    }
    cudaDeviceSynchronize();
}

void SymmetricShardedOptimizer::scatter_flat_into_params() {
    const size_t esz = elem_size(compute_dtype_);
    for (const auto& rec : records_) {
        const void* src = static_cast<const char*>(flat_stage_.data()) + rec.offset * esz;
        Tensor p = rec.param;
        d2d(p.data(), src, rec.numel * esz, device_);
    }
    sync_compute();
}

void SymmetricShardedOptimizer::issue_bucket_ag(size_t bidx) {
    CommBucket& b = buckets_[bidx];
    const size_t esz = elem_size(compute_dtype_);
    const void* send = param_wire_base() + wire_offset(b) * esz;
    void*       recv = static_cast<char*>(flat_stage_.data()) + b.begin * esz;
    b.ag_work = pg_->all_gather_async(send, recv, b.per_rank, compute_dtype_,
                                      /*sync=*/false, b.ag_work);
}

// ======================================================================
// Staged parameter refresh
// ======================================================================

void SymmetricShardedOptimizer::set_param_refresh_stages(const std::vector<int>& stage_of,
                                                         int num_stages) {
    if (!cfg_.overlap_param_refresh || W_ <= 1) return;   // stay on the sync path
    if (num_stages < 1) throw ZeroConfigError("num_stages must be >= 1");
    if (param_refresh_pending_) {
        throw ZeroConfigError("set_param_refresh_stages() called with a refresh in flight");
    }

    num_refresh_stages_ = num_stages;
    stage_records_.assign(num_stages, {});

    if (stage_of.size() < records_.size()) {
        // stage_of is indexed by the CONSTRUCTOR's parameter positions, so it
        // must be at least as long as that vector, not as long as records_.
        // records_[i].input_index is the bound we actually dereference.
    }
    for (size_t ri = 0; ri < records_.size(); ++ri) {
        const size_t ii = records_[ri].input_index;
        if (ii >= stage_of.size()) {
            throw ZeroConfigError("stage_of is shorter than the constructor parameter vector");
        }
        const int s = stage_of[ii];
        if (s < 0 || s >= num_stages) {
            throw ZeroConfigError("stage_of contains an out-of-range stage index");
        }
        stage_records_[s].push_back(static_cast<uint32_t>(ri));
    }

    // Assign each bucket the EARLIEST stage of any parameter overlapping it: the
    // bucket must have landed before the first stage that reads any of its bytes.
    // Buckets touched by no parameter (pure padding) ride the last stage.
    for (auto& b : buckets_) b.refresh_stage = num_stages - 1;
    std::vector<int> earliest(buckets_.size(), num_stages);
    for (int s = 0; s < num_stages; ++s) {
        for (uint32_t ri : stage_records_[s]) {
            for (uint32_t bidx : param_to_buckets_[ri]) {
                earliest[bidx] = std::min(earliest[bidx], s);
            }
        }
    }
    for (size_t bidx = 0; bidx < buckets_.size(); ++bidx) {
        if (earliest[bidx] < num_stages) buckets_[bidx].refresh_stage = earliest[bidx];
    }

    // Coverage invariant for adam_and_publish_staged()'s two-phase split. That
    // function partitions the shard by "is this bucket in stage 0", so the union of
    // the two phases must be exactly [0, S_) — otherwise some shard range gets no
    // Adam update and its never-written send buffer is published as zeros. This is
    // structurally guaranteed (build_buckets already verified the shard_offset
    // tiling), but it is the invariant the whole staged path silently depends on, so
    // check it once here rather than trusting it.
    {
        size_t covered = 0;
        for (const auto& b : buckets_) {
            if (b.refresh_stage < 0 || b.refresh_stage >= num_stages) {
                throw ZeroConfigError("internal: bucket refresh_stage out of range");
            }
            if (b.shard_offset != covered) {
                throw ZeroConfigError("internal: bucket shard pieces do not tile the "
                                      "shard in bucket order (staged refresh would "
                                      "skip part of the Adam update)");
            }
            covered += b.per_rank;
        }
        if (covered != S_) {
            throw ZeroConfigError("internal: bucket shard pieces do not cover the shard "
                                  "(staged refresh would skip part of the Adam update)");
        }
    }

    // Issue order: by (stage, bucket index). Derived from data that is identical
    // on every rank, so every rank issues the same all-gathers in the same order
    // on the same in-order comm stream — which is both the NCCL ordering
    // requirement and what makes "stage s landed" imply "all earlier stages landed".
    ag_issue_order_.resize(buckets_.size());
    std::iota(ag_issue_order_.begin(), ag_issue_order_.end(), 0u);
    std::stable_sort(ag_issue_order_.begin(), ag_issue_order_.end(),
                     [&](uint32_t a, uint32_t c) {
                         if (buckets_[a].refresh_stage != buckets_[c].refresh_stage)
                             return buckets_[a].refresh_stage < buckets_[c].refresh_stage;
                         return a < c;
                     });

    overlap_param_active_ = true;
    if (!param_sync_event_) {
        cudaEventCreateWithFlags(&param_sync_event_, cudaEventDisableTiming);
    }
}

void SymmetricShardedOptimizer::adam_and_publish_staged() {
    const float bc1 = 1.0f - std::pow(cfg_.beta1, static_cast<float>(step_count_));
    const float bc2 = 1.0f - std::pow(cfg_.beta2, static_cast<float>(step_count_));

    // Two-phase pipeline, same idea as v1: get stage 0's transfer moving before
    // the rest of Adam runs, because stage 0 (the token embedding) is read first
    // by the next forward and is the one piece that cannot hide behind it.
    //
    // Phase A = the buckets assigned to refresh stage 0; phase B = every other
    // bucket. The two sets are complementary and buckets tile [0,S_) in shard space
    // (build_buckets verifies shard_offset tiling and that the pieces sum to S_), so
    // the two phases' shard ranges partition the shard EXACTLY: every element gets
    // one Adam update and one down-cast, for any stage assignment the caller makes.
    //
    // This must NOT be collapsed back to "phase A = [min offset, max offset), phase
    // B = [that max, S_)". That is only equivalent when the stage-0 buckets happen
    // to be a PREFIX of buckets_. shard_offset being monotone in bucket index gives
    // a monotone SET of offsets, not a contiguous one — so a single stage-0
    // parameter sitting mid-model (which a caller's fallback mapping can easily
    // produce; bluscript_zero.cpp pins every parameter it cannot identify to stage
    // 0) makes min offset > 0, and the range [0, min offset) then receives no Adam
    // update AND no cast_master_shard_range. Since all_gather_params() — the only
    // other writer of shard_master_compute_ — is never called on this path,
    // the all-gather would publish that range's never-written send buffer, i.e.
    // ZEROS, to every rank.
    auto phase_runs = [&](bool want_stage0) {
        std::vector<std::pair<size_t, size_t>> runs;
        // buckets_ is ascending in shard_offset, so adjacent selected buckets merge
        // in a single forward scan — no sort needed.
        for (const auto& b : buckets_) {
            if ((b.refresh_stage == 0) != want_stage0) continue;
            const size_t lo = b.shard_offset;
            const size_t hi = b.shard_offset + b.per_rank;
            if (hi <= lo) continue;
            if (!runs.empty() && runs.back().second == lo) runs.back().second = hi;
            else runs.emplace_back(lo, hi);
        }
        return runs;
    };
    const std::vector<std::pair<size_t, size_t>> a_runs = phase_runs(true);
    const std::vector<std::pair<size_t, size_t>> b_runs = phase_runs(false);

    // ---- phase A: stage 0 only, published the moment its Adam + cast land ----
    if (!a_runs.empty()) {
        for (const auto& r : a_runs) {
            adam_shard_range(r.first, r.second, bc1, bc2);
            cast_master_shard_range(r.first, r.second);
        }
        // compute -> comm edge. Recording the shared event twice per step is safe:
        // phase A's cudaStreamWaitEvent captured this record before phase B
        // re-records it (program order).
        cudaEventRecord(param_sync_event_, ocuda::getCurrentStream());
        cudaStreamWaitEvent(pg_->get_stream(), param_sync_event_, 0);
        for (uint32_t bidx : ag_issue_order_) {
            if (buckets_[bidx].refresh_stage == 0) issue_bucket_ag(bidx);
        }
    }

    // ---- phase B: every remaining bucket, transferring under the next forward ----
    if (!b_runs.empty()) {
        for (const auto& r : b_runs) {
            adam_shard_range(r.first, r.second, bc1, bc2);
            cast_master_shard_range(r.first, r.second);
        }
        cudaEventRecord(param_sync_event_, ocuda::getCurrentStream());
        cudaStreamWaitEvent(pg_->get_stream(), param_sync_event_, 0);
        // In ag_issue_order_ (sorted by (stage, bucket index), identical on every
        // rank) on the one in-order comm stream, so "stage s landed" still implies
        // every earlier stage landed.
        for (uint32_t bidx : ag_issue_order_) {
            if (buckets_[bidx].refresh_stage != 0) issue_bucket_ag(bidx);
        }
    }

    param_refresh_pending_ = true;
    next_refresh_stage_    = 0;
}

void SymmetricShardedOptimizer::refresh_param_stage(int stage) {
    if (!param_refresh_pending_) return;
    const size_t esz = elem_size(compute_dtype_);

    for (int s = next_refresh_stage_; s <= stage && s < num_refresh_stages_; ++s) {
        // comm -> compute edge for every bucket this stage needs. Work::wait()
        // enqueues a cudaStreamWaitEvent on the compute stream; no host stall.
        // A failed all-gather here would leave flat_stage_ holding the previous
        // step's weights for that bucket, which then get scattered into the live
        // model tensors below — so the result is checked, not discarded.
        for (size_t bidx = 0; bidx < buckets_.size(); ++bidx) {
            if (buckets_[bidx].refresh_stage == s) {
                wait_or_throw(buckets_[bidx].ag_work, "parameter all-gather", bidx);
            }
        }
        for (uint32_t ri : stage_records_[s]) {
            const auto& rec = records_[ri];
            const void* src = static_cast<const char*>(flat_stage_.data()) + rec.offset * esz;
            Tensor p = rec.param;
            d2d(p.data(), src, rec.numel * esz, device_);
        }
        next_refresh_stage_ = s + 1;
    }
    if (next_refresh_stage_ >= num_refresh_stages_) param_refresh_pending_ = false;
}

void SymmetricShardedOptimizer::drain_param_refresh() {
    if (!param_refresh_pending_) return;
    refresh_param_stage(num_refresh_stages_ - 1);
}

// ======================================================================
// step()
// ======================================================================

void SymmetricShardedOptimizer::step() {
    last_step_skipped_ = false;

    if (param_refresh_pending_) {
        throw ZeroConfigError("parameter refresh stages were not consumed before "
                              "step() — every stage must be installed during the "
                              "forward pass (refresh_param_stage) or explicitly "
                              "drained (drain_param_refresh)");
    }

    if (cfg_.grad_accum_steps > 1) {
        ++micro_step_;
        if (micro_step_ < cfg_.grad_accum_steps) return;
        micro_step_ = 0;
    }

    if (!grads_reduced_) reduce_grads_now();
    grads_reduced_ = false;                       // one reduction per step

    /* edit */
    // Decided BEFORE the scale pass, because the decision is what tells that pass
    // whether to run at all. fused_shape_supports_clip() repeats the cheap,
    // purely local preconditions of try_fused_adam_and_publish(); it deliberately
    // does NOT probe the backend, because that probe costs a kernel launch and a
    // wrong guess here is recoverable -- apply_deferred_grad_scale() applies the
    // coefficient the old way if the backend turns the fusion down.
    scale_deferred_ = !overlap_param_active_
                      && fused_adamw_opt_in()
                      && fused_clip_opt_in()
                      && fused_shape_supports_clip();
    /* edit */

    const bool ok = apply_grad_scale_and_check();
    external_clip_coef_ = -1.0f;                  // valid for exactly one step
    cached_nsq_         = -1.0f;
    if (!ok) {                                    // non-finite: skip the step
        // The gradients are discarded with the step, so the deferred coefficient
        // must not survive into the next one -- it was computed from THIS step's
        // norm.
        pending_scale_  = nullptr;
        scale_deferred_ = false;
        return;
    }

    ++step_count_;

    if (overlap_param_active_) {
        // The staged path runs adam_shard_range() per refresh stage and never
        // reaches the fused kernel, so scale_deferred_ is false above and the
        // scale has already been applied in full by apply_grad_scale_and_check().
        adam_and_publish_staged();
    } else {
        /* edit */
        // BLU_FUSED_ADAMW=1 opts in; the call still declines on its own if the
        // shape or dtype cannot take it, so the flag alone never forces it.
        bool fused = false;
        if (fused_adamw_opt_in()) {
            fused = try_fused_adam_and_publish();
            // Say ONCE which path the flag actually got. A flag that silently
            // does nothing is the failure mode worth spending six lines on.
            static bool reported = false;
            if (!reported && rank_ == 0) {
                reported = true;
                if (fused) {
                    std::cout << "[ZeRO-1 v2] BLU_FUSED_ADAMW=1 ACTIVE: AdamW+AllGather fused ("
                              << buckets_.size() << " bucket(s), S=" << S_ << ")";
                    // Which of the two fusions actually engaged. scale_deferred_
                    // is the honest test: fused_clip_opt_in() alone would claim
                    // the clip is fused on a group whose shape declined it.
                    if (scale_deferred_) {
                        std::cout << " + BLU_FUSED_CLIP_ADAMW=1: grad scale folded in "
                                     "(no separate multi_tensor_scale pass)";
                    } else if (fused_clip_opt_in()) {
                        std::cout << "  [BLU_FUSED_CLIP_ADAMW=1 set but NOT engaged for "
                                     "this group]";
                    }
                    std::cout << std::endl;
                } else {
                    std::cout << "[ZeRO-1 v2] BLU_FUSED_ADAMW=1 DECLINED -- running the unfused "
                                 "adam+all_gather. compute_dtype="
                              << (compute_dtype_ == Dtype::Float32 ? "float32" : "non-fp32")
                              << " is_adamw=" << (is_adamw_ ? 1 : 0)
                              << " groups=" << groups_.size()
                              << " W=" << W_
                              << " (fp32 + AdamW + uniform wd/lr + per_rank%4==0 + BluCCL backend "
                                 "are all required)" << std::endl;
                }
            }
        }
        if (!fused) {
            // The fused kernel did not run, so nothing consumed a deferred
            // gradient scale. Apply it before Adam reads the gradients.
            apply_deferred_grad_scale();
            local_adam_step();
            all_gather_params();
        }
        pending_scale_  = nullptr;   // valid for exactly one step
        scale_deferred_ = false;
        /* edit */
        scatter_flat_into_params();
    }

    if (auto* sc = get_scaler()) sc->update(false);
}

void SymmetricShardedOptimizer::zero_grad() {
    // Stage 0 must be installed before the forward reads the embedding, and
    // zero_grad() is the first optimizer call of every iteration.
    if (param_refresh_pending_) refresh_param_stage(0);
    /* edit */
    // One memset over the whole flat buffer instead of N per-tensor fills; it
    // also covers the padding tail that arm_grad_sync() zeroes separately.
    if (grad_views_active_) {
        cudaMemsetAsync(flat_grad_.data(), 0,
                        padded_numel_ * elem_size(reduce_dtype_),
                        ocuda::getCurrentStream());
        return;
    }
    /* edit */
    for (auto& rec : records_) {
        Tensor p = rec.param;
        p.zero_grad();
    }
}

// ======================================================================
// Checkpoint
// ======================================================================

void SymmetricShardedOptimizer::save_state(std::ostream& os) {
    ShardLayoutDescriptor desc;
    desc.world_size   = W_;
    desc.total_numel  = static_cast<int64_t>(total_numel_);
    desc.padded_numel = static_cast<int64_t>(padded_numel_);
    desc.shard_numel  = static_cast<int64_t>(S_);
    desc.param_count  = static_cast<int32_t>(records_.size());
    desc.num_groups   = static_cast<int32_t>(groups_.size());
    desc.layout_hash  = layout_hash_;
    desc.ownership    = static_cast<int32_t>(ShardOwnership::PER_BUCKET);
    desc.bucket_elems = static_cast<int64_t>(bucket_elems_);

    os.write(reinterpret_cast<const char*>(&desc), sizeof(desc));
    os.write(reinterpret_cast<const char*>(&step_count_), sizeof(int64_t));

    OwnTensor::save_tensor(shard_master_, os);
    OwnTensor::save_tensor(shard_m_, os);
    OwnTensor::save_tensor(shard_v_, os);
}

// Convert a CONTIGUOUS (v1) shard into this rank's PER_BUCKET shard.
//
// The two schemes give the same rank DIFFERENT global elements, so this is not a
// local permutation — the data has to move between ranks. Reconstruct the full
// flat FP32 array with one all-gather (a v1 shard is contiguous and rank-major,
// so a single ncclAllGather of S elements IS the full array), then copy out the
// pieces this rank owns under PER_BUCKET.
//
// Peak extra memory: padded_numel_ * 4 bytes, allocated and freed per tensor
// (master, then m, then v) rather than all three at once. For a 1.6 B-parameter
// group that is ~6.5 GB during load only, with no activations live.
Tensor SymmetricShardedOptimizer::reshard_from_contiguous(const Tensor& src) {
    TensorOptions of = TensorOptions().with_dtype(Dtype::Float32).with_device(device_);
    Tensor dst = Tensor::zeros(flat_shape(S_), of);

    if (W_ == 1) {
        d2d(dst.data(), src.data(), S_ * sizeof(float), device_);
        return dst;
    }

    Tensor full = Tensor::zeros(flat_shape(padded_numel_), of);
    sync_compute();
    auto r = pg_->all_gather(src.data(), full.data(), S_, Dtype::Float32, /*sync=*/true);
    if (r != pgSuccess) throw ZeroCollectiveError("all_gather during checkpoint reshard failed");
    cudaDeviceSynchronize();

    for (const auto& b : buckets_) {
        const size_t g_lo = b.begin + static_cast<size_t>(rank_) * b.per_rank;
        d2d(dst.data<float>() + b.shard_offset,
            full.data<float>() + g_lo,
            b.per_rank * sizeof(float), device_);
    }
    sync_compute();
    return dst;   // `full` is released here
}

void SymmetricShardedOptimizer::load_state(std::istream& is) {
    drain_param_refresh();

    ShardLayoutDescriptor desc;
    is.read(reinterpret_cast<char*>(&desc), kShardLayoutDescriptorV1Bytes);
    if (desc.magic != ShardLayoutDescriptor{}.magic || desc.version < 1 || desc.version > 2) {
        throw ZeroCheckpointError("bad magic/version in ZeRO optimizer state");
    }
    if (desc.version >= 2) {
        is.read(reinterpret_cast<char*>(&desc) + kShardLayoutDescriptorV1Bytes,
                sizeof(desc) - kShardLayoutDescriptorV1Bytes);
    } else {
        // A version-1 record predates the ownership field and is, by definition,
        // v1's contiguous layout.
        desc.ownership    = static_cast<int32_t>(ShardOwnership::CONTIGUOUS);
        desc.bucket_elems = 0;
    }

    if (desc.world_size != W_ ||
        desc.total_numel != static_cast<int64_t>(total_numel_)) {
        throw ZeroCheckpointError("layout mismatch (cross-world-size resume not "
                                  "supported): checkpoint W=" +
                                  std::to_string(desc.world_size) + " current W=" +
                                  std::to_string(W_));
    }
    if (desc.param_count != static_cast<int32_t>(records_.size()) ||
        desc.num_groups  != static_cast<int32_t>(groups_.size())  ||
        desc.layout_hash != layout_hash_) {
        throw ZeroCheckpointError("layout mismatch: parameter ordering/grouping "
                                  "differs from the checkpoint. Resume with the "
                                  "identical model and ZeroConfig grouping.");
    }

    const auto own = static_cast<ShardOwnership>(desc.ownership);
    if (own == ShardOwnership::PER_BUCKET) {
        // Same scheme — but the bucket geometry must also match, or the same
        // shard_numel would name different global elements.
        if (desc.shard_numel != static_cast<int64_t>(S_) ||
            desc.bucket_elems != static_cast<int64_t>(bucket_elems_)) {
            throw ZeroCheckpointError("PER_BUCKET checkpoint has a different bucket "
                                      "geometry (checkpoint bucket_elems=" +
                                      std::to_string(desc.bucket_elems) + " current=" +
                                      std::to_string(bucket_elems_) +
                                      "). Resume with the same comm_bucket_bytes.");
        }
    } else if (desc.shard_numel != static_cast<int64_t>(S_)) {
        // v1 pads only to a multiple of W; v2 pads to a multiple of
        // W*(16/esz). When those differ the shards are different sizes and the
        // gathered array would be mis-sized.
        throw ZeroCheckpointError("CONTIGUOUS checkpoint shard size " +
                                  std::to_string(desc.shard_numel) +
                                  " does not match this run's " + std::to_string(S_) +
                                  " (v1 pads to a multiple of W, v2 to a multiple of "
                                  "W*16/elem_size). Reshard offline.");
    }

    is.read(reinterpret_cast<char*>(&step_count_), sizeof(int64_t));

    Tensor m_in = OwnTensor::load_tensor(is).to(device_);
    Tensor mm_in= OwnTensor::load_tensor(is).to(device_);
    Tensor v_in = OwnTensor::load_tensor(is).to(device_);

    if (own == ShardOwnership::CONTIGUOUS) {
        if (cfg_.verbose && rank_ == 0) {
            std::cout << "[ZeRO-1 v2] resharding a CONTIGUOUS (v1) checkpoint into "
                         "PER_BUCKET ownership" << std::endl;
        }
        shard_master_ = reshard_from_contiguous(m_in);
        shard_m_      = reshard_from_contiguous(mm_in);
        shard_v_      = reshard_from_contiguous(v_in);
    } else {
        shard_master_ = m_in;
        shard_m_      = mm_in;
        shard_v_      = v_in;
    }

    all_gather_params();
    scatter_flat_into_params();
}


// ======================================================================
// DCP flat-shard interface  (OwnTensor::dcp::ShardedOptimizerState)
// ======================================================================
//
// v2 ownership is PER_BUCKET: this rank owns a `per_rank`-long piece of EVERY
// bucket, so its shard is a union of strided ranges in the global flat space,
// not one contiguous run. dcp::OptimStateShardSpec can only describe a single
// contiguous [flat_offset, flat_offset+flat_numel) range, so emit ONE SPEC PER
// (state, bucket): each is contiguous globally, and together they cover exactly
// this rank's shard. DCP's load then intersects each saved range against the
// resuming rank's range, which is what makes a different world size work.
//
// Deliberately NOT resharding to a contiguous view at save time: that would
// need an all-gather into a full padded FP32 buffer (~6.5 GB for a 1.62 B
// group), which is the cost reshard_from_contiguous() already pays on the
// stream path and is not worth paying every checkpoint.
//
// Consequence: v1 and v2 write DIFFERENT tensor names, so DCP cannot resume one
// variant from the other's checkpoint. It fails as a coverage error, loudly.

namespace {
constexpr const char* kZeroMaster = "zero_master";
constexpr const char* kZeroM      = "zero_m";
constexpr const char* kZeroV      = "zero_v";

// param_fqn is "<scope>.b<bucket_index>". Returns false if it is not ours.
bool parse_bucket_fqn(const std::string& fqn, const std::string& scope,
                      size_t nbuckets, size_t& out_index) {
    const std::string prefix = scope + ".b";
    if (fqn.compare(0, prefix.size(), prefix) != 0) return false;
    const std::string digits = fqn.substr(prefix.size());
    if (digits.empty()) return false;
    for (char c : digits) if (c < '0' || c > '9') return false;
    out_index = static_cast<size_t>(std::stoull(digits));
    return out_index < nbuckets;
}
}  // namespace

std::vector<OwnTensor::dcp::OptimStateShardSpec>
SymmetricShardedOptimizer::dcp_state_shard_specs() {
    std::vector<OwnTensor::dcp::OptimStateShardSpec> out;
    out.reserve(3 * buckets_.size());
    for (const char* st : {kZeroMaster, kZeroM, kZeroV}) {
        for (size_t i = 0; i < buckets_.size(); ++i) {
            const auto& b = buckets_[i];
            if (b.per_rank == 0) continue;
            OwnTensor::dcp::OptimStateShardSpec s;
            s.state_type   = st;
            s.param_fqn    = dcp_scope_ + ".b" + std::to_string(i);
            s.kind         = OwnTensor::dcp::StateShardKind::FlatSharded;
            s.global_numel = static_cast<int64_t>(padded_numel_);
            s.flat_offset  = static_cast<int64_t>(b.begin +
                                 static_cast<size_t>(rank_) * b.per_rank);
            s.flat_numel   = static_cast<int64_t>(b.per_rank);
            out.push_back(std::move(s));
        }
    }
    return out;
}

OwnTensor::Tensor*
SymmetricShardedOptimizer::dcp_buffer_for(const std::string& state_type) {
    if (state_type == kZeroMaster) return &shard_master_;
    if (state_type == kZeroM)      return &shard_m_;
    if (state_type == kZeroV)      return &shard_v_;
    return nullptr;
}

OwnTensor::Tensor
SymmetricShardedOptimizer::dcp_get_state_shard(const std::string& state_type,
                                               const std::string& param_fqn) {
    size_t bi = 0;
    if (!parse_bucket_fqn(param_fqn, dcp_scope_, buckets_.size(), bi)) {
        throw ZeroCheckpointError("DCP: param_fqn '" + param_fqn +
                                  "' is not a bucket of scope '" + dcp_scope_ + "'");
    }
    Tensor* src = dcp_buffer_for(state_type);
    if (!src) throw ZeroCheckpointError("DCP: unknown state_type '" + state_type + "'");

    // The publish pipeline reads shard_master_; retire it before handing it out.
    drain_param_refresh();

    const auto& b = buckets_[bi];
    // A VIEW, not a copy. DCP stages whatever this returns with an ASYNCHRONOUS
    // D2H and does not synchronise until it has walked EVERY spec, so a freshly
    // allocated temporary here would be destroyed at the end of the caller's loop
    // iteration -- returning its block to the caching allocator while the copy is
    // still in flight. The next same-sized shard is then handed that same block,
    // overwrites it, and BOTH land on disk holding the second one's bytes. v1's
    // ShardedOptimizer::dcp_get_state_shard has always returned a view, which is
    // exactly why v1 checkpoints never showed this. Returning a view also drops a
    // shard-sized allocation and copy per state from every save.
    //
    // 1-D on purpose: DCP writes 1-D metadata for a FlatSharded spec and its
    // loader requests a 1-D region, so a [1,N] tensor would be rejected at load
    // with "Rank mismatch in load_tensor_region".
    //
    // The shard buffers are written on the compute stream, and the staging D2H
    // runs on the stager's own stream; sync so it reads this step's bytes.
    sync_compute();
    return src->slice_inplace(b.shard_offset, b.per_rank)
               .view(Shape{{static_cast<int64_t>(b.per_rank)}});
}

void SymmetricShardedOptimizer::dcp_set_state_shard(const std::string& state_type,
                                                    const std::string& param_fqn,
                                                    const Tensor& flat_slice) {
    size_t bi = 0;
    if (!parse_bucket_fqn(param_fqn, dcp_scope_, buckets_.size(), bi)) {
        throw ZeroCheckpointError("DCP: param_fqn '" + param_fqn +
                                  "' is not a bucket of scope '" + dcp_scope_ + "'");
    }
    Tensor* dst = dcp_buffer_for(state_type);
    if (!dst) throw ZeroCheckpointError("DCP: unknown state_type '" + state_type + "'");

    const auto& b = buckets_[bi];
    if (static_cast<size_t>(flat_slice.numel()) != b.per_rank) {
        throw ZeroCheckpointError("DCP: bucket element count mismatch for '" +
                                  param_fqn + "/" + state_type + "': checkpoint " +
                                  std::to_string(flat_slice.numel()) +
                                  " vs expected " + std::to_string(b.per_rank));
    }
    if (flat_slice.dtype() != Dtype::Float32) {
        throw ZeroCheckpointError("DCP: shard '" + state_type + "' must be FP32");
    }
    Tensor src = flat_slice.to(device_);
    d2d(dst->data<float>() + b.shard_offset, src.data(),
        b.per_rank * sizeof(float), device_);
    sync_compute();
}

void SymmetricShardedOptimizer::dcp_after_load() {
    all_gather_params();
    scatter_flat_into_params();
}

std::map<std::string, float> SymmetricShardedOptimizer::get_config() {
    return { { dcp_scope_ + ".step_count", static_cast<float>(step_count_) } };
}

void SymmetricShardedOptimizer::set_config(const std::map<std::string, float>& config) {
    auto it = config.find(dcp_scope_ + ".step_count");
    if (it != config.end()) step_count_ = static_cast<int64_t>(it->second);
}

} // namespace zero
} // namespace owt
