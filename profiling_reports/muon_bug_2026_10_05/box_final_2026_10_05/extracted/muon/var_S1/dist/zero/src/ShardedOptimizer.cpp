//======================================================================
// ShardedOptimizer.cpp — ZeRO Stage-1 implementation.
// See ShardedOptimizer.h and ZERO_OPTIMIZER_DESIGN.md.
//======================================================================

#include "ShardedOptimizer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>

#include <cuda_runtime.h>

#include "device/DeviceCore.h"        // OwnTensor::cuda::getCurrentStream()
#include "device/DeviceTransfer.h"    // OwnTensor::device::copy_memory
#include "ops/helpers/MultiTensorKernels.h"
#include "ops/helpers/GradNormKernels.h"
#include "ops/helpers/ConversionKernels.cuh"  // convert_type_cuda_generic (sub-range cast)
#include "core/Serialization.h"       // save_tensor / load_tensor
#include "autograd/Hooks.h"           // make_post_acc_hook (compute/comm overlap)
#include "autograd/Engine.h"          // get_execution_mode / ExecutionMode

namespace owt {
namespace zero {

using OwnTensor::Tensor;
using OwnTensor::Dtype;
using OwnTensor::Shape;
using OwnTensor::TensorOptions;
namespace dev = OwnTensor::device;
namespace ocuda = OwnTensor::cuda;

// -------- small local utilities ---------------------------------------

namespace {

inline size_t elem_size(Dtype d) { return Tensor::dtype_size(d); }

inline Shape flat_shape(size_t n) {
    return Shape{{1, static_cast<int64_t>(n)}};
}

// Device-to-device raw copy of `bytes` bytes on the current stream's device.
inline void d2d(void* dst, const void* src, size_t bytes, OwnTensor::DeviceIndex di) {
    if (bytes == 0) return;
    dev::copy_memory(dst, di.device, src, di.device, bytes);
}

inline void zero_async(void* ptr, size_t bytes) {
    if (bytes == 0) return;
    cudaMemsetAsync(ptr, 0, bytes, ocuda::getCurrentStream());
}

inline void sync_compute() { cudaStreamSynchronize(ocuda::getCurrentStream()); }

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

} // anonymous namespace

// ======================================================================
// Construction
// ======================================================================

ShardedOptimizer::ShardedOptimizer(const std::vector<Tensor>& model_parameters,
                                   std::shared_ptr<dist::comm::ProcessGroupBase> pg,
                                   ZeroConfig cfg)
    : ZeroOptimizerBase(),        // its ctor hands the base optimizer an EMPTY
      cfg_(cfg),                  // param list — handing it the real params would
      pg_(std::move(pg))          // build full-size FP32 masters, defeating ZeRO
{
    if (cfg_.strategy != ShardStrategy::STAGE_1) {
        throw ZeroConfigError("only ShardStrategy::STAGE_1 is implemented");
    }
    if (cfg_.variant == AdamVariant::ADAMW)      is_adamw_ = true;
    else if (cfg_.variant == AdamVariant::ADAM)  is_adamw_ = false;

    W_    = pg_ ? pg_->get_worldsize() : 1;
    rank_ = pg_ ? pg_->get_rank() : 0;
    if (W_ < 1) W_ = 1;
    if (W_ > 1 && !pg_) {
        throw ZeroConfigError("world_size > 1 requires a process group");
    }

    build_layout(model_parameters);
    allocate_buffers();
    init_master_from_params();
    if (cfg_.verify_consistency && W_ > 1) verify_consistency();

    // ---- optional compute/communication overlap setup ----
    // Only meaningful with real cross-rank comm (W>1). The bucketed-reduce
    // schedule fires NCCL collectives from inside the autograd backward hooks, so
    // it requires the SEQUENTIAL engine (hooks + the mop-up run inline on the
    // backward thread with a well-defined current stream). It also assumes the
    // caller drives gradient accumulation externally and arms exactly once before
    // the final micro-step (see arm_grad_sync); the optimizer's own internal
    // grad_accum gate would desynchronize arming from the reduce, so reject it.
    overlap_active_ = cfg_.overlap_backward_comm && W_ > 1;
    if (overlap_active_) {
        if (OwnTensor::autograd::get_execution_mode() !=
            OwnTensor::autograd::ExecutionMode::SEQUENTIAL) {
            throw ZeroConfigError("overlap_backward_comm requires the SEQUENTIAL "
                                  "autograd engine (backward hooks launch NCCL)");
        }
        if (cfg_.grad_accum_steps > 1) {
            throw ZeroConfigError("overlap_backward_comm is incompatible with the "
                                  "optimizer's internal grad_accum_steps>1; drive "
                                  "gradient accumulation in the training loop and "
                                  "call arm_grad_sync() before the last backward");
        }
        build_buckets();
        register_overlap_hooks();
        cudaEventCreateWithFlags(&compute_sync_event_, cudaEventDisableTiming);
    }

    if (cfg_.verbose && rank_ == 0) {
        double shard_mb = (double)S_ * sizeof(float) * 3.0 / (1024.0 * 1024.0);
        double full_mb  = (double)total_numel_ * sizeof(float) * 3.0 / (1024.0 * 1024.0);
        std::cout << "[ZeRO-1] params=" << total_numel_
                  << " padded=" << padded_numel_
                  << " W=" << W_ << " shard=" << S_
                  << " | optimizer state/GPU: " << shard_mb << " MB"
                  << " (vs " << full_mb << " MB unsharded)" << std::endl;
    }
}

ShardedOptimizer::~ShardedOptimizer() {
    if (d_norm_sq_)     cudaFree(d_norm_sq_);
    if (d_clip_coef_)   cudaFree(d_clip_coef_);
    if (h_norm_pinned_) cudaFreeHost(h_norm_pinned_);
    if (compute_sync_event_) cudaEventDestroy(compute_sync_event_);
    if (param_sync_event_)   cudaEventDestroy(param_sync_event_);
}

// ======================================================================
// Layout: filter trainable params, order group-contiguously, compute shards.
// ======================================================================

void ShardedOptimizer::build_layout(const std::vector<Tensor>& model_parameters) {
    // 1. Keep only trainable params (frozen params are never touched by the
    //    optimizer — they are simply absent from the flat buffer).
    std::vector<Tensor> trainable;
    std::vector<size_t> trainable_input_index;   // -> position in model_parameters
    trainable.reserve(model_parameters.size());
    trainable_input_index.reserve(model_parameters.size());
    for (size_t i = 0; i < model_parameters.size(); ++i) {
        if (model_parameters[i].requires_grad()) {
            trainable.push_back(model_parameters[i]);
            trainable_input_index.push_back(i);
        }
    }
    if (trainable.empty()) {
        throw ZeroConfigError("no trainable parameters supplied");
    }

    device_       = trainable[0].device();
    compute_dtype_ = trainable[0].dtype();
    // v1 reduces gradients ONLY in the compute dtype -- there is no separate
    // wire buffer at all (a differing reduce dtype, either direction, would
    // need a second full-size staging buffer; deferred -- see
    // SymmetricShardedOptimizer / ShardedMuon, which already carry one). Fail
    // loudly instead of silently ignoring an explicit reduce_dtype override
    // the compute dtype doesn't already satisfy.
    const Dtype requested_reduce_dtype =
        resolve_reduce_dtype(cfg_.reduce_dtype, compute_dtype_);
    if (requested_reduce_dtype != compute_dtype_) {
        throw ZeroConfigError("reduce_dtype=" + OwnTensor::get_dtype_name(requested_reduce_dtype) +
                              " is not supported in v1 for a model whose compute dtype is " +
                              OwnTensor::get_dtype_name(compute_dtype_) + " (gradients reduce in "
                              "the compute dtype only); use ZeRO-1 v2 (BLU_ZERO_V2=1) or "
                              "ShardedMuon instead");
    }
    reduce_dtype_  = compute_dtype_;

    for (const auto& p : trainable) {
        if (p.dtype() != compute_dtype_) {
            throw ZeroConfigError("all parameters must share one dtype");
        }
        if (p.device().device != device_.device ||
            p.device().index  != device_.index) {
            throw ZeroConfigError("all parameters must live on one device");
        }
    }

    // 2. Assign a weight-decay group id and order params so each group is a
    //    single contiguous run in the flat buffer (needed to pass a per-group
    //    scalar weight decay into the fused Adam kernel).
    //    group 0 -> weight_decay applies; group 1 -> weight_decay = 0.
    std::vector<size_t> idx(trainable.size());
    std::iota(idx.begin(), idx.end(), 0);
    auto group_of = [&](const Tensor& p) -> uint8_t {
        if (cfg_.exclude_1d_from_weight_decay && p.ndim() < 2) return 1;
        return 0;
    };
    std::stable_sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
        return group_of(trainable[a]) < group_of(trainable[b]);
    });

    // 3. Build records + running flat offsets.
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

    // 4. Pad to the W-INDEPENDENT checkpoint granule and compute this rank's
    //    contiguous shard.
    //
    //    This used to pad to a multiple of W, which made padded_numel_ -- and so
    //    the length of the single flat record this group writes -- a function of
    //    the world size. Resuming at a larger W then built a longer padded vector
    //    than the checkpoint held, and the tail past the saved end was covered by
    //    nothing; DCP reports that as a coverage failure. A fixed granule makes
    //    every world size describe the SAME [0, padded_numel) coordinate system,
    //    so the ranges simply intersect. See zero_ckpt_align_vecs() for why 3840
    //    vectors and how to override it.
    const size_t esz_min   = std::min(elem_size(compute_dtype_), elem_size(reduce_dtype_));
    const size_t vec_elems = (esz_min >= 16) ? 1 : (16 / esz_min);
    const size_t granule   = zero_ckpt_align_vecs() * vec_elems;
    if ((granule % static_cast<size_t>(W_)) != 0) {
        throw ZeroConfigError(
            "ZeRO-1: world size " + std::to_string(W_) + " does not divide the "
            "checkpoint alignment granule (" + std::to_string(zero_ckpt_align_vecs()) +
            " vectors). Set BLU_ZERO_CKPT_ALIGN to a multiple of " +
            std::to_string(W_) + " -- checkpoints only interchange between runs "
            "that agree on it, so keep one value for the whole run.");
    }
    padded_numel_ = ((total_numel_ + granule - 1) / granule) * granule;
    S_ = padded_numel_ / static_cast<size_t>(W_);
    shard_begin_ = static_cast<size_t>(rank_) * S_;

    // 5. Collapse records into contiguous (weight_decay, lr) group runs.
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

    // Layout identity: an FNV-1a hash over the ordered (numel, group_id)
    // sequence and each group's (begin,end,weight_decay). A checkpoint may only
    // be loaded into a run whose layout_hash matches, so any change to the flat
    // ordering or grouping (e.g. flipping exclude_1d_from_weight_decay, or
    // adding/removing/reordering params) is caught before optimizer state is
    // remapped onto the wrong parameters.
    auto fnv = [](uint64_t h, uint64_t x) {
        for (int b = 0; b < 8; ++b) { h ^= (x & 0xff); h *= 1099511628211ull; x >>= 8; }
        return h;
    };
    uint64_t h = 1469598103934665603ull;
    for (const auto& rec : records_) {
        h = fnv(h, rec.numel);
        h = fnv(h, rec.group_id);
    }
    for (const auto& g : groups_) {
        h = fnv(h, g.begin);
        h = fnv(h, g.end);
        uint32_t wd_bits; std::memcpy(&wd_bits, &g.weight_decay, sizeof(float));
        h = fnv(h, wd_bits);
    }
    layout_hash_ = h;
}

void ShardedOptimizer::allocate_buffers() {
    TensorOptions oc = TensorOptions().with_dtype(compute_dtype_).with_device(device_);
    TensorOptions of = TensorOptions().with_dtype(Dtype::Float32).with_device(device_);

    flat_stage_           = Tensor::zeros(flat_shape(padded_numel_), oc);
    shard_grad_           = Tensor::zeros(flat_shape(S_), oc);
    shard_grad_fp32_      = Tensor::zeros(flat_shape(S_), of);
    shard_master_         = Tensor::zeros(flat_shape(S_), of);
    shard_m_              = Tensor::zeros(flat_shape(S_), of);
    shard_v_              = Tensor::zeros(flat_shape(S_), of);
    shard_master_compute_ = Tensor::zeros(flat_shape(S_), oc);

    cudaMalloc(&d_norm_sq_,   sizeof(float));
    cudaMalloc(&d_clip_coef_, sizeof(float));
    cudaMallocHost(&h_norm_pinned_, sizeof(float));
    *h_norm_pinned_ = 0.0f;
}

// Copy the current model params into flat_stage_, (optionally) broadcast from
// rank 0 so all ranks agree, scatter back, then seed the FP32 master shard.
void ShardedOptimizer::init_master_from_params() {
    const size_t esz = elem_size(compute_dtype_);

    // params -> flat_stage_
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
        // scatter the synced params back into the model tensors
        for (const auto& rec : records_) {
            const void* src = static_cast<const char*>(flat_stage_.data()) + rec.offset * esz;
            Tensor p = rec.param;  // non-const handle (shares storage) for mutable data()
            d2d(p.data(), src, rec.numel * esz, device_);
        }
        sync_compute();
    }

    // seed FP32 master with this rank's shard of the (synced) params
    Tensor shard_view = flat_stage_.slice_inplace(shard_begin_, S_);   // compute-dtype view
    if (compute_dtype_ == Dtype::Float32) {
        d2d(shard_master_.data(), shard_view.data(), S_ * sizeof(float), device_);
    } else {
        Tensor f = shard_view.as_type(Dtype::Float32);
        d2d(shard_master_.data(), f.data(), S_ * sizeof(float), device_);
    }
    sync_compute();
}

// Cheap cross-rank check that everyone agrees on the flat size (F2). A
// mismatch here is the classic silent-divergence / hang bug.
void ShardedOptimizer::verify_consistency() {
    // NCCL collectives require DEVICE buffers — stage the scalar through the GPU
    // (d_norm_sq_ is already allocated and unused at construction time).
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
// Compute / communication overlap (ZeroConfig::overlap_backward_comm)
//
// Instead of one monolithic reduce-scatter issued after backward, the flat
// buffer is partitioned into contiguous byte-buckets, each kept inside a single
// owner rank's shard. A per-parameter post-accumulate-grad hook copies each
// gradient into its flat slice DURING the (armed) final micro-step's backward;
// when a bucket's last parameter lands, its ncclReduce(op=avg,root=owner) fires
// on the comm stream, overlapping with the remaining backward compute. Summing
// the per-bucket reduces reproduces the reduce-scatter result bit-for-bit
// (modulo BF16 reduction order), so shard_grad_ / Adam / all-gather downstream
// are unchanged.
//
// Correctness relies on the SEQUENTIAL engine: hooks fire in a deterministic,
// data-independent, cross-rank-identical order, so every rank issues the bucket
// reduces (different roots) in the same order — the NCCL ordering contract.
// ======================================================================

void ShardedOptimizer::build_buckets() {
    const size_t esz = elem_size(compute_dtype_);
    size_t bucket_elems = cfg_.comm_bucket_bytes / esz;
    if (bucket_elems == 0) bucket_elems = 1;

    // 1. Cut the flat buffer into ranges, splitting at every shard boundary
    //    (k*S_) so each bucket has exactly one owner, and again whenever a bucket
    //    would exceed comm_bucket_bytes.
    buckets_.clear();
    size_t pos = 0;
    while (pos < padded_numel_) {
        const size_t owner     = pos / S_;
        const size_t shard_end = (owner + 1) * S_;
        const size_t end = std::min(pos + bucket_elems, std::min(shard_end, padded_numel_));
        CommBucket b;
        b.begin = pos;
        b.end   = end;
        b.owner = static_cast<int>(owner);
        b.local_recv_offset = pos - owner * S_;
        buckets_.push_back(b);
        pos = end;
    }

    // 2. Map each parameter to the buckets its (contiguous) flat range overlaps,
    //    counting distinct params per bucket. A param straddling a shard boundary
    //    or bucket cut overlaps >1 bucket and is counted in each — so the bucket
    //    only fires once ALL its params (including the straddler) have landed.
    //    records_ and buckets_ are both in ascending flat order, so a single
    //    forward scan suffices.
    param_to_buckets_.assign(records_.size(), {});
    size_t bi = 0;
    for (size_t pidx = 0; pidx < records_.size(); ++pidx) {
        const size_t p_begin = records_[pidx].offset;
        const size_t p_end   = p_begin + records_[pidx].numel;
        while (bi < buckets_.size() && buckets_[bi].end <= p_begin) ++bi;
        for (size_t j = bi; j < buckets_.size() && buckets_[j].begin < p_end; ++j) {
            param_to_buckets_[pidx].push_back(static_cast<uint32_t>(j));
            buckets_[j].pending_init += 1;
        }
    }
    for (auto& b : buckets_) b.pending = b.pending_init;

    // 3. Scratch recv buffer for buckets this rank does not own: ncclReduce needs
    //    a valid recvbuff on non-root ranks even though the result is discarded.
    size_t max_count = 1;
    for (const auto& b : buckets_) max_count = std::max(max_count, b.end - b.begin);
    TensorOptions oc = TensorOptions().with_dtype(compute_dtype_).with_device(device_);
    scratch_recv_ = Tensor::zeros(flat_shape(max_count), oc);

    // 4. Partition invariants — the correctness-critical part. A bucket that
    //    straddled a shard boundary would have two owners and reduce into the
    //    wrong rank's shard, so verify at construction (cheap, once): buckets tile
    //    [0,padded) contiguously and each lies inside a single shard, the union of
    //    a rank's owned buckets is exactly its shard, and each param's buckets
    //    cover its whole flat range. Concatenating the per-bucket reduces then
    //    reproduces the monolithic reduce-scatter exactly.
    size_t cursor = 0;
    for (const auto& b : buckets_) {
        if (b.begin != cursor || b.end <= b.begin)
            throw ZeroConfigError("overlap bucket partition is not contiguous");
        if (static_cast<size_t>(b.owner) != b.begin / S_ ||
            b.end > (static_cast<size_t>(b.owner) + 1) * S_)
            throw ZeroConfigError("overlap bucket straddles a shard boundary");
        if (b.local_recv_offset + (b.end - b.begin) > S_)
            throw ZeroConfigError("overlap bucket recv slice exceeds shard");
        cursor = b.end;
    }
    if (cursor != padded_numel_)
        throw ZeroConfigError("overlap buckets do not tile the padded buffer");
    for (size_t pidx = 0; pidx < records_.size(); ++pidx) {
        size_t covered = records_[pidx].offset;
        for (uint32_t bidx : param_to_buckets_[pidx]) {
            const CommBucket& b = buckets_[bidx];
            const size_t a = std::max(b.begin, records_[pidx].offset);
            const size_t e = std::min(b.end, records_[pidx].offset + records_[pidx].numel);
            if (a != covered || e <= a)
                throw ZeroConfigError("overlap param->bucket map has a gap/overlap");
            covered = e;
        }
        if (covered != records_[pidx].offset + records_[pidx].numel)
            throw ZeroConfigError("overlap param->bucket map does not cover the param");
    }

    if (cfg_.verbose && rank_ == 0) {
        std::cout << "[ZeRO-1] overlap: " << buckets_.size() << " buckets, target "
                  << (cfg_.comm_bucket_bytes / (1024.0 * 1024.0)) << " MB/bucket"
                  << std::endl;
    }
}

void ShardedOptimizer::register_overlap_hooks() {
    for (size_t pidx = 0; pidx < records_.size(); ++pidx) {
        Tensor p = records_[pidx].param;   // leaf; requires_grad == true
        p.register_post_acc_hook(OwnTensor::make_post_acc_hook(
            [this, pidx](const Tensor& g) { this->on_grad_ready(pidx, g); }));
    }
}

void ShardedOptimizer::arm_grad_sync() {
    // Refuse to start a gradient-staging backward while a parameter refresh is
    // still outstanding. flat_stage_ is BOTH the param-out and the grad-in buffer:
    // the hooks below are about to overwrite it with gradients, which would both
    // discard the un-consumed parameters and later scatter gradient bytes into the
    // live weights. This fires loudly instead of silently diverging the ranks.
    if (param_refresh_pending_) {
        throw ZeroConfigError("parameter refresh stages were not consumed before "
                              "backward — call refresh_param_stage()/drain_param_refresh() "
                              "for every stage during the forward pass");
    }
    if (!overlap_active_) return;   // no-op when overlap disabled / W==1
    const size_t esz = elem_size(compute_dtype_);
    for (auto& b : buckets_) { b.pending = b.pending_init; b.launched = false; }
    buckets_launched_ = 0;
    grad_fired_.assign(records_.size(), false);
    // Zero the padded tail once per pass so pure-padding regions reduce to zero
    // (mirrors gather_grads_into_flat's tail handling). Enqueued on the compute
    // stream before any grad copy, so every later reduce sees it.
    if (padded_numel_ > total_numel_) {
        zero_async(static_cast<char*>(flat_stage_.data()) + total_numel_ * esz,
                   (padded_numel_ - total_numel_) * esz);
    }
    armed_ = true;
}

void ShardedOptimizer::on_grad_ready(size_t pidx, const Tensor& grad) {
    if (!armed_) return;   // inert on earlier micro-steps / unarmed backwards
    const size_t esz = elem_size(compute_dtype_);
    const auto& rec = records_[pidx];

    // Copy the fully-accumulated gradient into its flat slice on the compute
    // stream (identical op to gather_grads_into_flat's per-param copy).
    void* dst = static_cast<char*>(flat_stage_.data()) + rec.offset * esz;
    Tensor g = grad;
    if (!g.is_contiguous())          g = g.contiguous();
    if (g.dtype() != compute_dtype_) {
        // #2 family, site B' (W>1 overlap path).
        // What: convert the fp32 main-grad straight into its flat_stage_ slice.
        // Was:  g = g.as_type(compute_dtype_)  (a fresh temporary per param)  + d2d copy.
        // Why:  the temporary and the second copy did no useful work -- the same pattern
        //       is already fixed in cast_master_range (this file) and in
        //       SymmetricShardedOptimizer::stage_grad_into_flat. Same stream; fp32->bf16 is
        //       round-to-nearest-even on every kernel path, so the bits are unchanged
        //       (test_fix_family.cu: 0 mismatches, incl. misaligned slices and NaN/Inf).
        OwnTensor::convert_type_cuda_generic(g.data(), g.dtype(), dst, compute_dtype_,
                                             static_cast<int64_t>(rec.numel), ocuda::getCurrentStream());
    } else {
        d2d(dst, g.data(), rec.numel * esz, device_);
    }
    grad_fired_[pidx] = true;

    // Release each overlapping bucket; fire the ones that just filled.
    for (uint32_t bidx : param_to_buckets_[pidx]) {
        CommBucket& b = buckets_[bidx];
        if (!b.launched && b.pending > 0 && --b.pending == 0) launch_bucket(bidx);
    }
}

void ShardedOptimizer::launch_bucket(size_t bidx) {
    CommBucket& b = buckets_[bidx];
    const size_t esz = elem_size(compute_dtype_);

    // compute -> comm edge: the comm stream waits for the grad copies that filled
    // this bucket (all enqueued on the compute stream before this record). One
    // reused event is safe: each launch records-then-waits in program order.
    cudaEventRecord(compute_sync_event_, ocuda::getCurrentStream());
    cudaStreamWaitEvent(pg_->get_stream(), compute_sync_event_, 0);

    const void* send = static_cast<const char*>(flat_stage_.data()) + b.begin * esz;
    void* recv = (rank_ == b.owner)
        ? static_cast<char*>(shard_grad_.data()) + b.local_recv_offset * esz
        : scratch_recv_.data();
    // reduce-to-owner (avg): owner keeps the averaged bucket in its grad shard,
    // exactly the slice reduce_scatter would have delivered. Work recycled across
    // steps (event reuse) to keep the backward hook path free of event churn.
    b.work = pg_->reduce_async(send, recv, b.end - b.begin, reduce_dtype_,
                               /*op=*/avg, /*root=*/b.owner, /*sync=*/false, b.work);
    b.launched = true;
    ++buckets_launched_;
}

void ShardedOptimizer::finalize_overlap_reduce() {
    // Runs at step() entry, after backward has fully returned. Handle any bucket
    // not launched during backward: blocked by a parameter whose hook never fired
    // (unused in this step's graph — grad_fired_ is authoritative, NOT has_grad(),
    // which stays true from a previous step because zero_grad only zeroes the
    // existing buffer), or a pure-padding bucket with no owning parameter at all.
    const size_t esz = elem_size(compute_dtype_);
    for (size_t pidx = 0; pidx < records_.size(); ++pidx) {
        if (grad_fired_[pidx]) continue;   // hook ran this pass: already staged
        void* dst = static_cast<char*>(flat_stage_.data()) + records_[pidx].offset * esz;
        bool feeds_unlaunched = false;
        for (uint32_t bidx : param_to_buckets_[pidx])
            if (!buckets_[bidx].launched) { feeds_unlaunched = true; break; }
        if (!feeds_unlaunched) continue;
        zero_async(dst, records_[pidx].numel * esz);     // unused grad averages to 0
        for (uint32_t bidx : param_to_buckets_[pidx]) {
            CommBucket& b = buckets_[bidx];
            if (!b.launched && b.pending > 0 && --b.pending == 0) launch_bucket(bidx);
        }
    }
    for (size_t bidx = 0; bidx < buckets_.size(); ++bidx) {
        if (!buckets_[bidx].launched && buckets_[bidx].pending == 0)
            launch_bucket(bidx);          // pure-padding bucket
    }

    // Every bucket must have been launched by now — the mop-up above zeroes and
    // releases everything the hooks missed. A survivor would mean this rank waits
    // a stale event from a previous step and then reduces last step's gradient,
    // which is silent divergence; fail loudly instead.
    for (const auto& b : buckets_)
        if (!b.launched)
            throw ZeroCollectiveError("overlap reduce: a gradient bucket was never "
                                      "launched (missing grad-ready signal)");

    // comm -> compute edge: the compute-stream cast below reads shard_grad_, which
    // the owned-bucket reduces write on the comm stream. Non-owned reduces stay
    // ordered before the later all-gather on the same comm stream, so they need
    // no compute-side wait here.
    //
    // Only OWNED buckets are waited, so only they are error-checked here — a
    // non-owned reduce writes scratch_recv_ and its result is discarded, so a
    // failure there cannot corrupt this rank's state. A transport failure severe
    // enough to matter also fails the owned reduces (same communicator), and the
    // ProcessGroupNCCL watchdog catches an outright hang independently.
    for (size_t bidx = 0; bidx < buckets_.size(); ++bidx) {
        if (buckets_[bidx].owner == rank_) {
            wait_or_throw(buckets_[bidx].work, "gradient reduce-to-owner", bidx);
        }
    }

    armed_ = false;
}

// ======================================================================
// Staged parameter refresh (ZeroConfig::overlap_param_refresh)
//
// The reference path ends each step with one blocking all-gather that
// materializes every updated parameter into flat_stage_, followed by a copy of
// the whole model back into the live tensors. Both are pure critical path: by
// then backward is finished and Adam is done, so nothing overlaps them.
//
// Here the same data movement is split into consumption-ordered STAGES and
// issued asynchronously at the end of step(). The next forward consumes stage s
// (waits + copies into the model tensors) immediately before the first layer
// that reads it, so stage s+1..N-1 transfer while those layers compute. Stage 0
// is consumed by zero_grad(), before any forward op runs.
//
// Transport is one ncclBroadcast(root=owner) per (stage x owner-shard) piece,
// NOT ncclAllGather: an all-gather's output is rank-major, which only coincides
// with flat_stage_'s layout when the entire shard is gathered in a single call.
// A broadcast takes an arbitrary receive offset, so it can materialize any
// sub-range in place. Per-rank received volume is identical to the all-gather's;
// only the collective count changes.
//
// The broadcasts share the reduce path's comm stream deliberately. That stream
// is in-order, so a param broadcast cannot overwrite a flat_stage_ range that
// this step's gradient reduce is still reading (WAR), and it keeps scratch_recv_
// single-writer. Giving the param path its own stream would break both.
// ======================================================================

void ShardedOptimizer::set_param_refresh_stages(const std::vector<int>& stage_of,
                                                int num_stages) {
    if (!cfg_.overlap_param_refresh || W_ <= 1) return;   // stays on the sync path
    if (num_stages <= 0) throw ZeroConfigError("num_stages must be >= 1");
    if (param_refresh_pending_)
        throw ZeroConfigError("set_param_refresh_stages() called with a refresh in flight");

    num_refresh_stages_ = num_stages;
    stage_records_.assign(num_stages, {});
    stage_pieces_.assign(num_stages, {});
    refresh_pieces_.clear();

    // 1. Bin each record into its stage (indexing by the ctor-input position, so
    //    the caller never has to know the optimizer's internal flat ordering).
    for (size_t ri = 0; ri < records_.size(); ++ri) {
        const size_t ii = records_[ri].input_index;
        if (ii >= stage_of.size())
            throw ZeroConfigError("stage_of is shorter than the constructor's parameter vector");
        const int s = stage_of[ii];
        if (s < 0 || s >= num_stages)
            throw ZeroConfigError("stage_of contains an out-of-range stage index");
        stage_records_[s].push_back(static_cast<uint32_t>(ri));
    }

    // 2. Per stage: merge its records' flat ranges, then split each merged run at
    //    every shard boundary so a piece has exactly one owner. Only real
    //    parameter bytes are broadcast — the [total_numel_, padded_numel_) tail is
    //    never read back into a parameter, so it is excluded entirely and keeps
    //    arm_grad_sync's memset as its single writer.
    for (int s = 0; s < num_stages; ++s) {
        std::vector<std::pair<size_t, size_t>> ranges;
        ranges.reserve(stage_records_[s].size());
        for (uint32_t ri : stage_records_[s])
            ranges.emplace_back(records_[ri].offset, records_[ri].offset + records_[ri].numel);
        std::sort(ranges.begin(), ranges.end());

        std::vector<std::pair<size_t, size_t>> merged;
        for (const auto& r : ranges) {
            if (!merged.empty() && r.first <= merged.back().second) {
                merged.back().second = std::max(merged.back().second, r.second);
            } else {
                merged.push_back(r);
            }
        }

        for (const auto& r : merged) {
            size_t pos = r.first;
            while (pos < r.second) {
                const size_t owner     = pos / S_;
                const size_t shard_end = (owner + 1) * S_;
                const size_t end       = std::min(r.second, shard_end);
                RefreshPiece p;
                p.begin             = pos;
                p.end               = end;
                p.owner             = static_cast<int>(owner);
                p.local_send_offset = pos - owner * S_;
                stage_pieces_[s].push_back(static_cast<uint32_t>(refresh_pieces_.size()));
                refresh_pieces_.push_back(p);
                pos = end;
            }
        }
    }

    // 3. Invariants: every piece inside one shard and inside the real parameter
    //    range, and the stages together covering every parameter byte exactly once
    //    (so no parameter is left holding a stale value after a full drain).
    std::vector<std::pair<size_t, size_t>> all;
    for (const auto& p : refresh_pieces_) {
        if (p.end <= p.begin || p.end > total_numel_)
            throw ZeroConfigError("refresh piece is empty or runs past the parameter range");
        if (static_cast<size_t>(p.owner) != p.begin / S_ ||
            p.end > (static_cast<size_t>(p.owner) + 1) * S_)
            throw ZeroConfigError("refresh piece straddles a shard boundary");
        if (p.local_send_offset + (p.end - p.begin) > S_)
            throw ZeroConfigError("refresh piece send slice exceeds the shard");
        all.emplace_back(p.begin, p.end);
    }
    std::sort(all.begin(), all.end());
    size_t covered = 0;
    for (const auto& r : all) {
        if (r.first < covered) throw ZeroConfigError("refresh pieces overlap");
        covered += r.second - r.first;
    }
    size_t param_bytes = 0;
    for (const auto& rec : records_) param_bytes += rec.numel;
    if (covered != param_bytes)
        throw ZeroConfigError("refresh stages do not cover every parameter exactly once");

    overlap_param_active_ = true;
    if (!param_sync_event_)
        cudaEventCreateWithFlags(&param_sync_event_, cudaEventDisableTiming);

    if (cfg_.verbose && rank_ == 0) {
        std::cout << "[ZeRO-1] staged param refresh: " << num_stages << " stages, "
                  << refresh_pieces_.size() << " broadcasts/step" << std::endl;
    }
}

void ShardedOptimizer::adam_flat_range(size_t begin, size_t end, float bc1, float bc2) {
    const size_t lo = std::max(begin, shard_valid_begin());
    const size_t hi = std::min(end,   shard_valid_end());
    if (hi <= lo) return;                      // nothing of this run is ours

    for (const auto& g : groups_) {
        const size_t a = std::max(g.begin, lo);
        const size_t b = std::min(g.end,   hi);
        if (b <= a) continue;
        const size_t local = a - shard_begin_;
        const int64_t len  = static_cast<int64_t>(b - a);

        std::vector<ocuda::TensorInfo> p = {{shard_master_.data<float>()    + local, len}};
        std::vector<ocuda::TensorInfo> gr= {{shard_grad_fp32_.data<float>() + local, len}};
        std::vector<ocuda::TensorInfo> m = {{shard_m_.data<float>()         + local, len}};
        std::vector<ocuda::TensorInfo> v = {{shard_v_.data<float>()         + local, len}};

        ocuda::multi_tensor_adam_cuda(
            p, gr, m, v,
            cfg_.lr * g.lr_scale, cfg_.beta1, cfg_.beta2, cfg_.eps,
            g.weight_decay, bc1, bc2, is_adamw_);
        // Shared static pinned launch-metadata buffer — see local_adam_step().
        // This only drains the COMPUTE stream; broadcasts already in flight on
        // the comm stream keep running, which is what makes the staged pipeline
        // work despite the sync.
        sync_compute();
    }
}

void ShardedOptimizer::cast_master_range(size_t begin, size_t end) {
    // Clamped to the whole shard (not the VALID range): the padded tail carries
    // no parameter, but no refresh piece covers it either, so this is a no-op there.
    const size_t lo = std::max(begin, shard_begin_);
    const size_t hi = std::min(end,   shard_begin_ + S_);
    if (hi <= lo) return;

    const size_t esz   = elem_size(compute_dtype_);
    const size_t local = lo - shard_begin_;
    const int64_t n    = static_cast<int64_t>(hi - lo);

    if (compute_dtype_ == Dtype::Float32) {
        d2d(static_cast<char*>(shard_master_compute_.data()) + local * esz,
            shard_master_.data<float>() + local, n * sizeof(float), device_);
        return;
    }
    // Direct sub-range convert into the send buffer. Replaces the old
    // `shard_master_.as_type()` + full-shard d2d, which allocated a whole extra
    // S_-element temporary and copied it every step (114.2 MB / 0.36 ms in report5)
    // and could not express a partial range at all.
    OwnTensor::convert_type_cuda_generic(
        shard_master_.data<float>() + local, Dtype::Float32,
        static_cast<char*>(shard_master_compute_.data()) + local * esz,
        compute_dtype_, n, ocuda::getCurrentStream());
}

void ShardedOptimizer::issue_refresh_stage(int s) {
    const size_t esz = elem_size(compute_dtype_);
    for (uint32_t pi : stage_pieces_[s]) {
        RefreshPiece& p = refresh_pieces_[pi];
        const void* send = static_cast<const char*>(shard_master_compute_.data()) +
                           p.local_send_offset * esz;
        void* recv = static_cast<char*>(flat_stage_.data()) + p.begin * esz;
        p.work = pg_->broadcast_async(send, recv, p.end - p.begin, compute_dtype_,
                                      /*root=*/p.owner, /*sync=*/false, p.work);
    }
}

void ShardedOptimizer::adam_and_publish_staged() {
    const float bc1 = 1.0f - std::pow(cfg_.beta1, static_cast<float>(step_count_));
    const float bc2 = 1.0f - std::pow(cfg_.beta2, static_cast<float>(step_count_));

    // Coalesce a phase's piece ranges into maximal runs before launching. The
    // pieces of stages 1..N-1 are largely contiguous in flat order, so this keeps
    // the fused-Adam launch count (and its per-launch sync) at ~2-3 per phase
    // instead of one per piece — the launch latency would otherwise eat the win.
    auto merged = [](std::vector<std::pair<size_t,size_t>> r) {
        std::sort(r.begin(), r.end());
        std::vector<std::pair<size_t,size_t>> out;
        for (const auto& x : r) {
            if (!out.empty() && x.first <= out.back().second)
                out.back().second = std::max(out.back().second, x.second);
            else out.push_back(x);
        }
        return out;
    };
    auto ranges_of = [&](int lo_stage, int hi_stage) {
        std::vector<std::pair<size_t,size_t>> r;
        for (int s = lo_stage; s < hi_stage; ++s)
            for (uint32_t pi : stage_pieces_[s])
                r.emplace_back(refresh_pieces_[pi].begin, refresh_pieces_[pi].end);
        return merged(std::move(r));
    };

    // ---- phase A: stage 0 only, then publish it immediately ----
    for (const auto& r : ranges_of(0, 1)) {
        adam_flat_range(r.first, r.second, bc1, bc2);
        cast_master_range(r.first, r.second);
    }
    // compute -> comm edge for phase A. Recording the shared event twice per step
    // is safe: phase A's cudaStreamWaitEvent captured this record before phase B
    // re-records it (program order), exactly as Work::rearm relies on.
    cudaEventRecord(param_sync_event_, ocuda::getCurrentStream());
    cudaStreamWaitEvent(pg_->get_stream(), param_sync_event_, 0);
    issue_refresh_stage(0);
    // Stage 0 is now flying on the comm stream. Everything below runs underneath it.

    // ---- phase B: every remaining stage ----
    for (const auto& r : ranges_of(1, num_refresh_stages_)) {
        adam_flat_range(r.first, r.second, bc1, bc2);
        cast_master_range(r.first, r.second);
    }
    cudaEventRecord(param_sync_event_, ocuda::getCurrentStream());
    cudaStreamWaitEvent(pg_->get_stream(), param_sync_event_, 0);
    // In stage order on the one in-order comm stream, so a stage's completion
    // event still implies every earlier stage has landed.
    for (int s = 1; s < num_refresh_stages_; ++s) issue_refresh_stage(s);

    param_refresh_pending_ = true;
    next_refresh_stage_    = 0;
}

void ShardedOptimizer::refresh_param_stage(int stage) {
    if (!param_refresh_pending_) return;
    const size_t esz = elem_size(compute_dtype_);

    for (int s = next_refresh_stage_; s <= stage && s < num_refresh_stages_; ++s) {
        // Overlap diagnostic. "Resident on arrival" means the wait cost nothing,
        // which has TWO very different causes and this check cannot tell them
        // apart — only the profiler can:
        //   (a) the transfer finished inside the GPU-idle, host-bound gap between
        //       steps (logging, data loading). Then it cost no wall time at all,
        //       which is the best case, even though it overlapped no compute.
        //   (b) something drained the comm stream between issue and consumption
        //       (a cudaDeviceSynchronize, or a blocking collective queued behind
        //       these broadcasts on the same in-order stream). Then the transfer
        //       was fully exposed and the feature bought nothing.
        // To tell them apart, check in nsys whether a compute kernel was running
        // while the Broadcast kernels executed.
        if (cfg_.verbose && rank_ == 0 && s > 0 && step_count_ <= 3) {
            bool all_done = true;
            for (uint32_t pi : stage_pieces_[s])
                if (refresh_pieces_[pi].work && !refresh_pieces_[pi].work->query()) {
                    all_done = false; break;
                }
            // std::cout << "[ZeRO-1] param refresh stage " << s
            //           << (all_done ? ": resident on arrival, wait was free (transfer fit "
            //                          "the inter-step gap, or the comm stream was drained "
            //                          "— confirm which in the profile)"
            //                        : ": still in flight, overlapping compute as intended")
            //           << std::endl;
        }

        // comm -> compute edge for this stage's pieces. Work::wait() enqueues a
        // cudaStreamWaitEvent on the compute stream — no host stall — so the
        // copies below cannot start before the data has landed. A failed broadcast
        // would leave flat_stage_ holding the previous step's weights for that
        // piece, which then get copied into the live model tensors below, so the
        // result is checked rather than discarded.
        for (uint32_t pi : stage_pieces_[s])
            wait_or_throw(refresh_pieces_[pi].work, "parameter broadcast", pi);

        for (uint32_t ri : stage_records_[s]) {
            const auto& rec = records_[ri];
            const void* src = static_cast<const char*>(flat_stage_.data()) + rec.offset * esz;
            Tensor p = rec.param;   // non-const handle (shares storage) for mutable data()
            d2d(p.data(), src, rec.numel * esz, device_);
        }
        next_refresh_stage_ = s + 1;
    }

    if (next_refresh_stage_ >= num_refresh_stages_) param_refresh_pending_ = false;
}

void ShardedOptimizer::drain_param_refresh() {
    if (!param_refresh_pending_) return;
    refresh_param_stage(num_refresh_stages_ - 1);
}

// ======================================================================
// Step
// ======================================================================

size_t ShardedOptimizer::shard_valid_end() const {
    size_t end = std::min(shard_begin_ + S_, total_numel_);
    return end < shard_begin_ ? shard_begin_ : end;
}

void ShardedOptimizer::gather_grads_into_flat() {
    const size_t esz = elem_size(compute_dtype_);
    for (const auto& rec : records_) {
        void* dst = static_cast<char*>(flat_stage_.data()) + rec.offset * esz;
        if (rec.param.has_grad()) {
            Tensor g = rec.param.grad_view();
            if (!g.is_contiguous()) g = g.contiguous();
            if (g.dtype() != compute_dtype_) {
                // #2 family, site B (W=1 / reference path).
                // What: convert the fp32 main-grad straight into its flat_stage_ slice.
                // Was:  g = g.as_type(compute_dtype_) (temporary per param) + d2d copy.
                // Why:  same as on_grad_ready above -- one pass instead of two, bit-identical
                //       (test_fix_family.cu: 0 mismatches incl. misaligned slices).
                OwnTensor::convert_type_cuda_generic(g.data(), g.dtype(), dst, compute_dtype_,
                                                     static_cast<int64_t>(rec.numel), ocuda::getCurrentStream());
            } else {
                d2d(dst, g.data(), rec.numel * esz, device_);
            }
        } else {
            // param requires grad but produced none this step (unused): its
            // averaged gradient is 0, matching DDP semantics.
            zero_async(dst, rec.numel * esz);
        }
    }
    if (padded_numel_ > total_numel_) {
        zero_async(static_cast<char*>(flat_stage_.data()) + total_numel_ * esz,
                   (padded_numel_ - total_numel_) * esz);
    }
}

void ShardedOptimizer::reduce_scatter_grads() {
    const size_t esz = elem_size(compute_dtype_);
    if (W_ == 1) {
        // padded == total == S; averaging over one rank is the identity.
        d2d(shard_grad_.data(), flat_stage_.data(), S_ * esz, device_);
        return;
    }
    sync_compute();  // ensure gather copies are visible before the collective
    auto r = pg_->reduce_scatter(flat_stage_.data(), shard_grad_.data(),
                                 S_, reduce_dtype_, /*op=*/avg, /*sync=*/true);
    if (r != pgSuccess) throw ZeroCollectiveError("reduce_scatter(grads) failed");
    cudaDeviceSynchronize();
}

void ShardedOptimizer::cast_shard_grad_to_fp32() {
    if (compute_dtype_ == Dtype::Float32) {
        d2d(shard_grad_fp32_.data(), shard_grad_.data(), S_ * sizeof(float), device_);
    } else {
        // #2 family, site C.
        // What: convert shard_grad_ (bf16) straight into shard_grad_fp32_, which is
        //       already allocated in allocate_buffers().
        // Was:  Tensor f = shard_grad_.as_type(Float32); d2d(shard_grad_fp32_, f, ...)
        //       -- a whole extra S_-element fp32 temporary (5,888,802,816 B at W=1) plus a
        //       full copy, every optimizer step.
        // Why:  as_type launches exactly this kernel (AsTypeTensor.cpp:38-39), on the same
        //       stream as d2d and as every later reader of shard_grad_fp32_ (unscale, norm,
        //       clip, Adam), so writing directly is bit-identical and keeps the ordering.
        //       Proven: test_grad_cast_direct.cu 0/1,472,200,704 mismatches; disassembly shows
        //       as_type + copy_memory gone. Same fix cast_master_range already carries.
        OwnTensor::convert_type_cuda_generic(
            shard_grad_.data(), shard_grad_.dtype(),
            shard_grad_fp32_.data(), Dtype::Float32,
            static_cast<int64_t>(S_), ocuda::getCurrentStream());
    }
}

// F4: at W == 1 nothing is sent, so the gradient does not go through the wire buffer.
// What: copy each parameter's gradient straight into shard_grad_fp32_, the buffer the unscale,
//       the norm, the clip and Adam read. Autograd keeps the gradient of a bf16/fp16 parameter
//       in fp32, so this is an fp32 -> fp32 copy (a lower-precision gradient is widened, which is
//       exact). At W == 1 the shard is the whole flat space (shard_begin_ == 0, S_ ==
//       padded_numel_), so a parameter's flat offset is also its shard offset.
// Was:  gather_grads_into_flat() (fp32 -> compute dtype into flat_stage_), a plain W == 1 copy
//       into shard_grad_ (reduce_scatter_grads), then cast_shard_grad_to_fp32() (back to fp32).
//       On a bf16 group that rounded the whole gradient to bf16 once per step although nothing
//       was sent.
// Why:  the optimizer sees the exact accumulated gradient, and in one pass instead of three.
void ShardedOptimizer::stage_grads_direct_fp32() {
    float* base = shard_grad_fp32_.data<float>();
    for (const auto& rec : records_) {
        float* dst = base + rec.offset;
        if (rec.param.has_grad()) {
            Tensor g = rec.param.grad_view();
            if (!g.is_contiguous()) g = g.contiguous();
            if (g.dtype() == Dtype::Float32) {
                d2d(dst, g.data(), rec.numel * sizeof(float), device_);
            } else {
                OwnTensor::convert_type_cuda_generic(g.data(), g.dtype(), dst, Dtype::Float32,
                                                     static_cast<int64_t>(rec.numel),
                                                     ocuda::getCurrentStream());
            }
        } else {
            // param requires grad but produced none this step (unused): gradient 0,
            // exactly as gather_grads_into_flat() treats it.
            zero_async(dst, rec.numel * sizeof(float));
        }
    }
    if (padded_numel_ > total_numel_) {
        zero_async(base + total_numel_, (padded_numel_ - total_numel_) * sizeof(float));
    }
}

void ShardedOptimizer::local_adam_step() {
    const float bc1 = 1.0f - std::pow(cfg_.beta1, static_cast<float>(step_count_));
    const float bc2 = 1.0f - std::pow(cfg_.beta2, static_cast<float>(step_count_));

    const size_t vbeg = shard_valid_begin();
    const size_t vend = shard_valid_end();
    if (vend <= vbeg) return;   // this rank owns only padding

    for (const auto& g : groups_) {
        size_t a = std::max(g.begin, vbeg);
        size_t b = std::min(g.end, vend);
        if (b <= a) continue;
        size_t local = a - shard_begin_;
        int64_t len  = static_cast<int64_t>(b - a);

        std::vector<ocuda::TensorInfo> p = {{shard_master_.data<float>()   + local, len}};
        std::vector<ocuda::TensorInfo> gr= {{shard_grad_fp32_.data<float>()+ local, len}};
        std::vector<ocuda::TensorInfo> m = {{shard_m_.data<float>()        + local, len}};
        std::vector<ocuda::TensorInfo> v = {{shard_v_.data<float>()        + local, len}};

        ocuda::multi_tensor_adam_cuda(
            p, gr, m, v,
            cfg_.lr * g.lr_scale, cfg_.beta1, cfg_.beta2, cfg_.eps,
            g.weight_decay, bc1, bc2, is_adamw_);
        // The multi-tensor kernels stage their (ptr,numel) launch metadata
        // through a shared static pinned buffer that is consumed
        // asynchronously; a following multi-tensor launch can overwrite it
        // before this one reads it. Serialize each launch so the metadata is
        // never clobbered mid-flight. (Correctness in the reference schedule;
        // the overlap schedule will batch these instead.)
        sync_compute();
    }
}

void ShardedOptimizer::all_gather_params() {
    const size_t esz = elem_size(compute_dtype_);
    // master (FP32) -> compute-dtype staging
    if (compute_dtype_ == Dtype::Float32) {
        d2d(shard_master_compute_.data(), shard_master_.data(), S_ * sizeof(float), device_);
    } else {
        // #2 family, site D.
        // What: convert the fp32 master straight into shard_master_compute_.
        // Was:  Tensor c = shard_master_.as_type(compute_dtype_); d2d(...) -- an S_-element
        //       bf16 temporary (2.94 GB at W=1) plus a full copy, every optimizer step.
        // Why:  identical to cast_master_range(shard_begin_, shard_begin_ + S_) (same call,
        //       same stream), so bit-identical (test_fix_family.cu site D: 0 mismatches).
        OwnTensor::convert_type_cuda_generic(shard_master_.data(), Dtype::Float32,
                                             shard_master_compute_.data(), compute_dtype_,
                                             static_cast<int64_t>(S_), ocuda::getCurrentStream());
    }
    if (W_ == 1) {
        d2d(flat_stage_.data(), shard_master_compute_.data(), S_ * esz, device_);
        return;
    }
    sync_compute();
    auto r = pg_->all_gather(shard_master_compute_.data(), flat_stage_.data(),
                             S_, compute_dtype_, /*sync=*/true);
    if (r != pgSuccess) throw ZeroCollectiveError("all_gather(params) failed");
    cudaDeviceSynchronize();
}

void ShardedOptimizer::scatter_flat_into_params() {
    const size_t esz = elem_size(compute_dtype_);
    for (const auto& rec : records_) {
        const void* src = static_cast<const char*>(flat_stage_.data()) + rec.offset * esz;
        Tensor p = rec.param;  // non-const handle (shares storage) for mutable data()
        d2d(p.data(), src, rec.numel * esz, device_);
    }
    sync_compute();
}

// Gradient reduce + FP32 cast + loss-scaler unscale. Split out of step() so
// reduce_and_measure() can run it during the cross-group norm phase.
void ShardedOptimizer::reduce_grads_now() {
    if (overlap_active_ && armed_) {
        // Gradients were bucketed and reduced-to-owner DURING backward. Finish any
        // stragglers and insert the comm->compute dependency; shard_grad_ then
        // holds this rank's averaged gradient shard exactly as the monolithic
        // gather + reduce_scatter would have produced it.
        finalize_overlap_reduce();
    } else if (W_ == 1) {
        // F4: at W == 1 nothing is sent, so the gradients skip the wire buffer and go
        // straight into shard_grad_fp32_ (see stage_grads_direct_fp32). Nothing to cast.
        stage_grads_direct_fp32();
    } else {
        // Reference path: either overlap is off, or the caller did not arm this
        // pass — the (inert) hooks left grads untouched, so gather them normally.
        gather_grads_into_flat();
        reduce_scatter_grads();
    }
    if (W_ > 1) cast_shard_grad_to_fp32();   // W == 1 wrote fp32 directly (F4)

    // Loss-scaler unscale (grad *= 1/scale) belongs here, not after: the norm this
    // reduction feeds must be the norm of the TRUE gradient.
    if (OwnTensor::nn::LossScaler* sc = get_scaler()) {
        const size_t valid = shard_valid_end() - shard_valid_begin();
        float inv = 1.0f / sc->scale();
        *h_norm_pinned_ = inv;
        cudaMemcpy(d_clip_coef_, h_norm_pinned_, sizeof(float), cudaMemcpyHostToDevice);
        if (valid > 0) {
            std::vector<ocuda::TensorInfo> gi = {{shard_grad_fp32_.data<float>(), (int64_t)valid}};
            ocuda::multi_tensor_scale_cuda(gi, d_clip_coef_);
            sync_compute();
        }
    }
    grads_reduced_ = true;
}

// Phase 1 of cross-group global clipping — see ZeroOptimizerBase.
float ShardedOptimizer::reduce_and_measure() {
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
        return 0.0f;   // still accumulating
    }
    if (!grads_reduced_) reduce_grads_now();

    const size_t valid = shard_valid_end() - shard_valid_begin();
    cudaMemsetAsync(d_norm_sq_, 0, sizeof(float), ocuda::getCurrentStream());
    if (valid > 0) {
        std::vector<ocuda::TensorInfo> gi = {{shard_grad_fp32_.data<float>(), (int64_t)valid}};
        ocuda::multi_tensor_grad_norm_cuda(gi, d_norm_sq_);
        sync_compute();
    }
    if (W_ > 1) {   // COLLECTIVE: every rank participates, valid or not
        sync_compute();
        pg_->all_reduce(d_norm_sq_, d_norm_sq_, 1, Dtype::Float32, sum, true);
        cudaStreamSynchronize(ocuda::getCurrentStream());
    }
    cudaMemcpy(h_norm_pinned_, d_norm_sq_, sizeof(float), cudaMemcpyDeviceToHost);
    const float nsq = *h_norm_pinned_;
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
    last_grad_norm_ = (nsq > 0.0f) ? std::sqrt(nsq) : 0.0f;
    return nsq;
}

// F1. Text for ZeroNonFiniteGradError: which group, how big, which rank and
// step, so the log names the failing group instead of printing a norm of 0.
std::string ShardedOptimizer::nonfinite_grad_msg(float nsq) const {
    const char* g = compute_dtype_ == Dtype::Bfloat16 ? "bf16"
                  : compute_dtype_ == Dtype::Float32  ? "fp32" : "other";
    return "grad norm^2 = " + std::to_string(nsq) + " in the " + g + " group (" +
           std::to_string(total_numel_) + " params), rank " + std::to_string(rank_) +
           ", optimizer step " + std::to_string(step_count_ + 1) +
           "; no LossScaler installed, so the step cannot be skipped -- raised BEFORE any update";
}

void ShardedOptimizer::step() {
    last_step_skipped_ = false;

    // Every path below either writes flat_stage_ (gather_grads_into_flat on the
    // reference path) or re-arms the refresh Works (issue_param_refresh), so an
    // unconsumed refresh here is never recoverable: the parameters would be
    // overwritten by gradients and the stranded broadcasts silently forgotten.
    // arm_grad_sync() catches this earlier on the overlapped-backward path, but
    // the reference path has no such checkpoint — hence the guard here too.
    if (param_refresh_pending_) {
        throw ZeroConfigError("parameter refresh stages were not consumed before "
                              "step() — every stage must be installed during the "
                              "forward pass (refresh_param_stage) or explicitly "
                              "drained (drain_param_refresh)");
    }

    // --- gradient-accumulation gate: only reduce + step on the last micro-step
    if (cfg_.grad_accum_steps > 1) {
        ++micro_step_;
        if (micro_step_ < cfg_.grad_accum_steps) return;  // keep accumulating grads
        micro_step_ = 0;
    }

    if (!grads_reduced_) reduce_grads_now();
    grads_reduced_ = false;                       // one reduction per step

    OwnTensor::nn::LossScaler* sc = get_scaler();
    const size_t valid = shard_valid_end() - shard_valid_begin();

    const bool need_norm = (sc != nullptr) || (cfg_.max_grad_norm > 0.0f);
    if (need_norm && cached_nsq_ >= 0.0f) {
        // reduce_and_measure() already produced the norm this step (cross-group
        // clipping); reuse it instead of paying a second all-reduce.
        const float nsq = cached_nsq_;
        cached_nsq_ = -1.0f;
        // F1 (cached-norm path, BLU_GLOBAL_CLIP=1). Same change as the guard in
        // the per-group path below; see the full explanation there.
        //   was: last_grad_norm_ = (nsq > 0) ? sqrt(nsq) : 0   -> a NaN norm was LOGGED AS 0
        //   was: if (sc && !isfinite(nsq)) skip               -> no scaler installed = never fires
        last_grad_norm_ = (nsq > 0.0f || std::isnan(nsq)) ? std::sqrt(nsq) : 0.0f;   // NaN logs as nan, not 0
        if (!std::isfinite(nsq)) {
            if (!sc) throw ZeroNonFiniteGradError(nonfinite_grad_msg(nsq));
            sc->update(true);
            last_step_skipped_  = true;
            external_clip_coef_ = -1.0f;
            return;
        }
        if (cfg_.max_grad_norm > 0.0f && valid > 0) {
            float coef = external_clip_coef_ >= 0.0f
                       ? external_clip_coef_
                       : cfg_.max_grad_norm / (last_grad_norm_ + 1e-6f);
            if (coef > 1.0f) coef = 1.0f;
            *h_norm_pinned_ = coef;
            cudaMemcpy(d_clip_coef_, h_norm_pinned_, sizeof(float), cudaMemcpyHostToDevice);
            std::vector<ocuda::TensorInfo> gi = {{shard_grad_fp32_.data<float>(), (int64_t)valid}};
            ocuda::multi_tensor_scale_cuda(gi, d_clip_coef_);
            sync_compute();
        }
    } else if (need_norm) {
        cudaMemsetAsync(d_norm_sq_, 0, sizeof(float), ocuda::getCurrentStream());
        if (valid > 0) {
            std::vector<ocuda::TensorInfo> gi = {{shard_grad_fp32_.data<float>(), (int64_t)valid}};
            ocuda::multi_tensor_grad_norm_cuda(gi, d_norm_sq_);
            sync_compute();
        }
        if (W_ > 1) {
            sync_compute();
            pg_->all_reduce(d_norm_sq_, d_norm_sq_, 1, Dtype::Float32, sum, true);
            // Sync the COMPUTE stream, not the whole device. all_reduce(sync=true)
            // already enqueued a cudaStreamWaitEvent on the compute stream, so
            // draining it transitively waits for the collective — while leaving
            // unrelated comm-stream work alone. A cudaDeviceSynchronize() here
            // would drain another dtype group's in-flight staged parameter
            // broadcasts (see issue_param_refresh) and silently destroy the
            // overlap they were issued for.
            cudaStreamSynchronize(ocuda::getCurrentStream());
        }
        cudaMemcpy(h_norm_pinned_, d_norm_sq_, sizeof(float), cudaMemcpyDeviceToHost);
        float nsq = *h_norm_pinned_;
        last_grad_norm_ = (nsq > 0.0f || std::isnan(nsq)) ? std::sqrt(nsq) : 0.0f;   // NaN logs as nan, not 0

        // --- global non-finite guard: atomic step skip (params untouched) ---
        // F1 (the default per-group-clip path).
        // Was:  `if (sc && !std::isfinite(nsq)) { skip }`. The trainer never installs a
        //       LossScaler, so `sc` is null and this guard NEVER fired. A NaN/Inf gradient
        //       then went straight into Adam. Backtrace of what happened (TI 8808f89):
        //         1. nsq = NaN or Inf (multi_tensor_grad_norm keeps NaN/Inf, MultiTensorKernels.cu:136-184)
        //         2. last_grad_norm_ logged 0 for NaN (the old `(nsq > 0) ? .. : 0`), hiding it
        //         3. compute_clip_coef_cuda (GradNormKernels.cu:190-193): max/(NaN) = NaN, and
        //            `(NaN < 1) ? .. : 1.0f` gives coef = 1.0; for Inf, coef = 0
        //         4. multi_tensor_scale_cuda returns early at coef == 1.0 (MultiTensorKernels.cu:235-240),
        //            or multiplies Inf by 0 = NaN
        //         5. multi_tensor_adam_cuda (MultiTensorKernels.cu:321-334) writes NaN into master, m, v
        //         6. the next forward's loss is NaN -> the trainer aborts one step LATE; if this step
        //            was a checkpoint step, the NaN state was already saved and auto-resume reloads it.
        //       Reproduced on the H200 2026-09-25 (t12/T2a: NaN injected at step 2 -> "NaN/Inf loss
        //       at step 3", group norm printed 0.0000).
        // Why here: step() already computes this one number (the norm, needed for clipping) and
        //       all-reduces it, so checking it is free (one isfinite on a host float, once per group
        //       per step), it happens BEFORE any update, and every rank sees the same value and
        //       takes the same branch (no rank can hang in a later collective). A per-element check
        //       in the Adam kernel could not skip the step as a whole; the trainer's loss check runs
        //       before step() and a finite loss can still carry a NaN gradient.
        // Now:  with a LossScaler, unchanged (skip + back off). Without one, throw
        //       ZeroNonFiniteGradError before touching master/m/v; the trainer catches it and
        //       aborts cleanly, so no poisoned weights and no poisoned checkpoint.
        //       Verified: t12/T2b (NaN) and T2c (Inf) stop at step 2 with exit code 4.
        //       A finite norm takes exactly the old code path -> training is bit-identical.
        if (!std::isfinite(nsq)) {
            if (!sc) throw ZeroNonFiniteGradError(nonfinite_grad_msg(nsq));
            sc->update(true);
            last_step_skipped_  = true;
            external_clip_coef_ = -1.0f;
            return;   // master + model params unchanged; no all-gather issued
        }

        // --- global gradient clip (norm is exact across all shards) ---
        if (cfg_.max_grad_norm > 0.0f && valid > 0) {
            if (external_clip_coef_ >= 0.0f) {
                // Cross-group global clipping: the trainer combined every group's
                // norm and handed down one coefficient for all of them.
                float coef = std::min(1.0f, external_clip_coef_);
                *h_norm_pinned_ = coef;
                cudaMemcpy(d_clip_coef_, h_norm_pinned_, sizeof(float), cudaMemcpyHostToDevice);
            } else {
                ocuda::compute_clip_coef_cuda(d_norm_sq_, d_clip_coef_,
                                              cfg_.max_grad_norm, /*is_inf=*/false);
            }
            sync_compute();
            std::vector<ocuda::TensorInfo> gi = {{shard_grad_fp32_.data<float>(), (int64_t)valid}};
            ocuda::multi_tensor_scale_cuda(gi, d_clip_coef_);
            sync_compute();
        }
    }

    // Advance the Adam timestep only for a step that actually updates the
    // moments — a step skipped for gradient overflow (returned above) must NOT
    // consume a timestep, or the bias correction bc=1-beta^t drifts permanently.
    ++step_count_;
    if (overlap_param_active_) {
        // Adam, cast and broadcast run as a stage-ordered pipeline rather than
        // three whole-shard passes: stage 0 (the token embedding) is updated and
        // published FIRST so its transfer overlaps the Adam of everything else,
        // and the later stages' transfers overlap the next forward. Returns with
        // every broadcast still in flight — the next forward consumes the stages
        // (see refresh_param_stage); the model tensors still hold the PREVIOUS
        // step's weights until it does, which is why arm_grad_sync() refuses to
        // start a backward with a stage outstanding.
        adam_and_publish_staged();
    } else {
        local_adam_step();
        all_gather_params();
        scatter_flat_into_params();
    }

    if (sc) sc->update(false);
    external_clip_coef_ = -1.0f;      // valid for exactly one step
}

// ======================================================================
// zero_grad / checkpoint
// ======================================================================

void ShardedOptimizer::zero_grad() {
    // First optimizer call of the iteration, and it always precedes the forward
    // pass — the correct place to consume refresh stage 0 (the parameters the
    // forward reads before it reaches stage 1's consumption point).
    if (param_refresh_pending_) refresh_param_stage(0);

    for (auto& rec : records_) {
        if (rec.param.has_grad()) rec.param.zero_grad();
    }
}

void ShardedOptimizer::save_state(std::ostream& os) {
    ShardLayoutDescriptor desc;
    desc.world_size   = W_;
    desc.total_numel  = static_cast<int64_t>(total_numel_);
    desc.padded_numel = static_cast<int64_t>(padded_numel_);
    desc.shard_numel  = static_cast<int64_t>(S_);
    desc.param_count  = static_cast<int32_t>(records_.size());
    desc.num_groups   = static_cast<int32_t>(groups_.size());
    desc.layout_hash  = layout_hash_;
    // v1 is, by definition, whole-buffer contiguous ownership. Recording it is
    // what lets SymmetricShardedOptimizer (PER_BUCKET) detect a v1 checkpoint
    // and reshard instead of reinterpreting the bytes as its own layout.
    desc.ownership    = static_cast<int32_t>(ShardOwnership::CONTIGUOUS);
    desc.bucket_elems = 0;

    os.write(reinterpret_cast<const char*>(&desc), sizeof(desc));
    os.write(reinterpret_cast<const char*>(&step_count_), sizeof(int64_t));

    OwnTensor::save_tensor(shard_master_, os);
    OwnTensor::save_tensor(shard_m_, os);
    OwnTensor::save_tensor(shard_v_, os);
}

void ShardedOptimizer::load_state(std::istream& is) {
    // Any in-flight refresh targets flat_stage_ and the live params, both of which
    // this function overwrites from the checkpoint; retire it first.
    drain_param_refresh();

    // A version-1 record is a byte-exact prefix of a version-2 one, so read the
    // common prefix first and only pull the tail when the writer produced it.
    // `desc` keeps its defaults for the tail, which is exactly the version-1
    // meaning: CONTIGUOUS ownership, no bucketing.
    ShardLayoutDescriptor desc;
    is.read(reinterpret_cast<char*>(&desc), kShardLayoutDescriptorV1Bytes);
    if (desc.magic != ShardLayoutDescriptor{}.magic || desc.version < 1 || desc.version > 2) {
        throw ZeroCheckpointError("bad magic/version in ZeRO optimizer state");
    }
    if (desc.version >= 2) {
        is.read(reinterpret_cast<char*>(&desc) + kShardLayoutDescriptorV1Bytes,
                sizeof(desc) - kShardLayoutDescriptorV1Bytes);
    } else {
        desc.ownership    = static_cast<int32_t>(ShardOwnership::CONTIGUOUS);
        desc.bucket_elems = 0;
    }
    // v1 only understands its own layout. A PER_BUCKET checkpoint has the same
    // total_numel/shard_numel/layout_hash but its shard bytes name different
    // global elements, so without this check it would load cleanly and train
    // garbage. Resharding in this direction belongs in the v2 class, which owns
    // the bucket geometry; point the user at it rather than guessing.
    if (desc.ownership != static_cast<int32_t>(ShardOwnership::CONTIGUOUS)) {
        throw ZeroCheckpointError("this checkpoint uses PER_BUCKET shard ownership "
                                  "(written by SymmetricShardedOptimizer / ZeRO-1 v2). "
                                  "Resume it with v2, which can reshard, or convert it "
                                  "first — v1 cannot reinterpret those bytes.");
    }
    // v1: only same-world-size resume is supported. Cross-W resharding is a
    // documented follow-up (gather all shards -> re-partition).
    if (desc.world_size != W_ ||
        desc.total_numel != static_cast<int64_t>(total_numel_) ||
        desc.shard_numel != static_cast<int64_t>(S_)) {
        throw ZeroCheckpointError("layout mismatch (cross-world-size resume not "
                                  "supported in v1): checkpoint W=" +
                                  std::to_string(desc.world_size) + " current W=" +
                                  std::to_string(W_));
    }
    // Reject any change to the flat parameter ordering / grouping — otherwise
    // the saved shard would be remapped onto the wrong parameters, silently
    // corrupting both optimizer state and model weights.
    if (desc.param_count != static_cast<int32_t>(records_.size()) ||
        desc.num_groups  != static_cast<int32_t>(groups_.size())  ||
        desc.layout_hash != layout_hash_) {
        throw ZeroCheckpointError("layout mismatch: parameter ordering/grouping "
                                  "differs from the checkpoint (param_count/num_groups/"
                                  "layout hash). Resume with the identical model and "
                                  "ZeroConfig grouping.");
    }
    is.read(reinterpret_cast<char*>(&step_count_), sizeof(int64_t));

    shard_master_ = OwnTensor::load_tensor(is).to(device_);
    shard_m_      = OwnTensor::load_tensor(is).to(device_);
    shard_v_      = OwnTensor::load_tensor(is).to(device_);

    // Refresh the (lossy compute-dtype) model params from the restored FP32
    // master so the model reflects the true optimizer state immediately.
    all_gather_params();
    scatter_flat_into_params();
}


// ======================================================================
// DCP flat-shard interface  (OwnTensor::dcp::ShardedOptimizerState)
// ======================================================================
//
// v1 ownership is CONTIGUOUS: this rank owns exactly [rank*S, rank*S + S) of the
// padded flat vector, for each of the three FP32 state buffers. Each buffer is
// therefore ONE contiguous range in the global flat coordinate system, which is
// exactly the shape dcp::OptimStateShardSpec describes.
//
// Naming note: v1 emits three specs; v2 emits one per (state, bucket) because
// its shard is strided. The on-disk tensor names therefore DIFFER between the
// two variants — a v1 checkpoint cannot be resumed by v2 through DCP (or vice
// versa). That mismatch surfaces as a loud coverage failure at load time, not
// as silent garbage. Cross-variant conversion still goes through the stream
// path (load_state), which reshards deliberately.

namespace {
constexpr const char* kZeroMaster = "zero_master";
constexpr const char* kZeroM      = "zero_m";
constexpr const char* kZeroV      = "zero_v";
}  // namespace

std::vector<OwnTensor::dcp::OptimStateShardSpec>
ShardedOptimizer::dcp_state_shard_specs() {
    std::vector<OwnTensor::dcp::OptimStateShardSpec> out;
    out.reserve(3);
    for (const char* st : {kZeroMaster, kZeroM, kZeroV}) {
        OwnTensor::dcp::OptimStateShardSpec s;
        s.state_type   = st;
        s.param_fqn    = dcp_scope_;
        s.kind         = OwnTensor::dcp::StateShardKind::FlatSharded;
        s.global_numel = static_cast<int64_t>(padded_numel_);
        s.flat_offset  = static_cast<int64_t>(shard_begin_);
        s.flat_numel   = static_cast<int64_t>(S_);
        out.push_back(std::move(s));
    }
    return out;
}

OwnTensor::Tensor
ShardedOptimizer::dcp_get_state_shard(const std::string& state_type,
                                      const std::string& param_fqn) {
    if (param_fqn != dcp_scope_) {
        throw ZeroCheckpointError("DCP: param_fqn '" + param_fqn +
                                  "' does not match this optimizer's scope '" +
                                  dcp_scope_ + "'");
    }
    // A staged refresh reads shard_master_ while it publishes; retire it so the
    // bytes handed to the checkpoint are the ones this step actually produced.
    drain_param_refresh();
    // DCP describes a FlatSharded spec with 1-D metadata (global_shape =
    // {global_numel}), and its loader asks load_tensor_region for a 1-D region.
    // The shard buffers are allocated [1,S], so hand back a 1-D VIEW of the same
    // storage — otherwise the saved tensor has rank 2 and the load fails with
    // "Rank mismatch in load_tensor_region: file has 2 but requested 1".
    const Shape flat1d{{static_cast<int64_t>(S_)}};
    if (state_type == kZeroMaster) return shard_master_.view(flat1d);
    if (state_type == kZeroM)      return shard_m_.view(flat1d);
    if (state_type == kZeroV)      return shard_v_.view(flat1d);
    throw ZeroCheckpointError("DCP: unknown state_type '" + state_type + "'");
}

void ShardedOptimizer::dcp_set_state_shard(const std::string& state_type,
                                           const std::string& param_fqn,
                                           const Tensor& flat_slice) {
    if (param_fqn != dcp_scope_) {
        throw ZeroCheckpointError("DCP: param_fqn '" + param_fqn +
                                  "' does not match this optimizer's scope '" +
                                  dcp_scope_ + "'");
    }
    Tensor* dst = nullptr;
    if      (state_type == kZeroMaster) dst = &shard_master_;
    else if (state_type == kZeroM)      dst = &shard_m_;
    else if (state_type == kZeroV)      dst = &shard_v_;
    else throw ZeroCheckpointError("DCP: unknown state_type '" + state_type + "'");

    if (static_cast<size_t>(flat_slice.numel()) != S_) {
        throw ZeroCheckpointError("DCP: shard element count mismatch for '" +
                                  state_type + "': checkpoint " +
                                  std::to_string(flat_slice.numel()) +
                                  " vs expected " + std::to_string(S_));
    }
    if (flat_slice.dtype() != Dtype::Float32) {
        throw ZeroCheckpointError("DCP: shard '" + state_type +
                                  "' must be FP32");
    }
    // DCP hands back a 1-D host tensor; the shard buffers are [1,S] on device.
    // Copy by element count — the shapes differ only in rank, not in extent.
    Tensor src = flat_slice.to(device_);
    d2d(dst->data(), src.data(), S_ * sizeof(float), device_);
    sync_compute();
}

void ShardedOptimizer::dcp_after_load() {
    // dcp_set_state_shard only refilled the FP32 shard buffers. Republish them
    // to the live (compute-dtype) model tensors, the same way load_state ends.
    all_gather_params();
    scatter_flat_into_params();
}

std::map<std::string, float> ShardedOptimizer::get_config() {
    return { { dcp_scope_ + ".step_count", static_cast<float>(step_count_) } };
}

void ShardedOptimizer::set_config(const std::map<std::string, float>& config) {
    auto it = config.find(dcp_scope_ + ".step_count");
    if (it != config.end()) step_count_ = static_cast<int64_t>(it->second);
}

} // namespace zero
} // namespace owt
