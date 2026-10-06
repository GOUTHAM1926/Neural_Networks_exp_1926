//======================================================================
// ShardedMuon.cpp — ZeRO Stage-1 for Muon, over whole-parameter ownership.
// See ShardedMuon.h for why elementwise ZeRO-1 cannot be reused here.
//======================================================================

#include "ShardedMuon.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>

#include <cuda_runtime.h>

#include "device/DeviceCore.h"        // OwnTensor::cuda::getCurrentStream()
#include "device/DeviceTransfer.h"    // OwnTensor::device::copy_memory
#include "ops/helpers/AdamKernels.h"          // fused_adam_cuda (AdamW path)
#include "ops/helpers/MuonKernels.h"          // muon_* fused kernels
#include "ops/helpers/MultiTensorKernels.h"   // multi_tensor_grad_norm / _scale
#include "ops/helpers/ConversionKernels.cuh"  // convert_type_cuda_generic
#include "core/Serialization.h"       // save_tensor / load_tensor
#include "core/Views/ViewUtils.h"     // compute_strides (gradient bucket views)
#include "NcclSymmetricAllocator.hpp" // symmetric-memory backing for wire buffers
#include "BlucclSymmetricAllocator.hpp"
#include "autograd/Hooks.h"           // make_post_acc_hook
#include "autograd/Engine.h"          // get_execution_mode / ExecutionMode

namespace owt {
namespace zero {

using OwnTensor::Tensor;
using OwnTensor::Dtype;
using OwnTensor::Shape;
using OwnTensor::TensorOptions;
namespace dev   = OwnTensor::device;
namespace ocuda = OwnTensor::cuda;

namespace {

inline size_t elem_size(Dtype d) { return Tensor::dtype_size(d); }

// F1. Text for ZeroNonFiniteGradError: which optimizer and group, how big, which rank and
// step, so the log names the failing group. Same wording as v1's
// ShardedOptimizer::nonfinite_grad_msg. File-local on purpose: a new member function would
// change the header, and every trainer that includes it would need a rebuild.
std::string nonfinite_grad_msg(float nsq, Dtype dt, size_t numel, int rank, int64_t next_step) {
    const char* g = dt == Dtype::Bfloat16 ? "bf16" : dt == Dtype::Float32 ? "fp32" : "other";
    return std::string("ShardedMuon: grad norm^2 = ") + std::to_string(nsq) + " in the " + g +
           " group (" + std::to_string(numel) + " params), rank " + std::to_string(rank) +
           ", optimizer step " + std::to_string(next_step) +
           "; no LossScaler installed, so the step cannot be skipped -- raised BEFORE any update";
}

inline void d2d(void* dst, const void* src, size_t bytes, OwnTensor::DeviceIndex di) {
    if (bytes == 0) return;
    dev::copy_memory(dst, di.device, src, di.device, bytes);
}

// Host staging buffer -> device. Distinct from d2d: handing a host pointer to
// the d2d path tags both endpoints CUDA and the driver rejects the copy with
// "invalid argument" rather than reading through it.
inline void h2d(void* dst, const void* src, size_t bytes, OwnTensor::DeviceIndex di) {
    if (bytes == 0) return;
    dev::copy_memory(dst, di.device, src, OwnTensor::Device::CPU, bytes);
}

inline void sync_compute() { cudaStreamSynchronize(ocuda::getCurrentStream()); }

// Wait on an overlapped collective and turn a FAILURE into an exception.
// A discarded error here is silent corruption rather than a crash: the
// collective never wrote its destination, so the buffer keeps the PREVIOUS
// step's bytes, the update consumes them, and training continues with a wrong
// gradient (or publishes a stale weight) with nothing logged.
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

// Copy a checkpointed tensor INTO an owned-state view. The views alias one flat
// device buffer, so the old `owned_master_[s] = load_tensor(is).to(device_)`
// would silently detach the slot from that buffer — the update would then write
// a private allocation the all-gather never reads, and every other slot's view
// would still point at the (now stale) flat buffer.
inline void copy_into_owned(Tensor& dst, const Tensor& src,
                            OwnTensor::DeviceIndex device,
                            const char* what, size_t slot) {
    if (!dst.is_valid()) {
        throw ZeroCheckpointError(std::string("checkpoint carries ") + what +
                                  " for slot " + std::to_string(slot) +
                                  " but this rank has no buffer for it");
    }
    if (src.numel() != dst.numel()) {
        throw ZeroCheckpointError(std::string("checkpointed ") + what + " for slot " +
                                  std::to_string(slot) + " has " +
                                  std::to_string(src.numel()) + " elements, expected " +
                                  std::to_string(dst.numel()));
    }
    if (src.dtype() != Dtype::Float32) {
        throw ZeroCheckpointError(std::string("checkpointed ") + what + " for slot " +
                                  std::to_string(slot) + " must be FP32");
    }
    // load_tensor() always materializes on the CPU and every consumer below
    // dereferences these as DEVICE pointers.
    Tensor on_dev = src.to(device);
    dev::copy_memory(dst.data(), device.device, on_dev.data(), device.device,
                     dst.numel() * sizeof(float));
}

inline uint64_t fnv(uint64_t h, uint64_t x) {
    for (int b = 0; b < 8; ++b) { h ^= (x & 0xff); h *= 1099511628211ull; x >>= 8; }
    return h;
}

} // anonymous namespace

// ======================================================================
// Construction
// ======================================================================

ShardedMuon::ShardedMuon(const std::vector<Tensor>& model_parameters,
                         std::shared_ptr<dist::comm::ProcessGroupBase> pg,
                         MuonConfig cfg)
    : ZeroOptimizerBase(),   // EMPTY param list: handing the base the real
      cfg_(cfg),             // params would build full-size FP32 masters and
      pg_(std::move(pg))     // defeat the whole point of sharding
{
    if (cfg_.lr <= 0.0f) {
        throw ZeroConfigError("MuonConfig::lr must be > 0 (it anchors the "
                              "adamw_lr/lr ratio that set_lr preserves)");
    }
    if (cfg_.ns_steps < 1) {
        throw ZeroConfigError("MuonConfig::ns_steps must be >= 1");
    }
    adamw_lr_ratio_ = cfg_.adamw_lr / cfg_.lr;

    W_    = pg_ ? pg_->get_worldsize() : 1;
    rank_ = pg_ ? pg_->get_rank() : 0;
    if (W_ < 1) W_ = 1;
    if (W_ > 1 && !pg_) {
        throw ZeroConfigError("world_size > 1 requires a process group");
    }

    build_layout(model_parameters);
    // Ownership is derived DIFFERENTLY per collective mode: reduce/broadcast
    // packs greedily over all parameters at once, while reduce-scatter/all-gather
    // must pack within each bucket independently (a bucket is one collective, and
    // NCCL requires every rank's piece of it to be the same length). Both are
    // pure functions of records_ and W, so every rank derives the same map with
    // no communication.
    if (rsag()) build_bucketed_layout(); else assign_owners();
    allocate_owned_state();
    if (rsag()) { allocate_flat_buffers(); register_windows(); }
    /* edit */
    if (cfg_.grad_bucket_view) install_grad_views();
    /* edit */
    // Purely local preconditions; the backend decline is discovered at bucket 0.
    // overlap_param_refresh is refused because the publish IS the gather, and the
    // staged refresh expects to issue and consume gathers stage by stage.
    fused_publish_active_ = cfg_.fused_publish && rsag() && W_ > 1
                            && !cfg_.overlap_param_refresh;
    if (cfg_.fused_publish && !fused_publish_active_ && cfg_.verbose && rank_ == 0) {
        std::cout << "[ZeRO-1/Muon] BLU_FUSED_MUON_PUBLISH=1 INACTIVE: needs the "
                     "reduce-scatter/all-gather collective, world_size > 1, and "
                     "overlap_param_refresh off." << std::endl;
    }
    /* edit */
    /* edit */
    if (cfg_.verify_consistency && W_ > 1) verify_consistency();
    if (cfg_.init_broadcast && W_ > 1) broadcast_root_params();
    init_master_from_params();

    // Both the overlap and the reference path launch the SAME asynchronous
    // reduce-to-owner and need the same compute->comm ordering edge, so the event
    // is created whenever there is cross-rank comm at all — not only for overlap.
    if (W_ > 1) cudaEventCreateWithFlags(&compute_sync_event_, cudaEventDisableTiming);

    overlap_active_ = cfg_.overlap_backward_comm && W_ > 1;
    if (overlap_active_) {
        if (OwnTensor::autograd::get_execution_mode() !=
            OwnTensor::autograd::ExecutionMode::SEQUENTIAL) {
            throw ZeroConfigError("overlap_backward_comm requires the SEQUENTIAL "
                                  "autograd engine (backward hooks launch NCCL)");
        }
        register_overlap_hooks();
    }

    if (cfg_.verbose && rank_ == 0) {
        size_t n_muon = 0, n_adamw = 0, muon_elems = 0;
        for (const auto& r : records_) {
            if (r.use_muon) { ++n_muon; muon_elems += r.numel; } else ++n_adamw;
        }
        // Muon carries no second moment for orthogonalized params, so the
        // per-GPU state is 8 B/elem there against AdamW's 12.
        const double state_mb =
            (double)(master_numel_ * 2) * 4.0 / (1024.0 * 1024.0);
        // The FP32 gradient set this used to carry alongside master+momentum is
        // gone (see allocate_owned_state); what is left is one scratch the size
        // of the largest owned parameter, reported separately so an A/B against
        // the old layout is readable from the banner alone.
        const double scratch_mb =
            (double)max_owned_numel_ * 4.0 / (1024.0 * 1024.0);
        std::cout << "[ZeRO-1/Muon] adjust_lr="
                  << (cfg_.adjust_lr == MuonAdjustLr::MATCH_RMS_ADAMW
                          ? "match_rms_adamw" : "original")
                  << " | params=" << total_numel_
                  << " (" << n_muon << " orthogonalized / " << n_adamw << " AdamW)"
                  << " muon_elems=" << muon_elems
                  << " | W=" << W_ << " owned=" << owned_numel_
                  << " (" << owned_.size() << " tensors)"
                  << " | master+momentum: " << state_mb << " MB/GPU"
                  << " + grad scratch " << scratch_mb << " MB"
                  << std::endl;
    }
}

ShardedMuon::~ShardedMuon() {
    // Windows must be torn down BEFORE the tensors they cover are destroyed and
    // before ncclCommDestroy.
    if (pg_) {
        for (auto win : windows_) {
            if (!win) continue;
            try { pg_->deregister_symmetric(win); }
            catch (const std::exception& e) {
                std::cerr << "[ZeRO-1/Muon] deregister_symmetric failed: "
                          << e.what() << std::endl;
            }
        }
    }
    if (d_norm_sq_)     cudaFree(d_norm_sq_);
    if (h_norm_pinned_) cudaFreeHost(h_norm_pinned_);
    if (compute_sync_event_) cudaEventDestroy(compute_sync_event_);
    if (param_sync_event_)   cudaEventDestroy(param_sync_event_);
}

// ======================================================================
// Layout
// ======================================================================

void ShardedMuon::build_layout(const std::vector<Tensor>& model_parameters) {
    std::vector<size_t> force_adamw(cfg_.adamw_param_indices);
    std::sort(force_adamw.begin(), force_adamw.end());
    for (size_t i : force_adamw) {
        if (i >= model_parameters.size()) {
            throw ZeroConfigError("adamw_param_indices holds index " +
                                  std::to_string(i) + " but only " +
                                  std::to_string(model_parameters.size()) +
                                  " parameters were supplied");
        }
    }
    auto is_forced_adamw = [&](size_t i) {
        return std::binary_search(force_adamw.begin(), force_adamw.end(), i);
    };

    records_.clear();
    records_.reserve(model_parameters.size());
    for (size_t i = 0; i < model_parameters.size(); ++i) {
        // Frozen params are never touched by the optimizer — simply absent.
        if (!model_parameters[i].requires_grad()) continue;
        ParamRecord rec;
        rec.param       = model_parameters[i];
        rec.numel       = model_parameters[i].numel();
        rec.input_index = i;
        // Muon orthogonalizes a MATRIX. A 1-D parameter has no second axis to
        // orthogonalize against (Newton-Schulz on a vector collapses to a sign),
        // so norms/biases/gains always take the AdamW path — as does anything the
        // caller excluded (embedding / lm_head).
        rec.use_muon = (model_parameters[i].ndim() >= 2) && !is_forced_adamw(i);
        records_.push_back(std::move(rec));
    }
    if (records_.empty()) throw ZeroConfigError("no trainable parameters supplied");

    device_        = records_[0].param.device();
    compute_dtype_ = records_[0].param.dtype();
    for (const auto& r : records_) {
        if (r.param.dtype() != compute_dtype_) {
            throw ZeroConfigError("all parameters must share one dtype (group by "
                                  "dtype and give each group its own optimizer)");
        }
        if (r.param.device().device != device_.device ||
            r.param.device().index  != device_.index) {
            throw ZeroConfigError("all parameters must live on one device");
        }
    }
    reduce_dtype_ = resolve_reduce_dtype(cfg_.reduce_dtype, compute_dtype_);

    total_numel_ = 0;
    max_numel_   = 0;
    for (const auto& r : records_) {
        total_numel_ += r.numel;
        max_numel_    = std::max(max_numel_, r.numel);
    }

    // Layout identity: the ordered (numel, use_muon) sequence. A checkpoint may
    // only load into a run whose hash matches, so adding, removing, reordering or
    // reclassifying a parameter is caught before state is mapped onto the wrong
    // tensor. Owner assignment is a pure function of this sequence and W, so it
    // needs no separate term.
    uint64_t h = 1469598103934665603ull;
    for (const auto& r : records_) {
        h = fnv(h, r.numel);
        h = fnv(h, r.use_muon ? 1u : 0u);
    }
    // The collective mode changes the OWNER MAP (global vs per-bucket packing),
    // so state saved under one mode maps onto different tensors under the other.
    // Fold it in so a cross-mode load is rejected instead of silently wrong.
    // build_bucketed_layout() folds in the resolved bucket count on top.
    h = fnv(h, rsag() ? 2ull : 1ull);
    layout_hash_ = h;
}

void ShardedMuon::assign_owners() {
    // Whole tensors are indivisible, so the partition is balanced rather than
    // exactly equal. Greedy largest-first onto the least-loaded rank (the classic
    // LPT rule) bounds the worst case at 4/3 of optimal; on a uniform transformer
    // body it does far better — 216 tensors over 8 ranks lands within 0.52% of an
    // even split for this model's shapes.
    //
    // Balancing by ELEMENT COUNT balances memory. It also balances Newton-Schulz
    // cost closely enough: NS is O(steps * min(r,c) * r * c) per matrix, and with
    // a fixed min(r,c) across the transformer body that is proportional to numel.
    //
    // Deterministic on every rank (same records_, same tie-breaks), which is what
    // lets each rank compute the same owner map with no communication.
    std::vector<size_t> order(records_.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        if (records_[a].numel != records_[b].numel)
            return records_[a].numel > records_[b].numel;
        return a < b;
    });

    std::vector<size_t> load(static_cast<size_t>(W_), 0);
    for (size_t idx : order) {
        int best = 0;
        for (int r = 1; r < W_; ++r) if (load[r] < load[best]) best = r;
        records_[idx].owner = best;
        load[best] += records_[idx].numel;
    }

    owned_.clear();
    owned_numel_ = 0;
    for (size_t i = 0; i < records_.size(); ++i) {
        if (records_[i].owner != rank_) continue;
        records_[i].owned_slot = owned_.size();
        owned_.push_back(i);
        owned_numel_ += records_[i].numel;
    }

    if (cfg_.verbose && rank_ == 0 && W_ > 1) {
        const size_t lo = *std::min_element(load.begin(), load.end());
        const size_t hi = *std::max_element(load.begin(), load.end());
        std::cout << "[ZeRO-1/Muon] owner balance: min " << lo << " max " << hi
                  << " elems/rank (imbalance "
                  << (lo > 0 ? 100.0 * (double)(hi - lo) / (double)lo : 0.0)
                  << "%)" << std::endl;
    }
}

// ======================================================================
// REDUCE_SCATTER_ALL_GATHER layout
//
// Tiles the parameters into buckets, and within each bucket bin-packs WHOLE
// tensors onto the W ranks, padding every rank's group out to a common
// per_rank. The bucket's flat range is then exactly [begin, begin+per_rank*W)
// in rank-major order — which is the layout ncclReduceScatter produces and
// ncclAllGather consumes, so one bucket is one collective with no staging copy.
//
// The property that makes this legal for Muon: a tensor is assigned to exactly
// one rank's group and laid down contiguously inside it, so it never straddles
// a shard boundary. Rank r's reduce-scatter output is therefore its COMPLETE
// set of matrices, and Newton-Schulz still sees whole matrices — the thing a
// flat elementwise 1/W slice destroys.
//
// The padding is the cost. Because each bucket is packed independently it
// balances slightly worse than one global pass (the largest tensor in a bucket
// sets a floor on per_rank), and the excess rides the wire on both collectives.
// On a transformer body that is ~2-3%; the report below prints the real number.
// ======================================================================
void ShardedMuon::build_bucketed_layout() {
    buckets_.clear();
    owned_.clear();
    owned_numel_  = 0;
    S_            = 0;
    padded_numel_ = 0;

    const size_t n = records_.size();
    if (n == 0) return;

    // Keep every rank's sub-chunk starting on a 16-byte boundary — NCCL's
    // vectorized paths want it, and it costs at most a few elements per rank.
    //
    // per_rank is a shared ELEMENT count addressed by two buffers of different
    // element widths: flat_grad_ (reduce dtype) and flat_stage_ (compute dtype).
    // The binding constraint is therefore the SMALLER element type, which needs
    // the most elements to reach 16 bytes. Taking max() here would satisfy only
    // the wider buffer and leave the narrower one half-aligned — visible under
    // BLU_FP32_REDUCE=1 on a BF16 group, where per_rank=4 (mod 8) puts every odd
    // rank's slice of flat_stage_ 8 bytes past a boundary.
    const size_t esz   = std::min(elem_size(compute_dtype_), elem_size(reduce_dtype_));
    const size_t align = (esz >= 16) ? 1 : (16 / esz);

    // Bin-pack the records in [lo, hi) onto W ranks, largest-first onto the
    // least-loaded rank (LPT), and return the padded per-rank length. Pure:
    // callers use it both to PROBE a candidate bucket count and to commit one.
    auto pack = [&](size_t lo, size_t hi, std::vector<std::vector<size_t>>* out) {
        std::vector<size_t> order;
        order.reserve(hi - lo);
        for (size_t i = lo; i < hi; ++i) order.push_back(i);
        // Deterministic on every rank: same records_, same descending-numel
        // order, same lowest-index tie-break.
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t c) {
            if (records_[a].numel != records_[c].numel)
                return records_[a].numel > records_[c].numel;
            return a < c;
        });
        std::vector<size_t> load(static_cast<size_t>(W_), 0);
        if (out) out->assign(static_cast<size_t>(W_), {});
        for (size_t idx : order) {
            int best = 0;
            for (int r = 1; r < W_; ++r) if (load[r] < load[best]) best = r;
            load[static_cast<size_t>(best)] += records_[idx].numel;
            if (out) (*out)[static_cast<size_t>(best)].push_back(idx);
        }
        size_t per_rank = 0;
        for (int r = 0; r < W_; ++r) per_rank = std::max(per_rank, load[static_cast<size_t>(r)]);
        if (align > 1) per_rank = ((per_rank + align - 1) / align) * align;
        return per_rank;
    };
    auto bucket_bounds = [&](size_t nb, size_t b) {
        return std::pair<size_t, size_t>{ (n * b) / nb,
                                          (b + 1 == nb) ? n : (n * (b + 1)) / nb };
    };
    auto probe_padded = [&](size_t nb) {
        size_t flat = 0;
        for (size_t b = 0; b < nb; ++b) {
            auto [lo, hi] = bucket_bounds(nb, b);
            if (hi <= lo) continue;
            flat += pack(lo, hi, nullptr) * static_cast<size_t>(W_);
        }
        return flat;
    };

    // Choose the bucket count. More buckets means finer overlap and finer
    // refresh staging, but each is packed independently across all W ranks, so
    // the waste grows as the buckets shrink — and it grows FAST once a single
    // tensor exceeds the ideal per-rank share, since every other rank then pads
    // up to it. Halve until the padding fits the budget.
    size_t nb = static_cast<size_t>(std::max(1, cfg_.target_buckets));
    nb = std::max<size_t>(1, std::min(nb, n));
    const double budget = static_cast<double>(total_numel_) *
                          (1.0 + std::max(0.0f, cfg_.max_padding_frac));
    size_t chosen_pad = probe_padded(nb);
    while (nb > 1 && static_cast<double>(chosen_pad) > budget) {
        const size_t next = nb / 2;
        const size_t next_pad = probe_padded(next);
        if (cfg_.verbose && rank_ == 0) {
            std::cout << "[ZeRO-1/Muon] " << nb << " bucket(s) would pad "
                      << std::fixed
                      << (100.0 * (double)(chosen_pad - total_numel_) / (double)total_numel_)
                      << "% (> " << (100.0f * cfg_.max_padding_frac) << "% budget); "
                      << "falling back to " << next << std::endl;
        }
        nb = next;
        chosen_pad = next_pad;
    }

    size_t flat_cursor = 0, shard_cursor = 0;
    for (size_t b = 0; b < nb; ++b) {
        auto [start, stop] = bucket_bounds(nb, b);
        if (stop <= start) continue;

        std::vector<std::vector<size_t>> groups;
        const size_t per_rank = pack(start, stop, &groups);

        CommBucket cb;
        cb.begin        = flat_cursor;
        cb.per_rank     = per_rank;
        cb.end          = flat_cursor + per_rank * static_cast<size_t>(W_);
        cb.shard_offset = shard_cursor;

        for (int r = 0; r < W_; ++r) {
            // Ascending record index inside a rank's group, so the flat layout
            // still reads in model order and is trivially reproducible.
            auto& g = groups[static_cast<size_t>(r)];
            std::sort(g.begin(), g.end());
            size_t off = 0;
            for (size_t idx : g) {
                auto& rec = records_[idx];
                rec.owner        = r;
                rec.bucket       = buckets_.size();
                rec.flat_offset  = cb.begin + static_cast<size_t>(r) * per_rank + off;
                rec.shard_offset = cb.shard_offset + off;
                cb.records.push_back(static_cast<uint32_t>(idx));
                off += rec.numel;
            }
        }
        std::sort(cb.records.begin(), cb.records.end());
        cb.pending_init = cb.records.size();
        cb.pending      = cb.pending_init;
        buckets_.push_back(std::move(cb));

        flat_cursor  += per_rank * static_cast<size_t>(W_);
        shard_cursor += per_rank;
    }

    padded_numel_ = flat_cursor;
    S_            = shard_cursor;

    // Ascending record order, which keeps owned_slot ordering (and therefore the
    // checkpoint's slot sequence) identical in shape to the reduce/broadcast
    // path, and makes shard_offset monotonic across slots.
    for (size_t i = 0; i < records_.size(); ++i) {
        if (records_[i].owner != rank_) continue;
        records_[i].owned_slot = owned_.size();
        owned_.push_back(i);
        owned_numel_ += records_[i].numel;
    }

    /* edit */
    // Bucket -> this rank's owned slots in it. Used by the pipelined publish to
    // ask "what do I owe bucket k?" without rescanning owned_ per bucket.
    //
    // Built from owned_, which is filled in ascending RECORD order above, so the
    // slots inside each bucket come out ascending too -- the same order the
    // non-pipelined loop visits them in, which keeps update_one()'s per-tensor
    // diagnostics (ns_max_ratio_, rejected_ns_updates_) reporting identically.
    bucket_owned_slots_.assign(buckets_.size(), {});
    for (size_t s = 0; s < owned_.size(); ++s) {
        const size_t bidx = records_[owned_[s]].bucket;
        bucket_owned_slots_[bidx].push_back(static_cast<uint32_t>(s));
    }
    /* edit */

    // The resolved bucket count is part of the layout: it determines the owner
    // map, so a checkpoint written with a different target_buckets must not load.
    layout_hash_ = fnv(layout_hash_, static_cast<uint64_t>(buckets_.size()));

    // A collapse to one bucket is a PERFORMANCE CLIFF, not a tuning detail, so
    // it is reported whether or not verbose is on. Muon owns whole tensors, so
    // per_rank in a bucket is floored by the largest tensor IN that bucket; once
    // W is large enough that this floor exceeds the ideal per-rank share, every
    // finer bucketing blows the padding budget and the halving loop above walks
    // all the way down to 1. One bucket means one monolithic reduce-scatter
    // after the whole backward instead of several overlapped with it — exactly
    // the wrong shape on a multi-node job, where the collective is slowest and
    // hiding it matters most. Raise BLU_MUON_BUCKETS' padding budget
    // (MuonConfig::max_padding_frac) to trade wire bytes back for overlap, or
    // shard Muon over a smaller group than the full world.
    if (rank_ == 0 && W_ > 1 && buckets_.size() == 1 && cfg_.target_buckets > 1) {
        std::cout << "[ZeRO-1/Muon] NOTE: bucket count collapsed "
                  << cfg_.target_buckets << " -> 1 at W=" << W_
                  << " (finer bucketings exceed the "
                  << (100.0f * cfg_.max_padding_frac) << "% padding budget; the "
                     "largest owned tensor floors every bucket's per-rank size). "
                     "The gradient reduce is now a single collective and no "
                     "longer overlaps with backward." << std::endl;
    }

    if (cfg_.verbose && rank_ == 0) {
        const double pad_pct = total_numel_ > 0
            ? 100.0 * (double)(padded_numel_ - total_numel_) / (double)total_numel_
            : 0.0;
        std::cout << "[ZeRO-1/Muon] reduce-scatter/all-gather layout: "
                  << buckets_.size() << " bucket(s), "
                  << (2 * buckets_.size()) << " collectives/step, shard " << S_
                  << " elems/rank, padding " << std::fixed << pad_pct << "%"
                  << " (flat buffer "
                  << (double)padded_numel_ * (double)elem_size(compute_dtype_) / (1024.0*1024.0)
                  << " MB)" << std::endl;
    }
}

Tensor ShardedMuon::alloc_wire_tensor(size_t numel, Dtype dt) {
    const Shape sh{{1, static_cast<int64_t>(numel)}};
    if (!symmetric_active_) {
        return Tensor::zeros(sh, TensorOptions().with_dtype(dt).with_device(device_));
    }
    // Ask the GROUP for its allocator rather than naming NcclSymmetricAllocator:
    // NCCL hands back ncclMemAlloc memory (registrable as a symmetric window),
    // BluCCL hands back gas segments so its collectives run in place instead of
    // staging through two D2D copies per call. symmetric_active_ came from
    // pg_->symmetric_supported(), so a null here is a wiring bug, not a
    // fallback.
    OwnTensor::Allocator* sym_alloc = pg_->symmetric_allocator();
    if (!sym_alloc) {
        throw ZeroConfigError(
            std::string("process group backend '") + pg_->backend_str() +
            "' reported symmetric support but provides no symmetric allocator");
    }
    auto impl = OwnTensor::make_intrusive<OwnTensor::TensorImpl>(
        sh, dt, device_, /*requires_grad=*/false, sym_alloc);
    Tensor t(impl);
    // Symmetric memory is uninitialised and this path does not go through
    // Tensor::zeros()'s memset. The padding tail in particular MUST start at
    // zero: nothing writes it again, and it rides both collectives every step.
    cudaMemsetAsync(t.data(), 0, numel * elem_size(dt), ocuda::getCurrentStream());
    sync_compute();
    return t;
}

void ShardedMuon::allocate_flat_buffers() {
    if (padded_numel_ == 0) return;

    // Decided BEFORE allocating: the choice picks the allocator that backs each
    // wire buffer, and Storage captures its allocator in the deleter at
    // construction, so it cannot be changed afterwards.
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

    // Buffers that cross the wire — the symmetric-window candidates.
    flat_stage_ = alloc_wire_tensor(padded_numel_, compute_dtype_);
    // Aliases flat_stage_ at the default wire dtype. Safe because the gradient
    // bytes are dead by the time the all-gather writes parameters back into the
    // same buffer: the reduce-scatter has already been waited, and the update
    // consumed its output from shard_grad_. Across steps the ordering is enforced
    // by arm_grad_sync(), which refuses to begin a backward with an unconsumed
    // refresh — so the next pass cannot stage gradients over an in-flight gather.
    flat_grad_  = (reduce_dtype_ == compute_dtype_)
                      ? flat_stage_
                      : alloc_wire_tensor(padded_numel_, reduce_dtype_);
    shard_grad_  = alloc_wire_tensor(S_, reduce_dtype_);
    shard_param_ = alloc_wire_tensor(S_, compute_dtype_);
}

void ShardedMuon::register_windows() {
    // Windows are an NCCL concept. A backend with symmetric memory but no
    // window API (BluCCL) keeps its buffers and simply registers nothing.
    if (!symmetric_active_ || !pg_->symmetric_supported()) return;

    // COLLECTIVE and order-sensitive: every rank must register the same buffers
    // in the same sequence. All of them have identical sizes on every rank and
    // each window's head is the tensor's storage base, so every rank's offset
    // into a given window is identical — the NCCL_WIN_COLL_SYMMETRIC contract.
    auto reg = [&](Tensor& t, const char* what) {
        // Register the STORAGE BASE and the PADDED allocation size, not
        // data()/nbytes(): data() carries storage_offset for views, and
        // TensorImpl rounds allocations up to 256 bytes.
        void*  base  = t.unsafeGetTensorImpl()->mutable_storage().data_ptr();
        size_t bytes = t.unsafeGetTensorImpl()->storage().nbytes();
        auto win = pg_->register_symmetric(base, bytes);
        if (!win) {
            throw ZeroConfigError(std::string("symmetric window registration returned "
                                              "null for ") + what);
        }
        windows_.push_back(win);
    };

    reg(flat_stage_, "flat_stage");
    // Only when it is a distinct allocation — registering the same base twice
    // would be a second collective with no matching buffer on the other side.
    if (reduce_dtype_ != compute_dtype_) reg(flat_grad_, "flat_grad");
    reg(shard_grad_,  "shard_grad");
    reg(shard_param_, "shard_param");
}

// ======================================================================
// Owned optimizer state
//
// One flat allocation per state buffer, with a per-tensor VIEW carrying the
// parameter's own shape. See the block comment on flat_master_ in the header for
// why (216 size-class roundings and 216 allocator round-trips collapse to three).
//
// The gradient is NOT one of those buffers. It arrives in the wire dtype, already
// contiguous — shard_grad_ under RS/AG, flat_owned_grad_ under reduce/broadcast —
// and update_one() consumes one tensor at a time, so FP32 is materialized into a
// single max_owned_numel_ scratch by owned_grad_fp32().
// ======================================================================

void ShardedMuon::allocate_owned_state() {
    TensorOptions f32 = TensorOptions().with_dtype(Dtype::Float32).with_device(device_);
    TensorOptions red = TensorOptions().with_dtype(reduce_dtype_).with_device(device_);

    const size_t n = owned_.size();
    owned_master_.assign(n, Tensor());
    owned_m_.assign(n, Tensor());
    owned_v_.assign(n, Tensor());          // left invalid until allocate_v_state()
    owned_grad_.assign(n, Tensor());
    owned_offset_.assign(n, 0);
    owned_v_offset_.assign(n, static_cast<size_t>(-1));
    owned_grad_offset_.assign(n, 0);

    // Slot offsets, padded so every view starts on a kStateAlign boundary. The
    // optimizer kernels issue float4 loads off these pointers, so a view landing
    // mid-16-bytes would fault; kStateAlign is cudaMalloc's own guarantee, which
    // is exactly what a dedicated per-tensor allocation used to provide.
    auto align_up = [](size_t v, size_t a) { return ((v + a - 1) / a) * a; };
    const size_t f32_stride = kStateAlign / sizeof(float);
    const size_t red_stride = std::max<size_t>(1, kStateAlign / elem_size(reduce_dtype_));

    master_numel_    = 0;
    v_numel_         = 0;
    grad_numel_      = 0;
    max_owned_numel_ = 0;
    for (size_t s = 0; s < n; ++s) {
        const auto& rec = records_[owned_[s]];
        owned_offset_[s] = master_numel_;
        master_numel_    = align_up(master_numel_ + rec.numel, f32_stride);
        if (!rec.use_muon) {
            owned_v_offset_[s] = v_numel_;
            v_numel_           = align_up(v_numel_ + rec.numel, f32_stride);
        }
        if (!rsag()) {
            owned_grad_offset_[s] = grad_numel_;
            grad_numel_           = align_up(grad_numel_ + rec.numel, red_stride);
        }
        max_owned_numel_ = std::max(max_owned_numel_, rec.numel);
    }

    // A view is built as slice_inplace(offset, numel) -> view(param shape): the
    // slice shares the flat buffer's storage and carries the element offset, the
    // view puts the matrix shape back on it so Newton-Schulz still sees rows and
    // columns. Both are metadata-only.
    auto make_view = [&](Tensor& flat, size_t off, const Shape& sh, size_t numel) {
        return flat.slice_inplace(off, numel).view(sh);
    };

    if (master_numel_ > 0) {
        const Shape fsh{{1, static_cast<int64_t>(master_numel_)}};
        // master is fully overwritten by init_master_from_params(); m must start
        // at zero, and so must the alignment padding between slots — nothing ever
        // writes it, and a checkpoint save reads only the views, but a NaN there
        // would still surface through any future whole-buffer reduction.
        flat_master_ = Tensor::zeros(fsh, f32);
        flat_m_      = Tensor::zeros(fsh, f32);
        for (size_t s = 0; s < n; ++s) {
            const auto& rec = records_[owned_[s]];
            const Shape sh  = rec.param.shape();
            owned_master_[s] = make_view(flat_master_, owned_offset_[s], sh, rec.numel);
            owned_m_[s]      = make_view(flat_m_,      owned_offset_[s], sh, rec.numel);
        }
    }

    if (!rsag() && grad_numel_ > 0) {
        // REDUCE_BROADCAST has no shard buffer to read the reduced gradients out
        // of: ncclReduce writes one destination per parameter. Flatten those
        // destinations so they are still a single allocation.
        flat_owned_grad_ = Tensor::zeros(Shape{{1, static_cast<int64_t>(grad_numel_)}}, red);
        for (size_t s = 0; s < n; ++s) {
            const auto& rec = records_[owned_[s]];
            owned_grad_[s]  = make_view(flat_owned_grad_, owned_grad_offset_[s],
                                        rec.param.shape(), rec.numel);
        }
    }

    if (max_owned_numel_ > 0) {
        grad_scratch_ = Tensor::empty(Shape{{1, static_cast<int64_t>(max_owned_numel_)}}, f32);
    }

    if (!rsag() && W_ > 1 && max_numel_ > 0) {
        // ncclReduce writes the result only on the root, but every rank must
        // still pass a valid destination.
        scratch_recv_ = Tensor::empty(Shape{{1, static_cast<int64_t>(max_numel_)}}, red);
    }

    cudaMalloc(&d_norm_sq_, sizeof(float));
    cudaMallocHost(&h_norm_pinned_, sizeof(float));
}

void ShardedMuon::allocate_v_state() {
    if (flat_v_.is_valid() || v_numel_ == 0) return;
    TensorOptions f32 = TensorOptions().with_dtype(Dtype::Float32).with_device(device_);
    flat_v_ = Tensor::zeros(Shape{{1, static_cast<int64_t>(v_numel_)}}, f32);
    for (size_t s = 0; s < owned_.size(); ++s) {
        if (owned_v_offset_[s] == static_cast<size_t>(-1)) continue;   // Muon path
        const auto& rec = records_[owned_[s]];
        owned_v_[s] = flat_v_.slice_inplace(owned_v_offset_[s], rec.numel)
                             .view(rec.param.shape());
    }
}


void ShardedMuon::init_master_from_params() {
    for (size_t s = 0; s < owned_.size(); ++s) {
        const auto& rec = records_[owned_[s]];
        if (compute_dtype_ == Dtype::Float32) {
            d2d(owned_master_[s].data(), rec.param.data(),
                rec.numel * sizeof(float), device_);
        } else {
            OwnTensor::convert_type_cuda_generic(
                rec.param.data(), compute_dtype_,
                owned_master_[s].data<float>(), Dtype::Float32,
                static_cast<int64_t>(rec.numel), ocuda::getCurrentStream());
        }
    }
    sync_compute();
}

void ShardedMuon::verify_consistency() {
    // Every rank must agree on the parameter sequence, or the per-parameter
    // collectives below pair up different tensors on different ranks — which
    // NCCL cannot detect (the sizes may even match) and which silently trains
    // garbage. Compare by min/max so any disagreement at all surfaces.
    long long local[3] = { static_cast<long long>(records_.size()),
                           static_cast<long long>(total_numel_),
                           static_cast<long long>(layout_hash_ & 0x7fffffffffffffffull) };
    Tensor buf = Tensor::zeros(Shape{{1, 3}},
                               TensorOptions().with_dtype(Dtype::Float32).with_device(device_));
    // Hash comparison rides on FP64-free math: pack each value as a float pair
    // would lose bits, so compare through the process group in FP32 over a
    // reduced 24-bit digest instead — enough to catch a reordering, and exact
    // for the two counts.
    float h_local[3] = { static_cast<float>(local[0]),
                         static_cast<float>(local[1]),
                         static_cast<float>(static_cast<uint32_t>(layout_hash_ & 0xffffffull)) };
    h2d(buf.data(), h_local, sizeof(h_local), device_);
    sync_compute();

    Tensor mn = Tensor::zeros(buf.shape(), buf.opts());
    Tensor mx = Tensor::zeros(buf.shape(), buf.opts());
    if (pg_->all_reduce(buf.data(), mn.data(), 3, Dtype::Float32, op_min, true) != pgSuccess ||
        pg_->all_reduce(buf.data(), mx.data(), 3, Dtype::Float32, op_max, true) != pgSuccess) {
        throw ZeroCollectiveError("all_reduce(layout check) failed");
    }
    cudaStreamSynchronize(ocuda::getCurrentStream());

    float hmn[3], hmx[3];
    cudaMemcpy(hmn, mn.data(), sizeof(hmn), cudaMemcpyDeviceToHost);
    cudaMemcpy(hmx, mx.data(), sizeof(hmx), cudaMemcpyDeviceToHost);
    const char* what[3] = { "parameter count", "total element count", "layout hash" };
    for (int i = 0; i < 3; ++i) {
        if (hmn[i] != hmx[i]) {
            throw ZeroConsistencyError(std::string("ranks disagree on ") + what[i] +
                                       " (min " + std::to_string(hmn[i]) +
                                       ", max " + std::to_string(hmx[i]) + ")");
        }
    }
}

void ShardedMuon::broadcast_root_params() {
    for (size_t i = 0; i < records_.size(); ++i) {
        auto& rec = records_[i];
        Tensor live = rec.param;   // non-const handle (shares storage) for data()
        sync_compute();
        if (pg_->broadcast(live.data(), live.data(), rec.numel,
                           compute_dtype_, /*root=*/0, /*sync=*/true) != pgSuccess) {
            throw ZeroCollectiveError("broadcast(init params) failed at parameter " +
                                      std::to_string(i));
        }
    }
    cudaDeviceSynchronize();
}

void ShardedMuon::register_overlap_hooks() {
    for (size_t ridx = 0; ridx < records_.size(); ++ridx) {
        Tensor p = records_[ridx].param;   // leaf; requires_grad == true
        p.register_post_acc_hook(OwnTensor::make_post_acc_hook(
            [this, ridx](const Tensor& g) {
                if (this->rsag()) this->on_grad_ready_rsag(ridx, g);
                else              this->on_grad_ready(ridx, g);
            }));
    }
}

// ======================================================================
// Gradient reduce-to-owner
//
// One ncclReduce(op=avg, root=owner) per parameter. There is no bucketing and
// no flat staging buffer: the send side is the parameter's OWN gradient tensor
// (already contiguous and in the compute dtype on the normal path), and the
// receive side is the owner's per-parameter gradient tensor. Whole-parameter
// ownership is what makes that possible — under elementwise sharding a single
// parameter's gradient can span several owners, which is exactly why
// ShardedOptimizer has to stage through a padded flat buffer first.
//
// The collective count is one per parameter rather than one per ~50 MB bucket.
// On a transformer body that is a few hundred reduces of several MB each, which
// stays comfortably bandwidth-bound; a model with many small matrices would want
// owner-grouped bucketing instead.
// ======================================================================

void ShardedMuon::launch_reduce(size_t ridx, const Tensor& grad) {
    auto& rec = records_[ridx];

    Tensor send = grad;
    if (!send.is_contiguous())          send = send.contiguous();
    if (send.dtype() != reduce_dtype_)  send = send.as_type(reduce_dtype_);
    // Hold the storage alive until the wait. On the normal path this is the
    // parameter's own gradient tensor (a refcount bump, no copy); when a cast or
    // a compaction was needed it is that temporary, which would otherwise be
    // freed the moment this function returns while NCCL is still reading it.
    reduce_keepalive_.push_back(send);

    if (W_ == 1) {
        // Averaging over one rank is the identity.
        d2d(owned_grad_[rec.owned_slot].data(), send.data(),
            rec.numel * elem_size(reduce_dtype_), device_);
        return;
    }

    // compute -> comm edge: the comm stream waits for the backward kernels that
    // produced this gradient (all enqueued on the compute stream before this
    // record). One reused event is safe: each launch records-then-waits in
    // program order.
    cudaEventRecord(compute_sync_event_, ocuda::getCurrentStream());
    cudaStreamWaitEvent(pg_->get_stream(), compute_sync_event_, 0);

    void* recv = (rank_ == rec.owner) ? owned_grad_[rec.owned_slot].data()
                                      : scratch_recv_.data();
    rec.reduce_work = pg_->reduce_async(send.data(), recv, rec.numel, reduce_dtype_,
                                        /*op=*/avg, /*root=*/rec.owner,
                                        /*sync=*/false, rec.reduce_work);
}

void ShardedMuon::on_grad_ready(size_t ridx, const Tensor& grad) {
    if (!armed_) return;   // inert on earlier micro-steps / unarmed backwards
    grad_fired_[ridx] = true;
    launch_reduce(ridx, grad);
}

void ShardedMuon::reduce_all_grads_reference() {
    // Safe to release last step's send buffers here: this rank has since drained
    // the compute stream behind the norm all-reduce (which is ordered after every
    // one of those reduces on the in-order comm stream) and run a whole update.
    reduce_keepalive_.clear();
    for (size_t ridx = 0; ridx < records_.size(); ++ridx) {
        auto& rec = records_[ridx];
        if (rec.param.has_grad()) {
            launch_reduce(ridx, rec.param.grad_view());
        } else {
            // Unused this step: its averaged gradient is 0 across ranks, matching
            // DDP semantics. Still a collective — every rank must issue it, or the
            // communicator desynchronizes.
            Tensor z = Tensor::zeros(rec.param.shape(),
                                     TensorOptions().with_dtype(reduce_dtype_)
                                                    .with_device(device_));
            launch_reduce(ridx, z);
        }
    }
}

void ShardedMuon::finalize_reduces() {
    // Mop-up for the overlap path: a parameter unused in this step's graph never
    // fires its hook, but its reduce is still a collective every rank must issue.
    // grad_fired_ is authoritative, NOT has_grad(): zero_grad() only zeroes an
    // existing gradient buffer rather than dropping it, so a parameter absent from
    // this step's graph still reports has_grad() and would otherwise contribute a
    // stale gradient.
    //
    // Issued in ascending record order on every rank, after the hook-driven
    // launches, so the issue order still matches across the communicator.
    if (overlap_active_ && armed_) {
        for (size_t ridx = 0; ridx < records_.size(); ++ridx) {
            if (grad_fired_[ridx]) continue;
            Tensor z = Tensor::zeros(records_[ridx].param.shape(),
                                     TensorOptions().with_dtype(reduce_dtype_)
                                                    .with_device(device_));
            launch_reduce(ridx, z);
        }
    }

    // comm -> compute edge. Only OWNED reduces are waited: a non-owned reduce
    // writes scratch_recv_ and its result is discarded, so a failure there cannot
    // corrupt this rank's state, and it stays ordered ahead of the later
    // broadcasts on the same in-order comm stream anyway.
    for (size_t s = 0; s < owned_.size(); ++s) {
        wait_or_throw(records_[owned_[s]].reduce_work, "gradient reduce-to-owner",
                      owned_[s]);
    }

    // reduce_keepalive_ is deliberately NOT released here. Work::wait() only
    // enqueues a cudaStreamWaitEvent on the compute stream — it is not a host
    // stall — so at this point the host has no guarantee that any reduce has
    // actually finished on the device, and the NON-owned reduces were never
    // waited at all. Where BLU_FP32_REDUCE forced a cast, or a parameter was
    // unused and got a zero tensor, this vector is the ONLY owner of the send
    // buffer; dropping it now would hand that block back to the caching allocator
    // while NCCL is still reading it, and the next allocation would silently
    // corrupt an in-flight gradient. It is released at the start of the next
    // gradient pass instead (arm_grad_sync / reduce_all_grads_reference), by
    // which point the norm all-reduce has drained the comm stream.
    armed_ = false;
}

// ======================================================================
// Gradient reduce-scatter (REDUCE_SCATTER_ALL_GATHER)
//
// Every rank stages its FULL local gradient set into flat_grad_ at the layout's
// offsets, then one ncclReduceScatter(op=avg) per bucket hands each rank the
// averaged gradients for exactly the tensors it owns, contiguous in shard_grad_.
// ======================================================================

/* edit */
// =============================================================================
//  GRADIENT BUCKET VIEWS  (MuonConfig::grad_bucket_view, BLU_GRAD_VIEW)
//
//  WHAT IT REMOVES. Without this, autograd allocates a gradient tensor per
//  parameter and every backward hook then copies that gradient into this
//  optimizer's flat wire buffer. On this model's Muon group that copy reads
//  3.62e9 fp32 gradients (14.5 GB) and writes 3.62e9 bf16 (7.25 GB) EVERY step,
//  on the COMPUTE stream, inside the last backward -- competing for bandwidth
//  with exactly the GEMMs the reduce-scatter is trying to hide behind. It also
//  means the fp32 per-parameter gradients (14.5 GB resident) exist at all.
//
//  WHAT IT DOES. Hands autograd a VIEW into flat_grad_ as each parameter's
//  gradient before the first backward. AutogradMeta::accumulate_grad only
//  allocates when the gradient slot is empty (its Case-1.5 branch); with the slot
//  pre-filled it takes the `*grad_ += update` branch instead and accumulates
//  straight into the wire buffer. stage_grad_into_flat() then has nothing to do.
//
//  THE PRICE, AND IT IS NOT FREE. The view has to be in the WIRE dtype, because
//  that is what flat_grad_ is; on this model that is bf16, while autograd's own
//  slot would have been fp32 (it upcasts low-precision leaf gradients on
//  purpose). So micro-step gradient accumulation moves from fp32 to bf16. With
//  grad_accum_steps == 4 that is three bf16 adds instead of three fp32 adds --
//  small, but it is a REAL numerical change and it is why this is off by default
//  and behind its own flag rather than folded into an existing one.
//
//  Muon is the least bad place to spend that: the accumulated gradient goes into
//  a momentum EMA and is then Frobenius-normalised and orthogonalised, so a
//  common scale error cancels and the published recipes run the whole iteration
//  in bf16 anyway. That is an argument for trying it, not a proof it is safe --
//  A/B the loss curve.
//
//  WHEN IT DECLINES, and why each one matters:
//   - not the RS/AG path: the reduce/broadcast path has no flat buffer to view.
//   - a parameter whose dtype differs from the wire dtype: the += would then
//     materialise a full converted temporary per parameter per micro-step, which
//     costs more than the copy this is removing.
//   - overlap_param_refresh: flat_grad_ ALIASES flat_stage_ (allocate_flat_buffers),
//     and the staged refresh is still reading published parameters out of
//     flat_stage_ during the next forward -- exactly when gradients would be
//     accumulating into the same bytes. The two features need flat_grad_ to be a
//     separate 7.25 GB allocation to coexist; until that exists, they are
//     mutually exclusive and this one yields.
// =============================================================================
void ShardedMuon::install_grad_views() {
    auto decline = [&](const char* why) {
        if (cfg_.verbose && rank_ == 0) {
            std::cout << "[ZeRO-1/Muon] BLU_GRAD_VIEW=1 DECLINED: " << why
                      << " -- gradients stage through flat_grad_ as before."
                      << std::endl;
        }
        grad_views_active_ = false;
    };

    if (!rsag())                 return decline("collective is reduce/broadcast, no flat buffer");
    if (W_ <= 1)                 return decline("world_size == 1");
    if (!flat_grad_.is_valid())  return decline("flat_grad_ was not allocated");
    if (cfg_.overlap_param_refresh)
        return decline("overlap_param_refresh aliases flat_stage_ into flat_grad_");

    for (const auto& rec : records_) {
        if (rec.param.dtype() != reduce_dtype_)
            return decline("a parameter's dtype differs from the wire dtype");
    }

    // A view is a second TensorImpl over the SAME storage with a storage_offset;
    // TensorImpl::data() adds that offset, so every kernel that goes through
    // data() sees the right bytes. The shape is the PARAMETER's, not {1,n}: the
    // in-place += checks shapes, and the post-accumulation hook hands this tensor
    // straight back to us.
    auto base = flat_grad_.unsafeGetTensorImpl()->storage_ptr();
    const int64_t base_off = static_cast<int64_t>(flat_grad_.storage_offset());

    grad_views_.clear();
    grad_views_.reserve(records_.size());
    for (auto& rec : records_) {
        const Shape sh = rec.param.shape();
        auto impl = OwnTensor::make_intrusive<OwnTensor::TensorImpl>(
            base, sh, OwnTensor::ViewUtils::compute_strides(sh),
            base_off + static_cast<int64_t>(rec.flat_offset),
            reduce_dtype_, device_);
        Tensor view(std::move(impl));
        Tensor p = rec.param;
        p.set_grad(view);
        grad_views_.push_back(view);
    }
    grad_views_active_ = true;

    if (cfg_.verbose && rank_ == 0) {
        const double saved_mb = (double)total_numel_ * 4.0 / (1024.0 * 1024.0);
        std::cout << "[ZeRO-1/Muon] BLU_GRAD_VIEW=1 ACTIVE: " << records_.size()
                  << " gradients are views into flat_grad_ (no staging copy; "
                  << saved_mb << " MB of per-parameter fp32 gradients never "
                     "allocated). NOTE micro-step accumulation is now "
                  << OwnTensor::get_dtype_name(reduce_dtype_)
                  << ", not float32 -- A/B the loss curve." << std::endl;
    }
}
/* edit */

void ShardedMuon::stage_grad_into_flat(size_t ridx, const Tensor& grad) {
    auto& rec = records_[ridx];
    /* edit */
    // Bucket views: `grad` IS flat_grad_[rec.flat_offset ...]. Autograd
    // accumulated straight into the wire buffer, so there is nothing to stage --
    // copying here would be a self-copy. Only the bucket bookkeeping is left.
    if (grad_views_active_) {
        CommBucket& bv = buckets_[rec.bucket];
        if (bv.pending > 0 && --bv.pending == 0) issue_bucket_rs(rec.bucket);
        return;
    }
    /* edit */
    const size_t rsz = elem_size(reduce_dtype_);
    void* dst = static_cast<char*>(flat_grad_.data()) + rec.flat_offset * rsz;

    Tensor src = grad;
    if (!src.is_contiguous()) src = src.contiguous();
    // Hold the source alive until the next gradient pass. The copy below is
    // ENQUEUED on the compute stream, not completed by it, so a temporary from
    // contiguous() would otherwise be handed back to the caching allocator while
    // the copy is still pending. Released in arm_grad_sync/stage_all_grads.
    reduce_keepalive_.push_back(src);

    if (src.dtype() != reduce_dtype_) {
        OwnTensor::convert_type_cuda_generic(
            src.data(), src.dtype(), dst, reduce_dtype_,
            static_cast<int64_t>(rec.numel), ocuda::getCurrentStream());
    } else {
        d2d(dst, src.data(), rec.numel * rsz, device_);
    }

    CommBucket& b = buckets_[rec.bucket];
    if (b.pending > 0 && --b.pending == 0) issue_bucket_rs(rec.bucket);
}

void ShardedMuon::on_grad_ready_rsag(size_t ridx, const Tensor& grad) {
    if (!armed_) return;   // inert on earlier micro-steps / unarmed backwards
    grad_fired_[ridx] = true;
    stage_grad_into_flat(ridx, grad);
}

void ShardedMuon::issue_bucket_rs(size_t bidx) {
    CommBucket& b = buckets_[bidx];
    if (b.launched) return;
    b.launched = true;
    ++buckets_launched_;

    const size_t rsz = elem_size(reduce_dtype_);
    void* recv = static_cast<char*>(shard_grad_.data()) + b.shard_offset * rsz;
    const void* send = static_cast<const char*>(flat_grad_.data()) + b.begin * rsz;

    if (W_ == 1) {
        // Averaging over one rank is the identity, and the bucket's whole flat
        // range IS this rank's piece.
        d2d(recv, send, b.per_rank * rsz, device_);
        return;
    }

    // compute -> comm edge: the comm stream waits for the staging copies (and the
    // backward kernels behind them) enqueued on the compute stream so far.
    cudaEventRecord(compute_sync_event_, ocuda::getCurrentStream());
    cudaStreamWaitEvent(pg_->get_stream(), compute_sync_event_, 0);

    // op=SUM, not AVG: NCCL ships symmetric kernels only for the `_sum_`
    // variants, so an avg reduce-scatter would silently drop off the NVLS path —
    // which is most of the reason this mode exists. The 1/W rides grad_prescale_
    // instead, applied to each gradient as owned_grad_fp32() up-casts it. That is
    // also the more accurate place for it: the division happens in FP32 after the
    // wire cast rather than inside a BF16 reduction, and 1/W is a power of two so
    // it is exact either way.
    b.rs_work = pg_->reduce_scatter_async(send, recv, b.per_rank, reduce_dtype_,
                                          /*op=*/sum, /*sync=*/false, b.rs_work);
}

void ShardedMuon::stage_all_grads_reference() {
    reduce_keepalive_.clear();
    for (auto& b : buckets_) { b.pending = b.pending_init; b.launched = false; }
    buckets_launched_ = 0;
    for (size_t ridx = 0; ridx < records_.size(); ++ridx) {
        auto& rec = records_[ridx];
        if (rec.param.has_grad()) {
            stage_grad_into_flat(ridx, rec.param.grad_view());
        } else {
            // Unused this step: its averaged gradient is 0 across ranks, matching
            // DDP semantics. Zero the slot directly rather than materializing a
            // zero tensor to copy from.
            const size_t rsz = elem_size(reduce_dtype_);
            cudaMemsetAsync(static_cast<char*>(flat_grad_.data()) + rec.flat_offset * rsz,
                            0, rec.numel * rsz, ocuda::getCurrentStream());
            CommBucket& b = buckets_[rec.bucket];
            if (b.pending > 0 && --b.pending == 0) issue_bucket_rs(rec.bucket);
        }
    }
}

void ShardedMuon::finalize_reduce_scatter() {
    // Mop-up for the overlap path: a parameter unused in this step's graph never
    // fires its hook, so its bucket never fills and its slot still holds the
    // PREVIOUS step's gradient. grad_fired_ is authoritative, not has_grad():
    // zero_grad() zeroes an existing gradient buffer rather than dropping it, so
    // an absent parameter still reports has_grad().
    if (overlap_active_ && armed_) {
        const size_t rsz = elem_size(reduce_dtype_);
        for (size_t ridx = 0; ridx < records_.size(); ++ridx) {
            if (grad_fired_[ridx]) continue;
            auto& rec = records_[ridx];
            cudaMemsetAsync(static_cast<char*>(flat_grad_.data()) + rec.flat_offset * rsz,
                            0, rec.numel * rsz, ocuda::getCurrentStream());
            CommBucket& b = buckets_[rec.bucket];
            if (b.pending > 0 && --b.pending == 0) issue_bucket_rs(rec.bucket);
        }
    }

    // Any bucket still unlaunched (nothing in it fired, or we are on the W==1
    // path). Ascending bucket order on every rank, so the issue order stays
    // paired across the communicator.
    for (size_t b = 0; b < buckets_.size(); ++b) {
        if (!buckets_[b].launched) issue_bucket_rs(b);
    }

    // comm -> compute edge. EVERY bucket is waited, unlike the reduce/broadcast
    // path where a non-owned reduce wrote a discard buffer: here every rank
    // receives a real piece of every bucket.
    for (size_t b = 0; b < buckets_.size(); ++b) {
        wait_or_throw(buckets_[b].rs_work, "gradient reduce-scatter", b);
    }

    // reduce_keepalive_ is deliberately NOT released here — see the note in
    // finalize_reduces(). Work::wait() only enqueues a stream wait, so the host
    // has no guarantee the staging copies have run. Released at the start of the
    // next gradient pass.
    armed_ = false;
}

// ======================================================================
// Owned gradients
//
// This used to be a fan-out: every owned tensor's reduced gradient was up-cast
// into its own persistent FP32 buffer, all of them live at once, and the 1/W and
// clip factors were applied as in-place passes over that whole set. The FP32 set
// cost a second owned_numel_-wide allocation — 1,152 MiB on the reference config
// — for data the update reads one tensor at a time.
//
// It is now materialized on demand. The reduced gradients stay where the
// collective left them, in the wire dtype:
//
//   RS/AG             shard_grad_[rec.shard_offset ..]      (contiguous, [S_])
//   REDUCE_BROADCAST  flat_owned_grad_[owned_grad_offset_[slot] ..]
//
// and owned_grad_fp32() up-casts ONE of them into grad_scratch_ (sized to the
// largest owned parameter), applying grad_prescale_ and, on the AdamW path, the
// pending clip coefficient. The scalars ride the materialization because there is
// no longer a persistent FP32 buffer to scale in place — same arithmetic, same
// order (cast, then prescale, then clip), one tensor at a time.
// ======================================================================

Tensor ShardedMuon::owned_grad_fp32(size_t slot, bool apply_clip) {
    const auto& rec = records_[owned_[slot]];
    const size_t rsz = elem_size(reduce_dtype_);
    const void* src =
        rsag() ? static_cast<const char*>(shard_grad_.data()) + rec.shard_offset * rsz
               : static_cast<const char*>(flat_owned_grad_.data()) +
                     owned_grad_offset_[slot] * rsz;

    Tensor dst = grad_scratch_.slice_inplace(0, rec.numel).view(rec.param.shape());
    if (reduce_dtype_ == Dtype::Float32) {
        d2d(dst.data(), src, rec.numel * sizeof(float), device_);
    } else {
        OwnTensor::convert_type_cuda_generic(
            src, reduce_dtype_, dst.data<float>(), Dtype::Float32,
            static_cast<int64_t>(rec.numel), ocuda::getCurrentStream());
    }

    // muon_scale_inplace_cuda, not multi_tensor_scale_cuda: both compute
    // `x[i] *= s` with the same rounding, but the multi-tensor entry point stages
    // its descriptor through ONE static pinned host buffer and uploads it with
    // cudaMemcpyAsync, so back-to-back calls race (the host rewrites the staging
    // buffer before the DMA for the previous call has run). Every existing caller
    // gets away with it by synchronizing straight afterwards; this path is called
    // once per owned tensor and must not.
    if (grad_prescale_ != 1.0f)
        ocuda::muon_scale_inplace_cuda(dst.data<float>(), grad_prescale_,
                                       static_cast<int64_t>(rec.numel));
    // Clipping is withheld from Muon-path gradients on purpose — see the long
    // note in compute_norm_and_clip().
    if (apply_clip && clip_pending_ && !rec.use_muon)
        ocuda::muon_scale_inplace_cuda(dst.data<float>(), clip_coef_,
                                       static_cast<int64_t>(rec.numel));
    return dst;
}

void ShardedMuon::accumulate_owned_grad_norm(float* d_out) {
    // Pre-clip, by construction: the coefficient this norm produces cannot have
    // been applied yet.
    //
    // The squares are the same values the old full-width fan-out produced. What
    // changes is the summation tree: every owned tensor used to reach ONE
    // multi_tensor_grad_norm_cuda launch, now each reaches its own, so the norm
    // shifts in its last digits. The reduction itself stays as deterministic as
    // it was — the multi-tensor kernel writes per-block partials and a single
    // cleanup block folds them with ONE atomicAdd, and the per-tensor cleanups
    // are serialized on the compute stream, so no new cross-block atomic
    // ordering enters. (The gradients feeding it already vary run to run at
    // about the same magnitude.) Measured on the 3.62 B Muon group at W=4:
    // step-0 ||g|| 17.9589 both before and after.
    //
    // The sync is what makes the per-tensor call legal: multi_tensor_* stages its
    // (ptr,numel) descriptor through ONE static pinned host buffer and uploads it
    // with cudaMemcpyAsync, so a following launch clobbers it mid-flight unless
    // the host waits. SymmetricShardedOptimizer::adam_shard_range serializes its
    // launches for exactly this reason. It costs one sync per owned tensor
    // (~15 us each) against a step of tens of seconds. Removing it means giving
    // that staging buffer more than one bank, which belongs with the
    // sync-batching work in the audit's P3, not here.
    for (size_t s = 0; s < owned_.size(); ++s) {
        Tensor g = owned_grad_fp32(s, /*apply_clip=*/false);
        std::vector<ocuda::TensorInfo> gi{
            ocuda::TensorInfo{ g.data<float>(),
                               static_cast<int64_t>(records_[owned_[s]].numel) } };
        ocuda::multi_tensor_grad_norm_cuda(gi, d_out);
        sync_compute();
    }
}

// ======================================================================
// Global gradient norm + clip
// ======================================================================

bool ShardedMuon::compute_norm_and_clip() {
    OwnTensor::nn::LossScaler* sc = get_scaler();

    // The loss-scaler unscale (grad *= 1/scale) used to run here as an in-place
    // pass over the full FP32 gradient set. That set no longer exists; the factor
    // rides grad_prescale_, set by reduce_grads_now(), so it is already in every
    // gradient this function measures and every gradient update_one() reads.
    clip_pending_ = false;

    // reduce_and_measure() already produced the norm + all-reduce this step.
    if (cached_nsq_ >= 0.0f) {
        const float nsq = cached_nsq_;
        // F1 (cached-norm path, BLU_GLOBAL_CLIP=1). Same change as the guard in the
        // per-step path below; the full explanation is there.
        //   was: last_grad_norm_ = (nsq > 0) ? sqrt(nsq) : 0   -> a NaN norm was LOGGED AS 0
        //   was: if (sc && !isfinite(nsq)) skip               -> no scaler installed = never fires
        // reduce_and_measure() caches only a finite norm, so this guard cannot fire today; it is
        // kept in the same form as the one below so the two paths cannot drift apart.
        last_grad_norm_ = (nsq > 0.0f || std::isnan(nsq)) ? std::sqrt(nsq) : 0.0f;   // NaN logs as nan, not 0
        if (!std::isfinite(nsq)) {
            if (!sc) throw ZeroNonFiniteGradError(nonfinite_grad_msg(nsq, compute_dtype_, total_numel_,
                                                                     rank_, step_count_ + 1));
            sc->update(true); last_step_skipped_ = true; return false;
        }
        // Fall through to the clip below, which reads last_grad_norm_.
    } else {

    // Always computed, even with clipping off: last_grad_norm() is the metric the
    // training loop logs every step, and it is the only cross-rank signal that
    // separates "this group's gradients blew up" from "the update did".
    cudaMemsetAsync(d_norm_sq_, 0, sizeof(float), ocuda::getCurrentStream());
    if (!owned_.empty()) {
        accumulate_owned_grad_norm(d_norm_sq_);
        sync_compute();
    }
    if (W_ > 1) {
        // The owned sets are disjoint and cover every parameter exactly once, so
        // summing the per-rank partial sums of squares gives the EXACT global
        // norm — same guarantee as the elementwise path.
        sync_compute();
        pg_->all_reduce(d_norm_sq_, d_norm_sq_, 1, Dtype::Float32, sum, true);
        // Drain the COMPUTE stream, not the device: a cudaDeviceSynchronize here
        // would also drain another dtype group's in-flight staged parameter
        // broadcasts and silently destroy the overlap they were issued for.
        cudaStreamSynchronize(ocuda::getCurrentStream());
    }
    cudaMemcpy(h_norm_pinned_, d_norm_sq_, sizeof(float), cudaMemcpyDeviceToHost);
    const float nsq = *h_norm_pinned_;
    last_grad_norm_ = (nsq > 0.0f || std::isnan(nsq)) ? std::sqrt(nsq) : 0.0f;   // NaN logs as nan, not 0

    // --- global non-finite guard: atomic step skip (params untouched) ---
    // F1 (the default path; same fix as ShardedOptimizer::step() got in v1).
    // Was:  `if (sc && !std::isfinite(nsq)) { skip }`. set_scaler() is called only from
    //       dist/zero/tests, so in training `sc` is null and this guard NEVER fired. A NaN/Inf
    //       gradient then went on into the update:
    //         1. nsq = NaN or Inf (multi_tensor_grad_norm keeps NaN/Inf); the old
    //            `(nsq > 0) ? sqrt(nsq) : 0` logged a NaN norm as 0
    //         2. Muon-path tensors (every 2-D matrix, i.e. the body) -- SILENT: update_one() applies
    //            weight decay, then muon_momentum_cuda writes the NaN/Inf into the momentum m IN
    //            PLACE, and only then does muon_has_nonfinite_cuda reject the Newton-Schulz output.
    //            The weights stay finite, but m keeps the NaN (m = mu*m + ..., so it never leaves):
    //            every LATER step of that matrix is rejected too, and it silently stops learning
    //            (weight decay only). A checkpoint saves that m, and auto-resume reloads it.
    //         3. AdamW-path tensors: NaN -> last_grad_norm_ = 0 -> clip_coef_ = min(1, max/1e-6) = 1,
    //            no clip, fused Adam writes NaN into master/m/v. Inf -> clip_coef_ = 0, and
    //            Inf * 0 = NaN, same result.
    //         4. a NaN weight makes the next forward's loss NaN -> the trainer aborts one step LATE;
    //            if this step was a checkpoint step, the NaN state was already saved.
    //       Measured before this fix (test_f1_nonfinite, W=1, bf16 and fp32, 256x192 + 128x256
    //       matrices): NaN or Inf in one matrix gradient -> no error, 0 NaN weights, the NaN kept
    //       in m, and that matrix's update rejected on this step AND on the next finite step (the
    //       finite control: 0 and 0 rejected); NaN in a vector -> NaN written into the weight.
    //       BLU_GLOBAL_CLIP=1 did not help: reduce_and_measure() returns -1 for a non-finite norm,
    //       the trainer then only drops the global coefficient and still calls step(), which
    //       arrives here (cached_nsq_ = -1), measures the same NaN again, and went on.
    // Why here: this one number is already computed and all-reduced every step (it is logged
    //       even with clipping off), so the check is free (one isfinite on a host float, once
    //       per group per step), it runs BEFORE update_owned() and before step_count_ advances,
    //       and every rank holds the same all-reduced value, so every rank takes the same branch
    //       (no rank is left waiting in a later collective).
    // Now:  with a LossScaler, unchanged (skip + back off). Without one, throw
    //       ZeroNonFiniteGradError; bluscript_zero_ckpt catches it and aborts with exit code 4,
    //       so no poisoned weights and no poisoned checkpoint. A finite norm takes exactly the
    //       old code path, so training is bit-identical.
    if (!std::isfinite(nsq)) {
        if (!sc) throw ZeroNonFiniteGradError(nonfinite_grad_msg(nsq, compute_dtype_, total_numel_,
                                                                 rank_, step_count_ + 1));
        sc->update(true);
        last_step_skipped_ = true;
        return false;
    }
    }   // end of the "no cached norm" branch

    // --- global gradient clip: AdamW-path tensors ONLY ---
    //
    // This USED to scale every owned gradient, on the grounds that clipping is
    // "mathematically inert for the Muon path, since Newton-Schulz Frobenius-
    // normalizes its input, so a common scale factor cancels". The premise is
    // right and the conclusion does not follow, because the clip lands on the
    // gradient BEFORE it enters a stateful EMA, and Newton-Schulz normalizes the
    // EMA rather than the gradient:
    //
    //     m_t = mu * m_(t-1) + (1 - mu) * c_t * g_t          (muon_momentum_cuda)
    //     U_t = NewtonSchulz(m_t)                            (Frobenius-normalized)
    //
    // With a CONSTANT c, m_t = c * m'_t exactly, the normalization cancels it and
    // the update really is unchanged. With a VARYING c it does not: c_t reweights
    // this step's gradient against ~1/(1-mu) steps of accumulated history, so the
    // DIRECTION of m_t changes. Measured on this repo's 1.62 B Muon group at
    // max_grad_norm=1.0: the coefficient ran 0.042..1.000 (median 0.250, clipped
    // on 30 of 34 steps), a 9x spread of weights inside a single 20-step momentum
    // horizon. That is a large, unintended modification of the Muon update.
    //
    // The published Muon recipes (modded-nanogpt, Moonshot) do not clip the
    // orthogonalized parameters at all, which is what this now matches. The norm
    // is still computed over EVERY owned tensor and still reported through
    // last_grad_norm() — only the SCALE is withheld from Muon-path gradients, so
    // the diagnostic is unchanged and the AdamW arm still clips exactly as before.
    if (cfg_.max_grad_norm > 0.0f && !owned_.empty()) {
        bool any_adamw = false;
        for (size_t s = 0; s < owned_.size(); ++s)
            if (!records_[owned_[s]].use_muon) { any_adamw = true; break; }
        if (any_adamw) {
            if (external_clip_coef_ >= 0.0f) {
                // Cross-group global clipping: one coefficient for every group.
                clip_coef_ = std::min(1.0f, external_clip_coef_);
            } else {
                // The same expression compute_clip_coef_cuda evaluates
                // (max_norm / (sqrt(nsq) + 1e-6), clamped to 1), on the host. nsq
                // is already here — it was copied back to take last_grad_norm_
                // and the non-finite decision — so computing it on the device
                // would only add a round-trip, and the coefficient is now
                // consumed by a host-scalar kernel rather than a device pointer.
                clip_coef_ = std::min(1.0f, cfg_.max_grad_norm / (last_grad_norm_ + 1e-6f));
            }
            // The scale itself is deferred: there is no persistent FP32 gradient
            // set to walk any more, so owned_grad_fp32() applies clip_coef_ to
            // each AdamW-path tensor as it materializes it for update_one().
            // Same coefficient, same tensors, same point in the step — still
            // computed from the pre-clip norm, still landing before the gradient
            // enters the Adam moments.
            clip_pending_ = (clip_coef_ != 1.0f);
        }
    }
    return true;
}

// ======================================================================
// Update — runs only on the owner, over whole tensors
// ======================================================================

bool ShardedMuon::update_one(size_t slot,
                             dist::comm::ProcessGroupBase::MuonPublishSeg* defer) {
    const auto& rec = records_[owned_[slot]];
    Tensor& work = owned_master_[slot];         // FP32 master (view into flat_master_)
    Tensor& m    = owned_m_[slot];              //             (view into flat_m_)
    // Up-cast out of the wire buffer into grad_scratch_, prescaled and (AdamW
    // path) clipped. Valid until the next call, which is why update_owned()
    // finishes one tensor before starting the next.
    Tensor g32   = owned_grad_fp32(slot, /*apply_clip=*/true);
    const int64_t n = static_cast<int64_t>(rec.numel);

    if (rec.use_muon) {
        // 1. Decoupled weight decay on the FP32 master.
        if (cfg_.weight_decay > 0.0f) {
            ocuda::muon_scale_inplace_cuda(work.data<float>(),
                                           1.0f - cfg_.lr * cfg_.weight_decay, n);
        }

        // 2. First-moment EMA (+ optional Nesterov combine), fused.
        //    `g` must be a fresh tensor, never `m`: newton_schulz_orthogonalize
        //    Frobenius-scales its input IN PLACE (Tensor::contiguous() returns
        //    self when the input is already contiguous), so passing the momentum
        //    buffer would destroy it.
        Tensor g = Tensor::empty(m.shape(), m.opts());
        ocuda::muon_momentum_cuda(m.data<float>(), g32.data<float>(), g.data<float>(),
                                  cfg_.momentum, cfg_.nesterov, n);

        // 3. Orthogonalize the whole matrix — the step that cannot be sharded
        //    elementwise, and the reason this rank owns the tensor outright.
        Tensor update = OwnTensor::nn::newton_schulz_orthogonalize(
            g, cfg_.ns_steps, 1e-8f, /*use_bf16=*/true);
        Tensor up = update.is_contiguous() ? update : update.contiguous();

        // 4. Apply, with the reference recipe's aspect-ratio scale.
        //    The bf16 NS iteration only provably converges when every singular
        //    value of the normalized input is <= 1, which ordinary bf16 GEMM
        //    rounding can tip over for an ill-conditioned gradient; the result
        //    can then be finite but enormous. Skip such an update rather than
        //    corrupt the weight for the rest of training.
        if (ocuda::muon_has_nonfinite_cuda(up.data<float>(), up.numel())) {
            ++rejected_ns_updates_;
            ++ns_rejected_step_;
            /* edit */
            // A reject means "leave this tensor alone", and on the fused path it
            // still has to be PUBLISHED -- peers hold no copy of it, so skipping
            // the segment would leave their slice of `dest` holding whatever was
            // there before. alpha == 0 makes the kernel compute
            // `master - master*0 == master` and publish that: the update is
            // withheld, the value still reaches every rank, and the segment count
            // stays identical across ranks, which the per-block barrier requires.
            if (defer) {
                *defer = { work.data<float>(), work.data<float>(),
                           static_cast<std::int64_t>(rec.flat_offset),
                           static_cast<std::int64_t>(rec.numel), 0.0f, 0 };
                return true;
            }
            /* edit */
            return false;
        }
        // Newton-Schulz returns singular values ~1, so the update's ENTRIES have
        // RMS 1/sqrt(max(rows,cols)) — smaller for a bigger matrix. Both
        // published conventions correct for that; see MuonAdjustLr for why the
        // default is the one that puts lr on AdamW's scale.
        const int64_t rows = work.shape().dims[0];
        const int64_t cols = n / rows;

        // DIAGNOSTIC ONLY: Newton-Schulz returns singular values ~1, so a healthy
        // update has ||update||_F ~ sqrt(min(rows,cols)). Track the worst ratio
        // this step. Costs one reduction + one sync per owned tensor; the guard
        // above already syncs per tensor, so this does not change the step's
        // structure. Nothing here feeds the update.
        {
            cudaMemsetAsync(d_norm_sq_, 0, sizeof(float), ocuda::getCurrentStream());
            std::vector<ocuda::TensorInfo> ti{
                ocuda::TensorInfo{ up.data<float>(), static_cast<int64_t>(up.numel()) } };
            ocuda::multi_tensor_grad_norm_cuda(ti, d_norm_sq_);
            float nsq = 0.0f;
            cudaMemcpyAsync(&nsq, d_norm_sq_, sizeof(float),
                            cudaMemcpyDeviceToHost, ocuda::getCurrentStream());
            cudaStreamSynchronize(ocuda::getCurrentStream());
            const float expect =
                std::sqrt(static_cast<float>(std::min(rows, cols)));
            if (expect > 0.0f) {
                const float ratio = std::sqrt(std::max(0.0f, nsq)) / expect;
                if (ratio > ns_max_ratio_) ns_max_ratio_ = ratio;
            }
        }
        const float adjust =
            (cfg_.adjust_lr == MuonAdjustLr::MATCH_RMS_ADAMW)
                ? 0.2f * std::sqrt(static_cast<float>(std::max(rows, cols)))
                : std::sqrt(std::max(1.0f, static_cast<float>(rows) /
                                           static_cast<float>(cols)));
        /* edit */
        // Same arithmetic either way: muon_apply_kernel is `w -= u * coef` and
        // the fused kernel's muon_apply_elem is `w - u * alpha`. Bit-identical,
        // so BLU_FUSED_MUON_PUBLISH must not move the loss; if it does, that is a
        // defect and not a rounding difference.
        if (defer) {
            publish_up_.push_back(up);          // keepalive: consumed later
            *defer = { up.data<float>(), work.data<float>(),
                       static_cast<std::int64_t>(rec.flat_offset),
                       static_cast<std::int64_t>(rec.numel),
                       cfg_.lr * adjust, 0 };
            return true;
        }
        /* edit */
        ocuda::muon_apply_cuda(work.data<float>(), up.data<float>(),
                               cfg_.lr * adjust, n);
    } else {
        // AdamW path (1-D params, embedding, untied lm_head).
        // flat_v_ backs every AdamW-path slot at once; the first one to reach
        // here allocates it, and the Muon-only partition never does.
        if (!owned_v_[slot].is_valid()) allocate_v_state();
        const float bc1 = 1.0f - std::pow(cfg_.adamw_beta1, static_cast<float>(step_count_));
        const float bc2 = 1.0f - std::pow(cfg_.adamw_beta2, static_cast<float>(step_count_));
        const float wd = (cfg_.exclude_1d_from_weight_decay && rec.param.ndim() < 2)
                             ? 0.0f : cfg_.adamw_weight_decay;
        ocuda::fused_adam_cuda(work.data<float>(), g32.data<float>(),
                               m.data<float>(), owned_v_[slot].data<float>(), n,
                               cfg_.adamw_lr, cfg_.adamw_beta1, cfg_.adamw_beta2,
                               cfg_.adamw_eps, wd, bc1, bc2, /*is_adamw=*/true);
        /* edit */
        // fused_adam_cuda already wrote the master, so there is no pending apply
        // -- alpha == 0 turns the fused kernel into cast-and-publish for this
        // segment. (On this model the Muon group has zero AdamW-path tensors; the
        // branch exists for a stray 1-D parameter landing in a Muon dtype group.)
        if (defer) {
            *defer = { work.data<float>(), work.data<float>(),
                       static_cast<std::int64_t>(rec.flat_offset),
                       static_cast<std::int64_t>(rec.numel), 0.0f, 0 };
            return true;
        }
        /* edit */
    }
    return false;
}

/* edit */
// PIPELINED PUBLISH (MuonConfig::pipeline_publish, BLU_MUON_PIPELINE).
//
// The problem it solves. Newton-Schulz is by far the largest item in this
// optimizer's tail -- on a 36-layer/3072-wide model it is ~32 TFLOP per rank per
// step (5 iterations of 4m^2n + 2m^3 over this rank's owned matrices), and it
// runs entirely on the compute stream. The parameter all-gather is ~6 GB per
// rank on the comm stream. Run as written before this change, the two are
// strictly sequential: EVERY owned tensor is orthogonalized, THEN every bucket
// is gathered, THEN every tensor is scattered back. The comm stream is idle for
// the whole Newton-Schulz phase and the compute stream is idle for the whole
// gather.
//
// What changes. Bucket k's all-gather needs exactly one thing: every rank's
// slice of bucket k present in shard_param_. It does not care about bucket k+1.
// So the moment this rank has updated and cast the tensors it owns in bucket k,
// that bucket's all-gather can go -- and it then transfers underneath the
// Newton-Schulz of buckets k+1.. This hides the gather behind compute that was
// going to run anyway.
//
// What does NOT change: the numbers. Same update_one() calls, same order within
// a tensor, same destination offsets. Only the launch schedule moves.
//
// ORDERING, which is the part that can break the job rather than slow it. Every
// rank must issue the same all-gathers in the same sequence on its one in-order
// comm stream, or the communicator desynchronises and the job hangs. The loop
// below is driven by the BUCKET INDEX, never by which tensors this rank happens
// to own or by when they finish -- buckets_ is built identically on every rank,
// so the issue sequence is 0,1,..,n-1 everywhere regardless of ownership. A rank
// that owns nothing in bucket k still issues bucket k's all-gather at the same
// point in the sequence; it simply has no local work to do first.
//
// The compute->comm edge is a recorded event, not a host sync. Re-recording the
// one shared event per bucket is safe for the same reason adam_and_publish_staged()
// re-records it between its two phases: each cudaStreamWaitEvent captures the
// record that precedes it in program order, and this is a single-threaded path.
void ShardedMuon::update_owned() {
    // DIAGNOSTIC ONLY: per-step NS counters describe the step about to be taken.
    ns_rejected_step_ = 0;
    ns_max_ratio_     = 0.0f;

    // Pipelining only makes sense on the RS/AG path (the reduce/broadcast path
    // publishes per parameter, with no buckets to pipeline over) and only when
    // step() is about to call all_gather_params_sync(). Under
    // overlap_param_active_ the publish is issued by issue_refresh_stage() in
    // consumption order instead, and issuing here as well would double-issue
    // every bucket.
    const bool pipeline = cfg_.pipeline_publish && rsag() && !overlap_param_active_
                          && W_ > 1 && !buckets_.empty();

    if (!pipeline) {
        for (size_t s = 0; s < owned_.size(); ++s) {
            update_one(s);
            if (rsag()) continue;   // published via shard_param_ + all-gather instead
            // Cast the updated FP32 master straight into the LIVE parameter tensor.
            // There is no intermediate send buffer: the parameter is owned whole, so
            // it is itself a valid broadcast source, and NCCL supports an in-place
            // broadcast (sendbuff == recvbuff on the root).
            const auto& rec = records_[owned_[s]];
            Tensor live = rec.param;   // non-const handle (shares storage) for data()
            if (compute_dtype_ == Dtype::Float32) {
                d2d(live.data(), owned_master_[s].data<float>(),
                    rec.numel * sizeof(float), device_);
            } else {
                OwnTensor::convert_type_cuda_generic(
                    owned_master_[s].data<float>(), Dtype::Float32,
                    live.data(), compute_dtype_,
                    static_cast<int64_t>(rec.numel), ocuda::getCurrentStream());
            }
        }
        if (rsag()) cast_master_into_shard();
        return;
    }

    if (!param_sync_event_)
        cudaEventCreateWithFlags(&param_sync_event_, cudaEventDisableTiming);

    /* edit */
    // FUSED PUBLISH (BLU_FUSED_MUON_PUBLISH). Per bucket: run every owned tensor
    // up to but not including its apply, then hand the whole bucket to one kernel
    // that applies, casts to the wire dtype, and multicast-stores into
    // flat_stage_. That replaces muon_apply_cuda + cast_master_into_shard_one +
    // issue_bucket_ag for the bucket.
    //
    // Bucket 0 is the commit point, exactly as in
    // SymmetricShardedOptimizer::try_fused_adam_and_publish: if the backend
    // declines there, nothing has been applied and the whole step falls back. A
    // decline after bucket 0 cannot be recovered -- some masters are updated and
    // some are not -- so it throws.
    if (fused_publish_active_ && !fused_publish_declined_) {
        const size_t esz = elem_size(compute_dtype_);
        for (size_t b = 0; b < buckets_.size(); ++b) {
            publish_segs_.clear();
            publish_up_.clear();
            for (uint32_t slot : bucket_owned_slots_[b]) {
                dist::comm::ProcessGroupBase::MuonPublishSeg seg{};
                if (update_one(slot, &seg)) publish_segs_.push_back(seg);
            }
            // The compute -> comm edge still has to be recorded even for a bucket
            // this rank owns nothing in: the kernel runs on the group's stream and
            // reads masters written on the compute stream.
            cudaEventRecord(param_sync_event_, ocuda::getCurrentStream());
            cudaStreamWaitEvent(pg_->get_stream(), param_sync_event_, 0);

            // nsegs == 0 is NOT skipped -- every rank must enter the kernel at the
            // same point in the sequence or the per-block barrier desynchronises.
            const bool ok = pg_->fused_muon_publish(
                publish_segs_.data(), static_cast<int>(publish_segs_.size()),
                flat_stage_.data(), static_cast<int>(esz),
                ocuda::getCurrentStream());
            if (!ok) {
                if (b == 0) {
                    fused_publish_declined_ = true;
                    if (cfg_.verbose && rank_ == 0) {
                        std::cout << "[ZeRO-1/Muon] BLU_FUSED_MUON_PUBLISH DECLINED "
                                     "by the backend -- running the unfused "
                                     "apply + cast + all-gather." << std::endl;
                    }
                    publish_segs_.clear();
                    publish_up_.clear();
                    break;          // fall through to the unfused loop below
                }
                throw ZeroCollectiveError(
                    "fused_muon_publish declined at bucket " + std::to_string(b) +
                    " after bucket 0 succeeded; some masters are updated and some "
                    "are not, and the published parameters are now inconsistent "
                    "across ranks. Refusing to continue.");
            }
        }
        if (!fused_publish_declined_) {
            // The publish IS the all-gather. all_gather_params_sync() must not
            // issue or wait for one; it only needs to order the scatter after the
            // kernel, which ran on the group's stream.
            ag_pipelined_   = false;
            ag_fused_done_  = true;
            publish_up_.clear();
            return;
        }
    }
    /* edit */

    for (size_t b = 0; b < buckets_.size(); ++b) {
        for (uint32_t slot : bucket_owned_slots_[b]) {
            update_one(slot);
            cast_master_into_shard_one(slot);
        }
        cudaEventRecord(param_sync_event_, ocuda::getCurrentStream());
        cudaStreamWaitEvent(pg_->get_stream(), param_sync_event_, 0);
        issue_bucket_ag(b);
    }
    ag_pipelined_ = true;
}
/* edit */

// ======================================================================
// Parameter all-gather (REDUCE_SCATTER_ALL_GATHER)
//
// The owner casts its FP32 masters into its contiguous slice of shard_param_,
// one ncclAllGather per bucket reassembles the full flat buffer on every rank,
// and a scatter copies each tensor out into its live parameter. The scatter is
// the one thing a broadcast did not need — a broadcast lands directly in the
// destination tensor, an all-gather lands in the flat buffer.
// ======================================================================

/* edit */
// One owned tensor's master -> its slice of the all-gather send buffer. Split
// out of cast_master_into_shard() so the pipelined publish can cast a tensor the
// moment its update lands, instead of waiting for every other owned tensor to
// finish first. Identical work, identical destination offsets.
void ShardedMuon::cast_master_into_shard_one(size_t slot) {
    const size_t esz = elem_size(compute_dtype_);
    const auto& rec = records_[owned_[slot]];
    void* dst = static_cast<char*>(shard_param_.data()) + rec.shard_offset * esz;
    if (compute_dtype_ == Dtype::Float32) {
        d2d(dst, owned_master_[slot].data<float>(), rec.numel * sizeof(float), device_);
    } else {
        OwnTensor::convert_type_cuda_generic(
            owned_master_[slot].data<float>(), Dtype::Float32,
            dst, compute_dtype_,
            static_cast<int64_t>(rec.numel), ocuda::getCurrentStream());
    }
}
/* edit */

void ShardedMuon::cast_master_into_shard() {
    const size_t esz = elem_size(compute_dtype_);
    const size_t n   = owned_.size();
    // Both sides are monotonic in slot order: build_bucketed_layout lays a rank's
    // tensors down contiguously inside each bucket's shard region, and
    // allocate_owned_state lays flat_master_ out in the same slot order. So a
    // maximal run of slots that is contiguous in BOTH is one conversion instead
    // of one per tensor — 72 launches become one per bucket. Runs break at a
    // bucket boundary (the shard region is padded out to per_rank) and wherever
    // flat_master_'s kStateAlign padding sits.
    size_t s = 0;
    while (s < n) {
        const auto& first = records_[owned_[s]];
        size_t len = first.numel;
        size_t e   = s + 1;
        while (e < n &&
               records_[owned_[e]].shard_offset == first.shard_offset + len &&
               owned_offset_[e]                 == owned_offset_[s]    + len) {
            len += records_[owned_[e]].numel;
            ++e;
        }
        void* dst = static_cast<char*>(shard_param_.data()) + first.shard_offset * esz;
        const float* src = flat_master_.data<float>() + owned_offset_[s];
        if (compute_dtype_ == Dtype::Float32) {
            d2d(dst, src, len * sizeof(float), device_);
        } else {
            OwnTensor::convert_type_cuda_generic(
                src, Dtype::Float32, dst, compute_dtype_,
                static_cast<int64_t>(len), ocuda::getCurrentStream());
        }
        s = e;
    }
}

void ShardedMuon::issue_bucket_ag(size_t bidx) {
    CommBucket& b = buckets_[bidx];
    const size_t esz = elem_size(compute_dtype_);
    const void* send = static_cast<const char*>(shard_param_.data()) + b.shard_offset * esz;
    void*       recv = static_cast<char*>(flat_stage_.data()) + b.begin * esz;
    b.ag_work = pg_->all_gather_async(send, recv, b.per_rank, compute_dtype_,
                                      /*sync=*/false, b.ag_work);
}

void ShardedMuon::scatter_bucket_into_params(size_t bidx) {
    const size_t esz = elem_size(compute_dtype_);
    for (uint32_t ridx : buckets_[bidx].records) {
        const auto& rec = records_[ridx];
        Tensor live = rec.param;   // non-const handle (shares storage) for data()
        d2d(live.data(),
            static_cast<const char*>(flat_stage_.data()) + rec.flat_offset * esz,
            rec.numel * esz, device_);
    }
}

// Completes the publish: wait each bucket's all-gather, scatter that bucket into
// the live parameters, move on. Two things changed here.
//
// 1. The gathers are ASYNC and every bucket's scatter runs as soon as ITS gather
//    lands, rather than all gathers (each a blocking sync=true call, so a host
//    stall per bucket) followed by all scatters. Bucket k's scatter -- a read and
//    a write of that bucket's whole flat range -- now overlaps bucket k+1's
//    transfer instead of queueing behind every transfer in the step.
//
// 2. When update_owned() already pipelined the issue, this function only waits.
//    ag_pipelined_ is the handshake; it is consumed here so a later call on a
//    path that did not pipeline still issues for itself.
//
// Issue order is bucket-ascending in both branches and on every rank -- see the
// ordering note on update_owned().
void ShardedMuon::all_gather_params_sync() {
    const size_t esz = elem_size(compute_dtype_);

    if (W_ == 1) {
        // Degenerate case: the "gather" is a local copy. Kept exactly as it was.
        sync_compute();
        for (size_t b = 0; b < buckets_.size(); ++b) {
            const CommBucket& bk = buckets_[b];
            const void* send = static_cast<const char*>(shard_param_.data()) + bk.shard_offset * esz;
            void*       recv = static_cast<char*>(flat_stage_.data()) + bk.begin * esz;
            d2d(recv, send, bk.per_rank * esz, device_);
        }
        for (size_t b = 0; b < buckets_.size(); ++b) scatter_bucket_into_params(b);
        cudaDeviceSynchronize();
        return;
    }

    /* edit */
    // The fused publish multicast-stored straight into flat_stage_, so there is
    // no all-gather to issue or wait on -- only the host sync that orders the
    // scatter after a kernel that ran on the group's stream.
    if (ag_fused_done_) {
        ag_fused_done_ = false;
        cudaDeviceSynchronize();
        for (size_t b = 0; b < buckets_.size(); ++b) scatter_bucket_into_params(b);
        cudaDeviceSynchronize();
        return;
    }
    /* edit */

    if (!ag_pipelined_) {
        // compute -> comm edge for the whole send buffer, then issue in bucket
        // order. This is the un-pipelined path (BLU_MUON_PIPELINE=0, or a caller
        // that reached here without going through update_owned()).
        sync_compute();
        for (size_t b = 0; b < buckets_.size(); ++b) issue_bucket_ag(b);
    }
    ag_pipelined_ = false;

    for (size_t b = 0; b < buckets_.size(); ++b) {
        // wait_or_throw() RETURNS on a null Work -- that is correct for the
        // reduce/broadcast path, where a non-owned parameter legitimately has no
        // handle, and wrong here: every bucket must have been issued, so a null
        // means the all-gather never happened and the scatter below would publish
        // the previous step's weights with nothing logged. Check it explicitly
        // rather than inherit the permissive default.
        if (!buckets_[b].ag_work) {
            throw ZeroCollectiveError(
                "parameter all-gather #" + std::to_string(b) +
                " has no work handle -- it was never issued, so flat_stage_ still "
                "holds the previous step's parameters. Refusing to scatter them "
                "into the live model.");
        }
        wait_or_throw(buckets_[b].ag_work, "parameter all-gather", b);
        scatter_bucket_into_params(b);
    }
    // The gathers ran on the group's stream; Work::wait() only enqueues a stream
    // wait, so the host still needs this before anything reads the parameters.
    cudaDeviceSynchronize();
}

// ======================================================================
// Publish
// ======================================================================

void ShardedMuon::publish_sync() {
    if (W_ == 1) { sync_compute(); return; }
    sync_compute();
    for (size_t ridx = 0; ridx < records_.size(); ++ridx) {
        auto& rec = records_[ridx];
        Tensor live = rec.param;   // non-const handle (shares storage) for data()
        if (pg_->broadcast(live.data(), live.data(), rec.numel,
                           compute_dtype_, rec.owner, /*sync=*/true) != pgSuccess) {
            throw ZeroCollectiveError("broadcast(param publish) failed at parameter " +
                                      std::to_string(ridx));
        }
    }
    cudaDeviceSynchronize();
}

void ShardedMuon::issue_refresh_stage(int stage) {
    if (rsag()) {
        for (uint32_t bidx : stage_buckets_[stage]) issue_bucket_ag(bidx);
        return;
    }
    for (uint32_t ridx : stage_records_[stage]) {
        auto& rec = records_[ridx];
        Tensor live = rec.param;   // non-const handle (shares storage) for data()
        rec.bcast_work = pg_->broadcast_async(live.data(), live.data(),
                                              rec.numel, compute_dtype_, rec.owner,
                                              /*sync=*/false, rec.bcast_work);
    }
}

void ShardedMuon::set_param_refresh_stages(const std::vector<int>& stage_of,
                                           int num_stages) {
    if (!cfg_.overlap_param_refresh || W_ <= 1) return;   // stays on the sync path
    if (num_stages < 1) throw ZeroConfigError("num_stages must be >= 1");
    // Re-staging with a refresh outstanding would replace the map that
    // refresh_param_stage() is midway through consuming: next_refresh_stage_
    // stays where it was, so any bucket the NEW map places at an earlier stage is
    // never waited, and its scatter copies an in-flight buffer into live weights.
    if (param_refresh_pending_) {
        throw ZeroConfigError("set_param_refresh_stages() called with a parameter "
                              "refresh still in flight — drain_param_refresh() first");
    }

    stage_records_.assign(static_cast<size_t>(num_stages), {});
    for (size_t ridx = 0; ridx < records_.size(); ++ridx) {
        const size_t ii = records_[ridx].input_index;
        // A parameter the caller did not classify falls back to stage 0, which is
        // always CORRECT (stage 0 is consumed before the forward starts) — it only
        // forgoes the overlap for that tensor rather than risking a read before
        // its broadcast has landed.
        int s = 0;
        if (ii < stage_of.size()) s = stage_of[ii];
        if (s < 0 || s >= num_stages) s = 0;
        records_[ridx].stage = s;
        stage_records_[static_cast<size_t>(s)].push_back(static_cast<uint32_t>(ridx));
    }
    // RS/AG publishes a whole bucket per collective, so a bucket must be gathered
    // before the EARLIEST layer that reads any of its parameters. Pinning it to
    // min(stage) over its records is what makes waiting at that stage sufficient
    // for every other parameter it carries. Stage granularity is therefore bucket
    // granularity here, coarser than the per-parameter broadcast path — the
    // tradeoff target_buckets controls.
    if (rsag()) {
        stage_buckets_.assign(static_cast<size_t>(num_stages), {});
        for (size_t b = 0; b < buckets_.size(); ++b) {
            int earliest = num_stages - 1;
            for (uint32_t ridx : buckets_[b].records)
                earliest = std::min(earliest, records_[ridx].stage);
            buckets_[b].refresh_stage = earliest;
            stage_buckets_[static_cast<size_t>(earliest)].push_back(static_cast<uint32_t>(b));
        }
    }

    num_refresh_stages_  = num_stages;
    next_refresh_stage_  = 0;
    overlap_param_active_ = true;
    // Created once: a second call would leak the first event.
    if (!param_sync_event_)
        cudaEventCreateWithFlags(&param_sync_event_, cudaEventDisableTiming);
}

void ShardedMuon::refresh_param_stage(int stage) {
    if (!param_refresh_pending_) return;
    for (int s = next_refresh_stage_; s <= stage && s < num_refresh_stages_; ++s) {
        if (rsag()) {
            // Wait for the gather, THEN scatter the bucket out into the live
            // parameters. Both are ordered on the compute stream ahead of the
            // forward ops that read those weights: wait() enqueues a
            // cudaStreamWaitEvent (no host stall) and the scatter copies follow
            // it, so a layer cannot read a tensor whose bytes are still in the
            // flat buffer or still on the wire.
            for (uint32_t bidx : stage_buckets_[s]) {
                wait_or_throw(buckets_[bidx].ag_work, "parameter all-gather", bidx);
                scatter_bucket_into_params(bidx);
            }
        } else {
            // comm -> compute edge. Work::wait() enqueues a cudaStreamWaitEvent on
            // the compute stream (no host stall), so the forward ops that read
            // these parameters cannot start before the broadcast has landed IN
            // them — on this path there is no staging buffer between the wire and
            // the weight, so this wait is the only thing standing between a
            // half-transferred tensor and a forward pass reading it.
            for (uint32_t ridx : stage_records_[s])
                wait_or_throw(records_[ridx].bcast_work, "parameter broadcast", ridx);
        }
        next_refresh_stage_ = s + 1;
    }
    if (next_refresh_stage_ >= num_refresh_stages_) param_refresh_pending_ = false;
}

void ShardedMuon::drain_param_refresh() {
    if (!param_refresh_pending_) return;
    refresh_param_stage(num_refresh_stages_ - 1);
}

// ======================================================================
// Step
// ======================================================================

// Gradient reduce, split out of step(). The FP32 fan-out that used to follow is
// gone: the reduced gradients are consumed straight out of the wire buffer, one
// tensor at a time, by owned_grad_fp32().
void ShardedMuon::reduce_grads_now() {
    if (rsag()) {
        if (overlap_active_ && armed_) {
            finalize_reduce_scatter();   // mop-up + wait; staged during backward
        } else {
            stage_all_grads_reference();
            finalize_reduce_scatter();
        }
    } else {
        if (overlap_active_ && armed_) {
            finalize_reduces();      // mop-up + wait; grads were reduced in backward
        } else {
            reduce_all_grads_reference();
            finalize_reduces();
        }
    }

    // Everything that used to be an in-place pass over the FP32 gradient set,
    // resolved once here and applied per tensor at materialization:
    //   1/W       the SUM reduce-scatter deliberately did not average (see
    //             issue_bucket_rs). REDUCE_BROADCAST reduces with op=avg and
    //             needs none.
    //   1/scale   the loss-scaler unscale, so both the norm and the update see
    //             the TRUE gradient.
    // Folding them into one coefficient rather than two passes costs at most one
    // extra rounding, and none at all in the usual configuration (no scaler, and
    // 1/W a power of two).
    float pre = (rsag() && W_ > 1) ? (1.0f / static_cast<float>(W_)) : 1.0f;
    if (OwnTensor::nn::LossScaler* sc = get_scaler()) pre /= sc->scale();
    grad_prescale_ = pre;
    sync_compute();

    grads_reduced_ = true;
}

// Phase 1 of cross-group global clipping — see ZeroOptimizerBase.
//
// NOTE the returned ||g||^2 covers EVERY owned tensor, Muon-path included, so the
// combined cross-group norm matches what a single global clip_grad_norm_ would see.
// Whether the resulting coefficient is then APPLIED to a Muon-path gradient is a
// separate question, answered in compute_norm_and_clip(): it is not, because the
// clip would land ahead of a stateful momentum EMA that Newton-Schulz normalises.
float ShardedMuon::reduce_and_measure() {
    // Same precondition as step(): this reduces gradients into the staging buffer
    // that an unconsumed parameter refresh is still writing into (and, on the
    // aliased default, still holds the un-installed parameters). Fail loudly here
    // too rather than only at step().
    if (param_refresh_pending_) {
        throw ZeroConfigError("parameter refresh stages were not consumed before "
                              "reduce_and_measure() — install every stage during the "
                              "forward (refresh_param_stage) or drain explicitly");
    }

    if (!grads_reduced_) reduce_grads_now();
    // The loss-scaler unscale that used to run here — so that the norm measures
    // the TRUE gradient — is part of grad_prescale_, which reduce_grads_now()
    // has just set and which every materialized gradient carries.

    cudaMemsetAsync(d_norm_sq_, 0, sizeof(float), ocuda::getCurrentStream());
    if (!owned_.empty()) {
        accumulate_owned_grad_norm(d_norm_sq_);
        sync_compute();
    }
    if (W_ > 1) {   // COLLECTIVE: every rank participates, owned or not
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

void ShardedMuon::step() {
    last_step_skipped_ = false;

    // An unconsumed refresh here is unrecoverable: this step is about to
    // overwrite the parameter tensors that those broadcasts are still writing
    // into, so the two would race and the ranks would silently diverge.
    if (param_refresh_pending_) {
        throw ZeroConfigError("parameter refresh stages were not consumed before "
                              "step() — every stage must be installed during the "
                              "forward pass (refresh_param_stage) or explicitly "
                              "drained (drain_param_refresh)");
    }

    if (!grads_reduced_) reduce_grads_now();
    grads_reduced_ = false;                 // one reduction per step

    const bool ok = compute_norm_and_clip();
    external_clip_coef_ = -1.0f;            // valid for exactly one step
    cached_nsq_         = -1.0f;
    if (!ok) return;                        // overflow: params untouched

    // Advance the timestep only for a step that actually updates the moments —
    // a step skipped for gradient overflow (returned above) must NOT consume one,
    // or the AdamW path's bias correction drifts permanently.
    ++step_count_;

    update_owned();

    if (overlap_param_active_) {
        // compute -> comm edge, then issue every stage in order on the one
        // in-order comm stream, so waiting stage s also guarantees stages < s
        // have landed. Returns with the broadcasts still in flight: the next
        // forward consumes them stage by stage (refresh_param_stage), and until
        // it does, a non-owner's parameter tensor is being written by NCCL — the
        // reason step()/arm_grad_sync() refuse to proceed with a stage
        // outstanding, and the reason every non-forward read of the weights
        // (validation, checkpointing, inspection) must drain first.
        cudaEventRecord(param_sync_event_, ocuda::getCurrentStream());
        cudaStreamWaitEvent(pg_->get_stream(), param_sync_event_, 0);
        for (int s = 0; s < num_refresh_stages_; ++s) issue_refresh_stage(s);
        param_refresh_pending_ = true;
        next_refresh_stage_    = 0;
    } else if (rsag()) {
        all_gather_params_sync();
    } else {
        publish_sync();
    }

    if (OwnTensor::nn::LossScaler* sc = get_scaler()) sc->update(false);
}

void ShardedMuon::set_lr(float lr) {
    cfg_.lr       = lr;
    cfg_.adamw_lr = lr * adamw_lr_ratio_;
}

void ShardedMuon::arm_grad_sync() {
    // The hooks below reduce each gradient straight out of the parameter's own
    // gradient tensor, and the update writes the parameter tensors. Starting a
    // backward with a refresh outstanding would race NCCL's writes against the
    // forward's reads.
    if (param_refresh_pending_) {
        throw ZeroConfigError("parameter refresh stages were not consumed before "
                              "backward — call refresh_param_stage()/drain_param_refresh() "
                              "for every stage during the forward pass");
    }
    if (!overlap_active_) return;
    grad_fired_.assign(records_.size(), false);
    reduce_keepalive_.clear();
    // Re-arm the bucket fill counters so this pass's hooks launch each bucket
    // exactly once, when its last parameter is staged.
    for (auto& b : buckets_) { b.pending = b.pending_init; b.launched = false; }
    buckets_launched_ = 0;
    armed_ = true;
}

void ShardedMuon::zero_grad() {
    // First optimizer call of the iteration, always ahead of the forward — the
    // right place to consume refresh stage 0.
    if (param_refresh_pending_) refresh_param_stage(0);
    /* edit */
    // With bucket views every gradient lives in one contiguous buffer, so one
    // memset replaces N per-tensor fills -- and it covers the padding tail as
    // well, which arm_grad_sync() would otherwise have to zero separately.
    //
    // This must stay a full-buffer wipe, not a per-record one. flat_grad_ aliases
    // flat_stage_, whose parameter bytes were already consumed by
    // scatter_bucket_into_params() at the end of the previous step; leaving any
    // of it unzeroed would let last step's published weights be read as this
    // step's gradient for a parameter that never fires a hook.
    if (grad_views_active_) {
        cudaMemsetAsync(flat_grad_.data(), 0,
                        padded_numel_ * elem_size(reduce_dtype_),
                        ocuda::getCurrentStream());
        return;
    }
    /* edit */
    for (auto& rec : records_) {
        if (rec.param.has_grad()) rec.param.zero_grad();
    }
}

// ======================================================================
// Checkpoint
// ======================================================================

void ShardedMuon::save_state(std::ostream& os) {
    drain_param_refresh();

    ShardLayoutDescriptor desc;
    desc.world_size   = W_;
    desc.total_numel  = static_cast<int64_t>(total_numel_);
    desc.padded_numel = static_cast<int64_t>(rsag() ? padded_numel_ : total_numel_);
    desc.shard_numel  = static_cast<int64_t>(owned_numel_);
    desc.param_count  = static_cast<int32_t>(records_.size());
    desc.num_groups   = 1;
    desc.layout_hash  = layout_hash_;
    // Neither CONTIGUOUS nor PER_BUCKET: this optimizer owns whole tensors, so
    // reuse the field with a distinct value rather than claim a layout whose
    // bytes mean something else. -1 is reserved here for tensor-level ownership.
    desc.ownership    = -1;
    desc.bucket_elems = 0;

    os.write(reinterpret_cast<const char*>(&desc), sizeof(desc));
    os.write(reinterpret_cast<const char*>(&step_count_), sizeof(int64_t));

    const int64_t n_owned = static_cast<int64_t>(owned_.size());
    os.write(reinterpret_cast<const char*>(&n_owned), sizeof(int64_t));
    for (size_t s = 0; s < owned_.size(); ++s) {
        // Record which global parameter this slot is, so a load can verify the
        // owner map reproduced identically instead of trusting position alone.
        const int64_t ridx = static_cast<int64_t>(owned_[s]);
        os.write(reinterpret_cast<const char*>(&ridx), sizeof(int64_t));
        const int8_t has_v = owned_v_[s].is_valid() ? 1 : 0;
        os.write(reinterpret_cast<const char*>(&has_v), sizeof(int8_t));
        OwnTensor::save_tensor(owned_master_[s], os);
        OwnTensor::save_tensor(owned_m_[s], os);
        if (has_v) OwnTensor::save_tensor(owned_v_[s], os);
    }
}

void ShardedMuon::load_state(std::istream& is) {
    drain_param_refresh();

    ShardLayoutDescriptor desc{};
    is.read(reinterpret_cast<char*>(&desc), sizeof(desc));
    if (!is) throw ZeroCheckpointError("truncated ShardedMuon state (descriptor)");
    if (desc.magic != ShardLayoutDescriptor{}.magic) {
        throw ZeroCheckpointError("bad magic in ShardedMuon state");
    }
    if (desc.ownership != -1) {
        throw ZeroCheckpointError("checkpoint was written by an ELEMENTWISE-sharded "
                                  "optimizer (ownership=" + std::to_string(desc.ownership) +
                                  "); its bytes are a flat buffer slice, not "
                                  "whole-tensor Muon state, and cannot be reinterpreted");
    }
    if (desc.world_size != W_) {
        throw ZeroCheckpointError("world size changed (" + std::to_string(desc.world_size) +
                                  " -> " + std::to_string(W_) + "); resharding whole-tensor "
                                  "Muon state is not implemented");
    }
    if (desc.layout_hash != layout_hash_) {
        throw ZeroCheckpointError("layout hash mismatch — the parameter sequence or the "
                                  "Muon/AdamW classification changed since this "
                                  "checkpoint was written");
    }

    is.read(reinterpret_cast<char*>(&step_count_), sizeof(int64_t));
    int64_t n_owned = 0;
    is.read(reinterpret_cast<char*>(&n_owned), sizeof(int64_t));
    if (n_owned != static_cast<int64_t>(owned_.size())) {
        throw ZeroCheckpointError("owned-tensor count mismatch (" + std::to_string(n_owned) +
                                  " -> " + std::to_string(owned_.size()) + ")");
    }

    for (size_t s = 0; s < owned_.size(); ++s) {
        int64_t ridx = 0;
        is.read(reinterpret_cast<char*>(&ridx), sizeof(int64_t));
        if (ridx != static_cast<int64_t>(owned_[s])) {
            throw ZeroCheckpointError("owner map changed: slot " + std::to_string(s) +
                                      " held parameter " + std::to_string(ridx) +
                                      ", now holds " + std::to_string(owned_[s]));
        }
        int8_t has_v = 0;
        is.read(reinterpret_cast<char*>(&has_v), sizeof(int8_t));
        copy_into_owned(owned_master_[s], OwnTensor::load_tensor(is), device_, "master", s);
        copy_into_owned(owned_m_[s],      OwnTensor::load_tensor(is), device_, "m", s);
        if (has_v) {
            allocate_v_state();
            copy_into_owned(owned_v_[s], OwnTensor::load_tensor(is), device_, "v", s);
        } else if (owned_v_[s].is_valid()) {
            // The checkpoint predates this slot's first AdamW step. flat_v_ may
            // already exist (another slot carried a "v", or this instance has
            // stepped); reset this slot to the all-zero state update_one() would
            // have started it from rather than leave the previous run's moments.
            cudaMemsetAsync(owned_v_[s].data(), 0, owned_v_[s].numel() * sizeof(float),
                            ocuda::getCurrentStream());
        }
    }
    sync_compute();
    if (!is) throw ZeroCheckpointError("truncated ShardedMuon state (tensors)");

    // Republish the restored masters so every rank's live weights match the
    // checkpoint, not whatever the model was initialized with.
    if (rsag()) {
        cast_master_into_shard();
        all_gather_params_sync();
        return;
    }
    for (size_t s = 0; s < owned_.size(); ++s) {
        const auto& rec = records_[owned_[s]];
        Tensor live = rec.param;   // non-const handle (shares storage) for data()
        if (compute_dtype_ == Dtype::Float32) {
            d2d(live.data(), owned_master_[s].data<float>(),
                rec.numel * sizeof(float), device_);
        } else {
            OwnTensor::convert_type_cuda_generic(
                owned_master_[s].data<float>(), Dtype::Float32,
                live.data(), compute_dtype_,
                static_cast<int64_t>(rec.numel), ocuda::getCurrentStream());
        }
    }
    publish_sync();
}


// ====================================================================
// DCP flat-shard interface (OwnTensor::dcp::ShardedOptimizerState)
// ====================================================================

namespace {
constexpr const char* kZeroMaster = "zero_master";
constexpr const char* kZeroM      = "zero_m";
constexpr const char* kZeroV      = "zero_v";

// param_fqn is "<scope>.p<record_index>". Returns false if it is not ours /
// out of range.
bool parse_param_fqn(const std::string& fqn, const std::string& scope,
                     size_t nrecords, size_t& out_index) {
    const std::string prefix = scope + ".p";
    if (fqn.compare(0, prefix.size(), prefix) != 0) return false;
    const std::string digits = fqn.substr(prefix.size());
    if (digits.empty()) return false;
    for (char c : digits) if (c < '0' || c > '9') return false;
    out_index = static_cast<size_t>(std::stoull(digits));
    return out_index < nrecords;
}
}  // namespace
std::vector<OwnTensor::dcp::OptimStateShardSpec>
ShardedMuon::dcp_state_shard_specs() {
    std::vector<OwnTensor::dcp::OptimStateShardSpec> out;
    out.reserve(owned_.size() * 3);
    for (size_t s = 0; s < owned_.size(); ++s) {
        const size_t ridx = owned_[s];
        const auto& rec = records_[ridx];
        const int64_t n = static_cast<int64_t>(rec.numel);
        const std::string fqn = dcp_scope_ + ".p" + std::to_string(ridx);

        auto emit = [&](const char* state_type) {
            OwnTensor::dcp::OptimStateShardSpec spec;
            spec.state_type   = state_type;
            spec.param_fqn    = fqn;
            spec.kind         = OwnTensor::dcp::StateShardKind::FlatSharded;
            spec.global_numel = n;   // whole-tensor ownership: this IS the whole state
            spec.flat_offset  = 0;
            spec.flat_numel   = n;
            out.push_back(std::move(spec));
        };
        emit(kZeroMaster);
        emit(kZeroM);
        // Gated on the record's ROLE (use_muon), not on whether owned_v_[s] has
        // been allocated: this is called on a FRESHLY CONSTRUCTED instance to
        // determine what to request from a checkpoint (see
        // DistributedCheckpointing.cpp's load path), and owned_v_ is always
        // still invalid at that point (it is filled lazily by update_one() on
        // this record's first step) -- gating on validity here would silently
        // never request the AdamW-path "v" state at all on a fresh load.
        if (!rec.use_muon) emit(kZeroV);
    }
    return out;
}
OwnTensor::Tensor
ShardedMuon::dcp_get_state_shard(const std::string& state_type,
                                 const std::string& param_fqn) {
    size_t ridx = 0;
    if (!parse_param_fqn(param_fqn, dcp_scope_, records_.size(), ridx)) {
        throw ZeroCheckpointError("DCP: param_fqn '" + param_fqn +
                                  "' is not a parameter of scope '" + dcp_scope_ + "'");
    }
    const size_t s = records_[ridx].owned_slot;
    if (s == static_cast<size_t>(-1)) {
        throw ZeroCheckpointError("DCP: this rank does not own parameter " +
                                  std::to_string(ridx) + " ('" + param_fqn + "')");
    }
    // A staged refresh reads owned_master_ while it publishes; retire it so
    // the bytes handed to the checkpoint are the ones this step actually
    // produced.
    drain_param_refresh();

    Tensor* src = nullptr;
    if      (state_type == kZeroMaster) src = &owned_master_[s];
    else if (state_type == kZeroM)      src = &owned_m_[s];
    else if (state_type == kZeroV)      src = &owned_v_[s];
    else throw ZeroCheckpointError("DCP: unknown state_type '" + state_type + "'");

    // 1-D on purpose: DCP describes a FlatSharded spec with 1-D metadata and
    // its loader asks load_tensor_region for a 1-D region, while owned_master_/
    // m_/v_ are views carrying the parameter's own (possibly multi-D) shape.
    const int64_t n = static_cast<int64_t>(records_[ridx].numel);

    // The owned buffers are written on the compute stream, and the staging D2H
    // runs on the stager's own stream; sync so it reads this step's bytes.
    sync_compute();

    // A VIEW into flat_master_/flat_m_/flat_v_, not a copy. DCP stages whatever
    // this returns with an ASYNCHRONOUS D2H and does not synchronise until it has
    // walked EVERY spec, so a freshly allocated temporary here would be destroyed
    // at the end of the caller's loop iteration -- returning its block to the
    // caching allocator while the copy is still in flight. The next record of the
    // same size is then handed that same block, overwrites it, and BOTH land on
    // disk holding the second one's bytes. Returning a view also avoids copying
    // the whole master+momentum set (2,808 MB/rank on a 1.47 B-parameter group)
    // on every single checkpoint.
    if (src->is_valid()) return src->view(Shape{{n}});

    // "v" on an AdamW-path record that has not taken its first step yet
    // (owned_v_ is allocated lazily by update_one()) -- report the all-zero
    // state it would start from, exactly what update_one() itself initializes
    // it to on first use, rather than fail a checkpoint saved this early. This
    // one IS a temporary, and is safe only because build_staged_map retains the
    // tensors it stages from until the copies have landed.
    TensorOptions of = TensorOptions().with_dtype(Dtype::Float32).with_device(device_);
    return Tensor::zeros(Shape{{n}}, of);
}
void ShardedMuon::dcp_set_state_shard(const std::string& state_type,
                                      const std::string& param_fqn,
                                      const Tensor& flat_slice) {
    size_t ridx = 0;
    if (!parse_param_fqn(param_fqn, dcp_scope_, records_.size(), ridx)) {
        throw ZeroCheckpointError("DCP: param_fqn '" + param_fqn +
                                  "' is not a parameter of scope '" + dcp_scope_ + "'");
    }
    const size_t s = records_[ridx].owned_slot;
    if (s == static_cast<size_t>(-1)) {
        throw ZeroCheckpointError("DCP: this rank does not own parameter " +
                                  std::to_string(ridx) + " ('" + param_fqn + "')");
    }
    const auto& rec = records_[ridx];
    if (static_cast<size_t>(flat_slice.numel()) != rec.numel) {
        throw ZeroCheckpointError("DCP: element count mismatch for '" + param_fqn +
                                  "/" + state_type + "': checkpoint " +
                                  std::to_string(flat_slice.numel()) +
                                  " vs expected " + std::to_string(rec.numel));
    }
    if (flat_slice.dtype() != Dtype::Float32) {
        throw ZeroCheckpointError("DCP: shard '" + state_type + "' must be FP32");
    }

    // "v" is backed lazily (AdamW path only, on first update_one()); a checkpoint
    // written after that first step carries it before this run's copy exists.
    // Do this BEFORE taking the pointer — allocate_v_state() rewrites owned_v_.
    if (state_type == kZeroV) allocate_v_state();

    Tensor* dst = nullptr;
    if      (state_type == kZeroMaster) dst = &owned_master_[s];
    else if (state_type == kZeroM)      dst = &owned_m_[s];
    else if (state_type == kZeroV)      dst = &owned_v_[s];
    else throw ZeroCheckpointError("DCP: unknown state_type '" + state_type + "'");

    if (!dst->is_valid()) {
        throw ZeroCheckpointError("DCP: no '" + state_type + "' buffer for '" +
                                  param_fqn + "' — the parameter is on the " +
                                  (rec.use_muon ? "Muon" : "AdamW") + " path");
    }

    Tensor src = flat_slice.to(device_);
    d2d(dst->data(), src.data(), rec.numel * sizeof(float), device_);
    sync_compute();
}
void ShardedMuon::republish_owned_masters() {
    if (rsag()) {
        cast_master_into_shard();
        all_gather_params_sync();
        return;
    }
    for (size_t s = 0; s < owned_.size(); ++s) {
        const auto& rec = records_[owned_[s]];
        Tensor live = rec.param;   // non-const handle (shares storage) for data()
        if (compute_dtype_ == Dtype::Float32) {
            d2d(live.data(), owned_master_[s].data<float>(),
                rec.numel * sizeof(float), device_);
        } else {
            OwnTensor::convert_type_cuda_generic(
                owned_master_[s].data<float>(), Dtype::Float32,
                live.data(), compute_dtype_,
                static_cast<int64_t>(rec.numel), ocuda::getCurrentStream());
        }
    }
    publish_sync();
}

void ShardedMuon::dcp_after_load() {
    // dcp_set_state_shard only refilled the owned FP32 buffers. Republish them
    // to the live (compute-dtype) model tensors, the same way load_state ends.
    republish_owned_masters();
}

} // namespace zero
} // namespace owt
