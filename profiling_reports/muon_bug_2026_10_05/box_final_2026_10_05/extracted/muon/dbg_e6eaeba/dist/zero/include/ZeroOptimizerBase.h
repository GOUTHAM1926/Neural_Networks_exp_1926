#pragma once
//======================================================================
// ZeroOptimizerBase.h — the surface a training script needs in order to drive
// EITHER ZeRO-1 implementation without knowing which one it holds.
//
//   owt::zero::ShardedOptimizer           (v1) reduce-to-owner + staged broadcast
//   owt::zero::SymmetricShardedOptimizer  (v2) reduce-scatter + all-gather, NVLS
//
// Why this exists: OwnTensor::nn::Optimizer only declares step/set_lr/save_state/
// load_state/step_count/get_m_buffers/get_v_buffers. Every trainer in this repo
// also calls arm_grad_sync(), set_param_refresh_stages(), refresh_param_stage(),
// drain_param_refresh(), zero_grad() and last_grad_norm(), none of which the
// base optimizer knows about — so `std::vector<std::unique_ptr<ShardedOptimizer>>`
// hard-codes the choice of implementation at the declaration site.
//
// Deriving both from this class makes the choice a runtime branch at the single
// construction site while keeping nn::Optimizer in the hierarchy, which is
// required because CheckpointManager::save/load_latest take an nn::Optimizer&.
//
// Every method here is already implemented (with these exact signatures) by v1;
// making them virtual changes no behaviour.
//======================================================================

#include <vector>
#include <cstdint>
#include <string>
#include <utility>

#include "TensorLib.h"
#include "nn/optimizer/Optim.h"
#include "OptimShardSpec.h"   // OwnTensor::dcp::ShardedOptimizerState

namespace owt {
namespace zero {

// Also implements OwnTensor::dcp::ShardedOptimizerState so Distributed
// Checkpointing can see this rank's slice of the flattened optimizer state as a
// range in a GLOBAL flat coordinate system. That is what lets DCP write one
// shard per rank (instead of the old master-only stream dump, which persisted
// only rank 0's slice) and intersect-and-reshard those ranges on load.
//
// The three dcp_* methods are inherited pure-virtual from ShardedOptimizerState;
// each concrete variant implements them over its own ownership scheme (v1
// CONTIGUOUS, v2 PER_BUCKET).
class ZeroOptimizerBase : public OwnTensor::nn::Optimizer,
                          public OwnTensor::dcp::ShardedOptimizerState {
public:
    using Tensor = OwnTensor::Tensor;

    // Construct the base optimizer with an EMPTY parameter list: both ZeRO
    // implementations manage sharded state themselves and must not let the base
    // class allocate full-size FP32 masters.
    ZeroOptimizerBase() : OwnTensor::nn::Optimizer(std::vector<OwnTensor::Tensor>{}) {}
    ~ZeroOptimizerBase() override = default;

    // ---- gradient handling -------------------------------------------------
    // Zero the gradients of the managed model parameters. Non-virtual in the
    // base optimizer (it operates on the base's empty list), so both
    // implementations shadow it; this makes the override dispatch correctly
    // through a base pointer.
    virtual void zero_grad() = 0;

    // Arm the bucketed gradient reduce for ONE backward pass. Call immediately
    // before the FINAL micro-step's loss.backward(). No-op when overlap is off
    // or W==1. See each implementation for the exact contract.
    virtual void arm_grad_sync() = 0;
    virtual bool overlap_enabled() const = 0;

    // ---- staged parameter refresh -----------------------------------------
    // Assign every managed parameter to a refresh stage in forward-consumption
    // order. `stage_of` is index-aligned with the constructor's parameter
    // vector. Must be called once, before the first step().
    virtual void set_param_refresh_stages(const std::vector<int>& stage_of, int num_stages) = 0;

    // Consume refresh stages up to and including `stage`. Safe and cheap when
    // nothing is outstanding. Stages must be consumed in increasing order.
    virtual void refresh_param_stage(int stage) = 0;

    // Consume every outstanding stage. Required before ANY read of the model
    // parameters that is not the training forward pass.
    virtual void drain_param_refresh() = 0;
    virtual bool param_refresh_pending() const = 0;

    // ---- cross-group (global) gradient clipping ---------------------------
    // Each instance owns a DISJOINT subset of the model and computes the exact
    // global norm of that subset, so clipping inside step() clips each group
    // INDEPENDENTLY. DeepSpeed / Megatron / torch.clip_grad_norm_ instead clip on
    // the norm over ALL parameters. The two differ whenever the groups' norms
    // differ: on this repo's model the FP32 group's norm runs ~29x the 1.62 B
    // body's, so independent clipping leaves the body unclipped on 30 of 34 steps
    // while a global clip would have scaled it by ~0.38 — a ~2.6x difference in
    // effective step size on the bulk of the model.
    //
    // Clipping globally requires every group's norm BEFORE any group applies a
    // coefficient, so the step splits in two:
    //
    //     float nsq = 0;
    //     for (auto& o : opts) nsq += o->reduce_and_measure();   // grads reduced, ||g||^2
    //     float coef = std::min(1.f, max_norm / (std::sqrt(nsq) + 1e-6f));
    //     for (auto& o : opts) o->set_external_clip_coef(coef);
    //     for (auto& o : opts) o->step();                        // reuses the reduce
    //
    // reduce_and_measure() performs this group's gradient reduction and returns
    // its ||g||^2 (already all-reduced across ranks) WITHOUT clipping or updating.
    // The following step() detects that the reduction is already done and skips
    // it, so nothing is reduced twice. Returns a negative value when the step is
    // to be skipped (non-finite gradient under a loss scaler).
    //
    // Calling step() without these is unchanged: each group clips on its own norm.
    //
    // Side benefit: it also removes the head-of-line blocking that `step_order`
    // exists to work around, because every norm all-reduce now happens in phase 1,
    // before any group issues its staged parameter transfers.
    virtual float reduce_and_measure() = 0;
    virtual void  set_external_clip_coef(float coef) = 0;

    // ---- diagnostics -------------------------------------------------------
    virtual float get_lr() const = 0;
    virtual void  set_max_grad_norm(float m) = 0;
    virtual float last_grad_norm() const = 0;
    virtual bool  last_step_skipped() const = 0;

    // Layout info (tests / diagnostics).
    virtual int64_t total_numel()  const = 0;
    virtual int64_t padded_numel() const = 0;
    virtual int64_t shard_numel()  const = 0;
    virtual int     world_size()   const = 0;
    virtual int     rank()         const = 0;

    // Short human-readable name of the implementation, for logs ("zero1-v1
    // reduce/broadcast", "zero1-v2 reduce-scatter/all-gather+symmetric").
    virtual const char* variant_name() const = 0;

    // ---- distributed checkpointing (DCP) ----------------------------------
    // A trainer may hold SEVERAL ZeRO instances (bluscript_zero runs one per
    // dtype group), each with its own independent flat space. DCP names a state
    // tensor "optimizer.<state_type>.<param_fqn>", so without a distinct scope
    // per instance every group would write to the same name and silently
    // overwrite the others. Set this once, before the first save/load.
    void set_dcp_scope(std::string s) { dcp_scope_ = std::move(s); }
    const std::string& dcp_scope() const { return dcp_scope_; }

    // Call immediately BEFORE handing this optimizer to DCP's load: retires any
    // in-flight parameter refresh, whose target buffers the load overwrites.
    virtual void dcp_before_load() { drain_param_refresh(); }

    // Call immediately AFTER DCP's load returns. dcp_set_state_shard() only
    // repopulates the FP32 shard buffers; this republishes them to the live
    // model tensors (all-gather + scatter), which is the step the stream-based
    // load_state() used to do at the end of its own body.
    virtual void dcp_after_load() = 0;

    // ---- ADD NEW VIRTUALS BELOW THIS LINE, NEVER ABOVE IT --------------------
    // This class is inherited by objects that live in libdist as well as ones
    // compiled straight into the trainer. Inserting a virtual anywhere but the
    // END shifts the vtable slot of every virtual after it, and any object file
    // not recompiled against the new header then dispatches through the wrong
    // slot — a garbage `this`/pointer, not a link error. That is exactly what
    // adding clip_covered_numel() mid-class produced: a "GPU->GPU transfer
    // failed: invalid argument" out of copy_memory during ShardedMuon's
    // constructor. Appending keeps the change additive, so a stale object at
    // worst lacks the new slot instead of corrupting the existing ones.
    //
    // A full `make clean` is still required whenever this file changes.

    // How many of this group's elements the clip coefficient is actually APPLIED
    // to. It is not always total_numel(): ShardedMuon deliberately withholds the
    // scale from orthogonalized tensors (Newton-Schulz renormalizes the momentum,
    // so a varying coefficient would only distort the EMA's direction), and in the
    // usual hybrid partition that is every tensor it owns — so its coverage is 0.
    //
    // Exposed because the alternative is a lie in the log. A step that prints
    // `globalclip 0.0457` reads as "the model's gradients were scaled to 4.6%".
    // If coverage is 0 for the group holding 94% of the parameters, nothing of the
    // sort happened. The trainer sums this across groups and reports the fraction.
    virtual size_t clip_covered_numel() const = 0;

protected:
    std::string dcp_scope_ = "zero";
};

} // namespace zero
} // namespace owt
