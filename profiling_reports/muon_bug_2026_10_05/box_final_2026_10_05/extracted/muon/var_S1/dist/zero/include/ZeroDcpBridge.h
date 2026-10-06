#pragma once
//======================================================================
// ZeroDcpBridge.h — lets a plain nn::Module trainer that holds N ZeRO-1 dtype
// groups checkpoint through Distributed Checkpointing (DCP).
//
// DCP's entry points (DistCheckpoint::save/load) take exactly one
// dnn::DModule& and one nn::Optimizer*. A ZeRO-1 trainer such as
// bluscript_zero.cpp has neither shape: its model is an nn::Module, and it owns
// one ZeRO optimizer PER DTYPE GROUP. These two adapters supply the missing
// shapes without touching the training path.
//
//   ZeroDcpModule    — presents nn::Module parameters as REPLICATED DTensors.
//   ZeroDcpOptimizer — presents all ZeRO groups as one sharded optimizer.
//
// Why this matters: the stream path it replaces (CheckpointManager +
// Optimizer::save_state) wrote only the calling rank's ZeRO shard to one shared
// file. Under a master-only save that persisted rank 0's slice and every rank
// restored it, silently corrupting both optimizer state and weights on resume
// at world size > 1. Through DCP each rank writes its own slice, and the load
// intersects saved ranges against the resuming rank's range — so a resume at a
// DIFFERENT world size also works.
//======================================================================

#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "core/Tensor.h"
#include "nn/optimizer/Optim.h"

#include "OptimShardSpec.h"
#include "DistributedNN.h"   // OwnTensor::dnn::DModule
#include "dtensor.h"         // DTensor, Layout

#include "ZeroOptimizerBase.h"

namespace owt {
namespace zero {

// ---------------------------------------------------------------------
// ZeroDcpModule: nn::Module parameters as a DModule of replicated DTensors.
// ---------------------------------------------------------------------
class ZeroDcpModule : public OwnTensor::dnn::DModule {
public:
    // `named` maps checkpoint tensor name -> pointer to the LIVE model tensor.
    //
    // The DTensors built here are deliberately "dummy": no ProcessGroup, no
    // DeviceMesh, and a REPLICATED layout. That is safe because DCP never
    // dereferences the mesh on a replicated layout — Layout::get_local_shape()
    // and get_local_offsets() both return before touching mesh_ when the
    // placement is REPLICATE — and replicated is the truth here: ZeRO-1 shards
    // optimizer state, not parameters, so every rank holds the full parameter.
    explicit ZeroDcpModule(const std::map<std::string, OwnTensor::Tensor*>& named) {
        slots_.reserve(named.size());
        for (const auto& kv : named) {
            if (!kv.second || !kv.second->is_valid()) continue;
            auto dt = std::make_unique<DTensor>();
            Layout layout;                                   // REPLICATE, null mesh
            layout.set_global_shape(kv.second->shape().dims);
            dt->mutable_layout() = layout;
            dt->setData(*kv.second);                         // shares storage
            dt->set_name(kv.first);
            named_[kv.first] = dt.get();
            live_[kv.first]  = kv.second;
            slots_.push_back(std::move(dt));
        }
    }

    std::vector<DTensor*> parameters() override {
        std::vector<DTensor*> out;
        out.reserve(named_.size());
        for (const auto& kv : named_) out.push_back(kv.second);
        return out;
    }

    std::map<std::string, DTensor*> named_parameters() override { return named_; }

    // DCP's loader writes through DTensor::mutable_tensor(). For a GPU tensor it
    // ends with local_t.copy_(host_staging) — in place, so the model tensor we
    // share storage with already sees it. But when the on-disk dtype differs
    // from the live one it REBINDS local_t to a freshly allocated tensor, which
    // would leave the model untouched. Copy back for exactly that case.
    void sync_back_to_model() {
        for (auto& kv : named_) {
            OwnTensor::Tensor& src = kv.second->mutable_tensor();
            OwnTensor::Tensor* dst = live_[kv.first];
            if (!dst || !src.is_valid()) continue;
            if (src.unsafeGetTensorImpl() != dst->unsafeGetTensorImpl()) {
                dst->copy_(src);
            }
        }
    }

private:
    std::vector<std::unique_ptr<DTensor>>      slots_;
    std::map<std::string, DTensor*>            named_;
    std::map<std::string, OwnTensor::Tensor*>  live_;
};

// ---------------------------------------------------------------------
// ZeroDcpOptimizer: all ZeRO groups behind one optimizer pointer.
// ---------------------------------------------------------------------
class ZeroDcpOptimizer : public OwnTensor::nn::Optimizer,
                         public OwnTensor::dcp::ShardedOptimizerState {
public:
    // Non-owning. `groups` must outlive this object, and each must already have
    // a distinct dcp_scope() (see set_dcp_scope) — that scope is what routes a
    // param_fqn back to its group and what keeps two groups from writing the
    // same checkpoint tensor name.
    explicit ZeroDcpOptimizer(std::vector<ZeroOptimizerBase*> groups)
        : OwnTensor::nn::Optimizer(std::vector<OwnTensor::Tensor>{}),
          groups_(std::move(groups)) {}

    // This object is a checkpoint facade, never a training path. The real step()
    // is driven per group, in the trainer's own step order.
    void step() override {
        throw std::logic_error("ZeroDcpOptimizer::step(): checkpoint facade only");
    }
    void set_lr(float) override {}

    // Keep DCP's GENERIC per-parameter optimizer path inert: everything this
    // optimizer owns is reported through the flat-shard interface below.
    std::vector<OwnTensor::Tensor> get_params() override { return {}; }
    std::map<std::string, std::vector<OwnTensor::Tensor>> get_state_tensors() override {
        return {};
    }

    // ---- dcp::ShardedOptimizerState ----
    std::vector<OwnTensor::dcp::OptimStateShardSpec> dcp_state_shard_specs() override {
        std::vector<OwnTensor::dcp::OptimStateShardSpec> all;
        for (auto* g : groups_) {
            auto v = g->dcp_state_shard_specs();
            all.insert(all.end(), std::make_move_iterator(v.begin()),
                       std::make_move_iterator(v.end()));
        }
        return all;
    }

    OwnTensor::Tensor dcp_get_state_shard(const std::string& state_type,
                                          const std::string& param_fqn) override {
        return route(param_fqn)->dcp_get_state_shard(state_type, param_fqn);
    }

    void dcp_set_state_shard(const std::string& state_type,
                             const std::string& param_fqn,
                             const OwnTensor::Tensor& flat_slice) override {
        route(param_fqn)->dcp_set_state_shard(state_type, param_fqn, flat_slice);
    }

    // Side-channel for trainer-owned scalars that belong in the checkpoint but
    // live in no shard buffer -- the last validation loss, for instance, which
    // is only recomputed every val_freq steps and would otherwise come back as
    // its -1 sentinel on any resume that does not land on a validation step.
    // These ride the same optimizer_config map: saved from rank 0, restored on
    // every rank. Keys are namespaced so they cannot collide with a ZeRO
    // group's own scope-qualified entries.
    static constexpr const char* kExtraPrefix = "trainer.";

    void set_extra(const std::string& key, float value) {
        extras_[kExtraPrefix + key] = value;
    }
    // Returns false (leaving `out` untouched) when the checkpoint carried no
    // such key -- e.g. one written before this scalar was tracked.
    bool get_extra(const std::string& key, float& out) const {
        auto it = extras_.find(kExtraPrefix + key);
        if (it == extras_.end()) return false;
        out = it->second;
        return true;
    }

    // step_count_ lives in no shard buffer, so each group rides DCP's
    // optimizer_config map under its own scope-qualified key.
    std::map<std::string, float> get_config() override {
        std::map<std::string, float> merged;
        for (auto* g : groups_) {
            for (auto& kv : g->get_config()) merged[kv.first] = kv.second;
        }
        for (const auto& kv : extras_) merged[kv.first] = kv.second;
        return merged;
    }
    void set_config(const std::map<std::string, float>& config) override {
        for (auto* g : groups_) g->set_config(config);
        const std::string pre = kExtraPrefix;
        for (const auto& kv : config) {
            if (kv.first.compare(0, pre.size(), pre) == 0) extras_[kv.first] = kv.second;
        }
    }

    // Bracket a DCP load: retire in-flight refreshes first, republish the
    // restored shards to the live model tensors after.
    void dcp_before_load() { for (auto* g : groups_) g->dcp_before_load(); }
    void dcp_after_load()  { for (auto* g : groups_) g->dcp_after_load(); }

private:
    // A param_fqn is either exactly a group's scope (v1) or "<scope>.b<i>" (v2).
    ZeroOptimizerBase* route(const std::string& fqn) {
        for (auto* g : groups_) {
            const std::string& sc = g->dcp_scope();
            if (fqn == sc) return g;
            if (fqn.size() > sc.size() + 1 &&
                fqn.compare(0, sc.size(), sc) == 0 && fqn[sc.size()] == '.') {
                return g;
            }
        }
        throw std::runtime_error("ZeroDcpOptimizer: no ZeRO group owns '" + fqn + "'");
    }

    std::vector<ZeroOptimizerBase*> groups_;
    std::map<std::string, float> extras_;
};

}  // namespace zero
}  // namespace owt
