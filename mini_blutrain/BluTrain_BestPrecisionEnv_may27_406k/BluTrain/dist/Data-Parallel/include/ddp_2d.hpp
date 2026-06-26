#pragma once

#include "../../communication/include/ProcessGroupNCCL.h"
#include "../../Tensor-Parallelism/dnn/DistributedNN.h"
#include "../../Tensor-Parallelism/tensor/dtensor.h"
#include "../../Tensor-Parallelism/tensor/device_mesh.h"
#include "../../Tensor-Parallelism/tensor/layout.h"
#include "../../Tensor-Parallelism/tensor/placement.h"
#include "Error_logs.hpp"
#include "autograd/Variable.h"
#include "autograd/Engine.h"
#include "TensorLib.h"

#include <memory>
#include <vector>
#include <string>
#include <cmath>
#include <cuda_runtime.h>
#include <atomic>
#include <mutex>
#include <functional>
#include <optional>
#include <unordered_map>
#include <queue>
#include <unordered_set>
#include <algorithm>
#include <numeric>

// ---------------------------------------------------------------------------
// ParallelDims -- lightweight struct extracting DP/TP coordinates from a
//                 2D DeviceMesh. Mesh dim 0 = DP, dim 1 = TP.
// ---------------------------------------------------------------------------
struct ParallelDims {
    int dp_size      = 0;
    int tp_size      = 0;
    int dp_rank      = 0;
    int tp_rank      = 0;
    int global_rank  = 0;
    int world_size   = 0;

    std::shared_ptr<ProcessGroupNCCL> dp_pg;
    std::shared_ptr<ProcessGroupNCCL> tp_pg;

    ParallelDims() = default;

    explicit ParallelDims(DeviceMesh& mesh)
        : dp_size(mesh.shape()[0]),
          tp_size(mesh.shape()[1]),
          dp_rank(mesh.get_dim_rank(0)),
          tp_rank(mesh.get_dim_rank(1)),
          global_rank(mesh.rank()),
          world_size(mesh.world_size()),
          dp_pg(mesh.get_process_group(0)),
          tp_pg(mesh.get_process_group(1)) {}
};

// ---------------------------------------------------------------------------
// DDP2D_Options -- configuration for the 2D data-parallel wrapper.
//                  Builder pattern consistent with existing DDP_Options.
// ---------------------------------------------------------------------------
struct DDP2D_Options {
    std::shared_ptr<ProcessGroupNCCL> dp_pg;
    std::shared_ptr<ProcessGroupNCCL> tp_pg;

    int dp_world_size = 1;
    int tp_world_size = 1;
    int dp_rank       = 0;
    int tp_rank       = 0;
    int global_rank   = 0;
    int local_rank    = 0;

    bool   bucket_enabled = false;
    size_t bucket_size    = 25 * 1024 * 1024;

    bool   static_graph     = false;
    bool   grad_as_view     = false;
    bool   broadcast_buffer = false;

    size_t grad_accum_steps = 0;
    bool   is_accum_sync    = false;

    DDP2D_Options with_parallel_dims(const ParallelDims& dims) {
        DDP2D_Options o = *this;
        o.dp_pg         = dims.dp_pg;
        o.tp_pg         = dims.tp_pg;
        o.dp_world_size = dims.dp_size;
        o.tp_world_size = dims.tp_size;
        o.dp_rank       = dims.dp_rank;
        o.tp_rank       = dims.tp_rank;
        o.global_rank   = dims.global_rank;
        return o;
    }

    DDP2D_Options with_dp_process_group(std::shared_ptr<ProcessGroupNCCL> pg, int ws) {
        DDP2D_Options o = *this;
        o.dp_pg         = pg;
        o.dp_world_size = ws;
        o.dp_rank       = pg->get_rank();
        return o;
    }

    DDP2D_Options with_tp_process_group(std::shared_ptr<ProcessGroupNCCL> pg, int ws) {
        DDP2D_Options o = *this;
        o.tp_pg         = pg;
        o.tp_world_size = ws;
        o.tp_rank       = pg->get_rank();
        return o;
    }

    DDP2D_Options with_local_rank(int lr) {
        DDP2D_Options o = *this;
        o.local_rank = lr;
        return o;
    }

    DDP2D_Options with_bucket(bool enabled, size_t size = 25 * 1024 * 1024) {
        DDP2D_Options o = *this;
        o.bucket_enabled = enabled;
        o.bucket_size    = size;
        return o;
    }

    DDP2D_Options with_static_graph(bool sg) {
        DDP2D_Options o = *this;
        o.static_graph = sg;
        return o;
    }

    DDP2D_Options with_grad_view(bool gv) {
        DDP2D_Options o = *this;
        o.grad_as_view = gv;
        return o;
    }

    DDP2D_Options with_grad_accum(bool sync_flag, size_t steps) {
        DDP2D_Options o = *this;
        o.is_accum_sync    = sync_flag;
        o.grad_accum_steps = steps;
        return o;
    }
};

// ---------------------------------------------------------------------------
// Bucket2D -- gradient bucketing for 2D DDP.
//             Operates on the underlying Tensor of each DTensor parameter.
// ---------------------------------------------------------------------------
struct Bucket2D {
    OwnTensor::Tensor flatten_gradient;

    std::vector<OwnTensor::Tensor> grad_incoming;
    std::vector<OwnTensor::Tensor> grad_outgoing;

    std::vector<DTensor*>          param_dtensors;
    std::vector<OwnTensor::Tensor> param_tensors;
    std::vector<size_t>            param_indices;

    std::vector<size_t> offset;
    std::vector<size_t> length;
    std::vector<size_t> param_size;

    size_t bucket_byte_cap = 25 * 1024 * 1024;
    size_t bucket_pending  = 0;

    std::shared_ptr<Work> work_obj_;
};

// ---------------------------------------------------------------------------
// DataParallel2D -- the 2D parallelism wrapper for DModule (TP models).
//
// Responsibilities:
//   1. Classify DTensor parameters as sharded vs replicated.
//   2. Phase 1: TP gradient sync (replicated params all-reduce within TP group).
//   3. Phase 2: DDP gradient sync (all params all-reduce across DP replicas).
//   4. Broadcast parameters across DP replicas at init.
//   5. Bucket-based gradient aggregation (optional).
//   6. 2D-aware gradient norm clipping.
// ---------------------------------------------------------------------------
class DataParallel2D {
public:
    DataParallel2D() = default;

    DataParallel2D(OwnTensor::dnn::DModule* module, DDP2D_Options opts, bool init_sync = true);

    ~DataParallel2D();

    // Forward: delegates to module->forward(). Registers backward hooks.
    DTensor forward(DTensor& input);

    // Two-phase gradient synchronization (call after backward).
    void sync_gradients();

    // Phase 1: TP all-reduce for replicated parameters within TP group.
    void sync_tp_replicated_grads();

    // Phase 2: DDP all-reduce across DP replicas for all parameters.
    void sync_dp_grads();

    // Broadcast parameters from dp_rank 0 to all DP replicas.
    void broadcast_parameters();

    // Parameter access.
    std::vector<DTensor*> parameters();
    std::vector<DTensor*> sharded_parameters();
    std::vector<DTensor*> replicated_parameters();

    // No-sync mode for gradient accumulation.
    void no_sync();
    bool enable_sync();

    // 2D-aware gradient norm clipping.
    float clip_grad_norm_2d(float max_norm, float norm_type = 2.0f);

    DDP2D_Options options() const { return opts_; }

    void zero_grad();

    // Register autograd hooks for bucket-based async gradient sync.
    void register_hooks();

private:
    void categorize_parameters();

    // Bucket system
    void create_buckets();
    void create_bucket_views(Bucket2D& bucket);
    void grad_map_flatten_grad(Bucket2D& bucket);
    void mark_param_ready(size_t param_index);
    void mark_bucket_ready(size_t bucket_index);
    void all_reduce_bucket(Bucket2D& bucket);
    void finalize_backward();
    void mediator_hook(size_t variable_index);

    void search_for_unused(const std::vector<OwnTensor::Tensor>& outputs);

    void pre_forward();
    void pre_backward(const std::vector<OwnTensor::Tensor>& outputs);

    void rebuild_buckets();

    std::vector<std::vector<size_t>> decide_bucket_order(
        const std::vector<OwnTensor::Tensor>& params,
        const std::optional<std::vector<size_t>>& backward_order = std::nullopt);

private:
    DDP2D_Options opts_;
    OwnTensor::dnn::DModule* module_ = nullptr;
    bool init_sync_ = true;

    std::shared_ptr<ProcessGroupNCCL> dp_pg_;
    std::shared_ptr<ProcessGroupNCCL> tp_pg_;

    // All DTensor parameters from DModule::parameters()
    std::vector<DTensor*> all_params_;

    // Classified parameter lists
    std::vector<DTensor*> sharded_params_;
    std::vector<DTensor*> replicated_params_;

    // Per-parameter: true if replicated, false if sharded
    std::vector<bool> is_replicated_;

    // Underlying raw Tensors (extracted from DTensor for bucket system)
    std::vector<OwnTensor::Tensor> raw_tensors_;

    // Bucket state
    std::vector<Bucket2D> buckets_;
    struct ParamBucketLocator {
        size_t bucket_index;
        size_t intra_bucket_index;
    };
    std::unordered_map<size_t, ParamBucketLocator> variable_locator_;

    // Backward state
    bool    grad_sync_enabled_ = true;
    size_t  next_bucket_       = 0;
    bool    finalize_called_   = false;
    bool    first_backward_    = false;
    bool    rebuild_bucket_    = true;
    int64_t iteration_count_   = 0;

    std::vector<size_t>          unused_param_indices_;
    bool                         mark_unused_ = false;
    std::vector<OwnTensor::Tensor> rebuilt_params_order_;
    std::vector<size_t>          rebuilt_indices_order_;

    std::mutex  mutex_;
    cudaEvent_t compute_sync_event_ = nullptr;

    std::unordered_map<std::shared_ptr<OwnTensor::Node>, size_t> grad_acc_map_;
};
