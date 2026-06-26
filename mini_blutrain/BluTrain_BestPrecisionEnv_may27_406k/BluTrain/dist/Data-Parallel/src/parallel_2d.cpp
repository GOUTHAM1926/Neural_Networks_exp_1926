#include "parallel_2d.hpp"
#include "device/DeviceSet.h"

#include <cassert>
#include <iomanip>
#include <numeric>


// ====================================================================
// Constructor
// ====================================================================

Parallel2D::Parallel2D(const Parallel2D_Config& config)
    : config_(config)
{
    validate_config();
    create_mesh();
    set_cuda_device();
}

Parallel2D::~Parallel2D() = default;

// ====================================================================
// Configuration validation
// ====================================================================

void Parallel2D::validate_config() const {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int dp = config_.dp_size;
    int tp = config_.tp_size;

    // Auto-detect: if one dimension is set, derive the other
    if (dp > 0 && tp <= 0) {
        // tp will be set in create_mesh
    } else if (tp > 0 && dp <= 0) {
        // dp will be set in create_mesh
    } else if (dp > 0 && tp > 0) {
        if (dp * tp != world_size) {
            throw std::runtime_error(
                "Parallel2D: dp_size (" + std::to_string(dp) +
                ") * tp_size (" + std::to_string(tp) +
                ") = " + std::to_string(dp * tp) +
                " must equal world_size (" + std::to_string(world_size) + ")");
        }
    } else {
        throw std::runtime_error(
            "Parallel2D: at least one of dp_size or tp_size must be set (> 0)");
    }
}

// ====================================================================
// Mesh creation
// ====================================================================

void Parallel2D::create_mesh() {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int dp = config_.dp_size;
    int tp = config_.tp_size;

    // Auto-derive the unset dimension
    if (dp > 0 && tp <= 0) {
        tp = world_size / dp;
        if (dp * tp != world_size) {
            throw std::runtime_error(
                "Parallel2D: world_size (" + std::to_string(world_size) +
                ") is not divisible by dp_size (" + std::to_string(dp) + ")");
        }
    } else if (tp > 0 && dp <= 0) {
        dp = world_size / tp;
        if (dp * tp != world_size) {
            throw std::runtime_error(
                "Parallel2D: world_size (" + std::to_string(world_size) +
                ") is not divisible by tp_size (" + std::to_string(tp) + ")");
        }
    }

    // Store resolved sizes back into config
    config_.dp_size = dp;
    config_.tp_size = tp;

    // Device IDs: [0, 1, ..., world_size-1]
    std::vector<int> device_ids(world_size);
    std::iota(device_ids.begin(), device_ids.end(), 0);

    // Create the 2D mesh: dim 0 = DP, dim 1 = TP
    mesh_ = std::make_unique<DeviceMesh>(
        std::vector<int>{dp, tp}, device_ids);

    // Extract ParallelDims (PGs, ranks, sizes)
    dims_ = ParallelDims(*mesh_);
}

// ====================================================================
// CUDA device setup
// ====================================================================

void Parallel2D::set_cuda_device() {
    int gpus_per_node = 1;
    cudaGetDeviceCount(&gpus_per_node);

    const char* env = std::getenv("NO_GPUS_PER_NODE");
    if (env) {
        int parsed = std::atoi(env);
        if (parsed > 0) gpus_per_node = parsed;
    }

    int local_rank = dims_.global_rank % gpus_per_node;
    cudaSetDevice(local_rank);

    device_ = OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, local_rank);
}

// ====================================================================
// Model attachment
// ====================================================================

void Parallel2D::attach_model(OwnTensor::dnn::DModule* model, bool init_sync) {
    if (!model) {
        throw std::runtime_error("Parallel2D::attach_model: model cannot be null");
    }

    model_ = model;

    // Build DDP2D_Options from our config and ParallelDims
    DDP2D_Options ddp_opts;
    ddp_opts = ddp_opts.with_parallel_dims(dims_);
    ddp_opts = ddp_opts.with_local_rank(device_.index);
    ddp_opts = ddp_opts.with_bucket(config_.bucket_enabled, config_.bucket_size);
    ddp_opts = ddp_opts.with_static_graph(config_.static_graph);
    ddp_opts = ddp_opts.with_grad_view(config_.grad_as_view);

    if (config_.grad_accum_steps > 0) {
        ddp_opts = ddp_opts.with_grad_accum(true, config_.grad_accum_steps);
    }

    // Create the DataParallel2D wrapper
    ddp_ = std::make_unique<DataParallel2D>(model_, ddp_opts, init_sync);

    // Cache tensor parameters for optimizer
    tensor_params_.clear();
    auto dtensor_params = ddp_->parameters();
    tensor_params_.reserve(dtensor_params.size());
    for (auto* dt : dtensor_params) {
        tensor_params_.push_back(dt->mutable_tensor());
    }

    // Create optimizer if configured
    if (config_.optimizer_type != Parallel2D_Config::OptimizerType::NONE) {
        create_optimizer();
    }

    model_attached_ = true;
    micro_step_counter_ = 0;

    log("[Parallel2D] Model attached. Params: " +
        std::to_string(tensor_params_.size()) +
        " (sharded: " + std::to_string(ddp_->sharded_parameters().size()) +
        ", replicated: " + std::to_string(ddp_->replicated_parameters().size()) + ")");
}

// ====================================================================
// Optimizer creation
// ====================================================================

void Parallel2D::create_optimizer() {
    switch (config_.optimizer_type) {
        case Parallel2D_Config::OptimizerType::ADAMW: {
            optimizer_ = std::make_unique<OwnTensor::nn::AdamW>(
                tensor_params_,
                config_.learning_rate,
                config_.beta1,
                config_.beta2,
                config_.eps,
                config_.weight_decay);
            break;
        }
        case Parallel2D_Config::OptimizerType::ADAM: {
            optimizer_ = std::make_unique<OwnTensor::nn::Adam>(
                tensor_params_,
                config_.learning_rate,
                config_.beta1,
                config_.beta2,
                config_.eps);
            break;
        }
        case Parallel2D_Config::OptimizerType::SGD: {
            optimizer_ = std::make_unique<OwnTensor::nn::SGDOptimizer>(
                tensor_params_,
                config_.learning_rate);
            break;
        }
        case Parallel2D_Config::OptimizerType::ZERO_ADAMW: {
            // ZeRO-1 partitions optimizer state across the DP group.
            // Use the explicit-parameter constructor to avoid template
            // instantiation issues with the zero_ops overload.
            auto dp_pg = dims_.dp_pg;
            optimizer_.reset(new ZeROOptimizer<OwnTensor::nn::AdamW>(
                tensor_params_, dp_pg,
                config_.learning_rate,
                config_.beta1,
                config_.beta2,
                config_.eps,
                config_.weight_decay));
            break;
        }
        case Parallel2D_Config::OptimizerType::NONE:
            break;
    }
}

// ====================================================================
// Forward
// ====================================================================

DTensor Parallel2D::forward(DTensor& input) {
    if (!model_attached_) {
        throw std::runtime_error(
            "Parallel2D::forward: no model attached. Call attach_model() first.");
    }
    return ddp_->forward(input);
}

// ====================================================================
// Gradient synchronization
// ====================================================================

void Parallel2D::sync_gradients() {
    if (!ddp_) return;
    ddp_->sync_gradients();
}

// ====================================================================
// Gradient clipping
// ====================================================================

float Parallel2D::clip_grad_norm(float max_norm) {
    if (!ddp_ || max_norm <= 0.0f) return 0.0f;
    return ddp_->clip_grad_norm_2d(max_norm);
}

// ====================================================================
// Combined step: sync + clip + optimizer
// ====================================================================

float Parallel2D::step(int global_step) {
    if (!model_attached_) {
        throw std::runtime_error(
            "Parallel2D::step: no model attached. Call attach_model() first.");
    }

    micro_step_counter_++;

    // -- Gradient accumulation logic --
    // If grad_accum_steps > 0, we only sync + step on the last micro-step.
    if (config_.grad_accum_steps > 0) {
        bool is_sync_step = should_sync_at_micro_step(micro_step_counter_);

        if (!is_sync_step) {
            // Accumulation micro-step: skip sync and optimizer step.
            // DDP no_sync mode prevents communication.
            return 0.0f;
        }

        // Last micro-step: re-enable sync
        micro_step_counter_ = 0;
    }

    // Phase 1 + 2: TP sync + DDP sync
    sync_gradients();

    // Gradient clipping
    float grad_norm = 0.0f;
    if (config_.max_grad_norm > 0.0f) {
        grad_norm = clip_grad_norm(config_.max_grad_norm);
    }

    // Optimizer step
    if (optimizer_) {
        optimizer_->step();
    }

    return grad_norm;
}

// ====================================================================
// Zero gradients
// ====================================================================

void Parallel2D::zero_grad() {
    if (ddp_) {
        ddp_->zero_grad();
    } else if (model_) {
        model_->zero_grad();
    }
}

// ====================================================================
// Gradient accumulation helpers
// ====================================================================

bool Parallel2D::should_sync_at_micro_step(int micro_step) const {
    if (config_.grad_accum_steps == 0) return true;
    return (micro_step % static_cast<int>(config_.grad_accum_steps)) == 0;
}

void Parallel2D::no_sync() {
    if (ddp_) ddp_->no_sync();
}

void Parallel2D::enable_sync() {
    if (ddp_) ddp_->enable_sync();
}

// ====================================================================
// Optimizer / LR control
// ====================================================================

void Parallel2D::set_lr(float lr) {
    if (optimizer_) {
        optimizer_->set_lr(lr);
    }
}

float Parallel2D::get_lr() const {
    if (!optimizer_) return config_.learning_rate;

    // AdamW, Adam, SGD all store lr that can be accessed.
    // Use const_cast since the base Optimizer interface lacks a const get_lr.
    // The underlying implementations do not mutate state in get_lr-like accessors.
    return config_.learning_rate;
}

// ====================================================================
// Parameter accessors
// ====================================================================

std::vector<DTensor*> Parallel2D::parameters() {
    if (ddp_) return ddp_->parameters();
    if (model_) return model_->parameters();
    return {};
}

std::vector<OwnTensor::Tensor> Parallel2D::tensor_parameters() {
    return tensor_params_;
}

// ====================================================================
// Diagnostics
// ====================================================================

void Parallel2D::describe(std::ostream& os) const {
    if (dims_.global_rank != 0) return;

    os << "============================================================\n";
    os << "  Parallel2D Configuration\n";
    os << "============================================================\n";
    os << "  World Size     : " << dims_.world_size << "\n";
    os << "  Mesh Shape     : [" << dims_.dp_size << ", " << dims_.tp_size << "]\n";
    os << "                   dim 0 = DP (" << dims_.dp_size << " replicas)\n";
    os << "                   dim 1 = TP (" << dims_.tp_size << "-way sharding)\n";
    os << "------------------------------------------------------------\n";
    os << "  DP Groups (" << dims_.tp_size << " groups, " << dims_.dp_size << " ranks each):\n";

    // Print group membership
    for (int tp_idx = 0; tp_idx < dims_.tp_size; tp_idx++) {
        os << "    DP Group " << tp_idx << ": ranks [";
        for (int dp_idx = 0; dp_idx < dims_.dp_size; dp_idx++) {
            if (dp_idx > 0) os << ", ";
            os << mesh_->get_rank({static_cast<int64_t>(dp_idx),
                                   static_cast<int64_t>(tp_idx)});
        }
        os << "]\n";
    }

    os << "  TP Groups (" << dims_.dp_size << " groups, " << dims_.tp_size << " ranks each):\n";
    for (int dp_idx = 0; dp_idx < dims_.dp_size; dp_idx++) {
        os << "    TP Group " << dp_idx << ": ranks [";
        for (int tp_idx = 0; tp_idx < dims_.tp_size; tp_idx++) {
            if (tp_idx > 0) os << ", ";
            os << mesh_->get_rank({static_cast<int64_t>(dp_idx),
                                   static_cast<int64_t>(tp_idx)});
        }
        os << "]\n";
    }

    os << "------------------------------------------------------------\n";
    os << "  DDP Options:\n";
    os << "    Bucketing    : " << (config_.bucket_enabled ? "ON" : "OFF");
    if (config_.bucket_enabled) {
        os << " (" << (config_.bucket_size / (1024*1024)) << " MB)";
    }
    os << "\n";
    os << "    Static Graph : " << (config_.static_graph ? "ON" : "OFF") << "\n";
    os << "    Grad as View : " << (config_.grad_as_view ? "ON" : "OFF") << "\n";
    os << "    Grad Accum   : ";
    if (config_.grad_accum_steps > 0) {
        os << config_.grad_accum_steps << " micro-steps";
    } else {
        os << "OFF";
    }
    os << "\n";
    os << "    Max Grad Norm: ";
    if (config_.max_grad_norm > 0.0f) {
        os << config_.max_grad_norm;
    } else {
        os << "OFF";
    }
    os << "\n";

    os << "  Optimizer      : ";
    switch (config_.optimizer_type) {
        case Parallel2D_Config::OptimizerType::ADAMW:      os << "AdamW"; break;
        case Parallel2D_Config::OptimizerType::ADAM:        os << "Adam"; break;
        case Parallel2D_Config::OptimizerType::SGD:         os << "SGD"; break;
        case Parallel2D_Config::OptimizerType::ZERO_ADAMW:  os << "ZeRO-1 AdamW"; break;
        case Parallel2D_Config::OptimizerType::NONE:        os << "External"; break;
    }
    os << "\n";

    if (config_.optimizer_type != Parallel2D_Config::OptimizerType::NONE) {
        os << "    LR           : " << config_.learning_rate << "\n";
        os << "    Weight Decay : " << config_.weight_decay << "\n";
    }

    if (model_attached_) {
        os << "------------------------------------------------------------\n";
        os << "  Model Parameters:\n";

        int64_t total_params = 0, sharded_count = 0, replicated_count = 0;
        for (auto* dt : ddp_->parameters()) {
            total_params += dt->mutable_tensor().numel();
        }
        sharded_count = ddp_->sharded_parameters().size();
        replicated_count = ddp_->replicated_parameters().size();

        os << "    Total tensors    : " << (sharded_count + replicated_count) << "\n";
        os << "    Sharded (TP)     : " << sharded_count << "\n";
        os << "    Replicated       : " << replicated_count << "\n";
        os << "    Local elements   : " << total_params << "\n";
    }

    os << "============================================================\n";
}

std::string Parallel2D::summary() const {
    std::ostringstream oss;
    oss << "Parallel2D[" << dims_.dp_size << "DP x " << dims_.tp_size << "TP"
        << " | rank " << dims_.global_rank << "/" << dims_.world_size
        << " | dp_rank=" << dims_.dp_rank
        << " tp_rank=" << dims_.tp_rank << "]";
    return oss.str();
}

void Parallel2D::log(const std::string& msg) const {
    if (dims_.global_rank == 0) {
        std::cout << msg << std::endl;
    }
}

// ====================================================================
// Static utility: cosine LR schedule
// ====================================================================

float Parallel2D::cosine_lr(int step, float max_lr, float min_lr,
                             int warmup_steps, int max_steps) {
    if (step < warmup_steps) {
        return max_lr * static_cast<float>(step + 1) /
               static_cast<float>(warmup_steps);
    }
    if (step > max_steps) {
        return min_lr;
    }
    float decay_ratio = static_cast<float>(step - warmup_steps) /
                        static_cast<float>(max_steps - warmup_steps);
    float coeff = 0.5f * (1.0f + std::cos(M_PI * decay_ratio));
    return min_lr + coeff * (max_lr - min_lr);
}
