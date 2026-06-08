#pragma once

#include "ddp_2d.hpp"
#include "ZeRORedundancyOptimizer.hpp"
#include "nn/optimizer/Optim.h"

#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <functional>
#include <cmath>
#include <mpi.h>

// ============================================================================
// Parallel2D_Config -- user-facing configuration for the 2D parallelism setup.
//
// Users fill this in before constructing Parallel2D. Builder pattern keeps it
// consistent with DDP_Options / DDP2D_Options / zero_ops.
// ============================================================================
struct Parallel2D_Config {
    // --- Mesh topology ---
    int dp_size = -1;           // Data-parallel groups  (mesh dim 0)
    int tp_size = -1;           // Tensor-parallel groups (mesh dim 1)

    // --- DDP options ---
    bool   bucket_enabled   = false;
    size_t bucket_size      = 25 * 1024 * 1024;    // 25 MB
    bool   static_graph     = false;
    bool   grad_as_view     = false;
    bool   broadcast_buffer = false;

    // --- Gradient accumulation ---
    size_t grad_accum_steps = 0;

    // --- Gradient clipping ---
    float  max_grad_norm    = 0.0f;    // 0 = disabled

    // --- Optimizer choice (optional, can be managed externally) ---
    enum class OptimizerType { NONE, ADAMW, ADAM, SGD, ZERO_ADAMW };
    OptimizerType optimizer_type = OptimizerType::NONE;

    float learning_rate  = 3e-4f;
    float beta1          = 0.9f;
    float beta2          = 0.95f;
    float eps            = 1e-8f;
    float weight_decay   = 0.1f;

    // --- Builders ---
    Parallel2D_Config with_mesh(int dp, int tp) {
        auto c = *this; c.dp_size = dp; c.tp_size = tp; return c;
    }
    Parallel2D_Config with_bucket(bool enabled, size_t sz = 25*1024*1024) {
        auto c = *this; c.bucket_enabled = enabled; c.bucket_size = sz; return c;
    }
    Parallel2D_Config with_static_graph(bool sg) {
        auto c = *this; c.static_graph = sg; return c;
    }
    Parallel2D_Config with_grad_view(bool gv) {
        auto c = *this; c.grad_as_view = gv; return c;
    }
    Parallel2D_Config with_grad_accum(size_t steps) {
        auto c = *this; c.grad_accum_steps = steps; return c;
    }
    Parallel2D_Config with_max_grad_norm(float norm) {
        auto c = *this; c.max_grad_norm = norm; return c;
    }
    Parallel2D_Config with_adamw(float lr, float b1 = 0.9f, float b2 = 0.95f,
                                 float e = 1e-8f, float wd = 0.1f) {
        auto c = *this;
        c.optimizer_type = OptimizerType::ADAMW;
        c.learning_rate = lr; c.beta1 = b1; c.beta2 = b2;
        c.eps = e; c.weight_decay = wd;
        return c;
    }
    Parallel2D_Config with_zero_adamw(float lr, float b1 = 0.9f, float b2 = 0.95f,
                                      float e = 1e-8f, float wd = 0.1f) {
        auto c = *this;
        c.optimizer_type = OptimizerType::ZERO_ADAMW;
        c.learning_rate = lr; c.beta1 = b1; c.beta2 = b2;
        c.eps = e; c.weight_decay = wd;
        return c;
    }
};

// ============================================================================
// Parallel2D -- high-level orchestration class for 2D (DDP x TP) parallelism.
//
// Encapsulates:
//   - 2D DeviceMesh creation and process-group extraction
//   - Model wrapping with DataParallel2D
//   - Two-phase gradient synchronization
//   - Gradient accumulation
//   - 2D-aware gradient norm clipping
//   - Optimizer creation and LR updates
//   - Diagnostic / describe utilities
//
// Usage:
//   // 1. MPI_Init, get rank/world_size
//   // 2. Build config
//   Parallel2D_Config cfg = Parallel2D_Config()
//       .with_mesh(4, 2)
//       .with_bucket(true)
//       .with_adamw(3e-4f);
//
//   // 3. Construct
//   Parallel2D engine(cfg);
//
//   // 4. Build your model using tp_pg() and tp_mesh_view()
//   GPT model(gpt_cfg, engine.mesh(), engine.tp_pg(), device);
//
//   // 5. Attach model
//   engine.attach_model(&model);
//
//   // 6. Training loop
//   for (int step = 0; step < max_steps; ++step) {
//       engine.zero_grad();
//       DTensor logits = engine.forward(input);
//       loss.backward();
//       engine.step(step);   // sync_gradients + clip + optimizer.step
//   }
// ============================================================================

class Parallel2D {
public:
    // -----------------------------------------------------------------
    // Construction / Destruction
    // -----------------------------------------------------------------

    Parallel2D() = default;

    // Main constructor: creates 2D mesh, extracts PGs, sets CUDA device.
    // MPI must already be initialized.
    explicit Parallel2D(const Parallel2D_Config& config);

    ~Parallel2D();

    // Disable copy (owns CUDA resources via mesh / DDP)
    Parallel2D(const Parallel2D&) = delete;
    Parallel2D& operator=(const Parallel2D&) = delete;
    Parallel2D(Parallel2D&&) = default;
    Parallel2D& operator=(Parallel2D&&) = default;

    // -----------------------------------------------------------------
    // Model attachment
    // -----------------------------------------------------------------

    // Attach a DModule (TP-sharded model). Creates DataParallel2D wrapper
    // and optionally the optimizer.
    // Call this AFTER the model is fully constructed with tp_pg().
    void attach_model(OwnTensor::dnn::DModule* model, bool init_sync = true);

    // -----------------------------------------------------------------
    // Training step helpers
    // -----------------------------------------------------------------

    // Full forward pass through the 2D-wrapped model.
    DTensor forward(DTensor& input);

    // Two-phase gradient synchronization.
    // Phase 1: TP all-reduce for replicated params.
    // Phase 2: DDP all-reduce across DP replicas.
    void sync_gradients();

    // 2D-aware gradient norm clipping. Returns the total norm.
    float clip_grad_norm(float max_norm);

    // Combined: sync_gradients + clip + optimizer step.
    // Handles gradient accumulation if configured.
    // Returns the gradient norm (0 if this was an accumulation micro-step).
    float step(int global_step);

    // Zero all gradients in the model.
    void zero_grad();

    // -----------------------------------------------------------------
    // Gradient accumulation
    // -----------------------------------------------------------------

    // Returns true if this micro-step should trigger a sync + optimizer step.
    bool should_sync_at_micro_step(int micro_step) const;

    // Enter/exit no-sync mode explicitly.
    void no_sync();
    void enable_sync();

    // -----------------------------------------------------------------
    // Optimizer / LR control
    // -----------------------------------------------------------------

    // Set learning rate (e.g., from a scheduler).
    void set_lr(float lr);

    // Get current learning rate.
    float get_lr() const;

    // -----------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------

    // The 2D device mesh.
    DeviceMesh&       mesh()       { return *mesh_; }
    const DeviceMesh& mesh() const { return *mesh_; }

    // Process groups.
    std::shared_ptr<ProcessGroupNCCL> tp_pg() const { return dims_.tp_pg; }
    std::shared_ptr<ProcessGroupNCCL> dp_pg() const { return dims_.dp_pg; }

    // ParallelDims (dp/tp ranks, sizes, process groups).
    const ParallelDims& dims() const { return dims_; }

    // DP-specific: rank and world_size for data sharding.
    int dp_rank()       const { return dims_.dp_rank; }
    int dp_world_size() const { return dims_.dp_size; }

    // TP-specific: rank and world_size for model sharding.
    int tp_rank()       const { return dims_.tp_rank; }
    int tp_world_size() const { return dims_.tp_size; }

    // Global rank / world_size.
    int global_rank()   const { return dims_.global_rank; }
    int world_size()    const { return dims_.world_size; }

    // CUDA device for this rank.
    OwnTensor::DeviceIndex device() const { return device_; }

    // Underlying DataParallel2D wrapper (may be null before attach_model).
    DataParallel2D*  ddp()   { return ddp_.get(); }

    // All parameters (DTensor pointers from the model).
    std::vector<DTensor*> parameters();

    // Raw tensors extracted from DTensor params (for optimizer construction).
    std::vector<OwnTensor::Tensor> tensor_parameters();

    // Configuration.
    const Parallel2D_Config& config() const { return config_; }

    // -----------------------------------------------------------------
    // Diagnostics
    // -----------------------------------------------------------------

    // Print mesh topology, group membership, and parameter summary.
    void describe(std::ostream& os = std::cout) const;

    // Return a one-line summary string.
    std::string summary() const;

    // Print on rank 0 only.
    void log(const std::string& msg) const;

private:
    // -----------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------
    void validate_config() const;
    void create_mesh();
    void set_cuda_device();
    void create_optimizer();

    // Cosine LR schedule helper (utility, not required).
    static float cosine_lr(int step, float max_lr, float min_lr,
                           int warmup_steps, int max_steps);

private:
    Parallel2D_Config config_;

    // 2D DeviceMesh: shape = [dp_size, tp_size]
    std::unique_ptr<DeviceMesh> mesh_;
    ParallelDims                dims_;

    // CUDA device for this rank
    OwnTensor::DeviceIndex device_{OwnTensor::Device::CUDA, 0};

    // Model and DDP wrapper
    OwnTensor::dnn::DModule*           model_  = nullptr;
    std::unique_ptr<DataParallel2D>    ddp_;

    // Optimizer (optionally managed by Parallel2D)
    std::unique_ptr<OwnTensor::nn::Optimizer> optimizer_;

    // Cached tensor params (extracted once at attach_model)
    std::vector<OwnTensor::Tensor> tensor_params_;

    // Gradient accumulation state
    int micro_step_counter_ = 0;

    // Initialization flag
    bool model_attached_ = false;
};
