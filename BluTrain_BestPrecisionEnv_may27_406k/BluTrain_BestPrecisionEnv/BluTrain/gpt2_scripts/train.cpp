// =============================================================================
// train.cpp — Unified GPT-2 Training Entry Point
//
// All training parameters are driven by a config file.
// No source code changes needed to switch between:
//   - Single GPU / DDP / 2D Parallel (DDP+TP)
//   - FMHA / Normal / Fused Tril Softmax attention
//   - AdamW / Adam / SGD optimizers
//   - CUBLAS / BLU_BLAS backends
//   - Activation / Model checkpointing toggles
//
// Usage:
//   Single GPU:  make run-train CONFIG=gpt2_scripts/config/train_config.cfg
//   DDP:         make run-train-mpi CONFIG=gpt2_scripts/config/train_config.cfg NP=4
//   2D Parallel: make run-train-mpi CONFIG=gpt2_scripts/config/train_config.cfg NP=8
// =============================================================================

#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <mpi.h>

// Tensor library
#include "TensorLib.h"
#include "autograd/AutogradOps.h"
#include "autograd/operations/EmbeddingOps.h"
#include "autograd/operations/LossOps.h"
#include "autograd/operations/TrilOps.h"
#include "autograd/ops_template.h"
#include "checkpointing/Checkpointing.h"
#include "checkpointing/GradMode.h"
#include "device/CachingCudaAllocator.h"
#include "mlp/activation.h"
#include "nn/NN.h"
#include "nn/optimizer/Optim.h"

// Distributed
#include "../dist/distributed.h"

// Dataloader
#include "DataLoader.h"

// Config
#include "config/ConfigParser.h"

// Model components
#include "model/common.h"
#include "model/Embedding.h"
#include "model/Attention.h"
#include "model/MLP.h"
#include "model/GPT.h"
#include "model/GPT2D.h"

using namespace OwnTensor;

// =============================================================================
// Training loop: Single GPU
// =============================================================================
static int train_single_gpu(const config::TrainConfig& cfg) {
    std::cout << "=== GPT-2 Training (Single GPU) ===" << std::endl;
    cfg.print(0, 1);

    int gpu_device = cfg.gpu_device;
    cudaSetDevice(gpu_device);
    DeviceIndex device(Device::CUDA, gpu_device);

    int rank = 0, world_size = 1;

    // Build model
    std::cout << "Building model..." << std::endl;
    gpt2::GPT model(cfg, device, cfg.seed);

    auto params = model.parameters();
    int64_t num_params = 0;
    for (auto& p : params) num_params += p.numel();
    std::cout << "Parameters: " << num_params << std::endl;

    // Grad accumulation — same effective batch regardless of mode
    const int B = cfg.batch_size;
    const int T = cfg.token_length;
    const int GRAD_ACCUM_STEPS = cfg.grad_accum_steps(1);

    std::unique_ptr<nn::Optimizer> optimizer;
    switch (cfg.optimizer) {
        case config::OptimizerType::ADAMW:
            optimizer = std::make_unique<nn::AdamW>(
                params, cfg.max_lr, cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay);
            break;
        case config::OptimizerType::ADAM:
            optimizer = std::make_unique<nn::AdamW>(
                params, cfg.max_lr, cfg.beta1, cfg.beta2, cfg.eps, 0.0f);
            break;
        case config::OptimizerType::SGD:
            optimizer = std::make_unique<nn::SGDOptimizer>(
                params, cfg.max_lr, cfg.sgd_momentum, cfg.weight_decay);
            break;
    }

    // Data loaders
    DataLoaderLite train_loader(B, T, rank, world_size, "train",
                                cfg.data_root, true, cfg.max_seq_len_data, gpu_device);
    DataLoaderLite val_loader(B, T, rank, world_size, "val",
                              cfg.data_root, true, cfg.max_seq_len_data, gpu_device);

    // Checkpointing
    CheckpointManager ckpt_manager(cfg.checkpoint_dir, "gpt2", cfg.max_checkpoints);
    if (cfg.model_checkpointing)
        ckpt_manager.set_save_intervals(cfg.checkpoint_freq);

    int start_step = 0;
    float latest_loss = 0.0f;
    if (cfg.model_checkpointing &&
        ckpt_manager.load_latest(model, *optimizer,
                                 start_step, latest_loss)) {
        std::cout << "[Resume] step " << start_step << " loss " << latest_loss << std::endl;
        train_loader.skip_batches(static_cast<size_t>(start_step + 1) * GRAD_ACCUM_STEPS);
        start_step++;
    }

    // CSV log
    std::ofstream log_file;
    if (cfg.logging_enabled && cfg.log_csv) {
        log_file.open(cfg.log_csv_path);
        log_file << "step,loss,val_loss,lr,grad_norm,dt_ms,tok_per_sec\n";
        log_file << std::fixed << std::setprecision(6);
    }

    float val_loss_accum_log = -1.0f;

    // Pre-allocated tensors
    Tensor grad_scale = Tensor::full(Shape{{1}}, TensorOptions().with_device(device),
                                     1.0f / static_cast<float>(GRAD_ACCUM_STEPS));
    Tensor loss_accum_gpu = Tensor::zeros(Shape{{1}}, TensorOptions().with_device(device));

    bool train_loss_reached = false, val_loss_reached = false;
    auto training_start = std::chrono::high_resolution_clock::now();

    std::cout << "\nStarting training..." << std::endl;

    for (int step = start_step; step < cfg.max_steps; ++step) {
        auto t0 = std::chrono::high_resolution_clock::now();

        // --- Validation ---
        if (step % cfg.val_freq == 0 || step == cfg.max_steps - 1) {
            autograd::NoGradGuard no_grad;
            val_loader.reset();
            float val_loss_accum = 0.0f;

            for (int vs = 0; vs < cfg.val_steps; ++vs) {
                Batch batch = val_loader.next_batch();
                Tensor logits = model.forward(batch.input);
                Tensor loss = autograd::sparse_cross_entropy_loss(logits, batch.target);
                Tensor loss_cpu = loss.to_cpu();
                val_loss_accum += loss_cpu.data<float>()[0] / static_cast<float>(cfg.val_steps);
            }

            std::cout << "validation loss: " << std::fixed << std::setprecision(4)
                      << val_loss_accum << std::endl;
            val_loss_accum_log = val_loss_accum;

            if (!val_loss_reached && val_loss_accum <= cfg.target_val_loss) {
                val_loss_reached = true;
                double elapsed = std::chrono::duration<double>(
                    std::chrono::high_resolution_clock::now() - training_start).count();
                std::cout << ">>> TARGET VAL LOSS " << cfg.target_val_loss
                          << " reached at step " << step
                          << " | val_loss=" << val_loss_accum
                          << " | elapsed=" << std::fixed << std::setprecision(2) << elapsed << "s"
                          << std::endl;
            }
        }

        // --- Training step ---
        for (auto& param : params) {
            cudaMemsetAsync(param.grad(), 0.0f, param.nbytes());
        }
        if (model.train_cfg.weight_tying && model.lm_head->weight.has_grad()) {
            cudaMemsetAsync(model.lm_head->weight.grad(), 0.0f, model.lm_head->weight.nbytes());
        }

        loss_accum_gpu *= 0.0f;

        for (int micro = 0; micro < GRAD_ACCUM_STEPS; ++micro) {
            Batch batch = train_loader.next_batch();
            Tensor logits = model.forward(batch.input);
            Tensor loss = autograd::sparse_cross_entropy_loss(logits, batch.target);
            loss_accum_gpu += loss.detach();
            loss.backward(&grad_scale);
        }

        // Weight tying gradient accumulation
        if (model.train_cfg.weight_tying && model.lm_head->weight.has_grad()) {
            Tensor lm_grad_T = model.lm_head->weight.grad_view().transpose(0, 1).contiguous();
            Tensor wte_grad = model.wte.weight.grad_view();
            wte_grad += lm_grad_T;
        }

        float loss_accum;
        {
            Tensor loss_cpu = loss_accum_gpu.to_cpu();
            loss_accum = loss_cpu.data<float>()[0] / static_cast<float>(GRAD_ACCUM_STEPS);
        }

        if (std::isnan(loss_accum) || std::isinf(loss_accum)) {
            std::cerr << "ERROR: NaN/Inf loss at step " << step << std::endl;
            if (log_file.is_open()) log_file.close();
            return 1;
        }

        if (!train_loss_reached && loss_accum <= cfg.target_train_loss) {
            train_loss_reached = true;
            double elapsed = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - training_start).count();
            std::cout << ">>> TARGET TRAIN LOSS " << cfg.target_train_loss
                      << " reached at step " << step
                      << " | loss=" << std::fixed << std::setprecision(6) << loss_accum
                      << " | elapsed=" << std::fixed << std::setprecision(2) << elapsed << "s"
                      << std::endl;
        }

        float norm = nn::clip_grad_norm_(params, cfg.grad_clip_norm);

        float lr = gpt2::get_lr(step, cfg.max_lr, cfg.min_lr, cfg.warmup_steps, cfg.max_steps);
        optimizer->set_lr(lr);
        optimizer->step();

        // Checkpointing
        if (cfg.model_checkpointing && step == cfg.max_steps - 2) {
            ckpt_manager.save(step, model,
                              *optimizer, loss_accum);
        }
        if (cfg.model_checkpointing)
            ckpt_manager.step(step, model,
                              *optimizer, loss_accum);

        auto t1 = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double>(t1 - t0).count();
        int64_t tokens_processed = static_cast<int64_t>(B) * T * GRAD_ACCUM_STEPS;
        double tokens_per_sec = static_cast<double>(tokens_processed) / dt;

        std::cout << "step " << std::setw(5) << step
                  << " | loss: " << std::fixed << std::setprecision(6) << loss_accum
                  << " | lr " << std::scientific << std::setprecision(4) << lr
                  << " | norm: " << std::fixed << std::setprecision(4) << norm
                  << " | dt: " << std::fixed << std::setprecision(2) << (dt * 1000.0) << "ms"
                  << " | tok/sec: " << std::fixed << std::setprecision(2) << tokens_per_sec
                  << std::endl;

        if (cfg.logging_enabled && cfg.log_csv && log_file.is_open()) {
            log_file << step << "," << loss_accum << "," << val_loss_accum_log << ","
                     << lr << "," << norm << "," << (dt * 1000.0) << ","
                     << tokens_per_sec << "\n";
            log_file.flush();
        }
        val_loss_accum_log = -1.0f;
    }

    if (log_file.is_open()) log_file.close();
    std::cout << "\n=== Training Complete ===" << std::endl;
    return 0;
}

// =============================================================================
// Training loop: DDP (Data Parallel)
// =============================================================================
static int train_ddp(const config::TrainConfig& cfg, int rank, int world_size) {
    const bool is_master = (rank == 0);
    if (is_master) {
        std::cout << "=== GPT-2 Training (DDP, " << world_size << " GPUs) ===" << std::endl;
        cfg.print();
    }

    cudaSetDevice(rank);
    DeviceIndex device(Device::CUDA, rank);
    auto pg = init_process_group(world_size, rank);

    // Build model — same seed as single-GPU for identical weight init
    if (is_master) std::cout << "Building model..." << std::endl;
    gpt2::GPT model(cfg, device, cfg.seed);

    auto params = model.parameters();
    int64_t num_params = 0;
    for (auto& p : params) num_params += p.numel();

    // Wrap with DDP
    DDP_Options ddp_opts;
    ddp_opts = ddp_opts
        .with_process_group(pg)
        .with_world_size(world_size)
        .with_bucket_data(true, 25 * 1024 * 1024)
        .with_grad_view(false);
    DataParallel ddp_model(&model, ddp_opts, true);

    if (is_master) {
        std::cout << "Parameters: " << num_params << std::endl;
        cfg.print(0, world_size);
    }

    const int B = cfg.batch_size;
    const int T = cfg.token_length;
    // GRAD_ACCUM_STEPS shrinks with more GPUs so global_batch_size stays constant
    const int GRAD_ACCUM_STEPS = cfg.grad_accum_steps(world_size);

    // Optimizer
    std::unique_ptr<nn::Optimizer> optimizer;
    switch (cfg.optimizer) {
        case config::OptimizerType::ADAMW:
            optimizer = std::make_unique<nn::AdamW>(
                params, cfg.max_lr, cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay);
            break;
        case config::OptimizerType::ADAM:
            optimizer = std::make_unique<nn::AdamW>(
                params, cfg.max_lr, cfg.beta1, cfg.beta2, cfg.eps, 0.0f);
            break;
        case config::OptimizerType::SGD:
            optimizer = std::make_unique<nn::SGDOptimizer>(
                params, cfg.max_lr, cfg.sgd_momentum, cfg.weight_decay);
            break;
    }

    // Data loaders
    DataLoaderLite train_loader(B, T, rank, world_size, "train",
                                cfg.data_root, true, cfg.max_seq_len_data, rank);
    DataLoaderLite val_loader(B, T, rank, world_size, "val",
                              cfg.data_root, true, cfg.max_seq_len_data, rank);

    // Checkpointing
    CheckpointManager ckpt_manager(cfg.checkpoint_dir, "gpt2", cfg.max_checkpoints);
    if (cfg.model_checkpointing && is_master)
        ckpt_manager.set_save_intervals(cfg.checkpoint_freq);

    int start_step = 0;
    float latest_loss = 0.0f;
    if (cfg.model_checkpointing &&
        ckpt_manager.load_latest(model, *optimizer,
                                 start_step, latest_loss)) {
        if (is_master)
            std::cout << "[Resume] step " << start_step << " loss " << latest_loss << std::endl;
        train_loader.skip_batches(static_cast<size_t>(start_step + 1) * GRAD_ACCUM_STEPS);
        start_step++;
    }

    // CSV log
    std::ofstream log_file;
    if (is_master && cfg.logging_enabled && cfg.log_csv) {
        log_file.open(cfg.log_csv_path);
        log_file << "step,loss,val_loss,lr,grad_norm,dt_ms,tok_per_sec\n";
        log_file << std::fixed << std::setprecision(6);
    }

    float val_loss_accum_log = -1.0f;

    Tensor grad_scale = Tensor::full(Shape{{1}}, TensorOptions().with_device(device),
                                     1.0f / static_cast<float>(GRAD_ACCUM_STEPS * world_size));
    Tensor loss_accum_gpu = Tensor::zeros(Shape{{1}}, TensorOptions().with_device(device));

    bool train_loss_reached = false, val_loss_reached = false;
    auto training_start = std::chrono::high_resolution_clock::now();

    if (is_master) std::cout << "\nStarting training..." << std::endl;

    for (int step = start_step; step < cfg.max_steps; ++step) {
        auto t0 = std::chrono::high_resolution_clock::now();

        // --- Validation ---
        if (step % cfg.val_freq == 0 || step == cfg.max_steps - 1) {
            autograd::NoGradGuard no_grad;
            val_loader.reset();
            float val_loss_accum = 0.0f;

            for (int vs = 0; vs < cfg.val_steps; ++vs) {
                Batch batch = val_loader.next_batch();
                Tensor logits = model.forward(batch.input);
                Tensor loss = autograd::sparse_cross_entropy_loss(logits, batch.target);
                Tensor loss_cpu = loss.to_cpu();
                val_loss_accum += loss_cpu.data<float>()[0] / static_cast<float>(cfg.val_steps);
            }

            float global_val_loss = 0.0f;
            MPI_Allreduce(&val_loss_accum, &global_val_loss, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            global_val_loss /= world_size;

            if (is_master) {
                std::cout << "validation loss: " << std::fixed << std::setprecision(4)
                          << global_val_loss << std::endl;
                val_loss_accum_log = global_val_loss;

                if (!val_loss_reached && global_val_loss <= cfg.target_val_loss) {
                    val_loss_reached = true;
                    double elapsed = std::chrono::duration<double>(
                        std::chrono::high_resolution_clock::now() - training_start).count();
                    std::cout << ">>> TARGET VAL LOSS " << cfg.target_val_loss
                              << " reached at step " << step
                              << " | val_loss=" << global_val_loss
                              << " | elapsed=" << std::fixed << std::setprecision(2) << elapsed << "s"
                              << std::endl;
                }
            }
        }

        // --- Training step ---
        for (auto& param : params) {
            cudaMemsetAsync(param.grad(), 0.0f, param.nbytes());
        }
        if (model.train_cfg.weight_tying && model.lm_head->weight.has_grad()) {
            cudaMemsetAsync(model.lm_head->weight.grad(), 0.0f, model.lm_head->weight.nbytes());
        }

        loss_accum_gpu *= 0.0f;
        ddp_model.no_sync();

        for (int micro = 0; micro < GRAD_ACCUM_STEPS; ++micro) {
            Batch batch = train_loader.next_batch();
            Tensor logits = ddp_model.forward(batch.input);
            Tensor loss = autograd::sparse_cross_entropy_loss(logits, batch.target);
            loss_accum_gpu += loss.detach();
            if (micro == GRAD_ACCUM_STEPS - 1) ddp_model.sync();
            loss.backward(&grad_scale);
        }

        // Weight tying gradient accumulation
        if (model.train_cfg.weight_tying && model.lm_head->weight.has_grad()) {
            Tensor lm_grad_T = model.lm_head->weight.grad_view().transpose(0, 1).contiguous();
            Tensor wte_grad = model.wte.weight.grad_view();
            wte_grad += lm_grad_T;
        }

        float loss_accum;
        {
            Tensor loss_cpu = loss_accum_gpu.to_cpu();
            float local_loss = loss_cpu.data<float>()[0] / static_cast<float>(GRAD_ACCUM_STEPS);
            MPI_Allreduce(&local_loss, &loss_accum, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            loss_accum /= world_size;
        }

        if (std::isnan(loss_accum) || std::isinf(loss_accum)) {
            std::cerr << "ERROR: NaN/Inf loss at step " << step << std::endl;
            if (log_file.is_open()) log_file.close();
            MPI_Finalize();
            return 1;
        }

        if (!train_loss_reached && loss_accum <= cfg.target_train_loss && is_master) {
            train_loss_reached = true;
            double elapsed = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - training_start).count();
            std::cout << ">>> TARGET TRAIN LOSS " << cfg.target_train_loss
                      << " reached at step " << step
                      << " | loss=" << std::fixed << std::setprecision(6) << loss_accum
                      << " | elapsed=" << std::fixed << std::setprecision(2) << elapsed << "s"
                      << std::endl;
        }

        float norm = nn::clip_grad_norm_(params, cfg.grad_clip_norm);

        float lr = gpt2::get_lr(step, cfg.max_lr, cfg.min_lr, cfg.warmup_steps, cfg.max_steps);
        optimizer->set_lr(lr);
        optimizer->step();

        // Checkpointing
        if (cfg.model_checkpointing && step == cfg.max_steps - 2) {
            ckpt_manager.save(step, model,
                              *optimizer, loss_accum);
        }
        if (cfg.model_checkpointing)
            ckpt_manager.step(step, model,
                              *optimizer, loss_accum);

        auto t1 = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double>(t1 - t0).count();
        int64_t tokens_processed = static_cast<int64_t>(B) * T * GRAD_ACCUM_STEPS * world_size;
        double tokens_per_sec = static_cast<double>(tokens_processed) / dt;

        if (is_master) {
            std::cout << "step " << std::setw(5) << step
                      << " | loss: " << std::fixed << std::setprecision(6) << loss_accum
                      << " | lr " << std::scientific << std::setprecision(4) << lr
                      << " | norm: " << std::fixed << std::setprecision(4) << norm
                      << " | dt: " << std::fixed << std::setprecision(2) << (dt * 1000.0) << "ms"
                      << " | tok/sec: " << std::fixed << std::setprecision(2) << tokens_per_sec
                      << std::endl;

            if (cfg.logging_enabled && cfg.log_csv && log_file.is_open()) {
                log_file << step << "," << loss_accum << "," << val_loss_accum_log << ","
                         << lr << "," << norm << "," << (dt * 1000.0) << ","
                         << tokens_per_sec << "\n";
                log_file.flush();
            }
            val_loss_accum_log = -1.0f;
        }
    }

    if (log_file.is_open()) log_file.close();
    if (is_master) std::cout << "\n=== Training Complete ===" << std::endl;
    return 0;
}

// =============================================================================
// Training loop: 2D Parallel (DDP + TP)
// =============================================================================
static int train_2d_parallel(const config::TrainConfig& cfg, int global_rank, int world_size) {
    const bool is_master = (global_rank == 0);

    int tp_size = cfg.tp_size;
    int dp_size = world_size / tp_size;

    if (dp_size * tp_size != world_size) {
        if (is_master)
            std::cerr << "ERROR: world_size=" << world_size
                      << " not divisible by tp_size=" << tp_size << std::endl;
        return 1;
    }

    if (cfg.n_heads % tp_size != 0) {
        if (is_master)
            std::cerr << "ERROR: n_heads=" << cfg.n_heads
                      << " not divisible by tp_size=" << tp_size << std::endl;
        return 1;
    }

    int tp_rank = global_rank % tp_size;
    int dp_rank = global_rank / tp_size;

    cudaSetDevice(global_rank);
    DeviceIndex device(Device::CUDA, global_rank);

    // 2D DeviceMesh
    std::vector<int> device_ids(world_size);
    std::iota(device_ids.begin(), device_ids.end(), 0);
    DeviceMesh mesh({dp_size, tp_size}, device_ids);

    auto dp_pg = mesh.get_process_group(0);
    auto tp_pg = mesh.get_process_group(1);

    if (is_master) {
        std::cout << "=== GPT-2 Training (2D Parallel: DDP + TP) ===" << std::endl;
        std::cout << "  world_size=" << world_size
                  << "  dp_size=" << dp_size << "  tp_size=" << tp_size << std::endl;
        cfg.print(0, dp_size);
    }
    mesh.describe();

    // Build model — same seed as single-GPU for identical weight init
    if (is_master) std::cout << "Building model..." << std::endl;
    gpt2::GPT2D model(cfg, mesh, tp_pg, device, cfg.seed);

    auto params = model.collect_params();
    int64_t num_params_local = 0;
    for (auto& p : params) num_params_local += p.numel();
    if (is_master) std::cout << "Parameters per GPU: " << num_params_local << std::endl;

    const int B = cfg.batch_size;
    const int T = cfg.token_length;
    // DP replicas = dp_size; GRAD_ACCUM shrinks so global_batch_size stays constant
    const int GRAD_ACCUM_STEPS = cfg.grad_accum_steps(dp_size);

    // Optimizer
    std::unique_ptr<nn::Optimizer> optimizer;
    switch (cfg.optimizer) {
        case config::OptimizerType::ADAMW:
            optimizer = std::make_unique<nn::AdamW>(
                params, cfg.max_lr, cfg.beta1, cfg.beta2, cfg.eps, cfg.weight_decay);
            break;
        case config::OptimizerType::ADAM:
            optimizer = std::make_unique<nn::AdamW>(
                params, cfg.max_lr, cfg.beta1, cfg.beta2, cfg.eps, 0.0f);
            break;
        case config::OptimizerType::SGD:
            optimizer = std::make_unique<nn::SGDOptimizer>(
                params, cfg.max_lr, cfg.sgd_momentum, cfg.weight_decay);
            break;
    }

    // DP bucket-based gradient reducer
    DPBucketReducer dp_reducer(params, dp_pg, dp_size);

    // Data loaders (sharded by DP rank)
    DataLoaderLite train_loader(B, T, dp_rank, dp_size, "train",
                                cfg.data_root, is_master, cfg.max_seq_len_data, global_rank);
    DataLoaderLite val_loader(B, T, dp_rank, dp_size, "val",
                              cfg.data_root, is_master, cfg.max_seq_len_data, global_rank);

    // Checkpointing
    CheckpointManager ckpt_manager(cfg.checkpoint_dir, "gpt2_2d",
                                   cfg.max_checkpoints, global_rank, true);
    if (cfg.model_checkpointing)
        ckpt_manager.set_save_intervals(cfg.checkpoint_freq);

    // CSV log
    std::ofstream log_file;
    if (is_master && cfg.logging_enabled && cfg.log_csv) {
        log_file.open(cfg.log_csv_path);
        log_file << "step,loss,val_loss,lr,grad_norm,dt_ms,tok_per_sec\n";
        log_file << std::fixed << std::setprecision(6);
    }

    Tensor grad_scale = Tensor::full(Shape{{1}}, TensorOptions().with_device(device),
                                     1.0f / static_cast<float>(GRAD_ACCUM_STEPS));
    Tensor loss_accum_gpu = Tensor::zeros(Shape{{1}}, TensorOptions().with_device(device));

    float val_loss_accum_log = -1.0f;
    bool train_loss_reached = false, val_loss_reached = false;
    auto training_start = std::chrono::high_resolution_clock::now();

    if (is_master) std::cout << "\nStarting training..." << std::endl;

    for (int step = 0; step < cfg.max_steps; ++step) {
        auto t0 = std::chrono::high_resolution_clock::now();

        // --- Validation ---
        if (step % cfg.val_freq == 0 || step == cfg.max_steps - 1) {
            autograd::NoGradGuard no_grad;
            val_loader.reset();
            float val_loss_accum = 0.0f;

            for (int vs = 0; vs < cfg.val_steps; ++vs) {
                Batch batch = val_loader.next_batch();
                DTensor logits = model.forward(batch.input);
                Tensor loss = autograd::vocab_parallel_cross_entropy_v2(
                    logits, batch.target);
                Tensor loss_cpu = loss.to_cpu();
                val_loss_accum += loss_cpu.data<float>()[0] / static_cast<float>(cfg.val_steps);
            }

            // Val loss: TP ranks see identical data (already reduced by vocab_parallel_cross_entropy).
            // Average only across DP replicas — do NOT divide by tp_size again.
            float global_val_loss = 0.0f;
            MPI_Allreduce(&val_loss_accum, &global_val_loss, 1,
                          MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            // world_size = dp_size * tp_size; each TP group contributes tp_size identical copies
            // so SUM / world_size == SUM_per_dp_rank / dp_size ✓
            global_val_loss /= static_cast<float>(world_size);

            if (is_master) {
                std::cout << "validation loss: " << std::fixed << std::setprecision(4)
                          << global_val_loss << std::endl;
                val_loss_accum_log = global_val_loss;

                if (!val_loss_reached && global_val_loss <= cfg.target_val_loss) {
                    val_loss_reached = true;
                    double elapsed = std::chrono::duration<double>(
                        std::chrono::high_resolution_clock::now() - training_start).count();
                    std::cout << ">>> TARGET VAL LOSS " << cfg.target_val_loss
                              << " reached at step " << step
                              << " | val_loss=" << global_val_loss
                              << " | elapsed=" << std::fixed << std::setprecision(2) << elapsed << "s"
                              << std::endl;
                }
            }
        }

        // --- Training step ---
        model.zero_grad();
        loss_accum_gpu *= 0.0f;

        for (int micro = 0; micro < GRAD_ACCUM_STEPS; ++micro) {
            if (micro == GRAD_ACCUM_STEPS - 1) {
                dp_reducer.prepare_backward();
            }

            Batch batch = train_loader.next_batch();
            DTensor logits = model.forward(batch.input);
            Tensor loss = autograd::vocab_parallel_cross_entropy_v2(
                logits, batch.target);
            loss_accum_gpu += loss.detach();
            loss.backward(&grad_scale);
        }

        dp_reducer.finalize_backward();

        float loss_accum;
        {
            Tensor loss_cpu = loss_accum_gpu.to_cpu();
            loss_accum = loss_cpu.data<float>()[0] / static_cast<float>(GRAD_ACCUM_STEPS);
        }

        if (std::isnan(loss_accum) || std::isinf(loss_accum)) {
            std::cerr << "[Rank " << global_rank
                      << "] ERROR: NaN/Inf loss at step " << step << std::endl;
            MPI_Finalize();
            return 1;
        }

        float norm = nn::clip_grad_norm_(params, cfg.grad_clip_norm);

        float lr = gpt2::get_lr(step, cfg.max_lr, cfg.min_lr, cfg.warmup_steps, cfg.max_steps);
        optimizer->set_lr(lr);
        optimizer->step();

        auto t1 = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double>(t1 - t0).count();
        int64_t tokens_processed = static_cast<int64_t>(B) * T * GRAD_ACCUM_STEPS * dp_size;
        double tokens_per_sec = static_cast<double>(tokens_processed) / dt;

        // Average train loss across all ranks
        float global_train_loss = 0.0f;
        MPI_Allreduce(&loss_accum, &global_train_loss, 1,
                      MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        global_train_loss /= static_cast<float>(world_size);

        if (is_master) {
            std::cout << "step " << std::setw(5) << step
                      << " | loss: " << std::fixed << std::setprecision(6) << global_train_loss
                      << " | lr " << std::scientific << std::setprecision(4) << lr
                      << " | norm: " << std::fixed << std::setprecision(4) << norm
                      << " | dt: " << std::fixed << std::setprecision(2) << (dt * 1000.0) << "ms"
                      << " | tok/sec: " << std::fixed << std::setprecision(0) << tokens_per_sec
                      << std::endl;

            if (cfg.logging_enabled && cfg.log_csv && log_file.is_open()) {
                log_file << step << "," << global_train_loss << "," << val_loss_accum_log
                         << "," << lr << "," << norm << "," << (dt * 1000.0)
                         << "," << tokens_per_sec << "\n";
                log_file.flush();
            }
            val_loss_accum_log = -1.0f;

            if (!train_loss_reached && global_train_loss <= cfg.target_train_loss) {
                train_loss_reached = true;
                double elapsed = std::chrono::duration<double>(
                    std::chrono::high_resolution_clock::now() - training_start).count();
                std::cout << ">>> TARGET TRAIN LOSS " << cfg.target_train_loss
                          << " reached at step " << step
                          << " | loss=" << global_train_loss
                          << " | elapsed=" << std::fixed << std::setprecision(2) << elapsed << "s"
                          << std::endl;
            }
        }
    }

    if (log_file.is_open()) log_file.close();
    if (is_master) std::cout << "\n=== Training Complete ===" << std::endl;
    return 0;
}

// =============================================================================
// Main: parse config and dispatch to the right training loop
// =============================================================================
int main(int argc, char** argv) {
    // Parse config file path from command line
    std::string config_path = "gpt2_scripts/config/train_config.cfg";
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg.substr(0, 9) == "--config=") {
            config_path = arg.substr(9);
        }
    }

    // Parse configuration
    config::TrainConfig cfg;
    try {
        cfg = config::ConfigParser::parse(config_path);
    } catch (const std::exception& e) {
        std::cerr << "Config error: " << e.what() << std::endl;
        return 1;
    }

    // Set BLAS backend environment variable before any CUDA calls.
    // The runtime dispatchers in GenMatmul.cu / MatmulBackward.cu read
    // USE_BLU_BLAS on every matmul call to pick cuBLAS vs BluBridge-BLAS.
    const bool use_blublas = (cfg.blas_backend == config::BlasBackend::BLU_BLAS);
    setenv("USE_BLU_BLAS", use_blublas ? "1" : "0", 1);
    std::cout << "[BLAS] Requested backend: "
              << (use_blublas ? "BluBridge-BLAS" : "cuBLAS")
              << "  (USE_BLU_BLAS=" << (use_blublas ? "1" : "0") << ")"
#ifndef WITH_MYBLAS
              << "  [note: libtensor built without WITH_BLUBLAS — cuBLAS will be used regardless]"
#endif
              << std::endl;

    // Dispatch based on parallelism mode
    try {
        if (cfg.is_single_gpu()) {
            // Single GPU — MPI_Init is harmless if launched via mpirun
            MPI_Init(&argc, &argv);
            int ret = train_single_gpu(cfg);
            MPI_Finalize();
            return ret;

        } else if (cfg.has_ddp() && !cfg.has_tp()) {
            // DDP only
            MPI_Init(&argc, &argv);
            int rank, world_size;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            MPI_Comm_size(MPI_COMM_WORLD, &world_size);
            int ret = train_ddp(cfg, rank, world_size);
            MPI_Finalize();
            return ret;

        } else if (cfg.is_2d_parallel() || (cfg.has_tp() && !cfg.has_ddp())) {
            // 2D Parallel (DDP + TP), or TP-only (treated as dp_size=1)
            MPI_Init(&argc, &argv);
            int rank, world_size;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            MPI_Comm_size(MPI_COMM_WORLD, &world_size);
            int ret = train_2d_parallel(cfg, rank, world_size);
            MPI_Finalize();
            return ret;

        } else {
            std::cerr << "ERROR: Unsupported parallelism configuration" << std::endl;
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        MPI_Finalize();
        return 1;
    }
}
