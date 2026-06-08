/**
 * @file gpt2_tp_mlp_only_test.cpp
 * @brief GPT-2 training script with MLP-Only Tensor Parallelism
 * 
 * Architecture:
 * - GPU0: Embeddings (Token + Pos)
 * - Broadcast: Embedding output -> All GPUs
 * - All GPUs: MLP Blocks (Tensor Parallel)
 * - All-Reduce: MLP output is synchronized (summed) across GPUs
 * - GPU0: Final LayerNorm -> LM Head -> Loss
 * 
 * Gradient Flow:
 * - Loss (GPU0) -> Head -> LN -> MLP Output (GPU0)
 * - MLP Output Gradients:
 *   - GPU0: Valid gradient from LN/Head
 *   - GPU>0: Zero gradient (dummy)
 * - Distributed Backward:
 *   - All ranks run MLP.backward()
 *   - All-Reduce (Sum) happens internally for gradients
 *   - Resulting gradient at MLP Input is valid on all ranks
 * - Broadcast Backward (implicit aggregation):
 *   - We need to sum the gradients of the broadcasted input to get the proper gradient for Embeddings.
 *   - Since `x_dist` is replicated, `d_x_local = sum(d_x_dist_i)` over all ranks.
 *   - We perform an All-Reduce on `x_dist.grad` before feeding to Embeddings.
 */

#include <cstdint>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <filesystem> 

// Tensor library includes
#include "TensorLib.h"
#include "autograd/AutogradOps.h"
#include "autograd/operations/LossOps.h"
#include "nn/optimizer/Optim.h"
#include "mlp/activation.h"
#include "autograd/operations/EmbeddingOps.h"
#include "nn/NN.h"
#include "dnn/DistributedNN.h"


// Dataloader
#include "Data_Loader/dl_test.cpp"

using namespace OwnTensor;

// =============================================================================
// Configuration
// =============================================================================

int rank, world_size;

struct CudaTimer {
    cudaEvent_t start, stop;
    CudaTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    ~CudaTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    void start_timer() { cudaEventRecord(start); }
    float get_elapsed_ms() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
    double get_elapsed_seconds() {
        return get_elapsed_ms() / 1000.0;
    }
};

struct GPTConfig {
    int64_t batch_size = 8;
    int64_t context_length = 1024;
    int64_t vocab_size = 50304;  // GPT-2 vocab size
    int64_t n_embd = 384;
    int64_t n_layers = 3;
};

// =============================================================================
// Local Embedding Layer (GPU0 Only)
// =============================================================================

class Embedding : public nn::Module {
public:
    Tensor weight;  // [vocab_size, n_embd]
    Embedding() = default;
    Embedding(int64_t vocab_size, int64_t embed_dim, DeviceIndex device, uint64_t seed = 1234)
        : vocab_size_(vocab_size), embed_dim_(embed_dim)
    {
        // Initialize weight with small normal distribution
        TensorOptions opts = TensorOptions().with_dtype(Dtype::Float32)
                                          .with_device(device)
                                          .with_req_grad(true);
        weight = Tensor::randn<float>(Shape{{vocab_size, embed_dim}}, opts, seed, 0.02f);
        
        register_parameter(weight);
    }
    
    Tensor forward(const Tensor& indices) override {
        return autograd::embedding(weight, indices);
    }

    void clear_own_params() { params_.clear(); }
    
private:
    int64_t vocab_size_;
    int64_t embed_dim_;
};

// =============================================================================
// MLP Block (Distributed - Tensor Parallel)
// =============================================================================

class MLP : public dnn::DModule {
public:
    nn::LayerNorm ln;             // Pre-Norm (Replicated/Local per shard)
                                  // Wait, if input is replicated, LN is local.
                                  // Since we broadcast input to all ranks, LN can be local on each rank.
                                  
    dnn::DColumnLinear fc_up;     // Linear(n_embd, 4*n_embd) -> Splits output
    dnn::DRowLinear fc_down;      // Linear(4*n_embd, n_embd) -> Sums output (AllReduce)
    dnn::DGeLU gelu;
    
    MLP(GPTConfig config, DeviceMesh& mesh, std::shared_ptr<ProcessGroupNCCL>& pg,  DeviceIndex device, uint64_t seed = 1234)
        : ln(config.n_embd),
          fc_up(mesh, pg, config.batch_size, config.context_length, config.n_embd, 4 * config.n_embd,{}, false, 0.02f, seed),
          fc_down(mesh, pg, config.batch_size, config.context_length,  4 * config.n_embd, config.n_embd, {}, false, 0.02f * (1.0f / std::sqrt(2.0f * static_cast<float>(config.n_layers))), seed)

    {
        // Move LayerNorm to device
        ln.to(device);
        
        register_module(ln);
        register_module(fc_up);
        register_module(fc_down);
    }

    using dnn::DModule::register_module;
    
    void register_module(nn::LayerNorm& m) {
        register_parameter(&m.weight);
        if(m.bias.is_valid()) register_parameter(&m.bias);
    }
    
    // Forward: x [B, T, C] (Replicated) -> [B, T, C] (Replicated)
    DTensor forward( DTensor& x) {
        // Pre-Norm: ln(x)
        // Since x is replicated (same on all ranks), we can run LN locally on each rank
        DTensor h;
        h.mutable_tensor() = ln.forward(x.mutable_tensor()); // Local op
        // Note: x.mutable_tensor() is the local tensor on this rank

        // Up projection + GELU + Down projection
        h = fc_up.forward(h);    // Output is Sharded
        h = gelu.forward(h);     // Sharded
        h = fc_down.forward(h);  // Output is Replicated (Internal All-Reduced)
        
        // Residual connection: x + MLP(x)
        h.mutable_tensor() = autograd::add(x.mutable_tensor(), h.mutable_tensor());
        return h;
    }
    
private:
    int64_t n_embd_;
};

// =============================================================================
// GPT Model (Hybrid: Local GPU0 + Distributed MLP)
// =============================================================================

class GPT : public dnn::DModule {
public:
    GPTConfig config;
    DeviceMesh &mesh;
    
    // GPU0 Only Layers
    Embedding wte;  
    Embedding wpe;  
    nn::LayerNorm ln_f; 
    nn::Linear lm_head;  
    
    // Distributed Layers (All Ranks)
    dnn::DSequential mlps;

    // Tensors
    Tensor pos;
    Tensor logits;
    DTensor x_dist_input;  // Broadcasted input to MLP
    DTensor x_dist_output; // Output of MLP

    double t_tok_emb = 0, t_pos_emb = 0, t_mlp = 0, t_ln_f = 0, t_lm_head = 0;
    CudaTimer timer_tok_emb, timer_pos_emb, timer_mlp, timer_ln_f, timer_lm_head;
    
    // Helper to identify if we are Rank 0
    bool is_rank0() const { return mesh.get_rank() == 0; }

    GPT(GPTConfig cfg, DeviceMesh& mesh, std::shared_ptr<ProcessGroupNCCL>& pg, DeviceIndex device, uint64_t seed = 1234)
        : config(cfg), 
          mesh(mesh),
          // Initialize local layers only if Rank 0 (or strictly speaking, meaningful only on Rank 0)
          // We initialize them with default constructors first, then assign if Rank 0
          ln_f(cfg.n_embd),
          lm_head(cfg.n_embd, cfg.vocab_size, false)
    {
        if (is_rank0()) {
            wte = Embedding(cfg.vocab_size, cfg.n_embd, device, seed);
            wpe = Embedding(cfg.context_length, cfg.n_embd, device);
            ln_f.to(device);
            lm_head.to(device);
            
            wte.weight = lm_head.weight.t();
            wte.clear_own_params(); 
            
            // Register local params only on Rank 0
            // Note: DModule::register_parameter creates a DTensor wrapper.
            // Current DModule stores `vector<DTensor*> params_`.
            // If we register on Rank 0, these params exist in the list.
            // On Rank > 0, we don't register them, so parameters() list is shorter.
            // This is OK as long as optimizer handles it (it iterates simple vector).
            
            register_module(wpe); // Embedding (Local)
            // register_module(wte); // Shared with head
        }

        // Create MLP blocks (All Ranks)
        for (int i = 0; i < cfg.n_layers; ++i) {
            mlps.add(std::make_shared<MLP>(config, mesh, pg, device, 1234 + i));
        }
        register_module(mlps); // Distributed

        if (is_rank0()) {
            register_module(ln_f);  // Local
            register_module(lm_head); // Local
        }

        // Initialize x_dist_input with proper layout for broadcasting
        Layout replicated_layout(mesh, {config.batch_size, config.context_length, config.n_embd});
        x_dist_input = DTensor(mesh, pg, replicated_layout, "x_broadcasted");
        // We set requires_grad=true later during forward
    }

    using dnn::DModule::register_module;

    void register_module(Embedding& m) {
        if (is_rank0()) register_parameter(&m.weight);
    }

    void register_module(nn::LayerNorm& m) {
        if (is_rank0()) {
            register_parameter(&m.weight);
            if(m.bias.is_valid()) register_parameter(&m.bias);
        }
    }

    void register_module(nn::Linear& m) {
        if (is_rank0()) {
            register_parameter(&m.weight);
            if(m.bias.is_valid()) register_parameter(&m.bias);
        }
    }
    
    void reset_timing() {
        t_tok_emb = t_pos_emb = t_mlp = t_ln_f = t_lm_head = 0;
    }

    void print_timing(int rank) {
        if (rank == 0) {
            std::cout << "  [LAYER] tok_emb: " << std::fixed << std::setprecision(1) << (t_tok_emb * 1000.0) << "ms"
                      << " | pos_emb: " << (t_pos_emb * 1000.0) << "ms"
                      << " | mlps: " << (t_mlp * 1000.0) << "ms"
                      << " | ln_f: " << (t_ln_f * 1000.0) << "ms"
                      << " | lm_head: " << (t_lm_head * 1000.0) << "ms"
                      << std::endl;
        }
    }

    Tensor x_local_cache; // To keep embedding graph alive

    // Forward: indices [B, T] -> logits [B, T, vocab_size]
    // Returns logits on Rank 0, empty/dummy on Rank > 0
    Tensor forward(Tensor& idx) { // input idx is only valid on Rank 0
        
        Tensor x_local;
        
        // --- 1. Embeddings (GPU0 Only) ---
        if (is_rank0()) {
            timer_tok_emb.start_timer();
            
            auto shape = idx.shape().dims;
            int64_t B = shape[0];
            int64_t T = shape[1];
            
            // Pos indices
            pos = Tensor(Shape{{1, T}}, TensorOptions().with_dtype(Dtype::Int64).with_device(idx.device()));
            {
                Tensor pos_cpu(Shape{{1, T}}, TensorOptions().with_dtype(Dtype::Int64));
                std::vector<int64_t> pos_data(T);
                for (int64_t i = 0; i < T; ++i) pos_data[i] = i;
                pos_cpu.set_data(pos_data); 
                pos = pos_cpu.to(idx.device());
            }

            Tensor tok_emb = wte.forward(idx);  // [B, T, C]
            t_tok_emb += timer_tok_emb.get_elapsed_seconds();

            timer_pos_emb.start_timer();
            Tensor pos_emb = wpe.forward(pos);  // [1, T, C]
            t_pos_emb += timer_pos_emb.get_elapsed_seconds();

            x_local = autograd::add(tok_emb, pos_emb);
            
            // Store x_local to keep the graph alive for backward bridging
            x_local_cache = x_local;
        }
        
        // --- 2. Broadcast to all Ranks ---
        // Prepare x_dist_input storage
        // Need to ensure x_dist_input has correct shape if not initialized
        // Note: DTensor shape matches layout
        
        if (is_rank0()) {
             // Copy data to DTensor's local storage
             x_dist_input.mutable_tensor() = x_local;
        } else {
             // For other ranks, ensure mutable_tensor is allocated with correct shape/device
             // We can use a dummy allocation, replicate will overwrite
             // But size must match for buffer allocation inside broadcast
             // Layout is [B, T, C], already set in constructor
             // Make sure tensor_ is allocated
              if (x_dist_input.mutable_tensor().numel() == 0) {
                   // Allocate based on layout
                   std::vector<int64_t> global_shape = x_dist_input.get_layout().get_global_shape();
                   Shape shape_obj; shape_obj.dims.assign(global_shape.begin(), global_shape.end());
                    x_dist_input.mutable_tensor() = Tensor(shape_obj, TensorOptions().with_device(mesh.get_device(rank)).with_dtype(Dtype::Float32));
              }
        }

        // Broadcast from Root 0 to all
        // This makes x_dist_input.tensor_ valid on all ranks
        x_dist_input.replicate(0); 

        // CRITICAL: We need gradients to flow back to x_dist_input
        // replicate() does NOT set grad_fn or track history in current impl.
        // We must enable grad tracking manually on the RESULT of replicate.
        x_dist_input.mutable_tensor().set_requires_grad(true);
        x_dist_input.mutable_tensor().zero_grad(); // Clear old grads
        
        // --- 3. MLP Blocks (Distributed) ---
        timer_mlp.start_timer();
        x_dist_output = mlps.forward(x_dist_input); 
        // x_dist_output is Replicated (All-Reduced) on all ranks
        t_mlp += timer_mlp.get_elapsed_seconds();
        
        // --- 4. Final Layers (GPU0 Only) ---
        if (is_rank0()) {
            timer_ln_f.start_timer();
            Tensor h = x_dist_output.mutable_tensor(); // Take the local copy (valid on GPU0)
            
            // Important: We detach here implicitly if we continue with local ops?
            // No, we want to backprop through it.
            // h has a grad_fn (AllReduceSumBackward from the last MLP).
            // So executing h.backward() is valid.
            
            h = ln_f.forward(h);
            t_ln_f += timer_ln_f.get_elapsed_seconds();

            timer_lm_head.start_timer();
            logits = autograd::matmul(h, wte.weight.t()); // Tied weights
            t_lm_head += timer_lm_head.get_elapsed_seconds();
            
            return logits;
        } 
        
        return Tensor(); // Empty on other ranks
    }
    
    // Helper to backprop through embeddings using the reduced gradient
    void backward_embeddings(Tensor& grad_input) {
         if (is_rank0()) {
             // We need the graph from forward to be alive.
             // If we didn't save the graph root (x_local), the graph might be freed?
             // Tensor reference counting keeps it alive if we hold a reference.
             // Let's modify forward to store x_local in a member variable `x_local_cache`.
             if (x_local_cache.is_valid()) {
                 x_local_cache.backward(&grad_input);
             }
         }
    }
};


// =============================================================================
// Learning Rate Scheduler
// =============================================================================

float get_lr(int step, float max_lr, float min_lr, int warmup_steps, int max_steps) {
    if (step < warmup_steps) {
        return max_lr * static_cast<float>(step + 1) / static_cast<float>(warmup_steps);
    }
    if (step > max_steps) {
        return min_lr;
    }
    float decay_ratio = static_cast<float>(step - warmup_steps) / static_cast<float>(max_steps - warmup_steps);
    float coeff = 0.5f * (1.0f + std::cos(M_PI * decay_ratio));
    return min_lr + coeff * (max_lr - min_lr);
}

// =============================================================================
// Main Training Loop
// =============================================================================

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if(rank == 0){
        std::cout << "=== GPT-2 MLP-Only Tensor Parallel Training Script ===" << std::endl;
        std::cout << "Rank 0: Embeddings + Head + Loss" << std::endl;
        std::cout << "All Ranks: Distributed MLP (Broadcast Input -> TP -> AllReduce Output)" << std::endl;
    }
    try {
        
        // Configuration
        GPTConfig config;
        config.batch_size = 8;
        config.context_length = 1024;
        config.vocab_size = 50304;
        config.n_embd = 384;
        config.n_layers = 3;
        
        // Training hyperparameters
        const int B = 8;           // Batch size
        const int T = 1024;        // Sequence length
        const int global_batch = 65536;  
        const int grad_accum_steps = global_batch / (B * T);
        
        const float max_lr = 1e-4f;  
        const float min_lr = max_lr * 0.1f;
        
        DeviceIndex device(Device::CUDA, rank);
        cudaSetDevice(rank);
        
        std::vector<int> ranks_vec(world_size);
        for (int i = 0; i < world_size; i++) ranks_vec[i] = i;
        DeviceMesh mesh({world_size}, ranks_vec);
        auto pg = mesh.get_process_group(0);
        
        // Create model
        GPT model(config, mesh, pg, device);
        
        // Create optimizer (All ranks create it, but Rank > 0 has fewer params)
        std::vector<DTensor*> params = model.parameters();
        dnn::AdamW optimizer(max_lr, 0.9f, 0.95f, 1e-8f, 0.1f);
        
        int64_t num_params = 0;
        for(auto& p : params) num_params += p->mutable_tensor().numel();
        
        const int max_steps = 100; // Simplified for test
        const int warmup_steps = 10;

        if(rank == 0){
            std::cout << "Number of parameters (Rank 0 view): " << num_params << std::endl;
        }

        // Data Loader (Rank 0 Only)
        // Others need dummy loader or just wait? 
        // We only load data on Rank 0 and use it there.
        // But for code simplicity, we can init variables.
        std::string data_root = "/home/blu-bridge25/Study/Code/TensorParallelismBeta/DTensor/Data_Loader/Data/";
        DataLoaderLite* train_loader = nullptr;
        if (rank == 0) {
            train_loader = new DataLoaderLite(B, T, 0, 1, "train", data_root, true, 100000000);
        }

        CudaTimer timer_step;
        
        for (int step = 0; step < max_steps; ++step) {
            timer_step.start_timer();
            
            optimizer.zero_grad();
            float loss_accum = 0.0f;
            
            for (int micro_step = 0; micro_step < grad_accum_steps; ++micro_step) {
                
                Tensor x, y;
                if (rank == 0) {
                    Batch batch = train_loader->next_batch();
                    x = batch.input.to(device);
                    y = batch.target.to(device);
                    
                    // 1. Forward
                    Tensor logits = model.forward(x);
                    
                    // 5. Loss (Rank 0)
                    Tensor loss = autograd::sparse_cross_entropy_loss(logits, y);
                    Tensor divisor = Tensor::full(Shape{{1}}, TensorOptions().with_device(loss.device()), 
                                                  static_cast<float>(grad_accum_steps));
                    loss = autograd::div(loss, divisor);
                    
                    loss_accum += loss.to_cpu().data<float>()[0];
                    
                    // 6. Backward Part 1 (Rank 0): Loss -> Head -> MLP Output
                    loss.backward();
                    // Now model.x_dist_output on Rank 0 has grads.
                } else {
                    // Other ranks just run forward (waiting for broadcast)
                    // Input x is dummy, not used
                    Tensor dummy_idx; 
                    model.forward(dummy_idx);
                }
                
                // --- SYNCHRONIZED BACKWARD BRIDGE ---
                // All ranks must participate here
                
                // 7. Backward Part 2 (All Ranks): MLP Output -> MLP Input (x_dist_input)
                // Need to feed appropriate gradient to backprop through the distributed graph
                Tensor grad_for_mlp;
                if (rank == 0) {
                    // It's possible for grad to be missing if something went wrong, check it
                    if (model.x_dist_output.mutable_tensor().has_grad()) {
                        grad_for_mlp = model.x_dist_output.mutable_tensor().grad();
                    } else {
                        // If no loss backward reached here (e.g. detached), zero grad
                        grad_for_mlp = Tensor::zeros(model.x_dist_output.mutable_tensor().shape(), 
                                                   model.x_dist_output.mutable_tensor().opts());
                    }
                } else {
                    // Rank > 0: Zero gradient (dummy)
                    auto& t = model.x_dist_output.mutable_tensor();
                    grad_for_mlp = Tensor::zeros(t.shape(), t.opts());
                }
                
                // Trigger Distributed Backward
                // This calls backward() on the local tensor, which triggers the autograd engine.
                // The engine encounters AllReduceSumBackward (from fc_down), which syncs grads.
                model.x_dist_output.mutable_tensor().backward(&grad_for_mlp);
                
                // 8. Backward Part 3 (All Ranks): All-Reduce gradients of MLP Input
                // The gradient at model.x_dist_input is partial (from that rank's MLP shard).
                // d_x_local = Sum(d_x_dist_i)
                if (model.x_dist_input.mutable_tensor().has_grad()) {
                    Tensor input_grad = model.x_dist_input.mutable_tensor().grad();
                    
                    // In-place All-Reduce (Sum)
                    pg->all_reduce_async(input_grad.data<float>(), input_grad.data<float>(), 
                                        input_grad.numel(), Dtype::Float32, sum, false)->wait();
                                        
                    // 9. Backward Part 4 (Rank 0): MLP Input Grad -> Embeddings
                    model.backward_embeddings(input_grad);
                }
            }
            
            // Optimization
            // Only sync grads if needed? 
            // We already did all-reduce for MLP params inside their backward pass?
            // DModule::all_reduce_gradients is used when we accumulate gradients locally and then sync.
            // But here we use autograd-aware syncing. DRowLinear.backward syncs internally if it was used?
            // Wait, DRowLinear uses sync_w_autograd in forward.
            // sync_w_autograd registers AllReduceSumBackward.
            // Backward of AllReduceSumBackward is: AllReduce(grad).
            // So gradients of fc_down are ALREADY synchronized?
            // NO. AllReduceSumBackward synchronizes gradients of the INPUT to AllReduce.
            // It does NOT synchronize the PARAMETER gradients.
            // Parameter gradients are accumulated locally on each rank.
            // For FC layers, we need to handle gradient synch?
            // DColumnLinear: Weights are sharded. No sync needed (unless TP > 1 implies something else).
            // DRowLinear: Weights are sharded. No sync needed.
            // Bias in DRowLinear is REPLICATED. Gradients must be All-Reduced.
            // DEmbedding: Replicated. Gradients must be All-Reduced.
            // Our MLP has fc_up (Sharded), fc_down (Sharded), ln (Replicated).
            // So LN gradients need All-Reduce.
            // Biases (if any) might need All-Reduce.
            
            // clip_grad_norm_dtensor_nccl handles syncing internally?
            // No, it computes norm.
            // We usually call model.all_reduce_gradients(pg) before stepping.
            
            // Sync gradients for replicated parameters
            model.all_reduce_gradients(pg.get());
            
            float norm = dnn::clip_grad_norm_dtensor_nccl(params, 1.0f, pg);
            float lr = get_lr(step, max_lr, min_lr, warmup_steps, max_steps);
            optimizer.set_lr(lr);
            optimizer.step(params);
            
            double dt = timer_step.get_elapsed_seconds();
            
            if (rank == 0) {
                std::cout << "step " << std::setw(5) << step 
                          << " | loss: " << std::fixed << std::setprecision(6) << loss_accum 
                          << " | lr " << std::scientific << std::setprecision(4) << lr 
                          << " | norm: " << std::fixed << std::setprecision(4) << norm 
                          << " | dt: " << std::fixed << std::setprecision(2) << (dt * 1000.0) << "ms"
                          << std::endl;
            }
        }
        
        if (rank == 0) delete train_loader;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR Rank " << rank << ": " << e.what() << std::endl;
        return 1;
    }
}
