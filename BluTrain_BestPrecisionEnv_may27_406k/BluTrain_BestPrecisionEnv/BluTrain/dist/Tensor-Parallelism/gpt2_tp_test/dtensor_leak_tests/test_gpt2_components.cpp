#include "dtensor_test_utils.h"

// This file tests components from gpt2_entropy_parallel_test.cpp for memory leaks

using namespace OwnTensor;
using namespace OwnTensor::dnn;

// Forward declaration
void run_gpt2_component_tests(const DeviceMesh& mesh, std::shared_ptr<ProcessGroupNCCL> pg);




// =============================================================================
// AllGatherBackward - from gpt2_entropy_parallel_test.cpp
// =============================================================================
class AllGatherBackward : public Node {
public:
    AllGatherBackward(int rank, int world_size, std::shared_ptr<ProcessGroupNCCL> pg)
        : rank_(rank), world_size_(world_size), pg_(pg) {}

    std::vector<Tensor> apply(std::vector<Tensor>&& grad_outputs) override {
        Tensor grad_out = grad_outputs[0]; // [B, T, V]
        
        int64_t B = grad_out.shape().dims[0];
        int64_t T = grad_out.shape().dims[1];
        int64_t V = grad_out.shape().dims[2];
        int64_t local_v = V / world_size_;
        int64_t start_col = rank_ * local_v;

        Shape out_shape{{B, T, local_v}};
        Tensor grad_shard = Tensor::zeros(out_shape, grad_out.opts());

        float* src_ptr = grad_out.data<float>() + start_col;
        float* dst_ptr = grad_shard.data<float>();

        cudaMemcpy2D(
            dst_ptr, local_v * sizeof(float),
            src_ptr, V * sizeof(float),
            local_v * sizeof(float), B * T,
            cudaMemcpyDeviceToDevice
        );

        return {grad_shard};
    }
private:
    int rank_;
    int world_size_;
    std::shared_ptr<ProcessGroupNCCL> pg_;
};

void run_gpt2_component_tests(const DeviceMesh& mesh, std::shared_ptr<ProcessGroupNCCL> pg) {
    int rank = pg->get_rank();
    int world_size = pg->get_worldsize();
    
    if (rank == 0) {
        std::cout << "\n=== GPT-2 Component Tests ===" << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    int64_t B = 4, T = 64, C = 256;
    int64_t V = 1024;  // Small vocab for testing
    int64_t F = C * 4;
    int64_t local_v = V / world_size;
    
    // --- MLP Block (Column + Row Linear + GeLU) ---
    // This mirrors the MLP class from gpt2_entropy_parallel_test.cpp
    {
        DColumnLinear fc1(mesh, pg, B, T, C, F, std::vector<float>{}, true, 0.02f);
        DRowLinear fc2(mesh, pg, B, T, F, C, std::vector<float>{}, true, 0.02f, true);
        DGeLU gelu;
        nn::LayerNorm ln(C);
        ln.to(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank));
        
        Layout input_layout(mesh, {B, T, C});
        DTensor input(mesh, pg, input_layout, "mlp_input");
        input.mutable_tensor().set_requires_grad(true);
        
        auto run_fn = [&]() {
            // Clone input to avoid corrupting shared state
            DTensor x(mesh, pg, input_layout, "mlp_x");
            x.mutable_tensor() = input.mutable_tensor().clone();
            x.mutable_tensor().set_requires_grad(true);
            
            // MLP forward: LN -> fc1 -> GeLU -> fc2
            x.mutable_tensor() = ln.forward(x.mutable_tensor());
            DTensor h1 = fc1.forward(x);
            h1 = gelu.forward(h1);
            DTensor y = fc2.forward(h1);
            return y;
        };
        
        // benchmark_dtensor_op now measures per-iteration leaks internally
        DTensorTestMetrics m = benchmark_dtensor_op("MLP Block", run_fn, input, 5);
        print_dtensor_result("MLP Block [B,T,C]->[B,T,C]", m);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // --- DEmbeddingVParallel (Sharded Token Embedding) ---
    {
        Layout idx_layout(mesh, {B, T});
        DTensor indices(mesh, pg, idx_layout, "emb_indices");
        
        std::vector<float> idx_data(B * T);
        for (int64_t i = 0; i < B * T; i++) {
            idx_data[i] = static_cast<float>(i % V);
        }
        indices.setData(idx_data);
        indices.mutable_tensor() = indices.mutable_tensor().as_type(Dtype::Int32);
        
        DEmbeddingVParallel wte(mesh, pg, V, C);
        
        auto run_fn = [&]() { return wte.forward(indices); };
        
        DTensorTestMetrics m = benchmark_dtensor_op("DEmbeddingVParallel", run_fn, indices, 10);
        print_dtensor_result("DEmbeddingVParallel [B,T]->[B,T,C]", m);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // --- DEmbedding (Position Embedding) ---
    {
        Layout idx_layout(mesh, {1, T});
        DTensor indices(mesh, pg, idx_layout, "pos_indices");
        
        std::vector<float> idx_data(T);
        for (int64_t i = 0; i < T; i++) {
            idx_data[i] = static_cast<float>(i);
        }
        indices.setData(idx_data);
        indices.mutable_tensor() = indices.mutable_tensor().as_type(Dtype::Int32);
        
        DEmbedding wpe(mesh, pg, T, C);
        
        auto run_fn = [&]() { return wpe.forward(indices); };
        
        DTensorTestMetrics m = benchmark_dtensor_op("DEmbedding (pos)", run_fn, indices, 10);
        print_dtensor_result("DEmbedding (pos) [1,T]->[1,T,C]", m);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // --- DLMHead (LM Head with weight tying) ---
    {
        Layout input_layout(mesh, {B, T, C});
        DTensor input(mesh, pg, input_layout, "lmhead_input");
        input.mutable_tensor().set_requires_grad(true);
        
        // Create embedding for weight tying
        DEmbeddingVParallel wte(mesh, pg, V, C);
        DLMHead lm_head(mesh, pg, B, T, C, V, wte.weight.get());
        
        auto run_fn = [&]() { return lm_head.forward(input); };
        
        DTensorTestMetrics m = benchmark_dtensor_op("DLMHead", run_fn, input, 10);
        print_dtensor_result("DLMHead [B,T,C]->[B,T,V_local]", m);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // --- LayerNorm + Residual Pattern ---
    {
        Layout input_layout(mesh, {B, T, C});
        DTensor input(mesh, pg, input_layout, "ln_residual_input");
        input.mutable_tensor().set_requires_grad(true);
        
        nn::LayerNorm ln(C);
        ln.to(OwnTensor::DeviceIndex(OwnTensor::Device::CUDA, rank));
        
        auto run_fn = [&]() {
            Tensor x = input.mutable_tensor();
            Tensor norm = ln.forward(x);
            Tensor residual = autograd::add(x, norm);
            
            DTensor output(mesh, pg, input_layout, "ln_residual_out");
            output.mutable_tensor() = residual;
            return output;
        };
        
        DTensorTestMetrics m = benchmark_dtensor_op("LayerNorm+Residual", run_fn, input, 20);
        print_dtensor_result("LayerNorm+Residual [B,T,C]", m);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // --- AllGatherBackward (Gradient slicing for TP) ---
    {
        // Create a tensor simulating gathered logits [B, T, V]
        Tensor gathered = Tensor::randn<float>(Shape{{B, T, V}}, 
            TensorOptions().with_dtype(Dtype::Float32)
                          .with_device(DeviceIndex(Device::CUDA, rank))
                          .with_req_grad(true));
        
        Layout out_layout(mesh, {B, T, local_v});
        DTensor dummy_input(mesh, pg, out_layout, "dummy");
        
        auto run_fn = [&]() {
            // Simulate backward of all-gather
            Tensor grad_out = Tensor::ones(gathered.shape(), gathered.opts());
            
            AllGatherBackward backward_fn(rank, world_size, pg);
            std::vector<Tensor> grads = {grad_out};
            auto result = backward_fn.apply(std::move(grads));
            
            // Wrap in DTensor for return
            DTensor output(mesh, pg, out_layout, "ag_backward_out");
            output.mutable_tensor() = result[0];
            return output;
        };
        
        DTensorTestMetrics m = benchmark_dtensor_op("AllGatherBackward", run_fn, dummy_input, 10);
        print_dtensor_result("AllGatherBackward [B,T,V]->[B,T,V_local]", m);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // --- Token + Position Embedding Add ---
    {
        Layout tok_layout(mesh, {B, T, C});
        DTensor tok_emb(mesh, pg, tok_layout, "tok_emb");
        tok_emb.mutable_tensor().set_requires_grad(true);
        
        Layout pos_layout(mesh, {1, T, C});
        DTensor pos_emb(mesh, pg, pos_layout, "pos_emb");
        pos_emb.mutable_tensor().set_requires_grad(true);
        
        auto run_fn = [&]() {
            Tensor x = tok_emb.mutable_tensor() + pos_emb.mutable_tensor();
            
            DTensor output(mesh, pg, tok_layout, "combined_emb");
            output.mutable_tensor() = x;
            return output;
        };
        
        DTensorTestMetrics m = benchmark_dtensor_op("Tok+Pos Add", run_fn, tok_emb, 20);
        print_dtensor_result("Tok+Pos Embedding Add [B,T,C]", m);
    }
    
    if (rank == 0) {
        std::cout << "\n[INFO] GPT-2 component tests complete." << std::endl;
    }
}
