#include "dtensor_test_utils.h"

using namespace OwnTensor;
using namespace OwnTensor::dnn;

// Forward declaration
void run_dtensor_layer_tests(const DeviceMesh& mesh, std::shared_ptr<ProcessGroupNCCL> pg);

void run_dtensor_layer_tests(const DeviceMesh& mesh, std::shared_ptr<ProcessGroupNCCL> pg) {
    int rank = pg->get_rank();
    if (rank == 0) {
        std::cout << "\n=== DTensor Layer Tests ===" << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    int64_t B = 4, T = 64, C = 256;
    int64_t C4 = C * 4;  // Typical MLP expansion factor
    
    // --- DColumnLinear ---
    {
        Layout input_layout(mesh, {B, T, C});
        DTensor input(mesh, pg, input_layout, "column_input");
        input.mutable_tensor().set_requires_grad(true);
        
        DColumnLinear fc1(mesh, pg, B, T, C, C4, {}, true, 0.02);
        
        auto run_fn = [&]() { return fc1.forward(input); };
        
        DTensorTestMetrics m = benchmark_dtensor_op("DColumnLinear", run_fn, input, 20);
        print_dtensor_result("DColumnLinear [B,T,C]->[B,T,4C]", m);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // --- DRowLinear ---
    {
        Layout input_layout(mesh, {B, T, C4}, 2);  // Sharded on last dim
        DTensor input(mesh, pg, input_layout, "row_input");
        input.mutable_tensor().set_requires_grad(true);
        
        DRowLinear fc2(mesh, pg, B, T, C4, C, {}, true, 0.02, true);
        
        auto run_fn = [&]() { return fc2.forward(input); };
        
        DTensorTestMetrics m = benchmark_dtensor_op("DRowLinear", run_fn, input, 20);
        print_dtensor_result("DRowLinear [B,T,4C]->[B,T,C]", m);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // --- DEmbedding ---
    {
        int64_t vocab = 1000;
        int64_t dim = 256;
        
        Layout idx_layout(mesh, {B, T});
        DTensor indices(mesh, pg, idx_layout, "embedding_indices");
        
        std::vector<float> idx_data(B * T);
        for (int64_t i = 0; i < B * T; i++) {
            idx_data[i] = static_cast<float>(i % vocab);
        }
        indices.setData(idx_data);
        indices.mutable_tensor() = indices.mutable_tensor().as_type(Dtype::Int32);
        
        DEmbedding embedding(mesh, pg, vocab, dim);
        
        auto run_fn = [&]() { return embedding.forward(indices); };
        
        DTensorTestMetrics m = benchmark_dtensor_op("DEmbedding", run_fn, indices, 20);
        print_dtensor_result("DEmbedding [B,T]->[B,T,C]", m);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // --- DTensor.linear_w_autograd ---
    {
        Layout input_layout(mesh, {B, T, C});
        DTensor input(mesh, pg, input_layout, "linear_autograd_input");
        input.mutable_tensor().set_requires_grad(true);
        
        Layout weight_layout(mesh, {B, C, C});
        DTensor weight(mesh, pg, weight_layout, "linear_autograd_weight");
        weight.mutable_tensor().set_requires_grad(true);
        
        Layout bias_layout(mesh, {B, T, C});
        DTensor bias(mesh, pg, bias_layout, "linear_autograd_bias");
        bias.mutable_tensor().fill(0.0f);
        bias.mutable_tensor().set_requires_grad(true);
        
        auto run_fn = [&]() {
            Layout out_layout(mesh, {B, T, C});
            DTensor output(mesh, pg, out_layout, "linear_output");
            output.linear_w_autograd(input, weight, bias);
            return output;
        };
        
        DTensorTestMetrics m = benchmark_dtensor_op("linear_w_autograd", run_fn, input, 20);
        print_dtensor_result("DTensor.linear_w_autograd", m);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // --- DTensor.sync_w_autograd ---
    {
        Layout input_layout(mesh, {B, T, C});
        DTensor input(mesh, pg, input_layout, "sync_input");
        input.mutable_tensor().set_requires_grad(true);
        
        // Test sync without clone - just sync the input tensor directly
        auto run_fn = [&]() {
            input.sync_w_autograd();
            input.wait();
            return input;  // Return same tensor, not a new one
        };
        
        DTensorTestMetrics m = benchmark_dtensor_op("sync_w_autograd", run_fn, input, 20);
        print_dtensor_result("DTensor.sync_w_autograd", m);
    }
}
