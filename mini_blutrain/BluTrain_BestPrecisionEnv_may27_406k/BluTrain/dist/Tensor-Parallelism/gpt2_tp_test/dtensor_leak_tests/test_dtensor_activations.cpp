#include "dtensor_test_utils.h"

using namespace OwnTensor;
using namespace OwnTensor::dnn;

// Forward declaration
void run_dtensor_activation_tests(const DeviceMesh& mesh, std::shared_ptr<ProcessGroupNCCL> pg);

void run_dtensor_activation_tests(const DeviceMesh& mesh, std::shared_ptr<ProcessGroupNCCL> pg) {
    int rank = pg->get_rank();
    if (rank == 0) {
        std::cout << "\n=== DTensor Activation Tests ===" << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    int64_t B = 4, T = 64, C = 256;
    
    // --- DGeLU ---
    {
        Layout input_layout(mesh, {B, T, C});
        DTensor input(mesh, pg, input_layout, "gelu_input");
        input.mutable_tensor().set_requires_grad(true);
        
        DGeLU gelu;
        
        auto run_fn = [&]() { return gelu.forward(input); };
        
        DTensorTestMetrics m = benchmark_dtensor_op("DGeLU", run_fn, input, 50);
        print_dtensor_result("DGeLU [B,T,C]", m);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // --- Distributed ReLU (using underlying autograd::relu) ---
    {
        Layout input_layout(mesh, {B, T, C});
        DTensor input(mesh, pg, input_layout, "relu_input");
        input.mutable_tensor().set_requires_grad(true);
        
        auto run_fn = [&]() {
            Tensor& in_tensor = input.mutable_tensor();
            Tensor out_tensor = autograd::relu(in_tensor);
            
            DTensor output(input.get_device_mesh(), input.get_pg(), input.get_layout());
            output.mutable_tensor() = out_tensor;
            return output;
        };
        
        DTensorTestMetrics m = benchmark_dtensor_op("Distributed ReLU", run_fn, input, 50);
        print_dtensor_result("Distributed ReLU [B,T,C]", m);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // --- Distributed Softmax ---
    {
        Layout input_layout(mesh, {B, T, C});
        DTensor input(mesh, pg, input_layout, "softmax_input");
        input.mutable_tensor().set_requires_grad(true);
        
        auto run_fn = [&]() {
            Tensor& in_tensor = input.mutable_tensor();
            Tensor out_tensor = autograd::softmax(in_tensor, -1);
            
            DTensor output(input.get_device_mesh(), input.get_pg(), input.get_layout());
            output.mutable_tensor() = out_tensor;
            return output;
        };
        
        DTensorTestMetrics m = benchmark_dtensor_op("Distributed Softmax", run_fn, input, 50);
        print_dtensor_result("Distributed Softmax [B,T,C]", m);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // --- Distributed Sigmoid ---
    {
        Layout input_layout(mesh, {B, T, C});
        DTensor input(mesh, pg, input_layout, "sigmoid_input");
        input.mutable_tensor().set_requires_grad(true);
        
        auto run_fn = [&]() {
            Tensor& in_tensor = input.mutable_tensor();
            Tensor out_tensor = autograd::sigmoid(in_tensor);
            
            DTensor output(input.get_device_mesh(), input.get_pg(), input.get_layout());
            output.mutable_tensor() = out_tensor;
            return output;
        };
        
        DTensorTestMetrics m = benchmark_dtensor_op("Distributed Sigmoid", run_fn, input, 50);
        print_dtensor_result("Distributed Sigmoid [B,T,C]", m);
    }
}
