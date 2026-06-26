#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <chrono>
#include <stdexcept>
#include <cmath>
#include <sys/resource.h>
#include <unistd.h>

#include "TensorLib.h"
#include "autograd/operations/MatrixOps.h"
#include "autograd/operations/ActivationOps.h"
#include "autograd/operations/ReductionOps.h"

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

using namespace OwnTensor;


long get_peak_rss() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss; // in kilobytes
}

size_t get_cuda_used_memory() {
#ifdef WITH_CUDA
    size_t free, total;
    cudaError_t err = cudaMemGetInfo(&free, &total);
    if (err != cudaSuccess) return 0;
    return total - free;
#else
    return 0;
#endif
}


Tensor load_tensor(const std::string& filename, std::ifstream& f) {
    if (!f.is_open()) throw std::runtime_error("File not open");

    uint32_t num_dims;
    f.read(reinterpret_cast<char*>(&num_dims), sizeof(uint32_t));

    std::vector<int64_t> dims(num_dims);
    size_t num_elements = 1;
    for (uint32_t i = 0; i < num_dims; ++i) {
        uint32_t d;
        f.read(reinterpret_cast<char*>(&d), sizeof(uint32_t));
        dims[i] = static_cast<int64_t>(d);
        num_elements *= d;
    }

    std::vector<float> data(num_elements);
    f.read(reinterpret_cast<char*>(data.data()), num_elements * sizeof(float));

    Tensor t = Tensor::empty(Shape(dims), {Dtype::Float32, DeviceIndex(Device::CPU, 0)});
    t.set_data(data); 
    return t;
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    try {
        std::string data_file = "accuracy_test_data.bin";
        std::string results_file = "memory_test_results.bin";
        
        std::ifstream fin(data_file, std::ios::binary);
        if (!fin) throw std::runtime_error("Could not open data file: " + data_file);
        
        Tensor input = load_tensor(data_file, fin);
        Tensor target = load_tensor(data_file, fin);
        Tensor fc1_w_init = load_tensor(data_file, fin);
        Tensor fc1_b_init = load_tensor(data_file, fin);
        Tensor fc2_w_init = load_tensor(data_file, fin);
        Tensor fc2_b_init = load_tensor(data_file, fin);
        Tensor fc3_w_init = load_tensor(data_file, fin);
        Tensor fc3_b_init = load_tensor(data_file, fin);
        fin.close();
        
        Device dev_type = Device::CPU;
        const char* env_dev = std::getenv("TEST_DEVICE");
        if (env_dev && (std::string(env_dev) == "CUDA" || std::string(env_dev) == "cuda")) {
            if (device::cuda_available()) dev_type = Device::CUDA;
        }
        DeviceIndex dev_idx(dev_type, 0);
        
        input = input.to(dev_idx);
        target = target.to(dev_idx);
        Tensor w1 = fc1_w_init.to(dev_idx); w1.set_requires_grad(true);
        Tensor b1 = fc1_b_init.to(dev_idx); b1.set_requires_grad(true);
        Tensor w2 = fc2_w_init.to(dev_idx); w2.set_requires_grad(true);
        Tensor b2 = fc2_b_init.to(dev_idx); b2.set_requires_grad(true);
        Tensor w3 = fc3_w_init.to(dev_idx); w3.set_requires_grad(true);
        Tensor b3 = fc3_b_init.to(dev_idx); b3.set_requires_grad(true);
        
        auto update_param = [](Tensor& p, float lr) {
            if (p.has_grad()) {
                Tensor g = p.grad_view();
                p = (p - g * lr);
                p.set_grad_fn(nullptr);
                p.set_requires_grad(true);
            }
        };

        size_t peak_cuda = 0;
        long peak_cpu = 0;

        // Warmup and initial measurement
        for (int i = 0; i < 100; ++i) {
            Tensor x1 = autograd::matmul(input, w1.transpose(0, 1)) + b1;
            Tensor relu_out = autograd::relu(x1);
            Tensor x2 = autograd::matmul(relu_out, w2.transpose(0, 1)) + b2;
            Tensor sigmoid_out = autograd::sigmoid(x2);
            Tensor final_out = autograd::matmul(sigmoid_out, w3.transpose(0, 1)) + b3;
            Tensor diff = final_out - target;
            Tensor loss = autograd::mean(diff * diff);
            loss.backward();
            
            update_param(w1, 0.01f); update_param(b1, 0.01f);
            update_param(w2, 0.01f); update_param(b2, 0.01f);
            update_param(w3, 0.01f); update_param(b3, 0.01f);

            // Track peaks
            if (dev_type == Device::CUDA) {
                size_t current_cuda = get_cuda_used_memory();
                if (current_cuda > peak_cuda) peak_cuda = current_cuda;
            }
            long current_cpu = get_peak_rss();
            if (current_cpu > peak_cpu) peak_cpu = current_cpu;
        }

        std::cout << "[C++] Peak CPU RSS: " << peak_cpu << " KB" << std::endl;
        if (dev_type == Device::CUDA) {
            std::cout << "[C++] Peak CUDA Memory: " << peak_cuda / (1024 * 1024) << " MB" << std::endl;
        }

        // Save Results
        std::ofstream fout(results_file, std::ios::binary);
        if (!fout) throw std::runtime_error("Could not open results file");
        
        float cpu_mb = (float)peak_cpu / 1024.0f;
        float cuda_mb = (float)peak_cuda / (1024.0f * 1024.0f);
        
        fout.write(reinterpret_cast<const char*>(&cpu_mb), sizeof(float));
        fout.write(reinterpret_cast<const char*>(&cuda_mb), sizeof(float));
        fout.close();
        
    } catch (const std::exception& e) {
        std::cerr << "C++ Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
