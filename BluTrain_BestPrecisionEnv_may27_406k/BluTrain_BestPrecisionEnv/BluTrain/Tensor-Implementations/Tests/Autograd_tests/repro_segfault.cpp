#include "TensorLib.h"
#include <iostream>
#include <vector>

using namespace OwnTensor;

int main() {
    std::cout << "Starting BCE Repro..." << std::endl;
    
    int64_t B = 64, C = 10;
    Shape shape{{B, C}};
    
    TensorOptions opts = TensorOptions().with_device(DeviceIndex(Device::CPU)).with_dtype(Dtype::Float32).with_req_grad(true);
    
    for (int i = 0; i < 100; ++i) {
        if (i % 10 == 0) std::cout << "Iter " << i << std::endl;
        
        // BCE
        Tensor pred = autograd::sigmoid(Tensor::randn<float>(shape, opts));
        Tensor target = autograd::sigmoid(Tensor::randn<float>(shape, opts.with_req_grad(false))); 
        
        Tensor loss = autograd::binary_cross_entropy(pred, target);
        
        // std::cout << "Calling backward..." << std::endl;
        loss.backward();
        // std::cout << "Backward done." << std::endl;
    }
    
    std::cout << "BCE Repro Complete." << std::endl;
    return 0;
}
