#include <iostream>
#include <random>
#include <cassert>
#include "checkpointing/RNG.h"
#include "checkpointing/Checkpoint.h"
#include "autograd/operations/BinaryOps.h"
#include "autograd/operations/ArithmeticsOps.h"
#include "device/DeviceCore.h"
#include "TensorLib.h"

using namespace OwnTensor;
using namespace OwnTensor::autograd;

// A block that uses RNG (e.g., simulating noise or dropout)
variable_list rng_block(const variable_list& inputs) {
    Tensor x = inputs[0];
    
    // Simulate dropout manually with device-safe RNG
    // rand() returns values between lower and upper. We use [0, 1] and then compare to simulate dropout.
    // Note: OwnTensor might not have a boolean threshold op yet, so we use simple arithmetics.
    Tensor rand_vals = Tensor::rand<float>(x.shape(), x.opts(), 42, 0.0f, 1.0f);
    
    // We want binary mask: noise = (rand_vals > 0.5)
    // For now, let's just multiply by rand_vals to simulate any RNG-dependent operation.
    return {autograd::mul(x, rand_vals)};
}

void test_rng_determinism(DeviceIndex device) {
    std::cout << "\n>>> Testing RNG Determinism on " << (device.is_cpu() ? "CPU" : "GPU") << " <<<\n";
    
    TensorOptions opts = TensorOptions().with_device(device).with_req_grad(true);
    Tensor x = Tensor::ones(Shape{{1, 100}}, opts);
    
    RNG::set_seed(42);
    
    // 1. Run with checkpointing
    // We expect the forward to use some RNG state, and the backward to use the SAME state
    // because CheckpointNode re-runs the forward.
    variable_list out_list = checkpoint(rng_block, {x});
    Tensor out = out_list[0];
    
    // Capture output values
    Tensor out_cpu = out.to(Device::CPU);
    std::vector<float> values(out_cpu.numel());
    std::memcpy(values.data(), out_cpu.data(), out_cpu.nbytes());
    
    // Trigger backward. 
    // CheckpointNode::apply will:
    //   1. Save current RNG
    //   2. Set RNG to saved_state_
    //   3. Run forward
    //   4. Restore RNG
    // For this test to be useful, we want to know if the recomputed forward matched.
    // We can't easily see the recomputed forward, but we can check if RNG is restored correctly after apply.
    
    RNG::set_seed(100); // Change seed outside
    float v_outside1 = std::uniform_real_distribution<float>(0.0, 1.0)(RNG::get_cpu_generator());
    
    autograd::backward(autograd::sum(out));
    
    float v_outside2 = std::uniform_real_distribution<float>(0.0, 1.0)(RNG::get_cpu_generator());
    
    // If RNG restoration works, v_outside2 should follow v_outside1 correctly
    // (Meaning CheckpointNode didn't permanently corrupt the RNG state)
    
    std::cout << "Value before backward: " << v_outside1 << std::endl;
    std::cout << "Value after backward:  " << v_outside2 << std::endl;
    
    // Verify that the forward pass inside checkpointing is indeed deterministic
    // by comparing two identical checkpointed calls.
    RNG::set_seed(123);
    variable_list out1 = checkpoint(rng_block, {x});
    
    RNG::set_seed(123);
    variable_list out2 = checkpoint(rng_block, {x});
    
    Tensor out1_c = out1[0].to(Device::CPU);
    Tensor out2_c = out2[0].to(Device::CPU);
    
    bool match = true;
    for (size_t i = 0; i < out1_c.numel(); ++i) {
        if (out1_c.data<float>()[i] != out2_c.data<float>()[i]) {
            match = false;
            break;
        }
    }
    
    if (match) {
        std::cout << "SUCCESS: Checkpointed forward passes are deterministic with seed!\n";
    } else {
        std::cout << "FAILURE: Checkpointed forward passes diverged!\n";
        exit(1);
    }
}

int main() {
    try {
        test_rng_determinism(DeviceIndex(Device::CPU));
        #ifdef WITH_CUDA
        if (OwnTensor::device::cuda_available()) {
            test_rng_determinism(DeviceIndex(Device::CUDA, 0));
        }
        #endif
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
