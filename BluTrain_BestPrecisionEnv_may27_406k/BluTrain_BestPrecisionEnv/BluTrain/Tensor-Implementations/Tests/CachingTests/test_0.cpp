#include "TensorLib.h"
#include "device/AllocationTracker.h"

using namespace OwnTensor;

int main() {
  AllocationTracker::instance().init("allocations.csv");

  for (int i = 0; i < 5; ++i) {
    // Use scoped naming with lifetime hints
    {
      ScopedAllocName scope("input_tensor_" + std::to_string(i),
                            AllocLifetime::TEMPORARY);
      Tensor a = Tensor::rand(
          {{4, 4, 4}}, {Dtype::Float32, Device::CUDA, false}, i, -2.5f, 2.5f);

      AllocationTracker::set_thread_name("relu_output_" + std::to_string(i),
                                         AllocLifetime::ACTIVATION);
      Tensor b = ReLU(a);
      // Tensor c = b.t() + a.t();
      // b.display();

      a.reset();
      b.reset();
    }

    // Check for leaks after each iteration
    std::cerr << "\n--- After iteration " << i << " ---\n";
    std::cerr << "Current allocated (all): "
              << AllocationTracker::instance().get_current_allocated()
              << " bytes\n";
    std::cerr << "Peak allocated (all): "
              << AllocationTracker::instance().get_peak_allocated()
              << " bytes\n";
  }

  // Final leak report
  AllocationTracker::instance().print_leak_report();

  // Show total allocations made
  std::cerr << "Total allocations tracked: "
            << AllocationTracker::instance().get_total_allocations() << "\n";

  AllocationTracker::instance().shutdown();
  return 0;
}