#include <iostream>
#include "TensorLib.h"
#include "device/DeviceTransfer.h"
#include "utils/KernelUtils.cuh"
#include "ops/helpers/ActivationKernels.h"

using namespace OwnTensor;

void testSoftmaxValues(){
  //* needed parameters - input, output, rows, cols
  int64_t rows = 8192;
  int64_t cols = 1024;
  TensorOptions ops = TensorOptions().with_dtype(Dtype::Float32).with_device(Device::CUDA);
  Tensor input = Tensor::rand<float>(Shape{{rows,cols}}, ops);
  Tensor naive_output = Tensor::empty(Shape{{rows,cols}}, ops);
  Tensor online_output = Tensor::empty(Shape{{rows,cols}}, ops);

  std::cout << "Input Tensor:\n";
  input.display();

  std::cout << "calling the Naive approach:\n";
  cuda::softmax_forward_cuda(input.data<float>(), naive_output.data<float>(), rows, cols);
  std::cout << "naive_output:\n";
  naive_output.display();
  std::cout << "\n";

  std::cout << "calling the Online softmax kernel approach:\n";
  cuda::softmaxonline_forward_cuda(input.data<float>(), online_output.data<float>(), rows, cols);
  std::cout << "online_output:\n";
  online_output.display();
  std::cout << "\n";

  std::cout << "calling the softmax kernel approach from cpu:\n";
  auto cpu_input = input.to_cpu();
  Tensor cpu_output = autograd::softmax(cpu_input);
  std::cout << "cpu_output:\n";
  cpu_output.display();
  std::cout << "\n";

  std::cout << "Working fine till here...\n";
  //* move online result to cpu
  Tensor cpuOnlineRes = naive_output.to_cpu(); //* change here
  //* get the pointers
  float *hostRef = cpu_output.data<float>();
  float *gpuRef = cpuOnlineRes.data<float>();
  checkResult(hostRef, gpuRef, rows * cols);
}

int main(){
  testSoftmaxValues();
  return 0;
}