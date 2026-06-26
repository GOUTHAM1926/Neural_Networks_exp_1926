#include <iostream>
#include "TensorLib.h"
#include "ops/helpers/ActivationKernels.h"
#include "utils/KernelUtils.cuh"

using namespace OwnTensor;

float benchmarkNaiveKernel(Tensor& input, int64_t rows, int64_t cols, TensorOptions ops){
  //* get size of L2 cache
  int device = 0;
  int l2_size = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceGetAttribute(&l2_size, cudaDevAttrL2CacheSize, device);
  std::cout << "L2 cache size: " << l2_size / (1024 * 1024) << "mb" << "\n";
  size_t flushSize = l2_size * 2;
  // float* d_F = nullptr;
  void* d_F = nullptr;
  cudaMalloc(&d_F, flushSize);
  d_F = static_cast<float*>(d_F);
  cudaMemset(&d_F, 0, flushSize);

  Tensor naive_output = Tensor::empty(Shape{{rows,cols}}, ops);
  std::cout << "calling the Naive approach:\n";

  //* warm ups
  for(int i = 0; i < 10; ++i){
    cuda::softmax_forward_cuda(input.data<float>(), naive_output.data<float>(), rows, cols);
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  std::vector<float>runtime(100);
  for(int i = 0; i < 100; ++i){
    cudaMemsetAsync(&d_F, 0, flushSize);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    cuda::softmax_forward_cuda(input.data<float>(), naive_output.data<float>(), rows, cols);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float duration = 0;
    cudaEventElapsedTime(&duration, start, stop);
    // duration *= 1e6; //* for nano second
    runtime[i] = duration;
  }

  float duration = 0;
  for(int i = 0; i < 100; ++i){
    duration += runtime[i];
  }

  std::cout << "\033[32m" << "Naive Kernel Took: " << duration / 100 << "ms."  << "\033[0m\n\n";
  cudaFree(d_F);
  return duration / 100.0f;
}

float benchmarkOnlineKernel(Tensor& input, int64_t rows, int64_t cols, TensorOptions ops){
  //* get size of L2 cache
  int device = 0;
  int l2_size = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceGetAttribute(&l2_size, cudaDevAttrL2CacheSize, device);
  std::cout << "L2 cache size: " << l2_size / (1024 * 1024) << "mb" << "\n";
  size_t flushSize = l2_size * 2;
  // float* d_F = nullptr;
  void* d_F = nullptr;
  cudaMalloc(&d_F, flushSize);
  d_F = static_cast<float*>(d_F);

  cudaMemset(&d_F, 0, flushSize);

  Tensor online_output = Tensor::empty(Shape{{rows,cols}}, ops);
  std::cout << "calling the Naive approach:\n";

  //* warm ups
  for(int i = 0; i < 10; ++i){
    cuda::softmaxonline_forward_cuda(input.data<float>(), online_output.data<float>(), rows, cols);
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  std::vector<float>runtime(100);
  for(int i = 0; i < 100; ++i){
    cudaMemsetAsync(&d_F, 0, flushSize);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    cuda::softmaxonline_forward_cuda(input.data<float>(), online_output.data<float>(), rows, cols);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float duration = 0;
    cudaEventElapsedTime(&duration, start, stop);
    // duration *= 1e6; //* for nano second
    runtime[i] = duration;
  }

  float duration = 0;
  for(int i = 0; i < 100; ++i){
    duration += runtime[i];
  }

  std::cout << "\033[32m" << "Online Kernel Took: " << duration / 100 << "ms."  << "\033[0m\n\n";
  cudaFree(d_F);

  return duration / 100.0f;
}

int main(){
  //* needed parameters - input, output, rows, cols
  int n = 1;
  int64_t rows = 8192*n;
  int64_t cols = 1024*n;
  size_t kb = (rows*cols*sizeof(float))/(1024*1024);
  printf("\nRunning on the size [%d, %d] with size %ldmb...\n", rows, cols, kb); 
  TensorOptions ops = TensorOptions().with_dtype(Dtype::Float32).with_device(Device::CUDA);
  Tensor input = Tensor::rand<float>(Shape{{rows,cols}}, ops);
  
  printf("\nBenchmarking for size: [%ld, %ld]\n", rows, cols);
  std::cout << "Input Tensor:\n";
  input.display();
  std::cout << "\n";

  //* benchamark kernels
  float naiveAvg = benchmarkNaiveKernel(input, rows, cols, ops);
  float onlineAvg = benchmarkOnlineKernel(input, rows, cols, ops);

  std::cout << "\033[32m" << "Benchmarking done!\n" << "\033[0m";
  std::cout << "\033[32m" << "Average Time Taken By Naive Kernel: " << naiveAvg << "ms\n" << "\033[0m";
  std::cout << "\033[32m" << "Average Time Taken By Online Kernel: " << onlineAvg << "ms\n" << "\033[0m";
  
  if(onlineAvg < naiveAvg){
      std::cout << "Speedup: " << naiveAvg / onlineAvg << " times\n";
  } else{
      std::cout << "Slow by: " << onlineAvg - naiveAvg << " ms\n";
  }
  return 0;
}