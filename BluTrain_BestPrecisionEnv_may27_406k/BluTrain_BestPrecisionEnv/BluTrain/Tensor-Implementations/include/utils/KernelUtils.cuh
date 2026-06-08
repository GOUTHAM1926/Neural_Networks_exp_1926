#pragma once

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <iterator>
#include <stdlib.h>
#include <cuda_runtime.h>

//* Helper function to initialize random values to a vector
template<typename T>
void Init(std::vector<T>& vec){
  size_t n = vec.size();
  
  std::random_device rd;
  std::mt19937 mersenne_engine(rd());

  std::uniform_real_distribution<T> dist(0.0f,1.0f);
  auto gen = [&]() { return dist(mersenne_engine); };

  std::generate(vec.begin(), vec.end(), gen);
}

//* Helper function to display the vector
template<typename T>
void displayVector(std::vector<T> vec, size_t digits = 10){
  for(size_t i = 0; i < digits; ++i){
    std::cout << vec[i] << " ";
  } std::cout << "\n";
}

//* Helper function to display a matrix
template<typename T>
void displayMatrix(std::vector<T>Mat, int rows, int cols){

}

//! I need to work making this more robust
//* helper function to verify the result
void checkResult(float *hostRef, float *gpuRef, const size_t n){
  std::cout << "Calling the check result utility...\n";
  double epsilon = 1.0E-8;
  bool match = true;
  for(size_t i = 0; i < n; ++i){
    if(abs(hostRef[i] - gpuRef[i]) > epsilon){
      match = false;
      std::cout << "\033[31m" << "Values don't match!\n" << "\033[0m";
      break;
    }
  }
  if(match){
    std::cout << "\033[32m" << "Values match!\n" << "\033[0m";
  }
}

//* helper function to check the error status
#define CUDA_CHECK(call){ \
  const cudaError_t status = call; \
  if(status != cudaSuccess){ \
    printf("Error: %s : %d\n", __FILE__, __LINE__); \
    printf("code : %d , reason: %s\n", status, cudaGetErrorString(status)); \
    exit(1); \
  } \
}