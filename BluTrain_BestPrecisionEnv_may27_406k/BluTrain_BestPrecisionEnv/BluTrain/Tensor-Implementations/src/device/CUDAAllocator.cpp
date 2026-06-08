#include "device/CUDAAllocator.h"
#include "device/AllocationTracker.h"
#include <iostream>
#include <stdexcept>
#include "device/DeviceCore.h"

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace OwnTensor
{
    void* CUDAAllocator::allocate(size_t bytes) {
    #ifdef WITH_CUDA
        void* ptr = nullptr;
        cudaStream_t stream = OwnTensor::cuda::getCurrentStream(); //~change
        [[maybe_unused]] cudaError_t err = cudaMallocAsync(&ptr,bytes, stream); //~change
        if (err != cudaSuccess || ptr == nullptr)
        {
            std::cerr << "CUDA Allocation Failed: " << cudaGetErrorString(err) 
                    << "(requested " << bytes / (1024*1024) << " MB" << std::endl;
            throw std::runtime_error("CUDA allcation failed");
        }
            // std::cout << "CUDAAllocator: Allocating " << bytes << " bytes (" 
            //   << bytes / (1024*1024) << " MB)" << std::endl;    
            
            // Track allocation
            int device = 0;
            cudaGetDevice(&device);
            AllocationTracker::instance().on_alloc(ptr, bytes, bytes, device);
            
            return ptr;
    #else
        throw std::runtime_error("CUDA not available");
    #endif
    }

    // void CUDAAllocator::deallocate(void* ptr) {
    // #ifdef WITH_CUDA
    //     // cudaFree(ptr);
    //     if (ptr)
    //     {
    //         cudaError_t err = cudaFree(ptr);
    //         if (err != cudaSuccess) {
    //             std::cerr << "CUDA free failed: " << cudaGetErrorString(err) << std::endl;
    //         }
    //     }
    // #endif
    // }

    void CUDAAllocator::deallocate(void* ptr) {
    #ifdef WITH_CUDA
        if (ptr) {
            // cudaDeviceSynchronize();//✨✨✨
            
            // Track deallocation
            int device = 0;
            cudaGetDevice(&device);
            AllocationTracker::instance().on_free(ptr, device);

            cudaStream_t stream = OwnTensor::cuda::getCurrentStream(); //~change
            [[maybe_unused]] cudaError_t err = cudaFreeAsync(ptr, stream); //~change
            }
        #endif
    }
}