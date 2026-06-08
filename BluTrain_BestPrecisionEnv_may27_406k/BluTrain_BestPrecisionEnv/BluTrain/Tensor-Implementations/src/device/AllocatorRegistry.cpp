#include "device/AllocatorRegistry.h"
#include "device/CPUAllocator.h"
#include "device/CUDAAllocator.h"
#include "device/PinnedCPUAllocator.h"
#include "device/CachingCudaAllocator.h"
#include <stdexcept>
#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace OwnTensor
{ 
    namespace {
        CPUAllocator cpu_allocator;
        CUDAAllocator cuda_allocator;
        
        device::PinnedCPUAllocator pinned_default(cudaHostAllocDefault);
        device::PinnedCPUAllocator pinned_mapped(cudaHostAllocMapped);
        device::PinnedCPUAllocator pinned_portable(cudaHostAllocPortable);
        device::PinnedCPUAllocator pinned_wc(cudaHostAllocWriteCombined);
    }

    Allocator* AllocatorRegistry::get_allocator(Device device) {
        if (device == Device::CPU) {
            return &cpu_allocator;
        } else if (device == Device::CUDA){
           return &CachingCUDAAllocator::instance();
        } 
        else {  
            // return &cuda_allocator;
            throw std::runtime_error("There is no cuda support");
        }
    }

    Allocator* AllocatorRegistry::get_cpu_allocator() {
        return &cpu_allocator;
    }
    
    Allocator* AllocatorRegistry::get_pinned_cpu_allocator(Pinned_Flag flag) {
        switch (flag) {
            case Pinned_Flag::Default: return &pinned_default;
            case Pinned_Flag::Mapped: return &pinned_mapped;
            case Pinned_Flag::Portable: return &pinned_portable;
            case Pinned_Flag::WriteCombined: return &pinned_wc;
            default: return &pinned_portable;
        }
    }

    Allocator* AllocatorRegistry::get_cuda_allocator() {
        return &cuda_allocator;
        // Need to change 
        // &CachingAllocator::instance;
    }

    Allocator* AllocatorRegistry::get_caching_allocator()
    {
        return &CachingCUDAAllocator::instance();
    }

}