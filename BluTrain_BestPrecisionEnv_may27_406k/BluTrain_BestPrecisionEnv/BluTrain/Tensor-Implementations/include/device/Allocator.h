#pragma once
#include <cstddef>
//✨✨✨
#ifdef WITH_CUDA
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#endif
namespace OwnTensor
{
    class Allocator 
    {
        public:
            virtual ~Allocator() = default;
            virtual void* allocate(size_t bytes) = 0;
            virtual void deallocate(void* ptr) = 0;
    };
}