#pragma once
#include "device/Allocator.h"

namespace OwnTensor
{
    class CUDAAllocator : public Allocator
    {
        public:
            void* allocate(size_t bytes) override;
            void deallocate(void* ptr) override;

    };
}