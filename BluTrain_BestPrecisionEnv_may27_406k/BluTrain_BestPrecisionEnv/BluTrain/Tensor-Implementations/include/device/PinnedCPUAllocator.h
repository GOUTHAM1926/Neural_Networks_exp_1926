#pragma once
#include "device/Allocator.h"
#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace OwnTensor {
namespace device {

    class PinnedCPUAllocator : public Allocator {
    public:
        // Use explicit constructor to avoid implicit conversions
        explicit PinnedCPUAllocator(unsigned int flags = 0);
        
        void* allocate(size_t bytes) override;
        void deallocate(void* ptr) override;

        struct MemoryStats {
            size_t allocated = 0;
            size_t peak = 0;
        };
        static MemoryStats get_stats();
    private:
        unsigned int flags_;
    };

} // namespace device
} // namespace OwnTensor
