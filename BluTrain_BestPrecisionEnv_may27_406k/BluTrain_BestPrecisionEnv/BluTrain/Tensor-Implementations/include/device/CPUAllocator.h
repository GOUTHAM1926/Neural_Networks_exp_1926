#pragma once
#include "device/Allocator.h"

namespace OwnTensor
{
    class CPUAllocator : public Allocator {
    public:
        void* allocate(size_t bytes) override;
        void deallocate(void* ptr) override;

        struct MemoryStats {
            size_t allocated = 0;
            size_t peak = 0;
        };
        static MemoryStats get_stats();

    private:
        static void record_alloc(size_t bytes);
        static void record_free(size_t bytes);
    };
}