#pragma once
#include "device/Allocator.h"
#include "device/Device.h"

namespace OwnTensor
{
    enum class Pinned_Flag {
        None,
        Default,
        Mapped,
        Portable,
        WriteCombined
    };

    class AllocatorRegistry {
    public:
        static Allocator* get_allocator(Device device);
        static Allocator* get_cpu_allocator();
        static Allocator* get_pinned_cpu_allocator(Pinned_Flag flag);
        static Allocator* get_cuda_allocator();
        static Allocator* get_caching_allocator();
    };
}