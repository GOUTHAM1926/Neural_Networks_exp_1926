#pragma once
#include <cstddef>

namespace OwnTensor
{
    struct CachingAllocatorConfig {
        float max_split_size_mb = 512;
        bool enabled = true;
        float gc_threshold = 0.9;
        static CachingAllocatorConfig& instance() {
            static CachingAllocatorConfig config;
            return config;
        }
    };
}