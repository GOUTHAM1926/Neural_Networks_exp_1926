#pragma once

#include <cstddef>

namespace OwnTensor {

// Thin wrapper around NVML for querying per-process GPU memory.
// Used by benchmark mode to measure real GPU footprint per candidate config.
class NvmlMonitor {
public:
    // Initialize NVML. Returns false if init fails.
    bool init();

    // Shutdown NVML. Safe to call even if init() was not called.
    void shutdown();

    // Query GPU memory used by this process (bytes) on the given device.
    // Returns 0 if the process is not found or NVML query fails.
    size_t process_gpu_memory_bytes(int device = 0);

    bool is_initialized() const { return initialized_; }

private:
    bool initialized_ = false;
};

} // namespace OwnTensor
