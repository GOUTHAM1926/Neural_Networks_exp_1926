#pragma once
#include <cuda_runtime.h>
#include <mutex>
#include <array>

namespace OwnTensor {
namespace cuda {

// GPU architecture families, ordered by compute capability.
// Int values match SM major*10+minor for easy range comparisons.
enum class ArchFamily : int {
    Generic     = 0,   // unknown / fallback
    Ampere_A100 = 80,  // sm_80: A100, A30
    Ampere_A10  = 86,  // sm_86: RTX 3060/3080/3090, A10, A40  ← local dev
    Ada         = 89,  // sm_89: A6000 Ada, RTX 4090, L40S     ← server
    Hopper      = 90,  // sm_90: H100
    Blackwell   = 100, // sm_100: B100/B200 (future)
};

inline ArchFamily arch_family_from_sm(int major, int minor) {
    switch (major * 10 + minor) {
        case 80:  return ArchFamily::Ampere_A100;
        case 86:  return ArchFamily::Ampere_A10;
        case 89:  return ArchFamily::Ada;
        case 90:  return ArchFamily::Hopper;
        case 100: return ArchFamily::Blackwell;
        default:  return ArchFamily::Generic;
    }
}

// Returns the architecture family of the given CUDA device.
// Thread-safe. After the first call per device, cost is a single array lookup.
// Pass device_idx = -1 to auto-detect the current CUDA device.
inline ArchFamily get_arch(int device_idx = -1) {
    static std::array<ArchFamily, 8> cache{};
    static std::array<bool, 8>       ready{};
    static std::mutex                mtx;

    if (device_idx < 0) {
        cudaGetDevice(&device_idx);
    }

    if (!ready[device_idx]) {
        std::lock_guard<std::mutex> lock(mtx);
        if (!ready[device_idx]) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, device_idx);
            cache[device_idx] = arch_family_from_sm(prop.major, prop.minor);
            ready[device_idx] = true;
        }
    }
    return cache[device_idx];
}

} // namespace cuda
} // namespace OwnTensor
