#include "device/NvmlMonitor.h"
#include <nvml.h>
#include <unistd.h>
#include <iostream>
#include <vector>

namespace OwnTensor {

bool NvmlMonitor::init() {
    nvmlReturn_t result = nvmlInit_v2();
    if (result != NVML_SUCCESS) {
        std::cerr << "[NvmlMonitor] nvmlInit failed: "
                  << nvmlErrorString(result) << "\n";
        initialized_ = false;
        return false;
    }
    initialized_ = true;
    return true;
}

void NvmlMonitor::shutdown() {
    if (initialized_) {
        nvmlShutdown();
        initialized_ = false;
    }
}

size_t NvmlMonitor::process_gpu_memory_bytes(int device) {
    if (!initialized_) return 0;

    nvmlDevice_t nvml_device;
    nvmlReturn_t result = nvmlDeviceGetHandleByIndex_v2(
        static_cast<unsigned int>(device), &nvml_device);
    if (result != NVML_SUCCESS) return 0;

    // Query running compute processes
    unsigned int info_count = 0;
    result = nvmlDeviceGetComputeRunningProcesses_v2(nvml_device, &info_count, nullptr);
    if (result != NVML_ERROR_INSUFFICIENT_SIZE && result != NVML_SUCCESS) return 0;

    if (info_count == 0) return 0;

    std::vector<nvmlProcessInfo_t> infos(info_count);
    result = nvmlDeviceGetComputeRunningProcesses_v2(nvml_device, &info_count, infos.data());
    if (result != NVML_SUCCESS) return 0;

    pid_t my_pid = getpid();
    for (unsigned int i = 0; i < info_count; i++) {
        if (static_cast<pid_t>(infos[i].pid) == my_pid) {
            return infos[i].usedGpuMemory;
        }
    }

    return 0;
}

} // namespace OwnTensor
