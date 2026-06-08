#include "device/DeviceSet.h"
#include <cstring>
#include <stdexcept>
#ifdef WITH_CUDA
#include <cuda_runtime.h>
#include "device/DeviceCore.h"
#endif

namespace OwnTensor
{
    namespace device {
        void set_memory(void* ptr, Device device, int value, size_t bytes) {
            
            if (bytes == 0) {
                return;
            }

            // CPU: std::memset (inherently synchronous)
            if (device == Device::CPU) {
                std::memset(ptr, value, bytes);
                return;
            }

    #ifdef WITH_CUDA
            // GPU: cudaMemsetAsync (async, stream-ordered)
            cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
            cudaError_t err = cudaMemsetAsync(ptr, value, bytes, stream);
            if (err != cudaSuccess) {
                throw std::runtime_error(std::string("GPU memset failed: ") + 
                                       cudaGetErrorString(err));
            }
    #else
            throw std::runtime_error("CUDA not available for GPU memset");
    #endif
        }
    }
}
