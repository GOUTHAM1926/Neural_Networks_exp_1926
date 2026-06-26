#include "device/DeviceTransfer.h"
#include <stdexcept>
#include <cstring>
#ifdef WITH_CUDA
#include <cuda_runtime.h>
#include "device/DeviceCore.h"
#endif

namespace OwnTensor
{
    namespace device {
        // edits  — deduped. The Device-only overload now
        // forwards to the DeviceIndex overload using the current CUDA device
        // index for any CUDA endpoints. CPU is index 0 (unused on CPU side).
        // Single source of truth for transfer logic; matches PyTorch's design
        // where Device always carries an index — they never had two overloads.
        void copy_memory(void* dst, Device dst_device,
                         const void* src, Device src_device,
                         size_t bytes) {
            int idx = 0;
#ifdef WITH_CUDA
            if (dst_device == Device::CUDA || src_device == Device::CUDA) {
                cudaGetDevice(&idx);
            }
#endif
            DeviceIndex dst_di{ dst_device, dst_device == Device::CUDA ? idx : 0 };
            DeviceIndex src_di{ src_device, src_device == Device::CUDA ? idx : 0 };
            copy_memory(dst, dst_di, src, src_di, bytes);
        }
        void copy_memory(void* dst, DeviceIndex dst_device,
            const void* src, DeviceIndex src_device,
                        size_t bytes) {

                            if (bytes == 0) {
                                return;
                            }

                            // CPU → CPU: std::memcpy (inherently synchronous, no CUDA involved)
                            if (dst_device.device == Device::CPU && src_device.device == Device::CPU) {
                                std::memcpy(dst, src, bytes);
                                return;
                            }

                            #ifdef WITH_CUDA
                            // GPU → GPU: cudaMemcpyAsync (async, stream-ordered)
                            if (dst_device.device == Device::CUDA && src_device.device == Device::CUDA) {
                                if (dst_device.index == src_device.index) {
                                    OwnTensor::device::set_cuda_device(dst_device.index);
                                    cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
                                    cudaError_t err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream);
                                    if (err != cudaSuccess) {
                                        throw std::runtime_error(std::string("GPU→GPU transfer failed: ") +
                                        cudaGetErrorString(err));
                                    }
                                } else {
                                    // Cross-device transfer
                                    OwnTensor::device::set_cuda_device(dst_device.index);
                                    cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
                                    cudaError_t err = cudaMemcpyPeerAsync(dst, dst_device.index, src, src_device.index, bytes, stream);
                                    if (err != cudaSuccess) {
                                        throw std::runtime_error(std::string("Cross-GPU transfer failed: ") +
                                        cudaGetErrorString(err));
                                    }
                                }
                                return;
                            }

                            // CPU → GPU: cudaMemcpyAsync (async, stream-ordered)
                            if (dst_device.device == Device::CUDA && src_device.device == Device::CPU) {
                                OwnTensor::device::set_cuda_device(dst_device.index);
                                cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
                                cudaError_t err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream);
                                if (err != cudaSuccess) {
                                    throw std::runtime_error(std::string("CPU→GPU transfer failed: ") +
                                    cudaGetErrorString(err));
                                }
                                return;
                            }

                            // GPU → CPU: cudaMemcpyAsync (async, stream-ordered)
                            if (dst_device.device == Device::CPU && src_device.device == Device::CUDA) {
                                // For D2H, we must set to the source GPU device
                                OwnTensor::device::set_cuda_device(src_device.index);
                                cudaStream_t stream = OwnTensor::cuda::getCurrentStream();
                                cudaError_t err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream);
                                if (err != cudaSuccess) {
                                    throw std::runtime_error(std::string("GPU→CPU transfer failed: ") +
                                    cudaGetErrorString(err));
                                }
                                return;
                            }
                            #endif

            throw std::runtime_error("Unsupported device transfer");
        }
    }
}
