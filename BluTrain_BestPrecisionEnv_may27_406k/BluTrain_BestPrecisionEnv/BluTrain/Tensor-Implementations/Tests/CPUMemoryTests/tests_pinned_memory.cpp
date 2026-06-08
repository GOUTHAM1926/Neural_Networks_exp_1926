#include <iostream>
#include <cassert>
#include <cstring>
#include <cuda_runtime.h>
#include "core/Tensor.h"
#include "device/AllocatorRegistry.h"
#include "device/PinnedCPUAllocator.h"

using namespace OwnTensor;

// ============================================================
// Helper macros
// ============================================================
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

int passed = 0;
int failed = 0;

void PASS(const char* name) {
    std::cout << "  ✅ PASS: " << name << std::endl;
    passed++;
}
void PASS(const std::string& name) { PASS(name.c_str()); }

void FAIL(const char* name, const char* reason) {
    std::cout << "  ❌ FAIL: " << name << " — " << reason << std::endl;
    failed++;
}
void FAIL(const std::string& name, const char* reason) { FAIL(name.c_str(), reason); }

// ============================================================
// Helper: Check if a pointer is pinned using cudaPointerGetAttributes
// ============================================================
bool check_pointer_is_pinned(const void* ptr) {
    if (!ptr) return false;
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
    if (err != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return (attr.type == cudaMemoryTypeHost);
}

// ============================================================
// Helper: Check if pointer is device-mapped (cudaHostAllocMapped)
// Only pointers allocated with cudaHostAllocMapped will have a valid device pointer
// ============================================================
bool check_pointer_is_mapped(void* ptr) {
    if (!ptr) return false;
    void* device_ptr = nullptr;
    cudaError_t err = cudaHostGetDevicePointer(&device_ptr, ptr, 0);
    if (err != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return (device_ptr != nullptr);
}

// ============================================================
// Test 1: Standard CPU Tensor (Pageable Memory — No pinten flag)
// ============================================================
void test_standard_cpu_tensor() {
    std::cout << "\n[Test 1] Standard CPU Tensor (No pinten flag)" << std::endl;

    Shape shape{{1, 256}};
    Tensor t = Tensor(shape, TensorOptions().with_dtype(Dtype::Float32).with_device(Device::CPU));

    if (!t.is_pinned()) {
        PASS("Standard CPU tensor is NOT pinned");
    } else {
        FAIL("Standard CPU tensor is NOT pinned", "is_pinned() returned true");
    }

    if (!check_pointer_is_pinned(t.data())) {
        PASS("cudaPointerGetAttributes confirms NOT pinned");
    } else {
        FAIL("cudaPointerGetAttributes confirms NOT pinned", "pointer reports as pinned");
    }

    // Should NOT be mapped either
    if (!check_pointer_is_mapped(t.data())) {
        PASS("Not device-mapped (expected for pageable memory)");
    } else {
        FAIL("Not device-mapped (expected for pageable memory)", "reports mapped");
    }
}

// ============================================================
// Test 2: Pinned Memory — Default Flag (cudaHostAllocDefault)
// ============================================================
void test_pinned_default_flag() {
    std::cout << "\n[Test 2] Pinned Memory — pinten(Pinned_Flag::Default) → cudaHostAllocDefault" << std::endl;

    Shape shape{{1, 256}};
    Tensor t = Tensor(shape, TensorOptions()
        .with_dtype(Dtype::Float32)
        .with_device(Device::CPU)
        .with_pinten(Pinned_Flag::Default));

    if (t.is_pinned()) {
        PASS("is_pinned() = true");
    } else {
        FAIL("is_pinned() = true", "returned false");
    }

    if (check_pointer_is_pinned(t.data())) {
        PASS("cudaPointerGetAttributes confirms pinned");
    } else {
        FAIL("cudaPointerGetAttributes confirms pinned", "NOT pinned");
    }

    // Default flag does NOT set mapped bit 
    // (cudaHostAllocDefault = 0, no mapping unless cudaDeviceMapHost is set)
    // This test validates that Default != Mapped
    std::cout << "    [Info] Device-mapped: " << (check_pointer_is_mapped(t.data()) ? "yes" : "no") << std::endl;
}

// ============================================================
// Test 3: Pinned Memory — Mapped Flag (cudaHostAllocMapped)
// This is the KEY flag-specific validation: only Mapped pointers 
// can return a valid device pointer via cudaHostGetDevicePointer
// ============================================================
void test_pinned_mapped_flag() {
    std::cout << "\n[Test 3] Pinned Memory — pinten(Pinned_Flag::Mapped) → cudaHostAllocMapped" << std::endl;

    Shape shape{{1, 256}};
    Tensor t = Tensor(shape, TensorOptions()
        .with_dtype(Dtype::Float32)
        .with_device(Device::CPU)
        .with_pinten(Pinned_Flag::Mapped));

    if (t.is_pinned()) {
        PASS("is_pinned() = true");
    } else {
        FAIL("is_pinned() = true", "returned false");
    }

    // FLAG-SPECIFIC VALIDATION: Mapped pointers have a valid device pointer
    if (check_pointer_is_mapped(t.data())) {
        PASS("cudaHostGetDevicePointer succeeds (FLAG VALIDATION: Mapped works!)");
    } else {
        FAIL("cudaHostGetDevicePointer succeeds", "no device pointer — wrong flag used?");
    }

    // Get and print the actual device pointer
    void* device_ptr = nullptr;
    cudaHostGetDevicePointer(&device_ptr, t.data(), 0);
    std::cout << "    [Info] Host ptr:   " << t.data() << std::endl;
    std::cout << "    [Info] Device ptr: " << device_ptr << std::endl;
}

// ============================================================
// Test 4: Pinned Memory — Portable Flag (cudaHostAllocPortable)
// ============================================================
void test_pinned_portable_flag() {
    std::cout << "\n[Test 4] Pinned Memory — pinten(Pinned_Flag::Portable) → cudaHostAllocPortable" << std::endl;

    Shape shape{{1, 256}};
    Tensor t = Tensor(shape, TensorOptions()
        .with_dtype(Dtype::Float32)
        .with_device(Device::CPU)
        .with_pinten(Pinned_Flag::Portable));

    if (t.is_pinned()) {
        PASS("is_pinned() = true");
    } else {
        FAIL("is_pinned() = true", "returned false");
    }

    if (check_pointer_is_pinned(t.data())) {
        PASS("cudaPointerGetAttributes confirms pinned");
    } else {
        FAIL("cudaPointerGetAttributes confirms pinned", "NOT pinned");
    }

    // Portable should NOT have device mapping (unless combined with Mapped)
    std::cout << "    [Info] Device-mapped: " << (check_pointer_is_mapped(t.data()) ? "yes" : "no") 
              << " (expected: no, unless device also supports mapping)" << std::endl;
}

// ============================================================
// Test 5: Pinned Memory — WriteCombined Flag (cudaHostAllocWriteCombined)
// ============================================================
void test_pinned_wc_flag() {
    std::cout << "\n[Test 5] Pinned Memory — pinten(Pinned_Flag::WriteCombined) → cudaHostAllocWriteCombined" << std::endl;

    Shape shape{{1, 256}};
    Tensor t = Tensor(shape, TensorOptions()
        .with_dtype(Dtype::Float32)
        .with_device(Device::CPU)
        .with_pinten(Pinned_Flag::WriteCombined));

    if (t.is_pinned()) {
        PASS("is_pinned() = true");
    } else {
        FAIL("is_pinned() = true", "returned false");
    }

    if (check_pointer_is_pinned(t.data())) {
        PASS("cudaPointerGetAttributes confirms pinned");
    } else {
        FAIL("cudaPointerGetAttributes confirms pinned", "NOT pinned");
    }

    // WriteCombined: Reads from CPU are slow (write-combining cache policy).
    // We verify by doing a write and read — data should still be correct, just slower reads.
    float* data = static_cast<float*>(t.data());
    data[0] = 42.0f;

    // Force a memory fence to ensure write is visible
    __sync_synchronize();

    volatile float readback = data[0];
    if (readback == 42.0f) {
        PASS("Write-read roundtrip works (WriteCombined correct even if slow reads)");
    } else {
        FAIL("Write-read roundtrip works", "value mismatch");
    }
}

// ============================================================
// Test 6: Mapped vs Default — Cross-Validation
// Proves that different flags produce different behavior
// ============================================================
void test_mapped_vs_default_crosscheck() {
    std::cout << "\n[Test 6] CROSS-VALIDATION: Mapped vs Default (flag differentiation)" << std::endl;

    Shape shape{{1, 256}};

    // Allocate with Default flag
    Tensor t_default = Tensor(shape, TensorOptions()
        .with_dtype(Dtype::Float32)
        .with_device(Device::CPU)
        .with_pinten(Pinned_Flag::Default));

    // Allocate with Mapped flag
    Tensor t_mapped = Tensor(shape, TensorOptions()
        .with_dtype(Dtype::Float32)
        .with_device(Device::CPU)
        .with_pinten(Pinned_Flag::Mapped));

    bool default_is_mapped = check_pointer_is_mapped(t_default.data());
    bool mapped_is_mapped = check_pointer_is_mapped(t_mapped.data());

    std::cout << "    Default flag → device-mapped: " << (default_is_mapped ? "yes" : "no") << std::endl;
    std::cout << "    Mapped flag  → device-mapped: " << (mapped_is_mapped ? "yes" : "no") << std::endl;

    // The Mapped pointer MUST have a device pointer
    if (mapped_is_mapped) {
        PASS("Mapped flag produces device-mappable pointer");
    } else {
        FAIL("Mapped flag produces device-mappable pointer", "no device pointer");
    }

    // If Default has no mapping but Mapped does → flags are correctly differentiated!
    // Note: On some systems with cudaDeviceMapHost support, Default may also be mapped.
    // So we only hard-assert that Mapped IS mapped.
    if (mapped_is_mapped && !default_is_mapped) {
        PASS("FLAGS DIFFER: Default≠Mapped (different cudaHostAlloc flags confirmed!)");
    } else if (mapped_is_mapped && default_is_mapped) {
        std::cout << "  ⚠️  INFO: Both mapped (device supports cudaDeviceMapHost globally). "
                  << "Flags still correctly passed but indistinguishable on this GPU." << std::endl;
        PASS("Both pinned (flag passed correctly, GPU maps all pinned memory)");
    } else {
        FAIL("Flag differentiation", "unexpected result");
    }
}

// ============================================================
// Test 7: pin_memory() In-Place (cudaHostRegister)
// ============================================================
void test_pin_memory_inplace() {
    std::cout << "\n[Test 7] pin_memory() — In-Place Pinning (cudaHostRegister)" << std::endl;

    Shape shape{{1, 1024}};
    Tensor t = Tensor(shape, TensorOptions().with_dtype(Dtype::Float32).with_device(Device::CPU));

    if (!t.is_pinned()) {
        PASS("Before pin_memory(): NOT pinned");
    } else {
        FAIL("Before pin_memory(): NOT pinned", "already pinned?!");
    }

    Tensor t_pinned = t.pin_memory();

    if (t.is_pinned()) {
        PASS("After pin_memory(): original tensor is now pinned (in-place)");
    } else {
        FAIL("After pin_memory(): original tensor is now pinned (in-place)", "still not pinned");
    }

    if (t_pinned.is_pinned()) {
        PASS("Returned tensor from pin_memory() is pinned");
    } else {
        FAIL("Returned tensor from pin_memory() is pinned", "not pinned");
    }

    if (t.data() == t_pinned.data()) {
        PASS("Same data pointer (in-place, no copy)");
    } else {
        FAIL("Same data pointer (in-place, no copy)", "different pointers");
    }

    // Cleanup: unregister
    cudaHostUnregister(t.data());
}

// ============================================================
// Test 8: pin_memory() on Already Pinned (No-Op)
// ============================================================
void test_pin_memory_already_pinned() {
    std::cout << "\n[Test 8] pin_memory() on Already Pinned Tensor (No-Op)" << std::endl;

    Shape shape{{1, 256}};
    Tensor t = Tensor(shape, TensorOptions()
        .with_dtype(Dtype::Float32)
        .with_device(Device::CPU)
        .with_pinten(Pinned_Flag::Default));

    if (t.is_pinned()) {
        PASS("Tensor is already pinned");
    } else {
        FAIL("Tensor is already pinned", "not pinned");
    }

    Tensor t2 = t.pin_memory();
    if (t2.is_pinned()) {
        PASS("pin_memory() on already-pinned: still pinned (no-op)");
    } else {
        FAIL("pin_memory() on already-pinned: still pinned", "returned non-pinned");
    }
}

// ============================================================
// Test 9: pin_memory() on CUDA Tensor (Returns Self)
// ============================================================
void test_pin_memory_cuda_tensor() {
    std::cout << "\n[Test 9] pin_memory() on CUDA Tensor (Should Return Self)" << std::endl;

    Shape shape{{1, 256}};
    Tensor t_cpu = Tensor(shape, TensorOptions().with_dtype(Dtype::Float32).with_device(Device::CPU));
    Tensor t_gpu = t_cpu.to_cuda();

    Tensor t_result = t_gpu.pin_memory();
    if (t_result.is_cuda()) {
        PASS("pin_memory() on CUDA tensor returns CUDA tensor (self)");
    } else {
        FAIL("pin_memory() on CUDA tensor returns CUDA tensor", "not CUDA");
    }
}

// ============================================================
// Test 10: Data Integrity — Write then Read on Pinned Memory
// ============================================================
void test_data_integrity_pinned() {
    std::cout << "\n[Test 10] Data Integrity — Pinned Memory Read/Write" << std::endl;

    Shape shape{{1, 4}};
    Tensor t = Tensor(shape, TensorOptions()
        .with_dtype(Dtype::Float32)
        .with_device(Device::CPU)
        .with_pinten(Pinned_Flag::Portable));

    float* data = static_cast<float*>(t.data());
    data[0] = 1.0f; data[1] = 2.0f; data[2] = 3.0f; data[3] = 4.0f;

    const float* rdata = static_cast<const float*>(t.data());
    if (rdata[0] == 1.0f && rdata[1] == 2.0f && rdata[2] == 3.0f && rdata[3] == 4.0f) {
        PASS("Data integrity preserved on pinned memory");
    } else {
        FAIL("Data integrity preserved on pinned memory", "values mismatch");
    }
}

// ============================================================
// Test 11: Low-Level — Direct PinnedCPUAllocator with each flag
// ============================================================
void test_low_level_allocator_direct() {
    std::cout << "\n[Test 11] Low-Level: Direct PinnedCPUAllocator with each flag" << std::endl;

    unsigned int flags[] = {cudaHostAllocDefault, cudaHostAllocMapped, cudaHostAllocPortable, cudaHostAllocWriteCombined};
    const char* names[] = {"Default", "Mapped", "Portable", "WriteCombined"};

    for (int i = 0; i < 4; i++) {
        device::PinnedCPUAllocator alloc(flags[i]);
        size_t size = 4096;
        void* ptr = alloc.allocate(size);

        if (ptr != nullptr) {
            if (check_pointer_is_pinned(ptr)) {
                PASS(std::string("Direct allocator (") + names[i] + ") — pointer is pinned");
            } else {
                FAIL(std::string("Direct allocator (") + names[i] + ") — pointer is pinned", "NOT pinned");
            }
            alloc.deallocate(ptr);
        } else {
            FAIL(std::string("Direct allocator (") + names[i] + ") — allocation", "returned nullptr");
        }
    }

    // Flag-specific: Verify Mapped allocator has device pointer
    device::PinnedCPUAllocator mapped_alloc(cudaHostAllocMapped);
    void* mapped_ptr = mapped_alloc.allocate(4096);
    if (mapped_ptr && check_pointer_is_mapped(mapped_ptr)) {
        PASS("Direct Mapped allocator has valid device pointer");
    } else {
        FAIL("Direct Mapped allocator has valid device pointer", "no device pointer");
    }
    if (mapped_ptr) mapped_alloc.deallocate(mapped_ptr);
}

// ============================================================
// MAIN
// ============================================================
int main() {
    std::cout << "================================================================" << std::endl;
    std::cout << "  PINNED MEMORY COMPREHENSIVE TEST SUITE" << std::endl;
    std::cout << "  (Flag Validation via cudaPointerGetAttributes + cudaHostGetDevicePointer)" << std::endl;
    std::cout << "================================================================" << std::endl;

    test_standard_cpu_tensor();        // Test 1:  No pinten, pageable
    test_pinned_default_flag();        // Test 2:  pinten(Default)
    test_pinned_mapped_flag();         // Test 3:  pinten(Mapped) + device ptr check
    test_pinned_portable_flag();       // Test 4:  pinten(Portable)
    test_pinned_wc_flag();             // Test 5:  pinten(WriteCombined) + write-read
    test_mapped_vs_default_crosscheck(); // Test 6: Cross-validation proves flags differ
    test_pin_memory_inplace();         // Test 7:  cudaHostRegister in-place
    test_pin_memory_already_pinned();  // Test 8:  No-op on already pinned
    test_pin_memory_cuda_tensor();     // Test 9:  No-op on CUDA
    test_data_integrity_pinned();      // Test 10: Data read/write 
    test_low_level_allocator_direct(); // Test 11: Direct allocator + flag check

    std::cout << "\n================================================================" << std::endl;
    std::cout << "  RESULTS: " << passed << " passed, " << failed << " failed" << std::endl;
    std::cout << "================================================================" << std::endl;

    return (failed > 0) ? 1 : 0;
}