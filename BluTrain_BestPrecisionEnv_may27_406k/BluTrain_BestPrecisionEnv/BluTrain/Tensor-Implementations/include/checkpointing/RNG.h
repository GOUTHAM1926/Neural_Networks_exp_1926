#pragma once

#include <random>
#include <vector>
#include <memory>
#include <string>

#ifdef WITH_CUDA
#include <curand.h>
#endif

namespace OwnTensor {

#ifdef WITH_CUDA
/**
 * @brief RAII wrapper for curandGenerator_t.
 *
 * Ensures curandDestroyGenerator is called on thread exit, preventing the
 * per-thread handle leak that occurs with a raw thread_local curandGenerator_t.
 */
struct CurandGeneratorHandle {
    curandGenerator_t gen{};
    bool initialized = false;

    ~CurandGeneratorHandle() {
        if (initialized) {
            curandDestroyGenerator(gen);
            initialized = false;
        }
    }

    // No copy
    CurandGeneratorHandle() = default;
    CurandGeneratorHandle(const CurandGeneratorHandle&) = delete;
    CurandGeneratorHandle& operator=(const CurandGeneratorHandle&) = delete;
};
#endif

/**
 * @brief Container for RNG states (CPU and GPU).
 */
struct RNGState {
    std::string cpu_state;   // serialized via std::mt19937's operator<<
#ifdef WITH_CUDA
    unsigned long long gpu_seed;
    unsigned long long gpu_offset;
#endif
};

/**
 * @brief Global RNG management.
 */
class RNG {
public:
    /**
     * @brief Get the thread-local CPU generator.
     */
    static std::mt19937& get_cpu_generator();

    /**
     * @brief Get the current RNG state for the current thread.
     */
    static RNGState get_state();

    /**
     * @brief Restore RNG state for the current thread.
     */
    static void set_state(const RNGState& state);

    /**
     * @brief Set the seed for all generators in the current thread.
     */
    static void set_seed(unsigned long seed);

#ifdef WITH_CUDA
    /**
     * @brief Get the thread-local curand generator (host-API bulk generation).
     */
    static curandGenerator_t get_gpu_generator();

    /**
     * @brief Return the current GPU seed for passing directly into device kernels.
     */
    static unsigned long long get_gpu_seed();

    /**
     * @brief Return the current GPU offset and atomically advance it by @p count.
     *
     * Device kernels must call this instead of reading gpu_offset_ directly so
     * that consecutive kernel launches receive non-overlapping Philox counter
     * ranges and the offset is correctly tracked for checkpoint capture/restore.
     *
     * @param count Number of Philox 128-bit output blocks consumed by the kernel
     *              (typically 1 for kernels that call curand_uniform once per thread).
     * @return The offset value the kernel should pass to curand_init().
     */
    static unsigned long long get_gpu_offset_and_advance(size_t count);

    /**
     * @brief Increment the GPU offset counter after generating random numbers.
     * @param count Number of random values generated.
     */
    static void increment_gpu_offset(size_t count);
#endif

private:
    static thread_local std::unique_ptr<std::mt19937> cpu_gen_;
#ifdef WITH_CUDA
    // CurandGeneratorHandle owns the curandGenerator_t and calls
    // curandDestroyGenerator in its destructor, preventing the per-thread leak
    // that would occur with a raw thread_local curandGenerator_t.
    static thread_local CurandGeneratorHandle gpu_handle_;
    static thread_local unsigned long long gpu_seed_;
    static thread_local unsigned long long gpu_offset_;
#endif
};

/**
 * @brief RAII guard to save and restore RNG state.
 */
class RNGStateGuard {
public:
    RNGStateGuard() : saved_state_(RNG::get_state()) {}
    ~RNGStateGuard() {
        RNG::set_state(saved_state_);
    }

    // No copy or move
    RNGStateGuard(const RNGStateGuard&) = delete;
    RNGStateGuard& operator=(const RNGStateGuard&) = delete;

private:
    RNGState saved_state_;
};

} // namespace OwnTensor