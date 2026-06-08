#include "checkpointing/RNG.h"
#include <sstream>
#include <stdexcept>
#include <iostream>

namespace OwnTensor {

thread_local std::unique_ptr<std::mt19937> RNG::cpu_gen_ = nullptr;
#ifdef WITH_CUDA
thread_local CurandGeneratorHandle RNG::gpu_handle_;
thread_local unsigned long long RNG::gpu_seed_ = 1234ULL;
thread_local unsigned long long RNG::gpu_offset_ = 0ULL;
#endif

std::mt19937& RNG::get_cpu_generator() {
    if (!cpu_gen_) {
        cpu_gen_ = std::make_unique<std::mt19937>(5489u);  // mt19937 default seed
    }
    return *cpu_gen_;
}
#ifdef WITH_CUDA
curandGenerator_t RNG::get_gpu_generator() {
    if (!gpu_handle_.initialized) {
        curandCreateGenerator(&gpu_handle_.gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gpu_handle_.gen, gpu_seed_);
        curandSetGeneratorOffset(gpu_handle_.gen, gpu_offset_);
        gpu_handle_.initialized = true;
    }
    return gpu_handle_.gen;
}

unsigned long long RNG::get_gpu_seed() {
    return gpu_seed_;
}

unsigned long long RNG::get_gpu_offset_and_advance(size_t count) {
    unsigned long long offset = gpu_offset_;
    gpu_offset_ += count;
    return offset;
}

void RNG::increment_gpu_offset(size_t count) {
    gpu_offset_ += count;
}
#endif

RNGState RNG::get_state() {
    RNGState state;
    auto& gen = get_cpu_generator();
    std::ostringstream oss;
    oss << gen;
    state.cpu_state = oss.str();
#ifdef WITH_CUDA
    state.gpu_seed = gpu_seed_;
    state.gpu_offset = gpu_offset_;
#endif
    return state;
}

void RNG::set_state(const RNGState& state) {
    auto& gen = get_cpu_generator();
    std::istringstream iss(state.cpu_state);
    iss >> gen;
#ifdef WITH_CUDA
    gpu_seed_ = state.gpu_seed;
    gpu_offset_ = state.gpu_offset;
    if (gpu_handle_.initialized) {
        curandSetPseudoRandomGeneratorSeed(gpu_handle_.gen, gpu_seed_);
        curandSetGeneratorOffset(gpu_handle_.gen, gpu_offset_);
    }
#endif
}

void RNG::set_seed(unsigned long seed) {
    get_cpu_generator().seed(seed);
#ifdef WITH_CUDA
    gpu_seed_ = static_cast<unsigned long long>(seed);
    gpu_offset_ = 0ULL;
    if (gpu_handle_.initialized) {
        curandSetPseudoRandomGeneratorSeed(gpu_handle_.gen, gpu_seed_);
        curandSetGeneratorOffset(gpu_handle_.gen, gpu_offset_);
    }
#endif
}

} // namespace OwnTensor