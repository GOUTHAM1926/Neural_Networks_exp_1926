#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <cstddef>
#include <memory>

namespace OwnTensor {

/**
 * @brief A fast bump-pointer allocator for GPU memory.
 * 
 * CudaArena manages large blocks of GPU memory allocated from CachingCUDAAllocator.
 * It provides extremely fast sub-allocations by incrementing a pointer.
 * This is ideal for short-lived temporary buffers within a kernel or a step.
 */
class CudaArena {
public:
    struct Block {
        void* ptr;
        size_t size;
        size_t offset;

        Block(size_t sz, cudaStream_t stream = nullptr);
        ~Block();
        
        // Disable copying
        Block(const Block&) = delete;
        Block& operator=(const Block&) = delete;
        
        // Enable move
        Block(Block&& other) noexcept;
        Block& operator=(Block&& other) noexcept;
    };

private:
    std::vector<Block> blocks_;
    size_t current_block_idx_ = 0;
    size_t default_block_size_;

public:
    /**
     * @brief Construct a CudaArena.
     * @param block_size Default size for new GPU blocks (default 32MB).
     */
    explicit CudaArena(size_t block_size = 32 * 1024 * 1024);
    ~CudaArena() = default;

    /**
     * @brief Allocate GPU memory from the arena.
     * @param size Size in bytes.
     * @param stream CUDA stream for the allocation.
     * @param align Alignment in bytes.
     */
    void* allocate(size_t size, cudaStream_t stream = nullptr, size_t align = 256);

    /**
     * @brief Reset all blocks for reuse. Does not return memory to allocator.
     */
    void reset();

    /**
     * @brief Return all blocks to the CachingCUDAAllocator.
     */
    void clear();

    /**
     * @brief Get total bytes currently reserved from the OS.
     */
    size_t total_reserved_bytes() const;

    /**
     * @brief Get total bytes currently allocated (active) within the blocks.
     */
    size_t total_allocated_bytes() const;

    /**
     * @brief Get a thread-local instance of CudaArena.
     */
    static CudaArena& get_thread_local();
};

} // namespace OwnTensor
