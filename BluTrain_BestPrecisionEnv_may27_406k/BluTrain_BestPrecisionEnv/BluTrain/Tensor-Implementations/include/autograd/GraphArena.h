#pragma once

#include <cstddef>
#include <vector>
#include <memory>

namespace OwnTensor {
namespace autograd {

/**
 * @brief A fast bump-pointer allocator for autograd graph objects (Nodes, Edges).
 * 
 * This allocator reserves large blocks of memory and sub-allocates from them
 * using a simple pointer increment. This avoids the overhead of repeated 
 * heap allocations (malloc/new) for thousands of small graph objects.
 * 
 * All memory is "freed" and reused by simply resetting the pointer back to
 * the start of the blocks, typically at the end of each training step.
 */
class GraphArena {
public:
    struct Block {
        std::unique_ptr<uint8_t[]> data;
        size_t size;
        size_t offset;

        Block(size_t sz) : data(new uint8_t[sz]), size(sz), offset(0) {}
    };

private:
    std::vector<Block> blocks_;
    size_t current_block_idx_ = 0;
    size_t default_block_size_;

public:
    explicit GraphArena(size_t block_size = 64 * 1024 * 1024); // Default 64MB
    ~GraphArena() = default;

    /**
     * @brief Allocate memory from the arena.
     * @param size Size in bytes.
     * @param align Alignment required (default for pointers).
     */
    void* allocate(size_t size, size_t align = alignof(std::max_align_t));

    /**
     * @brief Reset all blocks for reuse. Does not deallocate from OS.
     */
    void reset();

    /**
     * @brief Release all memory back to the OS.
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
     * @brief Get thread-local instance for safe concurrent graph construction.
     */
    static GraphArena& get_thread_local();
};

} // namespace autograd
} // namespace OwnTensor
