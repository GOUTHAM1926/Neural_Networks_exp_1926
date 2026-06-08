#include "autograd/GraphArena.h"
#include <algorithm>

namespace OwnTensor {
namespace autograd {

GraphArena::GraphArena(size_t block_size) : default_block_size_(block_size) {
    blocks_.emplace_back(default_block_size_);
}

void* GraphArena::allocate(size_t size, size_t align) {
    if (blocks_.empty()) {
        blocks_.emplace_back(default_block_size_);
    }

    auto try_allocate = [&](Block& block) -> void* {
        uintptr_t base_addr = reinterpret_cast<uintptr_t>(block.data.get());
        uintptr_t current_addr = base_addr + block.offset;
        uintptr_t aligned_addr = (current_addr + align - 1) & ~(align - 1);
        size_t new_offset = (aligned_addr - base_addr) + size;

        if (new_offset <= block.size) {
            block.offset = new_offset;
            return reinterpret_cast<void*>(aligned_addr);
        }
        return nullptr;
    };

    // 1. Try to fit in current block
    void* ptr = try_allocate(blocks_[current_block_idx_]);
    if (ptr) return ptr;

    // 2. Try to move to the next existing block
    if (current_block_idx_ + 1 < blocks_.size()) {
        current_block_idx_++;
        ptr = try_allocate(blocks_[current_block_idx_]);
        if (ptr) return ptr;
    }

    // 3. Need to allocate a new block
    size_t new_block_size = std::max(default_block_size_, size + align);
    blocks_.emplace_back(new_block_size);
    current_block_idx_ = blocks_.size() - 1;
    
    return try_allocate(blocks_[current_block_idx_]);
}

void GraphArena::reset() {
    for (auto& block : blocks_) {
        block.offset = 0;
    }
    current_block_idx_ = 0;
}

void GraphArena::clear() {
    blocks_.clear();
    current_block_idx_ = 0;
}

size_t GraphArena::total_reserved_bytes() const {
    size_t total = 0;
    for (const auto& block : blocks_) {
        total += block.size;
    }
    return total;
}

size_t GraphArena::total_allocated_bytes() const {
    size_t total = 0;
    for (const auto& block : blocks_) {
        total += block.offset;
    }
    return total;
}

GraphArena& GraphArena::get_thread_local() {
    thread_local GraphArena arena;
    return arena;
}

} // namespace autograd
} // namespace OwnTensor
