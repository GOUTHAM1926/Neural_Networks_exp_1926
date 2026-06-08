#include "device/CudaArena.h"
#include "device/CachingCudaAllocator.h"
#include <algorithm>

namespace OwnTensor {

// --- Block Implementation ---

CudaArena::Block::Block(size_t sz, cudaStream_t stream) : size(sz), offset(0) {
    ptr = CachingCUDAAllocator::instance().allocate(size, stream);
}

CudaArena::Block::~Block() {
    if (ptr) {
        CachingCUDAAllocator::instance().deallocate(ptr);
    }
}

CudaArena::Block::Block(Block&& other) noexcept 
    : ptr(other.ptr), size(other.size), offset(other.offset) {
    other.ptr = nullptr;
    other.size = 0;
    other.offset = 0;
}

CudaArena::Block& CudaArena::Block::operator=(Block&& other) noexcept {
    if (this != &other) {
        if (ptr) CachingCUDAAllocator::instance().deallocate(ptr);
        ptr = other.ptr;
        size = other.size;
        offset = other.offset;
        other.ptr = nullptr;
        other.size = 0;
        other.offset = 0;
    }
    return *this;
}

// --- CudaArena Implementation ---

CudaArena::CudaArena(size_t block_size) : default_block_size_(block_size) {
    // Initial block is deferred until first allocation to avoid unnecessary GPU memory use
}

void* CudaArena::allocate(size_t size, cudaStream_t stream, size_t align) {
    if (blocks_.empty()) {
        blocks_.emplace_back(std::max(default_block_size_, size + align), stream);
    }

    auto try_alloc = [&](Block& b) -> void* {
        uintptr_t base_addr = reinterpret_cast<uintptr_t>(b.ptr);
        uintptr_t current_addr = base_addr + b.offset;
        uintptr_t aligned_addr = (current_addr + align - 1) & ~(align - 1);
        size_t new_offset = (aligned_addr - base_addr) + size;
        
        if (new_offset <= b.size) {
            b.offset = new_offset;
            return reinterpret_cast<void*>(aligned_addr);
        }
        return nullptr;
    };

    // 1. Try current block
    void* p = try_alloc(blocks_[current_block_idx_]);
    if (p) return p;

    // 2. Try subsequent existing blocks (if any)
    if (current_block_idx_ + 1 < blocks_.size()) {
        current_block_idx_++;
        // Reset offset of reused block just in case it wasn't reset (though reset() handles it)
        // Actually, we should only reset if we are moving forward after a reset() call.
        // If we are here, it means the current block is full.
        p = try_alloc(blocks_[current_block_idx_]);
        if (p) return p;
    }

    // 3. Allocate a new block
    size_t new_size = std::max(default_block_size_, size + align);
    blocks_.emplace_back(new_size, stream);
    current_block_idx_ = blocks_.size() - 1;
    
    return try_alloc(blocks_[current_block_idx_]);
}

void CudaArena::reset() {
    for (auto& b : blocks_) {
        b.offset = 0;
    }
    current_block_idx_ = 0;
}

void CudaArena::clear() {
    blocks_.clear();
    current_block_idx_ = 0;
}

CudaArena& CudaArena::get_thread_local() {
    thread_local CudaArena arena;
    return arena;
}

size_t CudaArena::total_reserved_bytes() const {
    size_t total = 0;
    for (const auto& block : blocks_) {
        total += block.size;
    }
    return total;
}

size_t CudaArena::total_allocated_bytes() const {
    size_t total = 0;
    for (const auto& block : blocks_) {
        total += block.offset;
    }
    return total;
}

} // namespace OwnTensor
