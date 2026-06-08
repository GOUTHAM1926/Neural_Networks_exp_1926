#pragma once

#include "device/Block.h"
#include <iterator>
#include <set>
#include <unordered_map>
// #include <mutex>

namespace OwnTensor {
class BlockPool {
public:
  // Free Blocks ordered by size
  std::set<Block *, BlockSizeComparator> free_blocks;

  // All blocks by pointer for lookup
  std::unordered_map<void *, Block *> allocated_blocks;

  size_t total_allocated = 0;
  size_t total_cached = 0;
  size_t peak_allocated = 0;
  size_t total_active_requested = 0;

  // Lookup allocated block by pointer (assumes caller holds device lock)
  Block *find_allocated_block(void *ptr);

  // Find best-fit free block of at least `size` bytes on the SAME cuda stream
  Block *find_free_block(size_t size, cudaStream_t stream);

  // return block to pool
  void return_block(Block *block);

  // Coalesce / merge with adjacent free blocks on the SAME stream
  Block *try_block_merge(Block *block);
};
} // namespace OwnTensor
