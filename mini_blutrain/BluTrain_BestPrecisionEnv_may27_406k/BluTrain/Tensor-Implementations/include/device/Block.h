#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <unordered_set>

namespace OwnTensor {
struct Block {
  // Memory Info
  void *ptr;       // GPU memory pointer
  size_t size;     // Actual Block size (after rounding)
  size_t req_size; // Actual requested size

  // Pool
  int device_id;       // GPU device index
  cudaStream_t stream; // CUDA stream for this allocation

  // for block splitting
  Block *prev; // Previous block in segment
  Block *next; // Next Block in segment

  // state of the block
  bool allocated; // Currently in use?
  bool is_split;  // Part of a split segment

  uint64_t alloc_id; // Unique allocation id for tracking

  /**
   * Need to add:
   *      - stream_uses - for multi-stream tracking
   *      - event_count - for deferred freeing (blocks are freed only after a
   * event ends)
   */
  std::unordered_set<cudaStream_t> recorded_streams;
  int event_count = 0; // Outstanding CUDA events guarding deallocation

  Block(void *p, size_t s, int dev, cudaStream_t str)
      : ptr(p), size(s), req_size(s), device_id(dev), stream(str),
        prev(nullptr), next(nullptr), allocated(false), is_split(false),
        alloc_id(0), event_count(0) {}
};

// for requesting block by size (size ordered lookup, ordered by size and then
// address)
struct BlockSizeComparator {
  bool operator()(const Block *a, const Block *b) const {
    if (a->size != b->size)
      return a->size < b->size;
    return a->ptr < b->ptr;
  }
};

// for requesting block by pointer (address ordered look up - for
// coalescing/merging)
struct BlockPtrComparator {
  bool operator()(const Block *a, const Block *b) const {
    return a->ptr < b->ptr;
  }
};
} // namespace OwnTensor
