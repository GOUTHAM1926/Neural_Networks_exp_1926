#!/usr/bin/env python3
"""TEST HARNESS ONLY -- box copy, never commit.

Adds an [ALLOC] line after the trainer's end-of-run [MEM] line in bluscript_zero_ckpt.cpp, printing the
caching allocator's own counters (CachingCudaAllocator.h MemoryStats: num_allocs, num_frees, ...), so the
F2 A/B can count the fp32 temporaries it removes (expected: 156 allocations and 156 frees fewer per
micro-step after the first).

Usage: python3 harness_alloc_print.py <BluTrain dir>
"""
import os, sys

root = sys.argv[1] if len(sys.argv) > 1 else "."
path = os.path.join(root, "bluscript_zero_ckpt.cpp")
text = open(path, encoding="utf-8").read()

if "[ALLOC] allocs" in text:
    sys.exit("already has the [ALLOC] print -- nothing to do")

anchor = 'std::cout << "[MEM] active_peak "'
n = text.count(anchor)
if n != 1:
    sys.exit(f"[MEM] anchor found {n} times, expected exactly 1 -- nothing written")

# Insert a new line right after the line that contains the [MEM] print.
start = text.index(anchor)
eol = text.index("\n", start)
alloc_line = (
    '        { auto st = CachingCUDAAllocator::instance().get_stats(); std::cout << "[ALLOC] allocs "'
    ' << st.num_allocs << " | frees " << st.num_frees << " | cache_hits " << st.num_cache_hits'
    ' << " | cache_misses " << st.num_cache_misses << " | cuda_mallocs " << st.num_cuda_mallocs'
    ' << " | splits " << st.num_splits << " | merges " << st.num_merges << std::endl; }'
)
text = text[: eol + 1] + alloc_line + "\n" + text[eol + 1 :]
open(path, "w", encoding="utf-8").write(text)
print("added [ALLOC] print after the [MEM] line in", path)
