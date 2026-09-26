#!/usr/bin/env python3
"""Reduction fixes, part 3 (applies ON TOP of patch_reduce_gpu.py and patch_reduce_gpu_2.py).

  Bug 7  CPU middle-axis reduction (outer > 1 and inner > 1): compute_reduction_layout tags it
         OuterContiguous with input_outer_stride = reduced * inner, but every CPU OuterContiguous
         consumer (scalar and SIMD) treats the output as ONE contiguous column block, which is
         right only when outer == 1. Now demoted to the CPU Generic carry-add path at all 6 CPU
         layout sites (the GPU handles the slab stride directly since patch 2).
  Race   GPU multi-CTA final combine: block_y_reduce then block_x_reduce reuse the same shared
         buffer with no barrier between them; compute-sanitizer racecheck reported 11,521 RAW
         hazards (write thread (0,1), read thread (0,0), same slot). One __syncthreads() there.

Run from the Tensor-Implementations root:  python3 patch_reduce_3.py
Refuses to write anything unless both files have the expected md5 and every pattern matches the
expected number of times.
"""
import hashlib, re, sys

EXPECT = {
    "include/ops/helpers/ReductionImpl.h":       "b42dc0b459330db6ba563fc5ce24e554",   # 7810aea
    "include/ops/helpers/ReductionKernels.cuh":  "3513e8e7251bcf605a620b6d6914899a",   # after patch 2
}

DEMOTE_FN = """// Modified: new helper, CPU only. A middle-axis reduction (kept dims both before AND after the
// reduced axes, e.g. axis 1 of [7, 300, 5]) is tagged OuterContiguous by compute_reduction_layout,
// with input_outer_stride = reduced * inner. The CPU OuterContiguous consumers (scalar and SIMD)
// all treat the output as ONE contiguous column block of inner_count values, which is right only
// when outer == 1 (a leading-axis reduction, where input_outer_stride == input_row_stride == inner).
// For outer > 1 they read the wrong elements: measured on 2026-09-26, sum fp32 [7, 300, 5] axis 1
// 30 of 35 outputs wrong, sum fp32 [4, 64, 128] axis 1 384 of 512, mean bf16 [3, 1000, 7] 8 of 21,
// max fp32 [2, 3, 4, 5] axes {1, 2} 5 of 10, mean int32 [6, 50, 9] 44 of 54.
// This demotes that case to the Generic carry-add path, which walks the real strides and is
// already correct for non-consecutive axes and transposed views (same test: 0 wrong). A
// leading-axis reduction keeps the vectorised OuterContiguous path. The GPU keeps OuterContiguous
// for the middle axis too: its kernel maps each output to its slab (ReduceOp::outer_stride).
// Cost: the middle-axis CPU case loses the SIMD path (correct first; a per-slab SIMD loop can
// come later if a CPU middle-axis reduction ever matters for speed).
inline void cpu_demote_middle_to_generic(ReductionLayout& layout) {
    if (layout.path == ReductionLayout::Path::OuterContiguous &&
        layout.input_outer_stride != layout.input_row_stride)
        layout.path = ReductionLayout::Path::Generic;
}

"""

def edit_impl(text):
    # 1. the helper, right before the first consumer (cascade_sum_kernel)
    anchor = """// =================================================================
// CASCADE SUM KERNEL"""
    if text.count(anchor) != 1:
        sys.exit(f"ABORT: ReductionImpl.h helper anchor found {text.count(anchor)} times (need 1). Nothing written.")
    text = text.replace(anchor, DEMOTE_FN + anchor)
    # 2. the 6 CPU layout sites (exact indentation, line-anchored)
    sites = [
        (r"(?m)^(    ReductionLayout layout = compute_reduction_layout\(input, normalized_axes\);\n)",
         r"\1    cpu_demote_middle_to_generic(layout);   // Modified: see cpu_demote_middle_to_generic\n", 3),
        (r"(?m)^(        ReductionLayout layout = compute_reduction_layout\(input, normalized_axes\);\n)",
         r"\1        cpu_demote_middle_to_generic(layout);   // Modified: see cpu_demote_middle_to_generic\n", 2),
        (r"(?m)^(    ReductionLayout layout_var = compute_reduction_layout\(input, normalized_axes\);\n)",
         r"\1    cpu_demote_middle_to_generic(layout_var);   // Modified: see cpu_demote_middle_to_generic\n", 1),
    ]
    for pat, rep, cnt in sites:
        text, n = re.subn(pat, rep, text)
        if n != cnt:
            sys.exit(f"ABORT: ReductionImpl.h site pattern matched {n} times (need {cnt}). Nothing written.")
    return text

def edit_kernels(text):
    old = """        value = block_y_reduce(value, op.ops, shared_memory);
        if (config.should_block_x_reduce())
            value = block_x_reduce(value, op.ops, shared_memory);
"""
    new = """        value = block_y_reduce(value, op.ops, shared_memory);
        if (config.should_block_x_reduce()) {
            // Modified: barrier between the two stages. block_y_reduce ends with the threadIdx.y == 0
            // row still reading the other rows' slots, while block_x_reduce starts by writing
            // smem[threadIdx.x + threadIdx.y * blockDim.x] -- those same slots. The value written is
            // the one the row stored in block_y_reduce, so the result was right, but it is an
            // unsynchronised write/read pair (a data race): compute-sanitizer racecheck on the H200
            // reported 11,521 RAW hazards here (sum fp32 [8, 1 << 20] axis 1: write thread (0,1),
            // read thread (0,0), same shared slot). Runs only in the last CTA of each output tile, so
            // it costs one barrier per tile. The main path calls the two in the other order
            // (block_x_reduce ends with its own barrier), so it needs none.
            __syncthreads();
            value = block_x_reduce(value, op.ops, shared_memory);
        }
"""
    n = text.count(old)
    if n != 1:
        sys.exit(f"ABORT: ReductionKernels.cuh combine pattern found {n} times (need 1). Nothing written.")
    return text.replace(old, new)

def main():
    out = {}
    for path, want in EXPECT.items():
        data = open(path, "rb").read()
        got = hashlib.md5(data).hexdigest()
        if got != want:
            sys.exit(f"ABORT: {path} md5 {got}, expected {want}. Nothing written.")
        text = data.decode()
        out[path] = edit_impl(text) if path.endswith("ReductionImpl.h") else edit_kernels(text)
    for path, text in out.items():
        open(path, "w").write(text)
        print(f"patched {path}")
    print("done: ReductionImpl.h (helper + 6 sites), ReductionKernels.cuh (1 barrier)")

if __name__ == "__main__":
    main()
