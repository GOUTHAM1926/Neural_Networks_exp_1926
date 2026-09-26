#!/usr/bin/env python3
"""GPU reduction fixes, part 2 (applies ON TOP of patch_reduce_gpu.py).

  Bug 5  middle-axis reduction (outer > 1, e.g. axis 1 of [7, 300, 5]): build_reduce_config reset
         num_outputs to inner_count and the kernel read src + output_idx, so only the first slab
         was computed and every later output was never written.
  Bug 6  Generic path (non-consecutive axes such as {0, 2}, or a non-contiguous input): the
         dispatchers passed EMPTY offset calculators and num_outputs / reduced_count = 0 (clamped
         to 1), so one output was computed from the wrong elements.
  Also:  removes a wrong line from the patch-1 comment in block_x_reduce (the middle-axis case is
         Bug 5, not block_x_reduce).

Run from the Tensor-Implementations root:  python3 patch_reduce_gpu_2.py
Refuses to write anything unless both files are exactly the patch-1 output and every pattern
matches the expected number of times.
"""
import hashlib, sys

EXPECT = {   # md5 AFTER patch_reduce_gpu.py (patch 1)
    "src/UnaryOps/cuda/ReductionImplGPU.cu":   "b91fe4e6790e4e11ded605b06c9b1c50",
    "include/ops/helpers/ReductionKernels.cuh": "fad00b91c498414fa3ab6c9bf21411d3",
}

SETUP_GENERIC = """
// Modified: new helper. Generic layouts (reduced axes that are not consecutive, e.g. {0, 2}, or a
// non-contiguous input such as a transposed view) arrive from compute_reduction_layout with only
// path = Generic: num_outputs = reduced_count = 0 and no offset information. The dispatchers used
// to launch them with EMPTY offset calculators, and build_reduce_config clamps both counts to 1, so
// the kernel computed a single output from src[0 .. ] -- wrong for every such reduction.
// This fills in the counts and builds the two calculators the Generic kernel path already uses
// (out_base = output_calc.get(output_idx), in_off = input_calc.get(i), value = src[out_base + in_off]):
//   output_calc walks the NON-reduced dims in order -> the output tensor's C order;
//   input_calc  walks the REDUCED dims             -> one reduction step each.
// Both use the input's real strides, so any view is addressed correctly. No-op for Tier 1/2.
// OffsetCalculator holds at most MAX_DIMS dims per side and silently ignores the rest, so more
// dims than that is refused here instead of computed wrongly.
static void setup_generic_path(detail::ReductionLayout& layout, const Tensor& input,
                               const std::vector<int64_t>& axes,
                               detail::OffsetCalculator<1, uint32_t>& input_calc,
                               detail::OffsetCalculator<1, uint32_t>& output_calc) {
    if (layout.path != detail::ReductionLayout::Path::Generic) return;
    const auto& dims    = input.shape().dims;
    const auto& strides = input.stride().strides;
    const int nd = static_cast<int>(dims.size());
    bool is_reduced[64] = {false};
    for (int64_t a : axes) is_reduced[a] = true;

    std::vector<int64_t> r_sz, r_st, o_sz, o_st;
    int64_t reduced = 1, outputs = 1;
    for (int d = 0; d < nd; ++d) {
        if (is_reduced[d]) { r_sz.push_back(dims[d]); r_st.push_back(strides[d]); reduced *= dims[d]; }
        else               { o_sz.push_back(dims[d]); o_st.push_back(strides[d]); outputs *= dims[d]; }
    }
    constexpr int kMaxDims = detail::OffsetCalculator<1, uint32_t>::MAX_DIMS;
    if (static_cast<int>(r_sz.size()) > kMaxDims || static_cast<int>(o_sz.size()) > kMaxDims)
        throw std::runtime_error("GPU reduction: a Generic-path reduction supports at most " +
                                 std::to_string(kMaxDims) + " reduced and " + std::to_string(kMaxDims) +
                                 " kept dims (got " + std::to_string(r_sz.size()) + " and " +
                                 std::to_string(o_sz.size()) + ")");
    layout.num_outputs   = outputs;
    layout.reduced_count = reduced;
    const int64_t* rsp = r_st.data();
    const int64_t* osp = o_st.data();
    input_calc  = detail::OffsetCalculator<1, uint32_t>(static_cast<int>(r_sz.size()), r_sz.data(), &rsp);
    output_calc = detail::OffsetCalculator<1, uint32_t>(static_cast<int>(o_sz.size()), o_sz.data(), &osp);
}
"""

# (old, new, expected count)
EDITS = {
"src/UnaryOps/cuda/ReductionImplGPU.cu": [
("""    num_mp = cached_num_mp; max_tpm = cached_max_tpm;
}
""", """    num_mp = cached_num_mp; max_tpm = cached_max_tpm;
}
""" + SETUP_GENERIC, 1),
# value dispatcher: calculators before the config, Generic filled in
("""    auto config = detail::build_reduce_config<arg_t>(layout, num_mp, max_tpm);

    // ── 5. OffsetCalculator (empty: Tier 1/2 use direct pointer math) ──
    detail::OffsetCalculator<1, uint32_t> input_calc{}, output_calc{};
""", """    // ── 5. OffsetCalculator (empty for Tier 1/2; built for Generic) ──
    // Modified: built BEFORE the config, because setup_generic_path also sets the Generic
    // layout's num_outputs / reduced_count, which the config is sized from.
    detail::OffsetCalculator<1, uint32_t> input_calc{}, output_calc{};
    setup_generic_path(layout, input, normalized_axes, input_calc, output_calc);
    auto config = detail::build_reduce_config<arg_t>(layout, num_mp, max_tpm);
""", 1),
# index, mean, variance dispatchers
("""    auto config = detail::build_reduce_config<arg_t>(layout, num_mp, max_tpm);
    detail::OffsetCalculator<1, uint32_t> input_calc{}, output_calc{};
""", """    // Modified: calculators built before the config (setup_generic_path sets the Generic counts).
    detail::OffsetCalculator<1, uint32_t> input_calc{}, output_calc{};
    setup_generic_path(layout, input, normalized_axes, input_calc, output_calc);
    auto config = detail::build_reduce_config<arg_t>(layout, num_mp, max_tpm);
""", 3),
# all 4 launches: pass the slab stride for OuterContiguous
("""layout.path, step_stride, smem, stream);""",
 """layout.path, step_stride, smem, stream,
        (layout.path == detail::ReductionLayout::Path::OuterContiguous) ? layout.input_outer_stride : 0);""", 4),
],
"include/ops/helpers/ReductionKernels.cuh": [
# config: keep num_outputs = outer * inner
("""    // OuterContiguous: inner_count is the "fast" dimension (outputs),
    // num_outputs is how many rows we reduce over.
    if (layout.path == ReductionLayout::Path::OuterContiguous) {
        num_outputs       = static_cast<int>(layout.inner_count);
        inputs_per_output = static_cast<int>(layout.reduced_count);
""", """    // Modified: OuterContiguous keeps num_outputs = layout.num_outputs (= outer * inner). It was
    // reset to inner_count, which is right only for a leading-axis reduction (outer == 1). For a
    // middle-axis reduction (outer > 1, e.g. axis 1 of [7, 300, 5]) only the first `inner` outputs
    // got a thread, so every later slab of the output was never written (measured on the H200:
    // 30 of 35 outputs wrong, sum fp32 and mean bf16). The kernel now maps each output index to its
    // slab (ReduceOp::outer_stride). For outer == 1 both values are equal, so nothing changes there.
    if (layout.path == ReductionLayout::Path::OuterContiguous) {
        num_outputs       = static_cast<int>(layout.num_outputs);
        inputs_per_output = static_cast<int>(layout.reduced_count);
""", 1),
# ReduceOp: outer_stride field
("""    int64_t step_stride;
    // Multi-CTA staging (populated only when config.should_global_reduce()).""",
 """    int64_t step_stride;
    // Modified: OuterContiguous only -- element stride between outer slabs (reduced * inner), so a
    // middle-axis output (o_outer, o_inner) reads the column starting at
    // src + o_outer * outer_stride + o_inner. For a leading-axis reduction every output_idx < inner,
    // so o_outer == 0 and the address is src + output_idx exactly as before.
    int64_t outer_stride = 0;
    // Multi-CTA staging (populated only when config.should_global_reduce()).""", 1),
# kernel: slab remap
("""        const scalar_t* __restrict__ base_ptr = op.src + output_idx;
        const index_t row_stride = (index_t)op.step_stride;
""", """        // Modified: map the flat output index to its slab. Outputs are ordered (o_outer, o_inner)
        // with inner_count == step_stride. It was src + output_idx, right only when outer == 1.
        const index_t row_stride = (index_t)op.step_stride;
        const index_t o_outer    = output_idx / row_stride;
        const index_t o_inner    = output_idx - o_outer * row_stride;
        const scalar_t* __restrict__ base_ptr = op.src + o_outer * (index_t)op.outer_stride + o_inner;
""", 1),
# launcher: new trailing parameter
("""    int smem,
    cudaStream_t stream)
{
    // Pack into single struct for minimal launch parameter overhead
    detail::ReduceOp<scalar_t, out_scalar_t, ops_t, index_t> op;
""", """    int smem,
    cudaStream_t stream,
    int64_t outer_stride = 0)   // Modified: OuterContiguous slab stride, see ReduceOp::outer_stride
{
    // Pack into single struct for minimal launch parameter overhead
    detail::ReduceOp<scalar_t, out_scalar_t, ops_t, index_t> op;
    op.outer_stride = outer_stride;
""", 1),
# correct the patch-1 comment: the middle-axis case is Bug 5, not block_x_reduce
("""    //   block_width > 32: sum [2, 65536] 2 / 2; multi-CTA sum [8, 1 << 20] 8 / 8;
    //   middle axis: sum fp32 [7, 300, 5] 30 / 35.
""", """    //   block_width > 32: sum [2, 65536] 2 / 2; multi-CTA sum [8, 1 << 20] 8 / 8.
""", 1),
],
}

def main():
    out = {}
    for path, want in EXPECT.items():
        data = open(path, "rb").read()
        got = hashlib.md5(data).hexdigest()
        if got != want:
            sys.exit(f"ABORT: {path} md5 {got}, expected {want} (patch 1 not applied?). Nothing written.")
        text = data.decode()
        for i, (old, new, cnt) in enumerate(EDITS[path], 1):
            n = text.count(old)
            if n != cnt:
                sys.exit(f"ABORT: {path} edit {i}: pattern found {n} times (need {cnt}). Nothing written.")
            text = text.replace(old, new)
        out[path] = text
    for path, text in out.items():
        open(path, "w").write(text)
        print(f"patched {path}: {len(EDITS[path])} edit(s)")
    print("done: 2 files, 9 edit groups")

if __name__ == "__main__":
    main()
