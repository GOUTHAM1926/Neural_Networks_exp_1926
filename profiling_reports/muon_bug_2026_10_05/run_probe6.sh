#!/usr/bin/env bash
# Build ns_probe6: the library's Frobenius kernels copied VERBATIM from the clone's MuonKernels.cu, plus the candidate
# fixes made from them by asserted text edits (only the across-block step changes). Standalone CUDA, no libtensor.
# Runs on GPU 2 with the real tensor-15 step-4 input first. Scratch only.
set -uo pipefail
BT=$HOME/gautam_k/BluTrain_bluccl; T=$HOME/gautam_k/muon; OUT=$T/logs/probe6; mkdir -p "$OUT"
export PATH=/usr/local/cuda/bin:$PATH CUDA_VISIBLE_DEVICES=2
echo "=== probe6 | TI $(git -C "$BT/Tensor-Implementations" rev-parse --short HEAD) | $(date '+%F %T') | GPU 2"
python3 - "$BT/Tensor-Implementations/src/Kernels/cuda/optimizer/MuonKernels.cu" "$T/ns_probe6.cu" "$T/ns_probe6_built.cu" <<'PY' || exit 1
import sys
src, tpl, out = sys.argv[1:]
s = open(src).read()
a = s.index("__inline__ __device__ float warpReduceSum(float val)")
b = s.index("// Persistent single-float scratch for the norm accumulator")
k = s[a:b]                                         # warpReduceSum + muon_sumsq_kernel + muon_scale_by_inv_frob_kernel
s1 = k.index("__global__ void muon_sumsq_kernel(")
s2 = k.index("__global__ void muon_scale_by_inv_frob_kernel(")
sumsq, scale = k[s1:s2], k[s2:]
def rep(text, old, new, what):
    c = text.count(old)
    assert c == 1, f"{what}: pattern found {c} times: {old[:70]!r}"
    return text.replace(old, new)
# V1/V3 partials: store the block sum instead of atomicAdd
partials = rep(sumsq, "muon_sumsq_kernel(", "muon_sumsq_partials(", "partials name")
partials = rep(partials, "atomicAdd(acc, thread_sum);", "acc[blockIdx.x] = thread_sum;", "partials store")
# V2 last block: store, fence, take a ticket; the last block adds all block sums in fixed order and resets the counter
last = rep(sumsq, "muon_sumsq_kernel(", "muon_sumsq_lastblock(", "last name")
last = rep(last, "    int64_t numel\n) {", "    int64_t numel,\n    float* __restrict__ partials,\n    unsigned int* __restrict__ counter\n) {\n    __shared__ bool s_last;", "last signature")
last = rep(last, "            atomicAdd(acc, thread_sum);\n        }\n    }\n}",
           "            partials[blockIdx.x] = thread_sum;\n            __threadfence();\n"
           "            s_last = (atomicInc(counter, gridDim.x - 1) == gridDim.x - 1);   // the last one also resets it to 0\n"
           "        }\n    }\n    __syncthreads();\n"
           "    if (s_last) {\n        const float t = fixed_order_total(partials, gridDim.x, true);\n        if (threadIdx.x == 0) *acc = t;\n    }\n}",
           "last tail")
# V3 scale: every block adds the block sums itself, in fixed order
fs = rep(scale, "muon_scale_by_inv_frob_kernel(", "muon_scale_fixedsum_kernel(", "fixedsum name")
fs = rep(fs, "    const float* __restrict__ acc,\n", "    const float* __restrict__ partials,\n    int nparts,\n", "fixedsum signature")
fs = rep(fs, "    float eps\n) {", "    float eps,\n    float* __restrict__ dbg_total\n) {", "fixedsum dbg param")
fs = rep(fs, "const float scale = rsqrtf(fmaxf(*acc, eps * eps));",
         "const float total_fixed = fixed_order_total(partials, nparts, false);\n"
         "    if (dbg_total && blockIdx.x == 0 && threadIdx.x == 0) *dbg_total = total_fixed;\n"
         "    const float scale = rsqrtf(fmaxf(total_fixed, eps * eps));", "fixedsum scale")
t = open(tpl).read()
assert t.count("//@@KERNELS@@") == 1
open(out, "w").write(t.replace("//@@KERNELS@@", k + "\n// ===== candidate fixes, made from the verbatim kernels above =====\n" + partials + "\n" + last + "\n" + fs))
print(f"verbatim kernels: {len(k.splitlines())} lines; V1/V3 partials, V2 last-block, V3 fixed-sum scale: all edits matched once")
PY
nvcc -std=c++17 -O3 --use_fast_math -arch=sm_90 -o "$T/ns_probe6" "$T/ns_probe6_built.cu" > "$OUT/compile.log" 2>&1 \
    || { echo "COMPILE FAILED"; grep -i error "$OUT/compile.log" | head; exit 1; }
echo "compiled"
"$T/ns_probe6" "$T/logs/dbg2/HEAD/nsin_step4.bin" 2>&1 | tee "$OUT/probe6.txt"
echo "=== done $(date '+%T')"
