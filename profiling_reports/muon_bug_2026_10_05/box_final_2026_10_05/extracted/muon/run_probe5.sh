#!/usr/bin/env bash
# Build ns_probe5 with the library's Frobenius kernels copied VERBATIM from the clone's MuonKernels.cu, then run it
# on tensor 15's step-4 Newton-Schulz input (dumped by run_debug2.sh). GPU 2. Scratch only.
set -uo pipefail
BT=$HOME/gautam_k/BluTrain_bluccl; T=$HOME/gautam_k/muon; OUT=$T/logs/dbg3
export PATH=/usr/local/cuda/bin:$PATH CUDA_VISIBLE_DEVICES=2
python3 - "$BT/Tensor-Implementations/src/Kernels/cuda/optimizer/MuonKernels.cu" "$T/ns_probe5.cu" "$T/ns_probe5_built.cu" <<'PY' || exit 1
import sys
src, tpl, out = sys.argv[1:]
s = open(src).read()
a = s.index("__inline__ __device__ float warpReduceSum(float val)")
b = s.index("// Persistent single-float scratch for the norm accumulator")
k = s[a:b]
assert k.count("__global__ void muon_sumsq_kernel(") == 1 and k.count("__global__ void muon_scale_by_inv_frob_kernel(") == 1
s1 = k.index("__global__ void muon_sumsq_kernel(")
s2 = k.index("__global__ void muon_scale_by_inv_frob_kernel(")
sumsq = k[s1:s2]
assert sumsq.count("atomicAdd(acc, thread_sum);") == 1
partials = sumsq.replace("muon_sumsq_kernel(", "muon_sumsq_partials(").replace("atomicAdd(acc, thread_sum);", "acc[blockIdx.x] = thread_sum;")
t = open(tpl).read()
assert t.count("//@@KERNELS@@") == 1
open(out, "w").write(t.replace("//@@KERNELS@@", k + "\n// ---- the same Pass 1, storing each block's sum instead of atomicAdd ----\n" + partials))
print(f"kernels copied verbatim: {len(k.splitlines())} lines; partials variant: 1 replacement")
PY
cd "$BT" || exit 1
nvcc -std=c++17 -O2 -arch=sm_90 -o "$T/ns_probe5" "$T/ns_probe5_built.cu" -LTensor-Implementations/lib -LProfiler/lib \
     -Xlinker -rpath,$PWD/Tensor-Implementations/lib -Xlinker -rpath,$PWD/Profiler/lib -ltensor -lprofiler -lcublas -lcuda \
     > "$OUT/compile_probe5.log" 2>&1 || { echo "COMPILE FAILED"; grep -i error "$OUT/compile_probe5.log" | head; exit 1; }
echo "compiled"
"$T/ns_probe5" "$T/logs/dbg2/HEAD/nsin_step4.bin" 2>&1 | tee "$OUT/probe5.txt"
