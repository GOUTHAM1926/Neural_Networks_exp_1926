#!/usr/bin/env bash
# Which operation INSIDE Newton-Schulz gives a different result in REDUCE_BROADCAST mode? (scratch only)
# Debug copy of HEAD's ShardedMuon.cpp: for tensor 15 (numel 9216) the call to the library
# newton_schulz_orthogonalize is replaced by dbg_ns(), a line-for-line copy of the library's bf16 branch
# (Optim.cpp, same calls in the same order) that prints, after every operation, a fingerprint of the result
# and the device address of the buffer. Other tensors still call the library.
#   usage: bash run_debug3.sh [runs]
set -uo pipefail
N=${1:-6}
BT=$HOME/gautam_k/BluTrain_bluccl; T=$HOME/gautam_k/muon; OUT=$T/logs/dbg3; SRC=$T/dbg3_HEAD
rm -rf "$OUT" "$SRC"; mkdir -p "$OUT" "$SRC"
export PATH=/usr/local/cuda/bin:$PATH CUDA_VISIBLE_DEVICES=2
cd "$BT" || exit 1
echo "=== debug3 | clone HEAD $(git rev-parse --short HEAD) | TI $(git -C Tensor-Implementations rev-parse --short HEAD) | $(date '+%F %T') | GPU 2 | $N runs"
cp -r "$BT/dist" "$SRC/"
python3 - "$SRC/dist/zero/src/ShardedMuon.cpp" <<'PY' || exit 1
import sys
p = sys.argv[1]
s = open(p).read()
helper = r'''
// ---- DEBUG (scratch copy only): line-for-line copy of newton_schulz_orthogonalize's bf16 branch with probes ----
static unsigned long long dbg_fpb(const void* d, size_t bytes) {
    cudaStreamSynchronize(ocuda::getCurrentStream());
    std::vector<unsigned char> h(bytes);
    cudaMemcpy(h.data(), d, bytes, cudaMemcpyDeviceToHost);
    unsigned long long x = 1469598103934665603ull;
    for (unsigned char c : h) { x ^= c; x *= 1099511628211ull; }
    return x;
}
static void dbg_p(int mode, long long step, const std::string& op, const Tensor& t) {
    const unsigned long long fp = dbg_fpb(t.data(), t.nbytes());
    std::printf("NSDBG mode=%d step=%lld op=%s fp=%016llx addr=%p mod256=%llu\n", mode, step, op.c_str(), fp, t.data(),
                static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(t.data()) % 256));
}
static Tensor dbg_ns(const Tensor& G, int steps, float eps, int mode, long long step) {
    Shape orig_shape = G.shape();
    Tensor X = G;
    if (X.ndim() != 2) { int64_t d0 = X.shape().dims[0]; X = X.view(Shape{{d0, X.numel() / d0}}); }
    bool transposed = false;
    if (X.shape().dims[0] > X.shape().dims[1]) { X = X.t().contiguous(); transposed = true; } else { X = X.contiguous(); }
    const float a = 3.4445f, b = -4.7750f, c = 2.0315f;
    dbg_p(mode, step, "in", X);
    ocuda::muon_frobenius_scale_cuda(X.data<float>(), X.numel(), eps);
    dbg_p(mode, step, "scaled", X);
    cudaStream_t stream = ocuda::getCurrentStream();
    int64_t k = X.shape().dims[0];
    TensorOptions bf16_opts = X.opts().with_dtype(Dtype::Bfloat16);
    Tensor Xb = X.as_type(Dtype::Bfloat16);
    dbg_p(mode, step, "Xb0", Xb);
    Tensor A = Tensor::empty(Shape{{k, k}}, bf16_opts);
    Tensor B = Tensor::empty(Shape{{k, k}}, bf16_opts);
    Tensor Xb_new = Tensor::empty(Xb.shape(), bf16_opts);
    for (int i = 0; i < steps; ++i) {
        const std::string it = std::to_string(i);
        OwnTensor::cuda_matmul(Xb, Xb.t(), A, stream);              dbg_p(mode, step, "it" + it + "_A=XXt", A);
        OwnTensor::cuda_addmm(A, A, A, c, b, B, stream);            dbg_p(mode, step, "it" + it + "_B=bA+cAA", B);
        OwnTensor::cuda_addmm(Xb, B, Xb, 1.0f, a, Xb_new, stream);  dbg_p(mode, step, "it" + it + "_Xn=aX+BX", Xb_new);
        std::swap(Xb, Xb_new);
    }
    X = Xb.as_type(Dtype::Float32);
    dbg_p(mode, step, "out", X);
    if (transposed) X = X.t().contiguous();
    if (orig_shape != X.shape()) X = X.view(orig_shape);
    return X;
}

bool ShardedMuon::update_one(size_t slot,'''
edits = [
 ("\nbool ShardedMuon::update_one(size_t slot,", helper),
 ("        Tensor update = OwnTensor::nn::newton_schulz_orthogonalize(\n            g, cfg_.ns_steps, 1e-8f, /*use_bf16=*/true);\n",
  "        Tensor update = (rec.numel == 9216)\n"
  "            ? dbg_ns(g, cfg_.ns_steps, 1e-8f, static_cast<int>(cfg_.collective), static_cast<long long>(step_count_))\n"
  "            : OwnTensor::nn::newton_schulz_orthogonalize(g, cfg_.ns_steps, 1e-8f, /*use_bf16=*/true);\n"),
]
for old, new in edits:
    c = s.count(old)
    assert c == 1, f"pattern found {c} times: {old[:60]!r}"
    s = s.replace(old, new)
s = "#include <vector>\n#include <cstdio>\n#include <cstdint>\n#include <string>\n#include \"ops/Matmul.cuh\"\n" + s
open(p, "w").write(s)
print("debug3 copy patched (2 edits, each matched once)")
PY
base=$(make --no-print-directory -n build-mpi FILE="$BT/dist/zero/tests/test_muon_rsag.cpp" SM_ARCH=90 WITH_CUBLAS=1 NO_TMUX=1 2>/dev/null | grep -- '-o _runner_ ' | tail -1)
cmd=${base/-o _runner_ /-o $T/test_muon_rsag_dbg3 }
cmd=${cmd//dist\/zero\/src\//$SRC\/dist\/zero\/src\/}
cmd=${cmd/-Idist\/zero\/include/-I$SRC\/dist\/zero\/include}
eval "$cmd" > "$OUT/compile.log" 2>&1 || { echo "COMPILE FAILED"; grep -i error "$OUT/compile.log" | head; exit 1; }
echo "compiled"
for i in $(seq 1 "$N"); do mpirun -np 1 "$T/test_muon_rsag_dbg3" > "$OUT/run_$i.txt" 2>&1; done
python3 - "$OUT" "$N" <<'PY'
import re, sys, os
out, n = sys.argv[1], int(sys.argv[2])
for i in range(1, n + 1):
    txt = open(os.path.join(out, f"run_{i}.txt")).read()
    v = " | ".join(f"{a} {b}" for a, b in re.findall(r"^\[(sync publish|staged refresh)\] (PASS|FAIL)", txt, re.M))
    rows = re.findall(r"^NSDBG mode=(\d+) step=(\d+) op=(\S+) fp=([0-9a-f]+) addr=(\S+) mod256=(\d+)", txt, re.M)
    # 4 blocks in order: sync RB, sync RS, staged RB, staged RS (mode changes between blocks)
    blocks, cur, last = [], [], None
    for r in rows:
        if last is not None and r[0] != last: blocks.append(cur); cur = []
        cur.append(r); last = r[0]
    if cur: blocks.append(cur)
    print(f"run {i}: {v}")
    for sc, (a, b) in zip(["sync", "staged"], [(0, 1), (2, 3)]):
        if b >= len(blocks): print(f"   {sc}: missing blocks"); continue
        d = next((x, y) for x, y in zip(blocks[a], blocks[b]) if x[3] != y[3]) if any(x[3] != y[3] for x, y in zip(blocks[a], blocks[b])) else None
        if d is None: print(f"   {sc}: RB and RS identical through every NS operation"); continue
        x, y = d
        print(f"   {sc}: FIRST DIFFERENT OPERATION step {x[1]} {x[2]}  | RB fp {x[3][:8]} at {x[4]} (mod256 {x[5]}) | RS fp {y[3][:8]} at {y[4]} (mod256 {y[5]})")
        # the inputs of that operation, both modes, for the same step
        for blk, name in [(blocks[a], "RB"), (blocks[b], "RS")]:
            st = [r for r in blk if r[1] == x[1]]
            print(f"      {name} step {x[1]}: " + " ".join(f"{r[2]}:{r[3][:6]}@{r[5]}" for r in st))
PY
echo "=== done $(date '+%T')"
