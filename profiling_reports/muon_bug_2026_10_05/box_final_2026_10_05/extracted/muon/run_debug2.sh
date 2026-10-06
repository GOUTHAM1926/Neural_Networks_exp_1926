#!/usr/bin/env bash
# Proof run for the test_muon_rsag W=1 failure (scratch only, nothing in the repo changes).
#  part A: the debug fingerprint copy (see run_debug.sh) for THREE commits' dist/zero: HEAD, a3aba3e (Navin, no F4),
#          e6eaeba (before both). Also dumps tensor 15's Newton-Schulz input per step, so part B can replay it.
#  part B: ns_probe4 on those exact inputs: is the library Newton-Schulz deterministic, and is the atomicAdd the cause?
#   usage: bash run_debug2.sh [runs per arm]
set -uo pipefail
N=${1:-10}
BT=$HOME/gautam_k/BluTrain_bluccl; T=$HOME/gautam_k/muon; OUT=$T/logs/dbg2; rm -rf "$OUT"; mkdir -p "$OUT"
export PATH=/usr/local/cuda/bin:$PATH CUDA_VISIBLE_DEVICES=2
ARMS="HEAD a3aba3e e6eaeba"
cd "$BT" || exit 1
echo "=== debug2 | clone HEAD $(git rev-parse --short HEAD) | TI $(git -C Tensor-Implementations rev-parse --short HEAD) | $(date '+%F %T') | GPU 2 | $N runs per arm"
base=$(make --no-print-directory -n build-mpi FILE="$BT/dist/zero/tests/test_muon_rsag.cpp" SM_ARCH=90 WITH_CUBLAS=1 NO_TMUX=1 2>/dev/null | grep -- '-o _runner_ ' | tail -1)
[ -z "$base" ] && { echo "ERROR: could not read the build-mpi compile line"; exit 1; }
for C in $ARMS; do
    SRC=$T/dbg_$C; rm -rf "$SRC"; mkdir -p "$SRC"
    if [ "$C" = HEAD ]; then cp -r "$BT/dist" "$SRC/" ; else git archive "$C" dist/zero | tar -x -C "$SRC" || { echo "ERROR: git archive $C"; exit 1; }; fi
    python3 - "$SRC/dist/zero/src/ShardedMuon.cpp" "$C" <<'PY' || exit 1
import sys
p, arm = sys.argv[1], sys.argv[2]
s = open(p).read()
helper = r'''
// ---- DEBUG (scratch copy only): fingerprint of n device floats; NSIN is also dumped to $DBG_DUMP/nsin_step<k>.bin ----
static unsigned long long dbg_fp(const float* d, int64_t n, const char* stage, long long step) {
    cudaStreamSynchronize(ocuda::getCurrentStream());
    std::vector<float> h(static_cast<size_t>(n));
    cudaMemcpy(h.data(), d, static_cast<size_t>(n) * sizeof(float), cudaMemcpyDeviceToHost);
    const char* dir = std::getenv("DBG_DUMP");
    if (dir && std::string(stage) == "NSIN") {
        std::string f = std::string(dir) + "/nsin_step" + std::to_string(step) + ".bin";
        if (FILE* fp = std::fopen(f.c_str(), "wb")) { std::fwrite(h.data(), sizeof(float), h.size(), fp); std::fclose(fp); }
    }
    unsigned long long x = 1469598103934665603ull;
    const unsigned char* b = reinterpret_cast<const unsigned char*>(h.data());
    for (size_t i = 0; i < h.size() * sizeof(float); ++i) { x ^= b[i]; x *= 1099511628211ull; }
    return x;
}
#define DBG_FP(stage, ptr, cnt) do { if (rec.numel == 9216) std::printf("DBG mode=%d step=%lld stage=%s fp=%016llx\n", \
    static_cast<int>(cfg_.collective), static_cast<long long>(step_count_), stage, \
    dbg_fp((ptr), (cnt), stage, static_cast<long long>(step_count_))); } while (0)

bool ShardedMuon::update_one(size_t slot,'''
edits = [
 ("\nbool ShardedMuon::update_one(size_t slot,", helper),
 ("    Tensor g32   = owned_grad_fp32(slot, /*apply_clip=*/true);\n    const int64_t n = static_cast<int64_t>(rec.numel);\n",
  "    Tensor g32   = owned_grad_fp32(slot, /*apply_clip=*/true);\n    const int64_t n = static_cast<int64_t>(rec.numel);\n"
  "    DBG_FP(\"W0\", work.data<float>(), n);\n    DBG_FP(\"G\", g32.data<float>(), n);\n"),
 ("        ocuda::muon_momentum_cuda(m.data<float>(), g32.data<float>(), g.data<float>(),\n                                  cfg_.momentum, cfg_.nesterov, n);\n",
  "        ocuda::muon_momentum_cuda(m.data<float>(), g32.data<float>(), g.data<float>(),\n                                  cfg_.momentum, cfg_.nesterov, n);\n"
  "        DBG_FP(\"M\", m.data<float>(), n);\n        DBG_FP(\"NSIN\", g.data<float>(), n);\n"),
 ("        Tensor up = update.is_contiguous() ? update : update.contiguous();\n",
  "        Tensor up = update.is_contiguous() ? update : update.contiguous();\n        DBG_FP(\"NSOUT\", up.data<float>(), n);\n"),
 ("        ocuda::muon_apply_cuda(work.data<float>(), up.data<float>(),\n                               cfg_.lr * adjust, n);\n",
  "        ocuda::muon_apply_cuda(work.data<float>(), up.data<float>(),\n                               cfg_.lr * adjust, n);\n        DBG_FP(\"W1\", work.data<float>(), n);\n"),
]
for old, new in edits:
    c = s.count(old)
    assert c == 1, f"{arm}: pattern found {c} times: {old[:60]!r}"
    s = s.replace(old, new)
s = "#include <vector>\n#include <cstdio>\n#include <cstdlib>\n#include <string>\n" + s
open(p, "w").write(s)
print(f"{arm}: debug copy patched (5 edits, each matched once)")
PY
    cmd=${base/-o _runner_ /-o $T/test_muon_rsag_dbg_$C }
    cmd=${cmd//dist\/zero\/src\//$SRC\/dist\/zero\/src\/}
    cmd=${cmd/-Idist\/zero\/include/-I$SRC\/dist\/zero\/include}
    for f in $(grep -o "$SRC/dist/zero/src/[A-Za-z_]*\.c[pu]*" <<<"$cmd" | sort -u); do
        [ -f "$f" ] || { echo "note: $f absent in $C, dropped from the compile line"; cmd=${cmd//$f/}; }
    done
    ( eval "$cmd" > "$OUT/compile_$C.log" 2>&1 && echo "$C compiled" || { echo "$C COMPILE FAILED"; grep -i error "$OUT/compile_$C.log" | head -5; } ) &
done
PROBE_CMD="g++ -std=c++2a -O2 -DWITH_CUDA -fopenmp -ITensor-Implementations/include -IProfiler/include -I/usr/local/cuda/include -o $T/ns_probe4 $T/ns_probe4.cpp -LTensor-Implementations/lib -LProfiler/lib -L/usr/local/cuda/lib64 -Wl,-rpath,$PWD/Tensor-Implementations/lib -Wl,-rpath,$PWD/Profiler/lib -ltensor -lprofiler -lcuda -lcudart -lcublas -lgomp -lpthread -ldl"
cp "$T/ns_probe4.cu" "$T/ns_probe4.cpp"
( eval "$PROBE_CMD" > "$OUT/compile_probe4.log" 2>&1 && echo "ns_probe4 compiled" || { echo "ns_probe4 COMPILE FAILED"; grep -i error "$OUT/compile_probe4.log" | head -5; } ) &
wait
for C in $ARMS; do [ -x "$T/test_muon_rsag_dbg_$C" ] || { echo "missing binary for $C"; exit 1; }; done

echo "--- part A: $N runs per arm, one arm after the other (never two at once)"
for C in $ARMS; do
    mkdir -p "$OUT/$C"
    for i in $(seq 1 "$N"); do DBG_DUMP=$OUT/$C mpirun -x DBG_DUMP -np 1 "$T/test_muon_rsag_dbg_$C" > "$OUT/$C/run_$i.txt" 2>&1; done
done
python3 - "$OUT" "$N" $ARMS <<'PY'
import re, sys, os
out, n, arms = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
nsin = {}
for arm in arms:
    fails = 0; firsts = {}; nsout = {}
    for i in range(1, n + 1):
        txt = open(os.path.join(out, arm, f"run_{i}.txt")).read()
        v = re.findall(r"^\[(sync publish|staged refresh)\] (PASS|FAIL)", txt, re.M)
        fails += any(x[1] == "FAIL" for x in v)
        lines = re.findall(r"^DBG mode=(\d+) step=(\d+) stage=(\w+) fp=([0-9a-f]+)", txt, re.M)
        runs, cur, last = [], [], None
        for mode, step, stage, fp in lines:
            if last is not None and mode != last: runs.append(cur); cur = []
            cur.append((int(step), stage, fp)); last = mode
        if cur: runs.append(cur)
        for a, b in [(0, 1), (2, 3)]:
            if b < len(runs):
                d = next((f"step {x[0]} {x[1]}" for x, y in zip(runs[a], runs[b]) if x[2] != y[2]), "identical")
                firsts[d] = firsts.get(d, 0) + 1
        for step, stage, fp in (t for r in runs for t in r):
            if stage == "NSIN": nsin.setdefault((arm, step), set()).add(fp)
            if stage == "NSOUT": nsout.setdefault(step, {}).setdefault(fp, 0); nsout[step][fp] += 1
    print(f"arm {arm:8s}: {fails} of {n} runs FAILED | RB-vs-RS first difference (2 scenarios x {n} runs): {firsts}")
    for step in sorted(nsout):
        print(f"      step {step}: NS output fingerprints seen {dict(nsout[step])}")
steps = sorted({s for (_, s) in nsin})
print("NS input (NSIN) fingerprint per step and arm (one value each = same input in every run and mode):")
for s in steps:
    print(f"   step {s}: " + " | ".join(f"{a} {','.join(sorted(nsin.get((a, s), {'-'})))}" for a in arms))
PY

echo "--- part B: ns_probe4 on the dumped inputs, 200 calls each"
for C in $ARMS; do
    for s in 1 2 3 4; do
        f=$OUT/$C/nsin_step$s.bin; [ -f "$f" ] && "$T/ns_probe4" "$f" 96 96 200
    done
done 2>&1 | tee "$OUT/probe4.txt"
echo "=== done $(date '+%T')"
