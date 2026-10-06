#!/usr/bin/env bash
# Debug copy of the LATEST ShardedMuon.cpp (scratch, outside the repo): for tensor 15 (the 96x96 weight, the only one
# that ever differs) print an FNV-1a fingerprint of every value along update_one(), in both collective modes, every
# step. In a failing run, the first stage whose fingerprint differs between REDUCE_BROADCAST and RS/AG is the bug.
#   stages: W0 master in | G gradient (g32) | M momentum | NSIN Newton-Schulz input | NSOUT its output | W1 master out
#   usage: bash run_debug.sh [runs]
set -uo pipefail
N=${1:-10}
BT=$HOME/gautam_k/BluTrain_bluccl; T=$HOME/gautam_k/muon; SRC=$T/var_DBG; mkdir -p "$T/logs/dbg"
export PATH=/usr/local/cuda/bin:$PATH CUDA_VISIBLE_DEVICES=2
cd "$BT" || exit 1
echo "=== debug fingerprints | clone HEAD $(git rev-parse --short HEAD) | $(date '+%F %T') | GPU 2 | $N runs"
python3 - "$BT" "$SRC" <<'PY' || exit 1
import os, shutil, sys
bt, src = sys.argv[1], sys.argv[2]
shutil.rmtree(src, ignore_errors=True)
shutil.copytree(os.path.join(bt, "dist/zero"), os.path.join(src, "dist/zero"))
p = os.path.join(src, "dist/zero/src/ShardedMuon.cpp")
s = open(p).read()
helper = r'''
// ---- DEBUG (scratch copy only): fingerprint of n floats on the device ----
static unsigned long long dbg_fp(const float* d, int64_t n) {
    cudaStreamSynchronize(ocuda::getCurrentStream());
    std::vector<float> h(static_cast<size_t>(n));
    cudaMemcpy(h.data(), d, static_cast<size_t>(n) * sizeof(float), cudaMemcpyDeviceToHost);
    unsigned long long x = 1469598103934665603ull;
    const unsigned char* b = reinterpret_cast<const unsigned char*>(h.data());
    for (size_t i = 0; i < h.size() * sizeof(float); ++i) { x ^= b[i]; x *= 1099511628211ull; }
    return x;
}
#define DBG_FP(stage, ptr, cnt) do { if (rec.numel == 9216) std::printf("DBG mode=%d step=%lld stage=%s fp=%016llx\n", \
    static_cast<int>(cfg_.collective), static_cast<long long>(step_count_), stage, dbg_fp((ptr), (cnt))); } while (0)

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
    assert c == 1, f"pattern found {c} times: {old[:60]!r}"
    s = s.replace(old, new)
if "#include <vector>" not in s: s = "#include <vector>\n#include <cstdio>\n" + s
open(p, "w").write(s)
print("debug copy patched (5 edits, each matched once)")
PY
base=$(make --no-print-directory -n build-mpi FILE="$BT/dist/zero/tests/test_muon_rsag.cpp" SM_ARCH=90 WITH_CUBLAS=1 NO_TMUX=1 2>/dev/null | grep -- '-o _runner_ ' | tail -1)
cmd=${base/-o _runner_ /-o $T/test_muon_rsag_DBG }
cmd=${cmd//dist\/zero\/src\//$SRC\/dist\/zero\/src\/}
cmd=${cmd/-Idist\/zero\/include/-I$SRC\/dist\/zero\/include}
eval "$cmd" > "$T/logs/compile_DBG.log" 2>&1 || { echo "COMPILE FAILED"; grep -i error "$T/logs/compile_DBG.log" | head; exit 1; }
echo "compiled"
for i in $(seq 1 "$N"); do mpirun -np 1 "$T/test_muon_rsag_DBG" > "$T/logs/dbg/run_$i.txt" 2>&1; done
python3 - "$T/logs/dbg" "$N" <<'PY'
import re, sys, os
d, n = sys.argv[1], int(sys.argv[2])
order = ["W0", "G", "M", "NSIN", "NSOUT", "W1"]
for i in range(1, n + 1):
    txt = open(os.path.join(d, f"run_{i}.txt")).read()
    verdict = " | ".join(re.findall(r"^\[(?:sync publish|staged refresh)\] (?:PASS|FAIL)", txt, re.M))
    lines = re.findall(r"^DBG mode=(\d+) step=(\d+) stage=(\w+) fp=([0-9a-f]+)", txt, re.M)
    # the test runs, per scenario: RB then RS. Split the DBG stream into 4 runs in order: sync-RB, sync-RS, staged-RB, staged-RS
    runs, cur, last_mode = [], [], None
    for mode, step, stage, fp in lines:
        if last_mode is not None and mode != last_mode: runs.append(cur); cur = []
        cur.append((int(step), stage, fp)); last_mode = mode
    if cur: runs.append(cur)
    msgs = []
    for sc, (a, b) in zip(["sync", "staged"], [(0, 1), (2, 3)]):
        if b >= len(runs): msgs.append(f"{sc}: missing"); continue
        first = next(((x[0], x[1]) for x, y in zip(runs[a], runs[b]) if x[2] != y[2]), None)
        msgs.append(f"{sc}: " + ("identical" if first is None else f"FIRST DIFFERENCE at step {first[0]} stage {first[1]}"))
    print(f"run {i:2d} | {verdict} | " + " ; ".join(msgs))
PY
echo "=== done $(date '+%T')"
