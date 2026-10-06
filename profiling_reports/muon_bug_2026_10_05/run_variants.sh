#!/usr/bin/env bash
# Which of a3aba3e's 3 changes causes test_muon_rsag's W=1 failure? Each variant = the LATEST dist/zero
# (clone HEAD, includes our F4) with exactly ONE change switched back, in a scratch copy outside the repo:
#   S1  gradient up-cast: new cast_to_f32_scaled kernel  -> old convert_type_cuda_generic + muon_scale_inplace
#   S2  W=1 gradient norm: new sumsq_accumulate kernels  -> old owned_grad_fp32 + multi_tensor_grad_norm + sync
#   S3  W=1 buffers: in place (flat_grad_/flat_stage_ shared) -> separate shard_grad_/shard_param_ again
# Same libtensor/libdist, same compile line as run_rsag_arm.sh. Runs all variants at once on GPU 2 (the test
# is tiny, and CUDA_LAUNCH_BLOCKING=1 showed the failure is not a timing effect).
#   usage: bash run_variants.sh [runs]
set -uo pipefail
N=${1:-10}
BT=$HOME/gautam_k/BluTrain_bluccl; T=$HOME/gautam_k/muon; mkdir -p "$T/logs"
export PATH=/usr/local/cuda/bin:$PATH CUDA_VISIBLE_DEVICES=2
cd "$BT" || exit 1
echo "=== variants | clone HEAD $(git rev-parse --short HEAD) | TI $(git -C Tensor-Implementations rev-parse --short HEAD) | $(date '+%F %T') | GPU 2 | $N runs each"
python3 - "$BT" "$T" <<'PY' || exit 1
import os, shutil, sys
bt, t = sys.argv[1], sys.argv[2]
patches = {
 "S1": ("    const kernels::ElemKind kind = elem_kind(src_dtype);\n    if (kind != kernels::ElemKind::UNSUPPORTED) {",
        "    const kernels::ElemKind kind = elem_kind(src_dtype);\n    if (false && kind != kernels::ElemKind::UNSUPPORTED) {   // EXPERIMENT S1: old up-cast"),
 "S2": ("            if (gk == kernels::ElemKind::UNSUPPORTED) {\n                // Generic fallback; the sync protects",
        "            if (true || gk == kernels::ElemKind::UNSUPPORTED) {   // EXPERIMENT S2: old norm\n                // Generic fallback; the sync protects"),
 "S3": ("    inplace_collectives_ = (W_ == 1) ||\n                           (pg_ && pg_->backend() == dist::comm::Backend::NCCL &&\n                            zero_inplace_collectives_enabled());",
        "    inplace_collectives_ = (W_ == 1) ||\n                           (pg_ && pg_->backend() == dist::comm::Backend::NCCL &&\n                            zero_inplace_collectives_enabled());\n    inplace_collectives_ = false;   // EXPERIMENT S3: separate buffers"),
}
for name, (old, new) in patches.items():
    root = os.path.join(t, "var_" + name)
    shutil.rmtree(root, ignore_errors=True)
    shutil.copytree(os.path.join(bt, "dist/zero"), os.path.join(root, "dist/zero"))
    p = os.path.join(root, "dist/zero/src/ShardedMuon.cpp")
    s = open(p).read()
    n = s.count(old)
    assert n == 1, f"{name}: pattern found {n} times, expected 1"
    open(p, "w").write(s.replace(old, new))
    print(f"{name}: patched (1 match) -> {p.replace(os.path.expanduser('~'), '~')}")
PY
base=$(make --no-print-directory -n build-mpi FILE="$BT/dist/zero/tests/test_muon_rsag.cpp" SM_ARCH=90 WITH_CUBLAS=1 NO_TMUX=1 2>/dev/null | grep -- '-o _runner_ ' | tail -1)
[ -z "$base" ] && { echo "ERROR: could not read the build-mpi compile line"; exit 1; }
for V in S1 S2 S3; do
    SRC=$T/var_$V; cmd=${base/-o _runner_ /-o $T/test_muon_rsag_$V }
    cmd=${cmd//dist\/zero\/src\//$SRC\/dist\/zero\/src\/}
    cmd=${cmd/-Idist\/zero\/include/-I$SRC\/dist\/zero\/include}
    cmd=${cmd//$BT\/dist\/zero\/tests\//$SRC\/dist\/zero\/tests\/}
    ( eval "$cmd" > "$T/logs/compile_$V.log" 2>&1 && echo "$V compiled" || { echo "$V COMPILE FAILED"; tail -20 "$T/logs/compile_$V.log"; } ) &
done
wait
for V in S1 S2 S3; do [ -x "$T/test_muon_rsag_$V" ] || { echo "missing binary $V"; exit 1; }; done
echo "--- running S1 S2 S3 at the same time on GPU 2, $N runs each"
for V in S1 S2 S3; do
  ( for i in $(seq 1 "$N"); do mpirun -np 1 "$T/test_muon_rsag_$V" > "$T/logs/run_${V}_$i.txt" 2>&1; echo $? > "$T/logs/rc_${V}_$i"; done ) &
done
wait
for V in S1 S2 S3; do
  f=0; for i in $(seq 1 "$N"); do [ "$(cat "$T/logs/rc_${V}_$i")" != 0 ] && f=$((f+1)); done
  echo "--- $V: $f of $N runs FAILED"
  grep -h -E "FAIL .*differ" "$T"/logs/run_${V}_*.txt | sed -E 's/ \(worst.*//' | sort | uniq -c
done
echo "=== done $(date '+%T')"
