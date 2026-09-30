#!/usr/bin/env bash
# Build one trainer binary with the Makefile's own build-mpi compile line, output renamed so the
# repo's _runner_ is never touched. CPU only (the libraries are already built).
#   usage: bash build_trainer_bin.sh <trainer.cpp> <output binary>
#   e.g.   bash build_trainer_bin.sh bluscript_zero_ckpt_bluccl_fused.cpp /root/gautam_k/f4/_runner_fused_old
set -uo pipefail
SRC=${1:?usage: build_trainer_bin.sh <trainer.cpp> <output>}; OUT=${2:?usage: build_trainer_bin.sh <trainer.cpp> <output>}
BT=/root/gautam_k/BluTrain
export PATH=/usr/local/cuda/bin:$PATH
cd $BT || exit 1
{
echo "=== build $SRC -> $OUT  $(date '+%F %T')  BluTrain $(git rev-parse --short HEAD)  TI $(git -C Tensor-Implementations rev-parse --short HEAD)"
git status --short
md5sum dist/zero/src/ShardedOptimizer.cpp dist/zero/src/SymmetricShardedOptimizer.cpp dist/zero/src/ShardedMuon.cpp $SRC
cmd=$(make --no-print-directory -n build-mpi FILE=$SRC SM_ARCH=90 2>/dev/null | grep -- '-o _runner_ ' | tail -1)
[ -z "$cmd" ] && { echo "ERROR: could not read the build-mpi compile line"; exit 1; }
cmd=${cmd/-o _runner_ /-o $OUT }
echo "--- compile (CPU): $cmd"
t0=$(date +%s)
if eval "$cmd"; then echo "--- OK in $(( $(date +%s) - t0 )) s"; ls -la $OUT; md5sum $OUT; else echo "--- COMPILE FAILED"; exit 1; fi
} 2>&1 | tee $OUT.build.log
