#!/bin/bash
# F1 test on the box.  usage:  bash run_f1_test.sh old|new
#   old = BEFORE the patch (v2 and Muon must FAIL the NaN/Inf cases, v1 must PASS)
#   new = AFTER  the patch (everything must PASS)
# Builds test_f1_nonfinite.cpp together with the 3 ZeRO sources exactly the way the trainer is
# built (the Makefile's own build-mpi compile line, output renamed so _runner_ is not touched),
# then runs it on GPU 0. One log per run: /root/gautam_k/f1/f1_<tag>.log
set -uo pipefail
tag=${1:?usage: run_f1_test.sh old|new}
BT=/root/gautam_k/BluTrain; T=/root/gautam_k/f1; LOG=$T/f1_$tag.log
cd $BT || exit 1
{
echo "=== F1 test ($tag)  $(date '+%F %T')  BluTrain $(git rev-parse --short HEAD)"
md5sum dist/zero/src/ShardedMuon.cpp dist/zero/src/SymmetricShardedOptimizer.cpp dist/zero/src/ShardedOptimizer.cpp
CMD=$(make --no-print-directory -n build-mpi FILE=$T/test_f1_nonfinite.cpp WITH_BLUBLAS=1 SM_ARCH=90 2>/dev/null | grep -- '-o _runner_ ' | tail -1)
if [ -z "$CMD" ]; then echo "ERROR: could not read the build-mpi compile line"; exit 1; fi
CMD=${CMD/-o _runner_ /-o $T/test_f1_$tag }
echo "--- compile (CPU only): $CMD"
t0=$(date +%s)
if ! eval "$CMD" > $T/compile_$tag.log 2>&1; then echo "COMPILE FAILED:"; tail -30 $T/compile_$tag.log; exit 1; fi
echo "--- compiled in $(( $(date +%s) - t0 )) s"
echo "--- new message strings in the binary (old: 0, new: 2): $(strings $T/test_f1_$tag | grep -c 'ShardedMuon: grad norm^2\|SymmetricShardedOptimizer: grad norm^2')"
echo "--- run on GPU 0"
CUDA_VISIBLE_DEVICES=0 $T/test_f1_$tag
echo "--- exit code $?"
} 2>&1 | tee $LOG
echo; echo "log: $LOG"
