#!/usr/bin/env bash
# Is test_muon_rsag's "staged refresh" failure at W=1 older than a3aba3e, or new in it?
# Builds Navin's test_muon_rsag.cpp against the dist/zero sources of a GIVEN BluTrain commit
# (headers + the three optimizer .cpp files), with everything else identical: the same TI
# (the clone's working copy), the same libtensor/libdist, the same compile line (the Makefile's
# own build-mpi line). The commit's dist/zero is extracted OUTSIDE the repo with git archive,
# which only reads. Then runs it the way his script does: mpirun -np 1.
#   usage: bash run_rsag_baseline.sh <commit>       e.g. e6eaeba (= before a3aba3e)
set -uo pipefail
C=${1:?usage: run_rsag_baseline.sh <commit>}
BT=/root/gautam_k/BluTrain; T=/root/gautam_k/f4; B=$T/base_$C; LOG=$T/rsag_base_$C.log
export PATH=/usr/local/cuda/bin:$PATH CUDA_VISIBLE_DEVICES=0
cd $BT || exit 1
{
echo "=== test_muon_rsag against dist/zero of $C  $(date '+%F %T')  clone HEAD $(git rev-parse --short HEAD)  TI $(git -C Tensor-Implementations rev-parse --short HEAD)"
rm -rf $B && mkdir -p $B && git archive $C dist/zero | tar -x -C $B || { echo "ERROR: git archive $C failed"; exit 1; }
md5sum $B/dist/zero/src/ShardedMuon.cpp $B/dist/zero/include/ShardedMuon.h $B/dist/zero/tests/test_muon_rsag.cpp dist/zero/tests/test_muon_rsag.cpp
cmd=$(make --no-print-directory -n build-mpi FILE=$B/dist/zero/tests/test_muon_rsag.cpp SM_ARCH=90 2>/dev/null | grep -- '-o _runner_ ' | tail -1)
[ -z "$cmd" ] && { echo "ERROR: could not read the build-mpi compile line"; exit 1; }
cmd=${cmd/-o _runner_ /-o $T/test_muon_rsag_base_$C }
cmd=${cmd//dist\/zero\/src\//$B\/dist\/zero\/src\/}          # that commit's optimizer sources
cmd=${cmd/-Idist\/zero\/include/-I$B\/dist\/zero\/include}   # ... and its headers, in place of the clone's
echo "--- compile (CPU): $cmd"
eval "$cmd" > $T/test_muon_rsag_base_$C.compile.log 2>&1 || { echo "COMPILE FAILED:"; tail -30 $T/test_muon_rsag_base_$C.compile.log; exit 1; }
echo "--- run: mpirun -np 1"
mpirun --allow-run-as-root -np 1 $T/test_muon_rsag_base_$C
echo "--- exit code $?"
} 2>&1 | tee $LOG
echo "log: $LOG"
