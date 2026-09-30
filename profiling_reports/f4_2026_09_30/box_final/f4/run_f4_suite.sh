#!/usr/bin/env bash
# F4 test suite on the box, one GPU.   usage:  bash run_f4_suite.sh old|new
#   old = BEFORE the patch: F4's bf16 cases must FAIL (proves the test sees F4); everything else PASS
#   new = AFTER  the patch: everything PASS
# Runs, in order, and writes ONE log (/root/gautam_k/f4/f4_suite_<tag>.log) ending in a SUMMARY block:
#   1. test_f4_w1_exact_grad  (new, this folder)
#   2. test_f1_nonfinite      (F1, 36 cases; must stay 36/36)
#   3. test_zero              (Navin's v1 test, W=1; run through a COPY of build_and_test.sh with this
#                              box's paths -- his file hard-codes his machine's ROOT and is not edited)
#   4. test_muon_rsag         (Navin's Muon test at W=1: RS/AG must equal reduce/broadcast bit for bit)
# 1 and 2 are compiled with the Makefile's own build-mpi line (output renamed, _runner_ untouched).
set -uo pipefail
tag=${1:?usage: run_f4_suite.sh old|new}
PARTS=${PARTS:-"1 2 3 4"}   # e.g. PARTS="3 4" runs only those parts
want() { [[ " $PARTS " == *" $1 "* ]]; }
BT=/root/gautam_k/BluTrain; T=/root/gautam_k/f4; LOG=$T/f4_suite_${tag}_parts${PARTS// /}.log
export PATH=/usr/local/cuda/bin:$PATH CUDA_VISIBLE_DEVICES=0
cd $BT || exit 1
declare -A RES
{
echo "=== F4 suite ($tag)  $(date '+%F %T')  host $(hostname)  BluTrain $(git rev-parse --short HEAD)  TI $(git -C Tensor-Implementations rev-parse --short HEAD)"
git status --short
md5sum dist/zero/src/ShardedOptimizer.cpp dist/zero/src/SymmetricShardedOptimizer.cpp dist/zero/src/ShardedMuon.cpp
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader

build_one() {   # $1 = source, $2 = output binary
    local cmd
    cmd=$(make --no-print-directory -n build-mpi FILE=$1 SM_ARCH=90 2>/dev/null | grep -- '-o _runner_ ' | tail -1)
    [ -z "$cmd" ] && { echo "ERROR: could not read the build-mpi compile line"; return 1; }
    cmd=${cmd/-o _runner_ /-o $2 }
    echo "--- compile (CPU): $cmd"
    eval "$cmd" > $2.compile.log 2>&1 || { echo "COMPILE FAILED:"; tail -30 $2.compile.log; return 1; }
}

if want 1; then
echo; echo "################ 1. test_f4_w1_exact_grad"
if build_one $T/test_f4_w1_exact_grad.cpp $T/test_f4_$tag; then
    $T/test_f4_$tag > $T/out_f4_$tag.txt 2>&1; RES[f4]=$?; cat $T/out_f4_$tag.txt
else RES[f4]=compile_failed; fi
fi

if want 2; then
echo; echo "################ 2. test_f1_nonfinite"
if build_one $T/test_f1_nonfinite.cpp $T/test_f1_$tag; then
    $T/test_f1_$tag > $T/out_f1_$tag.txt 2>&1; RES[f1]=$?; cat $T/out_f1_$tag.txt
else RES[f1]=compile_failed; fi
fi

# Navin's own build scripts (dist/zero/build_and_test.sh, run_muon_rsag_test.sh) no longer compile:
# ZeroOptimizerBase.h now includes OptimShardSpec.h and the scripts do not pass
# -Idist/Distributed_Checkpointing/include. So his tests are built with the Makefile's own
# build-mpi line, exactly like 1 and 2. His files are not edited.
if want 3; then
echo; echo "################ 3. test_zero (Navin, v1, W=1: null process group, no mpirun)"
if build_one $BT/dist/zero/tests/test_zero.cpp $T/test_zero_$tag; then
    $T/test_zero_$tag > $T/out_zero_$tag.txt 2>&1; RES[zero]=$?; cat $T/out_zero_$tag.txt
else RES[zero]=compile_failed; fi
fi

if want 4; then
echo; echo "################ 4. test_muon_rsag (Navin, W=1, launched as his script does: mpirun -np 1)"
if build_one $BT/dist/zero/tests/test_muon_rsag.cpp $T/test_muon_rsag_$tag; then
    mpirun --allow-run-as-root -np 1 $T/test_muon_rsag_$tag > $T/out_rsag_$tag.txt 2>&1; RES[rsag]=$?; cat $T/out_rsag_$tag.txt
else RES[rsag]=compile_failed; fi
fi

echo; echo "==================== SUMMARY ($tag) ===================="
echo "F4 test      exit=${RES[f4]:-not run}   $(grep -h '^cases:' $T/out_f4_$tag.txt 2>/dev/null)"
echo "F1 test      exit=${RES[f1]:-not run}   $(grep -h '^cases:' $T/out_f1_$tag.txt 2>/dev/null)"
echo "test_zero    exit=${RES[zero]:-not run}   $(grep -h -E 'ALL PASSED|ALL TESTS PASSED|FAILURES PRESENT' $T/out_zero_$tag.txt 2>/dev/null | tail -2 | tr '\n' ' ')"
echo "muon_rsag    exit=${RES[rsag]:-not run}   $(grep -h -E 'ALL PASSED|ALL TESTS PASSED|FAILURES PRESENT' $T/out_rsag_$tag.txt 2>/dev/null | tail -2 | tr '\n' ' ')"
echo "F4 FAIL lines:"; grep -h '^CASE|.*|FAIL|' $T/out_f4_$tag.txt 2>/dev/null | cut -d'|' -f2-5 | sed 's/^/   /'
echo "F1 FAIL lines:"; grep -h '^CASE|.*|FAIL|' $T/out_f1_$tag.txt 2>/dev/null | cut -d'|' -f2-5 | sed 's/^/   /'
} 2>&1 | tee $LOG
echo; echo "log: $LOG"
