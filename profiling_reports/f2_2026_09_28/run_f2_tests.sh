#!/usr/bin/env bash
# F2 (fused grad accumulation) + LinearBackward fix: standalone tests on the box.
#
#   bash run_f2_tests.sh before   # UNPATCHED lib: LinearBackward test only (expect the bug: L1/L2/L5 FAIL)
#   bash run_f2_tests.sh after    # PATCHED lib: both tests, fused path ON, then OFF (BLUTRAIN_NO_FUSED_GRAD_ACC=1)
#
# One log with the full output and a SUMMARY block at the end:
#   ~/gautam_k/f2_logs/<tag>/f2_tests_<tag>.log
# GPU: GPU=<index> (default 0). Build the lib first with `make -j$(nproc)`; this script never builds it.
# MAKE_ARGS="WITH_BLUBLAS=1" when the lib was built with BluBLAS (the trainer build), so the test links the same way.
set -u
TAG=${1:?usage: run_f2_tests.sh before|after}
TI=${TI:-$HOME/gautam_k/BluTrain/Tensor-Implementations}
OUT=$HOME/gautam_k/f2_logs/$TAG
mkdir -p "$OUT"
LOG=$OUT/f2_tests_$TAG.log
export CUDA_VISIBLE_DEVICES=${GPU:-0}
cd "$TI" || exit 1

SO=lib/libtensor.so
[ -f "$SO" ] || { echo "$SO missing -- build the lib first (make -j\$(nproc))"; exit 1; }
# Stale-library guard: the patched symbol must be present AFTER and absent BEFORE.
NSYM=$(nm -D --defined-only "$SO" 2>/dev/null | grep -c cuda_accumulate_lowp_into_fp32)
if [ "$TAG" = after ] && [ "$NSYM" -eq 0 ]; then echo "lib has no cuda_accumulate_lowp_into_fp32 -- rebuild after patching"; exit 1; fi
if [ "$TAG" = before ] && [ "$NSYM" -ne 0 ]; then echo "lib already contains the F2 symbol -- this is not an unpatched lib"; exit 1; fi
if [ "$TAG" = after ] && [ src/TensorOps/cuda/TensorOpsAdd.cu -nt "$SO" -o src/core/AutogradMeta.cpp -nt "$SO" -o src/autograd/backward/LinearBackward.cpp -nt "$SO" ]; then
  echo "a patched source is newer than $SO -- rebuild first"; exit 1
fi

{
  echo "==== F2 standalone tests ($TAG)  $(date '+%F %T')  host=$(hostname)  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
  echo "Tensor-Implementations: $(git log -1 --oneline)"
  git status --short | head -20
  md5sum src/core/AutogradMeta.cpp src/TensorOps/cuda/TensorOpsAdd.cu include/ops/TensorOps.cuh \
         include/ops/helpers/KernelPolicy.h src/autograd/backward/LinearBackward.cpp "$SO"
  echo "F2 symbol in $SO: $NSYM   MAKE_ARGS=${MAKE_ARGS:-<none>}"
  echo
  echo "==== test_linear_backward_3d"
  make run-cuda FILE=Tests/kernel_tests/test_linear_backward_3d.cu ${MAKE_ARGS:-}
  if [ "$TAG" = after ]; then
    echo
    echo "==== test_fused_grad_accum (fused path ON)"
    make run-cuda FILE=Tests/kernel_tests/test_fused_grad_accum.cu ${MAKE_ARGS:-}
    echo
    echo "==== test_fused_grad_accum (BLUTRAIN_NO_FUSED_GRAD_ACC=1, fused path OFF in autograd)"
    BLUTRAIN_NO_FUSED_GRAD_ACC=1 make run-cuda FILE=Tests/kernel_tests/test_fused_grad_accum.cu ${MAKE_ARGS:-}
  fi
} 2>&1 | tee "$LOG"

# Count BEFORE appending the summary, so the summary's own lines are not counted.
NPASS=$(grep -c '^PASS' "$LOG"); NFAIL=$(grep -c '^FAIL' "$LOG")
SUMMARY=$(grep -E "^==== test_|^FAIL|^RESULT|^P6 |^PASS (L[0-9]|P[2356])" "$LOG")
{
  echo
  echo "==== SUMMARY ($TAG)"
  echo "$SUMMARY"
  echo "PASS lines: $NPASS   FAIL lines: $NFAIL"
  echo "log: $LOG"
} | tee -a "$LOG"
