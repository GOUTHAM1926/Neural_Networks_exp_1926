#!/usr/bin/env bash
# F4 end-to-end check: ONE arm of the 1-GPU trainer A/B.
#   usage: bash run_trainer_ab.sh <binary> <arm name>        (STEPS=8 by default)
# The trainer is BLUCCL's bluscript_zero_ckpt_bluccl_fused.cpp (the only one that reads the flux
# shards this box has), with the flags of its own production arm, run_arm.sh, at ONE GPU:
# Muon for the body matrices, ZeRO v2 (forced by that trainer) for wte/lm_head/norms, all bf16,
# fused embedding backward. Checkpointing off. Runs in its own folder outside the repo; GPU memory
# is sampled every 4 s. Step 0 (train and val) is a forward on the same weights and data in every
# arm, so it must be IDENTICAL across arms; F4 can only change things from the first update on.
set -uo pipefail
BIN=${1:?usage: run_trainer_ab.sh <binary> <arm name>}; NAME=${2:?usage: run_trainer_ab.sh <binary> <arm name>}
BT=/root/gautam_k/BluTrain; T=/root/gautam_k/f4; D=$T/ab_$NAME; LOG=$D/$NAME.log
export PATH=/usr/local/cuda/bin:$PATH CUDA_VISIBLE_DEVICES=0
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
# ---- run_arm.sh's base + measurement flags, verbatim except NP ----
export BLU_SIZE_CLASS=exact BLU_OPTIMIZER=muon BLU_GLOBAL_CLIP=1
export BLU_ZERO_V2=1 BLU_SYMMETRIC=1 BLU_ALL_BF16=1 BLU_BF16_LOGITS=0
export BLUCCL_FUSED_EMB=1 BLU_FUSED_ADAMW=1
export BLU_MAX_STEPS=${STEPS:-8} BLU_CKPT=0
export BLU_DATA_ROOT=/root/.cache/huggingface/hub/datasets--navingv--flux/snapshots/5232b3d133d2eb71905725f288ff6b56dc0df773
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$BT/Tensor-Implementations/lib:$BT/Profiler/lib:$BT/dist/lib:${LD_LIBRARY_PATH:-}
mkdir -p $D && cd $D || exit 1
{
echo "=== trainer A/B arm '$NAME'  $(date '+%F %T')  host $(hostname)"
echo "binary $BIN  md5 $(md5sum < $BIN | cut -c1-32)"
echo "BluTrain $(git -C $BT rev-parse --short HEAD)  TI $(git -C $BT/Tensor-Implementations rev-parse --short HEAD)"
env | grep -E '^BLU' | sort
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
} > $LOG 2>&1
: > $D/$NAME.mem
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> $D/$NAME.mem; sleep 4; done ) &
SAMP=$!
t0=$(date +%s)
mpirun --allow-run-as-root -np 1 -x LD_LIBRARY_PATH -x CUDA_VISIBLE_DEVICES \
  $(env | sed -n 's/^\(BLU[A-Z_0-9]*\)=.*/-x \1/p' | tr '\n' ' ') $BIN >> $LOG 2>&1
RC=$?
kill $SAMP 2>/dev/null; wait $SAMP 2>/dev/null
echo "=== exit $RC  wall $(( $(date +%s) - t0 )) s  peak GPU MiB $(sort -n $D/$NAME.mem | tail -1)  $(date '+%F %T')" >> $LOG
grep -E '^step |\[val\]|DECLINED|INACTIVE|no effect|\(F4\)|ERROR|FATAL|Error|error:|out of memory|^=== exit' $LOG | cut -c1-260
