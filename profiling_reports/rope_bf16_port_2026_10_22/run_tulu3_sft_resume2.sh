#!/bin/bash
# Tulu 3 SFT run, RESUMED ON 2 GPUs (2026-10-22). The 1-GPU run (run_tulu3_sft.sh, GPU 0) was stopped at ~step 8.5k;
# this continues it from the newest complete checkpoint in the same BLU_CKPT_DIR (blumodel_step_8000) on GPUs 0 and 1.
#   Same settings as the 1-GPU run, except the split of the 20 rows per step: B 2 x GRAD_ACCUM 5 x W 2 (was B 4 x 5 x 1).
#   With B 2 x W 2 every step reads exactly the same rows as on 1 GPU (rank r reads rows ((g*W + r)*B), SftDataLoader.h:177,
#   and an epoch is 115,904 / 4 = 28,976 micro-batches either way). The trainer auto-resumes (bluscript_sft.cpp:2207-2242):
#   model + Muon/AdamW state re-split for W=2 by DCP (ShardedMuon.h:315-321), data position skipped, LR schedule continued.
#   Numbers: same math; rounding-level differences only (gradient summation order across 2 GPUs).
#   Logs: run_resume2.log (steps 8000..end) next to run.log (steps 0..~8.5k of the 1-GPU run; use its steps 0..7999).
# usage: setsid nohup bash ~/gautam_k/run_tulu3_sft_resume2.sh 5e-5 > /dev/null 2>&1 < /dev/null &
LR=$1
[ -n "$LR" ] || { echo "usage: $0 <peak lr, the same as the 1-GPU run: 5e-5>"; exit 1; }
GK=$HOME/gautam_k; M=$GK/BluTrain_sft; BIN=$M/bluscript_sft_bin
DATA=$GK/sft_data/tulu3_noaya; INIT=$GK/ckpt/fullchess; RUN=$GK/runs/tulu3_sft_1022; ST=$RUN/status_resume2.txt
LOG=$RUN/run_resume2.log
export PATH=/usr/local/cuda/bin:$PATH TMPDIR=$GK/tmp CUDA_VISIBLE_DEVICES=0,1
cd $RUN || exit 3
say() { echo "$(TZ=Asia/Kolkata date '+%F %H:%M:%S') IST $*" | tee -a $ST; }
[ ! -f $LOG ] || { say "ABORT: $LOG exists; move it away first"; exit 5; }
for g in 0 1; do
  [ -z "$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader)" ] || { say "ABORT: GPU $g is busy"; exit 2; }
done
ls -d $RUN/checkpoints/blumodel_step_* > /dev/null 2>&1 || { say "ABORT: no checkpoint in $RUN/checkpoints"; exit 6; }
free_gb=$(df -BG --output=avail $GK | tail -1 | tr -dc 0-9)
[ "$free_gb" -ge 40 ] || { say "ABORT: only ${free_gb} GB free"; exit 4; }
LDP=$(cd $M && make -s print_run_ld_path SM_ARCH=90 2>/dev/null | tail -n 1)
say "resume start on GPUs 0,1 | lr $LR | trainer $(md5sum < $BIN | cut -c1-8) | ${free_gb} GB free | checkpoints: $(ls $RUN/checkpoints | tr '\n' ' ')"
TZ=Asia/Kolkata nvidia-smi -i 0,1 --query-gpu=timestamp,index,memory.used,power.draw,clocks.sm,temperature.gpu,utilization.gpu \
  --format=csv -l 30 > $RUN/gpu_resume2.csv 2>&1 &
MON=$!
( for v in $(env | grep -o '^BLU_[A-Z0-9_]*'); do unset "$v"; done
  export LD_LIBRARY_PATH=$LDP:$LD_LIBRARY_PATH BLU_SFT_DATA=$DATA BLU_INIT_FROM=$INIT BLU_ALL_BF16=1 \
         BLU_OPTIMIZER=muon BLU_SFT_LR=$LR BLU_SFT_MUON_LR=$LR BLU_SFT_B=2 BLU_SFT_ROWS_PER_STEP=20 BLU_SFT_EPOCHS=2 \
         BLU_SFT_VAL_FREQ=200 BLU_N_LAYERS=28 BLU_ROPE_THETA=50000 BLU_CKPT=1 BLU_CKPT_DIR=$RUN/checkpoints \
         BLU_CKPT_FREQ=1000 BLU_CKPT_KEEP=2
  mpirun --allow-run-as-root -np 2 --bind-to none -x LD_LIBRARY_PATH -x CUDA_VISIBLE_DEVICES -x BLU_SFT_DATA \
    -x BLU_INIT_FROM -x BLU_ALL_BF16 -x BLU_OPTIMIZER -x BLU_SFT_LR -x BLU_SFT_MUON_LR -x BLU_SFT_B \
    -x BLU_SFT_ROWS_PER_STEP -x BLU_SFT_EPOCHS -x BLU_SFT_VAL_FREQ -x BLU_N_LAYERS -x BLU_ROPE_THETA -x BLU_CKPT \
    -x BLU_CKPT_DIR -x BLU_CKPT_FREQ -x BLU_CKPT_KEEP $BIN < /dev/null > $LOG 2>&1 )
rc=$?
kill $MON 2>/dev/null
say "end | exit $rc | last: $(grep -E '^step +[0-9]+ ' $LOG | tail -1 | cut -d'|' -f1-3)"
