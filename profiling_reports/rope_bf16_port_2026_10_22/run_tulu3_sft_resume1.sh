#!/bin/bash
# Tulu 3 SFT run, RESUMED ON 1 GPU (GPU 0; the other GPUs were taken) (2026-10-22). The 1-GPU run (run_tulu3_sft.sh, GPU 0) was stopped at step 8723 (20:43 IST);
# this continues it from the newest complete checkpoint in the same BLU_CKPT_DIR (blumodel_step_8000) on GPU 0.
#   EXACTLY the settings of the 1-GPU run (B 4 x GRAD_ACCUM 5 x W 1), so steps 8000.. read the same rows and should
#   match run.log up to rounding. The trainer auto-resumes from the newest complete checkpoint (bluscript_sft.cpp:2207-2242).
#   Logs: APPENDS to run.log and status.txt. In run.log, the lines after the last '=== RESUME' marker are the resumed run
#   (steps 8000..end); before it, use steps 0..7999 (8000..8723 there were redone after the resume).
# usage: setsid nohup bash ~/gautam_k/run_tulu3_sft_resume1.sh 5e-5 > /dev/null 2>&1 < /dev/null &
LR=$1
[ -n "$LR" ] || { echo "usage: $0 <peak lr, the same as the 1-GPU run: 5e-5>"; exit 1; }
GK=$HOME/gautam_k; M=$GK/BluTrain_sft; BIN=$M/bluscript_sft_bin
DATA=$GK/sft_data/tulu3_noaya; INIT=$GK/ckpt/fullchess; RUN=$GK/runs/tulu3_sft_1022; ST=$RUN/status.txt
LOG=$RUN/run.log   # the SAME log: the resume appends after a '=== RESUME' marker line
export PATH=/usr/local/cuda/bin:$PATH TMPDIR=$GK/tmp CUDA_VISIBLE_DEVICES=0
cd $RUN || exit 3
say() { echo "$(TZ=Asia/Kolkata date '+%F %H:%M:%S') IST $*" | tee -a $ST; }
for g in 0; do
  [ -z "$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader)" ] || { say "ABORT: GPU $g is busy"; exit 2; }
done
ls -d $RUN/checkpoints/blumodel_step_* > /dev/null 2>&1 || { say "ABORT: no checkpoint in $RUN/checkpoints"; exit 6; }
free_gb=$(df -BG --output=avail $GK | tail -1 | tr -dc 0-9)
[ "$free_gb" -ge 40 ] || { say "ABORT: only ${free_gb} GB free"; exit 4; }
LDP=$(cd $M && make -s print_run_ld_path SM_ARCH=90 2>/dev/null | tail -n 1)
say "resume start on GPU 0 | lr $LR | trainer $(md5sum < $BIN | cut -c1-8) | ${free_gb} GB free | checkpoints: $(ls $RUN/checkpoints | tr '\n' ' ')"
TZ=Asia/Kolkata nvidia-smi -i 0 --query-gpu=timestamp,index,memory.used,power.draw,clocks.sm,temperature.gpu,utilization.gpu \
  --format=csv -l 30 >> $RUN/gpu.csv 2>&1 &
MON=$!
echo "=== RESUME $(TZ=Asia/Kolkata date '+%F %H:%M:%S') IST: restarted from the newest complete checkpoint; the step lines below continue the run (steps after that checkpoint are redone) ===" >> $LOG
( for v in $(env | grep -o '^BLU_[A-Z0-9_]*'); do unset "$v"; done
  export LD_LIBRARY_PATH=$LDP:$LD_LIBRARY_PATH BLU_SFT_DATA=$DATA BLU_INIT_FROM=$INIT BLU_ALL_BF16=1 \
         BLU_OPTIMIZER=muon BLU_SFT_LR=$LR BLU_SFT_MUON_LR=$LR BLU_SFT_B=4 BLU_SFT_ROWS_PER_STEP=20 BLU_SFT_EPOCHS=2 \
         BLU_SFT_VAL_FREQ=200 BLU_N_LAYERS=28 BLU_ROPE_THETA=50000 BLU_CKPT=1 BLU_CKPT_DIR=$RUN/checkpoints \
         BLU_CKPT_FREQ=1000 BLU_CKPT_KEEP=2
  mpirun --allow-run-as-root -np 1 --bind-to none -x LD_LIBRARY_PATH -x CUDA_VISIBLE_DEVICES -x BLU_SFT_DATA \
    -x BLU_INIT_FROM -x BLU_ALL_BF16 -x BLU_OPTIMIZER -x BLU_SFT_LR -x BLU_SFT_MUON_LR -x BLU_SFT_B \
    -x BLU_SFT_ROWS_PER_STEP -x BLU_SFT_EPOCHS -x BLU_SFT_VAL_FREQ -x BLU_N_LAYERS -x BLU_ROPE_THETA -x BLU_CKPT \
    -x BLU_CKPT_DIR -x BLU_CKPT_FREQ -x BLU_CKPT_KEEP $BIN < /dev/null >> $LOG 2>&1 )
rc=$?
kill $MON 2>/dev/null
say "end | exit $rc | last: $(grep -E '^step +[0-9]+ ' $LOG | tail -1 | cut -d'|' -f1-3)"
