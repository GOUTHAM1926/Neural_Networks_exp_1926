#!/bin/bash
# Tulu 3 SFT run, RESUMED on the 4x H200 box 95.133.252.52, GPU 3 (2026-10-23). The 10-22 run stopped at step 9194
# (21:30 IST) and its box is gone; this continues it from blumodel_step_9000, downloaded from
# hf://buckets/gautam1926/checkpoints_sft (md5 404c2809.. / c14e3418.. = the 10-22 files).
#   EXACTLY the settings of the 10-22 run (B 4 x GRAD_ACCUM 5 x W 1, the same trainer binary md5 2e98b56b), so steps
#   9000..9194 should match run_resume1.log up to rounding. Two changes, neither touches the math:
#     - CUDA 13.0 libraries first in LD_LIBRARY_PATH: /usr/local/cuda on this box points to 13.4, the run used 13.0.
#     - BLU_CKPT_KEEP=1: the disk is shared and filling; 9000 and 6000 stay safe in the HF bucket.
#   Logs: run_resume_4x.log, status_resume_4x.txt, gpu_resume_4x.csv (new files; it refuses to start if the log exists).
# usage: setsid nohup bash ~/gautam_k/run_tulu3_sft_resume_4x.sh 5e-5 > /dev/null 2>&1 < /dev/null &
LR=$1
[ -n "$LR" ] || { echo "usage: $0 <peak lr, the same as the 10-22 run: 5e-5>"; exit 1; }
G=3
GK=$HOME/gautam_k; M=$GK/BluTrain_sft; BIN=$M/bluscript_sft_bin
DATA=$GK/sft_data/tulu3_noaya; INIT=$GK/ckpt/fullchess; RUN=$GK/runs/tulu3_sft_1022
LOG=$RUN/run_resume_4x.log; ST=$RUN/status_resume_4x.txt
C13=/usr/local/cuda-13.0
export PATH=$C13/bin:$PATH TMPDIR=$GK/tmp CUDA_VISIBLE_DEVICES=$G
cd $RUN || exit 3
say() { echo "$(TZ=Asia/Kolkata date '+%F %H:%M:%S') IST $*" | tee -a $ST; }
[ -e $LOG ] && { echo "ABORT: $LOG already exists (pick a new log name)"; exit 5; }
[ -d $C13/lib64 ] || { say "ABORT: no CUDA 13.0 at $C13"; exit 7; }
[ -z "$(nvidia-smi -i $G --query-compute-apps=pid --format=csv,noheader)" ] || { say "ABORT: GPU $G is busy"; exit 2; }
ls $RUN/checkpoints/blumodel_step_*/metadata.bin > /dev/null 2>&1 || { say "ABORT: no complete checkpoint in $RUN/checkpoints"; exit 6; }
free_gb=$(df -BG --output=avail $GK | tail -1 | tr -dc 0-9)
[ "$free_gb" -ge 25 ] || { say "ABORT: only ${free_gb} GB free (a checkpoint save needs 17.5 GB)"; exit 4; }
LDP=$(cd $M && make -s print_run_ld_path SM_ARCH=90 2>/dev/null | tail -n 1)
LDP=$C13/lib64:$LDP
say "resume start on GPU $G | lr $LR | trainer $(md5sum < $BIN | cut -c1-8) | ${free_gb} GB free | checkpoints: $(ls $RUN/checkpoints | tr '\n' ' ')| cublasLt: $(LD_LIBRARY_PATH=$LDP ldd $BIN | grep -m1 libcublasLt | awk '{print $3}')"
TZ=Asia/Kolkata nvidia-smi -i $G --query-gpu=timestamp,index,memory.used,power.draw,clocks.sm,temperature.gpu,utilization.gpu \
  --format=csv -l 30 >> $RUN/gpu_resume_4x.csv 2>&1 &
MON=$!
echo "=== RESUME $(TZ=Asia/Kolkata date '+%F %H:%M:%S') IST on 4x GPU $G: restarted from the newest complete checkpoint (blumodel_step_9000) ===" >> $LOG
( for v in $(env | grep -o '^BLU_[A-Z0-9_]*'); do unset "$v"; done
  export LD_LIBRARY_PATH=$LDP:$LD_LIBRARY_PATH BLU_SFT_DATA=$DATA BLU_INIT_FROM=$INIT BLU_ALL_BF16=1 \
         BLU_OPTIMIZER=muon BLU_SFT_LR=$LR BLU_SFT_MUON_LR=$LR BLU_SFT_B=4 BLU_SFT_ROWS_PER_STEP=20 BLU_SFT_EPOCHS=2 \
         BLU_SFT_VAL_FREQ=200 BLU_N_LAYERS=28 BLU_ROPE_THETA=50000 BLU_CKPT=1 BLU_CKPT_DIR=$RUN/checkpoints \
         BLU_CKPT_FREQ=1000 BLU_CKPT_KEEP=1
  mpirun --allow-run-as-root -np 1 --bind-to none -x LD_LIBRARY_PATH -x CUDA_VISIBLE_DEVICES -x BLU_SFT_DATA \
    -x BLU_INIT_FROM -x BLU_ALL_BF16 -x BLU_OPTIMIZER -x BLU_SFT_LR -x BLU_SFT_MUON_LR -x BLU_SFT_B \
    -x BLU_SFT_ROWS_PER_STEP -x BLU_SFT_EPOCHS -x BLU_SFT_VAL_FREQ -x BLU_N_LAYERS -x BLU_ROPE_THETA -x BLU_CKPT \
    -x BLU_CKPT_DIR -x BLU_CKPT_FREQ -x BLU_CKPT_KEEP $BIN < /dev/null >> $LOG 2>&1 )
rc=$?
kill $MON 2>/dev/null
say "end | exit $rc | last: $(grep -E '^step +[0-9]+ ' $LOG | tail -1 | cut -d'|' -f1-3)"
