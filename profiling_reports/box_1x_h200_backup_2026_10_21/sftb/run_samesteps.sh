#!/bin/bash
# 2026-10-21 (1x H200, GPU 0): the exact padding-vs-packing comparison. A packed SFT run whose step s holds EXACTLY the 64
# chats of step s of the padded run C_P_padded (data: build_samesteps.py -> stage0_5k_samesteps_as_padded; 28 rows per step,
# 1 epoch = the same 156 steps and learning rates; val = the same 496 val chats, dense). Everything else as phase C of
# run_all_sftb.sh. Compare step by step with runs/C_P_padded (compare_samesteps.py). ~11 min.
# usage: setsid nohup bash ~/gautam_k/sftb/run_samesteps.sh > /dev/null 2>&1 < /dev/null &
GK=$HOME/gautam_k; S=$GK/sftb; M=$GK/sft_v2/BluTrain_sft; BIN=$M/bluscript_sft_bin; INIT=$GK/ckpt/blumodel_step_8000
DATA=$GK/sft_data/stage0_5k_samesteps_as_padded; R=$S/runs; ST=$S/run_status; d=$R/C_S_samesteps
export PATH=/usr/local/cuda/bin:$PATH TMPDIR=$GK/tmp CUDA_VISIBLE_DEVICES=0
say() { echo "$(TZ=Asia/Kolkata date +%H:%M:%S) IST $*" | tee -a $ST; }
[ -z "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)" ] || { say "S ABORT: GPU 0 busy: $(nvidia-smi -i 0 --query-compute-apps=pid,process_name --format=csv,noheader | tr '\n' ' ')"; exit 2; }
[ -f $DATA/train/seq_offsets.bin ] && [ -f $DATA/val/labels.bin ] || { say "S ABORT: data missing in $DATA"; exit 3; }
mkdir -p $d && cd $d || exit 3
LDP=$(cd $M && make -s print_run_ld_path SM_ARCH=90 2>/dev/null | tail -n 1)
say "S samesteps start | data $(basename $DATA) | B 4 | rows per step 28 | epochs 1 | trainer $(md5sum < $BIN | cut -c1-8)"
TZ=Asia/Kolkata nvidia-smi -i 0 --query-gpu=timestamp,memory.used,power.draw,clocks.sm,temperature.gpu,utilization.gpu --format=csv -l 1 > $d/gpu.csv 2>&1 &
MEM=$!
( for v in $(env | grep -o '^BLU_[A-Z0-9_]*'); do unset "$v"; done
  export LD_LIBRARY_PATH=$LDP:$LD_LIBRARY_PATH BLU_SFT_DATA=$DATA BLU_INIT_FROM=$INIT BLU_ALL_BF16=1 BLU_CKPT=0 \
         BLU_OPTIMIZER=adamw BLU_SFT_B=4 BLU_SFT_ROWS_PER_STEP=28 BLU_SFT_EPOCHS=1
  timeout 7200 mpirun --allow-run-as-root -np 1 -x LD_LIBRARY_PATH -x CUDA_VISIBLE_DEVICES -x BLU_SFT_DATA -x BLU_INIT_FROM \
    -x BLU_ALL_BF16 -x BLU_CKPT -x BLU_OPTIMIZER -x BLU_SFT_B -x BLU_SFT_ROWS_PER_STEP -x BLU_SFT_EPOCHS $BIN < /dev/null > run.log 2>&1 )
rc=$?
kill $MEM 2>/dev/null; wait $MEM 2>/dev/null
say "S samesteps exit=$rc | steps $(grep -c '^step ' run.log) | $(grep -E 'SftLoader' run.log | head -1 | cut -c1-110) | last $(grep '^step ' run.log | tail -1 | cut -c1-105) | $(grep '\[val\]' run.log | tail -1 | sed 's/^ *//') | peak GPU memory $(awk -F', ' 'NR>1 {v=$2+0; if (v>m) m=v} END {print m}' gpu.csv) MiB | $(grep -c 'Training complete' run.log) 'Training complete'"
