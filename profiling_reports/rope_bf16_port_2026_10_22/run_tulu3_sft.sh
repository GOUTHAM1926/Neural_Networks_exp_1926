#!/bin/bash
# Tulu 3 SFT run (2026-10-22, box 86.38.182.101, GPU 0).
#   model : navingv/fullchess = Navin's pretraining run (log bluscript_zero_ckpt_022.log: 28 layers, d 2304, 18/6 heads,
#           ffn 6144, T 4096, rope_theta 50000 NeoX, vocab 50304 tied, resid/logits/embed bf16, 1,701,485,824 params,
#           Muon + AdamW, 22,600 steps x 1,048,576 tokens). The step-0 val of this run must be BELOW 1.2301 (the val
#           at theta 10000; 500000 gave 1.2596), confirming 50000.
#   data  : tulu-3-sft-olmo-2-mixture-0225 without Aya; train 741,413 chats -> 115,904 packed rows; val 2,000 chats
#   step  : 20 rows (B 4 x GRAD_ACCUM 5, ~128 chats); 2 epochs = ~11,590 steps; val every 200 steps and at the last step
#   optim : Muon (matrices) + AdamW (embedding, norms), both at peak lr = $1, min lr = peak/10, warmup 3%, then cosine.
#           Muon because the base was pretrained with Muon: Moonlight (arXiv 2502.16982, sec. 3.5, Table 6) finds
#           Muon-pretrained + Muon-SFT the best of the 4 combinations, SFT on the Tulu 3 mix, 2 epochs, lr 5e-5.
#           Our 10-07 sweep (work doc 37.5, 5,000 chats): Muon 2e-5 val 1.3162, 5e-5 1.2843 (still falling at the end),
#           1.2e-4 best 1.2747 at the end of epoch 1 then overfit in epoch 2 (train 1.08).
#   ckpt  : every 1000 steps + the last step, newest 2 kept, in $RUN/checkpoints
#   RoPE  : the bf16 kernel with per-chat positions (stages A-C passed)
#   time  : ~1.66 s/step -> ~5.5 h
# usage: setsid nohup bash ~/gautam_k/run_tulu3_sft.sh <peak lr, e.g. 2e-5> > /dev/null 2>&1 < /dev/null &
#        watch: tail -3 ~/gautam_k/runs/tulu3_sft_1022/status.txt ; grep -E '^step' ~/gautam_k/runs/tulu3_sft_1022/run.log | tail -3
LR=$1
[ -n "$LR" ] || { echo "usage: $0 <peak lr>"; exit 1; }
GK=$HOME/gautam_k; M=$GK/BluTrain_sft; BIN=$M/bluscript_sft_bin
DATA=$GK/sft_data/tulu3_noaya; INIT=$GK/ckpt/fullchess; RUN=$GK/runs/tulu3_sft_1022; ST=$RUN/status.txt
export PATH=/usr/local/cuda/bin:$PATH TMPDIR=$GK/tmp CUDA_VISIBLE_DEVICES=0
mkdir -p $RUN $GK/tmp && cd $RUN || exit 3
say() { echo "$(TZ=Asia/Kolkata date '+%F %H:%M:%S') IST $*" | tee -a $ST; }
[ ! -f $RUN/run.log ] || { say "ABORT: $RUN/run.log exists (an earlier run); move it away first"; exit 5; }
[ -z "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)" ] || { say "ABORT: GPU 0 is busy"; exit 2; }
free_gb=$(df -BG --output=avail $GK | tail -1 | tr -dc 0-9)
[ "$free_gb" -ge 80 ] || { say "ABORT: only ${free_gb} GB free (2 checkpoints need ~50 GB)"; exit 4; }
LDP=$(cd $M && make -s print_run_ld_path SM_ARCH=90 2>/dev/null | tail -n 1)
cp $BIN $RUN/bluscript_sft_bin.used   # the exact binary of this run
say "start | lr $LR | trainer $(md5sum < $BIN | cut -c1-8) | ${free_gb} GB free"
TZ=Asia/Kolkata nvidia-smi -i 0 --query-gpu=timestamp,memory.used,power.draw,clocks.sm,temperature.gpu,utilization.gpu \
  --format=csv -l 30 > $RUN/gpu.csv 2>&1 &
MON=$!
( for v in $(env | grep -o '^BLU_[A-Z0-9_]*'); do unset "$v"; done
  export LD_LIBRARY_PATH=$LDP:$LD_LIBRARY_PATH BLU_SFT_DATA=$DATA BLU_INIT_FROM=$INIT BLU_ALL_BF16=1 \
         BLU_OPTIMIZER=muon BLU_SFT_LR=$LR BLU_SFT_MUON_LR=$LR BLU_SFT_B=4 BLU_SFT_ROWS_PER_STEP=20 BLU_SFT_EPOCHS=2 BLU_SFT_VAL_FREQ=200 \
         BLU_N_LAYERS=28 BLU_ROPE_THETA=50000 BLU_CKPT=1 BLU_CKPT_DIR=$RUN/checkpoints BLU_CKPT_FREQ=1000 BLU_CKPT_KEEP=2
  mpirun --allow-run-as-root -np 1 -x LD_LIBRARY_PATH -x CUDA_VISIBLE_DEVICES -x BLU_SFT_DATA -x BLU_INIT_FROM \
    -x BLU_ALL_BF16 -x BLU_OPTIMIZER -x BLU_SFT_LR -x BLU_SFT_MUON_LR -x BLU_SFT_B -x BLU_SFT_ROWS_PER_STEP -x BLU_SFT_EPOCHS \
    -x BLU_SFT_VAL_FREQ -x BLU_N_LAYERS -x BLU_ROPE_THETA -x BLU_CKPT -x BLU_CKPT_DIR -x BLU_CKPT_FREQ -x BLU_CKPT_KEEP \
    $BIN < /dev/null > $RUN/run.log 2>&1 )
rc=$?
kill $MON 2>/dev/null
say "end | exit $rc | last: $(grep -E '^step +[0-9]+ ' $RUN/run.log | tail -1 | cut -d'|' -f1-3)"
