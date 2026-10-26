#!/bin/bash
# Stage C checks (2026-10-22, box 86.38.182.101, GPU 0). Short SFT runs on the Tulu data from navingv/fullchess
# (28 layers: BLU_N_LAYERS=28), each stopped after 4 minutes; then the step-0 val line and the first steps are printed.
#   bash ~/gautam_k/stagec_check.sh theta          two runs, bf16 RoPE, theta 10000 vs 500000. The checkpoint does not
#                                                  store theta; the one it was trained with gives a clearly LOWER
#                                                  step-0 val loss (the other one rotates q, k wrongly).   ~9 min
#   bash ~/gautam_k/stagec_check.sh rope <theta>   two runs at that theta: bf16 RoPE (new default) vs the old fp32
#                                                  path (BLU_QKROPE_FP32=1). train / val / norm must match exactly;
#                                                  only dt (step time) differs.                              ~9 min
# Logs: ~/gautam_k/stagec/<run>.log
GK=$HOME/gautam_k; M=$GK/BluTrain_sft; BIN=$M/bluscript_sft_bin
DATA=$GK/sft_data/tulu3_noaya; INIT=$GK/ckpt/fullchess; OUT=$GK/stagec
case "$1" in
  theta) RUNS="bf16_theta10000 bf16_theta500000" ;;
  rope)  [ -n "$2" ] || { echo "usage: $0 rope <theta>"; exit 1; }; RUNS="bf16_theta$2 fp32_theta$2" ;;
  *)     echo "usage: $0 theta | rope <theta>"; exit 1 ;;
esac
export PATH=/usr/local/cuda/bin:$PATH TMPDIR=$GK/tmp CUDA_VISIBLE_DEVICES=0
[ -z "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)" ] || { echo "ABORT: GPU 0 is busy"; exit 2; }
[ -f $DATA/train/seq_offsets.bin ] && [ -f $DATA/val/labels.bin ] || { echo "ABORT: data missing in $DATA"; exit 3; }
mkdir -p $OUT $GK/tmp
LDP=$(cd $M && make -s print_run_ld_path SM_ARCH=90 2>/dev/null | tail -n 1)
echo "trainer $(md5sum < $BIN | cut -c1-8) | start $(TZ=Asia/Kolkata date +%H:%M:%S) IST"
for run in $RUNS; do
  path=${run%%_*}; theta=${run##*_theta}
  ( for v in $(env | grep -o '^BLU_[A-Z0-9_]*'); do unset "$v"; done
    export LD_LIBRARY_PATH=$LDP:$LD_LIBRARY_PATH BLU_SFT_DATA=$DATA BLU_INIT_FROM=$INIT BLU_ALL_BF16=1 BLU_CKPT=0 \
           BLU_OPTIMIZER=adamw BLU_SFT_B=4 BLU_SFT_ROWS_PER_STEP=20 BLU_SFT_EPOCHS=2 BLU_SFT_VAL_FREQ=200 \
           BLU_N_LAYERS=28 BLU_ROPE_THETA=$theta BLU_QKROPE_FP32=$([ $path = fp32 ] && echo 1 || echo 0)
    timeout 240 mpirun --allow-run-as-root -np 1 -x LD_LIBRARY_PATH -x CUDA_VISIBLE_DEVICES -x BLU_SFT_DATA \
      -x BLU_INIT_FROM -x BLU_ALL_BF16 -x BLU_CKPT -x BLU_OPTIMIZER -x BLU_SFT_B -x BLU_SFT_ROWS_PER_STEP \
      -x BLU_SFT_EPOCHS -x BLU_SFT_VAL_FREQ -x BLU_N_LAYERS -x BLU_ROPE_THETA -x BLU_QKROPE_FP32 \
      $BIN < /dev/null > $OUT/$run.log 2>&1 )
  sleep 15   # let the stopped run release the GPU
done
for run in $RUNS; do
  echo "== $run"
  grep -E "n_layers +:|rope_theta +:" $OUT/$run.log | tr -s ' ' | tr '\n' ' '; echo
  grep -E "what\(\)" $OUT/$run.log
  grep -E "\[val\] step 0 " $OUT/$run.log
  grep -E "^step +[0-9]+ " $OUT/$run.log | head -6 | cut -d'|' -f1-3,5,6
done
echo "GPU 0 processes now: $(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader | wc -l) | end $(TZ=Asia/Kolkata date +%H:%M:%S) IST"
