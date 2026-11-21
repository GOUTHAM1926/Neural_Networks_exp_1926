#!/bin/bash
# Launch an SFT run with the settings of the 2026-10-22 Tülu 3 run (1.7B model navingv/fullchess, 1 GPU: 1.88 s per
# step on an H200, 11,590 steps, final val loss 0.8501). Every setting the run needs is set here; any other BLU_*
# variable in the caller's environment is removed first, so a leftover setting cannot silently change the run.
#
# usage (from the BluTrain root, after `bash sft/build_sft_trainer.sh`):
#   SFT_DATA=<dir with train/ and val/ from sft/prepare_sft_data.py> INIT_FROM=<DCP checkpoint dir> \
#   N_LAYERS=28 ROPE_THETA=50000 RUN_DIR=<new folder> [LR=5e-5] [NP=1] [GPUS=0] [ROWS=20] [B=4] \
#   [OPT=muon|adamw] [WARMUP=0.03] [VAL_FREQ=200] [EPOCHS=2] \
#   [D_MODEL= Q_HEADS= KV_HEADS= FFN_HIDDEN= VOCAB= EMBED_SCALE=1] \
#   [INIT_FROM_RAW=<dir from sft/hf_to_blutrain.py> instead of INIT_FROM] [EXPORT=1] \
#   [BIN=./bluscript_sft_bin] [RESUME=1] bash sft/run_sft.sh
#   EXPORT=1 writes the final weights to <RUN_DIR>/final.safetensors (sft/blutrain_to_hf.py makes a HF model of it).
#   The 135M base (2026-10-27): N_LAYERS=37 ROPE_THETA=50000 D_MODEL=512 Q_HEADS=4 KV_HEADS=2 FFN_HIDDEN=1408 VOCAB=49280
#   EMBED_SCALE=1 (the values of its pretraining run; like N_LAYERS / ROPE_THETA, they must match the checkpoint).
#   SFT_CP=U (2026-11-07): context parallelism, U GPUs share every row (e.g. SFT_CP=2 NP=2 GPUS=0,1); U must divide NP,
#   the query heads, the KV heads and T; not with SFT_TRAINED_ROWS (sft/README.md section 5.2).
#   SWA_WINDOW=W SWA_GLOBAL_EVERY=N (2026-11-20 step 9 (SWA in SFT)): sliding-window attention, the window of the run that made the base
#   (the 350M SWA base: SWA_WINDOW=1024 SWA_GLOBAL_EVERY=4); in packed rows each chat sees its own last W tokens. Like
#   ROPE_THETA, no weight depends on it: the trainer compares it with the base's record and stops on a mismatch;
#   SWA_MISMATCH_OK=1 allows a deliberate one (an ablation arm), printed as a warning (sft/README.md section 5.3).
#   watch:  tail -3 <RUN_DIR>/status.txt ; grep -E '^step' <RUN_DIR>/run.log | tail -3
#
# The settings, and why (sft/README.md section 8; README_SFT_checkpoint_for_DPO.md section 6):
#   ROWS 20 packed rows per optimizer step (B 4 x GRAD_ACCUM 5 x 1 GPU, ~128 chats). With NP > 1, ROWS must be a
#        multiple of B x NP (the trainer stops otherwise).
#   2 epochs (EPOCHS, 2026-10-30: the component-9 screen uses 1); the step count comes from the data (epochs x rows /
#        ROWS), BLU_MAX_STEPS is ignored in SFT mode.
#   Muon for the 2-D matrices + AdamW for the embedding and norms, both at peak LR (5e-5), 3% warm-up, cosine to LR/10:
#        the base was pretrained with Muon, and Moonlight (arXiv 2502.16982, sec. 3.5, Table 6) found Muon-pretrained +
#        Muon-SFT the best of 4 combinations on the Tülu 3 mix at lr 5e-5; our 10-07 sweep: 5e-5 no overfit.
#   BLU_ALL_BF16=1 (as the base was trained).
#   BLU_EMB_BWD=segmented (2026-10-30): the embedding backward pretraining uses (its YAML: embedding.backward:
#        segmented = sorted segments, fp32 sums, deterministic). Without it the library's built-in default runs: one
#        bf16 atomicAdd per token, not deterministic, and large errors on frequent tokens. Every SFT run before
#        2026-10-30 had that default because this script passes no YAML (SFT work doc section 45.108 part 5). Validation every 200 steps over the whole val set (the default 10 would
#        cost hours on a 2,000-chat val set). A checkpoint every 1000 steps and at the end, the newest 2 kept.
#   N_LAYERS / ROPE_THETA must be the values of the run that made INIT_FROM. A wrong layer count stops the load; a
#        wrong theta gives NO error, only a worse model (fullchess: 28 / 50000; this branch's built-in default theta is
#        10000). So both are required here.
#   RESUME=1 continues this RUN_DIR's own newest checkpoint (BLU_CKPT_LOAD=1), logging to a new run_resume_*.log.
set -u
cd "$(dirname "$0")/.." || exit 3
ROOT=$PWD
: "${SFT_DATA:?set SFT_DATA (folder with train/ and val/ written by sft/prepare_sft_data.py)}"
: "${RUN_DIR:?set RUN_DIR (a new folder for logs and checkpoints)}"
INIT_FROM=${INIT_FROM:-}; INIT_FROM_RAW=${INIT_FROM_RAW:-}; EXPORT=${EXPORT:-0}
[ -n "$INIT_FROM" ] && [ -n "$INIT_FROM_RAW" ] && { echo "set INIT_FROM or INIT_FROM_RAW, not both"; exit 1; }
LR=${LR:-5e-5}; NP=${NP:-1}; GPUS=${GPUS:-0}; ROWS=${ROWS:-20}; B=${B:-4}; RESUME=${RESUME:-0}
OPT=${OPT:-muon}; WARMUP=${WARMUP:-0.03}; VAL_FREQ=${VAL_FREQ:-200}; SCHEDULE=${SCHEDULE:-cosine}; EPOCHS=${EPOCHS:-2}
case "$OPT" in muon|adamw) ;; *) echo "OPT must be muon or adamw (got $OPT)"; exit 1 ;; esac
BIN=${BIN:-$ROOT/bluscript_sft_bin}
if [ -n "$INIT_FROM$INIT_FROM_RAW" ]; then
    : "${N_LAYERS:?set N_LAYERS to the layer count of the run that made INIT_FROM (fullchess: 28)}"
    : "${ROPE_THETA:?set ROPE_THETA to the RoPE theta of the run that made INIT_FROM (fullchess: 50000; a wrong one gives no error)}"
fi
[ -x "$BIN" ] || { echo "no trainer at $BIN: run bash sft/build_sft_trainer.sh first"; exit 1; }
[ -f "$SFT_DATA/train/meta.json" ] && [ -f "$SFT_DATA/val/meta.json" ] || { echo "$SFT_DATA needs train/ and val/ from prepare_sft_data.py"; exit 1; }
# 2026-10-31: the row length comes from the data (prepare_sft_data.py --seq-len writes meta.json "T"), so it is set in one place.
# An SFT_T given by hand must match it (the trainer's data loader stops on a mismatch anyway).
DATA_T=$(grep -m1 '"T":' "$SFT_DATA/train/meta.json" | tr -dc 0-9)
[ "$DATA_T" = "$(grep -m1 '"T":' "$SFT_DATA/val/meta.json" | tr -dc 0-9)" ] || { echo "train/ and val/ were made with different T"; exit 1; }
[ "${SFT_T:-$DATA_T}" = "$DATA_T" ] || { echo "SFT_T=$SFT_T but the data was made with T=$DATA_T"; exit 1; }
SFT_T=$DATA_T
mkdir -p "$RUN_DIR" && RUN_DIR=$(cd "$RUN_DIR" && pwd) && cd "$RUN_DIR" || exit 3
ST=$RUN_DIR/status.txt
say() { echo "$(TZ=Asia/Kolkata date '+%F %H:%M:%S') IST $*" | tee -a "$ST"; }
if [ "$RESUME" = 1 ]; then LOG=$RUN_DIR/run_resume_$(date +%m%d_%H%M).log
else LOG=$RUN_DIR/run.log; [ ! -f "$LOG" ] || { say "ABORT: $LOG exists (an earlier run): use a new RUN_DIR, or RESUME=1"; exit 5; }
fi
# The GPUs must not hold another training run (two would slow each other). Short eval jobs may share a GPU when 60 GB
# stay free (2026-10-28: the 135M grid on a node shared with colleagues' evals).
for g in ${GPUS//,/ }; do
    for p in $(nvidia-smi -i "$g" --query-compute-apps=pid --format=csv,noheader); do
        ps -o args= -p "$p" | grep -q bluscript && { say "ABORT: GPU $g holds another training run (pid $p)"; exit 2; }
    done
    [ "$(nvidia-smi -i "$g" --query-gpu=memory.free --format=csv,noheader,nounits)" -ge 60000 ] || { say "ABORT: GPU $g has < 60 GB free"; exit 2; }
done
free_gb=$(df -BG --output=avail "$RUN_DIR" | tail -1 | tr -dc 0-9)
[ "$free_gb" -ge 50 ] || { say "ABORT: only ${free_gb} GB free (two 1.7B checkpoints with optimizer state need ~35 GB)"; exit 4; }
LDP=$(cd "$ROOT" && make -s print_run_ld_path SM_ARCH=90 2>/dev/null | tail -n 1)   # libblublas.so is found only here
cp "$BIN" "$RUN_DIR/bluscript_sft_bin.used"   # the exact binary of this run
say "start | $OPT lr $LR warmup $WARMUP $SCHEDULE | epochs $EPOCHS | rows/step $ROWS (B $B x $NP GPU) | GPUs $GPUS | trainer $(md5sum < "$BIN" | cut -c1-8) | ${free_gb} GB free | log $(basename "$LOG")"
TZ=Asia/Kolkata nvidia-smi -i "$GPUS" --query-gpu=timestamp,index,memory.used,power.draw,clocks.sm,temperature.gpu,utilization.gpu \
    --format=csv -l 30 >> "$RUN_DIR/gpu.csv" 2>&1 &
MON=$!
VARS="BLU_SFT_DATA BLU_ALL_BF16 BLU_EMB_BWD BLU_OPTIMIZER BLU_SFT_LR BLU_SFT_MUON_LR BLU_SFT_B BLU_SFT_ROWS_PER_STEP BLU_SFT_EPOCHS BLU_SFT_WARMUP BLU_SFT_SCHEDULE
      BLU_SFT_VAL_FREQ BLU_CKPT BLU_CKPT_DIR BLU_CKPT_FREQ BLU_CKPT_KEEP BLU_CKPT_LOAD"
( for v in $(env | grep -o '^BLU_[A-Z0-9_]*'); do unset "$v"; done
  export LD_LIBRARY_PATH=$LDP:${LD_LIBRARY_PATH:-} CUDA_VISIBLE_DEVICES=$GPUS \
         BLU_SFT_DATA=$SFT_DATA BLU_ALL_BF16=1 BLU_EMB_BWD=segmented BLU_OPTIMIZER=$OPT BLU_SFT_LR=$LR BLU_SFT_MUON_LR=$LR \
         BLU_SFT_B=$B BLU_SFT_ROWS_PER_STEP=$ROWS BLU_SFT_EPOCHS=$EPOCHS BLU_SFT_VAL_FREQ=$VAL_FREQ BLU_SFT_WARMUP=$WARMUP BLU_SFT_SCHEDULE=$SCHEDULE \
         BLU_CKPT=1 BLU_CKPT_DIR=$RUN_DIR/checkpoints BLU_CKPT_FREQ=1000 BLU_CKPT_KEEP=2 BLU_CKPT_LOAD=$RESUME
  XS=""; for v in $VARS; do XS="$XS -x $v"; done
  if [ -n "$INIT_FROM" ]; then
      export BLU_INIT_FROM=$INIT_FROM BLU_N_LAYERS=$N_LAYERS BLU_ROPE_THETA=$ROPE_THETA
      XS="$XS -x BLU_INIT_FROM -x BLU_N_LAYERS -x BLU_ROPE_THETA"
  fi
  if [ -n "$INIT_FROM_RAW" ]; then
      export BLU_INIT_FROM_RAW=$INIT_FROM_RAW BLU_N_LAYERS=$N_LAYERS BLU_ROPE_THETA=$ROPE_THETA
      XS="$XS -x BLU_INIT_FROM_RAW -x BLU_N_LAYERS -x BLU_ROPE_THETA"
  fi
  if [ "$EXPORT" = 1 ]; then export BLU_EXPORT=$RUN_DIR/final.safetensors; XS="$XS -x BLU_EXPORT"; fi
  # 2026-11-06: + SFT_CP (context parallelism: U GPUs share each row, BLU_SFT_CP; unset = off). The previous line, kept:
  # for k in D_MODEL Q_HEADS KV_HEADS FFN_HIDDEN VOCAB EMBED_SCALE SFT_T SFT_TRAINED_ROWS; do   # the 135M base's shape (unset = built-in); SFT_T = row length; SFT_TRAINED_ROWS=1 = lm_head on trained rows only (2026-10-31, experimental, off by default)
  # 2026-11-20 step 9 (SWA in SFT): + SWA_WINDOW SWA_GLOBAL_EVERY SWA_MISMATCH_OK (sliding window; unset = off). The previous line, kept:
  # for k in D_MODEL Q_HEADS KV_HEADS FFN_HIDDEN VOCAB EMBED_SCALE SFT_T SFT_TRAINED_ROWS SFT_CP; do   # the 135M base's shape (unset = built-in); SFT_T = row length; SFT_TRAINED_ROWS=1 = lm_head on trained rows only (2026-10-31, experimental, off by default); SFT_CP = context parallelism
  for k in D_MODEL Q_HEADS KV_HEADS FFN_HIDDEN VOCAB EMBED_SCALE SFT_T SFT_TRAINED_ROWS SFT_CP SWA_WINDOW SWA_GLOBAL_EVERY SWA_MISMATCH_OK; do   # the base's shape (unset = built-in); SFT_T = row length; SFT_TRAINED_ROWS=1 = lm_head on trained rows only (2026-10-31, experimental, off by default); SFT_CP = context parallelism; SWA_* = sliding window (2026-11-20)
      if [ -n "${!k:-}" ]; then export "BLU_$k=${!k}"; XS="$XS -x BLU_$k"; fi
  done
  mpirun --allow-run-as-root -np "$NP" -x LD_LIBRARY_PATH -x CUDA_VISIBLE_DEVICES $XS "$BIN" < /dev/null > "$LOG" 2>&1 )
rc=$?
kill $MON 2>/dev/null
say "end | exit $rc | last: $(grep -E '^step +[0-9]+ ' "$LOG" | tail -1 | cut -d'|' -f1-3)"
