#!/bin/bash
# run_queue_1007.sh -- the 2026-10-07 SFT test queue on the H200 box (our GPUs: 4 and 5).
# Runs 4 tests one after another and stops at the first failure. Every run starts from a
# clean environment (all BLU_* variables removed, then only that run's set) in its own
# folder under ~/gautam_k/sft_runs/. Same _runner_ binary for all four: no rebuilds.
#   1. resume78   resume today's AdamW run from its step-78 checkpoint -> steps 78-155   (GPU 4)
#   2. pretrain4  the 10-06 task-2 pretraining command, unchanged, 4 steps             (GPU 4)
#   3. twogpu     today's AdamW SFT run on 2 GPUs, no checkpoints                       (GPUs 4,5)
#   4. muon       today's SFT run with Muon instead of AdamW, LR 2e-5, no checkpoints  (GPU 4)
# Progress, one line per start / finish / failure: ~/gautam_k/sft_runs/queue_1007.log
# Start:  nohup bash ~/gautam_k/sft_runs/run_queue_1007.sh > ~/gautam_k/sft_runs/queue_1007.out 2>&1 &
set -u
export PATH=/usr/local/cuda/bin:$PATH
G=$HOME/gautam_k
RUNS=$G/sft_runs
BIN=$G/BluTrain_sft/_runner_
QLOG=$RUNS/queue_1007.log
LDP=$(make -C "$G/BluTrain_sft" --no-print-directory print_run_ld_path)
SFT_DATA=$G/sft_data/stage0_ultrachat_5k_T4096
INIT=$G/ckpt/blumodel_step_8000

log() { echo "$(date '+%F %T') $*" | tee -a "$QLOG"; }

run() {   # run <name> <ranks> <gpus> VAR=value ...
    local name=$1 np=$2 gpus=$3; shift 3
    local dir=$RUNS/$name
    mkdir -p "$dir" && cd "$dir" || { log "FAIL $name: cannot enter $dir"; exit 1; }
    for v in $(env | grep -o '^BLU_[A-Z0-9_]*'); do unset "$v"; done      # nothing leaks between runs
    export CUDA_VISIBLE_DEVICES=$gpus LD_LIBRARY_PATH=$LDP:${LD_LIBRARY_PATH:-}
    local xs="-x LD_LIBRARY_PATH -x CUDA_VISIBLE_DEVICES"
    for kv in "$@"; do export "$kv"; xs="$xs -x ${kv%%=*}"; done
    for g in ${gpus//,/ }; do                                            # never start on a busy GPU
        local used
        used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits)
        if [ "$used" -gt 1000 ]; then log "FAIL $name: GPU $g already has $used MiB in use"; exit 1; fi
    done
    log "START $name  ranks=$np gpus=$gpus  $*"
    mpirun -np "$np" $xs "$BIN" < /dev/null > run.log 2>&1
    local rc=$?
    if [ $rc -ne 0 ] || ! grep -q "Training complete" run.log; then log "FAIL $name: exit code $rc"; exit 1; fi
    log "DONE  $name"
}

# 1. Resume: a new checkpoint folder holding only a LINK to today's step-78 checkpoint (no 22 GB copy).
mkdir -p "$RUNS/resume78/ckpt"
ln -sfn "$RUNS/stage0_adamw_full/ckpt/blumodel_step_78" "$RUNS/resume78/ckpt/blumodel_step_78"
run resume78 1 4 BLU_SFT_DATA=$SFT_DATA BLU_INIT_FROM=$INIT BLU_ALL_BF16=1 \
    BLU_CKPT=1 BLU_CKPT_FREQ=78 BLU_CKPT_DIR=$RUNS/resume78/ckpt
grep -q "\[resume\] loaded DCP checkpoint at step 78" "$RUNS/resume78/run.log" \
    || { log "FAIL resume78: it did not resume at step 78"; exit 1; }

# 2. Pretraining check: exactly the 10-06 task-2 command (no BLU_SFT_DATA = pretraining mode).
run pretrain4 1 4 BLU_INIT_FROM=$INIT BLU_INIT_VERIFY=1 BLU_ALL_BF16=1 BLU_CKPT=0 \
    BLU_MAX_STEPS=4 BLU_MIX_ROOT=$G/mix BLU_TOKEN_MAX=50304

# 3. Two GPUs: same SFT run, 64 rows per step = B 8 x GRAD_ACCUM 4 x W 2.
run twogpu 2 4,5 BLU_SFT_DATA=$SFT_DATA BLU_INIT_FROM=$INIT BLU_ALL_BF16=1 BLU_CKPT=0

# 4. Muon: same SFT run, Muon for the matrices (LR 2e-5, same scale as AdamW's), AdamW for the rest.
run muon 1 4 BLU_SFT_DATA=$SFT_DATA BLU_INIT_FROM=$INIT BLU_ALL_BF16=1 BLU_CKPT=0 \
    BLU_OPTIMIZER=muon BLU_SFT_MUON_LR=2e-5

log "QUEUE FINISHED"
