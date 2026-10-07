#!/bin/bash
# run_muon_sweep_1007.sh -- Muon learning-rate sweep for the stage-0 SFT run (2026-10-07).
# Three runs one after another, the same as today's 4-GPU AdamW run ("fourgpu", final val 1.280949 on
# 480 val rows) except BLU_OPTIMIZER=muon. Muon and the AdamW part (embeddings, norms) get the SAME LR,
# like Kathir's sft.py (one lr for ZeroMuonAdamW). His LR check: 2e-5 / 5e-5 / 1.2e-4 -> picked 5e-5.
#   muon_lr2e-5    redo of this afternoon's 1-GPU Muon run on the same GPU count, so its val is comparable
#   muon_lr5e-5    Kathir's pick
#   muon_lr1.2e-4  his highest
# No checkpoints (BLU_CKPT=0). Stops at the first failure, and refuses to start on a GPU that is in use.
# GPUs: default 3,4,5,6; override with GPUS=... (1, 2, 4 or 8 GPUs: 64 rows per step must split into
# micro-batches of 8 rows on every GPU).
# Progress: ~/gautam_k/sft_runs/muon_sweep_1007.log
# Start:  GPUS=3,4,5,6 nohup bash ~/gautam_k/sft_runs/run_muon_sweep_1007.sh > ~/gautam_k/sft_runs/muon_sweep_1007.out 2>&1 &
set -u
export PATH=/usr/local/cuda/bin:$PATH
G=$HOME/gautam_k
RUNS=$G/sft_runs
BIN=$G/BluTrain_sft/_runner_
QLOG=$RUNS/muon_sweep_1007.log
LDP=$(make -C "$G/BluTrain_sft" --no-print-directory print_run_ld_path)
SFT_DATA=$G/sft_data/stage0_ultrachat_5k_T4096
INIT=$G/ckpt/blumodel_step_8000
GPUS=${GPUS:-3,4,5,6}
NP=$(echo "$GPUS" | tr ',' '\n' | grep -c .)

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
    log "DONE  $name  final: $(grep '\[val\] step 155' run.log | tail -1)"
}

for LR in 2e-5 5e-5 1.2e-4; do
    run "muon_lr${LR}" "$NP" "$GPUS" BLU_SFT_DATA=$SFT_DATA BLU_INIT_FROM=$INIT BLU_ALL_BF16=1 BLU_CKPT=0 \
        BLU_OPTIMIZER=muon BLU_SFT_MUON_LR=$LR BLU_SFT_LR=$LR
done

log "SWEEP FINISHED"
