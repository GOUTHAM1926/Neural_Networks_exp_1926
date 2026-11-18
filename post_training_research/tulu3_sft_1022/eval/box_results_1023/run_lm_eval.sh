#!/bin/bash
# Dev / unseen suite runner (2026-10-23): EleutherAI lm-evaluation-harness on an exported model (export_to_hf.py),
# in our chat template (zero-shot / the task's default shots as chat turns), on ONE GPU (CUDA_VISIBLE_DEVICES).
#   The 4x box disk is shared and ~98% full, so every cache goes to RAM: HF_HOME (datasets, hub) under /dev/shm.
#   Results (small json + per-sample logs) go to /root/gautam_k/eval/results/<name>.
# usage: CUDA_VISIBLE_DEVICES=3 bash run_lm_eval.sh <model dir> <result name> <comma-separated tasks> [extra lm_eval args]
#   e.g. CUDA_VISIBLE_DEVICES=3 bash run_lm_eval.sh /dev/shm/gautam_eval/step_11590 smoke_11590 ifeval,arc_easy --limit 20
set -e
M=${1:?model dir}; NAME=${2:?result name}; TASKS=${3:?tasks}; shift 3
export HF_HOME=/dev/shm/gautam_eval/hf_home TOKENIZERS_PARALLELISM=false HF_DATASETS_TRUST_REMOTE_CODE=1
OUT=/root/gautam_k/eval/results/$NAME
mkdir -p "$OUT"
echo "$(TZ=Asia/Kolkata date '+%F %H:%M:%S') IST | model $M | tasks $TASKS | extra: $* | GPU ${CUDA_VISIBLE_DEVICES:-unset}" | tee "$OUT/run.txt"
/root/gautam_k/rivalenv/bin/lm_eval --model hf --model_args "pretrained=$M,dtype=bfloat16" \
    --tasks "$TASKS" --apply_chat_template --fewshot_as_multiturn --batch_size auto \
    --output_path "$OUT" --log_samples "$@" 2>&1 | tee -a "$OUT/run.txt"
echo "$(TZ=Asia/Kolkata date '+%F %H:%M:%S') IST | done" | tee -a "$OUT/run.txt"
