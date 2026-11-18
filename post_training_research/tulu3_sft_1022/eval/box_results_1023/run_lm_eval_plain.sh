#!/bin/bash
# Same as run_lm_eval.sh but WITHOUT the chat template (2026-10-23): for tasks built to score plain-text continuations
# (HellaSwag, ARC, WinoGrande, TruthfulQA MC2) and DROP, whose generation stops at the first "." (a chat-style answer
# "The first touchdown came from Russell, who ..." is cut before the answer; 11590 scored DROP F1 0.046 in chat mode).
# This is also how Tulu 3 / OLMES score these tasks, and the setting needed for the base-vs-SFT forgetting check.
# usage: CUDA_VISIBLE_DEVICES=3 bash run_lm_eval_plain.sh <model dir> <result name> <tasks> [extra lm_eval args]
set -e
M=${1:?model dir}; NAME=${2:?result name}; TASKS=${3:?tasks}; shift 3
export HF_HOME=/dev/shm/gautam_eval/hf_home TOKENIZERS_PARALLELISM=false HF_DATASETS_TRUST_REMOTE_CODE=1
OUT=/root/gautam_k/eval/results/$NAME
mkdir -p "$OUT"
echo "$(TZ=Asia/Kolkata date '+%F %H:%M:%S') IST | model $M | tasks $TASKS | NO chat template | extra: $* | GPU ${CUDA_VISIBLE_DEVICES:-unset}" | tee "$OUT/run.txt"
/root/gautam_k/rivalenv/bin/lm_eval --model hf --model_args "pretrained=$M,dtype=bfloat16" \
    --tasks "$TASKS" --batch_size auto --output_path "$OUT" --log_samples "$@" 2>&1 | tee -a "$OUT/run.txt"
echo "$(TZ=Asia/Kolkata date '+%F %H:%M:%S') IST | done" | tee -a "$OUT/run.txt"
