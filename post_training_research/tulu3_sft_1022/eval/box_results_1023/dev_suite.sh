#!/bin/bash
# Dev suite (plan of 2026-10-08, work_docs/overfitting_memorization_real_learning_check_val_dev_suite_and_unseen_suite.md
# section 6) on both checkpoints, one after the other, on GPU 3 only (2026-10-23):
#   lm-eval in our chat template: ifeval, gsm8k_cot (8 worked examples as chat turns), drop, truthfulqa_mc2, hellaswag,
#   arc_easy, arc_challenge, winogrande; then the end-token rate (endtoken_rate.py, 200 val prompts).
#   PopQA and IFBench are not in this lm-eval version: left out. The answer val loss was measured by verify_export.py
#   (0.859864 for 6000, 0.850077 for 11590).
# usage (box): setsid nohup bash /root/gautam_k/eval/dev_suite.sh > /dev/null 2>&1 < /dev/null &
#   progress: cat /root/gautam_k/eval/results/dev_suite_progress.txt
cd /root/gautam_k/eval
P=results/dev_suite_progress.txt
mkdir -p results
TASKS=ifeval,gsm8k_cot,drop,truthfulqa_mc2,hellaswag,arc_easy,arc_challenge,winogrande
now() { TZ=Asia/Kolkata date '+%F %H:%M:%S'; }
for s in 11590 6000; do
  echo "$(now) IST start lm-eval dev suite, step $s" >> $P
  CUDA_VISIBLE_DEVICES=3 bash run_lm_eval.sh /dev/shm/gautam_eval/step_$s dev_$s $TASKS > /dev/null 2>&1
  echo "$(now) IST end   lm-eval dev suite, step $s" >> $P
  echo "$(now) IST start end-token rate, step $s" >> $P
  CUDA_VISIBLE_DEVICES=3 HF_HOME=/dev/shm/gautam_eval/hf_home /root/gautam_k/rivalenv/bin/python endtoken_rate.py \
      /dev/shm/gautam_eval/step_$s /root/gautam_k/sft_data/tulu3_noaya/val --out results/endtoken_$s.json \
      > results/endtoken_$s.log 2>&1
  echo "$(now) IST end   end-token rate, step $s" >> $P
done
echo "$(now) IST ALL DEV SUITE DONE" >> $P
