#!/bin/bash
# Second dev-suite pass, no chat template (run_lm_eval_plain.sh), on both checkpoints, GPU 3 only (2026-10-23).
# It waits until dev_suite.sh has written "ALL DEV SUITE DONE", so the two never share the GPU.
# usage (box): setsid nohup bash /root/gautam_k/eval/dev_plain.sh > /dev/null 2>&1 < /dev/null &
cd /root/gautam_k/eval
P=results/dev_suite_progress.txt
now() { TZ=Asia/Kolkata date '+%F %H:%M:%S'; }
until grep -q "ALL DEV SUITE DONE" $P 2>/dev/null; do sleep 30; done
TASKS=drop,hellaswag,arc_easy,arc_challenge,winogrande,truthfulqa_mc2
for s in 11590 6000; do
  echo "$(now) IST start plain pass (no chat template), step $s" >> $P
  CUDA_VISIBLE_DEVICES=3 bash run_lm_eval_plain.sh /dev/shm/gautam_eval/step_$s plain_$s $TASKS > /dev/null 2>&1
  echo "$(now) IST end   plain pass, step $s" >> $P
done
echo "$(now) IST ALL PLAIN PASS DONE" >> $P
