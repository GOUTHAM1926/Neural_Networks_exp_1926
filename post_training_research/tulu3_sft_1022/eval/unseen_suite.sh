#!/bin/bash
# Unseen suite (plan of 2026-10-08, run ONCE on the chosen checkpoint; 2026-10-23), GPU 3 only.
#   Chosen checkpoint: blumodel_step_11590 (the final one: val 0.8501 vs 0.8599, HellaSwag +1.1 points (paired z +6.0),
#   TruthfulQA MC2 -0.9 (z -4.1), the rest equal; the model handed to the DPO team).
#   agieval_en: scores answer options as plain text -> no chat template (run_lm_eval_plain.sh; in chat format such tasks
#               drop toward chance, see WinoGrande 0.52 in dev_suite.sh).
#   mmlu_pro  : generative (5 worked examples as chat turns, then "The answer is (X)") -> our chat template.
#   IFBench is not in this lm-eval version: left out.
# usage (box): setsid nohup bash /root/gautam_k/eval/unseen_suite.sh > /dev/null 2>&1 < /dev/null &
cd /root/gautam_k/eval
P=results/dev_suite_progress.txt
M=/dev/shm/gautam_eval/step_11590
now() { TZ=Asia/Kolkata date '+%F %H:%M:%S'; }
echo "$(now) IST start UNSEEN suite, step 11590: agieval_en (no chat template)" >> $P
CUDA_VISIBLE_DEVICES=3 bash run_lm_eval_plain.sh $M unseen_11590_agieval agieval_en > /dev/null 2>&1
echo "$(now) IST end   agieval_en; start mmlu_pro (chat template)" >> $P
CUDA_VISIBLE_DEVICES=3 bash run_lm_eval.sh $M unseen_11590_mmlu_pro mmlu_pro > /dev/null 2>&1
echo "$(now) IST ALL UNSEEN SUITE DONE" >> $P
