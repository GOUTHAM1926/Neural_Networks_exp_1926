#!/bin/bash
# Base model (fullchess, before SFT) for the forgetting check of the dev-suite plan (2026-10-23), GPU 3 only.
# 1. export it (CPU, now); 2. after dev_plain.sh writes "ALL PLAIN PASS DONE": prove the export (val loss of the SFT
#    val set must equal the trainer's [val] step 0 = 1.175723, measured on these same weights), then the plain pass
#    (no chat template: a base model is not a chat model) on the same tasks as dev_plain.sh.
# usage (box): setsid nohup bash /root/gautam_k/eval/base_queue.sh > /dev/null 2>&1 < /dev/null &
cd /root/gautam_k/eval
P=results/dev_suite_progress.txt
now() { TZ=Asia/Kolkata date '+%F %H:%M:%S'; }
echo "$(now) IST start export of the base model (fullchess, CPU)" >> $P
/root/gautam_k/rivalenv/bin/python export_to_hf.py /root/gautam_k/ckpt/fullchess /dev/shm/gautam_eval/base_fullchess > export_base.log 2>&1
echo "$(now) IST end   export of the base model: $(tail -1 export_base.log | cut -c1-80)" >> $P
until grep -q "ALL PLAIN PASS DONE" $P 2>/dev/null; do sleep 30; done
echo "$(now) IST start base export proof (expect 1.175723)" >> $P
CUDA_VISIBLE_DEVICES=3 /root/gautam_k/rivalenv/bin/python verify_export.py /dev/shm/gautam_eval/base_fullchess \
    /root/gautam_k/sft_data/tulu3_noaya/val --expect 1.175723 > verify_base.log 2>&1
echo "$(now) IST start plain pass, base model" >> $P
CUDA_VISIBLE_DEVICES=3 bash run_lm_eval_plain.sh /dev/shm/gautam_eval/base_fullchess plain_base \
    drop,hellaswag,arc_easy,arc_challenge,winogrande,truthfulqa_mc2 > /dev/null 2>&1
echo "$(now) IST ALL BASE DONE" >> $P
