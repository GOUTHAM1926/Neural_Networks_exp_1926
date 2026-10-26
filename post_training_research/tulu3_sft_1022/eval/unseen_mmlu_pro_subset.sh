#!/bin/bash
# MMLU-Pro on a RANDOM subset (2026-10-23), part of the unseen suite on blumodel_step_11590, GPU 3 only.
#   Why a subset: the full 12,032 questions with written reasoning ran at 129 questions in 4.5 min (lm-eval projected
#   ~5 h): about 10% of this model's answers loop until the 2,048-token limit, and every batch waits for its longest answer.
#   150 random questions per subject (14 subjects, ~2,100 questions, seed 1023) give the score to about +-0.8 points.
#   Random, not lm-eval's --limit: --limit takes the FIRST n questions, and MMLU-Pro is ordered by source inside a subject.
#   The question numbers are written to results/mmlu_pro_subset_150.json, so the subset can be repeated exactly.
# usage (box): setsid nohup bash /root/gautam_k/eval/unseen_mmlu_pro_subset.sh > /dev/null 2>&1 < /dev/null &
cd /root/gautam_k/eval
P=results/dev_suite_progress.txt
now() { TZ=Asia/Kolkata date '+%F %H:%M:%S'; }
export HF_HOME=/dev/shm/gautam_eval/hf_home
/root/gautam_k/rivalenv/bin/python - > results/mmlu_pro_subset_150.json 2> results/mmlu_pro_subset_150.err <<'EOF'
import collections, json, random
from datasets import load_dataset
ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
by = collections.defaultdict(list)
for i, c in enumerate(ds["category"]):
    by[c].append(i)
rng = random.Random(1023)
out = {}
for c in sorted(by):
    n = len(by[c])
    out["mmlu_pro_" + c.replace(" ", "_")] = sorted(rng.sample(range(n), min(150, n)))   # positions inside the subject
print(json.dumps(out))
EOF
echo "$(now) IST start mmlu_pro RANDOM SUBSET ($(/root/gautam_k/rivalenv/bin/python -c "import json; d=json.load(open('results/mmlu_pro_subset_150.json')); print(len(d), 'subjects,', sum(map(len, d.values())), 'questions')"))" >> $P
CUDA_VISIBLE_DEVICES=3 bash run_lm_eval.sh /dev/shm/gautam_eval/step_11590 unseen_11590_mmlu_pro_subset mmlu_pro \
    --samples "$(cat results/mmlu_pro_subset_150.json)" > /dev/null 2>&1
echo "$(now) IST ALL UNSEEN SUITE DONE (agieval_en full + mmlu_pro random subset)" >> $P
