---
name: project-future-seq-len-no-default-no-cap
description: FUTURE (user 11-01) — prepare_sft_data.py --seq-len: no default (error if missing), no 131,072 cap, keep multiple-of-256; research deep first
metadata:
  type: project
---

prepare_sft_data.py today: `--seq-len` default 4096 (:668), `T % 256 or T > 131072` -> exit (:697-698). User wants:
no default (stop and ask for T), no fixed upper cap ("what if something new is invented"), keep the 256-multiple check.
Research first: other limits past 131k (grid / 32-bit indexes, memory: 135M at 131k B1 ~105-108 GB of 141 GB -> needs CP
or activation checkpointing), and loader / run_sft.sh / trainer. Work doc end of 45.131, plan Q&A 190. Not started.
Related: [[feedback_length_limits_follow_every_test]], [[project-future-cp-ulysses-sft]].
