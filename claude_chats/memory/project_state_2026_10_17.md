---
name: project_state_2026_10_17
description: "DONE 10-19 morning (new box 31.56.109.89, /root/gautam_K, CUDA 13.4; L1b raced = REJECTED). Was: we stopped at 'race L1b vs K5q' (not built on a box yet); loss items 2+3 DONE, 2 user decisions open; 2x box closed, all saved to box_2x_final_2026_10_17; new box setup steps in §45.40"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-17T15:43:44.031Z
---

**USER ASKED (10-17 ~21:15 IST): on Monday, FIRST remind them exactly where we stopped, then set everything up again on the
new box and continue from there.** Full write-up: work doc §45.40 (+ §45.38 loss item 3, §45.39 attention ncu).

**WHERE WE STOPPED:**
1. Attention forward: candidate **L1b** (profiling_reports/session_2026_10_17/candidates/fwd_L1b.cu, md5 069772b7) is written
   and its SASS checked on the laptop (QK issue 72 -> 42 instructions, bit-identical values) but **NOT built on a box and NOT
   raced**. Next action = race L1b vs K5q (see [[project_dense_fwd_gap_1017]]).
2. Loss kernel: items 2 and 3 DONE + verified ([[project_loss_bench_vs_others_plan]]). Waiting for the user: (a)
   BLU_LOSS_GRAD_INPLACE default on or opt-in; (b) commit the 8 files listed in §45.38.

**NUMBERS TO QUOTE:** attention backward beats all (10-16 same-box scoreboard: dense 0.963x FA3, packed 0.984x; fwd+bwd
1.009x / 1.012x). Forward: K5q vs F3c about -6.5% dense / -11.4% packed; ncu K5q vs FA3: packed 0.90x cycles (ours ahead),
dense 1.08x. Loss item 2: backward 1.6x faster, beats Liger everywhere; item 3: trainer peak -5,670 MiB, loss identical.

**MONDAY SETUP (new box, steps in §45.40):** copy the laptop master ~/Downloads/BluTrain_sft (all source; the closed box's tree
was identical to it), make -j, build tva_a, copy stage0 train/labels.bin; scripts from
profiling_reports/box_2x_final_2026_10_17/scripts/ (its cands/fwd_K5q.cu f35982af = the exact K5q source) + fwd_L1b.cu;
`build_cands.sh K5q L1b F3c` (F3c last); race: `AB_ROUNDS=8 AB_ENV="VARLEN_BENCH_DIR=fwd" AB_DENSE_ITERS=4000
AB_PACKED_ITERS=8000 ab_multi.sh <gpu> l1b vj2_K5q vj2_L1b`. Then: exp scheduling fix, Q / tile-0 prefetch, same-box
scoreboard vs FA3 / FA4 / cuDNN (rivals env must be rebuilt: FA3/FA4 flash-attention 94e22c9, torch 2.14.1+cu130).

**BOXES:** 2x H200 root@86.38.238.36 (tall-door-completes-fin-02) used 10-17 15:00-21:15, both GPUs ours in the evening (0 =
loss, 1 = attention), CLOSED by the user after the copy (622 files, md5 verified; libs 1.9 GB + lossenv 5.6 GB not copied,
manifests kept). The 2-node box operator@103.42.51.252 -p 2224 (GPU 7) was used in the morning. Laptop master
GQA_fwd_sm90.cu is still F3c; K5q / L1b only in candidates/ (user decides adoption).

**RISK:** the laptop tree holds ~5 weeks of UNCOMMITTED work (Tensor-Implementations 27 modified + 5 new, bluscript_sft.cpp and
sft/ untracked): remind the user to commit + push.

**Why:** the user explicitly asked to be reminded of the exact stop point on Monday and to resume from it.
**How to apply:** open Monday's session with this reminder (stop point + 2 open decisions + setup list), then follow §45.40's
order. GPU runs and box actions still need the user's OK each time ([[feedback_ask_before_every_gpu_run]]).
