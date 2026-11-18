---
name: project-cp-need-350m-1117
description: 2026-11-17 question "at what SFT row length does the 350M model need context parallelism"; calculated from measured 135M/1.6B runs (~121k tokens max on 1 H200); GPU confirmation ladder ready on P035, not yet run (GPUs taken)
metadata:
  type: project
---

User 2026-11-17: for the 350M model (Ram's torch_compile_fa4_350M.py = Navin's bluscript_zero_ckpt_bluccl_multi_135m_
eqab_fusedemb.cpp shape: d 1024, 28 layers, 8/4 heads, hd 128, ffn 2560, vocab 49280, theta 10000, tied lm_head; the
benchmark ran it with SWA window 1024 on 3 of 4 layers), at which SFT row length is CP needed? "test in backend or
calculate robustly, with proofs". Also: compare the training scripts (table given: all four share model + batch + LR;
differ in compile mode / precision / kernels). Nova JIT script: /localstorage/adwaid/BluTrain/bluscript_jit_zero_350M.cpp
on P035 (Aparna's copy differs in 14 lines, adwaid/swa copy in 16).
**Calculation (from MEASURED peaks of our SFT trainer, 07/11 cp_2026_11_07_2x/box_results/results_2x.txt):** 135M
0.7545 MiB/token (64k 61,503 / 131k 110,951 MiB, 1 GPU); 1.6B shape 2.11 MiB/token (CP-2 pair 16k 42,351 / 65k 96,635,
CP overhead x1.046 from the 135M CP pair); bf16-activation model (per layer x, n1, q, k, v, O, h, n2, gate_up, act +
bf16 logits) x factor 1.17 (135M) / 1.19 (1.6B) -> 350M 1.048 MiB/token; weights+optimizer 14.5 B/param + 9.6 GB fixed
-> 14.6 GB. Max row on one GPU (142,000 MiB): ~121,600 tokens: 114,688 fits (~134.8 GB), 122,880 OOM (~143.3 GB). CP 2:
~234k (85 GB per GPU at 131,072). 135M: ~173k on one GPU. SWA does not change stored activations (Q,K,V,O,LSE per
token), so the memory limit is the same with or without it. Our SFT trainer + attention kernels have NO SWA support.
**Confirmation ladder READY on P035 (not run: Kruthik took both nodes 11:31):** /localstorage/gautam_k/cp350/cp350.sh
(laptop copy profiling_reports/cp_need_350m_2026_11_17/); data prepared with the REAL prep (smol-smoltalk, ffd) at T
16,384 .. 131,072 (9 lengths, train 8,000 chats + val 300). Run: `rm STOP; setsid nohup bash cp350.sh mem <gpu>`
(stops at the first OOM). Prep gotcha: the 300-chat val split with --workers 16 hung 3 times -> 1 worker now.
Gotcha 11:38: my ssh one-liner killed its own shell with an unanchored `pgrep -f prepare_sft_data.py` (see
feedback_box_ops_traps): use anchored patterns or a script file.
