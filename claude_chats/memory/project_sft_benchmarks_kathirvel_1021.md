---
name: project_sft_benchmarks_kathirvel_1021
description: "DONE 10-21 18:36: SFT kernel stats vs peers for Kathirvel — report .docx built (my voice); loss/attn/RoPE ahead except lm_head+loss (CCE -9%) and dense fwd tie; packing = same learning in 0.30x time"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-21T13:38:32.304Z
---

Kathirvel's email 10-21: "need the kernels stats for SFT kernels against the peers". Ran on the **1x H200 root@86.38.238.55
only** (user-approved for this queue; GPU left free at 18:36). Full record: work doc §45.64.

- Deliverable (not yet sent by the user): `profiling_reports/sft_benchmarks_2026_10_21/sft_kernel_benchmarks.md` -> `.docx`
  (build: `PYTHONPATH=<scratchpad>/pylib python3 build_report_docx.py sft_kernel_benchmarks.md`; laptop has no LibreOffice,
  python-docx + matplotlib installed with `python3 -m pip install --target <dir>`). Figure `box_runs/sft_padding_vs_packing.png`.
- Results (ours = BluTrain; time ratios): loss F+B fastest in 6/6 cases (Liger 1.29-1.39x ours; it beat us on 10-17);
  **lm_head+loss: CCE 9% / Liger FLCE 5% faster on the median SFT batch** (we compute logits for 77% masked tokens) -> the
  "logits only at trained positions" future item; attention dense fwd tie with FA3 (1.006), all else ahead (dense f+b 0.982 /
  0.970 / 0.873 vs FA3/FA4/cuDNN; packed 0.935 / 0.923 / 0.881); RoPE 0.74/0.66 of torch.compile fwd/bwd; positions cost
  +0.7%/+2.5%; loss old->new 0.49-0.91, bit-identical; in-place == out-of-place bit-identical, within 1.7%.
- SFT runs (5,000 chats, 2 epochs): padding 156 steps 17.0 min vs packing 151 steps 5.1 min, val drop 0.1625 vs 0.1629;
  padded run repeats 10-07 (val 1.2839). Val gap 0.003 = different val subsets (loader drops last partial micro-batch).
- Method lessons: a one-process multi-library sweep ran the rivals at 1,980 MHz vs ours 1,635 MHz (power cap) — quote the
  one-library-per-process nsys scoreboard; `loss_compare.py` counts ours' peak with the out-of-place gradient.

- Test 1 curves differ per step because the runs trained on DIFFERENT chats per step (share 0.75 of 64; step-0 train
  1.4500 vs 1.4034 at identical weights). Test 2 (DONE 19:27): same 64 chats per step (`build_samesteps.py`, 28 rows,
  `run_samesteps.sh`) -> step 0 identical, train within 3e-4, val within 1e-4 at every step, epoch means equal: packing
  REPRODUCES padding. Report section 5 has both figures + graph-by-graph explanations. Work doc §45.64 addenda.

- Box backed up 10-21 ~19:45 (may be closed): `profiling_reports/box_1x_h200_backup_2026_10_21/` (checksum-verified,
  README_BACKUP.md lists what was skipped as rebuildable).

**Why:** deliverable for Kathirvel; the user wanted fresh, robust numbers on one box.
**How to apply:** if asked "what next": lm_head on trained tokens only (needs BluBLAS M/N%64 guard), dense-fwd diagonal-first.
Related: [[project_state_2026_10_21_lottn]], [[project_future_logits_trained_only]], [[project_dense_fwd_gap_1017]].
