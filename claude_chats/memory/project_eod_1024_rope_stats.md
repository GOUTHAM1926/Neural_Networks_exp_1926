---
name: project-eod-1024-rope-stats
description: "EOD 2026-10-24 MUST include the RoPE benefit explanation exactly as saved in work doc §45.91 (3 levels: standalone, in-trainer kernel, whole step) + 'bit-exact, no precision change'"
metadata:
  type: project
---

The user asked (2026-10-24 evening) to save the RoPE stats explanation exactly and to give it in **today's EOD** as the
explanation of the results. Source text: work doc `sft_work_documentation.md` §45.91.

Levels to include: (1) standalone vs Kruthik (SFT B4 2.32x, B1 3.97x, pretrain B8 2.03x) and vs rivals (16/16 fwd/bwd/total,
SFT B4 176.0 vs 243.2 µs compile); (2) in-trainer RoPE time per step SFT 38.0 -> 16.4 ms, pretraining 355.3 -> 176.0 ms;
(3) whole step SFT -12.9 ms (0.68%), pretraining -104 ms (0.49%), smaller than the kernel saving because of the 700 W power
cap (backward GEMMs 1-3% slower, clock median 1560 -> 1545). Precision: bit-exact (44/44, H200 + RTX 3060, step-0 loss
identical in both trainers).

**Also include the precision explanation exactly** (user, 2026-10-24: "this precision thing too exact, store for EOD"):
the §45.91 paragraph "Precision: no change, bit-exact": all 8 outputs bit-identical to Kruthik's original (fwd q/k out,
q/k rstd, dq, dk, dq_gamma, dk_gamma); 44/44 cases for every build (cp.async D3/D4/D6 + noon register version), 2 separate
runs, H200 and RTX 3060; same per-token math pinned to the original's compiled form, same lane mapping, same dgamma order;
only the loading and overlap changed, never the arithmetic. Trainer: step-0 loss identical (SFT 1.293106 / val 1.175723;
pretraining 11.290755 / 11.294598); later steps differ only as much as old-vs-old (the trainer's own run-to-run
non-determinism from other kernels, known since 10-22). "The speed changed, the numbers did not."

**How to apply:** when writing the 10-24 EOD, follow [[feedback_eod_report_format]] and put this in plain sentences with the
3 small tables; every % names its baseline.
