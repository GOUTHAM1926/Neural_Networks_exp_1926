---
name: project_remind_1022_rope_restart
description: "REMIND FIRST on 10-22 morning: re-explain the RoPE position restart (+ per-chat attention, loss changes) — user did not understand the 10-21 evening explanation; only this first, then other work"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-21T15:59:03.104Z
---

User, 2026-10-21 evening: "i couldn't understand this, so we will start with this tomorrow morning, right after tomorrow
morning give me this and explain this only, this rope thing, we will understand and then we will do other things".
**First thing on 10-22: explain ONLY this, before any other work.** Explain from zero, step by step, real terms, ASCII
pictures, small worked numbers, no analogies ([[feedback_no_analogies]], [[feedback_no_artifacts_for_explanations]]).

What has to come across (all checked on 10-21):
1. RoPE rotates each (x1, x2) pair of q and k by angle = position x frequency. q_i . k_j after rotation depends only on
   (i - j). Worked example: 2-D q = k = (1, 0), frequency 1 rad/position: token 10 vs token 5 -> score cos(10-5) = cos 5;
   same chat starting at 1200: tokens 1210 and 1205 -> cos(1210-1205) = cos 5. Same score.
2. So with PER-CHAT attention (packed rows, plan from seq_lengths, not from positions), restarting positions at 0 per
   chat does not change the math; without it only bf16 rounding differs (rotated fp32 values round differently).
3. Why BluTrain still restarts: packed == chat alone BIT-EXACT (attention G10/G11 tests, test_rope_positions 37/37,
   test 2 step 0 = 1.449952 both), training = inference (chat starts at 0), standard (HF DataCollatorWithFlattening
   position_ids, torchtitan rope_cache[positions]), cost +0.7% fwd / +2.5% bwd of the RoPE kernel (box_runs/A5_rope).
4. Per-chat attention IS required when packing (else chat B attends chat A: training != inference); restart WITHOUT
   per-chat attention is worse (B sees A at nonsense distances).
5. Padded SFT needs neither (pads come after the chat, causal; chat starts at 0) — only the loss changes.
6. Loss changes are needed for ANY SFT: masking prompt + pad (else it trains on prompt text / pad tokens, ~70% of padded
   positions are pad) and ONE denominator per optimizer step (else token weight = 1 / trained tokens of its
   micro-batch). 16-byte loads + in-place gradient = speed / memory only.
Related: [[project_sft_benchmarks_kathirvel_1021]], [[project_v2_build_progress]].
