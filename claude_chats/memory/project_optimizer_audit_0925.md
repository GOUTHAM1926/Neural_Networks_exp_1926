---
name: project-optimizer-audit-0925
description: 2026-09-25 optimizer/trainer audit (3 auditors + 5 skeptics) — F1 NaN/Inf grad written into weights is the one real production bug; #2 family = 4 sites; GRAD_ACCUM is 4 in prod, 64 in our 1-GPU test
metadata:
  type: project
---

Full table: `loss_kernel_dtype_bugs.md` §19.7. Raw verdicts: `profiling_reports/fix2_2026_09_25/audit_verdicts_F1-F9.json`.

**The real production bug is F1.** A NaN or Inf gradient is written into the fp32 master weights.
- The non-finite skip is gated on a LossScaler (`ShardedOptimizer.cpp:1160`, `:1201`), and none is installed.
- A NaN norm gives clip coef 1.0; an Inf norm gives coef 0, so Inf*0 = NaN.
- The trainer checks only the loss, and before `step()`.
- If it happens on a checkpoint step, the poisoned state is saved and auto-resume reloads it.
- Fix is pending the user's choice: skip the step, or stop before the update.

**Waste:**
- F2: `accumulate_grad` makes an fp32 temp + add every micro-step (about 0.7% at W=1, estimated). It needs a new fp32 += bf16 kernel.
- The #2 family is 4 sites: C `:977` (patched on the box, proven), B′ `:539`, B `:945`, D `:1022`.
- In production only `:539`, `:977` and `:1022` run, about 1.8 ms/step estimated.

**Left as is, per the user ("if harmless, leave it"):**
- the Adam `is_adamw` placeholder (the trainer hard-codes ADAMW at `:1108`; the colleague was right);
- per-group clipping;
- bf16 grad round trip in v1;
- DCP layout-hash check;
- empty-string env vars;
- the library's nn::Adam races (unreachable).

**GRAD_ACCUM = global_batch / (B*T*W)** (trainer `:926`): 64 in our 1-GPU test, 4 in production.

**Why:** the user asked for a deep 360° audit before fixing, and wants harmless things left alone.

**How to apply:**
- Don't re-audit these.
- Fix F1 once the user picks the option.
- Then patch B/B′/D and run `test_fix_family.cu`, then the A/B.
- The auditors read the colleague's ORIGINAL TI `8808f89`, which has no loss fix. Never tell the user "not applied" from that copy: that confused them on 2026-09-25.
