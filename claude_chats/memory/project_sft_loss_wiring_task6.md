---
name: project-sft-loss-wiring-task6
description: "Verified in code 2026-10-06: TI's global-denominator SFT loss is right (one denom tensor for fwd+bwd, 1/N applied once); the trainer is NOT wired (2-arg call); task-6 checklist + 4 hazards; written up in loss_calculation_in_pretraining_and_sft.md"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-06T14:04:00.217Z
---

Doc: `~/Downloads/sft_research/loss_calculation_in_pretraining_and_sft.md` (master) + the work_docs mirror, md5 `b41fe1a8…`, 777 lines. The user's requested name was "los_calculation_in pretraining_and_sft.md"; I fixed the typo and the space.

**Library (TI `d96b991`): correct.** The caller's `&denom` tensor is:
- used by the forward's last reduction (`LossKernels.cu:188`, launched at `:953-967`);
- stored in the backward node (`LossOps.cpp:493-495`);
- used by the backward kernel: `f_scale = *grad_output / *denom` (`LossKernels.cu:704`, `:813`).

The loss VALUE is never a backward input, so 1/N is applied once. PyTorch 2.14 CPU (in `~/Downloads/tf_issue_venv`) gives the same numbers: −0.05 / −0.45 for p = 0.9 / 0.1 at N = 2.

**Trainer (`909ab75`): NOT wired yet.** Committed line numbers:
- `:2164` loss call with 2 args;
- `:1796-1797` seed 1/GA;
- `:2218-2221` logged loss ÷ GA ÷ W;
- `:2102-2106` val = Method 1.

GA = 1,048,576 / (8·4096·W) = 32 on 1 GPU, 16 on 2 GPUs.

**Task-6 loss changes:**
- `denom` = `valid_tokens_ahead(GA)` + all-reduce SUM, put in a 1-element fp32 GPU tensor;
- loss call: `(logits, target, 65535, &denom)`;
- seed W. All 3 optimizers average over GPUs: v1 op=avg; v2 and Muon RS/AG use op=sum then ×1/W;
- logged loss: sum, with no ÷GA and no ÷W;
- val: global form as well.

**Hazards:**
- The SftDataLoader with the default −1 reads `row_logits[65535]` out of bounds, because there is no bounds check (`LossKernels.cu:318-319`). Ship the loader and the 65535 loss call in the SAME change.
- The B300 sm103 forward ignores `ignore_index` (TODO `:938-941`).
- The 2nd reduction stage drops partial sums past 256 when B·T > 65,536. This hits the loss value only. Found by reading, not run.

**Stage-0 measured:**
- micro-batch valid tokens range 2,920–12,729;
- within one step, the biggest/smallest micro-batch ratio is 2.54× (median) and 4.13× (max);
- step 0 has 243,536 trained tokens, out of 1M positions.

**F1 (for questions about it):** it checks the GRAD NORM in `step()`, which gives exit 4. It does not check the loss. The old loss check (`:2270`) gives exit 2.
- Before F1: the NaN went into the weights, and the run stopped at the next step's loss check, one step late. If it was a checkpoint step, the checkpoint was poisoned.
- Muon 2-D froze silently.

**Correction I made in the doc (§6.8):** "1 instead of 1/N = N× LR" is only true for SGD. AdamW and Muon mostly cancel a constant scale, but clip (1.0), eps and the logged norm still see it.

Related: [[project-sft-workflow-1006]], [[project-sft-loss-done]], [[project-future-dft-focal-loss]], [[project-f1-muon-v2-pushed]].
