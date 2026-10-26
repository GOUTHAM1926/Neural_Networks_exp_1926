---
name: no-redundant-boundary-fixes
description: "When fixing a bug at one boundary closes all upstream paths, do NOT add a \"defensive\" mirror fix at a downstream boundary just because it feels safer or because pytorch has a similar pattern. Verify the downstream code is actually reachable with the bad input."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
---

When fixing a layout/contract bug, before adding a second materialize/check at a downstream boundary for "defense-in-depth", trace every caller of the downstream code and verify that at least one of them can still hand it a bad input after the upstream fix.

**Why:** When debugging the strided `add_inplace_kernel_broadcast` perf cliff (May 2026), I correctly identified the fix at `Engine.cpp:309` (materialize the lead grad before the multi-grad sum loop). I then proposed ALSO materializing in `AutogradMeta::accumulate_grad`'s ELSE branch as a pytorch-style two-boundary contract. User pushed back: "if doing in engine.cpp is fine then why again in autograd meta?" — and they were right. `accumulate_grad` is only reachable via `GradAccumulator::apply`, which gets its grad from the engine reducer's `sum`, which the Engine.cpp:309 fix already materializes unconditionally. The AutogradMeta ELSE-branch addition was unreachable code. The pytorch "two-boundary" appearance is misleading: pytorch's `accumulateGrad` uses out-of-place `+` (always contig result) while `InputBuffer::add` uses in-place — the boundaries do different things and only one carries a materialize, not both.

**How to apply:** Before adding any defensive check that mirrors a real fix, trace upstream callers. If 100% of upstream paths are closed by the primary fix, the mirror is dead code — leave it out. Cite the chain in the doc instead of writing redundant code. Add the safety check ONLY if you can name a concrete caller that bypasses the primary fix (today, not "future hypothetical custom hook"). See [[resolving-strided-broadcast-inplace-add]] for the worked example.

Related codebase principle in `CLAUDE.md`: "Don't add features, refactor, or introduce abstractions beyond what the task requires." This applies to defensive code too.

**Second case (2026-09-29, the BLU_INIT_FROM weights-only loader):** I added a pre-flight check before the DCP load (weight count, shape and dtype per index, plus a friendly "run with BLU_ALL_BF16=1" hint). The user: "no need of this, nothing needed like this, just comment it ... remind in your memory". The failure was already loud: DCP stops on a missing weight or a different shape (`[Architecture Mismatch]`, LoadPlanning.cpp:33-41) and `Tensor::copy_` refuses a dtype mismatch.

**Rule:** when a failure is already loud, do not add a friendlier pre-check; write a comment naming the requirement and where it fails. Make verification-only code opt-in (e.g. `BLU_INIT_VERIFY=1`), not on by default.
