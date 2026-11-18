---
name: feedback_full_audit_before_plan_answers
description: "When a plan question comes in, audit the WHOLE choice space against primary files first; never drip new findings answer by answer"
metadata:
  type: feedback
---

On 2026-10-30 the user was frustrated ("you keep telling new things again"). In SFT component 9, each follow-up question
turned up something new: Cohere's role ids, Phi-3's per-message end, GLM-4, the default/merged system message, and the
lm-eval switch trap. The user wants one deep analysis, with the plan refactored once.

**Why:** piecemeal discoveries make the user doubt the plan and force them to re-approve it many times. They asked
explicitly: "deeply analyse everything and modify the plan".

**How to apply:**
- Before answering a "does the plan cover X?" question, list the full space of choices (every axis, not only the one
  asked about). Check each axis against the teams' own files: tokenizer.json special flags, chat_template,
  generation_config.
- Walk the implementation path id by id: build → train → export → eval backend.
- Put everything into ONE refactored plan section. Say plainly which earlier statements change.
- Related: [[feedback_no_assumptions_decide_by_tests]], [[feedback_completeness_goal]], [[project_sft_grid_plan_1028]].
