---
name: project-sft-plan-is-qa-questions
description: "From 2026-10-07 the SFT plan is the user's own 13 questions in the shared doc's SFT Q&A list, answered in depth and visually, not my task lists; evals are the tool for 6 of them"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-07T15:48:07.032Z
---

On 2026-10-07 the user set the SFT plan: answer all 13 questions they wrote in the shared Google Doc "PostTraining - Research and Planning" (the SFT Q&A list, Q1–Q13), in depth and visually, until everything is clear. This replaces my own "next steps" lists.

**Why:** the user wants full understanding of every SFT topic, not only a working pipeline.

**How to apply:**
- Frame every next step as "which question does this answer".
- Evals (generation plus benchmarks: the converter, then base vs SFT) are not a separate task. Q4, Q5, Q6, Q7, Q8 and Q12 can't be answered properly without them; they are the tool for those questions.
- Q1, Q2, Q3 and Q9 are mostly answered already (the doc's sections 1 and 3, `loss_calculation_in_pretraining_and_sft.md`). Q10, Q11 and Q13 need research plus speed/correctness runs, not evals.
- **10-08: the user dropped #5 (evals) and the Kathir PyTorch comparison as separate tasks** ("time waste, the questions will cover it"). Don't re-propose them as side tasks. Evals come up only inside Q7/Q8 when the user gets there.
- State of the pipeline: [[project-state-2026-10-06-eod-sft]].
