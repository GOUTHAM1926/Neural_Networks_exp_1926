---
name: feedback_no_assumptions_decide_by_tests
description: "Never decide or downplay anything on an assumption (e.g. 'no users so injection doesn't matter', 'safety = small share'); the model may be used or open-sourced later; every choice is made by our own tests"
metadata:
  node_type: memory
  type: feedback
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-29T14:19:34.423Z
---

Every choice in the SFT work is decided by our own tests, never by an assumption. Consider every case, including future
use by customers and open-sourcing.

**Why:** on 2026-10-29 night the user rejected two things I had written:
- "our 135M is a research model with no customers, so injection doesn't decide the format";
- "safety: in, small share" (factor 1 table).

Their words: "customers may use later right, now it's in developing phase but we should consider everything, later it
will go outside or open-sourced, so don't tell like that", and "considering all the things only we will do tests, not
assuming like these". They also said "don't assume like that ... we will decide clearly on that when we go to the next
component".

**How to apply:**
- Never write "X doesn't matter for us because <current situation>". Name the criterion, and add a test for it.
  Example: injection safety became a decision criterion (M7 plus the new M7b hidden-order test) in plan §4.28.
- Never state an amount or share (data mix, counts) as decided unless it was tested. Write instead that it is measured
  in factor N.
- When two criteria conflict (quality vs safety), put both sets of numbers to the user, who decides.

Related: [[project_sft_grid_plan_1028]], [[feedback_follow_recorded_decisions]], [[feedback_completeness_goal]].
