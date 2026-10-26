---
name: feedback-per-case-not-mean
description: Never report GEMM performance as a mean across shapes - the trainer runs individual shapes, so quote per-case numbers and the count below 100%.
metadata:
  type: feedback
---

**Never headline a mean.** On 2026-09-08 I reported "101.63% of cuBLAS, no variant below 100%".
Both halves misled:

- **101.63% was over 129 of 135 cases**, not 135. The 6 split-K cases were held out and the
  label did not say so. All 135 together = **100.89%**.
- **"No variant below 100%" grouped 135 cases into 16 variants and averaged inside each group.**
  15 of 16 group-means were >=100%, so it read as "everything is fine". Per *case*, **39 of 135
  are below 100%**. `NT_128x256x64_regsplit` averages 101.89% while **10 of its 38 shapes lose**.
- An **unweighted** mean also treats a shape called once like one called 48 times. Weighting the
  same data by GEMM time gives 101.69% (600M B=2) / 101.47% (4B B=2).

**Why:** Gautam's words — *"trainer wont run on those mean numbers"*. A mean is an average over
shapes; the trainer executes specific shapes at specific call counts. Averaging hides exactly the
losers that need work, which is the opposite of what the number is for.

**How to apply:** report **per-case**, and always state (a) how many cases are below 100%,
(b) the worst case, (c) the count the number covers. If a mean is genuinely wanted, give the
**time-weighted** one and label the denominator. See [[feedback_verify_claims_and_benchmark]]
and [[feedback_measure_against_dispatch_not_pins]].
