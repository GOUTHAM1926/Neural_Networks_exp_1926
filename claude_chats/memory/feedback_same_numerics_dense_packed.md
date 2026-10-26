---
name: feedback_same_numerics_dense_packed
description: Attention dense and packed (design A) must use IDENTICAL softmax arithmetic: the quick suite's G10/G11 checks require packed == alone bit for bit
metadata:
  type: feedback
---

2026-10-17: K8 (dense split 8 + conditional rescale, packed split 16 without it) failed 18 checks, K9/K9h (+ exp polynomial) 19/22,
all of them G10/G11 "every segment bitwise = alone" (+ G6c filler independence): a chat inside a packed row must produce EXACTLY the
result of the same chat run alone through the dense kernel. Differences were rounding-sized (dQ 0.25-0.5 of atol), but the
bit-exact guarantee is how packing is proven correct (packed SFT = unpacked SFT bit for bit).

**Why:** per-mode numerics (different split points with conditional rescale, polynomial exps on some register positions) make the
dense reference and the packed kernel round differently.
**How to apply:** any forward/backward variant must use the same softmax math in both MODE instances; position-dependent tricks
(e.g. exp polynomial chosen by register index) must give the same result for a chat wherever it sits. Run the quick suite on the
candidate itself, never infer correctness from 'identical SASS per mode' (K8's per-mode SASS matched two passing kernels and still
failed the cross-mode checks). Related: [[project_state_2026_10_17]].
