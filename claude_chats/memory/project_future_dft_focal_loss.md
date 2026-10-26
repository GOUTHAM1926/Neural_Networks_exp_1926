---
name: project-future-dft-focal-loss
description: FUTURE SFT experiments (user asked 2026-10-06) — try DFT and focal loss against plain cross-entropy after the v1 SFT runs work
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-06T13:13:19.230Z
---

FUTURE TASKS, added at the user's request on 2026-10-06. Do them after v1 SFT works (stage 0/1 runs). Both re-weight each token's loss by the model's probability p of the correct token:

| loss | per-token formula | effect | paper |
|---|---|---|---|
| plain cross-entropy (our v1) | −log p | every token has the same multiplier (1/N, the global denominator); hard tokens push harder only because their error (p − y) is bigger | — |
| **focal loss** | −(1−p)^γ · log p (γ ≈ 2) | down-weights tokens the model already gets right, so hard tokens dominate | Lin et al. 2017, "Focal Loss for Dense Object Detection", arXiv 1708.02002 (object detection, class imbalance: "down-weights the loss assigned to well-classified examples") |
| **DFT** (Dynamic Fine-Tuning) | −sg(p) · log p (no gradient through the p factor) | the OPPOSITE of focal: tokens the model finds unlikely count LESS | Wu et al. 2025, "On the Generalization of SFT: A Reinforcement Learning Perspective with Reward Rectification", arXiv 2508.05629 ("rescaling the objective function with the probability of this token", "single-line change", better generalization on math/code); code github.com/yongliang-wu/DFT |

**Worked example**, the loss of a hard token (p=0.1) versus an easy one (p=0.9):
- plain: 2.303 vs 0.105, about 22×;
- focal (γ=2): 1.865 vs 0.00105, about 1,770×;
- DFT: 0.230 vs 0.0948, about 2.4×.

**How to apply:**
- It is a change to the loss kernel / loss op only. Keep the global-token denominator for normalisation in all three.
- Already listed in SFT report §8.3 (DFT row). Focal loss is a new addition.
- Compare at an equal data size and LR, on held-out loss plus benchmarks.

Related: [[project-sft-workflow-1006]], [[project-sft-loss-done]].
