---
name: feedback-eod-only-what-user-lists
description: "EOD body must cover only the work the user's own draft describes; Kruthik's fused embedding kernel is the mandated one, so never write about the atomicAdd embedding kernel or \"deciding\" to use Kruthik's in reports"
metadata:
  node_type: memory
  type: feedback
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-08T16:17:40.906Z
---

On 2026-10-08 the user rejected my EOD continuation as "full bluff". Two problems:

1. **It added topics the user's draft didn't list.** I included Q9 (how SFT learns), which the user had not put in their "what i did / so today" lines.
2. **It discussed the bf16 atomicAdd embedding kernel and "decided to use Kruthik's fused kernel".** The user's words: "kruthik's kernel is the one which is used and mandated to use, not that atomic add one".

**Why:** the EOD goes to their boss and must match exactly what the user says they did that day, with no extra topics and no framing that contradicts the team's setup.

**How to apply:**
- When continuing an EOD draft, cover only the topics the draft itself lists. For 2026-10-08 that was kernel / SFT-data / SFT-model benchmarks, checkpoints and overfitting, the dev suite, and the decision to go to v2.
- Keep it factual and short. No extra side findings.
- In reports, treat Kruthik's fused embedding backward as the standard kernel. Don't present the atomicAdd kernel as the current path, and don't present adopting Kruthik's kernel as my decision.
- Related: [[feedback_eod_report_format]].
- **Same day, second rejection:** a sentence that listed method names ("alpagasus, deita, ifd, less, longest answers … on a million examples … tülu 3's 8-gram rule") was called unreadable: "how can the people reading this understand". In an EOD, never list paper or method names the reader won't know. Say what was done and found in plain words: what the method does, what the result was, and what I'll do because of it.
