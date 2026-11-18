---
name: project-all-sft-optimizations-doc-1118
description: "TODO after the 18/11 push + EOD, push 19/11 morning: write all_sft_optimizations.md (SFT package in work_docs) = every SFT optimization since the SFT start (~mid-Sept 2026), each with context, pre-state, post-state, numbers, shapes, diagrams, for colleagues new to SFT"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-11-18T16:18:10.947Z
---

User, 2026-11-18 ~21:55 IST: "I want a separate md file in the SFT package in the work docs folder, all_sft_optimizations.md,
documenting all the optimizations we did right from the SFT starting phase ... go through all the previous chats and our
work documentations right from the SFT starting day ... explaining each optimization in depth, giving context too,
pre-state, post-state, each thing in detail as if you explain to a kid and my colleagues who don't know anything about
SFT ... right from matmul shapes to all visualization things, mind maps and everything, in simple understandable English
with many examples ... do it after our remote pushing and EOD work, we will push this tomorrow morning."

**Why:** today the SFT optimizations are spread over the work log (sft_work_documentation.md section 45.x and earlier
26-44), sft/README.md (3.x, 5.1, 5.2, 6.x), sft_kernel_benchmarks.md, the attention write-ups and the chats; colleagues
need ONE place with context + before/after for each.

**How to apply:**
- Location: `work_docs(mark-down - (.md) files ) /SFT_package_2026_10_24/all_sft_optimizations.md` (the user said "a
  separate md file in sft package in work docs folder"). A separate file is the user's explicit wish (overrides the
  single-doc rule for this one).
- Sources, from the SFT start: SFT assignment 09-17 ([[project_sft_research_assignment]]), the work log (all SFT
  sections), plan_grid_search_sft.md, sft/README.md + sft/docs/*, sft_kernel_benchmarks.md, the attention docs
  (attention_kernel_optimizations_h200_2026_10_10_to_2026_10_21.md, attention_packed_rows_..._2026_11_13.md,
  attention_forward_and_backward_gqa_tests_and_benchmarks_2026_11_18.md), the claude_chats summaries and conversation
  files, memory notes.
- Per optimization: context (why), pre-state (code + numbers), post-state, how measured, numbers with source, shapes
  (e.g. 1.6B: d 2304, 18/6 heads, T 4096, B 8; 135M: d 512, 4/2 heads), diagrams (ASCII/mermaid mind map), examples.
  Written for colleagues who know nothing about SFT: simple English, no analogies ([[feedback_no_analogies]]), setup
  before numbers, every % with its baseline ([[feedback_explain_setup_before_numbers]]). In the user's voice ("I").
- Do it AFTER: the 18/11 TI + BluTrain push is verified, the Neural_Networks repo is pushed, EOD. Push it 19/11 morning.
