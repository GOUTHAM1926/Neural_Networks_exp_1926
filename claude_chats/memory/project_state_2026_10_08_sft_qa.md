---
name: project-state-2026-10-08-sft-qa
description: "STATE after 2026-10-08 (documented 10-09): Q9 measured + doc, Q7/Q8 pasted into the shared doc (3 sections), Q10 packing measured; where every record lives; what's next"
metadata:
  type: project
---

**Done by the end of 2026-10-08.** All of it is written up in work doc §38 (master `~/Downloads/sft_research/sft_work_documentation.md` + `work_docs` mirror).

- **Q9** (how SFT learns only from answers):
  - measured on a small model like ours;
  - doc `How_learning_happens_in_sft.md` (master + mirror);
  - records in `profiling_reports/q9_gradient_flow_2026_10_08/`.
- **Q7/Q8:** three sections pasted into the shared Google Doc tab `…/1nJ4utik9Jm7RT9YuaWD8UXcPpJRAj2WUrXnEdjJ_jlg/edit?tab=t.jzwfpg8vqqer`.
  - The sections: SFT-data evaluation plan, kernel benchmarks, SFT-model evaluation plan.
  - Final state: 86 links, 11 tables.
  - Readable without login via `/export?tab=…&format=txt|html|docx`.
  - Snapshot in `profiling_reports/sft_docs_snapshot_2026_10_08/`.
- **Q10:** v1/v1.5/v2 measured (×1 / ×2.84 / ×3.60); records in `profiling_reports/q10_packing_sim_2026_10_08/`. See [[project-future-v2-packing]].
- **Pending fix in the doc tab** (tell the user): kernel section, Step 5, "Two small things that change with v2" says "here 1–5" chats per row; the measured range is **1–12**.
- **Next:**
  1. the loss-kernel benchmark vs PyTorch / Liger / Apple CCE;
  2. v1.5, then v2 (Kruthik's fused embedding kernel with the real-N fix);
  3. Q11 after v2;
  4. the remaining Q&A (Q4, Q5, Q6, Q11, Q12, Q13).
- **Git:** nothing committed. New files are under `profiling_reports/` (gitignored, needs `git add -f`), `work_docs/How_learning_happens_in_sft.md`, and the modified `work_docs/sft_work_documentation.md`. The user does all commits.
- **Lesson:** the scratchpad is wiped between days. Save scripts and results under `profiling_reports/` the same day.
