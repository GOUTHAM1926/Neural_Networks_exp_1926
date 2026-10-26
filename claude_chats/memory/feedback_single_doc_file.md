---
name: feedback-single-doc-file
description: BLAS/cluster-arc docs live in blublas/docs/ as the master; work_docs/ holds a byte-identical mirror; never create new standalone .md plan/notes files
metadata: 
  node_type: memory
  type: feedback
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-09-18T14:10:24.380Z
---

User rule (2026-07-07, given firmly after I created 2x2_cluster_plan.md, 1x2_plan.md, all_tiles_plan.md as separate files): document EVERYTHING in the analyzations file — plans, research resolutions, run analyses, task lists. No new .md files.

**Why:** one single source of truth; scattered files lose learnings and the user has to hunt across files.

**How to apply:** append new numbered sections (`### §HNNN Title (date)`) to the analyzations file. The three old plan files were merged in full as §38–§40. Raw run logs may still be archived as separate files — those are data, not documentation.

## MASTER + MIRROR (confirmed 2026-09-18)

Documentation now lives in TWO places on purpose — edit ONE, copy to the other:

- **MASTER (edit here):** `~/Downloads/blublas/docs/` — 6 files: `analyzations_h200.md`,
  `analyzations_b300.md`, `B300_cublas_reveng_qwen_coverage_2026_07_01.md`,
  `1x2_plan.md`, `2x2_cluster_plan.md`, `all_tiles_plan.md`
- **MIRROR (never edit):** `Neural_Networks_exp_1926/work_docs(mark-down - (.md) files ) /`

Sync after every edit, then push BOTH repos:

    cp ~/Downloads/blublas/docs/*.md \
       ~/Downloads/Neural_Networks_exp_1926/"work_docs(mark-down - (.md) files ) /"

Drift check (silence = in sync):

    for f in ~/Downloads/blublas/docs/*.md; do
      cmp -s "$f" ~/Downloads/Neural_Networks_exp_1926/"work_docs(mark-down - (.md) files ) /$(basename $f)" \
        || echo "DRIFT: $(basename $f)"
    done

**Why two copies:** the user wants a second remote as backup. Commit `2bda064` had removed the
duplicate "so there is a single master and no drift" — the mirror restores the backup without
reintroducing drift, BECAUSE the mirror is never edited directly.

SFT documentation is separate: `work_docs/sft_work_documentation.md` (no blublas copy).

See [[project_active_repo]], [[feedback_push_pull_every_change]].
