---
name: feedback-box-backup-only-needed
description: Before a box is closed, copy ONLY code + important results/log reports that are not already on the laptop; never bulk-copy reproducible data, builds, envs, temp files
metadata:
  type: feedback
---

When the user says "copy all from the box, I will close the instance": do NOT rsync everything (10-31 I started a 17 GB
copy; the user stopped it: "we don't need unwanted or reproducible things ... copy only wanted things").

**Why:** prepared data, built trees, Python envs, nsys traces, inputs from seeds are reproducible; the laptop already has
the sources (BluTrain_push, lc_box/tests) and most records; the work gets pushed to the remote next.

**How to apply:** dry-run by CONTENT (`rsync -anc`) box -> laptop records to list only what is really missing; copy
those code files (into a separate box_only_code folder, never over laptop files) and result/log files; check big unique
items (e.g. a checkpoint) by md5 against the laptop copy (sort with LC_ALL=C on both sides before comparing); save env
package lists (pip freeze) instead of envs. Records: profiling_reports/<campaign>/. See [[feedback-pull-run-artifacts-immediately]].
