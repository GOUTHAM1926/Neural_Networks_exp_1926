---
name: feedback-pull-run-artifacts-immediately
description: After EVERY GPU run, copy the logs, profiles and configs off the box into the local Neural_Networks_exp_1926 folder without being asked — boxes get released and take everything with them.
metadata:
  type: feedback
---

The moment a run finishes, pull its whole output folder down to
`/home/blu-bridge016/Downloads/Neural_Networks_exp_1926/<run_folder>/`. Do not wait to be asked.
Do not batch it up for "later". The user has asked for this repeatedly and should not have to.

**Why:** rented Verda boxes disappear without warning. On 2026-09-01 the box was released with
~5 GB of nsys profiles on it — the production 4B capture, the whole old-tree cuBLAS/Kota/pinned
A/B, and 15 standalone shape captures. All of it was gone the next morning. The conclusions
survived only because they had been written into `analyzations_h200.md`; the raw profiles could
not be re-analysed and the 4B ones cannot even be regenerated without a 2-GPU box. See
[[feedback_edit_local_push_pull]], which is the same lesson for source.

**How to apply:**
- Structure every run to write into ONE folder: `train.log`, the `.nsys-rep`, `config_used.yaml`,
  `run_meta.txt` (host, date, arm, library md5, GPU/driver/clock), and any sampler csv.
- Pull with `rsync -az --partial --exclude='*.sqlite'` — `.sqlite` is regenerable from the
  `.nsys-rep` via `nsys stats`, and skipping it roughly halves the transfer.
- Transfers are slow (~350 MB took several minutes); run them detached with `--partial` so a
  dropped connection resumes instead of restarting.
- Say in the status report that the artefacts are down and where they are.
