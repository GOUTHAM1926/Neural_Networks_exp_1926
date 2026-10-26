---
name: reference_hf_token_and_bucket
description: "User's own HF token (account gautam1926) lives in ~/.config/blubridge/hf_token on the laptop (chmod 600, never print it); checkpoint bucket hf://buckets/gautam1926/checkpoints_sft; shared boxes' default hf login is a teammate's"
metadata:
  node_type: memory
  type: reference
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-23T07:11:42.392Z
---

- **Token file:** laptop `~/.config/blubridge/hf_token` (chmod 600, next to the org PAT, see [[feedback_commit_identity_company]]).
  Saved 2026-10-23 at the user's request. Account **gautam1926**, token name "sft_checkpoints work", fine-grained (Full
  Access preset). Never print it, never write it into a doc, a repo or a box file. The user pasted it in chat once
  (10-23), so it is in that transcript; rotate = make a new token and overwrite the file.
- **Bucket:** `hf://buckets/gautam1926/checkpoints_sft` (made by the user 10-23, HF "Buckets", xet). Upload with
  `hf sync <local_dir> hf://buckets/gautam1926/checkpoints_sft/<name>` (huggingface_hub 2.2.0 in box `~/gautam_k/prepenv`).
- **How to use it on a box without pasting it** (run from a laptop terminal; the token goes over ssh stdin, not the
  command line or the box's saved login):
  `ssh root@<box> 'export HF_TOKEN="$(cat)"; /root/gautam_k/prepenv/bin/hf auth whoami' < ~/.config/blubridge/hf_token`
- **Shared root boxes:** everyone logs in as root, so `/root/.cache/huggingface/token` is ONE login for all. On the 10-22
  disk it is Madhu's (`MADHU_h200`, saved 10-22 21:51 IST). Never `hf auth login` there (it would replace theirs);
  always pass HF_TOKEN for our command only.
