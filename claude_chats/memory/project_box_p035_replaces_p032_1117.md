---
name: project-box-p035-replaces-p032-1117
description: 2026-11-17 the maintainers replaced P032 (ssh port 2224) with P035 at port 2225, data backed up there; laptop needs the old 2225 host key removed; ~/.ssh/config has duplicate "Host 103.42.51.252" blocks
metadata:
  type: project
---

2026-11-17 (maintainer Dharun M S, team chat, pinned): "The p032 node is replaced with p035 at port 2225, all the data
which was on the node is fully backed up to the p035". Login: `ssh -p 2225 operator@103.42.51.252`.
- All our scripts so far use `ssh -p 2224` (P032): switch them to 2225. Our folder was /localstorage/gautam_k
  (76 GB: sft135 61 GB, attn_1113 7.7 GB with the backward experiment trees bt_bcomb / bt_bshare / bt_push, benv 6 GB);
  check it arrived on P035 before using it. P015 stays at port 2226.
- Laptop check 2026-11-17 (ssh-keyscan, no login): port 2225 answers with a NEW host key (saved one is old), 2224
  still answers with P032's old key, 2226 (P015) unchanged. Fix = `ssh-keygen -f ~/.ssh/known_hosts -R "[103.42.51.252]:2225"`
  (the user's first try used /home/blubridge-016/...; the real home is /home/blu-bridge016 -> "Cannot stat").
- ~/.ssh/config holds FIVE blocks all named `Host 103.42.51.252` (ports 2225, 2226, 2226, 2226, 2224): ssh uses only
  the first block's Port when no -p is given, and VS Code Remote-SSH treats them as ONE host (it reused the open P015
  connection when the user picked the 2225 entry). Suggested: one alias per node (p035 / p015 / p032).
Related: [[project_box_p015_1105]], [[project_p032_cleanup_1106]].
- VERIFIED 2026-11-17 10:38 IST: user removed the old 2225 key and accepted the new one (hostname innmi1srh2-p035).
  /localstorage/gautam_k on P035 = 77 GB, complete: libtensor bt_push a9c94831, bt_bcomb 788cc7ed, bt_bshare 58dbdad8,
  tva_push 5819f7a4, trainer 03fc1aef, sftdata (ckpt_A 514 MB, data_4k train/val meta.json), pkg14/status ends at the
  16/11 21:00:52 line. P035 /localstorage 98% full (530 GB free). Use `ssh -p 2225` from now on.
