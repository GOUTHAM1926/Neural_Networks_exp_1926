---
name: project_box_p015_1105
description: "P015 (replaces P022) = ssh -p 2226 operator@103.42.51.252, 8x H200: OUR GPU = 7 (more only if needed: 6, then 5, then 4); ~/gautam_k (localstorage not writable); yaml-cpp unpacked in ~/gautam_k/deps; 2x box CLOSED 11-05"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-11-05T12:49:21.456Z
---

**2026-11-05 ~17:05 IST the 2x box (86.38.238.51) closed.** User: no need to attach its disk or start an instance to copy
from it -- everything is reproducible (code on GitHub, data + scripts on the laptop). Lost only the raw SFT nsys files and
item-4 test logs of 11-05 (numbers are in work doc 45.165 / 45.166).

**P015 = `ssh -p 2226 operator@103.42.51.252`** (host innmi1srh2-p015; same IP as P032/P022, port 2226). Colleague notice
(Dharun M S): P022 is down for maintenance, do NOT use it even if it connects; P015 replaces it with P022's data.
- **Our GPU = 7** (user 11-05: "use the last GPU, ours needs only 1 GPU"; if 2 needed add 6, then 5, then 4 -- never
  0-3). Still free-check + permission rules ([[feedback-ask-before-every-gpu-run]]).
- Shared `operator` account: colleagues' Claude sessions run here too (Sanjay: /localstorage/Sanjay/135M_test; others
  on GPUs 0/1 from 18:15 on 11-05). Never touch their processes / folders; `pgrep -f` patterns can match theirs.
- **/localstorage is NOT writable** for us -> everything in **~/gautam_k** (root disk, 226 GB free on 11-05).
  `/localstorage/backup-p022` is a disk backup (rootfs etc.), not ours.
- Ubuntu 24.04.5, CUDA 13.0.88 (/usr/local/cuda), nsys 2025.3.2, OpenMPI 4.1.6, NCCL system, 224 CPUs (build -j96),
  python3 without numpy.
- **No yaml-cpp** (trainers' ConfigBridge.h needs it): `apt-get download libyaml-cpp-dev libyaml-cpp0.8` + `dpkg -x`
  into ~/gautam_k/deps/yaml (no root, nothing installed). Builds: CPATH / LIBRARY_PATH; runs: LD_LIBRARY_PATH
  (setup_p015.sh / test scripts in profiling_reports/pretrain_before_after_2026_11_05/items_3_4/p015/).
- Item 3 test setup 11-05: ~/gautam_k/item3_1105 (bt_old = pushed code via git archive, bt_new = + item 3; data_4k,
  ckpt_A, flux shards, mix/, cfg/). Trees build in ~10 min (libs) + ~1 min per trainer.
- ssh trap: a remote `sleep N` longer than the local `timeout` kills the ssh (the nohup job still runs); start
  background jobs with `setsid nohup ... &` and return at once. See [[feedback_box_ops_traps]].
