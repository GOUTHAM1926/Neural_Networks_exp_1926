---
name: project_state_2026_10_21_lottn
description: "10-21 pretraining hand-over branch dist_side_gautam_lottn (BLUCCL afc462d + QPU/Vj2 attention + loss fixes) — laptop copies, box A/B scripts, what is still to run"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-21T07:59:55.475Z
---

**Why:** colleagues start pretraining the "1435M" model; only pretraining-relevant work goes on a NEW branch from
`origin/dist_side_merge_BLUCCL` (BluTrain afc462d, TI 02d0d18, BLAS ea666abc, CP 661b6ef). User commits/pushes.

**Laptop (not committed, not pushed):**
- `~/Downloads/BluTrain_lottn` = branch `dist_side_gautam_lottn` in BluTrain AND TI (upstream tracking REMOVED so a bare
  `git push` cannot hit dist_side_merge_BLUCCL; push with an explicit refspec, TI first). 15 TI files ported from the master
  (11 attention: QPU fwd 7e4197ea + Vj2 default; 4 loss: one gradient allocation, in-place only, 16-byte loads fwd+bwd).
  Compile check 22 TUs OK. Norm/RoPE/test files NOT ported.
- `~/Downloads/BluTrain_lottn_old` = BLUCCL untouched (the OLD arm).
- Master `BluTrain_sft` got the same 3 edits: `if (inplace)` fold in LossBackward.cpp, Vj1d "DEFAULT" comment fixed,
  GQA_fwd_sm90.cu = QPU (F3c kept as scripts/cands/fwd_F3c.cu).
- git-lfs is NOT installed on the laptop and ~/.gitconfig marks filter.lfs required: BLUCCL's one LFS file
  (dist/Orchestrator/BluRun/deploy/obs/dcgm/counters.csv) broke the checkout; the copies carry local config
  `filter.lfs.process=""`, `filter.lfs.required=false` (file stays a pointer; not used by build/training).

**Box 1x H200 root@86.38.238.55 (GPU 0), layout `~/gautam_k/lottn/{old,new,ab,runs}`, data `~/gautam_k/pt_data/mix/<4 domains>`
-> links to 4 edufineweb train shards + 1 val (laptop `~/Downloads/master_gau_latest/data/`).** Build = `make -j22 profiler
tensor dist SM_ARCH=90` (NOT `all`: ensure-blurun/ensure-tmux touch PATH/apt), trainer = `make _ablation_compile
MODE=production_zero ABLATION_OUT=~/gautam_k/lottn/pz_<t>`.
Scripts (laptop `profiling_reports/pretrain_branch_2026_10_21/ab/`): run_loss_ab.sh (~3 min), run_attn_ab.sh quick|full,
ab_trainer.sh [STOP=12] (~30 min, production YAML + validation_freq 4 only, stop at step 12, never BLU_MAX_STEPS).

**Trainer facts learned:** production = `train/production_zero/bluscript_zero_ckpt.cpp` + `bluscript_zero_ckpt_config.yaml`
(2304 / 26 layers / 18:6 heads / hd 128 / T 4096 / B 8 / Muon / c2 mix; ~1.59B params with tied embedding — "1435M"
NOT confirmed by the user yet). Env wins over YAML (ConfigBridge set_env). `BLU_PARALLELISM` = parallelism.enabled (an int),
the parallelism YAML path comes from `parallelism.config` relative to the main YAML. The production YAML runs bf16
LOGITS (mixed_precision: enabled -> BLU_ALL_BF16=1, bf16_logits -> BLU_BF16_LOGITS=1; fp32 is only the built-in default):
OLD's loss backward briefly holds 2 x 3,144 MiB bf16 gradients, NEW none. BLUCCL afc462d BUG: the YAML's
monitoring.production (dba5cf4) is unknown to ConfigBridge.h -> trainer FATAL with the default YAML; A/B copy drops the
line (only Makefile + blutrain_monitor.py read BLU_PRODUCTION); 1-line env_table fix offered to the user.

**DONE + PUSHED 2026-10-21 ~16:10 IST.** GitHub `dist_side_gautam_lottn`: TI `b1443a5` (6 commits on 02d0d18: e857c77 single
gradient alloc, 43a5c3f bwd 16-byte, 37f2196 in-place only, fc4459b fwd 16-byte, c435578 Vj2+packed+F3c, b1443a5 QPU; every
commit compile-checked alone, tip = tested files), BluTrain `3b80f15` = merge of BLUCCL 4f45a88 (colleague's multinode
BluCCL commit, pushed 10-21 15:55 AFTER we branched from afc462d) + our TI pointer; differs from 4f45a88 by that 1 line.
Box results (runs pulled to profiling_reports/pretrain_branch_2026_10_21/box_runs/): A loss bit-identical, bf16 bwd kernel
4.35->1.64 ms, steady lm_head+loss bwd -2.7 ms/micro-step, -5.8 GiB alloc peak; B attention fwd +18.9%, bwd +40.5%
(507->605 / 360->506 TFLOP/s), quick suite 296/296; C production trainer +6.4% tok/s (45,936->48,855, ratio 0.9402 both
pairs, NEW at 2% lower clock under the 700 W cap), peak 103,893->96,725 MiB, loss/val/norm within OLD's own noise;
D full suite 1249/1249 (546 s). Open: colleague's monitoring.production bug (still in 4f45a88), work doc §45.62+.
**How to apply:** the branch is handed over; later SFT push = CudaNativeType in loss + packed RoPE + logits-only-trained. Related: [[project_state_2026_10_19_eod]],
[[feedback_user_runs_all_box_commands]], [[feedback_never_shorten_max_steps_to_benchmark]], [[project_laptop_glibc243_cuda13_break]] (resolved 10-21).
