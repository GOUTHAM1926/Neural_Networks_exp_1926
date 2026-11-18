---
name: project_p022_box_1030
description: "P022 node (ssh p022 = operator@103.42.51.252 -p 2225) for the component-9 study; GPU owners; user STOPPED the setup 10-30 13:05 — wait for the user's word"
metadata:
  type: project
---

2026-10-30 ~13:00 IST the user moved the component-9 tests to **P022** (not P032). GPUs: 6 is fully ours; 0 and 1 belong to
Surendra and 3 to Ananya (usable only in their free time, with the user's word); 7 had someone's libbench; 5 harshini.

P022 state when checked (read-only): no /localstorage/gautam_k folder; 5.9 TB free on /localstorage; 224 cores; 2 TB RAM;
python 3.12.3; nvcc present. P032's sft135 tree has the same bluscript_sft.cpp / SftDataLoader.h md5 as the laptop.

Plan I proposed (NOT done): copy BluTrain tree + prepenv + ckpt_A + base_hf + tokenizer_base from P032 via the laptop,
rebuild data_fmt/init_fmt on P022 with the verified scripts and compare checksums with fmt_study_manifest.json, build the
trainer, create a vLLM env (lm_eval[vllm,ifeval]==0.4.12), then gates -> parity -> smoke -> screen.

**The user rejected the first copy and said: "stop, we will do these later, I will tell; don't do anything regarding
this chat-template test; don't use any GPU."** Nothing was written to P022 or P032 (only ls / nvidia-smi / md5sum).

**How to apply:** do nothing on P022/P032 for this study until the user says so; then start with the setup plan above,
step by step. Related: [[project_sft_grid_plan_1028]], [[feedback_user_runs_all_box_commands]].

**2026-10-30 evening (loss-fix runs on GPU 6, given by the user):**
- P022 = 8x **H200**, CUDA 13.4, every GPU in **Exclusive_Process** mode: only one process can hold a GPU at a time.
- Aparna runs short 8-GPU debug jobs on ALL GPUs, including our GPU 6, every ~7 min
  (`/localstorage/Aparna/BluTrain_Eager/logs/diag_*_1030_*`, mpirun + `runner_bluccl`).
  - Her job held GPU 6 -> our run failed with "busy or unavailable" (harmless to her).
  - Our timing held GPU 6 at 19:33 -> HER run `diag_gdb2_1030_1933` aborted with the same error.
- The free check passing a minute earlier does NOT protect her next launch.
- Our work dir: `/localstorage/gautam_k/loss_fix_1030/`. `/localstorage/gautam_k/sft135/` is a partial BluTrain copy (1.1 GB), left untouched.

**How to apply:** before any GPU-6 run on P022, the user must agree a slot with Aparna (or she must leave GPU 6 out). Never run in the gaps of her job: a collision kills her run.
