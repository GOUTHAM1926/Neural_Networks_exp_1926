---
name: project_muon_gpu_guard_1104
description: "11-04 item 1 DONE: Muon NS guard on GPU, VERSION 2 = no flag, old CPU check removed in 3 places; tested 1 GPU + 2 GPU NCCL/BluCCL pipelined/fused; push commands ready (push_1104), user pushes"
metadata:
  type: project
---

Item 1 of the per-step CUDA-call list (work doc §45.140) built 11-04 and tested the same day on the 2x H200 box
86.38.238.65 (GPU 1 ours; GPU 0 used by colleague Adwaid). Results in work doc §45.142 and
`work_docs/muon_optimizer_synchronize_checks_removal_optimization.md` §10:
- unit test ALL PASS (90 cases + 222-matrix step, weights bitwise equal);
- trainer old vs new within old-vs-old noise (trainer is NOT bit-reproducible run to run: 298/372 tensors differ even old-old);
- cudaStreamSynchronize -26,640 and cudaMemcpyAsync -53,220 per 60 steps (exactly as expected);
- step 134.38 -> 127.53 ms (-5.1%).

Local uncommitted code: ~/Downloads/BluTrain_push (+ its Tensor-Implementations); patches in
profiling_reports/muon_gpu_guard_2026_11_04/. Box copy: ~/BluTrain_mg with patches applied.

**Why:** the user decides the flag default and pushes (branch rule [[feedback_branch_rule_sft_vs_experimental]]).
**How to apply:** next = item 2 (per-parameter copy loops), one item at a time, in depth.

New-box traps: libyaml-cpp-dev missing; unit test needs -lcuda and LD_LIBRARY_PATH from `make -s print_run_ld_path`;
run_sft.sh does not detect other people's non-bluscript GPU jobs. Related: [[project_next_cudafree_normal_path]].

**Version 2 (11-04 afternoon, user: "no flag, delete the old method"):** old check removed from ShardedMuon normal +
fused-publish path (BluCCL kernel reads `MuonPublishSeg::bad`) + TI nn::Muon; muon_has_nonfinite_cuda deleted. Patches
ti v2 7b28f915 / blutrain v2 97fee8fc. Tests all PASS: 2x box 1 GPU (-5.4%), P032 bt_old/bt_new side by side
(/localstorage/gautam_k/muon_guard_p032, KEEP: user said no cleanup, reuse bt_old as baseline): NCCL 2 GPU, BluCCL 2 GPU
pipelined (-6.4%) and fused. Fused speed single-run noisy (others' jobs) - optional repeat. Work doc 45.143.
P032 is busy with others (Aparna GPU 6 scaling runs, Mahesh SFT GPU 5, Ananya GPU 0): always check GPUs before runs.

**PUSHED 11-04 17:15 IST + verified on GitHub:** TI d98d6ee, BluTrain 37514ff (dist_side_gautam_lottn). Local BluTrain_push still has unrelated uncommitted sft/* format-study files and the Context_Parallelism pointer (not pushed on purpose).

**CLEAN rerun 11-04 17:00-17:32 (P032 GPUs 4/5, one at a time, nobody else on them):** all pass; step -7.1% (1 GPU), -6.8% NCCL, -7.5% BluCCL, -6.8% fused. Group message for colleagues: user will send it AFTER all redundant-call items are done and pushed (one pull for everyone).
