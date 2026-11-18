---
name: project_cp_sft_impl_1106
description: "11-06: SFT context parallelism (BLU_SFT_CP=U) wired into our SFT trainer with Adhitya's Ulysses all-to-alls (his submodule, as-is); FORWARD exact, BACKWARD loses gradient (open bug); tests so far + next steps"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-11-07T08:27:52.581Z
---

**ROOT CAUSE FOUND + FIXED 11-07 (laptop, CPU analysis + RTX 3060):** TI autograd::contiguous
(ReshapeOps.cpp).
- Tensor::contiguous() returns *this for a contiguous offset-0 tensor.
- make_unary_op then stamped a new ContiguousBackward onto that same object (= the input), keeping its output_nr.
- k_bf is output 1 of fused_rms_rope_bf16, so cp_combine's edge became (ContiguousBackward, slot 1). The engine puts dK
  in slot 1, and ContiguousBackward::apply returns nothing when slot 0 is empty: **dK and k_gamma are dropped in every
  layer**.
- Why Adhitya never hit it: his q/k/v are transpose outputs (output 0, non-contiguous -> real copies).

**Fix PUSHED 11-07 (user ran the commands) + VERIFIED on GitHub file by file:** TI dist_side_gautam_lottn 66a32a7 -> d59740b (ReshapeOps.cpp +14, test_contig_alias.cu +66; remote blobs == tested files), BluTrain accc6c8 -> 8086e78 (TI pointer only). Author Goutham Kumar, no AI trailer. CP code + 2 sft docs stay local (not pushed, on purpose). Bug-fix table row 26 given to the user.
**Fix (user OK 11-07):** ReshapeOps.cpp: `if (input.is_contiguous() && input.storage_offset() == 0)
return input;` (like to_dtype / PyTorch). The old body is kept as a comment.
- Verified on the RTX 3060: test_contig_alias.cu before = k 0 / k_gamma 0, after = equal to the no-contiguous run
  (k 156.991745, k_gamma 350.833604); TI autocheck 43/43 pass.
- Regression test added: TI Tests/kernel_tests/test_contig_alias.cu.
- All callers checked: every other call passes a transpose view (unchanged).
- Laptop build: scratchpad/ti_repro/bt (BluTrain accc6c8 + TI 66a32a7 + the fix, SM_ARCH=86).

**Pending:** a 2-GPU CP=2 rerun (step-0 grad norm must be 2.8002); TI fix + pointer already pushed (d59740b / 8086e78).

**11-11 UPDATE: items 1 + 2 below are DONE.**
- BluTrain 8086e78 -> 125ae9f: CP code + README 5.2 + the 11-07 docs. Personal repo 6fc9f20: work docs. Both verified
  on the remotes file by file.
- Still open: CP + data parallel (4+ H200), the Adhitya follow-up, the preproduction-135M push (the user's word).
**RESUME WED 2026-11-11 (user on Diwali leave 11-07 evening to 11-10). Open, in order:**
1. **Docs push:** the user ran nothing yet (BluTrain head was still 8086e78 at 13:05 IST). The commands were given 11-07:
   - BluTrain: sft/README.md, sft/docs/sft_kernel_benchmarks.md, sft/docs/kernel_diff_sft_pretraining.md;
   - personal repo: work_docs sft_work_documentation.md, plan_grid_search_sft.md, kernel_diff_sft_pretraining.md,
     sft_kernel_benchmarks.md.
   Verify on GitHub after.
2. **CP code push** (bluscript_sft.cpp, Makefile, run_sft.sh; local only): first add the CP section to sft/README.md (and
   update the README section 9 "CP in progress" bullet), then the user pushes.
3. **Tests not done yet:** CP + data parallel (needs 4+ GPUs).
   - The 1.6B with CP is DONE (11-07 13:52-13:57 IST, 2x box, random init):
     - 4k: CP off step-0 norm 41.0146; CP=2 41.0215 (+0.017%), G2 equal 0.0338; 30-step train diff 0.000737.
       Step time 480.0 -> 274.5 ms (1.75x); peak 66.7 GB on 1 GPU vs 42.4 GB per GPU.
     - 64k: CP=2 runs (96.6 GB per GPU, 861 ms per step).
     - 64k on ONE GPU: out of memory ("cudaMallocAsync failed: out of memory" at 143.1 GB).
     - So CP is what makes the 1.6B at 64k possible. Results: box_results/results_2x.txt (appended).
4. **Adhitya:**
   - follow-up question: is his forward_ulysses_gqa KV copy (rep = P/nkv) fine for U > KV heads, or only the hybrid?
   - when he pushes: diff BlubridgeAI/Context_Parallelism 73b3bdd..new.
5. **preproduction-135M push:** when the user says; tell Navin about the segmented switch first.
6. **11-07 EOD** given. The 2x box results are verified on the laptop: 116 files identical. Our box folder
   /root/gautam_k/cp2x (trees + 3.5 GB data) can be deleted when the user says.

**CP=2 PROVEN 11-07 12:55-12:59 IST (2x box, GPUs locked by holders, released after). Results:
profiling_reports/cp_2026_11_07_2x/box_results/results_2x.txt.**
- 4k (data_4k = 480-row train set, same val as m1_smol_A):
  - CP off: step-0 grad norm 3.0473.
  - CP=2 with the fix: 3.0473 (+0.000%), G2 0.0458 = 0.0458; largest train-loss difference over 30 steps 0.000997,
    against CP off vs CP off 0.000813 and data-parallel 0.001163.
  - CP=2 WITHOUT the fix (control, same box): 2.8367 (-6.9%), G2 -31%; 11-06: -7.46%.
  - (2.8002 was m1_smol_A's first 4 rows.)
- 64k: CP=1 387.05 / 389.22 (same-box variation +0.56%); CP=2 386.98 (-0.02%).
  - Step time 250.7 -> 158.2 ms (1.58x); peak per GPU 61.5 -> 36.3 GB.
- 128k: CP=1 34.6985 / 34.7813 (+0.24%); CP=2 34.7619 (+0.18%).
  - Step time 492.9 -> 297.9 ms (1.65x); peak per GPU 111.0 -> 62.2 GB (predicted ~61).
- The 0.24% open point (128k vs 10-31) = normal run-to-run variation (two identical runs differ by 0.24%).
- Not done yet: CP + data parallel (needs 4+ GPUs); 1.6B with CP; pushing the CP code (user decides); README CP section
  with the push.

**11-07 ~12:35 IST: 2x H200 box 86.38.238.112 (shy-earth-writes-fin-02; SHARED: /root/adwaid, /root/ram = colleagues,
never touch).**
- Ours: /root/gautam_k/cp2x; yaml-cpp in cp2x/deps (apt-get download + dpkg -x, NOT installed system-wide).
- CUDA 13.2, NCCL 2.30.4, openmpi 4.1.6.
- Scripts: profiling_reports/cp_2026_11_07_2x/pkg/tests_2x.sh (stages setup / smoke / cp4k / long) + compare_2x.py.
- Docs push commands given to the user (BluTrain: 3 sft doc files; personal repo: 4 work_docs files). Verify on GitHub
  after the push.

**NEXT (user 11-07: "if GPUs available I will tell, then we do our tests"):**
- 2 GPUs: CP=2 4k vs CP off (step-0 norm must be 2.8002, not 2.5912) + CP=2 64k/128k vs CP=1 on the SAME box +
  test_cp_exchange.
- 8 GPUs: CP=2 x 4 DP.
- 1 GPU only: the forced-U=1 CP-path test (test switch, local only; fixed TI -> 2.8002, original TI -> about 2.5912
  proves the bug is the whole gap).
- CP=4 is impossible: the 135M has 2 KV heads and the 1.6B has 6 (U must divide them, bluscript_sft.cpp:1516). On 8
  GPUs, U = 2 only.

**11-07 1x H200 (86.38.238.69, CLOSED by the user after T3; all 55 result files verified identical on the laptop;
/root/gautam_k/cp1x; records profiling_reports/cp_2026_11_07_1x/):**
- T1: test_contig_alias, fix tree A = B (k 156.991745, k_gamma 350.833604). The nofix tree's B gives k 0 / k_gamma 0 (link
  ContiguousBackward, slot 1). autocheck 43/43.
- T2: normal SFT (CP off, data_4k, 30 steps). Step 0 is 1.530339 in all three runs.
  - Max train-loss difference: fix vs nofix 0.000598 / 0.000604; nofix vs nofix 0.000630.
  - Conclusion: no effect.
- T3: CP=1 references (135M ckpt_A, data_long_T, B 1, 1 row per step, 30 steps); both runs clean.

  | T | step-0 val loss | step-0 train loss | step-0 grad norm | peak memory (nvidia-smi) |
  | --- | --- | --- | --- | --- |
  | 65,536 | 3.216672 | 7.105492 | 398.4949 | 60,913 MiB |
  | 131,072 | 3.216672 | 4.334521 | 34.7171 | 110,361 MiB |

  - 131k step-0 train loss = the 10-31 gate.
  - The loss spikes (5-9, norm up to 1,117) are rows whose trained tokens sit past position 4096. ckpt_A meta has
    context_length 4096.
    - Loss is about 1.70 before position 4096 and about 6.9 after it (correlation 0.95 / 0.92).
    - Rows with no tokens past 4096 have loss 1.63-2.43.
    - pkg/rows_vs_steps.py; the loader uses rows in file order, so step s = row s (trained counts match 30/30).
  - Step-0 norm vs 10-31 at 131k: 34.7171 vs 34.8020 (-0.24%). Train loss is identical.
    - Same-build repeats vary by 0.004% (4k) to 0.06% (16k). There is no 131k repeat yet, so noise vs a change is
      undecided.
    - A T3 repeat was offered to the user (it also gives the noise floor the CP=2 comparison needs).
- Prediction to test: CP=2 at 128k per GPU is about the same as CP=1 at 64k (about 61 GB nvidia-smi).
- Adhitya 11-07: "Dividing kv heads is not available in Ulysses, use ring-Ulysses hybrid", the answer to our Q5.
  - His code cf1372d does replicate KV heads (ContextParallel.h:1713-1740, rep = P/nkv, DeepSpeed rule). Follow-up
    question: is that path OK for U > Nkv, or is only the hybrid recommended?
  - His fixes from our issues doc are NOT pushed: BlubridgeAI/Context_Parallelism (repo moved from adhityablubridge)
    has only main @ 73b3bdd (11-01), 0 PRs, 0 forks.
  - All 3 issues are in code we do not call (gather_seq, head_repeat_interleave, his trainer's loss). Diff
    73b3bdd..new when he pushes.

**User decisions (11-06):**
- Use Adhitya's Context_Parallelism submodule AS IS: dist/Context_Parallelism, pinned cf1372d. The local checkout was moved
  from 661b6ef to cf1372d; it was clean, and the two headers we use are identical in both.
- Work on our branch dist_side_gautam_lottn (~/Downloads/BluTrain_push). The work is local and uncommitted; the user
  pushes later.
- No kernel changes are needed (explained to the user).

**Code (local, NOT pushed):**
- Makefile: TRAIN_INCLUDES + -I$(DIST_DIR)/Context_Parallelism.
- sft/run_sft.sh: SFT_CP -> BLU_SFT_CP.
- bluscript_sft.cpp, behind BLU_SFT_CP (default 1 = off):
  - the SftCp struct;
  - cp_make_group: ncclCommSplit plus a BLOCKING stream, like his DeviceMesh (his all-to-all does not wait on the
    compute stream);
  - cp_combine / cp_partition: his combine_ag / partition_ag lines around his public ulysses_* functions and his
    UlyssesCombine/PartitionBackward nodes;
  - cp_slice;
  - attention: our RoPE on the slice -> combine q/k/v -> our SDPA with the plan built for Nq/U heads -> partition;
  - the loader runs with dp_rank / dp_world; the counts are /U; embedding install N = B*T/U; tokens are /U;
  - checks: U divides world, Nq, Nkv and T; no trained-rows switch.
- P032 build: cp_1106/bt (a copy of pvo_1106/ours plus the 3 files), bluscript_sft_bin e2c0aac1.

**Tests 11-06 (P032 GPUs 2+3; 135M ckpt_A, m1_smol_A, B 4, 4 rows/step):**

| run | val loss | grad norm |
| --- | --- | --- |
| A (CP off, 1 GPU) | 1.424576 | 2.8002 |
| C (2 GPUs data-parallel, no CP) | 1.424576 | 2.8003 |
| B (CP=2) | 1.424576 | 2.5912 |
| D (CP=2, BLU_NO_OVERLAP=1) | 1.424576 | 2.5917 |

- The forward is exact. The backward is wrong in a deterministic way: group norms G0 -7.8%, G1 -5.9%, G2 (gammas/norms)
  -36%.
- Ruled out:
  - the attention kernel at 2/1 heads (test_varlen_attention VARLEN_HEADS=2/1,4/2: 515 pass);
  - the RoPE bf16 backward on a mid-chat slice (test_rope_slice.cu: dq/dk bit-identical, gamma sums equal);
  - the ZeRO overlap (run D);
  - the engine's dependency counting (per edge, FIFO; looks right).
- **Next (needs 2 GPUs; P032/P015 are gone until next Wed, Verda maybe):**
  - a 2-GPU unit test of cp_combine/cp_partition forward + backward against a known gradient;
  - per-tensor dQ/dK/dV around the exchanges, CP=2 vs CP=1, for one layer;
  - try his monolithic-node pattern (all three partitions in one backward node, as in UlyssesAttentionBackward).
- Records: profiling_reports/cp_2026_11_06/ (cp_test.sh, test_rope_slice.cu, p032_results/).
- READY for the next session (CPU work, 11-06 night):
  - `test_cp_exchange.cpp` (2 GPUs: T1 round trip, T2 the backward of the exchanges, T3 exchange + attention vs the full
    row). Build it like a trainer (`make _ablation_compile TRAIN_SRC=...`).
  - `cp_module_analysis.md`: the deep read of his module, with line numbers.
- Also checked that night:
  - his gamma AVG rule is for his fused path only (QK-norm after the all-to-all), not ours;
  - Work::wait() = cudaStreamWaitEvent (GPU-side);
  - his mesh sub-groups use blocking streams (device_mesh.cpp:184);
  - our attention op's autograd hookup is right (AttentionOps.cpp:418-430).
Related: [[project_future_cp_ulysses_sft]], [[project_p032_cleanup_1106]].
