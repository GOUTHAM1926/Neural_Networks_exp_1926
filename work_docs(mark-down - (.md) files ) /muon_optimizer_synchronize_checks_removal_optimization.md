# Muon optimizer: removing the per-matrix CPU waits (synchronize checks) — explanation, change, tests

Status (2026-11-04, final): **done.** (Today's other two Muon-step changes and the before/after of all three: section 13; new code
-19.6% step time vs the 11-03 code.) The old CPU-decided check and the flag are removed everywhere; the GPU-decided
check is the only path. **Pushed and verified on GitHub: Tensor-Implementations `d98d6ee`, BluTrain `37514ff`**
(branch `dist_side_gautam_lottn`, section 12). Final clean test results: section "Final results" right below and 10.12.

## 1. In short

Every training step, the Muon optimizer updates the model's 222 weight matrices one after another, and for each one
the CPU stopped twice to wait for the GPU: once for a NaN check and once for a log-only number (`nsratio`). That is
**444 CPU waits per step**, about **16 ms of CPU waiting per ~142 ms step** (measured 2026-11-03 with my own call
tracer, no profiler). I moved both onto the GPU, the way PyTorch's fused Adam does its skip check: the checks still run
per matrix with the same rule, but the decision is made by the GPU and the CPU only reads two numbers per step for the
log. First built behind a flag (`BLU_MUON_GPU_GUARD`, version 1); after the tests the old CPU path and the flag were
removed, so the GPU-decided check is now the only one (version 2, section 9).

Result: same decisions and same weights as before, all 444 waits per step gone, and **7% faster steps** (clean
rerun below).

### Final results (clean rerun, P032, 2026-11-04 17:00-17:32 IST; details in 10.12)

How it ran:
- GPUs 4 and 5 only, 16 runs, **one at a time** (our own runs never overlapped).
- Each test 4 runs in this order: old_a, new_a, new_b, old_b. old = the code before the change (BluTrain 7184fe1 /
  TI d18f820), new = the pushed change (37514ff / d98d6ee). The two values in each row below are two separate runs of
  the same code; their difference is the run-to-run noise. Old first and last, new in the middle: if the node got
  slower or faster during the test, both sides were affected equally.
- Before every run the script checked that no one else had a process on GPU 4 or 5 (it never had to wait). During every
  run a recorder listed all GPU processes of the node about every 10 s; it saw our own processes on GPUs 4 and 5 in every
  run (so it worked) and **no one else on GPU 4 or 5 in any run**. Other people's jobs ran only on other GPUs.
- All 16 runs: 60 steps, 0 errors; the fused runs really used the fused kernel (not refused).
- Same model (135M, 37 layers), data (`data_4k`), start weights (`ckpt_A`), Muon lr 5e-5, 60 steps in every run.

Correctness (max train-loss difference over the 60 steps; `nsrej` = matrices rejected by the guard):

| test | old vs old (noise) | new vs new (noise) | **old vs new** | `nsrej` |
| --- | --- | --- | --- | --- |
| 1 GPU | 0.000976 | 0.000777 | 0.001027 / 0.000700 | equal on every step |
| 2 GPUs NCCL | 0.000995 | 0.000737 | 0.000703 / 0.000839 | equal on every step |
| 2 GPUs BluCCL pipelined publish | 0.000803 | 0.000689 | 0.000535 / 0.000585 | equal on every step |
| 2 GPUs BluCCL fused publish | 0.000763 | 0.000585 | 0.000821 / 0.000963 | equal on every step |

Old vs new differs no more than two runs of the same code. (The trainer is not bit-reproducible from run to run, so
the bit-level proof is the unit test: same decisions, weights bitwise equal, sections 10.2 and 10.8.)

Step time (median of steps 11-60):

| test | old (old_a / old_b) | new (new_a / new_b) | change |
| --- | --- | --- | --- |
| 1 GPU | 137.2 / 139.1 ms | 128.7 / 128.1 ms | **-9.8 ms (-7.1%)** |
| 2 GPUs NCCL | 72.5 / 72.4 ms | 67.4 / 67.7 ms | **-4.9 ms (-6.8%)** |
| 2 GPUs BluCCL pipelined publish | 73.2 / 73.3 ms | 67.8 / 67.7 ms | **-5.5 ms (-7.5%)** |
| 2 GPUs BluCCL fused publish | 72.3 / 72.7 ms | 67.8 / 67.3 ms | **-5.0 ms (-6.8%)** |

- Two runs of the same code differ by 0.1-1.9 ms; every gain (4.9-9.8 ms) is far above that.
- CUDA calls per 60 steps (1 GPU, call tracer): `cudaStreamSynchronize` 28,794 -> 2,154 (-26,640 = 222 matrices x 2
  x 60 steps), `cudaMemcpyAsync` 90,402 -> 37,182 (-53,220). No per-matrix CPU wait is left in Muon.
- The fused path's 75.2 ms in the first P032 attempt (10.9) came from other jobs loading the node, not from the change.
- On 1 GPU the 2x H200 box gained -5.4% (134.4 -> 127.1 ms); P032 -7.1%: a different machine, where the removed CPU
  waits cost a different amount.

## 2. Where everything lives

All the numbers of training live in **GPU memory** the whole time:

- **W**: the "master weights", in fp32. For example `layers.0.wq`, a 512 x 512 matrix = 262,144 numbers.
- **G**: the gradient of that matrix from the backward pass.
- **M**: the momentum buffer of that matrix (Muon keeps one per matrix).

The **CPU never holds these numbers**. The CPU only runs the program, which decides **which job the GPU should do
next**.

## 3. How the CPU makes the GPU work

- Each GPU job (a "kernel") is started by the CPU sending one command into a **queue** on the GPU.
- The GPU takes the commands out of the queue **in order** and runs them.
- Sending a command takes the CPU a few microseconds, and the CPU does **not** wait for the job to finish. It sends the
  next command right away.
- So normally the CPU is ahead: the queue always has jobs waiting, and the GPU is never idle.
- **The CPU has to wait only when it wants to read a number from GPU memory.** It can only get that number after the
  GPU has finished every job queued before it.

## 4. One matrix in one step, in order (example: `layers.0.wq`, 512 x 512)

The model has 37 layers with 6 Muon matrices each (`wq`, `wk`, `wv`, `wo`, `gate_up`, `down`) = 222 matrices
(the trainer log prints `222 orthogonalized`). Code: `dist/zero/src/ShardedMuon.cpp`, function `update_one`.

1. **Weight decay:** W = W x (1 - lr x wd). This makes the weights a tiny bit smaller.
2. **Momentum:** M is updated with the new gradient G, and from M a new matrix is made: the input to step 3.
3. **Newton-Schulz:** 5 rounds of matrix multiplications turn that input into **U, the "update"**: a 512 x 512 matrix
   of the same shape as W. A healthy U has values of about +-0.04.
4. **The check, on U.** Not on the gradient, not on the weights. It looks at **all 262,144 numbers of U**. If any
   number is NaN, Inf, or bigger than 1000 in size, the flag is set: "this U is broken". Newton-Schulz in bf16 can very
   rarely blow up like that.
5. **`nsratio`:** the size of U, compared with what a healthy U should have (sqrt(min(rows, cols))). This is only for
   the log line.
6. **Apply:** W = W - lr x adjust x U. **This is the actual weight update.**

After all 222 matrices, the GPU converts the new fp32 W into the bf16 copies the model uses. That also happens in GPU
memory; nothing goes back to the CPU.

## 5. What happens when U is broken (flag set)

- **Only step 6 is skipped, for only this matrix, for only this step.** Its W keeps the value it had after step 1 (the
  weight decay), exactly as in the old code.
- The other 221 matrices update normally.
- **Training does not stop.** The log counts it (`nsrej 1/1`). In the next step this matrix gets a new gradient and a
  new U, and is checked again.
- In all my runs it never happened (`nsrej 0/0`).

## 6. The two different checks

| | my F1 check (2026-09-26) | the per-matrix guard (this document) |
| --- | --- | --- |
| what it checks | the total gradient norm, once per step | each matrix's Newton-Schulz output U |
| when it fires | the gradient is NaN or Inf | U has NaN, Inf or a value above 1000 |
| what happens | `opt->step()` throws `ZeroNonFiniteGradError` before any weight, momentum or optimizer state is touched; the trainer prints `[FATAL] ... stopped before the weight update; no checkpoint written` and calls `MPI_Abort`: **the run stops** (`bluscript_sft.cpp` ~3633) | **only that matrix's update is skipped** for that step; training continues |
| changed here | no | only where the decision is made (GPU instead of CPU) |

**Does a stop by F1 mean starting again from step 0?** No. The trainer saves checkpoints every `BLU_CKPT_FREQ` steps
(1000 in `run_sft.sh`, the last 2 kept), and `RESUME=1` continues from the run folder's newest checkpoint. What F1
prevents is saving the step whose gradient is NaN, so the saved state is never poisoned; the steps since the last
checkpoint are lost. A NaN gradient usually means something went wrong in that batch or the steps just before it (bad
data, too high a learning rate, an overflow), so before resuming, the cause is looked at (for example skip that batch
or lower the learning rate); resuming unchanged may hit the same NaN again.

Who wrote what (git history): the per-matrix guard `muon_has_nonfinite_cuda` by kathir-23s (Tensor-Implementations
927d0a9, 2026-08-13), wired into `ShardedMuon` by kiruthikkanna-m (BluTrain c2eaffc, 2026-08-31); the `nsratio`
diagnostic by kiruthikkanna-m (7e2264f, 2026-09-07); the F1 check by me (2026-09-26).

## 7. The OLD way and the NEW way, in time order

**Old, for one matrix:**

```text
CPU: send 1, 2, 3 (Newton-Schulz), send "check U -> flag", send "copy flag to CPU"
CPU: WAIT ......... (until the GPU has finished all of Newton-Schulz and the check, and the 4 bytes arrive)
CPU: read flag.  If broken: count it, go to the next matrix (send nothing more).
                 If fine:   send "measure size of U", send "copy to CPU", WAIT again ......, compute nsratio,
                            then send 6 (apply).
GPU: during each WAIT it finishes its queue and then sits idle until the CPU sends the next command
```

Why the CPU waits: it wants to decide by itself whether to send job 6. To decide, it must have the flag in its own
memory, so it must wait for the GPU. That is 2 waits x 222 matrices = **444 waits per step**.

**New, for one matrix:**

```text
CPU: send 1, 2, 3, send "check U -> flag[k]" (the flag stays in GPU memory),
     send "nsratio of U (skip if flag[k] is set) -> maximum in GPU memory",
     send "apply: if flag[k] is set -> do nothing, count +1; otherwise W = W - lr x adjust x U"
CPU: immediately continues with the next matrix. No waiting.
GPU: runs them in order. When the apply job runs, the check job before it in the queue has
     already finished, so flag[k] is ready to read.
```

**How does the CPU know whether to send the apply job?** It does not need to know. **It always sends it.** The apply
job itself looks at the flag in GPU memory when it runs, and decides there: update, or do nothing. Same rule, same
decision, now made by the GPU.

**When does the CPU learn what happened?** At the end of the step. The step already has to wait for the GPU there, to
hand the new weights to the model. Right then the CPU reads **2 numbers** from GPU memory: how many of the 222
matrices were skipped, and the largest `nsratio`. It prints them in the log line exactly as before. Since it was
waiting there anyway, this read stops nothing extra.

| | old | new |
| --- | --- | --- |
| check per matrix, on U, before its update | yes | yes (same rule) |
| who decides "update or skip" | the CPU, after copying the flag | the GPU apply job, reading the flag |
| CPU waits per step | 444 | 0 (plus 1 read at the end, where the step waits anyway) |
| what goes back to the CPU | 444 small numbers per step | 2 numbers per step (for the log) |
| weights | always in GPU memory | always in GPU memory |

## 8. How others do it (their source code, read 2026-11-04)

| library | how a "skip if bad" decision is made | does the CPU wait? |
| --- | --- | --- |
| PyTorch fused Adam (`aten/src/ATen/native/cuda/fused_adam_utils.cuh`) | a `found_inf` flag in GPU memory is passed to the kernel, which starts with `if (found_inf_ptr && *found_inf_ptr == 1) { return; }` | no: no copy to the CPU, no sync in the file; its comment says fused optimizers are built this way "in order to remove the device sync" |
| NVIDIA Apex (`csrc/multi_tensor_l2norm_kernel.cu`) | the norm kernel sets a GPU flag on a non-finite value: `if (!isfinite(final)) *noop_gmem = 1;` | no: the CPU never reads the flag in those files |
| Keller Jordan's Muon, PyTorch `torch.optim.Muon` | no per-matrix check at all; the norm before Newton-Schulz is computed on the GPU | no |

So the new path is the PyTorch fused-Adam pattern, applied to Muon's check.

## 9. The change in the code (final = version 2)

History: version 1 (morning of 11-04) added the GPU path next to the old one, switched by `BLU_MUON_GPU_GUARD`
(default 0 = old code), and kept the old CPU check on the fused-publish path. After the version-1 tests passed, the user
decided: no flag, delete the old method. Version 2 removes the old CPU check from **all three places** it existed:

| # | where | before | version 2 |
| --- | --- | --- | --- |
| 1 | `ShardedMuon::update_one`, normal path (every ZeRO trainer: SFT and pretraining `bluscript_zero*`) | 2 CPU waits per matrix | GPU flag; the apply kernel reads it |
| 2 | the same function, fused-publish path (`BLU_FUSED_MUON_PUBLISH=1`, BluCCL, off by default) | CPU check; a rejected matrix sent as `alpha = 0` | GPU flag; the BluCCL fused kernel reads it |
| 3 | Tensor-Implementations' own non-sharded `nn::Muon::step` (`Optim.cpp`; no BluTrain trainer uses it) | 1 CPU wait per matrix | GPU flag; the guarded apply kernel |

Tensor-Implementations (`include/ops/helpers/MuonKernels.h`, `src/Kernels/cuda/optimizer/MuonKernels.cu`,
`src/nn/optimizer/Optim.cpp`):

- **Removed:** `bool muon_has_nonfinite_cuda()` (the host function that copied the flag to the CPU and waited). Its
  check kernel `muon_isnonfinite_kernel` and the rule `muon_update_elem_is_bad` (`!isfinite(v) || fabsf(v) > 1000.0f`)
  are unchanged and used by the new function.
- `muon_flag_bad_cuda(x, n, flag)`: the same check kernel, writing into a flag the caller gives (the caller zeroes it).
- `muon_apply_guarded_cuda(work, update, coef, n, bad_flag)`: if the flag is set, the kernel returns **without any
  multiply** (a "multiply by 0" would turn an Inf in U into a NaN in W, because Inf x 0 = NaN); otherwise the same
  arithmetic and the same launch as `muon_apply_cuda`, so the weights are bit-identical.
- `muon_update_ratio_cuda(update, n, expect, bad_flag, max_ratio, rejected)`: `nsratio` on the GPU with the old
  formula `sqrt(max(0, sum of squares)) / sqrt(min(rows, cols))`, kept as a maximum in GPU memory and skipped for a
  broken U; it also **counts the rejected matrix** (one place for the count, used by both ShardedMuon paths). It uses the
  existing deterministic sum kernel and takes its inputs as kernel arguments. The old code used a multi-tensor norm
  that uploads its input list through one shared pinned CPU buffer; back-to-back calls would race on that buffer, and
  the per-matrix wait was what kept it safe. The new path needs no upload, so no race.
- `muon_apply_unless_bad_cuda(work, update, coef, n)`: for a caller without its own flag buffer (`nn::Muon`): zero a
  per-GPU flag, check, guarded apply, all queued on the current stream, nothing read back. Reusing one flag matrix after
  matrix is safe because the zeroing, the check and the apply of one matrix are queued on the same stream, in that
  order, before the next matrix's zeroing.
- `nn::Muon::step`: `if (!muon_has_nonfinite_cuda(u)) muon_apply_cuda(w, u, coef, n);` became
  `muon_apply_unless_bad_cuda(w, u, coef, n);`.

BluTrain (`dist/zero/include/ShardedMuon.h`, `dist/zero/src/ShardedMuon.cpp`,
`dist/communication/include/ProcessGroupBase.h`, `dist/communication/BluCCL/kernels/fusions/fused_muon_publish.cuh`,
`dist/communication/src/processGroupBluCCL.cu`):

- `update_owned`: `reset_ns_gpu_counters()` at the start of every step: makes the buffers on the first step (one flag
  per owned matrix, the two counters, a pinned CPU copy, an event) and zeroes the flags and counters on the compute
  stream (no CPU wait).
- `update_one` (both paths): check -> ratio (+ count) -> apply. Normal path: `muon_apply_guarded_cuda`. Fused path: the
  publish segment carries the flag (`MuonPublishSeg::bad`, a new field; the layout of the BluCCL copy of the struct is
  checked at compile time with `static_assert`, now including the new field). The fused kernel reads the flag at the
  start of each segment; when it is set it publishes the master unchanged exactly as the old host-decided reject did
  (`up = master`, `alpha = 0`, i.e. `master - master*0`) and never reads U. The segment is always published, so the
  segment count stays the same on every GPU (the kernel's barrier needs that).
- If BluCCL refuses the fused path at bucket 0 (it falls back to the normal path for the whole step), the flags and
  counters are zeroed again before the fallback, because bucket 0's matrices already wrote them once.
- `step()`: after the updates, one copy of the two counters into pinned memory plus an event, every step; the log
  functions (`ns_rejected_last_step`, `rejected_ns_updates`, `ns_max_update_ratio`) wait on that event (already
  finished when the log line asks) and add the numbers in.
- Removed: the flag `BLU_MUON_GPU_GUARD`, its member and its start-up message, the old CPU check and the old CPU ratio.

Every change has a comment at its place (what, why, date).

Patches (version 2, against BluTrain 7184fe1 / TI d18f820):
`profiling_reports/muon_gpu_guard_2026_11_04/ti_muon_gpu_guard_v2.patch` (md5 7b28f915...),
`blutrain_muon_gpu_guard_v2.patch` (md5 97fee8fc...). Version 1: `ti_muon_gpu_guard.patch`,
`blutrain_muon_gpu_guard.patch`. Unit test: `test_muon_gpu_guard.cu` (version 2), `test_muon_gpu_guard_v1_with_flag.cu`.

## 10. Tests and results

Cost before the change (2026-11-03, my CUDA call tracer, 60 steps, no profiler): the guard's 13,320 copies to the CPU
held the CPU 683 ms and the `nsratio` copies 229 ms, plus 26,640 stream syncs and 26,640 small uploads: about 16 ms of
CPU waiting per step.

Prepared (runbook `~/Downloads/lc_box/muon_guard_box/RUNBOOK.md`):

1. **Unit test** (`test_muon_gpu_guard.cu`): the old and the new path on the same data. 10 cases (clean, NaN, +Inf,
   -Inf, 1000.5, -1000.5, exactly 1000, 999.9, 1e30, all zero), each in the vector part and in the last element, on 5
   shapes; then a 222-matrix step with 3 broken matrices. Checks: same decision, same count, weights bitwise equal,
   broken matrices untouched, `nsratio` within 1e-5, and the CPU time of the step.
2. **Trainer A/B**: 2 runs with the old path and 2 with the new one, 60 steps, the final weights exported. Compared:
   loss, `nsrej` and `nsratio` on every step, and the final weights bit by bit; the old-vs-old difference shows the
   trainer's own run-to-run noise.
3. **CUDA call counts** with the tracer, old vs new.
4. **Step time**, old vs new.

### 10.1 Setup of the runs (2026-11-04)

- Box: 2x H200, 86.38.238.65. Only **GPU 1** was used (GPU 0 was running a colleague's `ncu` profiling); the runs were
  given the upper half of the CPU cores (`taskset`) and mpirun's core binding was switched off
  (`OMPI_MCA_hwloc_base_binding_policy=none`; `mpirun -np 1` pins its process to core 0 otherwise).
- Code: BluTrain `dist_side_gautam_lottn` 7184fe1 + Tensor-Implementations d18f820 + BluBridge-BLAS 297f372, plus the
  two patches. Trainer `bluscript_sft_bin` md5 `2dc5e28d`. Checked before the runs: the flag name is in the trainer
  binary, and the 3 new functions (`muon_flag_bad_cuda`, `muon_apply_guarded_cuda`, `muon_update_ratio_cuda`) are
  exported by `libtensor.so`.
- Trainer runs: the 135M model (37 layers, d_model 512), the 4k-row test data `data_4k`, start weights `ckpt_A`,
  60 steps, B 4, 8 rows per step, Muon lr 5e-5, cosine schedule, final weights exported (`EXPORT=1`). The flag was set
  in a wrapper script (`wrap_g0.sh` = old path, `wrap_g1.sh` = new path).
- Order of the runs: old_a, new_a, new_b, old_b (one after the other, so a slow drift of the box hits both sides
  equally), then trace_old and trace_new (with the call tracer).

### 10.2 Unit test: ALL PASS

The old path (the CPU code of `update_one`) and the new path (the three new GPU functions) run on the same data,
then their results are compared.

- **90 single-matrix cases**: 5 shapes (512x512, 256x512, 1408x512, 512x1408, and 37x13 whose size is not a multiple
  of 4) x (8 broken or edge values, each once inside the vectorised part and once in the last element, + a clean
  matrix + an all-zero matrix).
  - Rejected by both paths: NaN, +Inf, -Inf, 1000.5, -1000.5, 1e30. Accepted by both: clean, exactly 1000, 999.9,
    all zero. The rule (`!isfinite(v) || |v| > 1000`) is the same in both.
  - In every case: same decision, same rejected count, the weights **bitwise equal**, a rejected matrix left
    **untouched**, and `nsratio` within 1e-5 (rejected matrices give 0 in both). The largest ratio difference was
    277.3518066 (old) vs 277.3517761 (new), a relative difference of 1.1e-7: the two reduction kernels add the
    squares in a different order. The trainer's log prints `nsratio` with 3 decimals, so this cannot be seen there.
- **A 222-matrix step** (37 layers x the 6 shapes of the model) with 3 broken matrices (an Inf, a NaN, a 2000):
  both paths rejected 3, the largest ratio is 1.0061849 in both, and **0 matrices** have different weights.
- **CPU time of that step** (guard + ratio + update only, without Newton-Schulz): old 10.252 ms with the CPU busy the
  whole time (the GPU was done 0.006 ms later); new 1.830 ms to queue all the work, then 1.135 ms in the one read at
  the end (the GPU finishing). The old path's time is almost all round trips: 444 waits.

### 10.3 Trainer A/B: same behaviour as before, within the trainer's own noise

| pair | max train-loss difference over 60 steps | `nsrej` equal on every step | max `nsratio` difference | final weights |
| --- | --- | --- | --- | --- |
| old_a vs old_b (old vs old = the noise) | 0.000815 | yes | 0.001 | 298 of 372 tensors differ |
| new_a vs new_b (new vs new) | 0.001014 | yes | 0.001 | 299 differ |
| old_a vs new_a | 0.000850 | yes | 0.001 | 298 differ |
| old_b vs new_b | 0.000939 | yes | 0.001 | 299 differ |

- Old vs new differs exactly as much as old vs old.
- The weights are not bitwise equal even between two runs of the **same** old code: the trainer is not bit-reproducible
  from run to run (its backward pass, seen before on 2026-10-22). So the trainer can only show "the same within its
  own noise"; the bitwise proof is the unit test, where both paths get identical inputs.
- `nsrej` was `0/0` on all 60 steps in every run: in healthy training no matrix is rejected, so the reject path in the
  real trainer is covered by the unit test only.

### 10.4 CUDA calls (call tracer, 60 steps)

| call | old | new | removed | expected |
| --- | --- | --- | --- | --- |
| `cudaStreamSynchronize` | 28,794 | 2,154 | **26,640** | 222 matrices x 2 waits x 60 steps = 26,640 |
| `cudaMemcpyAsync` | 90,402 | 37,182 | **53,220** | 13,320 guard reads + 13,320 ratio reads + 26,640 ratio uploads = 53,280, minus the 60 new per-step reads = 53,220 |

- Both match the expected numbers exactly.
- Where the old CPU time went (the time the CPU spent inside each call, 60 steps): the guard's read
  (`muon_has_nonfinite_cuda`) 13,320 calls, 801.7 ms; the ratio read (`ShardedMuon.cpp:1772`) 13,320 calls, 205.4 ms;
  the ratio's uploads (`multi_tensor_grad_norm_cuda`) 26,640 calls, 45.9 ms; the two sync sites 15.2 + 17.9 ms. In
  total about 1.09 s, **about 18 ms per step**.
- New: nothing per matrix is left in `ShardedMuon::update_one`. The only new call from it is one `cudaMalloc` of
  3,072 bytes for the whole run (the ratio kernel's scratch buffer, 768 floats).

### 10.5 Step time

Mean step time over steps 11-60 (the first 10 left out: start-up):

| | run a | run b | mean |
| --- | --- | --- | --- |
| old | 134.65 ms | 134.10 ms | 134.38 ms |
| new | 127.99 ms | 127.06 ms | 127.53 ms |

- **6.85 ms faster per step, -5.1%.** Two runs of the same code differ by 0.55 ms (old) and 0.93 ms (new), so the
  gain is far above the noise.
- Why 6.85 ms and not the ~18 ms of waiting: while the CPU waited, the GPU was mostly busy with that same matrix's
  Newton-Schulz anyway. The step only got shorter by the time the GPU sat **idle** after each wait, while the CPU was
  queuing the next matrix's work. The unit test without Newton-Schulz shows the same size: 10.26 ms -> 2.97 ms for the
  guard + ratio + update of 222 matrices, a 7.3 ms difference.

### 10.6 Conclusion

The new path makes the same decisions (unit test: all 90 cases and the 222-matrix step), leaves the same weights
(bitwise in the unit test; within the trainer's own noise in the real runs), prints the same `nsrej` and `nsratio`,
removes all 26,640 syncs and 53,220 copies per 60 steps, and makes the step 5.1% faster. (Version 1; the user then
decided to remove the flag and the old method: section 9 and 10.8-10.11.)

### 10.7 Notes for re-running on a new box

- The trainer needs `yaml-cpp`: `apt-get install -y libyaml-cpp-dev` (a fresh box does not have it; the trainer build
  stops at `ConfigBridge.h: fatal error: yaml-cpp/yaml.h`).
- The unit test links with `-ltensor -lcuda` (TI and BluBLAS call CUDA driver functions) and runs with
  `LD_LIBRARY_PATH=$(cd BluTrain && make -s print_run_ld_path SM_ARCH=90 | tail -n 1)`, because `libtensor.so` itself
  does not say where `libblublas.so` is.
- Check the GPUs first: `run_sft.sh` stops only for another `bluscript` run or < 60 GB free, not for other people's
  jobs (here a colleague's `ncu` on GPU 0).

### 10.8 Version 2 (old check removed everywhere) on 1 GPU, 2x H200 box, GPU 1

Same box, same data and settings as 10.1. The box's code was set back to the pushed files, then the two version-2
patches applied (`git diff --stat` and the patch md5 identical to the laptop), library and trainer rebuilt (trainer md5
`98b6481e`). Checked in the built library: `muon_has_nonfinite` 0 symbols (gone), the 4 new functions present.
"Old" = this morning's old_a / old_b (the pushed code path).

- **Unit test version 2: ALL PASS.** The reference is now the old rule written out on the CPU (the old library
  function no longer exists), compared with the new ShardedMuon sequence AND with `muon_apply_unless_bad_cuda`
  (`nn::Muon`): 90 single-matrix cases, then the 222-matrix step with 3 broken matrices: rejected 3 / 3, max ratio
  1.0061849 / 1.0061849, **0 matrices with different weights** in both new functions. CPU time of the new path: 1.514 ms
  to queue the step + 1.342 ms for the one read (removed path: 10.252 ms).
- Trainer, 60 steps:

| pair | max train-loss difference | `nsrej` equal on every step | max `nsratio` difference | final weights |
| --- | --- | --- | --- | --- |
| old_a vs old_b (noise) | 0.000815 | yes | 0.001 | 298 of 372 differ |
| v2_a vs v2_b (noise) | 0.000676 | yes | 0.002 | 299 differ |
| old_a vs v2_a | 0.000972 | yes | 0.002 | 300 differ |
| old_b vs v2_b | 0.000788 | yes | 0.001 | 299 differ |

- Step time (steps 11-60): old 134.65 / 134.10 ms, v2 127.04 / 127.17 ms: **-7.27 ms, -5.4%**.
- Calls (tracer, 60 steps): `cudaStreamSynchronize` 28,794 -> 2,154, `cudaMemcpyAsync` 90,402 -> 37,182, exactly as
  version 1; `ShardedMuon::update_one` appears in one entry only (the one 3,072-byte scratch allocation per run).

### 10.9 Version 2 on several GPUs (P032, 8x H200)

Two builds side by side in `/localstorage/gautam_k/muon_guard_p032`: `bt_old` = the pushed code (7184fe1 / d18f820 /
297f372, trainer `98574f5f`, library still has `muon_has_nonfinite`), `bt_new` = the same + the version-2 patches
(trainer `c525a1e5`, the old function gone). Same model, data and start weights as 10.1; 60 steps; mpirun core binding
off. Unit test version 2 on P032 GPU 0: **ALL PASS** (same numbers as 10.8).

| test | GPUs | settings | runs | result |
| --- | --- | --- | --- | --- |
| NCCL (default) | 0, 2 (2 ranks, B 4) | none | old_a, new_a, old_b | all 60 steps, 0 errors. Noise old_a vs old_b: loss 0.000566, nsratio 0.006; **old_a vs new_a: loss 0.000822, nsrej equal, nsratio 0.005** |
| BluCCL, pipelined publish | 4, 7 (2 ranks, B 4) | `BLU_BACKEND=bluccl` | old, new | 60 steps each, 0 errors, nobody else on 4/7. **old vs new: loss 0.000625, nsrej equal, nsratio 0.001** |
| BluCCL, fused publish kernel | 4, 7 | `BLU_BACKEND=bluccl BLU_FUSED_MUON_PUBLISH=1` | old, new | 60 steps each, 0 errors, the fused path not refused. **old vs new: loss 0.000794, nsrej equal, nsratio 0.001** |
| cross-checks | 4, 7 | | | old pipelined vs old fused: 0.000671 (how much the two paths differ anyway); new pipelined vs new fused: 0.000734 |

- Every run printed `reduce-scatter/all-gather layout`: this trainer's NCCL default also uses the reduce-scatter /
  all-gather publish (not a broadcast per matrix, as I first wrote).
- No matrix was rejected in any run (`nsrej 0/0`), as on 1 GPU.
- Step time of this first attempt (NOT clean: other jobs on the node; replaced by 10.12): BluCCL pipelined old 72.4 ms
  -> new 67.8 ms. Fused: old
  73.3 ms, new 75.2 ms, but the new run had two bursts of very slow steps (150-262 ms at steps 41-46 and 51-55) while the
  rest ran at ~68 ms (faster than old's 72-73 ms); at that time other people's jobs ran at 100% on GPUs 1, 2 and 5.
  Single runs on a shared node; not repeated (the fused path is off by default).

### 10.10 Pipelined and fused publish: what they are

On several GPUs (ZeRO), each Muon matrix is owned by one GPU, which updates it; afterwards every GPU needs the new
values of all matrices for the next forward. Sending them is the **publish** (an all-gather). The matrices are grouped
into buckets; per matrix: (1) apply `w = w - u * lr` on the fp32 master, (2) cast to bf16 into a send buffer, (3)
all-gather.

- **Pipelined publish** (default, `BLU_MUON_PIPELINE=1`, NCCL or BluCCL): the three steps are three separate GPU
  operations (apply kernel, cast kernel, all-gather). As soon as one bucket is updated and cast, its all-gather starts on
  the communication stream while the compute stream already runs Newton-Schulz for the next bucket. The guard flag is
  read by the apply kernel.
- **Fused publish** (`BLU_FUSED_MUON_PUBLISH=1`, BluCCL only, off by default): one BluCCL kernel per bucket does all
  three: reads u and w, computes `w - u * lr`, writes the master, and writes the bf16 value directly into every GPU's
  memory through the NVSwitch (multicast store). Fewer launches and memory passes; needs the NVSwitch (an 8-GPU node
  like P032). The guard flag is read inside that kernel.
- Both are in `ShardedMuon::update_owned`. On 1 GPU there is no publish.

### 10.11 Running next to other people's jobs (P032, 2026-11-04 16:20-16:30)

The first P032 attempt (2 GPUs NCCL on 0/2 and 4 GPUs BluCCL on 4-7 at the same time) overlapped with other jobs:
Aparna's `scaling_runs/runner.fused` (20 GB, then 92 GB) and later `runner.batched` started on GPU 6, Mahesh's SFT run
(`./bluscript_sft_bin`, 51-122 GB) on GPU 5, other `_jit_runner_` / `snippet_runner` jobs on GPUs 0, 1, 2.

- The 4-GPU BluCCL **new** run stopped at start-up with `ZeroConsistencyError: ranks disagree on total parameter count`
  on all 4 ranks, while GPU 6 was shared with Aparna's 92 GB job. That check runs before the first step: the ranks
  exchange their parameter counts over BluCCL and compare them; the change does not touch it. Rerun on GPUs 4 and 7 with
  nobody else on them: the same new build started and trained 60 steps without error. So the failure came from the
  shared GPU, not from the change.
- `run_sft.sh` refused the last run by itself (`ABORT: GPU 5 holds another training run`) when Mahesh's run had started
  there; it does not detect other kinds of jobs.
- Why overlapping is avoided: (1) runs can fail or hang (BluCCL uses the NVSwitch directly; two 135M runs on one GPU hung
  twice on 2026-10-28); (2) speed numbers are meaningless; (3) it slows the other person's job and spoils their
  measurements (Aparna's scaling runs measure speed). A run that finishes all 60 steps is still a valid correctness
  check (sharing slows the math but does not change it).
- The final script (`p032_final.sh`) checks before every run that no one else has a process on its GPUs and stops if
  someone does; a read-only watcher on the laptop checked every 20 s.

### 10.12 CLEAN rerun of every trainer test (P032, GPUs 4 and 5 only, 2026-11-04 17:00-17:32 IST)

Why: in the first P032 attempt (10.9, 10.11) other people's jobs ran on the node and partly on our GPUs, and our own two
test groups ran at the same time, so the 2-GPU step times were not clean. The user asked for a clean rerun of everything.

How (`p032_clean.sh`, 16 runs, **one at a time**, never two of ours together):
- GPUs 4 and 5 only (the 1-GPU test uses GPU 4); old = `bt_old` (pushed 7184fe1 / d18f820), new = `bt_new` (version 2
  = what is pushed now, 37514ff / d98d6ee). Same model, data, start weights, 60 steps as before; mpirun binding off.
- Each test in the order old_a, new_a, new_b, old_b (a drift of the node hits both sides equally; old-old and new-new give
  the noise).
- Before every run the script waited until no one else had a process on GPU 4 or 5 (it never had to wait).
- During every run a recorder listed every GPU process of the node about every 10 s (6-10 samples per run). Checked that
  the recorder works: it saw our own processes on GPU 4 (1-GPU runs) and on GPUs 4 and 5 (2-GPU runs) in every run, and
  **no other process on GPU 4 or 5 in any sample of any run**. Elsewhere on the node: `_jit_runner_` jobs (other GPUs) in
  every run, Aparna's `runner.fused` (GPUs 6, 7) during the first two runs only.
- A GPU lock (exclusive compute mode on GPUs 4/5 with sudo) was considered and NOT applied: Claude Code's permission
  check refused changing a shared node setting; it was not needed in the end (no one came onto GPUs 4/5).
- All 16 runs: 60 steps, 0 errors; the fused runs really ran the fused path (never refused).

Correctness (max train-loss difference over 60 steps; `nsrej` = rejected matrices per step; `nsratio` printed with 3
decimals):

| test | old_a vs old_b (noise) | new_a vs new_b (noise) | **old_a vs new_a** | **old_b vs new_b** | `nsrej` | `nsratio` diff |
| --- | --- | --- | --- | --- | --- | --- |
| 1 GPU | 0.000976 | 0.000777 | 0.001027 | 0.000700 | equal on every step in all pairs (0/0) | 0.001 in all pairs |
| 2 GPUs NCCL | 0.000995 | 0.000737 | 0.000703 | 0.000839 | equal | 0.001-0.002 |
| 2 GPUs BluCCL pipelined | 0.000803 | 0.000689 | 0.000535 | 0.000585 | equal | 0.001-0.003 |
| 2 GPUs BluCCL fused | 0.000763 | 0.000585 | 0.000821 | 0.000963 | equal | 0.001-0.003 |

Old vs new differs as much as old vs old and new vs new (0.0005-0.0010 in every column; the 1-GPU old_a vs new_a 0.001027
is at the top of the same range: the 2x box saw 0.001014 between two runs of the same code). Final weights differ in
297-299 of 372 tensors in every pair, same-code pairs included (the trainer is not bit-reproducible run to run); the bit
proof is the unit test.

Step time (median of steps 11-60, ms):

| test | old_a | new_a | new_b | old_b | old mean -> new mean | change |
| --- | --- | --- | --- | --- | --- | --- |
| 1 GPU | 137.2 | 128.7 | 128.1 | 139.1 | 138.2 -> 128.4 | **-9.8 ms, -7.1%** |
| 2 GPUs NCCL | 72.5 | 67.4 | 67.7 | 72.4 | 72.5 -> 67.6 | **-4.9 ms, -6.8%** |
| 2 GPUs BluCCL pipelined | 73.2 | 67.8 | 67.7 | 73.3 | 73.3 -> 67.8 | **-5.5 ms, -7.5%** |
| 2 GPUs BluCCL fused | 72.3 | 67.8 | 67.3 | 72.7 | 72.5 -> 67.6 | **-5.0 ms, -6.8%** |

- Two runs of the same code differ by 0.1-1.9 ms; every old -> new gain is 5-10 ms: far above the noise.
- The fused path is faster too (-6.8%): its 75.2 ms in the first attempt (10.9) came from the node load at that time, not
  from the change.
- 1 GPU on P032 gains more than on the 2x box (-7.1% vs -5.4%): a different machine (CPU, driver), so the CPU waits that
  were removed cost a different amount there.

**This table replaces the speed numbers of 10.9.** Run logs: P032 `/localstorage/gautam_k/muon_guard_p032/runs/c*`,
`samples/c*.txt`, `p032_clean.log`.

## 11. What the trace still shows after the change, and why

After version 2 the tracer still counts 2,154 `cudaStreamSynchronize` and 37,182 `cudaMemcpyAsync` in 60 steps. They
are NOT from the Muon guard (its 26,640 syncs and 53,280 copies are all gone); they come from other places, which are
the next items of the CUDA-call list (work doc 45.140). From the 2026-11-03 per-call-site accounting:

| calls (60 steps) | from | how often | item |
| --- | --- | --- | --- |
| ~865 syncs | `empty_cache` at steps 0 and 5 (a sync before every cudaFree of a cached block) | once per run | 10 |
| ~444 syncs | reading the checkpoint shards at start-up | once per run | per run |
| ~360 syncs | AdamW group (`ShardedOptimizer::step`, `local_adam_step`) | 6 per step | 5 |
| ~480 syncs | Muon all-gather, gradient-norm clip, reduce, step end | 8 per step | 5, 6, 7 |
| ~31,650 copies | GPU->GPU copies, one per parameter: `scatter_bucket_into_params` (222 per step), `stage_grads_direct_fp32` (150), `scatter_flat_into_params` (150); no CPU wait, only many small launches (~1.6 ms of CPU per step) | 527 per step | 3 |
| ~4,400 copies | start-up (weights to the GPU) and end (weights back to the CPU for export / checkpoint) | once per run | per run |
| ~1,400 copies | small uploads of tensor lists for the multi-tensor kernels | ~23 per step | 8 |
| 60 copies | the new per-step read of the two Muon counters | 1 per step | 1 |

So in normal training about 14 syncs per step are left (items 5-7), and the 527 per-parameter copies (item 3) are the
largest remaining count.

## 12. Push (2026-11-04)

- Branch `dist_side_gautam_lottn` in both repos (where the 11-03 BluBLAS / trained-rows work went). Order: TI first, then
  BluTrain with the new TI pointer.
- Tensor-Implementations: `include/ops/helpers/MuonKernels.h`, `src/Kernels/cuda/optimizer/MuonKernels.cu`,
  `src/nn/optimizer/Optim.cpp`. BluTrain: `dist/zero/include/ShardedMuon.h`, `dist/zero/src/ShardedMuon.cpp`,
  `dist/communication/include/ProcessGroupBase.h`, `dist/communication/BluCCL/kernels/fusions/fused_muon_publish.cuh`,
  `dist/communication/src/processGroupBluCCL.cu` + the `Tensor-Implementations` pointer. Nothing else (the local
  `dist/Context_Parallelism` pointer and `sft/*` changes are not part of this).
- Before the push the local changes were checked byte-identical to the tested patches (md5 of the diff = patch md5:
  TI `7b28f915...`, BluTrain `97fee8fc...`).
- Commands and commit messages: `profiling_reports/muon_gpu_guard_2026_11_04/push_1104/` (`commands.sh`,
  `msg_ti.txt`, `msg_blutrain.txt`). After the push: every file checked on the remote against the local file.
- Open after the push: a repeat of the fused-publish speed runs on a quiet node (10.9), optional.
- **Pushed 2026-11-04 (17:05-17:15 IST):** Tensor-Implementations `d18f820..d98d6ee`, BluTrain `7184fe1..37514ff` (the
  first TI push attempts failed with `Internal Server Error` / `fatal error in commit_refs_rules` during a GitHub
  incident, "Git Operations degraded" from 16:47 IST; the retry went through). **Verified on GitHub:** both branch heads
  are the new commits; author Goutham Kumar <goutham.kumar@blubridge.com>; the commits change exactly the 3 TI files and
  the 5 BluTrain files + the TI pointer; the TI pointer in BluTrain 37514ff = d98d6ee; all 8 files on GitHub are
  byte-identical to the tested local files (md5: MuonKernels.h 68592bfd, MuonKernels.cu 551d53a9, Optim.cpp 38e0eb28,
  ShardedMuon.h e85e161d, ShardedMuon.cpp b524f26e, ProcessGroupBase.h 549f8d6c, fused_muon_publish.cuh 4b41503b,
  processGroupBluCCL.cu 645677da); `muon_has_nonfinite_cuda` and `BLU_MUON_GPU_GUARD` are gone from the pushed code.

## 13. After this change: the other two Muon-step changes of 2026-11-04, and all three measured together

The Muon guard was item 1 of the CUDA-call list (SFT work doc 45.140). Two more changes followed the same day:
- **empty_cache at step 0 only** (item 10): the trainers emptied the allocator cache at steps 0 and 5; step 5 only forced
  a slow refill. Now step 0 only, old block kept as a comment. **Pushed: BluTrain `e9e310c`** (7 trainers). Work doc 45.145.
- **Batched Newton-Schulz**: Newton-Schulz ran once per matrix (222 matrices, about 28 small kernels each); now the
  matrices of the same shape run as one batch (6 groups of 37 on the 135M model; memory cap `BLU_MUON_NS_BATCH_MB`,
  default 512 MB, 0 = off). It needed a per-batch bias mode in BluBLAS addmm, and testing that found and fixed old addmm
  bias bugs (the fallback computed `alpha*A@B + beta*C_old + bias`). **Tested, not pushed.** Work doc 45.146-45.151.

Before/after of all three (P032, SFT trainer, 135M, 1 H200, B=4, 60 steps, median step time over steps 11-59, each
pair on the same GPU; work doc 45.152):

| change | GPU | before (ms) | after (ms) | change |
| --- | --- | --- | --- | --- |
| all three (11-03 code -> new code) | 0 | 137.41 | 110.83 | -19.3% |
| all three (11-03 code -> new code) | 1 | 137.01 | 109.79 | -19.9% |
| Muon GPU guard (this document) | 4 | 138.05 | 128.48 | -6.9% |
| empty_cache step 0 only | 2 | 128.15 | 128.69 | same (step 6: 2357 -> 128 ms, once) |
| empty_cache step 0 only | 3 | 129.59 | 129.14 | same (step 6: 3308 -> 127 ms, once) |
| batched Newton-Schulz + addmm fix | 5 | 128.87 | 110.02 | -14.6% |

Losses: old vs new differ by at most 0.00084, the same build on two GPUs by up to 0.00097 (noise). nsrej equal on every
step. The pretraining before/after is planned for 2026-11-05 (no pretraining data on P032).

## 14. Next (2026-11-05)

Fix 4 old BluBLAS GEMM bugs the addmm tests exposed (work doc 45.151), push the batched Newton-Schulz + addmm fix,
measure pretraining before/after, then the remaining CUDA-call items, then the context-parallel wrappers (45.153).
