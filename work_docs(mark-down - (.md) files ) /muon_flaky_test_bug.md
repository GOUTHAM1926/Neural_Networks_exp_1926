# The flaky Muon test (`test_muon_rsag`): what broke, why, and how it was proven

- **Date:** 2026-10-05.
- **Machine:** H200 box (`ssh -p 2224`), GPU 2 only.
- **Code:** BluTrain branch `dist_side_merge_BLUCCL` at `b9b567f`, with Tensor-Implementations (TI) at `a60099b`.
- **Build:** cuBLAS build (`WITH_CUBLAS=1 SM_ARCH=90`).
- **Repos:** nothing was changed in any repo. Every experiment ran on a scratch copy outside the repo.

---

## 0. The answer in five sentences

1. The test runs the same Muon optimizer in two ways and demands the two results be **identical to the last bit**. Sometimes they are not.
2. The difference starts in **one line of the TI library**: `atomicAdd(acc, thread_sum)` at `src/Kernels/cuda/optimizer/MuonKernels.cu:77`. That line adds up the pieces of a sum **in whatever order the GPU finishes them**. With computer numbers, a different order can give a total that is 1 bit different.
3. For one tensor (number 15, 96×96) at step 4, 1 of the 3 possible totals changes the scale used by Newton-Schulz. That 1-bit change grows to 7,077 different numbers in the final weights, at most 2.7e-5 apart.
4. **Navin's Muon code is correct.** Every value that goes into the update is bit-identical in both modes. His commit `a3aba3e` only changes how busy and how warm the GPU is just before that sum. That changes the finishing order, so the bad total shows up more often.
5. The fix belongs in the library: add the pieces **in one fixed order**. Only loosening the test would hide the problem. The chosen fix (**V1n**, Navin's own two-step pattern) is measured and written: see section 12.

---

## 1. Words you need (explained simply)

| Word | What it means here |
|---|---|
| **GPU** | A chip with thousands of small calculators that work at the same time. |
| **Kernel** | One small program that the GPU runs on many calculators at once. |
| **Block** | A team of 256 GPU threads (workers). A kernel is split into blocks. Blocks run in parallel and finish in any order. |
| **SM** | One "room" on the GPU where blocks run. The H200 has 132 rooms. |
| **L2 cache** | A small, fast memory on the GPU. If data is already there ("warm"), reading it is fast. If not ("cold"), reading takes longer. |
| **float (fp32)** | How the computer stores a decimal number: about 7 correct digits, then it rounds. |
| **bf16** | A smaller number format: only about 3 correct digits. It is used to make the math faster. |
| **bit** | The smallest piece of a stored number. "Differ by 1 bit" = the smallest possible difference. |
| **Rounding** | Cutting a number to the digits the format can keep. 46.8725519 becomes 46.87255 in float. |
| **`atomicAdd`** | A GPU instruction: "add my number into this shared box". Many workers can use it safely, but **the order they arrive in is not fixed**. |
| **Deterministic** | Same input → same output, every time, to the last bit. |
| **Weights / gradients** | Weights are the numbers the model learns. Gradients say which way to move them. |
| **Optimizer** | The code that moves the weights using the gradients. |
| **Muon** | One kind of optimizer. For each weight matrix it takes the momentum (a running average of gradients) and "straightens" it with Newton-Schulz before applying it. |
| **Newton-Schulz (NS)** | A 5-round recipe of matrix multiplications that makes a matrix "orthogonal". Before it starts, the matrix is divided by its size (the Frobenius norm). Here it runs in bf16. |
| **Frobenius norm** | The "size" of a matrix: square every number, add them all up, take the square root. |
| **Fingerprint (hash)** | A 16-character code computed from all the bytes of a buffer. If even 1 bit changes, the code changes completely. We use it to ask "are these two buffers exactly equal?". |
| **W** | The number of GPUs (ranks) taking part. This test runs with W=1. |
| **Collective mode** | How ShardedMuon moves data between GPUs. There are two modes, explained in section 2. |
| **Commit** | One saved change in git, named by a short code like `a3aba3e`. |

---

## 2. What the test does

The file is `dist/zero/tests/test_muon_rsag.cpp`.

ShardedMuon can move data between GPUs in two ways (`ShardedMuon.h:167`: `enum class MuonCollective { REDUCE_SCATTER_ALL_GATHER, REDUCE_BROADCAST }`, so in the logs **mode 0 = RS/AG and mode 1 = REDUCE_BROADCAST**):

- **REDUCE_BROADCAST (RB):** one reduce and one broadcast per parameter. The original way.
- **REDUCE_SCATTER_ALL_GATHER (RS/AG):** tensors are packed into one flat buffer, then one reduce-scatter and one all-gather per bucket. This way is faster on NVLink, but a wrong offset would silently mix up tensors.

The test's job is to catch a wrong offset. So it runs the **same** training twice, once per mode, and compares every final weight.

- 22 tensors (`kShapes`, 20 matrices + 2 vectors) = 82,120 numbers.
- 4 optimizer steps (`kSteps = 4`, line 324).
- 2 scenarios: **sync publish** and **staged refresh** (two ways of putting the new weights back).
- For each scenario: run RB first, then RS/AG (lines 332-333), then compare.
- At W ≤ 2 the rule is **bit-exact**: any difference at all = FAIL.

The test explains its own reason (lines 26-31):

> "Both modes average the same gradients and run the same Newton-Schulz on the same FP32 master, so at W<=2 there is no reordering anywhere that could legitimately change a bit."

**This sentence is the assumption that turns out false.** There is a reordering. It is inside the library's Frobenius norm (section 4).

A simple picture: two cooks bake the same cake with the same recipe, in two different kitchen layouts. The judge says the cakes must match to the last crumb. But one recipe step says "add the helpers' measurements in whatever order they arrive". Neither cook is wrong. That step is just not exactly repeatable, and the judge's rule assumed it was.

---

## 3. The symptom

A failing run prints this (always the same numbers):

```
[staged refresh] FAIL 7077/82120 elements differ | max_abs 2.679e-05 max_rel 1.719e-02
                 (worst tensor 15 elem 7659: 0.33138907 vs 0.331362277)
```

- **Always tensor 15**, the 96×96 matrix (`kShapes[15] = {96, 96}`).
- 7,077 of its 9,216 numbers differ. All other tensors match exactly.
- **Biggest difference:** 0.33138907 (RB) vs 0.331362277 (RS/AG), which is 2.7e-5.
- `max_rel` 1.7e-2 is a relative difference measured on a weight whose value is small.
- **How often:** on HEAD, 17 of 20 runs had at least one scenario FAIL. Which scenario fails changes from run to run. The same binary sometimes passes. That is why it is called **flaky**.

---

## 4. The cause, explained like you are 10

### 4.1 Order matters when your calculator keeps few digits

Imagine a calculator that can only show **3 digits**. It rounds everything else.

- `(1000 + 4) + 4`: first 1000 + 4 = 1004, which rounds to **1000**. Then 1000 + 4 = 1004, which rounds to **1000** again.
- `1000 + (4 + 4)`: first 4 + 4 = 8. Then 1000 + 8 = 1008, which rounds to **1010**.

Same three numbers, different order, different answer. Real computer floats keep about 7 digits, not 3, but the same thing happens in the 7th digit, which is the last bit.

### 4.2 Nine workers shouting their part to one box

To get the size of tensor 15, the library must add up 9,216 squared numbers. It splits the job across **9 blocks**:

```
need4      = (numel/4 + 255) / 256 = (9216/4 + 255) / 256 = 9      (MuonKernels.cu, inside muon_frobenius_scale_cuda)
red_blocks = min(blocks-per-SM × 132, need4) = 9
```

Each block adds its own part. Then each block does `atomicAdd(acc, its_part)` into one shared box (`MuonKernels.cu:77`). The box starts at 0. The adds happen **in the order the blocks finish**, and the GPU does not promise any order.

It is like 9 kids counting marbles in 9 bags. Each kid shouts their count to the teacher as soon as they finish. The teacher types each number into the 7-digit calculator in the order they hear it.

### 4.3 Our real numbers

The 9 block parts for tensor 15 at step 4 (from `ns_probe5`, run on the GPU, then recomputed on the laptop with the same result):

```
4.99930286  5.05442524  5.3774209  5.29176378  5.1028614  5.25180912  4.82695436  5.43447971  5.53353453
```

There are 9! = 362,880 possible orders to add them. Trying **all** of them gives only **3 different totals**:

| Total (float) | Bits | Number of orders that give it | One example order (block numbers) | Effect |
|---|---|---|---|---|
| 46.872547 | `423b7d7d` | 58,600 | 8,7,6,5,4,3,2,1,0 | normal result |
| 46.87255 | `423b7d7e` | 274,544 | 0,1,2,3,4,5,6,7,8 | normal result |
| 46.872555 | `423b7d7f` | **29,736** (8.2%) | 0,1,3,5,2,6,7,4,8 | **the odd result** |

The mathematically exact sum is 46.8725519…. The first two totals give the same scale, `1/√total = 0.14606316`. The third gives `0.14606315`, **1 bit smaller**. Only that third total changes anything downstream.

### 4.4 How 1 bit becomes 7,077 different numbers

1. Every number in the matrix is multiplied by the scale. With the odd scale, many of them come out 1 bit different.
2. The matrix is cut down to **bf16** (about 3 digits). Most numbers still round to the same bf16 value. But a number sitting right at a halfway point now rounds the other way. So **some bf16 numbers flip**.
3. Newton-Schulz then does 5 rounds of matrix multiplication. In a matrix multiplication, every output number mixes a whole row with a whole column, so one flipped input touches many outputs. After 5 rounds, thousands of numbers differ a little.
4. The update is multiplied by the learning rate and applied to the weights. **Final result:** 7,077 weights differ, by at most 2.7e-5.

The fingerprints show exactly this chain. In a failing run, the input to the scale step is identical, the scaled matrix differs, its bf16 copy differs, and every Newton-Schulz round after that differs (section 6, test T9).

### 4.5 Why it shows up only in REDUCE_BROADCAST mode, and only after Navin's commit

The order the 9 blocks finish depends on the **state of the GPU** at that moment:

- Is the data already in the L2 cache (warm) or not (cold)?
- Are some SMs still busy with other work?

The two modes run different copy and collective code just before the update, so they leave the GPU in a different state. Navin's commit `a3aba3e` changed that code for W=1 (in-place buffers and new kernels). After it, the state that RB mode leaves behind makes the "odd" order happen often.

We proved that GPU state alone flips the result, with no Muon code at all (test T11). The real library function on the same input gave the odd result:

- 0 of 200 times on an idle GPU;
- 16 of 200 times when the L2 cache was flushed first;
- 38 of 200 times when other work kept a changing number of SMs busy.

**Not found:** which exact line of `a3aba3e` makes RB mode's state "bad". Switching back any one of his three changes still failed (test T4). This is expected: any change in what runs before can move the order. It is not a logic switch.

---

## 5. The exact code

### 5.1 The line that adds in random order

`Tensor-Implementations/src/Kernels/cuda/optimizer/MuonKernels.cu` (TI `a60099b`):

```cpp
// :22
__global__ void muon_sumsq_kernel(const float* __restrict__ x, float* __restrict__ acc, int64_t numel) {
    ...                                   // each thread squares and adds its numbers (float4 loads)
    thread_sum = warpReduceSum(thread_sum);   // fixed order inside a warp
    ...                                   // fixed order inside the block
        if (tid == 0) {
            atomicAdd(acc, thread_sum);   // :77  <-- one add per block, in the order blocks finish
        }
}

// :123
void muon_frobenius_scale_cuda(float* x, int64_t numel, float eps) {
    if (!s_muon_norm) cudaMalloc(&s_muon_norm, sizeof(float));
    cudaMemsetAsync(s_muon_norm, 0, sizeof(float), stream);          // :128  box = 0
    const int64_t need4 = (numel / 4 + threads - 1) / threads;
    int red_blocks = max(1, min(nbps_red * sm_count, need4));         // = 9 for 96x96
    muon_sumsq_kernel<<<red_blocks, threads, 0, stream>>>(x, s_muon_norm, numel);   // :161
    ...                                   // then muon_scale_by_inv_frob_kernel: x *= rsqrtf(*acc)
}
```

Everything **inside** a block is added in a fixed order: the warp shuffle and then the shared-memory step. Only the **across-blocks** step is unordered. Any tensor that needs more than 1 block can be affected: that is any tensor with more than 1,024 numbers.

### 5.2 How the test reaches it

```
ShardedMuon::update_one()                         dist/zero/src/ShardedMuon.cpp
  g32 = owned_grad_fp32(...)                      gradient, fp32
  muon_momentum_cuda(m, g32, g, ...)              momentum -> g
  newton_schulz_orthogonalize(g, 5, 1e-8, true)   :1642
    -> muon_frobenius_scale_cuda(X, numel, eps)   TI Optim.cpp:1548
         -> muon_sumsq_kernel ... atomicAdd       TI MuonKernels.cu:77   <-- HERE
    -> X.as_type(bf16), 5 rounds of cuda_matmul / cuda_addmm
  muon_apply_cuda(work, up, lr * adjust, n)       new weights
```

---

## 6. How it was proven, test by test

Each test answers one question. All of them ran on GPU 2, on scratch copies, with the same `libtensor`/`libdist` build.

### T1. Is the bug still there today?

- **What:** built `test_muon_rsag` from the latest code (HEAD `b9b567f`) and ran it 20 times.
- **Result:** **17 of 20 runs FAILED**, always with the same signature (7,077/82,120, tensor 15 elem 7659).
- **Meaning:** yes, it is real and frequent. It is a cuBLAS build, so it is not caused by BluBLAS.

### T2. Which commit started it?

Between the last passing commit and HEAD, two commits touch `dist/zero`:

```
149249d  zero: at W == 1 the gradient no longer goes through the wire buffer (F4)   <- ours
a3aba3e  Zero memory buffer inplace change                                         <- Navin (bruce-yeager), 2026-09-30
e6eaeba  (before both)
```

| Code built | Runs that FAILED |
|---|---|
| e6eaeba | **0 of 20**, then 0 of 10 again |
| a3aba3e alone (no F4) | **7 of 10** |
| HEAD (a3aba3e + F4) | 17 of 20, then 10 of 10 |

- **Meaning:** it starts at `a3aba3e`. Our F4 is not needed for it to fail.

### T3. Is it a timing race between GPU launches?

- **What:** ran HEAD with `CUDA_LAUNCH_BLOCKING=1`, which makes every GPU launch wait for the previous one to finish.
- **Result:** still **17 of 20** failed.
- **Meaning:** it is not two of our launches overlapping. (The real cause, the order *inside* one kernel, is not affected by this setting.)

### T4. Which of Navin's three changes?

Took HEAD and switched back **one** change at a time:

| Variant | What was switched back | Runs that FAILED |
|---|---|---|
| S1 | gradient up-cast: new `cast_to_f32_scaled` → old convert + scale | 1 of 10 |
| S2 | W=1 gradient norm: new `sumsq_accumulate` → old `multi_tensor_grad_norm` + sync | 1 of 10 |
| S3 | W=1 buffers: in place → separate buffers again | 4 of 10 |

- **Meaning:** none of them alone is "the switch". S2 also rules out the earlier idea about a missing sync and a pinned buffer.
- **Note:** these three ran at the same time on GPU 2, which itself changes GPU state. That fits section 4.5.

### T5. Do the CUDA sanitizers find a memory bug?

| Tool | Result | What it means |
|---|---|---|
| memcheck | 0 errors | No out-of-bounds reads or writes. |
| initcheck | 18,560 "uninitialized read" reports in `broadcast_scale_kernel` (from `cuda_addmm`) | **False alarm for this bug.** `ns_probe3` filled every Newton-Schulz buffer with garbage first and still got identical outputs. |
| racecheck | 20 **Warnings** (not Errors) of shared-memory hazards inside TI's `matmul_bf16_optimized<128,128,32,32,32>` | **Not this bug.** In more than 12,000 calls during these tests, the same input always gave the same matmul output (T8, T9). It is listed as a separate item in section 9. |

### T6. Where exactly do the two modes first differ?

- **What:** a debug copy of HEAD's `ShardedMuon.cpp` (`run_debug.sh`). For tensor 15, at every step, in both modes, it prints a fingerprint of each stage:
  - **W0:** weights before;
  - **G:** gradient;
  - **M:** momentum;
  - **NSIN:** Newton-Schulz input;
  - **NSOUT:** Newton-Schulz output;
  - **W1:** weights after.
- **Result:** 10 runs, 12 failing comparisons. **Every one first differs at step 4, stage NSOUT.** W0, G, M and NSIN were identical in every run, at every step, in both modes.
- **Meaning:** Navin's code feeds Newton-Schulz exactly the same input in both modes. The difference is born **inside** Newton-Schulz.

### T7. Same check on all three commits

- **What:** `run_debug2.sh`, 10 runs each of HEAD, a3aba3e and e6eaeba.
- **Result:**
  - The NS **input** fingerprints are **identical across all three commits** at every step (step 4: `6b6023cd51360133`).
  - The NS **output** at step 4 only ever takes 2 values: `57bf40c3114993ed` (normal) or `0ce46e2b51fb0c33` (odd).

| Commit | Odd output in RB mode | Odd output in RS/AG mode |
|---|---|---|
| HEAD | 17 of 20 | **0 of 20** |
| a3aba3e | 11 of 20 | **0 of 20** |
| e6eaeba | 0 of 20 | 0 of 20 |

- **Meaning:** the old and new code give the optimizer exactly the same numbers. Only RB mode, and only from `a3aba3e`, gets the odd output.

### T8. Is Newton-Schulz itself repeatable on that exact input?

- **What:** `ns_probe4` loaded the dumped step 1-4 inputs (all 3 commits) and ran 200 calls each of: the Frobenius scale alone, the full library Newton-Schulz, and the same Newton-Schulz with the sum done in one fixed order.
- **Result:** **always 1 result per input**, 200 of 200. The full NS gave the normal value `57bf40…` at step 4.
- **Meaning:** on a quiet GPU, in a fresh program, the order of the 9 blocks happens to stay the same. So this test alone could not show the problem. T11 shows what changes when the GPU is not quiet.

### T9. Which operation inside Newton-Schulz goes first?

- **What:** `run_debug3.sh` replaced the library call for tensor 15 with a line-for-line copy of the library's bf16 Newton-Schulz. It prints a fingerprint and the memory address after **every** operation.
- **Result:** 6 runs, 8 failing comparisons. **Every one first differs at the operation "scaled"**, the output of `muon_frobenius_scale_cuda`:

```
                 in        scaled     Xb0 (bf16)  ... 15 matmul steps ...  out
RB (odd)   :  6b6023 ->  309adc  ->  1f60a3   ->  (all differ)        -> 0ce46e
RS/AG      :  6b6023 ->  b3f448  ->  4e5432   ->  (all differ)        -> 57bf40
```

- Same input, **same memory address** (`0x526200000` in both modes), different scaled result.
- **Meaning:** the split happens in the Frobenius-norm step. Everything after it is only the knock-on effect.

### T10. Is the odd value exactly "another add order"?

- **What:** `ns_probe5` part 1 copies the library's two Frobenius kernels **verbatim** from `MuonKernels.cu` at build time. One copy stores each block's part instead of using `atomicAdd`. The probe then adds the 9 parts in all 362,880 orders, the way `atomicAdd` does (one float add at a time onto 0). For each distinct total it runs the verbatim scale kernel.
- **Result:** 3 totals (the table in 4.3):
  - `…7d` and `…7e` give scaled `b3f448f1a3cf2a55`, **exactly the normal value**;
  - `…7f` gives `309adc60d0b89f1f`, **exactly the odd RB value from the test**.
- **Meaning:** the odd value in the test is precisely "the 9 blocks were added in a different order". Nothing else is needed to explain it.

### T11. Can we make it happen outside the test?

- **What:** `ns_probe5` part 2: the **real library function** `muon_frobenius_scale_cuda`, on the same input, 200 calls per condition.

| Condition | Odd result `309adc…` |
|---|---|
| idle GPU | 0 of 200 |
| L2 cache flushed before each call | **16 of 200** |
| all but 4 SMs busy with other work | 0 of 200 |
| a changing number of SMs (1 to 132) busy | **38 of 200** |

- **Meaning:** no Muon code and no test are involved here. The GPU's state alone decides the order, and the order decides the result. This is the final proof.

---

## 7. Ideas that were checked and ruled out

| Idea | Why it is ruled out |
|---|---|
| A missing sync / pinned-buffer race in Navin's norm rewrite (the 2026-09-30 lead) | S2 put the old norm back and it still failed (T4). The NS input is identical anyway (T6, T7). |
| Our F4 change caused it | a3aba3e alone fails 7/10 (T2), and the NS input is identical with and without F4 (T7). |
| Two GPU launches overlapping (timing race) | Still 17/20 with `CUDA_LAUNCH_BLOCKING=1` (T3). |
| Uninitialized memory (initcheck reports) | Garbage-filled NS buffers give identical outputs (`ns_probe3`). |
| Uninitialized class members in ShardedMuon | The new members are initialized in the header (`bool inplace_collectives_ = false; float* d_sumsq_partials_ = nullptr;`). |
| The matmul not writing its whole output | `ns_probe` part 1: every output entry is written and correct for all 11 tested shapes. |
| cuBLAS picking a different kernel because of buffer addresses | The address was the same in both modes (T9), and moving buffers did not change results (`ns_probe3` part B). |
| Wrong data reaching the optimizer (a layout bug) | Weights, gradient, momentum and NS input are bit-identical in both modes (T6). The test's layout checks pass. |

---

## 8. What it means

**Is Navin's Muon code wrong?** No. It gives Newton-Schulz exactly the same input in both modes (T6, T7). His commit only changes the GPU's state before the sum. That makes an existing non-repeatable step show up.

**Is the test wrong?** Its rule ("identical to the last bit") is a good rule. It is the right way to catch silent layout bugs. But its stated assumption ("no reordering anywhere", lines 26-31) is false while the library's Frobenius norm uses an unordered `atomicAdd`. So today the test can fail even when ShardedMuon is perfect.

**Does it hurt training?** The math is not wrong. The difference is the size of normal rounding noise (a 1-bit change in the scale). Two facts matter:

- **Training is not exactly repeatable.** In principle, two runs with the same seed can end with slightly different weights. This was not measured in the trainer.
- **No rank-to-rank drift.** Each Muon tensor's Newton-Schulz runs on **one** owner rank, and the result is sent to the others ("the SAME whole-parameter ownership", test header). So all GPUs still hold the same weights.

**Why this matters beyond the test:** a test that fails randomly gets ignored. Then a real layout bug, which is what the test exists to catch, would also be ignored.

---

## 9. How to fix it

**Make the Frobenius sum add in one fixed order.** This is a library change in `MuonKernels.cu`.

The first idea was to do the fixed-order add inside the scale kernel (that is "V3" in section 12). After measuring five versions, the chosen fix is **V1n**: each block writes its own slot, one small block adds the slots in a fixed order, and the number of blocks depends only on the matrix size. It is the same two-step pattern Navin already uses in `dist/zero/src/ZeroKernels.cu`. The design, the measurements and the patch are in **section 12**.

**Not recommended on its own:** giving the test a tolerance at W=1. The test would pass, but it would also become blind to small real layout bugs, and training would still not be repeatable.

**Who owns it:** the file is in Tensor-Implementations. Its last commit is `927d0a9` (kathir-23s, 2026-08-13); this does not say who wrote line 77. Navin should know the result, because his test is the one that fails.

---

## 10. Still open / not checked

- **Which exact part of `a3aba3e`** puts RB mode in the "bad" GPU state. Not found. No single change is the switch (T4).
- **Why only tensor 15 and only step 4.** Only step 4's input was put through the all-orders check (T10). For other tensors and steps, either no order changes the scale, or the change flips no bf16 rounding. Not checked one by one.
- ~~Other unordered float sums in the library.~~ **Checked (section 12.4):** the optimizer path has no other live one.
- **The racecheck warnings** in TI's `matmul_bf16_optimized<128,128,32,32,32>` (T5). They never changed a result here, but nobody has looked at them. Worth a separate look.
- **W=2.** Not run. At W=2 the test is also bit-exact, so the same flakiness is expected but not measured.

---

## 11. How to run it again, and where everything is

**Scripts and probes on the laptop:** `Neural_Networks_exp_1926/profiling_reports/muon_bug_2026_10_05/`.

| File | What it does |
|---|---|
| `run_rsag_arm.sh <commit> [runs]` | Builds the test against one commit's `dist/zero` (via `git archive`, outside the repo), runs it N times (T1, T2, T3). |
| `run_variants.sh [runs]` | The S1/S2/S3 switch-back variants (T4). |
| `run_debug.sh [runs]` | Stage fingerprints, HEAD only (T6). |
| `run_debug2.sh [runs]` | Stage fingerprints for HEAD, a3aba3e and e6eaeba, plus dumps the NS input; then runs `ns_probe4` (T7, T8). |
| `run_debug3.sh [runs]` | Line-for-line NS copy with a fingerprint after every operation (T9). |
| `run_probe5.sh` | Builds `ns_probe5` with the library kernels copied verbatim; all add orders + real-function conditions (T10, T11). |
| `ns_probe.cu`, `ns_probe2.cu`, `ns_probe3.cu` | Earlier checks: matmul writes all of A; initcheck buffer map; garbage-filled buffers. |
| `run_probe6.sh` + `ns_probe6.cu` | The five fix versions vs today's code: determinism, precision, speed (section 12.3). `ns_probe6 - <cap>` sets the numel-only cap. |
| `run_probe6_caps.sh` | The same with caps 256 / 512 / 768 (section 12.3). |
| `fix_v1n/patch_muon_v1n.py` | **The fix** (md5-guarded, 4 edits, each must match once). |
| `fix_v1n/muon_v1n.diff` | The resulting diff of `MuonKernels.cu`. |
| `fix_v1n/run_fix_v1n.sh` | Validation on a scratch copy of TI (section 12.6). |

**On the box:**
- Scripts and binaries are in `~/gautam_k/muon/`.
- The clone is `~/gautam_k/BluTrain_bluccl` (HEAD `b9b567f`, TI `a60099b`, libraries already built).
- Example: `cd ~/gautam_k/muon && bash run_debug2.sh 10`. It pins `CUDA_VISIBLE_DEVICES=2` itself.

**All logs** (pulled to the laptop): `profiling_reports/muon_bug_2026_10_05/box_logs/muon_logs_2026_10_05.tgz`, unpacked in `box_logs/extracted/`. They include the full sanitizer logs, every run's output, `probe4.txt`, `probe5.txt`, and the step-4 NS input `nsin_step4.bin`.

**The fingerprints quoted above:**

| What | Normal (RS/AG, idle GPU) | Odd (RB mode) |
|---|---|---|
| NS input, step 4 | `6b6023cd51360133` | `6b6023cd51360133` (same) |
| after Frobenius scale | `b3f448f1a3cf2a55` | `309adc60d0b89f1f` |
| NS output | `57bf40c3114993ed` | `0ce46e2b51fb0c33` |
| norm total (bits) | `423b7d7d` or `423b7d7e` | `423b7d7f` |

---

## 12. The fix: V1n (Navin's two-step pattern)

### 12.1 What it does, simply

**Before:** 9 kids shout their marble counts to one teacher. The teacher adds them in the order they are heard, so the order changes every time.

**After:** each kid writes their count on **their own numbered card** (card 0, card 1, card 2 …). Then one person adds card 0, card 1, card 2 … **always in the same order, the same way**. Who finished first no longer matters, so the total is the same every time.

The change is in `Tensor-Implementations/src/Kernels/cuda/optimizer/MuonKernels.cu` only. Nothing else in TI or dist uses these kernels or the static buffer.

1. **Sum kernel** (`muon_sumsq_kernel`): `atomicAdd(acc, thread_sum)` becomes `partials[blockIdx.x] = thread_sum`. Each block writes its own slot.
2. **New add-the-slots kernel** (`muon_sumsq_finalize_kernel`, 1 block): each thread adds a fixed share of the slots, then the same warp-and-shared-memory tree the sum kernel already uses. It writes the total with `=` (not `+=`), so the old `cudaMemsetAsync` that cleared the box is gone.
3. **Number of blocks** for the sum = `min(768, ceil(numel / 1024))`. This depends **only on the matrix size**. Before, it was `blocks-per-SM × SM count`, which depends on the GPU. The scale kernel is unchanged.

GPU steps per matrix: 3 before (clear, sum, scale) and 3 after (sum, add slots, scale).

### 12.2 The five versions compared, and why V1n

| Version | How the slots are added | GPU steps | Robust to later edits? |
|---|---|---|---|
| V0 (today) | `atomicAdd`, in finishing order | 3 | not repeatable |
| V1 | 1-block kernel, fixed order | 3 | yes |
| V2 | the last block to finish adds them (counter + memory fence) | 2 | **no**: if a block ever exits early, the counter stays wrong and the next call silently uses a stale norm |
| V3 | every scale block re-adds the slots itself | 2 | weaker: the scale kernel changes, and two kernels must agree on the block count |
| V3n | V3 + block count from matrix size only | 2 | same as V3 |
| **V1n** | V1 + block count from matrix size only | 3 | **yes**, and the same bits on any GPU model |

**Why V1n:**
- It is the pattern already used twice in our code: Navin's `sumsq_partial_kernel` + `sumsq_finalize_kernel` in `dist/zero/src/ZeroKernels.cu:131-182`, and TI's `multi_tensor_grad_norm` with its 1-block cleanup kernel. So it is one pattern everywhere.
- It has no hidden state, and the scale kernel is unchanged.
- The total is stored as one value (useful later for logging or NaN checks).
- It is the smallest diff.
- **Precision is identical** for V1/V2/V3, which add the same slots in the same order, so they give bit-identical totals. V2/V3 are only a few µs faster (12.3).

### 12.3 Measured (`ns_probe6`, H200, 2026-10-05)

**How it was measured:**
- The library's kernels were copied **verbatim** from `MuonKernels.cu` and built with the library's own flags (`--use_fast_math -O3`).
- The fixes were made from those copies by small asserted edits, so only the across-block step differs.
- Each version on each matrix: **400 calls** in 4 GPU conditions (idle, L2 cache flushed, most SMs busy, a changing number of SMs busy).
- **Exact reference:** the sum of squares in 64-bit `long double` on the CPU.
- **Speed:** median of 5 rounds of back-to-back calls, versions interleaved.
- This H200 runs **792** blocks of the sum kernel at once (6 per SM × 132 SMs).

**Determinism and precision** (V1n with cap 768; the first row is the real tensor-15 step-4 input from the test):

| Matrix | Today: different totals in 400 calls | Today: worst error | V1n: different totals | V1n: worst error |
|---|---|---|---|---|
| 96×96 real tensor 15 | 3 | 1.13 ulp | **1** | **0.13 ulp** |
| 96×96 | 3 | 1.28 ulp | **1** | 0.72 ulp |
| 768×768 | 20 | 9.76 ulp | **1** | 0.24 ulp |
| 768×3072 | 21 | 13.1 ulp | **1** | 0.94 ulp |
| 2048×2048 | 18 | 10.0 ulp | **1** | 0.04 ulp |
| 2048×8192 | 21 | 12.4 ulp | **1** | 0.60 ulp |
| 4096×4096 | 16 | 9.69 ulp | **1** | 0.31 ulp |
| 4096×14336 | 26 | 18.4 ulp | **1** | 0.40 ulp |

**What this shows:**
- Today's code is far more random on big matrices (16 to 26 different totals) than on our test tensor (3). So in real training, **every** Muon step of every large matrix had this randomness, not only the test's tensor.
- The fix gives one total every time, always within about 1 ulp of the exact answer, which is **10 to 45 times more accurate** in the worst case.
- On the real tensor 15 it gives `423b7d7e`: the float closest to the exact sum, and today's "normal" value.

**Speed,** µs per call (V1n uses cap 768):

| Matrix | V0 today | V1 | V2 | V3 | **V1n** | V1n − today |
|---|---|---|---|---|---|---|
| 96×96 | 6.41 | 6.70 | 5.72 | 5.04 | **6.70** | +0.3 |
| 768×768 | 8.19 | 8.61 | 7.79 | 7.35 | **8.62** | +0.4 |
| 768×3072 | 11.04 | 11.52 | 10.65 | 9.82 | **11.38** | +0.3 |
| 2048×2048 | 13.58 | 14.45 | 13.55 | 12.61 | **14.30** | +0.7 |
| 2048×8192 | 56.06 | 59.07 | 57.62 | 56.95 | **58.85** | +2.8 |
| 4096×4096 | 56.16 | 59.13 | 57.64 | 56.95 | **58.82** | +2.7 |
| 4096×14336 | 183.0 | 185.8 | 184.4 | 183.7 | **185.4** | +2.4 (+1.3%) |

**Why the cap is 768:** this H200 fits 792 sum blocks at once. With a cap of 1024, a second, mostly empty round was needed. Caps 256 and 512 left the GPU under-used on big matrices. Speed of V1n per cap, in µs:

| Matrix | cap 256 | cap 512 | **cap 768** | cap 1024 |
|---|---|---|---|---|
| 2048×8192 | 67.20 | 59.93 | **58.85** | 60.01 |
| 4096×4096 | 67.22 | 60.00 | **58.82** | 60.02 |
| 4096×14336 | 197.17 | 186.59 | **185.37** | 190.64 |

**What the speed cost means:**
- The fix adds +0.3 to +2.8 µs per Muon matrix per step. That is the small "add the slots" step replacing the old "clear the box" step.
- Example estimate (not measured end to end): GPT-2 small has about 48 Muon matrices, which is about 17 µs per training step. With ZeRO each GPU only handles its own matrices, so less.
- The matrix multiplications in Newton-Schulz take far longer than this step.

### 12.4 Other sums of the same kind (checked)

| Kernel | Used by | Status |
|---|---|---|
| `multi_tensor_grad_norm` (`MultiTensorKernels.cu`) | grad clipping (`Optim.cpp:629`, `:750`), ShardedMuon at W>1 | **safe**: block partials, then `grad_norm_cleanup_kernel<<<1,256>>>`; only one add per call, in stream order |
| `sumsq_accumulate` (`dist/zero/src/ZeroKernels.cu:131-182`, Navin) | ShardedMuon gradient norm at W=1 | **safe**: the same two-step pattern; block count from `n` only |
| `grad_norm_squared_kernel` (`GradNormKernels.cu:51`) | **nothing calls it** | same unsafe pattern, but dead code; left alone |
| **`muon_sumsq_kernel` (`MuonKernels.cu:77`)** | Newton-Schulz (`Optim.cpp:1548` bf16, `:1580` fp32) | **the only live one; fixed by V1n** |

### 12.5 Comments in the code

The patch documents itself in `MuonKernels.cu` (no author tags):
- above the add-the-slots kernel: why a fixed order, what went wrong with `atomicAdd` (the measured totals and the `test_muon_rsag` failure), and that it is the same pattern as `ZeroKernels.cu`;
- above `kMuonSumsqMaxBlocks`: why the block count depends on the matrix size only, and why 768 (with the measured numbers);
- on the static buffers: they are shared by every call, so calls must stay on one stream (as before);
- on the scale grid: why it can stay occupancy-based (it is elementwise).

### 12.6 Validation on the real library

`fix_v1n/run_fix_v1n.sh`, run on **GPU 1** (GPU 2 was taken by another job at 21:35):

1. Copy the clone's TI to a scratch folder (timestamps kept) and apply the patch (md5-guarded). Rebuild; the script refuses to start unless `make -n` shows that **only** `MuonKernels.cu` will compile.
2. Check the new kernel is in the fixed `libtensor.so` and not in the original, and that each run loads the intended library (`ldd`).
3. Run the **same** `test_muon_rsag` binary 20 times with the original library and 20 times with the fixed one, alternating. Only the library differs.
4. Put the real patched function through the stress conditions (`ns_probe5` part 2).
5. Run the library Newton-Schulz on the dumped step 1-4 inputs, 200 calls each, original vs fixed.

**Results (2026-10-05, 21:39-21:53, GPU 1):**

| Check | Original library | **Fixed library** |
|---|---|---|
| Rebuild (`make -n` before building) | n/a | exactly 1 file (`MuonKernels.cu`), then device link and `libtensor.so` link |
| New kernel present (`strings`, mangled name) | 0 | 21 |
| Library each run loads (`ldd`, binary uses RUNPATH) | clone's `Tensor-Implementations/lib` | scratch `fixroot/Tensor-Implementations/lib` |
| `test_muon_rsag`, same binary, 20 alternating rounds | **19 of 20 runs FAILED** | **0 of 20 FAILED** (40 of 40 scenarios "PASS bit-identical") |
| Real function under stress, 200 calls per condition: idle / L2 flushed / most SMs busy / varying SMs busy | odd value 0 / 16 / 0 / 38 times (T11) | **1 result in all 800 calls** (`b3f448f1a3cf2a55`, the normal value) |
| Library Newton-Schulz on the dumped step 1-4 inputs, 200 calls each | `3346bd7b…`, `ee3db3a1…`, `30da315e…`, `57bf40c3…` | **the same four values**, 1 result per input |

**What this proves:**
- With only the library changed, the flaky test goes from 19/20 failing to 0/20.
- The patched function gives one result under every GPU condition that used to flip it.
- On the test's own inputs, the fixed Newton-Schulz gives exactly the old "normal" outputs. So for these inputs the fix changes nothing except removing the random "odd" result.

**Not covered by this run:**
- W=2 (needs 2 GPUs).
- The single-GPU `nn::Muon` optimizer path (same function, not separately run).
- A full training run.

### 12.7 How to apply

**PUSHED 2026-10-05** (author Goutham Kumar <goutham.kumar@blubridge.com>, no co-author lines):
- Tensor-Implementations `dist_side_merge_BLUCCL`: `a60099b..a9bc08c`, "Muon: deterministic Frobenius norm (fixed-order sum, no atomicAdd)". It changes only `MuonKernels.cu` (+71 / −19). The committed file's md5 is `4e1d9393…`, the same as the tested file.
- BluTrain `dist_side_merge_BLUCCL`: `b9b567f..35df307`, "Point Tensor-Implementations to the Muon norm fix". It changes only the TI pointer (`a60099b` → `a9bc08c`); `dist/Context_Parallelism` is unchanged.

- **Patch:** `fix_v1n/patch_muon_v1n.py <path to MuonKernels.cu>`. It refuses unless the file's md5 is `f6d821c2c64cb1f39e04625e2a74ee47` (TI `a60099b`) and each of the 4 edits matches exactly once. The new md5 is `4e1d93931f0411b1d6b062fcbe5a0089`. The diff is `fix_v1n/muon_v1n.diff`.
- **Where:** commit to the TI branch `dist_side_merge_BLUCCL`. Then move the TI pointer in BluTrain `dist_side_merge_BLUCCL`. Colleagues run `git submodule update` and rebuild `libtensor`; only `MuonKernels.cu` recompiles.
- **Who to tell:** Navin (his test now has a true assumption) and Kathir (last committer of the file).
- **What people will notice:** Muon weights change in their last bits once, compared to before. After that, runs repeat exactly.
