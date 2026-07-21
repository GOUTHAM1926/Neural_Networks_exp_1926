# BluBridge-BLAS on H200 (Hopper) — analysis + build log

Companion to `analyzations_b300.md` (all prior B300/Blackwell/SM103 work — §1-§69).
This file starts the H200/Hopper/SM90 track from 2026-07-20 onward.

## §H1. 2026-07-20 — H200 box setup, CUDA switch, and the multi-arch Makefile fix

### H1.1 Goal
Build a genuine SM90 (Hopper) BF16 GEMM module in `src/level3/batched_gemm/BF16/SM90/`,
mirroring the SM89/SM100/SM103 folder convention. Before writing any kernel code, the
build system itself needed fixing — it had never been tested on non-Blackwell hardware.

### H1.2 Environment audit + CUDA 12.4 -> 13.0 switch
- H200 confirmed via `nvidia-smi`: compute capability **9.0**, driver 580.126.09, 143771 MiB.
- CUDA 13.0 was **already installed** on the box (`/usr/local/cuda-13.0`, full toolkit +
  nsight-compute/nsight-systems) — no download needed. `/usr/local/cuda` was symlinked to
  the older `cuda-12.4` (a colleague's setup), which is why `nvcc` resolved to 12.4.
- Switched the symlink: `sudo ln -sfn /usr/local/cuda-13.0 /usr/local/cuda`. Verified:
  `nvcc --version` -> release 13.0, V13.0.88.
- `LD_LIBRARY_PATH` was built from 3 stacked `.bashrc` lines that each re-prepended without
  clearing the previous ones (cuda-13.0 first, cuda-12.4 duplicated twice after) — messy but
  functionally harmless since 13.0 already won resolution priority; left as a known cleanup
  item, not urgent.

### H1.3 Why CUDA 13.0 was necessary, not optional
Hopper's `wgmma.mma_async` requires the family-specific `sm_90a` target, exactly the same
class of requirement as Blackwell's `tcgen05.mma` needing `sm_103a` (both proven from the
local PTX ISA 9.3 spec: `ptx_docs/ptx_isa_9.3.txt`, "Target ISA Notes: Requires sm_90a."
for wgmma, "requires sm_100a" for tcgen05's .f32 form — each gated to ONLY its own arch,
neither is a subset/superset of the other). An old toolkit not recognizing `compute_90a`
would hard-fail the same way `compute_103a` once failed on a stale CUDA 12.8 box earlier
in the B300 campaign (see `analyzations_b300.md` §60+ era).

### H1.4 Two distinct SM90-vs-SM100/103 facts, both verified from the PTX spec (not assumed)
1. **`wgmma` and `tcgen05` are NOT a subset/superset chain.** Blackwell didn't extend
   Hopper's warpgroup-MMA mechanism — it replaced it. Hopper's `wgmma` accumulates directly
   in registers; Blackwell's `tcgen05` accumulates in a dedicated on-chip Tensor Memory
   (TMEM) that doesn't physically exist on Hopper silicon. Neither instruction is valid
   under the other's `"a"` arch.
2. **Both `sm_90` (plain) and `sm_90a` (accelerated) are real, separate, valid compile
   targets for the SAME chip** — not two different GPUs. Plain `sm_90` is the portable
   baseline (works on any 9.0+ chip); `sm_90a` additionally unlocks `wgmma` but ties the
   binary to the Hopper family specifically. Same relationship holds for `sm_103`/`sm_103a`.

### H1.5 The Makefile's GPU-arch auto-detect: from a stale hardcode to an empirical probe
Original Makefile hardcoded `GENCODE_FLAGS := -gencode arch=compute_103a,code=sm_103a` —
correct only for B300, silently wrong (or simply never tested) for anything else.

Considered and rejected two lazier options first, both checked against real prior art:
- **PyTorch's own `native` arch-detect** (`torch/utils/cpp_extension.py`,
  `_get_cuda_arch_flags`): reads `torch.cuda.get_device_capability()` but ALWAYS emits the
  plain `-gencode=arch=compute_N,code=sm_N` — it never appends `"a"`, for exactly the same
  reason we discovered: whether `"a"` is valid isn't a hardware-queryable fact.
- **CUTLASS's own `CMakeLists.txt`** (`CUTLASS_NVCC_ARCHS_SUPPORTED`): a hand-maintained,
  CUDA-toolkit-version-gated list (`90a` from CUDA>=12.0; `100a/103a` etc. from >=12.8/12.9).
  Same fundamental limitation — no design can pre-know an unpublished future ISA.
- Also checked `nvcc --list-gpu-code`/`--list-gpu-arch` directly: neither lists ANY `"a"`
  variant (confirmed empirically on this box: full list came back with only plain `sm_XY`
  entries), so there is no compiler-introspection shortcut either.

**Final design: empirically probe.** At `make`-parse time, for every unique
`nvidia-smi --query-gpu=compute_cap` result, write a 1-line trivial `.cu` kernel and try
compiling it with `-gencode arch=compute_XXa,code=sm_XXa` using the *actual active* nvcc;
use `"a"` only if that compile succeeds, else fall back to the plain form. This is ground
truth for the exact toolkit on the exact box, needs no maintained list, self-adapts to any
future CUDA release or chip. Cost: ~1-2s one-time per `make` invocation (negligible next to
multi-minute full rebuilds — deliberately left uncached; see H1.9).

Verified live on H200: `make info` -> `GENCODE: -gencode arch=compute_90a,code=sm_90a`,
`GPU caps: 9.0`. Confirmed B300 still resolves to `compute_103a` (mechanism unchanged for it).

### H1.6 The deeper problem the arch-detect alone didn't solve: per-file exclusion
Auto-detecting the right flags for the CURRENT GPU only fixes what flags get PASSED — the
Makefile's `CU_SOURCES := $(shell find $(SRCDIR) -name '*.cu')` still unconditionally
globs the ENTIRE tree. Even with `compute_90a` correctly selected, `make all` would still
try to compile `BF16/SM100`, `BF16/SM103` (+`kota_addmm`), `FP8/SM103`, `FP16/B300` — all
`tcgen05`-based, Blackwell-exclusive — and hard-fail on Hopper. This is NOT a new bug: it
was equally broken for any non-B300 box the whole campaign, just never surfaced because
every prior session ran on B300, and B300's `103a` target happens to be a superset that
also swallows the portable SM89-era files fine (older/backward-compatible instructions
generally forward-compile onto newer architectures; that asymmetry is exactly why the bug
was invisible until now).

Independently found: **BluTrain's own `Tensor-Implementations/Makefile`** (embedded copy at
`BluTrain/Tensor-Implementations/`) had ALREADY solved this identical problem, with an
explicit code comment: "Fix: build fat binaries. Legacy sources compile for every supported
legacy arch...; Blackwell-only sources compile only for the Blackwell targets they require.
Selection is per source file, based on which directory it lives in." Its
`GENCODE_LEGACY`/`GENCODE_BLACKWELL`/`BLACKWELL_MARKERS`/`GENCODE_FOR` pattern was the
direct reference ported into BluBLAS's own Makefile (see H1.7).

### H1.7 Design iteration — from a naive threshold to a safe enumerated list
First draft used `HAVE_BLACKWELL := (any detected cap >= 100)` — an open-ended numeric
threshold. **User caught the flaw directly**: this repeats the exact "higher number implies
compatible" assumption already disproven for wgmma-vs-tcgen05. A genuinely new future
architecture (whatever comes after Blackwell, however numbered) is NOT provably
tcgen05-compatible just because its cap is bigger, and `>=100` would silently misclassify it.

**Fix: replaced the threshold with an EXPLICIT ENUMERATED list** of the specific caps we
have real evidence for — `BLACKWELL_CAPS := 100 101 103 110 120 121` (pulled directly from
CUTLASS's own `CUTLASS_NVCC_ARCHS_SUPPORTED` "a"-suffixed entries) and `HOPPER_CAPS := 90`.
Verified via standalone bash simulation across 5 cases (B300=10.3, H200=9.0, an SM89-only
box=8.9, a HYPOTHETICAL future arch=13.0, a known-but-untested Blackwell cap=10.0): only the
enumerated caps match; the hypothetical future arch correctly falls through to "excluded" —
i.e. the unknown case fails SAFE (only portable/legacy files build) rather than being wrongly
assumed compatible. This is the honest ceiling for ANY design (ours or CUTLASS's): nobody can
pre-support an ISA that doesn't exist yet; the best a Makefile can do is make the unknown
case safe and require one deliberate line added once a new architecture's real compatibility
is documented.

### H1.8 The dispatcher-side half of the fix: link-time guards
Excluding a Blackwell-only `.cu` file from the build doesn't help if some OTHER, always-
compiled file still declares+calls its symbols unconditionally — that produces an undefined-
symbol LINK error, not a graceful skip. Audited every dispatcher for this:
- `BF16/Bgemm_dispatcher.cu` — **6 real references** (3 `extern "C"` forward-declarations +
  3 calls inside `if (actual_sm_ver >= 103) {...}`) to `blublasBgemmSM103_dispatch_nn/nt/tn`.
  This is the ONE file that actually needed guarding.
- `FP16/Hgemm_dispatcher.cu`, `FP32/Sgemm_dispatcher.cu`, `FP8/SM103/Bgemm_sm103_fp8_dispatcher.cu`
  — checked precisely (not just a combined grep count, which was initially misleading): ZERO
  actual `SM103_dispatch`/`SM100_dispatch` references. Repo-wide search confirmed no caller
  OUTSIDE their own Blackwell-marked folder references `blublasHgemmSM103_dispatch_*` or the
  FP8 dispatcher's symbols — both are self-contained and safe to exclude wholesale.
- Fix applied: Makefile now defines `-DBLUBLAS_HAS_SM103`/`-DBLUBLAS_HAS_SM90` (`ARCH_DEFINES`)
  exactly when the matching files are included (same enumerated-cap logic, single source of
  truth). `Bgemm_dispatcher.cu`'s 3 declarations + the whole `if (actual_sm_ver >= 103) {...}`
  block are wrapped in `#ifdef BLUBLAS_HAS_SM103` — declaration and call site guarded together
  so they can never drift out of lockstep.

### H1.9 Two real mistakes made and caught mid-session (both fixed, worth recording)
1. **Wholesale file overwrite from a differently-formatted working copy.** Read+edited a
   local mirror of `Bgemm_dispatcher.cu` that had different line-wrapping than the actual
   push-repo HEAD version; the resulting `cp` produced a 620-line diff for a 12-line change
   (a full reformat, not the intended edit). Caught via `git diff --cached --stat` showing an
   implausible line count before pushing; fixed by reverting and re-reading+editing the REAL
   current push-repo file directly.
2. **Fragile inline Make/shell filtering corrupted the build.** The first working version of
   the per-file exclusion was written as deeply nested `for`/`if`/`case` shell control flow
   directly inside Makefile text (`$(shell ...)`), mixing Make's `$(VAR)` substitution with
   shell's `$$`-escaping in the same block. It passed a small 4-file standalone test locally,
   but at the real repo's ~85-90 file scale it corrupted the output: a stray `excl=1`
   fragment leaked into the file list, and Make tried to build a target literally named
   `excl=1`, hard-failing with `No rule to make target 'excl=1'`. Root-caused and fixed by
   moving the filtering logic entirely OUT of inline Makefile shell and into a standalone,
   independently-testable script: `tools/filter_cu_sources.sh` (plain positional args + env
   vars only, no Make-substitution-inside-shell-control-flow at all). Re-verified at the
   exact real scale (85 files) with zero garbage output before handing back.
3. **The Makefile change sat staged-but-unpushed for a while without me verifying it had
   actually reached the remote.** Caught by direct three-way check
   (`git diff` working-tree vs `git show HEAD:Makefile` vs `git show origin/main:Makefile`)
   after a "deep analyze everything" request — the empirical-probe fix existed only locally
   until that point despite earlier assuming it was "done." Lesson: verify a fix landed on
   origin, don't just trust that staging + a described push command means it happened.

### H1.10 Final validated result — first successful multi-architecture build in this repo's history
`git pull` + `make info` on H200 (commit `16313f0`):
```
GENCODE      : -gencode arch=compute_90a,code=sm_90a
GPU caps     : 9.0
Have Blackwell (SM100/SM103/B300) : no
Have Hopper (SM90)                : yes
ARCH_DEFINES : -DBLUBLAS_HAS_SM90
Excluded .cu files : 19
```
`make clean && make all -j126`: 19 Blackwell-only files correctly excluded (FP8/SM103 x3,
FP16/B300 x5, BF16/SM100 x1, BF16/SM103 x9 incl. kota_addmm), remaining ~66 files (SM89/SM86
legacy BF16/FP16/FP32/FP64 + generic gemm/TRMM) compiled clean under `compute_90a`/
`BLUBLAS_HAS_SM90`, zero errors (only pre-existing harmless "declared but never referenced"
warnings in old SM89/SM86 code, unrelated to today's changes), and the shared library
**linked successfully**: `✅ Library built: build/libmyblas.so`. This is the first time this
Makefile has produced a working build for anything other than B300.

### H1.11 Open items / next steps
- Functional verification still pending: the .so LINKS, but hasn't been exercised (no test
  run yet) — next session should run an existing SM89-path test
  (`make test TEST=examples/...`) on H200 to confirm the legacy BF16/FP32 kernels actually
  execute correctly on Hopper hardware, not just that they compiled.
- `.bashrc` `LD_LIBRARY_PATH` cleanup (3 stacked lines -> 1, symlink-based) — cosmetic, not
  urgent, flagged in H1.2.
- **The actual mission — write the real SM90 (wgmma-based) BF16 GEMM kernel — has not
  started yet.** Everything above was build-system prerequisite work; scaffolding
  `src/level3/batched_gemm/BF16/SM90/` with a genuine warp-specialized wgmma+TMA kernel
  (referencing CUTLASS's `sm90_gemm_tma_warpspecialized*.hpp` collective mainloop as the
  architectural model, the same way SM103's design leaned on CUTLASS's sm100 code) is the
  next real task.
- Colleague's BluTrain/SM89-SGEMM-under-autocast question (raised mid-session, investigated
  via `BluTrain/Tensor-Implementations/src/Kernels/cuda/matmul/GenMatmul.cu`): traced to
  `BLAS_BACKEND` defaulting to plain `cublas` unless the training script is launched with
  `WITH_BLUBLAS=1` explicitly — if that flag wasn't passed, BluBLAS was never linked at all,
  and the observed "SM89 SGEMM" in the nsys trace is almost certainly cuBLAS's own internal
  architecture-tagged kernel, unrelated to our repo's SM89 folder. Not yet confirmed with
  the colleague directly — worth a one-line follow-up ("did you build with WITH_BLUBLAS=1?").

## §H2. 2026-07-21 — module design survey, Kota's SM90 base wired in, repo/sync note

### H2.1 Cross-module design survey (before writing/accepting any SM90 code)

Went through BF16/SM89, FP32/SM86, FP32/SM89, and BF16/SM103 file-by-file. Finding: the
repo does NOT have one uniform module pattern — it has exactly two, split by whether the
hardware generation needs warp-collective/async orchestration or not:

- **Ampere-style** (BF16/SM89, FP32/SM86, FP32/SM89): one generic `*_core_template.cuh`
  kernel templated on `<Config, IsAligned, SplitK, Layout>` (`Layout` an enum `{NN,NT,TN}`
  handled via `if constexpr` inside a single kernel body, simple synchronous
  `mma.sync`+`ldmatrix`). Thin `*_variants.cu` files just instantiate tile shapes. **No
  internal module dispatcher** — shape routing lives directly in the top-level
  `Bgemm_dispatcher.cu`/`Sgemm_dispatcher.cu` as an if/else cascade (confirmed by grep:
  zero dispatcher files in any of these three directories).
- **Cluster/async-style** (BF16/SM103): a shared giant `*_cluster_template.cuh` (2612
  lines) PLUS separate substantial per-layout files (`_NN.cu`, `_NN_cluster.cu`, etc.,
  700-870 lines each — TMA descriptor setup and cluster launch genuinely differ per
  layout, unlike Ampere's simple index-permutation). Thin `*_variants.cu` wrappers.
  **Its own internal `Bgemm_sm103_dispatcher.cu`** with a measured `kTnWinners[]` (N,K)
  winner-table for TN (12+ variants) and a simpler M/N%256 check for NN/NT.
- Real principle: simple synchronous hardware -> one generic kernel, no internal
  dispatcher. Async/warp-collective hardware (TMA, clusters, wgmma) -> per-layout
  launcher files + its own internal dispatcher, because the orchestration complexity
  genuinely differs, not because of an inconsistent codebase.

### H2.2 Kota's SM90 base setup (`BluBridge-BLAS_21_07_2026_kota_H200_setup`) — verdict

Reviewed strictly as a base lookup, not copied blind. Correctly follows the
**cluster/async-style** pattern (right call — `wgmma` is warp-group-collective like
`tcgen05`, not simple `mma.sync`): `Bgemm_sm90_bf16.cuh` (public decls, mirrors
`Bgemm_sm103_bf16.cuh`'s role and true/false decline contract), `Bgemm_sm90_warpspecialized
_template.cuh` (947 lines, the shared kernel), `Bgemm_sm90_{nn,nt,tn}_variants.cu` (thin
one-line instantiation wrappers, same style as SM103's), `Bgemm_sm90_dispatcher.cu` (own
internal alignment-gated dispatcher, M%128/N%128/K%64, with measured rationale in
comments — e.g. why TN always uses 128x128x64 regardless of N, explained via shared-memory
double-buffer budget for staging both raw operands, not guessed). Test files
(`test_bgemm_sm90_{bf16,vs_cublas,nn_tn}.cu`) exist; code comments claim H200 hardware
validation (NT 19/19, NN/TN 22/22) — not independently re-verified this session, that's
next. **Verdict: well-designed, structurally consistent with SM103's precedent — adopted.**

### H2.3 Makefile comparison — Kota's fat-binary vs ours (detect-and-exclude)

Kota's Makefile is a genuinely different, more portable strategy: per-file `GENCODE_FOR`
(keyed on path markers) gives every file its own gencode always, building ONE universal
`.so` (Ampere through Blackwell) that runs on any GPU model without rebuild; a separate
`NVCC_BLACKWELL` toolchain path lets Blackwell files use a newer nvcc than the rest of the
build without needing the whole box's active CUDA symlinked (exactly the friction we
manually solved this week via symlink-switching). Confirmed via direct grep: his Makefile
never calls `nvidia-smi` at all — gating is purely on toolchain presence
(`HAVE_BLACKWELL_TOOLCHAIN`), not on attached GPU, so on a normal box **every** legacy +
Hopper + Blackwell file compiles on every `make clean && make all`, always, regardless of
what's plugged in (85 files, full multi-gencode legacy fat list, every time).
Ours only ever compiles for the one GPU physically in the box (66/85 files on this H200,
single gencode target, no fat legacy list) — much faster iteration, but the resulting
`.so` cannot run on a different GPU model without a rebuild (confirmed: `GENCODE_FLAGS` is
derived purely from `nvidia-smi`-detected caps present on THIS box, not a portable list).
**Decision: keep our Makefile unchanged.** Matches actual usage (always rebuild fresh on
the literal target box over SSH, no fleet-wide artifact distribution today) and avoids
paying Kota's always-compile-everything cost on every iteration. Revisit only if BluBLAS
is ever meant to ship as one build artifact across mixed GPU models.

### H2.4 SM90 module brought in, wired into the top-level dispatcher

Copied the 6 files from H2.2 into `src/level3/batched_gemm/BF16/SM90/` plus the 3 test
files into `examples/`. Wired `Bgemm_dispatcher.cu`'s top-level
`blublasBgemmStridedBatched_dispatcher` with a `0b. SM90 Path` block: `#ifdef
BLUBLAS_HAS_SM90` guard (needed because our Makefile still detect-and-excludes files —
Kota's originals had zero `#ifdef` guards anywhere, since his Makefile always compiles
SM90 files unconditionally; that guard is the one real adaptation, not a straight copy),
runtime-gated to `actual_sm_ver >= 90 && actual_sm_ver < 100` (exact bound Kota's own
top-level dispatcher already used, so it can never shadow the `>=103` Blackwell block),
same true/false decline-and-fall-through convention as SM103. Zero Makefile changes
needed — `HOPPER_MARKERS := SM90` already classifies the new folder correctly by design
(the whole point of that marker-based filter, see H1.5-H1.6). Compile-only sanity-checked
against `sm_90a` on a local RTX 3060 box (can compile for Hopper without one attached —
compilation doesn't require the target GPU physically present); real execution
correctness still needs the actual H200: `make clean && make all -j126`, then run the 3
copied test files there.

### H2.5 Repo/sync structure — NOTE FOR LATER, not resolved this session

Discovered mid-session that there are (at least) three separate local checkouts of this
codebase, easy to edit the wrong one by mistake (did exactly that once this session,
caught before it mattered):

1. `~/Downloads/Neural_Networks_exp_1926/BLUBLAS_kota/BluBridge-BLAS` — **not its own git
   repo**; a plain nested folder inside the bigger personal monorepo
   (`origin = GOUTHAM1926/Neural_Networks_exp_1926`). Playground copy only — edits here
   don't reach any BLAS-specific remote on their own.
2. `~/Downloads/BLUBLAS/BLUBLAS_kota/BluBridge-BLAS` — a real, separate git repo,
   `origin = github.com/GOUTHAM1926/blublas.git` (personal). **This is the one actually
   used to push/pull this session** — today's SM90 files + dispatcher wiring live here.
3. `~/Downloads/BluBridge-BLAS_21_07_2026_kota_H200_setup/BluBridge-BLAS` — Kota's
   handoff copy, `origin = github.com/Blubridge-Tech/BluBridge-BLAS` (the team/org repo).
   This looks like the actual eventual submission target ("main repo" / "others" the user
   mentioned), separate from #2 above.

Open question, explicitly deferred: how work done in #2 (personal repo, where the daily
work happens) is meant to reach #3 (`Blubridge-Tech/BluBridge-BLAS`, the shared/team repo)
is NOT settled — no PR flow, fork-sync, or manual-copy convention has been confirmed yet,
and the user expects this to change/get clarified later. **Action for a future session:
don't assume push-to-personal-origin is the finish line — check with the user whether a
Blubridge-Tech PR (or some other sync step) is also required before considering any given
change actually "shipped."**

### H2.6 First `make all -j126` on H200 with SM90 wired in — clean build

`git pull` + `make clean && make all -j126` on H200: same 19 files excluded (correct,
unchanged classification), `-DBLUBLAS_HAS_SM90` present on every compile line, all 4 new
SM90 files (`Bgemm_sm90_{nn,nt,tn}_variants.cu`, `Bgemm_sm90_dispatcher.cu`) and the edited
top-level `Bgemm_dispatcher.cu` compiled with **zero errors, zero warnings** — every warning
in the log is pre-existing noise in `FP32/SM89`, `FP32/SM86`, `FP64` (unused-variable
notices, same files/lines as before today). Linked successfully: `build/libmyblas.so`,
with all 4 new `.o` files confirmed present in the final link command.

### H2.7 `setmaxnreg` "Potential Performance Loss" investigated and resolved

Build log showed 6× `ptxas info (C7507) Potential Performance Loss: 'setmaxnreg' ignored
to maintain minimum register requirements` (2 per SM90 variant file × 3 files, one per tile
shape). Traced to `Bgemm_sm90_warpspecialized_template.cuh`'s warp-specialized register
split (`PRODUCER_REGS=72`/`CONSUMER_REGS=240`, via `setmaxnreg.dec`/`.inc` for the 1
producer + 2 consumer warpgroups). Two initial hypotheses were tested and both **empirically
refuted** (not just theorized — verified via `-Xptxas -v` on real compiles):

1. "Total register-file budget" theory (128×(72+240+240)=70656 > 65536 SM90 register file)
   — dropping `PRODUCER_REGS` to 24 (the documented minimum) still didn't clear the warning.
2. "Producer giveback must cover consumer demand" pool-balance theory — tested `24/232`
   against the measured natural baselines (126, 167 registers/thread per tile shape via
   `ptxas -v`), still didn't clear.

Deep-researched properly (per explicit user direction — CUTLASS/DeepGEMM/ThunderKittens
source + PTX ISA docs + NVIDIA developer forums/GitHub, not further guessing) and found the
real, authoritative root cause via `NVIDIA/cutlass` GitHub issue #2007 (fetched via GitHub's
public REST API) — maintainer `thakkarV`'s comment gives the actual constraint:

```text
232 * 2 + 24 != 168 * 3   <- must be an EXACT equality, not a budget inequality
```

i.e. `PRODUCER_REGS + num_consumer_warpgroups * CONSUMER_REGS` must exactly equal
`baseline_registers_per_thread * total_warpgroups`, where the baseline is whatever ptxas
naturally compiles the (single, branching) kernel function to — `setmaxnreg` redistributes
a fixed-size CTA register pool, it cannot create or destroy registers. Confirmed CUTLASS's
own shipped Hopper cooperative-GEMM kernel
(`sm90_gemm_array_tma_warpspecialized_cooperative.hpp:188-189`) uses `LoadRegisterRequirement
=40`/`MmaRegisterRequirement=232` for the identical 1-producer+2-consumer/384-thread
structure — but porting those exact values to Kota's kernel (tested empirically) still
didn't clear it, because the natural baseline is a property of the ACTUAL compiled code
(different between CUTLASS's more complex kernel and Kota's simpler one), not a portable
constant. Also confirmed: `--maxrregcount` is silently overridden by an explicit per-kernel
`__launch_bounds__` (documented nvcc behavior, verified empirically — the flag had zero
effect on the reported register count). Also confirmed the two tile shapes this file
instantiates have genuinely different natural baselines (126 vs 167 registers/thread, from
`CREGS=BN/2` scaling), and their exact-equality targets (378, 501) aren't even multiples of
8 at those baselines — meaning no fixed `(P,C)` pair could ever satisfy both simultaneously,
and the true baseline is itself a fixed-point function of which `(P,C)` you try, not a fact
you can measure once and solve for (matches why even the CUTLASS issue's own reporter needed
several rounds, hanging when off by just a few registers).

Built a small reusable autotuning probe (`tools/tune_sm90_regsplit.sh`, since deleted — see
below) modeled on this repo's own `GENCODE_FLAGS` empirical-probe philosophy (test-compile,
check the real ptxas result, don't assume) to sweep candidate `(P,C)` pairs per isolated
tile shape. Swept 91 combinations per shape (both combined and isolated) — **zero cleared
the warning**, for either tile shape, in either the CUTLASS-adjacent range or the full valid
range.

**Resolution — matches user's direct question and instinct:** since (a) the redistribution
was already being silently ignored by ptxas regardless (confirmed unchanged register counts
with or without the calls present), (b) SM103's own kernel in this exact codebase never
attempts this optimization either and ships correct + hardware-validated without it, and (c)
further empirical tuning is genuinely open-ended (no formula, real trial-and-error even for
NVIDIA's own engineers) — **removed the two `setmaxnreg_inc`/`setmaxnreg_dec` calls and
their now-dead helper functions/config constants entirely**, matching SM103's proven
precedent rather than forcing an unproven optimization. Zero functional change (confirmed:
register counts 124-126/167 after removal vs 126/167 before, trivial variance from removed
dead code) — the warp specialization itself (separate producer/consumer code paths, so TMA
loads and wgmma issue overlap) still provides its own benefit independent of the register
split. All 4 SM90 files + the edited top-level dispatcher now compile with zero errors AND
zero warnings. `tools/tune_sm90_regsplit.sh` and the temp isolated-shape test files were
deleted after use — not left in the repo since the fix path taken didn't end up needing them,
but the sweep methodology is documented here in full if a future session wants to revisit
giving SM90 (or SM103) this optimization properly, per-tile-shape, with its own dedicated
compilation unit so each shape can get its own forced register target.

**Not yet done:** actually running the 4 SM90 test files on real H200 hardware — everything
above is compile-only verification (this session's environment has an RTX 3060, not an
H200). This is still the next real step.

### H2.8 Real H200 hardware verification of the setmaxnreg removal — SASS + ncu + nsys

User (correctly) was not satisfied with a single TFLOPS run as proof (an earlier A/B attempt
briefly showed a ~2x regression on large shapes that turned out to be a one-off anomaly, not
reproduced on a second run of either version — see the conversation; not repeated here since
this section supersedes it with the actual rigorous verification). Built
`tools/capture_sm90_proof.sh` (deleted after use, methodology documented here) to capture, for
both the original (commit `72da35f`, dead `setmaxnreg` calls present) and fixed (current HEAD)
versions, on the real H200: SASS disassembly of the built objects, `ncu` hardware-counter
metrics (register count, occupancy, SM/DRAM throughput, instructions executed, duration), and
`nsys` system-wide kernel timing over the full 37-shape benchmark. First attempt used
`ncu --set full`, which is extremely slow (replays each kernel 15-30+ times, one pass per
profiling section) — switched to `ncu --metrics <specific 6 counters>` directly on the
profiling run itself, which only needs 1-3 replay passes and finishes in a fraction of the time
for the exact same proof.

**Results — three independent tools, one conclusion, real H200 hardware:**

- **SASS** (NT layout, the layout the benchmark exercises): 3192 instructions in both versions
  (exact match), zero register spills (`STL`/`LDL`) in either, opcode-type histogram identical
  except 2 instructions shifting which predicate guards them (trivial scheduler reshuffling,
  not a different program).
- **ncu**, per distinct kernel config actually launched:
  - `Sm90WarpspecConfig<128,256,64,0>` NN: registers 167=167, occupancy 18.63%=18.63%,
    SM throughput 7.83%≈7.84%, instructions 24,224,000≈24,223,488.
  - Same config NT: registers 167=167, occupancy 14.80%≈14.81%, SM throughput 2.37%=2.37%,
    instructions 82,264=82,264 (exact).
  - `Sm90WarpspecConfig<128,128,64,0>` TN: registers 126=126, occupancy 18.49%=18.49% (exact),
    SM throughput 1.08%=1.08% (exact), instructions 858,624≈858,752.
- **nsys**, real system-wide trace of the full 37-shape benchmark: 888 kernel launches in both
  (exact match), total/avg/median duration matching within 0.2-1.2% (normal run-to-run noise).

Combined with 37/37 correctness at `0.00e+00` MaxRelErr in both versions across all runs this
session, this is now a properly, independently verified result across compile-time (SASS),
hardware-counter (ncu), and real-timeline (nsys) evidence — not a claim resting on a single
TFLOPS number.

### H2.9 FUTURE TASK — implement the setmaxnreg register-split PROPERLY (not now, explicitly deferred)

Explicit user decision: leave `setmaxnreg` out for now (matches SM103's own precedent, zero
proven downside per H2.8) — but this is NOT a closed door. When picked up again, do it
properly, not by porting someone else's numbers:

1. The exact-equality constraint (`PRODUCER_REGS + num_consumer_warpgroups * CONSUMER_REGS ==
   baseline_registers_per_thread * total_warpgroups`) must be satisfied for the SPECIFIC
   kernel's actual compiled baseline, which is a property of that kernel's real code — not
   portable from CUTLASS or any other kernel, confirmed by directly testing CUTLASS's own
   canonical values (40/232) against Kota's kernel and having it still fail (§H2.7).
2. The two tile shapes in this file have genuinely different natural baselines (126 vs 167
   registers/thread) and need their OWN register-split values — this likely requires splitting
   them into separate compilation units/translation units so each can get its own forced
   register target (`--maxrregcount` is silently overridden by an explicit per-kernel
   `__launch_bounds__`, confirmed empirically, so the two shapes currently sharing one TU
   can't be tuned independently via that flag alone).
3. Use `tools/tune_sm90_regsplit.sh`'s methodology (deleted, but documented in H2.7 above) as
   the starting point for an actual empirical sweep per shape, rather than guessing values.
4. Per H2.7's honest caveat: confirmed via SASS that this kernel currently has ZERO register
   spilling under uniform allocation at either baseline — meaning the classic motivating case
   for this optimization (avoiding spills on a register-starved consumer) doesn't clearly apply
   to these specific (128x256, 128x128) tile shapes. Worth checking whether the realistic
   upside is actually meaningful for THESE shapes before sinking more tuning time in, or
   whether it matters more for larger future tile configs (CUTLASS's own use case tends to be
   bigger tiles with genuinely register-constrained accumulators).
5. Same technique is a legitimate future candidate for SM103 too (Blackwell CUTLASS kernels
   use it with their own values, 48/256 — also not portable, would need its own from-scratch
   empirical derivation per H2.7's lesson).
6. **Why "one universal value for every shape" doesn't work here** (user asked directly):
   CUTLASS's own kernel hardcodes one fixed `(40, 232)` pair across its whole tile-shape
   catalog, which looks universal but isn't — it works only because CUTLASS's heavier CuTe
   abstraction overhead keeps their natural compiled baseline clustered near ~168 regardless
   of tile size. Kota's leaner kernel's baseline visibly SCALES with tile size (126 vs 167,
   tracking `CREGS = BN/2`), so a single pair can't satisfy the exact-equality for every shape
   — confirmed this is arithmetic, not a tuning gap (§H2.7's 378 vs 501 result). Any future
   attempt needs a value PER shape, and that requirement only grows as more shapes are added.
7. **The "ignored" fallback is graceful but NOT unconditionally safe** — worth knowing before
   attempting this again. When the exact-equality doesn't hold, ptxas's default behavior (all
   of our testing) is to silently keep uniform allocation and just warn — genuinely safe
   degradation. But the CUTLASS issue #2007 thread shows this ISN'T guaranteed: a value close
   to but not exactly satisfying the equation caused an actual kernel HANG at runtime for that
   reporter, not just a silent no-op. This means any future per-shape sweep MUST validate each
   candidate by actually running it on the H200 (checking for hangs + correctness), not just by
   checking that the ptxas "ignored" warning disappeared at compile time — compile-time
   clearing of the warning is necessary but not sufficient proof of safety.

## §H3. Qwen dense-model matmul shape research — full independent redo (2026-07-21)

### H3.1 Why this redo happened, and what it explicitly does NOT carry over

The prior campaign (`analyzations_b300.md` §8-11, §35, §16.3) researched 8 Qwen models
(Qwen2.5: 0.5B/1.5B/3B/7B, Qwen3: 0.6B/1.7B/4B/8B) and built a 42-(N,K)-shape catalog from
them. That catalog was re-audited during this session (via a dedicated research agent reading
the full 6483-line document) and found to be **internally correct for what it covered** — no
section ever found the 42 shapes themselves to be wrong, only that the *scope* (2 generations,
8 fixed sizes, forward-pass only, M=4096 only) was narrower than the full picture.

Per explicit instruction, this redo does NOT reuse a single number from that catalog. Every
model in this section was independently re-fetched from the official Hugging Face `config.json`
this session, every architectural claim (fused vs separate QKV, MHA vs GQA, FFN structure) was
verified from actual model source code (`modeling_qwen.py` / `modeling_qwen2.py`), not assumed
from general knowledge or from the prior document. Where a number in this section happens to
match the old catalog, that is because both are correct, not because this one copied the other.

**Scope, explicitly locked before this research began:** dense, standard-transformer-architecture
Qwen models only. Excluded by explicit decision: all MoE variants (Qwen1.5-MoE-A2.7B,
Qwen2-57B-A14B, Qwen3-30B-A3B, Qwen3-235B-A22B) — same GEMM math per expert, but a materially
different execution pattern (grouped/variable-M GEMM dispatch) that's a separate research task.
Also excluded: the new hybrid linear-attention generations discovered mid-research (Qwen3-Next,
Qwen3.5, Qwen3.6 — see H3.1.1) — architecturally a different computation entirely (Gated
DeltaNet-style state-recurrence, not standard QKV/O attention), out of scope until the standard
dense-transformer catalog is solid. Also excluded: multimodal towers (VL/Audio/Omni variants) —
these share their parent generation's base transformer config for the text backbone but add a
separate vision/audio encoder with its own distinct matmul shapes, not researched here.

#### H3.1.1 Important discovery made during this redo: two entire generations released after prior knowledge

A dedicated enumeration pass (official GitHub `QwenLM` org + official Hugging Face `Qwen` org +
official blog, cross-checked against live `config.json` fetches) found that Alibaba released
**Qwen3.5 (Feb 2026)** and **Qwen3.6 (Apr 2026)** — entire new generations, confirmed via live
`model_type: qwen3_5`/`qwen3_5_moe` config fields, not rumor. These use a hybrid architecture:
most layers are `linear_attention` (Gated DeltaNet-style recurrent state update), with a standard
`full_attention` layer only every `full_attention_interval=4` layers, plus a larger vocab
(248,320 vs Qwen3's 151,936) and native 262,144-token context. **Qwen3-Next-80B-A3B** (Sept 2025)
is a transitional version of this same hybrid pattern. All three are explicitly OUT OF SCOPE for
this redo per the scope-lock above, but are flagged here because a future BluTrain target-model
decision needs to know they exist — this redo's 168-shape catalog does NOT cover them at all.

### H3.2 Cross-generation architecture findings (apply across multiple models, not per-model quirks)

These are the load-bearing, non-obvious findings that a shape catalog built from `config.json`
values alone — without reading actual model source code — would have gotten wrong or missed
entirely. Each was independently verified from the official `modeling_*.py` source this session.

**1. Qwen1's `intermediate_size` config field is NOT the real per-branch FFN width — the single
most consequential catch of this whole research effort.** Verified directly from
`QWenMLP.__init__` in the official `modeling_qwen.py` (byte-identical across all 4 Qwen1 repos):
```python
self.w1 = nn.Linear(config.hidden_size, config.intermediate_size // 2, bias=not config.no_bias)
self.w2 = nn.Linear(config.hidden_size, config.intermediate_size // 2, bias=not config.no_bias)
ff_dim_in = config.intermediate_size // 2
self.c_proj = nn.Linear(ff_dim_in, config.hidden_size, bias=not config.no_bias)
```
Both `w1` (up) and `w2` (gate, SiLU-activated) actually project to `intermediate_size // 2`, not
the raw config value. Any shape catalog built by naively reading `config.json`'s
`intermediate_size` field for Qwen1 would have every Gate/Up/Down matmul shape wrong by exactly
2x. All Qwen1 shapes in this document already use the corrected (halved) width.

**2. Qwen1 uses a fused, single-matmul QKV projection — not three separate matrices.** Verified
from `QWenAttention.__init__`:
```python
self.projection_size = config.kv_channels * config.num_attention_heads
self.c_attn = nn.Linear(config.hidden_size, 3 * self.projection_size)
...
mixed_x_layer = self.c_attn(hidden_states)
query, key, value = mixed_x_layer.split(self.split_size, dim=2)
```
One GEMM produces Q+K+V concatenated, split afterward — genuinely one matmul with 3x the output
width, not three separate GEMMs. Every subsequent generation (Qwen1.5 onward) switched to three
independent `nn.Linear` layers (`q_proj`, `k_proj`, `v_proj`), confirmed from
`Qwen2Attention.__init__` (`modeling_qwen2.py`, identical structure through Qwen1.5/2/2.5/3). This
means a shape-catalog generator or a dispatcher's variant-selection logic CANNOT treat "QKV
projection" as one architectural pattern across the whole Qwen family — Qwen1 needs a genuinely
different matmul (one wide fused GEMM) than every later generation (three separate GEMMs).

**3. GQA (grouped-query attention) presence and ratio is NOT a clean function of generation or
model size — verified per-model, not assumed:**
- **Qwen1**: no GQA anywhere (confirmed: `num_key_value_heads` is absent not just from
  `config.json` but from the `QWenConfig` constructor signature itself — there is no code path
  for it in this generation at all). All 4 sizes are plain MHA.
- **Qwen1.5**: GQA appears in only 2 of 8 sizes (32B: 40→8 heads; 110B: 64→8 heads). The other
  six (0.5B, 1.8B, 4B, 7B, 14B, and notably **72B**) are all plain MHA — 72B sitting between two
  GQA models (32B, 110B) while itself using full MHA breaks any "bigger = more compressed"
  assumption.
- **Qwen2**: GQA in all 4 sizes, but the compression ratio itself varies (7:1, 6:1, 7:1, 8:1) —
  not a fixed constant across the generation.
- **Qwen2.5**: GQA in all 7 sizes, ratio varies non-monotonically (7:1, 6:1, 8:1, 7:1, 5:1, 5:1,
  8:1) — 14B and 32B both sit at 5:1 despite being different sizes, while 3B (8:1) and 72B (8:1)
  match each other despite being far apart in scale.
- **Qwen3**: `num_key_value_heads=8` fixed across all 6 sizes, so the GQA ratio itself scales
  cleanly with `num_attention_heads` (2:1 for 0.6B/1.7B, 4:1 for 4B/8B, 5:1 for 14B, 8:1 for
  32B) — the one generation where KV head count, not the ratio, is the fixed quantity.

**4. Non-square Q/O projections (`Q_dim = num_attention_heads * head_dim ≠ hidden_size`) exist
ONLY in Qwen3, and affect 3 of its 6 sizes, not 2.** Qwen3 is the only generation with an explicit
`head_dim` config override (=128 for every size); every other generation derives head_dim as
`hidden_size / num_attention_heads` with no remainder, making Q_dim always equal hidden_size by
construction. Independently re-verified for all 6 Qwen3 sizes this session: **0.6B** (H=1024,
Q_dim=2048, 2.0x), **4B** (H=2560, Q_dim=4096, 1.6x), and **32B** (H=5120, Q_dim=8192, 1.6x) all
have non-square Q/O. 1.7B, 8B, and 14B do not (Q_dim=H exactly). The prior campaign, having only
checked 0.6B/1.7B/4B/8B, found the first two correctly but never had the chance to catch 32B —
confirming the user's suspicion that scope, not correctness, was the gap in the old work.

**5. `tie_word_embeddings` and `vocab_size` splits are clean, model-count-based cutoffs within
each generation, but the exact cutoff differs by generation — verified per-model, not inferred
from one example:**
- Qwen1: all 4 sizes untied (no split to find).
- Qwen1.5: only 0.5B tied; all other 7 sizes (including 1.8B) untied — a 1-model phenomenon,
  not a size-based cutoff.
- Qwen2 / Qwen2.5: tied for the two/three smallest sizes, untied above — Qwen2.5's exact cutoff
  is between 3B (tied) and 7B (untied), applying uniformly to all four larger sizes (7B/14B/32B/72B).
- Qwen3: tied for 0.6B/1.7B/4B, untied from 8B upward (8B/14B/32B).
- `vocab_size` itself changes mid-generation in three of the five families (Qwen1: 151936→152064
  at 14B; Qwen2: same cutoff at 7B; Qwen2.5: same cutoff at 7B; Qwen1.5: 151936→152064 at 14B) —
  Qwen3 alone keeps a single vocab_size (151936) across all 6 sizes.

**6. Attention QK^T/softmax/@V matmuls are excluded from this catalog on the same grounds the
prior campaign used** (fused into FlashAttention/SDPA kernels in every real training/inference
path, never surfacing as a standalone GEMM call) — not re-verified against BluTrain's own
attention implementation in this session; flagged as an open item in H3.7.

### H3.3 Backward-pass derivation — methodology

For a forward linear layer `Y[M,N] = X[M,K] @ Wᵀ`, with `W` stored `[N,K]` (PyTorch `nn.Linear`
convention), training requires two additional GEMMs per layer during backpropagation:

1. **Input gradient**: `dX[M,K] = dY[M,N] @ W`. Same token-batch `M`, but `N` and `K` swap roles
   relative to the forward shape — reported here as `(M, K, N)`.
2. **Weight gradient**: `dW[N,K] = dYᵀ[N,M] @ X[M,K]`. This is the shape that's easy to get wrong
   by analogy — it is NOT simply "the forward shape transposed." The token-batch dimension `M`
   (4096 in this catalog) becomes the GEMM's **contraction dimension**, sitting in the position
   a hidden/intermediate dimension normally occupies in the forward pass. Reported here as
   `(N, K, M)`. This is a fundamentally different tiling problem than either the forward pass or
   the input-gradient pass — a K=4096 contraction is "deep" relative to the forward pass's
   typical K range (896–29568 for FFN, but much smaller for attention projections), which likely
   changes which tile shapes/pipeline depths perform best, and is exactly the kind of shape a
   kernel catalog built only from forward-pass reasoning would never surface.

Computed over all 29 models: 75 unique forward (N,K) shapes, +18 additional unique shapes from
`dX` not already covered by a forward shape (the two coincide whenever N=K, i.e. every square
Q/O projection — but diverge for every non-square shape: Gate/Up/Down/LMhead, and the 3 Qwen3
models with non-square Q/O), +75 structurally distinct `dW` shapes (all 75, since the batch-M
sitting in the contraction position is a different GEMM configuration even on the models where
the raw (N,K) numbers happen to coincide with an existing forward shape). **Total: 168 unique
GEMM configurations** needed for full forward+backward pretraining coverage of the 29-model dense
Qwen catalog, at the M=4096 token-batch convention (see H3.7 for why M itself is not yet a closed
question).

### H3.4 Cross-check against the old 42-shape catalog

Confirmed: **all 8 models from the old catalog are a strict subset of this redo's 29 models** —
nothing in the prior work was factually wrong, it was narrower in scope (2 generations instead of
5, forward-pass only, no backward-pass shapes at all). This redo adds 21 models the old catalog
never touched: all 4 Qwen1 sizes, all 8 Qwen1.5 sizes, all 4 Qwen2 sizes, plus the 14B/32B/72B
sizes of Qwen2.5 and the 14B/32B sizes of Qwen3 that the old work stopped short of. Findings this
redo caught that the old catalog structurally could not have, given its narrower scope: the Qwen1
FFN-halving quirk (H3.2.1 — Qwen1 was never in the old catalog at all), Qwen3-32B's non-square
Q/O (old catalog stopped at Qwen3-8B), and the full non-uniform-GQA-ratio picture across every
generation (old catalog only ever saw Qwen2.5 and Qwen3's ratios, not Qwen1/1.5/2's).

### H3.5 Master shape table — all 696 shape-instances (29 models × 8 ops × 3 pass-types), ordered by model then op then pass

Convention: (M, N, K) where `C[M,N] = A[M,K] @ B[K,N]`. FWD = forward pass. BWD-dX = input
gradient. BWD-dW = weight gradient (M-column here is the ORIGINAL forward M=4096, sitting in
the K/contraction position of this specific GEMM — see H3.3).

| # | Model | Generation | Op | Pass | M | N | K | Notes |
|---:|---|---|---|---|---:|---:|---:|---|
| 1 | Qwen1-1.8B | Qwen1 | Q | FWD | 4096 | 2048 | 2048 |  |
| 2 | Qwen1-1.8B | Qwen1 | Q | BWD-dX | 4096 | 2048 | 2048 | dX=dY@W, N/K swap |
| 3 | Qwen1-1.8B | Qwen1 | Q | BWD-dW | 2048 | 2048 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 4 | Qwen1-1.8B | Qwen1 | K | FWD | 4096 | 2048 | 2048 |  |
| 5 | Qwen1-1.8B | Qwen1 | K | BWD-dX | 4096 | 2048 | 2048 | dX=dY@W, N/K swap |
| 6 | Qwen1-1.8B | Qwen1 | K | BWD-dW | 2048 | 2048 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 7 | Qwen1-1.8B | Qwen1 | V | FWD | 4096 | 2048 | 2048 |  |
| 8 | Qwen1-1.8B | Qwen1 | V | BWD-dX | 4096 | 2048 | 2048 | dX=dY@W, N/K swap |
| 9 | Qwen1-1.8B | Qwen1 | V | BWD-dW | 2048 | 2048 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 10 | Qwen1-1.8B | Qwen1 | O | FWD | 4096 | 2048 | 2048 |  |
| 11 | Qwen1-1.8B | Qwen1 | O | BWD-dX | 4096 | 2048 | 2048 | dX=dY@W, N/K swap |
| 12 | Qwen1-1.8B | Qwen1 | O | BWD-dW | 2048 | 2048 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 13 | Qwen1-1.8B | Qwen1 | Gate | FWD | 4096 | 5504 | 2048 |  |
| 14 | Qwen1-1.8B | Qwen1 | Gate | BWD-dX | 4096 | 2048 | 5504 | dX=dY@W, N/K swap |
| 15 | Qwen1-1.8B | Qwen1 | Gate | BWD-dW | 5504 | 2048 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 16 | Qwen1-1.8B | Qwen1 | Up | FWD | 4096 | 5504 | 2048 |  |
| 17 | Qwen1-1.8B | Qwen1 | Up | BWD-dX | 4096 | 2048 | 5504 | dX=dY@W, N/K swap |
| 18 | Qwen1-1.8B | Qwen1 | Up | BWD-dW | 5504 | 2048 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 19 | Qwen1-1.8B | Qwen1 | Down | FWD | 4096 | 2048 | 5504 |  |
| 20 | Qwen1-1.8B | Qwen1 | Down | BWD-dX | 4096 | 5504 | 2048 | dX=dY@W, N/K swap |
| 21 | Qwen1-1.8B | Qwen1 | Down | BWD-dW | 2048 | 5504 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 22 | Qwen1-1.8B | Qwen1 | LMhead | FWD | 4096 | 151936 | 2048 |  |
| 23 | Qwen1-1.8B | Qwen1 | LMhead | BWD-dX | 4096 | 2048 | 151936 | dX=dY@W, N/K swap |
| 24 | Qwen1-1.8B | Qwen1 | LMhead | BWD-dW | 151936 | 2048 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 25 | Qwen1-7B | Qwen1 | Q | FWD | 4096 | 4096 | 4096 |  |
| 26 | Qwen1-7B | Qwen1 | Q | BWD-dX | 4096 | 4096 | 4096 | dX=dY@W, N/K swap |
| 27 | Qwen1-7B | Qwen1 | Q | BWD-dW | 4096 | 4096 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 28 | Qwen1-7B | Qwen1 | K | FWD | 4096 | 4096 | 4096 |  |
| 29 | Qwen1-7B | Qwen1 | K | BWD-dX | 4096 | 4096 | 4096 | dX=dY@W, N/K swap |
| 30 | Qwen1-7B | Qwen1 | K | BWD-dW | 4096 | 4096 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 31 | Qwen1-7B | Qwen1 | V | FWD | 4096 | 4096 | 4096 |  |
| 32 | Qwen1-7B | Qwen1 | V | BWD-dX | 4096 | 4096 | 4096 | dX=dY@W, N/K swap |
| 33 | Qwen1-7B | Qwen1 | V | BWD-dW | 4096 | 4096 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 34 | Qwen1-7B | Qwen1 | O | FWD | 4096 | 4096 | 4096 |  |
| 35 | Qwen1-7B | Qwen1 | O | BWD-dX | 4096 | 4096 | 4096 | dX=dY@W, N/K swap |
| 36 | Qwen1-7B | Qwen1 | O | BWD-dW | 4096 | 4096 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 37 | Qwen1-7B | Qwen1 | Gate | FWD | 4096 | 11008 | 4096 |  |
| 38 | Qwen1-7B | Qwen1 | Gate | BWD-dX | 4096 | 4096 | 11008 | dX=dY@W, N/K swap |
| 39 | Qwen1-7B | Qwen1 | Gate | BWD-dW | 11008 | 4096 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 40 | Qwen1-7B | Qwen1 | Up | FWD | 4096 | 11008 | 4096 |  |
| 41 | Qwen1-7B | Qwen1 | Up | BWD-dX | 4096 | 4096 | 11008 | dX=dY@W, N/K swap |
| 42 | Qwen1-7B | Qwen1 | Up | BWD-dW | 11008 | 4096 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 43 | Qwen1-7B | Qwen1 | Down | FWD | 4096 | 4096 | 11008 |  |
| 44 | Qwen1-7B | Qwen1 | Down | BWD-dX | 4096 | 11008 | 4096 | dX=dY@W, N/K swap |
| 45 | Qwen1-7B | Qwen1 | Down | BWD-dW | 4096 | 11008 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 46 | Qwen1-7B | Qwen1 | LMhead | FWD | 4096 | 151936 | 4096 |  |
| 47 | Qwen1-7B | Qwen1 | LMhead | BWD-dX | 4096 | 4096 | 151936 | dX=dY@W, N/K swap |
| 48 | Qwen1-7B | Qwen1 | LMhead | BWD-dW | 151936 | 4096 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 49 | Qwen1-14B | Qwen1 | Q | FWD | 4096 | 5120 | 5120 |  |
| 50 | Qwen1-14B | Qwen1 | Q | BWD-dX | 4096 | 5120 | 5120 | dX=dY@W, N/K swap |
| 51 | Qwen1-14B | Qwen1 | Q | BWD-dW | 5120 | 5120 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 52 | Qwen1-14B | Qwen1 | K | FWD | 4096 | 5120 | 5120 |  |
| 53 | Qwen1-14B | Qwen1 | K | BWD-dX | 4096 | 5120 | 5120 | dX=dY@W, N/K swap |
| 54 | Qwen1-14B | Qwen1 | K | BWD-dW | 5120 | 5120 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 55 | Qwen1-14B | Qwen1 | V | FWD | 4096 | 5120 | 5120 |  |
| 56 | Qwen1-14B | Qwen1 | V | BWD-dX | 4096 | 5120 | 5120 | dX=dY@W, N/K swap |
| 57 | Qwen1-14B | Qwen1 | V | BWD-dW | 5120 | 5120 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 58 | Qwen1-14B | Qwen1 | O | FWD | 4096 | 5120 | 5120 |  |
| 59 | Qwen1-14B | Qwen1 | O | BWD-dX | 4096 | 5120 | 5120 | dX=dY@W, N/K swap |
| 60 | Qwen1-14B | Qwen1 | O | BWD-dW | 5120 | 5120 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 61 | Qwen1-14B | Qwen1 | Gate | FWD | 4096 | 13696 | 5120 |  |
| 62 | Qwen1-14B | Qwen1 | Gate | BWD-dX | 4096 | 5120 | 13696 | dX=dY@W, N/K swap |
| 63 | Qwen1-14B | Qwen1 | Gate | BWD-dW | 13696 | 5120 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 64 | Qwen1-14B | Qwen1 | Up | FWD | 4096 | 13696 | 5120 |  |
| 65 | Qwen1-14B | Qwen1 | Up | BWD-dX | 4096 | 5120 | 13696 | dX=dY@W, N/K swap |
| 66 | Qwen1-14B | Qwen1 | Up | BWD-dW | 13696 | 5120 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 67 | Qwen1-14B | Qwen1 | Down | FWD | 4096 | 5120 | 13696 |  |
| 68 | Qwen1-14B | Qwen1 | Down | BWD-dX | 4096 | 13696 | 5120 | dX=dY@W, N/K swap |
| 69 | Qwen1-14B | Qwen1 | Down | BWD-dW | 5120 | 13696 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 70 | Qwen1-14B | Qwen1 | LMhead | FWD | 4096 | 152064 | 5120 |  |
| 71 | Qwen1-14B | Qwen1 | LMhead | BWD-dX | 4096 | 5120 | 152064 | dX=dY@W, N/K swap |
| 72 | Qwen1-14B | Qwen1 | LMhead | BWD-dW | 152064 | 5120 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 73 | Qwen1-72B | Qwen1 | Q | FWD | 4096 | 8192 | 8192 |  |
| 74 | Qwen1-72B | Qwen1 | Q | BWD-dX | 4096 | 8192 | 8192 | dX=dY@W, N/K swap |
| 75 | Qwen1-72B | Qwen1 | Q | BWD-dW | 8192 | 8192 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 76 | Qwen1-72B | Qwen1 | K | FWD | 4096 | 8192 | 8192 |  |
| 77 | Qwen1-72B | Qwen1 | K | BWD-dX | 4096 | 8192 | 8192 | dX=dY@W, N/K swap |
| 78 | Qwen1-72B | Qwen1 | K | BWD-dW | 8192 | 8192 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 79 | Qwen1-72B | Qwen1 | V | FWD | 4096 | 8192 | 8192 |  |
| 80 | Qwen1-72B | Qwen1 | V | BWD-dX | 4096 | 8192 | 8192 | dX=dY@W, N/K swap |
| 81 | Qwen1-72B | Qwen1 | V | BWD-dW | 8192 | 8192 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 82 | Qwen1-72B | Qwen1 | O | FWD | 4096 | 8192 | 8192 |  |
| 83 | Qwen1-72B | Qwen1 | O | BWD-dX | 4096 | 8192 | 8192 | dX=dY@W, N/K swap |
| 84 | Qwen1-72B | Qwen1 | O | BWD-dW | 8192 | 8192 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 85 | Qwen1-72B | Qwen1 | Gate | FWD | 4096 | 24576 | 8192 |  |
| 86 | Qwen1-72B | Qwen1 | Gate | BWD-dX | 4096 | 8192 | 24576 | dX=dY@W, N/K swap |
| 87 | Qwen1-72B | Qwen1 | Gate | BWD-dW | 24576 | 8192 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 88 | Qwen1-72B | Qwen1 | Up | FWD | 4096 | 24576 | 8192 |  |
| 89 | Qwen1-72B | Qwen1 | Up | BWD-dX | 4096 | 8192 | 24576 | dX=dY@W, N/K swap |
| 90 | Qwen1-72B | Qwen1 | Up | BWD-dW | 24576 | 8192 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 91 | Qwen1-72B | Qwen1 | Down | FWD | 4096 | 8192 | 24576 |  |
| 92 | Qwen1-72B | Qwen1 | Down | BWD-dX | 4096 | 24576 | 8192 | dX=dY@W, N/K swap |
| 93 | Qwen1-72B | Qwen1 | Down | BWD-dW | 8192 | 24576 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 94 | Qwen1-72B | Qwen1 | LMhead | FWD | 4096 | 152064 | 8192 |  |
| 95 | Qwen1-72B | Qwen1 | LMhead | BWD-dX | 4096 | 8192 | 152064 | dX=dY@W, N/K swap |
| 96 | Qwen1-72B | Qwen1 | LMhead | BWD-dW | 152064 | 8192 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 97 | Qwen1.5-0.5B | Qwen1.5 | Q | FWD | 4096 | 1024 | 1024 |  |
| 98 | Qwen1.5-0.5B | Qwen1.5 | Q | BWD-dX | 4096 | 1024 | 1024 | dX=dY@W, N/K swap |
| 99 | Qwen1.5-0.5B | Qwen1.5 | Q | BWD-dW | 1024 | 1024 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 100 | Qwen1.5-0.5B | Qwen1.5 | K | FWD | 4096 | 1024 | 1024 |  |
| 101 | Qwen1.5-0.5B | Qwen1.5 | K | BWD-dX | 4096 | 1024 | 1024 | dX=dY@W, N/K swap |
| 102 | Qwen1.5-0.5B | Qwen1.5 | K | BWD-dW | 1024 | 1024 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 103 | Qwen1.5-0.5B | Qwen1.5 | V | FWD | 4096 | 1024 | 1024 |  |
| 104 | Qwen1.5-0.5B | Qwen1.5 | V | BWD-dX | 4096 | 1024 | 1024 | dX=dY@W, N/K swap |
| 105 | Qwen1.5-0.5B | Qwen1.5 | V | BWD-dW | 1024 | 1024 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 106 | Qwen1.5-0.5B | Qwen1.5 | O | FWD | 4096 | 1024 | 1024 |  |
| 107 | Qwen1.5-0.5B | Qwen1.5 | O | BWD-dX | 4096 | 1024 | 1024 | dX=dY@W, N/K swap |
| 108 | Qwen1.5-0.5B | Qwen1.5 | O | BWD-dW | 1024 | 1024 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 109 | Qwen1.5-0.5B | Qwen1.5 | Gate | FWD | 4096 | 2816 | 1024 |  |
| 110 | Qwen1.5-0.5B | Qwen1.5 | Gate | BWD-dX | 4096 | 1024 | 2816 | dX=dY@W, N/K swap |
| 111 | Qwen1.5-0.5B | Qwen1.5 | Gate | BWD-dW | 2816 | 1024 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 112 | Qwen1.5-0.5B | Qwen1.5 | Up | FWD | 4096 | 2816 | 1024 |  |
| 113 | Qwen1.5-0.5B | Qwen1.5 | Up | BWD-dX | 4096 | 1024 | 2816 | dX=dY@W, N/K swap |
| 114 | Qwen1.5-0.5B | Qwen1.5 | Up | BWD-dW | 2816 | 1024 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 115 | Qwen1.5-0.5B | Qwen1.5 | Down | FWD | 4096 | 1024 | 2816 |  |
| 116 | Qwen1.5-0.5B | Qwen1.5 | Down | BWD-dX | 4096 | 2816 | 1024 | dX=dY@W, N/K swap |
| 117 | Qwen1.5-0.5B | Qwen1.5 | Down | BWD-dW | 1024 | 2816 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 118 | Qwen1.5-0.5B | Qwen1.5 | LMhead | FWD | 4096 | 151936 | 1024 |  |
| 119 | Qwen1.5-0.5B | Qwen1.5 | LMhead | BWD-dX | 4096 | 1024 | 151936 | dX=dY@W, N/K swap |
| 120 | Qwen1.5-0.5B | Qwen1.5 | LMhead | BWD-dW | 151936 | 1024 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 121 | Qwen1.5-1.8B | Qwen1.5 | Q | FWD | 4096 | 2048 | 2048 |  |
| 122 | Qwen1.5-1.8B | Qwen1.5 | Q | BWD-dX | 4096 | 2048 | 2048 | dX=dY@W, N/K swap |
| 123 | Qwen1.5-1.8B | Qwen1.5 | Q | BWD-dW | 2048 | 2048 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 124 | Qwen1.5-1.8B | Qwen1.5 | K | FWD | 4096 | 2048 | 2048 |  |
| 125 | Qwen1.5-1.8B | Qwen1.5 | K | BWD-dX | 4096 | 2048 | 2048 | dX=dY@W, N/K swap |
| 126 | Qwen1.5-1.8B | Qwen1.5 | K | BWD-dW | 2048 | 2048 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 127 | Qwen1.5-1.8B | Qwen1.5 | V | FWD | 4096 | 2048 | 2048 |  |
| 128 | Qwen1.5-1.8B | Qwen1.5 | V | BWD-dX | 4096 | 2048 | 2048 | dX=dY@W, N/K swap |
| 129 | Qwen1.5-1.8B | Qwen1.5 | V | BWD-dW | 2048 | 2048 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 130 | Qwen1.5-1.8B | Qwen1.5 | O | FWD | 4096 | 2048 | 2048 |  |
| 131 | Qwen1.5-1.8B | Qwen1.5 | O | BWD-dX | 4096 | 2048 | 2048 | dX=dY@W, N/K swap |
| 132 | Qwen1.5-1.8B | Qwen1.5 | O | BWD-dW | 2048 | 2048 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 133 | Qwen1.5-1.8B | Qwen1.5 | Gate | FWD | 4096 | 5504 | 2048 |  |
| 134 | Qwen1.5-1.8B | Qwen1.5 | Gate | BWD-dX | 4096 | 2048 | 5504 | dX=dY@W, N/K swap |
| 135 | Qwen1.5-1.8B | Qwen1.5 | Gate | BWD-dW | 5504 | 2048 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 136 | Qwen1.5-1.8B | Qwen1.5 | Up | FWD | 4096 | 5504 | 2048 |  |
| 137 | Qwen1.5-1.8B | Qwen1.5 | Up | BWD-dX | 4096 | 2048 | 5504 | dX=dY@W, N/K swap |
| 138 | Qwen1.5-1.8B | Qwen1.5 | Up | BWD-dW | 5504 | 2048 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 139 | Qwen1.5-1.8B | Qwen1.5 | Down | FWD | 4096 | 2048 | 5504 |  |
| 140 | Qwen1.5-1.8B | Qwen1.5 | Down | BWD-dX | 4096 | 5504 | 2048 | dX=dY@W, N/K swap |
| 141 | Qwen1.5-1.8B | Qwen1.5 | Down | BWD-dW | 2048 | 5504 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 142 | Qwen1.5-1.8B | Qwen1.5 | LMhead | FWD | 4096 | 151936 | 2048 |  |
| 143 | Qwen1.5-1.8B | Qwen1.5 | LMhead | BWD-dX | 4096 | 2048 | 151936 | dX=dY@W, N/K swap |
| 144 | Qwen1.5-1.8B | Qwen1.5 | LMhead | BWD-dW | 151936 | 2048 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 145 | Qwen1.5-4B | Qwen1.5 | Q | FWD | 4096 | 2560 | 2560 |  |
| 146 | Qwen1.5-4B | Qwen1.5 | Q | BWD-dX | 4096 | 2560 | 2560 | dX=dY@W, N/K swap |
| 147 | Qwen1.5-4B | Qwen1.5 | Q | BWD-dW | 2560 | 2560 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 148 | Qwen1.5-4B | Qwen1.5 | K | FWD | 4096 | 2560 | 2560 |  |
| 149 | Qwen1.5-4B | Qwen1.5 | K | BWD-dX | 4096 | 2560 | 2560 | dX=dY@W, N/K swap |
| 150 | Qwen1.5-4B | Qwen1.5 | K | BWD-dW | 2560 | 2560 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 151 | Qwen1.5-4B | Qwen1.5 | V | FWD | 4096 | 2560 | 2560 |  |
| 152 | Qwen1.5-4B | Qwen1.5 | V | BWD-dX | 4096 | 2560 | 2560 | dX=dY@W, N/K swap |
| 153 | Qwen1.5-4B | Qwen1.5 | V | BWD-dW | 2560 | 2560 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 154 | Qwen1.5-4B | Qwen1.5 | O | FWD | 4096 | 2560 | 2560 |  |
| 155 | Qwen1.5-4B | Qwen1.5 | O | BWD-dX | 4096 | 2560 | 2560 | dX=dY@W, N/K swap |
| 156 | Qwen1.5-4B | Qwen1.5 | O | BWD-dW | 2560 | 2560 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 157 | Qwen1.5-4B | Qwen1.5 | Gate | FWD | 4096 | 6912 | 2560 |  |
| 158 | Qwen1.5-4B | Qwen1.5 | Gate | BWD-dX | 4096 | 2560 | 6912 | dX=dY@W, N/K swap |
| 159 | Qwen1.5-4B | Qwen1.5 | Gate | BWD-dW | 6912 | 2560 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 160 | Qwen1.5-4B | Qwen1.5 | Up | FWD | 4096 | 6912 | 2560 |  |
| 161 | Qwen1.5-4B | Qwen1.5 | Up | BWD-dX | 4096 | 2560 | 6912 | dX=dY@W, N/K swap |
| 162 | Qwen1.5-4B | Qwen1.5 | Up | BWD-dW | 6912 | 2560 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 163 | Qwen1.5-4B | Qwen1.5 | Down | FWD | 4096 | 2560 | 6912 |  |
| 164 | Qwen1.5-4B | Qwen1.5 | Down | BWD-dX | 4096 | 6912 | 2560 | dX=dY@W, N/K swap |
| 165 | Qwen1.5-4B | Qwen1.5 | Down | BWD-dW | 2560 | 6912 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 166 | Qwen1.5-4B | Qwen1.5 | LMhead | FWD | 4096 | 151936 | 2560 |  |
| 167 | Qwen1.5-4B | Qwen1.5 | LMhead | BWD-dX | 4096 | 2560 | 151936 | dX=dY@W, N/K swap |
| 168 | Qwen1.5-4B | Qwen1.5 | LMhead | BWD-dW | 151936 | 2560 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 169 | Qwen1.5-7B | Qwen1.5 | Q | FWD | 4096 | 4096 | 4096 |  |
| 170 | Qwen1.5-7B | Qwen1.5 | Q | BWD-dX | 4096 | 4096 | 4096 | dX=dY@W, N/K swap |
| 171 | Qwen1.5-7B | Qwen1.5 | Q | BWD-dW | 4096 | 4096 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 172 | Qwen1.5-7B | Qwen1.5 | K | FWD | 4096 | 4096 | 4096 |  |
| 173 | Qwen1.5-7B | Qwen1.5 | K | BWD-dX | 4096 | 4096 | 4096 | dX=dY@W, N/K swap |
| 174 | Qwen1.5-7B | Qwen1.5 | K | BWD-dW | 4096 | 4096 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 175 | Qwen1.5-7B | Qwen1.5 | V | FWD | 4096 | 4096 | 4096 |  |
| 176 | Qwen1.5-7B | Qwen1.5 | V | BWD-dX | 4096 | 4096 | 4096 | dX=dY@W, N/K swap |
| 177 | Qwen1.5-7B | Qwen1.5 | V | BWD-dW | 4096 | 4096 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 178 | Qwen1.5-7B | Qwen1.5 | O | FWD | 4096 | 4096 | 4096 |  |
| 179 | Qwen1.5-7B | Qwen1.5 | O | BWD-dX | 4096 | 4096 | 4096 | dX=dY@W, N/K swap |
| 180 | Qwen1.5-7B | Qwen1.5 | O | BWD-dW | 4096 | 4096 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 181 | Qwen1.5-7B | Qwen1.5 | Gate | FWD | 4096 | 11008 | 4096 |  |
| 182 | Qwen1.5-7B | Qwen1.5 | Gate | BWD-dX | 4096 | 4096 | 11008 | dX=dY@W, N/K swap |
| 183 | Qwen1.5-7B | Qwen1.5 | Gate | BWD-dW | 11008 | 4096 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 184 | Qwen1.5-7B | Qwen1.5 | Up | FWD | 4096 | 11008 | 4096 |  |
| 185 | Qwen1.5-7B | Qwen1.5 | Up | BWD-dX | 4096 | 4096 | 11008 | dX=dY@W, N/K swap |
| 186 | Qwen1.5-7B | Qwen1.5 | Up | BWD-dW | 11008 | 4096 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 187 | Qwen1.5-7B | Qwen1.5 | Down | FWD | 4096 | 4096 | 11008 |  |
| 188 | Qwen1.5-7B | Qwen1.5 | Down | BWD-dX | 4096 | 11008 | 4096 | dX=dY@W, N/K swap |
| 189 | Qwen1.5-7B | Qwen1.5 | Down | BWD-dW | 4096 | 11008 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 190 | Qwen1.5-7B | Qwen1.5 | LMhead | FWD | 4096 | 151936 | 4096 |  |
| 191 | Qwen1.5-7B | Qwen1.5 | LMhead | BWD-dX | 4096 | 4096 | 151936 | dX=dY@W, N/K swap |
| 192 | Qwen1.5-7B | Qwen1.5 | LMhead | BWD-dW | 151936 | 4096 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 193 | Qwen1.5-14B | Qwen1.5 | Q | FWD | 4096 | 5120 | 5120 |  |
| 194 | Qwen1.5-14B | Qwen1.5 | Q | BWD-dX | 4096 | 5120 | 5120 | dX=dY@W, N/K swap |
| 195 | Qwen1.5-14B | Qwen1.5 | Q | BWD-dW | 5120 | 5120 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 196 | Qwen1.5-14B | Qwen1.5 | K | FWD | 4096 | 5120 | 5120 |  |
| 197 | Qwen1.5-14B | Qwen1.5 | K | BWD-dX | 4096 | 5120 | 5120 | dX=dY@W, N/K swap |
| 198 | Qwen1.5-14B | Qwen1.5 | K | BWD-dW | 5120 | 5120 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 199 | Qwen1.5-14B | Qwen1.5 | V | FWD | 4096 | 5120 | 5120 |  |
| 200 | Qwen1.5-14B | Qwen1.5 | V | BWD-dX | 4096 | 5120 | 5120 | dX=dY@W, N/K swap |
| 201 | Qwen1.5-14B | Qwen1.5 | V | BWD-dW | 5120 | 5120 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 202 | Qwen1.5-14B | Qwen1.5 | O | FWD | 4096 | 5120 | 5120 |  |
| 203 | Qwen1.5-14B | Qwen1.5 | O | BWD-dX | 4096 | 5120 | 5120 | dX=dY@W, N/K swap |
| 204 | Qwen1.5-14B | Qwen1.5 | O | BWD-dW | 5120 | 5120 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 205 | Qwen1.5-14B | Qwen1.5 | Gate | FWD | 4096 | 13696 | 5120 |  |
| 206 | Qwen1.5-14B | Qwen1.5 | Gate | BWD-dX | 4096 | 5120 | 13696 | dX=dY@W, N/K swap |
| 207 | Qwen1.5-14B | Qwen1.5 | Gate | BWD-dW | 13696 | 5120 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 208 | Qwen1.5-14B | Qwen1.5 | Up | FWD | 4096 | 13696 | 5120 |  |
| 209 | Qwen1.5-14B | Qwen1.5 | Up | BWD-dX | 4096 | 5120 | 13696 | dX=dY@W, N/K swap |
| 210 | Qwen1.5-14B | Qwen1.5 | Up | BWD-dW | 13696 | 5120 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 211 | Qwen1.5-14B | Qwen1.5 | Down | FWD | 4096 | 5120 | 13696 |  |
| 212 | Qwen1.5-14B | Qwen1.5 | Down | BWD-dX | 4096 | 13696 | 5120 | dX=dY@W, N/K swap |
| 213 | Qwen1.5-14B | Qwen1.5 | Down | BWD-dW | 5120 | 13696 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 214 | Qwen1.5-14B | Qwen1.5 | LMhead | FWD | 4096 | 152064 | 5120 |  |
| 215 | Qwen1.5-14B | Qwen1.5 | LMhead | BWD-dX | 4096 | 5120 | 152064 | dX=dY@W, N/K swap |
| 216 | Qwen1.5-14B | Qwen1.5 | LMhead | BWD-dW | 152064 | 5120 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 217 | Qwen1.5-32B | Qwen1.5 | Q | FWD | 4096 | 5120 | 5120 |  |
| 218 | Qwen1.5-32B | Qwen1.5 | Q | BWD-dX | 4096 | 5120 | 5120 | dX=dY@W, N/K swap |
| 219 | Qwen1.5-32B | Qwen1.5 | Q | BWD-dW | 5120 | 5120 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 220 | Qwen1.5-32B | Qwen1.5 | K | FWD | 4096 | 1024 | 5120 |  |
| 221 | Qwen1.5-32B | Qwen1.5 | K | BWD-dX | 4096 | 5120 | 1024 | dX=dY@W, N/K swap |
| 222 | Qwen1.5-32B | Qwen1.5 | K | BWD-dW | 1024 | 5120 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 223 | Qwen1.5-32B | Qwen1.5 | V | FWD | 4096 | 1024 | 5120 |  |
| 224 | Qwen1.5-32B | Qwen1.5 | V | BWD-dX | 4096 | 5120 | 1024 | dX=dY@W, N/K swap |
| 225 | Qwen1.5-32B | Qwen1.5 | V | BWD-dW | 1024 | 5120 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 226 | Qwen1.5-32B | Qwen1.5 | O | FWD | 4096 | 5120 | 5120 |  |
| 227 | Qwen1.5-32B | Qwen1.5 | O | BWD-dX | 4096 | 5120 | 5120 | dX=dY@W, N/K swap |
| 228 | Qwen1.5-32B | Qwen1.5 | O | BWD-dW | 5120 | 5120 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 229 | Qwen1.5-32B | Qwen1.5 | Gate | FWD | 4096 | 27392 | 5120 |  |
| 230 | Qwen1.5-32B | Qwen1.5 | Gate | BWD-dX | 4096 | 5120 | 27392 | dX=dY@W, N/K swap |
| 231 | Qwen1.5-32B | Qwen1.5 | Gate | BWD-dW | 27392 | 5120 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 232 | Qwen1.5-32B | Qwen1.5 | Up | FWD | 4096 | 27392 | 5120 |  |
| 233 | Qwen1.5-32B | Qwen1.5 | Up | BWD-dX | 4096 | 5120 | 27392 | dX=dY@W, N/K swap |
| 234 | Qwen1.5-32B | Qwen1.5 | Up | BWD-dW | 27392 | 5120 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 235 | Qwen1.5-32B | Qwen1.5 | Down | FWD | 4096 | 5120 | 27392 |  |
| 236 | Qwen1.5-32B | Qwen1.5 | Down | BWD-dX | 4096 | 27392 | 5120 | dX=dY@W, N/K swap |
| 237 | Qwen1.5-32B | Qwen1.5 | Down | BWD-dW | 5120 | 27392 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 238 | Qwen1.5-32B | Qwen1.5 | LMhead | FWD | 4096 | 152064 | 5120 |  |
| 239 | Qwen1.5-32B | Qwen1.5 | LMhead | BWD-dX | 4096 | 5120 | 152064 | dX=dY@W, N/K swap |
| 240 | Qwen1.5-32B | Qwen1.5 | LMhead | BWD-dW | 152064 | 5120 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 241 | Qwen1.5-72B | Qwen1.5 | Q | FWD | 4096 | 8192 | 8192 |  |
| 242 | Qwen1.5-72B | Qwen1.5 | Q | BWD-dX | 4096 | 8192 | 8192 | dX=dY@W, N/K swap |
| 243 | Qwen1.5-72B | Qwen1.5 | Q | BWD-dW | 8192 | 8192 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 244 | Qwen1.5-72B | Qwen1.5 | K | FWD | 4096 | 8192 | 8192 |  |
| 245 | Qwen1.5-72B | Qwen1.5 | K | BWD-dX | 4096 | 8192 | 8192 | dX=dY@W, N/K swap |
| 246 | Qwen1.5-72B | Qwen1.5 | K | BWD-dW | 8192 | 8192 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 247 | Qwen1.5-72B | Qwen1.5 | V | FWD | 4096 | 8192 | 8192 |  |
| 248 | Qwen1.5-72B | Qwen1.5 | V | BWD-dX | 4096 | 8192 | 8192 | dX=dY@W, N/K swap |
| 249 | Qwen1.5-72B | Qwen1.5 | V | BWD-dW | 8192 | 8192 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 250 | Qwen1.5-72B | Qwen1.5 | O | FWD | 4096 | 8192 | 8192 |  |
| 251 | Qwen1.5-72B | Qwen1.5 | O | BWD-dX | 4096 | 8192 | 8192 | dX=dY@W, N/K swap |
| 252 | Qwen1.5-72B | Qwen1.5 | O | BWD-dW | 8192 | 8192 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 253 | Qwen1.5-72B | Qwen1.5 | Gate | FWD | 4096 | 24576 | 8192 |  |
| 254 | Qwen1.5-72B | Qwen1.5 | Gate | BWD-dX | 4096 | 8192 | 24576 | dX=dY@W, N/K swap |
| 255 | Qwen1.5-72B | Qwen1.5 | Gate | BWD-dW | 24576 | 8192 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 256 | Qwen1.5-72B | Qwen1.5 | Up | FWD | 4096 | 24576 | 8192 |  |
| 257 | Qwen1.5-72B | Qwen1.5 | Up | BWD-dX | 4096 | 8192 | 24576 | dX=dY@W, N/K swap |
| 258 | Qwen1.5-72B | Qwen1.5 | Up | BWD-dW | 24576 | 8192 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 259 | Qwen1.5-72B | Qwen1.5 | Down | FWD | 4096 | 8192 | 24576 |  |
| 260 | Qwen1.5-72B | Qwen1.5 | Down | BWD-dX | 4096 | 24576 | 8192 | dX=dY@W, N/K swap |
| 261 | Qwen1.5-72B | Qwen1.5 | Down | BWD-dW | 8192 | 24576 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 262 | Qwen1.5-72B | Qwen1.5 | LMhead | FWD | 4096 | 152064 | 8192 |  |
| 263 | Qwen1.5-72B | Qwen1.5 | LMhead | BWD-dX | 4096 | 8192 | 152064 | dX=dY@W, N/K swap |
| 264 | Qwen1.5-72B | Qwen1.5 | LMhead | BWD-dW | 152064 | 8192 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 265 | Qwen1.5-110B | Qwen1.5 | Q | FWD | 4096 | 8192 | 8192 |  |
| 266 | Qwen1.5-110B | Qwen1.5 | Q | BWD-dX | 4096 | 8192 | 8192 | dX=dY@W, N/K swap |
| 267 | Qwen1.5-110B | Qwen1.5 | Q | BWD-dW | 8192 | 8192 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 268 | Qwen1.5-110B | Qwen1.5 | K | FWD | 4096 | 1024 | 8192 |  |
| 269 | Qwen1.5-110B | Qwen1.5 | K | BWD-dX | 4096 | 8192 | 1024 | dX=dY@W, N/K swap |
| 270 | Qwen1.5-110B | Qwen1.5 | K | BWD-dW | 1024 | 8192 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 271 | Qwen1.5-110B | Qwen1.5 | V | FWD | 4096 | 1024 | 8192 |  |
| 272 | Qwen1.5-110B | Qwen1.5 | V | BWD-dX | 4096 | 8192 | 1024 | dX=dY@W, N/K swap |
| 273 | Qwen1.5-110B | Qwen1.5 | V | BWD-dW | 1024 | 8192 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 274 | Qwen1.5-110B | Qwen1.5 | O | FWD | 4096 | 8192 | 8192 |  |
| 275 | Qwen1.5-110B | Qwen1.5 | O | BWD-dX | 4096 | 8192 | 8192 | dX=dY@W, N/K swap |
| 276 | Qwen1.5-110B | Qwen1.5 | O | BWD-dW | 8192 | 8192 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 277 | Qwen1.5-110B | Qwen1.5 | Gate | FWD | 4096 | 49152 | 8192 |  |
| 278 | Qwen1.5-110B | Qwen1.5 | Gate | BWD-dX | 4096 | 8192 | 49152 | dX=dY@W, N/K swap |
| 279 | Qwen1.5-110B | Qwen1.5 | Gate | BWD-dW | 49152 | 8192 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 280 | Qwen1.5-110B | Qwen1.5 | Up | FWD | 4096 | 49152 | 8192 |  |
| 281 | Qwen1.5-110B | Qwen1.5 | Up | BWD-dX | 4096 | 8192 | 49152 | dX=dY@W, N/K swap |
| 282 | Qwen1.5-110B | Qwen1.5 | Up | BWD-dW | 49152 | 8192 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 283 | Qwen1.5-110B | Qwen1.5 | Down | FWD | 4096 | 8192 | 49152 |  |
| 284 | Qwen1.5-110B | Qwen1.5 | Down | BWD-dX | 4096 | 49152 | 8192 | dX=dY@W, N/K swap |
| 285 | Qwen1.5-110B | Qwen1.5 | Down | BWD-dW | 8192 | 49152 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 286 | Qwen1.5-110B | Qwen1.5 | LMhead | FWD | 4096 | 152064 | 8192 |  |
| 287 | Qwen1.5-110B | Qwen1.5 | LMhead | BWD-dX | 4096 | 8192 | 152064 | dX=dY@W, N/K swap |
| 288 | Qwen1.5-110B | Qwen1.5 | LMhead | BWD-dW | 152064 | 8192 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 289 | Qwen2-0.5B | Qwen2 | Q | FWD | 4096 | 896 | 896 |  |
| 290 | Qwen2-0.5B | Qwen2 | Q | BWD-dX | 4096 | 896 | 896 | dX=dY@W, N/K swap |
| 291 | Qwen2-0.5B | Qwen2 | Q | BWD-dW | 896 | 896 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 292 | Qwen2-0.5B | Qwen2 | K | FWD | 4096 | 128 | 896 |  |
| 293 | Qwen2-0.5B | Qwen2 | K | BWD-dX | 4096 | 896 | 128 | dX=dY@W, N/K swap |
| 294 | Qwen2-0.5B | Qwen2 | K | BWD-dW | 128 | 896 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 295 | Qwen2-0.5B | Qwen2 | V | FWD | 4096 | 128 | 896 |  |
| 296 | Qwen2-0.5B | Qwen2 | V | BWD-dX | 4096 | 896 | 128 | dX=dY@W, N/K swap |
| 297 | Qwen2-0.5B | Qwen2 | V | BWD-dW | 128 | 896 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 298 | Qwen2-0.5B | Qwen2 | O | FWD | 4096 | 896 | 896 |  |
| 299 | Qwen2-0.5B | Qwen2 | O | BWD-dX | 4096 | 896 | 896 | dX=dY@W, N/K swap |
| 300 | Qwen2-0.5B | Qwen2 | O | BWD-dW | 896 | 896 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 301 | Qwen2-0.5B | Qwen2 | Gate | FWD | 4096 | 4864 | 896 |  |
| 302 | Qwen2-0.5B | Qwen2 | Gate | BWD-dX | 4096 | 896 | 4864 | dX=dY@W, N/K swap |
| 303 | Qwen2-0.5B | Qwen2 | Gate | BWD-dW | 4864 | 896 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 304 | Qwen2-0.5B | Qwen2 | Up | FWD | 4096 | 4864 | 896 |  |
| 305 | Qwen2-0.5B | Qwen2 | Up | BWD-dX | 4096 | 896 | 4864 | dX=dY@W, N/K swap |
| 306 | Qwen2-0.5B | Qwen2 | Up | BWD-dW | 4864 | 896 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 307 | Qwen2-0.5B | Qwen2 | Down | FWD | 4096 | 896 | 4864 |  |
| 308 | Qwen2-0.5B | Qwen2 | Down | BWD-dX | 4096 | 4864 | 896 | dX=dY@W, N/K swap |
| 309 | Qwen2-0.5B | Qwen2 | Down | BWD-dW | 896 | 4864 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 310 | Qwen2-0.5B | Qwen2 | LMhead | FWD | 4096 | 151936 | 896 |  |
| 311 | Qwen2-0.5B | Qwen2 | LMhead | BWD-dX | 4096 | 896 | 151936 | dX=dY@W, N/K swap |
| 312 | Qwen2-0.5B | Qwen2 | LMhead | BWD-dW | 151936 | 896 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 313 | Qwen2-1.5B | Qwen2 | Q | FWD | 4096 | 1536 | 1536 |  |
| 314 | Qwen2-1.5B | Qwen2 | Q | BWD-dX | 4096 | 1536 | 1536 | dX=dY@W, N/K swap |
| 315 | Qwen2-1.5B | Qwen2 | Q | BWD-dW | 1536 | 1536 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 316 | Qwen2-1.5B | Qwen2 | K | FWD | 4096 | 256 | 1536 |  |
| 317 | Qwen2-1.5B | Qwen2 | K | BWD-dX | 4096 | 1536 | 256 | dX=dY@W, N/K swap |
| 318 | Qwen2-1.5B | Qwen2 | K | BWD-dW | 256 | 1536 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 319 | Qwen2-1.5B | Qwen2 | V | FWD | 4096 | 256 | 1536 |  |
| 320 | Qwen2-1.5B | Qwen2 | V | BWD-dX | 4096 | 1536 | 256 | dX=dY@W, N/K swap |
| 321 | Qwen2-1.5B | Qwen2 | V | BWD-dW | 256 | 1536 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 322 | Qwen2-1.5B | Qwen2 | O | FWD | 4096 | 1536 | 1536 |  |
| 323 | Qwen2-1.5B | Qwen2 | O | BWD-dX | 4096 | 1536 | 1536 | dX=dY@W, N/K swap |
| 324 | Qwen2-1.5B | Qwen2 | O | BWD-dW | 1536 | 1536 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 325 | Qwen2-1.5B | Qwen2 | Gate | FWD | 4096 | 8960 | 1536 |  |
| 326 | Qwen2-1.5B | Qwen2 | Gate | BWD-dX | 4096 | 1536 | 8960 | dX=dY@W, N/K swap |
| 327 | Qwen2-1.5B | Qwen2 | Gate | BWD-dW | 8960 | 1536 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 328 | Qwen2-1.5B | Qwen2 | Up | FWD | 4096 | 8960 | 1536 |  |
| 329 | Qwen2-1.5B | Qwen2 | Up | BWD-dX | 4096 | 1536 | 8960 | dX=dY@W, N/K swap |
| 330 | Qwen2-1.5B | Qwen2 | Up | BWD-dW | 8960 | 1536 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 331 | Qwen2-1.5B | Qwen2 | Down | FWD | 4096 | 1536 | 8960 |  |
| 332 | Qwen2-1.5B | Qwen2 | Down | BWD-dX | 4096 | 8960 | 1536 | dX=dY@W, N/K swap |
| 333 | Qwen2-1.5B | Qwen2 | Down | BWD-dW | 1536 | 8960 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 334 | Qwen2-1.5B | Qwen2 | LMhead | FWD | 4096 | 151936 | 1536 |  |
| 335 | Qwen2-1.5B | Qwen2 | LMhead | BWD-dX | 4096 | 1536 | 151936 | dX=dY@W, N/K swap |
| 336 | Qwen2-1.5B | Qwen2 | LMhead | BWD-dW | 151936 | 1536 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 337 | Qwen2-7B | Qwen2 | Q | FWD | 4096 | 3584 | 3584 |  |
| 338 | Qwen2-7B | Qwen2 | Q | BWD-dX | 4096 | 3584 | 3584 | dX=dY@W, N/K swap |
| 339 | Qwen2-7B | Qwen2 | Q | BWD-dW | 3584 | 3584 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 340 | Qwen2-7B | Qwen2 | K | FWD | 4096 | 512 | 3584 |  |
| 341 | Qwen2-7B | Qwen2 | K | BWD-dX | 4096 | 3584 | 512 | dX=dY@W, N/K swap |
| 342 | Qwen2-7B | Qwen2 | K | BWD-dW | 512 | 3584 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 343 | Qwen2-7B | Qwen2 | V | FWD | 4096 | 512 | 3584 |  |
| 344 | Qwen2-7B | Qwen2 | V | BWD-dX | 4096 | 3584 | 512 | dX=dY@W, N/K swap |
| 345 | Qwen2-7B | Qwen2 | V | BWD-dW | 512 | 3584 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 346 | Qwen2-7B | Qwen2 | O | FWD | 4096 | 3584 | 3584 |  |
| 347 | Qwen2-7B | Qwen2 | O | BWD-dX | 4096 | 3584 | 3584 | dX=dY@W, N/K swap |
| 348 | Qwen2-7B | Qwen2 | O | BWD-dW | 3584 | 3584 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 349 | Qwen2-7B | Qwen2 | Gate | FWD | 4096 | 18944 | 3584 |  |
| 350 | Qwen2-7B | Qwen2 | Gate | BWD-dX | 4096 | 3584 | 18944 | dX=dY@W, N/K swap |
| 351 | Qwen2-7B | Qwen2 | Gate | BWD-dW | 18944 | 3584 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 352 | Qwen2-7B | Qwen2 | Up | FWD | 4096 | 18944 | 3584 |  |
| 353 | Qwen2-7B | Qwen2 | Up | BWD-dX | 4096 | 3584 | 18944 | dX=dY@W, N/K swap |
| 354 | Qwen2-7B | Qwen2 | Up | BWD-dW | 18944 | 3584 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 355 | Qwen2-7B | Qwen2 | Down | FWD | 4096 | 3584 | 18944 |  |
| 356 | Qwen2-7B | Qwen2 | Down | BWD-dX | 4096 | 18944 | 3584 | dX=dY@W, N/K swap |
| 357 | Qwen2-7B | Qwen2 | Down | BWD-dW | 3584 | 18944 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 358 | Qwen2-7B | Qwen2 | LMhead | FWD | 4096 | 152064 | 3584 |  |
| 359 | Qwen2-7B | Qwen2 | LMhead | BWD-dX | 4096 | 3584 | 152064 | dX=dY@W, N/K swap |
| 360 | Qwen2-7B | Qwen2 | LMhead | BWD-dW | 152064 | 3584 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 361 | Qwen2-72B | Qwen2 | Q | FWD | 4096 | 8192 | 8192 |  |
| 362 | Qwen2-72B | Qwen2 | Q | BWD-dX | 4096 | 8192 | 8192 | dX=dY@W, N/K swap |
| 363 | Qwen2-72B | Qwen2 | Q | BWD-dW | 8192 | 8192 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 364 | Qwen2-72B | Qwen2 | K | FWD | 4096 | 1024 | 8192 |  |
| 365 | Qwen2-72B | Qwen2 | K | BWD-dX | 4096 | 8192 | 1024 | dX=dY@W, N/K swap |
| 366 | Qwen2-72B | Qwen2 | K | BWD-dW | 1024 | 8192 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 367 | Qwen2-72B | Qwen2 | V | FWD | 4096 | 1024 | 8192 |  |
| 368 | Qwen2-72B | Qwen2 | V | BWD-dX | 4096 | 8192 | 1024 | dX=dY@W, N/K swap |
| 369 | Qwen2-72B | Qwen2 | V | BWD-dW | 1024 | 8192 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 370 | Qwen2-72B | Qwen2 | O | FWD | 4096 | 8192 | 8192 |  |
| 371 | Qwen2-72B | Qwen2 | O | BWD-dX | 4096 | 8192 | 8192 | dX=dY@W, N/K swap |
| 372 | Qwen2-72B | Qwen2 | O | BWD-dW | 8192 | 8192 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 373 | Qwen2-72B | Qwen2 | Gate | FWD | 4096 | 29568 | 8192 |  |
| 374 | Qwen2-72B | Qwen2 | Gate | BWD-dX | 4096 | 8192 | 29568 | dX=dY@W, N/K swap |
| 375 | Qwen2-72B | Qwen2 | Gate | BWD-dW | 29568 | 8192 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 376 | Qwen2-72B | Qwen2 | Up | FWD | 4096 | 29568 | 8192 |  |
| 377 | Qwen2-72B | Qwen2 | Up | BWD-dX | 4096 | 8192 | 29568 | dX=dY@W, N/K swap |
| 378 | Qwen2-72B | Qwen2 | Up | BWD-dW | 29568 | 8192 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 379 | Qwen2-72B | Qwen2 | Down | FWD | 4096 | 8192 | 29568 |  |
| 380 | Qwen2-72B | Qwen2 | Down | BWD-dX | 4096 | 29568 | 8192 | dX=dY@W, N/K swap |
| 381 | Qwen2-72B | Qwen2 | Down | BWD-dW | 8192 | 29568 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 382 | Qwen2-72B | Qwen2 | LMhead | FWD | 4096 | 152064 | 8192 |  |
| 383 | Qwen2-72B | Qwen2 | LMhead | BWD-dX | 4096 | 8192 | 152064 | dX=dY@W, N/K swap |
| 384 | Qwen2-72B | Qwen2 | LMhead | BWD-dW | 152064 | 8192 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 385 | Qwen2.5-0.5B | Qwen2.5 | Q | FWD | 4096 | 896 | 896 |  |
| 386 | Qwen2.5-0.5B | Qwen2.5 | Q | BWD-dX | 4096 | 896 | 896 | dX=dY@W, N/K swap |
| 387 | Qwen2.5-0.5B | Qwen2.5 | Q | BWD-dW | 896 | 896 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 388 | Qwen2.5-0.5B | Qwen2.5 | K | FWD | 4096 | 128 | 896 |  |
| 389 | Qwen2.5-0.5B | Qwen2.5 | K | BWD-dX | 4096 | 896 | 128 | dX=dY@W, N/K swap |
| 390 | Qwen2.5-0.5B | Qwen2.5 | K | BWD-dW | 128 | 896 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 391 | Qwen2.5-0.5B | Qwen2.5 | V | FWD | 4096 | 128 | 896 |  |
| 392 | Qwen2.5-0.5B | Qwen2.5 | V | BWD-dX | 4096 | 896 | 128 | dX=dY@W, N/K swap |
| 393 | Qwen2.5-0.5B | Qwen2.5 | V | BWD-dW | 128 | 896 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 394 | Qwen2.5-0.5B | Qwen2.5 | O | FWD | 4096 | 896 | 896 |  |
| 395 | Qwen2.5-0.5B | Qwen2.5 | O | BWD-dX | 4096 | 896 | 896 | dX=dY@W, N/K swap |
| 396 | Qwen2.5-0.5B | Qwen2.5 | O | BWD-dW | 896 | 896 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 397 | Qwen2.5-0.5B | Qwen2.5 | Gate | FWD | 4096 | 4864 | 896 |  |
| 398 | Qwen2.5-0.5B | Qwen2.5 | Gate | BWD-dX | 4096 | 896 | 4864 | dX=dY@W, N/K swap |
| 399 | Qwen2.5-0.5B | Qwen2.5 | Gate | BWD-dW | 4864 | 896 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 400 | Qwen2.5-0.5B | Qwen2.5 | Up | FWD | 4096 | 4864 | 896 |  |
| 401 | Qwen2.5-0.5B | Qwen2.5 | Up | BWD-dX | 4096 | 896 | 4864 | dX=dY@W, N/K swap |
| 402 | Qwen2.5-0.5B | Qwen2.5 | Up | BWD-dW | 4864 | 896 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 403 | Qwen2.5-0.5B | Qwen2.5 | Down | FWD | 4096 | 896 | 4864 |  |
| 404 | Qwen2.5-0.5B | Qwen2.5 | Down | BWD-dX | 4096 | 4864 | 896 | dX=dY@W, N/K swap |
| 405 | Qwen2.5-0.5B | Qwen2.5 | Down | BWD-dW | 896 | 4864 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 406 | Qwen2.5-0.5B | Qwen2.5 | LMhead | FWD | 4096 | 151936 | 896 |  |
| 407 | Qwen2.5-0.5B | Qwen2.5 | LMhead | BWD-dX | 4096 | 896 | 151936 | dX=dY@W, N/K swap |
| 408 | Qwen2.5-0.5B | Qwen2.5 | LMhead | BWD-dW | 151936 | 896 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 409 | Qwen2.5-1.5B | Qwen2.5 | Q | FWD | 4096 | 1536 | 1536 |  |
| 410 | Qwen2.5-1.5B | Qwen2.5 | Q | BWD-dX | 4096 | 1536 | 1536 | dX=dY@W, N/K swap |
| 411 | Qwen2.5-1.5B | Qwen2.5 | Q | BWD-dW | 1536 | 1536 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 412 | Qwen2.5-1.5B | Qwen2.5 | K | FWD | 4096 | 256 | 1536 |  |
| 413 | Qwen2.5-1.5B | Qwen2.5 | K | BWD-dX | 4096 | 1536 | 256 | dX=dY@W, N/K swap |
| 414 | Qwen2.5-1.5B | Qwen2.5 | K | BWD-dW | 256 | 1536 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 415 | Qwen2.5-1.5B | Qwen2.5 | V | FWD | 4096 | 256 | 1536 |  |
| 416 | Qwen2.5-1.5B | Qwen2.5 | V | BWD-dX | 4096 | 1536 | 256 | dX=dY@W, N/K swap |
| 417 | Qwen2.5-1.5B | Qwen2.5 | V | BWD-dW | 256 | 1536 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 418 | Qwen2.5-1.5B | Qwen2.5 | O | FWD | 4096 | 1536 | 1536 |  |
| 419 | Qwen2.5-1.5B | Qwen2.5 | O | BWD-dX | 4096 | 1536 | 1536 | dX=dY@W, N/K swap |
| 420 | Qwen2.5-1.5B | Qwen2.5 | O | BWD-dW | 1536 | 1536 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 421 | Qwen2.5-1.5B | Qwen2.5 | Gate | FWD | 4096 | 8960 | 1536 |  |
| 422 | Qwen2.5-1.5B | Qwen2.5 | Gate | BWD-dX | 4096 | 1536 | 8960 | dX=dY@W, N/K swap |
| 423 | Qwen2.5-1.5B | Qwen2.5 | Gate | BWD-dW | 8960 | 1536 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 424 | Qwen2.5-1.5B | Qwen2.5 | Up | FWD | 4096 | 8960 | 1536 |  |
| 425 | Qwen2.5-1.5B | Qwen2.5 | Up | BWD-dX | 4096 | 1536 | 8960 | dX=dY@W, N/K swap |
| 426 | Qwen2.5-1.5B | Qwen2.5 | Up | BWD-dW | 8960 | 1536 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 427 | Qwen2.5-1.5B | Qwen2.5 | Down | FWD | 4096 | 1536 | 8960 |  |
| 428 | Qwen2.5-1.5B | Qwen2.5 | Down | BWD-dX | 4096 | 8960 | 1536 | dX=dY@W, N/K swap |
| 429 | Qwen2.5-1.5B | Qwen2.5 | Down | BWD-dW | 1536 | 8960 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 430 | Qwen2.5-1.5B | Qwen2.5 | LMhead | FWD | 4096 | 151936 | 1536 |  |
| 431 | Qwen2.5-1.5B | Qwen2.5 | LMhead | BWD-dX | 4096 | 1536 | 151936 | dX=dY@W, N/K swap |
| 432 | Qwen2.5-1.5B | Qwen2.5 | LMhead | BWD-dW | 151936 | 1536 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 433 | Qwen2.5-3B | Qwen2.5 | Q | FWD | 4096 | 2048 | 2048 |  |
| 434 | Qwen2.5-3B | Qwen2.5 | Q | BWD-dX | 4096 | 2048 | 2048 | dX=dY@W, N/K swap |
| 435 | Qwen2.5-3B | Qwen2.5 | Q | BWD-dW | 2048 | 2048 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 436 | Qwen2.5-3B | Qwen2.5 | K | FWD | 4096 | 256 | 2048 |  |
| 437 | Qwen2.5-3B | Qwen2.5 | K | BWD-dX | 4096 | 2048 | 256 | dX=dY@W, N/K swap |
| 438 | Qwen2.5-3B | Qwen2.5 | K | BWD-dW | 256 | 2048 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 439 | Qwen2.5-3B | Qwen2.5 | V | FWD | 4096 | 256 | 2048 |  |
| 440 | Qwen2.5-3B | Qwen2.5 | V | BWD-dX | 4096 | 2048 | 256 | dX=dY@W, N/K swap |
| 441 | Qwen2.5-3B | Qwen2.5 | V | BWD-dW | 256 | 2048 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 442 | Qwen2.5-3B | Qwen2.5 | O | FWD | 4096 | 2048 | 2048 |  |
| 443 | Qwen2.5-3B | Qwen2.5 | O | BWD-dX | 4096 | 2048 | 2048 | dX=dY@W, N/K swap |
| 444 | Qwen2.5-3B | Qwen2.5 | O | BWD-dW | 2048 | 2048 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 445 | Qwen2.5-3B | Qwen2.5 | Gate | FWD | 4096 | 11008 | 2048 |  |
| 446 | Qwen2.5-3B | Qwen2.5 | Gate | BWD-dX | 4096 | 2048 | 11008 | dX=dY@W, N/K swap |
| 447 | Qwen2.5-3B | Qwen2.5 | Gate | BWD-dW | 11008 | 2048 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 448 | Qwen2.5-3B | Qwen2.5 | Up | FWD | 4096 | 11008 | 2048 |  |
| 449 | Qwen2.5-3B | Qwen2.5 | Up | BWD-dX | 4096 | 2048 | 11008 | dX=dY@W, N/K swap |
| 450 | Qwen2.5-3B | Qwen2.5 | Up | BWD-dW | 11008 | 2048 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 451 | Qwen2.5-3B | Qwen2.5 | Down | FWD | 4096 | 2048 | 11008 |  |
| 452 | Qwen2.5-3B | Qwen2.5 | Down | BWD-dX | 4096 | 11008 | 2048 | dX=dY@W, N/K swap |
| 453 | Qwen2.5-3B | Qwen2.5 | Down | BWD-dW | 2048 | 11008 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 454 | Qwen2.5-3B | Qwen2.5 | LMhead | FWD | 4096 | 151936 | 2048 |  |
| 455 | Qwen2.5-3B | Qwen2.5 | LMhead | BWD-dX | 4096 | 2048 | 151936 | dX=dY@W, N/K swap |
| 456 | Qwen2.5-3B | Qwen2.5 | LMhead | BWD-dW | 151936 | 2048 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 457 | Qwen2.5-7B | Qwen2.5 | Q | FWD | 4096 | 3584 | 3584 |  |
| 458 | Qwen2.5-7B | Qwen2.5 | Q | BWD-dX | 4096 | 3584 | 3584 | dX=dY@W, N/K swap |
| 459 | Qwen2.5-7B | Qwen2.5 | Q | BWD-dW | 3584 | 3584 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 460 | Qwen2.5-7B | Qwen2.5 | K | FWD | 4096 | 512 | 3584 |  |
| 461 | Qwen2.5-7B | Qwen2.5 | K | BWD-dX | 4096 | 3584 | 512 | dX=dY@W, N/K swap |
| 462 | Qwen2.5-7B | Qwen2.5 | K | BWD-dW | 512 | 3584 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 463 | Qwen2.5-7B | Qwen2.5 | V | FWD | 4096 | 512 | 3584 |  |
| 464 | Qwen2.5-7B | Qwen2.5 | V | BWD-dX | 4096 | 3584 | 512 | dX=dY@W, N/K swap |
| 465 | Qwen2.5-7B | Qwen2.5 | V | BWD-dW | 512 | 3584 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 466 | Qwen2.5-7B | Qwen2.5 | O | FWD | 4096 | 3584 | 3584 |  |
| 467 | Qwen2.5-7B | Qwen2.5 | O | BWD-dX | 4096 | 3584 | 3584 | dX=dY@W, N/K swap |
| 468 | Qwen2.5-7B | Qwen2.5 | O | BWD-dW | 3584 | 3584 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 469 | Qwen2.5-7B | Qwen2.5 | Gate | FWD | 4096 | 18944 | 3584 |  |
| 470 | Qwen2.5-7B | Qwen2.5 | Gate | BWD-dX | 4096 | 3584 | 18944 | dX=dY@W, N/K swap |
| 471 | Qwen2.5-7B | Qwen2.5 | Gate | BWD-dW | 18944 | 3584 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 472 | Qwen2.5-7B | Qwen2.5 | Up | FWD | 4096 | 18944 | 3584 |  |
| 473 | Qwen2.5-7B | Qwen2.5 | Up | BWD-dX | 4096 | 3584 | 18944 | dX=dY@W, N/K swap |
| 474 | Qwen2.5-7B | Qwen2.5 | Up | BWD-dW | 18944 | 3584 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 475 | Qwen2.5-7B | Qwen2.5 | Down | FWD | 4096 | 3584 | 18944 |  |
| 476 | Qwen2.5-7B | Qwen2.5 | Down | BWD-dX | 4096 | 18944 | 3584 | dX=dY@W, N/K swap |
| 477 | Qwen2.5-7B | Qwen2.5 | Down | BWD-dW | 3584 | 18944 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 478 | Qwen2.5-7B | Qwen2.5 | LMhead | FWD | 4096 | 152064 | 3584 |  |
| 479 | Qwen2.5-7B | Qwen2.5 | LMhead | BWD-dX | 4096 | 3584 | 152064 | dX=dY@W, N/K swap |
| 480 | Qwen2.5-7B | Qwen2.5 | LMhead | BWD-dW | 152064 | 3584 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 481 | Qwen2.5-14B | Qwen2.5 | Q | FWD | 4096 | 5120 | 5120 |  |
| 482 | Qwen2.5-14B | Qwen2.5 | Q | BWD-dX | 4096 | 5120 | 5120 | dX=dY@W, N/K swap |
| 483 | Qwen2.5-14B | Qwen2.5 | Q | BWD-dW | 5120 | 5120 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 484 | Qwen2.5-14B | Qwen2.5 | K | FWD | 4096 | 1024 | 5120 |  |
| 485 | Qwen2.5-14B | Qwen2.5 | K | BWD-dX | 4096 | 5120 | 1024 | dX=dY@W, N/K swap |
| 486 | Qwen2.5-14B | Qwen2.5 | K | BWD-dW | 1024 | 5120 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 487 | Qwen2.5-14B | Qwen2.5 | V | FWD | 4096 | 1024 | 5120 |  |
| 488 | Qwen2.5-14B | Qwen2.5 | V | BWD-dX | 4096 | 5120 | 1024 | dX=dY@W, N/K swap |
| 489 | Qwen2.5-14B | Qwen2.5 | V | BWD-dW | 1024 | 5120 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 490 | Qwen2.5-14B | Qwen2.5 | O | FWD | 4096 | 5120 | 5120 |  |
| 491 | Qwen2.5-14B | Qwen2.5 | O | BWD-dX | 4096 | 5120 | 5120 | dX=dY@W, N/K swap |
| 492 | Qwen2.5-14B | Qwen2.5 | O | BWD-dW | 5120 | 5120 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 493 | Qwen2.5-14B | Qwen2.5 | Gate | FWD | 4096 | 13824 | 5120 |  |
| 494 | Qwen2.5-14B | Qwen2.5 | Gate | BWD-dX | 4096 | 5120 | 13824 | dX=dY@W, N/K swap |
| 495 | Qwen2.5-14B | Qwen2.5 | Gate | BWD-dW | 13824 | 5120 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 496 | Qwen2.5-14B | Qwen2.5 | Up | FWD | 4096 | 13824 | 5120 |  |
| 497 | Qwen2.5-14B | Qwen2.5 | Up | BWD-dX | 4096 | 5120 | 13824 | dX=dY@W, N/K swap |
| 498 | Qwen2.5-14B | Qwen2.5 | Up | BWD-dW | 13824 | 5120 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 499 | Qwen2.5-14B | Qwen2.5 | Down | FWD | 4096 | 5120 | 13824 |  |
| 500 | Qwen2.5-14B | Qwen2.5 | Down | BWD-dX | 4096 | 13824 | 5120 | dX=dY@W, N/K swap |
| 501 | Qwen2.5-14B | Qwen2.5 | Down | BWD-dW | 5120 | 13824 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 502 | Qwen2.5-14B | Qwen2.5 | LMhead | FWD | 4096 | 152064 | 5120 |  |
| 503 | Qwen2.5-14B | Qwen2.5 | LMhead | BWD-dX | 4096 | 5120 | 152064 | dX=dY@W, N/K swap |
| 504 | Qwen2.5-14B | Qwen2.5 | LMhead | BWD-dW | 152064 | 5120 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 505 | Qwen2.5-32B | Qwen2.5 | Q | FWD | 4096 | 5120 | 5120 |  |
| 506 | Qwen2.5-32B | Qwen2.5 | Q | BWD-dX | 4096 | 5120 | 5120 | dX=dY@W, N/K swap |
| 507 | Qwen2.5-32B | Qwen2.5 | Q | BWD-dW | 5120 | 5120 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 508 | Qwen2.5-32B | Qwen2.5 | K | FWD | 4096 | 1024 | 5120 |  |
| 509 | Qwen2.5-32B | Qwen2.5 | K | BWD-dX | 4096 | 5120 | 1024 | dX=dY@W, N/K swap |
| 510 | Qwen2.5-32B | Qwen2.5 | K | BWD-dW | 1024 | 5120 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 511 | Qwen2.5-32B | Qwen2.5 | V | FWD | 4096 | 1024 | 5120 |  |
| 512 | Qwen2.5-32B | Qwen2.5 | V | BWD-dX | 4096 | 5120 | 1024 | dX=dY@W, N/K swap |
| 513 | Qwen2.5-32B | Qwen2.5 | V | BWD-dW | 1024 | 5120 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 514 | Qwen2.5-32B | Qwen2.5 | O | FWD | 4096 | 5120 | 5120 |  |
| 515 | Qwen2.5-32B | Qwen2.5 | O | BWD-dX | 4096 | 5120 | 5120 | dX=dY@W, N/K swap |
| 516 | Qwen2.5-32B | Qwen2.5 | O | BWD-dW | 5120 | 5120 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 517 | Qwen2.5-32B | Qwen2.5 | Gate | FWD | 4096 | 27648 | 5120 |  |
| 518 | Qwen2.5-32B | Qwen2.5 | Gate | BWD-dX | 4096 | 5120 | 27648 | dX=dY@W, N/K swap |
| 519 | Qwen2.5-32B | Qwen2.5 | Gate | BWD-dW | 27648 | 5120 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 520 | Qwen2.5-32B | Qwen2.5 | Up | FWD | 4096 | 27648 | 5120 |  |
| 521 | Qwen2.5-32B | Qwen2.5 | Up | BWD-dX | 4096 | 5120 | 27648 | dX=dY@W, N/K swap |
| 522 | Qwen2.5-32B | Qwen2.5 | Up | BWD-dW | 27648 | 5120 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 523 | Qwen2.5-32B | Qwen2.5 | Down | FWD | 4096 | 5120 | 27648 |  |
| 524 | Qwen2.5-32B | Qwen2.5 | Down | BWD-dX | 4096 | 27648 | 5120 | dX=dY@W, N/K swap |
| 525 | Qwen2.5-32B | Qwen2.5 | Down | BWD-dW | 5120 | 27648 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 526 | Qwen2.5-32B | Qwen2.5 | LMhead | FWD | 4096 | 152064 | 5120 |  |
| 527 | Qwen2.5-32B | Qwen2.5 | LMhead | BWD-dX | 4096 | 5120 | 152064 | dX=dY@W, N/K swap |
| 528 | Qwen2.5-32B | Qwen2.5 | LMhead | BWD-dW | 152064 | 5120 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 529 | Qwen2.5-72B | Qwen2.5 | Q | FWD | 4096 | 8192 | 8192 |  |
| 530 | Qwen2.5-72B | Qwen2.5 | Q | BWD-dX | 4096 | 8192 | 8192 | dX=dY@W, N/K swap |
| 531 | Qwen2.5-72B | Qwen2.5 | Q | BWD-dW | 8192 | 8192 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 532 | Qwen2.5-72B | Qwen2.5 | K | FWD | 4096 | 1024 | 8192 |  |
| 533 | Qwen2.5-72B | Qwen2.5 | K | BWD-dX | 4096 | 8192 | 1024 | dX=dY@W, N/K swap |
| 534 | Qwen2.5-72B | Qwen2.5 | K | BWD-dW | 1024 | 8192 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 535 | Qwen2.5-72B | Qwen2.5 | V | FWD | 4096 | 1024 | 8192 |  |
| 536 | Qwen2.5-72B | Qwen2.5 | V | BWD-dX | 4096 | 8192 | 1024 | dX=dY@W, N/K swap |
| 537 | Qwen2.5-72B | Qwen2.5 | V | BWD-dW | 1024 | 8192 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 538 | Qwen2.5-72B | Qwen2.5 | O | FWD | 4096 | 8192 | 8192 |  |
| 539 | Qwen2.5-72B | Qwen2.5 | O | BWD-dX | 4096 | 8192 | 8192 | dX=dY@W, N/K swap |
| 540 | Qwen2.5-72B | Qwen2.5 | O | BWD-dW | 8192 | 8192 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 541 | Qwen2.5-72B | Qwen2.5 | Gate | FWD | 4096 | 29568 | 8192 |  |
| 542 | Qwen2.5-72B | Qwen2.5 | Gate | BWD-dX | 4096 | 8192 | 29568 | dX=dY@W, N/K swap |
| 543 | Qwen2.5-72B | Qwen2.5 | Gate | BWD-dW | 29568 | 8192 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 544 | Qwen2.5-72B | Qwen2.5 | Up | FWD | 4096 | 29568 | 8192 |  |
| 545 | Qwen2.5-72B | Qwen2.5 | Up | BWD-dX | 4096 | 8192 | 29568 | dX=dY@W, N/K swap |
| 546 | Qwen2.5-72B | Qwen2.5 | Up | BWD-dW | 29568 | 8192 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 547 | Qwen2.5-72B | Qwen2.5 | Down | FWD | 4096 | 8192 | 29568 |  |
| 548 | Qwen2.5-72B | Qwen2.5 | Down | BWD-dX | 4096 | 29568 | 8192 | dX=dY@W, N/K swap |
| 549 | Qwen2.5-72B | Qwen2.5 | Down | BWD-dW | 8192 | 29568 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 550 | Qwen2.5-72B | Qwen2.5 | LMhead | FWD | 4096 | 152064 | 8192 |  |
| 551 | Qwen2.5-72B | Qwen2.5 | LMhead | BWD-dX | 4096 | 8192 | 152064 | dX=dY@W, N/K swap |
| 552 | Qwen2.5-72B | Qwen2.5 | LMhead | BWD-dW | 152064 | 8192 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 553 | Qwen3-0.6B | Qwen3 | Q | FWD | 4096 | 2048 | 1024 | non-square (Q_dim=2048 vs H=1024) |
| 554 | Qwen3-0.6B | Qwen3 | Q | BWD-dX | 4096 | 1024 | 2048 | dX=dY@W, N/K swap |
| 555 | Qwen3-0.6B | Qwen3 | Q | BWD-dW | 2048 | 1024 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 556 | Qwen3-0.6B | Qwen3 | K | FWD | 4096 | 1024 | 1024 |  |
| 557 | Qwen3-0.6B | Qwen3 | K | BWD-dX | 4096 | 1024 | 1024 | dX=dY@W, N/K swap |
| 558 | Qwen3-0.6B | Qwen3 | K | BWD-dW | 1024 | 1024 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 559 | Qwen3-0.6B | Qwen3 | V | FWD | 4096 | 1024 | 1024 |  |
| 560 | Qwen3-0.6B | Qwen3 | V | BWD-dX | 4096 | 1024 | 1024 | dX=dY@W, N/K swap |
| 561 | Qwen3-0.6B | Qwen3 | V | BWD-dW | 1024 | 1024 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 562 | Qwen3-0.6B | Qwen3 | O | FWD | 4096 | 1024 | 2048 | non-square (Q_dim=2048 vs H=1024) |
| 563 | Qwen3-0.6B | Qwen3 | O | BWD-dX | 4096 | 2048 | 1024 | dX=dY@W, N/K swap |
| 564 | Qwen3-0.6B | Qwen3 | O | BWD-dW | 1024 | 2048 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 565 | Qwen3-0.6B | Qwen3 | Gate | FWD | 4096 | 3072 | 1024 |  |
| 566 | Qwen3-0.6B | Qwen3 | Gate | BWD-dX | 4096 | 1024 | 3072 | dX=dY@W, N/K swap |
| 567 | Qwen3-0.6B | Qwen3 | Gate | BWD-dW | 3072 | 1024 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 568 | Qwen3-0.6B | Qwen3 | Up | FWD | 4096 | 3072 | 1024 |  |
| 569 | Qwen3-0.6B | Qwen3 | Up | BWD-dX | 4096 | 1024 | 3072 | dX=dY@W, N/K swap |
| 570 | Qwen3-0.6B | Qwen3 | Up | BWD-dW | 3072 | 1024 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 571 | Qwen3-0.6B | Qwen3 | Down | FWD | 4096 | 1024 | 3072 |  |
| 572 | Qwen3-0.6B | Qwen3 | Down | BWD-dX | 4096 | 3072 | 1024 | dX=dY@W, N/K swap |
| 573 | Qwen3-0.6B | Qwen3 | Down | BWD-dW | 1024 | 3072 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 574 | Qwen3-0.6B | Qwen3 | LMhead | FWD | 4096 | 151936 | 1024 |  |
| 575 | Qwen3-0.6B | Qwen3 | LMhead | BWD-dX | 4096 | 1024 | 151936 | dX=dY@W, N/K swap |
| 576 | Qwen3-0.6B | Qwen3 | LMhead | BWD-dW | 151936 | 1024 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 577 | Qwen3-1.7B | Qwen3 | Q | FWD | 4096 | 2048 | 2048 |  |
| 578 | Qwen3-1.7B | Qwen3 | Q | BWD-dX | 4096 | 2048 | 2048 | dX=dY@W, N/K swap |
| 579 | Qwen3-1.7B | Qwen3 | Q | BWD-dW | 2048 | 2048 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 580 | Qwen3-1.7B | Qwen3 | K | FWD | 4096 | 1024 | 2048 |  |
| 581 | Qwen3-1.7B | Qwen3 | K | BWD-dX | 4096 | 2048 | 1024 | dX=dY@W, N/K swap |
| 582 | Qwen3-1.7B | Qwen3 | K | BWD-dW | 1024 | 2048 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 583 | Qwen3-1.7B | Qwen3 | V | FWD | 4096 | 1024 | 2048 |  |
| 584 | Qwen3-1.7B | Qwen3 | V | BWD-dX | 4096 | 2048 | 1024 | dX=dY@W, N/K swap |
| 585 | Qwen3-1.7B | Qwen3 | V | BWD-dW | 1024 | 2048 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 586 | Qwen3-1.7B | Qwen3 | O | FWD | 4096 | 2048 | 2048 |  |
| 587 | Qwen3-1.7B | Qwen3 | O | BWD-dX | 4096 | 2048 | 2048 | dX=dY@W, N/K swap |
| 588 | Qwen3-1.7B | Qwen3 | O | BWD-dW | 2048 | 2048 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 589 | Qwen3-1.7B | Qwen3 | Gate | FWD | 4096 | 6144 | 2048 |  |
| 590 | Qwen3-1.7B | Qwen3 | Gate | BWD-dX | 4096 | 2048 | 6144 | dX=dY@W, N/K swap |
| 591 | Qwen3-1.7B | Qwen3 | Gate | BWD-dW | 6144 | 2048 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 592 | Qwen3-1.7B | Qwen3 | Up | FWD | 4096 | 6144 | 2048 |  |
| 593 | Qwen3-1.7B | Qwen3 | Up | BWD-dX | 4096 | 2048 | 6144 | dX=dY@W, N/K swap |
| 594 | Qwen3-1.7B | Qwen3 | Up | BWD-dW | 6144 | 2048 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 595 | Qwen3-1.7B | Qwen3 | Down | FWD | 4096 | 2048 | 6144 |  |
| 596 | Qwen3-1.7B | Qwen3 | Down | BWD-dX | 4096 | 6144 | 2048 | dX=dY@W, N/K swap |
| 597 | Qwen3-1.7B | Qwen3 | Down | BWD-dW | 2048 | 6144 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 598 | Qwen3-1.7B | Qwen3 | LMhead | FWD | 4096 | 151936 | 2048 |  |
| 599 | Qwen3-1.7B | Qwen3 | LMhead | BWD-dX | 4096 | 2048 | 151936 | dX=dY@W, N/K swap |
| 600 | Qwen3-1.7B | Qwen3 | LMhead | BWD-dW | 151936 | 2048 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 601 | Qwen3-4B | Qwen3 | Q | FWD | 4096 | 4096 | 2560 | non-square (Q_dim=4096 vs H=2560) |
| 602 | Qwen3-4B | Qwen3 | Q | BWD-dX | 4096 | 2560 | 4096 | dX=dY@W, N/K swap |
| 603 | Qwen3-4B | Qwen3 | Q | BWD-dW | 4096 | 2560 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 604 | Qwen3-4B | Qwen3 | K | FWD | 4096 | 1024 | 2560 |  |
| 605 | Qwen3-4B | Qwen3 | K | BWD-dX | 4096 | 2560 | 1024 | dX=dY@W, N/K swap |
| 606 | Qwen3-4B | Qwen3 | K | BWD-dW | 1024 | 2560 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 607 | Qwen3-4B | Qwen3 | V | FWD | 4096 | 1024 | 2560 |  |
| 608 | Qwen3-4B | Qwen3 | V | BWD-dX | 4096 | 2560 | 1024 | dX=dY@W, N/K swap |
| 609 | Qwen3-4B | Qwen3 | V | BWD-dW | 1024 | 2560 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 610 | Qwen3-4B | Qwen3 | O | FWD | 4096 | 2560 | 4096 | non-square (Q_dim=4096 vs H=2560) |
| 611 | Qwen3-4B | Qwen3 | O | BWD-dX | 4096 | 4096 | 2560 | dX=dY@W, N/K swap |
| 612 | Qwen3-4B | Qwen3 | O | BWD-dW | 2560 | 4096 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 613 | Qwen3-4B | Qwen3 | Gate | FWD | 4096 | 9728 | 2560 |  |
| 614 | Qwen3-4B | Qwen3 | Gate | BWD-dX | 4096 | 2560 | 9728 | dX=dY@W, N/K swap |
| 615 | Qwen3-4B | Qwen3 | Gate | BWD-dW | 9728 | 2560 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 616 | Qwen3-4B | Qwen3 | Up | FWD | 4096 | 9728 | 2560 |  |
| 617 | Qwen3-4B | Qwen3 | Up | BWD-dX | 4096 | 2560 | 9728 | dX=dY@W, N/K swap |
| 618 | Qwen3-4B | Qwen3 | Up | BWD-dW | 9728 | 2560 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 619 | Qwen3-4B | Qwen3 | Down | FWD | 4096 | 2560 | 9728 |  |
| 620 | Qwen3-4B | Qwen3 | Down | BWD-dX | 4096 | 9728 | 2560 | dX=dY@W, N/K swap |
| 621 | Qwen3-4B | Qwen3 | Down | BWD-dW | 2560 | 9728 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 622 | Qwen3-4B | Qwen3 | LMhead | FWD | 4096 | 151936 | 2560 |  |
| 623 | Qwen3-4B | Qwen3 | LMhead | BWD-dX | 4096 | 2560 | 151936 | dX=dY@W, N/K swap |
| 624 | Qwen3-4B | Qwen3 | LMhead | BWD-dW | 151936 | 2560 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 625 | Qwen3-8B | Qwen3 | Q | FWD | 4096 | 4096 | 4096 |  |
| 626 | Qwen3-8B | Qwen3 | Q | BWD-dX | 4096 | 4096 | 4096 | dX=dY@W, N/K swap |
| 627 | Qwen3-8B | Qwen3 | Q | BWD-dW | 4096 | 4096 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 628 | Qwen3-8B | Qwen3 | K | FWD | 4096 | 1024 | 4096 |  |
| 629 | Qwen3-8B | Qwen3 | K | BWD-dX | 4096 | 4096 | 1024 | dX=dY@W, N/K swap |
| 630 | Qwen3-8B | Qwen3 | K | BWD-dW | 1024 | 4096 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 631 | Qwen3-8B | Qwen3 | V | FWD | 4096 | 1024 | 4096 |  |
| 632 | Qwen3-8B | Qwen3 | V | BWD-dX | 4096 | 4096 | 1024 | dX=dY@W, N/K swap |
| 633 | Qwen3-8B | Qwen3 | V | BWD-dW | 1024 | 4096 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 634 | Qwen3-8B | Qwen3 | O | FWD | 4096 | 4096 | 4096 |  |
| 635 | Qwen3-8B | Qwen3 | O | BWD-dX | 4096 | 4096 | 4096 | dX=dY@W, N/K swap |
| 636 | Qwen3-8B | Qwen3 | O | BWD-dW | 4096 | 4096 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 637 | Qwen3-8B | Qwen3 | Gate | FWD | 4096 | 12288 | 4096 |  |
| 638 | Qwen3-8B | Qwen3 | Gate | BWD-dX | 4096 | 4096 | 12288 | dX=dY@W, N/K swap |
| 639 | Qwen3-8B | Qwen3 | Gate | BWD-dW | 12288 | 4096 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 640 | Qwen3-8B | Qwen3 | Up | FWD | 4096 | 12288 | 4096 |  |
| 641 | Qwen3-8B | Qwen3 | Up | BWD-dX | 4096 | 4096 | 12288 | dX=dY@W, N/K swap |
| 642 | Qwen3-8B | Qwen3 | Up | BWD-dW | 12288 | 4096 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 643 | Qwen3-8B | Qwen3 | Down | FWD | 4096 | 4096 | 12288 |  |
| 644 | Qwen3-8B | Qwen3 | Down | BWD-dX | 4096 | 12288 | 4096 | dX=dY@W, N/K swap |
| 645 | Qwen3-8B | Qwen3 | Down | BWD-dW | 4096 | 12288 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 646 | Qwen3-8B | Qwen3 | LMhead | FWD | 4096 | 151936 | 4096 |  |
| 647 | Qwen3-8B | Qwen3 | LMhead | BWD-dX | 4096 | 4096 | 151936 | dX=dY@W, N/K swap |
| 648 | Qwen3-8B | Qwen3 | LMhead | BWD-dW | 151936 | 4096 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 649 | Qwen3-14B | Qwen3 | Q | FWD | 4096 | 5120 | 5120 |  |
| 650 | Qwen3-14B | Qwen3 | Q | BWD-dX | 4096 | 5120 | 5120 | dX=dY@W, N/K swap |
| 651 | Qwen3-14B | Qwen3 | Q | BWD-dW | 5120 | 5120 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 652 | Qwen3-14B | Qwen3 | K | FWD | 4096 | 1024 | 5120 |  |
| 653 | Qwen3-14B | Qwen3 | K | BWD-dX | 4096 | 5120 | 1024 | dX=dY@W, N/K swap |
| 654 | Qwen3-14B | Qwen3 | K | BWD-dW | 1024 | 5120 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 655 | Qwen3-14B | Qwen3 | V | FWD | 4096 | 1024 | 5120 |  |
| 656 | Qwen3-14B | Qwen3 | V | BWD-dX | 4096 | 5120 | 1024 | dX=dY@W, N/K swap |
| 657 | Qwen3-14B | Qwen3 | V | BWD-dW | 1024 | 5120 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 658 | Qwen3-14B | Qwen3 | O | FWD | 4096 | 5120 | 5120 |  |
| 659 | Qwen3-14B | Qwen3 | O | BWD-dX | 4096 | 5120 | 5120 | dX=dY@W, N/K swap |
| 660 | Qwen3-14B | Qwen3 | O | BWD-dW | 5120 | 5120 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 661 | Qwen3-14B | Qwen3 | Gate | FWD | 4096 | 17408 | 5120 |  |
| 662 | Qwen3-14B | Qwen3 | Gate | BWD-dX | 4096 | 5120 | 17408 | dX=dY@W, N/K swap |
| 663 | Qwen3-14B | Qwen3 | Gate | BWD-dW | 17408 | 5120 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 664 | Qwen3-14B | Qwen3 | Up | FWD | 4096 | 17408 | 5120 |  |
| 665 | Qwen3-14B | Qwen3 | Up | BWD-dX | 4096 | 5120 | 17408 | dX=dY@W, N/K swap |
| 666 | Qwen3-14B | Qwen3 | Up | BWD-dW | 17408 | 5120 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 667 | Qwen3-14B | Qwen3 | Down | FWD | 4096 | 5120 | 17408 |  |
| 668 | Qwen3-14B | Qwen3 | Down | BWD-dX | 4096 | 17408 | 5120 | dX=dY@W, N/K swap |
| 669 | Qwen3-14B | Qwen3 | Down | BWD-dW | 5120 | 17408 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 670 | Qwen3-14B | Qwen3 | LMhead | FWD | 4096 | 151936 | 5120 |  |
| 671 | Qwen3-14B | Qwen3 | LMhead | BWD-dX | 4096 | 5120 | 151936 | dX=dY@W, N/K swap |
| 672 | Qwen3-14B | Qwen3 | LMhead | BWD-dW | 151936 | 5120 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 673 | Qwen3-32B | Qwen3 | Q | FWD | 4096 | 8192 | 5120 | non-square (Q_dim=8192 vs H=5120) |
| 674 | Qwen3-32B | Qwen3 | Q | BWD-dX | 4096 | 5120 | 8192 | dX=dY@W, N/K swap |
| 675 | Qwen3-32B | Qwen3 | Q | BWD-dW | 8192 | 5120 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 676 | Qwen3-32B | Qwen3 | K | FWD | 4096 | 1024 | 5120 |  |
| 677 | Qwen3-32B | Qwen3 | K | BWD-dX | 4096 | 5120 | 1024 | dX=dY@W, N/K swap |
| 678 | Qwen3-32B | Qwen3 | K | BWD-dW | 1024 | 5120 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 679 | Qwen3-32B | Qwen3 | V | FWD | 4096 | 1024 | 5120 |  |
| 680 | Qwen3-32B | Qwen3 | V | BWD-dX | 4096 | 5120 | 1024 | dX=dY@W, N/K swap |
| 681 | Qwen3-32B | Qwen3 | V | BWD-dW | 1024 | 5120 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 682 | Qwen3-32B | Qwen3 | O | FWD | 4096 | 5120 | 8192 | non-square (Q_dim=8192 vs H=5120) |
| 683 | Qwen3-32B | Qwen3 | O | BWD-dX | 4096 | 8192 | 5120 | dX=dY@W, N/K swap |
| 684 | Qwen3-32B | Qwen3 | O | BWD-dW | 5120 | 8192 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 685 | Qwen3-32B | Qwen3 | Gate | FWD | 4096 | 25600 | 5120 |  |
| 686 | Qwen3-32B | Qwen3 | Gate | BWD-dX | 4096 | 5120 | 25600 | dX=dY@W, N/K swap |
| 687 | Qwen3-32B | Qwen3 | Gate | BWD-dW | 25600 | 5120 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 688 | Qwen3-32B | Qwen3 | Up | FWD | 4096 | 25600 | 5120 |  |
| 689 | Qwen3-32B | Qwen3 | Up | BWD-dX | 4096 | 5120 | 25600 | dX=dY@W, N/K swap |
| 690 | Qwen3-32B | Qwen3 | Up | BWD-dW | 25600 | 5120 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 691 | Qwen3-32B | Qwen3 | Down | FWD | 4096 | 5120 | 25600 |  |
| 692 | Qwen3-32B | Qwen3 | Down | BWD-dX | 4096 | 25600 | 5120 | dX=dY@W, N/K swap |
| 693 | Qwen3-32B | Qwen3 | Down | BWD-dW | 5120 | 25600 | 4096 | dW=dY^T@X, batch M becomes contraction |
| 694 | Qwen3-32B | Qwen3 | LMhead | FWD | 4096 | 151936 | 5120 |  |
| 695 | Qwen3-32B | Qwen3 | LMhead | BWD-dX | 4096 | 5120 | 151936 | dX=dY@W, N/K swap |
| 696 | Qwen3-32B | Qwen3 | LMhead | BWD-dW | 151936 | 5120 | 4096 | dW=dY^T@X, batch M becomes contraction |

### H3.6 Per-model detailed reference — full config + forward + backward shapes, one subsection per model

#### Qwen1-1.8B

Source: `https://huggingface.co/Qwen/Qwen-1_8B/raw/main/config.json` (fetched and verified this session)


| Config field | Value |
|---|---|
| hidden_size (H) | 2048 |
| intermediate_size (real per-branch FFN width, I) | 5504 |
| num_hidden_layers | 24 |
| num_attention_heads (QH) | 16 |
| num_key_value_heads (KVH) | 16 |
| head_dim (HD) | 128 |
| Q_dim = QH*HD | 2048 (=H, square) |
| KV_dim = KVH*HD | 2048 (plain MHA, no GQA) |
| vocab_size (V) | 151936 |
| tie_word_embeddings | False |
| QKV projection style | fused (single c_attn matmul, split post-hoc) |

**Forward-pass GEMM shapes (M=4096 tokens):**

| Op | (M, N, K) | Notes |
|---|---|---|
| Q | (4096, 2048, 2048) |  |
| K | (4096, 2048, 2048) |  |
| V | (4096, 2048, 2048) |  |
| O | (4096, 2048, 2048) |  |
| Gate | (4096, 5504, 2048) |  |
| Up | (4096, 5504, 2048) |  |
| Down | (4096, 2048, 5504) |  |
| LMhead | (4096, 151936, 2048) |  |

**Backward-pass GEMM shapes derived from the above:**

| Op | dX = dY@W (M,N,K) | dW = dY^T@X (M,N,K) |
|---|---|---|
| Q | (4096, 2048, 2048) | (2048, 2048, 4096) |
| K | (4096, 2048, 2048) | (2048, 2048, 4096) |
| V | (4096, 2048, 2048) | (2048, 2048, 4096) |
| O | (4096, 2048, 2048) | (2048, 2048, 4096) |
| Gate | (4096, 2048, 5504) | (5504, 2048, 4096) |
| Up | (4096, 2048, 5504) | (5504, 2048, 4096) |
| Down | (4096, 5504, 2048) | (2048, 5504, 4096) |
| LMhead | (4096, 2048, 151936) | (151936, 2048, 4096) |

#### Qwen1-7B

Source: `https://huggingface.co/Qwen/Qwen-7B/raw/main/config.json` (fetched and verified this session)


| Config field | Value |
|---|---|
| hidden_size (H) | 4096 |
| intermediate_size (real per-branch FFN width, I) | 11008 |
| num_hidden_layers | 32 |
| num_attention_heads (QH) | 32 |
| num_key_value_heads (KVH) | 32 |
| head_dim (HD) | 128 |
| Q_dim = QH*HD | 4096 (=H, square) |
| KV_dim = KVH*HD | 4096 (plain MHA, no GQA) |
| vocab_size (V) | 151936 |
| tie_word_embeddings | False |
| QKV projection style | fused (single c_attn matmul, split post-hoc) |

**Forward-pass GEMM shapes (M=4096 tokens):**

| Op | (M, N, K) | Notes |
|---|---|---|
| Q | (4096, 4096, 4096) |  |
| K | (4096, 4096, 4096) |  |
| V | (4096, 4096, 4096) |  |
| O | (4096, 4096, 4096) |  |
| Gate | (4096, 11008, 4096) |  |
| Up | (4096, 11008, 4096) |  |
| Down | (4096, 4096, 11008) |  |
| LMhead | (4096, 151936, 4096) |  |

**Backward-pass GEMM shapes derived from the above:**

| Op | dX = dY@W (M,N,K) | dW = dY^T@X (M,N,K) |
|---|---|---|
| Q | (4096, 4096, 4096) | (4096, 4096, 4096) |
| K | (4096, 4096, 4096) | (4096, 4096, 4096) |
| V | (4096, 4096, 4096) | (4096, 4096, 4096) |
| O | (4096, 4096, 4096) | (4096, 4096, 4096) |
| Gate | (4096, 4096, 11008) | (11008, 4096, 4096) |
| Up | (4096, 4096, 11008) | (11008, 4096, 4096) |
| Down | (4096, 11008, 4096) | (4096, 11008, 4096) |
| LMhead | (4096, 4096, 151936) | (151936, 4096, 4096) |

#### Qwen1-14B

Source: `https://huggingface.co/Qwen/Qwen-14B/raw/main/config.json` (fetched and verified this session)


| Config field | Value |
|---|---|
| hidden_size (H) | 5120 |
| intermediate_size (real per-branch FFN width, I) | 13696 |
| num_hidden_layers | 40 |
| num_attention_heads (QH) | 40 |
| num_key_value_heads (KVH) | 40 |
| head_dim (HD) | 128 |
| Q_dim = QH*HD | 5120 (=H, square) |
| KV_dim = KVH*HD | 5120 (plain MHA, no GQA) |
| vocab_size (V) | 152064 |
| tie_word_embeddings | False |
| QKV projection style | fused (single c_attn matmul, split post-hoc) |

**Forward-pass GEMM shapes (M=4096 tokens):**

| Op | (M, N, K) | Notes |
|---|---|---|
| Q | (4096, 5120, 5120) |  |
| K | (4096, 5120, 5120) |  |
| V | (4096, 5120, 5120) |  |
| O | (4096, 5120, 5120) |  |
| Gate | (4096, 13696, 5120) |  |
| Up | (4096, 13696, 5120) |  |
| Down | (4096, 5120, 13696) |  |
| LMhead | (4096, 152064, 5120) |  |

**Backward-pass GEMM shapes derived from the above:**

| Op | dX = dY@W (M,N,K) | dW = dY^T@X (M,N,K) |
|---|---|---|
| Q | (4096, 5120, 5120) | (5120, 5120, 4096) |
| K | (4096, 5120, 5120) | (5120, 5120, 4096) |
| V | (4096, 5120, 5120) | (5120, 5120, 4096) |
| O | (4096, 5120, 5120) | (5120, 5120, 4096) |
| Gate | (4096, 5120, 13696) | (13696, 5120, 4096) |
| Up | (4096, 5120, 13696) | (13696, 5120, 4096) |
| Down | (4096, 13696, 5120) | (5120, 13696, 4096) |
| LMhead | (4096, 5120, 152064) | (152064, 5120, 4096) |

#### Qwen1-72B

Source: `https://huggingface.co/Qwen/Qwen-72B/raw/main/config.json` (fetched and verified this session)


| Config field | Value |
|---|---|
| hidden_size (H) | 8192 |
| intermediate_size (real per-branch FFN width, I) | 24576 |
| num_hidden_layers | 80 |
| num_attention_heads (QH) | 64 |
| num_key_value_heads (KVH) | 64 |
| head_dim (HD) | 128 |
| Q_dim = QH*HD | 8192 (=H, square) |
| KV_dim = KVH*HD | 8192 (plain MHA, no GQA) |
| vocab_size (V) | 152064 |
| tie_word_embeddings | False |
| QKV projection style | fused (single c_attn matmul, split post-hoc) |

**Forward-pass GEMM shapes (M=4096 tokens):**

| Op | (M, N, K) | Notes |
|---|---|---|
| Q | (4096, 8192, 8192) |  |
| K | (4096, 8192, 8192) |  |
| V | (4096, 8192, 8192) |  |
| O | (4096, 8192, 8192) |  |
| Gate | (4096, 24576, 8192) |  |
| Up | (4096, 24576, 8192) |  |
| Down | (4096, 8192, 24576) |  |
| LMhead | (4096, 152064, 8192) |  |

**Backward-pass GEMM shapes derived from the above:**

| Op | dX = dY@W (M,N,K) | dW = dY^T@X (M,N,K) |
|---|---|---|
| Q | (4096, 8192, 8192) | (8192, 8192, 4096) |
| K | (4096, 8192, 8192) | (8192, 8192, 4096) |
| V | (4096, 8192, 8192) | (8192, 8192, 4096) |
| O | (4096, 8192, 8192) | (8192, 8192, 4096) |
| Gate | (4096, 8192, 24576) | (24576, 8192, 4096) |
| Up | (4096, 8192, 24576) | (24576, 8192, 4096) |
| Down | (4096, 24576, 8192) | (8192, 24576, 4096) |
| LMhead | (4096, 8192, 152064) | (152064, 8192, 4096) |

#### Qwen1.5-0.5B

Source: `https://huggingface.co/Qwen/Qwen1.5-0.5B/raw/main/config.json` (fetched and verified this session)


| Config field | Value |
|---|---|
| hidden_size (H) | 1024 |
| intermediate_size (real per-branch FFN width, I) | 2816 |
| num_hidden_layers | 24 |
| num_attention_heads (QH) | 16 |
| num_key_value_heads (KVH) | 16 |
| head_dim (HD) | 64 |
| Q_dim = QH*HD | 1024 (=H, square) |
| KV_dim = KVH*HD | 1024 (plain MHA, no GQA) |
| vocab_size (V) | 151936 |
| tie_word_embeddings | True |
| QKV projection style | separate (three independent nn.Linear layers) |

**Forward-pass GEMM shapes (M=4096 tokens):**

| Op | (M, N, K) | Notes |
|---|---|---|
| Q | (4096, 1024, 1024) |  |
| K | (4096, 1024, 1024) |  |
| V | (4096, 1024, 1024) |  |
| O | (4096, 1024, 1024) |  |
| Gate | (4096, 2816, 1024) |  |
| Up | (4096, 2816, 1024) |  |
| Down | (4096, 1024, 2816) |  |
| LMhead | (4096, 151936, 1024) |  |

**Backward-pass GEMM shapes derived from the above:**

| Op | dX = dY@W (M,N,K) | dW = dY^T@X (M,N,K) |
|---|---|---|
| Q | (4096, 1024, 1024) | (1024, 1024, 4096) |
| K | (4096, 1024, 1024) | (1024, 1024, 4096) |
| V | (4096, 1024, 1024) | (1024, 1024, 4096) |
| O | (4096, 1024, 1024) | (1024, 1024, 4096) |
| Gate | (4096, 1024, 2816) | (2816, 1024, 4096) |
| Up | (4096, 1024, 2816) | (2816, 1024, 4096) |
| Down | (4096, 2816, 1024) | (1024, 2816, 4096) |
| LMhead | (4096, 1024, 151936) | (151936, 1024, 4096) |

#### Qwen1.5-1.8B

Source: `https://huggingface.co/Qwen/Qwen1.5-1.8B/raw/main/config.json` (fetched and verified this session)


| Config field | Value |
|---|---|
| hidden_size (H) | 2048 |
| intermediate_size (real per-branch FFN width, I) | 5504 |
| num_hidden_layers | 24 |
| num_attention_heads (QH) | 16 |
| num_key_value_heads (KVH) | 16 |
| head_dim (HD) | 128 |
| Q_dim = QH*HD | 2048 (=H, square) |
| KV_dim = KVH*HD | 2048 (plain MHA, no GQA) |
| vocab_size (V) | 151936 |
| tie_word_embeddings | False |
| QKV projection style | separate (three independent nn.Linear layers) |

**Forward-pass GEMM shapes (M=4096 tokens):**

| Op | (M, N, K) | Notes |
|---|---|---|
| Q | (4096, 2048, 2048) |  |
| K | (4096, 2048, 2048) |  |
| V | (4096, 2048, 2048) |  |
| O | (4096, 2048, 2048) |  |
| Gate | (4096, 5504, 2048) |  |
| Up | (4096, 5504, 2048) |  |
| Down | (4096, 2048, 5504) |  |
| LMhead | (4096, 151936, 2048) |  |

**Backward-pass GEMM shapes derived from the above:**

| Op | dX = dY@W (M,N,K) | dW = dY^T@X (M,N,K) |
|---|---|---|
| Q | (4096, 2048, 2048) | (2048, 2048, 4096) |
| K | (4096, 2048, 2048) | (2048, 2048, 4096) |
| V | (4096, 2048, 2048) | (2048, 2048, 4096) |
| O | (4096, 2048, 2048) | (2048, 2048, 4096) |
| Gate | (4096, 2048, 5504) | (5504, 2048, 4096) |
| Up | (4096, 2048, 5504) | (5504, 2048, 4096) |
| Down | (4096, 5504, 2048) | (2048, 5504, 4096) |
| LMhead | (4096, 2048, 151936) | (151936, 2048, 4096) |

#### Qwen1.5-4B

Source: `https://huggingface.co/Qwen/Qwen1.5-4B/raw/main/config.json` (fetched and verified this session)


| Config field | Value |
|---|---|
| hidden_size (H) | 2560 |
| intermediate_size (real per-branch FFN width, I) | 6912 |
| num_hidden_layers | 40 |
| num_attention_heads (QH) | 20 |
| num_key_value_heads (KVH) | 20 |
| head_dim (HD) | 128 |
| Q_dim = QH*HD | 2560 (=H, square) |
| KV_dim = KVH*HD | 2560 (plain MHA, no GQA) |
| vocab_size (V) | 151936 |
| tie_word_embeddings | False |
| QKV projection style | separate (three independent nn.Linear layers) |

**Forward-pass GEMM shapes (M=4096 tokens):**

| Op | (M, N, K) | Notes |
|---|---|---|
| Q | (4096, 2560, 2560) |  |
| K | (4096, 2560, 2560) |  |
| V | (4096, 2560, 2560) |  |
| O | (4096, 2560, 2560) |  |
| Gate | (4096, 6912, 2560) |  |
| Up | (4096, 6912, 2560) |  |
| Down | (4096, 2560, 6912) |  |
| LMhead | (4096, 151936, 2560) |  |

**Backward-pass GEMM shapes derived from the above:**

| Op | dX = dY@W (M,N,K) | dW = dY^T@X (M,N,K) |
|---|---|---|
| Q | (4096, 2560, 2560) | (2560, 2560, 4096) |
| K | (4096, 2560, 2560) | (2560, 2560, 4096) |
| V | (4096, 2560, 2560) | (2560, 2560, 4096) |
| O | (4096, 2560, 2560) | (2560, 2560, 4096) |
| Gate | (4096, 2560, 6912) | (6912, 2560, 4096) |
| Up | (4096, 2560, 6912) | (6912, 2560, 4096) |
| Down | (4096, 6912, 2560) | (2560, 6912, 4096) |
| LMhead | (4096, 2560, 151936) | (151936, 2560, 4096) |

#### Qwen1.5-7B

Source: `https://huggingface.co/Qwen/Qwen1.5-7B/raw/main/config.json` (fetched and verified this session)


| Config field | Value |
|---|---|
| hidden_size (H) | 4096 |
| intermediate_size (real per-branch FFN width, I) | 11008 |
| num_hidden_layers | 32 |
| num_attention_heads (QH) | 32 |
| num_key_value_heads (KVH) | 32 |
| head_dim (HD) | 128 |
| Q_dim = QH*HD | 4096 (=H, square) |
| KV_dim = KVH*HD | 4096 (plain MHA, no GQA) |
| vocab_size (V) | 151936 |
| tie_word_embeddings | False |
| QKV projection style | separate (three independent nn.Linear layers) |

**Forward-pass GEMM shapes (M=4096 tokens):**

| Op | (M, N, K) | Notes |
|---|---|---|
| Q | (4096, 4096, 4096) |  |
| K | (4096, 4096, 4096) |  |
| V | (4096, 4096, 4096) |  |
| O | (4096, 4096, 4096) |  |
| Gate | (4096, 11008, 4096) |  |
| Up | (4096, 11008, 4096) |  |
| Down | (4096, 4096, 11008) |  |
| LMhead | (4096, 151936, 4096) |  |

**Backward-pass GEMM shapes derived from the above:**

| Op | dX = dY@W (M,N,K) | dW = dY^T@X (M,N,K) |
|---|---|---|
| Q | (4096, 4096, 4096) | (4096, 4096, 4096) |
| K | (4096, 4096, 4096) | (4096, 4096, 4096) |
| V | (4096, 4096, 4096) | (4096, 4096, 4096) |
| O | (4096, 4096, 4096) | (4096, 4096, 4096) |
| Gate | (4096, 4096, 11008) | (11008, 4096, 4096) |
| Up | (4096, 4096, 11008) | (11008, 4096, 4096) |
| Down | (4096, 11008, 4096) | (4096, 11008, 4096) |
| LMhead | (4096, 4096, 151936) | (151936, 4096, 4096) |

#### Qwen1.5-14B

Source: `https://huggingface.co/Qwen/Qwen1.5-14B/raw/main/config.json` (fetched and verified this session)


| Config field | Value |
|---|---|
| hidden_size (H) | 5120 |
| intermediate_size (real per-branch FFN width, I) | 13696 |
| num_hidden_layers | 40 |
| num_attention_heads (QH) | 40 |
| num_key_value_heads (KVH) | 40 |
| head_dim (HD) | 128 |
| Q_dim = QH*HD | 5120 (=H, square) |
| KV_dim = KVH*HD | 5120 (plain MHA, no GQA) |
| vocab_size (V) | 152064 |
| tie_word_embeddings | False |
| QKV projection style | separate (three independent nn.Linear layers) |

**Forward-pass GEMM shapes (M=4096 tokens):**

| Op | (M, N, K) | Notes |
|---|---|---|
| Q | (4096, 5120, 5120) |  |
| K | (4096, 5120, 5120) |  |
| V | (4096, 5120, 5120) |  |
| O | (4096, 5120, 5120) |  |
| Gate | (4096, 13696, 5120) |  |
| Up | (4096, 13696, 5120) |  |
| Down | (4096, 5120, 13696) |  |
| LMhead | (4096, 152064, 5120) |  |

**Backward-pass GEMM shapes derived from the above:**

| Op | dX = dY@W (M,N,K) | dW = dY^T@X (M,N,K) |
|---|---|---|
| Q | (4096, 5120, 5120) | (5120, 5120, 4096) |
| K | (4096, 5120, 5120) | (5120, 5120, 4096) |
| V | (4096, 5120, 5120) | (5120, 5120, 4096) |
| O | (4096, 5120, 5120) | (5120, 5120, 4096) |
| Gate | (4096, 5120, 13696) | (13696, 5120, 4096) |
| Up | (4096, 5120, 13696) | (13696, 5120, 4096) |
| Down | (4096, 13696, 5120) | (5120, 13696, 4096) |
| LMhead | (4096, 5120, 152064) | (152064, 5120, 4096) |

#### Qwen1.5-32B

Source: `https://huggingface.co/Qwen/Qwen1.5-32B/raw/main/config.json` (fetched and verified this session)


| Config field | Value |
|---|---|
| hidden_size (H) | 5120 |
| intermediate_size (real per-branch FFN width, I) | 27392 |
| num_hidden_layers | 64 |
| num_attention_heads (QH) | 40 |
| num_key_value_heads (KVH) | 8 |
| head_dim (HD) | 128 |
| Q_dim = QH*HD | 5120 (=H, square) |
| KV_dim = KVH*HD | 1024 (GQA ratio 5:1)
| vocab_size (V) | 152064 |
| tie_word_embeddings | False |
| QKV projection style | separate (three independent nn.Linear layers) |

**Forward-pass GEMM shapes (M=4096 tokens):**

| Op | (M, N, K) | Notes |
|---|---|---|
| Q | (4096, 5120, 5120) |  |
| K | (4096, 1024, 5120) |  |
| V | (4096, 1024, 5120) |  |
| O | (4096, 5120, 5120) |  |
| Gate | (4096, 27392, 5120) |  |
| Up | (4096, 27392, 5120) |  |
| Down | (4096, 5120, 27392) |  |
| LMhead | (4096, 152064, 5120) |  |

**Backward-pass GEMM shapes derived from the above:**

| Op | dX = dY@W (M,N,K) | dW = dY^T@X (M,N,K) |
|---|---|---|
| Q | (4096, 5120, 5120) | (5120, 5120, 4096) |
| K | (4096, 5120, 1024) | (1024, 5120, 4096) |
| V | (4096, 5120, 1024) | (1024, 5120, 4096) |
| O | (4096, 5120, 5120) | (5120, 5120, 4096) |
| Gate | (4096, 5120, 27392) | (27392, 5120, 4096) |
| Up | (4096, 5120, 27392) | (27392, 5120, 4096) |
| Down | (4096, 27392, 5120) | (5120, 27392, 4096) |
| LMhead | (4096, 5120, 152064) | (152064, 5120, 4096) |

#### Qwen1.5-72B

Source: `https://huggingface.co/Qwen/Qwen1.5-72B/raw/main/config.json` (fetched and verified this session)


| Config field | Value |
|---|---|
| hidden_size (H) | 8192 |
| intermediate_size (real per-branch FFN width, I) | 24576 |
| num_hidden_layers | 80 |
| num_attention_heads (QH) | 64 |
| num_key_value_heads (KVH) | 64 |
| head_dim (HD) | 128 |
| Q_dim = QH*HD | 8192 (=H, square) |
| KV_dim = KVH*HD | 8192 (plain MHA, no GQA) |
| vocab_size (V) | 152064 |
| tie_word_embeddings | False |
| QKV projection style | separate (three independent nn.Linear layers) |

**Forward-pass GEMM shapes (M=4096 tokens):**

| Op | (M, N, K) | Notes |
|---|---|---|
| Q | (4096, 8192, 8192) |  |
| K | (4096, 8192, 8192) |  |
| V | (4096, 8192, 8192) |  |
| O | (4096, 8192, 8192) |  |
| Gate | (4096, 24576, 8192) |  |
| Up | (4096, 24576, 8192) |  |
| Down | (4096, 8192, 24576) |  |
| LMhead | (4096, 152064, 8192) |  |

**Backward-pass GEMM shapes derived from the above:**

| Op | dX = dY@W (M,N,K) | dW = dY^T@X (M,N,K) |
|---|---|---|
| Q | (4096, 8192, 8192) | (8192, 8192, 4096) |
| K | (4096, 8192, 8192) | (8192, 8192, 4096) |
| V | (4096, 8192, 8192) | (8192, 8192, 4096) |
| O | (4096, 8192, 8192) | (8192, 8192, 4096) |
| Gate | (4096, 8192, 24576) | (24576, 8192, 4096) |
| Up | (4096, 8192, 24576) | (24576, 8192, 4096) |
| Down | (4096, 24576, 8192) | (8192, 24576, 4096) |
| LMhead | (4096, 8192, 152064) | (152064, 8192, 4096) |

#### Qwen1.5-110B

Source: `https://huggingface.co/Qwen/Qwen1.5-110B/raw/main/config.json` (fetched and verified this session)


| Config field | Value |
|---|---|
| hidden_size (H) | 8192 |
| intermediate_size (real per-branch FFN width, I) | 49152 |
| num_hidden_layers | 80 |
| num_attention_heads (QH) | 64 |
| num_key_value_heads (KVH) | 8 |
| head_dim (HD) | 128 |
| Q_dim = QH*HD | 8192 (=H, square) |
| KV_dim = KVH*HD | 1024 (GQA ratio 8:1)
| vocab_size (V) | 152064 |
| tie_word_embeddings | False |
| QKV projection style | separate (three independent nn.Linear layers) |

**Forward-pass GEMM shapes (M=4096 tokens):**

| Op | (M, N, K) | Notes |
|---|---|---|
| Q | (4096, 8192, 8192) |  |
| K | (4096, 1024, 8192) |  |
| V | (4096, 1024, 8192) |  |
| O | (4096, 8192, 8192) |  |
| Gate | (4096, 49152, 8192) |  |
| Up | (4096, 49152, 8192) |  |
| Down | (4096, 8192, 49152) |  |
| LMhead | (4096, 152064, 8192) |  |

**Backward-pass GEMM shapes derived from the above:**

| Op | dX = dY@W (M,N,K) | dW = dY^T@X (M,N,K) |
|---|---|---|
| Q | (4096, 8192, 8192) | (8192, 8192, 4096) |
| K | (4096, 8192, 1024) | (1024, 8192, 4096) |
| V | (4096, 8192, 1024) | (1024, 8192, 4096) |
| O | (4096, 8192, 8192) | (8192, 8192, 4096) |
| Gate | (4096, 8192, 49152) | (49152, 8192, 4096) |
| Up | (4096, 8192, 49152) | (49152, 8192, 4096) |
| Down | (4096, 49152, 8192) | (8192, 49152, 4096) |
| LMhead | (4096, 8192, 152064) | (152064, 8192, 4096) |

#### Qwen2-0.5B

Source: `https://huggingface.co/Qwen/Qwen2-0.5B/raw/main/config.json` (fetched and verified this session)


| Config field | Value |
|---|---|
| hidden_size (H) | 896 |
| intermediate_size (real per-branch FFN width, I) | 4864 |
| num_hidden_layers | 24 |
| num_attention_heads (QH) | 14 |
| num_key_value_heads (KVH) | 2 |
| head_dim (HD) | 64 |
| Q_dim = QH*HD | 896 (=H, square) |
| KV_dim = KVH*HD | 128 (GQA ratio 7:1)
| vocab_size (V) | 151936 |
| tie_word_embeddings | True |
| QKV projection style | separate (three independent nn.Linear layers) |

**Forward-pass GEMM shapes (M=4096 tokens):**

| Op | (M, N, K) | Notes |
|---|---|---|
| Q | (4096, 896, 896) |  |
| K | (4096, 128, 896) |  |
| V | (4096, 128, 896) |  |
| O | (4096, 896, 896) |  |
| Gate | (4096, 4864, 896) |  |
| Up | (4096, 4864, 896) |  |
| Down | (4096, 896, 4864) |  |
| LMhead | (4096, 151936, 896) |  |

**Backward-pass GEMM shapes derived from the above:**

| Op | dX = dY@W (M,N,K) | dW = dY^T@X (M,N,K) |
|---|---|---|
| Q | (4096, 896, 896) | (896, 896, 4096) |
| K | (4096, 896, 128) | (128, 896, 4096) |
| V | (4096, 896, 128) | (128, 896, 4096) |
| O | (4096, 896, 896) | (896, 896, 4096) |
| Gate | (4096, 896, 4864) | (4864, 896, 4096) |
| Up | (4096, 896, 4864) | (4864, 896, 4096) |
| Down | (4096, 4864, 896) | (896, 4864, 4096) |
| LMhead | (4096, 896, 151936) | (151936, 896, 4096) |

#### Qwen2-1.5B

Source: `https://huggingface.co/Qwen/Qwen2-1.5B/raw/main/config.json` (fetched and verified this session)


| Config field | Value |
|---|---|
| hidden_size (H) | 1536 |
| intermediate_size (real per-branch FFN width, I) | 8960 |
| num_hidden_layers | 28 |
| num_attention_heads (QH) | 12 |
| num_key_value_heads (KVH) | 2 |
| head_dim (HD) | 128 |
| Q_dim = QH*HD | 1536 (=H, square) |
| KV_dim = KVH*HD | 256 (GQA ratio 6:1)
| vocab_size (V) | 151936 |
| tie_word_embeddings | True |
| QKV projection style | separate (three independent nn.Linear layers) |

**Forward-pass GEMM shapes (M=4096 tokens):**

| Op | (M, N, K) | Notes |
|---|---|---|
| Q | (4096, 1536, 1536) |  |
| K | (4096, 256, 1536) |  |
| V | (4096, 256, 1536) |  |
| O | (4096, 1536, 1536) |  |
| Gate | (4096, 8960, 1536) |  |
| Up | (4096, 8960, 1536) |  |
| Down | (4096, 1536, 8960) |  |
| LMhead | (4096, 151936, 1536) |  |

**Backward-pass GEMM shapes derived from the above:**

| Op | dX = dY@W (M,N,K) | dW = dY^T@X (M,N,K) |
|---|---|---|
| Q | (4096, 1536, 1536) | (1536, 1536, 4096) |
| K | (4096, 1536, 256) | (256, 1536, 4096) |
| V | (4096, 1536, 256) | (256, 1536, 4096) |
| O | (4096, 1536, 1536) | (1536, 1536, 4096) |
| Gate | (4096, 1536, 8960) | (8960, 1536, 4096) |
| Up | (4096, 1536, 8960) | (8960, 1536, 4096) |
| Down | (4096, 8960, 1536) | (1536, 8960, 4096) |
| LMhead | (4096, 1536, 151936) | (151936, 1536, 4096) |

#### Qwen2-7B

Source: `https://huggingface.co/Qwen/Qwen2-7B/raw/main/config.json` (fetched and verified this session)


| Config field | Value |
|---|---|
| hidden_size (H) | 3584 |
| intermediate_size (real per-branch FFN width, I) | 18944 |
| num_hidden_layers | 28 |
| num_attention_heads (QH) | 28 |
| num_key_value_heads (KVH) | 4 |
| head_dim (HD) | 128 |
| Q_dim = QH*HD | 3584 (=H, square) |
| KV_dim = KVH*HD | 512 (GQA ratio 7:1)
| vocab_size (V) | 152064 |
| tie_word_embeddings | False |
| QKV projection style | separate (three independent nn.Linear layers) |

**Forward-pass GEMM shapes (M=4096 tokens):**

| Op | (M, N, K) | Notes |
|---|---|---|
| Q | (4096, 3584, 3584) |  |
| K | (4096, 512, 3584) |  |
| V | (4096, 512, 3584) |  |
| O | (4096, 3584, 3584) |  |
| Gate | (4096, 18944, 3584) |  |
| Up | (4096, 18944, 3584) |  |
| Down | (4096, 3584, 18944) |  |
| LMhead | (4096, 152064, 3584) |  |

**Backward-pass GEMM shapes derived from the above:**

| Op | dX = dY@W (M,N,K) | dW = dY^T@X (M,N,K) |
|---|---|---|
| Q | (4096, 3584, 3584) | (3584, 3584, 4096) |
| K | (4096, 3584, 512) | (512, 3584, 4096) |
| V | (4096, 3584, 512) | (512, 3584, 4096) |
| O | (4096, 3584, 3584) | (3584, 3584, 4096) |
| Gate | (4096, 3584, 18944) | (18944, 3584, 4096) |
| Up | (4096, 3584, 18944) | (18944, 3584, 4096) |
| Down | (4096, 18944, 3584) | (3584, 18944, 4096) |
| LMhead | (4096, 3584, 152064) | (152064, 3584, 4096) |

#### Qwen2-72B

Source: `https://huggingface.co/Qwen/Qwen2-72B/raw/main/config.json` (fetched and verified this session)


| Config field | Value |
|---|---|
| hidden_size (H) | 8192 |
| intermediate_size (real per-branch FFN width, I) | 29568 |
| num_hidden_layers | 80 |
| num_attention_heads (QH) | 64 |
| num_key_value_heads (KVH) | 8 |
| head_dim (HD) | 128 |
| Q_dim = QH*HD | 8192 (=H, square) |
| KV_dim = KVH*HD | 1024 (GQA ratio 8:1)
| vocab_size (V) | 152064 |
| tie_word_embeddings | False |
| QKV projection style | separate (three independent nn.Linear layers) |

**Forward-pass GEMM shapes (M=4096 tokens):**

| Op | (M, N, K) | Notes |
|---|---|---|
| Q | (4096, 8192, 8192) |  |
| K | (4096, 1024, 8192) |  |
| V | (4096, 1024, 8192) |  |
| O | (4096, 8192, 8192) |  |
| Gate | (4096, 29568, 8192) |  |
| Up | (4096, 29568, 8192) |  |
| Down | (4096, 8192, 29568) |  |
| LMhead | (4096, 152064, 8192) |  |

**Backward-pass GEMM shapes derived from the above:**

| Op | dX = dY@W (M,N,K) | dW = dY^T@X (M,N,K) |
|---|---|---|
| Q | (4096, 8192, 8192) | (8192, 8192, 4096) |
| K | (4096, 8192, 1024) | (1024, 8192, 4096) |
| V | (4096, 8192, 1024) | (1024, 8192, 4096) |
| O | (4096, 8192, 8192) | (8192, 8192, 4096) |
| Gate | (4096, 8192, 29568) | (29568, 8192, 4096) |
| Up | (4096, 8192, 29568) | (29568, 8192, 4096) |
| Down | (4096, 29568, 8192) | (8192, 29568, 4096) |
| LMhead | (4096, 8192, 152064) | (152064, 8192, 4096) |

#### Qwen2.5-0.5B

Source: `https://huggingface.co/Qwen/Qwen2.5-0.5B/raw/main/config.json` (fetched and verified this session)


| Config field | Value |
|---|---|
| hidden_size (H) | 896 |
| intermediate_size (real per-branch FFN width, I) | 4864 |
| num_hidden_layers | 24 |
| num_attention_heads (QH) | 14 |
| num_key_value_heads (KVH) | 2 |
| head_dim (HD) | 64 |
| Q_dim = QH*HD | 896 (=H, square) |
| KV_dim = KVH*HD | 128 (GQA ratio 7:1)
| vocab_size (V) | 151936 |
| tie_word_embeddings | True |
| QKV projection style | separate (three independent nn.Linear layers) |

**Forward-pass GEMM shapes (M=4096 tokens):**

| Op | (M, N, K) | Notes |
|---|---|---|
| Q | (4096, 896, 896) |  |
| K | (4096, 128, 896) |  |
| V | (4096, 128, 896) |  |
| O | (4096, 896, 896) |  |
| Gate | (4096, 4864, 896) |  |
| Up | (4096, 4864, 896) |  |
| Down | (4096, 896, 4864) |  |
| LMhead | (4096, 151936, 896) |  |

**Backward-pass GEMM shapes derived from the above:**

| Op | dX = dY@W (M,N,K) | dW = dY^T@X (M,N,K) |
|---|---|---|
| Q | (4096, 896, 896) | (896, 896, 4096) |
| K | (4096, 896, 128) | (128, 896, 4096) |
| V | (4096, 896, 128) | (128, 896, 4096) |
| O | (4096, 896, 896) | (896, 896, 4096) |
| Gate | (4096, 896, 4864) | (4864, 896, 4096) |
| Up | (4096, 896, 4864) | (4864, 896, 4096) |
| Down | (4096, 4864, 896) | (896, 4864, 4096) |
| LMhead | (4096, 896, 151936) | (151936, 896, 4096) |

#### Qwen2.5-1.5B

Source: `https://huggingface.co/Qwen/Qwen2.5-1.5B/raw/main/config.json` (fetched and verified this session)


| Config field | Value |
|---|---|
| hidden_size (H) | 1536 |
| intermediate_size (real per-branch FFN width, I) | 8960 |
| num_hidden_layers | 28 |
| num_attention_heads (QH) | 12 |
| num_key_value_heads (KVH) | 2 |
| head_dim (HD) | 128 |
| Q_dim = QH*HD | 1536 (=H, square) |
| KV_dim = KVH*HD | 256 (GQA ratio 6:1)
| vocab_size (V) | 151936 |
| tie_word_embeddings | True |
| QKV projection style | separate (three independent nn.Linear layers) |

**Forward-pass GEMM shapes (M=4096 tokens):**

| Op | (M, N, K) | Notes |
|---|---|---|
| Q | (4096, 1536, 1536) |  |
| K | (4096, 256, 1536) |  |
| V | (4096, 256, 1536) |  |
| O | (4096, 1536, 1536) |  |
| Gate | (4096, 8960, 1536) |  |
| Up | (4096, 8960, 1536) |  |
| Down | (4096, 1536, 8960) |  |
| LMhead | (4096, 151936, 1536) |  |

**Backward-pass GEMM shapes derived from the above:**

| Op | dX = dY@W (M,N,K) | dW = dY^T@X (M,N,K) |
|---|---|---|
| Q | (4096, 1536, 1536) | (1536, 1536, 4096) |
| K | (4096, 1536, 256) | (256, 1536, 4096) |
| V | (4096, 1536, 256) | (256, 1536, 4096) |
| O | (4096, 1536, 1536) | (1536, 1536, 4096) |
| Gate | (4096, 1536, 8960) | (8960, 1536, 4096) |
| Up | (4096, 1536, 8960) | (8960, 1536, 4096) |
| Down | (4096, 8960, 1536) | (1536, 8960, 4096) |
| LMhead | (4096, 1536, 151936) | (151936, 1536, 4096) |

#### Qwen2.5-3B

Source: `https://huggingface.co/Qwen/Qwen2.5-3B/raw/main/config.json` (fetched and verified this session)


| Config field | Value |
|---|---|
| hidden_size (H) | 2048 |
| intermediate_size (real per-branch FFN width, I) | 11008 |
| num_hidden_layers | 36 |
| num_attention_heads (QH) | 16 |
| num_key_value_heads (KVH) | 2 |
| head_dim (HD) | 128 |
| Q_dim = QH*HD | 2048 (=H, square) |
| KV_dim = KVH*HD | 256 (GQA ratio 8:1)
| vocab_size (V) | 151936 |
| tie_word_embeddings | True |
| QKV projection style | separate (three independent nn.Linear layers) |

**Forward-pass GEMM shapes (M=4096 tokens):**

| Op | (M, N, K) | Notes |
|---|---|---|
| Q | (4096, 2048, 2048) |  |
| K | (4096, 256, 2048) |  |
| V | (4096, 256, 2048) |  |
| O | (4096, 2048, 2048) |  |
| Gate | (4096, 11008, 2048) |  |
| Up | (4096, 11008, 2048) |  |
| Down | (4096, 2048, 11008) |  |
| LMhead | (4096, 151936, 2048) |  |

**Backward-pass GEMM shapes derived from the above:**

| Op | dX = dY@W (M,N,K) | dW = dY^T@X (M,N,K) |
|---|---|---|
| Q | (4096, 2048, 2048) | (2048, 2048, 4096) |
| K | (4096, 2048, 256) | (256, 2048, 4096) |
| V | (4096, 2048, 256) | (256, 2048, 4096) |
| O | (4096, 2048, 2048) | (2048, 2048, 4096) |
| Gate | (4096, 2048, 11008) | (11008, 2048, 4096) |
| Up | (4096, 2048, 11008) | (11008, 2048, 4096) |
| Down | (4096, 11008, 2048) | (2048, 11008, 4096) |
| LMhead | (4096, 2048, 151936) | (151936, 2048, 4096) |

#### Qwen2.5-7B

Source: `https://huggingface.co/Qwen/Qwen2.5-7B/raw/main/config.json` (fetched and verified this session)


| Config field | Value |
|---|---|
| hidden_size (H) | 3584 |
| intermediate_size (real per-branch FFN width, I) | 18944 |
| num_hidden_layers | 28 |
| num_attention_heads (QH) | 28 |
| num_key_value_heads (KVH) | 4 |
| head_dim (HD) | 128 |
| Q_dim = QH*HD | 3584 (=H, square) |
| KV_dim = KVH*HD | 512 (GQA ratio 7:1)
| vocab_size (V) | 152064 |
| tie_word_embeddings | False |
| QKV projection style | separate (three independent nn.Linear layers) |

**Forward-pass GEMM shapes (M=4096 tokens):**

| Op | (M, N, K) | Notes |
|---|---|---|
| Q | (4096, 3584, 3584) |  |
| K | (4096, 512, 3584) |  |
| V | (4096, 512, 3584) |  |
| O | (4096, 3584, 3584) |  |
| Gate | (4096, 18944, 3584) |  |
| Up | (4096, 18944, 3584) |  |
| Down | (4096, 3584, 18944) |  |
| LMhead | (4096, 152064, 3584) |  |

**Backward-pass GEMM shapes derived from the above:**

| Op | dX = dY@W (M,N,K) | dW = dY^T@X (M,N,K) |
|---|---|---|
| Q | (4096, 3584, 3584) | (3584, 3584, 4096) |
| K | (4096, 3584, 512) | (512, 3584, 4096) |
| V | (4096, 3584, 512) | (512, 3584, 4096) |
| O | (4096, 3584, 3584) | (3584, 3584, 4096) |
| Gate | (4096, 3584, 18944) | (18944, 3584, 4096) |
| Up | (4096, 3584, 18944) | (18944, 3584, 4096) |
| Down | (4096, 18944, 3584) | (3584, 18944, 4096) |
| LMhead | (4096, 3584, 152064) | (152064, 3584, 4096) |

#### Qwen2.5-14B

Source: `https://huggingface.co/Qwen/Qwen2.5-14B/raw/main/config.json` (fetched and verified this session)


| Config field | Value |
|---|---|
| hidden_size (H) | 5120 |
| intermediate_size (real per-branch FFN width, I) | 13824 |
| num_hidden_layers | 48 |
| num_attention_heads (QH) | 40 |
| num_key_value_heads (KVH) | 8 |
| head_dim (HD) | 128 |
| Q_dim = QH*HD | 5120 (=H, square) |
| KV_dim = KVH*HD | 1024 (GQA ratio 5:1)
| vocab_size (V) | 152064 |
| tie_word_embeddings | False |
| QKV projection style | separate (three independent nn.Linear layers) |

**Forward-pass GEMM shapes (M=4096 tokens):**

| Op | (M, N, K) | Notes |
|---|---|---|
| Q | (4096, 5120, 5120) |  |
| K | (4096, 1024, 5120) |  |
| V | (4096, 1024, 5120) |  |
| O | (4096, 5120, 5120) |  |
| Gate | (4096, 13824, 5120) |  |
| Up | (4096, 13824, 5120) |  |
| Down | (4096, 5120, 13824) |  |
| LMhead | (4096, 152064, 5120) |  |

**Backward-pass GEMM shapes derived from the above:**

| Op | dX = dY@W (M,N,K) | dW = dY^T@X (M,N,K) |
|---|---|---|
| Q | (4096, 5120, 5120) | (5120, 5120, 4096) |
| K | (4096, 5120, 1024) | (1024, 5120, 4096) |
| V | (4096, 5120, 1024) | (1024, 5120, 4096) |
| O | (4096, 5120, 5120) | (5120, 5120, 4096) |
| Gate | (4096, 5120, 13824) | (13824, 5120, 4096) |
| Up | (4096, 5120, 13824) | (13824, 5120, 4096) |
| Down | (4096, 13824, 5120) | (5120, 13824, 4096) |
| LMhead | (4096, 5120, 152064) | (152064, 5120, 4096) |

#### Qwen2.5-32B

Source: `https://huggingface.co/Qwen/Qwen2.5-32B/raw/main/config.json` (fetched and verified this session)


| Config field | Value |
|---|---|
| hidden_size (H) | 5120 |
| intermediate_size (real per-branch FFN width, I) | 27648 |
| num_hidden_layers | 64 |
| num_attention_heads (QH) | 40 |
| num_key_value_heads (KVH) | 8 |
| head_dim (HD) | 128 |
| Q_dim = QH*HD | 5120 (=H, square) |
| KV_dim = KVH*HD | 1024 (GQA ratio 5:1)
| vocab_size (V) | 152064 |
| tie_word_embeddings | False |
| QKV projection style | separate (three independent nn.Linear layers) |

**Forward-pass GEMM shapes (M=4096 tokens):**

| Op | (M, N, K) | Notes |
|---|---|---|
| Q | (4096, 5120, 5120) |  |
| K | (4096, 1024, 5120) |  |
| V | (4096, 1024, 5120) |  |
| O | (4096, 5120, 5120) |  |
| Gate | (4096, 27648, 5120) |  |
| Up | (4096, 27648, 5120) |  |
| Down | (4096, 5120, 27648) |  |
| LMhead | (4096, 152064, 5120) |  |

**Backward-pass GEMM shapes derived from the above:**

| Op | dX = dY@W (M,N,K) | dW = dY^T@X (M,N,K) |
|---|---|---|
| Q | (4096, 5120, 5120) | (5120, 5120, 4096) |
| K | (4096, 5120, 1024) | (1024, 5120, 4096) |
| V | (4096, 5120, 1024) | (1024, 5120, 4096) |
| O | (4096, 5120, 5120) | (5120, 5120, 4096) |
| Gate | (4096, 5120, 27648) | (27648, 5120, 4096) |
| Up | (4096, 5120, 27648) | (27648, 5120, 4096) |
| Down | (4096, 27648, 5120) | (5120, 27648, 4096) |
| LMhead | (4096, 5120, 152064) | (152064, 5120, 4096) |

#### Qwen2.5-72B

Source: `https://huggingface.co/Qwen/Qwen2.5-72B/raw/main/config.json` (fetched and verified this session)


| Config field | Value |
|---|---|
| hidden_size (H) | 8192 |
| intermediate_size (real per-branch FFN width, I) | 29568 |
| num_hidden_layers | 80 |
| num_attention_heads (QH) | 64 |
| num_key_value_heads (KVH) | 8 |
| head_dim (HD) | 128 |
| Q_dim = QH*HD | 8192 (=H, square) |
| KV_dim = KVH*HD | 1024 (GQA ratio 8:1)
| vocab_size (V) | 152064 |
| tie_word_embeddings | False |
| QKV projection style | separate (three independent nn.Linear layers) |

**Forward-pass GEMM shapes (M=4096 tokens):**

| Op | (M, N, K) | Notes |
|---|---|---|
| Q | (4096, 8192, 8192) |  |
| K | (4096, 1024, 8192) |  |
| V | (4096, 1024, 8192) |  |
| O | (4096, 8192, 8192) |  |
| Gate | (4096, 29568, 8192) |  |
| Up | (4096, 29568, 8192) |  |
| Down | (4096, 8192, 29568) |  |
| LMhead | (4096, 152064, 8192) |  |

**Backward-pass GEMM shapes derived from the above:**

| Op | dX = dY@W (M,N,K) | dW = dY^T@X (M,N,K) |
|---|---|---|
| Q | (4096, 8192, 8192) | (8192, 8192, 4096) |
| K | (4096, 8192, 1024) | (1024, 8192, 4096) |
| V | (4096, 8192, 1024) | (1024, 8192, 4096) |
| O | (4096, 8192, 8192) | (8192, 8192, 4096) |
| Gate | (4096, 8192, 29568) | (29568, 8192, 4096) |
| Up | (4096, 8192, 29568) | (29568, 8192, 4096) |
| Down | (4096, 29568, 8192) | (8192, 29568, 4096) |
| LMhead | (4096, 8192, 152064) | (152064, 8192, 4096) |

#### Qwen3-0.6B

Source: `https://huggingface.co/Qwen/Qwen3-0.6B/raw/main/config.json` (fetched and verified this session)


| Config field | Value |
|---|---|
| hidden_size (H) | 1024 |
| intermediate_size (real per-branch FFN width, I) | 3072 |
| num_hidden_layers | 28 |
| num_attention_heads (QH) | 16 |
| num_key_value_heads (KVH) | 8 |
| head_dim (HD) | 128 |
| Q_dim = QH*HD | 2048 **(non-square vs H)** |
| KV_dim = KVH*HD | 1024 (GQA ratio 2:1)
| vocab_size (V) | 151936 |
| tie_word_embeddings | True |
| QKV projection style | separate (three independent nn.Linear layers) |

**Forward-pass GEMM shapes (M=4096 tokens):**

| Op | (M, N, K) | Notes |
|---|---|---|
| Q | (4096, 2048, 1024) | non-square |
| K | (4096, 1024, 1024) |  |
| V | (4096, 1024, 1024) |  |
| O | (4096, 1024, 2048) | non-square |
| Gate | (4096, 3072, 1024) |  |
| Up | (4096, 3072, 1024) |  |
| Down | (4096, 1024, 3072) |  |
| LMhead | (4096, 151936, 1024) |  |

**Backward-pass GEMM shapes derived from the above:**

| Op | dX = dY@W (M,N,K) | dW = dY^T@X (M,N,K) |
|---|---|---|
| Q | (4096, 1024, 2048) | (2048, 1024, 4096) |
| K | (4096, 1024, 1024) | (1024, 1024, 4096) |
| V | (4096, 1024, 1024) | (1024, 1024, 4096) |
| O | (4096, 2048, 1024) | (1024, 2048, 4096) |
| Gate | (4096, 1024, 3072) | (3072, 1024, 4096) |
| Up | (4096, 1024, 3072) | (3072, 1024, 4096) |
| Down | (4096, 3072, 1024) | (1024, 3072, 4096) |
| LMhead | (4096, 1024, 151936) | (151936, 1024, 4096) |

#### Qwen3-1.7B

Source: `https://huggingface.co/Qwen/Qwen3-1.7B/raw/main/config.json` (fetched and verified this session)


| Config field | Value |
|---|---|
| hidden_size (H) | 2048 |
| intermediate_size (real per-branch FFN width, I) | 6144 |
| num_hidden_layers | 28 |
| num_attention_heads (QH) | 16 |
| num_key_value_heads (KVH) | 8 |
| head_dim (HD) | 128 |
| Q_dim = QH*HD | 2048 (=H, square) |
| KV_dim = KVH*HD | 1024 (GQA ratio 2:1)
| vocab_size (V) | 151936 |
| tie_word_embeddings | True |
| QKV projection style | separate (three independent nn.Linear layers) |

**Forward-pass GEMM shapes (M=4096 tokens):**

| Op | (M, N, K) | Notes |
|---|---|---|
| Q | (4096, 2048, 2048) |  |
| K | (4096, 1024, 2048) |  |
| V | (4096, 1024, 2048) |  |
| O | (4096, 2048, 2048) |  |
| Gate | (4096, 6144, 2048) |  |
| Up | (4096, 6144, 2048) |  |
| Down | (4096, 2048, 6144) |  |
| LMhead | (4096, 151936, 2048) |  |

**Backward-pass GEMM shapes derived from the above:**

| Op | dX = dY@W (M,N,K) | dW = dY^T@X (M,N,K) |
|---|---|---|
| Q | (4096, 2048, 2048) | (2048, 2048, 4096) |
| K | (4096, 2048, 1024) | (1024, 2048, 4096) |
| V | (4096, 2048, 1024) | (1024, 2048, 4096) |
| O | (4096, 2048, 2048) | (2048, 2048, 4096) |
| Gate | (4096, 2048, 6144) | (6144, 2048, 4096) |
| Up | (4096, 2048, 6144) | (6144, 2048, 4096) |
| Down | (4096, 6144, 2048) | (2048, 6144, 4096) |
| LMhead | (4096, 2048, 151936) | (151936, 2048, 4096) |

#### Qwen3-4B

Source: `https://huggingface.co/Qwen/Qwen3-4B/raw/main/config.json` (fetched and verified this session)


| Config field | Value |
|---|---|
| hidden_size (H) | 2560 |
| intermediate_size (real per-branch FFN width, I) | 9728 |
| num_hidden_layers | 36 |
| num_attention_heads (QH) | 32 |
| num_key_value_heads (KVH) | 8 |
| head_dim (HD) | 128 |
| Q_dim = QH*HD | 4096 **(non-square vs H)** |
| KV_dim = KVH*HD | 1024 (GQA ratio 4:1)
| vocab_size (V) | 151936 |
| tie_word_embeddings | True |
| QKV projection style | separate (three independent nn.Linear layers) |

**Forward-pass GEMM shapes (M=4096 tokens):**

| Op | (M, N, K) | Notes |
|---|---|---|
| Q | (4096, 4096, 2560) | non-square |
| K | (4096, 1024, 2560) |  |
| V | (4096, 1024, 2560) |  |
| O | (4096, 2560, 4096) | non-square |
| Gate | (4096, 9728, 2560) |  |
| Up | (4096, 9728, 2560) |  |
| Down | (4096, 2560, 9728) |  |
| LMhead | (4096, 151936, 2560) |  |

**Backward-pass GEMM shapes derived from the above:**

| Op | dX = dY@W (M,N,K) | dW = dY^T@X (M,N,K) |
|---|---|---|
| Q | (4096, 2560, 4096) | (4096, 2560, 4096) |
| K | (4096, 2560, 1024) | (1024, 2560, 4096) |
| V | (4096, 2560, 1024) | (1024, 2560, 4096) |
| O | (4096, 4096, 2560) | (2560, 4096, 4096) |
| Gate | (4096, 2560, 9728) | (9728, 2560, 4096) |
| Up | (4096, 2560, 9728) | (9728, 2560, 4096) |
| Down | (4096, 9728, 2560) | (2560, 9728, 4096) |
| LMhead | (4096, 2560, 151936) | (151936, 2560, 4096) |

#### Qwen3-8B

Source: `https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json` (fetched and verified this session)


| Config field | Value |
|---|---|
| hidden_size (H) | 4096 |
| intermediate_size (real per-branch FFN width, I) | 12288 |
| num_hidden_layers | 36 |
| num_attention_heads (QH) | 32 |
| num_key_value_heads (KVH) | 8 |
| head_dim (HD) | 128 |
| Q_dim = QH*HD | 4096 (=H, square) |
| KV_dim = KVH*HD | 1024 (GQA ratio 4:1)
| vocab_size (V) | 151936 |
| tie_word_embeddings | False |
| QKV projection style | separate (three independent nn.Linear layers) |

**Forward-pass GEMM shapes (M=4096 tokens):**

| Op | (M, N, K) | Notes |
|---|---|---|
| Q | (4096, 4096, 4096) |  |
| K | (4096, 1024, 4096) |  |
| V | (4096, 1024, 4096) |  |
| O | (4096, 4096, 4096) |  |
| Gate | (4096, 12288, 4096) |  |
| Up | (4096, 12288, 4096) |  |
| Down | (4096, 4096, 12288) |  |
| LMhead | (4096, 151936, 4096) |  |

**Backward-pass GEMM shapes derived from the above:**

| Op | dX = dY@W (M,N,K) | dW = dY^T@X (M,N,K) |
|---|---|---|
| Q | (4096, 4096, 4096) | (4096, 4096, 4096) |
| K | (4096, 4096, 1024) | (1024, 4096, 4096) |
| V | (4096, 4096, 1024) | (1024, 4096, 4096) |
| O | (4096, 4096, 4096) | (4096, 4096, 4096) |
| Gate | (4096, 4096, 12288) | (12288, 4096, 4096) |
| Up | (4096, 4096, 12288) | (12288, 4096, 4096) |
| Down | (4096, 12288, 4096) | (4096, 12288, 4096) |
| LMhead | (4096, 4096, 151936) | (151936, 4096, 4096) |

#### Qwen3-14B

Source: `https://huggingface.co/Qwen/Qwen3-14B/raw/main/config.json` (fetched and verified this session)


| Config field | Value |
|---|---|
| hidden_size (H) | 5120 |
| intermediate_size (real per-branch FFN width, I) | 17408 |
| num_hidden_layers | 40 |
| num_attention_heads (QH) | 40 |
| num_key_value_heads (KVH) | 8 |
| head_dim (HD) | 128 |
| Q_dim = QH*HD | 5120 (=H, square) |
| KV_dim = KVH*HD | 1024 (GQA ratio 5:1)
| vocab_size (V) | 151936 |
| tie_word_embeddings | False |
| QKV projection style | separate (three independent nn.Linear layers) |

**Forward-pass GEMM shapes (M=4096 tokens):**

| Op | (M, N, K) | Notes |
|---|---|---|
| Q | (4096, 5120, 5120) |  |
| K | (4096, 1024, 5120) |  |
| V | (4096, 1024, 5120) |  |
| O | (4096, 5120, 5120) |  |
| Gate | (4096, 17408, 5120) |  |
| Up | (4096, 17408, 5120) |  |
| Down | (4096, 5120, 17408) |  |
| LMhead | (4096, 151936, 5120) |  |

**Backward-pass GEMM shapes derived from the above:**

| Op | dX = dY@W (M,N,K) | dW = dY^T@X (M,N,K) |
|---|---|---|
| Q | (4096, 5120, 5120) | (5120, 5120, 4096) |
| K | (4096, 5120, 1024) | (1024, 5120, 4096) |
| V | (4096, 5120, 1024) | (1024, 5120, 4096) |
| O | (4096, 5120, 5120) | (5120, 5120, 4096) |
| Gate | (4096, 5120, 17408) | (17408, 5120, 4096) |
| Up | (4096, 5120, 17408) | (17408, 5120, 4096) |
| Down | (4096, 17408, 5120) | (5120, 17408, 4096) |
| LMhead | (4096, 5120, 151936) | (151936, 5120, 4096) |

#### Qwen3-32B

Source: `https://huggingface.co/Qwen/Qwen3-32B/raw/main/config.json` (fetched and verified this session)


| Config field | Value |
|---|---|
| hidden_size (H) | 5120 |
| intermediate_size (real per-branch FFN width, I) | 25600 |
| num_hidden_layers | 64 |
| num_attention_heads (QH) | 64 |
| num_key_value_heads (KVH) | 8 |
| head_dim (HD) | 128 |
| Q_dim = QH*HD | 8192 **(non-square vs H)** |
| KV_dim = KVH*HD | 1024 (GQA ratio 8:1)
| vocab_size (V) | 151936 |
| tie_word_embeddings | False |
| QKV projection style | separate (three independent nn.Linear layers) |

**Forward-pass GEMM shapes (M=4096 tokens):**

| Op | (M, N, K) | Notes |
|---|---|---|
| Q | (4096, 8192, 5120) | non-square |
| K | (4096, 1024, 5120) |  |
| V | (4096, 1024, 5120) |  |
| O | (4096, 5120, 8192) | non-square |
| Gate | (4096, 25600, 5120) |  |
| Up | (4096, 25600, 5120) |  |
| Down | (4096, 5120, 25600) |  |
| LMhead | (4096, 151936, 5120) |  |

**Backward-pass GEMM shapes derived from the above:**

| Op | dX = dY@W (M,N,K) | dW = dY^T@X (M,N,K) |
|---|---|---|
| Q | (4096, 5120, 8192) | (8192, 5120, 4096) |
| K | (4096, 5120, 1024) | (1024, 5120, 4096) |
| V | (4096, 5120, 1024) | (1024, 5120, 4096) |
| O | (4096, 8192, 5120) | (5120, 8192, 4096) |
| Gate | (4096, 5120, 25600) | (25600, 5120, 4096) |
| Up | (4096, 5120, 25600) | (25600, 5120, 4096) |
| Down | (4096, 25600, 5120) | (5120, 25600, 4096) |
| LMhead | (4096, 5120, 151936) | (151936, 5120, 4096) |

### H3.7 Open items / next steps

- **M-axis is still not closed** — every shape above uses M=4096 (matching the prior
  campaign's convention for comparability). The prior campaign's own audit (§23.4 of
  `analyzations_b300.md`) explicitly flagged this as the one real gap in ITS catalog and it
  was never closed there either — decode-regime M (1-128) and larger training M (8192,
  16384, 32768) are not yet represented anywhere in either catalog. Needs its own sweep.
- **Attention QK^T/softmax/@V exclusion not re-verified against BluTrain's own attention
  kernel specifically** (only verified as reasonable by analogy to the prior campaign's
  finding for a different, cuBLAS-fronting training stack) — worth a direct check of
  BluTrain's own attention implementation before fully trusting this exclusion.
- **MoE and hybrid-architecture (Qwen3-Next/3.5/3.6) shapes are entirely out of scope**
  here per explicit decision (H3.1) — not forgotten, deliberately deferred.
- **Next phase** (not started): for each of the 168 shapes above, research what kernel
  variant each reference library (cuBLAS, CUTLASS, DeepGEMM, ThunderKittens — PyTorch/
  TensorFlow pending a quick check on whether they do anything beyond delegating to
  cuBLAS/CUTLASS on Hopper) actually dispatches, using the same trace-based methodology
  already proven this session (nsys kernel-name extraction) and in the prior B300 campaign
  (§16.3's cuBLAS-kernel-per-shape table).

## §H4. Cross-library reference sweep — infrastructure prepared (2026-07-21), execution deferred to next session

### H4.1 Goal and explicit ground rules

For every one of the 164 distinct (M,N,K) shapes from §H3, capture — with nsys/ncu proof, not
guesses — exactly which kernel variant each of cuBLAS, CUTLASS, PyTorch, TensorFlow, DeepGEMM,
and ThunderKittens actually dispatches, with **zero hardcoding of which algorithm/variant to
use**: every library call lets that library's own internal choice happen naturally (no
`CUBLAS_GEMM_ALGO*` override, no CUTLASS tile pinned by us, no PyTorch/TF backend override).
Explicitly deferred to a later phase (to document then, not now): building our own cost-model/
heuristic dispatcher from this data — today's work is observation-only infrastructure.

**Explicit instruction being followed:** do not assume PyTorch/TensorFlow "just call cuBLAS so
they'll pick the same kernel." Both are tested as their own independent library entries in this
sweep specifically so that assumption gets checked with real profiling data, not asserted.

### H4.2 Feasibility check performed before building anything (all verified this session, not assumed)

- **PyTorch 2.7.1+cu126** — installed locally, `torch.cuda.is_available()=True`. Confirmed
  working end-to-end on this box (RTX 3060) as a syntax/logic sanity check; real numbers need
  the H200.
- **TensorFlow 2.19.0** — installed locally, confirmed running (GPU device detected, executes
  correctly). Note from the local run: oneDNN warns it only supports bf16 on AVX-512 platforms
  and falls back to Eigen otherwise — worth checking whether the H200 box's CPU has AVX-512,
  since a fallback path could mean TF's bf16 matmul here reflects a CPU-library limitation
  rather than a GPU kernel choice; flagged for verification during the actual sweep.
- **DeepGEMM** — `deep_gemm` Python module is NOT pip-installed in the current environment
  (`ModuleNotFoundError` on local test) — confirmed real prerequisite, not yet done:
  `pip install -e <DeepGEMM repo root>` (has `setup.py`) needed on the H200 before this part of
  the sweep can run.
- **CUTLASS** — `tools/profiler` source exists but `cutlass_profiler` binary is NOT built yet —
  confirmed via direct filesystem check. Needs a real CMake build first (`cmake ..
  -DCUTLASS_NVCC_ARCHS=90a -DCUTLASS_LIBRARY_KERNELS=all && make cutlass_profiler -j$(nproc)`)
  — budget real wall-clock time for this, CUTLASS's template instantiation surface makes this a
  genuinely slow build, not a quick step.
- **ThunderKittens** — `kernels/gemm/bf16_h100/bf16_h100_gemm.cu` exists and is directly usable.
  Important architectural difference from every other library in this sweep, confirmed by
  reading the source: ThunderKittens has **no runtime shape-based kernel dispatcher at all** —
  `bf16_h100_gemm.cu` is one hand-written kernel with a fixed tile config
  (`matmul_template<2,4,8>`, i.e. `M_BLOCK=2, N_BLOCK=4, SUPER_M=8`) compiled in via template
  arguments; only the actual M/N/K problem size is a runtime value
  (`run_benchmark<matmul_template<2,4,8>>(M, N, K)` inside `main()`). Testing ThunderKittens per
  shape means "run their one compiled kernel against this shape and measure it," NOT "see which
  variant they picked" the way that framing applies to cuBLAS/CUTLASS/DeepGEMM. Documented here
  explicitly so this distinction isn't lost or forced into the wrong framing later.

### H4.3 Corrected finding: DeepGEMM has its own genuine BF16 kernel, not just a cuBLASLt wrapper

An earlier assumption in this conversation (before reading DeepGEMM's actual source) was that
DeepGEMM might be FP8/FP4-focused with BF16 just passed through to cuBLASLt. Reading
`DeepGEMM/tests/test_bf16.py` directly disproved this: DeepGEMM exposes **two distinct entry
points** for BF16 — `deep_gemm.bf16_gemm_nt(a, b, d, c=c)` (DeepGEMM's own hand-written kernel)
and `deep_gemm.cublaslt_gemm_nt(a, b, d, c=c)` (an explicit cuBLASLt pass-through, used as
**DeepGEMM's own reference baseline** — their test file literally computes and prints "Average
speedup over cuBLASLt", i.e. DeepGEMM positions itself as a cuBLASLt competitor with a genuinely
distinct kernel, the same way BluBLAS does, not a wrapper around it). The `qwen_shape_deepgemm.py`
test built for this sweep (H4.4) benchmarks BOTH — DeepGEMM's own choice (what the sweep actually
wants) and their own cuBLASLt baseline call, for direct within-tool cross-checking. Confirmed
from source: `'nt'` layout means A is K-major/non-transposed ([M,K] row-major) and B is
transposed ([N,K] used as Bᵀ) — matches this whole sweep's `out = in @ Wᵀ` convention exactly.

### H4.4 Infrastructure built this session (all files, all pushed to the repo)

**`tools/qwen_shapes.json`** — the single source of truth every test script reads from (well,
currently each Python/shell script takes M/N/K as explicit CLI args, driven by the master
script's own read of this JSON — no shape numbers are hardcoded inside any individual library
test file). Contains all 164 distinct (M,N,K) tuples derived from §H3's 168 fwd+bwd configs
(164, not 168, because a few configs coincide numerically across different models/pass-types —
each JSON entry lists every `(model:op, pass-type)` combination that maps to it, so no
information from §H3 is lost, just deduplicated for testing purposes).

**`examples/qwen_shape_cublas.cu`** — single-shape test, `cublasGemmStridedBatchedEx` with
`CUBLAS_GEMM_DEFAULT` (cuBLAS's own heuristic picks the algorithm, matches the exact call
pattern already proven in `test_bgemm_sm90_vs_cublas.cu` this session, which is what produced
the `nvjet_sm90_*` kernel names we've already learned to decode via nsys). TN layout (A=[M,K]
row-major non-transposed, B=[N,K] row-major transposed) matches every shape's derivation
convention. Locally sanity-checked (M=N=K=128 on the RTX 3060): compiles and runs correctly.

**`examples/qwen_shape_pytorch.py`** — single-shape test via `torch.nn.functional.linear(x, w)`
— deliberately the literal call every real Qwen `nn.Linear` layer makes internally, not a
synthetic `torch.matmul`, so this sweep measures the actual code path a real forward pass
exercises. No backend/algorithm override set. Locally sanity-checked: runs correctly.

**`examples/qwen_shape_tensorflow.py`** — single-shape test via `tf.matmul(x, w,
transpose_b=True)` inside a `@tf.function` (mirrors `tf.keras.layers.Dense`'s actual internal
call). No device/algorithm override set. Locally sanity-checked: runs correctly (see H4.2's
AVX-512/oneDNN note for a caveat to verify during the real sweep).

**`examples/qwen_shape_deepgemm.py`** — single-shape test benchmarking both
`deep_gemm.bf16_gemm_nt` (their own kernel) and `deep_gemm.cublaslt_gemm_nt` (their own
baseline) — see H4.3. Not yet locally testable (`deep_gemm` not installed in this environment,
confirmed real prerequisite for the H200 run, not a script bug).

**`tools/qwen_shape_cutlass.sh`** — drives the `cutlass_profiler` CLI tool (`--operation=Gemm
--m=.. --n=.. --k=.. --A=bf16:row --B=bf16:column --C=bf16:row --accumulator-type=f32
--providers=cutlass`), which sweeps CUTLASS's own registered kernel instantiations for the given
shape and reports the fastest — the correct "let CUTLASS choose" interpretation given CUTLASS
itself has no runtime dispatcher (see H4.2's build prerequisite).

**`tools/qwen_shape_thunderkittens.sh`** — copies (never modifies in place) the vendored
`bf16_h100_gemm.cu`, patches only the `main()` body to call
`run_benchmark<matmul_template<2,4,8>>(M, N, K)` with the sweep's shape instead of the repo's
own hardcoded square-N test, compiles fresh per shape (`-arch=sm_90a`, matching the repo's own
Makefile flags), and runs from a scratch build dir — vendored ThunderKittens repo stays
untouched throughout.

**`tools/run_qwen_shape_sweep.sh`** — master orchestrator. For every (shape, library) pair not
already profiled (resumable — checks for existing report files and skips them, so this can be
safely stopped and restarted across sessions, since a full ~164×6 sweep with both nsys and ncu
per pair is genuinely many hours of wall-clock time), runs nsys (full trace + `--stats=true`)
and ncu (the same targeted 6-metric set already proven this session — register count,
occupancy, SM/DRAM throughput, instructions executed, duration — NOT `--set full`, which was
already found this session to be far slower for no extra proof value here) individually, saving
labeled report files (`shape_<id>_M<m>_N<n>_K<k>.{nsys-rep,ncu-rep,stdout.txt}` per library).

### H4.5 Explicit prerequisites before running the sweep on the H200 (none of these are done yet)

1. `pip install -e <DeepGEMM repo root>` — `deep_gemm` module not currently installed.
2. Build the CUTLASS profiler: `cmake .. -DCUTLASS_NVCC_ARCHS=90a -DCUTLASS_LIBRARY_KERNELS=all
   && make cutlass_profiler -j$(nproc)` inside a `cutlass/build/` dir — not yet built, budget
   real time for this.
3. Set `CUTLASS_BUILD_DIR` and `THUNDERKITTENS_ROOT` environment variables (or pass them
   directly) before invoking `run_qwen_shape_sweep.sh`.
4. Push this session's 8 new files (`tools/qwen_shapes.json`, `tools/qwen_shape_cutlass.sh`,
   `tools/qwen_shape_thunderkittens.sh`, `tools/run_qwen_shape_sweep.sh`,
   `examples/qwen_shape_{cublas.cu,pytorch.py,tensorflow.py,deepgemm.py}`) and pull on the H200.
5. `make test TEST=examples/qwen_shape_cublas.cu` (or equivalent build step) so
   `build/qwen_shape_cublas` exists before the sweep script tries to invoke it.

### H4.6 Explicitly NOT done yet (deferred to the actual run + next analysis phase)

- No shapes have been run on real H200 hardware for this sweep yet — everything in this section
  is infrastructure + local (RTX 3060) syntax/logic sanity checks only.
- No kernel names have been extracted or compared across libraries yet.
- The PyTorch-vs-cuBLAS / TensorFlow-vs-cuBLAS "same variant or different" question (explicitly
  asked to be answered with proof, not assumed) is NOT yet answered — that's exactly what
  running this harness and diffing the resulting kernel names will answer.
- Building our own dispatcher/heuristic/cost-model from whatever this sweep finds is explicitly
  a separate, later task (per direct instruction) — not started, not planned in detail yet
  beyond "this is next after the sweep."

### H4.7 Added after user correction: clock-locking, preheat, and a real clone-on-H200 setup script

User correctly pushed back on running any of this locally — every one of §H4.2's feasibility
checks was done on the local (non-H200) box out of necessity (no H200 access from this
environment), but NOTHING about the actual sweep should run anywhere but the H200, including
cloning the open-source libraries themselves. `tools/setup_qwen_sweep_env.sh` was added to do
exactly that: clones CUTLASS/DeepGEMM/ThunderKittens fresh into a workdir on the H200 (skips
re-cloning if already present, safe to re-run), builds the CUTLASS profiler, `pip install -e`'s
DeepGEMM, and does one small smoke-test compile of each before the real sweep starts.

`run_qwen_shape_sweep.sh` was also updated to bake in the two most consequential measurement
lessons from the prior B300 campaign (`analyzations_b300.md` §47.1, §47.4) directly, rather than
risk re-learning them the hard way on a fresh library-comparison campaign:
- **Clock-locking** (`nvidia-smi -pm 1` + `-lgc 1400`, restored via a `trap ... EXIT` on script
  exit) — §47.1 found cuBLAS-vs-ours comparisons swing ±30% by measurement order alone under
  unlocked boost clocks; 1400MHz specifically was the value that avoided the power-cap engaging
  in that campaign, carried forward here as a starting point, not yet re-validated for H200
  specifically.
- **60-second preheat** before any real measurement — §47.4 found HBM/memory clocks decay over
  the first ~2 minutes independent of the graphics-clock lock, biasing whichever shape/library
  happens to be measured first. Run once at the very start of the sweep script, not per-library.

Both are exactly the kind of thing that would otherwise silently bias a 6-library comparison in
favor of whichever library's test script happens to run first for a given shape — worth having
locked in before a single real measurement is taken, not discovered after the fact.

### H4.8 Session paused mid-setup (2026-07-21 end of day) — resume point for next session

`setup_qwen_sweep_env.sh` was started on the H200 and interrupted (user-quit) during step 1 —
neither PyTorch nor TensorFlow were already present on this H200 box (confirmed, not assumed),
and installing them (`tensorflow[and-cuda]` especially) pulls a large separate CUDA/cuDNN/cuBLAS
package stack on top of the PyTorch wheel, which was taking too long to sit and wait through.
Nothing else in the setup ran yet (CUTLASS clone/build, DeepGEMM clone/install, ThunderKittens
clone all still pending too, in step order after step 1). No damage from the interrupt — the
script is idempotent (checks before cloning/installing anything), so re-running it from scratch
tomorrow picks up cleanly rather than redoing finished work. Recommended for tomorrow: run it
under `nohup`/`tmux` from the start so the slow PyTorch/TensorFlow download doesn't block
anything else, and treat CUTLASS's profiler build (step 2) as the next likely slow step to
budget time for once step 1 clears.
