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
