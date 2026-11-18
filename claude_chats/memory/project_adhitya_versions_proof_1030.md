---
name: project_adhitya_versions_proof_1030
description: "Proven 10-30: Adhitya's GitHub CP branch pins TI master, but his D150 runs on P022/P032 loaded our dist_side_merge TI d96b991 (09-28) + BLAS PR #2 + pre-hook embedding (bf16 atomicAdd); pt/synagg = his 03d3790"
metadata:
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-10-30T13:19:33.170Z
---

Checked 2026-10-30. Full proof is in sft_work_documentation.md §45.108 parts 10 and 11.

**GitHub (fetched):**
- TI master = `1c84859` (08-24).
- BluTrain `context-parallel-integration` (`0df6ecd`, 09-26) pins TI `1c84859` and CP `5472925`.
- His 10-22 push = TI `03d3790` + BluTrain `f37c616` (135M trainers). It moved BLAS back to PR #2 (`35e51a0`).
- GitHub CP main ends at `cf1372d`. His `009b371` and his box trainer sources are not on GitHub.

**Servers (user ran read-only commands; I matched `git hash-object` IDs against all 1,247 TI commits):**
- His work is in `/localstorage/adhi_l32/` on both boxes. Every copy is a plain copy, not a checkout.
- D150 run scripts use `BT=/localstorage/adhi_l32/lc_D150_2026-10-26/BluTrain` with `LD_LIBRARY_PATH=$BT/Tensor-Implementations/lib`, and all set `CP_BF16_EMBED=1`.
- That copy = our `dist_side_merge` TI `ad10e7d`..`d96b991` (09-25..28).
  - Exact commit = TI `d96b991` (our dist_side_merge tip, 09-28). Nine fingerprints match it and no other commit.
  - It has BLAS PR #2 and the pre-hook embedding file `1d61b0d` (bf16 -> atomicAdd).
- `pt_D150_2026-10-27/src8k` = exactly `03d3790`.
  - The pt / synagg `libtensor.so` files are md5-identical copies of the `src8k` build.
  - The LC-8k runs (10-24) are not on P022/P032; scripts mention a machine "p15: ~/Adhi/BluTrain".
  - His run trees have no `.git`, so a `git pull` cannot deliver the fix: he needs a fresh clone, then a rebuild.
- `lc_D150_2026-10-26/BluTrain_failed_localsrc` (P032) = the same mixed tree his reply quoted:
  - LossKernels from `ad10e7d`;
  - LossOps from `1c84859`;
  - BLAS from before PR #2.
- The binaries load `libtensor.so` at run time (RUNPATH `$ORIGIN/../../Tensor-Implementations/lib`). Checking the program file alone with `cuobjdump` proves nothing.

**Why:** the user demands proof for every claim about other people's versions.

**How to apply:**
- Never state another person's version without `ls-tree` / `hash-object` proof.
- For a dynamically linked program, follow `LD_LIBRARY_PATH` from the run script to the `.so`.
- Paste only command lines on boxes: the user's pasted output ran as commands twice.
- Avoid `git remote -v` in `~/Downloads/blublas`: its URL embeds a token.

Related: [[project_kernel_plan_1030]], [[project_preproduction_135m_state_1029]].
