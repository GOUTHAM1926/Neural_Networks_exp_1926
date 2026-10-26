---
name: feedback-verify-reachability-not-presence
description: A symbol/count check proves presence, never reachability - verify the call site and the actual instructions, not the inventory
metadata:
  type: feedback
---

Counting symbols, files, or objects is NOT verification. It proves a thing EXISTS, never that it
is REACHED. Verify the call site, the compile flags, and the emitted instructions.

**The case that cost hours (2026-09-07, 600M trainer):** our BLAS vendored into the trainer ran
at 60% of cuBLAS. My link check reported "4628 SM90 kernels present in libblublas.so, 3 _kota
exports, 0 unresolved symbols" and I called the wiring verified. It was not. The trainer builds
the vendored BLAS with its OWN Makefile, which never defines `BLUBLAS_HAS_SM90`; our BF16
dispatcher wraps its whole SM90 routing block in `#ifdef BLUBLAS_HAS_SM90`, so the CALL SITE was
preprocessed out. The kernels were all present and completely unreachable - every BF16 GEMM fell
to the SM89 legacy path. nsys showed `bgemm_backward_template_kernel` (SM89) instead of
`bgemm_sm90_kernel`. Adding the define: 44,090 -> 70,300 tok/s, 60% -> 95.6% of cuBLAS.

**Why:** presence and reachability are independent. A preprocessor branch, a runtime arch gate,
a decline predicate, or a dead `if` can all remove the path while every symbol stays in the
binary. So can the reverse - a symbol can be absent yet never needed.

**How to apply:**
* The check that would have caught it: `nm -u` on the CALLER'S OBJECT for the callee symbol.
  0 undefined refs = the call site is gone. 3 = it is really compiled in. Presence in the .so
  tells you nothing about this.
* For any `#ifdef`-gated feature, grep the ACTUAL build flags of the ACTUAL Makefile that
  compiles it - not the repo's own Makefile if something else drives the build.
* Confirm with the emitted artefact: nsys/ncu kernel NAMES for what really ran, `cuobjdump
  --list-elf` for the target arch, SASS for the instructions. Names and counts of what is
  installed are the weakest evidence available.
* Regex counts are their own trap: `dispatch_..._kota` needs 3 chars and the names are `nn`/
  `nt`/`tn`, so a correct build read as 0. Use `grep -F` and print the matches, never just `-c`.
* Honest exit codes: `echo "[$(date)] exit=$?"` RESETS `$?` via the substitution - a failed
  build/run reports exit=0. Capture `rc=$?` on its own line first. This masked two real
  failures in one day.

See [[feedback_backend_ab_snapshot]], [[feedback_vacuous_checks]], [[feedback_samples_not_static_counts]].
