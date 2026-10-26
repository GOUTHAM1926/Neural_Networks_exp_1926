---
name: feedback-smoke-gate-before-every-campaign
description: Run a 4-step smoke gate before any measurement campaign -- a 9-check static verification passed 9/9 on a build whose GEMMs all silently took the SM89 legacy path
metadata: 
  node_type: memory
  type: feedback
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-09-10T07:53:26.323Z
---

**2026-09-10, cost half a day.** A build was verified nine ways and passed 9/9 — 4,756 SM90
kernels present in the cubin, sm_90a image present, H163 pin found in machine code, libblublas
resolving at runtime, libtensor's undefined refs proving it *calls* blublas, the two libtensor.so
confirmed distinct. Every check true. **Every BF16 GEMM was still taking the SM89 legacy path**
(21,449 tok/s instead of 34,264, -37%), because the dispatcher's `#ifdef BLUBLAS_HAS_SM90` guard
was never satisfied and the whole SM90 block had been preprocessed away.

**Presence of kernels in a binary is NOT reachability. Only a RUN proves the path.**
This is [[feedback_verify_reachability_not_presence]] again, and static checks kept passing
because I kept inventing new ways to count symbols instead of executing one step.

## The rule

Before committing GPU time to a campaign, run `h188_smoke.sh` — 4 steps per arm, ~90 s — and
hard-fail on:
1. `WARNING|SM89 legacy|not supported|fallback|no kernel` anywhere in the log
2. the binary's own config echo not matching what was asked (e.g. `[ZeRO-1] variant:` must show
   v2 + symmetric when those flags are set)
3. throughput outside the known-good band (fallback ~21.4k vs healthy ~34-36k on 4B)
4. `mem_alloc` not matching the known-good footprint for that model

It caught the broken build in 4 minutes; the campaign would have wasted an hour and produced
numbers that looked plausible.

## Two verification idioms that produce FALSE PASSES — never reuse

- `ldd X | grep -c libblublas` counts `libblublas.so => not found` as a hit. Grep for
  `libblublas.so => /` and separately assert `=> not found` is zero — and do it under the real
  `RUN_LD_PATH`, not a bare `ldd`.
- `nm -D libblublas.so` shows 0 device kernels — they live in the cubin, not the host symbol
  table. Use `cuobjdump --dump-elf-symbols`.

## The decisive check for a compiled-out guard

`strings libblublas.so | grep -c "built WITHOUT -DBLUBLAS_HAS_SM90"` must be **0** — that string
exists only in the `#ifndef` branch, so its absence proves the macro was defined. Plus
`nm -u Bgemm_dispatcher.o | grep -c SM90` must be > 0. Generally: assert on something that
EXISTS ONLY IN THE FAILURE BRANCH, not on something present either way.

Related: [[feedback_vacuous_checks]], [[feedback_backend_ab_snapshot]],
[[project_regime_effect_is_blas_specific]].

## 2026-10-17: a script that MAKES its inputs must check them, and must stop at the first failed run

run_item3.sh built a tiny SFT set on the box with `head -c src > dst`. The sources did not exist on that box, `head`'s
error went to /dev/null, and the redirect still created EMPTY files. The script checked only one source file and logged
each trainer's exit code instead of stopping, so all 6 runs died in 3 s each and the script still printed PACKED.
- Check the SIZE of every file a script creates (`stat -c %s` == expected), before any build or run.
- Stop at the first run that exits non-zero or misses its summary line (`|| fail`), never log-and-continue.
- A tiny trainer run must still pass the trainer's own config checks: SFT with 64 rows needs BLU_SFT_EPOCHS >= 3
  (warmup + stable < max_steps - 1); read those checks before choosing the size.
