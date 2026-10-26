---
name: project_stmatrix_reopen
description: CLOSED — the stmatrix epilogue had a real bug (ignored the column-chunk index), now fixed and bit-exact; but as an optimisation it is a power null
metadata:
  type: project
---

2026-09-11, §H220/§H221. Was an open lead on the strength of cuBLAS emitting 414 STSM in its
epilogue where we emit plain STS.

**THE BUG (found and fixed).** The stmatrix store hardcoded its shared-memory region as
`epi_slot_wg`, where the scalar STS path uses `cbuf_ch = USE_CHUNKED_EPILOGUE ? epi_slot_wg :
ch` -- i.e. by the COLUMN CHUNK when the epilogue is not chunked. `USE_CHUNKED_EPILOGUE ==
(EPI_TILE_N > 0)`, and the ONE config stmatrix had ever been validated on had EPI_TILE_N=64, so
the two coincided and the hardcode was accidentally correct. Every production config has
EPI_TILE_N=0, so BN=256 over C_BOX_N=64 gives four column chunks that were all written on top
of each other: ~24.9 M of 25.2 M elements wrong, every shape, every layout. Fix is one line --
`grp_ch` was already computed two lines above for the skip test. Now 15/15 bit-exact.

**Its three static_asserts all passed.** The relevant one tests `C_BOX_N == EPI_SPAN*8` while
its message claims `EPI_TILE_N*2 == EPI_SWIZZLE_BYTES`; those coincide only when the epilogue
is chunked, because only then is C_BOX_N derived from EPI_TILE_N. On our configs C_BOX_N is 64
because the REBIND sets it. **An assertion that tests a proxy is only as good as the proxy.**

**AS AN OPTIMISATION: NULL.** +0.34% FLOP-weighted power, 6/15 shapes lower, 180 measurements
at a locked uncapped 1200 MHz with independent operands. The old "1.93x" was STORE THROUGHPUT;
the store is a small share of kernel energy.

**Why:** stops this being re-opened as an untested lead, and leaves a correct baseline.

**How to apply:** the fix is kept, the knob stays off. Two reusable lessons: a knob validated
on ONE config is not validated -- diff the code paths it switches against the path in use
before enabling it on a new config family; and never read a partial ABBA (`gate_up wgrad`
showed -2.85% at 13/15 shapes and settled to -0.51% by the end).
See [[project_power_is_operand_toggling]], [[feedback_profiler_measurement_traps]].
