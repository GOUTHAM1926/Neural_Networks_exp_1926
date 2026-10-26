---
name: feedback_make_lib_vs_tests
description: In blublas, plain `make` is all:lib and never relinks the test binaries -- new configs stay invisible
metadata:
  type: feedback
---

`make` in the blublas repo is `all: lib`. It rebuilds `build/libmyblas.so` ONLY. The test binaries
in `examples/` are a SEPARATE target: `make tests`.

**Why it matters:** `examples/test_bgemm_sm90_matrix.cu` embeds its own copy of the config registry
(`BLUBLAS_SM90_CFG_FOREACH*(ADD_ONE, ...)`). A config added to the header compiles into the library
but is INVISIBLE to a stale test binary, so `--cfg=<new>` silently matches nothing and the harness
prints no row at all — not an error, just an absent line.

That cost a full clock-locked GPU pass on 2026-08-20: all four arms printed `NA / REJECT err=`
while the pin measured fine, and I misread it as a registration bug.

Worse, the `tests:` recipe ends in `| grep -v "warning:" || true`, so a FAILED test compile is
swallowed and the previous binary is left in place while the build reports success.

**How to apply:** after touching any config header, build with
`make tests TEST_SOURCES=examples/test_bgemm_sm90_matrix.cu`, then VERIFY the binary rather than
trusting the exit status: `strings build/test_bgemm_sm90_matrix | grep -c <cfg>` must be non-zero.
`tools/h27zo_v1_gate.sh` now aborts on this check before locking clocks.

See [[feedback_separate_build_and_run]], [[project_reduction_build_gotcha]].
