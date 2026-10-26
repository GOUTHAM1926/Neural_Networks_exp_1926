---
name: feedback-i-was-wrong-on-with-blublas-flag
description: CORRECTION - flipping WITH_BLUBLAS on the RUN command IS enough; Kota's Makefile has a stamp mechanism that rebuilds exactly GenMatmul.cu and MatmulBackward.cu. No clean-tensor needed. I claimed otherwise and was wrong.
metadata:
  type: feedback
---

**I asserted that `make` could not notice a `WITH_BLUBLAS` flag change and that `clean-tensor` was
required before switching arms. That was WRONG**, and I asserted it before testing the actual
workflow. The user's colleague was right.

Kota's `Tensor-Implementations/Makefile` has a stamp mechanism built for exactly this:

```make
BLAS_STAMP := $(OBJDIR)/.blas_backend
LAST_BLAS  := $(shell cat $(BLAS_STAMP) 2>/dev/null)
ifneq ($(LAST_BLAS),$(BLAS_BACKEND))
  .PHONY: $(BLAS_OBJ_GENMATMUL) $(BLAS_OBJ_BACKWARD) $(TARGET_SO)
endif
write-blas-stamp:
	@echo "$(BLAS_BACKEND)" > $(BLAS_STAMP)
```

Only `GenMatmul.cu` and `MatmulBackward.cu` are `#ifdef WITH_MYBLAS`-guarded, so only those two
need rebuilding, and the stamp forces exactly those two.

**PROVEN in the real workflow** - `make run-mpi ... WITH_BLUBLAS=1` from a cuBLAS state printed
`Compiling ... GenMatmul.cu` and `Compiling ... MatmulBackward.cu`, and the run came up on our BLAS.

So: **`make clean && make all -jN`, then flip the flag on each run command. That is the workflow.**

**Why I got it wrong:** I grepped for `flagstamp`/`.flags` and missed the target because it is named
`write-blas-stamp`; then I tested a bare `make tensor` instead of the real `make run-mpi` path and
generalised from it. Grep for the MECHANISM (a stamp, a FORCE target, an order-only prereq), not a
guessed name, and always test the user's ACTUAL command.

**The standing check that beats all of this:** the nsys kernel names.
`blublas_ws::bgemm_sm90_kernel` vs `nvjet_sm90_tst_*`, or `[BluBLAS] SM90 dispatcher` in the log.
Symbols and `ldd` say what COULD be called; the profile says what WAS. See
[[feedback-verify-reachability-not-presence]].
