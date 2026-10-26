---
name: feedback-make-parallel-j
description: "Every make command I give must carry -j$(nproc); the Tensor-Implementations Makefile sets no -j, so a plain `make tensor` compiles one file at a time on a 44-core box"
metadata:
  node_type: memory
  type: feedback
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-09-26T16:20:19.209Z
---

Always write builds as `make -j$(nproc) tensor WITH_BLUBLAS=1 SM_ARCH=90`, and the same for
`build-mpi` and `dist`.

**Why (2026-09-26):** I gave `make tensor ...` with no -j for the reduction fix. The user saw it taking
too long and asked, "are you building in one thread?" They were right:
- The Makefile has no MAKEFLAGS or -j.
- The 12 dependent files compiled serially on 44 cores.
- `ReductionImplGPU.cu` alone takes more than 4 minutes. It is one translation unit that instantiates
  every reduction op for every dtype, and -j cannot split a single file.

**How to apply:**
- Put -j on every make I hand the user.
- When the slow part is a single heavy .cu, say so up front, with the expected time.
- Don't restart a running serial build just to add -j when only that one file is left; it would not
  finish sooner.
