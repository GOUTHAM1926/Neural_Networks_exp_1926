---
name: project-where-the-work-lives
description: Map of which tree holds the REAL hybrid BluBLAS work vs older/other copies - three BF16 SM90 BLAS copies exist and they are NOT the same.
metadata:
  type: project
---

There are **three** copies of the BF16 SM90 BLAS in `~/gautam_k/blublas`. They are different
trees, not links. Measured 2026-09-12:

| path | template lines | variant entry points | our knobs |
|---|---|---|---|
| **`src/`** (standalone) | **10,970** | **161** | **all** - DESC_HOIST, stsm_cbuf_ch, EPI_A1B0, 192192_CM |
| `BluTrain/Tensor-Implementations/BluBridge-BLAS/src/` | 8,263 | 121 | none of the latest |
| `blutrain_prod/Tensor-Implementations/BluBridge-BLAS/src/` | 2,356 | 117 | none - this is KOTA's, reference only |

* **THE LATEST WORK IS `blublas/src/`.** That is the tuned BLAS. It is committed and pushed.
* **THE TRAINER IS `BluTrain/`** - this is OUR tree, not Kota's. Its Makefile carries
  `-DBLUBLAS_HAS_SM90` (line 179) and the header-dependency fix; without the first, the entire
  SM90 module is silently preprocessed away and the trainer runs the Ampere path on a Hopper card.
* **`blutrain_prod/` is Kota's, for reference only.** The user does not want runs in it.
* `BluTrain/`'s bundled BLAS is OUR work but an OLDER snapshot (121 vs 161 variants, 33 files
  differ). To run the trainer against the current BLAS, `src/` must be synced into it first -
  ASK before doing that, it overwrites files.

**`blutrain_ours/` is GONE** - it lived only on the old box (86.38.238.116, cut 2026-09-11) and
was never committed. Only 2 Makefiles survived, in `h211_2026_09_10/box_src.tgz`. It is no loss:
those Makefiles LACK both `-DBLUBLAS_HAS_SM90` and the header-dep fix that `BluTrain/` has, so
the repo copy is strictly newer.

**Build the trainer with `WITH_BLUBLAS=1`** - the Makefile defaults to `WITH_BLUBLAS ?= 0`
which is cuBLAS, so a plain `make all` measures cuBLAS while looking like it built ours. There is
one `lib/libtensor.so`, so the two arms cannot both be built then both run - see
[[feedback-backend-ab-snapshot]]. Box has **88 cores** (not 126): use `-j88`.
