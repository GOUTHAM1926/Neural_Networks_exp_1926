---
name: project-hybrid-workflow
description: The agreed hybrid-BluBLAS workflow — modify and validate in the vendored production tree, diff against a pristine baseline, then PR to the production repo with the COMPANY account.
metadata:
  type: project
---

Agreed 2026-09-02.

**Base tree.** All hybrid work happens in
`blutrain_prod/Tensor-Implementations/BluBridge-BLAS` — the vendored production BLAS
(ea666ab). It is the tree production actually builds, so a PR can land against it, and it can
be A/B'd on the production trainer immediately.

**Pristine baseline.** `hybrid_baseline_ea666ab/` is an immutable copy taken before any change.
Never edit it. The final change set comes out mechanically:

    diff -ruN hybrid_baseline_ea666ab blutrain_prod/Tensor-Implementations/BluBridge-BLAS

Hand-written modification lists drift; a diff cannot.

**Performance floor is KOTA'S numbers, not ours** (corrected 2026-09-02). Because every change
is made ON TOP OF his tree, his current metrics are the thing to hold or beat:

    1.7B   KOTA 36,099   <- the floor      (cuBLAS 36,770 for reference)
    600M   KOTA 80,612   <- the floor      (cuBLAS 81,387 for reference)

Maintain or improve, never decay. Our own 36,190 on 1.7B is NOT the target: it came from a
shape-specific pin table that does not survive into his generic design.

**Our BLAS on 600M is not worth measuring** — we never pinned those shapes (0/15) and all
future optimisation happens on the hybrid, so the number would describe a tree we are
abandoning.

**Why his tree is the base, in his words:** he worked across all three models and his
dispatcher is GENERIC and suits all of them; ours is model- and shape-specific. So the hybrid
is: his version + our contributions — extra features, reworked existing ones, better code
practice, redundancy removal, templatisation of features, future-proofing for later additions.

**Which test to run is the USER'S CALL, per modification** (corrected 2026-09-02). Do not
assume a fixed gate — not every change warrants both tests, and the right one depends on what
changed. Ask, and run what he says. The instruments available: standalone matrix test (one
config per process, bit-exact maxRelErr, nsys kernel time, minutes) and the 30-step trainer
A/B with the clock/power sampler (~1 h for a pair).

**Push at the END, not incrementally.** Clone production BluBLAS fresh with the PAT, apply the
diff, open a PR.

**COMMIT AS THE COMPANY ACCOUNT for that PR:** `goutham.kumar@blubridge.com` — NOT the personal
`goutham64840@gmail.com` that VS Code is signed in with. The user holds push access under the
company account. This is specific to the production-repo PR; our own repo keeps using
[[feedback_git_no_ai_contributor]] rules.

## HIS DISPATCHER LOGIC IS FROZEN (decided 2026-09-02)

Kota's dispatcher stays. Ours is shape- and model-specific — tuned pins for the exact 1.7B/4B
shapes — while his is GENERIC: every path ends in a real kernel, so any (M,N,K) lands somewhere.
Ours returns 0xFFFF on a miss and falls through to a weak fitted tree. Our tier-1 pin table stays
in OUR local repo only; it is not going into production. It may be added later as a layer ON TOP
of his logic, but not now.

**What may change in his dispatcher:** only what our modifications force — entry-point NAMES and
added template parameters. **What may not change: the decision logic itself.**

## THE RULE THAT MAKES "NO PERFORMANCE LOSS" TRUE

Same dispatcher logic means the same entry point is chosen by name — but that alone does NOT
guarantee identical performance, because the kernel BEHIND that name would be our implementation
(different register allocation, schedule, defaults).

It is only safe if we **templatise by addition with behaviour-preserving defaults** — the
discipline our own template already follows: every new knob goes LAST in the parameter list with
a default that reproduces today's behaviour byte-for-byte, so existing configs compile to
identical code and new capability is strictly opt-in.

Verify in three levels, cheapest first:
  1. symbol diff — the same entry points still exist (used on the coopB removal)
  2. SASS / instruction diff — unchanged configs generate identical code. **If the SASS is
     identical the performance cannot differ**, which removes the need for a GPU run.
  3. kernel-name check in nsys + 30-step A/B — same variants selected, same throughput.

## HOW HIS DISPATCHER WORKS (for reference)

    entry: validate M,N,K > 0
           N_padded = (N%8==0) ? N : round up to 64      <- padding costs a copy, avoided when possible
           K_padded = (K%16==0 && N%128==0) ? K : round up to 64
           non-packed ld: 16-byte-aligned stride -> TMA carries it, zero copy;
                          misaligned -> cuTensorMapEncodeTiled CANNOT express it, force padding
    model: waves = ceil(total_tiles/132); utilization = total_tiles/(waves*132)
           pick the WIDEST BN that still reaches full SM occupancy — a wide tile does more work
           per CTA but yields fewer CTAs, and fewer tiles than SMs leaves SMs idle all kernel
           (N=768 with BN=256 -> 96 tiles on 132 SMs; those shapes measured 42-54% of reference)
    gates: BN=192 has priority (measured to beat 256/128/cluster), occupancy bar 3/4 of SM count
           (99) not 132 — interpolated between a confirmed win at 97% and a confirmed loss at 3%.
           WIDE-N EXCLUSION: skip the BN=192 gate when N%256==0 && N>=3072 (costs 3-5% there).
    route (do_nn): narrow-K -> bm192_ok -> bm256_narrow_ok -> bn==192 -> depth4_retest_ok
                   -> switch(bn): 256/160/128/64/32/16/8, every branch a real kernel.

## THE PER-KNOB DECISION RULE (agreed 2026-09-03)

For every feature where his tree and ours differ, decide in this order:

**1. Compare by CODE MECHANISM, never by name.** Read what the code actually does -- which branches,
which instructions, what accumulation order, what memory it touches -- then look for THAT mechanism
in the other tree however it is spelled. "coopB" is the standing example: the name implied a
cooperative feature, the code was plain B-multicast on an M-axis cluster, which we already had as
MULTICAST_B. A name-and-grep comparison produced a wrong answer once and must not be trusted again.

**2. Check whether the feature is ACTUALLY EXERCISED**, using the nsys kernel census from real
trainer runs -- not the dispatcher source. A path can exist in the switch and never be selected:
the cluster kernels are reachable in his dispatcher yet 600M and 1.7B both ran 100%
Sm90WarpspecConfig with zero Sm90ClusterNN*. If a feature never fires for the real model shapes,
the choice barely matters and it should not consume effort or risk.

**3. If it IS exercised, and ours is better on precision and performance, take ours.** Better must
be demonstrated, not asserted -- bit-exactness against cuBLAS for precision, and a measurement for
performance. "Unmeasured" is a valid finding and must be reported as such rather than guessed.

**4. Whatever is taken, verify afterwards that the same variant is still being selected**, by
reading the kernel name out of an nsys profile. The dispatcher chooses entry points by NAME, so a
port can keep every name and still change the kernel behind it.
