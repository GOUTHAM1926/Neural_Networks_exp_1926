---
name: project-noswizzle-descriptor-fix
description: Our make_gmma_desc_mn_major never swapped stride/leading in the no-swizzle tier -- the real cause of the old 128x168 wrong answers; fixed, MN floor now 8 elements
metadata:
  type: project
---

`sw_pick` has always been able to return 16 (the no-swizzle / INTERLEAVE tier) for any MN extent
that is not a multiple of 16, but our `make_gmma_desc_mn_major` built that case like a swizzled
one. **In the no-swizzle encoding the two descriptor fields are TRANSPOSED** -- the reference's
own `kSwizzleMode==16` special case. Our raw values already matched his; only the swap was missing.

That is exactly why our 128x168 tile returned MaxRelErr ~1e4 on 12/12 nn+tn cases while all 6 nt
cases PASSED: in NT that operand is k-major and never reaches the tier. We had reacted by adding
a `static_assert` that REFUSES the case (BM/BN must be % 16). Kota's tree carries the correct
encoding and hardware-confirmed it 2026-07-27 (test_bgemm_sm90_narrow_n_tiers).

Fixed 2026-09-04. The MN-extent floor drops from 16 elements to 8 (the no-swizzle atom, 16 bytes),
which is what makes BN=8 legal in nn/tn -- entries 25/43 of the 44 now PASS -- and retires the
H27y.30 "pad 168 to 176 and mask the store" workaround. No-op for everything shipped before, since
the old assert guaranteed no MN-major extent ever reached the 16 tier.

Lesson worth keeping: we hit a silent wrong answer and responded by forbidding the input; the
right move was to read his implementation of the same tier. See [[feedback-diff-against-proven-code-first]]
and [[project-44way-baseline]]. Detail: analyzations_h200.md §H92.
