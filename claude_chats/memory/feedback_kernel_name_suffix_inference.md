---
name: feedback-kernel-name-suffix-inference
description: Never infer which layout a cuBLAS kernel serves from its name suffix or from FLOPs; pin it by running that layout standalone and reading the profile.
metadata:
  type: feedback
---

I read cuBLAS's nvjet kernel names out of a trainer trace and inferred TNT=tn(wgrad),
NTN=nt(dgrad) from instance counts and per-kernel FLOPs. It is exactly INVERTED: ask cuBLAS
for `nt` and it runs a TNT kernel; ask for `tn` and it runs NTN.

**Why:** every shape in a fwd/dgrad/wgrad triple is the same three dimensions multiplied, so
all three have IDENTICAL FLOPs. FLOPs can never separate them — the derivation was unsound,
not unlucky. Instance counts tie equally across the three. And the suffix is cuBLAS's internal
naming, which need not match the caller's transa/transb (it may reflect an internal A/B swap).

**How to apply:** to learn which variant cuBLAS uses for a (shape, layout), RUN that shape and
layout standalone and read the kernel out of the profile. One 5-minute run beats any amount of
name reading. Cost of skipping it: 10 of 15 pins in the 4B apples-to-apples test were the
variant for the *other* pass, and the whole run had to be repeated. Extends
[[feedback_measure_against_dispatch_not_pins]] and the "never propose a mechanism from a name"
rule in [[project_l2_read_excess_coop]].
