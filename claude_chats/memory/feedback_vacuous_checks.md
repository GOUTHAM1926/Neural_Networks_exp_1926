---
name: feedback-vacuous-checks
description: A check whose passing state is reachable without the thing under test being correct proves nothing; run the arm that would look identical on failure.
metadata:
  type: feedback
---

FOUR checks in two days passed while testing nothing:

1. Clock lock verified `clocks.applications.graphics` — 1980 is this H200's default, so the
   check passed whether or not the lock took. Correct probe: sample `clocks.sm` WHILE A KERNEL
   IS RUNNING. (The lock also has to be BELOW the sustainable ceiling — 1785 sagged to 1710 on
   a 700 W-capped card; 1600 held 69/69.)
2. "rank-4 is bit-identical, 5/5" was validated on NT, where the knob is a no-op — the SASS for
   MN4D=0 and MN4D=1 hashes identically. The feature was 50% wrong on NN. See [[project-mn4d-multicast-incompatible]].
3. A sweep logged the shape NAME into the checksum column, so "checksums identical" was
   comparing a constant to itself.
4. A byte-identity re-check reported "BYTE-IDENTICAL 16/16" while the comparison arm had FAILED
   to compile (32 errors) -- it compared against a STALE .o from an earlier successful build.
   Fix: rm -f the artefacts first, and withhold the verdict on any nonzero error count.

**Why:** the common shape is a check that can pass for reasons unrelated to correctness. Each
one hid a real defect or produced a number later voided.

**How to apply:** before trusting a check, ask "if the thing under test were broken/absent,
would this still pass?" If yes, it is not a check. Concretely: run the arm that would look
IDENTICAL on failure — for a Hilbert knob that is an explicit plain-column-major arm, because
a curve that fails to engage falls back to exactly that; if the two differ, the feature ran.
Also prefer a probe that LOCALISES (count corrupt elements and print their value) over one that
merely detects (a checksum), since "exactly 50% wrong, each exactly half" named the mechanism
immediately. Related: [[feedback-verify-claims-and-benchmark]], [[feedback-isolated-measurements-mislead]].

**Case (2026-10-10, test compile):** I built the test with `make -n run-cuda | grep -m1 "nvcc.*-o "` and reported "exit 0". The library was stale, so `make -n` listed library recompiles FIRST and the grep picked a library object line -- the test was never compiled (binary still had the old 21:05 timestamp). Fix: select the test line by `-- "-ltensor"`, `rm -f` the output first, and print the binary's timestamp. Rule: a build check must show the artifact it claims (fresh timestamp / symbol), not just an exit code.
