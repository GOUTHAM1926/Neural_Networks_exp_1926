---
name: project_laptop_glibc243_cuda13_break
description: RESOLVED 10-21 (libc6-dev back to 2.39, plain nvcc works); 10-19..10-20 every laptop CUDA 13.0 compile failed on glibc 2.43 rsqrt; workaround if it returns = shadow bits/mathcalls.h
metadata:
  type: project
---

The laptop got a big package upgrade on 2026-10-19 14:56-15:00 IST (glibc 2.39 -> 2.43, libc6-dev 2.43-2ubuntu2.4,
linux-libc-dev 7.0.0-38). glibc 2.43 declares the C23 functions `rsqrt` / `rsqrtf` in bits/mathcalls.h (line 206,
`__MATHCALL_VEC (rsqrt,, ...)`); CUDA 13.0's crt/math_functions.h declares them too with another exception spec, so EVERY
local nvcc compile that pulls in <math.h> fails: "exception specification is incompatible with that of previous function
rsqrt".

**Workaround used for local SASS checks:** a copy of /usr/include/x86_64-linux-gnu/bits/mathcalls.h with that one line
removed, in `<scratch>/sysfix/bits/`, and `-I<scratch>/sysfix` on the nvcc line. Proven harmless: LHOd built with it has
the same SASS as before the upgrade (0 lines differ). A `#define rsqrt ...` rename does NOT work (glibc token-pastes the name).

**Why:** local SASS checks are a standard step before every box build ([[feedback_smoke_gate_before_every_campaign]]).
**How to apply:** if a local compile fails with the rsqrt message, recreate the shadow header (scratch is per session);
never "fix" the system headers; tell the user if they build CUDA on the laptop themselves. The box (CUDA 13.4) is not affected.

**2026-10-21:** `dpkg -l libc6-dev` = 2.39-0ubuntu8.9 again (the user's machine was restored); 22 TUs compiled with plain
nvcc, no sysfix. Keep this note for the day glibc 2.43 comes back.
