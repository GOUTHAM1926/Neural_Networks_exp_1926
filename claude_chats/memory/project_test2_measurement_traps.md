---
name: project-test2-measurement-traps
description: Benchmark-harness traps found in Test 2 — our leak corrupted cuBLAS's numbers, and multi-run agreement does not prove correctness
metadata:
  type: project
---

THREE measurement traps that invalidated headline results. Two found 2026-08-04, the third
2026-08-24 -- and the third is the same lesson as the second, hit again a second time.

**1. A leaking harness degrades the COMPETITOR, and it looks like a win.** Our split-K workspace
was a `static` inside a *template*, so sk2/4/8/16/32 each held their own grow-only cache. That
starved cuBLAS's `CUBLAS_GEMM_DEFAULT` heuristic of workspace, so it fell back to non-split-K and
lost 3.6x on small-N shapes. Reported as us "beating cuBLAS by 288%". Isolation on an idle GPU:
cuBLAS 185 TF, us 150 — we actually **lose** by 19%.

**2. Run-to-run agreement does NOT prove a number is right.** `nt 4096x256x2048` read 103.7% with
only 1.9% spread across three runs; isolation measured 90.4%. All three runs shared the same
systematic bias, so consistency detected nothing.

**3. Passing MANY configs to one process deflates SOME of them 6-8% and spares others (2026-08-24,
§H27zF/G).** Same config, same shape, same box, clock flat at 1410 (`clkMin==clkMax`):
alone **713.24** TFLOPS, but **648.73** when measured 3rd of 9 in one invocation. Its challenger
was unaffected in every context, so the bias does NOT cancel out of a ranking -- it invented
"+10.28% better than the pin", and that number reproduced to two decimal places across two
independent runs. Bisected: **1 predecessor is safe (714.52), 2 is not (656.42)**; no single
config is responsible; mechanism unidentified (there is no `cudaMalloc` in the SM90 sources, so
the workspace theory is dead). nsys and run ORDER were both tested and are innocent.

**THE FIX IS THE DESIGN: one config per process.** That context reproduced within 0.1% across six
independent arms and agreed across GPUs (713.24 on GPU 3 vs 713.82 on GPU 1). Measured properly
the "+10.28%" was **+0.31%** -- a tie. Also: state the win threshold BEFORE seeing data, and make
the run measure its own floor (re-measure the pin in a fresh process; that gap IS the noise).

**How to apply:** in-sweep `%cuB` is unreliable for small-N shapes — verify any beat-cuBLAS claim on
an idle GPU before stating it. Of 8 wins originally claimed, 4 survived mean-of-3 and only **2**
survived isolation. Our own measurements were never the unstable part (median 0.70% run-to-run), so
this affects the scoreboard, never the dispatcher table ([[project-sm90-dispatcher-done]]).
