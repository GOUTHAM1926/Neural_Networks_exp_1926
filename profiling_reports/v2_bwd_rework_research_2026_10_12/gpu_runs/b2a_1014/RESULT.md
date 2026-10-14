# B-2a — FA3's wait placement between the dQ and dK GEMMs (2026-10-14)

Code: Vj2 backward `85eb2d0f` (snapshot b2a; line-info copy b2ali, kernel SASS identical 1760/1760). One change: a
`wgmma_wait1()` (dV done, dQ still running) before the dK GEMM is sent, and the wait after dK now waits for dQ only.
FA3 does the same (`mainloop_bwd_sm90_tma_gmma_ws.hpp:941-951`).

## Test 1 — quick CLASSIC, GPU 7 (11:35)
144/144 pass; every error value identical to g123.

## Test 2 — speed, 6 alternating rounds g123 / b2a, same session, GPU 7 (11:42-11:48) — `2_ab_g123_vs_b2a.txt`
- Backward: b2a/g123 paired mean 0.9806 (t = -8.7), faster on 13/13; B8 18/6 3.202 -> 3.166 ms (0.9889).
- Forward (identical code, noise check): 1.0016 (t = 0.6); B8 shapes +0.5..1.4% (likely clock/power after a faster backward).
- Forward + backward: 0.9861, faster on 13/13.

## Test 3 — ncu --set full, B8 18/6, 11th launch (14:23-14:25, GPU 0) — `3_before_after_fa3.txt`, `3_line_diff.txt`
Before = test 1.4 (g123, GPU 7). FA3 = prof_1013 (GPU 2). Tools: `tools/before_after_ncu.py`, `tools/line_diff.py`.

| | before g123 | after b2a | FA3 |
|---|---|---|---|
| main kernel time | 2526.3 us | **2478.0 us** (-48.3, -1.9%) | 2481.4 us |
| SM clock under ncu | 1.742 GHz | 1.727 GHz (lower, so not a clock gain) | 1.740 GHz |
| tensor pipe busy | 67.0% | 68.9% | 68.3% |
| warps waiting for each other at GEMM starts (warpgroup arrive) | 16.7% = 422 us-eq | **12.2% = 303** | 10.4% = 259 |
| ... at the dK GEMM start | 215.5 | **148.2** | 141.1 |
| ... at the dV GEMM start | 96.8 | 54.7 | 39.5 |
| ... at the dP GEMM start | 96.8 | 87.2 | 76.8 |
| ... at the S GEMM start | 13.4 | 12.1 | 1.1 |
| the new wait (dV done) before dK | — | 50.1 | 22.6 |
| wait for S (line 519) | 77.5 | 52.4 | — |
| both-groups barrier before dQ | 54.0 | 71.7 | 50.4 |
| short scoreboard (exp line 527, dS lines 550-551) | 78.7 | 142.6 | 137.1 |
| mio throttle (shared-memory queue) | 237.1 | 191.8 | 98.8 |
| dispatch stall | 155.9 | 136.8 | 86.2 |

us-eq = share of the kernel's stall samples x kernel time.

**Verdict: KEEP.** The predicted cause is confirmed: the wait at the dK GEMM start fell 215.5 -> 148.2 (FA3 141.1). The
new wait costs 50.1, the both-groups barrier +17.7 and short scoreboard +64 (exp and dS lines now wait for their
shared-memory values / exp results); the net is -48.3 us. The main kernel now equals FA3 (2478.0 vs 2481.4 us; different
days and GPUs, so "equal", not "faster").

**Remaining gaps to FA3 in this profile:** mio throttle +93 (B-3), dispatch +51, warpgroup arrive +44 (dP/dV/S starts;
B-2b early dO release), helpers + fills +8 (test 1.3; B-1 memset into prep).
