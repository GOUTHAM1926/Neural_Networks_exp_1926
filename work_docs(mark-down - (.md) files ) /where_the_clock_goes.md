# Where the clock goes — why our SM90 GEMM loses to cuBLAS at free clocks

**H200 · loud-tree-grows-fin-03 · GPU 1 · 2026-08-26**
Log sections §H27ac–§H27ah in `cublas_rev_eng_binary/analyzations_h200.md`

At a **locked** clock our GEMM beats cuBLAS. At **free** clocks it loses — and the entire loss is
clock, not kernel. We draw the **same power**, sit at the **same cap**, and are handed **fewer
cycles per second**. This is the measurement chain that found why.

---

## 1. The gap is the clock

Per-kernel clock, both libraries measured **inside one process** so they are sampled at the same
instant on the same card.

| | value |
|---|---|
| Locked 1400, mean over the 13 shapes where the lock held | **100.42%** of cuBLAS |
| Free clocks, mean over 15 shapes | **97.15%** of cuBLAS |
| Free-clock deficit, measured per kernel | **−3.42%** |
| Check: `100.42 × (1 − 0.0342)` | **97.0** vs measured **97.15** |

The two numbers close to fifteen hundredths of a point. **There is nothing else in the gap** — no
kernel deficit, no memory effect, no scheduling loss. Whatever explains the lost clock explains
the whole thing.

---

## 2. What it is NOT — each eliminated by direct measurement

| candidate | verdict | evidence |
|---|---|---|
| Thermal | **ruled out** | `hw_thermal_slowdown` never fires; 52 °C on a card that throttles in the 80s |
| Current limit | **ruled out** | `hw_power_brake_slowdown` never fires on any shape |
| We draw more power | **ruled out** | measured **1.00×** on all 15 shapes at free clocks, settled, phase-separated |
| Our kernel is slower | **ruled out** | 2.43× the instructions in **1.01× the cycles** at identical tensor utilisation |
| **`sw_power_cap`** | **THE LIMITER** | the only reason that ever engages — and at free clocks it is active **100% of the time for BOTH libraries** |

Both of us are pinned against the same ceiling, drawing the same watts, and one of us is given
less frequency for it. **That single observation forces the explanation in §5.**

---

## 3. Free clocks — the full 15-shape result

Power **1.00×** and cap **100% vs 100%** on *every* shape. Only the clock differs.

| V | shape | ours MHz | cuBLAS MHz | clock loss | instr ratio |
|---|---|---|---|---|---|
| V1 | 2048x512x16384 tn | 1426 | 1530 | **−6.80%** | 2.07× |
| V2 | 2048x2048x16384 tn | 1474 | 1528 | −3.53% | 2.27× |
| V2b | 5632x2048x16384 tn | 1507 | 1543 | −2.33% | 2.20× |
| V2c | 2048x50304x16384 tn | 1386 | 1467 | **−5.52%** | 2.18× |
| V3 | 16384x11264x2048 nn | 1492 | 1496 | −0.27% | 1.56× |
| V4 | 16384x50304x2048 nn | 1467 | 1476 | −0.61% | 1.59× |
| V5 | 2048x11264x16384 tn | 1461 | 1468 | −0.48% | 1.61× |
| V6 | 16384x5632x2048 nt | 1503 | 1505 | −0.13% | 1.61× |
| V7 | 16384x2048x2048 nn | 1551 | 1568 | −1.08% | 1.80× |
| V7b | 16384x2048x5632 nt | 1490 | 1503 | −0.86% | 1.81× |
| V7c | 16384x512x2048 nn | 1486 | 1528 | −2.75% | 1.83× |
| V8 | 16384x2048x512 nt | 1638 | 1650 | −0.73% | 1.82× |
| V8b | 16384x2048x2048 nt | 1554 | 1572 | −1.15% | 1.79× |
| V8c | 16384x2048x11264 nt | 1477 | 1481 | −0.27% | 1.76× |
| V8d | 16384x2048x50304 nt | 1485 | 1476 | +0.61% | 1.81× |

**Pearson r = −0.742 · slope −6.82% of clock per 1.0× of instruction ratio · mean loss −1.73%**

Grouped, the monotonicity is clean:

```
instr 1.56-1.61x   (V3,V4,V5,V6)                    mean loss -0.37%
instr 1.76-1.83x   (V7,V7b,V7c,V8,V8b,V8c,V8d)      mean loss -0.75%
instr 2.07-2.27x   (V1,V2,V2b,V2c)                  mean loss -4.55%
```

---

## 4. The two measurement bugs that had to die first

**A 400 ms window never lets the power sensor settle.** Version 1 of the probe reported ours at
**2.3–2.6× cuBLAS's power** and 73.2% cap-active on V2c against cuBLAS's 0.0%. All of it was an
artefact — the per-instant data showed every window still climbing at its end:

```
V4  cublas   103 -> 140 -> 201 -> 263 W   across its 400 ms window
V4  ours     368 -> 423 -> 486 -> 555 W   across its 400 ms window
```

Each rep runs `idle → cuBLAS → ours`, so power climbs monotonically and **whoever goes second looks
worse**. The aggregate table was entirely persuasive and entirely wrong; only the per-instant
series exposed it. Fixed with 10 s windows and the first 4 s discarded.

**`MAX_IT = 2000` silently truncated the window.** Asking for 10 s delivered 500 ms on fast shapes
(2000 iters × ~0.25 ms). Harmless for TFLOPS, fatal for power.

*(And `strings "$BIN" | grep -q X` under `set -o pipefail` rejected the binary **because** the match
was found — SIGPIPE promoted to failure. Third occurrence of that trap.)*

---

## 5. The mechanism — why more instructions cost clock

Dynamic switching power on any CMOS chip:

```
P  =  α · C · V²· f        (+ leakage)
```

| term | what it is | fixed or variable |
|---|---|---|
| **C** | **Capacitance.** Every wire and gate holds a tiny charge, like a small bucket. Flipping a bit fills or empties one. | **Fixed** — etched into the silicon, identical for both libraries |
| **V** | **Voltage.** Energy per flip goes as V². | The card picks V and f together (DVFS) |
| **f** | **Frequency.** Cycles per second — the thing we are losing. | the outcome |
| **α** | **Activity factor.** Of all those buckets, the fraction that actually flip each cycle. | **This is the term the workload moves** |

Every instruction issued makes hardware switch — decode logic, register-file reads, operand
collectors, the ALU. More instructions per cycle means a larger share of the *fixed* capacitance
charging and discharging each cycle. **C stays the same; α goes up.**

Rearranged:

```
f  =  P / (α · C · V²)
```

Hold **P** at the cap, **C** at the silicon, **V** at the same operating point, and **f falls as α
rises.** We are not allowed to draw more power — we are already at the wall — so the chip charges
us in clock instead.

### What is assumption, not measurement

- **α was never measured.** No instrument exposes it. The instruction ratio is a **proxy**.
- **V may also differ**, since DVFS selects (V, f) pairs together and V is squared.
- **r = −0.742, not −1.0** — real scatter remains unexplained.

---

## 6. Which instructions — dynamic per-opcode counts

Real execution counts from ncu's SASS source page, **not** a static census. Our mainloop *loops*
where cuBLAS *unrolls*, so static ratios would have been meaningless.

| shape | ours | cuBLAS | ratio |
|---|---|---|---|
| V1 2048x512x16384 tn | 12,590,432 | 5,190,716 | 2.43× |
| V2c 2048x50304x16384 tn | 923,599,623 | 397,157,914 | 2.33× |
| V7 16384x2048x2048 nn | 33,969,919 | 18,271,094 | 1.86× |
| V8d 16384x2048x50304 nt | 677,560,476 | 382,502,330 | 1.77× |

### The diagnostic pair

| opcode | shape | BluBLAS | cuBLAS |
|---|---|---|---|
| R2UR | V1 | 1,050,112 | 4,096 |
| R2UR | V2c | 103,123,200 | 106,322 |
| R2UR | V7 | 2,113,536 | 26,970 |
| R2UR | V8d | 51,527,680 | 26,970 |
| ULEA | V1 | 1,025,792 | **0** |
| ULEA | V2c | 43,349,784 | **0** |
| ULEA | V7 | 1,788,360 | **0** |
| ULEA | V8d | 42,487,240 | **0** |

`R2UR` moves a value from a per-thread register into a **uniform** register. `ULEA` then does
uniform address arithmetic on it.

**cuBLAS issues `ULEA` exactly zero times on every shape tested**, across three tile configurations
and three layouts. The reading: **we compute warp-uniform addresses per thread and then promote the
result to uniform. cuBLAS's addresses are uniform from the start**, so it needs neither instruction.
A warp-uniform address computed per-thread is the same value calculated thirty-two times.

Corroborating, on V1: cuBLAS issues **6,272** plain `IMAD` where we issue **1,517,056** — a factor
of **242**.

### Top contributors per shape

**V1**, gap +7,399,716:
```
IMAD      1,517,056        6,272   +1,510,784   20.4%
R2UR      1,050,112        4,096   +1,046,016   14.1%
ULEA      1,025,792            0   +1,025,792   13.9%
BRA       1,152,980      388,591     +764,389   10.3%
UMOV        875,648      185,472     +690,176    9.3%
```

**V2c**, gap +526,441,709:
```
LOP3    115,959,168    7,527,742  +108,431,426   20.6%
R2UR    103,123,200      106,322  +103,016,878   19.6%
ULOP3    61,173,360       30,480   +61,142,880   11.6%
SHF      61,241,520      153,088   +61,088,432   11.6%
```

**V8d**, gap +295,058,146:
```
ULOP3    61,173,360       11,328   +61,162,032   20.7%
LOP3     64,668,960    7,342,454   +57,326,506   19.4%
R2UR     51,527,680       26,970   +51,500,710   17.5%
ULEA     42,487,240            0   +42,487,240   14.4%
```

Also recorded: `BRA` is 8.3–10.3% on every shape (loop vs unroll). `BSSY`/`BSYNC` at 235,008 and
284,544 on V1 against cuBLAS's **zero** — our code has branch divergence theirs does not. And
`UIADD3` runs the *other* way (ours 1,121,920 vs cuBLAS 1,887,808) — not a defect, that is cuBLAS
doing on the cheap uniform path what we do on the expensive per-thread one.

---

## 7. The prediction, and how to kill it

The slope is quantitative, so the next change is a **test**, not a hope.

| shape | ratio now | target | predicted clock gain |
|---|---|---|---|
| V1 | 2.07× | 1.50× | **+3.9%** |
| V2 | 2.27× | 1.50× | **+5.3%** |
| V2c | 2.18× | 1.50× | **+4.6%** |

**Change the address path, re-measure the clock, compare against the prediction.** If the clock
does not move as predicted, the correlation was spurious and this whole chain is bookkeeping. That
is the same standard that killed PDL, the backward secondary stream, and the split-K reduce scare —
each looked real until the measurement said otherwise.

**Still unknown:** which source-level construct emits the `R2UR`/`ULEA` pattern. The histogram
names the instructions; it does not name the line of CUDA that generates them. That mapping is the
next measurement and needs ncu's SASS-to-source correlation.

---

## Provenance

```
h27af locked 1400    15 shapes, 10 s windows, 3 reps,  94,049 per-instant samples @ 99 Hz
h27af free clocks    15 shapes, 10 s windows, 3 reps, 103,027 per-instant samples @ 99 Hz
h27ah opcode hist    4 shapes x 2 arms, ncu --section SourceCounters, dynamic per-line counts
h27zw clock matrix   15 shapes x {free,1400}, nsys --gpu-metrics-devices @ 10 kHz
```

Raw data: `profiling_reports/h27af_free/`, `h27af_locked1400/`, `h27ah/`
Tools: `tools/h27af_power_attrib.sh`, `tools/h27ah_opcode_hist.sh`, `tools/h27zw_clock_matrix.sh`
