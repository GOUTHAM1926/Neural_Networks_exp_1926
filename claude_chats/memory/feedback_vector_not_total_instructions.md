---
name: feedback-vector-not-total-instructions
description: Count instructions by PIPE, not as a total - but the old "uniform is at parity" claim is WRONG on the 192x192 pins, where uniform is the LARGEST contributor at 37-44% of the excess.
metadata:
  type: feedback
---

Count instruction issue **per pipe**, never as a bare total: a vector op drives 32 lanes where a
uniform op drives 1, so the total hides which datapath is actually costing power.
`smsp__inst_executed_pipe_uniform.sum` gives the uniform count directly, and
**VECTOR = total - uniform**.

**CORRECTION (§H50, 2026-08-29).** The earlier note here said "uniform is already at parity
(r=+0.061); the excess is VECTOR at 2.35-3.67x". That is **wrong for the 192x192 pins**:

    pipe      gate_up-fwd            gate_up-wgrad        share of the 62M excess
    uniform   1.436x                 1.500x               37.5%  /  43.6%   <- LARGEST
    alu       2.03x                  2.48x                19.9%  /  20.1%
    cbu       3.46x                  3.49x                 5.3%  /   5.5%
    fma      30.5x                  58.5x                 17.6%  /   4.8%   <- inconsistent
    tensor    1.000x  (IDENTICAL)     1.000x               0%
    VECTOR    1.694x                 1.696x

**Why:** `uniform` + `alu` are the only contributors that are both large AND consistent across
shape and layout - together 57-64% of the excess. `fma` has a huge ratio but swings 17.6% to
4.8% between shapes, matching §H27ao's finding that it is worst on the shape with NO deficit.

**How to apply:** target `uniform` and `alu`. Ignore the `fma` ratio's size. And always report
the tensor-pipe count as the control - it came out identical to the instruction on both shapes,
which is what makes every other difference pure overhead.
