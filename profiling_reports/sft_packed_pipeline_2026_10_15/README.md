# 2026-10-15: packed SFT pipeline, design A (seqlens) everywhere

Work doc: `sft_work_documentation.md` §45.3-45.10.

- `code_snapshot/`: every changed or new source file of the SFT work since the last commits (Tensor-Implementations
  `d96b991`, BluTrain_sft `909ab75`, both `dist_side_merge`, 2026-09-28), as full copies, plus
  `tensor_implementations_vs_d96b991.diff` and `blutrain_sft_vs_909ab75.diff` (tracked files) and `md5.txt`.
  Nothing of it is committed in those repos yet.
- `tools/`: `p1`-`p9` patch scripts (every edit of the evening, exact-match replacements), `plan_cpu_test.py` (the real
  plan builder on the CPU, 254 layouts), `loader_cpu_test.py` (the real loader code vs the old seg_start file),
  `sass.sh` / `dump.sh` (local sm_90a SASS comparison).
- `box_runs/` (when pulled): the logs of the 10-15 box runs (`sft_runs/`: e2e_v1, e2e_packed, packed64, packed16;
  `a1_1015_tests/`: the morning's test_varlen_attention runs).
