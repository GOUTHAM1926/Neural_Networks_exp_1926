---
name: project-kota-run-gap-resolved
description: "Kota's 4B cuBLAS 35,478 vs our 36,354 was OUR benchmark flag, not his setup -- his tree/binary/flags/data are identical and reproduce our number exactly"
metadata: 
  node_type: memory
  type: project
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-09-09T15:06:07.497Z
---

RESOLVED 2026-09-09. Kota's 4B run reported ~35,470 tok/s on cuBLAS while ours
reported ~36,250. Nothing was wrong with his setup.

**Ruled out by measurement, in this order:**
- heat soak -- his 40-step curve is FLAT from step 1 (35,562 -> 35,474)
- his tree/build -- Makefile byte-identical, libtensor.so identical size with 0
  blublas syms, libprofiler.so md5-identical, same libcublas.so.13
- his script -- differs in 3 lines (include path, a comment, data_root)
- his launch -- `mpirun -np 2 -x LD_LIBRARY_PATH ./_runner_`, NP=2, mem 101.787 GB
  identical to ours; ranks bound to core 0 / core 1, same as ours
- his ZeRO flags -- he exports `BLU_ZERO_V2=1 BLU_SYMMETRIC=1` (v2
  SymmetricShardedOptimizer + NCCL symmetric/NVLS; both default 0 and we never set
  them). Ablation with his exact flags: **36,317 vs 36,354 = 0.1%, no effect.**
- power cap / thermal -- both pin the 700 W factory default (power.limit ==
  default_limit == enforced), swPowerCap Active 100% of samples in BOTH,
  hwThermal/hwBrake/swThermal never active, same temps

**The cause: our own `BLU_MAX_STEPS=15/20`.** See
[[feedback_never_shorten_max_steps_to_benchmark]] for the mechanism and the
locked-clock proof. Crossover (order B,A,A,B): 36,354 / 35,486.7 / 35,488.6 /
36,366.5. Our tree at the 2000 default = 35,487, matching his 35,478 to 0.02%.

**Still open:** every BluBLAS-vs-cuBLAS number measured today used
BLU_MAX_STEPS=15/20 and therefore the wrong numerical regime -- re-baseline at the
production schedule before trusting any of them. That likely also explains the
unresolved our-BluBLAS anomaly (35,596 at 15 steps vs 34,283 at 20 steps on an
identical binary).

Artefacts pulled to `kota_backtrack_2026_09_09/` (his CSV + our 4 arm CSVs +
h180/h182/h183/h184 scripts). Box scripts under /root/gautam_k/blublas/.
Relates to [[project_trainer_parity_1p7b]] and [[project_kota_blas_not_at_cublas_parity]].
