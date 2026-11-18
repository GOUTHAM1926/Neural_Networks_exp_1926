---
name: feedback-mpirun-core0-pinning
description: "mpirun -np 1/2 (Open MPI 4.1.6 default bind-to-core) pins rank 0 to core 0 (+HT 112 on P032); parallel single-GPU runs, other users' jobs and monitoring daemons all share it -> pin each run to its own cores with taskset + OMPI_MCA_hwloc_base_binding_policy=none"
metadata:
  node_type: memory
  type: feedback
  originSessionId: d36b6b0d-fbee-47c9-8b10-a9773d29d8dc
  modified: 2026-11-03T10:58:16.295Z
---

Found 2026-11-03 16:24 on P032 (trainer lag test): both our trainer runs (GPU 0 and GPU 1, each `run_sft.sh` ->
`mpirun -np 1`) had Cpus_allowed_list 0,112 = ONE physical core. Also on core 0: Surendra's single-rank mpirun job and
5 monitoring daemons (dcgm-exporter, otelcol, rdma_exporter, neysamon, journalctl). Open MPI binds to core by default
for <= 2 ranks, rank 0 -> core 0, per mpirun, so every parallel single-GPU job lands on the same core.

**Why:** under nsys (extra CPU per CUDA call) the step becomes CPU-bound and GPU idle then depends on what the other
processes on core 0 do (D2f_sw0 showed +10 ms/micro-batch idle vs A_sw0 with identical call counts and kernel time).
Without nsys the 135M step is mostly GPU-bound (~145 ms = 2 x 73 ms kernels), so it shows as noise (+-10 ms).

**How to apply:**
- For any P032 measurement via mpirun with 1-2 ranks: `export OMPI_MCA_hwloc_base_binding_policy=none` and start the
  binary under `taskset -c <own free cores>` (GPUs 0-3 = NUMA 0 = even cores, HT sibling = core + 112; GPUs 4-7 =
  NUMA 1 = odd cores). Verified on the CPU: `mpirun -np 1 taskset -c 40,42 ...` -> 40,42; plain mpirun -> 0,112.
- Log Cpus_allowed_list of the running trainer in every timing/nsys run.
- Call counts and kernel times are unaffected; GPU-idle comparisons between runs are not valid without this.
Related: [[feedback_isolated_measurements_mislead]], [[project_test2_measurement_traps]], [[feedback_profiler_measurement_traps]].

(Moved from the MEMORY.md index line on 2026-11-05:) fix = taskset + mpirun binding none (--bind-to none).
