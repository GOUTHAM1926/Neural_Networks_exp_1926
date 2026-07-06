// test_clc_dynamic_probe.cu
// Device probe: CLC try_cancel semantics under a PREFERRED(4)/FALLBACK(2)
// cluster launch on B300 (sm_103a). Answers the M2 design question (§31.2):
//
//   When a cluster steals a pending cluster via clusterlaunchcontrol.
//   try_cancel, what is the CANCELLED UNIT's size/alignment?
//     (B) always a full PREFERRED unit  → first_ctaid % 4 == 0 always
//     (D) sized like the REQUESTER      → fallback stealers see % 4 == 2 too
//
//   Also measures: how many clusters get granted PREFERRED (4) vs FALLBACK
//   (2) on this GPU — the occupancy story (cuBLAS f-kernel: 74×2 vs 33×4).
//
// Usage:  ./build/test_clc_dynamic_probe
// Safe to run on a busy GPU (tiny kernel, ~2s).

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t e = (call);                                                    \
    if (e != cudaSuccess) {                                                    \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e),       \
              __FILE__, __LINE__);                                             \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// Record: x = type (0=cluster-launch record, 1=steal record)
//         y = granted cluster size (%cluster_nctaid.x of the recorder)
//         z = type0: first blockIdx of this cluster | type1: cancelled ctaid
//         w = type0: 0                              | type1: cancelled ctaid %
//         PREFERRED
struct Rec {
  int type, granted, ctaid, misalign;
};

__device__ __forceinline__ unsigned p_cluster_nctas() {
  unsigned n;
  asm volatile("mov.u32 %0, %%cluster_nctaid.x;" : "=r"(n));
  return n;
}
__device__ __forceinline__ unsigned p_cluster_rank() {
  unsigned r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ void p_mbar_init(unsigned addr, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(addr),
               "r"(count));
}
__device__ __forceinline__ void p_mbar_arrive_expect(unsigned addr, int tx) {
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, "
               "[%0], %1;" ::"r"(addr),
               "r"(tx)
               : "memory");
}
__device__ __forceinline__ void p_mbar_wait(unsigned addr, int phase) {
  unsigned ticks = 0x989680;
  asm volatile("{\n\t"
               ".reg .pred P;\n\t"
               "LAB_WAIT:\n\t"
               "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P, [%0], "
               "%1, %2;\n\t"
               "@P bra.uni DONE;\n\t"
               "bra.uni LAB_WAIT;\n\t"
               "DONE:\n\t"
               "}" ::"r"(addr),
               "r"(phase), "r"(ticks));
}
// NON-multicast try_cancel: response + completion land on THIS CTA only, so
// peer CTAs need no barrier participation (multicast::all would require every
// cluster CTA to arm/consume).
__device__ __forceinline__ void p_clc_try_cancel(unsigned resp, unsigned mbar) {
  asm volatile("clusterlaunchcontrol.try_cancel.async.shared::cta"
               ".mbarrier::complete_tx::bytes.b128 [%0], [%1];" ::"r"(resp),
               "r"(mbar)
               : "memory");
}
__device__ __forceinline__ void p_clc_query(unsigned resp, unsigned &valid,
                                            unsigned &ctaid) {
  asm volatile("{\n\t"
               ".reg .b128 handle;\n\t"
               ".reg .pred p;\n\t"
               "mov.s32 %0, 0;\n\t"
               "ld.shared.b128 handle, [%2];\n\t"
               "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p, "
               "handle;\n\t"
               "@p mov.s32 %0, 1;\n\t"
               "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 "
               "%1, handle;\n\t"
               "}"
               : "=r"(valid), "=r"(ctaid)
               : "r"(resp)
               : "memory");
}
__device__ __forceinline__ void p_cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n"
               "barrier.cluster.wait.acquire.aligned;\n" ::
                   : "memory");
}

__global__ void __launch_bounds__(32)
    clc_probe_kernel(Rec *out, int *out_n, int max_out, long long spin_cycles,
                     int preferred) {
  __shared__ __align__(16) unsigned char resp[16];
  __shared__ __align__(8) unsigned long long mbar[1];
  const unsigned granted = p_cluster_nctas();
  const unsigned rank = p_cluster_rank();
  const unsigned resp_a = (unsigned)__cvta_generic_to_shared(resp);
  const unsigned mbar_a = (unsigned)__cvta_generic_to_shared(mbar);

  if (rank == 0 && threadIdx.x == 0) {
    p_mbar_init(mbar_a, 1);
    // one record per CLUSTER: granted size + where it sits in the grid
    int i = atomicAdd(out_n, 1);
    if (i < max_out)
      out[i] = {0, (int)granted, (int)blockIdx.x,
                (int)(blockIdx.x % preferred)};
  }
  p_cluster_sync();

  // burn time so later clusters stay PENDING and steals actually happen
  long long t0 = clock64();
  while (clock64() - t0 < spin_cycles)
    ;

  if (rank == 0 && threadIdx.x == 0) {
    int phase = 0;
    while (true) {
      p_mbar_arrive_expect(mbar_a, 16);
      p_clc_try_cancel(resp_a, mbar_a);
      p_mbar_wait(mbar_a, phase);
      phase ^= 1;
      unsigned valid, ctaid;
      p_clc_query(resp_a, valid, ctaid);
      if (!valid)
        break;
      int i = atomicAdd(out_n, 1);
      if (i < max_out)
        out[i] = {1, (int)granted, (int)ctaid, (int)(ctaid % preferred)};
    }
  }
  p_cluster_sync();
}

int main() {
  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
  printf("Device: %s (SM %d.%d, %d SMs)\n", prop.name, prop.major, prop.minor,
         prop.multiProcessorCount);
  if (prop.major != 10 || prop.minor != 3) {
    printf("[SKIP] needs sm_103\n");
    return 0;
  }

  const int PREFERRED = 4, FALLBACK = 2;
  const int UNITS = 2048; // pending pool ≫ resident capacity
  const int GRID = UNITS * PREFERRED;
  const int MAX_OUT = GRID * 2;

  Rec *d_out;
  int *d_n;
  CHECK_CUDA(cudaMalloc(&d_out, MAX_OUT * sizeof(Rec)));
  CHECK_CUDA(cudaMalloc(&d_n, sizeof(int)));
  CHECK_CUDA(cudaMemset(d_n, 0, sizeof(int)));

  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(GRID);
  cfg.blockDim = dim3(32);
  cudaLaunchAttribute attrs[2] = {};
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = FALLBACK;
  attrs[0].val.clusterDim.y = 1;
  attrs[0].val.clusterDim.z = 1;
  attrs[1].id = cudaLaunchAttributePreferredClusterDimension;
  attrs[1].val.preferredClusterDim.x = PREFERRED;
  attrs[1].val.preferredClusterDim.y = 1;
  attrs[1].val.preferredClusterDim.z = 1;
  cfg.attrs = attrs;
  cfg.numAttrs = 2;

  long long spin = 2000000; // ~1ms at ~2GHz
  CHECK_CUDA(cudaLaunchKernelEx(&cfg, clc_probe_kernel, d_out, d_n, MAX_OUT,
                                spin, PREFERRED));
  CHECK_CUDA(cudaDeviceSynchronize());

  int n = 0;
  CHECK_CUDA(cudaMemcpy(&n, d_n, sizeof(int), cudaMemcpyDeviceToHost));
  if (n > MAX_OUT)
    n = MAX_OUT;
  Rec *h = (Rec *)malloc(n * sizeof(Rec));
  CHECK_CUDA(cudaMemcpy(h, d_out, n * sizeof(Rec), cudaMemcpyDeviceToHost));

  // ── analysis ──
  long launched[2] = {0, 0};   // granted 2 / granted 4 cluster counts
  long launch_mis[2] = {0, 0}; // fallback clusters at offset 0 vs 2
  long steals[2][2] = {{0, 0},
                       {0, 0}}; // [requester 2/4][cancelled %4: 0 / !=0]
  long steal_mis_vals[4] = {0, 0, 0, 0};
  for (int i = 0; i < n; ++i) {
    int g = (h[i].granted == 2) ? 0 : 1;
    if (h[i].type == 0) {
      launched[g]++;
      if (g == 0)
        launch_mis[h[i].misalign ? 1 : 0]++;
    } else {
      steals[g][h[i].misalign ? 1 : 0]++;
      steal_mis_vals[h[i].misalign & 3]++;
    }
  }
  printf("\n=== LAUNCH GRANTS ===\n");
  printf("clusters granted FALLBACK(2): %ld  (at preferred-offset 0: %ld, "
         "offset 2: %ld)\n",
         launched[0], launch_mis[0], launch_mis[1]);
  printf("clusters granted PREFERRED(4): %ld\n", launched[1]);
  printf("resident CTA estimate: %ld (of %d SMs)\n",
         launched[0] * 2 + launched[1] * 4, prop.multiProcessorCount);
  printf("\n=== CLC STEALS (cancelled first_ctaid alignment vs preferred=4) "
         "===\n");
  printf("requester FALLBACK(2): aligned %%4==0: %ld   misaligned: %ld\n",
         steals[0][0], steals[0][1]);
  printf("requester PREFERRED(4): aligned %%4==0: %ld   misaligned: %ld\n",
         steals[1][0], steals[1][1]);
  printf("misalignment histogram (ctaid%%4): [0]=%ld [1]=%ld [2]=%ld [3]=%ld\n",
         steal_mis_vals[0], steal_mis_vals[1], steal_mis_vals[2],
         steal_mis_vals[3]);
  long total_steals = steals[0][0] + steals[0][1] + steals[1][0] + steals[1][1];
  long total_clusters = launched[0] + launched[1];
  printf("\ntotal clusters launched: %ld   total steals: %ld   (units: %d)\n",
         total_clusters, total_steals, UNITS);
  printf("coverage check: launched + stolen units = %ld (expect %d if every "
         "unit is preferred-sized)\n",
         (launched[0] + 1) / 2 + launched[1] + total_steals, UNITS);
  printf("\nVERDICT: %s\n",
         (steals[0][1] + steals[1][1]) == 0
             ? "ALL cancelled units PREFERRED-aligned -> M2 strategy: any "
               "stealer covers the FULL 2-N-block unit (sequential on "
               "fallback stealers)"
             : "MISALIGNED cancellations exist -> cancelled units are "
               "requester-sized pieces; M2 must size stolen work by "
               "(ctaid%%4, granted)");

  free(h);
  cudaFree(d_out);
  cudaFree(d_n);
  return 0;
}
