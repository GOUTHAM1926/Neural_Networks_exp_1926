// ============================================================================
// test_varlen_attention.cu -- v2 packing, Step 2: the attention test harness.
//
// WHAT IT TESTS
//   The causal GQA attention BluTrain runs on H200:
//     gqa_flash_attn_forward_qkv  -> kernel v62  (GQA_fwd_sm90.cu)
//     gqa_flash_attn_backward_qkv -> kernel Vj1d (GQA_bwd_vj1d_sm90.cu, default mode)
//   Step 2 runs ONLY today's kernels. It proves the harness itself (references, pass
//   rule, data loads) before any kernel changes. Steps 3-5 add the packed-row variants
//   (B: packed row + chat mask, A: per-chat offsets) to the same groups.
//
// REFERENCES (built from the SAME bf16 inputs the kernel gets)
//   truth : fp64 attention, restricted to each segment (a chat, or a row's pad tail),
//           causal inside it. FlashAttention's tests use fp32 for this
//           (attention_ref(upcast=True), hopper/test_util.py); fp64 is a cleaner truth.
//   floor : plain PyTorch bf16 math. It rounds to bf16 at the same points as
//           FlashAttention's attention_ref(upcast=False, reorder_ops=True):
//             forward : k*scale, scores, softmax output, P@V
//             backward: dP, dV, dS (softmax backward), dQ, dK*scale, the GQA group sum
//           Inside one op, reductions run in index order here; PyTorch reduces in
//           parallel. That changes values at the fp32 rounding level only, far below
//           the bf16 steps. exp/log of the floor go through double and are rounded to
//           float (PyTorch: expf) -- at most 1 float ulp apart, also far below bf16.
//   Both are built per segment from tiled GEMMs in which each output element adds its
//   terms in index order. So a plain CPU loop over the FULL row with an explicit
//   block-diagonal causal mask must reproduce them: group G1 checks exactly that.
//
// PASS RULE (per chat; PyTorch's varlen test also checks per sequence)
//   max|kernel - truth| <= 2 * max|floor - truth| + 2 * max|(x + 0.3) - 0.3 - x|
//   for O, dQ, dK, dV:
//     FA3 hopper/test_flash_attn.py:535-536, 579, 655-660;
//     PyTorch test/test_varlen_attention.py:508-516, 583-595.
//     FA2 uses 3x for the gradients (tests/test_flash_attn.py:1449-1451); we keep 2x.
//   LSE gets the same rule against the fp32 logsumexp of the bf16 scores. This is our
//   addition: FA does not gate LSE directly.
//   G5 gates the gradients with FA2's rule for that test, 5x + 1e-3
//   (tests/test_flash_attn.py:2287-2295), and still prints the 2x result.
//   KNOWN issues (kKnown): explained deviations kept by decision are reported as KNOWN in
//   every run; they match only one exact case/tensor/chat up to the recorded ratio (+1%).
//   Anything else fails as NEW.
//   1-token chats: the exact dQ and dK are 0 (P = 1, dS = 0) and both references give
//   exactly 0, so the 2x rule would allow 0. There dQ/dK are checked against a derived
//   fp32 rounding bound instead and printed on their own lines (one_token_region).
//
// GROUPS (Step 2)
//   G1 reference self-test : GPU references (per segment) vs a CPU full-row masked loop.
//                            Covers 1-token chats, zero-length entries, pad tails, an
//                            all-pad row, GQA and MQA. Runs on any GPU.
//   G2 dense rows          : today's kernels, one chat = the whole row.
//                            T in {256, 1024, 4096} x heads {18/6, 12/4, 8/8, 32/8, 8/1}
//                            x B in {1, 2, 8}. Outputs are filled with NaN before every
//                            call, so an unwritten element shows up.
//   G3 v1 SFT rows         : real chat + pad tail (today's SFT layout, dO = 0 on pads).
//                            (a) whole rows vs the dense reference;
//                            (b) chat rows vs the per-chat reference -- proves v1's
//                                computation of chat rows IS per-chat attention.
//   G4 repeat (race) test  : 1000 runs at 8x4096 and 250 runs at 60x1024. O, LSE, dK, dV
//                            must be bitwise equal; dQ within FA's atol
//                            2*max|(dq + 0.3) - 0.3 - dq| in bf16
//                            (FA2 tests/test_flash_attn.py:2199-2240,
//                             FA3 hopper/test_flash_attn.py:1133-1165).
//   G5 large inputs        : q*5, k*3, v*3 (FA2 test_flash_attn_bwd_overflow, :2247-2295).
//   G6 chat alone          : layouts D2-D10. Every chat runs through today's kernel ALONE
//                            (row of roundup(L, 256), chat at offset 0, random filler
//                            after it with dO = 0) and must pass the rule against the
//                            per-chat reference of the packed layout. Causal attention
//                            cannot see the filler, so this is exact per-chat attention.
//                            G6c: the same chats alone at T=4096 must be bitwise equal
//                            to the short run (proves the filler trick).
//   G7 timing              : today's kernels, cudaEvent median/mean/min, FA's FLOP
//                            formula: causal fwd = B*Hq*2*T*(T/2)*(2*hd)
//                            (hopper/benchmark_attn.py:62-73); bwd = 2.5 x fwd (:387).
//   G8 B forward (Step 3)  : packed rows through gqa_flash_attn_forward_qkv(seg_start):
//                            O/LSE of every chat AND pad tail vs the per-segment reference;
//                            tile-aligned chats bitwise = alone; isolation (one chat changed,
//                            the others bitwise unchanged); offsets (D11); API refusals.
//   G9 B backward (Step 4) : packed rows through gqa_flash_attn_backward_qkv(seg_start), after
//                            the G8 forward: dQ/dK/dV (+O/LSE) of every chat AND pad tail (random
//                            dO everywhere) vs the per-segment reference; 1-token segments vs the
//                            derived bound; tile-aligned chats = alone (dK/dV bitwise, dQ atol);
//                            isolation incl. dO; refusals (needs the seg_start message); repeat
//                            x1000 on 8 real rows; tile-visit model + timing packed vs dense.
//   G10 A forward (Step 5) : packed rows through gqa_flash_attn_forward_qkv(GqaChatPlan): O/LSE of
//                            every segment vs the reference; EVERY segment bitwise = alone (A cuts
//                            tiles at the segment start like dense at row 0); isolation; refusals
//                            incl. the plan builder; forward time dense vs B vs A on real rows.
//   G11 A backward (Step 5): the same plan through gqa_flash_attn_backward_qkv, G9's inputs: dQ/dK/dV
//                            of every segment vs the reference; EVERY segment = alone (O/LSE/dK/dV
//                            bitwise, dQ atol); isolation incl. dO; refusals; repeat; bwd time.
//   PENDING items are printed in the SUMMARY.
//
// DATA LOADS
//   D1  real v1 rows (1 chat + pad)    D2  2048+2048          D3  1000+3096
//   D4  1+4095, 4095+1                 D5  64x64, 32x128       D6  4096 chats of 1 token
//   D7  63,65,127,129,255,257 repeated D8  random lengths, pad tails 0/1/16/63/64/65/255
//   D9  all-pad row                    D10 real stage-0 rows, packed first-fit decreasing
//                                          (same rule as pack_sim.py, 1517 rows)
//
// ENV
//   VARLEN_STAGE0_LABELS  path to stage-0 train/labels.bin (D1, D10); the laptop path
//                         is the default. On the box:
//                         /home/operator/gautam_k/sft_data/stage0_ultrachat_5k_T4096/train/labels.bin
//   VARLEN_QUICK=1        smoke subset
//   VARLEN_D10_ROWS=n     use only n packed rows of D10 (default: all)
//   VARLEN_D10_FIRST=r    start D10 at row r (multiple of 8); the inputs equal the full run's
//   VARLEN_REPEATS=n      repeats for G4 (default 1000 / 250)
//   VARLEN_ONLY=D10       run only G6 D10 (for diagnostics)
//   VARLEN_ONLY=G8        run only G8 (design B forward); with VARLEN_QUICK=1 it is ~1-2 min
//   VARLEN_ONLY=G7        run only G7 (timing), seconds per pass: for same-session A/B runs that
//                         alternate two library builds (one pass per library per round)
//   VARLEN_ONLY=G10       run only G10 (design A forward); VARLEN_CASE=<name part> limits the layouts
//   VARLEN_ONLY=BENCH     Step 6 timing of our dense / B / A (VARLEN_BENCH_ITERS, VARLEN_BENCH_EXPORT=<file>)
//   VARLEN_ONLY=G9        run only G9 (design B backward); with USE_FLASH=1 or USE_DETERMINISTIC_DQ=1
//                         it runs only the mode refusal (seconds)
//   VARLEN_DIAG_ROW=r, VARLEN_DIAG_T0=t, VARLEN_DIAG_REPEAT=n
//                         dQ diagnostic of the D10 chat at row r, start t (see diag_chat)
//
// Build + run (from Tensor-Implementations/):
//   make run-cuda FILE=Tests/kernel_tests/test_varlen_attention.cu
// ============================================================================
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "ops/helpers/AttentionKernels.h"
#include "ops/helpers/KernelDispatch.h"   // get_arch(): the same check the dispatcher makes
#include "ops/helpers/KernelPolicy.h"     // disable_sm90_gqa(): env BLUTRAIN_NO_SM90[_GQA]
#include "dtype/Types.h"

using OwnTensor::bfloat16_t;
using bf16 = __nv_bfloat16;
namespace otc = OwnTensor::cuda;

#define CUDA_CHECK(call)                                                          \
  do {                                                                            \
    cudaError_t e_ = (call);                                                      \
    if (e_ != cudaSuccess) {                                                      \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,          \
                   cudaGetErrorString(e_));                                       \
      std::exit(2);                                                               \
    }                                                                             \
  } while (0)
#define KERNEL_CHECK() CUDA_CHECK(cudaGetLastError())

static bool g_quick = false;
// Diagnostic of one D10 chat (VARLEN_DIAG_ROW = absolute D10 row, VARLEN_DIAG_T0 = chat start).
static int g_diag_row = -1, g_diag_t0 = -1, g_diag_rep = 50;
// G9: VARLEN_DIAG_CASE = which layout the diagnostic row belongs to (substring of its name, default
// "D10"); VARLEN_CASE = run only the G9 layouts whose name contains it (no D10/repeat/timing/refusals).
static std::string g_diag_case = "D10", g_case;
// G9: VARLEN_DIAG_LIST="row:t0,row:t0,..." = segments to diagnose (packed vs alone for dQ/dK/dV + a dV
// arithmetic model). With a list, G9 runs only the 8-row batches that contain those rows (same seeds
// as the full run) and skips the repeat/timing/refusal parts.
static std::vector<std::pair<int, int>> g_diag_list;
// false in USE_FLASH=1 runs: Vj1 sums the heads' dK/dV with bf16 atomics, so they differ run to run by
// one bf16 step (GQA_bwd_generic.cu header). Bitwise dK/dV checks then use FA's atol on dK/dV instead.
static bool g_kv_deterministic = true;

// ---------------------------------------------------------------------------
// Exact arithmetic helpers.
// The test is compiled with --use_fast_math (TENSOR_NVCCFLAGS, Makefile:142-143):
// float division and logf become approximations and a*b+c may be fused. So every
// rounding of the references is spelled out: IEEE intrinsics on the device, plain
// operators on the host (the default x86-64 target has no FMA, so no contraction),
// fma() where a fused multiply-add is wanted on both sides.
// ---------------------------------------------------------------------------
__host__ __device__ inline float f_add(float a, float b) {
#ifdef __CUDA_ARCH__
  return __fadd_rn(a, b);
#else
  return a + b;
#endif
}
__host__ __device__ inline float f_sub(float a, float b) {
#ifdef __CUDA_ARCH__
  return __fsub_rn(a, b);
#else
  return a - b;
#endif
}
__host__ __device__ inline float f_mul(float a, float b) {
#ifdef __CUDA_ARCH__
  return __fmul_rn(a, b);
#else
  return a * b;
#endif
}
__host__ __device__ inline float f_div(float a, float b) {
#ifdef __CUDA_ARCH__
  return __fdiv_rn(a, b);
#else
  return a / b;
#endif
}
__host__ __device__ inline double d_add(double a, double b) {
#ifdef __CUDA_ARCH__
  return __dadd_rn(a, b);
#else
  return a + b;
#endif
}
__host__ __device__ inline double d_sub(double a, double b) {
#ifdef __CUDA_ARCH__
  return __dsub_rn(a, b);
#else
  return a - b;
#endif
}
__host__ __device__ inline double d_mul(double a, double b) {
#ifdef __CUDA_ARCH__
  return __dmul_rn(a, b);
#else
  return a * b;
#endif
}
__host__ __device__ inline double d_div(double a, double b) {
#ifdef __CUDA_ARCH__
  return __ddiv_rn(a, b);
#else
  return a / b;
#endif
}
__host__ __device__ inline float  add_t(float a, float b)   { return f_add(a, b); }
__host__ __device__ inline double add_t(double a, double b) { return d_add(a, b); }
__host__ __device__ inline float  sub_t(float a, float b)   { return f_sub(a, b); }
__host__ __device__ inline double sub_t(double a, double b) { return d_sub(a, b); }
__host__ __device__ inline float  mul_t(float a, float b)   { return f_mul(a, b); }
__host__ __device__ inline double mul_t(double a, double b) { return d_mul(a, b); }
__host__ __device__ inline float  quot_t(float a, float b)   { return f_div(a, b); }
__host__ __device__ inline double quot_t(double a, double b) { return d_div(a, b); }
__host__ __device__ inline float  fma_t(float a, float b, float c)    { return fmaf(a, b, c); }
__host__ __device__ inline double fma_t(double a, double b, double c) { return fma(a, b, c); }
__host__ __device__ inline double exp_t(double x) { return exp(x); }
__host__ __device__ inline float  exp_t(float x)  { return (float)exp((double)x); }
__host__ __device__ inline double log_t(double x) { return log(x); }
__host__ __device__ inline float  log_t(float x)  { return (float)log((double)x); }
__host__ __device__ inline float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
template <bool R> __host__ __device__ inline float  rnd(float x)  { return R ? bf16r(x) : x; }
template <bool R> __host__ __device__ inline double rnd(double x) { return x; }

__host__ __device__ inline uint64_t splitmix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

static int grid_for(size_t n, int block = 256) {
  size_t g = (n + block - 1) / block;
  return (int)std::max<size_t>(1, std::min<size_t>(g, 65535));
}

// ---------------------------------------------------------------------------
// Device buffer. ensure() only grows the allocation; n is the live element count.
// ---------------------------------------------------------------------------
template <typename T>
struct DBuf {
  T* p = nullptr;
  size_t n = 0, cap = 0;
  DBuf() = default;
  ~DBuf() { release(); }
  DBuf(const DBuf&) = delete;
  DBuf& operator=(const DBuf&) = delete;
  void ensure(size_t m) {
    if (m > cap) {
      release();
      CUDA_CHECK(cudaMalloc(&p, m * sizeof(T)));
      cap = m;
    }
    n = m;
  }
  void release() {
    if (p) CUDA_CHECK(cudaFree(p));
    p = nullptr;
    n = cap = 0;
  }
  void fill_nan() { if (n) CUDA_CHECK(cudaMemset(p, 0xFF, n * sizeof(T))); }   // bf16/fp32/fp64 NaN
  std::vector<T> to_host() const {
    std::vector<T> h(n);
    if (n) CUDA_CHECK(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
  }
  void copy_from(const DBuf& o) {
    ensure(o.n);
    if (n) CUDA_CHECK(cudaMemcpy(p, o.p, n * sizeof(T), cudaMemcpyDeviceToDevice));
  }
};

// ---------------------------------------------------------------------------
// Problem, layouts, segments
// ---------------------------------------------------------------------------
struct Problem {
  int B = 0, T = 0, Hq = 0, Hkv = 0, hd = 128;
  int G() const { return Hq / Hkv; }
  size_t nQ() const { return (size_t)B * Hq * T * hd; }
  size_t nKV() const { return (size_t)B * Hkv * T * hd; }
  size_t nL() const { return (size_t)B * Hq * T; }
};

// A segment is one attention block: a chat, or a row's pad tail (its own segment, the
// way torchtitan keeps padding apart: tests/unit_tests/cpu/test_varlen_attention.py:64-78).
struct Segment {
  int b, t0, L;
  bool pad;
};

// rows[r] = chat lengths in order. 0 = an empty entry (skipped, like torchtitan's
// fixed-size cu_seqlens [0,3,5,9,9,9]); the rest of the row is pad.
struct Layout {
  std::string name;
  int T = 4096;
  std::vector<std::vector<int>> rows;
};

static std::vector<Segment> packed_segments(const Layout& lo) {
  std::vector<Segment> s;
  for (int b = 0; b < (int)lo.rows.size(); ++b) {
    int t = 0;
    for (int L : lo.rows[b]) {
      if (L < 0) throw std::runtime_error(lo.name + ": negative chat length");
      if (L == 0) continue;
      s.push_back({b, t, L, false});
      t += L;
    }
    if (t > lo.T) throw std::runtime_error(lo.name + ": row " + std::to_string(b) + " is longer than T");
    if (t < lo.T) s.push_back({b, t, lo.T - t, true});
  }
  return s;
}
static std::vector<Segment> dense_segments(int B, int T) {
  std::vector<Segment> s;
  for (int b = 0; b < B; ++b) s.push_back({b, 0, T, false});
  return s;
}

// ---------------------------------------------------------------------------
// Inputs: bf16 [B, H, T, hd], the layout gqa_flash_attn_*_qkv takes
// ---------------------------------------------------------------------------
struct Inputs {
  Problem p;
  DBuf<bf16> q, k, v, dO;
};

// Counter-based normal(0,1) * scale, rounded to bf16. Deterministic for a given seed.
__global__ void k_fill_normal(bf16* x, size_t n, uint64_t seed, float scale) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x) {
    const uint64_t a = splitmix64(seed ^ splitmix64(2 * i));
    const uint64_t c = splitmix64(seed ^ splitmix64(2 * i + 1));
    const double u1 = ((double)(a >> 11) + 1.0) * 0x1.0p-53;   // (0, 1]
    const double u2 = (double)(c >> 11) * 0x1.0p-53;           // [0, 1)
    const double z = sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
    x[i] = __float2bfloat16_rn((float)(z * scale));
  }
}
static void fill_normal(DBuf<bf16>& x, uint64_t seed, float scale) {
  k_fill_normal<<<grid_for(x.n), 256>>>(x.p, x.n, seed, scale);
  KERNEL_CHECK();
}

// Zero tokens [t0, t1) of row b in every head (dO of pad tokens: ignored labels give
// exactly zero output gradients there in training).
__global__ void k_zero_tokens(bf16* x, int H, int T, int hd, int b, int t0, int t1) {
  const size_t n = (size_t)H * (t1 - t0) * hd;
  for (size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x; e < n;
       e += (size_t)gridDim.x * blockDim.x) {
    const int d = (int)(e % hd);
    const size_t r = e / hd;
    const int t = t0 + (int)(r % (t1 - t0));
    const int h = (int)(r / (t1 - t0));
    x[(((size_t)b * H + h) * T + t) * hd + d] = __float2bfloat16_rn(0.f);
  }
}
static void zero_tokens(DBuf<bf16>& x, int H, int T, int hd, int b, int t0, int t1) {
  if (t1 <= t0) return;
  k_zero_tokens<<<grid_for((size_t)H * (t1 - t0) * hd), 256>>>(x.p, H, T, hd, b, t0, t1);
  KERNEL_CHECK();
}

// Copy tokens [s0, s0+L) of row bs into tokens [d0, d0+L) of row bd, every head.
__global__ void k_copy_tokens(const bf16* src, int Ts, int bs, int s0, bf16* dst, int Td, int bd,
                              int d0, int H, int L, int hd) {
  const size_t n = (size_t)H * L * hd;
  for (size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x; e < n;
       e += (size_t)gridDim.x * blockDim.x) {
    const int d = (int)(e % hd);
    const size_t r = e / hd;
    const int i = (int)(r % L), h = (int)(r / L);
    dst[(((size_t)bd * H + h) * Td + d0 + i) * hd + d] = src[(((size_t)bs * H + h) * Ts + s0 + i) * hd + d];
  }
}

// G13 switches (2026-10-12): default input scales for make_inputs, FA2's large-input gate for the
// gradients in g9_layout/g11_layout, and the number of rows per kernel call there (normally 8).
static float g_in_qs = 1.f, g_in_ks = 1.f, g_in_vs = 1.f;
static bool g_large_rule = false;
static size_t g_rows_per_call = 8;
static void make_inputs(Inputs& in, const Problem& p, uint64_t seed, float qs = g_in_qs, float ks = g_in_ks,
                        float vs = g_in_vs) {
  in.p = p;
  in.q.ensure(p.nQ());
  in.k.ensure(p.nKV());
  in.v.ensure(p.nKV());
  in.dO.ensure(p.nQ());
  fill_normal(in.q, splitmix64(seed * 4 + 1), qs);
  fill_normal(in.k, splitmix64(seed * 4 + 2), ks);
  fill_normal(in.v, splitmix64(seed * 4 + 3), vs);
  fill_normal(in.dO, splitmix64(seed * 4 + 4), 1.f);
}
// Training layout: pad tails get dO = 0.
static void zero_pad_grads(Inputs& in, const std::vector<Segment>& segs) {
  for (const Segment& s : segs)
    if (s.pad) zero_tokens(in.dO, in.p.Hq, in.p.T, in.p.hd, s.b, s.t0, s.t0 + s.L);
}

// ===========================================================================
// GPU references (truth: T=double, R=false; floor: T=float, R=true)
// ===========================================================================
template <typename T>
__global__ void k_gather(const bf16* __restrict__ src, int H, int Ts, int hd, int b, int t0, int L,
                         T* __restrict__ dst) {
  const size_t n = (size_t)H * L * hd;
  for (size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x; e < n;
       e += (size_t)gridDim.x * blockDim.x) {
    const int d = (int)(e % hd);
    const size_t r = e / hd;
    const int i = (int)(r % L), h = (int)(r / L);
    dst[e] = (T)__bfloat162float(src[(((size_t)b * H + h) * Ts + t0 + i) * hd + d]);
  }
}
template <typename T>
__global__ void k_scatter(const T* __restrict__ src, int H, int L, int D, T* __restrict__ dst, int Td,
                          int b, int t0) {
  const size_t n = (size_t)H * L * D;
  for (size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x; e < n;
       e += (size_t)gridDim.x * blockDim.x) {
    const int d = (int)(e % D);
    const size_t r = e / D;
    const int i = (int)(r % L), h = (int)(r / L);
    dst[(((size_t)b * H + h) * Td + t0 + i) * D + d] = src[e];
  }
}
// Floor only: Ks = bf16(k * scale)  (attention_ref with reorder_ops scales k).
__global__ void k_make_ks(const float* K, float* Ks, size_t n, float scale) {
  for (size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x; e < n;
       e += (size_t)gridDim.x * blockDim.x)
    Ks[e] = bf16r(f_mul(K[e], scale));
}

// C[z][m][n] = R( alpha * sum_{k=0}^{K-1} A(z,m,k) * B(z/zdiv,n,k) )
// A(z,m,k) = A[z*sAz + m*sAm + k*sAk], B(zb,n,k) = B[zb*sBz + n*sBn + k*sBk].
// Each thread owns one C element and adds k = 0, 1, ..., K-1 in that order with fma,
// so the sum order is the plain triple loop of the CPU self-test.
template <typename T, bool R>
__global__ void k_gemm(const T* __restrict__ A, long long sAz, long long sAm, long long sAk,
                       const T* __restrict__ B, long long sBz, long long sBn, long long sBk, int zdiv,
                       T* __restrict__ C, long long sCz, long long sCm, int M, int N, int K, T alpha) {
  __shared__ T As[16][17];
  __shared__ T Bs[16][17];
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int z = blockIdx.z, zb = z / zdiv;
  const int m = blockIdx.y * 16 + ty, n = blockIdx.x * 16 + tx, nb = blockIdx.x * 16 + ty;
  const T* Az = A + (long long)z * sAz;
  const T* Bz = B + (long long)zb * sBz;
  T acc = T(0);
  for (int k0 = 0; k0 < K; k0 += 16) {
    const int kk0 = k0 + tx;
    As[ty][tx] = (m < M && kk0 < K) ? Az[(long long)m * sAm + (long long)kk0 * sAk] : T(0);
    Bs[ty][tx] = (nb < N && kk0 < K) ? Bz[(long long)nb * sBn + (long long)kk0 * sBk] : T(0);
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < 16; ++kk) acc = fma_t(As[ty][kk], Bs[tx][kk], acc);
    __syncthreads();
  }
  if (m < M && n < N) C[(long long)z * sCz + (long long)m * sCm + n] = rnd<R>(mul_t(alpha, acc));
}
template <typename T, bool R>
static void gemm(const T* A, long long sAz, long long sAm, long long sAk, const T* B, long long sBz,
                 long long sBn, long long sBk, int zdiv, T* C, long long sCz, long long sCm, int M, int N,
                 int K, T alpha, int Z) {
  dim3 blk(16, 16), grd((N + 15) / 16, (M + 15) / 16, Z);
  k_gemm<T, R><<<grd, blk>>>(A, sAz, sAm, sAk, B, sBz, sBn, sBk, zdiv, C, sCz, sCm, M, N, K, alpha);
  KERNEL_CHECK();
}

// Row softmax of S[h][i][0..L) in place (keys j <= i are visible: causal inside the
// segment). Writes P (rounded for the floor) and LSE = max + log(sum).
// The sum runs in index order in one thread, like the CPU self-test.
template <typename T, bool R>
__global__ void k_softmax(T* S, T* LSE, int L) {
  const int i = blockIdx.x, h = blockIdx.y;
  T* row = S + ((size_t)h * L + i) * L;
  __shared__ T red[256];
  __shared__ T m_sh, l_sh;
  T m = T(-INFINITY);
  for (int j = threadIdx.x; j <= i; j += blockDim.x) m = row[j] > m ? row[j] : m;
  red[threadIdx.x] = m;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] = red[threadIdx.x + s] > red[threadIdx.x] ? red[threadIdx.x + s] : red[threadIdx.x];
    __syncthreads();
  }
  if (threadIdx.x == 0) m_sh = red[0];
  __syncthreads();
  const T mm = m_sh;
  for (int j = threadIdx.x; j < L; j += blockDim.x) row[j] = (j <= i) ? exp_t(sub_t(row[j], mm)) : T(0);
  __syncthreads();
  if (threadIdx.x == 0) {
    T l = T(0);
    for (int j = 0; j <= i; ++j) l = add_t(l, row[j]);
    l_sh = l;
    LSE[(size_t)h * L + i] = add_t(mm, log_t(l));
  }
  __syncthreads();
  const T l = l_sh;
  for (int j = threadIdx.x; j <= i; j += blockDim.x) row[j] = rnd<R>(quot_t(row[j], l));
}
// rs[r] = sum_j P[r][j] * dP[r][j], index order (softmax backward's row term).
template <typename T>
__global__ void k_rowsum(const T* P, const T* dP, T* rs, size_t rows, int L) {
  for (size_t r = (size_t)blockIdx.x * blockDim.x + threadIdx.x; r < rows; r += (size_t)gridDim.x * blockDim.x) {
    const T* p = P + r * L;
    const T* d = dP + r * L;
    T acc = T(0);
    for (int j = 0; j < L; ++j) acc = fma_t(p[j], d[j], acc);
    rs[r] = acc;
  }
}
// dS = R( P * (dP - rs) ) on visible entries, 0 elsewhere; written over dP.
template <typename T, bool R>
__global__ void k_ds(const T* P, T* dPdS, const T* rs, size_t n, int L) {
  for (size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x; e < n; e += (size_t)gridDim.x * blockDim.x) {
    const size_t r = e / L;
    const int j = (int)(e % L), i = (int)(r % L);
    dPdS[e] = (j <= i) ? rnd<R>(mul_t(P[e], sub_t(dPdS[e], rs[r]))) : T(0);
  }
}
// Floor only: dK_rep = bf16(dKs * scale)  (backward of k * scale).
__global__ void k_scale_round(float* x, size_t n, float scale) {
  for (size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x; e < n; e += (size_t)gridDim.x * blockDim.x)
    x[e] = bf16r(f_mul(x[e], scale));
}
// X[hkv] = R( sum_{g=0}^{G-1} Xh[hkv*G + g] )  (backward of the GQA repeat: a sum over the group).
template <typename T, bool R>
__global__ void k_group_sum(const T* Xh, T* X, int Hkv, int G, int L, int hd) {
  const size_t plane = (size_t)L * hd, n = (size_t)Hkv * plane;
  for (size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x; e < n; e += (size_t)gridDim.x * blockDim.x) {
    const size_t hkv = e / plane, off = e % plane;
    T acc = T(0);
    for (int g = 0; g < G; ++g) acc = add_t(acc, Xh[(hkv * G + g) * plane + off]);
    X[e] = rnd<R>(acc);
  }
}

template <typename T>
struct RefScratch {
  DBuf<T> Q, K, Ks, V, dO, S, dP, O, LSE, rs, dQ, dKh, dVh, dK, dV;
  void ensure(const Problem& p, int L) {
    const size_t qd = (size_t)p.Hq * L * p.hd, kd = (size_t)p.Hkv * L * p.hd;
    const size_t ll = (size_t)p.Hq * L * L, ql = (size_t)p.Hq * L;
    Q.ensure(qd); K.ensure(kd); Ks.ensure(kd); V.ensure(kd); dO.ensure(qd);
    S.ensure(ll); dP.ensure(ll); O.ensure(qd); LSE.ensure(ql); rs.ensure(ql);
    dQ.ensure(qd); dKh.ensure(qd); dVh.ensure(qd); dK.ensure(kd); dV.ensure(kd);
  }
};

// Full-size reference outputs: [B, H, T, hd] and LSE [B, Hq, T].
struct RefOut {
  DBuf<double> O, LSE, dQ, dK, dV;     // truth (fp64)
  DBuf<float> Of, LSEf, dQf, dKf, dVf;  // floor (bf16 values held in fp32; LSE fp32)
  void ensure(const Problem& p) {
    O.ensure(p.nQ()); LSE.ensure(p.nL()); dQ.ensure(p.nQ()); dK.ensure(p.nKV()); dV.ensure(p.nKV());
    Of.ensure(p.nQ()); LSEf.ensure(p.nL()); dQf.ensure(p.nQ()); dKf.ensure(p.nKV()); dVf.ensure(p.nKV());
  }
  void fill_nan() {
    O.fill_nan(); LSE.fill_nan(); dQ.fill_nan(); dK.fill_nan(); dV.fill_nan();
    Of.fill_nan(); LSEf.fill_nan(); dQf.fill_nan(); dKf.fill_nan(); dVf.fill_nan();
  }
};

// One segment, all heads: gather -> forward -> backward -> scatter into the full arrays.
template <typename T, bool R>
static void ref_segment(const Inputs& in, const Segment& sg, RefScratch<T>& s, T alpha_qk, float scale_f,
                        T* O, T* LSE, T* dQ, T* dK, T* dV) {
  const Problem& p = in.p;
  const int L = sg.L, hd = p.hd, Hq = p.Hq, Hkv = p.Hkv, G = p.G();
  s.ensure(p, L);
  const size_t qd = (size_t)Hq * L * hd, kd = (size_t)Hkv * L * hd, ll = (size_t)Hq * L * L, ql = (size_t)Hq * L;
  k_gather<T><<<grid_for(qd), 256>>>(in.q.p, Hq, p.T, hd, sg.b, sg.t0, L, s.Q.p);
  k_gather<T><<<grid_for(kd), 256>>>(in.k.p, Hkv, p.T, hd, sg.b, sg.t0, L, s.K.p);
  k_gather<T><<<grid_for(kd), 256>>>(in.v.p, Hkv, p.T, hd, sg.b, sg.t0, L, s.V.p);
  k_gather<T><<<grid_for(qd), 256>>>(in.dO.p, Hq, p.T, hd, sg.b, sg.t0, L, s.dO.p);
  KERNEL_CHECK();
  const T* Ku = s.K.p;
  if constexpr (R) {
    k_make_ks<<<grid_for(kd), 256>>>(s.K.p, s.Ks.p, kd, scale_f);
    KERNEL_CHECK();
    Ku = s.Ks.p;
  }
  const long long Ld = (long long)L * hd, LL = (long long)L * L;
  // S = alpha * Q Ku^T                 [Hq, L, L]
  gemm<T, R>(s.Q.p, Ld, hd, 1, Ku, Ld, hd, 1, G, s.S.p, LL, L, L, L, hd, alpha_qk, Hq);
  k_softmax<T, R><<<dim3(L, Hq), 256>>>(s.S.p, s.LSE.p, L);
  KERNEL_CHECK();
  // O = P V          A(i,j) = P[h][i][j], B(d,j) = V[hkv][j][d]
  gemm<T, R>(s.S.p, LL, L, 1, s.V.p, Ld, 1, hd, G, s.O.p, Ld, hd, L, hd, L, T(1), Hq);
  // dP = dO V^T      A(i,d) = dO[h][i][d], B(j,d) = V[hkv][j][d]
  gemm<T, R>(s.dO.p, Ld, hd, 1, s.V.p, Ld, hd, 1, G, s.dP.p, LL, L, L, L, hd, T(1), Hq);
  // dVh = P^T dO     A(j,i) = P[h][i][j], B(d,i) = dO[h][i][d]
  gemm<T, R>(s.S.p, LL, 1, L, s.dO.p, Ld, 1, hd, 1, s.dVh.p, Ld, hd, L, hd, L, T(1), Hq);
  k_rowsum<T><<<grid_for(ql), 256>>>(s.S.p, s.dP.p, s.rs.p, ql, L);
  k_ds<T, R><<<grid_for(ll), 256>>>(s.S.p, s.dP.p, s.rs.p, ll, L);
  KERNEL_CHECK();
  // dQ = alpha * dS Ku     A(i,j) = dS[h][i][j], B(d,j) = Ku[hkv][j][d]
  gemm<T, R>(s.dP.p, LL, L, 1, Ku, Ld, 1, hd, G, s.dQ.p, Ld, hd, L, hd, L, alpha_qk, Hq);
  // dKh = alpha * dS^T Q   A(j,i) = dS[h][i][j], B(d,i) = Q[h][i][d]
  gemm<T, R>(s.dP.p, LL, 1, L, s.Q.p, Ld, 1, hd, 1, s.dKh.p, Ld, hd, L, hd, L, alpha_qk, Hq);
  if constexpr (R) {
    k_scale_round<<<grid_for(qd), 256>>>(s.dKh.p, qd, scale_f);
    KERNEL_CHECK();
  }
  k_group_sum<T, R><<<grid_for(kd), 256>>>(s.dKh.p, s.dK.p, Hkv, G, L, hd);
  k_group_sum<T, R><<<grid_for(kd), 256>>>(s.dVh.p, s.dV.p, Hkv, G, L, hd);
  KERNEL_CHECK();
  k_scatter<T><<<grid_for(qd), 256>>>(s.O.p, Hq, L, hd, O, p.T, sg.b, sg.t0);
  k_scatter<T><<<grid_for(ql), 256>>>(s.LSE.p, Hq, L, 1, LSE, p.T, sg.b, sg.t0);
  k_scatter<T><<<grid_for(qd), 256>>>(s.dQ.p, Hq, L, hd, dQ, p.T, sg.b, sg.t0);
  k_scatter<T><<<grid_for(kd), 256>>>(s.dK.p, Hkv, L, hd, dK, p.T, sg.b, sg.t0);
  k_scatter<T><<<grid_for(kd), 256>>>(s.dV.p, Hkv, L, hd, dV, p.T, sg.b, sg.t0);
  KERNEL_CHECK();
}

// Reference for every segment of a batch. Positions not covered by a segment stay NaN.
static void reference(const Inputs& in, const std::vector<Segment>& segs, RefOut& out) {
  static RefScratch<double> sd;
  static RefScratch<float> sf;
  out.ensure(in.p);
  out.fill_nan();
  const double scale = 1.0 / std::sqrt((double)in.p.hd);
  const float scale_f = (float)(1.0 / std::sqrt((double)in.p.hd));   // what PyTorch's k * (1/sqrt(d)) uses
  for (const Segment& sg : segs) {
    if (sg.L <= 0) continue;
    ref_segment<double, false>(in, sg, sd, scale, scale_f, out.O.p, out.LSE.p, out.dQ.p, out.dK.p, out.dV.p);
    ref_segment<float, true>(in, sg, sf, 1.0f, scale_f, out.Of.p, out.LSEf.p, out.dQf.p, out.dKf.p, out.dVf.p);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
}

// ===========================================================================
// CPU reference for G1: the FULL row with an explicit block-diagonal causal mask
// (FlexAttention-style mask: same segment and j <= i), no gathering.
// ===========================================================================
template <typename T, bool R>
static void cpu_reference(const Problem& p, const std::vector<int>& seg_id, const std::vector<float>& q,
                          const std::vector<float>& k, const std::vector<float>& v,
                          const std::vector<float>& dO, T alpha, float scale_f, std::vector<T>& O,
                          std::vector<T>& LSE, std::vector<T>& dQ, std::vector<T>& dK, std::vector<T>& dV) {
  const int Tn = p.T, hd = p.hd, G = p.G();
  O.assign(p.nQ(), T(0)); LSE.assign(p.nL(), T(0)); dQ.assign(p.nQ(), T(0));
  dK.assign(p.nKV(), T(0)); dV.assign(p.nKV(), T(0));
  auto qi = [&](int b, int h, int t, int d) { return (((size_t)b * p.Hq + h) * Tn + t) * hd + d; };
  auto ki = [&](int b, int h, int t, int d) { return (((size_t)b * p.Hkv + h) * Tn + t) * hd + d; };
  std::vector<T> P((size_t)Tn * Tn), dS((size_t)Tn * Tn), Ks((size_t)Tn * hd);
  std::vector<std::vector<T>> dKh(G, std::vector<T>((size_t)Tn * hd)), dVh(G, std::vector<T>((size_t)Tn * hd));
  for (int b = 0; b < p.B; ++b) {
#ifndef VARLEN_SELFTEST_MUTANT
    auto vis = [&](int i, int j) { return j <= i && seg_id[(size_t)b * Tn + j] == seg_id[(size_t)b * Tn + i]; };
#else
    // Negative control: plain causal over the whole row (chat borders ignored).
    // Built with -DVARLEN_SELFTEST_MUTANT, G1 must FAIL on every multi-segment row.
    auto vis = [&](int i, int j) { (void)seg_id; return j <= i; };
#endif
    for (int hkv = 0; hkv < p.Hkv; ++hkv) {
      for (int j = 0; j < Tn; ++j)
        for (int d = 0; d < hd; ++d) {
          const float kv = k[ki(b, hkv, j, d)];
          Ks[(size_t)j * hd + d] = R ? (T)bf16r(f_mul(kv, scale_f)) : (T)kv;
        }
      for (int g = 0; g < G; ++g) {
        const int h = hkv * G + g;
        std::fill(P.begin(), P.end(), T(0));
        std::fill(dS.begin(), dS.end(), T(0));
        for (int i = 0; i < Tn; ++i) {   // forward
          T m = T(-INFINITY);
          for (int j = 0; j <= i; ++j) {
            if (!vis(i, j)) continue;
            T acc = T(0);
            for (int d = 0; d < hd; ++d) acc = fma_t((T)q[qi(b, h, i, d)], Ks[(size_t)j * hd + d], acc);
            const T sv = rnd<R>(mul_t(alpha, acc));
            P[(size_t)i * Tn + j] = sv;
            m = sv > m ? sv : m;
          }
          T l = T(0);
          for (int j = 0; j <= i; ++j) {
            if (!vis(i, j)) continue;
            const T e = exp_t(sub_t(P[(size_t)i * Tn + j], m));
            P[(size_t)i * Tn + j] = e;
            l = add_t(l, e);
          }
          LSE[((size_t)b * p.Hq + h) * Tn + i] = add_t(m, log_t(l));
          for (int j = 0; j <= i; ++j)
            if (vis(i, j)) P[(size_t)i * Tn + j] = rnd<R>(quot_t(P[(size_t)i * Tn + j], l));
          for (int d = 0; d < hd; ++d) {
            T acc = T(0);
            for (int j = 0; j <= i; ++j)
              if (vis(i, j)) acc = fma_t(P[(size_t)i * Tn + j], (T)v[ki(b, hkv, j, d)], acc);
            O[qi(b, h, i, d)] = rnd<R>(mul_t(T(1), acc));
          }
        }
        for (int i = 0; i < Tn; ++i) {   // backward: dP, row term, dS
          T rs = T(0);
          std::vector<T> dP(Tn, T(0));
          for (int j = 0; j <= i; ++j) {
            if (!vis(i, j)) continue;
            T acc = T(0);
            for (int d = 0; d < hd; ++d) acc = fma_t((T)dO[qi(b, h, i, d)], (T)v[ki(b, hkv, j, d)], acc);
            dP[j] = rnd<R>(mul_t(T(1), acc));
            rs = fma_t(P[(size_t)i * Tn + j], dP[j], rs);
          }
          for (int j = 0; j <= i; ++j)
            if (vis(i, j)) dS[(size_t)i * Tn + j] = rnd<R>(mul_t(P[(size_t)i * Tn + j], sub_t(dP[j], rs)));
        }
        for (int i = 0; i < Tn; ++i)     // dQ
          for (int d = 0; d < hd; ++d) {
            T acc = T(0);
            for (int j = 0; j <= i; ++j)
              if (vis(i, j)) acc = fma_t(dS[(size_t)i * Tn + j], Ks[(size_t)j * hd + d], acc);
            dQ[qi(b, h, i, d)] = rnd<R>(mul_t(alpha, acc));
          }
        for (int j = 0; j < Tn; ++j)     // dK, dV for this query head
          for (int d = 0; d < hd; ++d) {
            T ak = T(0), av = T(0);
            for (int i = j; i < Tn; ++i) {
              if (!vis(i, j)) continue;
              ak = fma_t(dS[(size_t)i * Tn + j], (T)q[qi(b, h, i, d)], ak);
              av = fma_t(P[(size_t)i * Tn + j], (T)dO[qi(b, h, i, d)], av);
            }
            T dk = rnd<R>(mul_t(alpha, ak));
            if (R) dk = (T)bf16r(f_mul((float)dk, scale_f));
            dKh[g][(size_t)j * hd + d] = dk;
            dVh[g][(size_t)j * hd + d] = rnd<R>(mul_t(T(1), av));
          }
      }
      for (int j = 0; j < Tn; ++j)       // GQA group sums
        for (int d = 0; d < hd; ++d) {
          T ak = T(0), av = T(0);
          for (int g = 0; g < G; ++g) {
            ak = add_t(ak, dKh[g][(size_t)j * hd + d]);
            av = add_t(av, dVh[g][(size_t)j * hd + d]);
          }
          dK[ki(b, hkv, j, d)] = rnd<R>(ak);
          dV[ki(b, hkv, j, d)] = rnd<R>(av);
        }
    }
  }
}

// ===========================================================================
// Comparison: per chat region, max |kernel - truth|, max |floor - truth|,
// max |(x + 0.3) - 0.3 - x| (fp32), and the count of non-finite kernel values.
// ===========================================================================
struct DevErr {
  unsigned long long ek, ef, er, nan;
};
struct Err {
  double ek = 0, ef = 0, er = 0;
  unsigned long long nan = 0;
};
__device__ inline double to_d(bf16 x) { return (double)__bfloat162float(x); }
__device__ inline double to_d(float x) { return (double)x; }

__device__ inline void block_max3(double& a, double& b, double& c, unsigned long long& n, DevErr* out) {
  __shared__ double sa[256], sb[256], sc[256];
  __shared__ unsigned long long sn[256];
  sa[threadIdx.x] = a; sb[threadIdx.x] = b; sc[threadIdx.x] = c; sn[threadIdx.x] = n;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      sa[threadIdx.x] = fmax(sa[threadIdx.x], sa[threadIdx.x + s]);
      sb[threadIdx.x] = fmax(sb[threadIdx.x], sb[threadIdx.x + s]);
      sc[threadIdx.x] = fmax(sc[threadIdx.x], sc[threadIdx.x + s]);
      sn[threadIdx.x] += sn[threadIdx.x + s];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {   // non-negative doubles order like their bit patterns
    atomicMax(&out->ek, (unsigned long long)__double_as_longlong(sa[0]));
    atomicMax(&out->ef, (unsigned long long)__double_as_longlong(sb[0]));
    atomicMax(&out->er, (unsigned long long)__double_as_longlong(sc[0]));
    atomicAdd(&out->nan, sn[0]);
  }
}

template <typename KT>
__global__ void k_compare(const KT* __restrict__ Kp, int bk, int Tk, int tk0, const double* __restrict__ Tp,
                          const float* __restrict__ Fp, int br, int Tr, int tr0, int H, int L, int D,
                          DevErr* out) {
  double ek = 0, ef = 0, er = 0;
  unsigned long long nan = 0;
  const size_t n = (size_t)H * L * D;
  for (size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x; e < n; e += (size_t)gridDim.x * blockDim.x) {
    const int d = (int)(e % D);
    const size_t r = e / D;
    const int i = (int)(r % L), h = (int)(r / L);
    const double kv = to_d(Kp[(((size_t)bk * H + h) * Tk + tk0 + i) * D + d]);
    const size_t ti = (((size_t)br * H + h) * Tr + tr0 + i) * D + d;
    const double tv = Tp[ti], fv = (double)Fp[ti];
    if (!isfinite(kv)) { ++nan; continue; }
    ek = fmax(ek, fabs(kv - tv));
    ef = fmax(ef, fabs(fv - tv));
    const float tf = (float)tv;
    er = fmax(er, (double)fabsf(f_sub(f_sub(f_add(tf, 0.3f), 0.3f), tf)));
  }
  block_max3(ek, ef, er, nan, out);
}

template <typename KT>
static Err compare_region(const KT* kern, int bk, int Tk, int tk0, const double* truth, const float* floor_,
                          int br, int Tr, int tr0, int H, int L, int D) {
  static DevErr* d = nullptr;
  if (!d) CUDA_CHECK(cudaMalloc(&d, sizeof(DevErr)));
  CUDA_CHECK(cudaMemset(d, 0, sizeof(DevErr)));
  const size_t n = (size_t)H * L * D;
  k_compare<KT><<<std::min(grid_for(n), 1024), 256>>>(kern, bk, Tk, tk0, truth, floor_, br, Tr, tr0, H, L, D, d);
  KERNEL_CHECK();
  DevErr h;
  CUDA_CHECK(cudaMemcpy(&h, d, sizeof(DevErr), cudaMemcpyDeviceToHost));
  Err e;
  std::memcpy(&e.ek, &h.ek, 8);
  std::memcpy(&e.ef, &h.ef, 8);
  std::memcpy(&e.er, &h.er, 8);
  e.nan = h.nan;
  return e;
}

// Non-finite count of a whole buffer (unwritten NaN-filled elements included).
template <typename X>
__global__ void k_nonfinite(const X* x, size_t n, unsigned long long* cnt) {
  unsigned long long c = 0;
  for (size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x; e < n; e += (size_t)gridDim.x * blockDim.x)
    if (!isfinite(to_d(x[e]))) ++c;
  if (c) atomicAdd(cnt, c);
}
template <typename X>
static unsigned long long nonfinite(const DBuf<X>& x) {
  static unsigned long long* d = nullptr;
  if (!d) CUDA_CHECK(cudaMalloc(&d, sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemset(d, 0, sizeof(unsigned long long)));
  k_nonfinite<X><<<std::min(grid_for(x.n), 1024), 256>>>(x.p, x.n, d);
  KERNEL_CHECK();
  unsigned long long h = 0;
  CUDA_CHECK(cudaMemcpy(&h, d, sizeof(h), cudaMemcpyDeviceToHost));
  return h;
}

// Region bit compare: element (h, i, d) of row ra/Ta vs row rb/Tb. Counts differing
// bit patterns and the max |a - b|.
struct BitDiff {
  unsigned long long mism = 0;
  double maxabs = 0;
};
__device__ inline unsigned int bits_of(bf16 x) { return __bfloat16_as_ushort(x); }
__device__ inline unsigned int bits_of(float x) { return __float_as_uint(x); }
template <typename X>
__global__ void k_bitdiff(const X* a, int ra, int Ta, int ta0, const X* b, int rb, int Tb, int tb0, int H, int L,
                          int D, unsigned long long* mism, unsigned long long* maxabs) {
  unsigned long long c = 0;
  double m = 0;
  const size_t n = (size_t)H * L * D;
  for (size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x; e < n; e += (size_t)gridDim.x * blockDim.x) {
    const int d = (int)(e % D);
    const size_t r = e / D;
    const int i = (int)(r % L), h = (int)(r / L);
    const X xa = a[(((size_t)ra * H + h) * Ta + ta0 + i) * D + d];
    const X xb = b[(((size_t)rb * H + h) * Tb + tb0 + i) * D + d];
    if (bits_of(xa) != bits_of(xb)) {
      ++c;
      const double da = to_d(xa), db = to_d(xb);
      m = fmax(m, isfinite(da) && isfinite(db) ? fabs(da - db) : INFINITY);
    }
  }
  if (c) {
    atomicAdd(mism, c);
    atomicMax(maxabs, (unsigned long long)__double_as_longlong(m));
  }
}
template <typename X>
static BitDiff bitdiff(const X* a, int ra, int Ta, int ta0, const X* b, int rb, int Tb, int tb0, int H, int L, int D) {
  static unsigned long long* d = nullptr;
  if (!d) CUDA_CHECK(cudaMalloc(&d, 2 * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemset(d, 0, 2 * sizeof(unsigned long long)));
  const size_t n = (size_t)H * L * D;
  k_bitdiff<X><<<std::min(grid_for(n), 1024), 256>>>(a, ra, Ta, ta0, b, rb, Tb, tb0, H, L, D, d, d + 1);
  KERNEL_CHECK();
  unsigned long long h[2];
  CUDA_CHECK(cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost));
  BitDiff r;
  r.mism = h[0];
  std::memcpy(&r.maxabs, &h[1], 8);
  return r;
}

// FA's dQ tolerance for repeat tests: 2 * max |bf16(bf16(dq + 0.3) - 0.3) - dq| over a region.
__global__ void k_dq_atol(const bf16* x, int r0, int T, int t0, int H, int L, int D, unsigned long long* out) {
  double m = 0;
  const size_t n = (size_t)H * L * D;
  for (size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x; e < n; e += (size_t)gridDim.x * blockDim.x) {
    const int d = (int)(e % D);
    const size_t r = e / D;
    const int i = (int)(r % L), h = (int)(r / L);
    const float v = __bfloat162float(x[(((size_t)r0 * H + h) * T + t0 + i) * D + d]);
    const float y = bf16r(f_sub(bf16r(f_add(v, 0.3f)), 0.3f));
    m = fmax(m, (double)fabsf(f_sub(y, v)));
  }
  atomicMax(out, (unsigned long long)__double_as_longlong(m));
}
static double dq_atol(const bf16* x, int r0, int T, int t0, int H, int L, int D) {
  static unsigned long long* d = nullptr;
  if (!d) CUDA_CHECK(cudaMalloc(&d, sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemset(d, 0, sizeof(unsigned long long)));
  const size_t n = (size_t)H * L * D;
  k_dq_atol<<<std::min(grid_for(n), 1024), 256>>>(x, r0, T, t0, H, L, D, d);
  KERNEL_CHECK();
  unsigned long long h;
  CUDA_CHECK(cudaMemcpy(&h, d, sizeof(h), cudaMemcpyDeviceToHost));
  double v;
  std::memcpy(&v, &h, 8);
  return 2.0 * v;
}

// ===========================================================================
// Kernels under test (today: the dense causal path)
// ===========================================================================
struct KOut {
  DBuf<bf16> O, dQ, dK, dV;
  DBuf<float> LSE;
  void ensure(const Problem& p) {
    O.ensure(p.nQ()); dQ.ensure(p.nQ()); dK.ensure(p.nKV()); dV.ensure(p.nKV()); LSE.ensure(p.nL());
  }
  void fill_nan() { O.fill_nan(); dQ.fill_nan(); dK.fill_nan(); dV.fill_nan(); LSE.fill_nan(); }
  unsigned long long nonfinite_all() const {
    return nonfinite(O) + nonfinite(dQ) + nonfinite(dK) + nonfinite(dV) + nonfinite(LSE);
  }
};
static const bfloat16_t* cb(const bf16* x) { return reinterpret_cast<const bfloat16_t*>(x); }
static bfloat16_t* mb(bf16* x) { return reinterpret_cast<bfloat16_t*>(x); }

static void today_fwd(const Inputs& in, KOut& o) {
  const Problem& p = in.p;
  otc::gqa_flash_attn_forward_qkv(cb(in.q.p), cb(in.k.p), cb(in.v.p), mb(o.O.p), o.LSE.p, p.B, p.Hq, p.Hkv, p.T,
                                  p.hd, /*is_causal=*/true);
}
static void today_bwd(const Inputs& in, KOut& o) {
  const Problem& p = in.p;
  otc::gqa_flash_attn_backward_qkv(cb(in.q.p), cb(in.k.p), cb(in.v.p), cb(o.O.p), cb(in.dO.p), o.LSE.p,
                                   mb(o.dQ.p), mb(o.dK.p), mb(o.dV.p), p.B, p.Hq, p.Hkv, p.T, p.hd,
                                   /*is_causal=*/true);
}
static bool run_today(const Inputs& in, KOut& o, std::string& err) {
  o.ensure(in.p);
  o.fill_nan();
  try {
    today_fwd(in, o);
    today_bwd(in, o);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  } catch (const std::exception& e) {
    err = e.what();
    return false;
  }
  return true;
}

// ===========================================================================
// Bookkeeping
// ===========================================================================
// ---------------------------------------------------------------------------
// Known, explained deviations, kept by the user's decision (2026-10-10) and observed in every
// run until the end of the v2 work. A failure counts as KNOWN only if the case, heads,
// tensor and chat all match AND its ratio is no worse than recorded (+1%), with no
// non-finite values. Anything else -- another chat, another tensor, a worse ratio -- is NEW.
// ---------------------------------------------------------------------------
struct KnownIssue {
  const char *case_prefix, *heads, *tensor, *chat;
  double max_ratio;
  const char* why;
};
static const KnownIssue kKnown[] = {
    {"G6 D10", "18/6", "dQ", "row 1022 chat@2552 L=1276", 1.156 * 1.01,
     "Vj1d computes dS with the bf16 P it re-reads from shared memory (fuse_dS_ldstsm, "
     "GQA_bwd_vj1d_sm90.cu:311-341); FA3 uses the fp32 P (hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp:893). "
     "2026-10-10 DIAG: the bf16-P model reproduces the kernel's dQ row exactly; the fp32-P model gives "
     "1.09x the floor. Hits rows with few keys (chat token 1) and large dP - D."},
    // Same mechanism, found by G9 quick on 2026-10-10 (random inputs, seed 906). Proven NOT a packing
    // effect: the same chat ALONE through today's dense kernels gives the same ratio.
    {"G9 D8", "18/6", "dQ", "chat row 1 @664 L=303", 1.049 * 1.01,
     "same Vj1d bf16-P dS issue as the G6 D10 entry. 2026-10-10 DIAG (G9_diag_D8_row1_t664_h200_gpu7_1010_1932.log): worst "
     "element at chat token 2 (3 keys), D = 12.9, sum_j bf16(P_j) = 0.99902; the bf16-P model reproduces the "
     "kernel's dQ row exactly, the fp32-P models give 0.74x the floor's row error; the SAME chat alone through "
     "today's dense kernels gives the same ratio 1.049 in 50/50 runs -> not caused by packing."},
    // G9 full on the real packed rows, 2026-10-10: 3 dQ (bf16-P dS) + 4 dV (per-head bf16 partials, a second
    // mechanism of today's Vj1d). Each proven NOT packing: same ratio alone through the dense kernels.
    {"G9 D10", "18/6", "dQ", "pad row 492 @4094 L=2", 1.062 * 1.01,
     "Vj1d bf16-P dS (same as the G6 D10 entry). Found by G9 full (G9_full_h200_gpu7_1010_1939.log); DIAG (G9_diag7_D10_h200_gpu7_1010_1955.log): the SAME segment alone through today's dense kernels gives the same ratio (not packing); for chat 791 the bf16-P model reproduces the kernel's dQ row exactly and fp32 P gives 0.92x the floor."},
    {"G9 D10", "18/6", "dQ", "chat row 791 @2998 L=1098", 1.121 * 1.01,
     "Vj1d bf16-P dS (same as the G6 D10 entry). Found by G9 full (G9_full_h200_gpu7_1010_1939.log); DIAG (G9_diag7_D10_h200_gpu7_1010_1955.log): the SAME segment alone through today's dense kernels gives the same ratio (not packing); for chat 791 the bf16-P model reproduces the kernel's dQ row exactly and fp32 P gives 0.92x the floor."},
    {"G9 D10", "18/6", "dQ", "chat row 1102 @3537 L=559", 1.096 * 1.01,
     "Vj1d bf16-P dS (same as the G6 D10 entry). Found by G9 full (G9_full_h200_gpu7_1010_1939.log); DIAG (G9_diag7_D10_h200_gpu7_1010_1955.log): the SAME segment alone through today's dense kernels gives the same ratio (not packing); for chat 791 the bf16-P model reproduces the kernel's dQ row exactly and fp32 P gives 0.92x the floor."},
    {"G9 D10", "18/6", "dV", "pad row 455 @4057 L=39", 1.031 * 1.01,
     "Vj1d rounds each query head's dV partial to bf16 before the fixed-order head sum (one-writer partials + gqa_dkdv_greduce, the price of bit-identical dK/dV); FA3 sums heads in fp32 and rounds once. Found by G9 full (G9_full_h200_gpu7_1010_1939.log); DIAG (G9_diag7_D10_h200_gpu7_1010_1955.log): the SAME segment alone through today's dense kernels gives the same ratio (not packing); the Vj1d arithmetic model reproduces the kernel's worst dV element exactly; an fp32 head sum brings the error to the floor's."},
    {"G9 D10", "18/6", "dV", "chat row 1140 @2283 L=1141", 1.171 * 1.01,
     "Vj1d rounds each query head's dV partial to bf16 before the fixed-order head sum (one-writer partials + gqa_dkdv_greduce, the price of bit-identical dK/dV); FA3 sums heads in fp32 and rounds once. Found by G9 full (G9_full_h200_gpu7_1010_1939.log); DIAG (G9_diag7_D10_h200_gpu7_1010_1955.log): the SAME segment alone through today's dense kernels gives the same ratio (not packing); the Vj1d arithmetic model reproduces the kernel's worst dV element exactly; an fp32 head sum brings the error to the floor's."},
    {"G9 D10", "18/6", "dV", "chat row 1262 @1005 L=1005", 1.033 * 1.01,
     "Vj1d rounds each query head's dV partial to bf16 before the fixed-order head sum (one-writer partials + gqa_dkdv_greduce, the price of bit-identical dK/dV); FA3 sums heads in fp32 and rounds once. Found by G9 full (G9_full_h200_gpu7_1010_1939.log); DIAG (G9_diag7_D10_h200_gpu7_1010_1955.log): the SAME segment alone through today's dense kernels gives the same ratio (not packing); the Vj1d arithmetic model reproduces the kernel's worst dV element exactly; an fp32 head sum brings the error to the floor's."},
    {"G9 D10", "18/6", "dV", "chat row 1359 @1714 L=857", 1.014 * 1.01,
     "Vj1d rounds each query head's dV partial to bf16 before the fixed-order head sum (one-writer partials + gqa_dkdv_greduce, the price of bit-identical dK/dV); FA3 sums heads in fp32 and rounds once. Found by G9 full (G9_full_h200_gpu7_1010_1939.log); DIAG (G9_diag7_D10_h200_gpu7_1010_1955.log): the SAME segment alone through today's dense kernels gives the same ratio (not packing); the Vj1d arithmetic model reproduces the kernel's worst dV element exactly; an fp32 head sum brings the error to the floor's."},
};
static const KnownIssue* match_known(const std::string& tag0, const std::string& tensor, const std::string& at,
                                     double ratio, unsigned long long nan) {
  if (nan != 0) return nullptr;
  // G11 (design A backward) runs exactly G9's inputs (same seeds) and A == alone == today's kernel per
  // segment, so a G9 entry covers the same segment in G11.
  const std::string tag = tag0.rfind("G11 ", 0) == 0 ? "G9 " + tag0.substr(4) : tag0;
  for (const KnownIssue& k : kKnown)
    if (tag.find(k.case_prefix) != std::string::npos && tag.find(k.heads) != std::string::npos &&
        tensor == k.tensor && at == k.chat && ratio <= k.max_ratio)
      return &k;
  return nullptr;
}

struct Summary {
  int pass = 0, fail = 0, skip = 0;
  std::vector<std::string> failed, skipped, known;
  void record(bool ok, const std::string& what) {
    if (ok) ++pass;
    else { ++fail; failed.push_back(what); }
  }
  void skip_(const std::string& what) { ++skip; skipped.push_back(what); }
};

// One tensor (O / LSE / dQ / dK / dV) across all chats of a case.
struct Check {
  std::string name;
  double mult = 2.0, extra = 0.0;   // bound = mult*err_floor + 2*fp32_round + extra
  int n = 0, fails = 0;
  double worst = -1, wk = 0, wf = 0;
  unsigned long long nan = 0;
  struct Fail {
    std::string at, detail;
    double ratio;
    unsigned long long nan;
  };
  std::vector<Fail> where;      // every failing entry (failures are rare)
  std::vector<double> ratios;   // every entry's ratio, for the percentile line
  explicit Check(std::string nm, double m = 2.0, double x = 0.0) : name(std::move(nm)), mult(m), extra(x) {}
  std::string pct() const {     // p50 / p90 / p99 / max of the per-chat ratios
    if (ratios.size() < 20) return "";
    std::vector<double> r = ratios;
    std::sort(r.begin(), r.end());
    auto at = [&](double f) { return r[std::min(r.size() - 1, (size_t)(f * (r.size() - 1) + 0.5))]; };
    char b[96];
    std::snprintf(b, sizeof b, "  [p50 %.3f p90 %.3f p99 %.3f max %.3f]", at(0.5), at(0.9), at(0.99), r.back());
    return b;
  }
  void add(const Err& e, const std::string& at) {
    ++n;
    nan += e.nan;
    const double bound = mult * e.ef + 2.0 * e.er + extra;
    const double ratio = bound > 0 ? e.ek / bound : (e.ek > 0 ? INFINITY : 0.0);
    ratios.push_back(ratio);
    const bool ok = e.nan == 0 && e.ek <= bound;
    if (!ok) {
      ++fails;
      char b[160];
      std::snprintf(b, sizeof b, "%s (err %.3e, floor %.3e, nan %llu)", at.c_str(), e.ek, e.ef, e.nan);
      where.push_back({at, b, ratio, e.nan});
    }
    if (ratio > worst) { worst = ratio; wk = e.ek; wf = e.ef; }
  }
  // Entry checked against a derived bound instead of the 2x rule (exact answer 0):
  // ratio = max |kernel| / bound, maxk = max |kernel|.
  void add_ratio(double ratio, double maxk, unsigned long long nn, const std::string& at) {
    ++n;
    nan += nn;
    ratios.push_back(ratio);
    const bool ok = nn == 0 && ratio <= 1.0;
    if (!ok) {
      ++fails;
      char b[160];
      std::snprintf(b, sizeof b, "%s (|kernel| %.3e, |kernel|/bound %.3e, nan %llu)", at.c_str(), maxk, ratio, nn);
      where.push_back({at, b, ratio, nn});
    }
    if (ratio > worst) { worst = ratio; wk = maxk; wf = 0; }
  }
  bool ok() const { return fails == 0 && nan == 0; }
};
struct Checks {
  Check O{"O"}, LSE{"LSE"}, dQ{"dQ"}, dK{"dK"}, dV{"dV"};
  std::vector<Check*> all() { return {&O, &LSE, &dQ, &dK, &dV}; }
};
static void print_checks(const std::string& tag, Checks& c, Summary& sum) {
  for (Check* k : c.all()) {
    if (k->n == 0) continue;   // a tensor this group does not check (G8 is forward only)
    int known = 0;
    std::vector<const KnownIssue*> kinds;
    for (const Check::Fail& f : k->where) {
      const KnownIssue* ki = match_known(tag, k->name, f.at, f.ratio, f.nan);
      kinds.push_back(ki);
      if (ki) {
        ++known;
        char b[200];
        std::snprintf(b, sizeof b, "%s %s %s: ratio %.3f (recorded <= %.3f)", tag.c_str(), k->name.c_str(),
                      f.at.c_str(), f.ratio, ki->max_ratio);
        sum.known.push_back(std::string(b) + " -- " + ki->why);
      }
    }
    const bool ok = (k->fails - known) == 0 && k->nan == 0;
    const char* status = (k->fails == 0 && k->nan == 0) ? "PASS" : (ok ? "KNOWN" : "FAIL");
    std::printf("  %-44s %-4s chats %5d  worst ratio %7.3f  (err %.3e vs floor %.3e)  fails %d (known %d)  nan %llu  %s%s\n",
                tag.c_str(), k->name.c_str(), k->n, k->worst, k->wk, k->wf, k->fails, known, k->nan, status,
                k->pct().c_str());
    for (size_t i = 0; i < k->where.size() && i < 20; ++i)
      std::printf("        %s %s\n", kinds[i] ? "known:" : "FAIL: ", k->where[i].detail.c_str());
    sum.record(ok, tag + " " + k->name);
  }
}

// Compare kernel outputs (row bk, token tk0, layout Tk) with reference (row br, token tr0).
static void add_region(Checks& c, const KOut& ko, int bk, int Tk, int tk0, const RefOut& ref, int br, int Tr,
                       int tr0, int L, const Problem& p, const std::string& at) {
  c.O.add(compare_region(ko.O.p, bk, Tk, tk0, ref.O.p, ref.Of.p, br, Tr, tr0, p.Hq, L, p.hd), at);
  c.LSE.add(compare_region(ko.LSE.p, bk, Tk, tk0, ref.LSE.p, ref.LSEf.p, br, Tr, tr0, p.Hq, L, 1), at);
  c.dQ.add(compare_region(ko.dQ.p, bk, Tk, tk0, ref.dQ.p, ref.dQf.p, br, Tr, tr0, p.Hq, L, p.hd), at);
  c.dK.add(compare_region(ko.dK.p, bk, Tk, tk0, ref.dK.p, ref.dKf.p, br, Tr, tr0, p.Hkv, L, p.hd), at);
  c.dV.add(compare_region(ko.dV.p, bk, Tk, tk0, ref.dV.p, ref.dVf.p, br, Tr, tr0, p.Hkv, L, p.hd), at);
}

// ---------------------------------------------------------------------------
// 1-token chats. The exact dQ and dK are 0: the token sees only itself, P = 1, so
// dS = P * (dP - D) = 0. Both references give exactly 0, so the 2x rule allows 0 and
// any fp32 rounding left in the kernel fails it (first H200 run, 2026-10-10: 4.5e-7 to
// 1.3e-6 on D4/D6). Vj1d builds dS from two sums of the same 128 products, added in
// different kernels and orders: D in compute_drowsum_v22 (GQA_bwd_vj1d_sm90.cu:283-306),
// dP in the wgmma, dS = P * (dP - D) (:333-334), P = ex2.approx of the saved LSE
// (:246-256). Its dS is therefore a rounding residue, not 0.
// So for these chats dQ and dK are checked against a derived bound instead (Higham's
// dot-product bound, any summation order; u = 2^-23 also covers truncating tensor-core
// accumulation):
//   |dS_h|   <= 2*gamma_hd * sum_d |dO_hd * v_d|  +  sum_d |dO_hd| * |O_hd - v_d|
//   |dQ_hd|  <= c * scale * |k_d| * |dS_h|
//   |dK_d|   <= c * scale * sum_g |q_gd| * |dS_g|           (sum over the G heads of the group)
// The O term covers a kernel O that is not exactly v. c = 1.02 covers P <= 1 + 2^-10 and the
// bf16 roundings of dS, the per-head dK partials and the outputs (product ~1.013).
// O, LSE and dV of these chats keep the normal 2x rule.
// ---------------------------------------------------------------------------
__global__ void k_maxabs(const double* Tp, int br, int Tr, int tr0, int H, int L, int D, unsigned long long* out) {
  double m = 0;
  const size_t n = (size_t)H * L * D;
  for (size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x; e < n; e += (size_t)gridDim.x * blockDim.x) {
    const int d = (int)(e % D);
    const size_t rr = e / D;
    const int i = (int)(rr % L), h = (int)(rr / L);
    m = fmax(m, fabs(Tp[(((size_t)br * H + h) * Tr + tr0 + i) * D + d]));
  }
  atomicMax(out, (unsigned long long)__double_as_longlong(m));
}
static double maxabs_region(const double* Tp, int br, int Tr, int tr0, int H, int L, int D) {
  static unsigned long long* d = nullptr;
  if (!d) CUDA_CHECK(cudaMalloc(&d, sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemset(d, 0, sizeof(unsigned long long)));
  k_maxabs<<<std::min(grid_for((size_t)H * L * D), 1024), 256>>>(Tp, br, Tr, tr0, H, L, D, d);
  KERNEL_CHECK();
  unsigned long long h;
  CUDA_CHECK(cudaMemcpy(&h, d, sizeof(h), cudaMemcpyDeviceToHost));
  double v;
  std::memcpy(&v, &h, 8);
  return v;
}
// One block per query head, blockDim = hd (a power of two <= 256).
// out[0] = max |dQ|/bound (bits), out[1] = max |dQ| (bits), out[2] = non-finite count.
__global__ void k_one_token_dq(const bf16* v, const bf16* k, const bf16* dO, int Tin, int b, int t, const bf16* Ok,
                               const bf16* dQk, int Tk, int r, int tk0, int Hq, int Hkv, int G, double scale, double c,
                               double gamma, double* dsb, unsigned long long* out) {
  const int h = blockIdx.x, hkv = h / G, d = threadIdx.x, hd = blockDim.x;
  __shared__ double s1[256], s2[256];
  const size_t iq = (((size_t)b * Hq + h) * Tin + t) * hd + d;
  const size_t ikv = (((size_t)b * Hkv + hkv) * Tin + t) * hd + d;
  const size_t io = (((size_t)r * Hq + h) * Tk + tk0) * hd + d;   // kernel outputs: row r, token tk0
  const double vv = (double)__bfloat162float(v[ikv]), g = (double)__bfloat162float(dO[iq]);
  const double oo = (double)__bfloat162float(Ok[io]);
  s1[d] = fabs(g * vv);
  s2[d] = fabs(g) * fabs(oo - vv);
  __syncthreads();
  for (int s = hd / 2; s > 0; s >>= 1) {
    if (d < s) { s1[d] += s1[d + s]; s2[d] += s2[d + s]; }
    __syncthreads();
  }
  const double dsr = 2.0 * gamma * s1[0] + s2[0];
  if (d == 0) dsb[h] = dsr;
  const double bound = c * scale * fabs((double)__bfloat162float(k[ikv])) * dsr;
  const double x = (double)__bfloat162float(dQk[io]);
  if (!isfinite(x)) { atomicAdd(&out[2], 1ull); return; }
  const double ax = fabs(x), ratio = bound > 0 ? ax / bound : (ax > 0 ? INFINITY : 0.0);
  atomicMax(&out[0], (unsigned long long)__double_as_longlong(ratio));
  atomicMax(&out[1], (unsigned long long)__double_as_longlong(ax));
}
// One block per KV head, blockDim = hd. Uses the per-head dS bounds written by k_one_token_dq.
__global__ void k_one_token_dk(const bf16* q, int Tin, int b, int t, const bf16* dKk, int Tk, int r, int tk0, int Hq,
                               int Hkv, int G, double scale, double c, const double* dsb, unsigned long long* out) {
  const int hkv = blockIdx.x, d = threadIdx.x, hd = blockDim.x;
  double acc = 0;
  for (int g = 0; g < G; ++g) {
    const int h = hkv * G + g;
    acc += fabs((double)__bfloat162float(q[(((size_t)b * Hq + h) * Tin + t) * hd + d])) * dsb[h];
  }
  const double bound = c * scale * acc;
  const double x = (double)__bfloat162float(dKk[(((size_t)r * Hkv + hkv) * Tk + tk0) * hd + d]);
  if (!isfinite(x)) { atomicAdd(&out[2], 1ull); return; }
  const double ax = fabs(x), ratio = bound > 0 ? ax / bound : (ax > 0 ? INFINITY : 0.0);
  atomicMax(&out[0], (unsigned long long)__double_as_longlong(ratio));
  atomicMax(&out[1], (unsigned long long)__double_as_longlong(ax));
}

struct OneTok {
  Check q{"dQ"}, k{"dK"};
  int n = 0, not_zero = 0;   // not_zero: chats whose truth was not exactly 0 (they fall back to the 2x rule)
};
// Kernel outputs: row r, token tk0 of a batch with row length Tk (tk0 = 0 for a chat run alone,
// tk0 = s.t0 for a chat inside a packed row, G9). Inputs/reference: segment s of `in`.
static void one_token_region(Checks& c, OneTok& ot, const KOut& ko, int r, int Tk, const Inputs& in,
                             const RefOut& ref, const Segment& s, const std::string& at, int tk0 = 0) {
  const Problem& p = in.p;
  c.O.add(compare_region(ko.O.p, r, Tk, tk0, ref.O.p, ref.Of.p, s.b, p.T, s.t0, p.Hq, 1, p.hd), at);
  c.LSE.add(compare_region(ko.LSE.p, r, Tk, tk0, ref.LSE.p, ref.LSEf.p, s.b, p.T, s.t0, p.Hq, 1, 1), at);
  c.dV.add(compare_region(ko.dV.p, r, Tk, tk0, ref.dV.p, ref.dVf.p, s.b, p.T, s.t0, p.Hkv, 1, p.hd), at);
  ++ot.n;
  if (maxabs_region(ref.dQ.p, s.b, p.T, s.t0, p.Hq, 1, p.hd) != 0.0 ||
      maxabs_region(ref.dK.p, s.b, p.T, s.t0, p.Hkv, 1, p.hd) != 0.0) {
    ++ot.not_zero;   // not the exact-zero case after all: the normal rule decides
    c.dQ.add(compare_region(ko.dQ.p, r, Tk, tk0, ref.dQ.p, ref.dQf.p, s.b, p.T, s.t0, p.Hq, 1, p.hd), at);
    c.dK.add(compare_region(ko.dK.p, r, Tk, tk0, ref.dK.p, ref.dKf.p, s.b, p.T, s.t0, p.Hkv, 1, p.hd), at);
    return;
  }
  static double* dsb = nullptr;
  static unsigned long long* d = nullptr;
  if (!dsb) {
    CUDA_CHECK(cudaMalloc(&dsb, 256 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d, 6 * sizeof(unsigned long long)));
  }
  CUDA_CHECK(cudaMemset(d, 0, 6 * sizeof(unsigned long long)));
  const double u = 0x1.0p-23, gamma = p.hd * u / (1.0 - p.hd * u), cfac = 1.02;
  const double scale = (double)(1.0f / std::sqrt((float)p.hd));   // the fp32 scale the launcher uses
  k_one_token_dq<<<p.Hq, p.hd>>>(in.v.p, in.k.p, in.dO.p, p.T, s.b, s.t0, ko.O.p, ko.dQ.p, Tk, r, tk0, p.Hq, p.Hkv,
                                 p.G(), scale, cfac, gamma, dsb, d);
  k_one_token_dk<<<p.Hkv, p.hd>>>(in.q.p, p.T, s.b, s.t0, ko.dK.p, Tk, r, tk0, p.Hq, p.Hkv, p.G(), scale, cfac, dsb,
                                  d + 3);
  KERNEL_CHECK();
  unsigned long long h[6];
  CUDA_CHECK(cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost));
  double rq, mq, rk, mk;
  std::memcpy(&rq, &h[0], 8); std::memcpy(&mq, &h[1], 8);
  std::memcpy(&rk, &h[3], 8); std::memcpy(&mk, &h[4], 8);
  ot.q.add_ratio(rq, mq, h[2], at);
  ot.k.add_ratio(rk, mk, h[5], at);
}

static double secs_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}
static int env_int(const char* n, int dflt) {
  const char* e = std::getenv(n);
  return (e && *e) ? std::atoi(e) : dflt;
}

// ===========================================================================
// G1: reference self-test (GPU per-segment references vs CPU full-row masked loop)
// ===========================================================================
static void g1_selftest(Summary& sum) {
  std::printf("\n===== G1 reference self-test: GPU per-segment vs CPU full-row explicit mask =====\n");
  struct Case { Problem p; Layout lo; };
  std::vector<Case> cases = {
      {{4, 320, 4, 2, 128}, {"GQA G=2, T=320", 320, {{320}, {100, 1, 57, 130}, {0, 50, 0, 270}, {}}}},
      {{2, 192, 3, 1, 128}, {"MQA G=3, T=192", 192, {{1, 190, 1}, {64, 64, 63}}}},
  };
  for (Case& c : cases) {
    const Problem& p = c.p;
    const auto t0 = std::chrono::steady_clock::now();
    Inputs in;
    make_inputs(in, p, 1234 + p.T);
    // dO stays random on pad rows here: the self-test exercises the math of every segment.
    const std::vector<Segment> segs = packed_segments(c.lo);
    RefOut ref;
    reference(in, segs, ref);
    std::vector<int> seg_id((size_t)p.B * p.T, -1);
    for (size_t s = 0; s < segs.size(); ++s)
      for (int t = 0; t < segs[s].L; ++t) seg_id[(size_t)segs[s].b * p.T + segs[s].t0 + t] = (int)s;
    auto tofloat = [](const DBuf<bf16>& x) {
      std::vector<bf16> h = x.to_host();
      std::vector<float> f(h.size());
      for (size_t i = 0; i < h.size(); ++i) f[i] = __bfloat162float(h[i]);
      return f;
    };
    const std::vector<float> q = tofloat(in.q), k = tofloat(in.k), v = tofloat(in.v), dO = tofloat(in.dO);
    const double scale = 1.0 / std::sqrt((double)p.hd);
    const float scale_f = (float)scale;
    std::vector<double> O, LSE, dQ, dK, dV;
    std::vector<float> Of, LSEf, dQf, dKf, dVf;
    cpu_reference<double, false>(p, seg_id, q, k, v, dO, scale, scale_f, O, LSE, dQ, dK, dV);
    cpu_reference<float, true>(p, seg_id, q, k, v, dO, 1.0f, scale_f, Of, LSEf, dQf, dKf, dVf);
    auto cmp_truth = [&](const char* nm, const DBuf<double>& g, const std::vector<double>& h) {
      const std::vector<double> gh = g.to_host();
      double md = 0, mx = 0;
      size_t nan = 0;
      for (size_t i = 0; i < h.size(); ++i) {
        if (!std::isfinite(gh[i])) { ++nan; continue; }
        md = std::max(md, std::fabs(gh[i] - h[i]));
        mx = std::max(mx, std::fabs(h[i]));
      }
      const double rel = mx > 0 ? md / mx : md;
      const bool ok = nan == 0 && rel <= 1e-12;
      std::printf("  %-18s truth %-4s max|gpu-cpu|/max|cpu| = %.3e   uncovered %zu   %s\n", c.lo.name.c_str(), nm,
                  rel, nan, ok ? "PASS" : "FAIL");
      sum.record(ok, "G1 " + c.lo.name + " truth " + nm);
    };
    auto cmp_floor = [&](const char* nm, const DBuf<float>& g, const std::vector<float>& h, bool is_lse) {
      const std::vector<float> gh = g.to_host();
      size_t mism = 0, nan = 0;
      double worst = 0;   // in units of 1 bf16 ulp of the CPU value (LSE: relative)
      for (size_t i = 0; i < h.size(); ++i) {
        if (!std::isfinite(gh[i])) { ++nan; continue; }
        if (gh[i] == h[i]) continue;
        ++mism;
        const double a = std::max(std::fabs((double)h[i]), 1e-30);
        int ex = 0;
        std::frexp(a, &ex);   // a in [2^(ex-1), 2^ex); bf16 keeps 8 significant bits -> ulp = 2^(ex-8)
        const double unit = is_lse ? a : std::ldexp(1.0, ex - 8);
        worst = std::max(worst, std::fabs((double)gh[i] - (double)h[i]) / unit);
      }
      const bool ok = nan == 0 && (is_lse ? worst <= 1e-6 : worst <= 1.0);
      std::printf("  %-18s floor %-4s mismatches %zu / %zu  worst %s %.3e   uncovered %zu   %s\n",
                  c.lo.name.c_str(), nm, mism, h.size(), is_lse ? "rel" : "bf16-ulp", worst, nan,
                  ok ? "PASS" : "FAIL");
      sum.record(ok, "G1 " + c.lo.name + " floor " + nm);
    };
    cmp_truth("O", ref.O, O); cmp_truth("LSE", ref.LSE, LSE); cmp_truth("dQ", ref.dQ, dQ);
    cmp_truth("dK", ref.dK, dK); cmp_truth("dV", ref.dV, dV);
    cmp_floor("O", ref.Of, Of, false); cmp_floor("LSE", ref.LSEf, LSEf, true); cmp_floor("dQ", ref.dQf, dQf, false);
    cmp_floor("dK", ref.dKf, dKf, false); cmp_floor("dV", ref.dVf, dVf, false);
    std::printf("  (%.1f s)\n", secs_since(t0));
  }
}

// ===========================================================================
// G2: dense rows (one chat = the whole row), today's kernels
// ===========================================================================
struct Heads { int Hq, Hkv; };
static const std::vector<Heads> kHeads = {{18, 6}, {12, 4}, {8, 8}, {32, 8}, {8, 1}};

static void g2_dense(Summary& sum) {
  std::printf("\n===== G2 dense rows: today's kernels vs per-row reference (outputs NaN-filled first) =====\n");
  std::vector<int> Ts = {256, 1024, 4096}, Bs = {1, 2, 8};
  std::vector<Heads> hs = kHeads;
  if (g_quick) { Ts = {256, 4096}; Bs = {2}; hs = {{18, 6}, {8, 1}}; }
  for (int T : Ts)
    for (const Heads& h : hs)
      for (int B : Bs) {
        const Problem p{B, T, h.Hq, h.Hkv, 128};
        const auto t0 = std::chrono::steady_clock::now();
        Inputs in;
        make_inputs(in, p, 1000003ull * T + 1009ull * h.Hq + 17ull * h.Hkv + B);
        const std::vector<Segment> segs = dense_segments(B, T);
        RefOut ref;
        reference(in, segs, ref);
        KOut ko;
        std::string err;
        char tag[96];
        std::snprintf(tag, sizeof tag, "G2 B=%d T=%d %d/%d", B, T, h.Hq, h.Hkv);
        if (!run_today(in, ko, err)) {
          std::printf("  %s  kernel threw: %s  FAIL\n", tag, err.c_str());
          sum.record(false, std::string(tag) + " threw: " + err);
          continue;
        }
        const unsigned long long nf = ko.nonfinite_all();
        Checks c;
        for (const Segment& s : segs)
          add_region(c, ko, s.b, T, s.t0, ref, s.b, T, s.t0, s.L, p, "row " + std::to_string(s.b));
        print_checks(tag, c, sum);
        std::printf("  %-44s non-finite/unwritten outputs: %llu  %s  (%.1f s)\n", tag, nf, nf ? "FAIL" : "PASS",
                    secs_since(t0));
        sum.record(nf == 0, std::string(tag) + " all outputs written + finite");
      }
}

// ===========================================================================
// Real stage-0 chat lengths (D1, D10)
// ===========================================================================
static bool read_stage0(std::vector<int>& L, std::string& why) {
  const char* env = std::getenv("VARLEN_STAGE0_LABELS");
  const std::string path = env ? env : "/home/blu-bridge016/Downloads/sft_data/stage0_ultrachat_5k_T4096/train/labels.bin";
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) { why = "cannot open " + path + " (set VARLEN_STAGE0_LABELS)"; return false; }
  std::fseek(f, 0, SEEK_END);
  const long bytes = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  const int T = 4096;
  if (bytes <= 0 || bytes % (T * 2) != 0) { std::fclose(f); why = path + ": size is not N x 4096 x uint16"; return false; }
  std::vector<uint16_t> lab((size_t)bytes / 2);
  const size_t got = std::fread(lab.data(), 2, lab.size(), f);
  std::fclose(f);
  if (got != lab.size()) { why = path + ": short read"; return false; }
  const size_t N = lab.size() / T;
  L.assign(N, 0);
  for (size_t r = 0; r < N; ++r)   // chat length = last trained label index + 1 (pack_sim.py)
    for (int t = T - 1; t >= 0; --t)
      if (lab[r * T + t] != 65535) { L[r] = t + 1; break; }
  return true;
}
// First-fit decreasing, the rule of pack_sim.py (it sorts descending, then first fit).
static std::vector<std::vector<int>> pack_ffd(std::vector<int> L, int T) {
  std::sort(L.begin(), L.end(), std::greater<int>());
  std::vector<std::vector<int>> rows;
  std::vector<int> used;
  for (int x : L) {
    size_t r = 0;
    while (r < rows.size() && used[r] + x > T) ++r;
    if (r == rows.size()) { rows.push_back({}); used.push_back(0); }
    rows[r].push_back(x);
    used[r] += x;
  }
  return rows;
}

// ===========================================================================
// G3: v1 SFT rows (real chat + pad tail), today's kernels
// ===========================================================================
static void g3_v1_rows(const std::vector<int>& Lreal, Summary& sum) {
  std::printf("\n===== G3 v1 SFT rows: real chat + pad (dO = 0 on pads), 18/6 =====\n");
  const int T = 4096, nrows = g_quick ? 8 : 16;
  for (int r0 = 0; r0 < nrows; r0 += 8) {
    Layout lo{"v1", T, {}};
    for (int r = r0; r < r0 + 8; ++r) lo.rows.push_back({Lreal[r]});
    const Problem p{8, T, 18, 6, 128};
    Inputs in;
    make_inputs(in, p, 77 + r0);
    const std::vector<Segment> pk = packed_segments(lo);
    zero_pad_grads(in, pk);
    KOut ko;
    std::string err;
    const std::string tag = "G3 v1 rows " + std::to_string(r0) + "-" + std::to_string(r0 + 7);
    if (!run_today(in, ko, err)) { sum.record(false, tag + " threw: " + err); continue; }
    RefOut dense_ref, packed_ref;
    const std::vector<Segment> dn = dense_segments(8, T);
    reference(in, dn, dense_ref);
    reference(in, pk, packed_ref);
    Checks a, b;
    for (const Segment& s : dn) add_region(a, ko, s.b, T, 0, dense_ref, s.b, T, 0, T, p, "row " + std::to_string(s.b));
    for (const Segment& s : pk)
      if (!s.pad) add_region(b, ko, s.b, T, s.t0, packed_ref, s.b, T, s.t0, s.L, p, "chat row " + std::to_string(s.b));
    print_checks(tag + " (a) whole rows", a, sum);
    print_checks(tag + " (b) chat rows = per-chat", b, sum);
    const unsigned long long nf = ko.nonfinite_all();
    sum.record(nf == 0, tag + " all outputs written + finite");
  }
}

// ===========================================================================
// G4: repeat (race) test
// ===========================================================================
// Mirrors gqa_bwd_mode() in GQA_bwd_generic.cu. USE_VJ2=1 (the FA3-structured Vj2, added 2026-10-12) is new: before
// it, a Vj2 run would have been labelled "Vj1d" and checked as Vj1d.
static std::string bwd_mode_name() {
  auto on = [](const char* n) { const char* e = std::getenv(n); return e && e[0] == '1'; };
  const bool f = on("USE_FLASH"), d = on("USE_DETERMINISTIC_DQ"), v = on("USE_VJ2");
  if ((int)f + (int)d + (int)v > 1) return "Vj1d (more than one of USE_FLASH / USE_DETERMINISTIC_DQ / USE_VJ2 set -> dispatcher falls back to Vj1d)";
  if (f) return "Vj1 (USE_FLASH=1)";
  if (d) return "v44 (USE_DETERMINISTIC_DQ=1)";
  if (v) return "Vj2 (USE_VJ2=1)";
  return "Vj1d (default)";
}
static void g4_repeat(Summary& sum) {
  std::printf("\n===== G4 repeat (race) test: O, LSE, dK, dV bitwise; dQ within FA's atol =====\n");
  // dK/dV are checked bitwise when the kernel is meant to give them bit-identical every run: Vj1d and Vj2 (both use
  // one-writer per-head partials + the fixed-order head sum). Before Vj2 this was Vj1d only.
  const bool vj1d = bwd_mode_name().rfind("Vj1d", 0) == 0 || bwd_mode_name().rfind("Vj2", 0) == 0;
  struct RC { Problem p; int reps; };
  const int r1 = env_int("VARLEN_REPEATS", g_quick ? 50 : 1000), r2 = env_int("VARLEN_REPEATS", g_quick ? 20 : 250);
  const std::vector<RC> cs = {{{8, 4096, 18, 6, 128}, r1}, {{60, 1024, 18, 6, 128}, r2}};
  for (const RC& c : cs) {
    const Problem& p = c.p;
    const auto t0 = std::chrono::steady_clock::now();
    Inputs in;
    make_inputs(in, p, 4242 + p.B);
    KOut base, cur;
    std::string err;
    char tag[96];
    std::snprintf(tag, sizeof tag, "G4 B=%d T=%d 18/6 x%d", p.B, p.T, c.reps);
    if (!run_today(in, base, err)) { sum.record(false, std::string(tag) + " threw: " + err); continue; }
    const double atol = dq_atol(base.dQ.p, 0, p.T, 0, (int)(p.nQ() / ((size_t)p.T * p.hd)), p.T, p.hd);
    // FA's atol computed on dK/dV themselves; before 2026-10-10 the non-Vj1d branch used the dQ atol, which
    // is too tight for dK/dV (larger values: one bf16 step at |x| in [4,8) is 0.031 > the dQ atol 0.0156).
    const int HKB0 = (int)(p.nKV() / ((size_t)p.T * p.hd));
    const double atol_kv = std::max(dq_atol(base.dK.p, 0, p.T, 0, HKB0, p.T, p.hd), dq_atol(base.dV.p, 0, p.T, 0, HKB0, p.T, p.hd));
    unsigned long long mO = 0, mL = 0, mK = 0, mV = 0, mQ = 0;
    double worst_dq = 0, worst_kv = 0;
    for (int r = 0; r < c.reps; ++r) {
      if (!run_today(in, cur, err)) { sum.record(false, std::string(tag) + " threw: " + err); break; }
      const int HB = p.B * p.Hq, HKB = p.B * p.Hkv;   // treat [B,H] as one axis
      mO += bitdiff(base.O.p, 0, p.T, 0, cur.O.p, 0, p.T, 0, HB, p.T, p.hd).mism;
      mL += bitdiff(base.LSE.p, 0, p.T, 0, cur.LSE.p, 0, p.T, 0, HB, p.T, 1).mism;
      const BitDiff k = bitdiff(base.dK.p, 0, p.T, 0, cur.dK.p, 0, p.T, 0, HKB, p.T, p.hd);
      const BitDiff v = bitdiff(base.dV.p, 0, p.T, 0, cur.dV.p, 0, p.T, 0, HKB, p.T, p.hd);
      mK += k.mism; mV += v.mism;
      worst_kv = std::max({worst_kv, k.maxabs, v.maxabs});
      const BitDiff q = bitdiff(base.dQ.p, 0, p.T, 0, cur.dQ.p, 0, p.T, 0, HB, p.T, p.hd);
      mQ += q.mism;
      worst_dq = std::max(worst_dq, q.maxabs);
    }
    const bool okO = mO == 0 && mL == 0;
    const bool okKV = vj1d ? (mK == 0 && mV == 0) : worst_kv <= atol_kv;
    const bool okQ = worst_dq <= atol;
    std::printf("  %-34s O/LSE mismatched elements %llu/%llu %s | dK/dV %llu/%llu (max %.3e) %s | dQ max diff %.3e vs atol %.3e (%llu elems differ) %s  (%.1f s)\n",
                tag, mO, mL, okO ? "PASS" : "FAIL", mK, mV, worst_kv,
                okKV ? "PASS" : "FAIL", worst_dq, atol, mQ, okQ ? "PASS" : "FAIL", secs_since(t0));
    if (!vj1d) std::printf("        note: backward mode is %s -- dK/dV gated by FA's atol on dK/dV (%.3e), not bitwise\n", bwd_mode_name().c_str(), atol_kv);
    sum.record(okO, std::string(tag) + " O/LSE bitwise");
    sum.record(okKV, std::string(tag) + " dK/dV");
    sum.record(okQ, std::string(tag) + " dQ");
  }
}

// ===========================================================================
// G5: large inputs (FA2 test_flash_attn_bwd_overflow: q*5, k*3, v*3)
// ===========================================================================
static void g5_large(Summary& sum) {
  std::printf("\n===== G5 large inputs: q*5, k*3, v*3 (grads gated 5x + 1e-3 as in FA2; 2x printed too) =====\n");
  for (int T : {256, 512})
    for (const Heads& h : std::vector<Heads>{{18, 6}, {8, 1}}) {
      const Problem p{2, T, h.Hq, h.Hkv, 128};
      Inputs in;
      make_inputs(in, p, 5150 + T + h.Hq, 5.f, 3.f, 3.f);
      const std::vector<Segment> segs = dense_segments(2, T);
      RefOut ref;
      reference(in, segs, ref);
      KOut ko;
      std::string err;
      char tag[96];
      std::snprintf(tag, sizeof tag, "G5 B=2 T=%d %d/%d", T, h.Hq, h.Hkv);
      if (!run_today(in, ko, err)) { sum.record(false, std::string(tag) + " threw: " + err); continue; }
      Checks gate;   // FA2's gate for this test
      gate.dQ = Check("dQ", 5.0, 1e-3); gate.dK = Check("dK", 5.0, 1e-3); gate.dV = Check("dV", 5.0, 1e-3);
      Checks strict;
      for (const Segment& s : segs) {
        add_region(gate, ko, s.b, T, 0, ref, s.b, T, 0, T, p, "row " + std::to_string(s.b));
        add_region(strict, ko, s.b, T, 0, ref, s.b, T, 0, T, p, "row " + std::to_string(s.b));
      }
      print_checks(std::string(tag) + " [gate]", gate, sum);
      for (Check* k : strict.all())
        std::printf("  %-44s %-4s 2x-rule worst ratio %.3f (info)\n", tag, k->name.c_str(), k->worst);
      const unsigned long long nf = ko.nonfinite_all();
      sum.record(nf == 0, std::string(tag) + " all outputs finite");
    }
}

// ===========================================================================
// G6: every chat ALONE through today's kernel vs the per-chat reference of the
// packed layout. G6c: alone at T=4096 must be bitwise equal to alone at roundup(L,256).
// ===========================================================================
static Layout make_D7() {
  Layout lo{"D7 63/65/127/129/255/257", 4096, {{}}};
  const int pat[6] = {63, 65, 127, 129, 255, 257};
  int t = 0;
  for (int k = 0;; ++k) {
    const int L = pat[k % 6];
    if (t + L > 4096) break;
    lo.rows[0].push_back(L);
    t += L;
  }
  return lo;
}
static Layout make_D8() {
  Layout lo{"D8 random + pad tails", 4096, {}};
  std::mt19937 rng(8888);
  std::uniform_int_distribution<int> U(1, 1500);
  for (int pad : {0, 1, 16, 63, 64, 65, 255}) {
    std::vector<int> row;
    int rem = 4096 - pad;
    while (rem > 0) {
      int L = std::min(U(rng), rem);
      row.push_back(L);
      rem -= L;
    }
    lo.rows.push_back(row);
  }
  return lo;
}

// ---------------------------------------------------------------------------
// Diagnostic of one chat's dQ error (VARLEN_DIAG_ROW / VARLEN_DIAG_T0).
//  (a) re-run the chat g_diag_rep times (dQ is reduced with atomics, so it varies run to run)
//  (b) the worst dQ element of the first run: kernel, truth, floor, bf16 ulp
//  (c) test of one mechanism at that row. Vj1d (like FlashAttention) uses
//      D = rowsum(dO * O) with the bf16-rounded O (compute_drowsum_v22,
//      GQA_bwd_vj1d_sm90.cu:283-306), while the exact D is sum_j P_ij dP_ij. The rounding
//      dD = dO . (O_bf16 - O_exact) enters dS_ij = P_ij (dP_ij - D) for every key of the
//      row in the same direction, so it adds the coherent error  -scale * dD * sum_j P_ij K_j
//      to dQ_i. If that is the excess, e = dQ_kernel - dQ_truth lines up with that vector.
// ---------------------------------------------------------------------------
template <typename X>
static std::vector<X> d2h(const X* base, size_t off, size_t n) {
  std::vector<X> h(n);
  CUDA_CHECK(cudaMemcpy(h.data(), base + off, n * sizeof(X), cudaMemcpyDeviceToHost));
  return h;
}
static double bf16_ulp(double a) {
  int ex = 0;
  std::frexp(std::max(std::fabs(a), 1e-30), &ex);
  return std::ldexp(1.0, ex - 8);
}
// tk0: token where the chat starts inside the kernel output (0 for a chat run alone, s.t0 in a packed row).
static void diag_chat(const Inputs& in, const RefOut& ref, const KOut& ko, int r, int Ta, const Segment& s,
                      const std::function<bool(uint64_t, Inputs&, KOut&)>& rerun, int tk0 = 0) {
  const Problem& p = in.p;
  const int L = s.L, hd = p.hd, Hq = p.Hq, Hkv = p.Hkv, G = p.G(), T = p.T;
  std::printf("\n  ---- DIAG dQ: row %d, chat at %d, L=%d (batch row %d; kernel output row %d, token %d, row length %d) ----\n",
              g_diag_row, s.t0, L, s.b, r, tk0, Ta);
  // (a) repeats
  std::vector<double> rr;
  int over = 0;
  for (int k = 0; k < g_diag_rep; ++k) {
    Inputs ai;
    KOut kk;
    if (!rerun(777 + k, ai, kk)) { std::printf("  rerun %d threw\n", k); break; }
    const Err e = compare_region(kk.dQ.p, r, Ta, tk0, ref.dQ.p, ref.dQf.p, s.b, T, s.t0, Hq, L, hd);
    const double ratio = e.ek / (2.0 * e.ef + 2.0 * e.er);
    rr.push_back(ratio);
    if (ratio > 1.0) ++over;
  }
  if (!rr.empty()) {
    std::vector<double> q = rr;
    std::sort(q.begin(), q.end());
    std::printf("  (a) %zu re-runs of this chat: dQ ratio min %.3f  median %.3f  max %.3f  -> %d of %zu above 1\n",
                q.size(), q.front(), q[q.size() / 2], q.back(), over, q.size());
  }
  // (b) worst element of the first run
  int wh = 0, wi = 0, wd = 0;
  double we = -1, wt = 0, wk = 0, wf = 0, fe_max = 0;
  for (int h = 0; h < Hq; ++h) {
    const std::vector<bf16> k = d2h(ko.dQ.p, (((size_t)r * Hq + h) * Ta + tk0) * hd, (size_t)L * hd);
    const std::vector<double> t = d2h(ref.dQ.p, (((size_t)s.b * Hq + h) * T + s.t0) * hd, (size_t)L * hd);
    const std::vector<float> f = d2h(ref.dQf.p, (((size_t)s.b * Hq + h) * T + s.t0) * hd, (size_t)L * hd);
    for (size_t e = 0; e < t.size(); ++e) {
      const double kv = __bfloat162float(k[e]);
      const double err = std::fabs(kv - t[e]);
      fe_max = std::max(fe_max, std::fabs((double)f[e] - t[e]));
      if (err > we) { we = err; wh = h; wi = (int)(e / hd); wd = (int)(e % hd); wt = t[e]; wk = kv; wf = f[e]; }
    }
  }
  std::printf("  (b) worst dQ element: head %d, chat token %d, dim %d: kernel %.6f  truth %.6f  floor %.6f\n", wh, wi,
              wd, wk, wt, wf);
  std::printf("      |kernel-truth| %.3e = %.2f bf16 ulp of the truth;  |floor-truth| there %.3e;  floor max over the chat %.3e\n",
              we, we / bf16_ulp(wt), std::fabs(wf - wt), fe_max);
  // (c) the coherent-D test at that row
  const int hkv = wh / G;
  const std::vector<bf16> qr = d2h(in.q.p, (((size_t)s.b * Hq + wh) * T + s.t0 + wi) * hd, hd);
  const std::vector<bf16> Kr = d2h(in.k.p, (((size_t)s.b * Hkv + hkv) * T + s.t0) * hd, (size_t)(wi + 1) * hd);
  const std::vector<bf16> gr = d2h(in.dO.p, (((size_t)s.b * Hq + wh) * T + s.t0 + wi) * hd, hd);
  const std::vector<bf16> Ok = d2h(ko.O.p, (((size_t)r * Hq + wh) * Ta + tk0 + wi) * hd, hd);
  const std::vector<double> Ot = d2h(ref.O.p, (((size_t)s.b * Hq + wh) * T + s.t0 + wi) * hd, hd);
  const std::vector<bf16> Qk = d2h(ko.dQ.p, (((size_t)r * Hq + wh) * Ta + tk0 + wi) * hd, hd);
  const std::vector<double> Qt = d2h(ref.dQ.p, (((size_t)s.b * Hq + wh) * T + s.t0 + wi) * hd, hd);
  const std::vector<float> Qf = d2h(ref.dQf.p, (((size_t)s.b * Hq + wh) * T + s.t0 + wi) * hd, hd);
  const double scale = 1.0 / std::sqrt((double)hd);
  std::vector<double> P(wi + 1), v(hd, 0.0);
  double m = -INFINITY, Z = 0;
  for (int j = 0; j <= wi; ++j) {
    double a = 0;
    for (int d = 0; d < hd; ++d) a += (double)__bfloat162float(qr[d]) * (double)__bfloat162float(Kr[(size_t)j * hd + d]);
    P[j] = a * scale;
    m = std::max(m, P[j]);
  }
  for (int j = 0; j <= wi; ++j) { P[j] = std::exp(P[j] - m); Z += P[j]; }
  for (int j = 0; j <= wi; ++j) {
    P[j] /= Z;
    for (int d = 0; d < hd; ++d) v[d] += P[j] * (double)__bfloat162float(Kr[(size_t)j * hd + d]);
  }
  double dD = 0;
  for (int d = 0; d < hd; ++d) dD += (double)__bfloat162float(gr[d]) * ((double)__bfloat162float(Ok[d]) - Ot[d]);
  double ee = 0, hh = 0, eh = 0, ev = 0, vv = 0, emax = 0, hmax = 0, rmax = 0, fmax_row = 0;
  for (int d = 0; d < hd; ++d) {
    const double e = (double)__bfloat162float(Qk[d]) - Qt[d], pr = -scale * dD * v[d];
    ee += e * e; hh += pr * pr; eh += e * pr; ev += e * v[d]; vv += v[d] * v[d];
    emax = std::max(emax, std::fabs(e)); hmax = std::max(hmax, std::fabs(pr));
    rmax = std::max(rmax, std::fabs(e - pr)); fmax_row = std::max(fmax_row, std::fabs((double)Qf[d] - Qt[d]));
  }
  const double cosv = (ee > 0 && hh > 0) ? eh / std::sqrt(ee * hh) : 0.0;
  const double dD_fit = vv > 0 ? -ev / (scale * vv) : 0.0;
  std::printf("  (c) row test at head %d token %d (keys 0..%d): predicted dD = dO.(O_bf16 - O_exact) = %.4e, fitted from the error = %.4e\n",
              wh, wi, wi, dD, dD_fit);
  std::printf("      kernel error on this row: max %.3e;  predicted coherent error max %.3e;  cosine(error, prediction) %.4f\n",
              emax, hmax, cosv);
  std::printf("      after removing the prediction: max %.3e   (floor's own max error on this row %.3e)\n", rmax, fmax_row);
  // (d) the backward's dS arithmetic on this row, modelled three ways. The kernel's P is
  // exp(scale*q.k - LSE) with the kernel's own LSE (ex2 of the same exponent, :246-256).
  //   model 0 = Vj1d: dS = bf16( bf16(P) * (dP - D_O) )   (fuse_dS_ldstsm, :311-341: P read back
  //             as bf16 from sP), D_O = dO . O_kernel (compute_drowsum_v22)
  //   model 1 = FA3:  dS = bf16( P * (dP - D_O) ), P in fp32 (hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp:893;
  //             P is rounded to bf16 only afterwards, :897-898)
  //   model 2 = fp32 P with the consistent row term D_P = sum_j P_j dP_j
  // dQ = bf16( scale * sum_j dS_j K_j )  (scale applied when dQ is stored, :514).
  const std::vector<bf16> Vr = d2h(in.v.p, (((size_t)s.b * Hkv + hkv) * T + s.t0) * hd, (size_t)(wi + 1) * hd);
  const std::vector<float> Lk = d2h(ko.LSE.p, ((size_t)r * Hq + wh) * Ta + tk0 + wi, 1);
  const float scale_f = 1.0f / std::sqrt((float)hd);
  std::vector<double> sc(wi + 1), dPr(wi + 1), Pk(wi + 1);
  double D_O = 0, D_P = 0;
  for (int d = 0; d < hd; ++d) D_O += (double)__bfloat162float(gr[d]) * (double)__bfloat162float(Ok[d]);
  for (int j = 0; j <= wi; ++j) {
    double a = 0, b2 = 0;
    for (int d = 0; d < hd; ++d) {
      a += (double)__bfloat162float(qr[d]) * (double)__bfloat162float(Kr[(size_t)j * hd + d]);
      b2 += (double)__bfloat162float(gr[d]) * (double)__bfloat162float(Vr[(size_t)j * hd + d]);
    }
    sc[j] = a * (double)scale_f;
    dPr[j] = b2;
    Pk[j] = std::exp(sc[j] - (double)Lk[0]);
    D_P += Pk[j] * dPr[j];
  }
  auto model = [&](int mode) {
    std::vector<double> acc(hd, 0.0), out(hd);
    for (int j = 0; j <= wi; ++j) {
      const double Pu = (mode == 0) ? (double)bf16r((float)Pk[j]) : Pk[j];
      const double Dm = (mode == 2) ? D_P : D_O;
      const double dS = (double)bf16r((float)(Pu * (dPr[j] - Dm)));
      for (int d = 0; d < hd; ++d) acc[d] += dS * (double)__bfloat162float(Kr[(size_t)j * hd + d]);
    }
    for (int d = 0; d < hd; ++d) out[d] = (double)bf16r((float)((double)scale_f * acc[d]));
    return out;
  };
  const char* names[3] = {"Vj1d   (bf16 P, D from O)", "FA3    (fp32 P, D from O)", "fp32 P + consistent D   "};
  std::printf("  (d) dS arithmetic models on this row (D_O = %.6f, D_P = %.6f, sum_j bf16(P_j) = %.8f):\n", D_O, D_P,
              [&] { double s2 = 0; for (double x : Pk) s2 += (double)bf16r((float)x); return s2; }());
  for (int mode = 0; mode < 3; ++mode) {
    const std::vector<double> mq = model(mode);
    double vs_kernel = 0, vs_truth = 0;
    for (int d = 0; d < hd; ++d) {
      vs_kernel = std::max(vs_kernel, std::fabs(mq[d] - (double)__bfloat162float(Qk[d])));
      vs_truth = std::max(vs_truth, std::fabs(mq[d] - Qt[d]));
    }
    std::printf("      model %d %s: max |model - kernel| %.3e   max |model - truth| %.3e  (%.2f x the floor's row error)\n",
                mode, names[mode], vs_kernel, vs_truth, fmax_row > 0 ? vs_truth / fmax_row : 0.0);
  }
  // (e) export the chat for cross-library runs (VARLEN_EXPORT_DIR). Every tensor is written
  // contiguous as [H, L, D] for this chat only: inputs bf16, our kernel's outputs (bf16; LSE
  // fp32), truth fp64, floor fp32. meta.json gives the shapes. Read by crosslib_chat_check.py.
  if (const char* ex = std::getenv("VARLEN_EXPORT_DIR")) {
    const std::string dir = ex;
    if (std::system(("mkdir -p '" + dir + "'").c_str()) != 0) {
      std::printf("  (e) export: cannot create %s\n", dir.c_str());
    } else {
      auto put = [&](const std::string& name, const void* p, size_t bytes) {
        FILE* f = std::fopen((dir + "/" + name).c_str(), "wb");
        if (!f || std::fwrite(p, 1, bytes, f) != bytes) std::printf("  (e) export: write failed for %s\n", name.c_str());
        if (f) std::fclose(f);
      };
      // gather [H, L, D] for row b, tokens t0..t0+L of a [B, H, Tsrc, D] device array
      auto grab = [&](auto* dev, int H, int Tsrc, int b, int t0, int D, const std::string& name) {
        using X = std::remove_const_t<std::remove_pointer_t<decltype(dev)>>;
        std::vector<X> all;
        for (int h = 0; h < H; ++h) {
          std::vector<X> part = d2h(dev, (((size_t)b * H + h) * Tsrc + t0) * D, (size_t)L * D);
          all.insert(all.end(), part.begin(), part.end());
        }
        put(name, all.data(), all.size() * sizeof(X));
      };
      grab(in.q.p, Hq, T, s.b, s.t0, hd, "q.bf16");
      grab(in.k.p, Hkv, T, s.b, s.t0, hd, "k.bf16");
      grab(in.v.p, Hkv, T, s.b, s.t0, hd, "v.bf16");
      grab(in.dO.p, Hq, T, s.b, s.t0, hd, "dO.bf16");
      grab(ko.O.p, Hq, Ta, r, tk0, hd, "ours_O.bf16");
      grab(ko.LSE.p, Hq, Ta, r, tk0, 1, "ours_LSE.f32");
      grab(ko.dQ.p, Hq, Ta, r, tk0, hd, "ours_dQ.bf16");
      grab(ko.dK.p, Hkv, Ta, r, tk0, hd, "ours_dK.bf16");
      grab(ko.dV.p, Hkv, Ta, r, tk0, hd, "ours_dV.bf16");
      grab(ref.O.p, Hq, T, s.b, s.t0, hd, "truth_O.f64");
      grab(ref.LSE.p, Hq, T, s.b, s.t0, 1, "truth_LSE.f64");
      grab(ref.dQ.p, Hq, T, s.b, s.t0, hd, "truth_dQ.f64");
      grab(ref.dK.p, Hkv, T, s.b, s.t0, hd, "truth_dK.f64");
      grab(ref.dV.p, Hkv, T, s.b, s.t0, hd, "truth_dV.f64");
      grab(ref.Of.p, Hq, T, s.b, s.t0, hd, "floor_O.f32");
      grab(ref.LSEf.p, Hq, T, s.b, s.t0, 1, "floor_LSE.f32");
      grab(ref.dQf.p, Hq, T, s.b, s.t0, hd, "floor_dQ.f32");
      grab(ref.dKf.p, Hkv, T, s.b, s.t0, hd, "floor_dK.f32");
      grab(ref.dVf.p, Hkv, T, s.b, s.t0, hd, "floor_dV.f32");
      char meta[512];
      std::snprintf(meta, sizeof meta,
                    "{\"L\": %d, \"Hq\": %d, \"Hkv\": %d, \"hd\": %d, \"d10_row\": %d, \"t0\": %d, "
                    "\"worst\": {\"head\": %d, \"token\": %d, \"dim\": %d}, \"scale\": %.17g}\n",
                    L, Hq, Hkv, hd, g_diag_row, s.t0, wh, wi, wd, (double)(1.0f / std::sqrt((float)hd)));
      put("meta.json", meta, std::strlen(meta));
      std::printf("  (e) exported this chat to %s\n", dir.c_str());
    }
  }
  std::printf("  ---- end DIAG ----\n\n");
}

static void chat_alone(const Layout& lo, const Heads& h, int bitwise_chats, Summary& sum, uint64_t seed,
                       int row_base = 0) {
  const int T = lo.T;
  Checks c;
  OneTok ot;   // 1-token chats: dQ/dK against the derived bound (see one_token_region)
  unsigned long long nonfin = 0;
  int nchats = 0, bit_done = 0;
  unsigned long long bmis = 0;
  double bq = 0, bq_atol_min = INFINITY;
  bool bq_ok = true;
  std::string err;
  bool threw = false;
  const auto t0 = std::chrono::steady_clock::now();
  for (size_t r0 = 0; r0 < lo.rows.size(); r0 += 8) {
    Layout sub{lo.name, T, {}};
    for (size_t r = r0; r < std::min(lo.rows.size(), r0 + 8); ++r) sub.rows.push_back(lo.rows[r]);
    const Problem p{(int)sub.rows.size(), T, h.Hq, h.Hkv, 128};
    Inputs in;
    make_inputs(in, p, seed + r0);
    const std::vector<Segment> segs = packed_segments(sub);
    zero_pad_grads(in, segs);
    RefOut ref;
    reference(in, segs, ref);
    std::map<int, std::vector<Segment>> by_ta;   // group chats by their alone row length
    for (const Segment& s : segs)
      if (!s.pad) by_ta[std::max(256, (s.L + 255) / 256 * 256)].push_back(s);
    for (auto& kv : by_ta) {
      const int Ta = kv.first;
      const std::vector<Segment>& chats = kv.second;
      for (size_t c0 = 0; c0 < chats.size(); c0 += 8) {
        const int nb = (int)std::min<size_t>(8, chats.size() - c0);
        auto run_alone = [&](int Tal, uint64_t fseed, Inputs& ai, KOut& ko) -> bool {
          const Problem ap{nb, Tal, h.Hq, h.Hkv, 128};
          make_inputs(ai, ap, fseed);   // random filler everywhere first
          for (int r = 0; r < nb; ++r) {
            const Segment& s = chats[c0 + r];
            k_copy_tokens<<<grid_for((size_t)h.Hq * s.L * 128), 256>>>(in.q.p, T, s.b, s.t0, ai.q.p, Tal, r, 0, h.Hq, s.L, 128);
            k_copy_tokens<<<grid_for((size_t)h.Hkv * s.L * 128), 256>>>(in.k.p, T, s.b, s.t0, ai.k.p, Tal, r, 0, h.Hkv, s.L, 128);
            k_copy_tokens<<<grid_for((size_t)h.Hkv * s.L * 128), 256>>>(in.v.p, T, s.b, s.t0, ai.v.p, Tal, r, 0, h.Hkv, s.L, 128);
            k_copy_tokens<<<grid_for((size_t)h.Hq * s.L * 128), 256>>>(in.dO.p, T, s.b, s.t0, ai.dO.p, Tal, r, 0, h.Hq, s.L, 128);
            KERNEL_CHECK();
            zero_tokens(ai.dO, h.Hq, Tal, 128, r, s.L, Tal);   // filler gets dO = 0
          }
          return run_today(ai, ko, err);
        };
        Inputs ai;
        KOut ko;
        if (!run_alone(Ta, seed * 31 + Ta + c0, ai, ko)) { threw = true; break; }
        nonfin += ko.nonfinite_all();
        for (int r = 0; r < nb; ++r) {
          const Segment& s = chats[c0 + r];
          const int abs_row = row_base + (int)r0 + s.b;
          const std::string at =
              "row " + std::to_string(abs_row) + " chat@" + std::to_string(s.t0) + " L=" + std::to_string(s.L);
          if (s.L == 1) one_token_region(c, ot, ko, r, Ta, in, ref, s, at);
          else add_region(c, ko, r, Ta, 0, ref, s.b, T, s.t0, s.L, ai.p, at);
          ++nchats;
          if (abs_row == g_diag_row && s.t0 == g_diag_t0)
            diag_chat(in, ref, ko, r, Ta, s, [&](uint64_t k, Inputs& a2, KOut& k2) {
              return run_alone(Ta, seed * 31 + Ta + c0 + k, a2, k2);
            });
        }
        if (bit_done < bitwise_chats) {   // G6c
          Inputs aw;
          KOut kw;
          if (!run_alone(4096, seed * 37 + 4096 + c0, aw, kw)) { threw = true; break; }
          for (int r = 0; r < nb && bit_done < bitwise_chats; ++r, ++bit_done) {
            const int L = chats[c0 + r].L;
            bmis += bitdiff(ko.O.p, r, Ta, 0, kw.O.p, r, 4096, 0, h.Hq, L, 128).mism;
            bmis += bitdiff(ko.LSE.p, r, Ta, 0, kw.LSE.p, r, 4096, 0, h.Hq, L, 1).mism;
            if (g_kv_deterministic) {
              bmis += bitdiff(ko.dK.p, r, Ta, 0, kw.dK.p, r, 4096, 0, h.Hkv, L, 128).mism;
              bmis += bitdiff(ko.dV.p, r, Ta, 0, kw.dV.p, r, 4096, 0, h.Hkv, L, 128).mism;
            } else {   // Vj1: one bf16 step of atomic-order noise is expected
              for (const DBuf<bf16>* x : {&ko.dK, &ko.dV}) {
                const DBuf<bf16>* y = (x == &ko.dK) ? &kw.dK : &kw.dV;
                const double atk = dq_atol(x->p, r, Ta, 0, h.Hkv, L, 128);
                const BitDiff dd = bitdiff(x->p, r, Ta, 0, y->p, r, 4096, 0, h.Hkv, L, 128);
                if (dd.maxabs > atk) ++bmis;   // counted as a mismatch only beyond the atol
              }
            }
            const double at = dq_atol(ko.dQ.p, r, Ta, 0, h.Hq, L, 128);
            const BitDiff dq = bitdiff(ko.dQ.p, r, Ta, 0, kw.dQ.p, r, 4096, 0, h.Hq, L, 128);
            bq = std::max(bq, dq.maxabs);
            bq_atol_min = std::min(bq_atol_min, at);
            if (dq.maxabs > at) bq_ok = false;
          }
        }
      }
      if (threw) break;
    }
    if (threw) break;
  }
  char tag[120];
  std::snprintf(tag, sizeof tag, "G6 %s %d/%d", lo.name.c_str(), h.Hq, h.Hkv);
  if (threw) {
    std::printf("  %s  kernel threw: %s  FAIL\n", tag, err.c_str());
    sum.record(false, std::string(tag) + " threw: " + err);
    return;
  }
  if (nchats == 0) {
    std::printf("  %-44s no chats (all-pad rows): nothing to run alone; the kernels see this case in Steps 3-5\n", tag);
    return;
  }
  print_checks(tag, c, sum);
  if (ot.n > 0) {
    std::printf("  %-44s 1-token chats: %d (exact dQ = dK = 0; truth exactly 0 for %d of them, the rest use the 2x rule)\n",
                tag, ot.n, ot.n - ot.not_zero);
    for (Check* k : {&ot.q, &ot.k}) {
      if (k->n == 0) continue;
      std::printf("  %-44s %-4s 1-token  chats %5d  worst |kernel|/bound %.3e  (max |kernel| %.3e)  fails %d  nan %llu  %s\n",
                  tag, k->name.c_str(), k->n, k->worst, k->wk, k->fails, k->nan, k->ok() ? "PASS" : "FAIL");
      for (size_t i = 0; i < k->where.size() && i < 20; ++i)
        std::printf("        FAIL:  %s\n", k->where[i].detail.c_str());
      sum.record(k->ok(), std::string(tag) + " " + k->name + " 1-token bound");
    }
  }
  std::printf("  %-44s %d chats alone, non-finite outputs %llu %s  (%.1f s)\n", tag, nchats, nonfin,
              nonfin ? "FAIL" : "PASS", secs_since(t0));
  sum.record(nonfin == 0, std::string(tag) + " alone outputs finite");
  if (bit_done > 0) {
    const bool ok = bmis == 0 && bq_ok;
    std::printf("  %-44s G6c %d chats: alone@4096 vs alone@roundup -- O/LSE/dK/dV mismatched elements %llu, dQ max diff %.3e (smallest atol %.3e)%s  %s\n",
                tag, bit_done, bmis, bq, bq_atol_min,
                g_kv_deterministic ? "" : " [Vj1: dK/dV count only beyond FA's atol]", ok ? "PASS" : "FAIL");
    sum.record(ok, std::string(tag) + " G6c bitwise filler independence");
  }
}

static void g6_chat_alone(const std::vector<int>* Lreal, Summary& sum) {
  std::printf("\n===== G6 chat alone: each chat through today's kernel alone vs the packed per-chat reference =====\n");
  std::vector<Layout> los = {
      {"D2 2048+2048", 4096, {{2048, 2048}}},
      {"D3 1000+3096", 4096, {{1000, 3096}}},
      {"D4 1+4095, 4095+1", 4096, {{1, 4095}, {4095, 1}}},
      {"D5 64x64, 32x128", 4096, {std::vector<int>(64, 64), std::vector<int>(32, 128)}},
      {"D6 4096 x 1", 4096, {std::vector<int>(4096, 1)}},
      make_D7(),
      make_D8(),
      {"D9 all pad", 4096, {{}}},
  };
  std::vector<Heads> hs = kHeads;
  if (g_quick) hs = {{18, 6}};
  uint64_t seed = 600;
  for (const Layout& lo : los)
    for (const Heads& h : hs) {
      if (lo.name.rfind("D6", 0) == 0 && !(h.Hq == 18 && h.Hkv == 6) && g_quick) continue;
      chat_alone(lo, h, /*bitwise_chats=*/64, sum, seed++);
    }
}

// D10: the real stage-0 rows. VARLEN_D10_FIRST / VARLEN_D10_ROWS select a window of rows;
// the seeds are the ones the full run uses for those rows (seed 1010 + absolute row of the
// batch start), so a window reproduces exactly the same inputs.
static void g6_d10(const std::vector<int>* Lreal, Summary& sum) {
  if (!Lreal) { sum.skip_("G6 D10 real stage-0 rows (labels.bin not found)"); return; }
  const std::vector<std::vector<int>> rows = pack_ffd(*Lreal, 4096);
  long long tok = 0;
  for (int x : *Lreal) tok += x;
  const bool data_ok = Lreal->size() == 5000 && tok == 6188911 && rows.size() == 1517;
  std::printf("  D10 data: %zu chats, %lld real tokens, %zu packed rows (pack_sim.py: 5000 / 6,188,911 / 1517)  %s\n",
              Lreal->size(), tok, rows.size(), data_ok ? "MATCH" : "MISMATCH");
  sum.record(data_ok, "D10 stage-0 lengths + packing match pack_sim.py");
  const int first = std::max(0, env_int("VARLEN_D10_FIRST", 0));
  if (first % 8 != 0 || first >= (int)rows.size()) {
    sum.record(false, "VARLEN_D10_FIRST must be a multiple of 8 below " + std::to_string(rows.size()));
    return;
  }
  size_t nr = rows.size() - first;
  const int lim = env_int("VARLEN_D10_ROWS", g_quick ? 16 : 0);
  if (lim > 0) nr = std::min(nr, (size_t)lim);
  Layout d10{"D10 stage-0 packed (rows " + std::to_string(first) + "-" + std::to_string(first + nr - 1) + ")", 4096, {}};
  d10.rows.assign(rows.begin() + first, rows.begin() + first + nr);
  chat_alone(d10, {18, 6}, /*bitwise_chats=*/16, sum, 1010 + first, first);
}

// ===========================================================================
// G8 (Step 3, 2026-10-10): design B forward on PACKED rows
// (gqa_flash_attn_forward_qkv with seg_start -> gqa_v62_sm90<PACKED=true>).
//   a) O and LSE of every segment -- chats AND pad tails -- vs the same per-segment reference,
//      2x rule; every output written (NaN-filled first) and finite.
//   b) chats that start on a K/V tile edge (t0 % 128 == 0) must equal the chat run ALONE
//      through the dense kernel bit for bit: the same K/V tiles in the same order, and the
//      fully masked tiles before the chat add exact zeros (the -inf guard).
//   c) isolation: new q/k/v for one chat -> every OTHER segment's O and LSE bitwise unchanged;
//      the changed chat's O must change (so the check cannot pass vacuously).
//   d) the API refuses seg_start where it cannot be honoured (it throws before any launch).
// The backward does not take seg_start until Step 4, so only O and LSE are checked here.
// ===========================================================================
static std::vector<int32_t> build_seg_start(const Layout& lo) {
  std::vector<int32_t> s((size_t)lo.rows.size() * lo.T, -1);
  for (const Segment& g : packed_segments(lo))
    for (int t = 0; t < g.L; ++t) s[(size_t)g.b * lo.T + g.t0 + t] = g.t0;
  return s;
}
// The API contract: seg_start[t] <= t, and it is t or seg_start[t-1]; every row starts at 0.
static bool seg_start_ok(const std::vector<int32_t>& s, int B, int T, std::string& why) {
  for (int b = 0; b < B; ++b)
    for (int t = 0; t < T; ++t) {
      const int32_t v = s[(size_t)b * T + t];
      const bool ok = v >= 0 && v <= t && (t == 0 ? v == 0 : (v == t || v == s[(size_t)b * T + t - 1]));
      if (!ok) {
        why = "row " + std::to_string(b) + " token " + std::to_string(t) + " has seg_start " + std::to_string(v);
        return false;
      }
    }
  return true;
}
static bool run_packed_fwd(const Inputs& in, const int32_t* d_seg, KOut& o, std::string& err) {
  const Problem& p = in.p;
  o.ensure(p);
  o.O.fill_nan();
  o.LSE.fill_nan();
  try {
    otc::gqa_flash_attn_forward_qkv(cb(in.q.p), cb(in.k.p), cb(in.v.p), mb(o.O.p), o.LSE.p, p.B, p.Hq, p.Hkv, p.T,
                                    p.hd, /*is_causal=*/true, d_seg);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  } catch (const std::exception& e) {
    err = e.what();
    return false;
  }
  return true;
}
// New random values for tokens [t0, t1) of row b, every head (isolation test).
__global__ void k_refill_tokens(bf16* x, int H, int T, int hd, int b, int t0, int t1, uint64_t seed) {
  const size_t n = (size_t)H * (t1 - t0) * hd;
  for (size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x; e < n; e += (size_t)gridDim.x * blockDim.x) {
    const int d = (int)(e % hd);
    const size_t r = e / hd;
    const int t = t0 + (int)(r % (t1 - t0)), h = (int)(r / (t1 - t0));
    const size_t idx = (((size_t)b * H + h) * T + t) * hd + d;
    const uint64_t a = splitmix64(seed ^ splitmix64(2 * idx)), c = splitmix64(seed ^ splitmix64(2 * idx + 1));
    const double u1 = ((double)(a >> 11) + 1.0) * 0x1.0p-53, u2 = (double)(c >> 11) * 0x1.0p-53;
    x[idx] = __float2bfloat16_rn((float)(sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2)));
  }
}
// Chats [c0, c0+nb) of `chats` ALONE through the dense kernel, each at token 0 of its own row of
// length Tal, random filler after it with dO = 0 there (the G6 construction).
static bool run_alone_dense(const Inputs& in, const std::vector<Segment>& chats, size_t c0, int nb, int Tal,
                            uint64_t fseed, Inputs& ai, KOut& ko, std::string& err, bool with_bwd = true) {
  const Problem& p = in.p;
  const Problem ap{nb, Tal, p.Hq, p.Hkv, p.hd};
  make_inputs(ai, ap, fseed);
  for (int r = 0; r < nb; ++r) {
    const Segment& s = chats[c0 + r];
    k_copy_tokens<<<grid_for((size_t)p.Hq * s.L * p.hd), 256>>>(in.q.p, p.T, s.b, s.t0, ai.q.p, Tal, r, 0, p.Hq, s.L, p.hd);
    k_copy_tokens<<<grid_for((size_t)p.Hkv * s.L * p.hd), 256>>>(in.k.p, p.T, s.b, s.t0, ai.k.p, Tal, r, 0, p.Hkv, s.L, p.hd);
    k_copy_tokens<<<grid_for((size_t)p.Hkv * s.L * p.hd), 256>>>(in.v.p, p.T, s.b, s.t0, ai.v.p, Tal, r, 0, p.Hkv, s.L, p.hd);
    k_copy_tokens<<<grid_for((size_t)p.Hq * s.L * p.hd), 256>>>(in.dO.p, p.T, s.b, s.t0, ai.dO.p, Tal, r, 0, p.Hq, s.L, p.hd);
    KERNEL_CHECK();
    zero_tokens(ai.dO, p.Hq, Tal, p.hd, r, s.L, Tal);
  }
  if (with_bwd) return run_today(ai, ko, err);
  ko.ensure(ai.p);   // forward only (G10)
  ko.O.fill_nan();
  ko.LSE.fill_nan();
  try {
    today_fwd(ai, ko);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  } catch (const std::exception& e) {
    err = e.what();
    return false;
  }
  return true;
}

static void g8_layout(const Layout& lo, const Heads& h, Summary& sum, uint64_t seed, int row_base = 0) {
  const int T = lo.T;
  Checks c;   // only O and LSE are filled (forward only)
  unsigned long long nonfin = 0, align_mis = 0, iso_other = 0;
  int nseg = 0, n_aligned = 0, iso_runs = 0, iso_unchanged = 0;
  std::string err;
  bool threw = false;
  DBuf<int32_t> d_seg;
  const auto t0c = std::chrono::steady_clock::now();
  for (size_t r0 = 0; r0 < lo.rows.size() && !threw; r0 += 8) {
    Layout sub{lo.name, T, {}};
    for (size_t r = r0; r < std::min(lo.rows.size(), r0 + 8); ++r) sub.rows.push_back(lo.rows[r]);
    const Problem p{(int)sub.rows.size(), T, h.Hq, h.Hkv, 128};
    Inputs in;
    make_inputs(in, p, seed + r0);
    const std::vector<Segment> segs = packed_segments(sub);
    const std::vector<int32_t> hs = build_seg_start(sub);
    std::string why;
    if (!seg_start_ok(hs, p.B, T, why)) {
      sum.record(false, "G8 " + lo.name + ": seg_start breaks the contract: " + why);
      return;
    }
    d_seg.ensure(hs.size());
    CUDA_CHECK(cudaMemcpy(d_seg.p, hs.data(), hs.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    RefOut ref;
    reference(in, segs, ref);
    KOut ko;
    if (!run_packed_fwd(in, d_seg.p, ko, err)) { threw = true; break; }
    nonfin += nonfinite(ko.O) + nonfinite(ko.LSE);
    // a) every segment, chats and pad tails
    for (const Segment& s : segs) {
      const std::string at = std::string(s.pad ? "pad" : "chat") + " row " + std::to_string(row_base + (int)r0 + s.b) +
                             " @" + std::to_string(s.t0) + " L=" + std::to_string(s.L);
      c.O.add(compare_region(ko.O.p, s.b, T, s.t0, ref.O.p, ref.Of.p, s.b, T, s.t0, p.Hq, s.L, p.hd), at);
      c.LSE.add(compare_region(ko.LSE.p, s.b, T, s.t0, ref.LSE.p, ref.LSEf.p, s.b, T, s.t0, p.Hq, s.L, 1), at);
      ++nseg;
    }
    // b) tile-aligned chats vs the same chat alone through the dense kernel, bit for bit
    std::map<int, std::vector<Segment>> al;
    for (const Segment& s : segs)
      if (!s.pad && s.t0 % 128 == 0) al[std::max(256, (s.L + 255) / 256 * 256)].push_back(s);
    for (auto& kv : al) {
      for (size_t c0 = 0; c0 < kv.second.size(); c0 += 8) {
        const int nb = (int)std::min<size_t>(8, kv.second.size() - c0);
        Inputs ai;
        KOut ka;
        if (!run_alone_dense(in, kv.second, c0, nb, kv.first, seed * 41 + kv.first + c0, ai, ka, err)) {
          threw = true;
          break;
        }
        for (int r = 0; r < nb; ++r) {
          const Segment& s = kv.second[c0 + r];
          align_mis += bitdiff(ko.O.p, s.b, T, s.t0, ka.O.p, r, kv.first, 0, p.Hq, s.L, p.hd).mism;
          align_mis += bitdiff(ko.LSE.p, s.b, T, s.t0, ka.LSE.p, r, kv.first, 0, p.Hq, s.L, 1).mism;
          ++n_aligned;
        }
      }
      if (threw) break;
    }
    if (threw) break;
    // c) isolation: the 2nd chat of the first row that has >= 2 chats gets new q/k/v
    const Segment* target = nullptr;
    for (int b = 0; b < p.B && !target; ++b) {
      int k2 = 0;
      for (const Segment& s : segs)
        if (s.b == b && !s.pad && ++k2 == 2) { target = &s; break; }
    }
    if (target) {
      Inputs in2;
      in2.p = p;
      in2.q.copy_from(in.q); in2.k.copy_from(in.k); in2.v.copy_from(in.v); in2.dO.copy_from(in.dO);
      const int tt0 = target->t0, tt1 = target->t0 + target->L;
      k_refill_tokens<<<grid_for((size_t)p.Hq * target->L * p.hd), 256>>>(in2.q.p, p.Hq, T, p.hd, target->b, tt0, tt1, seed ^ 0xA5A5ull);
      k_refill_tokens<<<grid_for((size_t)p.Hkv * target->L * p.hd), 256>>>(in2.k.p, p.Hkv, T, p.hd, target->b, tt0, tt1, seed ^ 0x5A5Aull);
      k_refill_tokens<<<grid_for((size_t)p.Hkv * target->L * p.hd), 256>>>(in2.v.p, p.Hkv, T, p.hd, target->b, tt0, tt1, seed ^ 0x3C3Cull);
      KERNEL_CHECK();
      KOut k2;
      if (!run_packed_fwd(in2, d_seg.p, k2, err)) { threw = true; break; }
      ++iso_runs;
      for (const Segment& s : segs) {
        const unsigned long long m = bitdiff(ko.O.p, s.b, T, s.t0, k2.O.p, s.b, T, s.t0, p.Hq, s.L, p.hd).mism +
                                     bitdiff(ko.LSE.p, s.b, T, s.t0, k2.LSE.p, s.b, T, s.t0, p.Hq, s.L, 1).mism;
        if (&s == target) { if (m == 0) ++iso_unchanged; }
        else iso_other += m;
      }
    }
  }
  char tag[140];
  std::snprintf(tag, sizeof tag, "G8 %s %d/%d", lo.name.c_str(), h.Hq, h.Hkv);
  if (threw) {
    std::printf("  %s  threw: %s  FAIL\n", tag, err.c_str());
    sum.record(false, std::string(tag) + " threw: " + err);
    return;
  }
  print_checks(tag, c, sum);
  std::printf("  %-44s %d segments (chats + pad tails), non-finite/unwritten O+LSE %llu %s  (%.1f s)\n", tag, nseg, nonfin,
              nonfin ? "FAIL" : "PASS", secs_since(t0c));
  sum.record(nonfin == 0, std::string(tag) + " packed O/LSE written + finite");
  if (n_aligned > 0) {
    std::printf("  %-44s %d chats at t0 %% 128 == 0: packed vs alone (dense) -- O/LSE mismatched elements %llu  %s\n", tag,
                n_aligned, align_mis, align_mis ? "FAIL" : "PASS");
    sum.record(align_mis == 0, std::string(tag) + " tile-aligned chats bitwise = alone");
  }
  if (iso_runs > 0) {
    const bool ok = iso_other == 0 && iso_unchanged == 0;
    std::printf("  %-44s isolation (%d runs): other segments' O/LSE mismatched elements %llu, changed chat unchanged in %d run(s)  %s\n",
                tag, iso_runs, iso_other, iso_unchanged, ok ? "PASS" : "FAIL");
    sum.record(ok, std::string(tag) + " isolation");
  }
}

static void g8_packed_forward(const std::vector<int>* Lreal, Summary& sum) {
  std::printf("\n===== G8 design B forward: packed rows (seg_start) vs the per-segment reference =====\n");
  std::vector<Layout> los = {
      {"D2 2048+2048", 4096, {{2048, 2048}}},
      {"D3 1000+3096", 4096, {{1000, 3096}}},
      {"D4 1+4095, 4095+1", 4096, {{1, 4095}, {4095, 1}}},
      {"D5 64x64, 32x128", 4096, {std::vector<int>(64, 64), std::vector<int>(32, 128)}},
      {"D6q 512 x 1 + pad", 4096, {std::vector<int>(512, 1)}},
      make_D7(),
      make_D8(),
      {"D9 all pad", 4096, {{}}},
      // D11: the same-length chat (300 tokens) at offsets 0/1/63/64/65/127/128/3796, behind a
      // prefix chat; the rest of each row is pad (research item 4: offsets).
      {"D11 offsets", 4096, {{300}, {1, 300}, {63, 300}, {64, 300}, {65, 300}, {127, 300}, {128, 300}, {3796, 300}}},
  };
  std::vector<Heads> hs = kHeads;
  if (g_quick) hs = {{18, 6}};
  uint64_t seed = 800;
  for (const Layout& lo : los)
    for (const Heads& h : hs) g8_layout(lo, h, sum, seed++);
  if (Lreal) {
    const std::vector<std::vector<int>> rows = pack_ffd(*Lreal, 4096);
    const int first = std::max(0, env_int("VARLEN_D10_FIRST", 0)) / 8 * 8;
    size_t nr = rows.size() - std::min<size_t>(first, rows.size());
    const int lim = env_int("VARLEN_D10_ROWS", g_quick ? 16 : 0);
    if (lim > 0) nr = std::min(nr, (size_t)lim);
    Layout d10{"D10 stage-0 packed (rows " + std::to_string(first) + "-" + std::to_string(first + nr - 1) + ")", 4096, {}};
    d10.rows.assign(rows.begin() + first, rows.begin() + first + nr);
    g8_layout(d10, {18, 6}, sum, 1810 + first, first);
  } else {
    sum.skip_("G8 D10 real stage-0 rows (labels.bin not found)");
  }
  // d) refusals: seg_start where the v62 path cannot honour it must throw before any launch
  auto refuse = [&](const char* what, int Hq, int Hkv, int T, int hd, bool causal) {
    const Problem p{1, T, Hq, Hkv, hd};
    Inputs in;
    make_inputs(in, p, 99);
    KOut o;
    o.ensure(p);
    std::vector<int32_t> hsg((size_t)T, 0);   // one segment: a valid array
    DBuf<int32_t> ds;
    ds.ensure(hsg.size());
    CUDA_CHECK(cudaMemcpy(ds.p, hsg.data(), hsg.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    bool threw = false;
    std::string msg;
    try {
      otc::gqa_flash_attn_forward_qkv(cb(in.q.p), cb(in.k.p), cb(in.v.p), mb(o.O.p), o.LSE.p, 1, Hq, Hkv, T, hd,
                                      causal, ds.p);
      CUDA_CHECK(cudaDeviceSynchronize());
    } catch (const std::exception& e) {
      threw = true;
      msg = e.what();
    }
    // Only the seg_start refusal counts. Some of these calls would ALSO throw further down for an
    // unrelated reason (T = 320 falls to the generic path, which throws for causal hd 128), so a bare
    // "it threw" would pass even with the refusal missing.
    const bool ok = threw && msg.find("seg_start") != std::string::npos;
    std::printf("  G8 refusal: %-34s %s  %s\n", what, threw ? ("threw: " + msg).c_str() : "DID NOT THROW",
                ok ? "PASS" : "FAIL");
    sum.record(ok, std::string("G8 refusal ") + what);
  };
  refuse("is_causal = false", 2, 1, 256, 128, false);
  refuse("head_dim 64", 2, 1, 256, 64, true);
  refuse("T = 320 (not a multiple of 256)", 2, 1, 320, 128, true);
}

// ===========================================================================
// G9 (Step 4, 2026-10-10): design B BACKWARD on PACKED rows
// (gqa_flash_attn_backward_qkv with seg_start -> gqa_bwd_vj1d<PACKED=true>, fed by the G8 forward).
//   a) dQ, dK, dV (and O, LSE again) of every segment -- chats AND pad tails, random dO on all of
//      them, so a pad tail is checked as the segment of its own that the kernel sees -- vs the
//      per-segment reference, 2x rule. 1-token segments: dQ/dK against the derived bound
//      (one_token_region; the exact answer is 0). Every output written (NaN-filled first), finite.
//   b) chats that start on a 128-token edge: packed == the chat ALONE through the dense kernels:
//      O, LSE, dK, dV bit for bit (same tiles in the same order; rows of the next chat add exact
//      zeros where the alone run has filler rows with dO = 0); dQ within FA's atol, because dQ is
//      reduced with atomics and differs run to run.
//   c) isolation: new q/k/v AND dO for one chat -> every OTHER segment's O/LSE/dK/dV bitwise
//      unchanged and its dQ within the atol; the changed chat's dK and dV must change.
//   d) refusals: the backward throws for a seg_start it cannot honour, with the seg_start message
//      (is_causal=false, hd 64, T % 64 != 0; a non-Hopper GPU or BLUTRAIN_NO_SM90[_GQA]; and in
//      USE_FLASH=1 / USE_DETERMINISTIC_DQ=1 runs any seg_start at all).
//   e) repeat: 8 real rows, packed fwd+bwd N times: O/LSE/dK/dV bitwise, dQ within the atol
//      (the new PACKED code paths under the same race test as G4).
//   f) work + time: tiles the packed kernels visit (host model of their trip counts) vs dense, and
//      the measured fwd/bwd medians on the same 8 rows (reported, not gated).
// ===========================================================================
static bool run_packed(const Inputs& in, const int32_t* d_seg, KOut& o, std::string& err) {
  const Problem& p = in.p;
  o.ensure(p);
  o.fill_nan();
  try {
    otc::gqa_flash_attn_forward_qkv(cb(in.q.p), cb(in.k.p), cb(in.v.p), mb(o.O.p), o.LSE.p, p.B, p.Hq, p.Hkv, p.T,
                                    p.hd, /*is_causal=*/true, d_seg);
    otc::gqa_flash_attn_backward_qkv(cb(in.q.p), cb(in.k.p), cb(in.v.p), cb(o.O.p), cb(in.dO.p), o.LSE.p,
                                     mb(o.dQ.p), mb(o.dK.p), mb(o.dV.p), p.B, p.Hq, p.Hkv, p.T, p.hd,
                                     /*is_causal=*/true, d_seg);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  } catch (const std::exception& e) {
    err = e.what();
    return false;
  }
  return true;
}
// Host models of the packed trip counts, written from the kernels' own rules:
//  backward (GQA_bwd_vj1d_sm90.cu, packed_nq): key tile kt visits q = kt, kt+1, ... while
//    seg_start[q*64] <= kt*64 + 63; dense visits q = kt .. T/64 - 1.
//  forward (GQA_fwd_sm90.cu, my_nkv): 64-row q-tile qt of pair qt/2 visits the 128-wide key tiles
//    n0 .. (qt*64 + 63)/128 with n0 = seg_start[pair*128]/128; dense starts at 0.
// Returned as (packed, dense) tile visits for one head of every row.
static std::pair<long long, long long> bwd_tiles(const std::vector<int32_t>& ss, int B, int T) {
  const int n = T / 64;
  long long pk = 0, de = 0;
  for (int b = 0; b < B; ++b)
    for (int kt = 0; kt < n; ++kt) {
      for (int q = kt; q < n && ss[(size_t)b * T + (size_t)q * 64] <= kt * 64 + 63; ++q) ++pk;
      de += n - kt;
    }
  return {pk, de};
}
static std::pair<long long, long long> fwd_tiles(const std::vector<int32_t>& ss, int B, int T) {
  long long pk = 0, de = 0;
  for (int b = 0; b < B; ++b)
    for (int qt = 0; qt < T / 64; ++qt) {
      const int last = (qt * 64 + 63) / 128, n0 = ss[(size_t)b * T + (size_t)(qt / 2) * 128] / 128;
      pk += last + 1 - n0;
      de += last + 1;
    }
  return {pk, de};
}

// (f) of the diagnostic, G9 only: the SAME chat (same inputs) run ALONE through today's dense kernels,
// g_diag_rep times, against the same per-chat reference. Same ratios here as in the packed row means the
// excess comes from the shared backward arithmetic, not from the packing code.
static void g9_diag_alone(const Inputs& in, const RefOut& ref, const KOut& ko, const Segment& s, uint64_t seed) {
  const Problem& p = in.p;
  const int Tal = std::max(256, (s.L + 255) / 256 * 256);
  const std::vector<Segment> one = {s};
  std::vector<double> rr;
  double dmax = 0;
  int over = 0;
  for (int k = 0; k < g_diag_rep; ++k) {
    Inputs ai;
    KOut ka;
    std::string err;
    if (!run_alone_dense(in, one, 0, 1, Tal, seed * 47 + 1, ai, ka, err)) { std::printf("  (f) alone run threw: %s\n", err.c_str()); return; }
    const Err e = compare_region(ka.dQ.p, 0, Tal, 0, ref.dQ.p, ref.dQf.p, s.b, p.T, s.t0, p.Hq, s.L, p.hd);
    const double ratio = e.ek / (2.0 * e.ef + 2.0 * e.er);
    rr.push_back(ratio);
    if (ratio > 1.0) ++over;
    if (k == 0) dmax = bitdiff(ko.dQ.p, s.b, p.T, s.t0, ka.dQ.p, 0, Tal, 0, p.Hq, s.L, p.hd).maxabs;
  }
  if (rr.empty()) return;
  std::sort(rr.begin(), rr.end());
  std::printf("  (f) the same chat ALONE through today's dense kernels (same inputs, %zu runs): dQ ratio min %.3f  median %.3f  max %.3f  -> %d above 1\n",
              rr.size(), rr.front(), rr[rr.size() / 2], rr.back(), over);
  std::printf("      packed vs alone (first runs): max |dQ difference| %.3e\n", dmax);
}

// VARLEN_DIAG_LIST target: packed vs alone ratios for dQ/dK/dV (same inputs), and at the worst dV
// element of the packed run, Vj1d's dV arithmetic recomputed on the host:
//   P_ij  = bf16( ex2( fma(fp32(q_i.k_j), scale*log2e, -LSE_i*log2e) ) )   (fused_p_stsm, kernel LSE)
//   dVh   = bf16( fp32( sum_i P_ij dO_i ) )          per query head (one-writer bf16 partial, :546-556)
//   dV    = bf16( sum_g float(dVh_g) ), g in order    (gqa_dkdv_greduce)
// "model A" = exactly that; "model B" = the same but the per-head partial kept unrounded and rounded
// once after the head sum (how FA3 sums GQA heads in fp32). model A == kernel proves the kernel does
// its designed arithmetic on the right keys/queries (no leak, nothing missing) at that element.
static void g9_diag_target(const Inputs& in, const RefOut& ref, const KOut& ko, const Segment& s, int abs_row,
                           uint64_t seed) {
  const Problem& p = in.p;
  const int L = s.L, hd = p.hd, Hq = p.Hq, Hkv = p.Hkv, G = p.G(), T = p.T;
  std::printf("\n  ---- DIAG target: %s row %d @%d L=%d ----\n", s.pad ? "pad" : "chat", abs_row, s.t0, L);
  auto ratio3 = [&](const KOut& k, int r, int Tk, int tk0, double out[3]) {
    const Err q = compare_region(k.dQ.p, r, Tk, tk0, ref.dQ.p, ref.dQf.p, s.b, T, s.t0, Hq, L, hd);
    const Err kk = compare_region(k.dK.p, r, Tk, tk0, ref.dK.p, ref.dKf.p, s.b, T, s.t0, Hkv, L, hd);
    const Err v = compare_region(k.dV.p, r, Tk, tk0, ref.dV.p, ref.dVf.p, s.b, T, s.t0, Hkv, L, hd);
    out[0] = q.ek / (2 * q.ef + 2 * q.er); out[1] = kk.ek / (2 * kk.ef + 2 * kk.er); out[2] = v.ek / (2 * v.ef + 2 * v.er);
  };
  double rp[3], ra[3];
  ratio3(ko, s.b, T, s.t0, rp);
  const int Tal = std::max(256, (L + 255) / 256 * 256);
  Inputs ai;
  KOut ka;
  std::string err;
  if (!run_alone_dense(in, std::vector<Segment>{s}, 0, 1, Tal, seed * 47 + 1, ai, ka, err)) {
    std::printf("  alone run threw: %s\n", err.c_str());
    return;
  }
  ratio3(ka, 0, Tal, 0, ra);
  std::printf("  ratios (2x rule)   dQ      dK      dV\n");
  std::printf("    packed       %7.3f %7.3f %7.3f\n    alone(dense) %7.3f %7.3f %7.3f   (same inputs; t0 %% 64 = %d packed, 0 alone)\n",
              rp[0], rp[1], rp[2], ra[0], ra[1], ra[2], s.t0 % 64);
  // worst dV element of the packed run
  int wh = 0, wj = 0, wd = 0;
  double we = -1;
  for (int hk = 0; hk < Hkv; ++hk) {
    const std::vector<bf16> k = d2h(ko.dV.p, (((size_t)s.b * Hkv + hk) * T + s.t0) * hd, (size_t)L * hd);
    const std::vector<double> t = d2h(ref.dV.p, (((size_t)s.b * Hkv + hk) * T + s.t0) * hd, (size_t)L * hd);
    for (size_t e = 0; e < t.size(); ++e) {
      const double x = std::fabs((double)__bfloat162float(k[e]) - t[e]);
      if (x > we) { we = x; wh = hk; wj = (int)(e / hd); wd = (int)(e % hd); }
    }
  }
  const size_t ivk = (((size_t)s.b * Hkv + wh) * T + s.t0 + wj) * hd;
  const double kv = __bfloat162float(d2h(ko.dV.p, ivk + wd, 1)[0]), tv = d2h(ref.dV.p, ivk + wd, 1)[0],
               fv = d2h(ref.dVf.p, ivk + wd, 1)[0];
  const std::vector<bf16> kr = d2h(in.k.p, ivk, hd);
  const float scale_f = 1.0f / std::sqrt((float)hd), LOG2E_f = 1.4426950408889634f, scale2 = scale_f * LOG2E_f;
  float accA = 0.f;
  double sumB = 0;
  for (int g = 0; g < G; ++g) {
    const int h = wh * G + g;
    const int nq = L - wj;   // queries wj..L-1 of the segment see key wj
    const std::vector<bf16> q = d2h(in.q.p, (((size_t)s.b * Hq + h) * T + s.t0 + wj) * hd, (size_t)nq * hd);
    const std::vector<bf16> g_ = d2h(in.dO.p, (((size_t)s.b * Hq + h) * T + s.t0 + wj) * hd, (size_t)nq * hd);
    const std::vector<float> lse = d2h(ko.LSE.p, ((size_t)s.b * Hq + h) * T + s.t0 + wj, (size_t)nq);
    double part = 0;
    for (int i = 0; i < nq; ++i) {
      double sc = 0;
      for (int d = 0; d < hd; ++d) sc += (double)__bfloat162float(q[(size_t)i * hd + d]) * (double)__bfloat162float(kr[d]);
      const float x = fmaf((float)sc, scale2, -(lse[i] * LOG2E_f));
      const float P = bf16r((float)std::exp2((double)x));
      part += (double)P * (double)__bfloat162float(g_[(size_t)i * hd + wd]);
    }
    accA = f_add(accA, bf16r((float)part));   // greduce: fp32 sum of the bf16 partials, g in order
    sumB += part;
  }
  const double mA = bf16r(accA), mB = bf16r((float)sumB);
  std::printf("  worst dV element (packed): kv head %d, segment token %d, dim %d\n", wh, wj, wd);
  std::printf("    kernel %.6f  model A (Vj1d) %.6f  model B (fp32 head sum) %.6f  truth %.6f  floor %.6f\n", kv, mA, mB, tv, fv);
  std::printf("    |model A - kernel| %.3e (0 = the kernel did exactly its designed arithmetic here);  |kernel-truth| %.3e = %.2f bf16 ulp;  |model B - truth| %.3e;  |floor-truth| %.3e\n",
              std::fabs(mA - kv), std::fabs(kv - tv), std::fabs(kv - tv) / bf16_ulp(tv), std::fabs(mB - tv), std::fabs(fv - tv));
}

static void g9_layout(const Layout& lo, const Heads& h, Summary& sum, uint64_t seed, int row_base = 0) {
  const int T = lo.T;
  Checks c;
  if (g_large_rule) {   // G13 large inputs: FA2's gate for this test (tests/test_flash_attn.py:2287-2295), as in G5
    c.dQ = Check("dQ", 5.0, 1e-3); c.dK = Check("dK", 5.0, 1e-3); c.dV = Check("dV", 5.0, 1e-3);
  }
  OneTok ot;   // 1-token segments: dQ/dK against the derived bound
  unsigned long long nonfin = 0, align_mis = 0, iso_other = 0;
  double align_dq = 0, align_atol_min = INFINITY, iso_dq_ratio = 0;
  bool align_dq_ok = true, iso_dq_ok = true;
  int nseg = 0, n_aligned = 0, iso_runs = 0, iso_unchanged = 0;
  long long bt_p = 0, bt_d = 0;
  std::string err;
  bool threw = false;
  DBuf<int32_t> d_seg;
  const auto t0c = std::chrono::steady_clock::now();
  for (size_t r0 = 0; r0 < lo.rows.size() && !threw; r0 += g_rows_per_call) {
    if (!g_diag_list.empty()) {   // VARLEN_DIAG_LIST: only the batches that hold a target row
      bool want = false;
      for (const auto& tg : g_diag_list) want |= tg.first >= row_base + (int)r0 && tg.first < row_base + (int)r0 + 8;
      if (!want) continue;
    }
    Layout sub{lo.name, T, {}};
    for (size_t r = r0; r < std::min(lo.rows.size(), r0 + g_rows_per_call); ++r) sub.rows.push_back(lo.rows[r]);
    const Problem p{(int)sub.rows.size(), T, h.Hq, h.Hkv, 128};
    Inputs in;
    make_inputs(in, p, seed + r0);   // random dO on every segment, pad tails included
    const std::vector<Segment> segs = packed_segments(sub);
    const std::vector<int32_t> hs = build_seg_start(sub);
    std::string why;
    if (!seg_start_ok(hs, p.B, T, why)) {
      sum.record(false, "G9 " + lo.name + ": seg_start breaks the contract: " + why);
      return;
    }
    const auto bt = bwd_tiles(hs, p.B, T);
    bt_p += bt.first;
    bt_d += bt.second;
    d_seg.ensure(hs.size());
    CUDA_CHECK(cudaMemcpy(d_seg.p, hs.data(), hs.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    RefOut ref;
    reference(in, segs, ref);
    KOut ko;
    if (!run_packed(in, d_seg.p, ko, err)) { threw = true; break; }
    nonfin += ko.nonfinite_all();
    // a) every segment, chats and pad tails
    for (const Segment& s : segs) {
      const std::string at = std::string(s.pad ? "pad" : "chat") + " row " + std::to_string(row_base + (int)r0 + s.b) +
                             " @" + std::to_string(s.t0) + " L=" + std::to_string(s.L);
      if (s.L == 1) one_token_region(c, ot, ko, s.b, T, in, ref, s, at, /*tk0=*/s.t0);
      else add_region(c, ko, s.b, T, s.t0, ref, s.b, T, s.t0, s.L, p, at);
      ++nseg;
      if (!s.pad && lo.name.find(g_diag_case) != std::string::npos && row_base + (int)r0 + s.b == g_diag_row &&
          s.t0 == g_diag_t0) {
        diag_chat(in, ref, ko, s.b, T, s, [&](uint64_t, Inputs&, KOut& kk) { return run_packed(in, d_seg.p, kk, err); },
                  /*tk0=*/s.t0);
        g9_diag_alone(in, ref, ko, s, seed);
      }
      for (const auto& tg : g_diag_list)
        if (tg.first == row_base + (int)r0 + s.b && tg.second == s.t0) g9_diag_target(in, ref, ko, s, tg.first, seed);
    }
    // b) chats on a 128-token edge vs the same chat alone through the dense kernels
    std::map<int, std::vector<Segment>> al;
    for (const Segment& s : segs)
      if (!s.pad && s.t0 % 128 == 0) al[std::max(256, (s.L + 255) / 256 * 256)].push_back(s);
    for (auto& kv : al) {
      for (size_t c0 = 0; c0 < kv.second.size(); c0 += 8) {
        const int nb = (int)std::min<size_t>(8, kv.second.size() - c0);
        Inputs ai;
        KOut ka;
        if (!run_alone_dense(in, kv.second, c0, nb, kv.first, seed * 43 + kv.first + c0, ai, ka, err)) {
          threw = true;
          break;
        }
        for (int r = 0; r < nb; ++r) {
          const Segment& s = kv.second[c0 + r];
          align_mis += bitdiff(ko.O.p, s.b, T, s.t0, ka.O.p, r, kv.first, 0, p.Hq, s.L, p.hd).mism;
          align_mis += bitdiff(ko.LSE.p, s.b, T, s.t0, ka.LSE.p, r, kv.first, 0, p.Hq, s.L, 1).mism;
          align_mis += bitdiff(ko.dK.p, s.b, T, s.t0, ka.dK.p, r, kv.first, 0, p.Hkv, s.L, p.hd).mism;
          align_mis += bitdiff(ko.dV.p, s.b, T, s.t0, ka.dV.p, r, kv.first, 0, p.Hkv, s.L, p.hd).mism;
          const double atol = dq_atol(ka.dQ.p, r, kv.first, 0, p.Hq, s.L, p.hd);
          const BitDiff dq = bitdiff(ko.dQ.p, s.b, T, s.t0, ka.dQ.p, r, kv.first, 0, p.Hq, s.L, p.hd);
          align_dq = std::max(align_dq, dq.maxabs);
          align_atol_min = std::min(align_atol_min, atol);
          if (dq.maxabs > atol) align_dq_ok = false;
          ++n_aligned;
        }
      }
      if (threw) break;
    }
    if (threw) break;
    // c) isolation: the 2nd chat of the first row that has >= 2 chats gets new q/k/v AND dO
    const Segment* target = nullptr;
    for (int b = 0; b < p.B && !target; ++b) {
      int k2 = 0;
      for (const Segment& s : segs)
        if (s.b == b && !s.pad && ++k2 == 2) { target = &s; break; }
    }
    if (target) {
      Inputs in2;
      in2.p = p;
      in2.q.copy_from(in.q); in2.k.copy_from(in.k); in2.v.copy_from(in.v); in2.dO.copy_from(in.dO);
      const int tt0 = target->t0, tt1 = target->t0 + target->L;
      k_refill_tokens<<<grid_for((size_t)p.Hq * target->L * p.hd), 256>>>(in2.q.p, p.Hq, T, p.hd, target->b, tt0, tt1, seed ^ 0xA5A5ull);
      k_refill_tokens<<<grid_for((size_t)p.Hkv * target->L * p.hd), 256>>>(in2.k.p, p.Hkv, T, p.hd, target->b, tt0, tt1, seed ^ 0x5A5Aull);
      k_refill_tokens<<<grid_for((size_t)p.Hkv * target->L * p.hd), 256>>>(in2.v.p, p.Hkv, T, p.hd, target->b, tt0, tt1, seed ^ 0x3C3Cull);
      k_refill_tokens<<<grid_for((size_t)p.Hq * target->L * p.hd), 256>>>(in2.dO.p, p.Hq, T, p.hd, target->b, tt0, tt1, seed ^ 0x9696ull);
      KERNEL_CHECK();
      KOut k2;
      if (!run_packed(in2, d_seg.p, k2, err)) { threw = true; break; }
      ++iso_runs;
      for (const Segment& s : segs) {
        const unsigned long long mk = bitdiff(ko.dK.p, s.b, T, s.t0, k2.dK.p, s.b, T, s.t0, p.Hkv, s.L, p.hd).mism;
        const unsigned long long mv = bitdiff(ko.dV.p, s.b, T, s.t0, k2.dV.p, s.b, T, s.t0, p.Hkv, s.L, p.hd).mism;
        if (&s == target) {
          if (mk == 0 || mv == 0) ++iso_unchanged;
          continue;
        }
        iso_other += mk + mv + bitdiff(ko.O.p, s.b, T, s.t0, k2.O.p, s.b, T, s.t0, p.Hq, s.L, p.hd).mism +
                     bitdiff(ko.LSE.p, s.b, T, s.t0, k2.LSE.p, s.b, T, s.t0, p.Hq, s.L, 1).mism;
        const double atol = dq_atol(ko.dQ.p, s.b, T, s.t0, p.Hq, s.L, p.hd);
        const BitDiff dq = bitdiff(ko.dQ.p, s.b, T, s.t0, k2.dQ.p, s.b, T, s.t0, p.Hq, s.L, p.hd);
        if (dq.maxabs > atol) iso_dq_ok = false;
        iso_dq_ratio = std::max(iso_dq_ratio, atol > 0 ? dq.maxabs / atol : (dq.maxabs > 0 ? INFINITY : 0.0));
      }
    }
  }
  char tag[140];
  std::snprintf(tag, sizeof tag, "G9 %s %d/%d", lo.name.c_str(), h.Hq, h.Hkv);
  if (threw) {
    std::printf("  %s  threw: %s  FAIL\n", tag, err.c_str());
    sum.record(false, std::string(tag) + " threw: " + err);
    return;
  }
  print_checks(tag, c, sum);
  if (ot.n > 0) {
    std::printf("  %-44s 1-token segments: %d (exact dQ = dK = 0; truth exactly 0 for %d of them, the rest use the 2x rule)\n",
                tag, ot.n, ot.n - ot.not_zero);
    for (Check* k : {&ot.q, &ot.k}) {
      if (k->n == 0) continue;
      std::printf("  %-44s %-4s 1-token  segs  %5d  worst |kernel|/bound %.3e  (max |kernel| %.3e)  fails %d  nan %llu  %s\n",
                  tag, k->name.c_str(), k->n, k->worst, k->wk, k->fails, k->nan, k->ok() ? "PASS" : "FAIL");
      for (size_t i = 0; i < k->where.size() && i < 20; ++i)
        std::printf("        FAIL:  %s\n", k->where[i].detail.c_str());
      sum.record(k->ok(), std::string(tag) + " " + k->name + " 1-token bound");
    }
  }
  std::printf("  %-44s %d segments (chats + pad tails), non-finite/unwritten O+LSE+dQ+dK+dV %llu %s  (%.1f s)\n", tag, nseg,
              nonfin, nonfin ? "FAIL" : "PASS", secs_since(t0c));
  sum.record(nonfin == 0, std::string(tag) + " packed outputs written + finite");
  std::printf("  %-44s backward q-tile visits (host model): packed %lld vs dense %lld = %.1f%%\n", tag, bt_p, bt_d,
              bt_d ? 100.0 * bt_p / bt_d : 0.0);
  if (n_aligned > 0) {
    const bool ok = align_mis == 0 && align_dq_ok;
    std::printf("  %-44s %d chats at t0 %% 128 == 0: packed vs alone (dense) -- O/LSE/dK/dV mismatched elements %llu, dQ max diff %.3e (smallest atol %.3e)  %s\n",
                tag, n_aligned, align_mis, align_dq, align_atol_min, ok ? "PASS" : "FAIL");
    sum.record(ok, std::string(tag) + " tile-aligned chats = alone");
  }
  if (iso_runs > 0) {
    const bool ok = iso_other == 0 && iso_unchanged == 0 && iso_dq_ok;
    std::printf("  %-44s isolation incl. dO (%d runs): other segments' O/LSE/dK/dV mismatched elements %llu, worst dQ diff/atol %.3f, changed chat's dK/dV unchanged in %d run(s)  %s\n",
                tag, iso_runs, iso_other, iso_dq_ratio, iso_unchanged, ok ? "PASS" : "FAIL");
    sum.record(ok, std::string(tag) + " isolation incl. dO");
  }
}

// d) refusals. Every call gets a VALID seg_start; it must throw with the seg_start message (a throw for
// any other reason does not count, see the note in G8's refusals).
static void g9_refuse(Summary& sum, const char* what, bool forward, int Hq, int Hkv, int T, int hd, bool causal) {
  const Problem p{1, T, Hq, Hkv, hd};
  Inputs in;
  make_inputs(in, p, 99);
  KOut o;
  o.ensure(p);   // O/LSE contents do not matter: the call must refuse before any launch
  std::vector<int32_t> hsg((size_t)T, 0);   // one segment: a valid array
  DBuf<int32_t> ds;
  ds.ensure(hsg.size());
  CUDA_CHECK(cudaMemcpy(ds.p, hsg.data(), hsg.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
  bool threw = false;
  std::string msg;
  try {
    if (forward)
      otc::gqa_flash_attn_forward_qkv(cb(in.q.p), cb(in.k.p), cb(in.v.p), mb(o.O.p), o.LSE.p, 1, Hq, Hkv, T, hd, causal,
                                      ds.p);
    else
      otc::gqa_flash_attn_backward_qkv(cb(in.q.p), cb(in.k.p), cb(in.v.p), cb(o.O.p), cb(in.dO.p), o.LSE.p, mb(o.dQ.p),
                                       mb(o.dK.p), mb(o.dV.p), 1, Hq, Hkv, T, hd, causal, ds.p);
    CUDA_CHECK(cudaDeviceSynchronize());
  } catch (const std::exception& e) {
    threw = true;
    msg = e.what();
  }
  const bool ok = threw && msg.find("seg_start") != std::string::npos;
  std::printf("  G9 refusal: %-8s %-40s %s  %s\n", forward ? "forward" : "backward", what,
              threw ? ("threw: " + msg).c_str() : "DID NOT THROW", ok ? "PASS" : "FAIL");
  sum.record(ok, std::string("G9 refusal ") + (forward ? "forward " : "backward ") + what);
}

// e) + f) on 8 packed rows: repeat (race) test and work/time, packed vs dense on the same inputs.
static void g9_repeat_and_time(const Layout& lo8, Summary& sum) {
  const Problem p{(int)lo8.rows.size(), lo8.T, 18, 6, 128};
  Inputs in;
  make_inputs(in, p, 9090);
  const std::vector<int32_t> hs = build_seg_start(lo8);
  std::string why, err;
  if (!seg_start_ok(hs, p.B, p.T, why)) { sum.record(false, "G9 repeat: seg_start breaks the contract: " + why); return; }
  DBuf<int32_t> d_seg;
  d_seg.ensure(hs.size());
  CUDA_CHECK(cudaMemcpy(d_seg.p, hs.data(), hs.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
  const int HB = p.B * p.Hq, HKB = p.B * p.Hkv;   // treat [B,H] as one axis
  // e) repeat
  {
    const int reps = env_int("VARLEN_REPEATS", g_quick ? 50 : 1000);
    const auto t0 = std::chrono::steady_clock::now();
    KOut base, cur;
    if (!run_packed(in, d_seg.p, base, err)) { sum.record(false, "G9 repeat threw: " + err); return; }
    const double atol = dq_atol(base.dQ.p, 0, p.T, 0, HB, p.T, p.hd);
    unsigned long long mO = 0, mL = 0, mK = 0, mV = 0, mQ = 0;
    double worst_dq = 0;
    for (int r = 0; r < reps; ++r) {
      if (!run_packed(in, d_seg.p, cur, err)) { sum.record(false, "G9 repeat threw: " + err); return; }
      mO += bitdiff(base.O.p, 0, p.T, 0, cur.O.p, 0, p.T, 0, HB, p.T, p.hd).mism;
      mL += bitdiff(base.LSE.p, 0, p.T, 0, cur.LSE.p, 0, p.T, 0, HB, p.T, 1).mism;
      mK += bitdiff(base.dK.p, 0, p.T, 0, cur.dK.p, 0, p.T, 0, HKB, p.T, p.hd).mism;
      mV += bitdiff(base.dV.p, 0, p.T, 0, cur.dV.p, 0, p.T, 0, HKB, p.T, p.hd).mism;
      const BitDiff q = bitdiff(base.dQ.p, 0, p.T, 0, cur.dQ.p, 0, p.T, 0, HB, p.T, p.hd);
      mQ += q.mism;
      worst_dq = std::max(worst_dq, q.maxabs);
    }
    const bool okOL = mO == 0 && mL == 0, okKV = mK == 0 && mV == 0, okQ = worst_dq <= atol;
    std::printf("  G9 repeat packed x%-5d (%s)  O/LSE mismatched elements %llu/%llu %s | dK/dV %llu/%llu %s | dQ max diff %.3e vs atol %.3e (%llu elems differ) %s  (%.1f s)\n",
                reps, lo8.name.c_str(), mO, mL, okOL ? "PASS" : "FAIL", mK, mV, okKV ? "PASS" : "FAIL", worst_dq, atol,
                mQ, okQ ? "PASS" : "FAIL", secs_since(t0));
    sum.record(okOL, "G9 repeat packed O/LSE bitwise");
    sum.record(okKV, "G9 repeat packed dK/dV bitwise");
    sum.record(okQ, "G9 repeat packed dQ");
  }
  // f) work + time (free clocks: reported, not gated)
  {
    const auto fw = fwd_tiles(hs, p.B, p.T);
    const auto bw = bwd_tiles(hs, p.B, p.T);
    KOut kd, kp;
    kd.ensure(p);
    kp.ensure(p);
    auto pk_fwd = [&] {
      otc::gqa_flash_attn_forward_qkv(cb(in.q.p), cb(in.k.p), cb(in.v.p), mb(kp.O.p), kp.LSE.p, p.B, p.Hq, p.Hkv, p.T,
                                      p.hd, true, d_seg.p);
    };
    auto pk_bwd = [&] {
      otc::gqa_flash_attn_backward_qkv(cb(in.q.p), cb(in.k.p), cb(in.v.p), cb(kp.O.p), cb(in.dO.p), kp.LSE.p,
                                       mb(kp.dQ.p), mb(kp.dK.p), mb(kp.dV.p), p.B, p.Hq, p.Hkv, p.T, p.hd, true, d_seg.p);
    };
    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    auto timed = [&](const std::function<void()>& f) {
      float ms = 0;
      CUDA_CHECK(cudaEventRecord(a));
      f();
      CUDA_CHECK(cudaEventRecord(b));
      CUDA_CHECK(cudaEventSynchronize(b));
      CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
      return ms;
    };
    std::vector<float> df, db, pf, pb;
    try {
      for (int i = 0; i < 5; ++i) { today_fwd(in, kd); today_bwd(in, kd); pk_fwd(); pk_bwd(); }
      CUDA_CHECK(cudaDeviceSynchronize());
      for (int i = 0; i < 20; ++i) {   // interleaved, so clock drift hits both alike
        df.push_back(timed([&] { today_fwd(in, kd); }));
        db.push_back(timed([&] { today_bwd(in, kd); }));
        pf.push_back(timed(pk_fwd));
        pb.push_back(timed(pk_bwd));
      }
    } catch (const std::exception& e) {
      sum.record(false, std::string("G9 timing threw: ") + e.what());
      return;
    }
    auto med = [](std::vector<float> v) {
      std::sort(v.begin(), v.end());
      return v.size() % 2 ? (double)v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
    };
    std::printf("  G9 work (host model, %s): forward key-tile visits packed/dense %.1f%%, backward q-tile visits %.1f%%\n",
                lo8.name.c_str(), 100.0 * fw.first / fw.second, 100.0 * bw.first / bw.second);
    std::printf("  G9 time B=%d T=%d 18/6 (median of 20, interleaved): fwd dense %.4f ms, packed %.4f ms (%.1f%%) | bwd dense %.4f ms, packed %.4f ms (%.1f%%)\n",
                p.B, p.T, med(df), med(pf), 100.0 * med(pf) / med(df), med(db), med(pb), 100.0 * med(pb) / med(db));
    std::printf("  (timing is reported, not gated; free clocks)\n");
    CUDA_CHECK(cudaEventDestroy(a));
    CUDA_CHECK(cudaEventDestroy(b));
  }
}

static void g9_packed_backward(const std::vector<int>* Lreal, Summary& sum, bool kernels_ok) {
  std::printf("\n===== G9 design B backward: packed rows (seg_start) vs the per-segment reference =====\n");
  if (!kernels_ok) {   // non-Hopper GPU or BLUTRAIN_NO_SM90[_GQA]: only the refusals can run, and must hold
    std::printf("  kernels under test are not reachable here: only the refusals run (both must throw)\n");
    g9_refuse(sum, "this GPU / BLUTRAIN_NO_SM90[_GQA]", true, 2, 1, 256, 128, true);
    g9_refuse(sum, "this GPU / BLUTRAIN_NO_SM90[_GQA]", false, 2, 1, 256, 128, true);
    return;
  }
  const std::string mode = bwd_mode_name();
  if (mode.rfind("Vj1d", 0) != 0) {   // USE_FLASH=1 or USE_DETERMINISTIC_DQ=1: no packed backward there
    std::printf("  backward mode is %s: packed rows must be refused\n", mode.c_str());
    g9_refuse(sum, mode.c_str(), false, 2, 1, 256, 128, true);
    return;
  }
  std::vector<Layout> los = {
      {"D2 2048+2048", 4096, {{2048, 2048}}},
      {"D3 1000+3096", 4096, {{1000, 3096}}},
      {"D4 1+4095, 4095+1", 4096, {{1, 4095}, {4095, 1}}},
      {"D5 64x64, 32x128", 4096, {std::vector<int>(64, 64), std::vector<int>(32, 128)}},
      {"D6q 512 x 1 + pad", 4096, {std::vector<int>(512, 1)}},
      make_D7(),
      make_D8(),
      {"D9 all pad", 4096, {{}}},
      {"D11 offsets", 4096, {{300}, {1, 300}, {63, 300}, {64, 300}, {65, 300}, {127, 300}, {128, 300}, {3796, 300}}},
  };
  std::vector<Heads> hs = kHeads;
  if (g_quick) hs = {{18, 6}};
  uint64_t seed = 900;
  for (const Layout& lo : los)
    for (const Heads& h : hs) {
      const uint64_t sd = seed++;   // the seed of a layout does not depend on the VARLEN_CASE filter
      if (g_case.empty() || lo.name.find(g_case) != std::string::npos) g9_layout(lo, h, sum, sd);
    }
  if (!g_case.empty() && g_case.find("D10") == std::string::npos) {
    std::printf("  VARLEN_CASE=%s: D10, repeat, timing and refusals skipped\n", g_case.c_str());
    return;
  }
  if (Lreal) {
    const std::vector<std::vector<int>> rows = pack_ffd(*Lreal, 4096);
    const int first = std::max(0, env_int("VARLEN_D10_FIRST", 0)) / 8 * 8;
    size_t nr = rows.size() - std::min<size_t>(first, rows.size());
    const int lim = env_int("VARLEN_D10_ROWS", g_quick ? 16 : 0);
    if (lim > 0) nr = std::min(nr, (size_t)lim);
    Layout d10{"D10 stage-0 packed (rows " + std::to_string(first) + "-" + std::to_string(first + nr - 1) + ")", 4096, {}};
    d10.rows.assign(rows.begin() + first, rows.begin() + first + nr);
    g9_layout(d10, {18, 6}, sum, 1910 + first, first);
    if (!g_diag_list.empty()) {
      std::printf("  VARLEN_DIAG_LIST set: repeat, timing and refusals skipped\n");
      return;
    }
    Layout r8{"D10 rows 0-7", 4096, {}};
    r8.rows.assign(rows.begin(), rows.begin() + std::min<size_t>(8, rows.size()));
    g9_repeat_and_time(r8, sum);
  } else {
    sum.skip_("G9 D10 real stage-0 rows + repeat + timing (labels.bin not found)");
  }
  g9_refuse(sum, "is_causal = false", false, 2, 1, 256, 128, false);
  g9_refuse(sum, "head_dim 64", false, 2, 1, 256, 64, true);
  g9_refuse(sum, "T = 288 (not a multiple of 64)", false, 2, 1, 288, 128, true);
}

// ===========================================================================
// G10 (Step 5, 2026-10-10): design A forward on PACKED rows (gqa_flash_attn_forward_qkv with a
// GqaChatPlan -> gqa_v62_sm90<MODE_CHATS>).
//   a) O and LSE of every segment (chats AND pad tails) vs the per-segment reference, 2x rule; every
//      output written (NaN-filled first) and finite -- the plan covers every row.
//   b) EVERY segment (not only tile-aligned ones) == the segment ALONE through the dense kernel, bit
//      for bit. A cuts its tiles at the segment start exactly as the dense kernel cuts them at row 0,
//      so the arithmetic is the same; keys past the segment end sit only in diagonal tiles, where
//      causal hides them in both runs.
//   c) isolation: new q/k/v for one chat -> every OTHER segment's O/LSE bitwise unchanged; the
//      changed chat's O must change.
//   d) refusals, each with its own message: plan + seg_start; plan on paths v62 cannot take; plan
//      built for another shape; plan builder: gap, overlap, segment outside its row, row not covered.
//   e) time: dense vs B vs A forward on 8 real rows from three places of the packed set (first =
//      longest chats, middle, last = shortest), median of 20, interleaved (reported, not gated).
// ===========================================================================
struct PlanHolder {   // frees the plan's device lists on scope exit
  otc::GqaChatPlan p;
  PlanHolder() = default;
  PlanHolder(const PlanHolder&) = delete;
  PlanHolder& operator=(const PlanHolder&) = delete;
  ~PlanHolder() { otc::gqa_chat_plan_free(p); }
};
static void build_plan(const Layout& lo, int Hq, otc::GqaChatPlan& plan) {
  std::vector<int32_t> b, t0, L;
  for (const Segment& g : packed_segments(lo)) { b.push_back(g.b); t0.push_back(g.t0); L.push_back(g.L); }
  otc::gqa_chat_plan_build(plan, b.data(), t0.data(), L.data(), (int)b.size(), (int)lo.rows.size(), lo.T, Hq);
}
static bool run_chats_fwd(const Inputs& in, const otc::GqaChatPlan* plan, KOut& o, std::string& err) {
  const Problem& p = in.p;
  o.ensure(p);
  o.O.fill_nan();
  o.LSE.fill_nan();
  try {
    otc::gqa_flash_attn_forward_qkv(cb(in.q.p), cb(in.k.p), cb(in.v.p), mb(o.O.p), o.LSE.p, p.B, p.Hq, p.Hkv, p.T,
                                    p.hd, /*is_causal=*/true, /*seg_start=*/nullptr, plan);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  } catch (const std::exception& e) {
    err = e.what();
    return false;
  }
  return true;
}

static void g10_layout(const Layout& lo, const Heads& h, Summary& sum, uint64_t seed, int row_base = 0) {
  const int T = lo.T;
  Checks c;   // only O and LSE are filled (forward)
  unsigned long long nonfin = 0, alone_mis = 0, iso_other = 0;
  int nseg = 0, n_alone = 0, iso_runs = 0, iso_unchanged = 0;
  std::string err;
  bool threw = false;
  const auto t0c = std::chrono::steady_clock::now();
  for (size_t r0 = 0; r0 < lo.rows.size() && !threw; r0 += 8) {
    Layout sub{lo.name, T, {}};
    for (size_t r = r0; r < std::min(lo.rows.size(), r0 + 8); ++r) sub.rows.push_back(lo.rows[r]);
    const Problem p{(int)sub.rows.size(), T, h.Hq, h.Hkv, 128};
    Inputs in;
    make_inputs(in, p, seed + r0);
    const std::vector<Segment> segs = packed_segments(sub);
    PlanHolder plan;
    try {
      build_plan(sub, p.Hq, plan.p);
    } catch (const std::exception& e) {
      err = std::string("plan: ") + e.what();
      threw = true;
      break;
    }
    RefOut ref;
    reference(in, segs, ref);
    KOut ko;
    if (!run_chats_fwd(in, &plan.p, ko, err)) { threw = true; break; }
    nonfin += nonfinite(ko.O) + nonfinite(ko.LSE);
    // a) every segment, chats and pad tails
    for (const Segment& g : segs) {
      const std::string at = std::string(g.pad ? "pad" : "chat") + " row " + std::to_string(row_base + (int)r0 + g.b) +
                             " @" + std::to_string(g.t0) + " L=" + std::to_string(g.L);
      c.O.add(compare_region(ko.O.p, g.b, T, g.t0, ref.O.p, ref.Of.p, g.b, T, g.t0, p.Hq, g.L, p.hd), at);
      c.LSE.add(compare_region(ko.LSE.p, g.b, T, g.t0, ref.LSE.p, ref.LSEf.p, g.b, T, g.t0, p.Hq, g.L, 1), at);
      ++nseg;
    }
    // b) every segment alone through the dense kernel, bit for bit (pad tails included)
    std::map<int, std::vector<Segment>> al;
    for (const Segment& g : segs) al[std::max(256, (g.L + 255) / 256 * 256)].push_back(g);
    for (auto& kv : al) {
      for (size_t c0 = 0; c0 < kv.second.size(); c0 += 8) {
        const int nb = (int)std::min<size_t>(8, kv.second.size() - c0);
        Inputs ai;
        KOut ka;
        if (!run_alone_dense(in, kv.second, c0, nb, kv.first, seed * 53 + kv.first + c0, ai, ka, err, /*with_bwd=*/false)) {
          threw = true;
          break;
        }
        for (int r = 0; r < nb; ++r) {
          const Segment& g = kv.second[c0 + r];
          alone_mis += bitdiff(ko.O.p, g.b, T, g.t0, ka.O.p, r, kv.first, 0, p.Hq, g.L, p.hd).mism;
          alone_mis += bitdiff(ko.LSE.p, g.b, T, g.t0, ka.LSE.p, r, kv.first, 0, p.Hq, g.L, 1).mism;
          ++n_alone;
        }
      }
      if (threw) break;
    }
    if (threw) break;
    // c) isolation: the 2nd chat of the first row that has >= 2 chats gets new q/k/v
    const Segment* target = nullptr;
    for (int b = 0; b < p.B && !target; ++b) {
      int k2 = 0;
      for (const Segment& g : segs)
        if (g.b == b && !g.pad && ++k2 == 2) { target = &g; break; }
    }
    if (target) {
      Inputs in2;
      in2.p = p;
      in2.q.copy_from(in.q); in2.k.copy_from(in.k); in2.v.copy_from(in.v); in2.dO.copy_from(in.dO);
      const int tt0 = target->t0, tt1 = target->t0 + target->L;
      k_refill_tokens<<<grid_for((size_t)p.Hq * target->L * p.hd), 256>>>(in2.q.p, p.Hq, T, p.hd, target->b, tt0, tt1, seed ^ 0xA5A5ull);
      k_refill_tokens<<<grid_for((size_t)p.Hkv * target->L * p.hd), 256>>>(in2.k.p, p.Hkv, T, p.hd, target->b, tt0, tt1, seed ^ 0x5A5Aull);
      k_refill_tokens<<<grid_for((size_t)p.Hkv * target->L * p.hd), 256>>>(in2.v.p, p.Hkv, T, p.hd, target->b, tt0, tt1, seed ^ 0x3C3Cull);
      KERNEL_CHECK();
      KOut k2;
      if (!run_chats_fwd(in2, &plan.p, k2, err)) { threw = true; break; }
      ++iso_runs;
      for (const Segment& g : segs) {
        const unsigned long long m = bitdiff(ko.O.p, g.b, T, g.t0, k2.O.p, g.b, T, g.t0, p.Hq, g.L, p.hd).mism +
                                     bitdiff(ko.LSE.p, g.b, T, g.t0, k2.LSE.p, g.b, T, g.t0, p.Hq, g.L, 1).mism;
        if (&g == target) { if (m == 0) ++iso_unchanged; }
        else iso_other += m;
      }
    }
  }
  char tag[140];
  std::snprintf(tag, sizeof tag, "G10 %s %d/%d", lo.name.c_str(), h.Hq, h.Hkv);
  if (threw) {
    std::printf("  %s  threw: %s  FAIL\n", tag, err.c_str());
    sum.record(false, std::string(tag) + " threw: " + err);
    return;
  }
  print_checks(tag, c, sum);
  std::printf("  %-44s %d segments (chats + pad tails), non-finite/unwritten O+LSE %llu %s  (%.1f s)\n", tag, nseg, nonfin,
              nonfin ? "FAIL" : "PASS", secs_since(t0c));
  sum.record(nonfin == 0, std::string(tag) + " A O/LSE written + finite");
  std::printf("  %-44s %d segments: A vs the segment alone (dense kernel) -- O/LSE mismatched elements %llu  %s\n", tag,
              n_alone, alone_mis, alone_mis ? "FAIL" : "PASS");
  sum.record(alone_mis == 0, std::string(tag) + " every segment bitwise = alone");
  if (iso_runs > 0) {
    const bool ok = iso_other == 0 && iso_unchanged == 0;
    std::printf("  %-44s isolation (%d runs): other segments' O/LSE mismatched elements %llu, changed chat unchanged in %d run(s)  %s\n",
                tag, iso_runs, iso_other, iso_unchanged, ok ? "PASS" : "FAIL");
    sum.record(ok, std::string(tag) + " isolation");
  }
}

// d) refusals. `what` must appear in the exception message (a throw for another reason does not count).
static void g10_refuse(Summary& sum, const char* name, const char* expect, const std::function<void()>& call) {
  bool threw = false;
  std::string msg;
  try {
    call();
    CUDA_CHECK(cudaDeviceSynchronize());
  } catch (const std::exception& e) {
    threw = true;
    msg = e.what();
  }
  const bool ok = threw && msg.find(expect) != std::string::npos;
  std::printf("  G10 refusal: %-44s %s  %s\n", name, threw ? ("threw: " + msg).c_str() : "DID NOT THROW", ok ? "PASS" : "FAIL");
  sum.record(ok, std::string("G10 refusal ") + name);
}
static void g10_refusals(Summary& sum, bool kernels_ok) {
  auto fwd = [&](int B, int Hq, int Hkv, int T, int hd, bool causal, const int32_t* seg, const otc::GqaChatPlan* pl) {
    const Problem p{B, T, Hq, Hkv, hd};
    Inputs in;
    make_inputs(in, p, 77);
    KOut o;
    o.ensure(p);
    otc::gqa_flash_attn_forward_qkv(cb(in.q.p), cb(in.k.p), cb(in.v.p), mb(o.O.p), o.LSE.p, B, Hq, Hkv, T, hd, causal, seg, pl);
  };
  auto plan_for = [&](int B, int T, int Hq, otc::GqaChatPlan& pl) {
    std::vector<int32_t> b, t0, L;
    for (int r = 0; r < B; ++r) { b.push_back(r); t0.push_back(0); L.push_back(T); }
    otc::gqa_chat_plan_build(pl, b.data(), t0.data(), L.data(), B, B, T, Hq);
  };
  auto build = [&](std::vector<int32_t> b, std::vector<int32_t> t0, std::vector<int32_t> L, int B, int T) {
    otc::GqaChatPlan pl;
    otc::gqa_chat_plan_build(pl, b.data(), t0.data(), L.data(), (int)b.size(), B, T, 2);
    otc::gqa_chat_plan_free(pl);
  };
  // the builder (host checks, before any device work)
  g10_refuse(sum, "builder: gap in a row", "gap or overlap", [&] { build({0, 0}, {0, 120}, {100, 136}, 1, 256); });
  g10_refuse(sum, "builder: overlap in a row", "gap or overlap", [&] { build({0, 0}, {0, 100}, {120, 156}, 1, 256); });
  g10_refuse(sum, "builder: segment past the row end", "outside its row", [&] { build({0}, {0}, {300}, 1, 256); });
  g10_refuse(sum, "builder: row 1 not covered", "is covered up to", [&] { build({0}, {0}, {256}, 2, 256); });
  if (!kernels_ok) {
    PlanHolder pl;
    plan_for(1, 256, 2, pl.p);
    g10_refuse(sum, "this GPU / BLUTRAIN_NO_SM90[_GQA]", "chats", [&] { fwd(1, 2, 1, 256, 128, true, nullptr, &pl.p); });
    return;
  }
  {
    PlanHolder pl;
    plan_for(1, 256, 2, pl.p);
    DBuf<int32_t> ds;
    std::vector<int32_t> hsg(256, 0);
    ds.ensure(hsg.size());
    CUDA_CHECK(cudaMemcpy(ds.p, hsg.data(), hsg.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    g10_refuse(sum, "plan AND seg_start", "not both", [&] { fwd(1, 2, 1, 256, 128, true, ds.p, &pl.p); });
    g10_refuse(sum, "is_causal = false", "chats", [&] { fwd(1, 2, 1, 256, 128, false, nullptr, &pl.p); });
    g10_refuse(sum, "head_dim 64", "chats", [&] { fwd(1, 2, 1, 256, 64, true, nullptr, &pl.p); });
    g10_refuse(sum, "plan for B=1 used with B=2", "another shape", [&] { fwd(2, 2, 1, 256, 128, true, nullptr, &pl.p); });
    g10_refuse(sum, "plan for 2 heads used with 4", "another shape", [&] { fwd(1, 4, 1, 256, 128, true, nullptr, &pl.p); });
  }
  {
    PlanHolder pl;
    plan_for(1, 320, 2, pl.p);
    g10_refuse(sum, "T = 320 (not a multiple of 256)", "chats", [&] { fwd(1, 2, 1, 320, 128, true, nullptr, &pl.p); });
  }
}

// e) forward time: dense vs B vs A on the same 8 rows (free clocks; reported, not gated)
static void g10_time(const std::vector<std::vector<int>>& rows, int first, Summary& sum) {
  Layout lo{"D10 rows " + std::to_string(first) + "-" + std::to_string(first + 7), 4096, {}};
  lo.rows.assign(rows.begin() + first, rows.begin() + first + 8);
  const Problem p{8, 4096, 18, 6, 128};
  Inputs in;
  make_inputs(in, p, 1010 + first);
  const std::vector<int32_t> hs = build_seg_start(lo);
  DBuf<int32_t> d_seg;
  d_seg.ensure(hs.size());
  CUDA_CHECK(cudaMemcpy(d_seg.p, hs.data(), hs.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
  PlanHolder pl;
  build_plan(lo, p.Hq, pl.p);
  // A's work in the same unit as fwd_tiles: per 64-row q-tile of a segment, the 128-key tiles it visits
  long long wa = 0;
  for (const Segment& g : packed_segments(lo)) {
    const int nq = 2 * ((g.L + 127) / 128);
    for (int qt = 0; qt < nq; ++qt) wa += (qt * 64 >= g.L) ? 0 : (qt * 64 + 63) / 128 + 1;
  }
  const auto fw = fwd_tiles(hs, p.B, p.T);
  KOut kd, kb, ka;
  kd.ensure(p); kb.ensure(p); ka.ensure(p);
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  auto timed = [&](const std::function<void()>& f) {
    float ms = 0;
    CUDA_CHECK(cudaEventRecord(a));
    f();
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    return ms;
  };
  auto f_dense = [&] { today_fwd(in, kd); };
  auto f_b = [&] {
    otc::gqa_flash_attn_forward_qkv(cb(in.q.p), cb(in.k.p), cb(in.v.p), mb(kb.O.p), kb.LSE.p, p.B, p.Hq, p.Hkv, p.T, p.hd, true, d_seg.p);
  };
  auto f_a = [&] {
    otc::gqa_flash_attn_forward_qkv(cb(in.q.p), cb(in.k.p), cb(in.v.p), mb(ka.O.p), ka.LSE.p, p.B, p.Hq, p.Hkv, p.T, p.hd, true, nullptr, &pl.p);
  };
  std::vector<float> td, tb, ta;
  try {
    for (int i = 0; i < 5; ++i) { f_dense(); f_b(); f_a(); }
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int i = 0; i < 20; ++i) { td.push_back(timed(f_dense)); tb.push_back(timed(f_b)); ta.push_back(timed(f_a)); }
  } catch (const std::exception& e) {
    sum.record(false, std::string("G10 timing threw: ") + e.what());
    return;
  }
  auto med = [](std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return v.size() % 2 ? (double)v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
  };
  const double md = med(td), mbb = med(tb), maa = med(ta);
  std::printf("  G10 %-16s work (key-tile visits vs dense): B %.1f%%, A %.1f%% | fwd median: dense %.4f ms, B %.4f ms (%.1f%%), A %.4f ms (%.1f%%)\n",
              lo.name.c_str(), 100.0 * fw.first / fw.second, 100.0 * wa / fw.second, md, mbb, 100.0 * mbb / md, maa, 100.0 * maa / md);
  CUDA_CHECK(cudaEventDestroy(a));
  CUDA_CHECK(cudaEventDestroy(b));
}

static void g10_chats_forward(const std::vector<int>* Lreal, Summary& sum, bool kernels_ok) {
  std::printf("\n===== G10 design A forward: per-segment plan (chats + pad tails) vs reference, alone, B =====\n");
  g10_refusals(sum, kernels_ok);
  if (!kernels_ok) return;
  std::vector<Layout> los = {
      {"D2 2048+2048", 4096, {{2048, 2048}}},
      {"D3 1000+3096", 4096, {{1000, 3096}}},
      {"D4 1+4095, 4095+1", 4096, {{1, 4095}, {4095, 1}}},
      {"D5 64x64, 32x128", 4096, {std::vector<int>(64, 64), std::vector<int>(32, 128)}},
      {"D6q 512 x 1 + pad", 4096, {std::vector<int>(512, 1)}},
      make_D7(),
      make_D8(),
      {"D9 all pad", 4096, {{}}},
      {"D11 offsets", 4096, {{300}, {1, 300}, {63, 300}, {64, 300}, {65, 300}, {127, 300}, {128, 300}, {3796, 300}}},
  };
  std::vector<Heads> hs = kHeads;
  if (g_quick) hs = {{18, 6}};
  uint64_t seed = 1000;
  for (const Layout& lo : los)
    for (const Heads& h : hs) {
      const uint64_t sd = seed++;
      if (g_case.empty() || lo.name.find(g_case) != std::string::npos) g10_layout(lo, h, sum, sd);
    }
  if (!g_case.empty() && g_case.find("D10") == std::string::npos) return;
  if (!Lreal) { sum.skip_("G10 D10 real stage-0 rows + timing (labels.bin not found)"); return; }
  const std::vector<std::vector<int>> rows = pack_ffd(*Lreal, 4096);
  const int first = std::max(0, env_int("VARLEN_D10_FIRST", 0)) / 8 * 8;
  size_t nr = rows.size() - std::min<size_t>(first, rows.size());
  const int lim = env_int("VARLEN_D10_ROWS", g_quick ? 16 : 0);
  if (lim > 0) nr = std::min(nr, (size_t)lim);
  Layout d10{"D10 stage-0 packed (rows " + std::to_string(first) + "-" + std::to_string(first + nr - 1) + ")", 4096, {}};
  d10.rows.assign(rows.begin() + first, rows.begin() + first + nr);
  g10_layout(d10, {18, 6}, sum, 2010 + first, first);
  for (int f : {0, (int)rows.size() / 2 / 8 * 8, ((int)rows.size() - 8) / 8 * 8}) g10_time(rows, f, sum);
  std::printf("  (timing is reported, not gated; free clocks)\n");
}

// ===========================================================================
// G11 (Step 5, 2026-10-10): design A BACKWARD (gqa_flash_attn_backward_qkv with the same GqaChatPlan
// -> gqa_bwd_vj1d<MODE_CHATS>), after the A forward. Same inputs as G9 (same seeds).
//   a) dQ/dK/dV (+O/LSE) of every segment, random dO everywhere, vs the per-segment reference, 2x
//      rule; 1-token segments vs the derived bound; every output written and finite.
//   b) EVERY segment == the segment ALONE through today's dense kernels: O/LSE/dK/dV bit for bit,
//      dQ within FA's atol (atomics). A cuts its tiles at the segment start like dense at row 0.
//   c) isolation incl. dO: other segments' O/LSE/dK/dV bitwise unchanged, dQ within the atol; the
//      changed chat's dK and dV must change.
//   d) refusals (plan + seg_start, paths Vj1d cannot take, wrong shape; USE_FLASH / DET_DQ runs).
//   e) repeat x N on 8 real rows (bitwise O/LSE/dK/dV), f) backward time dense vs B vs A.
// ===========================================================================
static bool run_chats(const Inputs& in, const otc::GqaChatPlan* plan, KOut& o, std::string& err) {
  const Problem& p = in.p;
  o.ensure(p);
  o.fill_nan();
  try {
    otc::gqa_flash_attn_forward_qkv(cb(in.q.p), cb(in.k.p), cb(in.v.p), mb(o.O.p), o.LSE.p, p.B, p.Hq, p.Hkv, p.T,
                                    p.hd, true, nullptr, plan);
    otc::gqa_flash_attn_backward_qkv(cb(in.q.p), cb(in.k.p), cb(in.v.p), cb(o.O.p), cb(in.dO.p), o.LSE.p, mb(o.dQ.p),
                                     mb(o.dK.p), mb(o.dV.p), p.B, p.Hq, p.Hkv, p.T, p.hd, true, nullptr, plan);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  } catch (const std::exception& e) {
    err = e.what();
    return false;
  }
  return true;
}

static void g11_layout(const Layout& lo, const Heads& h, Summary& sum, uint64_t seed, int row_base = 0) {
  const int T = lo.T;
  Checks c;
  if (g_large_rule) {   // G13 large inputs: FA2's gate for this test (tests/test_flash_attn.py:2287-2295), as in G5
    c.dQ = Check("dQ", 5.0, 1e-3); c.dK = Check("dK", 5.0, 1e-3); c.dV = Check("dV", 5.0, 1e-3);
  }
  OneTok ot;
  unsigned long long nonfin = 0, alone_mis = 0, iso_other = 0;
  double alone_dq = 0, iso_dq_ratio = 0;
  bool alone_dq_ok = true, iso_dq_ok = true;
  int nseg = 0, n_alone = 0, iso_runs = 0, iso_unchanged = 0;
  long long wa = 0, wd = 0;
  std::string err;
  bool threw = false;
  const auto t0c = std::chrono::steady_clock::now();
  for (size_t r0 = 0; r0 < lo.rows.size() && !threw; r0 += g_rows_per_call) {
    Layout sub{lo.name, T, {}};
    for (size_t r = r0; r < std::min(lo.rows.size(), r0 + g_rows_per_call); ++r) sub.rows.push_back(lo.rows[r]);
    const Problem p{(int)sub.rows.size(), T, h.Hq, h.Hkv, 128};
    Inputs in;
    make_inputs(in, p, seed + r0);   // G9's inputs for this batch
    const std::vector<Segment> segs = packed_segments(sub);
    PlanHolder plan;
    try { build_plan(sub, p.Hq, plan.p); } catch (const std::exception& e) { err = std::string("plan: ") + e.what(); threw = true; break; }
    for (const Segment& g : segs) { const int nk = (g.L + 63) / 64; wa += (long long)nk * (nk + 1) / 2; }
    wd += (long long)p.B * (T / 64) * (T / 64 + 1) / 2;
    RefOut ref;
    reference(in, segs, ref);
    KOut ko;
    if (!run_chats(in, &plan.p, ko, err)) { threw = true; break; }
    nonfin += ko.nonfinite_all();
    for (const Segment& g : segs) {   // a)
      const std::string at = std::string(g.pad ? "pad" : "chat") + " row " + std::to_string(row_base + (int)r0 + g.b) +
                             " @" + std::to_string(g.t0) + " L=" + std::to_string(g.L);
      if (g.L == 1) one_token_region(c, ot, ko, g.b, T, in, ref, g, at, g.t0);
      else add_region(c, ko, g.b, T, g.t0, ref, g.b, T, g.t0, g.L, p, at);
      ++nseg;
    }
    std::map<int, std::vector<Segment>> al;   // b) every segment alone, pads included
    for (const Segment& g : segs) al[std::max(256, (g.L + 255) / 256 * 256)].push_back(g);
    for (auto& kv : al) {
      for (size_t c0 = 0; c0 < kv.second.size(); c0 += 8) {
        const int nb = (int)std::min<size_t>(8, kv.second.size() - c0);
        Inputs ai;
        KOut ka;
        if (!run_alone_dense(in, kv.second, c0, nb, kv.first, seed * 59 + kv.first + c0, ai, ka, err)) { threw = true; break; }
        for (int r = 0; r < nb; ++r) {
          const Segment& g = kv.second[c0 + r];
          alone_mis += bitdiff(ko.O.p, g.b, T, g.t0, ka.O.p, r, kv.first, 0, p.Hq, g.L, p.hd).mism;
          alone_mis += bitdiff(ko.LSE.p, g.b, T, g.t0, ka.LSE.p, r, kv.first, 0, p.Hq, g.L, 1).mism;
          alone_mis += bitdiff(ko.dK.p, g.b, T, g.t0, ka.dK.p, r, kv.first, 0, p.Hkv, g.L, p.hd).mism;
          alone_mis += bitdiff(ko.dV.p, g.b, T, g.t0, ka.dV.p, r, kv.first, 0, p.Hkv, g.L, p.hd).mism;
          const double atol = dq_atol(ka.dQ.p, r, kv.first, 0, p.Hq, g.L, p.hd);
          const BitDiff dq = bitdiff(ko.dQ.p, g.b, T, g.t0, ka.dQ.p, r, kv.first, 0, p.Hq, g.L, p.hd);
          alone_dq = std::max(alone_dq, atol > 0 ? dq.maxabs / atol : (dq.maxabs > 0 ? INFINITY : 0.0));
          if (dq.maxabs > atol) alone_dq_ok = false;
          ++n_alone;
        }
      }
      if (threw) break;
    }
    if (threw) break;
    const Segment* target = nullptr;   // c) isolation incl. dO
    for (int b = 0; b < p.B && !target; ++b) {
      int k2 = 0;
      for (const Segment& g : segs)
        if (g.b == b && !g.pad && ++k2 == 2) { target = &g; break; }
    }
    if (target) {
      Inputs in2;
      in2.p = p;
      in2.q.copy_from(in.q); in2.k.copy_from(in.k); in2.v.copy_from(in.v); in2.dO.copy_from(in.dO);
      const int tt0 = target->t0, tt1 = target->t0 + target->L;
      k_refill_tokens<<<grid_for((size_t)p.Hq * target->L * p.hd), 256>>>(in2.q.p, p.Hq, T, p.hd, target->b, tt0, tt1, seed ^ 0xA5A5ull);
      k_refill_tokens<<<grid_for((size_t)p.Hkv * target->L * p.hd), 256>>>(in2.k.p, p.Hkv, T, p.hd, target->b, tt0, tt1, seed ^ 0x5A5Aull);
      k_refill_tokens<<<grid_for((size_t)p.Hkv * target->L * p.hd), 256>>>(in2.v.p, p.Hkv, T, p.hd, target->b, tt0, tt1, seed ^ 0x3C3Cull);
      k_refill_tokens<<<grid_for((size_t)p.Hq * target->L * p.hd), 256>>>(in2.dO.p, p.Hq, T, p.hd, target->b, tt0, tt1, seed ^ 0x9696ull);
      KERNEL_CHECK();
      KOut k2;
      if (!run_chats(in2, &plan.p, k2, err)) { threw = true; break; }
      ++iso_runs;
      for (const Segment& g : segs) {
        const unsigned long long mk = bitdiff(ko.dK.p, g.b, T, g.t0, k2.dK.p, g.b, T, g.t0, p.Hkv, g.L, p.hd).mism;
        const unsigned long long mv = bitdiff(ko.dV.p, g.b, T, g.t0, k2.dV.p, g.b, T, g.t0, p.Hkv, g.L, p.hd).mism;
        if (&g == target) { if (mk == 0 || mv == 0) ++iso_unchanged; continue; }
        iso_other += mk + mv + bitdiff(ko.O.p, g.b, T, g.t0, k2.O.p, g.b, T, g.t0, p.Hq, g.L, p.hd).mism +
                     bitdiff(ko.LSE.p, g.b, T, g.t0, k2.LSE.p, g.b, T, g.t0, p.Hq, g.L, 1).mism;
        const double atol = dq_atol(ko.dQ.p, g.b, T, g.t0, p.Hq, g.L, p.hd);
        const BitDiff dq = bitdiff(ko.dQ.p, g.b, T, g.t0, k2.dQ.p, g.b, T, g.t0, p.Hq, g.L, p.hd);
        if (dq.maxabs > atol) iso_dq_ok = false;
        iso_dq_ratio = std::max(iso_dq_ratio, atol > 0 ? dq.maxabs / atol : (dq.maxabs > 0 ? INFINITY : 0.0));
      }
    }
  }
  char tag[140];
  std::snprintf(tag, sizeof tag, "G11 %s %d/%d", lo.name.c_str(), h.Hq, h.Hkv);
  if (threw) {
    std::printf("  %s  threw: %s  FAIL\n", tag, err.c_str());
    sum.record(false, std::string(tag) + " threw: " + err);
    return;
  }
  print_checks(tag, c, sum);
  for (Check* k : {&ot.q, &ot.k}) {
    if (k->n == 0) continue;
    std::printf("  %-44s %-4s 1-token  segs  %5d  worst |kernel|/bound %.3e  fails %d  nan %llu  %s\n", tag, k->name.c_str(),
                k->n, k->worst, k->fails, k->nan, k->ok() ? "PASS" : "FAIL");
    sum.record(k->ok(), std::string(tag) + " " + k->name + " 1-token bound");
  }
  std::printf("  %-44s %d segments, non-finite/unwritten O+LSE+dQ+dK+dV %llu %s  (%.1f s); backward q-tile visits A/dense %.1f%%\n",
              tag, nseg, nonfin, nonfin ? "FAIL" : "PASS", secs_since(t0c), wd ? 100.0 * wa / wd : 0.0);
  sum.record(nonfin == 0, std::string(tag) + " A outputs written + finite");
  const bool ok_alone = alone_mis == 0 && alone_dq_ok;
  std::printf("  %-44s %d segments: A vs alone (dense kernels) -- O/LSE/dK/dV mismatched elements %llu, worst dQ diff/atol %.3f  %s\n",
              tag, n_alone, alone_mis, alone_dq, ok_alone ? "PASS" : "FAIL");
  sum.record(ok_alone, std::string(tag) + " every segment = alone");
  if (iso_runs > 0) {
    const bool ok = iso_other == 0 && iso_unchanged == 0 && iso_dq_ok;
    std::printf("  %-44s isolation incl. dO (%d runs): other segments' O/LSE/dK/dV mismatched elements %llu, worst dQ diff/atol %.3f, changed chat's dK/dV unchanged in %d run(s)  %s\n",
                tag, iso_runs, iso_other, iso_dq_ratio, iso_unchanged, ok ? "PASS" : "FAIL");
    sum.record(ok, std::string(tag) + " isolation incl. dO");
  }
}

static void g11_repeat_and_time(const std::vector<std::vector<int>>& rows, Summary& sum) {
  for (int first : {0, (int)rows.size() / 2 / 8 * 8, ((int)rows.size() - 8) / 8 * 8}) {
    Layout lo{"D10 rows " + std::to_string(first) + "-" + std::to_string(first + 7), 4096, {}};
    lo.rows.assign(rows.begin() + first, rows.begin() + first + 8);
    const Problem p{8, 4096, 18, 6, 128};
    Inputs in;
    make_inputs(in, p, 9090 + first);
    const std::vector<int32_t> hs = build_seg_start(lo);
    DBuf<int32_t> d_seg;
    d_seg.ensure(hs.size());
    CUDA_CHECK(cudaMemcpy(d_seg.p, hs.data(), hs.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    PlanHolder pl;
    build_plan(lo, p.Hq, pl.p);
    std::string err;
    const int HB = p.B * p.Hq, HKB = p.B * p.Hkv;
    if (first == 0) {   // e) repeat on the longest rows
      const int reps = env_int("VARLEN_REPEATS", g_quick ? 50 : 1000);
      KOut base, cur;
      if (!run_chats(in, &pl.p, base, err)) { sum.record(false, "G11 repeat threw: " + err); return; }
      const double atol = dq_atol(base.dQ.p, 0, p.T, 0, HB, p.T, p.hd);
      unsigned long long m = 0;
      double wq = 0;
      for (int r = 0; r < reps; ++r) {
        if (!run_chats(in, &pl.p, cur, err)) { sum.record(false, "G11 repeat threw: " + err); return; }
        m += bitdiff(base.O.p, 0, p.T, 0, cur.O.p, 0, p.T, 0, HB, p.T, p.hd).mism + bitdiff(base.LSE.p, 0, p.T, 0, cur.LSE.p, 0, p.T, 0, HB, p.T, 1).mism +
             bitdiff(base.dK.p, 0, p.T, 0, cur.dK.p, 0, p.T, 0, HKB, p.T, p.hd).mism + bitdiff(base.dV.p, 0, p.T, 0, cur.dV.p, 0, p.T, 0, HKB, p.T, p.hd).mism;
        wq = std::max(wq, bitdiff(base.dQ.p, 0, p.T, 0, cur.dQ.p, 0, p.T, 0, HB, p.T, p.hd).maxabs);
      }
      const bool ok = m == 0 && wq <= atol;
      std::printf("  G11 repeat A x%d (%s): O/LSE/dK/dV mismatched elements %llu, dQ max diff %.3e vs atol %.3e  %s\n", reps,
                  lo.name.c_str(), m, wq, atol, ok ? "PASS" : "FAIL");
      sum.record(ok, "G11 repeat A");
    }
    // f) backward time: each backward gets its own forward's O/LSE
    KOut kd, kb, ka;
    kd.ensure(p); kb.ensure(p); ka.ensure(p);
    today_fwd(in, kd);
    otc::gqa_flash_attn_forward_qkv(cb(in.q.p), cb(in.k.p), cb(in.v.p), mb(kb.O.p), kb.LSE.p, p.B, p.Hq, p.Hkv, p.T, p.hd, true, d_seg.p);
    otc::gqa_flash_attn_forward_qkv(cb(in.q.p), cb(in.k.p), cb(in.v.p), mb(ka.O.p), ka.LSE.p, p.B, p.Hq, p.Hkv, p.T, p.hd, true, nullptr, &pl.p);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto bwd = [&](KOut& o, const int32_t* sg, const otc::GqaChatPlan* pp) {
      otc::gqa_flash_attn_backward_qkv(cb(in.q.p), cb(in.k.p), cb(in.v.p), cb(o.O.p), cb(in.dO.p), o.LSE.p, mb(o.dQ.p), mb(o.dK.p),
                                       mb(o.dV.p), p.B, p.Hq, p.Hkv, p.T, p.hd, true, sg, pp);
    };
    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    auto timed = [&](const std::function<void()>& f) {
      float ms = 0;
      CUDA_CHECK(cudaEventRecord(a)); f(); CUDA_CHECK(cudaEventRecord(b));
      CUDA_CHECK(cudaEventSynchronize(b)); CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
      return ms;
    };
    std::vector<float> td, tb, ta;
    for (int i = 0; i < 5; ++i) { bwd(kd, nullptr, nullptr); bwd(kb, d_seg.p, nullptr); bwd(ka, nullptr, &pl.p); }
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int i = 0; i < 20; ++i) {
      td.push_back(timed([&] { bwd(kd, nullptr, nullptr); }));
      tb.push_back(timed([&] { bwd(kb, d_seg.p, nullptr); }));
      ta.push_back(timed([&] { bwd(ka, nullptr, &pl.p); }));
    }
    auto med = [](std::vector<float> v) { std::sort(v.begin(), v.end()); return v.size() % 2 ? (double)v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]); };
    const auto bw = bwd_tiles(hs, p.B, p.T);
    long long wa = 0;
    for (const Segment& g : packed_segments(lo)) { const int nk = (g.L + 63) / 64; wa += (long long)nk * (nk + 1) / 2; }
    const double md = med(td), mbb = med(tb), maa = med(ta);
    std::printf("  G11 %-16s work (q-tile visits vs dense): B %.1f%%, A %.1f%% | bwd median: dense %.4f ms, B %.4f ms (%.1f%%), A %.4f ms (%.1f%%)\n",
                lo.name.c_str(), 100.0 * bw.first / bw.second, 100.0 * wa / bw.second, md, mbb, 100.0 * mbb / md, maa, 100.0 * maa / md);
    CUDA_CHECK(cudaEventDestroy(a));
    CUDA_CHECK(cudaEventDestroy(b));
  }
  std::printf("  (timing is reported, not gated; free clocks)\n");
}

static void g11_chats_backward(const std::vector<int>* Lreal, Summary& sum, bool kernels_ok) {
  std::printf("\n===== G11 design A backward: per-segment plan vs reference, alone, isolation (G9's inputs) =====\n");
  auto refuse = [&](const char* name, const char* expect, int B, int Hq, int T, int hd, bool causal, bool with_seg, int planB) {
    const Problem p{B, T, Hq, 1, hd};
    Inputs in;
    make_inputs(in, p, 78);
    KOut o;
    o.ensure(p);
    PlanHolder pl;
    std::vector<int32_t> bb, t0, L;
    for (int r = 0; r < planB; ++r) { bb.push_back(r); t0.push_back(0); L.push_back(T); }
    otc::gqa_chat_plan_build(pl.p, bb.data(), t0.data(), L.data(), planB, planB, T, Hq);
    DBuf<int32_t> ds;
    std::vector<int32_t> hsg((size_t)B * T, 0);
    ds.ensure(hsg.size());
    CUDA_CHECK(cudaMemcpy(ds.p, hsg.data(), hsg.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    g10_refuse(sum, (std::string("bwd ") + name).c_str(), expect, [&] {
      otc::gqa_flash_attn_backward_qkv(cb(in.q.p), cb(in.k.p), cb(in.v.p), cb(o.O.p), cb(in.dO.p), o.LSE.p, mb(o.dQ.p), mb(o.dK.p),
                                       mb(o.dV.p), B, Hq, 1, T, hd, causal, with_seg ? ds.p : nullptr, &pl.p);
    });
  };
  if (!kernels_ok) { refuse("this GPU / BLUTRAIN_NO_SM90[_GQA]", "chats", 1, 2, 256, 128, true, false, 1); return; }
  const std::string mode = bwd_mode_name();
  if (mode.rfind("Vj1d", 0) != 0) { refuse(mode.c_str(), "chats", 1, 2, 256, 128, true, false, 1); return; }
  refuse("plan AND seg_start", "not both", 1, 2, 256, 128, true, true, 1);
  refuse("is_causal = false", "chats", 1, 2, 256, 128, false, false, 1);
  refuse("head_dim 64", "chats", 1, 2, 256, 64, true, false, 1);
  refuse("T = 288 (not a multiple of 64)", "chats", 1, 2, 288, 128, true, false, 1);
  refuse("plan for B=1 used with B=2", "another shape", 2, 2, 256, 128, true, false, 1);
  std::vector<Layout> los = {
      {"D2 2048+2048", 4096, {{2048, 2048}}},
      {"D3 1000+3096", 4096, {{1000, 3096}}},
      {"D4 1+4095, 4095+1", 4096, {{1, 4095}, {4095, 1}}},
      {"D5 64x64, 32x128", 4096, {std::vector<int>(64, 64), std::vector<int>(32, 128)}},
      {"D6q 512 x 1 + pad", 4096, {std::vector<int>(512, 1)}},
      make_D7(),
      make_D8(),
      {"D9 all pad", 4096, {{}}},
      {"D11 offsets", 4096, {{300}, {1, 300}, {63, 300}, {64, 300}, {65, 300}, {127, 300}, {128, 300}, {3796, 300}}},
  };
  std::vector<Heads> hs = kHeads;
  if (g_quick) hs = {{18, 6}};
  uint64_t seed = 900;   // G9's seeds: the same inputs, layout by layout
  for (const Layout& lo : los)
    for (const Heads& h : hs) {
      const uint64_t sd = seed++;
      if (g_case.empty() || lo.name.find(g_case) != std::string::npos) g11_layout(lo, h, sum, sd);
    }
  if (!g_case.empty() && g_case.find("D10") == std::string::npos) return;
  if (!Lreal) { sum.skip_("G11 D10 real stage-0 rows + repeat + timing (labels.bin not found)"); return; }
  const std::vector<std::vector<int>> rows = pack_ffd(*Lreal, 4096);
  const int first = std::max(0, env_int("VARLEN_D10_FIRST", 0)) / 8 * 8;
  size_t nr = rows.size() - std::min<size_t>(first, rows.size());
  const int lim = env_int("VARLEN_D10_ROWS", g_quick ? 16 : 0);
  if (lim > 0) nr = std::min(nr, (size_t)lim);
  Layout d10{"D10 stage-0 packed (rows " + std::to_string(first) + "-" + std::to_string(first + nr - 1) + ")", 4096, {}};
  d10.rows.assign(rows.begin() + first, rows.begin() + first + nr);
  g11_layout(d10, {18, 6}, sum, 1910 + first, first);
  g11_repeat_and_time(rows, sum);
}

// ===========================================================================
// BENCH (Step 6, 2026-10-12): our three modes timed on fixed layouts, for the cross-library comparison.
//   layouts: 3 sets of 8 real D10 rows (first = longest chats, middle, last = shortest) and synthetic
//   rows where every chat has the same length (512 / 1024 / 2048 / 4096), B = 8, T = 4096, 18/6, hd 128.
//   per layout: dense (full-row causal), B (seg_start), A (plan); forward and backward timed separately,
//   10 warm-up + VARLEN_BENCH_ITERS (default 50) interleaved rounds, cudaEvent around each call.
//   useful FLOPs = the real per-segment causal attention only, FA's formula per segment:
//   fwd = 4 * Hq * hd * L^2 / 2 (hopper/benchmark_attn.py:62-73), bwd = 2.5 x fwd (:387). Dense is
//   charged the same useful FLOPs (it computes more, so its TFLOP/s reads as useful work per second).
//   VARLEN_BENCH_EXPORT=<file>: writes "layout b t0 L" lines so the Python side packs the same segments.
//   Output lines "BENCH layout=.. impl=.. dir=.. med_ms= p10_ms= p90_ms= tflops=" (same format in Python).
// ===========================================================================
static void bench_mode(const std::vector<int>* Lreal, Summary& sum) {
  std::printf("\n===== BENCH: ours (dense / B / A), fwd and bwd, interleaved =====\n");
  const int iters = env_int("VARLEN_BENCH_ITERS", 50);
  std::vector<Layout> los;
  if (Lreal) {
    const std::vector<std::vector<int>> rows = pack_ffd(*Lreal, 4096);
    for (int f : {0, (int)rows.size() / 2 / 8 * 8, ((int)rows.size() - 8) / 8 * 8}) {
      Layout lo{"real_rows_" + std::to_string(f) + "-" + std::to_string(f + 7), 4096, {}};
      lo.rows.assign(rows.begin() + f, rows.begin() + f + 8);
      los.push_back(lo);
    }
  } else {
    sum.skip_("BENCH real rows (labels.bin not found)");
  }
  for (int L : {512, 1024, 2048, 4096}) los.push_back({"uniform_" + std::to_string(L), 4096, std::vector<std::vector<int>>(8, std::vector<int>(4096 / L, L))});
  FILE* ex = nullptr;
  if (const char* e = std::getenv("VARLEN_BENCH_EXPORT")) ex = std::fopen(e, "w");
  for (const Layout& lo : los) {
    if (!g_case.empty() && lo.name.find(g_case) == std::string::npos) continue;   // VARLEN_CASE: one layout (profiling)
    const Problem p{8, 4096, 18, 6, 128};
    Inputs in;
    make_inputs(in, p, 6060);
    const std::vector<Segment> segs = packed_segments(lo);
    double fl = 0;
    for (const Segment& g : segs) fl += 4.0 * p.Hq * p.hd * (double)g.L * g.L / 2.0;
    if (ex) for (const Segment& g : segs) std::fprintf(ex, "%s %d %d %d\n", lo.name.c_str(), g.b, g.t0, g.L);
    const std::vector<int32_t> hs = build_seg_start(lo);
    DBuf<int32_t> d_seg;
    d_seg.ensure(hs.size());
    CUDA_CHECK(cudaMemcpy(d_seg.p, hs.data(), hs.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    PlanHolder pl;
    build_plan(lo, p.Hq, pl.p);
    KOut kd, kb, ka;
    kd.ensure(p); kb.ensure(p); ka.ensure(p);
    auto fwd = [&](KOut& o, const int32_t* sg, const otc::GqaChatPlan* pp) {
      otc::gqa_flash_attn_forward_qkv(cb(in.q.p), cb(in.k.p), cb(in.v.p), mb(o.O.p), o.LSE.p, p.B, p.Hq, p.Hkv, p.T, p.hd, true, sg, pp);
    };
    auto bwd = [&](KOut& o, const int32_t* sg, const otc::GqaChatPlan* pp) {
      otc::gqa_flash_attn_backward_qkv(cb(in.q.p), cb(in.k.p), cb(in.v.p), cb(o.O.p), cb(in.dO.p), o.LSE.p, mb(o.dQ.p), mb(o.dK.p),
                                       mb(o.dV.p), p.B, p.Hq, p.Hkv, p.T, p.hd, true, sg, pp);
    };
    struct Impl { const char* name; KOut* o; const int32_t* sg; const otc::GqaChatPlan* pp; std::vector<float> tf, tb; };
    std::vector<Impl> im = {{"ours_dense_fullrow", &kd, nullptr, nullptr, {}, {}},
                            {"ours_B_segstart", &kb, d_seg.p, nullptr, {}, {}},
                            {"ours_A_plan", &ka, nullptr, &pl.p, {}, {}}};
    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    auto timed = [&](const std::function<void()>& f) {
      float ms = 0;
      CUDA_CHECK(cudaEventRecord(a)); f(); CUDA_CHECK(cudaEventRecord(b));
      CUDA_CHECK(cudaEventSynchronize(b)); CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
      return ms;
    };
    try {
      for (int w = 0; w < 10; ++w) for (Impl& x : im) { fwd(*x.o, x.sg, x.pp); bwd(*x.o, x.sg, x.pp); }
      CUDA_CHECK(cudaDeviceSynchronize());
      for (int it = 0; it < iters; ++it) {
        for (Impl& x : im) x.tf.push_back(timed([&] { fwd(*x.o, x.sg, x.pp); }));
        for (Impl& x : im) x.tb.push_back(timed([&] { bwd(*x.o, x.sg, x.pp); }));
      }
    } catch (const std::exception& e) {
      sum.record(false, std::string("BENCH threw: ") + e.what());
      return;
    }
    auto q = [](std::vector<float> v, double f) { std::sort(v.begin(), v.end()); return (double)v[std::min(v.size() - 1, (size_t)(f * (v.size() - 1) + 0.5))]; };
    for (Impl& x : im)
      for (int d = 0; d < 2; ++d) {
        const std::vector<float>& t = d ? x.tb : x.tf;
        const double med = q(t, 0.5), flops = d ? 2.5 * fl : fl;
        std::printf("BENCH layout=%s impl=%s dir=%s med_ms=%.4f p10_ms=%.4f p90_ms=%.4f tflops=%.1f\n", lo.name.c_str(), x.name,
                    d ? "bwd" : "fwd", med, q(t, 0.1), q(t, 0.9), flops / (med * 1e-3) / 1e12);
      }
    CUDA_CHECK(cudaEventDestroy(a));
    CUDA_CHECK(cudaEventDestroy(b));
  }
  if (ex) std::fclose(ex);
  std::printf("  (free clocks; record nvidia-smi clocks next to it)\n");
}

// BENCHDENSE (Step 6a, 2026-10-12): our dense causal kernels (v62 fwd / Vj1d bwd) on the shapes of the
// colleagues' cuDNN comparison (T = 4096, hd 128, B 2/4/8 x heads 12/4, 16/4, 24/8, 32/8) plus our SFT
// shape 18/6 B=8. Same timing rules as BENCH; FLOPs = FA's causal formula 4*B*Hq*hd*T^2/2 (bwd 2.5x).
static const int kDenseShapes[][3] = {{2, 12, 4}, {4, 12, 4}, {8, 12, 4}, {2, 16, 4}, {4, 16, 4}, {8, 16, 4}, {2, 24, 8},
                                      {4, 24, 8}, {8, 24, 8}, {2, 32, 8}, {4, 32, 8}, {8, 32, 8}, {8, 18, 6}};
static void bench_dense_mode(Summary& sum) {
  std::printf("\n===== BENCHDENSE: ours dense causal, colleagues' shapes =====\n");
  const int iters = env_int("VARLEN_BENCH_ITERS", 50);
  for (const auto& sh : kDenseShapes) {
    const Problem p{sh[0], 4096, sh[1], sh[2], 128};
    char name[64];
    std::snprintf(name, sizeof name, "dense_B%d_H%dx%d", sh[0], sh[1], sh[2]);
    if (!g_case.empty() && std::string(name) != g_case) continue;   // VARLEN_CASE: one shape (profiling)
    Inputs in;
    make_inputs(in, p, 6161);
    KOut ko;
    ko.ensure(p);
    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    auto timed = [&](const std::function<void()>& f) {
      float ms = 0;
      CUDA_CHECK(cudaEventRecord(a)); f(); CUDA_CHECK(cudaEventRecord(b));
      CUDA_CHECK(cudaEventSynchronize(b)); CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
      return ms;
    };
    std::vector<float> tf, tb;
    try {
      for (int w = 0; w < 10; ++w) { today_fwd(in, ko); today_bwd(in, ko); }
      CUDA_CHECK(cudaDeviceSynchronize());
      for (int it = 0; it < iters; ++it) {
        tf.push_back(timed([&] { today_fwd(in, ko); }));
        tb.push_back(timed([&] { today_bwd(in, ko); }));
      }
    } catch (const std::exception& e) {
      sum.record(false, std::string("BENCHDENSE threw: ") + e.what());
      return;
    }
    auto q = [](std::vector<float> v, double f) { std::sort(v.begin(), v.end()); return (double)v[std::min(v.size() - 1, (size_t)(f * (v.size() - 1) + 0.5))]; };
    const double fl = 4.0 * p.B * p.Hq * p.hd * (double)p.T * p.T / 2.0;
    for (int d = 0; d < 2; ++d) {
      const std::vector<float>& t = d ? tb : tf;
      const double med = q(t, 0.5);
      std::printf("BENCH layout=%s impl=ours_dense dir=%s med_ms=%.4f p10_ms=%.4f p90_ms=%.4f tflops=%.1f\n", name, d ? "bwd" : "fwd",
                  med, q(t, 0.1), q(t, 0.9), (d ? 2.5 : 1.0) * fl / (med * 1e-3) / 1e12);
    }
    CUDA_CHECK(cudaEventDestroy(a));
    CUDA_CHECK(cudaEventDestroy(b));
  }
}

// ===========================================================================
// G12 (Step 6, 2026-10-12): edge cases G8-G11 never covered, run through the SAME full checks
// (g9_layout = design B fwd+bwd, g11_layout = design A fwd+bwd: per-segment accuracy vs the reference,
// 1-token bound, A == alone bitwise, isolation incl. dO, outputs written + finite):
//   - row lengths other than 4096: T = 1024, 2048, 8192;
//   - one chat of 16384 tokens (2/1 heads, to keep the fp64 reference small): B's backward then needs a
//     2nd pass of its __syncthreads_count prefix loop (more than 128 q-tiles after a key tile) -- the
//     only path of that loop no test reached before;
//   - a whole row of 4096 one-token chats (G8-G11 used 512): the largest unit/tile counts.
// Tags start "G9 [G12]" / "G11 [G12]", so no known-list entry can match them.
// ===========================================================================
static void g12_edges(Summary& sum) {
  std::printf("\n===== G12 edge cases: other row lengths, a 16384-token chat, 4096 one-token chats (B and A) =====\n");
  struct EC { Layout lo; Heads h; };
  std::vector<EC> cs = {
      {{"[G12] T1024 300+724", 1024, {{300, 724}}}, {18, 6}},
      {{"[G12] T1024 1+1023 / 64x16 / all pad", 1024, {{1, 1023}, std::vector<int>(16, 64), {}}}, {18, 6}},
      {{"[G12] T2048 777+1271 / 129x15 + pad", 2048, {{777, 1271}, std::vector<int>(15, 129)}}, {18, 6}},
      {{"[G12] T8192 5000+3192 / 8192 / 4096x2", 8192, {{5000, 3192}, {8192}, {4096, 4096}}}, {18, 6}},
      {{"[G12] T8192 offsets 1/63/65/127 + 8000", 8192, {{1, 8000}, {63, 8000}, {65, 8000}, {127, 8000}}}, {18, 6}},
      {{"[G12] T16384 one chat (2nd prefix pass)", 16384, {{16384}}}, {2, 1}},
      {{"[G12] T16384 100 + 16000 + pad", 16384, {{100, 16000}}}, {2, 1}},
      {{"[G12] D6 4096 x 1", 4096, {std::vector<int>(4096, 1)}}, {18, 6}},
  };
  uint64_t seed = 1200;
  for (const EC& c : cs) {
    const uint64_t sd = seed++;
    if (!g_case.empty() && c.lo.name.find(g_case) == std::string::npos) continue;
    g9_layout(c.lo, c.h, sum, sd);    // design B
    g11_layout(c.lo, c.h, sum, sd);   // design A (same inputs)
  }
}

// G13 (Step 6, 2026-10-12): the last two gaps of the coverage audit, through the same full B and A checks.
//   a) large inputs on packed rows (q*5, k*3, v*3, as G5 does for dense rows): gradients gated with FA2's rule
//      for this test (5x + 1e-3); O/LSE keep the 2x rule; A == alone and isolation stay exact.
//   b) 60 packed rows in ONE call (T = 1024, random chats with pad tails), like G4's 60-row dense case; the
//      other packed groups call with at most 8 rows.
static void g13_more(const std::vector<int>* Lreal, Summary& sum) {
  std::printf("\n===== G13: large inputs on packed rows; 60 packed rows in one call =====\n");
  std::vector<Layout> big = {make_D8(), {"D11 offsets", 4096, {{300}, {1, 300}, {63, 300}, {64, 300}, {65, 300}, {127, 300}, {128, 300}, {3796, 300}}}};
  if (Lreal) {
    const std::vector<std::vector<int>> rows = pack_ffd(*Lreal, 4096);
    Layout d{"D10 rows 752-767", 4096, {}};
    d.rows.assign(rows.begin() + 752, rows.begin() + 768);
    big.push_back(d);
  }
  g_in_qs = 5.f; g_in_ks = 3.f; g_in_vs = 3.f; g_large_rule = true;
  uint64_t seed = 1300;
  for (Layout lo : big) {
    lo.name = "[G13 large q*5 k*3 v*3] " + lo.name;
    g9_layout(lo, {18, 6}, sum, seed);
    g11_layout(lo, {18, 6}, sum, seed);
    ++seed;
  }
  g_in_qs = g_in_ks = g_in_vs = 1.f; g_large_rule = false;
  Layout m{"[G13] 60 rows x 1024 in one call", 1024, {}};
  std::mt19937 rng(1360);
  std::uniform_int_distribution<int> U(1, 600), P(0, 64);
  for (int r = 0; r < 60; ++r) {
    std::vector<int> row;
    int rem = 1024 - P(rng);
    while (rem > 0) { const int L = std::min(U(rng), rem); row.push_back(L); rem -= L; }
    m.rows.push_back(row);
  }
  g_rows_per_call = 60;
  g9_layout(m, {18, 6}, sum, 1399);
  g11_layout(m, {18, 6}, sum, 1399);
  g_rows_per_call = 8;
}

// ===========================================================================
// G7: timing of today's kernels (free clocks; FA's FLOP formula)
// ===========================================================================
static void g7_timing(Summary& sum) {
  std::printf("\n===== G7 timing: today's kernels, 5 warm-up + 20 timed, cudaEvent =====\n");
  for (const Heads& h : std::vector<Heads>{{18, 6}, {12, 4}}) {
    const Problem p{8, 4096, h.Hq, h.Hkv, 128};
    Inputs in;
    make_inputs(in, p, 7070 + h.Hq);
    KOut ko;
    ko.ensure(p);
    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    std::vector<float> tf, tb;
    try {
      for (int i = 0; i < 5; ++i) { today_fwd(in, ko); today_bwd(in, ko); }
      CUDA_CHECK(cudaDeviceSynchronize());
      for (int i = 0; i < 20; ++i) {
        float ms = 0;
        CUDA_CHECK(cudaEventRecord(a));
        today_fwd(in, ko);
        CUDA_CHECK(cudaEventRecord(b));
        CUDA_CHECK(cudaEventSynchronize(b));
        CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
        tf.push_back(ms);
        CUDA_CHECK(cudaEventRecord(a));
        today_bwd(in, ko);
        CUDA_CHECK(cudaEventRecord(b));
        CUDA_CHECK(cudaEventSynchronize(b));
        CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
        tb.push_back(ms);
      }
    } catch (const std::exception& e) {
      sum.record(false, std::string("G7 threw: ") + e.what());
      continue;
    }
    auto stats = [](std::vector<float> v, double& med, double& mean, double& mn) {
      std::sort(v.begin(), v.end());
      med = v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
      mean = 0;
      for (float x : v) mean += x;
      mean /= v.size();
      mn = v.front();
    };
    double fm, fa, fn, bm, ba, bn;
    stats(tf, fm, fa, fn);
    stats(tb, bm, ba, bn);
    const double flop = (double)p.B * p.Hq * 2.0 * p.T * (p.T / 2.0) * (2.0 * p.hd);
    std::printf("  B=8 T=4096 %d/%d  fwd median %.4f ms (mean %.4f, min %.4f) = %.1f TFLOP/s | bwd median %.4f ms (mean %.4f, min %.4f) = %.1f TFLOP/s\n",
                h.Hq, h.Hkv, fm, fa, fn, flop / (fm * 1e-3) / 1e12, bm, ba, bn, 2.5 * flop / (bm * 1e-3) / 1e12);
    CUDA_CHECK(cudaEventDestroy(a));
    CUDA_CHECK(cudaEventDestroy(b));
  }
  std::printf("  (timing is reported, not gated; record nvidia-smi clocks next to it)\n");
}

// ===========================================================================
int main() {
  g_quick = env_int("VARLEN_QUICK", 0) != 0;
  g_diag_row = env_int("VARLEN_DIAG_ROW", -1);
  g_diag_t0 = env_int("VARLEN_DIAG_T0", -1);
  g_diag_rep = env_int("VARLEN_DIAG_REPEAT", 50);
  if (const char* e = std::getenv("VARLEN_DIAG_CASE")) g_diag_case = e;
  if (const char* e = std::getenv("VARLEN_CASE")) g_case = e;
  if (const char* e = std::getenv("VARLEN_DIAG_LIST")) {
    std::string t = e;
    for (size_t a = 0; a < t.size();) {
      size_t b = t.find(',', a);
      if (b == std::string::npos) b = t.size();
      const std::string it = t.substr(a, b - a);
      const size_t c = it.find(':');
      if (c != std::string::npos) g_diag_list.push_back({std::atoi(it.substr(0, c).c_str()), std::atoi(it.substr(c + 1).c_str())});
      a = b + 1;
    }
  }
  const char* only_env = std::getenv("VARLEN_ONLY");
  g_kv_deterministic = bwd_mode_name().rfind("Vj1 (", 0) != 0;   // only Vj1 (USE_FLASH=1) is non-deterministic
  const std::string only = only_env ? only_env : "";   // "D10": run only G6 D10 (diagnostics)
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
  const bool hopper = otc::get_arch() == otc::ArchFamily::Hopper;
  const bool sm90_on = !otc::disable_sm90_gqa();
  const bool kernels_ok = hopper && sm90_on;
  std::printf("===== test_varlen_attention (v2 Step 2 harness) =====\n");
  std::printf("GPU %d: %s  sm_%d%d  %d SMs  |  arch family Hopper: %s  |  BLUTRAIN_NO_SM90[_GQA]: %s\n", dev,
              prop.name, prop.major, prop.minor, prop.multiProcessorCount, hopper ? "yes" : "no",
              sm90_on ? "off" : "ON");
  std::printf("backward mode: %s  |  quick: %s\n", bwd_mode_name().c_str(), g_quick ? "yes" : "no");

  Summary sum;
  const auto t_all = std::chrono::steady_clock::now();
  if (only.empty()) g1_selftest(sum);

  std::vector<int> Lreal;
  std::string why;
  const bool have_real = read_stage0(Lreal, why);
  if (!have_real) std::printf("\nstage-0 data: %s\n", why.c_str());

  if (only == "D10") {
    std::printf("\n===== VARLEN_ONLY=D10 (diagnostic run: G6 D10 only) =====\n");
    if (kernels_ok) g6_d10(have_real ? &Lreal : nullptr, sum);
    else sum.skip_("G6 D10: kernels under test are Hopper-only");
  } else if (only == "G8") {
    std::printf("\n===== VARLEN_ONLY=G8 (Step 3: design B forward only) =====\n");
    if (kernels_ok) g8_packed_forward(have_real ? &Lreal : nullptr, sum);
    else sum.skip_("G8: kernels under test are Hopper-only");
  } else if (only == "G7") {   // timing only: one pass is seconds, so an A/B can alternate libraries
    std::printf("\n===== VARLEN_ONLY=G7 (timing only) =====\n");
    if (kernels_ok) g7_timing(sum);
    else sum.skip_("G7: kernels under test are Hopper-only");
  } else if (only == "CLASSIC") {   // G1-G7 only (the Step 2 groups), for running groups as parallel processes
    std::printf("\n===== VARLEN_ONLY=CLASSIC (G1-G7) =====\n");
    g1_selftest(sum);
    if (kernels_ok) {
      g2_dense(sum);
      if (have_real) g3_v1_rows(Lreal, sum);
      else sum.skip_("G3 v1 rows: " + why);
      g4_repeat(sum);
      g5_large(sum);
      g6_chat_alone(have_real ? &Lreal : nullptr, sum);
      g6_d10(have_real ? &Lreal : nullptr, sum);
      g7_timing(sum);
    } else {
      sum.skip_("G2-G7: kernels under test are Hopper-only");
    }
  } else if (only == "G13") {
    if (kernels_ok) g13_more(have_real ? &Lreal : nullptr, sum);
    else sum.skip_("G13: kernels under test are Hopper-only");
  } else if (only == "G12") {
    std::printf("\n===== VARLEN_ONLY=G12 (Step 6: edge cases) =====\n");
    if (kernels_ok) g12_edges(sum);
    else sum.skip_("G12: kernels under test are Hopper-only");
  } else if (only == "BENCHDENSE") {
    if (kernels_ok) bench_dense_mode(sum);
    else sum.skip_("BENCHDENSE: kernels under test are Hopper-only");
  } else if (only == "BENCH") {
    if (kernels_ok) bench_mode(have_real ? &Lreal : nullptr, sum);
    else sum.skip_("BENCH: kernels under test are Hopper-only");
  } else if (only == "G11") {
    std::printf("\n===== VARLEN_ONLY=G11 (Step 5: design A backward) =====\n");
    g11_chats_backward(have_real ? &Lreal : nullptr, sum, kernels_ok);
  } else if (only == "G10") {
    std::printf("\n===== VARLEN_ONLY=G10 (Step 5: design A forward) =====\n");
    g10_chats_forward(have_real ? &Lreal : nullptr, sum, kernels_ok);
  } else if (only == "G9") {
    std::printf("\n===== VARLEN_ONLY=G9 (Step 4: design B backward) =====\n");
    g9_packed_backward(have_real ? &Lreal : nullptr, sum, kernels_ok);
  } else if (!only.empty()) {
    sum.record(false, "unknown VARLEN_ONLY=" + only + " (supported: CLASSIC, D10, G7, G8, G9, G10, G11, G12, G13, BENCH, BENCHDENSE)");
  } else if (!kernels_ok) {
    const std::string r = !hopper ? std::string("kernels under test are Hopper-only (v62/Vj1d); this GPU is ") + prop.name
                                  : "BLUTRAIN_NO_SM90[_GQA] is set";
    for (const char* g : {"G2", "G3", "G4", "G5", "G6", "G7", "G8"}) sum.skip_(std::string(g) + ": " + r);
    g9_packed_backward(nullptr, sum, /*kernels_ok=*/false);   // only the refusals; they must hold here too
    g10_chats_forward(nullptr, sum, /*kernels_ok=*/false);
    g11_chats_backward(nullptr, sum, /*kernels_ok=*/false);
  } else {
    g2_dense(sum);
    if (have_real) g3_v1_rows(Lreal, sum);
    else sum.skip_("G3 v1 rows: " + why);
    g4_repeat(sum);
    g5_large(sum);
    g6_chat_alone(have_real ? &Lreal : nullptr, sum);
    g6_d10(have_real ? &Lreal : nullptr, sum);
    g8_packed_forward(have_real ? &Lreal : nullptr, sum);
    g9_packed_backward(have_real ? &Lreal : nullptr, sum, kernels_ok);
    g10_chats_forward(have_real ? &Lreal : nullptr, sum, kernels_ok);
    g11_chats_backward(have_real ? &Lreal : nullptr, sum, kernels_ok);
    g7_timing(sum);
  }

  std::printf("\n===== SUMMARY =====\n");
  std::printf("checks passed %d, failed %d, skipped groups %d   (%.0f s total)\n", sum.pass, sum.fail, sum.skip,
              secs_since(t_all));
  for (const std::string& f : sum.failed) std::printf("  FAIL  %s\n", f.c_str());
  for (const std::string& s : sum.skipped) std::printf("  SKIP  %s\n", s.c_str());
  std::printf("  KNOWN issues observed this run: %zu (kept by decision, watched in every run)\n", sum.known.size());
  for (const std::string& k : sum.known) std::printf("  KNOWN %s\n", k.c_str());
  std::printf("  PENDING (Step 5; design B is covered by G8 (forward) and G9 (backward)):\n"
              "    G9 mode refusals need their own runs: USE_FLASH=1 and USE_DETERMINISTIC_DQ=1 with VARLEN_ONLY=G9\n"
              "    G11 mode refusals need their own runs: USE_FLASH=1 and USE_DETERMINISTIC_DQ=1 with VARLEN_ONLY=G11\n"
              "    empty-row contract (A's pad rows): O = 0, dQ/dK/dV = 0, fixed LSE value\n"
              "    tile-count check (debug build): tiles processed == host model\n");
  std::printf("%s\n", sum.fail != 0 ? "FAILURES ABOVE (NEW, not on the known list)"
                      : sum.known.empty() ? "ALL RUN CHECKS PASS"
                                          : "ALL RUN CHECKS PASS (known issues observed: see KNOWN above)");
  return sum.fail == 0 ? 0 : 1;
}
