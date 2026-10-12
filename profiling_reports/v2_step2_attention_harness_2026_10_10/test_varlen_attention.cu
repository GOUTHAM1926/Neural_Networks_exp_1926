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
//   PENDING (need the B / A kernels, Steps 3-5) are printed in the SUMMARY.
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

static void make_inputs(Inputs& in, const Problem& p, uint64_t seed, float qs = 1.f, float ks = 1.f,
                        float vs = 1.f) {
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
};
static const KnownIssue* match_known(const std::string& tag, const std::string& tensor, const std::string& at,
                                     double ratio, unsigned long long nan) {
  if (nan != 0) return nullptr;
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
                               const bf16* dQk, int Tk, int r, int Hq, int Hkv, int G, double scale, double c,
                               double gamma, double* dsb, unsigned long long* out) {
  const int h = blockIdx.x, hkv = h / G, d = threadIdx.x, hd = blockDim.x;
  __shared__ double s1[256], s2[256];
  const size_t iq = (((size_t)b * Hq + h) * Tin + t) * hd + d;
  const size_t ikv = (((size_t)b * Hkv + hkv) * Tin + t) * hd + d;
  const size_t io = ((size_t)r * Hq + h) * Tk * hd + d;   // kernel outputs: row r, token 0
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
__global__ void k_one_token_dk(const bf16* q, int Tin, int b, int t, const bf16* dKk, int Tk, int r, int Hq, int Hkv,
                               int G, double scale, double c, const double* dsb, unsigned long long* out) {
  const int hkv = blockIdx.x, d = threadIdx.x, hd = blockDim.x;
  double acc = 0;
  for (int g = 0; g < G; ++g) {
    const int h = hkv * G + g;
    acc += fabs((double)__bfloat162float(q[(((size_t)b * Hq + h) * Tin + t) * hd + d])) * dsb[h];
  }
  const double bound = c * scale * acc;
  const double x = (double)__bfloat162float(dKk[((size_t)r * Hkv + hkv) * Tk * hd + d]);
  if (!isfinite(x)) { atomicAdd(&out[2], 1ull); return; }
  const double ax = fabs(x), ratio = bound > 0 ? ax / bound : (ax > 0 ? INFINITY : 0.0);
  atomicMax(&out[0], (unsigned long long)__double_as_longlong(ratio));
  atomicMax(&out[1], (unsigned long long)__double_as_longlong(ax));
}

struct OneTok {
  Check q{"dQ"}, k{"dK"};
  int n = 0, not_zero = 0;   // not_zero: chats whose truth was not exactly 0 (they fall back to the 2x rule)
};
// Kernel outputs: row r, token 0 of a batch with row length Tk. Inputs/reference: segment s of `in`.
static void one_token_region(Checks& c, OneTok& ot, const KOut& ko, int r, int Tk, const Inputs& in,
                             const RefOut& ref, const Segment& s, const std::string& at) {
  const Problem& p = in.p;
  c.O.add(compare_region(ko.O.p, r, Tk, 0, ref.O.p, ref.Of.p, s.b, p.T, s.t0, p.Hq, 1, p.hd), at);
  c.LSE.add(compare_region(ko.LSE.p, r, Tk, 0, ref.LSE.p, ref.LSEf.p, s.b, p.T, s.t0, p.Hq, 1, 1), at);
  c.dV.add(compare_region(ko.dV.p, r, Tk, 0, ref.dV.p, ref.dVf.p, s.b, p.T, s.t0, p.Hkv, 1, p.hd), at);
  ++ot.n;
  if (maxabs_region(ref.dQ.p, s.b, p.T, s.t0, p.Hq, 1, p.hd) != 0.0 ||
      maxabs_region(ref.dK.p, s.b, p.T, s.t0, p.Hkv, 1, p.hd) != 0.0) {
    ++ot.not_zero;   // not the exact-zero case after all: the normal rule decides
    c.dQ.add(compare_region(ko.dQ.p, r, Tk, 0, ref.dQ.p, ref.dQf.p, s.b, p.T, s.t0, p.Hq, 1, p.hd), at);
    c.dK.add(compare_region(ko.dK.p, r, Tk, 0, ref.dK.p, ref.dKf.p, s.b, p.T, s.t0, p.Hkv, 1, p.hd), at);
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
  k_one_token_dq<<<p.Hq, p.hd>>>(in.v.p, in.k.p, in.dO.p, p.T, s.b, s.t0, ko.O.p, ko.dQ.p, Tk, r, p.Hq, p.Hkv,
                                 p.G(), scale, cfac, gamma, dsb, d);
  k_one_token_dk<<<p.Hkv, p.hd>>>(in.q.p, p.T, s.b, s.t0, ko.dK.p, Tk, r, p.Hq, p.Hkv, p.G(), scale, cfac, dsb, d + 3);
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
static std::string bwd_mode_name() {
  auto on = [](const char* n) { const char* e = std::getenv(n); return e && e[0] == '1'; };
  const bool f = on("USE_FLASH"), d = on("USE_DETERMINISTIC_DQ");
  if (f && d) return "Vj1d (both USE_FLASH and USE_DETERMINISTIC_DQ set -> dispatcher falls back to Vj1d)";
  if (f) return "Vj1 (USE_FLASH=1)";
  if (d) return "v44 (USE_DETERMINISTIC_DQ=1)";
  return "Vj1d (default)";
}
static void g4_repeat(Summary& sum) {
  std::printf("\n===== G4 repeat (race) test: O, LSE, dK, dV bitwise; dQ within FA's atol =====\n");
  const bool vj1d = bwd_mode_name().rfind("Vj1d", 0) == 0;
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
    const bool okKV = vj1d ? (mK == 0 && mV == 0) : worst_kv <= atol;
    const bool okQ = worst_dq <= atol;
    std::printf("  %-34s O/LSE mismatched elements %llu/%llu %s | dK/dV %llu/%llu (max %.3e) %s | dQ max diff %.3e vs atol %.3e (%llu elems differ) %s  (%.1f s)\n",
                tag, mO, mL, okO ? "PASS" : "FAIL", mK, mV, worst_kv,
                okKV ? "PASS" : "FAIL", worst_dq, atol, mQ, okQ ? "PASS" : "FAIL", secs_since(t0));
    if (!vj1d) std::printf("        note: backward mode is %s -- dK/dV gated by the dQ atol, not bitwise\n", bwd_mode_name().c_str());
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
static void diag_chat(const Inputs& in, const RefOut& ref, const KOut& ko, int r, int Ta, const Segment& s,
                      const std::function<bool(uint64_t, Inputs&, KOut&)>& rerun) {
  const Problem& p = in.p;
  const int L = s.L, hd = p.hd, Hq = p.Hq, Hkv = p.Hkv, G = p.G(), T = p.T;
  std::printf("\n  ---- DIAG dQ: D10 row %d, chat at %d, L=%d (batch row %d, alone row %d, Ta=%d) ----\n", g_diag_row,
              s.t0, L, s.b, r, Ta);
  // (a) repeats
  std::vector<double> rr;
  int over = 0;
  for (int k = 0; k < g_diag_rep; ++k) {
    Inputs ai;
    KOut kk;
    if (!rerun(777 + k, ai, kk)) { std::printf("  rerun %d threw\n", k); break; }
    const Err e = compare_region(kk.dQ.p, r, Ta, 0, ref.dQ.p, ref.dQf.p, s.b, T, s.t0, Hq, L, hd);
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
    const std::vector<bf16> k = d2h(ko.dQ.p, ((size_t)r * Hq + h) * Ta * hd, (size_t)L * hd);
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
  const std::vector<bf16> Ok = d2h(ko.O.p, (((size_t)r * Hq + wh) * Ta + wi) * hd, hd);
  const std::vector<double> Ot = d2h(ref.O.p, (((size_t)s.b * Hq + wh) * T + s.t0 + wi) * hd, hd);
  const std::vector<bf16> Qk = d2h(ko.dQ.p, (((size_t)r * Hq + wh) * Ta + wi) * hd, hd);
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
  const std::vector<float> Lk = d2h(ko.LSE.p, ((size_t)r * Hq + wh) * Ta + wi, 1);
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
      grab(ko.O.p, Hq, Ta, r, 0, hd, "ours_O.bf16");
      grab(ko.LSE.p, Hq, Ta, r, 0, 1, "ours_LSE.f32");
      grab(ko.dQ.p, Hq, Ta, r, 0, hd, "ours_dQ.bf16");
      grab(ko.dK.p, Hkv, Ta, r, 0, hd, "ours_dK.bf16");
      grab(ko.dV.p, Hkv, Ta, r, 0, hd, "ours_dV.bf16");
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
            bmis += bitdiff(ko.dK.p, r, Ta, 0, kw.dK.p, r, 4096, 0, h.Hkv, L, 128).mism;
            bmis += bitdiff(ko.dV.p, r, Ta, 0, kw.dV.p, r, 4096, 0, h.Hkv, L, 128).mism;
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
    std::printf("  %-44s G6c %d chats: alone@4096 vs alone@roundup -- O/LSE/dK/dV mismatched elements %llu, dQ max diff %.3e (smallest atol %.3e)  %s\n",
                tag, bit_done, bmis, bq, bq_atol_min, ok ? "PASS" : "FAIL");
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
  const char* only_env = std::getenv("VARLEN_ONLY");
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
  } else if (!only.empty()) {
    sum.record(false, "unknown VARLEN_ONLY=" + only + " (supported: D10)");
  } else if (!kernels_ok) {
    const std::string r = !hopper ? std::string("kernels under test are Hopper-only (v62/Vj1d); this GPU is ") + prop.name
                                  : "BLUTRAIN_NO_SM90[_GQA] is set";
    for (const char* g : {"G2", "G3", "G4", "G5", "G6", "G7"}) sum.skip_(std::string(g) + ": " + r);
  } else {
    g2_dense(sum);
    if (have_real) g3_v1_rows(Lreal, sum);
    else sum.skip_("G3 v1 rows: " + why);
    g4_repeat(sum);
    g5_large(sum);
    g6_chat_alone(have_real ? &Lreal : nullptr, sum);
    g6_d10(have_real ? &Lreal : nullptr, sum);
    g7_timing(sum);
  }

  std::printf("\n===== SUMMARY =====\n");
  std::printf("checks passed %d, failed %d, skipped groups %d   (%.0f s total)\n", sum.pass, sum.fail, sum.skip,
              secs_since(t_all));
  for (const std::string& f : sum.failed) std::printf("  FAIL  %s\n", f.c_str());
  for (const std::string& s : sum.skipped) std::printf("  SKIP  %s\n", s.c_str());
  std::printf("  KNOWN issues observed this run: %zu (kept by decision, watched in every run)\n", sum.known.size());
  for (const std::string& k : sum.known) std::printf("  KNOWN %s\n", k.c_str());
  std::printf("  PENDING (need the packed-row kernels, Steps 3-5):\n"
              "    B/A on packed rows vs the same per-chat reference (all data loads, all head configs)\n"
              "    isolation: change one chat's q/k/v/dO -> every other chat bitwise unchanged\n"
              "    same chat at offsets 0/1/63/64/65/127/128/end; zero-length entries; pad segments\n"
              "    empty-row contract (A's pad rows): O = 0, dQ/dK/dV = 0, fixed LSE value\n"
              "    varlen API rejections; non-varlen paths (Vj1, v44, generic, B300) must refuse chat info\n"
              "    tile-count check (debug build): tiles processed == host model\n");
  std::printf("%s\n", sum.fail != 0 ? "FAILURES ABOVE (NEW, not on the known list)"
                      : sum.known.empty() ? "ALL RUN CHECKS PASS"
                                          : "ALL RUN CHECKS PASS (known issues observed: see KNOWN above)");
  return sum.fail == 0 ? 0 : 1;
}
