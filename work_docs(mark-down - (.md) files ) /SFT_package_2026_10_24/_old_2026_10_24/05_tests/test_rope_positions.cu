// ============================================================================
// test_rope_positions.cu -- v2 packing, Step 1: per-token RoPE positions in the
// fused QK-norm + RoPE kernels (forward + backward), both kernel variants.
//
// The change (2026-10-10): fused_qk_norm_rope_{forward,backward}[_sm90] take an
// optional int32 positions[B, T]. With it, the cos/sin cache row of token (b, t)
// is positions[b*T + t] instead of t + pos_offset, so several chats packed into
// one row can each restart at position 0 (torchtitan does the same with
// rope_cache[positions], models/common/rope.py:382-385).
//
// Kernels under test. Both contain no Hopper-only instructions, so both run on
// any GPU when called directly; the library only SELECTS the Hopper-tuned one on
// Hopper (dispatcher in FusedRMSRoPE_fwd_sm90.cu / FusedRMSRoPE_bwd_sm90.cu):
//   dispatch = cuda::fused_qk_norm_rope_{forward,backward}       (portable kernel
//              off Hopper, the sm90 kernel on Hopper)
//   sm90     = cuda::fused_qk_norm_rope_{forward,backward}_sm90  called directly
//
// Tests (every variant x every config):
//   T1 identity : positions[b][t] = t      must equal positions=nullptr, pos_offset=0  (bit-exact)
//   T2 shift    : positions[b][t] = t + 7  must equal positions=nullptr, pos_offset=7  (bit-exact)
//   T3 packed   : chats restart at 0       must equal running EACH CHAT ALONE in its own tensor
//                 with no positions: q_out, k_out, rstd, dq, dk bit-exact per token; dgamma
//                 vs the sum of the per-chat dgammas (only the summation order differs)
//   T4 random   : arbitrary positions      vs an fp64 CPU reference of the same math
//   T5 API      : the Tensor-level wrapper accepts valid positions and rejects wrong dtype,
//                 shape, device, and positions together with a nonzero pos_offset
// Note: dgamma on the portable path is accumulated with atomicAdd
// (FusedRMSRoPE_bwd_sm90.cu), so its summation order is not fixed; it is
// compared with a tolerance everywhere it is not bit-exact by construction.
//
// Build + run (from Tensor-Implementations/):
//   make run-cuda FILE=Tests/kernel_tests/test_rope_positions.cu
// ============================================================================
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "ops/helpers/FusedRMSRoPEKernels.h"
#include "ops/helpers/KernelDispatch.h"   // get_arch(): same check the dispatcher makes
#include "ops/helpers/KernelPolicy.h"     // disable_sm90_qknorm_rope(): env override BLUTRAIN_NO_SM90_QKNORM_ROPE
#include "ops/UnaryOps/Normalizations.h"
#include "autograd/operations/RoPEOps.h"
#include "core/Tensor.h"
#include "device/Device.h"

namespace OwnTensor {
namespace cuda {
// Hopper-tuned entry points (defined in FusedRMSRoPE_{fwd,bwd}_sm90_kernel.cu, not in
// the public header). Declared here so the test can run them on any GPU.
void fused_qk_norm_rope_forward_sm90(
    const float* q_in, const float* k_in, float* q_out, float* k_out,
    const float* q_gamma, const float* k_gamma, const float* cos_sin,
    float* q_rstd, float* k_rstd, int B, int Hq, int Hkv, int T, int hd,
    int pos_offset, float eps, bool interleaved, cudaStream_t stream,
    const int32_t* positions);
void fused_qk_norm_rope_backward_sm90(
    const float* dz_q, const float* dz_k, const float* q_in, const float* k_in,
    const float* q_gamma, const float* k_gamma, const float* cos_sin,
    const float* q_rstd, const float* k_rstd, float* dq_in, float* dk_in,
    float* dq_gamma, float* dk_gamma, int B, int Hq, int Hkv, int T, int hd,
    int pos_offset, bool interleaved, cudaStream_t stream, const int32_t* positions);
}  // namespace cuda
}  // namespace OwnTensor

using namespace OwnTensor;

#define CK(x)                                                                    \
  do {                                                                           \
    cudaError_t e_ = (x);                                                        \
    if (e_ != cudaSuccess) {                                                     \
      fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__,    \
              cudaGetErrorString(e_));                                           \
      exit(2);                                                                   \
    }                                                                            \
  } while (0)

static const float kEps = 1e-6f;        // the trainer's qk_eps (bluscript_zero_ckpt.cpp:142)
static const float kTheta = 500000.f;   // the trainer's rope_theta (bluscript_zero_ckpt.cpp:137)

struct Cfg {
  int B, Hq, Hkv, T, hd;
  bool interleaved;
  std::vector<std::vector<int>> chats;  // per row: chat lengths (a pad tail is its own segment)
};

enum Variant { DISPATCH = 0, SM90 = 1 };
static const char* vname(Variant v) { return v == DISPATCH ? "dispatch" : "sm90"; }

struct Inputs {  // host data for one problem
  int B, Hq, Hkv, T, hd;
  std::vector<float> q, k, dzq, dzk, qg, kg;
};
struct Outputs {
  std::vector<float> qo, ko, qr, kr, dq, dk, dqg, dkg;
};

// One forward + backward on the GPU. positions empty = nullptr.
static Outputs run(Variant v, const Inputs& in, const float* d_cache, bool interleaved,
                   const std::vector<int32_t>& positions, int pos_offset) {
  const size_t nq = in.q.size(), nk = in.k.size();
  const size_t nqr = (size_t)in.B * in.Hq * in.T, nkr = (size_t)in.B * in.Hkv * in.T;
  float *q, *k, *qo, *ko, *qr, *kr, *dzq, *dzk, *dq, *dk, *dqg, *dkg, *qg, *kg;
  int32_t* pos = nullptr;
  CK(cudaMalloc(&q, nq * 4)); CK(cudaMalloc(&k, nk * 4));
  CK(cudaMalloc(&qo, nq * 4)); CK(cudaMalloc(&ko, nk * 4));
  CK(cudaMalloc(&qr, nqr * 4)); CK(cudaMalloc(&kr, nkr * 4));
  CK(cudaMalloc(&dzq, nq * 4)); CK(cudaMalloc(&dzk, nk * 4));
  CK(cudaMalloc(&dq, nq * 4)); CK(cudaMalloc(&dk, nk * 4));
  CK(cudaMalloc(&dqg, in.hd * 4)); CK(cudaMalloc(&dkg, in.hd * 4));
  CK(cudaMalloc(&qg, in.hd * 4)); CK(cudaMalloc(&kg, in.hd * 4));
  CK(cudaMemcpy(q, in.q.data(), nq * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(k, in.k.data(), nk * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dzq, in.dzq.data(), nq * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dzk, in.dzk.data(), nk * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(qg, in.qg.data(), in.hd * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(kg, in.kg.data(), in.hd * 4, cudaMemcpyHostToDevice));
  if (!positions.empty()) {
    CK(cudaMalloc(&pos, positions.size() * 4));
    CK(cudaMemcpy(pos, positions.data(), positions.size() * 4, cudaMemcpyHostToDevice));
  }
  if (v == DISPATCH) {
    cuda::fused_qk_norm_rope_forward(q, k, qo, ko, qg, kg, d_cache, qr, kr, in.B, in.Hq, in.Hkv,
                                     in.T, in.hd, pos_offset, kEps, interleaved, 0, pos);
    cuda::fused_qk_norm_rope_backward(dzq, dzk, q, k, qg, kg, d_cache, qr, kr, dq, dk, dqg, dkg,
                                      in.B, in.Hq, in.Hkv, in.T, in.hd, pos_offset, interleaved, 0,
                                      pos);
  } else {
    cuda::fused_qk_norm_rope_forward_sm90(q, k, qo, ko, qg, kg, d_cache, qr, kr, in.B, in.Hq,
                                          in.Hkv, in.T, in.hd, pos_offset, kEps, interleaved, 0,
                                          pos);
    cuda::fused_qk_norm_rope_backward_sm90(dzq, dzk, q, k, qg, kg, d_cache, qr, kr, dq, dk, dqg,
                                           dkg, in.B, in.Hq, in.Hkv, in.T, in.hd, pos_offset,
                                           interleaved, 0, pos);
  }
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  Outputs o;
  o.qo.resize(nq); o.ko.resize(nk); o.qr.resize(nqr); o.kr.resize(nkr);
  o.dq.resize(nq); o.dk.resize(nk); o.dqg.resize(in.hd); o.dkg.resize(in.hd);
  CK(cudaMemcpy(o.qo.data(), qo, nq * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(o.ko.data(), ko, nk * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(o.qr.data(), qr, nqr * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(o.kr.data(), kr, nkr * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(o.dq.data(), dq, nq * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(o.dk.data(), dk, nk * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(o.dqg.data(), dqg, in.hd * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(o.dkg.data(), dkg, in.hd * 4, cudaMemcpyDeviceToHost));
  for (void* p : {(void*)q, (void*)k, (void*)qo, (void*)ko, (void*)qr, (void*)kr, (void*)dzq,
                  (void*)dzk, (void*)dq, (void*)dk, (void*)dqg, (void*)dkg, (void*)qg, (void*)kg})
    CK(cudaFree(p));
  if (pos) CK(cudaFree(pos));
  return o;
}

// ---- comparisons ------------------------------------------------------------------------------
static size_t bit_mismatch(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) return (size_t)-1;
  size_t n = 0;
  for (size_t i = 0; i < a.size(); ++i) n += std::memcmp(&a[i], &b[i], 4) != 0;
  return n;
}
// max |a-b| / max |b|  (0 when both are all-zero)
static double rel_err(const std::vector<float>& a, const std::vector<double>& b) {
  double num = 0, den = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    num = std::max(num, std::fabs((double)a[i] - b[i]));
    den = std::max(den, std::fabs(b[i]));
  }
  return den > 0 ? num / den : num;
}
static double rel_err_f(const std::vector<float>& a, const std::vector<float>& b) {
  std::vector<double> bd(b.begin(), b.end());
  return rel_err(a, bd);
}

static int g_fail = 0, g_cases = 0;
static void report(bool ok, const std::string& line) {
  ++g_cases;
  g_fail += !ok;
  printf("  %s  %s\n", ok ? "PASS" : "FAIL", line.c_str());
}

// ---- inputs -------------------------------------------------------------------------------------
static Inputs make_inputs(int B, int Hq, int Hkv, int T, int hd, uint32_t seed) {
  Inputs in{B, Hq, Hkv, T, hd, {}, {}, {}, {}, {}, {}};
  std::mt19937 rng(seed);
  std::normal_distribution<float> n01(0.f, 1.f);
  auto fill = [&](std::vector<float>& v, size_t n, float s, float off) {
    v.resize(n);
    for (auto& x : v) x = off + s * n01(rng);
  };
  fill(in.q, (size_t)B * Hq * T * hd, 1.f, 0.f);
  fill(in.k, (size_t)B * Hkv * T * hd, 1.f, 0.f);
  fill(in.dzq, (size_t)B * Hq * T * hd, 1.f, 0.f);
  fill(in.dzk, (size_t)B * Hkv * T * hd, 1.f, 0.f);
  fill(in.qg, hd, 0.1f, 1.f);
  fill(in.kg, hd, 0.1f, 1.f);
  return in;
}

// copy tokens [s, s+L) of row b into a [1, H, L, hd] problem
static Inputs slice_chat(const Inputs& in, int b, int s, int L) {
  Inputs o{1, in.Hq, in.Hkv, L, in.hd, {}, {}, {}, {}, in.qg, in.kg};
  auto cut = [&](const std::vector<float>& src, int H, std::vector<float>& dst) {
    dst.resize((size_t)H * L * in.hd);
    for (int h = 0; h < H; ++h)
      std::memcpy(&dst[(size_t)h * L * in.hd],
                  &src[(((size_t)b * H + h) * in.T + s) * in.hd], (size_t)L * in.hd * 4);
  };
  cut(in.q, in.Hq, o.q); cut(in.k, in.Hkv, o.k);
  cut(in.dzq, in.Hq, o.dzq); cut(in.dzk, in.Hkv, o.dzk);
  return o;
}

// ---- fp64 CPU reference of the same math (FusedRMSRoPE kernels) ---------------------------------
static void ref_side(const std::vector<float>& x, const std::vector<float>& dz,
                     const std::vector<float>& gamma, const std::vector<float>& cache, int B, int H,
                     int T, int hd, bool interleaved, const std::vector<int32_t>& pos,
                     std::vector<double>& y, std::vector<double>& dx, std::vector<double>& dg) {
  const int half = hd / 2;
  y.assign(x.size(), 0); dx.assign(x.size(), 0); dg.assign(hd, 0);
  std::vector<double> xh(hd), dy(hd), g(hd);
  for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h)
      for (int t = 0; t < T; ++t) {
        const size_t o = (((size_t)b * H + h) * T + t) * hd;
        const int p = pos[(size_t)b * T + t];
        double ss = 0;
        for (int j = 0; j < hd; ++j) ss += (double)x[o + j] * x[o + j];
        const double rstd = 1.0 / std::sqrt(ss / hd + (double)kEps);
        for (int j = 0; j < hd; ++j) xh[j] = x[o + j] * rstd;
        double S = 0;
        for (int i = 0; i < half; ++i) {
          const int ia = interleaved ? 2 * i : i, ib = interleaved ? 2 * i + 1 : i + half;
          const double c = cache[(size_t)p * hd + i], s = cache[(size_t)p * hd + i + half];
          const double ya = xh[ia] * gamma[ia], yb = xh[ib] * gamma[ib];
          y[o + ia] = ya * c - yb * s;
          y[o + ib] = ya * s + yb * c;
          const double dza = dz[o + ia], dzb = dz[o + ib];
          dy[ia] = dza * c + dzb * s;   // R^T
          dy[ib] = -dza * s + dzb * c;
          g[ia] = dy[ia] * gamma[ia];
          g[ib] = dy[ib] * gamma[ib];
          S += g[ia] * xh[ia] + g[ib] * xh[ib];
        }
        S /= hd;
        for (int j = 0; j < hd; ++j) {
          dx[o + j] = rstd * (g[j] - xh[j] * S);
          dg[j] += dy[j] * xh[j];
        }
      }
}

static std::vector<int32_t> packed_positions(const Cfg& c) {
  std::vector<int32_t> pos((size_t)c.B * c.T);
  for (int b = 0; b < c.B; ++b) {
    int t = 0;
    for (int L : c.chats[b])
      for (int i = 0; i < L; ++i) pos[(size_t)b * c.T + t++] = i;
    if (t != c.T) { fprintf(stderr, "bad chat layout in row %d (%d != %d)\n", b, t, c.T); exit(2); }
  }
  return pos;
}

// ---- T5: Tensor-level wrapper argument checks ----------------------------------------------------
static void test_api() {
  printf("\n== T5 API (Tensor-level fused_rms_rope_forward)\n");
  const int B = 1, H = 2, T = 8, hd = 64;
  DeviceIndex gpu(Device::CUDA, 0), cpu(Device::CPU, 0);
  auto mk = [&](Shape s, Dtype d, DeviceIndex dev) {
    return Tensor::empty(s, TensorOptions().with_dtype(d).with_device(dev));
  };
  Inputs in = make_inputs(B, H, H, T, hd, 7);
  Tensor q = mk(Shape{{B, H, T, hd}}, Dtype::Float32, gpu), k = mk(Shape{{B, H, T, hd}}, Dtype::Float32, gpu);
  Tensor qg = mk(Shape{{hd}}, Dtype::Float32, gpu), kg = mk(Shape{{hd}}, Dtype::Float32, gpu);
  q.set_data(in.q); k.set_data(in.k); qg.set_data(in.qg); kg.set_data(in.kg);
  Tensor cache = autograd::build_rope_cache(T + 16, hd, kTheta, gpu);
  std::vector<int32_t> ident(T);
  for (int t = 0; t < T; ++t) ident[t] = t;
  Tensor pos_ok = mk(Shape{{B, T}}, Dtype::Int32, gpu);
  pos_ok.set_data(ident);

  auto expect_throw = [&](const char* what, auto&& fn) {
    bool threw = false;
    try { fn(); } catch (const std::exception&) { threw = true; }
    report(threw, std::string("rejects ") + what);
  };
  // valid positions: accepted, and identical to the no-positions path
  {
    auto a = fused_rms_rope_forward(q, k, qg, kg, cache, 0, kEps, false);
    auto b = fused_rms_rope_forward(q, k, qg, kg, cache, 0, kEps, false, pos_ok);
    std::vector<float> ha(a.q_out.numel()), hb(b.q_out.numel());
    CK(cudaMemcpy(ha.data(), a.q_out.data<float>(), ha.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hb.data(), b.q_out.data<float>(), hb.size() * 4, cudaMemcpyDeviceToHost));
    report(bit_mismatch(ha, hb) == 0, "accepts valid int32 [B,T] positions; identity == no positions (bit-exact)");
  }
  Tensor pos_i64 = mk(Shape{{B, T}}, Dtype::Int64, gpu);
  expect_throw("int64 positions", [&] { fused_rms_rope_forward(q, k, qg, kg, cache, 0, kEps, false, pos_i64); });
  Tensor pos_shape = mk(Shape{{B, T + 1}}, Dtype::Int32, gpu);
  expect_throw("wrong shape [B, T+1]", [&] { fused_rms_rope_forward(q, k, qg, kg, cache, 0, kEps, false, pos_shape); });
  Tensor pos_cpu = mk(Shape{{B, T}}, Dtype::Int32, cpu);
  expect_throw("CPU positions", [&] { fused_rms_rope_forward(q, k, qg, kg, cache, 0, kEps, false, pos_cpu); });
  expect_throw("positions + pos_offset=3", [&] { fused_rms_rope_forward(q, k, qg, kg, cache, 3, kEps, false, pos_ok); });
}

int main() {
  int dev = 0;
  CK(cudaGetDevice(&dev));
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, dev));
  // Which kernel the dispatcher really runs: exactly the condition in FusedRMSRoPE_fwd_sm90.cu /
  // FusedRMSRoPE_bwd_sm90.cu (Hopper AND not disabled by BLUTRAIN_NO_SM90_QKNORM_ROPE / the
  // all-sm90 switch). The first H200 run of this test inferred it from the GPU alone and so
  // expected bit-exact dgamma under the env override, where the portable atomicAdd kernel runs.
  const bool dispatch_is_sm90 =
      !cuda::disable_sm90_qknorm_rope() && cuda::get_arch() == cuda::ArchFamily::Hopper;
  printf("device: %s (sm_%d%d)  -- 'dispatch' runs the %s kernel on this GPU\n", prop.name, prop.major,
         prop.minor, dispatch_is_sm90 ? "sm90 (Hopper)" : "portable");

  std::vector<Cfg> cfgs = {
      // our model's shape: 18 q heads, 6 kv heads, hd 128, T 4096, NeoX
      {2, 18, 6, 4096, 128, false, {{5, 1, 1000, 3090}, {1000, 3000, 96}}},
      // odd T, interleaved, hd 64, a row of 333 one-token chats
      {3, 4, 2, 333, 64, true, {{100, 233}, std::vector<int>(333, 1), {64, 64, 64, 64, 64, 13}}},
      {1, 8, 8, 64, 128, false, {{7, 57}}},
      {2, 6, 2, 256, 64, false, {{256}, {128, 128}}},
  };

  for (const Cfg& c : cfgs) {
    printf("\n== config B=%d Hq=%d Hkv=%d T=%d hd=%d %s\n", c.B, c.Hq, c.Hkv, c.T, c.hd,
           c.interleaved ? "interleaved" : "NeoX");
    const int Lc = c.T + 16;  // cache long enough for the +7 shift
    Tensor cache = autograd::build_rope_cache(Lc, c.hd, kTheta, DeviceIndex(Device::CUDA, 0));
    const float* d_cache = cache.data<float>();
    std::vector<float> h_cache((size_t)Lc * c.hd);
    CK(cudaMemcpy(h_cache.data(), d_cache, h_cache.size() * 4, cudaMemcpyDeviceToHost));
    Inputs in = make_inputs(c.B, c.Hq, c.Hkv, c.T, c.hd, 1234u + c.T);

    for (Variant v : {DISPATCH, SM90}) {
      const char* vn = vname(v);
      // ---- T1 identity
      std::vector<int32_t> ident((size_t)c.B * c.T);
      for (int b = 0; b < c.B; ++b)
        for (int t = 0; t < c.T; ++t) ident[(size_t)b * c.T + t] = t;
      Outputs base = run(v, in, d_cache, c.interleaved, {}, 0);
      Outputs idp = run(v, in, d_cache, c.interleaved, ident, 0);
      size_t m = bit_mismatch(base.qo, idp.qo) + bit_mismatch(base.ko, idp.ko) +
                 bit_mismatch(base.qr, idp.qr) + bit_mismatch(base.kr, idp.kr) +
                 bit_mismatch(base.dq, idp.dq) + bit_mismatch(base.dk, idp.dk);
      // dgamma is order-fixed (bit-reproducible) only in the sm90 kernel; the portable one uses atomics.
      const bool dg_exact = (v == SM90) || dispatch_is_sm90;
      size_t mg = bit_mismatch(base.dqg, idp.dqg) + bit_mismatch(base.dkg, idp.dkg);
      double eg = std::max(rel_err_f(idp.dqg, base.dqg), rel_err_f(idp.dkg, base.dkg));
      report(m == 0 && (dg_exact ? mg == 0 : eg < 1e-5),
             std::string(vn) + " T1 identity == no positions: " + std::to_string(m) +
                 " mismatching values (q/k out, rstd, dq/dk); dgamma " +
                 (dg_exact ? std::to_string(mg) + " mismatching" : "rel err " + std::to_string(eg)));
      // ---- T2 shift by 7
      std::vector<int32_t> sh = ident;
      for (auto& p : sh) p += 7;
      Outputs off = run(v, in, d_cache, c.interleaved, {}, 7);
      Outputs shp = run(v, in, d_cache, c.interleaved, sh, 0);
      m = bit_mismatch(off.qo, shp.qo) + bit_mismatch(off.ko, shp.ko) + bit_mismatch(off.dq, shp.dq) +
          bit_mismatch(off.dk, shp.dk);
      report(m == 0, std::string(vn) + " T2 positions t+7 == pos_offset 7: " + std::to_string(m) +
                         " mismatching values");
      // ---- T3 packed vs each chat alone
      std::vector<int32_t> pk = packed_positions(c);
      Outputs P = run(v, in, d_cache, c.interleaved, pk, 0);
      size_t mism = 0, nchats = 0;
      std::vector<double> dqg_sum(c.hd, 0), dkg_sum(c.hd, 0);
      for (int b = 0; b < c.B; ++b) {
        int s = 0;
        for (int L : c.chats[b]) {
          Inputs one = slice_chat(in, b, s, L);
          Outputs O = run(v, one, d_cache, c.interleaved, {}, 0);
          auto cmp_side = [&](const std::vector<float>& pk_, const std::vector<float>& one_, int H, int per) {
            for (int h = 0; h < H; ++h)
              mism += std::memcmp(&pk_[(((size_t)b * H + h) * c.T + s) * per], &one_[((size_t)h * L) * per],
                                  (size_t)L * per * 4) != 0;
          };
          cmp_side(P.qo, O.qo, c.Hq, c.hd); cmp_side(P.ko, O.ko, c.Hkv, c.hd);
          cmp_side(P.qr, O.qr, c.Hq, 1);    cmp_side(P.kr, O.kr, c.Hkv, 1);
          cmp_side(P.dq, O.dq, c.Hq, c.hd); cmp_side(P.dk, O.dk, c.Hkv, c.hd);
          for (int j = 0; j < c.hd; ++j) { dqg_sum[j] += O.dqg[j]; dkg_sum[j] += O.dkg[j]; }
          s += L;
          ++nchats;
        }
      }
      double eg3 = std::max(rel_err(P.dqg, dqg_sum), rel_err(P.dkg, dkg_sum));
      // dgamma tolerance: the sm90 kernel reduces dgamma in a fixed order (its T1 shows 0
      // mismatching values), so only the packed-vs-per-chat summation order differs: 1e-5.
      // The portable kernel adds dgamma with float atomicAdd: two runs with IDENTICAL inputs
      // already differ (T1 above prints that run-to-run difference, 4e-6 at B=2 T=4096) and its
      // error vs the fp64 reference is ~1.2e-5 at that size (T4). A 1e-5 bound sits below that
      // noise floor (first run of this test, 2026-10-10: 1.1e-5 with 0 per-token mismatches),
      // so the atomic path uses T4's 1e-4.
      const double tol_dg3 = dg_exact ? 1e-5 : 1e-4;
      report(mism == 0 && eg3 < tol_dg3,
             std::string(vn) + " T3 packed == each of " + std::to_string(nchats) +
                 " chats alone: " + std::to_string(mism) +
                 " mismatching (chat, head, tensor) blocks; dgamma vs sum of chats rel err " +
                 std::to_string(eg3));
      // ---- T4 random positions vs fp64 reference
      std::mt19937 rng(99u + c.T);
      std::uniform_int_distribution<int> up(0, Lc - 1);
      std::vector<int32_t> rp((size_t)c.B * c.T);
      for (auto& p : rp) p = up(rng);
      Outputs R = run(v, in, d_cache, c.interleaved, rp, 0);
      std::vector<double> yq, dxq, dgq, yk, dxk, dgk;
      ref_side(in.q, in.dzq, in.qg, h_cache, c.B, c.Hq, c.T, c.hd, c.interleaved, rp, yq, dxq, dgq);
      ref_side(in.k, in.dzk, in.kg, h_cache, c.B, c.Hkv, c.T, c.hd, c.interleaved, rp, yk, dxk, dgk);
      const double ef = std::max(rel_err(R.qo, yq), rel_err(R.ko, yk));
      const double eb = std::max(rel_err(R.dq, dxq), rel_err(R.dk, dxk));
      const double eg4 = std::max(rel_err(R.dqg, dgq), rel_err(R.dkg, dgk));
      report(ef < 1e-5 && eb < 1e-5 && eg4 < 1e-4,
             std::string(vn) + " T4 random positions vs fp64 reference: fwd " + std::to_string(ef) +
                 ", dq/dk " + std::to_string(eb) + ", dgamma " + std::to_string(eg4));
    }
  }

  test_api();

  printf("\n================================================================\n");
  printf("SUMMARY: %d cases, %d failed  -> %s\n", g_cases, g_fail, g_fail ? "FAIL" : "ALL PASS");
  printf("================================================================\n");
  return g_fail ? 1 : 0;
}
