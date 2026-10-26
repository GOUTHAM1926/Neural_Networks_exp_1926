// =============================================================================
// test_rope_bf16_positions.cu  (2026-10-22, stage B of the bf16 RoPE port)
//
// Stage B added per-chat positions to Kruthik's bf16 fused QK-RMSNorm + RoPE (the same HAS_POS scheme as our fp32
// kernels). Four checks, every output compared bit for bit (q_out, k_out, q_rstd, k_rstd, dq, dk, dgamma_q, dgamma_k):
//   1. NONE vs IDENTITY: positions = 0..T-1 in every row must give exactly the same bits as no positions at all
//      (proves the HAS_POS=true path reads the right cache row for every token).
//   2. PACKED bf16 vs PACKED fp32: rows of several chats, positions restarting at 0 in every chat (as SftDataLoader
//      builds them: every chat, then the pad tail as its own segment). bf16 kernel with positions vs the fp32 path with
//      the same positions, as bluscript_sft.cpp runs it today (its copies included).
//   3. PACKED vs ALONE: the second chat of row 0, cut out and run alone at the start of a row WITHOUT positions, must
//      give the same bits as inside the packed row (dgamma is a sum over all tokens, so it is not part of this check).
//   4. TIME: bf16 without positions (= pretraining), bf16 with positions (= SFT from now on), fp32 path with positions
//      (= SFT today).
// Shape: T 4096, 18 query / 6 KV heads, head dim 128, NeoX, eps 1e-6, theta 10000; B = 4 and 8.
// usage: test_rope_bf16_positions [--iters N]
// =============================================================================
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "core/Tensor.h"
#include "device/Device.h"
#include "ops/UnaryOps/Normalizations.h"
#include "autograd/operations/ReshapeOps.h"
#include "autograd/operations/RoPEOps.h"

using namespace OwnTensor;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA error %s at %s:%d\n", \
    cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(2); } } while (0)

static constexpr int64_t kT = 4096, kHq = 18, kHkv = 6, kHd = 128;
static constexpr float kEps = 1e-6f, kTheta = 10000.0f;

static uint16_t f2bf(float f) {   // round to nearest even, as __float2bfloat16_rn (no NaN in these inputs)
  uint32_t u; std::memcpy(&u, &f, 4);
  return static_cast<uint16_t>((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
template <typename T> static std::vector<T> host(const Tensor& t) {
  std::vector<T> h(t.numel());
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(h.data(), t.data(), h.size() * sizeof(T), cudaMemcpyDeviceToHost));
  return h;
}
template <typename T> static int64_t mismatches(const std::vector<T>& a, const std::vector<T>& b) {
  int64_t n = 0;
  for (size_t i = 0; i < a.size(); ++i) n += (std::memcmp(&a[i], &b[i], sizeof(T)) != 0);
  return n;
}
static float median(std::vector<float> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

// one compared output: name, differing elements, total elements
struct Row { std::string name; int64_t bad, n; };
static bool report(const char* check, int64_t B, const std::vector<Row>& rows) {
  bool ok = true;
  for (const Row& r : rows) {
    std::printf("%-8s B=%ld %-16s differing %ld of %ld%s\n", check, (long)B, r.name.c_str(), (long)r.bad, (long)r.n,
                r.bad ? "   <-- FAIL" : "  (bit-identical)");
    ok = ok && r.bad == 0;
  }
  return ok;
}

int main(int argc, char** argv) {
  int iters = 200;
  for (int i = 1; i + 1 < argc; i += 2) if (std::string(argv[i]) == "--iters") iters = std::atoi(argv[i + 1]);
  const DeviceIndex gpu(Device::CUDA, 0);
  auto opts = [&](Dtype d) { return TensorOptions().with_dtype(d).with_device(gpu); };
  Tensor cache = autograd::build_rope_cache(kT, kHd, kTheta, gpu);
  cudaDeviceProp prop{}; CK(cudaGetDeviceProperties(&prop, 0));
  std::printf("device %s | T %ld, %ld / %ld heads, hd %ld, NeoX, eps %g, theta %.0f | %d timed calls\n", prop.name,
              (long)kT, (long)kHq, (long)kHkv, (long)kHd, kEps, kTheta, iters);
  bool all_ok = true;

  for (int64_t B : {4, 8}) {
    // ---- inputs: bf16 q, k token-major (what the projections give), bf16 grads head-major, fp32 gammas ----
    std::vector<uint16_t> hq(B * kT * kHq * kHd), hk(B * kT * kHkv * kHd), hgq(B * kHq * kT * kHd), hgk(B * kHkv * kT * kHd);
    uint64_t s = 0x9E3779B97F4A7C15ull + B;
    auto next = [&]() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; };
    auto rnd = [&]() {   // ~N(0,1): sum of 4 uniforms
      float acc = 0.f;
      for (int j = 0; j < 4; ++j) acc += (float)((next() >> 40) * (1.0 / 16777216.0)) - 0.5f;
      return acc * 1.7320508f;
    };
    for (auto* v : {&hq, &hk, &hgq, &hgk}) for (auto& x : *v) x = f2bf(rnd());
    std::vector<float> hg(kHd);
    for (int i = 0; i < kHd; ++i) hg[i] = 1.0f + 0.01f * static_cast<float>((i * 37) % 17 - 8);

    // ---- packed positions, as SftDataLoader builds them: chats of 1..1500 tokens (Tulu mean is ~640), each
    //      restarting at 0, then the pad tail as a segment of its own (also from 0) ----
    std::vector<int32_t> hpos_packed(B * kT), hpos_ident(B * kT);
    int64_t chat1_start = -1, chat1_len = 0, n_chats = 0;   // row 0's second chat, for check 3
    for (int64_t b = 0; b < B; ++b) {
      int64_t t = 0, c = 0;
      while (t < kT) {
        int64_t len = 1 + (int64_t)(next() % 1500);
        if (t + len > kT) len = kT - t;   // the last segment = the pad tail, positions 0..len-1 like a chat
        if (b == 0 && c == 1) { chat1_start = t; chat1_len = len; }
        for (int64_t i = 0; i < len; ++i) hpos_packed[b * kT + t + i] = (int32_t)i;
        t += len; ++c;
      }
      n_chats += c;
      for (int64_t i = 0; i < kT; ++i) hpos_ident[b * kT + i] = (int32_t)i;
    }
    std::printf("SETUP    B=%ld %ld segments in %ld rows (chats + pad tails); row 0 chat 2 = tokens %ld..%ld\n",
                (long)B, (long)n_chats, (long)B, (long)chat1_start, (long)(chat1_start + chat1_len - 1));

    auto up = [&](const auto& h, Shape sh, Dtype d) {
      Tensor t = Tensor::empty(sh, opts(d));
      CK(cudaMemcpy(t.data(), h.data(), h.size() * sizeof(h[0]), cudaMemcpyHostToDevice));
      return t;
    };
    Tensor q  = up(hq,  Shape{{B, kT, kHq * kHd}},  Dtype::Bfloat16);
    Tensor k  = up(hk,  Shape{{B, kT, kHkv * kHd}}, Dtype::Bfloat16);
    Tensor gq = up(hgq, Shape{{B, kHq, kT, kHd}},   Dtype::Bfloat16);
    Tensor gk = up(hgk, Shape{{B, kHkv, kT, kHd}},  Dtype::Bfloat16);
    Tensor qg = up(hg, Shape{{kHd}}, Dtype::Float32), kg = up(hg, Shape{{kHd}}, Dtype::Float32);
    Tensor pos_packed = up(hpos_packed, Shape{{B, kT}}, Dtype::Int32);
    Tensor pos_ident  = up(hpos_ident,  Shape{{B, kT}}, Dtype::Int32);
    const Tensor none;   // invalid Tensor = no positions (the pretraining call)

    // ---- the bf16 kernel: forward + backward, all 8 outputs ----
    struct Out { Tensor qo, ko, qr, kr, dq, dk, dgq, dgk; };
    auto run16 = [&](const Tensor& qq, const Tensor& kk, const Tensor& gqq, const Tensor& gkk, const Tensor& pos) {
      auto f = fused_rms_rope_forward_bf16_tm(qq, kk, kHq, kHkv, qg, kg, cache, 0, kEps, false, pos);
      auto r = fused_rms_rope_backward_bf16_tm(gqq, gkk, qq, kk, qg, kg, cache, f.q_rstd, f.k_rstd, 0, false, pos);
      return Out{f.q_out, f.k_out, f.q_rstd, f.k_rstd, r.grad_q, r.grad_k, r.grad_q_gamma, r.grad_k_gamma};
    };
    // ---- the fp32 path with positions, as bluscript_sft.cpp runs it today ----
    auto to4d = [&](const Tensor& x, int64_t H) {
      return autograd::to_dtype(autograd::contiguous(autograd::transpose(autograd::reshape(x, Shape{{B, kT, H, kHd}}), 1, 2)),
                                Dtype::Float32);
    };
    auto back = [&](const Tensor& d, int64_t H) {   // fp32 [B,H,T,hd] -> bf16 [B,T,H*hd]
      return autograd::reshape(autograd::contiguous(autograd::transpose(autograd::to_dtype(d, Dtype::Bfloat16), 1, 2)),
                               Shape{{B, kT, H * kHd}});
    };
    auto run32 = [&](const Tensor& pos) {
      Tensor q4 = to4d(q, kHq), k4 = to4d(k, kHkv);
      auto f = fused_rms_rope_forward(q4, k4, qg, kg, cache, 0, kEps, false, pos);
      auto r = fused_rms_rope_backward(autograd::to_dtype(gq, Dtype::Float32), autograd::to_dtype(gk, Dtype::Float32),
                                       q4, k4, qg, kg, cache, f.q_rstd, f.k_rstd, 0, false, pos);
      return Out{autograd::to_dtype(f.q_out, Dtype::Bfloat16), autograd::to_dtype(f.k_out, Dtype::Bfloat16),
                 f.q_rstd, f.k_rstd, back(r.grad_q, kHq), back(r.grad_k, kHkv), r.grad_q_gamma, r.grad_k_gamma};
    };
    auto compare_all = [&](const Out& a, const Out& b) {
      std::vector<Row> rows;
      auto c16 = [&](const char* nm, const Tensor& x, const Tensor& y) {
        auto hx = host<uint16_t>(x), hy = host<uint16_t>(y); rows.push_back({nm, mismatches(hx, hy), (int64_t)hx.size()});
      };
      auto c32 = [&](const char* nm, const Tensor& x, const Tensor& y) {
        auto hx = host<float>(x), hy = host<float>(y); rows.push_back({nm, mismatches(hx, hy), (int64_t)hx.size()});
      };
      c16("q_out (bf16)", a.qo, b.qo);   c16("k_out (bf16)", a.ko, b.ko);
      c32("q_rstd (fp32)", a.qr, b.qr);  c32("k_rstd (fp32)", a.kr, b.kr);
      c16("dq (bf16)", a.dq, b.dq);      c16("dk (bf16)", a.dk, b.dk);
      c32("dgamma_q (fp32)", a.dgq, b.dgq); c32("dgamma_k (fp32)", a.dgk, b.dgk);
      return rows;
    };

    // ---- check 1: no positions vs identity positions ----
    const Out o_none = run16(q, k, gq, gk, none);
    const Out o_ident = run16(q, k, gq, gk, pos_ident);
    all_ok = report("CHECK1", B, compare_all(o_none, o_ident)) && all_ok;

    // ---- check 2: packed positions, bf16 kernel vs fp32 path ----
    const Out o_packed = run16(q, k, gq, gk, pos_packed);
    const Out o_packed32 = run32(pos_packed);
    all_ok = report("CHECK2", B, compare_all(o_packed, o_packed32)) && all_ok;
    {   // the packed result must differ from no positions, or check 2 proves nothing about positions
      auto a = host<uint16_t>(o_packed.qo), b = host<uint16_t>(o_none.qo);
      const int64_t d = mismatches(a, b);
      std::printf("CHECK2   B=%ld packed q_out vs NO positions: differing %ld of %ld%s\n", (long)B, (long)d, (long)a.size(),
                  d ? "  (positions are used)" : "   <-- FAIL: positions had no effect");
      all_ok = all_ok && d > 0;
    }

    // ---- check 3: row 0's second chat, alone at the start of a row, no positions ----
    {
      std::vector<uint16_t> aq = hq, ak = hk, agq = hgq, agk = hgk;
      const int64_t L = chat1_len, S = chat1_start;
      std::memcpy(&aq[0], &hq[S * kHq * kHd], L * kHq * kHd * 2);      // token-major: row 0 tokens S..S+L-1 -> 0..L-1
      std::memcpy(&ak[0], &hk[S * kHkv * kHd], L * kHkv * kHd * 2);
      for (int64_t h = 0; h < kHq; ++h)                                    // head-major grads: per head
        std::memcpy(&agq[h * kT * kHd], &hgq[(h * kT + S) * kHd], L * kHd * 2);
      for (int64_t h = 0; h < kHkv; ++h)
        std::memcpy(&agk[h * kT * kHd], &hgk[(h * kT + S) * kHd], L * kHd * 2);
      const Out o_alone = run16(up(aq, q.shape(), Dtype::Bfloat16), up(ak, k.shape(), Dtype::Bfloat16),
                                up(agq, gq.shape(), Dtype::Bfloat16), up(agk, gk.shape(), Dtype::Bfloat16), none);
      std::vector<Row> rows;
      // head-major [B,H,T,x]: row 0, every head, tokens S.. (packed) vs 0.. (alone)
      auto hm = [&](const char* nm, const Tensor& p, const Tensor& a, int64_t H, int64_t W, auto tag) {
        using T = decltype(tag);
        auto hp = host<T>(p), ha = host<T>(a); int64_t bad = 0;
        for (int64_t h = 0; h < H; ++h)
          for (int64_t i = 0; i < L * W; ++i)
            bad += std::memcmp(&hp[(h * kT + S) * W + i], &ha[h * kT * W + i], sizeof(T)) != 0;
        rows.push_back({nm, bad, H * L * W});
      };
      // token-major [B,T,H*hd]: row 0, tokens S.. vs 0..
      auto tm = [&](const char* nm, const Tensor& p, const Tensor& a, int64_t H) {
        auto hp = host<uint16_t>(p), ha = host<uint16_t>(a); int64_t bad = 0;
        for (int64_t i = 0; i < L * H * kHd; ++i) bad += hp[S * H * kHd + i] != ha[i];
        rows.push_back({nm, bad, L * H * kHd});
      };
      hm("q_out (bf16)", o_packed.qo, o_alone.qo, kHq, kHd, uint16_t{});
      hm("k_out (bf16)", o_packed.ko, o_alone.ko, kHkv, kHd, uint16_t{});
      hm("q_rstd (fp32)", o_packed.qr, o_alone.qr, kHq, 1, float{});
      hm("k_rstd (fp32)", o_packed.kr, o_alone.kr, kHkv, 1, float{});
      tm("dq (bf16)", o_packed.dq, o_alone.dq, kHq);
      tm("dk (bf16)", o_packed.dk, o_alone.dk, kHkv);
      all_ok = report("CHECK3", B, rows) && all_ok;
    }

    // ---- check 4: time (median of `iters` forward+backward calls after 20 warm-up calls) ----
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    auto time_it = [&](auto&& body) {
      for (int i = 0; i < 20; ++i) body();
      CK(cudaDeviceSynchronize());
      std::vector<float> t;
      for (int i = 0; i < iters; ++i) {
        CK(cudaEventRecord(e0)); body(); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); t.push_back(ms);
      }
      return median(t);
    };
    const float t_none = time_it([&] { auto r = run16(q, k, gq, gk, none); (void)r; });
    const float t_pos  = time_it([&] { auto r = run16(q, k, gq, gk, pos_packed); (void)r; });
    const float t_32   = time_it([&] { auto r = run32(pos_packed); (void)r; });
    std::printf("TIME     B=%ld fwd+bwd | bf16 no positions (pretraining) %.4f ms | bf16 with positions (SFT new) %.4f ms "
                "(%+.1f%%) | fp32 path with positions (SFT today) %.4f ms | new / today = %.3f\n",
                (long)B, t_none, t_pos, 100.0 * (t_pos - t_none) / t_none, t_32, t_pos / t_32);
    CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
  }
  std::printf("%s\n", all_ok ? "RESULT: ALL CHECKS PASS (every compared value bit-identical)"
                             : "RESULT: FAIL (see the lines marked FAIL)");
  CK(cudaGetLastError());
  return all_ok ? 0 : 3;
}
