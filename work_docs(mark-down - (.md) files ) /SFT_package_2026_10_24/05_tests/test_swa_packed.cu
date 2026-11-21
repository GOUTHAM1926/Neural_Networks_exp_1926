// ============================================================================
// test_swa_packed.cu -- 2026-11-20, step 8 of the SWA plan: packed rows (SFT chats, a GqaChatPlan) with a sliding
// window, i.e. forward modes 4 / 5 and the backward WIN copies on the plan's lists (step 7), plus item A (a batch whose
// segments all fit in the window runs the kernels of before). The checks come from the 20/11 study of how the other
// libraries implement and test it (SWA analysis doc, section 11.3):
//   P1 alone     every chat of a packed batch == the same chat ALONE through the DENSE kernels with the same window (a
//                row of roundup(L, 256), the chat at offset 0, random filler after it with dO = 0: the suite's G6 / G10 /
//                G11 convention): O, LSE, dK, dV bit for bit; dQ within FA's atol 2 x max|bf16(bf16(dq + 0.3) - 0.3) - dq|
//                (the dQ accumulator is a TMA reduce-add). The dense window kernel is the one validated in step 4.
//   P2 same      the same batch with window W and with window 0: every chat of L <= W is bit for bit unchanged (O, LSE,
//                dK, dV; dQ atol), every chat of L > W changes. In a batch with a long chat the window kernels run for ALL
//                chats, so this is the proof behind item A.
//   P3 fp64      every chat of L > W against the fp64 reference of that chat alone (Navin's ref kernels, as copied into
//                test_swa_extra.cu), his T4 rule: windowed error <= 2 x causal error + 2e-3, for O, LSE, dQ, dK, dV.
//   P4 item A    a batch whose segments (chats and pad tails) all fit in W: window W == window 0 bit for bit.
//   P5 refusals  packed + window 1000 and -1 throw (forward and backward); window >= T runs and == window 0; a
//                zero-length segment is refused by the plan builder.
//   P6 repeat    the same packed run twice: O, LSE, dK, dV bit for bit, dQ within FA's atol.
//   P7 long row  one row of 8,192 tokens (a chat of 8,000 + a pad tail of 192), W 1,024: P1 + P3.
//   P8 autograd  autograd::scaled_dot_product_attention(..., plan, W) + backward == the direct calls (O, dK, dV bit for
//                bit; dQ max difference printed).
// Layouts (rows of 4,096; the rest of a row is its pad tail, a segment of its own):
//   edge128  W 128 / 256: chats of W-1, W, W+1, 2W, 2W+1 for both windows (127 128 129 256 257 / 255 256 257 512 513),
//            3W-1 .. 3W+1 of W 128, and 1, 2, 63, 64, 65 tokens (TE found its window off-by-one only with a small window:
//            TransformerEngine c3b3cd21);
//   w1024    W 512 / 1024 (the 350M): 1023, 1024, 1025, 2047, 2049, a whole-row chat, short odd chats, a pad tail;
//   mx       W 1024 / 128: many chats with an odd number of 64-row tiles, so the plan holds mixed units (forward mode 5).
// Heads: 8/4 (the 350M; idea 4 on), 4/1 (one KV head: idea 4 off, the per-query-head path), 8/8 (G = 1: no mixed units).
// 2026-11-20 step 9: + the 350M's per-GPU head shapes under SFT context parallelism (BLU_SFT_CP=U: each GPU holds whole rows for
// Hq/U query and Hkv/U KV heads): 4/2 (U = 2, idea 4 on) and 2/1 (U = 4, idea 4 off), on the w1024 and mx layouts.
// The backward variant is chosen once per process: run it three times (default, BLU_VJ2_HSUM=0, BLU_VJ2_DKDV_F32=1).
// ============================================================================
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "core/Tensor.h"
#include "device/Device.h"
#include "autograd/operations/AttentionOps.h"
#include "ops/helpers/AttentionKernels.h"

using namespace OwnTensor;
using bf16 = __nv_bfloat16;
constexpr int D = 128;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2); } } while (0)

static int g_fail = 0, g_pass = 0;
static void check(bool ok, const char* fmt, ...) __attribute__((format(printf, 2, 3)));
static void check(bool ok, const char* fmt, ...) {
    va_list ap; va_start(ap, fmt);
    printf("  [%s] ", ok ? "PASS" : "FAIL"); vprintf(fmt, ap); printf("\n");
    va_end(ap);
    if (ok) ++g_pass; else ++g_fail;
    fflush(stdout);
}

// ---- device buffers of one attention call (dense layout [B, H, T, D]; copied from test_swa_extra.cu) ----------------------
struct Problem { int B, Hq, Hkv, T; };
struct Data {
    Problem p;
    bf16 *q, *k, *v, *o, *dO, *dq, *dk, *dv;
    float* lse;
    size_t nq, nk;
    explicit Data(const Problem& p_) : p(p_) {
        nq = (size_t)p.B * p.Hq * p.T * D; nk = (size_t)p.B * p.Hkv * p.T * D;
        for (bf16** b : {&q, &o, &dO, &dq}) CK(cudaMalloc(b, nq * sizeof(bf16)));
        for (bf16** b : {&k, &v, &dk, &dv}) CK(cudaMalloc(b, nk * sizeof(bf16)));
        CK(cudaMalloc(&lse, (size_t)p.B * p.Hq * p.T * sizeof(float)));
    }
    ~Data() { for (void* b : {(void*)q, (void*)k, (void*)v, (void*)o, (void*)dO, (void*)dq, (void*)dk, (void*)dv,
                              (void*)lse}) cudaFree(b); }
    Data(const Data&) = delete; Data& operator=(const Data&) = delete;
    size_t nl() const { return (size_t)p.B * p.Hq * p.T; }
};
static const bfloat16_t* cb(const bf16* x) { return reinterpret_cast<const bfloat16_t*>(x); }
static bfloat16_t* mb(bf16* x) { return reinterpret_cast<bfloat16_t*>(x); }
static void forward(Data& d, int window, const cuda::GqaChatPlan* chats = nullptr) {
    CK(cudaMemset(d.o, 0xFF, d.nq * sizeof(bf16)));
    CK(cudaMemset(d.lse, 0xFF, d.nl() * sizeof(float)));
    cuda::gqa_flash_attn_forward_qkv(cb(d.q), cb(d.k), cb(d.v), mb(d.o), d.lse,
                                     d.p.B, d.p.Hq, d.p.Hkv, d.p.T, D, true, chats, window);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
}
static void backward(Data& d, int window, const cuda::GqaChatPlan* chats = nullptr) {
    CK(cudaMemset(d.dq, 0xFF, d.nq * sizeof(bf16)));
    CK(cudaMemset(d.dk, 0xFF, d.nk * sizeof(bf16)));
    CK(cudaMemset(d.dv, 0xFF, d.nk * sizeof(bf16)));
    cuda::gqa_flash_attn_backward_qkv(cb(d.q), cb(d.k), cb(d.v), cb(d.o), cb(d.dO), d.lse, mb(d.dq), mb(d.dk), mb(d.dv),
                                      d.p.B, d.p.Hq, d.p.Hkv, d.p.T, D, true, chats, window);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
}
template <class T> static std::vector<T> dl(const T* x, size_t n) {
    std::vector<T> h(n); CK(cudaMemcpy(h.data(), x, n * sizeof(T), cudaMemcpyDeviceToHost)); return h;
}
template <class T> static void ul(T* x, const std::vector<T>& h) {
    CK(cudaMemcpy(x, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
}
static std::vector<bf16> randn_bf16(size_t n, uint64_t seed, float s) {
    std::mt19937_64 g(seed); std::normal_distribution<float> nd(0.f, s);
    std::vector<bf16> h(n); for (auto& x : h) x = __float2bfloat16(nd(g)); return h;
}
static float f(bf16 x) { return __bfloat162float(x); }

// ---- fp64-accumulated reference (copied unchanged from test_swa_attention.cu, as in test_swa_extra.cu) --------------------
__device__ int ref_lo(int i, int W) { return W > 0 ? max(0, i - W + 1) : 0; }
__global__ void ref_fwd(const bf16* Q, const bf16* K, const bf16* V, float* O, float* LSE,
                        int B, int Hq, int Hkv, int T, int W) {
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)B * Hq * T) return;
    const int i = idx % T, bh = idx / T, hq = bh % Hq, b = bh / Hq, hkv = hq / (Hq / Hkv);
    const bf16* q = Q + idx * D;
    const bf16* Kb = K + (long)(b * Hkv + hkv) * T * D;
    const bf16* Vb = V + (long)(b * Hkv + hkv) * T * D;
    const double sc = 1.0 / sqrt((double)D);
    double m = -INFINITY;
    for (int j = ref_lo(i, W); j <= i; ++j) {
        double s = 0; for (int x = 0; x < D; ++x) s += (double)__bfloat162float(q[x]) * __bfloat162float(Kb[(long)j * D + x]);
        m = fmax(m, s * sc);
    }
    double l = 0, o[D];
    for (int x = 0; x < D; ++x) o[x] = 0;
    for (int j = ref_lo(i, W); j <= i; ++j) {
        double s = 0; for (int x = 0; x < D; ++x) s += (double)__bfloat162float(q[x]) * __bfloat162float(Kb[(long)j * D + x]);
        const double p = exp(s * sc - m);
        l += p;
        for (int x = 0; x < D; ++x) o[x] += p * __bfloat162float(Vb[(long)j * D + x]);
    }
    for (int x = 0; x < D; ++x) O[idx * D + x] = (float)(o[x] / l);
    LSE[idx] = (float)(m + log(l));
}
__global__ void ref_dq(const bf16* Q, const bf16* K, const bf16* V, const bf16* dO, const float* O, const float* LSE,
                       float* Drow, float* dQ, int B, int Hq, int Hkv, int T, int W) {
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)B * Hq * T) return;
    const int i = idx % T, bh = idx / T, hq = bh % Hq, b = bh / Hq, hkv = hq / (Hq / Hkv);
    const bf16* q = Q + idx * D; const bf16* g = dO + idx * D;
    const bf16* Kb = K + (long)(b * Hkv + hkv) * T * D;
    const bf16* Vb = V + (long)(b * Hkv + hkv) * T * D;
    const double sc = 1.0 / sqrt((double)D);
    double Di = 0; for (int x = 0; x < D; ++x) Di += (double)__bfloat162float(g[x]) * O[idx * D + x];
    Drow[idx] = (float)Di;
    double dq[D]; for (int x = 0; x < D; ++x) dq[x] = 0;
    for (int j = ref_lo(i, W); j <= i; ++j) {
        double s = 0, dp = 0;
        for (int x = 0; x < D; ++x) {
            s  += (double)__bfloat162float(q[x]) * __bfloat162float(Kb[(long)j * D + x]);
            dp += (double)__bfloat162float(g[x]) * __bfloat162float(Vb[(long)j * D + x]);
        }
        const double ds = exp(s * sc - LSE[idx]) * (dp - Di);
        for (int x = 0; x < D; ++x) dq[x] += ds * __bfloat162float(Kb[(long)j * D + x]);
    }
    for (int x = 0; x < D; ++x) dQ[idx * D + x] = (float)(dq[x] * sc);
}
__global__ void ref_dkdv(const bf16* Q, const bf16* K, const bf16* V, const bf16* dO, const float* LSE, const float* Drow,
                         float* dK, float* dV, int B, int Hq, int Hkv, int T, int W) {
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)B * Hkv * T) return;
    const int j = idx % T, bh = idx / T, hkv = bh % Hkv, b = bh / Hkv, G = Hq / Hkv;
    const bf16* k = K + idx * D; const bf16* v = V + idx * D;
    const double sc = 1.0 / sqrt((double)D);
    const int hi = W > 0 ? min(T - 1, j + W - 1) : T - 1;
    double dk[D], dv[D]; for (int x = 0; x < D; ++x) { dk[x] = 0; dv[x] = 0; }
    for (int gq = 0; gq < G; ++gq) {
        const long rb = (long)(b * Hq + hkv * G + gq) * T;
        for (int i = j; i <= hi; ++i) {
            const bf16* q = Q + (rb + i) * D; const bf16* g = dO + (rb + i) * D;
            double s = 0, dp = 0;
            for (int x = 0; x < D; ++x) {
                s  += (double)__bfloat162float(q[x]) * __bfloat162float(k[x]);
                dp += (double)__bfloat162float(g[x]) * __bfloat162float(v[x]);
            }
            const double p = exp(s * sc - LSE[rb + i]), ds = p * (dp - Drow[rb + i]);
            for (int x = 0; x < D; ++x) { dv[x] += p * __bfloat162float(g[x]); dk[x] += ds * __bfloat162float(q[x]); }
        }
    }
    for (int x = 0; x < D; ++x) { dK[idx * D + x] = (float)(dk[x] * sc); dV[idx * D + x] = (float)dv[x]; }
}
struct Ref { std::vector<float> O, LSE, dQ, dK, dV; };
static Ref reference(const Data& d, int W) {
    const Problem& p = d.p;
    float *O, *LSE, *Dr, *dQ, *dK, *dV;
    CK(cudaMalloc(&O, d.nq * 4)); CK(cudaMalloc(&LSE, d.nl() * 4)); CK(cudaMalloc(&Dr, d.nl() * 4));
    CK(cudaMalloc(&dQ, d.nq * 4)); CK(cudaMalloc(&dK, d.nk * 4)); CK(cudaMalloc(&dV, d.nk * 4));
    const int tb = 64;
    ref_fwd<<<(unsigned)((d.nl() + tb - 1) / tb), tb>>>(d.q, d.k, d.v, O, LSE, p.B, p.Hq, p.Hkv, p.T, W);
    ref_dq<<<(unsigned)((d.nl() + tb - 1) / tb), tb>>>(d.q, d.k, d.v, d.dO, O, LSE, Dr, dQ, p.B, p.Hq, p.Hkv, p.T, W);
    const size_t nkr = (size_t)p.B * p.Hkv * p.T;
    ref_dkdv<<<(unsigned)((nkr + tb - 1) / tb), tb>>>(d.q, d.k, d.v, d.dO, LSE, Dr, dK, dV, p.B, p.Hq, p.Hkv, p.T, W);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    Ref r{dl(O, d.nq), dl(LSE, d.nl()), dl(dQ, d.nq), dl(dK, d.nk), dl(dV, d.nk)};
    for (float* x : {O, LSE, Dr, dQ, dK, dV}) cudaFree(x);
    return r;
}
static double rel_err(const std::vector<float>& k, const std::vector<float>& r) {
    double e = 0, m = 1e-30;
    for (size_t i = 0; i < r.size(); ++i) {
        if (!std::isfinite(k[i])) return INFINITY;
        e = std::max(e, std::fabs((double)k[i] - r[i])); m = std::max(m, (double)std::fabs(r[i]));
    }
    return e / m;
}

// ---- packed layouts --------------------------------------------------------------------------------------------------
struct LayoutDef { const char* name; int T; std::vector<std::vector<int>> rows; };   // chat lengths per row
struct Seg { int b, t0, L; bool pad; };
static std::vector<Seg> segments(const LayoutDef& lo) {
    std::vector<Seg> s;
    for (int b = 0; b < (int)lo.rows.size(); ++b) {
        int t = 0;
        for (int L : lo.rows[b]) { s.push_back({b, t, L, false}); t += L; }
        if (t > lo.T) { fprintf(stderr, "layout %s row %d is longer than T\n", lo.name, b); exit(3); }
        if (t < lo.T) s.push_back({b, t, lo.T - t, true});            // the row's pad tail: a segment of its own
    }
    return s;
}
static cuda::GqaChatPlanPtr make_plan(const LayoutDef& lo, const std::vector<Seg>& s, int Hq, int Hkv) {
    std::vector<int32_t> sb, st, sl;
    for (const Seg& g : s) { sb.push_back(g.b); st.push_back(g.t0); sl.push_back(g.L); }
    return cuda::gqa_chat_plan_make(sb.data(), st.data(), sl.data(), (int)s.size(), (int)lo.rows.size(), lo.T, Hq, Hkv);
}

// Host copy of a packed batch's inputs and the outputs of one call.
struct HostIn { Problem p; std::vector<bf16> q, k, v, dO; };
struct Out { std::vector<bf16> o, dq, dk, dv; std::vector<float> lse; };
static HostIn make_inputs(const Problem& p, uint64_t seed) {
    const size_t nq = (size_t)p.B * p.Hq * p.T * D, nk = (size_t)p.B * p.Hkv * p.T * D;
    return {p, randn_bf16(nq, seed + 1, 1.f), randn_bf16(nk, seed + 2, 1.f), randn_bf16(nk, seed + 3, 1.f),
            randn_bf16(nq, seed + 4, 1.f)};
}
static void upload(Data& d, const HostIn& h) { ul(d.q, h.q); ul(d.k, h.k); ul(d.v, h.v); ul(d.dO, h.dO); }
static Out run(Data& d, int W, const cuda::GqaChatPlan* plan) {
    forward(d, W, plan); backward(d, W, plan);
    return {dl(d.o, d.nq), dl(d.dq, d.nq), dl(d.dk, d.nk), dl(d.dv, d.nk), dl(d.lse, d.nl())};
}

// rows [r0, r0 + L) of every head of batch row b, from a [B, H, T, D] (or [B, H, T] when E = 1) host tensor
template <class X> static std::vector<X> rows_of(const std::vector<X>& t, int H, int T, int E, int b, int r0, int L) {
    std::vector<X> out((size_t)H * L * E);
    for (int h = 0; h < H; ++h)
        std::memcpy(&out[(size_t)h * L * E], &t[(((size_t)b * H + h) * T + r0) * E], (size_t)L * E * sizeof(X));
    return out;
}
template <class X> static size_t nbitdiff(const std::vector<X>& a, const std::vector<X>& b) {
    size_t c = 0; for (size_t i = 0; i < a.size(); ++i) c += std::memcmp(&a[i], &b[i], sizeof(X)) != 0; return c;
}
static double maxdiff(const std::vector<bf16>& a, const std::vector<bf16>& b) {
    double m = 0; for (size_t i = 0; i < a.size(); ++i) m = std::max(m, (double)std::fabs(f(a[i]) - f(b[i])));
    return m;
}
static float bf16r(float x) { return __bfloat162float(__float2bfloat16(x)); }
static double fa_atol(const std::vector<bf16>& dq) {   // FA's dQ tolerance: 2 x max |bf16(bf16(dq + 0.3) - 0.3) - dq|
    double m = 0;
    for (bf16 x : dq) { const float v = f(x); m = std::max(m, (double)std::fabs(bf16r(bf16r(v + 0.3f) - 0.3f) - v)); }
    return 2 * m;
}
static std::vector<float> to_f(const std::vector<bf16>& x) { std::vector<float> o(x.size()); for (size_t i = 0; i < x.size(); ++i) o[i] = f(x[i]); return o; }

// Compares the slices of one chat in two outputs (packed vs packed, or packed vs alone).
struct Cmp { size_t o = 0, lse = 0, dk = 0, dv = 0; double dq = 0, atol = 0; size_t n = 0; };
static Cmp compare_chat(const Out& a, const Problem& pa, int ba, int ra, const Out& b, const Problem& pb, int bb, int rb, int L) {
    Cmp c;
    c.o   = nbitdiff(rows_of(a.o,   pa.Hq,  pa.T, D, ba, ra, L), rows_of(b.o,   pb.Hq,  pb.T, D, bb, rb, L));
    c.lse = nbitdiff(rows_of(a.lse, pa.Hq,  pa.T, 1, ba, ra, L), rows_of(b.lse, pb.Hq,  pb.T, 1, bb, rb, L));
    c.dk  = nbitdiff(rows_of(a.dk,  pa.Hkv, pa.T, D, ba, ra, L), rows_of(b.dk,  pb.Hkv, pb.T, D, bb, rb, L));
    c.dv  = nbitdiff(rows_of(a.dv,  pa.Hkv, pa.T, D, ba, ra, L), rows_of(b.dv,  pb.Hkv, pb.T, D, bb, rb, L));
    const std::vector<bf16> qa = rows_of(a.dq, pa.Hq, pa.T, D, ba, ra, L), qb = rows_of(b.dq, pb.Hq, pb.T, D, bb, rb, L);
    c.dq = maxdiff(qa, qb); c.atol = fa_atol(qb);
    c.n = (size_t)L;
    return c;
}
static bool same(const Cmp& c) { return c.o == 0 && c.lse == 0 && c.dk == 0 && c.dv == 0 && c.dq <= c.atol; }

// The chat alone through the DENSE kernels: a row of roundup(L, 256), the chat at offset 0, random filler after it with
// dO = 0 (causal attention never lets the chat see the filler, and the filler's dS = 0 adds nothing to the chat's dK / dV).
static Out run_alone(const HostIn& h, const Seg& s, int W, uint64_t seed, int& Ta) {
    const Problem& pp = h.p;
    Ta = std::max(256, (s.L + 255) / 256 * 256);
    const Problem pa{1, pp.Hq, pp.Hkv, Ta};
    HostIn a{pa, randn_bf16((size_t)pa.Hq * Ta * D, seed + 11, 1.f), randn_bf16((size_t)pa.Hkv * Ta * D, seed + 12, 1.f),
             randn_bf16((size_t)pa.Hkv * Ta * D, seed + 13, 1.f), std::vector<bf16>((size_t)pa.Hq * Ta * D, __float2bfloat16(0.f))};
    auto put = [&](std::vector<bf16>& dst, const std::vector<bf16>& src, int H) {
        for (int hh = 0; hh < H; ++hh)
            std::memcpy(&dst[(size_t)hh * Ta * D], &src[(((size_t)s.b * H + hh) * pp.T + s.t0) * D], (size_t)s.L * D * sizeof(bf16));
    };
    put(a.q, h.q, pp.Hq); put(a.k, h.k, pp.Hkv); put(a.v, h.v, pp.Hkv); put(a.dO, h.dO, pp.Hq);
    Data d(pa); upload(d, a);
    return run(d, W, nullptr);
}

// The fp64 reference of one chat alone (its own T = L), with window W.
static Ref chat_reference(const HostIn& h, const Seg& s, int W) {
    const Problem& pp = h.p;
    const Problem pc{1, pp.Hq, pp.Hkv, s.L};
    Data d(pc);
    ul(d.q, rows_of(h.q, pp.Hq, pp.T, D, s.b, s.t0, s.L)); ul(d.k, rows_of(h.k, pp.Hkv, pp.T, D, s.b, s.t0, s.L));
    ul(d.v, rows_of(h.v, pp.Hkv, pp.T, D, s.b, s.t0, s.L)); ul(d.dO, rows_of(h.dO, pp.Hq, pp.T, D, s.b, s.t0, s.L));
    return reference(d, W);
}
// rel errors of a packed output's chat slice against a chat reference: O, LSE, dQ, dK, dV
static void chat_errors(const Out& o, const Problem& p, const Seg& s, const Ref& r, double e[5]) {
    e[0] = rel_err(to_f(rows_of(o.o, p.Hq, p.T, D, s.b, s.t0, s.L)), r.O);
    e[1] = rel_err(rows_of(o.lse, p.Hq, p.T, 1, s.b, s.t0, s.L), r.LSE);
    e[2] = rel_err(to_f(rows_of(o.dq, p.Hq, p.T, D, s.b, s.t0, s.L)), r.dQ);
    e[3] = rel_err(to_f(rows_of(o.dk, p.Hkv, p.T, D, s.b, s.t0, s.L)), r.dK);
    e[4] = rel_err(to_f(rows_of(o.dv, p.Hkv, p.T, D, s.b, s.t0, s.L)), r.dV);
}

// ---- P1 + P2 + P3 on one layout, one window, one head shape ------------------------------------------------------------
static void layout_window(const LayoutDef& lo, int W, int Hq, int Hkv, uint64_t seed, bool expect_mx) {
    const std::vector<Seg> segs = segments(lo);
    const Problem p{(int)lo.rows.size(), Hq, Hkv, lo.T};
    cuda::GqaChatPlanPtr plan = make_plan(lo, segs, Hq, Hkv);
    const HostIn h = make_inputs(p, seed);
    Data d(p); upload(d, h);
    const Out pw = run(d, W, plan.get());
    const Out p0 = run(d, 0, plan.get());
    int nchat = 0, nlong = 0, bad1 = 0, bad2s = 0, bad2l = 0, bad3 = 0;
    double worst_q1 = 0, worst_q2 = 0, worst3 = -1e30; std::string first1, first2, first3;
    const char* nm[5] = {"O", "LSE", "dQ", "dK", "dV"};
    for (size_t i = 0; i < segs.size(); ++i) {
        const Seg& s = segs[i];
        if (s.pad) continue;
        ++nchat;
        // P1: the chat alone through the dense kernels
        int Ta = 0;
        const Out al = run_alone(h, s, W, seed * 131 + i, Ta);
        const Problem pa{1, Hq, Hkv, Ta};
        const Cmp c1 = compare_chat(pw, p, s.b, s.t0, al, pa, 0, 0, s.L);
        worst_q1 = std::max(worst_q1, c1.atol > 0 ? c1.dq / c1.atol : (c1.dq > 0 ? 1e9 : 0));
        if (!same(c1)) {
            if (!bad1++) { char b[256]; snprintf(b, sizeof b, " first: chat %zu (L %d, row %d @%d, alone T %d): O %zu LSE %zu dK %zu dV %zu differ, dQ %.2e vs atol %.2e",
                                                   i, s.L, s.b, s.t0, Ta, c1.o, c1.lse, c1.dk, c1.dv, c1.dq, c1.atol); first1 = b; }
        }
        // P2: window W vs window 0 in the same batch
        const Cmp c2 = compare_chat(pw, p, s.b, s.t0, p0, p, s.b, s.t0, s.L);
        if (s.L <= W) {
            worst_q2 = std::max(worst_q2, c2.atol > 0 ? c2.dq / c2.atol : (c2.dq > 0 ? 1e9 : 0));
            if (!same(c2) && !bad2s++) { char b[256]; snprintf(b, sizeof b, " first: chat %zu (L %d <= W): O %zu LSE %zu dK %zu dV %zu differ, dQ %.2e vs atol %.2e",
                                                               i, s.L, c2.o, c2.lse, c2.dk, c2.dv, c2.dq, c2.atol); first2 = b; }
        } else {
            ++nlong;
            if (c2.o == 0 && !bad2l++) { char b[160]; snprintf(b, sizeof b, " first: chat %zu (L %d > W): O identical to window 0", i, s.L); first2 += b; }
            // P3: fp64, his T4 rule
            const Ref rw = chat_reference(h, s, W), r0 = chat_reference(h, s, 0);
            double ew[5], ec[5];
            chat_errors(pw, p, s, rw, ew); chat_errors(p0, p, s, r0, ec);
            for (int t = 0; t < 5; ++t) {
                const double margin = ew[t] - (2 * ec[t] + 2e-3);
                if (margin > worst3) worst3 = margin;
                if (margin > 0 && !bad3++) { char b[200]; snprintf(b, sizeof b, " first: chat %zu (L %d) %s window %.2e vs causal %.2e", i, s.L, nm[t], ew[t], ec[t]); first3 = b; }
            }
        }
    }
    char tag[160];
    snprintf(tag, sizeof tag, "%-8s W%-4d heads %d/%d (%d chats, %d longer than W; plan mx_hkv %d)", lo.name, W, Hq, Hkv,
             nchat, nlong, plan->mx_hkv);
    check(bad1 == 0, "P1 alone %s: every chat == alone (dense kernels), bit for bit (O, LSE, dK, dV); dQ worst %.2f of atol%s",
          tag, worst_q1, first1.c_str());
    check(bad2s == 0 && bad2l == 0, "P2 same  %s: chats <= W unchanged by the window (dQ worst %.2f of atol), chats > W changed%s",
          tag, worst_q2, first2.c_str());
    if (nlong > 0)
        check(bad3 == 0, "P3 fp64  %s: windowed error <= 2 x causal error + 2e-3 for every chat > W (worst margin %.2e)%s",
              tag, worst3, first3.c_str());
    if (expect_mx && Hq / Hkv >= 2 && getenv("BLU_GQA_MX") == nullptr)
        check(plan->mx_hkv != 0, "P1 mx    %s: the plan holds mixed units, so forward mode 5 ran", tag);
}

// ---- P4: every segment fits in W (item A): window W == window 0 ---------------------------------------------------------
static void p4_item_a(int Hq, int Hkv) {
    const LayoutDef lo{"short", 4096, {{1000, 1024, 600, 1000, 472}, {512, 1023, 1, 960, 600, 1000}}};   // pad tails 0 / 0
    const std::vector<Seg> segs = segments(lo);
    const Problem p{2, Hq, Hkv, lo.T};
    cuda::GqaChatPlanPtr plan = make_plan(lo, segs, Hq, Hkv);
    Data d(p); upload(d, make_inputs(p, 501));
    const Out pw = run(d, 1024, plan.get()), p0 = run(d, 0, plan.get());
    const size_t no = nbitdiff(pw.o, p0.o), nl = nbitdiff(pw.lse, p0.lse), nk = nbitdiff(pw.dk, p0.dk), nv = nbitdiff(pw.dv, p0.dv);
    const double dq = maxdiff(pw.dq, p0.dq), at = fa_atol(p0.dq);
    check(no == 0 && nl == 0 && nk == 0 && nv == 0 && dq <= at,
          "P4 item A heads %d/%d: longest segment %d <= W 1024: window == no window (O %zu LSE %zu dK %zu dV %zu differ; dQ %.2e vs atol %.2e)",
          Hq, Hkv, plan->gen_max_len, no, nl, nk, nv, dq, at);
}

// ---- P5: refusals ------------------------------------------------------------------------------------------------------
template <class F> static bool throws_with(F&& fn, const char* word, std::string& msg) {
    try { fn(); } catch (const std::runtime_error& e) { msg = e.what(); return word == nullptr || msg.find(word) != std::string::npos; }
    msg = "(no exception)"; return false;
}
static void p5_refusals() {
    const LayoutDef lo{"w1024", 4096, {{1023, 1024, 1025, 1024}, {2047, 2049}}};
    const std::vector<Seg> segs = segments(lo);
    const Problem p{2, 8, 4, lo.T};
    cuda::GqaChatPlanPtr plan = make_plan(lo, segs, 8, 4);
    Data d(p); upload(d, make_inputs(p, 601));
    std::string m;
    for (int W : {1000, -1}) {
        bool fw = throws_with([&] { forward(d, W, plan.get()); }, "window", m);
        check(fw, "P5 refuse packed forward  window %d throws: %s", W, m.c_str());
        forward(d, 0, plan.get());
        bool bw = throws_with([&] { backward(d, W, plan.get()); }, "window", m);
        check(bw, "P5 refuse packed backward window %d throws: %s", W, m.c_str());
    }
    const Out pT = run(d, 4096, plan.get()), p0 = run(d, 0, plan.get());
    const size_t n = nbitdiff(pT.o, p0.o) + nbitdiff(pT.lse, p0.lse) + nbitdiff(pT.dk, p0.dk) + nbitdiff(pT.dv, p0.dv);
    check(n == 0 && maxdiff(pT.dq, p0.dq) <= fa_atol(p0.dq),
          "P5 refuse window 4096 >= T runs and == window 0 (%zu elements of O/LSE/dK/dV differ)", n);
    const int32_t sb[3] = {0, 0, 0}, st[3] = {0, 2048, 2048}, sl[3] = {2048, 0, 2048};
    bool z = throws_with([&] { cuda::gqa_chat_plan_make(sb, st, sl, 3, 1, 4096, 8, 4); }, nullptr, m);
    check(z, "P5 refuse a zero-length segment is refused by the plan builder: %s", m.c_str());
}

// ---- P6: repeat ---------------------------------------------------------------------------------------------------------
static void p6_repeat() {
    const LayoutDef lo{"w1024", 4096, {{1023, 1024, 1025, 1024}, {2047, 2049}, {4096}, {1, 63, 65, 127, 129, 191, 193, 255, 257, 64, 2000}}};
    const std::vector<Seg> segs = segments(lo);
    const Problem p{4, 8, 4, lo.T};
    cuda::GqaChatPlanPtr plan = make_plan(lo, segs, 8, 4);
    Data d(p); upload(d, make_inputs(p, 701));
    const Out a = run(d, 1024, plan.get()), b = run(d, 1024, plan.get());
    const size_t n = nbitdiff(a.o, b.o) + nbitdiff(a.lse, b.lse) + nbitdiff(a.dk, b.dk) + nbitdiff(a.dv, b.dv);
    const double dq = maxdiff(a.dq, b.dq), at = fa_atol(a.dq);
    check(n == 0 && dq <= at, "P6 repeat the same packed run twice (W 1024, heads 8/4): O/LSE/dK/dV %zu elements differ; dQ %.2e vs atol %.2e", n, dq, at);
}

// ---- P8: through the autograd op ----------------------------------------------------------------------------------------
static std::vector<bf16> grad_bf16(const Tensor& t, size_t n) {
    Tensor g = t.grad_view();
    std::vector<bf16> out(n);
    CK(cudaDeviceSynchronize());
    if (g.dtype() == Dtype::Bfloat16) CK(cudaMemcpy(out.data(), g.data(), n * 2, cudaMemcpyDeviceToHost));
    else if (g.dtype() == Dtype::Float32) {
        std::vector<float> h(n); CK(cudaMemcpy(h.data(), g.data(), n * 4, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < n; ++i) out[i] = __float2bfloat16(h[i]);
    } else { printf("  (unexpected grad dtype)\n"); }
    return out;
}
static void p8_autograd(int W) {
    const LayoutDef lo{"w1024", 4096, {{1023, 1024, 1025, 1024}, {2047, 2049}}};
    const std::vector<Seg> segs = segments(lo);
    const Problem p{2, 8, 4, lo.T};
    cuda::GqaChatPlanPtr plan = make_plan(lo, segs, 8, 4);
    const HostIn h = make_inputs(p, 801);
    Data d(p); upload(d, h);
    const Out direct = run(d, W, plan.get());
    const DeviceIndex dev(Device::CUDA, 0);
    auto make = [&](const Shape& s, const std::vector<bf16>& x, bool rg) {
        Tensor t(s, Dtype::Bfloat16, dev, rg);
        CK(cudaMemcpy(t.data(), x.data(), x.size() * sizeof(bf16), cudaMemcpyHostToDevice));
        return t;
    };
    const Shape sq{{p.B, p.Hq, p.T, D}}, sk{{p.B, p.Hkv, p.T, D}};
    Tensor q = make(sq, h.q, true), k = make(sk, h.k, true), v = make(sk, h.v, true), g = make(sq, h.dO, false);
    Tensor o = autograd::scaled_dot_product_attention(q, k, v, p.Hq, p.Hkv, /*is_causal=*/true, /*dropout_p=*/0.0f,
                                                      autograd::SDPBackend::FlashAttention, plan, W);
    o.backward(&g);
    CK(cudaDeviceSynchronize());
    std::vector<bf16> oa(d.nq);
    if (o.dtype() == Dtype::Bfloat16) CK(cudaMemcpy(oa.data(), o.data(), d.nq * 2, cudaMemcpyDeviceToHost));
    else { std::vector<float> t(d.nq); CK(cudaMemcpy(t.data(), o.data(), d.nq * 4, cudaMemcpyDeviceToHost));
           for (size_t i = 0; i < d.nq; ++i) oa[i] = __float2bfloat16(t[i]); }
    const std::vector<bf16> dqa = grad_bf16(q, d.nq), dka = grad_bf16(k, d.nk), dva = grad_bf16(v, d.nk);
    const size_t no = nbitdiff(oa, direct.o), nk = nbitdiff(dka, direct.dk), nv = nbitdiff(dva, direct.dv);
    check(no == 0 && nk == 0 && nv == 0, "P8 autograd W%d with the chat plan: O / dK / dV equal the direct calls bit for bit "
          "(%zu / %zu / %zu differ); dQ max diff %.1e", W, no, nk, nv, maxdiff(dqa, direct.dq));
}

int main() {
    cudaDeviceProp prop{}; CK(cudaGetDeviceProperties(&prop, 0));
    const char* hs = getenv("BLU_VJ2_HSUM"); const char* f32 = getenv("BLU_VJ2_DKDV_F32"); const char* mx = getenv("BLU_GQA_MX");
    printf("test_swa_packed on %s (sm_%d%d) | BLU_VJ2_HSUM=%s BLU_VJ2_DKDV_F32=%s BLU_GQA_MX=%s\n", prop.name, prop.major,
           prop.minor, hs ? hs : "(unset)", f32 ? f32 : "(unset)", mx ? mx : "(unset)");
    if (prop.major != 9) { printf("  [SKIP] the window kernels are Hopper only\n"); return 0; }
    const LayoutDef edge128{"edge128", 4096, {{127, 128, 129, 256, 257, 1, 2, 63, 64, 65, 1000, 2000},
                                             {255, 512, 513, 383, 384, 385, 640, 1024}}};
    const LayoutDef w1024{"w1024", 4096, {{1023, 1024, 1025, 1024}, {2047, 2049}, {4096},
                                          {1, 63, 65, 127, 129, 191, 193, 255, 257, 64, 2000}}};
    const LayoutDef mxl{"mx", 4096, {{33, 161, 289, 417, 63, 31, 545, 1, 191, 319, 447, 15, 177, 305},
                                     {3000, 63, 31, 161, 289}}};
    const LayoutDef long8k{"long8k", 8192, {{8000}}};
    const int heads[3][2] = {{8, 4}, {4, 1}, {8, 8}};
    uint64_t seed = 1000;
    for (const auto& hh : heads) {
        for (int W : {128, 256}) layout_window(edge128, W, hh[0], hh[1], seed++, false);
        for (int W : {512, 1024}) layout_window(w1024, W, hh[0], hh[1], seed++, false);
        for (int W : {1024, 128}) layout_window(mxl, W, hh[0], hh[1], seed++, true);
        p4_item_a(hh[0], hh[1]);
    }
    // 2026-11-20 step 9: the CP shapes of the 350M (U = 2 -> 4/2, U = 4 -> 2/1)
    const int cp_heads[2][2] = {{4, 2}, {2, 1}};
    for (const auto& hh : cp_heads) {
        for (int W : {512, 1024}) layout_window(w1024, W, hh[0], hh[1], seed++, false);
        layout_window(mxl, 1024, hh[0], hh[1], seed++, true);
        p4_item_a(hh[0], hh[1]);
    }
    layout_window(long8k, 1024, 8, 4, seed++, false);   // P7
    p5_refusals();
    p6_repeat();
    p8_autograd(1024);
    printf("%s (%d passed, %d failed)\n", g_fail ? "FAILED" : "ALL PASSED", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
