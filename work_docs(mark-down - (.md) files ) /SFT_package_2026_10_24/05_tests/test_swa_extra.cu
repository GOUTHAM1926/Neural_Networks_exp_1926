// ============================================================================
// test_swa_extra.cu -- 2026-11-20: extra checks for the SWA port (Navin's b913f32 onto TI 334b244), next to his
// Tests/kernel_tests/test_swa_attention.cu (kept unchanged). The fp64 reference (ref_fwd / ref_dq / ref_dkdv), rel_err
// and the random-input rule are copied unchanged from his file.
//
//   E1 random   the 350M shape (8 query / 4 KV heads, T 4096, W 1024), q x1 and x3, against the fp64 reference, his T4
//               rule: PASS if the windowed error <= 2 x the causal error + 2e-3 (relative), for O, LSE, dQ, dK, dV.
//   E2 throws   window 1000 (not a multiple of 128), window -1, and window 1024 with a chats plan (packed rows) must each
//               throw std::runtime_error naming the window, in the forward and in the backward; the same plan with
//               window 0 must NOT throw (so the throw comes from the window, not from the plan).
//               2026-11-20 step 7: packed rows + window run now (forward modes 4 / 5). So with the chats plan, window 1024
//               must RUN (forward + backward) and window 1000 must still throw; the full packed checks are in
//               test_swa_packed.cu.
//   E3 autograd autograd::scaled_dot_product_attention(..., window 1024) + backward: O and dK / dV bit for bit equal to
//               the direct kernel calls with window 1024 (dQ: max difference reported, the dQ accumulator is a TMA
//               reduce-add), and dK differs from the window-0 result (the backward node really got the window).
//
// The backward variant is chosen once per process, so run it three times: default (idea 4 on), BLU_VJ2_HSUM=0
// (per-query-head path), BLU_VJ2_DKDV_F32=1 (fp32 dK / dV partials).
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

static int g_fail = 0;
static void check(bool ok, const char* fmt, ...) __attribute__((format(printf, 2, 3)));
static void check(bool ok, const char* fmt, ...) {
    va_list ap; va_start(ap, fmt);
    printf("  [%s] ", ok ? "PASS" : "FAIL"); vprintf(fmt, ap); printf("\n");
    va_end(ap);
    if (!ok) ++g_fail;
}

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
static std::vector<bf16> to_bf16(const std::vector<float>& h) {
    std::vector<bf16> b(h.size()); for (size_t i = 0; i < h.size(); ++i) b[i] = __float2bfloat16(h[i]); return b;
}
static void ul(bf16* x, const std::vector<float>& h) {
    const std::vector<bf16> b = to_bf16(h);
    CK(cudaMemcpy(x, b.data(), b.size() * sizeof(bf16), cudaMemcpyHostToDevice));
}
static std::vector<float> randn(size_t n, uint64_t seed, float s) {
    std::mt19937_64 g(seed); std::normal_distribution<float> nd(0.f, s);
    std::vector<float> h(n); for (auto& x : h) x = nd(g); return h;
}

// ---- fp64-accumulated reference (copied unchanged from test_swa_attention.cu) ----------------------------------------
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
template <class K> static double rel_err(const std::vector<K>& k, const std::vector<float>& r) {
    double e = 0, m = 1e-30;
    for (size_t i = 0; i < r.size(); ++i) {
        double kv;
        if constexpr (std::is_same<K, bf16>::value) kv = __bfloat162float(k[i]); else kv = k[i];
        if (!std::isfinite(kv)) return INFINITY;
        e = std::max(e, std::fabs(kv - r[i])); m = std::max(m, (double)std::fabs(r[i]));
    }
    return e / m;
}
static void fill_inputs(Data& d, uint64_t seed, float qs) {
    ul(d.q, randn(d.nq, seed + 1, qs)); ul(d.k, randn(d.nk, seed + 2, 1.f));
    ul(d.v, randn(d.nk, seed + 3, 1.f)); ul(d.dO, randn(d.nq, seed + 4, 1.f));
}

// ---- E1: the 350M shape against the fp64 reference (his T4 rule) ----------------------------------------------------------
static void e1_random(const Problem& p, int W, float qs, uint64_t seed) {
    Data d(p);
    fill_inputs(d, seed, qs);
    double ec[5], ew[5];
    for (int pass = 0; pass < 2; ++pass) {
        const int win = pass ? W : 0;
        const Ref r = reference(d, win);
        forward(d, win); backward(d, win);
        double* e = pass ? ew : ec;
        e[0] = rel_err(dl(d.o, d.nq), r.O);   e[1] = rel_err(dl(d.lse, d.nl()), r.LSE);
        e[2] = rel_err(dl(d.dq, d.nq), r.dQ); e[3] = rel_err(dl(d.dk, d.nk), r.dK); e[4] = rel_err(dl(d.dv, d.nk), r.dV);
    }
    const char* nm[5] = {"O", "LSE", "dQ", "dK", "dV"};
    bool ok = true; char buf[512]; int n = 0;
    for (int t = 0; t < 5; ++t) {
        ok &= ew[t] <= 2 * ec[t] + 2e-3;
        n += snprintf(buf + n, sizeof(buf) - n, " %s %.1e/%.1e", nm[t], ew[t], ec[t]);
    }
    check(ok, "E1 random  B%d %d/%d T%d W%d q x%.0f: window/causal rel err%s", p.B, p.Hq, p.Hkv, p.T, W, qs, buf);
}

// ---- E2: settings that must throw ---------------------------------------------------------------------------------------
template <class F> static bool throws_window(F&& fn, std::string& msg) {
    try { fn(); } catch (const std::runtime_error& e) { msg = e.what(); return msg.find("window") != std::string::npos; }
    msg = "(no exception)"; return false;
}
static void e2_must_throw() {
    const Problem p{2, 8, 4, 4096};
    Data d(p); fill_inputs(d, 7, 1.f);
    forward(d, 0);                                              // valid O / LSE for the backward calls below
    std::string m;
    for (int W : {1000, -1}) {
        bool f = throws_window([&] { forward(d, W); }, m);
        check(f, "E2 forward  window %d throws: %s", W, m.c_str());
        forward(d, 0);
        bool b = throws_window([&] { backward(d, W); }, m);
        check(b, "E2 backward window %d throws: %s", W, m.c_str());
    }
    // a valid plan: 2 rows, each split into 2 chats of 2048
    const int32_t sb[4] = {0, 0, 1, 1}, st[4] = {0, 2048, 0, 2048}, sl[4] = {2048, 2048, 2048, 2048};
    cuda::GqaChatPlanPtr plan = cuda::gqa_chat_plan_make(sb, st, sl, 4, p.B, p.T, p.Hq, 0);
    bool ok0 = true;
    try { forward(d, 0, plan.get()); backward(d, 0, plan.get()); } catch (const std::exception& e) { ok0 = false; m = e.what(); }
    check(ok0, "E2 the same chats plan with window 0 runs (forward + backward)%s", ok0 ? "" : (": " + m).c_str());
    // Old checks (kept, 2026-11-20 step 7: packed rows + window are implemented now, so window 1024 + plan must run):
    // bool f = throws_window([&] { forward(d, 1024, plan.get()); }, m);
    // check(f, "E2 forward  window 1024 + chats plan throws: %s", m.c_str());
    // forward(d, 0, plan.get());
    // bool b = throws_window([&] { backward(d, 1024, plan.get()); }, m);
    // check(b, "E2 backward window 1024 + chats plan throws: %s", m.c_str());
    bool ok1 = true;
    try { forward(d, 1024, plan.get()); backward(d, 1024, plan.get()); } catch (const std::exception& e) { ok1 = false; m = e.what(); }
    check(ok1, "E2 the chats plan with window 1024 runs (forward + backward; packed rows + window, step 7)%s",
          ok1 ? "" : (": " + m).c_str());
    bool f = throws_window([&] { forward(d, 1000, plan.get()); }, m);
    check(f, "E2 forward  window 1000 + chats plan throws: %s", m.c_str());
    forward(d, 0, plan.get());
    bool b = throws_window([&] { backward(d, 1000, plan.get()); }, m);
    check(b, "E2 backward window 1000 + chats plan throws: %s", m.c_str());
}

// ---- E3: through the autograd op ------------------------------------------------------------------------------------------
static std::vector<float> grad_floats(const Tensor& t, size_t n) {
    Tensor g = t.grad_view();
    std::vector<float> out(n);
    CK(cudaDeviceSynchronize());
    if (g.dtype() == Dtype::Float32) {
        CK(cudaMemcpy(out.data(), g.data(), n * 4, cudaMemcpyDeviceToHost));
    } else if (g.dtype() == Dtype::Bfloat16) {
        std::vector<bf16> h(n); CK(cudaMemcpy(h.data(), g.data(), n * 2, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < n; ++i) out[i] = __bfloat162float(h[i]);
    } else {
        printf("  (unexpected grad dtype)\n"); out.assign(n, NAN);
    }
    return out;
}
static std::vector<float> bf16_floats(const bf16* x, size_t n) {
    const std::vector<bf16> h = dl(x, n); std::vector<float> f(n);
    for (size_t i = 0; i < n; ++i) f[i] = __bfloat162float(h[i]);
    return f;
}
static size_t ndiff(const std::vector<float>& a, const std::vector<float>& b) {
    size_t c = 0; for (size_t i = 0; i < a.size(); ++i) c += std::memcmp(&a[i], &b[i], 4) != 0; return c;
}
static double maxdiff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0; for (size_t i = 0; i < a.size(); ++i) m = std::max(m, (double)std::fabs(a[i] - b[i])); return m;
}
static void e3_autograd(int W) {
    const Problem p{2, 8, 4, 4096};
    Data d(p);
    const std::vector<float> hq = randn(d.nq, 41, 1.f), hk = randn(d.nk, 42, 1.f), hv = randn(d.nk, 43, 1.f),
                             hg = randn(d.nq, 44, 1.f);
    ul(d.q, hq); ul(d.k, hk); ul(d.v, hv); ul(d.dO, hg);
    forward(d, 0); backward(d, 0);
    const std::vector<float> dk0 = bf16_floats(d.dk, d.nk);
    forward(d, W); backward(d, W);
    const std::vector<float> o1 = bf16_floats(d.o, d.nq), dq1 = bf16_floats(d.dq, d.nq),
                             dk1 = bf16_floats(d.dk, d.nk), dv1 = bf16_floats(d.dv, d.nk);

    const DeviceIndex dev(Device::CUDA, 0);
    auto make = [&](const Shape& s, const std::vector<float>& h, bool rg) {
        Tensor t(s, Dtype::Bfloat16, dev, rg);
        const std::vector<bf16> b = to_bf16(h);
        CK(cudaMemcpy(t.data(), b.data(), b.size() * sizeof(bf16), cudaMemcpyHostToDevice));
        return t;
    };
    const Shape sq{{p.B, p.Hq, p.T, D}}, sk{{p.B, p.Hkv, p.T, D}};
    Tensor q = make(sq, hq, true), k = make(sk, hk, true), v = make(sk, hv, true), g = make(sq, hg, false);
    Tensor o = autograd::scaled_dot_product_attention(q, k, v, p.Hq, p.Hkv, /*is_causal=*/true, /*dropout_p=*/0.0f,
                                                      autograd::SDPBackend::FlashAttention, /*chats=*/nullptr, W);
    o.backward(&g);
    CK(cudaDeviceSynchronize());
    std::vector<float> oa(d.nq);
    if (o.dtype() == Dtype::Bfloat16) oa = bf16_floats(reinterpret_cast<const bf16*>(o.data()), d.nq);
    else CK(cudaMemcpy(oa.data(), o.data(), d.nq * 4, cudaMemcpyDeviceToHost));
    const std::vector<float> dqa = grad_floats(q, d.nq), dka = grad_floats(k, d.nk), dva = grad_floats(v, d.nk);
    const size_t no = ndiff(oa, o1), nk = ndiff(dka, dk1), nv = ndiff(dva, dv1), n0 = ndiff(dka, dk0);
    check(no == 0, "E3 autograd W%d: O equals the direct kernel call bit for bit (%zu of %zu differ)", W, no, d.nq);
    check(nk == 0 && nv == 0, "E3 autograd W%d: dK / dV equal the direct backward with window %d bit for bit (%zu / %zu differ); "
          "dQ max diff %.1e", W, W, nk, nv, maxdiff(dqa, dq1));
    check(n0 > 0, "E3 autograd W%d: dK differs from the window-0 backward (%zu of %zu differ): the backward node got the window",
          W, n0, d.nk);
}

int main() {
    cudaDeviceProp prop{}; CK(cudaGetDeviceProperties(&prop, 0));
    const char* hs = getenv("BLU_VJ2_HSUM"); const char* f32 = getenv("BLU_VJ2_DKDV_F32");
    printf("test_swa_extra on %s (sm_%d%d) | BLU_VJ2_HSUM=%s BLU_VJ2_DKDV_F32=%s\n", prop.name, prop.major, prop.minor,
           hs ? hs : "(unset)", f32 ? f32 : "(unset)");
    if (prop.major != 9) { printf("  [SKIP] the window kernels are Hopper only\n"); return 0; }
    e1_random({2, 8, 4, 4096}, 1024, 1.f, 101);
    e1_random({2, 8, 4, 4096}, 1024, 3.f, 102);
    e2_must_throw();
    e3_autograd(1024);
    printf("%s (%d failed)\n", g_fail ? "FAILED" : "ALL PASSED", g_fail);
    return g_fail ? 1 : 0;
}
