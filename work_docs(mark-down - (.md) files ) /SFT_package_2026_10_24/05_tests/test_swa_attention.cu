// ============================================================================
// test_swa_attention.cu -- sliding-window GQA attention (2026-11-04): the v62 forward's MODE_WIN and the Vj2
// backward's WIN, through cuda::gqa_flash_attn_forward_qkv / gqa_flash_attn_backward_qkv with window > 0.
// A query at row i sees keys max(0, i - W + 1) .. i (HF sliding_window = W).
//
//   T1 count     K = 0, so every score is exactly 0 and LSE_i = log(number of keys row i sees). Checks
//                LSE_i == log(min(i + 1, W)) for every row (fp32, |err| < 1e-4): the window holds exactly W keys.
//   T2 spike fwd K = 0 except one key per (batch, kv head) with score 8 * 64 / sqrt(128) = 45.25 (exact in bf16),
//                V of that key = 64 e_0: O_i[0] ~ 64 and LSE_i = 45.25 for exactly the rows whose window holds it.
//                Spikes sit on tile edges.
//   T3 spike bwd one kv head per query head, W in {128, 256}, dO = e_0: dV[spike][0] = sum of P over the rows that see
//                it = min(W, T - spike) (P ~ 1 there), an integer bf16 holds exactly. A row too many or too few
//                changes it by 1.
//   T4 random    random Q/K/V/dO against an fp64-accumulated reference built from the same bf16 inputs. The causal
//                kernels' error against the causal reference on the same inputs sets the scale: PASS if the windowed
//                error <= 2 x causal + 2e-3 (relative to the tensor's max |reference|), for O, LSE, dQ, dK, dV.
//                Outputs are filled with NaN first, so an unwritten element fails.
//   T5 dispatch  window >= T is the causal kernel (bitwise); windowed O / LSE / dK / dV bitwise run to run.
//   T6 bench     forward and backward at the 135M training shape (B 16, 4 / 2 heads, T 4096), causal vs W 1024.
//
// Build + run (from Tensor-Implementations/, after `make tensor`):
//   make run-cuda FILE=Tests/kernel_tests/test_swa_attention.cu
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
#include <string>
#include <type_traits>
#include <vector>

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

// Device buffers of one problem. q/o/dO/dq: [B, Hq, T, D]; k/v/dk/dv: [B, Hkv, T, D]; lse: [B, Hq, T].
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

static void forward(Data& d, int window) {
    CK(cudaMemset(d.o, 0xFF, d.nq * sizeof(bf16)));             // NaN: an unwritten element shows up
    CK(cudaMemset(d.lse, 0xFF, d.nl() * sizeof(float)));
    cuda::gqa_flash_attn_forward_qkv(cb(d.q), cb(d.k), cb(d.v), mb(d.o), d.lse,
                                     d.p.B, d.p.Hq, d.p.Hkv, d.p.T, D, true, nullptr, window);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
}
static void backward(Data& d, int window) {
    CK(cudaMemset(d.dq, 0xFF, d.nq * sizeof(bf16)));
    CK(cudaMemset(d.dk, 0xFF, d.nk * sizeof(bf16)));
    CK(cudaMemset(d.dv, 0xFF, d.nk * sizeof(bf16)));
    cuda::gqa_flash_attn_backward_qkv(cb(d.q), cb(d.k), cb(d.v), cb(d.o), cb(d.dO), d.lse, mb(d.dq), mb(d.dk), mb(d.dv),
                                      d.p.B, d.p.Hq, d.p.Hkv, d.p.T, D, true, nullptr, window);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
}

template <class T> static std::vector<T> dl(const T* x, size_t n) {
    std::vector<T> h(n); CK(cudaMemcpy(h.data(), x, n * sizeof(T), cudaMemcpyDeviceToHost)); return h;
}
static void ul(bf16* x, const std::vector<float>& h) {
    std::vector<bf16> b(h.size());
    for (size_t i = 0; i < h.size(); ++i) b[i] = __float2bfloat16(h[i]);
    CK(cudaMemcpy(x, b.data(), b.size() * sizeof(bf16), cudaMemcpyHostToDevice));
}
static std::vector<float> randn(size_t n, uint64_t seed, float s) {
    std::mt19937_64 g(seed); std::normal_distribution<float> nd(0.f, s);
    std::vector<float> h(n); for (auto& x : h) x = nd(g); return h;
}
static float f(bf16 x) { return __bfloat162float(x); }

// ---- fp64-accumulated reference (same bf16 inputs), one thread per (b, head, row) -------------------------------
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
// D_i = dO_i . O_i (reference O), then dQ_i = scale * sum_j dS_ij K_j.
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
// dK_j = scale * sum_{group heads, rows i that see j} dS_ij Q_i;  dV_j = sum P_ij dO_i.
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

// max |kernel - ref| / max |ref|; non-finite kernel values count as an infinite error.
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

// ---- T1: the window holds exactly W keys ------------------------------------------------------------------------------
static void t1_count(const Problem& p, int W) {
    Data d(p);
    std::vector<float> q(d.nq, 0.f), z(d.nk, 0.f);
    for (size_t r = 0; r < d.nq / D; ++r) q[r * D] = 1.f;      // any q: K = 0 makes every score 0
    ul(d.q, q); ul(d.k, z); ul(d.v, randn(d.nk, 7, 1.f));
    forward(d, W);
    auto lse = dl(d.lse, d.nl());
    double worst = 0; long bad = -1;
    for (size_t r = 0; r < lse.size(); ++r) {
        const int i = r % p.T;
        const int n = (W > 0 && W < p.T) ? std::min(i + 1, W) : i + 1;
        const double e = std::fabs(lse[r] - std::log((double)n));
        if (!(e <= worst)) { worst = e; bad = (long)r; }
    }
    check(worst < 1e-4, "T1 count  B%d %d/%d T%-5d W%-5d: max |LSE - log(keys seen)| = %.2e%s", p.B, p.Hq, p.Hkv, p.T, W,
          worst, worst < 1e-4 ? "" : (" at row " + std::to_string(bad % p.T)).c_str());
}

// ---- T2: which keys a row sees -------------------------------------------------------------------------------------------
static void t2_spike_fwd(int T, int W, const std::vector<int>& spikes) {
    const Problem p{(int)spikes.size(), 4, 2, T};               // batch b holds spike spikes[b] in both kv heads
    Data d(p);
    const float a = 8.f, c = 64.f;                              // exact in bf16: score a * c / sqrt(D) on the spike, 0 elsewhere
    const float s = a * c / std::sqrt((float)D);
    std::vector<float> q(d.nq, 0.f), k(d.nk, 0.f), v = randn(d.nk, 11, 0.5f);
    for (size_t r = 0; r < d.nq / D; ++r) q[r * D] = a;
    for (int b = 0; b < p.B; ++b)
        for (int h = 0; h < p.Hkv; ++h) {
            const size_t row = ((size_t)(b * p.Hkv + h) * T + spikes[b]) * D;
            k[row] = c; v[row] = 64.f;
        }
    ul(d.q, q); ul(d.k, k); ul(d.v, v);
    forward(d, W);
    auto o = dl(d.o, d.nq); auto lse = dl(d.lse, d.nl());
    long wrong = 0; std::string first;
    for (int b = 0; b < p.B; ++b)
        for (int h = 0; h < p.Hq; ++h)
            for (int i = 0; i < T; ++i) {
                const size_t r = (size_t)(b * p.Hq + h) * T + i;
                const bool sees = spikes[b] <= i && i - spikes[b] < W;
                const bool got  = f(o[r * D]) > 32.f && std::fabs(lse[r] - s) < 1e-3f;
                const bool none = f(o[r * D]) < 32.f && lse[r] < 10.f;   // log(keys seen) <= log(4096) = 8.3
                if (sees ? !got : !none) {
                    if (!wrong) first = " (first: spike " + std::to_string(spikes[b]) + ", row " + std::to_string(i) + ")";
                    ++wrong;
                }
            }
    check(wrong == 0, "T2 spike fwd T%d W%-5d %zu spike positions: %ld rows disagree with the window%s", T, W,
          spikes.size(), wrong, first.c_str());
}

// ---- T3: how many rows reach a key in the backward -------------------------------------------------------------------
static void t3_spike_bwd(int T, int W, const std::vector<int>& spikes) {
    const Problem p{(int)spikes.size(), 4, 4, T};               // one kv head per query head: counts stay exact
    Data d(p);
    const float a = 8.f, c = 64.f;                              // spike score 45.25: P = 1 in fp32 where it is seen
    std::vector<float> q(d.nq, 0.f), k(d.nk, 0.f), g(d.nq, 0.f), v = randn(d.nk, 13, 0.5f);
    for (size_t r = 0; r < d.nq / D; ++r) { q[r * D] = a; g[r * D] = 1.f; }
    for (int b = 0; b < p.B; ++b)
        for (int h = 0; h < p.Hkv; ++h) k[((size_t)(b * p.Hkv + h) * T + spikes[b]) * D] = c;
    ul(d.q, q); ul(d.k, k); ul(d.v, v); ul(d.dO, g);
    forward(d, W); backward(d, W);
    auto dv = dl(d.dv, d.nk);
    long wrong = 0; std::string first;
    for (int b = 0; b < p.B; ++b)
        for (int h = 0; h < p.Hkv; ++h) {
            const double want = std::min(W, T - spikes[b]);
            const double got = f(dv[((size_t)(b * p.Hkv + h) * T + spikes[b]) * D]);
            if (std::fabs(got - want) > 0.25) {
                if (!wrong) first = " (first: spike " + std::to_string(spikes[b]) + ": dV " + std::to_string(got) +
                                    ", want " + std::to_string((int)want) + ")";
                ++wrong;
            }
        }
    check(wrong == 0, "T3 spike bwd T%d W%-5d %zu spike positions: %ld dV counts off%s", T, W, spikes.size(), wrong,
          first.c_str());
}

// ---- T4: random inputs against the reference ---------------------------------------------------------------------------
static void t4_random(const Problem& p, int W, float qs, uint64_t seed) {
    Data d(p);
    fill_inputs(d, seed, qs);
    double ec[5], ew[5];
    for (int pass = 0; pass < 2; ++pass) {                      // 0: causal (the scale), 1: windowed
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
    check(ok, "T4 random  B%d %2d/%d T%-5d W%-5d q x%.0f: window/causal rel err%s", p.B, p.Hq, p.Hkv, p.T, W, qs, buf);
}

// ---- T5: dispatch and determinism ----------------------------------------------------------------------------------------
static void t5_dispatch(const Problem& p, int W) {
    Data d(p);
    fill_inputs(d, 99, 1.f);
    forward(d, 0); backward(d, 0);
    auto o0 = dl(d.o, d.nq); auto l0 = dl(d.lse, d.nl()); auto k0 = dl(d.dk, d.nk); auto v0 = dl(d.dv, d.nk);
    auto q0 = dl(d.dq, d.nq);
    forward(d, p.T); backward(d, p.T);                          // window == T: the causal kernels
    bool same = !memcmp(o0.data(), dl(d.o, d.nq).data(), d.nq * 2) && !memcmp(l0.data(), dl(d.lse, d.nl()).data(), d.nl() * 4)
             && !memcmp(k0.data(), dl(d.dk, d.nk).data(), d.nk * 2) && !memcmp(v0.data(), dl(d.dv, d.nk).data(), d.nk * 2);
    check(same, "T5 window = T (%d) gives the causal kernels' O, LSE, dK, dV bitwise", p.T);
    forward(d, W); backward(d, W);
    auto o1 = dl(d.o, d.nq); auto l1 = dl(d.lse, d.nl()); auto k1 = dl(d.dk, d.nk); auto v1 = dl(d.dv, d.nk);
    auto q1 = dl(d.dq, d.nq);
    forward(d, W); backward(d, W);
    auto q2 = dl(d.dq, d.nq);
    same = !memcmp(o1.data(), dl(d.o, d.nq).data(), d.nq * 2) && !memcmp(l1.data(), dl(d.lse, d.nl()).data(), d.nl() * 4)
        && !memcmp(k1.data(), dl(d.dk, d.nk).data(), d.nk * 2) && !memcmp(v1.data(), dl(d.dv, d.nk).data(), d.nk * 2);
    double dqd = 0; for (size_t i = 0; i < q1.size(); ++i) dqd = std::max(dqd, (double)std::fabs(f(q1[i]) - f(q2[i])));
    check(same, "T5 W%d twice: O, LSE, dK, dV bitwise equal (dQ max diff %.1e: TMA-reduced, as the causal kernel)", W, dqd);
    bool differs = memcmp(o0.data(), o1.data(), d.nq * 2) != 0;
    check(differs, "T5 W%d output differs from causal (the window is applied at all)", W);
}

// ---- T6: timing -------------------------------------------------------------------------------------------------------------
static void t6_bench(const Problem& p, int W) {
    Data d(p);
    fill_inputs(d, 5, 1.f);
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    auto time = [&](auto&& fn) {
        for (int i = 0; i < 5; ++i) fn();
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(e0));
        for (int i = 0; i < 20; ++i) fn();
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
        float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); return ms / 20;
    };
    float tf[2], tb[2];
    for (int k = 0; k < 2; ++k) {
        const int win = k ? W : 0;
        tf[k] = time([&] { cuda::gqa_flash_attn_forward_qkv(cb(d.q), cb(d.k), cb(d.v), mb(d.o), d.lse, p.B, p.Hq, p.Hkv,
                                                             p.T, D, true, nullptr, win); });
        tb[k] = time([&] { cuda::gqa_flash_attn_backward_qkv(cb(d.q), cb(d.k), cb(d.v), cb(d.o), cb(d.dO), d.lse,
                                                              mb(d.dq), mb(d.dk), mb(d.dv), p.B, p.Hq, p.Hkv, p.T, D, true,
                                                              nullptr, win); });
    }
    printf("  [INFO] T6 bench B%d %d/%d T%d: forward %.3f ms causal, %.3f ms W%d (%.2fx); backward %.3f / %.3f ms (%.2fx)\n",
           p.B, p.Hq, p.Hkv, p.T, tf[0], tf[1], W, tf[0] / tf[1], tb[0], tb[1], tb[0] / tb[1]);
}

int main() {
    cudaDeviceProp prop{}; CK(cudaGetDeviceProperties(&prop, 0));
    printf("test_swa_attention on %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    if (prop.major != 9) { printf("  [SKIP] the window kernels are Hopper only\n"); return 0; }

    for (int W : {0, 128, 256, 1024, 2048, 3968}) t1_count({2, 4, 2, 4096}, W);
    t1_count({3, 12, 4, 1024}, 512);
    t1_count({1, 8, 8, 2048}, 1024);

    const std::vector<int> sp{0, 1, 63, 64, 127, 128, 129, 1000, 1023, 1024, 1025, 2047, 2048, 3071, 3072, 4095};
    for (int W : {128, 1024, 2048}) t2_spike_fwd(4096, W, sp);
    t2_spike_fwd(1024, 256, {0, 127, 128, 255, 256, 511, 767, 768, 1023});

    for (int W : {128, 256}) t3_spike_bwd(4096, W, sp);
    t3_spike_bwd(1024, 128, {0, 127, 128, 896, 897, 1023});

    t4_random({2, 4, 2, 4096}, 1024, 1.f, 1);                  // the 135M shape, per rank row count reduced
    t4_random({2, 4, 2, 4096}, 1024, 3.f, 2);                  // peaked softmax
    t4_random({2, 4, 2, 4096}, 128, 1.f, 3);
    t4_random({1, 12, 4, 2048}, 512, 1.f, 4);
    t4_random({3, 8, 8, 1024}, 256, 2.f, 5);
    t4_random({2, 8, 1, 2048}, 1024, 1.f, 6);

    t5_dispatch({2, 4, 2, 4096}, 1024);

    t6_bench({16, 4, 2, 4096}, 1024);
    t6_bench({16, 4, 2, 4096}, 512);

    printf("%s (%d failed)\n", g_fail ? "FAILED" : "ALL PASSED", g_fail);
    return g_fail ? 1 : 0;
}
