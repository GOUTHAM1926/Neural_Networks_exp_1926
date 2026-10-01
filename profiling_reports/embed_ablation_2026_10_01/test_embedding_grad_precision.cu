// =============================================================================
// test_embedding_grad_precision.cu -- how much does a bf16 embedding table cost in gradient precision?
//
// The embedding table wte [50,304 x 2,304] gets its gradient in 3 places where an fp32 table and a bf16 table differ
// (embedding_optimizer_zero_1_parallelism.md section 7):
//   D1  the lookup backward: the K vectors of a token that appears K times are summed into one row
//   D2  the engine adds the lookup part and the tied lm_head part (Engine.cpp:341-344)
//   D3  the way to the optimizer: the micro-steps are summed into the fp32 gradient buffer, then the send buffer
//       rounds the sum to bf16 (at W = 1 too on dist_side_merge: F4 is fixed only on dist_side_merge_BLUCCL)
// This test measures each one with the REAL library code of dist_side_merge (built for this GPU), against an exact
// fp64 reference computed on the CPU from the same input numbers.
//
// PART D1 -- the lookup backward alone, direct kernel calls (cuda::embedding_backward_cuda, EmbeddingKernels.cu):
//   arms   fp32_kernel     the fp32 path: the cooperative kernel (or fp32 atomicAdd if BLUTRAIN_EMB_BWD_COOPERATIVE=0)
//          bf16_atomic     the bf16 path: atomicAdd into a bf16 table, run 5 times (its order is not fixed)
//          sorted_fp32_once  CPU EMULATION (not the kernel) of the BLUCCL EmbeddingBwdFused arithmetic: positions in
//                          ascending order, rows of <= 32 summed whole, longer rows in chunks of 32 then folded, all in
//                          fp32, one round-to-nearest to bf16 at the end
//          floor           the exact sum rounded to bf16 once: the error no bf16 table can avoid
//   token ids  uniform, 4 real flux micro-batches, Zipf 1.0 / 1.3, SFT-style padded and packed (synthetic template +
//              flux text + EOT), one document repeated, one token everywhere, and a K sweep (one token each with
//              K = 1, 2, 4 ... 16,384); flux also at N = 8,192 / 16,384 (B = 2 / 4)
//   gradient vectors (every value rounded to bf16 first, so all arms add exactly the same numbers):
//              iid (random signs), aligned rho = 0.5 / 0.9 / 0.99 / 1.0 (the K vectors of a token share a direction),
//              lognormal magnitudes (few big, many small), big+small (first occurrence 1, the rest 0.003, same
//              direction), and scale 1e-8 / 1e+2 (scale invariance); pad positions of the SFT sets get exact zeros
//   metrics    per row: relative L2 error vs fp64; by K bucket: median, p95, max, and the median ratio to the floor;
//              the 5 most frequent tokens; run-to-run bit differences of bf16_atomic
//
// PART D2 -- the tied lm_head add, through REAL autograd in the trainer's op order (bluscript_zero_ckpt.cpp:748, :786-788):
//   x = embedding(wte, ids) -> to_dtype(bf16) -> sum(x * G1)   and   logits = matmul(H, contiguous(transpose(
//   to_dtype(wte, bf16)))) -> sum(logits * G2); backward; read wte's fp32 gradient buffer.
//   Arms: wte fp32 (bf16 residual, as BLU_ALL_BF16=1 BLU_BF16_EMBED=0) and wte bf16 (BLU_ALL_BF16=1). Each runs 3 times:
//   lookup part only (G2 = 0), lm_head part only (G1 = 0), both. G2 is a realistic dlogits (softmax - one-hot of the
//   next flux token, / N), H a unit-RMS hidden state, both bf16. The lookup part's size is swept against the lm_head
//   part's: s = 0.01 / 0.1 / 1 / 10 / 100. Errors on 256 sampled rows (the 64 most frequent tokens, 128 random present,
//   64 absent), against fp64 references of both parts.
//
// PART D3 -- micro-step accumulation and the send rounding, real kernels: M = 4 and 32 micro-steps of N = 8,192
//   flux tokens. fp32 table: fp32 lookup + fp32 add. bf16 table: bf16 lookup + F2's cuda_accumulate_lowp_into_fp32.
//   bf16 + BLU_GRAD_VIEW (micro-steps added in bf16, library bf16 +=). Then the send rounding (library as_type(bf16))
//   for the bf16 table, and for the fp32 table with BLU_BF16_REDUCE=1.
//
// SELF-CHECKS (if one fails, the numbers are not reported as results):
//   SC1 plumbing: values +-2^-10, K <= 4 -> every arm equals the fp64 sum EXACTLY, absent rows exactly 0
//   SC2 every K = 1 row of bf16_atomic equals its single input exactly, in every case
//   SC3 every absent row is exactly 0 in every arm, in every case
//   SC4 fp32_kernel relative error < 1e-4 in every case (fp32 sanity, not a precision claim)
//   SC5 fp32 cooperative kernel bit-identical across 2 runs (skipped when BLUTRAIN_EMB_BWD_COOPERATIVE=0)
//   SC6 D2: the combined gradient equals round(lookup part + lm_head part) element for element (one engine add)
//
// PREDICTIONS, written before the first run:
//   P1 fp32_kernel: relative error about 1e-7 to 1e-5 at every K.
//   P2 floor: about 1e-3 (bf16 keeps 8 bits) at every K, independent of K.
//   P3 sorted_fp32_once: within ~1.1x of the floor at every K (one rounding).
//   P4 bf16_atomic: grows with K. iid vectors: roughly 2^-9 * sqrt(K), so ~1e-2 at K ~ 100 and several % at K ~ 1,000.
//      Aligned vectors much worse: once the total reaches ~256 x one value, new values round away, so at rho = 1 and
//      K >> 256 a large part of the sum is lost (tens of %). The ratio to the floor grows with K.
//   P5 bf16_atomic differs from run to run once K > 1 (random order); fp32 cooperative never does.
//   P6 D2: the engine add costs at most one rounding of the sum; the smaller part loses the most when the parts differ
//      by >> 256x.
//   P7 D3: the send rounding adds ~1 floor; BLU_GRAD_VIEW adds up to M-1 more roundings (grows with M for aligned).
//
// Build (from Tensor-Implementations/), see run_embed_ablation.sh: the Makefile's run-cuda nvcc line, plus OpenMP for
// the CPU references. Inputs: FLUX_DIR (flux_train_*.bin, raw uint16 tokens). Output: one log + CSVs in OUT_DIR.
// =============================================================================
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <omp.h>

#include <cuda_runtime.h>

#include "TensorLib.h"
#include "ops/TensorOps.h"
#include "ops/TensorOps.cuh"
#include "ops/helpers/EmbeddingKernels.h"
#include "autograd/operations/EmbeddingOps.h"
#include "autograd/operations/ReshapeOps.h"
#include "autograd/operations/MatrixOps.h"
#include "autograd/operations/BinaryOps.h"
#include "autograd/operations/ReductionOps.h"

using namespace OwnTensor;

namespace {

constexpr int64_t V = 50304;   // vocabulary (bluscript_zero_ckpt.cpp:137)
constexpr int64_t C = 2304;    // d_model     (bluscript_zero_ckpt.cpp:129)
constexpr int NBUCKET = 7;
const char* BUCKET_NAME[NBUCKET] = {"K=1", "K=2-3", "K=4-15", "K=16-63", "K=64-255", "K=256-1023", "K>=1024"};
int bucket_of(int64_t K) { return K <= 1 ? 0 : K < 4 ? 1 : K < 16 ? 2 : K < 64 ? 3 : K < 256 ? 4 : K < 1024 ? 5 : 6; }

static_assert(sizeof(bfloat16_t) == 2, "bfloat16_t must be 2 bytes");

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    std::printf("CUDA ERROR %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); std::exit(2); } } while (0)

// ---------------------------------------------------------------- numbers
inline uint32_t f2u(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
inline float u2f(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
// round-to-nearest-even fp32 -> bf16 -> fp32 (the GPU's __float2bfloat16_rn on finite values)
inline float bf16_rn(float x) {
    uint32_t b = f2u(x);
    if ((b & 0x7f800000u) == 0x7f800000u) return x;
    b = (b + 0x7FFFu + ((b >> 16) & 1u)) & 0xFFFF0000u;
    return u2f(b);
}
inline uint16_t bf16_bits(float bf16_exact) { return static_cast<uint16_t>(f2u(bf16_exact) >> 16); }
inline float bits_to_f(uint16_t b) { return u2f(static_cast<uint32_t>(b) << 16); }

struct Rng {   // splitmix64: one stream per (seed, index), so results do not depend on the thread count
    uint64_t s;
    explicit Rng(uint64_t seed, uint64_t idx = 0) : s(seed * 0x9E3779B97F4A7C15ull ^ (idx + 0x632BE59BD9B4E019ull)) { next(); }
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    double uni() { return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0); }
    void normal2(double& a, double& b) {
        double u1 = uni(); if (u1 < 1e-300) u1 = 1e-300;
        const double r = std::sqrt(-2.0 * std::log(u1)), t = 6.283185307179586 * uni();
        a = r * std::cos(t); b = r * std::sin(t);
    }
    double normal() { double a, b; normal2(a, b); return a; }
};
void fill_normal(float* out, int64_t n, Rng& r) {
    int64_t i = 0;
    for (; i + 1 < n; i += 2) { double a, b; r.normal2(a, b); out[i] = (float)a; out[i + 1] = (float)b; }
    if (i < n) out[i] = (float)r.normal();
}

// ---------------------------------------------------------------- token ids
std::string g_flux_dir;
std::vector<uint16_t> read_flux(int file, int64_t offset, int64_t n) {
    char name[64];
    std::snprintf(name, sizeof name, "/flux_train_%06d.bin", file);
    const std::string p = g_flux_dir + name;
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) { std::printf("cannot open %s\n", p.c_str()); std::exit(2); }
    std::vector<uint16_t> v(n);
    std::fseek(f, offset * 2, SEEK_SET);
    if (std::fread(v.data(), 2, n, f) != static_cast<size_t>(n)) { std::printf("short read %s\n", p.c_str()); std::exit(2); }
    std::fclose(f);
    for (auto& t : v) if (t >= V) t = 50256;
    return v;
}

struct IdSet {
    std::string name;
    std::vector<uint16_t> ids;
    std::vector<uint8_t> zero;   // 1 = this position's gradient is exactly 0 (SFT padding after the last real token)
};

IdSet ids_uniform(int64_t n, uint64_t seed) {
    IdSet s{"uniform", std::vector<uint16_t>(n), std::vector<uint8_t>(n, 0)};
    Rng r(seed);
    for (auto& t : s.ids) t = static_cast<uint16_t>(r.next() % V);
    return s;
}
IdSet ids_flux(int file, int64_t off, int64_t n, const char* tag) {
    return IdSet{std::string("flux_") + tag, read_flux(file, off, n), std::vector<uint8_t>(n, 0)};
}
IdSet ids_zipf(int64_t n, double expo, uint64_t seed) {
    std::vector<double> cdf(V);
    double acc = 0;
    for (int64_t k = 0; k < V; ++k) { acc += 1.0 / std::pow(static_cast<double>(k + 1), expo); cdf[k] = acc; }
    std::vector<uint16_t> perm(V);
    for (int64_t k = 0; k < V; ++k) perm[k] = static_cast<uint16_t>(k);
    Rng r(seed);
    for (int64_t k = V - 1; k > 0; --k) std::swap(perm[k], perm[r.next() % (k + 1)]);
    char nm[32]; std::snprintf(nm, sizeof nm, "zipf%.1f", expo);
    IdSet s{nm, std::vector<uint16_t>(n), std::vector<uint8_t>(n, 0)};
    for (auto& t : s.ids) {
        const double u = r.uni() * acc;
        t = perm[std::lower_bound(cdf.begin(), cdf.end(), u) - cdf.begin()];
    }
    return s;
}
// SFT-style rows of T = 4,096: [template][prompt text][template][answer text][EOT], from flux text. The template ids
// are a fixed stand-in (the real template is not chosen yet); only the repetition pattern matters here.
IdSet ids_sft(int64_t n, bool packed, uint64_t seed) {
    const std::vector<uint16_t> pre = {27, 91, 320, 62, 9688, 91, 29, 7220, 198};
    const std::vector<uint16_t> mid = {27, 91, 320, 62, 437, 91, 29, 198, 27, 91, 320, 62, 9688, 91, 29, 562, 10167, 198};
    const std::vector<uint16_t> text = read_flux(2, 0, 4000000);
    IdSet s{packed ? "sft_packed" : "sft_padded", {}, {}};
    Rng r(seed);
    int64_t tp = static_cast<int64_t>(r.next() % 1000000);
    const int64_t T = 4096;
    while (static_cast<int64_t>(s.ids.size()) < n) {
        std::vector<uint16_t> row;
        while (true) {
            std::vector<uint16_t> smp(pre);
            const int64_t lp = 30 + static_cast<int64_t>(r.next() % 200), la = 50 + static_cast<int64_t>(r.next() % 600);
            for (int64_t i = 0; i < lp; ++i) smp.push_back(text[(tp++) % text.size()]);
            smp.insert(smp.end(), mid.begin(), mid.end());
            for (int64_t i = 0; i < la; ++i) smp.push_back(text[(tp++) % text.size()]);
            smp.push_back(50256);
            if (static_cast<int64_t>(row.size() + smp.size()) > T) break;
            row.insert(row.end(), smp.begin(), smp.end());
            if (!packed) break;   // padded: one sample per row
        }
        const int64_t real = static_cast<int64_t>(row.size());
        while (static_cast<int64_t>(row.size()) < T) row.push_back(50256);   // pad with EOT
        for (int64_t i = 0; i < T && static_cast<int64_t>(s.ids.size()) < n; ++i) {
            s.ids.push_back(row[i]);
            s.zero.push_back(i >= real ? 1 : 0);   // after the last real token: no loss reaches it, gradient 0
        }
    }
    return s;
}
IdSet ids_repeat_doc(int64_t n) {
    const std::vector<uint16_t> doc = read_flux(3, 12345, 512);
    IdSet s{"repeat_doc", std::vector<uint16_t>(n), std::vector<uint8_t>(n, 0)};
    for (int64_t i = 0; i < n; ++i) s.ids[i] = doc[i % 512];
    return s;
}
IdSet ids_one_token(int64_t n) { return IdSet{"one_token", std::vector<uint16_t>(n, 262), std::vector<uint8_t>(n, 0)}; }
IdSet ids_ksweep(int64_t n, uint64_t seed) {   // tokens 1000+i with K = 2^i, i = 0..14, + fillers, shuffled
    IdSet s{"k_sweep", {}, std::vector<uint8_t>(n, 0)};
    for (int i = 0; i < 15; ++i) for (int64_t k = 0; k < (int64_t(1) << i); ++k) s.ids.push_back(static_cast<uint16_t>(1000 + i));
    while (static_cast<int64_t>(s.ids.size()) < n) s.ids.push_back(static_cast<uint16_t>(2000 + s.ids.size() % 7));
    s.ids.resize(n);
    Rng r(seed);
    for (int64_t k = n - 1; k > 0; --k) std::swap(s.ids[k], s.ids[r.next() % (k + 1)]);
    return s;
}

// ---------------------------------------------------------------- compact token index
struct Compact {
    std::vector<int32_t> row_of;   // [V] -> compact row or -1
    std::vector<int32_t> tok;      // compact row -> token id
    std::vector<int64_t> K;        // occurrences (all positions; the kernel sees them all)
    std::vector<int64_t> Keff;     // occurrences with a non-zero gradient (SFT pads add exact zeros, no rounding)
    std::vector<int64_t> start;    // CSR into pos (positions in ascending order)
    std::vector<int64_t> pos;
};
Compact make_compact(const std::vector<uint16_t>& ids, const std::vector<uint8_t>* zero = nullptr) {
    Compact c;
    c.row_of.assign(V, -1);
    for (uint16_t t : ids) if (c.row_of[t] < 0) { c.row_of[t] = static_cast<int32_t>(c.tok.size()); c.tok.push_back(t); }
    const int64_t U = static_cast<int64_t>(c.tok.size());
    c.K.assign(U, 0);
    for (uint16_t t : ids) c.K[c.row_of[t]]++;
    c.Keff.assign(U, 0);
    for (size_t p = 0; p < ids.size(); ++p) if (!zero || !(*zero)[p]) c.Keff[c.row_of[ids[p]]]++;
    c.start.assign(U + 1, 0);
    for (int64_t r = 0; r < U; ++r) c.start[r + 1] = c.start[r] + c.K[r];
    c.pos.resize(ids.size());
    std::vector<int64_t> fill(c.start.begin(), c.start.end() - 1);
    for (int64_t p = 0; p < static_cast<int64_t>(ids.size()); ++p) c.pos[fill[c.row_of[ids[p]]]++] = p;
    return c;
}

// ---------------------------------------------------------------- gradient vectors (bf16-exact values)
struct Grad { std::string name; double sigma = 1e-3; double rho = 0; double logsig = 0; int kind = 0; };
// kind: 0 iid, 1 aligned(rho), 2 lognormal(logsig), 3 big+small, 4 exact (+-2^-10)
void make_grad(const IdSet& s, const Compact& cp, const Grad& g, uint64_t seed, std::vector<float>& G, uint64_t dir_seed = 0) {
    if (dir_seed == 0) dir_seed = seed;
    const int64_t N = static_cast<int64_t>(s.ids.size());
    G.assign(N * C, 0.0f);
    const int64_t U = static_cast<int64_t>(cp.tok.size());
    std::vector<float> dir;
    if (g.kind == 1 || g.kind == 3) {   // one direction per token, shared by all its occurrences
        dir.resize(U * C);
        #pragma omp parallel for schedule(static)
        for (int64_t r = 0; r < U; ++r) { Rng rr(dir_seed ^ 0xD1Bull, cp.tok[r]); fill_normal(&dir[r * C], C, rr); }
    }
    std::vector<uint8_t> first(N, 0);
    if (g.kind == 3) for (int64_t r = 0; r < U; ++r) first[cp.pos[cp.start[r]]] = 1;
    const double a = std::sqrt(std::max(0.0, 1.0 - g.rho * g.rho));
    #pragma omp parallel for schedule(static)
    for (int64_t p = 0; p < N; ++p) {
        float* out = &G[p * C];
        if (s.zero[p]) continue;
        Rng rr(seed, static_cast<uint64_t>(p));
        if (g.kind == 4) {
            for (int64_t c = 0; c < C; ++c) out[c] = (rr.next() & 1) ? 0.0009765625f : -0.0009765625f;
            continue;
        }
        fill_normal(out, C, rr);
        const float* d = (g.kind == 1 || g.kind == 3) ? &dir[static_cast<int64_t>(cp.row_of[s.ids[p]]) * C] : nullptr;
        double scale = g.sigma;
        if (g.kind == 2) scale *= std::exp(g.logsig * rr.normal());
        for (int64_t c = 0; c < C; ++c) {
            double v = out[c];
            if (g.kind == 1) v = g.rho * d[c] + a * v;
            if (g.kind == 3) v = (first[p] ? 1.0 : 0.003) * d[c];
            out[c] = bf16_rn(static_cast<float>(scale * v));
        }
    }
}

// ---------------------------------------------------------------- references
void fp64_truth(const Compact& cp, const std::vector<float>& G, std::vector<double>& T) {
    const int64_t U = static_cast<int64_t>(cp.tok.size());
    T.assign(U * C, 0.0);
    #pragma omp parallel for schedule(dynamic, 16)
    for (int64_t r = 0; r < U; ++r) {
        double* t = &T[r * C];
        for (int64_t i = cp.start[r]; i < cp.start[r + 1]; ++i) {
            const float* g = &G[cp.pos[i] * C];
            for (int64_t c = 0; c < C; ++c) t[c] += g[c];
        }
    }
}
// CPU emulation of EmbeddingBwdFused's order (embedding_fused.cu: bounded_direct / chunks / fold, EMB_T = 32)
void sorted_fp32_once(const Compact& cp, const std::vector<float>& G, std::vector<float>& out) {
    const int64_t U = static_cast<int64_t>(cp.tok.size());
    out.assign(U * C, 0.0f);
    #pragma omp parallel
    {
        std::vector<float> acc(C), part(C);
        #pragma omp for schedule(dynamic, 16)
        for (int64_t r = 0; r < U; ++r) {
            const int64_t b = cp.start[r], K = cp.K[r];
            std::fill(acc.begin(), acc.end(), 0.0f);
            if (K <= 32) {
                for (int64_t i = 0; i < K; ++i) { const float* g = &G[cp.pos[b + i] * C]; for (int64_t c = 0; c < C; ++c) acc[c] += g[c]; }
            } else {
                for (int64_t j = 0; j * 32 < K; ++j) {
                    std::fill(part.begin(), part.end(), 0.0f);
                    const int64_t e = std::min<int64_t>(K, (j + 1) * 32);
                    for (int64_t i = j * 32; i < e; ++i) { const float* g = &G[cp.pos[b + i] * C]; for (int64_t c = 0; c < C; ++c) part[c] += g[c]; }
                    for (int64_t c = 0; c < C; ++c) acc[c] += part[c];
                }
            }
            for (int64_t c = 0; c < C; ++c) out[r * C + c] = bf16_rn(acc[c]);
        }
    }
}

// ---------------------------------------------------------------- metrics
struct Metrics {
    double global = 0;                       // sqrt(sum |x-t|^2 / sum |t|^2) over present rows
    std::vector<double> rel;                 // per compact row (NaN for an all-zero true row)
    int64_t zero_true_rows_nonzero = 0;      // all-zero true rows that are not exactly 0
};
Metrics measure(const Compact& cp, const std::vector<double>& T, const float* X /* compact rows */) {
    const int64_t U = static_cast<int64_t>(cp.tok.size());
    Metrics m;
    m.rel.assign(U, NAN);
    double se = 0, st = 0;
    int64_t bad0 = 0;
    #pragma omp parallel for reduction(+ : se, st, bad0) schedule(static)
    for (int64_t r = 0; r < U; ++r) {
        double e = 0, t2 = 0;
        for (int64_t c = 0; c < C; ++c) { const double d = static_cast<double>(X[r * C + c]) - T[r * C + c]; e += d * d; t2 += T[r * C + c] * T[r * C + c]; }
        se += e; st += t2;
        if (t2 > 0) m.rel[r] = std::sqrt(e / t2);
        else if (e > 0) bad0++;
    }
    m.global = st > 0 ? std::sqrt(se / st) : 0;
    m.zero_true_rows_nonzero = bad0;
    return m;
}
struct BucketStat { int64_t n = 0; double med = NAN, p95 = NAN, mx = NAN; };
void buckets(const Compact& cp, const Metrics& m, BucketStat out[NBUCKET]) {
    std::vector<double> v[NBUCKET];
    for (size_t r = 0; r < cp.tok.size(); ++r) if (!std::isnan(m.rel[r])) v[bucket_of(cp.Keff[r])].push_back(m.rel[r]);
    for (int b = 0; b < NBUCKET; ++b) {
        out[b] = BucketStat{};
        if (v[b].empty()) continue;
        std::sort(v[b].begin(), v[b].end());
        out[b].n = static_cast<int64_t>(v[b].size());
        out[b].med = v[b][v[b].size() / 2];
        out[b].p95 = v[b][std::min(v[b].size() - 1, (v[b].size() * 95) / 100)];
        out[b].mx = v[b].back();
    }
}

// ---------------------------------------------------------------- bookkeeping
FILE* g_csv1 = nullptr; FILE* g_csv2 = nullptr; FILE* g_csv3 = nullptr;
int g_sc_fail = 0;
std::vector<std::string> g_summary;
void sc(bool ok, const std::string& what) {
    std::printf("  SELF-CHECK %s: %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) { g_sc_fail++; g_summary.push_back("SELF-CHECK FAIL: " + what); }
}
std::string fmt(double x) { if (std::isnan(x)) return "-"; char b[32]; std::snprintf(b, sizeof b, "%.3g", x); return b; }
double nanmax(double a, double b) { return std::isnan(a) ? b : std::isnan(b) ? a : std::max(a, b); }

// ---------------------------------------------------------------- library tensors
DeviceIndex dev0() { return DeviceIndex{Device::CUDA, 0}; }
Tensor gpu_from(const std::vector<float>& h, std::vector<int64_t> dims, Dtype dt) {
    Shape sh; sh.dims = dims;
    Tensor t = Tensor::zeros(sh, TensorOptions().with_dtype(Dtype::Float32).with_device(dev0()));
    t.set_data(h.data(), h.size());
    return dt == Dtype::Float32 ? t : t.as_type(dt);   // values are bf16-exact -> exact
}
std::vector<float> host_f32(const Tensor& t) {
    Tensor f = t.dtype() == Dtype::Float32 ? t : t.as_type(Dtype::Float32);
    Tensor c = f.to_cpu();
    const float* p = c.data<float>();
    return std::vector<float>(p, p + c.numel());
}

// ---------------------------------------------------------------- PART D1
struct D1Buffers {
    uint16_t* ids = nullptr; float* g32 = nullptr; uint16_t* g16 = nullptr; float* t32 = nullptr; uint16_t* t16 = nullptr;
    std::vector<float> h32; std::vector<uint16_t> h16;
    void alloc(int64_t nmax) {
        if (ids) return;
        CK(cudaMalloc(&ids, nmax * 2)); CK(cudaMalloc(&g32, nmax * C * 4)); CK(cudaMalloc(&g16, nmax * C * 2));
        CK(cudaMalloc(&t32, V * C * 4)); CK(cudaMalloc(&t16, V * C * 2));
        h32.resize(V * C); h16.resize(V * C);
    }
};
D1Buffers g_b;
bool g_coop = true;

// run one arm on the GPU, return compact rows (float) + count of non-zero elements in absent rows
int64_t run_kernel(bool bf16, int64_t N, const Compact& cp, std::vector<float>& rows) {
    if (bf16) {
        CK(cudaMemset(g_b.t16, 0, V * C * 2));
        cuda::embedding_backward_cuda(g_b.ids, reinterpret_cast<const bfloat16_t*>(g_b.g16),
                                      reinterpret_cast<bfloat16_t*>(g_b.t16), N, C, V, -1, C, 1);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(g_b.h16.data(), g_b.t16, V * C * 2, cudaMemcpyDeviceToHost));
    } else {
        CK(cudaMemset(g_b.t32, 0, V * C * 4));
        cuda::embedding_backward_cuda(g_b.ids, g_b.g32, g_b.t32, N, C, V, -1, C, 1);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(g_b.h32.data(), g_b.t32, V * C * 4, cudaMemcpyDeviceToHost));
    }
    const int64_t U = static_cast<int64_t>(cp.tok.size());
    rows.resize(U * C);
    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < U; ++r) {
        const int64_t base = static_cast<int64_t>(cp.tok[r]) * C;
        for (int64_t c = 0; c < C; ++c) rows[r * C + c] = bf16 ? bits_to_f(g_b.h16[base + c]) : g_b.h32[base + c];
    }
    int64_t nz = 0;
    #pragma omp parallel for reduction(+ : nz) schedule(static)
    for (int64_t t = 0; t < V; ++t) {
        if (cp.row_of[t] >= 0) continue;
        for (int64_t c = 0; c < C; ++c) nz += bf16 ? (g_b.h16[t * C + c] != 0) : (g_b.h32[t * C + c] != 0.0f);
    }
    return nz;
}

void d1_case(const IdSet& s, const Grad& g, uint64_t seed) {
    const int64_t N = static_cast<int64_t>(s.ids.size());
    const Compact cp = make_compact(s.ids, &s.zero);
    const int64_t U = static_cast<int64_t>(cp.tok.size());
    int64_t maxK = 0; for (auto k : cp.Keff) maxK = std::max(maxK, k);
    std::vector<float> G; make_grad(s, cp, g, seed, G);
    std::vector<double> T; fp64_truth(cp, G, T);
    // uploads: the same bf16-exact numbers as fp32 and as bf16
    CK(cudaMemcpy(g_b.ids, s.ids.data(), N * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(g_b.g32, G.data(), N * C * 4, cudaMemcpyHostToDevice));
    {
        std::vector<uint16_t> g16(N * C);
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < N * C; ++i) g16[i] = bf16_bits(G[i]);
        CK(cudaMemcpy(g_b.g16, g16.data(), N * C * 2, cudaMemcpyHostToDevice));
    }
    std::printf("\nCASE|D1|%s|%s|N=%lld|unique=%lld|maxK=%lld\n", s.name.c_str(), g.name.c_str(),
                (long long)N, (long long)U, (long long)maxK);
    // fp32 kernel, 2 runs
    std::vector<float> f32a, f32b, b16[5], emu, flo;
    const int64_t nz32 = run_kernel(false, N, cp, f32a) + run_kernel(false, N, cp, f32b);
    int64_t d32 = 0;
    for (int64_t i = 0; i < U * C; ++i) d32 += (f2u(f32a[i]) != f2u(f32b[i]));
    int64_t nz16 = 0;
    for (int k = 0; k < 5; ++k) nz16 += run_kernel(true, N, cp, b16[k]);
    int64_t d16 = 0;
    for (int k = 1; k < 5; ++k) { int64_t d = 0; for (int64_t i = 0; i < U * C; ++i) d += (f2u(b16[0][i]) != f2u(b16[k][i])); d16 = std::max(d16, d); }
    sorted_fp32_once(cp, G, emu);
    flo.resize(U * C);
    for (int64_t i = 0; i < U * C; ++i) flo[i] = bf16_rn(static_cast<float>(T[i]));

    const char* a32 = g_coop ? "fp32_cooperative" : "fp32_atomic";
    struct Arm { const char* name; const float* x; } arms[] = {
        {a32, f32a.data()}, {"bf16_atomic", b16[0].data()}, {"sorted_fp32_once(emu)", emu.data()}, {"floor(round once)", flo.data()}};
    Metrics m[4]; BucketStat bs[4][NBUCKET];
    for (int a = 0; a < 4; ++a) { m[a] = measure(cp, T, arms[a].x); buckets(cp, m[a], bs[a]); }
    double g16min = m[1].global, g16max = m[1].global;
    for (int k = 1; k < 5; ++k) { const Metrics mk = measure(cp, T, b16[k].data()); g16min = std::min(g16min, mk.global); g16max = std::max(g16max, mk.global); }
    // print
    for (int a = 0; a < 4; ++a) {
        std::printf("  ARM %-22s global rel err %-9s", arms[a].name, fmt(m[a].global).c_str());
        if (a == 1) std::printf(" (5 runs: %s .. %s)", fmt(g16min).c_str(), fmt(g16max).c_str());
        std::printf("\n");
    }
    std::printf("  %-12s", "bucket");
    for (int a = 0; a < 4; ++a) std::printf(" | %-26s", arms[a].name);
    std::printf(" | bf16/floor\n");
    for (int b = 0; b < NBUCKET; ++b) {
        if (bs[0][b].n == 0) continue;
        std::printf("  %-12s", BUCKET_NAME[b]);
        for (int a = 0; a < 4; ++a)
            std::printf(" | med %-7s max %-8s", fmt(bs[a][b].med).c_str(), fmt(bs[a][b].mx).c_str());
        std::printf(" | %-6s  (rows %lld)\n", fmt(bs[1][b].med / bs[3][b].med).c_str(), (long long)bs[0][b].n);
        for (int a = 0; a < 4; ++a)
            std::fprintf(g_csv1, "%s,%s,%s,%s,%lld,%.6g,%.6g,%.6g\n", s.name.c_str(), g.name.c_str(), arms[a].name,
                         BUCKET_NAME[b], (long long)bs[a][b].n, bs[a][b].med, bs[a][b].p95, bs[a][b].mx);
    }
    // the 5 most frequent tokens
    std::vector<int64_t> order(U); for (int64_t r = 0; r < U; ++r) order[r] = r;
    std::partial_sort(order.begin(), order.begin() + std::min<int64_t>(5, U), order.end(), [&](int64_t x, int64_t y) { return cp.Keff[x] > cp.Keff[y]; });
    std::printf("  top tokens (id K: fp32 / bf16 / emu / floor):");
    for (int64_t i = 0; i < std::min<int64_t>(5, U); ++i) {
        const int64_t r = order[i];
        std::printf("  [%d K=%lld: %s / %s / %s / %s]", cp.tok[r], (long long)cp.Keff[r], fmt(m[0].rel[r]).c_str(),
                    fmt(m[1].rel[r]).c_str(), fmt(m[2].rel[r]).c_str(), fmt(m[3].rel[r]).c_str());
    }
    std::printf("\n  run-to-run: bf16_atomic elements differing (max over runs 2-5 vs 1): %lld of %lld; %s 2 runs: %lld\n",
                (long long)d16, (long long)(U * C), a32, (long long)d32);
    // self-checks
    int64_t k1bad = 0;
    for (int64_t r = 0; r < U; ++r) {
        if (cp.K[r] != 1) continue;
        const float* in = &G[cp.pos[cp.start[r]] * C];
        for (int64_t c = 0; c < C; ++c) k1bad += (f2u(b16[0][r * C + c]) != f2u(in[c])) && !(in[c] == 0.0f && b16[0][r * C + c] == 0.0f);
    }
    sc(k1bad == 0, "SC2 " + s.name + "/" + g.name + ": every K=1 row of bf16_atomic equals its input (" + std::to_string(k1bad) + " mismatches)");
    const int64_t z0 = m[0].zero_true_rows_nonzero + m[1].zero_true_rows_nonzero;
    sc(nz32 == 0 && nz16 == 0 && z0 == 0, "SC3 " + s.name + "/" + g.name + ": absent rows exactly 0 (non-zero elements: fp32 " +
       std::to_string(nz32) + ", bf16 " + std::to_string(nz16) + "); present rows with an all-zero sum exactly 0 (bad rows " + std::to_string(z0) + ")");
    sc(m[0].global < 1e-4, "SC4 " + s.name + "/" + g.name + ": fp32 kernel rel err " + fmt(m[0].global) + " < 1e-4");
    if (g_coop) sc(d32 == 0, "SC5 " + s.name + "/" + g.name + ": fp32 cooperative bit-identical across 2 runs (" + std::to_string(d32) + " diffs)");
    char line[512];
    std::snprintf(line, sizeof line, "D1 %-12s %-14s maxK %-6lld | global rel err: %s %-9s bf16_atomic %-9s emu %-9s floor %-9s | bf16 K>=256 med %s",
                  s.name.c_str(), g.name.c_str(), (long long)maxK, a32, fmt(m[0].global).c_str(), fmt(m[1].global).c_str(),
                  fmt(m[2].global).c_str(), fmt(m[3].global).c_str(),
                  fmt(nanmax(bs[1][5].med, bs[1][6].med)).c_str());
    g_summary.push_back(line);
}

void d1_plumbing() {   // SC1: exact sums
    IdSet s = ids_uniform(32768, 7);
    for (int64_t p = 0; p < 32768; ++p) s.ids[p] = static_cast<uint16_t>(p / 4);   // every token K = 4
    const Compact cp = make_compact(s.ids);
    Grad g{"exact_pm2^-10", 0, 0, 0, 4};
    std::vector<float> G; make_grad(s, cp, g, 99, G);
    std::vector<double> T; fp64_truth(cp, G, T);
    const int64_t N = 32768, U = static_cast<int64_t>(cp.tok.size());
    CK(cudaMemcpy(g_b.ids, s.ids.data(), N * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(g_b.g32, G.data(), N * C * 4, cudaMemcpyHostToDevice));
    std::vector<uint16_t> g16(N * C); for (int64_t i = 0; i < N * C; ++i) g16[i] = bf16_bits(G[i]);
    CK(cudaMemcpy(g_b.g16, g16.data(), N * C * 2, cudaMemcpyHostToDevice));
    std::vector<float> a, b, e;
    const int64_t nz = run_kernel(false, N, cp, a) + run_kernel(true, N, cp, b);
    sorted_fp32_once(cp, G, e);
    int64_t bad[3] = {0, 0, 0};
    for (int64_t i = 0; i < U * C; ++i) {
        bad[0] += (static_cast<double>(a[i]) != T[i]); bad[1] += (static_cast<double>(b[i]) != T[i]); bad[2] += (static_cast<double>(e[i]) != T[i]);
    }
    std::printf("\nCASE|D1|plumbing|exact +-2^-10, every token K=4, N=32768\n");
    {   // SC7: 4M floats through the library's fp32 -> bf16 conversion: random normal numbers of every exponent,
        // exact ties (low 16 bits = 0x8000), and subnormals (reported only: no gradient in this test is subnormal)
        const int64_t n = 4 << 20;
        std::vector<float> x(n);
        Rng r(321);
        for (int64_t i = 0; i < n; ++i) {
            uint32_t u = static_cast<uint32_t>(r.next());
            if (((u >> 23) & 0xFFu) == 0xFFu) u &= ~(1u << 23);              // no inf / NaN
            if (((u >> 23) & 0xFFu) == 0x00u) u |= (1u << 23);               // no subnormal here
            if (i % 4 == 1) u = (u & 0xFFFF0000u) | 0x8000u;                  // exact tie
            if (i % 4 == 2) u = u & 0x807FFFFFu;                              // subnormal (or 0)
            x[i] = u2f(u);
        }
        Tensor h = gpu_from(x, {n}, Dtype::Float32);
        const std::vector<float> back = host_f32(h.as_type(Dtype::Bfloat16));
        int64_t bad = 0, bad_sub = 0;
        for (int64_t i = 0; i < n; ++i) {
            const bool d = (f2u(back[i]) != f2u(bf16_rn(x[i]))) && !(back[i] == 0.0f && bf16_rn(x[i]) == 0.0f);
            (i % 4 == 2 ? bad_sub : bad) += d;
        }
        sc(bad == 0, "SC7 the CPU bf16 rounding used for floor / emulation equals the library's as_type(bf16) on 3M normal floats incl. 1M exact ties (" + std::to_string(bad) + " differ)");
        std::printf("  info: subnormal inputs where the library differs from round-to-nearest: %lld of %lld\n", (long long)bad_sub, (long long)(n / 4));
    }
    sc(bad[0] == 0 && bad[1] == 0 && bad[2] == 0 && nz == 0,
       "SC1 plumbing: fp32 / bf16 / emu equal the fp64 sum exactly (mismatches " + std::to_string(bad[0]) + " / " +
       std::to_string(bad[1]) + " / " + std::to_string(bad[2]) + "), absent rows non-zero " + std::to_string(nz));
}

void part_d1() {
    std::printf("\n################ PART D1: the lookup backward (fp32 path: %s)\n", g_coop ? "cooperative kernel" : "fp32 atomicAdd, BLUTRAIN_EMB_BWD_COOPERATIVE=0");
    g_b.alloc(32768);
    d1_plumbing();
    const Grad iid{"iid", 1e-3, 0, 0, 0}, al5{"aligned0.5", 1e-3, 0.5, 0, 1}, al9{"aligned0.9", 1e-3, 0.9, 0, 1},
        al99{"aligned0.99", 1e-3, 0.99, 0, 1}, al1{"aligned1.0", 1e-3, 1.0, 0, 1}, lgn{"lognormal1.5", 1e-3, 0, 1.5, 2},
        bs{"big+small", 1e-3, 0, 0, 3}, sm{"iid_scale1e-8", 1e-8, 0, 0, 0}, lg{"iid_scale1e+2", 1e2, 0, 0, 0};
    const int64_t N = 32768;
    std::vector<IdSet> sets;
    sets.push_back(ids_uniform(N, 1));
    sets.push_back(ids_flux(1, 0, N, "a"));
    sets.push_back(ids_flux(1, 1000000, N, "b"));
    sets.push_back(ids_flux(1, 50000000, N, "c"));
    sets.push_back(ids_flux(50, 7000000, N, "d"));
    sets.push_back(ids_zipf(N, 1.0, 2));
    sets.push_back(ids_zipf(N, 1.3, 3));
    sets.push_back(ids_sft(N, false, 4));
    sets.push_back(ids_sft(N, true, 5));
    sets.push_back(ids_repeat_doc(N));
    sets.push_back(ids_one_token(N));
    sets.push_back(ids_ksweep(N, 6));
    uint64_t seed = 1000;
    for (const auto& s : sets) for (const Grad* g : {&iid, &al9, &lgn}) d1_case(s, *g, seed++);
    for (const Grad* g : {&al5, &al99, &al1, &bs}) { d1_case(sets[1], *g, seed++); d1_case(sets[11], *g, seed++); }
    d1_case(sets[1], sm, seed++); d1_case(sets[1], lg, seed++);
    for (int64_t n : {8192, 16384}) {
        char tag[16]; std::snprintf(tag, sizeof tag, "a_N%lld", (long long)n);
        const IdSet s = ids_flux(1, 0, n, tag);
        d1_case(s, iid, seed++); d1_case(s, al9, seed++);
    }
}

// ---------------------------------------------------------------- PART D2 (real autograd)

// One forward + backward in the trainer's op order. Returns wte's gradient [V x C] as float.
std::vector<float> d2_graph(bool bf16_table, const Tensor& ids_t, const Tensor& G1t, const Tensor& Ht, const Tensor& G2t) {
    Shape sh; sh.dims = {V, C};
    Tensor wte = Tensor::zeros(sh, TensorOptions().with_dtype(bf16_table ? Dtype::Bfloat16 : Dtype::Float32).with_device(dev0()));
    wte.set_requires_grad(true);
    Tensor x   = autograd::embedding(wte, ids_t, -1);                                  // op 1
    Tensor xr  = autograd::to_dtype(x, Dtype::Bfloat16);                               // op 2 (bf16 residual)
    Tensor l1  = autograd::sum(autograd::mul(xr, G1t));                                // injects G1 at the lookup output
    Tensor wbf = autograd::to_dtype(wte, Dtype::Bfloat16);                             // :786
    Tensor wt  = autograd::contiguous(autograd::transpose(wbf, 0, 1));                 // :787
    Tensor lg  = autograd::matmul(Ht, wt);                                             // :788
    Tensor l2  = autograd::sum(autograd::mul(lg, G2t));                                // injects dlogits = G2
    Tensor loss = autograd::add(l1, l2);
    loss.backward();
    CK(cudaDeviceSynchronize());
    return host_f32(wte.grad_view());
}

void part_d2() {
    std::printf("\n################ PART D2: lookup part + tied lm_head part, real autograd\n");
    const int64_t N = 4096;
    const std::vector<uint16_t> tok = read_flux(4, 3000000, N + 1);
    IdSet s{"flux_e", std::vector<uint16_t>(tok.begin(), tok.begin() + N), std::vector<uint8_t>(N, 0)};
    const Compact cp = make_compact(s.ids);
    const int64_t U = static_cast<int64_t>(cp.tok.size());
    // H: unit-RMS hidden states; G2: dlogits = (softmax(z) - onehot(next token)) / N
    std::vector<float> H(N * C), G2(N * V);
    #pragma omp parallel for schedule(static)
    for (int64_t p = 0; p < N; ++p) {
        Rng r(777, p);
        fill_normal(&H[p * C], C, r);
        for (int64_t c = 0; c < C; ++c) H[p * C + c] = bf16_rn(H[p * C + c]);
        float* z = &G2[p * V];
        fill_normal(z, V, r);
        double mx = -1e300; for (int64_t v = 0; v < V; ++v) { z[v] *= 2.0f; mx = std::max(mx, static_cast<double>(z[v])); }
        double sum = 0; for (int64_t v = 0; v < V; ++v) { const double e = std::exp(z[v] - mx); z[v] = static_cast<float>(e); sum += e; }
        for (int64_t v = 0; v < V; ++v) {
            double q = z[v] / sum - (v == tok[p + 1] ? 1.0 : 0.0);
            z[v] = bf16_rn(static_cast<float>(q / N));
        }
    }
    // sampled rows: 64 most frequent present, 128 random present, 64 absent
    std::vector<int64_t> order(U); for (int64_t r = 0; r < U; ++r) order[r] = r;
    std::sort(order.begin(), order.end(), [&](int64_t a, int64_t b) { return cp.K[a] > cp.K[b]; });
    std::vector<int32_t> rows;   // token ids
    for (int64_t i = 0; i < std::min<int64_t>(64, U); ++i) rows.push_back(cp.tok[order[i]]);
    Rng rr(4242);
    while (static_cast<int64_t>(rows.size()) < 192) { int32_t t = cp.tok[rr.next() % U]; if (std::find(rows.begin(), rows.end(), t) == rows.end()) rows.push_back(t); }
    while (static_cast<int64_t>(rows.size()) < 256) { int32_t t = static_cast<int32_t>(rr.next() % V); if (cp.row_of[t] < 0 && std::find(rows.begin(), rows.end(), t) == rows.end()) rows.push_back(t); }
    const int64_t S = static_cast<int64_t>(rows.size());
    // fp64 lm_head part for the sampled rows: B[v][c] = sum_p G2[p][v] * H[p][c]
    std::vector<double> Btrue(S * C, 0.0);
    #pragma omp parallel for schedule(dynamic, 1)
    for (int64_t i = 0; i < S; ++i) {
        double* b = &Btrue[i * C];
        for (int64_t p = 0; p < N; ++p) { const double g = G2[p * V + rows[i]]; if (g == 0) continue; const float* h = &H[p * C]; for (int64_t c = 0; c < C; ++c) b[c] += g * h[c]; }
    }
    double brms = 0; for (int64_t i = 0; i < S * C; ++i) brms += Btrue[i] * Btrue[i]; brms = std::sqrt(brms / (S * C));
    std::vector<int64_t> ks(cp.K); std::sort(ks.begin(), ks.end()); const double kmed = static_cast<double>(ks[ks.size() / 2]);
    std::printf("  N=%lld positions, %lld unique tokens, median K %.0f, max K %lld; lm_head part RMS (sampled rows) %s\n",
                (long long)N, (long long)U, kmed, (long long)ks.back(), fmt(brms).c_str());
    // tensors
    Shape sid; sid.dims = {N};
    Tensor ids_t = Tensor::zeros(sid, TensorOptions().with_dtype(Dtype::UInt16).with_device(dev0()));
    CK(cudaMemcpy(ids_t.data(), s.ids.data(), N * sizeof(uint16_t), cudaMemcpyHostToDevice));   // set_data<uint16_t> is not exported
    Tensor Ht = gpu_from(H, {N, C}, Dtype::Bfloat16), G2t = gpu_from(G2, {N, V}, Dtype::Bfloat16);
    Shape s1; s1.dims = {N, C};
    Shape s2; s2.dims = {N, V};
    const Tensor G1zero = Tensor::zeros(s1, TensorOptions().with_dtype(Dtype::Bfloat16).with_device(dev0()));
    const Tensor G2zero = Tensor::zeros(s2, TensorOptions().with_dtype(Dtype::Bfloat16).with_device(dev0()));
    std::fprintf(g_csv2, "grad,s_target,s_actual,table,group,quantity,rows,median,max\n");
    const char* GROUP[3] = {"top64(K high)", "random present", "absent"};
    const char* QTY[6] = {"combined", "lookup only", "lm_head only", "lookup kept", "lm_head kept", "floor(bf16)"};
    const Grad gens[2] = {Grad{"iid", 1.0, 0, 0, 0}, Grad{"aligned0.9", 1.0, 0.9, 0, 1}};
    uint64_t seed = 5000;
    for (const Grad& gen : gens) {
        std::vector<float> G1base; make_grad(s, cp, gen, seed++, G1base);
        std::vector<double> Abase(S * C, 0.0);   // fp64 lookup part of the sampled rows at scale 1
        #pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < S; ++i) {
            const int32_t r = cp.row_of[rows[i]];
            if (r < 0) continue;
            for (int64_t j = cp.start[r]; j < cp.start[r + 1]; ++j) {
                const float* g = &G1base[cp.pos[j] * C];
                for (int64_t c = 0; c < C; ++c) Abase[i * C + c] += g[c];
            }
        }
        double arms = 0; for (int64_t i = 0; i < 192 * C; ++i) arms += Abase[i] * Abase[i]; arms = std::sqrt(arms / (192.0 * C));
        for (double st : {0.01, 0.1, 1.0, 10.0, 100.0}) {
            // scale G1 by a power of 2 (keeps every value bf16-exact) so the lookup part's RMS is ~st x the lm_head part's
            const int e = static_cast<int>(std::lround(std::log2(st * brms / arms)));
            const double scale = std::ldexp(1.0, e), sa = arms * scale / brms;
            std::vector<float> G1(G1base.size());
            for (size_t i = 0; i < G1.size(); ++i) G1[i] = static_cast<float>(G1base[i] * scale);
            const Tensor G1t = gpu_from(G1, {N, C}, Dtype::Bfloat16);
            std::printf("\nCASE|D2|%s|s=%g|actual s %.3g (lookup part RMS / lm_head part RMS, sampled present rows)\n", gen.name.c_str(), st, sa);
            for (int tb = 0; tb < 2; ++tb) {
                const bool b16 = tb == 1;
                const char* tname = b16 ? "bf16 table" : "fp32 table";
                const std::vector<float> ga = d2_graph(b16, ids_t, G1t, Ht, G2zero);    // lookup part only
                const std::vector<float> gb = d2_graph(b16, ids_t, G1zero, Ht, G2t);    // lm_head part only
                const std::vector<float> gc = d2_graph(b16, ids_t, G1t, Ht, G2t);       // both (the real case)
                // SC6 on the whole table: combined == round(lookup + lm_head) wherever the arms are deterministic
                int64_t bad = 0, nd_diff = 0, nd_n = 0, absent_nz = 0;
                #pragma omp parallel for reduction(+ : bad, nd_diff, nd_n, absent_nz) schedule(static)
                for (int64_t t = 0; t < V; ++t) {
                    const int32_t r = cp.row_of[t];
                    const bool det = !b16 || r < 0 || cp.K[r] == 1;   // fp32 cooperative + cuBLAS are deterministic; bf16 atomics only at K=1
                    for (int64_t c = 0; c < C; ++c) {
                        const float a = ga[t * C + c], b = gb[t * C + c], got = gc[t * C + c];
                        const float want = b16 ? bf16_rn(a + b) : a + b;
                        const bool d = f2u(got) != f2u(want) && !(got == 0.0f && want == 0.0f);
                        if (det) bad += d; else { nd_diff += d; nd_n++; }
                        if (r < 0) absent_nz += (a != 0.0f);
                    }
                }
                sc(bad == 0 && absent_nz == 0, std::string("SC6 D2 ") + gen.name + " s=" + fmt(st) + " " + tname +
                   ": combined == round(lookup part + lm_head part) on every deterministic row (" + std::to_string(bad) +
                   " differ); lookup part exactly 0 on absent rows (" + std::to_string(absent_nz) + " non-zero)");
                if (b16) std::printf("  info: bf16 table, K>1 rows (atomic order differs between runs): %lld of %lld elements differ from round(a+b) of the separate runs\n",
                                     (long long)nd_diff, (long long)nd_n);
                // errors on the sampled rows
                std::vector<double> rel[6][3];
                for (int64_t i = 0; i < S; ++i) {
                    const int64_t t = rows[i];
                    double e[6] = {0, 0, 0, 0, 0, 0}, n[6] = {0, 0, 0, 0, 0, 0};
                    for (int64_t c = 0; c < C; ++c) {
                        const double A = Abase[i * C + c] * scale, B = Btrue[i * C + c], a = ga[t * C + c], b = gb[t * C + c], x = gc[t * C + c];
                        const double got[6] = {x, a, b, x - b, x - a, static_cast<double>(bf16_rn(static_cast<float>(A + B)))};
                        const double ref[6] = {A + B, A, B, A, B, A + B};
                        for (int q = 0; q < 6; ++q) { e[q] += (got[q] - ref[q]) * (got[q] - ref[q]); n[q] += ref[q] * ref[q]; }
                    }
                    for (int q = 0; q < 6; ++q) if (n[q] > 0) rel[q][i < 64 ? 0 : i < 192 ? 1 : 2].push_back(std::sqrt(e[q] / n[q]));
                }
                std::printf("  %-10s %-15s", tname, "group");
                for (int q = 0; q < 6; ++q) std::printf(" | %-24s", QTY[q]);
                std::printf("\n");
                double med[6][3];
                for (int gq = 0; gq < 3; ++gq) {
                    std::printf("  %-10s %-15s", tname, GROUP[gq]);
                    for (int q = 0; q < 6; ++q) {
                        auto& v = rel[q][gq];
                        std::sort(v.begin(), v.end());
                        med[q][gq] = v.empty() ? NAN : v[v.size() / 2];
                        const double mx = v.empty() ? NAN : v.back();
                        std::printf(" | med %-7s max %-8s", fmt(med[q][gq]).c_str(), fmt(mx).c_str());
                        std::fprintf(g_csv2, "%s,%g,%.4g,%s,%s,%s,%zu,%.6g,%.6g\n", gen.name.c_str(), st, sa, tname, GROUP[gq], QTY[q], v.size(), med[q][gq], mx);
                    }
                    std::printf("\n");
                }
                char line[512];
                std::snprintf(line, sizeof line, "D2 %-10s s=%-5s %-10s | combined med: top64 %-9s random %-9s absent %-9s | lookup kept med top64 %-9s | lm_head kept med top64 %-9s absent %-9s",
                              gen.name.c_str(), fmt(st).c_str(), tname, fmt(med[0][0]).c_str(), fmt(med[0][1]).c_str(), fmt(med[0][2]).c_str(),
                              fmt(med[3][0]).c_str(), fmt(med[4][0]).c_str(), fmt(med[4][2]).c_str());
                g_summary.push_back(line);
            }
        }
    }
}

// ---------------------------------------------------------------- PART D3 (micro-steps + send)
std::vector<float> rows_of(const Tensor& t, const Compact& cp) {
    const std::vector<float> h = host_f32(t);
    const int64_t U = static_cast<int64_t>(cp.tok.size());
    std::vector<float> out(U * C);
    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < U; ++r) std::memcpy(&out[r * C], &h[static_cast<int64_t>(cp.tok[r]) * C], C * sizeof(float));
    return out;
}

void part_d3() {
    std::printf("\n################ PART D3: micro-step accumulation + the send rounding (real kernels, lookup part only)\n");
    g_b.alloc(32768);
    const int64_t N = 8192;
    Shape svc; svc.dims = {V, C};
    auto mk = [&](Dtype d) { return Tensor::zeros(svc, TensorOptions().with_dtype(d).with_device(dev0())); };
    Tensor upd32 = mk(Dtype::Float32), upd16 = mk(Dtype::Bfloat16);
    std::fprintf(g_csv3, "M,grad,arm,bucket,rows,median,p95,max\n");
    const char* ARM[6] = {"fp32 table: buffer=sent", "fp32 + BLU_BF16_REDUCE: sent", "bf16 table: fp32 buffer", "bf16 table: sent bf16",
                          "bf16 running sum(assumed GRAD_VIEW)", "floor(round once)"};
    const Grad gens[2] = {Grad{"iid", 1e-3, 0, 0, 0}, Grad{"aligned0.9", 1e-3, 0.9, 0, 1}};
    uint64_t seed = 9000;
    for (int M : {4, 32}) for (const Grad& gen : gens) {
        Tensor acc32 = mk(Dtype::Float32), acc16 = mk(Dtype::Float32), accgv = mk(Dtype::Bfloat16);
        std::vector<std::vector<uint16_t>> mids(M);
        std::vector<uint16_t> all;
        for (int m = 0; m < M; ++m) { mids[m] = read_flux(5, static_cast<int64_t>(m) * N, N); all.insert(all.end(), mids[m].begin(), mids[m].end()); }
        const Compact ca = make_compact(all);
        const int64_t U = static_cast<int64_t>(ca.tok.size());
        std::vector<double> T(U * C, 0.0);
        const uint64_t dseed = seed++;   // a token keeps its direction across micro-steps (aligned)
        bool f2ok = true;
        for (int m = 0; m < M; ++m) {
            const IdSet sm{"m", mids[m], std::vector<uint8_t>(N, 0)};
            const Compact cm = make_compact(sm.ids);
            std::vector<float> G; make_grad(sm, cm, gen, seed++, G, dseed);
            #pragma omp parallel for schedule(static)
            for (int64_t c0 = 0; c0 < C; c0 += 64)   // split by columns: no two threads touch the same T element
                for (int64_t p = 0; p < N; ++p) {
                    double* t = &T[static_cast<int64_t>(ca.row_of[sm.ids[p]]) * C];
                    const float* g = &G[p * C];
                    for (int64_t c = c0; c < c0 + 64; ++c) t[c] += g[c];
                }
            std::vector<uint16_t> g16(N * C);
            for (int64_t i = 0; i < N * C; ++i) g16[i] = bf16_bits(G[i]);
            CK(cudaMemcpy(g_b.ids, sm.ids.data(), N * 2, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(g_b.g32, G.data(), N * C * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(g_b.g16, g16.data(), N * C * 2, cudaMemcpyHostToDevice));
            CK(cudaMemset(upd32.data(), 0, V * C * 4));
            CK(cudaMemset(upd16.data(), 0, V * C * 2));
            cuda::embedding_backward_cuda(g_b.ids, g_b.g32, static_cast<float*>(upd32.data()), N, C, V, -1, C, 1);
            cuda::embedding_backward_cuda(g_b.ids, reinterpret_cast<const bfloat16_t*>(g_b.g16), static_cast<bfloat16_t*>(upd16.data()), N, C, V, -1, C, 1);
            CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
            acc32 += upd32;                                           // fp32 table: AutogradMeta `*grad_ += update`
            f2ok &= cuda_accumulate_lowp_into_fp32(acc16, upd16);     // bf16 table: AutogradMeta F2 path
            accgv += upd16;                                           // a bf16 running sum
            CK(cudaDeviceSynchronize());
        }
        sc(f2ok, "SC8 D3 M=" + std::to_string(M) + " " + gen.name + ": F2 cuda_accumulate_lowp_into_fp32 took every call (no fallback)");
        std::vector<float> x[6];
        x[0] = rows_of(acc32, ca);
        x[1] = rows_of(acc32.as_type(Dtype::Bfloat16), ca);
        x[2] = rows_of(acc16, ca);
        x[3] = rows_of(acc16.as_type(Dtype::Bfloat16), ca);
        x[4] = rows_of(accgv, ca);
        x[5].resize(U * C);
        for (int64_t i = 0; i < U * C; ++i) x[5][i] = bf16_rn(static_cast<float>(T[i]));
        int64_t maxK = 0; for (auto k : ca.K) maxK = std::max(maxK, k);
        std::printf("\nCASE|D3|M=%d|%s|N=%lld per micro-step|unique=%lld|maxK(total)=%lld\n", M, gen.name.c_str(), (long long)N, (long long)U, (long long)maxK);
        Metrics mt[6]; BucketStat bs[6][NBUCKET];
        for (int a = 0; a < 6; ++a) { mt[a] = measure(ca, T, x[a].data()); buckets(ca, mt[a], bs[a]); }
        for (int a = 0; a < 6; ++a) std::printf("  ARM %-36s global rel err %s\n", ARM[a], fmt(mt[a].global).c_str());
        std::printf("  %-12s", "bucket");
        for (int a = 0; a < 6; ++a) std::printf(" | %-24s", ARM[a]);
        std::printf("\n");
        bool k1exact = true;
        for (int b = 0; b < NBUCKET; ++b) {
            if (bs[0][b].n == 0) continue;
            std::printf("  %-12s", BUCKET_NAME[b]);
            for (int a = 0; a < 6; ++a) {
                std::printf(" | med %-7s max %-8s", fmt(bs[a][b].med).c_str(), fmt(bs[a][b].mx).c_str());
                std::fprintf(g_csv3, "%d,%s,%s,%s,%lld,%.6g,%.6g,%.6g\n", M, gen.name.c_str(), ARM[a], BUCKET_NAME[b],
                             (long long)bs[a][b].n, bs[a][b].med, bs[a][b].p95, bs[a][b].mx);
                if (b == 0 && bs[a][b].mx != 0.0) k1exact = false;
            }
            std::printf("  (rows %lld)\n", (long long)bs[0][b].n);
        }
        sc(k1exact, "SC9 D3 M=" + std::to_string(M) + " " + gen.name + ": every arm exact on rows seen once in the whole step");
        sc(mt[0].global < 1e-4, "SC4 D3 M=" + std::to_string(M) + " " + gen.name + ": fp32 arm rel err " + fmt(mt[0].global) + " < 1e-4");
        char line[512];
        std::snprintf(line, sizeof line, "D3 M=%-2d %-10s | global rel err: fp32 %-9s fp32+BF16_REDUCE %-9s bf16 buffer %-9s bf16 sent %-9s bf16 running %-9s floor %-9s",
                      M, gen.name.c_str(), fmt(mt[0].global).c_str(), fmt(mt[1].global).c_str(), fmt(mt[2].global).c_str(),
                      fmt(mt[3].global).c_str(), fmt(mt[4].global).c_str(), fmt(mt[5].global).c_str());
        g_summary.push_back(line);
    }
}

}  // namespace

int main(int argc, char** argv) {
    const char* fd = std::getenv("FLUX_DIR");
    g_flux_dir = fd ? fd : "/home/blu-bridge016/Downloads/Neural_Networks_exp_1926/flux_bin_data";
    const char* od = std::getenv("OUT_DIR");
    const std::string out = od ? od : ".";
    const char* ce = std::getenv("BLUTRAIN_EMB_BWD_COOPERATIVE");
    g_coop = !(ce && ce[0] == '0');
    const std::string parts = argc > 1 ? argv[1] : "d1,d2,d3";
    const std::string sfx = g_coop ? "" : "_fp32atomic";
    g_csv1 = std::fopen((out + "/embed_d1_buckets" + sfx + ".csv").c_str(), "w");
    g_csv2 = std::fopen((out + "/embed_d2" + sfx + ".csv").c_str(), "w");
    g_csv3 = std::fopen((out + "/embed_d3" + sfx + ".csv").c_str(), "w");
    std::fprintf(g_csv1, "ids,grad,arm,bucket,rows,median,p95,max\n");
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    std::printf("=== test_embedding_grad_precision  gpu %s (sm_%d%d)  omp threads %d  parts %s  fp32 path %s\n",
                prop.name, prop.major, prop.minor, omp_get_max_threads(), parts.c_str(), g_coop ? "cooperative" : "atomicAdd");
    if (parts.find("d1") != std::string::npos) part_d1();
    if (parts.find("d2") != std::string::npos) part_d2();
    if (parts.find("d3") != std::string::npos) part_d3();
    std::printf("\n==================== SUMMARY ====================\n");
    for (const auto& l : g_summary) std::printf("%s\n", l.c_str());
    std::printf("self-check failures: %d\nRESULT: %s\n", g_sc_fail, g_sc_fail == 0 ? "SELF-CHECKS PASS" : "SELF-CHECKS FAIL");
    std::fclose(g_csv1); std::fclose(g_csv2); std::fclose(g_csv3);
    return g_sc_fail == 0 ? 0 : 1;
}
