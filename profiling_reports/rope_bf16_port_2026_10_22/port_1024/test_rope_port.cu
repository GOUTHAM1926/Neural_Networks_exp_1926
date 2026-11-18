// Port test (2026-10-24, from the 10-23 one): the library's bf16 fused QK-RMSNorm + RoPE with the 10-24 backward
// (hd 128 = fused_qknorm_rope_bwd_pp_bf16_tm, DEPTH tokens in flight per warp, with and without positions; dgamma =
// reduce_dgamma_qk_bf16_tm<HD, U, STAGES>, register-buffered) against the stage-B library file (md5 5338bd7b, renamed
// into namespace OwnTensor::cuda::ref), both called through the public launchers and built with the LIBRARY's flags
// (--use_fast_math). Added cases (10-24): T 40 (warps with fewer tokens than DEPTH) and 5 / 1 and 7 / 1 heads (dgamma row
// counts that are not a multiple of U * STAGES, so the reduce's last partial buffer runs).
//   CHECK: every output bit for bit -- forward q_out, k_out, q_rstd, k_rstd; backward dq, dk, dq_gamma, dk_gamma --
//          for NeoX / interleaved x hd 128 / 64 x B 1 2 4 8 x no positions / packed positions (chats restart at 0),
//          T 4096, 18 / 6 heads; plus T 1000 (not a multiple of 32: the backward's tail tokens).
//   TIME : the production shape (NeoX, hd 128, 18 / 6 heads, T 4096), B 1 2 4 8, no positions / packed positions:
//          forward launcher and backward launcher (backward kernel + dgamma reduce), stage B vs port, 2 s warm-up, then
//          9 rounds of 200 launches each, the two alternating, median per side.
// build + run: run_port_test.sh <Tensor-Implementations dir> (copies the library file to new_lib.cu; sed makes ref_ns.cu
// from ref/FusedRMSRoPE_bf16_stageB.cu; nvcc 13.0, library flags).
#include "ref_ns.cu"     // stage B, namespace OwnTensor::cuda::ref
#include "new_lib.cu"    // the library file under test, namespace OwnTensor::cuda

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <random>
#include <vector>
#include <cuda_profiler_api.h>

using namespace OwnTensor;
using bf16h = __nv_bfloat16;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA error %s at %s:%d\n", \
    cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(2); } } while (0)

template <class T> static T* dalloc(size_t n) { T* p; CK(cudaMalloc(&p, n * sizeof(T))); return p; }
template <class T> static std::vector<T> fetch(const T* d, size_t n) {
    std::vector<T> h(n); CK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost)); return h;
}
template <class T> static size_t n_diff(const std::vector<T>& a, const std::vector<T>& b) {   // bitwise
    size_t n = 0;
    for (size_t i = 0; i < a.size(); ++i) n += std::memcmp(&a[i], &b[i], sizeof(T)) != 0;
    return n;
}

struct Case {
    int B, Hq = 18, Hkv = 6, T, hd; bool interleaved;
    size_t nq() const { return (size_t)B * T * Hq * hd; }
    size_t nk() const { return (size_t)B * T * Hkv * hd; }
};

struct Data {   // inputs, shared by both sides
    bf16h *q, *k, *dzq, *dzk; float *qg, *kg, *cs; int32_t* pos;
};
struct Outs {   // outputs of one side
    bf16h *qo, *ko, *dq, *dk; float *qr, *kr, *dqg, *dkg;
};

static void fill_bf16(bf16h* d, size_t n, float lo, float hi, std::mt19937& g) {
    std::uniform_real_distribution<float> u(lo, hi); std::vector<bf16h> h(n);
    for (auto& x : h) x = __float2bfloat16_rn(u(g));
    CK(cudaMemcpy(d, h.data(), n * sizeof(bf16h), cudaMemcpyHostToDevice));
}
static void fill_f32(float* d, size_t n, float lo, float hi, std::mt19937& g) {
    std::uniform_real_distribution<float> u(lo, hi); std::vector<float> h(n);
    for (auto& x : h) x = u(g);
    CK(cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
}
// packed rows: chats of 1..2048 tokens, each chat's positions restart at 0
static std::vector<int32_t> packed_positions(int B, int T, std::mt19937& g) {
    std::uniform_int_distribution<int> len(1, 2048); std::vector<int32_t> p((size_t)B * T);
    for (int b = 0; b < B; ++b) for (int t = 0; t < T;) { int L = std::min(len(g), T - t); for (int i = 0; i < L; ++i) p[(size_t)b * T + t + i] = i; t += L; }
    return p;
}

static Data make_data(const Case& c, bool with_pos, std::mt19937& g) {
    Data d{};
    d.q = dalloc<bf16h>(c.nq()); d.k = dalloc<bf16h>(c.nk()); d.dzq = dalloc<bf16h>(c.nq()); d.dzk = dalloc<bf16h>(c.nk());
    d.qg = dalloc<float>(c.hd); d.kg = dalloc<float>(c.hd); d.cs = dalloc<float>((size_t)c.T * c.hd); d.pos = nullptr;
    fill_bf16(d.q, c.nq(), -3.f, 3.f, g); fill_bf16(d.k, c.nk(), -3.f, 3.f, g);
    fill_bf16(d.dzq, c.nq(), -1.f, 1.f, g); fill_bf16(d.dzk, c.nk(), -1.f, 1.f, g);
    fill_f32(d.qg, c.hd, 0.5f, 1.5f, g); fill_f32(d.kg, c.hd, 0.5f, 1.5f, g);
    std::vector<float> cs((size_t)c.T * c.hd);   // the real cache: row p = cos(p f_i) | sin(p f_i), theta 50000
    const int half = c.hd / 2;
    for (int p = 0; p < c.T; ++p) for (int i = 0; i < half; ++i) {
        const double f = std::pow(50000.0, -2.0 * i / c.hd);
        cs[(size_t)p * c.hd + i] = (float)std::cos(p * f); cs[(size_t)p * c.hd + half + i] = (float)std::sin(p * f);
    }
    CK(cudaMemcpy(d.cs, cs.data(), cs.size() * sizeof(float), cudaMemcpyHostToDevice));
    if (with_pos) {
        auto p = packed_positions(c.B, c.T, g); d.pos = dalloc<int32_t>(p.size());
        CK(cudaMemcpy(d.pos, p.data(), p.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    }
    return d;
}
static Outs make_outs(const Case& c) {
    Outs o{};
    o.qo = dalloc<bf16h>(c.nq()); o.ko = dalloc<bf16h>(c.nk()); o.dq = dalloc<bf16h>(c.nq()); o.dk = dalloc<bf16h>(c.nk());
    o.qr = dalloc<float>((size_t)c.B * c.Hq * c.T); o.kr = dalloc<float>((size_t)c.B * c.Hkv * c.T);
    o.dqg = dalloc<float>(c.hd); o.dkg = dalloc<float>(c.hd);
    return o;
}
template <class T> static const bfloat16_t* cb(const T* p) { return reinterpret_cast<const bfloat16_t*>(p); }
template <class T> static bfloat16_t* mb(T* p) { return reinterpret_cast<bfloat16_t*>(p); }

// side 0 = stage B (ref), side 1 = port
static void fwd(int side, const Case& c, const Data& d, const Outs& o) {
    auto f = side ? cuda::fused_qk_norm_rope_forward_bf16_tm : cuda::ref::fused_qk_norm_rope_forward_bf16_tm;
    f(cb(d.q), cb(d.k), mb(o.qo), mb(o.ko), d.qg, d.kg, d.cs, o.qr, o.kr, c.B, c.Hq, c.Hkv, c.T, c.hd, 0, 1e-6f,
      c.interleaved, 0, d.pos);
}
static void bwd(int side, const Case& c, const Data& d, const Outs& o, const Outs& rstd_from) {
    auto f = side ? cuda::fused_qk_norm_rope_backward_bf16_tm : cuda::ref::fused_qk_norm_rope_backward_bf16_tm;
    f(cb(d.dzq), cb(d.dzk), cb(d.q), cb(d.k), d.qg, d.kg, d.cs, rstd_from.qr, rstd_from.kr, mb(o.dq), mb(o.dk),
      o.dqg, o.dkg, c.B, c.Hq, c.Hkv, c.T, c.hd, 0, c.interleaved, 0, d.pos);
}

// --ncu: no checks, no timing; one backward call of the library under test per case (B 1 and B 4, without and with
// positions, production shape), each after 3 warm-up calls, inside cudaProfilerStart/Stop, for
// ncu --profile-from-start off (10-24: why does B 1 with positions stay at ~90 us while B 1 without reaches 56 us?).
static int ncu_mode() {
    std::mt19937 g(1023);
    for (int B : {1, 4}) for (bool with_pos : {false, true}) {
        Case c{B, 18, 6, 4096, 128, false};
        Data d = make_data(c, with_pos, g); Outs o = make_outs(c);
        fwd(1, c, d, o);
        for (int i = 0; i < 3; ++i) bwd(1, c, d, o, o);
        CK(cudaDeviceSynchronize());
        cudaProfilerStart(); bwd(1, c, d, o, o); CK(cudaDeviceSynchronize()); cudaProfilerStop();
        std::printf("ncu: profiled one backward, B %d %s\n", B, with_pos ? "positions" : "none");
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--ncu") == 0) return ncu_mode();
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    std::printf("device %s | library flags (--use_fast_math -O3) | stage B (5338bd7b) vs port\n", prop.name);
    std::mt19937 g(1023);
    int fails = 0, checks = 0;

    // ---------------------------------------------------------------- CHECK
    std::vector<Case> cases;
    for (int hd : {128, 64}) for (bool il : {false, true}) for (int B : {1, 2, 4, 8}) cases.push_back({B, 18, 6, 4096, hd, il});
    cases.push_back({4, 18, 6, 1000, 128, false}); cases.push_back({4, 18, 6, 1000, 128, true});
    cases.push_back({2, 18, 6, 40, 128, false});   cases.push_back({2, 18, 6, 40, 128, true});    // 1-2 tokens per warp
    cases.push_back({1, 5, 1, 4096, 128, false});  cases.push_back({3, 7, 1, 1000, 128, true});   // reduce tail buffers
    for (const Case& c : cases) for (bool with_pos : {false, true}) {
        Data d = make_data(c, with_pos, g);
        Outs r = make_outs(c), n = make_outs(c);
        fwd(0, c, d, r); fwd(1, c, d, n);
        bwd(0, c, d, r, r); bwd(1, c, d, n, r);   // both backwards read the SAME rstd (the reference forward's)
        CK(cudaDeviceSynchronize());
        size_t df[8] = {
            n_diff(fetch(r.qo, c.nq()), fetch(n.qo, c.nq())), n_diff(fetch(r.ko, c.nk()), fetch(n.ko, c.nk())),
            n_diff(fetch(r.qr, (size_t)c.B * c.Hq * c.T), fetch(n.qr, (size_t)c.B * c.Hq * c.T)),
            n_diff(fetch(r.kr, (size_t)c.B * c.Hkv * c.T), fetch(n.kr, (size_t)c.B * c.Hkv * c.T)),
            n_diff(fetch(r.dq, c.nq()), fetch(n.dq, c.nq())), n_diff(fetch(r.dk, c.nk()), fetch(n.dk, c.nk())),
            n_diff(fetch(r.dqg, c.hd), fetch(n.dqg, c.hd)), n_diff(fetch(r.dkg, c.hd), fetch(n.dkg, c.hd))};
        size_t tot = 0; for (size_t x : df) tot += x;
        // a vacuous-check guard: the outputs must not be all zero (a launch that did nothing would compare equal)
        auto dqg = fetch(n.dqg, c.hd); bool nz = false; for (float x : dqg) nz |= (x != 0.f);
        ++checks; if (tot != 0 || !nz) ++fails;
        std::printf("CHECK %-11s hd %3d B %d H %2d/%d T %4d %-9s | q_out %zu k_out %zu q_rstd %zu k_rstd %zu | dq %zu dk %zu dq_gamma %zu dk_gamma %zu | %s\n",
                    c.interleaved ? "interleaved" : "NeoX", c.hd, c.B, c.Hq, c.Hkv, c.T, with_pos ? "positions" : "none",
                    df[0], df[1], df[2], df[3], df[4], df[5], df[6], df[7],
                    (tot == 0 && nz) ? "bit-identical" : (nz ? "DIFFERENT" : "ALL-ZERO OUTPUT"));
        for (void* p : {(void*)d.q, (void*)d.k, (void*)d.dzq, (void*)d.dzk, (void*)d.qg, (void*)d.kg, (void*)d.cs, (void*)d.pos,
                        (void*)r.qo, (void*)r.ko, (void*)r.dq, (void*)r.dk, (void*)r.qr, (void*)r.kr, (void*)r.dqg, (void*)r.dkg,
                        (void*)n.qo, (void*)n.ko, (void*)n.dq, (void*)n.dk, (void*)n.qr, (void*)n.kr, (void*)n.dqg, (void*)n.dkg})
            if (p) CK(cudaFree(p));
    }
    std::printf("CHECK RESULT: %d of %d cases bit-identical in all 8 outputs%s\n", checks - fails, checks, fails ? "  <-- NOT PASSED" : "");
    if (fails) return 1;

    // ---------------------------------------------------------------- TIME
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    const int reps = 200, rounds = 9;
    std::printf("TIME production shape: NeoX, hd 128, 18 / 6 heads, T 4096 | %d launches per round, median of %d rounds (us per call)\n", reps, rounds);
    for (int B : {1, 2, 4, 8}) for (bool with_pos : {false, true}) {
        Case c{B, 18, 6, 4096, 128, false};
        Data d = make_data(c, with_pos, g);
        Outs r = make_outs(c), n = make_outs(c);
        fwd(0, c, d, r);
        auto time_us = [&](auto&& body) {
            CK(cudaEventRecord(e0)); for (int i = 0; i < reps; ++i) body(); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
            float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); return 1000.0f * ms / reps;
        };
        const auto t_end = std::chrono::steady_clock::now() + std::chrono::seconds(2);   // warm-up (clocks, caches)
        while (std::chrono::steady_clock::now() < t_end) { fwd(0, c, d, r); bwd(0, c, d, r, r); fwd(1, c, d, n); bwd(1, c, d, n, r); }
        CK(cudaDeviceSynchronize());
        std::vector<float> f0, f1, b0, b1;
        for (int k = 0; k < rounds; ++k) {   // alternate the sides inside every round
            f0.push_back(time_us([&] { fwd(0, c, d, r); }));       f1.push_back(time_us([&] { fwd(1, c, d, n); }));
            b1.push_back(time_us([&] { bwd(1, c, d, n, r); }));    b0.push_back(time_us([&] { bwd(0, c, d, r, r); }));
        }
        auto med = [](std::vector<float> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
        const float F0 = med(f0), F1 = med(f1), B0 = med(b0), B1 = med(b1);
        std::printf("  B %d %-9s | forward %7.1f -> %7.1f (%.2fx) | backward+reduce %7.1f -> %7.1f (%.2fx) | total %7.1f -> %7.1f (%.2fx)\n",
                    B, with_pos ? "positions" : "none", F0, F1, F0 / F1, B0, B1, B0 / B1, F0 + B0, F1 + B1, (F0 + B0) / (F1 + B1));
        for (void* p : {(void*)d.q, (void*)d.k, (void*)d.dzq, (void*)d.dzk, (void*)d.qg, (void*)d.kg, (void*)d.cs, (void*)d.pos,
                        (void*)r.qo, (void*)r.ko, (void*)r.dq, (void*)r.dk, (void*)r.qr, (void*)r.kr, (void*)r.dqg, (void*)r.dkg,
                        (void*)n.qo, (void*)n.ko, (void*)n.dq, (void*)n.dk, (void*)n.qr, (void*)n.kr, (void*)n.dqg, (void*)n.dkg})
            if (p) CK(cudaFree(p));
    }
    std::printf("RESULT: all checks bit-identical; times above\n");
    return 0;
}
