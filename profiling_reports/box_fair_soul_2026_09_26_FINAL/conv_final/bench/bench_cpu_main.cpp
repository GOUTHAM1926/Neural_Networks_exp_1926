// bench_cpu_main.cpp -- CPU micro-benchmark: OLD vs NEW conversion code (2026-09-26).
//
// For every conversion path x every input kind x {1 thread, all threads}:
//   * time OLD and NEW back to back, 12 rounds, order alternating (old,new / new,old),
//     report the median ns per element and the paired new/old ratio (median, min, max);
//   * check precision of BOTH against an independent reference (not the new code):
//     exact-by-value fp32->bf16, x86 F16C for fp16, GCC __bf16 / _Float16 for double,
//     integers and products (products of two bf16 / fp16 values are exact in fp32).
// A fingerprint first proves the two copies really are the old and the new code.
//
// Build: see build_cpu.sh (library CPU flags: -O3 -mavx2 -mfma -mf16c -fopenmp).
#include "bench_cpu_api.h"
#include <immintrin.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

template <class To, class From> static To bc(const From& f) { To t; std::memcpy(&t, &f, sizeof(To)); return t; }

// ---------------- independent references ----------------
static uint16_t ex_f2bf(uint32_t u) {                       // round to nearest even, by value
    uint32_t sign = (u >> 16) & 0x8000u, e = (u >> 23) & 0xFF, m = u & 0x7FFFFFu;
    if (e == 0xFF) return m ? (uint16_t)(sign | 0x7FC0u) : (uint16_t)(sign | 0x7F80u);
    uint64_t sig = (e == 0) ? m : (m | 0x800000u), q = sig >> 16, rem = sig & 0xFFFFu;
    if (rem > 0x8000u || (rem == 0x8000u && (q & 1))) q++;
    uint32_t bits;
    if (e == 0) bits = (uint32_t)q;
    else { uint32_t ee = e, qq = (uint32_t)q; if (qq == 0x100) { qq = 0x80; ee++; } bits = (ee >= 0xFF) ? 0x7F80u : ((ee << 7) | (qq & 0x7F)); }
    return (uint16_t)(sign | bits);
}
static uint16_t hw_f2h(float f) { return (uint16_t)_cvtss_sh(f, _MM_FROUND_TO_NEAREST_INT); }
static float hw_h2f(uint16_t h) { return _cvtsh_ss(h); }
static float bf2f(uint16_t b) { return bc<float>((uint32_t)b << 16); }
template <class T> static uint16_t gcc_bf(T v) { __bf16 h = (__bf16)v; return bc<uint16_t>(h); }
template <class T> static uint16_t gcc_h(T v) { _Float16 h = (_Float16)v; return bc<uint16_t>(h); }
static bool nan_bf(uint16_t h) { return (h & 0x7F80u) == 0x7F80u && (h & 0x7Fu); }
static bool nan_h(uint16_t h) { return (h & 0x7C00u) == 0x7C00u && (h & 0x3FFu); }

// ---------------- inputs ----------------
enum InKind { K_F32, K_BF16, K_F16, K_F64, K_I32, K_I64, K_I8, K_BF16PAIR, K_F16PAIR };
struct Case { int fn; const char* fname; int kind; const char* cname; int cls; };
static const size_t N = 1u << 22;                          // 4,194,304 elements per case
static std::mt19937_64 rng(20260926);
static double logu(double lo, double hi) { std::uniform_real_distribution<double> u(std::log(lo), std::log(hi)); return std::exp(u(rng)); }
static double sgn() { return (rng() & 1) ? -1.0 : 1.0; }

// class ids (meaning depends on the input kind; the table below names them)
static void make_input(const Case& c, std::vector<unsigned char>& buf) {
    auto rnd32 = [] { return (uint32_t)rng(); };
    switch (c.kind) {
    case K_F32: {
        buf.resize(N * 4); float* a = (float*)buf.data();
        for (size_t i = 0; i < N; ++i) {
            float v;
            switch (c.cls) {
                case 0: v = (float)(sgn() * logu(1e-3, 1e3)); break;                      // normal
                case 1: v = (float)(sgn() * logu(65536.0, 1e30)); break;                  // large >= 65,536
                case 2: v = (float)(sgn() * logu(0x1p-24, 0x1p-14)); break;               // fp16 subnormal range
                case 3: v = bc<float>((rnd32() & 0x807FFFFFu) | 1u); break;               // fp32 subnormal = bf16 subnormal range
                case 4: { const float s[] = {0.0f, -0.0f, INFINITY, -INFINITY, NAN, -NAN}; v = s[rng() % 6]; } break;  // specials
                default: v = bc<float>(rnd32()); break;                                   // 5/6: random bits (all kinds)
            }
            a[i] = v;
        }
        if (c.cls == 6) std::sort(a, a + N, [](float x, float y) { return (bc<uint32_t>(x) & 0x7F800000u) < (bc<uint32_t>(y) & 0x7F800000u); });
    } break;
    case K_BF16: case K_F16: {
        buf.resize(N * 2); uint16_t* a = (uint16_t*)buf.data();
        for (size_t i = 0; i < N; ++i) {
            const float v = (float)(sgn() * logu(1e-3, 1e3));
            if (c.cls == 0) a[i] = c.kind == K_BF16 ? ex_f2bf(bc<uint32_t>(v)) : hw_f2h(v);     // normal
            else if (c.cls == 1) a[i] = (uint16_t)((rng() & 0x83FFu) | 1u);                  // fp16 subnormal bits (exp 0)
            else a[i] = (uint16_t)rng();                                                      // random bits
        }
    } break;
    case K_BF16PAIR: case K_F16PAIR: {
        buf.resize(N * 4); uint16_t* a = (uint16_t*)buf.data();
        for (size_t i = 0; i < 2 * N; ++i) {
            const float v = (float)(sgn() * logu(c.cls == 0 ? 1e-2 : 1.0, c.cls == 0 ? 1e2 : (c.kind == K_BF16PAIR ? 1e20 : 250.0)));
            a[i] = c.cls == 2 ? (uint16_t)rng() : (c.kind == K_BF16PAIR ? ex_f2bf(bc<uint32_t>(v)) : hw_f2h(v));
        }
    } break;
    case K_F64: {
        buf.resize(N * 8); double* a = (double*)buf.data();
        for (size_t i = 0; i < N; ++i) {
            switch (c.cls) {
                case 0: a[i] = sgn() * logu(1e-3, 1e3); break;
                case 1: a[i] = sgn() * logu(65536.0, 1e300); break;
                case 2: a[i] = sgn() * logu(1e-45, 1e-5); break;
                default: a[i] = bc<double>(rng()); break;
            }
        }
    } break;
    case K_I32: {
        buf.resize(N * 4); int32_t* a = (int32_t*)buf.data();
        for (size_t i = 0; i < N; ++i) a[i] = c.cls == 0 ? (int32_t)(rng() % 511) - 255 : c.cls == 1 ? (int32_t)(rng() % (1u << 25)) - (1 << 24) : (int32_t)rng();
    } break;
    case K_I64: {
        buf.resize(N * 8); int64_t* a = (int64_t*)buf.data();
        for (size_t i = 0; i < N; ++i) a[i] = c.cls == 0 ? (int64_t)(rng() % 511) - 255 : (int64_t)rng();
    } break;
    case K_I8: { buf.resize(N); for (size_t i = 0; i < N; ++i) buf[i] = (unsigned char)rng(); } break;
    }
}

// reference for element i; returns false for "NaN expected" cases handled by the caller
static void check(const Case& c, const std::vector<unsigned char>& in, const std::vector<unsigned char>& out, long long& wrong, long long& nanbad) {
    wrong = nanbad = 0;
    for (size_t i = 0; i < N; ++i) {
        uint32_t got32 = 0; uint16_t got = 0, ref = 0; bool isnan_in = false, is16 = true, nanout = false;
        switch (c.fn) {
        case F_F32_BF16: case F_VEC_F32_BF16: { const float f = ((const float*)in.data())[i]; isnan_in = std::isnan(f); ref = ex_f2bf(bc<uint32_t>(f)); got = ((const uint16_t*)out.data())[i]; nanout = nan_bf(got); } break;
        case F_F32_F16: { const float f = ((const float*)in.data())[i]; isnan_in = std::isnan(f); ref = hw_f2h(f); got = ((const uint16_t*)out.data())[i]; nanout = nan_h(got); } break;
        case F_BF16_F32: { const uint16_t b = ((const uint16_t*)in.data())[i]; is16 = false; got32 = ((const uint32_t*)out.data())[i];
                           if (got32 != ((uint32_t)b << 16)) ++wrong; continue; }
        case F_F16_F32: { const uint16_t h = ((const uint16_t*)in.data())[i]; is16 = false; const float r = hw_h2f(h); const float g = ((const float*)out.data())[i];
                          if (std::isnan(r)) { if (!std::isnan(g)) ++nanbad; } else if (bc<uint32_t>(g) != bc<uint32_t>(r)) ++wrong; continue; }
        case F_F64_BF16: { const double d = ((const double*)in.data())[i]; isnan_in = std::isnan(d); ref = gcc_bf(d); got = ((const uint16_t*)out.data())[i]; nanout = nan_bf(got); } break;
        case F_F64_F16:  { const double d = ((const double*)in.data())[i]; isnan_in = std::isnan(d); ref = gcc_h(d); got = ((const uint16_t*)out.data())[i]; nanout = nan_h(got); } break;
        case F_I32_BF16: ref = gcc_bf(((const int32_t*)in.data())[i]); got = ((const uint16_t*)out.data())[i]; break;
        case F_I32_F16:  ref = gcc_h(((const int32_t*)in.data())[i]); got = ((const uint16_t*)out.data())[i]; break;
        case F_I64_BF16: ref = gcc_bf((long long)((const int64_t*)in.data())[i]); got = ((const uint16_t*)out.data())[i]; break;
        case F_I8_BF16:  ref = gcc_bf((int)((const int8_t*)in.data())[i]); got = ((const uint16_t*)out.data())[i]; break;
        case F_BF16_MUL: { const uint16_t* a = (const uint16_t*)in.data(); const float p = bf2f(a[2 * i]) * bf2f(a[2 * i + 1]);   // exact in fp32
                           isnan_in = std::isnan(p); ref = gcc_bf(p); got = ((const uint16_t*)out.data())[i]; nanout = nan_bf(got); } break;
        case F_F16_MUL:  { const uint16_t* a = (const uint16_t*)in.data(); const float p = hw_h2f(a[2 * i]) * hw_h2f(a[2 * i + 1]); // exact in fp32
                           isnan_in = std::isnan(p); ref = gcc_h(p); got = ((const uint16_t*)out.data())[i]; nanout = nan_h(got); } break;
        case F_BF16_TO_F16: { const float v = bf2f(((const uint16_t*)in.data())[i]); isnan_in = std::isnan(v); ref = gcc_h(v); got = ((const uint16_t*)out.data())[i]; nanout = nan_h(got); } break;
        }
        (void)is16;
        if (isnan_in) { if (!nanout) ++nanbad; } else if (got != ref) ++wrong;
    }
}

static size_t out_bytes(int fn) { return (fn == F_BF16_F32 || fn == F_F16_F32) ? N * 4 : N * 2; }
static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main() {
    const int T_ALL = omp_get_max_threads();
    std::printf("CPU micro-benchmark, old vs new conversion code. N = %zu elements per case, threads: 1 and %d\n", N, T_ALL);
    // fingerprint: the two copies must really be old and new
    const uint16_t o1 = bench_old::probe_f2bf(70000.0f), n1 = bench_new::probe_f2bf(70000.0f);
    const uint16_t o2 = bench_old::probe_f2h(0.00012204051f), n2 = bench_new::probe_f2h(0.00012204051f);
    std::printf("fingerprint: f2bf(70000) old=0x%04x (expect 0x7f80 = Inf, Bug D)  new=0x%04x (expect 0x4789)\n", o1, n1);
    std::printf("             f2h(0.00012204051) old=0x%04x (expect 0x0400, halved)  new=0x%04x (expect 0x0800)\n", o2, n2);
    if (o1 != 0x7F80 || n1 != 0x4789 || o2 != 0x0400 || n2 != 0x0800) { std::printf("FINGERPRINT MISMATCH: the build is not old-vs-new. Stop.\n"); return 2; }

    const std::vector<Case> cases = {
        {F_F32_BF16, "f32->bf16", K_F32, "normal", 0}, {F_F32_BF16, "f32->bf16", K_F32, "large>=65536", 1}, {F_F32_BF16, "f32->bf16", K_F32, "bf16-subnormal", 3},
        {F_F32_BF16, "f32->bf16", K_F32, "specials", 4}, {F_F32_BF16, "f32->bf16", K_F32, "random-bits", 5}, {F_F32_BF16, "f32->bf16", K_F32, "random-sorted", 6},
        {F_VEC_F32_BF16, "vec f32->bf16", K_F32, "normal", 0}, {F_VEC_F32_BF16, "vec f32->bf16", K_F32, "large>=65536", 1}, {F_VEC_F32_BF16, "vec f32->bf16", K_F32, "random-bits", 5},
        {F_BF16_F32, "bf16->f32", K_BF16, "normal", 0}, {F_BF16_F32, "bf16->f32", K_BF16, "random-bits", 2},
        {F_F32_F16, "f32->f16", K_F32, "normal", 0}, {F_F32_F16, "f32->f16", K_F32, "large>=65536", 1}, {F_F32_F16, "f32->f16", K_F32, "f16-subnormal", 2},
        {F_F32_F16, "f32->f16", K_F32, "specials", 4}, {F_F32_F16, "f32->f16", K_F32, "random-bits", 5}, {F_F32_F16, "f32->f16", K_F32, "random-sorted", 6},
        {F_F16_F32, "f16->f32", K_F16, "normal", 0}, {F_F16_F32, "f16->f32", K_F16, "f16-subnormal", 1}, {F_F16_F32, "f16->f32", K_F16, "random-bits", 2},
        {F_F64_BF16, "f64->bf16", K_F64, "normal", 0}, {F_F64_BF16, "f64->bf16", K_F64, "large>=65536", 1}, {F_F64_BF16, "f64->bf16", K_F64, "tiny", 2}, {F_F64_BF16, "f64->bf16", K_F64, "random-bits", 3},
        {F_F64_F16, "f64->f16", K_F64, "normal", 0}, {F_F64_F16, "f64->f16", K_F64, "large>=65536", 1}, {F_F64_F16, "f64->f16", K_F64, "tiny", 2}, {F_F64_F16, "f64->f16", K_F64, "random-bits", 3},
        {F_I32_BF16, "i32->bf16", K_I32, "|v|<256", 0}, {F_I32_BF16, "i32->bf16", K_I32, "|v|<2^24", 1}, {F_I32_BF16, "i32->bf16", K_I32, "full-range", 2},
        {F_I32_F16, "i32->f16", K_I32, "|v|<256", 0}, {F_I32_F16, "i32->f16", K_I32, "full-range", 2},
        {F_I64_BF16, "i64->bf16", K_I64, "|v|<256", 0}, {F_I64_BF16, "i64->bf16", K_I64, "full-range", 1},
        {F_I8_BF16, "i8->bf16", K_I8, "full-range", 0},
        {F_BF16_MUL, "bf16*bf16", K_BF16PAIR, "normal", 0}, {F_BF16_MUL, "bf16*bf16", K_BF16PAIR, "big (overflow 65536)", 1}, {F_BF16_MUL, "bf16*bf16", K_BF16PAIR, "random-bits", 2},
        {F_F16_MUL, "f16*f16", K_F16PAIR, "normal", 0}, {F_F16_MUL, "f16*f16", K_F16PAIR, "random-bits", 2},
        {F_BF16_TO_F16, "bf16->f16", K_BF16, "normal", 0}, {F_BF16_TO_F16, "bf16->f16", K_BF16, "random-bits", 2},
    };
    const int ROUNDS = 12;
    struct Res { std::string name; int threads; double old_med, new_med, r_med, r_min, r_max; long long ow, on, nw, nn; };
    std::vector<Res> all;
    std::vector<unsigned char> in, out_old, out_new;
    std::printf("\n%-15s %-21s %3s %11s %11s %8s %19s   %-24s %s\n", "function", "input", "thr", "old ns/el", "new ns/el", "speedup", "new/old min..max", "old: wrong / NaN-bad", "new: wrong / NaN-bad");
    for (const Case& c : cases) {
        make_input(c, in); out_old.assign(out_bytes(c.fn), 0); out_new.assign(out_bytes(c.fn), 0xAB);
        // precision (one pass each, all threads)
        bench_old::run(c.fn, in.data(), out_old.data(), N, T_ALL);
        bench_new::run(c.fn, in.data(), out_new.data(), N, T_ALL);
        long long ow, on, nw, nn; check(c, in, out_old, ow, on); check(c, in, out_new, nw, nn);
        for (int thr : {1, T_ALL}) {
            // calibrate repetitions to ~15 ms per measurement (on the new code)
            double t0 = now_ms(); bench_new::run(c.fn, in.data(), out_new.data(), N, thr); double one = now_ms() - t0;
            const int reps = std::max(1, (int)(15.0 / std::max(one, 1e-3)));
            for (int w = 0; w < 2; ++w) { bench_old::run(c.fn, in.data(), out_old.data(), N, thr); bench_new::run(c.fn, in.data(), out_new.data(), N, thr); }
            std::vector<double> to, tn, ratio;
            for (int r = 0; r < ROUNDS; ++r) {
                double a = 0, b = 0;
                for (int k = 0; k < 2; ++k) {
                    const bool do_old = ((r & 1) == 0) ? (k == 0) : (k == 1);   // old,new then new,old
                    double s = now_ms();
                    for (int q = 0; q < reps; ++q) (do_old ? bench_old::run : bench_new::run)(c.fn, in.data(), (do_old ? out_old : out_new).data(), N, thr);
                    const double e = (now_ms() - s) * 1e6 / ((double)reps * N);   // ns per element
                    (do_old ? a : b) = e;
                }
                to.push_back(a); tn.push_back(b); ratio.push_back(b / a);
            }
            auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]); };
            Res R{std::string(c.fname) + " | " + c.cname, thr, med(to), med(tn), med(ratio), *std::min_element(ratio.begin(), ratio.end()), *std::max_element(ratio.begin(), ratio.end()), ow, on, nw, nn};
            std::printf("%-15s %-21s %3d %11.4f %11.4f %7.2fx %8.3f..%-8.3f   %11lld / %-10lld %lld / %lld\n", c.fname, c.cname, thr, R.old_med, R.new_med, R.old_med / R.new_med, R.r_min, R.r_max, ow, on, nw, nn);
            std::fflush(stdout);
            all.push_back(R);
        }
    }
    // ---------------- SUMMARY ----------------
    std::printf("\n==================== SUMMARY ====================\n");
    long long new_wrong = 0, new_nan = 0, old_wrong = 0; for (auto& r : all) if (r.threads == 1) { new_wrong += r.nw; new_nan += r.nn; old_wrong += r.ow; }
    std::printf("precision: old code wrong on %lld inputs across all cases, new code wrong on %lld (NaN-bad %lld)\n", old_wrong, new_wrong, new_nan);
    std::printf("slower cases (paired new/old: every round > 1.03):\n"); int slow = 0;
    for (auto& r : all) if (r.r_min > 1.03) { ++slow; std::printf("   %-40s thr %3d  new/old median %.3f (min %.3f)\n", r.name.c_str(), r.threads, r.r_med, r.r_min); }
    if (!slow) std::printf("   none\n");
    std::printf("faster cases (paired new/old: every round < 0.97): ");
    int fast = 0; for (auto& r : all) fast += r.r_max < 0.97; std::printf("%d of %zu\n", fast, all.size());
    std::printf("unclear (rounds straddle 0.97..1.03): %zu\n", all.size() - fast - slow);
    std::printf("RESULT: precision %s\n", new_wrong || new_nan ? "FAIL" : "PASS");
    return (new_wrong || new_nan) ? 1 : 0;
}
