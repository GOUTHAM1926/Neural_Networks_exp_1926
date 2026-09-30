// test_f4_w1_exact_grad.cpp -- F4 on all three ZeRO optimizers, one GPU (W=1, no mpirun).
//
// F4: at W=1 nothing is sent, yet every optimizer still turns the fp32 gradient of a bf16 weight
// into bf16 (the send dtype) and back to fp32 before Adam/Muon reads it. That rounds the gradient
// once per step for no reason. After the fix the optimizer must see the fp32 gradient EXACTLY.
//
//   v1  ShardedOptimizer           (default)
//   v2  SymmetricShardedOptimizer  (BLU_ZERO_V2=1)
//   mu  ShardedMuon                (BLU_OPTIMIZER=muon; 2 matrices on the Muon path + 1 vector on
//                                   its AdamW path)
//
// The gradients are chosen so bf16 CANNOT hold them: g = v * (1 + 2^-10), v a full-rank bf16
// pattern. bf16 rounds every such g back to exactly v (2^-10 is below half a bf16 ulp), so the old
// code hands the optimizer v instead of g.
//
// Checks (one "CASE|opt|dtype|case|PASS/FAIL|detail" line each, then a SUMMARY block):
//   norm_exact    last_grad_norm() (computed from the buffer the update reads) equals the fp64 norm
//                 of g to 1e-5 relative. Old code on bf16: off by ~2^-10 = 9.8e-4 -> FAIL.
//   sees_exact    two fresh optimizers, one given g and one given v = bf16(g): after one step their
//                 state (fp32 master + moments) must DIFFER. Old code on bf16: both saw v, the state
//                 is byte-identical -> FAIL.
//   autograd2     the real trainer path: two backward passes accumulate 1.0*c + 2^-10*c into the
//                 fp32 gradient buffer of a bf16 weight (autograd keeps it fp32), then the norm check.
//   nograd        one weight gets no gradient at all; must behave exactly like a zero gradient.
//   flag_bf16     reduce_dtype forced to BF16 (= BLU_BF16_REDUCE=1). At W=1 nothing is sent, so the
//                 fixed code must still see g exactly (and print one notice). v1 refuses a reduce
//                 dtype that differs from its weights: that refusal must stay (expect the throw).
// fp32 weights are the CONTROL for the first four checks: they never rounded, so they must PASS
// before and after the fix. (flag_bf16 on fp32 v2/mu is expected to FAIL on the old code: the flag
// itself forces the rounding there.)
//
// Run it against the OLD build first: the bf16 cases must FAIL (the test can see F4), the fp32
// controls must PASS. Then against the patched build: all PASS.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "TensorLib.h"
#include "autograd/operations/BinaryOps.h"
#include "autograd/operations/ReductionOps.h"
#include "nn/optimizer/Optim.h"

#include "ShardedOptimizer.h"
#include "SymmetricShardedOptimizer.h"
#include "ShardedMuon.h"

using OwnTensor::Dtype;
using OwnTensor::Shape;
using OwnTensor::Tensor;
using OwnTensor::TensorOptions;
using namespace owt::zero;

namespace {

OwnTensor::DeviceIndex cuda0() { return OwnTensor::DeviceIndex{OwnTensor::Device::CUDA, 0}; }
TensorOptions f32opts() { return TensorOptions().with_dtype(Dtype::Float32).with_device(cuda0()); }

// round-to-nearest-even fp32 -> bf16 -> fp32, on the host
float bf16_round(float x) {
    uint32_t b; std::memcpy(&b, &x, 4);
    b = (b + 0x7FFFu + ((b >> 16) & 1u)) & 0xFFFF0000u;
    float r; std::memcpy(&r, &b, 4);
    return r;
}

Tensor from_host(const std::vector<float>& h, const std::vector<int64_t>& dims) {
    Shape s; s.dims = dims;
    Tensor t = Tensor::zeros(s, f32opts());
    t.set_data(h.data(), h.size());
    return t;
}

Tensor make_param(const std::vector<int64_t>& dims, float base, Dtype dt) {
    int64_t n = 1; for (auto d : dims) n *= d;
    std::vector<float> init(n);
    for (int64_t i = 0; i < n; ++i) init[i] = base + 0.001f * static_cast<float>(i % 97);
    Tensor f = from_host(init, dims);
    Tensor t = (dt == Dtype::Float32) ? f : f.as_type(dt);
    t.set_requires_grad(true);
    return t;
}

// sizes multiple of 64 (Muon's bf16 Newton-Schulz rejects 64x48 / 32x64, see the F1 test)
std::vector<std::vector<int64_t>> shapes() { return {{256, 192}, {128, 256}, {192}}; }
std::vector<Tensor> make_params(Dtype dt) {
    const auto sh = shapes();
    return {make_param(sh[0], 0.5f, dt), make_param(sh[1], -0.25f, dt), make_param(sh[2], 1.0f, dt)};
}

// v: full-rank pseudo-random pattern in [-0.01, 0.01), every element exactly representable in bf16
std::vector<float> pattern_v(int64_t n, int64_t salt) {
    std::vector<float> v(n);
    for (int64_t i = 0; i < n; ++i) {
        uint32_t h = static_cast<uint32_t>(i + 7919 * n + 104729 * salt) * 2654435761u;
        float x = 0.01f * (static_cast<float>((h >> 8) & 0xFFFF) / 32768.0f - 1.0f);
        v[i] = bf16_round(x == 0.0f ? 0.001f : x);
    }
    return v;
}
// g = v * (1 + 2^-10): exact in fp32, NOT representable in bf16 (bf16 rounds it back to v)
std::vector<float> pattern_g(const std::vector<float>& v) {
    std::vector<float> g(v.size());
    for (size_t i = 0; i < v.size(); ++i) g[i] = v[i] * (1.0f + 1.0f / 1024.0f);
    return g;
}

// fp32 gradient on every weight (bf16 weights too: that IS the trainer's state, autograd keeps
// the gradient of a bf16 weight in fp32)
void set_grads(std::vector<Tensor>& ps, const std::vector<std::vector<float>>& gs, int skip = -1) {
    for (size_t i = 0; i < ps.size(); ++i)
        if (static_cast<int>(i) != skip) ps[i].set_grad(from_host(gs[i], ps[i].shape().dims));
}

double fp64_norm(const std::vector<std::vector<float>>& gs, int skip = -1) {
    double s = 0.0;
    for (size_t i = 0; i < gs.size(); ++i)
        if (static_cast<int>(i) != skip) for (float x : gs[i]) s += static_cast<double>(x) * x;
    return std::sqrt(s);
}

std::string state_bytes(ZeroOptimizerBase& opt) {   // fp32 master + moments + step count
    std::ostringstream os;
    try { opt.save_state(os); } catch (const std::exception& e) { return std::string("ERR:") + e.what(); }
    return os.str();
}

std::unique_ptr<ZeroOptimizerBase> make_opt(const std::string& which, const std::vector<Tensor>& ps,
                                            bool force_bf16_reduce) {
    if (which == "mu") {                          // as the trainer's ShardedMuon config, W=1
        MuonConfig m;
        m.lr = 0.02f; m.momentum = 0.95f; m.weight_decay = 0.1f; m.ns_steps = 5; m.nesterov = true;
        m.adamw_lr = 1e-2f; m.adamw_beta1 = 0.9f; m.adamw_beta2 = 0.95f; m.adamw_eps = 1e-8f;
        m.adamw_weight_decay = 0.1f; m.exclude_1d_from_weight_decay = true;
        m.max_grad_norm = 1e6f;                   // norm is measured, never clips
        m.init_broadcast = false;
        if (force_bf16_reduce) m.reduce_dtype = ReduceDtype::BF16;
        return std::make_unique<ShardedMuon>(ps, nullptr, m);
    }
    ZeroConfig z;
    z.variant = AdamVariant::ADAMW; z.lr = 1e-2f; z.beta1 = 0.9f; z.beta2 = 0.95f; z.eps = 1e-8f;
    z.weight_decay = 0.1f; z.exclude_1d_from_weight_decay = true;
    z.max_grad_norm = 1e6f;                       // norm is measured, never clips
    z.init_broadcast = false;
    if (force_bf16_reduce) z.reduce_dtype = ReduceDtype::BF16;
    if (which == "v2") return std::make_unique<SymmetricShardedOptimizer>(ps, nullptr, z);
    return std::make_unique<ShardedOptimizer>(ps, nullptr, z);
}

int g_pass = 0, g_fail = 0;
void report(const std::string& opt, const char* dt, const char* cs, bool ok, const std::string& detail) {
    std::printf("CASE|%s|%s|%s|%s|%s\n", opt.c_str(), dt, cs, ok ? "PASS" : "FAIL", detail.c_str());
    std::fflush(stdout);
    (ok ? g_pass : g_fail)++;
}

std::string fmt(double x) { char b[64]; std::snprintf(b, sizeof b, "%.9g", x); return b; }

// kind: "norm_exact", "sees_exact", "autograd2", "nograd", "flag_bf16"
void run_case(const std::string& which, Dtype dt, const char* dts, const std::string& k) {
    const auto sh = shapes();
    std::vector<std::vector<float>> vs, gs;
    for (size_t i = 0; i < sh.size(); ++i) {
        int64_t n = 1; for (auto d : sh[i]) n *= d;
        vs.push_back(pattern_v(n, static_cast<int64_t>(i)));
        gs.push_back(pattern_g(vs.back()));
    }
    const double tol = 1e-5;

    if (k == "norm_exact" || k == "flag_bf16") {
        const bool force = (k == "flag_bf16");
        std::vector<Tensor> ps = make_params(dt);
        std::unique_ptr<ZeroOptimizerBase> opt;
        try { opt = make_opt(which, ps, force); }
        catch (const std::exception& e) {
            // v1 refuses a reduce dtype different from its weights (constructor check, kept as is)
            const bool expected = force && which == "v1" && dt == Dtype::Float32;
            report(which, dts, k.c_str(), expected, std::string(expected ? "refused as expected: " : "UNEXPECTED: ") + e.what());
            return;
        }
        set_grads(ps, gs);
        cudaDeviceSynchronize();
        opt->step();
        cudaDeviceSynchronize();
        const double ref = fp64_norm(gs), got = opt->last_grad_norm(), rel = std::fabs(got - ref) / ref;
        const double ref_v = fp64_norm(vs), rel_v = std::fabs(got - ref_v) / ref_v;
        report(which, dts, k.c_str(), rel < tol,
               "grad_norm " + fmt(got) + " | fp64 norm of g " + fmt(ref) + " (rel err " + fmt(rel) +
               ") | fp64 norm of bf16(g) " + fmt(ref_v) + " (rel err " + fmt(rel_v) + ")");
        return;
    }

    if (k == "sees_exact") {
        std::vector<Tensor> pa = make_params(dt), pb = make_params(dt);
        auto oa = make_opt(which, pa, false), ob = make_opt(which, pb, false);
        const std::string s0a = state_bytes(*oa), s0b = state_bytes(*ob);
        set_grads(pa, gs);        // g
        set_grads(pb, vs);        // v = bf16(g), as fp32 values
        cudaDeviceSynchronize();
        oa->step(); ob->step();
        cudaDeviceSynchronize();
        const std::string sa = state_bytes(*oa), sb = state_bytes(*ob);
        const bool same_start = (s0a == s0b), differ = (sa != sb);
        report(which, dts, k.c_str(), same_start && differ,
               std::string("start states ") + (same_start ? "identical" : "DIFFERENT (setup bug)") +
               "; after one step with g vs bf16(g): " + (differ ? "states DIFFER (optimizer saw g)"
                                                                 : "states IDENTICAL (optimizer saw bf16(g))") +
               " | norms " + fmt(oa->last_grad_norm()) + " vs " + fmt(ob->last_grad_norm()));
        return;
    }

    if (k == "autograd2") {
        // Two backward passes into the same weights, as two micro-steps: c1 = v (bf16 values), then
        // c2 = v * 2^-10 (also exact in bf16). loss_k = sum(w * c_k) -> dL/dw = c_k. Autograd keeps
        // the gradient of a bf16 weight in fp32, so it ends as v + v*2^-10 = g exactly.
        std::vector<Tensor> ps = make_params(dt);
        auto opt = make_opt(which, ps, false);
        for (int pass = 0; pass < 2; ++pass) {
            for (size_t i = 0; i < ps.size(); ++i) {
                std::vector<float> c(vs[i]);
                if (pass == 1) for (auto& x : c) x *= 1.0f / 1024.0f;
                Tensor ct = from_host(c, ps[i].shape().dims);
                if (dt != Dtype::Float32) ct = ct.as_type(dt);
                Tensor loss = OwnTensor::autograd::sum(OwnTensor::autograd::mul(ps[i], ct));
                loss.backward();
            }
        }
        cudaDeviceSynchronize();
        std::string gd;
        for (size_t i = 0; i < ps.size(); ++i)
            gd += (i ? "," : "") + std::string(OwnTensor::get_dtype_name(ps[i].grad_view().dtype()));
        opt->step();
        cudaDeviceSynchronize();
        const double ref = fp64_norm(gs), got = opt->last_grad_norm(), rel = std::fabs(got - ref) / ref;
        report(which, dts, k.c_str(), rel < tol,
               "grad buffer dtypes [" + gd + "] | grad_norm " + fmt(got) + " | fp64 norm of g " + fmt(ref) +
               " (rel err " + fmt(rel) + ")");
        return;
    }

    if (k == "nograd") {
        // weight 1 gets NO gradient in run A, an explicit zero gradient in run B: identical result
        std::vector<Tensor> pa = make_params(dt), pb = make_params(dt);
        auto oa = make_opt(which, pa, false), ob = make_opt(which, pb, false);
        std::vector<std::vector<float>> gz = gs;
        std::fill(gz[1].begin(), gz[1].end(), 0.0f);
        set_grads(pa, gs, /*skip=*/1);
        set_grads(pb, gz);
        cudaDeviceSynchronize();
        oa->step(); ob->step();
        cudaDeviceSynchronize();
        const bool same = (state_bytes(*oa) == state_bytes(*ob));
        const double ref = fp64_norm(gs, 1), got = oa->last_grad_norm(), rel = std::fabs(got - ref) / ref;
        report(which, dts, k.c_str(), same && rel < tol,
               std::string("no-grad vs zero-grad state ") + (same ? "identical" : "DIFFERENT") +
               " | grad_norm " + fmt(got) + " vs fp64 " + fmt(ref) + " (rel err " + fmt(rel) + ")");
        return;
    }
}

}  // namespace

int main(int argc, char** argv) {
    const std::vector<std::string> opts = argc > 1 ? std::vector<std::string>(argv + 1, argv + argc)
                                                   : std::vector<std::string>{"v1", "v2", "mu"};
    const char* kinds[] = {"norm_exact", "sees_exact", "autograd2", "nograd", "flag_bf16"};
    for (const auto& o : opts)
        for (int d = 0; d < 2; ++d)
            for (const char* k : kinds) {
                const Dtype dt = d == 0 ? Dtype::Bfloat16 : Dtype::Float32;
                const char* dts = d == 0 ? "bf16" : "fp32";
                try { run_case(o, dt, dts, k); }
                catch (const std::exception& e) { report(o, dts, k, false, std::string("EXCEPTION: ") + e.what()); }
            }
    std::printf("\n==================== SUMMARY ====================\n");
    std::printf("cases: %d, PASS %d, FAIL %d\n", g_pass + g_fail, g_pass, g_fail);
    std::printf("RESULT: %s\n", g_fail == 0 ? "PASS" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}
