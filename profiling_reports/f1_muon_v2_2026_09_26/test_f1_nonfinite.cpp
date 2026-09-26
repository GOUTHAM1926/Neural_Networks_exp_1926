// test_f1_nonfinite.cpp -- F1 on all three ZeRO optimizers, one GPU (W=1, no mpirun).
//
// A NaN/Inf gradient with NO LossScaler must raise ZeroNonFiniteGradError from step() and leave
// the model parameters, the optimizer state and the step count untouched. With a LossScaler the
// old behaviour must stay: skip the step, no throw. A finite step must still run.
//
//   v1  ShardedOptimizer           (default)          -- already fixed in a264586: the CONTROL
//   v2  SymmetricShardedOptimizer  (BLU_ZERO_V2=1)
//   mu  ShardedMuon                (BLU_OPTIMIZER=muon)
//
// Run it against the OLD libdist first: v2/mu must FAIL cases 2-5 (NaN written into the weights),
// v1 must PASS. That proves the test can see the bug. Then against the patched libdist: all PASS.
//
// Output: one "CASE|opt|dtype|case|PASS/FAIL|detail" line per case, then a SUMMARY block.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "TensorLib.h"
#include "nn/optimizer/Optim.h"
#include "nn/optimizer/LossScaler.h"

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

Tensor make_param(const std::vector<int64_t>& dims, float base, Dtype dt) {
    Shape s; s.dims = dims;
    int64_t n = 1; for (auto d : dims) n *= d;
    std::vector<float> init(n);
    for (int64_t i = 0; i < n; ++i) init[i] = base + 0.001f * static_cast<float>(i % 97);
    Tensor f = Tensor::zeros(s, TensorOptions().with_dtype(Dtype::Float32).with_device(cuda0()));
    f.set_data(init.data(), init.size());
    Tensor t = (dt == Dtype::Float32) ? f : f.as_type(dt);
    t.set_requires_grad(true);
    return t;
}

// Gradient = val * a fixed pseudo-random pattern in [-1, 1), full rank like a real gradient.
// NOT a constant: a constant matrix is rank 1, and Muon's Newton-Schulz output for it is rejected
// by update_one() on every step (measured), which would hide what a NaN does.
void set_grad(Tensor& p, float val, float bad, Dtype dt) {   // bad != 0 -> element 0 = bad
    std::vector<float> g(p.numel());
    for (int64_t i = 0; i < p.numel(); ++i) {
        uint32_t h = static_cast<uint32_t>(i + 7919 * p.numel()) * 2654435761u;
        g[i] = val * (static_cast<float>((h >> 8) & 0xFFFF) / 32768.0f - 1.0f);
    }
    if (bad != 0.0f) g[0] = bad;
    Shape s; s.dims = p.shape().dims;
    Tensor gf = Tensor::zeros(s, TensorOptions().with_dtype(Dtype::Float32).with_device(cuda0()));
    gf.set_data(g.data(), g.size());
    p.set_grad(dt == Dtype::Float32 ? gf : gf.as_type(dt));
}

// raw bytes of every parameter, for a bit-exact before/after compare
std::string param_bytes(const std::vector<Tensor>& ps) {
    std::string out;
    for (const auto& p : ps) {
        Tensor c = p.to_cpu();
        out.append(static_cast<const char*>(c.data()), c.numel() * Tensor::dtype_size(c.dtype()));
    }
    return out;
}

int64_t count_nonfinite(const std::vector<Tensor>& ps) {
    int64_t bad = 0;
    for (const auto& p : ps) {
        Tensor c = p.to_cpu();
        if (c.dtype() != Dtype::Float32) c = c.as_type(Dtype::Float32);
        const float* d = c.data<float>();
        for (int64_t i = 0; i < c.numel(); ++i) if (!std::isfinite(d[i])) ++bad;
    }
    return bad;
}

// NaN/Inf elements in the optimizer's moment buffers (m, v): proves whether bad state was kept
int64_t count_nonfinite_state(const ZeroOptimizerBase& opt) {
    int64_t bad = 0;
    for (const auto& v : {opt.get_m_buffers(), opt.get_v_buffers()})
        for (const auto& t : v) if (t.is_valid() && t.numel() > 0) bad += count_nonfinite({t});
    return bad;
}

std::string state_bytes(ZeroOptimizerBase& opt) {   // master + moments + step count
    std::ostringstream os;
    try { opt.save_state(os); } catch (const std::exception& e) { return std::string("ERR:") + e.what(); }
    return os.str();
}

// 2 matrices (Muon path in ShardedMuon) + 1 vector (AdamW path in ShardedMuon)
std::vector<Tensor> make_params(Dtype dt) {
    // sizes multiple of 64: with 64x48 / 32x64, Muon's bf16 Newton-Schulz output was rejected on
    // EVERY step, healthy ones included (measured), so the test could not see what a NaN does
    return {make_param({256, 192}, 0.5f, dt), make_param({128, 256}, -0.25f, dt), make_param({192}, 1.0f, dt)};
}

std::unique_ptr<ZeroOptimizerBase> make_opt(const std::string& which, const std::vector<Tensor>& ps) {
    if (which == "mu") {                          // as bluscript_zero_ckpt.cpp:1333-1361, W=1
        MuonConfig m;
        m.lr = 0.02f; m.momentum = 0.95f; m.weight_decay = 0.1f; m.ns_steps = 5; m.nesterov = true;
        m.adamw_lr = 3e-4f; m.adamw_beta1 = 0.9f; m.adamw_beta2 = 0.95f; m.adamw_eps = 1e-8f;
        m.adamw_weight_decay = 0.1f; m.exclude_1d_from_weight_decay = true;
        m.max_grad_norm = 1.0f;                   // = cfg.grad_clip default
        m.init_broadcast = false;
        return std::make_unique<ShardedMuon>(ps, nullptr, m);
    }
    ZeroConfig z;                                 // as bluscript_zero_ckpt.cpp:1108-1121, W=1
    // lr 1e-2, not the trainer's 3e-4: one 3e-4 Adam step is below half a bf16 ulp at |x| ~ 0.5-1.1,
    // so bf16 params would legitimately not move and the finite case could not see the step.
    z.variant = AdamVariant::ADAMW; z.lr = 1e-2f; z.beta1 = 0.9f; z.beta2 = 0.95f; z.eps = 1e-8f;
    z.weight_decay = 0.1f; z.exclude_1d_from_weight_decay = true;
    z.max_grad_norm = 1.0f;                       // = cfg.grad_clip default
    z.init_broadcast = false;
    if (which == "v2") return std::make_unique<SymmetricShardedOptimizer>(ps, nullptr, z);
    return std::make_unique<ShardedOptimizer>(ps, nullptr, z);
}

int g_pass = 0, g_fail = 0;
void report(const std::string& opt, const char* dt, const char* cs, bool ok, const std::string& detail) {
    std::printf("CASE|%s|%s|%s|%s|%s\n", opt.c_str(), dt, cs, ok ? "PASS" : "FAIL", detail.c_str());
    std::fflush(stdout);
    (ok ? g_pass : g_fail)++;
}

// kind: "finite", "nan_matrix", "nan_vector", "inf_matrix", "nan_globalclip", "nan_scaler"
void run_case(const std::string& which, Dtype dt, const char* dts, const char* kind) {
    const std::string k = kind;
    std::vector<Tensor> ps = make_params(dt);
    std::unique_ptr<ZeroOptimizerBase> opt = make_opt(which, ps);
    OwnTensor::nn::LossScaler scaler(1.0f);       // init scale 1: the grads are not pre-scaled
    if (k == "nan_scaler") opt->set_scaler(&scaler);

    const float nan = std::numeric_limits<float>::quiet_NaN(), inf = std::numeric_limits<float>::infinity();
    const float bad = k == "finite" ? 0.0f : (k == "inf_matrix" ? inf : nan);
    const size_t bad_at = (k == "nan_vector") ? 2 : 0;   // 2 = the 1-D vector
    for (size_t i = 0; i < ps.size(); ++i) set_grad(ps[i], 0.01f, i == bad_at ? bad : 0.0f, dt);
    cudaDeviceSynchronize();

    const std::string p0 = param_bytes(ps), s0 = state_bytes(*opt);
    const int64_t c0 = opt->step_count();

    bool threw = false, other = false; std::string msg;
    try {
        if (k == "nan_globalclip") {
            const float nsq = opt->reduce_and_measure();   // the trainer's phase 1 (BLU_GLOBAL_CLIP=1)
            msg = "reduce_and_measure=" + std::to_string(nsq) + "; ";
        }
        opt->step();
        cudaDeviceSynchronize();
    } catch (const ZeroNonFiniteGradError& e) { threw = true; msg += e.what(); }
      catch (const std::exception& e)          { other = true; msg += std::string("OTHER EXCEPTION: ") + e.what(); }
    cudaDeviceSynchronize();

    const bool params_same = (param_bytes(ps) == p0);
    const bool state_same  = (state_bytes(*opt) == s0);
    const int64_t nf = count_nonfinite(ps), nfs = count_nonfinite_state(*opt);
    std::string st = "step_count " + std::to_string(c0) + "->" + std::to_string(opt->step_count()) +
                     ", params " + (params_same ? "unchanged" : "CHANGED") +
                     ", state " + (state_same ? "unchanged" : "CHANGED") +
                     ", non-finite params " + std::to_string(nf) + ", non-finite m/v " + std::to_string(nfs);
    // first-step results, taken BEFORE the optional follow-up step below changes them
    const int64_t c1 = opt->step_count(); const bool sk1 = opt->last_step_skipped();
    const float gn1 = opt->last_grad_norm();
    // Muon, step NOT stopped (incl. the finite CONTROL): take one more FINITE step and report how
    // many Newton-Schulz updates update_one() rejected in step 1 and in that next step. The finite
    // case is the control: it must show 0, or the toy data itself is being rejected.
    if (which == "mu" && k != "nan_scaler" && !threw && !other) {
        auto* sm = dynamic_cast<ShardedMuon*>(opt.get());
        st += "; step-1 NS updates rejected " + std::to_string(sm->ns_rejected_last_step());
        for (size_t i = 0; i < ps.size(); ++i) set_grad(ps[i], 0.01f, 0.0f, dt);
        try { sm->step(); cudaDeviceSynchronize();
              st += "; NEXT finite step: NS updates rejected " + std::to_string(sm->ns_rejected_last_step()) +
                    ", non-finite m/v " + std::to_string(count_nonfinite_state(*opt)); }
        catch (const std::exception& e) { st += std::string("; NEXT finite step threw: ") + e.what(); }
    }
    if (k == "finite") {
        const bool ok = !threw && !other && !sk1 && c1 == 1 && !params_same && nf == 0 && nfs == 0;
        report(which, dts, kind, ok, st + ", grad_norm " + std::to_string(gn1) + (msg.empty() ? "" : "; " + msg));
    } else if (k == "nan_scaler") {           // with a scaler: skip, no throw (old behaviour kept)
        const bool ok = !threw && !other && sk1 && c1 == c0 && params_same && state_same && nfs == 0;
        report(which, dts, kind, ok, st + ", skipped " + (sk1 ? "yes" : "no") + (msg.empty() ? "" : "; " + msg));
    } else {                                  // no scaler: must throw, nothing touched
        const bool ok = threw && c1 == c0 && params_same && state_same && nf == 0 && nfs == 0;
        report(which, dts, kind, ok, std::string(threw ? "threw ZeroNonFiniteGradError" : "DID NOT THROW") +
               ", " + st + "; " + msg);
    }
}

}  // namespace

int main(int argc, char** argv) {
    const std::vector<std::string> opts = argc > 1 ? std::vector<std::string>(argv + 1, argv + argc)
                                                   : std::vector<std::string>{"v1", "v2", "mu"};
    const char* kinds[] = {"finite", "nan_matrix", "nan_vector", "inf_matrix", "nan_globalclip", "nan_scaler"};
    for (const auto& o : opts)
        for (int d = 0; d < 2; ++d)
            for (const char* k : kinds) {
                const Dtype dt = d == 0 ? Dtype::Bfloat16 : Dtype::Float32;
                try { run_case(o, dt, d == 0 ? "bf16" : "fp32", k); }
                catch (const std::exception& e) { report(o, d == 0 ? "bf16" : "fp32", k, false, std::string("SETUP EXCEPTION: ") + e.what()); }
            }
    std::printf("\n==================== SUMMARY ====================\n");
    std::printf("cases: %d, PASS %d, FAIL %d\n", g_pass + g_fail, g_pass, g_fail);
    std::printf("RESULT: %s\n", g_fail == 0 ? "PASS" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}
