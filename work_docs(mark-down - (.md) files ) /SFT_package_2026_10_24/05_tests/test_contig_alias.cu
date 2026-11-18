// test_contig_alias.cu -- 2026-11-07, one GPU (RTX 3060 / any sm_80+): does autograd::contiguous() on an output that is
// ALREADY contiguous and is output #1 of a two-output op lose that output's gradient? This is exactly the SFT CP code:
//   auto [q_bf, k_bf] = fused_rms_rope_bf16(...);          // q_bf = output 0, k_bf = output 1
//   cp_combine(autograd::contiguous(q_bf)); cp_combine(autograd::contiguous(k_bf));
// Run A: loss straight on (q_bf, k_bf)        -- what the trainer does without CP.
// Run B: loss on autograd::contiguous(q_bf) / autograd::contiguous(k_bf) -- what the CP code does (no all-to-all here:
//        one GPU, so any difference comes from autograd::contiguous alone).
// Same inputs, same loss weights. Printed: the backward link of k_bf before / after the contiguous call, then the
// gradient norms of q, k (inputs) and q_gamma, k_gamma.
#include "TensorLib.h"
#include "autograd/AutogradOps.h"
#include "autograd/operations/RoPEOps.h"
#include "autograd/operations/NormalizationOps.h"
#include "autograd/operations/ReshapeOps.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <string>
using namespace OwnTensor;

static std::string gnorm(const Tensor& t) {
    try {
        Tensor g = t.grad_view().as_type(Dtype::Float32).to_cpu();
        const float* p = g.data<float>(); double s = 0;
        for (int64_t i = 0; i < g.numel(); ++i) s += (double)p[i] * p[i];
        char b[64]; std::snprintf(b, sizeof b, "%.6f", std::sqrt(s)); return b;
    } catch (const std::exception&) { return "NO GRADIENT"; }
}
static std::string link(const Tensor& t) {
    auto f = t.grad_fn();
    char b[160]; std::snprintf(b, sizeof b, "%s, output_nr %u", f ? f->name() : "(none)", t.output_nr()); return b;
}

int main() {
    const int64_t B = 2, T = 256, Nq = 4, Nkv = 2, hd = 128;
    const DeviceIndex dev(Device::CUDA, 0);
    auto f32 = TensorOptions().with_dtype(Dtype::Float32).with_device(dev);
    Tensor cache = autograd::build_rope_cache(T, hd, 50000.0f, dev);
    Tensor Gq = Tensor::randn<float>(Shape{{B, Nq, T, hd}}, f32, 3);
    Tensor Gk = Tensor::randn<float>(Shape{{B, Nkv, T, hd}}, f32, 4);
    for (int use_contig = 0; use_contig < 2; ++use_contig) {
        Tensor q = Tensor::randn<float>(Shape{{B, T, Nq * hd}}, f32, 1).as_type(Dtype::Bfloat16);
        Tensor k = Tensor::randn<float>(Shape{{B, T, Nkv * hd}}, f32, 2).as_type(Dtype::Bfloat16);
        Tensor qg = Tensor::randn<float>(Shape{{hd}}, f32, 7, 0.5f), kg = Tensor::randn<float>(Shape{{hd}}, f32, 8, 0.5f);
        q.set_requires_grad(true); k.set_requires_grad(true); qg.set_requires_grad(true); kg.set_requires_grad(true);
        auto [qo, ko] = autograd::fused_rms_rope_bf16(q, k, Nq, Nkv, qg, kg, cache, 0, 1e-6f, false, Tensor());
        std::printf("\n%s\n", use_contig ? "B: loss on autograd::contiguous(q_bf), autograd::contiguous(k_bf)  (the CP code)"
                                          : "A: loss straight on q_bf, k_bf  (the trainer without CP)");
        std::printf("  k_bf link before        : %s   (k_bf contiguous: %d)\n", link(ko).c_str(), (int)ko.is_contiguous());
        Tensor qu = qo, ku = ko;
        if (use_contig) {
            qu = autograd::contiguous(qo);
            ku = autograd::contiguous(ko);
            std::printf("  k_bf link after         : %s\n", link(ko).c_str());
            std::printf("  contiguous(k_bf) link   : %s   (same storage as k_bf: %d)\n", link(ku).c_str(),
                        (int)(ku.data() == ko.data()));
        }
        Tensor L = autograd::add(autograd::sum(autograd::mul(autograd::to_dtype(qu, Dtype::Float32), Gq)),
                                 autograd::sum(autograd::mul(autograd::to_dtype(ku, Dtype::Float32), Gk)));
        L.backward();
        cudaDeviceSynchronize();
        std::printf("  grad norms: q %s | k %s | q_gamma %s | k_gamma %s\n", gnorm(q).c_str(), gnorm(k).c_str(),
                    gnorm(qg).c_str(), gnorm(kg).c_str());
    }
    return 0;
}
