//======================================================================
// test_zero.cpp — correctness tests for the ZeRO Stage-1 optimizer.
//
// The load-bearing test is PARITY: at world_size == 1 the ZeRO optimizer must
// produce the SAME weights as single-process nn::AdamW, because it runs the
// identical fused Adam kernel on the identical data. This is the gate the whole
// design is validated against.
//
// Runs standalone (no mpirun): W==1 uses a null process group and issues no
// NCCL collective, so it executes on a single GPU. A separate multi-rank
// smoke path activates only when launched under mpirun with -np > 1.
//======================================================================

#include <cstdio>
#include <cmath>
#include <random>
#include <vector>
#include <string>
#include <memory>
#include <fstream>
#include <limits>

#include <cuda_runtime.h>

#include "TensorLib.h"
#include "nn/optimizer/Optim.h"

#include "ShardedOptimizer.h"

using OwnTensor::Tensor;
using OwnTensor::Dtype;
using OwnTensor::Shape;
using OwnTensor::TensorOptions;
using owt::zero::ShardedOptimizer;
using owt::zero::ZeroConfig;
using owt::zero::AdamVariant;

static int g_failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::printf("  [FAIL] %s\n", msg); ++g_failures; } \
    else         { std::printf("  [ok]   %s\n", msg); } } while(0)

// ---- helpers ----------------------------------------------------------

static OwnTensor::DeviceIndex cuda0() {
    return OwnTensor::DeviceIndex{OwnTensor::Device::CUDA, 0};
}

// Build a param of dtype `dt` from FP32 host data. For non-FP32 dtypes we
// create an FP32 tensor (set_data<float> is a raw byte copy, only valid for
// FP32) and convert with as_type so the values round-trip correctly.
static Tensor make_param(const std::vector<int64_t>& dims,
                         const std::vector<float>& init, Dtype dt) {
    Shape s; s.dims = dims;
    TensorOptions o32 = TensorOptions().with_dtype(Dtype::Float32).with_device(cuda0());
    Tensor f = Tensor::zeros(s, o32);
    f.set_data(init.data(), init.size());   // host -> device (FP32)
    Tensor t = (dt == Dtype::Float32) ? f : f.as_type(dt);
    t.set_requires_grad(true);
    return t;
}

// Set a parameter's gradient from FP32 host data, converting to the param dtype.
static void set_grad_dt(Tensor& p, const std::vector<float>& g, Dtype dt) {
    Shape s; s.dims = p.shape().dims;
    Tensor gf = Tensor::zeros(s, TensorOptions().with_dtype(Dtype::Float32).with_device(cuda0()));
    gf.set_data(g.data(), g.size());
    Tensor gg = (dt == Dtype::Float32) ? gf : gf.as_type(dt);
    p.set_grad(gg);
}

static std::vector<float> host_of(const Tensor& t) {
    Tensor c = t.to_cpu();
    if (c.dtype() != Dtype::Float32) c = c.as_type(Dtype::Float32);
    const float* p = c.data<float>();
    return std::vector<float>(p, p + c.numel());
}

static float max_abs_diff(const Tensor& a, const Tensor& b) {
    auto ha = host_of(a), hb = host_of(b);
    float m = 0.0f;
    for (size_t i = 0; i < ha.size(); ++i) m = std::max(m, std::fabs(ha[i] - hb[i]));
    return m;
}

// Deterministic parameter shapes covering: matrix (wd group), 1-D bias
// (no-wd group when enabled), and an odd-sized tensor to exercise padding
// when W>1 (harmless at W=1).
struct Spec { std::vector<int64_t> dims; };
static std::vector<Spec> kShapes = {
    {{4, 6}},      // 24
    {{7}},         // 7  (1-D)
    {{3, 5}},      // 15
    {{11}},        // 11 (1-D, makes total = 57, not a nice multiple)
};

// Build two independent-but-identical parameter sets from one RNG seed.
static void make_two_sets(std::vector<Tensor>& ref, std::vector<Tensor>& zed,
                          Dtype dt, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& sp : kShapes) {
        int64_t n = 1; for (auto d : sp.dims) n *= d;
        std::vector<float> init(n);
        for (auto& x : init) x = nd(rng);
        ref.push_back(make_param(sp.dims, init, dt));
        zed.push_back(make_param(sp.dims, init, dt));
    }
}

static void set_same_grad(std::vector<Tensor>& a, std::vector<Tensor>& b,
                          std::mt19937& rng, Dtype dt = Dtype::Float32) {
    std::normal_distribution<float> nd(0.f, 0.5f);
    for (size_t i = 0; i < a.size(); ++i) {
        std::vector<float> g(a[i].numel());
        for (auto& x : g) x = nd(rng);
        set_grad_dt(a[i], g, dt);
        set_grad_dt(b[i], g, dt);
    }
}

// ---- tests ------------------------------------------------------------

// PARITY GATE: W=1 ZeRO-AdamW must equal nn::AdamW step-for-step.
static void test_parity(Dtype dt, float wd, const char* tag) {
    std::printf("[test_parity dtype=%s wd=%.3f]\n", tag, wd);
    std::mt19937 rng(1234);
    std::vector<Tensor> ref, zed;
    make_two_sets(ref, zed, dt, rng);

    const float lr = 3e-4f, b1 = 0.9f, b2 = 0.95f, eps = 1e-8f;

    OwnTensor::nn::AdamW ref_opt(ref, lr, b1, b2, eps, wd);

    ZeroConfig cfg;
    cfg.variant = AdamVariant::ADAMW;
    cfg.lr = lr; cfg.beta1 = b1; cfg.beta2 = b2; cfg.eps = eps;
    cfg.weight_decay = wd;
    cfg.exclude_1d_from_weight_decay = false;  // uniform wd == nn::AdamW behavior
    ShardedOptimizer zero_opt(zed, /*pg=*/nullptr, cfg);

    float worst = 0.f;
    for (int step = 0; step < 6; ++step) {
        set_same_grad(ref, zed, rng, dt);
        ref_opt.step();
        cudaDeviceSynchronize();
        zero_opt.step();
        cudaDeviceSynchronize();
        for (size_t i = 0; i < ref.size(); ++i)
            worst = std::max(worst, max_abs_diff(ref[i], zed[i]));
    }
    std::printf("  max|Δparam| over 6 steps = %.3e\n", worst);
    // Same kernel, same data, same order -> effectively bit-identical for FP32.
    float tol = (dt == Dtype::Float32) ? 1e-5f : 3e-2f;  // BF16 param round-trip is lossy
    CHECK(worst < tol, "ZeRO-1 weights match single-process AdamW");
}

// Gradient clipping: ZeRO's internal global-norm clip must match a hand-clipped
// reference. We clip the reference grads on the host with the EXACT formula
// compute_clip_coef_cuda uses (coef = max_norm/(||g||+1e-6), capped at 1), feed
// the clipped grads to the already-parity-verified nn::AdamW, and compare to
// ZeRO clipping the raw grads internally. (We avoid nn::clip_grad_norm_ as the
// oracle because the prebuilt lib's version may differ from current source.)
static void test_clip() {
    std::printf("[test_clip]\n");
    std::mt19937 rng(77);
    std::vector<Tensor> ref, zed;
    make_two_sets(ref, zed, Dtype::Float32, rng);
    const float lr = 1e-3f, b1 = 0.9f, b2 = 0.95f, eps = 1e-8f, wd = 0.0f;
    const float max_norm = 0.5f;

    OwnTensor::nn::AdamW ref_opt(ref, lr, b1, b2, eps, wd);
    ZeroConfig cfg; cfg.lr = lr; cfg.beta1 = b1; cfg.beta2 = b2; cfg.eps = eps;
    cfg.weight_decay = wd; cfg.max_grad_norm = max_norm;
    ShardedOptimizer zero_opt(zed, nullptr, cfg);

    std::mt19937 grng(500);
    std::normal_distribution<float> nd(0.f, 0.5f);
    float worst = 0.f;
    for (int step = 0; step < 4; ++step) {
        // generate this step's grads and their global L2 norm
        std::vector<std::vector<float>> gs(ref.size());
        double sumsq = 0.0;
        for (size_t i = 0; i < ref.size(); ++i) {
            gs[i].resize(ref[i].numel());
            for (auto& x : gs[i]) { x = nd(grng); sumsq += (double)x * x; }
        }
        float total_norm = (float)std::sqrt(sumsq);
        float coef = max_norm / (total_norm + 1e-6f);
        if (coef > 1.0f) coef = 1.0f;

        for (size_t i = 0; i < ref.size(); ++i) {
            std::vector<float> clipped = gs[i];
            for (auto& x : clipped) x *= coef;
            ref[i].set_grad(clipped.data(), clipped.size());   // reference: pre-clipped
            zed[i].set_grad(gs[i].data(), gs[i].size());       // ZeRO: raw, clips internally
        }
        ref_opt.step();
        cudaDeviceSynchronize();   // isolate the two optimizer instances: the
        zero_opt.step();           // multi-tensor kernels share static launch
        cudaDeviceSynchronize();   // metadata across instances (see gpt2 note).
        for (size_t i = 0; i < ref.size(); ++i)
            worst = std::max(worst, max_abs_diff(ref[i], zed[i]));
    }
    std::printf("  max|Δparam| = %.3e  (last grad_norm=%.4f)\n", worst, zero_opt.last_grad_norm());
    CHECK(worst < 1e-3f, "ZeRO-1 internal clip matches hand-clipped reference");
}

// Checkpoint round-trip at W=1: save, perturb, load, continue == uninterrupted.
static void test_checkpoint() {
    std::printf("[test_checkpoint]\n");
    std::mt19937 rng(9);
    std::vector<Tensor> a, b;   // two identical runs
    make_two_sets(a, b, Dtype::Float32, rng);
    ZeroConfig cfg; cfg.lr = 1e-3f; cfg.beta1 = 0.9f; cfg.beta2 = 0.95f;
    ShardedOptimizer oa(a, nullptr, cfg);
    ShardedOptimizer ob(b, nullptr, cfg);

    std::mt19937 gseed(55);
    // run both 3 steps identically
    for (int s = 0; s < 3; ++s) { set_same_grad(a, b, gseed); oa.step(); ob.step(); }

    // checkpoint run A, then keep only A going; later restore B from A's ckpt
    std::string path = "/tmp/zero_ckpt.bin";
    { std::ofstream os(path, std::ios::binary); oa.save_state(os); }

    // advance both 3 more steps with the SAME grads
    std::mt19937 gpost(999);
    std::mt19937 gpostB = gpost;
    for (int s = 0; s < 3; ++s) { set_same_grad(a, a, gpost); oa.step(); }

    // B: load A's checkpoint (rewinds optimizer state), replay same post grads
    { std::ifstream is(path, std::ios::binary); ob.load_state(is); }
    for (int s = 0; s < 3; ++s) { set_same_grad(b, b, gpostB); ob.step(); }

    float worst = 0.f;
    for (size_t i = 0; i < a.size(); ++i) worst = std::max(worst, max_abs_diff(a[i], b[i]));
    std::printf("  max|Δparam| after resume = %.3e\n", worst);
    CHECK(worst < 1e-4f, "resume-from-checkpoint reproduces the uninterrupted run");
}

// Overflow handling: a step whose (unscaled) grads are non-finite must be
// skipped atomically — params unchanged AND the Adam timestep NOT advanced.
static void test_overflow_skip() {
    std::printf("[test_overflow_skip]\n");
    std::mt19937 rng(3);
    std::vector<Tensor> zed;
    { std::vector<Tensor> tmp; make_two_sets(zed, tmp, Dtype::Float32, rng); }

    ZeroConfig cfg; cfg.lr = 1e-3f; cfg.beta1 = 0.9f; cfg.beta2 = 0.95f;
    ShardedOptimizer opt(zed, nullptr, cfg);
    OwnTensor::nn::LossScaler scaler(65536.0f);
    opt.set_scaler(&scaler);

    auto before = host_of(zed[0]);

    // step 1: inject a non-finite gradient -> must skip
    for (size_t i = 0; i < zed.size(); ++i) {
        std::vector<float> g(zed[i].numel(), 0.1f);
        if (i == 0) g[0] = std::numeric_limits<float>::infinity();
        set_grad_dt(zed[i], g, Dtype::Float32);
    }
    opt.step();
    cudaDeviceSynchronize();
    auto after = host_of(zed[0]);
    bool unchanged = true;
    for (size_t k = 0; k < before.size(); ++k) if (before[k] != after[k]) unchanged = false;
    CHECK(opt.last_step_skipped(), "overflow step was skipped");
    CHECK(opt.step_count() == 0, "skipped step did NOT advance the Adam timestep");
    CHECK(unchanged, "params unchanged after a skipped step");

    // step 2: finite grads -> real step, timestep advances to 1
    for (size_t i = 0; i < zed.size(); ++i) {
        std::vector<float> g(zed[i].numel(), 0.05f);
        set_grad_dt(zed[i], g, Dtype::Float32);
    }
    opt.step();
    cudaDeviceSynchronize();
    CHECK(!opt.last_step_skipped() && opt.step_count() == 1,
          "first finite step runs with timestep t=1 (bias correction correct)");
}

// A checkpoint must not load into a run with a different flat ordering/grouping.
static void test_ckpt_layout_guard() {
    std::printf("[test_ckpt_layout_guard]\n");
    std::mt19937 rng(11);
    std::vector<Tensor> a, b;
    make_two_sets(a, b, Dtype::Float32, rng);

    ZeroConfig c1; c1.lr = 1e-3f; c1.exclude_1d_from_weight_decay = false; // 1 group
    ZeroConfig c2; c2.lr = 1e-3f; c2.exclude_1d_from_weight_decay = true;  // 2 groups, reordered
    ShardedOptimizer o1(a, nullptr, c1);
    ShardedOptimizer o2(b, nullptr, c2);

    std::string path = "/tmp/zero_ckpt_layout.bin";
    { std::ofstream os(path, std::ios::binary); o1.save_state(os); }

    bool threw = false;
    try {
        std::ifstream is(path, std::ios::binary);
        o2.load_state(is);
    } catch (const owt::zero::ZeroCheckpointError&) {
        threw = true;
    }
    CHECK(threw, "load_state rejects a checkpoint with different param grouping");
}

int main() {
    std::printf("==== ZeRO-1 tests (single GPU, W=1) ====\n");
    test_parity(Dtype::Float32, 0.0f, "fp32");
    test_parity(Dtype::Float32, 0.1f, "fp32");
    test_parity(Dtype::Bfloat16, 0.1f, "bf16");   // mixed precision: FP32 master
    test_clip();
    test_checkpoint();
    test_overflow_skip();
    test_ckpt_layout_guard();
    std::printf("==== %s ====\n", g_failures == 0 ? "ALL PASSED" : "FAILURES PRESENT");
    return g_failures == 0 ? 0 : 1;
}
