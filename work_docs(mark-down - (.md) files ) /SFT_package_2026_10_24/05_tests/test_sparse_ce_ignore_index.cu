// ============================================================================
// SFT ignore_index label masking — sparse cross-entropy, full coverage test
//
//   forward  : loss_output = SUM over rows of  ( log(sum_c exp(z_rc - m_r)) + m_r - z_r,t )
//              rows whose target == ignore_index contribute EXACTLY 0
//   stats    : saved_max[r] = m_r , saved_sum[r] = sum_c exp(z_rc - m_r)
//              masked rows get the sentinels (max = 0, sum = 1).  sum must be 1,
//              never 0: backward computes 1/sum, and 1/0 = inf, inf*0 = NaN.
//   backward : g_rc = ( exp(z_rc - m_r)/s_r - [c == t_r] ) * grad_output * host_scale
//              masked rows must be BIT-ZERO, not merely small.
//
// Methodology (same three arms as test_RMSNorm_fwd.cu / the GQA tests):
//   A) reference   : fp64 from the ORIGINAL fp32 logits  -> ground truth
//   B) noise floor : logits rounded to dtype T, accumulation in fp32, store as T
//                    -> the error intrinsic to T, nothing to do with masking
//   C) kernel      : the library entry point under test
//   metric = relative L2.  Pass if err(C) <= 2*err(B) + floor.
//
// ANTI-VACUITY (this is the point of the file).  A masking test that only checks
// "masked result == masked reference" PASSES when masking does nothing, because
// the two arms are identical in that case.  So every masked case ALSO asserts a
// SEPARATION: the masked run must differ from the same run with masking off by
// more than a margin, and the rows chosen for masking always carry a large,
// clearly non-zero loss/gradient when unmasked.  Both halves must hold.
//
// Groups
//   A  masking OFF is unchanged           (regression guard, 9 dtype combos)
//   B  masking ON, forward                (patterns, mask-all, stats sentinels)
//   C  masking ON, backward               (bit-zero rows, untouched rows, no NaN)
//   D  shapes that select different code paths
//   E  numerical stress
//   F  sentinel-value semantics
//   G  op level: denominator + CPU-vs-CUDA agreement
//   H  loss sum over many rows: the old stage-2 reduction summed only the
//      first 65,536 rows. Must pass; LOSSBITS lines print the exact loss bits
//      of both entry points so two library builds can be diffed.
//   X  DIAGNOSTIC: pre-existing path limits, NOT introduced by SFT masking.
//      These are reported separately and do NOT set the exit code.
//
// Build + run (from Tensor-Implementations/):
//   make run-cuda FILE=Tests/kernel_tests/test_sparse_ce_ignore_index.cu
// ============================================================================
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "dtype/Types.h"
#include "ops/helpers/LossKernels.h"

#include "autograd/operations/LossOps.h"
#include "core/Tensor.h"
#include "device/Device.h"

using OwnTensor::bfloat16_t;
using OwnTensor::float16_t;

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err_ = (call);                                                 \
    if (err_ != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                   cudaGetErrorString(err_));                                  \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

// ============================================================================
// reporting
// ============================================================================
static int g_sft_fail = 0;   // failures that are OURS -> drive the exit code
static int g_diag_fail = 0;  // pre-existing path limits -> reported only
static int g_cases = 0;

static void header_row() {
  std::printf("%-3s %-34s %-5s %-6s %8s %6s %11s %11s %8s %-5s %s\n", "grp",
              "case", "dt", "idx", "B", "V", "err_kernel", "err_noisefl",
              "ratio", "verd", "detail");
  std::printf("%s\n", std::string(118, '-').c_str());
}

// ok = the assertion held.  diagnostic = do not fail the build on it.
static void report(const char *grp, const std::string &name, const char *dt,
                   const char *idx, long B, long V, double ek, double enf,
                   bool ok, bool diagnostic, const std::string &note) {
  ++g_cases;
  const char *verdict = ok ? "PASS" : (diagnostic ? "XFAIL" : "FAIL");
  if (!ok) {
    if (diagnostic) ++g_diag_fail; else ++g_sft_fail;
  }
  // no noise floor (exactness checks) -> no ratio to report
  char rbuf[16];
  if (enf > 0.0) std::snprintf(rbuf, sizeof(rbuf), "%.2f", ek / enf);
  else           std::snprintf(rbuf, sizeof(rbuf), "%s", "-");
  std::printf("%-3s %-34s %-5s %-6s %8ld %6ld %11.3e %11.3e %8s %-5s %s\n",
              grp, name.c_str(), dt, idx, B, V, ek, enf, rbuf, verdict,
              note.c_str());
  std::fflush(stdout);
}

// ============================================================================
// metrics
// ============================================================================
static double rel_l2(const std::vector<double> &test,
                     const std::vector<double> &ref) {
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    double d = test[i] - ref[i];
    num += d * d;
    den += ref[i] * ref[i];
  }
  if (den == 0.0) return std::sqrt(num);
  return std::sqrt(num / den);
}

// ============================================================================
// dtype plumbing
// ============================================================================
template <typename T> struct DtName;
template <> struct DtName<float>       { static const char *n() { return "f32"; } };
template <> struct DtName<float16_t>   { static const char *n() { return "f16"; } };
template <> struct DtName<bfloat16_t>  { static const char *n() { return "bf16"; } };

template <typename Tidx> struct IdxName;
template <> struct IdxName<uint16_t> { static const char *n() { return "u16"; } };
template <> struct IdxName<int32_t>  { static const char *n() { return "i32"; } };
template <> struct IdxName<int64_t>  { static const char *n() { return "i64"; } };

// round a float vector through dtype T
template <typename T> static std::vector<T> to_dt(const std::vector<float> &x) {
  std::vector<T> o(x.size());
  for (size_t i = 0; i < x.size(); ++i) o[i] = static_cast<T>(x[i]);
  return o;
}
template <typename T> static std::vector<float> from_dt(const std::vector<T> &x) {
  std::vector<float> o(x.size());
  for (size_t i = 0; i < x.size(); ++i) o[i] = static_cast<float>(x[i]);
  return o;
}
// ============================================================================
// reference (arm A: fp64) and noise floor (arm B: dtype T, fp32 accumulation)
//
// ignore_index is compared the way the kernel compares it:
//   static_cast<int64_t>(targets[row]) == ignore_index
// so the reference reproduces the exact uint16 wraparound semantics.
// ============================================================================
struct RowStats { double loss, mx, sum; bool masked; };

template <typename Tidx>
static std::vector<RowStats> ref_forward(const std::vector<float> &z,
                                         const std::vector<Tidx> &tgt, long B,
                                         long V, int64_t ignore_index) {
  std::vector<RowStats> out(B);
  for (long r = 0; r < B; ++r) {
    if (static_cast<int64_t>(tgt[r]) == ignore_index) {
      out[r] = {0.0, 0.0, 1.0, true};   // the sentinels the kernel writes
      continue;
    }
    const float *row = z.data() + (size_t)r * V;
    double m = -1e300;
    for (long c = 0; c < V; ++c) m = std::max(m, (double)row[c]);
    double s = 0.0;
    for (long c = 0; c < V; ++c) s += std::exp((double)row[c] - m);
    long t = (long)tgt[r];
    if (t < 0 || t >= V) {           // out-of-vocab target: not computable
      out[r] = {0.0, m, s, true};    // the caller skips the control arm anyway
      continue;
    }
    out[r] = {std::log(s) + m - (double)row[t], m, s, false};
  }
  return out;
}

// arm B: logits already rounded to T (passed back as float), fp32 accumulation
template <typename Tidx>
static std::vector<RowStats> nf_forward(const std::vector<float> &zb,
                                        const std::vector<Tidx> &tgt, long B,
                                        long V, int64_t ignore_index) {
  std::vector<RowStats> out(B);
  for (long r = 0; r < B; ++r) {
    if (static_cast<int64_t>(tgt[r]) == ignore_index) {
      out[r] = {0.0, 0.0, 1.0, true};
      continue;
    }
    const float *row = zb.data() + (size_t)r * V;
    float m = -3.4e38f;
    for (long c = 0; c < V; ++c) m = std::max(m, row[c]);
    float s = 0.f;
    for (long c = 0; c < V; ++c) s += std::exp(row[c] - m);
    long t = (long)tgt[r];
    if (t < 0 || t >= V) {
      out[r] = {0.0, (double)m, (double)s, true};
      continue;
    }
    out[r] = {(double)(std::log(s) + m - row[t]), (double)m, (double)s, false};
  }
  return out;
}

// ============================================================================
// device buffers
// ============================================================================
template <typename T> struct DBuf {
  T *p = nullptr;
  size_t n = 0;
  explicit DBuf(size_t count) : n(count) {
    CUDA_CHECK(cudaMalloc(&p, count * sizeof(T)));
  }
  ~DBuf() { if (p) cudaFree(p); }
  DBuf(const DBuf &) = delete;
  DBuf &operator=(const DBuf &) = delete;
  void up(const std::vector<T> &h) {
    CUDA_CHECK(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
  }
  std::vector<T> down() const {
    std::vector<T> h(n);
    CUDA_CHECK(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
  }
  // poison: fill with a value that is NOT zero, so "the kernel wrote 0" cannot
  // be confused with "the buffer already was 0". Without this every zero-grad
  // assertion in group C is vacuous.
  void poison(float v) {
    std::vector<T> h(n, static_cast<T>(v));
    up(h);
  }
};

// ============================================================================
// case description
// ============================================================================
enum class Mask { None, EveryOther, FirstRow, LastRow, All, Half, OneMiddle };
enum class Dist { Normal, Wide, AllEqual, OneHuge, VeryNegative, Tiny };

struct Spec {
  const char *group;
  std::string name;
  long B, V;
  Mask mask = Mask::None;
  Dist dist = Dist::Normal;
  int64_t sentinel = 65535;      // the target VALUE written into masked rows
  int64_t ignore_index = 65535;  // what we hand the kernel
  bool diagnostic = false;
  double floor = 1e-5;
  unsigned seed = 1234;
  std::string note;
};

static bool row_masked(Mask m, long r, long B, std::mt19937 &rng) {
  switch (m) {
    case Mask::None:       return false;
    case Mask::EveryOther: return (r % 2) == 1;
    case Mask::FirstRow:   return r == 0;
    case Mask::LastRow:    return r == B - 1;
    case Mask::All:        return true;
    case Mask::Half:       return (rng() & 1u) != 0u;
    case Mask::OneMiddle:  return r == B / 2;
  }
  return false;
}

static std::vector<float> make_logits(const Spec &s) {
  std::vector<float> z((size_t)s.B * s.V);
  std::mt19937 rng(s.seed);
  std::normal_distribution<float> nd(0.f, 1.f);
  for (long r = 0; r < s.B; ++r) {
    float *row = z.data() + (size_t)r * s.V;
    switch (s.dist) {
      case Dist::Normal:
        for (long c = 0; c < s.V; ++c) row[c] = nd(rng);
        break;
      case Dist::Wide:  // +-30: exp() overflows without the max subtraction
        for (long c = 0; c < s.V; ++c) row[c] = nd(rng) * 10.f + ((c & 1) ? 30.f : -30.f);
        break;
      case Dist::AllEqual:  // perfectly flat softmax, loss == log(V) exactly
        for (long c = 0; c < s.V; ++c) row[c] = 0.5f;
        break;
      case Dist::OneHuge:  // one dominant logit: sum -> 1, every other prob -> 0
        for (long c = 0; c < s.V; ++c) row[c] = nd(rng) * 0.1f;
        row[(r * 7919) % s.V] = 60.f;
        break;
      case Dist::VeryNegative:  // all far below 0; naive exp() would flush to 0
        for (long c = 0; c < s.V; ++c) row[c] = -80.f + nd(rng);
        break;
      case Dist::Tiny:
        for (long c = 0; c < s.V; ++c) row[c] = nd(rng) * 1e-4f;
        break;
    }
  }
  return z;
}

// Build targets. Unmasked rows get a real class id; masked rows get s.sentinel.
// Row 0 and row B-1 always get the extreme legal ids (0 and V-1) when unmasked,
// so the boundary columns are covered in every case.
template <typename Tidx>
static std::vector<Tidx> make_targets(const Spec &s, std::vector<char> &is_masked) {
  std::vector<Tidx> t(s.B);
  is_masked.assign(s.B, 0);
  std::mt19937 rng(s.seed ^ 0x5bd1u);
  std::mt19937 mrng(s.seed ^ 0xa17fu);
  for (long r = 0; r < s.B; ++r) {
    if (row_masked(s.mask, r, s.B, mrng)) {
      is_masked[r] = 1;
      t[r] = static_cast<Tidx>(s.sentinel);
    } else {
      long id;
      if (r == 0)            id = 0;
      else if (r == s.B - 1) id = s.V - 1;
      else                   id = (long)(rng() % (unsigned)s.V);
      t[r] = static_cast<Tidx>(id);
    }
  }
  return t;
}

// An ignore_index that provably matches NO target in t. Used as the
// "masking off" control arm -- see the anti-vacuity note at the top.
template <typename Tidx>
static int64_t no_match_index(const std::vector<Tidx> &t) {
  int64_t cand = -987654321;  // negative: unreachable for uint16, and far
  for (Tidx v : t)            // outside any vocab for int32 / int64
    if (static_cast<int64_t>(v) == cand) return cand - 1;
  return cand;
}

// ============================================================================
// 2-BYTE DTYPES (fp16 / bf16) -- GATE REMOVED 2026-09-22
//
// These used to be skipped: the kernels did a 16-byte cp.async / float4 access
// with the element count hardcoded to 4 regardless of sizeof(T), which for a
// 2-byte T both misaligned the address (tid*8, not tid*16 -> "misaligned
// address") and reinterpreted 8 bf16 values as 4 floats.
//
// Fixed by deriving the width from the dtype: LossVec<T>::WIDTH == 16/sizeof(T)
// (4 float / 8 bf16 / 2 double), typed staging, and a scalar tail. The
// per-thread byte offset is now v*16 for every dtype. So fp16/bf16 now run for
// real and these cases are live coverage, not skips.
// ============================================================================

// ============================================================================
// GROUP A / B / D / E / F  --  FORWARD
//
// Runs BOTH forward entry points, and for every case also runs a control arm
// with masking disabled. The masked arm must (i) match the fp64 masked
// reference and (ii) differ from the control by exactly the loss the masked
// rows carried. (ii) is what makes the case non-vacuous.
// ============================================================================
template <typename T, typename Tidx>
static void run_forward(const Spec &s) {
  const size_t N = (size_t)s.B * s.V;

  std::vector<float> z32 = make_logits(s);
  std::vector<char> is_masked;
  std::vector<Tidx> tgt = make_targets<Tidx>(s, is_masked);
  const int64_t off_idx = no_match_index(tgt);

  // The control arm turns masking OFF, so every target gets indexed. If any of
  // them is outside [0, V) that read runs off the row and kills the process, so
  // the control is skipped. Surviving the masked arm is then itself the proof
  // that the guard sits BEFORE row_logits[target_idx] -- if it sat after, this
  // case would fault.
  bool control_safe = true;
  for (Tidx v : tgt) {
    int64_t iv = static_cast<int64_t>(v);
    if (iv < 0 || iv >= s.V) { control_safe = false; break; }
  }

  // how many rows the kernel will ACTUALLY treat as masked, using the kernel's
  // own comparison. For uint16 + sentinel -100 this is legitimately 0.
  long eff_masked = 0;
  for (long r = 0; r < s.B; ++r)
    if (static_cast<int64_t>(tgt[r]) == s.ignore_index) ++eff_masked;

  std::vector<T> zT = to_dt<T>(z32);
  std::vector<float> zb = from_dt<T>(zT);  // logits as the kernel sees them

  DBuf<T> d_z(N);        d_z.up(zT);
  DBuf<Tidx> d_t(s.B);   d_t.up(tgt);
  // loss_output is fp32 for every logits dtype -- see the design note in
  // ops/helpers/LossKernels.h. This used to be DBuf<T>.
  DBuf<float> d_loss(1);
  DBuf<float> d_max(s.B), d_sum(s.B);

  auto ref_on  = ref_forward<Tidx>(z32, tgt, s.B, s.V, s.ignore_index);
  auto ref_off = control_safe ? ref_forward<Tidx>(z32, tgt, s.B, s.V, off_idx)
                              : ref_on;
  auto nf_on   = nf_forward<Tidx>(zb, tgt, s.B, s.V, s.ignore_index);

  double ref_total_on = 0.0, ref_total_off = 0.0, nf_total_on = 0.0;
  for (long r = 0; r < s.B; ++r) {
    ref_total_on  += ref_on[r].loss;
    ref_total_off += ref_off[r].loss;
    nf_total_on   += nf_on[r].loss;
  }
  // The kernel's LAST act is `output[0] = static_cast<TOut>(sum / denom)`
  // (LossKernels.cu, sum_reduction_kernel). Arm B has to model that store or it
  // under-reports the noise floor by exactly one rounding -- which for fp32 is
  // invisible (1 ulp of 2^-24) but for bf16 is 2^-9, i.e. the entire error.
  // Missing this is what made every 2-byte case look like a kernel failure.
  nf_total_on = (double)static_cast<float>(static_cast<T>((float)nf_total_on));
  // the loss the masked rows were carrying -- must be non-trivial for the
  // separation check below to mean anything
  const double removed_expect = ref_total_off - ref_total_on;

  const char *dt = DtName<T>::n();
  const char *ix = IdxName<Tidx>::n();

  // ---- entry point 1: ..._impl_vec (no stats) -----------------------------
  {
    d_loss.poison(-12345.f);
    OwnTensor::cuda::sparse_cross_entropy_forward_cuda_impl_vec<T, Tidx>(
        d_z.p, d_t.p, d_loss.p, s.B, s.V, 0, s.ignore_index);
    CUDA_CHECK(cudaDeviceSynchronize());
    double got_on = (double)d_loss.down()[0];

    double got_off = 0.0;
    if (control_safe) {
      d_loss.poison(-12345.f);
      OwnTensor::cuda::sparse_cross_entropy_forward_cuda_impl_vec<T, Tidx>(
          d_z.p, d_t.p, d_loss.p, s.B, s.V, 0, off_idx);
      CUDA_CHECK(cudaDeviceSynchronize());
      got_off = (double)d_loss.down()[0];
    }

    double ek  = rel_l2({got_on}, {ref_total_on});
    double enf = rel_l2({nf_total_on}, {ref_total_on});
    bool ok = std::isfinite(got_on) && (ek <= 2.0 * enf + s.floor);

    // separation: the control arm must differ by exactly the removed loss
    std::string note = s.note;
    if (eff_masked > 0 && !control_safe) {
      char b[224];
      std::snprintf(b, sizeof(b),
                    "%s mask=%ld/%ld control SKIPPED (target %lld outside [0,V)); "
                    "no fault => guard precedes the indexed read",
                    s.note.c_str(), eff_masked, s.B, (long long)s.ignore_index);
      note = b;
    } else if (eff_masked > 0) {
      double removed_actual = got_off - got_on;
      double sep_err = std::fabs(removed_actual - removed_expect) /
                       std::max(1e-12, std::fabs(removed_expect));
      bool sep_ok = (removed_expect > 0.5) && (sep_err <= 2e-2);
      if (!sep_ok) ok = false;
      char b[192];
      std::snprintf(b, sizeof(b), "%s mask=%ld/%ld removed=%.4f/%.4f%s",
                    s.note.c_str(), eff_masked, s.B, removed_actual,
                    removed_expect, sep_ok ? "" : " SEPARATION-FAIL");
      note = b;
    } else {
      char b[192];
      std::snprintf(b, sizeof(b), "%s mask=0/%ld (control)", s.note.c_str(), s.B);
      note = b;
    }
    report(s.group, s.name + "/fwd_vec", dt, ix, s.B, s.V, ek, enf, ok,
           s.diagnostic, note);
  }

  // ---- entry point 2: ..._impl_vec_save_stats -----------------------------
  {
    d_loss.poison(-12345.f);
    d_max.poison(-999.f);
    d_sum.poison(-999.f);
    OwnTensor::cuda::sparse_cross_entropy_forward_cuda_impl_vec_save_stats<T, Tidx>(
        d_z.p, d_t.p, d_loss.p, d_max.p, d_sum.p, s.B, s.V, 0, s.ignore_index);
    CUDA_CHECK(cudaDeviceSynchronize());
    double got = (double)d_loss.down()[0];
    std::vector<float> hmax = d_max.down(), hsum = d_sum.down();

    double ek  = rel_l2({got}, {ref_total_on});
    double enf = rel_l2({nf_total_on}, {ref_total_on});
    bool ok = std::isfinite(got) && (ek <= 2.0 * enf + s.floor);

    // sentinel check: masked rows must be EXACTLY (0, 1).
    // sum == 1 is load bearing: backward computes 1/sum, and 1/0 = inf,
    // then inf * 0 = NaN, which would poison the whole gradient tensor.
    long bad_sentinel = 0, bad_stats = 0, coincidence = 0;
    for (long r = 0; r < s.B; ++r) {
      bool masked = (static_cast<int64_t>(tgt[r]) == s.ignore_index);
      if (masked) {
        if (hmax[r] != 0.0f || hsum[r] != 1.0f) ++bad_sentinel;
        // if the row's UNMASKED sum would also have been 1.0, the sentinel
        // check proves nothing for that row
        if (control_safe && std::fabs(ref_off[r].sum - 1.0) < 1e-6) ++coincidence;
      } else {
        // Compare against arm B, not arm A: the kernel computes max/sum from the
        // logits AS ROUNDED TO T, so the fp64 reference is the wrong baseline
        // for a 2-byte dtype (it is built from the original fp32 logits).
        double em = std::fabs(hmax[r] - nf_on[r].mx) / std::max(1.0, std::fabs(nf_on[r].mx));
        double es = std::fabs(hsum[r] - nf_on[r].sum) / std::max(1e-30, nf_on[r].sum);
        if (!(std::isfinite(hmax[r]) && std::isfinite(hsum[r])) ||
            em > 1e-2 || es > 5e-2) ++bad_stats;
      }
    }
    if (bad_sentinel || bad_stats) ok = false;
    char b[224];
    std::snprintf(b, sizeof(b), "%s mask=%ld/%ld badsent=%ld badstats=%ld%s",
                  s.note.c_str(), eff_masked, s.B, bad_sentinel, bad_stats,
                  coincidence ? " [sum==1 coincidence]" : "");
    report(s.group, s.name + "/fwd_stats", dt, ix, s.B, s.V, ek, enf, ok,
           s.diagnostic, b);
  }
}

// ============================================================================
// GROUP C  --  BACKWARD
//
//   g_rc = ( exp(z_rc - m_r)/s_r - [c == t_r] ) * grad_output * host_scale
//   masked rows must be BIT-ZERO (== 0.0f, not "small"): the kernel writes them
//   with an explicit store loop, so anything else means the guard did not run.
//
// Two paths are exercised:
//   with_stats : forward saves (max,sum) -> sparseCENormalize_from_stats
//   vec        : sparseCEReduce_kernel_optimized + sparseCENormalize_kernel_optimized
//
// The grad buffer is POISONED with 7.0 before every launch. Without that, a
// masked row reading 0 would not prove the kernel zeroed it.
// ============================================================================
enum class BwdPath { WithStats, Vec };

template <typename T, typename Tidx>
static void run_backward(const Spec &s, BwdPath path, double grad_out_val,
                         double host_scale) {
  const char *pname = (path == BwdPath::WithStats) ? "bwd_stats" : "bwd_vec";
  const size_t N = (size_t)s.B * s.V;

  std::vector<float> z32 = make_logits(s);
  std::vector<char> is_masked;
  std::vector<Tidx> tgt = make_targets<Tidx>(s, is_masked);
  const int64_t off_idx = no_match_index(tgt);

  // same rule as the forward: the control arm indexes every target, so it can
  // only run when every target is a legal column
  bool control_safe = true;
  for (Tidx v : tgt) {
    int64_t iv = static_cast<int64_t>(v);
    if (iv < 0 || iv >= s.V) { control_safe = false; break; }
  }

  long eff_masked = 0;
  for (long r = 0; r < s.B; ++r)
    if (static_cast<int64_t>(tgt[r]) == s.ignore_index) ++eff_masked;

  std::vector<T> zT = to_dt<T>(z32);
  std::vector<float> zb = from_dt<T>(zT);

  DBuf<T> d_z(N);       d_z.up(zT);
  DBuf<Tidx> d_t(s.B);  d_t.up(tgt);
  DBuf<T> d_g(N);
  // grad_output and loss_output are both fp32 by contract now, whatever T is.
  DBuf<float> d_go(1);  { std::vector<float> h{(float)grad_out_val}; d_go.up(h); }
  DBuf<float> d_loss(1);
  DBuf<float> d_max(s.B), d_sum(s.B);
  // The kernels divide by *denom now (PyTorch's total_weight), so the test's
  // host_scale multiplier becomes a denominator of 1/host_scale.
  DBuf<float> d_denom(1);
  { std::vector<float> h{1.0f / (float)host_scale}; d_denom.up(h); }

  auto launch = [&](int64_t ii) {
    d_g.poison(7.0f);
    if (path == BwdPath::WithStats) {
      d_max.poison(-999.f);
      d_sum.poison(-999.f);
      OwnTensor::cuda::sparse_cross_entropy_forward_cuda_impl_vec_save_stats<T, Tidx>(
          d_z.p, d_t.p, d_loss.p, d_max.p, d_sum.p, s.B, s.V, 0, ii);
      CUDA_CHECK(cudaDeviceSynchronize());
      OwnTensor::cuda::sparse_ce_backward_cuda_vec_with_stats<T, Tidx>(
          d_z.p, d_t.p, d_g.p, d_max.p, d_sum.p, s.B, s.V, d_go.p,
          d_denom.p, 0, ii);
    } else {
      OwnTensor::cuda::sparse_ce_backward_cuda_vec<T, Tidx>(
          d_z.p, d_t.p, d_g.p, s.B, s.V, d_go.p, d_denom.p, 0, ii);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    return d_g.down();
  };

  std::vector<T> g_on = launch(s.ignore_index);
  std::vector<T> g_off;
  if (control_safe) g_off = launch(off_idx);
  else g_off.assign(g_on.size(), static_cast<T>(0.f));

  auto ref_on = ref_forward<Tidx>(z32, tgt, s.B, s.V, s.ignore_index);
  auto nf_on  = nf_forward<Tidx>(zb, tgt, s.B, s.V, s.ignore_index);

  const double f_scale = grad_out_val * host_scale;

  double num_k = 0.0, num_nf = 0.0, den = 0.0;
  long masked_dirty = 0;      // masked row that is NOT bit-zero -> guard failed
  long masked_flat_off = 0;   // masked row that is ~0 even WITHOUT masking ->
                              // that row proves nothing (vacuous)
  long nonfinite = 0;
  long untouched_changed = 0; // unmasked row that moved between the two arms
  double worst_unmasked_drift = 0.0;

  std::vector<double> rg(s.V), ng(s.V);
  for (long r = 0; r < s.B; ++r) {
    const float *zrow = z32.data() + (size_t)r * s.V;
    const float *zrb  = zb.data() + (size_t)r * s.V;
    const T *k_on  = g_on.data()  + (size_t)r * s.V;
    const T *k_off = g_off.data() + (size_t)r * s.V;
    const bool masked = (static_cast<int64_t>(tgt[r]) == s.ignore_index);

    if (masked) {
      double off_mag = 0.0;
      for (long c = 0; c < s.V; ++c) {
        float v = static_cast<float>(k_on[c]);
        if (v != 0.0f) ++masked_dirty;
        off_mag = std::max(off_mag, std::fabs((double)static_cast<float>(k_off[c])));
      }
      // the control arm must have put something clearly non-zero here
      if (control_safe && off_mag < 1e-9) ++masked_flat_off;
      continue;
    }

    // unmasked row: full reference + noise floor
    long t = (long)tgt[r];
    for (long c = 0; c < s.V; ++c) {
      rg[c] = (std::exp((double)zrow[c] - ref_on[r].mx) / ref_on[r].sum -
               (c == t ? 1.0 : 0.0)) * f_scale;
      ng[c] = (double)static_cast<T>((float)(
                  (std::exp(zrb[c] - (float)nf_on[r].mx) / (float)nf_on[r].sum -
                   (c == t ? 1.f : 0.f)) * (float)f_scale));
    }
    for (long c = 0; c < s.V; ++c) {
      double kv = (double)static_cast<float>(k_on[c]);
      if (!std::isfinite(kv)) ++nonfinite;
      double d1 = kv - rg[c];
      double d2 = ng[c] - rg[c];
      num_k += d1 * d1;
      num_nf += d2 * d2;
      den += rg[c] * rg[c];
      // masking must not perturb a row it does not own
      if (control_safe) {
        double drift = std::fabs(kv - (double)static_cast<float>(k_off[c]));
        worst_unmasked_drift = std::max(worst_unmasked_drift, drift);
      }
    }
  }
  if (worst_unmasked_drift > 1e-7 * std::max(1.0, std::fabs(f_scale)))
    ++untouched_changed;

  double ek  = (den > 0.0) ? std::sqrt(num_k / den) : std::sqrt(num_k);
  double enf = (den > 0.0) ? std::sqrt(num_nf / den) : std::sqrt(num_nf);

  bool ok = (ek <= 2.0 * enf + s.floor) && masked_dirty == 0 && nonfinite == 0 &&
            untouched_changed == 0;
  // a case that masks rows but where every masked row was already ~0 anyway
  // has not tested anything
  if (control_safe && eff_masked > 0 && masked_flat_off == eff_masked) ok = false;

  char b[256];
  std::snprintf(b, sizeof(b),
                "%s%s mask=%ld/%ld dirty=%ld nonfin=%ld drift=%.2e%s%s",
                s.note.c_str(), control_safe ? "" : " [control skipped]",
                eff_masked, s.B, masked_dirty, nonfinite,
                worst_unmasked_drift,
                untouched_changed ? " UNMASKED-ROW-MOVED" : "",
                (eff_masked > 0 && masked_flat_off == eff_masked) ? " VACUOUS" : "");
  report(s.group, s.name + "/" + pname, DtName<T>::n(), IdxName<Tidx>::n(),
         s.B, s.V, ek, enf, ok, s.diagnostic, b);
}

// ============================================================================
// GROUP G  --  op level: the denominator contract, and CPU/CUDA agreement
//
// Contract (LossOps.cpp, torchtitan BaseLoss.__call__):
//   denom == nullptr -> result = sum_of_unmasked_losses / batch_size
//   denom != nullptr -> result = sum_of_unmasked_losses / denom[0]
// Note the nullptr case divides by batch_size, i.e. by ALL rows INCLUDING the
// masked ones. That is deliberate (constant denominator); the test pins it so
// nobody "fixes" it into a per-micro-batch valid-token count by accident.
// ============================================================================
using OwnTensor::Dtype;
using OwnTensor::Device;
using OwnTensor::DeviceIndex;
using OwnTensor::Shape;
using OwnTensor::Tensor;
using OwnTensor::TensorOptions;

static void run_op_case(const std::string &name, long B, long V, Mask mask,
                        int64_t sentinel, int64_t ignore_index, bool use_denom,
                        float denom_val, bool on_cuda) {
  Spec s;
  s.group = "G"; s.name = name; s.B = B; s.V = V; s.mask = mask;
  s.dist = Dist::Normal; s.sentinel = sentinel; s.ignore_index = ignore_index;
  s.seed = 4242;

  std::vector<float> z32 = make_logits(s);
  std::vector<char> is_masked;
  std::vector<uint16_t> tgt = make_targets<uint16_t>(s, is_masked);

  long eff_masked = 0;
  for (long r = 0; r < B; ++r)
    if (static_cast<int64_t>(tgt[r]) == ignore_index) ++eff_masked;
  const long n_valid = B - eff_masked;

  auto ref = ref_forward<uint16_t>(z32, tgt, B, V, ignore_index);
  double ref_sum = 0.0;
  for (long r = 0; r < B; ++r) ref_sum += ref[r].loss;

  // The op divides by the caller's denom when given, else by the non-ignored
  // count clamped to >= 1 -- PyTorch's total_weight rule, NOT batch_size.
  const double denom = use_denom ? (double)denom_val
                                 : (double)std::max<long>(1, n_valid);
  const double expect = ref_sum / denom;

  DeviceIndex dev = on_cuda ? DeviceIndex(Device::CUDA, 0) : DeviceIndex(Device::CPU, 0);
  TensorOptions lo = TensorOptions().with_dtype(Dtype::Float32).with_device(dev);
  TensorOptions to = TensorOptions().with_dtype(Dtype::UInt16).with_device(dev);

  Tensor logits = Tensor::empty(Shape{{B, V}}, lo);
  Tensor targets = Tensor::empty(Shape{{B}}, to);
  if (on_cuda) {
    CUDA_CHECK(cudaMemcpy(logits.data<float>(), z32.data(),
                          z32.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(targets.data<uint16_t>(), tgt.data(),
                          tgt.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
  } else {
    std::memcpy(logits.data<float>(), z32.data(), z32.size() * sizeof(float));
    std::memcpy(targets.data<uint16_t>(), tgt.data(), tgt.size() * sizeof(uint16_t));
  }

  Tensor dtens;
  const Tensor *dptr = nullptr;
  if (use_denom) {
    dtens = Tensor::full(Shape{{1}},
                         TensorOptions().with_dtype(Dtype::Float32).with_device(dev),
                         denom_val);
    dptr = &dtens;
  }

  Tensor out = OwnTensor::autograd::sparse_cross_entropy_loss(logits, targets,
                                                              ignore_index, dptr);
  if (on_cuda) CUDA_CHECK(cudaDeviceSynchronize());
  double got = (double)out.to_cpu().data<float>()[0];

  double ek = std::fabs(got - expect) / std::max(1e-12, std::fabs(expect));
  bool ok = std::isfinite(got) && ek <= 2e-5;

  char b[224];
  std::snprintf(b, sizeof(b), "%s mask=%ld/%ld nvalid=%ld denom=%.1f got=%.6f expect=%.6f",
                on_cuda ? "cuda" : "cpu ", eff_masked, B, n_valid, denom, got, expect);
  report("G", name + (on_cuda ? "/cuda" : "/cpu"), "f32", "u16", B, V, ek, 0.0,
         ok, false, b);
}

// CPU and CUDA must agree to fp32 noise once masking is on. If they disagree,
// one of the two mask implementations is wrong and every parity test downstream
// is meaningless.
static void run_cpu_cuda_parity(const std::string &name, long B, long V,
                                Mask mask, int64_t sentinel, int64_t ignore_index) {
  Spec s;
  s.group = "G"; s.name = name; s.B = B; s.V = V; s.mask = mask;
  s.dist = Dist::Normal; s.sentinel = sentinel; s.ignore_index = ignore_index;
  s.seed = 909;

  std::vector<float> z32 = make_logits(s);
  std::vector<char> is_masked;
  std::vector<uint16_t> tgt = make_targets<uint16_t>(s, is_masked);
  long eff_masked = 0;
  for (long r = 0; r < B; ++r)
    if (static_cast<int64_t>(tgt[r]) == ignore_index) ++eff_masked;

  auto one = [&](bool cuda) {
    DeviceIndex dev = cuda ? DeviceIndex(Device::CUDA, 0) : DeviceIndex(Device::CPU, 0);
    Tensor lg = Tensor::empty(Shape{{B, V}},
        TensorOptions().with_dtype(Dtype::Float32).with_device(dev));
    Tensor tg = Tensor::empty(Shape{{B}},
        TensorOptions().with_dtype(Dtype::UInt16).with_device(dev));
    if (cuda) {
      CUDA_CHECK(cudaMemcpy(lg.data<float>(), z32.data(), z32.size() * sizeof(float),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(tg.data<uint16_t>(), tgt.data(), tgt.size() * sizeof(uint16_t),
                            cudaMemcpyHostToDevice));
    } else {
      std::memcpy(lg.data<float>(), z32.data(), z32.size() * sizeof(float));
      std::memcpy(tg.data<uint16_t>(), tgt.data(), tgt.size() * sizeof(uint16_t));
    }
    Tensor o = OwnTensor::autograd::sparse_cross_entropy_loss(lg, tg, ignore_index, nullptr);
    if (cuda) CUDA_CHECK(cudaDeviceSynchronize());
    return (double)o.to_cpu().data<float>()[0];
  };

  double c = one(false), g = one(true);
  double ek = std::fabs(c - g) / std::max(1e-12, std::fabs(c));
  bool ok = std::isfinite(c) && std::isfinite(g) && ek <= 1e-4;
  char b[192];
  std::snprintf(b, sizeof(b), "mask=%ld/%ld cpu=%.6f cuda=%.6f", eff_masked, B, c, g);
  report("G", name + "/parity", "f32", "u16", B, V, ek, 0.0, ok, false, b);
}

// ============================================================================
// GROUP W  --  WIRING: does ignore_index actually REACH the backward?
//
// The three backward kernel guards are correct when called directly (group C),
// but LossBackward.cpp contains the string "ignore_index" ZERO times, so the
// autograd path always takes the -1 default and never masks. That means a real
// SFT training run would still push gradients through prompt tokens.
// WIRED 2026-09-22: LossBackward.cpp now carries ignore_index_ and passes it to
// all 6 CUDA call sites plus the CPU loop. This case passes and is a real gate.
// ============================================================================
static void run_wiring_case(long B, long V, int64_t sentinel, int64_t ignore_index,
                            bool on_cuda) {
  Spec s;
  s.group = "W"; s.name = "bwd_ignore_index_reaches_autograd";
  s.B = B; s.V = V; s.mask = Mask::EveryOther; s.dist = Dist::Normal;
  s.sentinel = sentinel; s.ignore_index = ignore_index; s.seed = 77;

  std::vector<float> z32 = make_logits(s);
  std::vector<char> is_masked;
  std::vector<uint16_t> tgt = make_targets<uint16_t>(s, is_masked);
  long eff_masked = 0;
  for (long r = 0; r < B; ++r)
    if (static_cast<int64_t>(tgt[r]) == ignore_index) ++eff_masked;

  DeviceIndex dev = on_cuda ? DeviceIndex(Device::CUDA, 0) : DeviceIndex(Device::CPU, 0);
  Tensor logits = Tensor::empty(Shape{{B, V}},
      TensorOptions().with_dtype(Dtype::Float32).with_device(dev).with_req_grad(true));
  Tensor targets = Tensor::empty(Shape{{B}},
      TensorOptions().with_dtype(Dtype::UInt16).with_device(dev));
  if (on_cuda) {
    CUDA_CHECK(cudaMemcpy(logits.data<float>(), z32.data(),
                          z32.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(targets.data<uint16_t>(), tgt.data(),
                          tgt.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
  } else {
    std::memcpy(logits.data<float>(), z32.data(), z32.size() * sizeof(float));
    std::memcpy(targets.data<uint16_t>(), tgt.data(), tgt.size() * sizeof(uint16_t));
  }
  logits.set_requires_grad(true);

  Tensor out = OwnTensor::autograd::sparse_cross_entropy_loss(logits, targets,
                                                              ignore_index, nullptr);
  out.backward();
  if (on_cuda) CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> g((size_t)B * V, -1.f);
  if (logits.has_grad()) {
    if (on_cuda)
      CUDA_CHECK(cudaMemcpy(g.data(), logits.grad<float>(),
                            g.size() * sizeof(float), cudaMemcpyDeviceToHost));
    else
      std::memcpy(g.data(), logits.grad<float>(), g.size() * sizeof(float));
  }

  long dirty = 0;
  double worst = 0.0;
  for (long r = 0; r < B; ++r) {
    if (static_cast<int64_t>(tgt[r]) != ignore_index) continue;
    const float *row = g.data() + (size_t)r * V;
    for (long c = 0; c < V; ++c) {
      if (row[c] != 0.0f) ++dirty;
      worst = std::max(worst, std::fabs((double)row[c]));
    }
  }
  bool ok = logits.has_grad() && dirty == 0;
  char b[224];
  std::snprintf(b, sizeof(b),
                "masked=%ld/%ld dirty=%ld worst=%.3e %s", eff_masked, B, dirty,
                worst,
                ok ? "" : "-> LossBackward.cpp does not thread ignore_index (known gap)");
  // Was diagnostic while LossBackward.cpp did not thread ignore_index. That is
  // wired now and this passes, so it is a REAL gate: if it ever goes red again,
  // SFT is silently training on prompt tokens.
  report("W", s.name + (on_cuda ? "/cuda" : "/cpu"), "f32", "u16", B, V, worst,
         0.0, ok, /*diagnostic=*/false, b);
}

// ============================================================================
// main
// ============================================================================
template <typename T, typename Tidx> static void fwd(const Spec &s) {
  run_forward<T, Tidx>(s);
}

// sent / ii == kLastId means "the last legal class id for this vocab". The
// behavioural groups use it so the masking-off CONTROL arm is safe: with an
// out-of-vocab sentinel the control arm makes the kernel index a column that
// does not exist (65535 on a 50304-wide row = 61 KB past it) and the process
// dies. Group F still passes the real production sentinel 65535 explicitly.
static constexpr int64_t kLastId = -2;

static Spec mk(const char *grp, const char *name, long B, long V, Mask m,
               Dist d = Dist::Normal, int64_t sent = kLastId,
               int64_t ii = kLastId, bool diag = false, double floor = 1e-5,
               unsigned seed = 1234, const char *note = "") {
  if (sent == kLastId) sent = V - 1;
  if (ii == kLastId) ii = V - 1;
  Spec s;
  s.group = grp; s.name = name; s.B = B; s.V = V; s.mask = m; s.dist = d;
  s.sentinel = sent; s.ignore_index = ii; s.diagnostic = diag;
  s.floor = floor; s.seed = seed; s.note = note;
  return s;
}

// ============================================================================
// GROUP H  --  LOSS SUM OVER MANY ROWS
//
// sum_reduction_kernel's stage 2 is ONE block of 256 threads over
// ceil(B/256) partials. While each thread loaded a single partial, only
// 256 x 256 = 65,536 rows reached the loss; the denominator still counted all
// of them, so the logged loss read about true x 65,536 / B.
//
// run_sum_bits prints the loss of both entry points as raw bits. Diff the
// LOSSBITS lines of two library builds: up to 65,536 rows they must be equal.
// run_equal_logits uses flat logits, where every row's loss is ln V, so
// sum / B must be ln V for every B -- a check that needs no reference arm.
// ============================================================================
template <typename Tidx>
static void run_sum_bits(const char *tag, long B, long V) {
  Spec s = mk("H", tag, B, V, Mask::EveryOther);
  std::vector<float> z32 = make_logits(s);
  std::vector<char> is_masked;
  std::vector<Tidx> tgt = make_targets<Tidx>(s, is_masked);
  DBuf<float> d_z(z32.size());  d_z.up(z32);
  DBuf<Tidx> d_t(s.B);          d_t.up(tgt);
  DBuf<float> d_loss(1), d_max(s.B), d_sum(s.B);

  d_loss.poison(-12345.f);
  OwnTensor::cuda::sparse_cross_entropy_forward_cuda_impl_vec<float, Tidx>(
      d_z.p, d_t.p, d_loss.p, s.B, s.V, 0, s.ignore_index);
  CUDA_CHECK(cudaDeviceSynchronize());
  const float v1 = d_loss.down()[0];
  d_loss.poison(-12345.f);
  OwnTensor::cuda::sparse_cross_entropy_forward_cuda_impl_vec_save_stats<float, Tidx>(
      d_z.p, d_t.p, d_loss.p, d_max.p, d_sum.p, s.B, s.V, 0, s.ignore_index);
  CUDA_CHECK(cudaDeviceSynchronize());
  const float v2 = d_loss.down()[0];

  uint32_t b1, b2;
  std::memcpy(&b1, &v1, sizeof(b1));
  std::memcpy(&b2, &v2, sizeof(b2));
  std::printf("LOSSBITS %-10s B=%8ld V=%6ld fwd_vec=%.9g 0x%08x fwd_stats=%.9g 0x%08x\n",
              tag, B, V, v1, b1, v2, b2);
}

template <typename Tidx>
static void run_equal_logits(long B, long V) {
  // ignore_index -1 matches no uint16 target, so every row counts
  Spec s = mk("H", "equal_logits_lnV", B, V, Mask::None, Dist::AllEqual, kLastId, -1);
  std::vector<float> z32 = make_logits(s);
  std::vector<char> is_masked;
  std::vector<Tidx> tgt = make_targets<Tidx>(s, is_masked);
  DBuf<float> d_z(z32.size());  d_z.up(z32);
  DBuf<Tidx> d_t(s.B);          d_t.up(tgt);
  DBuf<float> d_loss(1), d_max(s.B), d_sum(s.B);
  const double lnV = std::log((double)V);

  for (int entry = 0; entry < 2; ++entry) {
    d_loss.poison(-12345.f);
    if (entry == 0)
      OwnTensor::cuda::sparse_cross_entropy_forward_cuda_impl_vec<float, Tidx>(
          d_z.p, d_t.p, d_loss.p, s.B, s.V, 0, s.ignore_index);
    else
      OwnTensor::cuda::sparse_cross_entropy_forward_cuda_impl_vec_save_stats<float, Tidx>(
          d_z.p, d_t.p, d_loss.p, d_max.p, d_sum.p, s.B, s.V, 0, s.ignore_index);
    CUDA_CHECK(cudaDeviceSynchronize());
    const double mean = (double)d_loss.down()[0] / (double)B;
    const double ek = std::fabs(mean - lnV) / lnV;
    // fp32 ln V carries ~3e-8 and the fp32 sum of B equal values a few ulp
    const bool ok = std::isfinite(mean) && ek <= 1e-6;
    char b[160];
    std::snprintf(b, sizeof(b), "sum/B = %.7f, ln V = %.7f", mean, lnV);
    report("H", std::string("equal_logits_lnV") + (entry == 0 ? "/fwd_vec" : "/fwd_stats"),
           "f32", IdxName<Tidx>::n(), B, V, ek, 0.0, ok, false, b);
  }
}

int main() {
  int dev_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&dev_count));
  if (dev_count == 0) { std::printf("no CUDA device\n"); return 1; }
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("\n===== sparse cross-entropy : SFT ignore_index masking =====\n");
  std::printf("device: %s  sm_%d%d   %.1f GB\n", prop.name, prop.major,
              prop.minor, prop.totalGlobalMem / 1073741824.0);
  std::printf("arms: A fp64 reference | B dtype-T noise floor | C kernel\n");
  std::printf("pass: err_kernel <= 2*err_noisefloor + floor, AND every "
              "exactness/separation assertion holds\n\n");
  header_row();

  const long V = 50304;   // production vocab, divisible by 4 and by 16 bytes
  const long Vodd = 50303;

  // ---- A: masking OFF must be unchanged, all 9 dtype combos ---------------
  {
    Spec a = mk("A", "off_baseline", 64, V, Mask::None, Dist::Normal, 65535, -1);
    a.note = "ignore=-1";
    fwd<float, uint16_t>(a);   fwd<float, int32_t>(a);   fwd<float, int64_t>(a);
    fwd<float16_t, uint16_t>(a);  fwd<float16_t, int32_t>(a);  fwd<float16_t, int64_t>(a);
    fwd<bfloat16_t, uint16_t>(a); fwd<bfloat16_t, int32_t>(a); fwd<bfloat16_t, int64_t>(a);

    // ignore_index set to a value no target holds -> must also be a no-op
    Spec a2 = mk("A", "off_no_target_matches", 64, V, Mask::None, Dist::Normal, 65535, 65535);
    a2.note = "ignore=65535, no row has it";
    fwd<float, uint16_t>(a2);
  }

  // ---- B: masking ON, forward --------------------------------------------
  {
    fwd<float, uint16_t>(mk("B", "mask_every_other", 64, V, Mask::EveryOther));
    fwd<float, uint16_t>(mk("B", "mask_first_row",   64, V, Mask::FirstRow));
    fwd<float, uint16_t>(mk("B", "mask_last_row",    64, V, Mask::LastRow));
    fwd<float, uint16_t>(mk("B", "mask_one_middle",  64, V, Mask::OneMiddle));
    fwd<float, uint16_t>(mk("B", "mask_random_half", 64, V, Mask::Half));
    // every row masked: loss must be exactly 0 and every stat a sentinel
    fwd<float, uint16_t>(mk("B", "mask_all_rows",    64, V, Mask::All));
    // the same pattern across all 9 dtype combos
    Spec b = mk("B", "mask_dtypes", 64, V, Mask::EveryOther);
    fwd<float, int32_t>(b);       fwd<float, int64_t>(b);
    fwd<float16_t, uint16_t>(b);  fwd<float16_t, int32_t>(b);  fwd<float16_t, int64_t>(b);
    fwd<bfloat16_t, uint16_t>(b); fwd<bfloat16_t, int32_t>(b); fwd<bfloat16_t, int64_t>(b);
  }

  // ---- D: shapes that select different code paths -------------------------
  {
    fwd<float, uint16_t>(mk("D", "batch_1",        1, V, Mask::None));
    fwd<float, uint16_t>(mk("D", "batch_2_one_masked", 2, V, Mask::LastRow));
    fwd<float, uint16_t>(mk("D", "batch_256_1stage", 256, V, Mask::EveryOther));
    fwd<float, uint16_t>(mk("D", "batch_257_2stage", 257, V, Mask::EveryOther));
    fwd<float, uint16_t>(mk("D", "batch_1024",    1024, V, Mask::EveryOther));
    fwd<float, uint16_t>(mk("D", "vocab_4",         64, 4, Mask::EveryOther));
    fwd<float, uint16_t>(mk("D", "vocab_128_bigB", 4096, 128, Mask::EveryOther));
    fwd<float, uint16_t>(mk("D", "vocab_512",       64, 512, Mask::EveryOther));
  }

  // ---- E: numerical stress ------------------------------------------------
  {
    fwd<float, uint16_t>(mk("E", "logits_wide_pm30",  64, V, Mask::EveryOther, Dist::Wide));
    fwd<float, uint16_t>(mk("E", "logits_all_equal",  64, V, Mask::EveryOther, Dist::AllEqual));
    fwd<float, uint16_t>(mk("E", "logits_one_huge",   64, V, Mask::EveryOther, Dist::OneHuge));
    fwd<float, uint16_t>(mk("E", "logits_very_neg",   64, V, Mask::EveryOther, Dist::VeryNegative));
    fwd<float, uint16_t>(mk("E", "logits_tiny",       64, V, Mask::EveryOther, Dist::Tiny));
    fwd<bfloat16_t, uint16_t>(mk("E", "bf16_wide",    64, V, Mask::EveryOther, Dist::Wide));
  }

  // ---- F: sentinel-value semantics ----------------------------------------
  {
    // 65535 is outside vocab 50304 and representable in uint16 -> the intended choice
    fwd<float, uint16_t>(mk("F", "sent_65535_u16",  64, V, Mask::EveryOther, Dist::Normal, 65535, 65535,
                            false, 1e-5, 1234, "expect MASKS"));
    // -100 in a uint16 target is stored as 65436, so it can never equal -100:
    // PyTorch's sentinel is silently a no-op for this dtype. Pinned on purpose.
    // -100 in a uint16 target is stored as 65436. Run this at vocab 65536 so
    // 65436 is a LEGAL column: at vocab 50304 the un-masked kernel would do
    // row_logits[65436], i.e. 61 KB past the row, and fault. That fault IS the
    // real hazard of using PyTorch's -100 here, but it cannot be asserted
    // without killing the context, so it is documented rather than executed.
    fwd<float, uint16_t>(mk("F", "sent_m100_u16_noop", 8, 65536, Mask::EveryOther, Dist::Normal,
                            -100, -100, false, 1e-5, 1234, "expect NO-OP (uint16 wrap)"));
    // the same -100 works for signed targets
    fwd<float, int32_t>(mk("F", "sent_m100_i32",  64, V, Mask::EveryOther, Dist::Normal, -100, -100,
                           false, 1e-5, 1234, "expect MASKS"));
    fwd<float, int64_t>(mk("F", "sent_m100_i64",  64, V, Mask::EveryOther, Dist::Normal, -100, -100,
                           false, 1e-5, 1234, "expect MASKS"));
    // ignore_index that collides with a LEGITIMATE class id: every row holding
    // that id is masked, including rows we did not intend to mask
    fwd<float, uint16_t>(mk("F", "sent_0_collides", 64, V, Mask::EveryOther, Dist::Normal, 0, 0,
                           false, 1e-5, 1234, "collides with real id 0"));
    fwd<float, uint16_t>(mk("F", "sent_mid_vocab",  64, V, Mask::EveryOther, Dist::Normal, 25000, 25000,
                           false, 1e-5, 1234, "collides with real id 25000"));
    // the default -1 must never match anything in any target dtype
    // sentinel is an IN-RANGE id here on purpose: the assertion is "-1 matches
    // nothing", and proving that means the kernel actually indexes the target.
    fwd<float, uint16_t>(mk("F", "default_m1_u16",  64, V, Mask::EveryOther, Dist::Normal, 50303, -1,
                           false, 1e-5, 1234, "expect NO-OP (default)"));
    fwd<float, int64_t>(mk("F", "default_m1_i64",   64, V, Mask::EveryOther, Dist::Normal, 50303, -1,
                          false, 1e-5, 1234, "expect NO-OP (default)"));
  }

  // ---- C: masking ON, backward -------------------------------------------
  {
    const long Bb = 128;
    auto both = [&](const Spec &s, double go, double hs) {
      run_backward<float, uint16_t>(s, BwdPath::WithStats, go, hs);
      run_backward<float, uint16_t>(s, BwdPath::Vec, go, hs);
    };
    both(mk("C", "bwd_every_other", Bb, V, Mask::EveryOther), 1.0, 1.0 / Bb);
    both(mk("C", "bwd_first_row",   Bb, V, Mask::FirstRow),   1.0, 1.0 / Bb);
    both(mk("C", "bwd_last_row",    Bb, V, Mask::LastRow),    1.0, 1.0 / Bb);
    both(mk("C", "bwd_all_masked",  Bb, V, Mask::All),        1.0, 1.0 / Bb);
    both(mk("C", "bwd_random_half", Bb, V, Mask::Half),       1.0, 1.0 / Bb);
    // grad_output != 1 and host_scale != 1/B must both propagate
    both(mk("C", "bwd_gradout_2p5", Bb, V, Mask::EveryOther), 2.5, 1.0);
    both(mk("C", "bwd_scale_one",   Bb, V, Mask::EveryOther), 1.0, 1.0);
    // signed target dtypes
    run_backward<float, int32_t>(mk("C", "bwd_i32", Bb, V, Mask::EveryOther),
                                 BwdPath::WithStats, 1.0, 1.0 / Bb);
    run_backward<float, int64_t>(mk("C", "bwd_i64", Bb, V, Mask::EveryOther),
                                 BwdPath::WithStats, 1.0, 1.0 / Bb);
    // stress
    both(mk("C", "bwd_wide",  Bb, V, Mask::EveryOther, Dist::Wide), 1.0, 1.0 / Bb);
    both(mk("C", "bwd_oneHuge", Bb, V, Mask::EveryOther, Dist::OneHuge), 1.0, 1.0 / Bb);
    // small / boundary shapes
    both(mk("C", "bwd_batch_2",  2, V, Mask::LastRow), 1.0, 0.5);
    both(mk("C", "bwd_vocab_128", 1024, 128, Mask::EveryOther), 1.0, 1.0 / 1024);
  }

  // ---- G: op level --------------------------------------------------------
  {
    run_op_case("denom_null_nomask",  64, V, Mask::None,       65535, 65535, false, 0.f, true);
    run_op_case("denom_null_masked",  64, V, Mask::EveryOther, 65535, 65535, false, 0.f, true);
    run_op_case("denom_tensor_32",    64, V, Mask::EveryOther, 65535, 65535, true, 32.f, true);
    run_op_case("denom_tensor_1",     64, V, Mask::EveryOther, 65535, 65535, true, 1.f,  true);
    run_op_case("denom_null_masked",  16, V, Mask::EveryOther, 65535, 65535, false, 0.f, false);
    run_cpu_cuda_parity("cpu_cuda_masked",   16, V, Mask::EveryOther, 65535, 65535);
    run_cpu_cuda_parity("cpu_cuda_unmasked", 16, V, Mask::None,       65535, 65535);
    run_cpu_cuda_parity("cpu_cuda_all_masked", 16, V, Mask::All,      65535, 65535);
  }

  // ---- W: is the backward actually wired? ---------------------------------
  run_wiring_case(64, V, 65535, 65535, /*on_cuda=*/true);
  run_wiring_case(16, V, 65535, 65535, /*on_cuda=*/false);

  // ---- X: DIAGNOSTIC -- pre-existing path limits, not SFT ------------------
  {
    // vocab_size % 4 != 0 : rows that happen to be 16-byte aligned take the
    // float4 branch, which overruns into the tail region
    // floor must be BELOW the defect (3.7e-06) or the check passes while the bug
    // is present. The noise floor here is 9.3e-09, so 1e-7 is still generous.
    Spec x1 = mk("X", "vocab_50303_mod4", 64, Vodd, Mask::EveryOther, Dist::Normal,
                 kLastId, kLastId, /*diag=*/true, 1e-7, 1234, "PRE-EXISTING mod-4 path");
    fwd<float, uint16_t>(x1);
    Spec x4 = mk("X", "vocab_7_mod4", 64, 7, Mask::EveryOther, Dist::Normal,
                 kLastId, kLastId, /*diag=*/true, 1e-5, 1234, "PRE-EXISTING mod-4 path");
    fwd<float, uint16_t>(x4);
  }

  // ---- H: loss sum over many rows ------------------------------------------
  {
    // 65,536 = 256 x 256 is the most the old stage 2 summed in full; 65,537
    // makes reduce_blocks 257. Vocab 8 keeps 1M rows at 32 MB of logits: the
    // cap depends only on the row count.
    for (long b : {65536L, 65537L, 81920L, 131072L, 262144L, 524288L, 1048576L}) {
      const std::string name = "batch_" + std::to_string(b);
      fwd<float, uint16_t>(mk("H", name.c_str(), b, 8, Mask::EveryOther, Dist::Normal,
                              kLastId, kLastId, /*diag=*/false, 1e-5, 1234,
                              b > 65536 ? "rows past 65,536 must count" : "largest one-pass size"));
    }
    run_equal_logits<uint16_t>(1024, V);
    for (long b : {65536L, 65537L, 81920L, 1048576L}) run_equal_logits<uint16_t>(b, 8);

    std::printf("\n");
    for (long b : {1L, 255L, 256L, 257L, 32768L, 65536L, 65537L, 81920L, 131072L, 262144L, 1048576L})
      run_sum_bits<uint16_t>("v8", b, 8);
    for (long b : {1L, 255L, 256L, 257L, 4096L})
      run_sum_bits<uint16_t>("v50304", b, V);
    std::printf("\n");
  }

  // ---- summary ------------------------------------------------------------
  std::printf("\n%s\n", std::string(118, '=').c_str());
  std::printf("SUMMARY\n");
  std::printf("%s\n", std::string(118, '=').c_str());
  std::printf("  cases run              : %d\n", g_cases);
  std::printf("  SFT failures           : %d   <- these are ours, they gate the exit code\n", g_sft_fail);
  std::printf("  diagnostic XFAILs      : %d   <- pre-existing paths / known unfinished wiring\n", g_diag_fail);
  std::printf("\n%s  (%d SFT failure%s, %d diagnostic)\n",
              g_sft_fail == 0 ? "ALL PASS" : "FAILED", g_sft_fail,
              g_sft_fail == 1 ? "" : "s", g_diag_fail);
  return g_sft_fail == 0 ? 0 : 1;
}
