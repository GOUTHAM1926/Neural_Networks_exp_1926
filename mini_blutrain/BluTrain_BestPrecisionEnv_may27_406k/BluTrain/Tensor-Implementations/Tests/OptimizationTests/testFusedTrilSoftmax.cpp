#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <string>
#include <random>

#include "TensorLib.h"
#include "ops/FusedKernels.cuh"

using namespace OwnTensor;

static const float NEG_INF = -std::numeric_limits<float>::infinity();

// =============================================================================
// CPU Reference: fused tril + softmax on a [B*H, W] flat layout.
//   local_row = global_row % H  →  same per-batch masking the kernel uses.
// =============================================================================
static void cpu_fused_tril_softmax(
    const std::vector<float>& in,
    std::vector<float>&       out,
    int64_t H, int64_t cols,
    int64_t trilDiag, float mask_val)
{
  int64_t rows = static_cast<int64_t>(in.size()) / cols;
  out.assign(rows * cols, 0.0f);

  for (int64_t row = 0; row < rows; ++row) {
    int64_t local_row = row % H;

    // Step 1 – apply tril mask, find row max for numerical stability
    float max_val = NEG_INF;
    for (int64_t col = 0; col < cols; ++col) {
      float v = (col > local_row + trilDiag) ? mask_val : in[row * cols + col];
      out[row * cols + col] = v;
      if (v > max_val) max_val = v;
    }

    // Step 2 – exp(v - max) and partial sum
    float sum = 0.0f;
    for (int64_t col = 0; col < cols; ++col) {
      float e = std::exp(out[row * cols + col] - max_val);
      out[row * cols + col] = e;
      sum += e;
    }

    // Step 3 – normalize
    for (int64_t col = 0; col < cols; ++col)
      out[row * cols + col] /= sum;
  }
}

// =============================================================================
// Helpers
// =============================================================================
static bool check_close(
    const std::vector<float>& ref,
    const std::vector<float>& got,
    float threshold,
    const std::string& label)
{
  bool    pass      = true;
  int64_t mismatches = 0;

  for (size_t i = 0; i < ref.size(); ++i) {
    // Both NaN is acceptable: an all-masked row gives 0/0 = NaN on both sides.
    if (std::isnan(ref[i]) && std::isnan(got[i])) continue;

    float diff = std::abs(ref[i] - got[i]);
    if (std::isnan(got[i]) || diff > threshold) {
      if (mismatches < 5)
        printf("    [%zu]  ref=%.7f  got=%.7f  diff=%.2e\n",
               i, ref[i], got[i], (double)diff);
      ++mismatches;
      pass = false;
    }
  }

  printf("  %-35s %s", label.c_str(),
         pass ? "\033[32mPASS\033[0m" : "\033[31mFAIL\033[0m");
  if (!pass)
    printf("  (%ld / %zu differ by > %.0e)",
           mismatches, ref.size(), (double)threshold);
  printf("\n");
  return pass;
}

// Check that every non-NaN row in `data` sums to 1.0 within `threshold`.
static bool check_row_sums(
    const std::vector<float>& data,
    int64_t rows, int64_t cols,
    float threshold,
    const std::string& label)
{
  bool    pass = true;
  int64_t bad  = 0;

  for (int64_t r = 0; r < rows; ++r) {
    float sum     = 0.0f;
    bool  all_nan = true;
    for (int64_t c = 0; c < cols; ++c) {
      float v = data[r * cols + c];
      if (!std::isnan(v)) { sum += v; all_nan = false; }
    }
    if (all_nan) continue;  // fully-masked row → NaN expected

    if (std::abs(sum - 1.0f) > threshold) {
      if (bad < 3) printf("    row[%ld]: sum=%.7f\n", r, sum);
      ++bad;
      pass = false;
    }
  }

  printf("  %-35s %s", label.c_str(),
         pass ? "\033[32mPASS\033[0m" : "\033[31mFAIL\033[0m");
  if (!pass) printf("  (%ld rows have |sum - 1| > %.0e)", bad, (double)threshold);
  printf("\n");
  return pass;
}

// Run the GPU kernel and return the host output.
static std::vector<float> run_gpu(
    const std::vector<float>& h_in,
    int64_t H, int64_t W,
    int64_t rows, int64_t cols,
    int64_t trilDiag, float mask_val)
{
  TensorOptions opts =
      TensorOptions().with_dtype(Dtype::Float32).with_device(Device::CUDA);

  // Shape must encode H in shape[-2] so the kernel derives the correct
  // per-batch tril row index via  local_row = global_row % H.
  // Shape{{rows, cols}} would set H = rows (wrong for multi-batch inputs).
  int64_t B   = rows / H;
  Tensor d_in = Tensor::empty(Shape{{B, H, W}}, opts);
  d_in.set_data(h_in);

  Tensor d_out   = fused_tril_softmax(d_in, trilDiag, static_cast<double>(mask_val));
  Tensor cpu_out = d_out.to_cpu();
  float* ptr     = cpu_out.data<float>();
  return std::vector<float>(ptr, ptr + static_cast<size_t>(rows * cols));
}

// =============================================================================
// Test cases
// =============================================================================

// Accuracy test: compare GPU output to CPU reference element-wise.
static bool test_accuracy(
    const std::string& name,
    int B, int64_t H, int64_t W,
    int64_t trilDiag, float mask_val,
    float threshold = 1e-5f,
    uint64_t seed   = 42)
{
  printf("\n[Accuracy] %s  B=%d H=%ld W=%ld trilDiag=%ld\n",
         name.c_str(), B, H, W, trilDiag);

  int64_t rows = static_cast<int64_t>(B) * H;
  int64_t cols = W;
  size_t  n    = static_cast<size_t>(rows * cols);

  std::vector<float> h_in(n);
  {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (auto& v : h_in) v = dist(rng);
  }

  std::vector<float> cpu_ref;
  cpu_fused_tril_softmax(h_in, cpu_ref, H, cols, trilDiag, mask_val);

  std::vector<float> gpu_out = run_gpu(h_in, H, W, rows, cols, trilDiag, mask_val);

  return check_close(cpu_ref, gpu_out, threshold, "vs CPU ref");
}

// Row-sum invariant: every output row (after softmax) must sum to 1.
static bool test_row_sum(
    const std::string& name,
    int B, int64_t H, int64_t W,
    int64_t trilDiag, float mask_val,
    float threshold = 1e-5f)
{
  printf("\n[RowSum]   %s  B=%d H=%ld W=%ld trilDiag=%ld\n",
         name.c_str(), B, H, W, trilDiag);

  int64_t rows = static_cast<int64_t>(B) * H;
  int64_t cols = W;
  size_t  n    = static_cast<size_t>(rows * cols);

  std::vector<float> h_in(n);
  {
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (auto& v : h_in) v = dist(rng);
  }

  std::vector<float> gpu_out = run_gpu(h_in, H, W, rows, cols, trilDiag, mask_val);
  return check_row_sums(gpu_out, rows, cols, threshold, "row sums ≈ 1.0");
}

// Spot-check: first row of every batch has exactly 1 valid element (col 0).
// After softmax that element must equal 1.0.
static bool test_first_row_is_one(int B, int64_t H, int64_t W)
{
  printf("\n[Spot]     First-row-of-batch = 1.0  B=%d H=%ld W=%ld\n", B, H, W);

  int64_t rows = static_cast<int64_t>(B) * H;
  int64_t cols = W;
  size_t  n    = static_cast<size_t>(rows * cols);

  std::vector<float> h_in(n);
  {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (auto& v : h_in) v = dist(rng);
  }

  std::vector<float> gpu_out = run_gpu(h_in, H, W, rows, cols, 0, NEG_INF);

  bool pass = true;
  for (int b = 0; b < B; ++b) {
    int64_t row = static_cast<int64_t>(b) * H;  // first row of each batch item
    float   v   = gpu_out[row * cols + 0];       // col 0 is the only valid element
    if (std::abs(v - 1.0f) > 1e-5f) {
      printf("    batch[%d] row[%ld][0] = %.7f  (expected 1.0)\n", b, row, v);
      pass = false;
    }
    // All other cols in this row must be ≈ 0
    for (int64_t c = 1; c < cols; ++c) {
      float vc = gpu_out[row * cols + c];
      if (!std::isnan(vc) && std::abs(vc) > 1e-5f) {
        printf("    batch[%d] row[%ld][%ld] = %.7f  (expected 0)\n", b, row, c, vc);
        pass = false;
      }
    }
  }

  printf("  %-35s %s\n", "col-0 == 1.0, rest == 0",
         pass ? "\033[32mPASS\033[0m" : "\033[31mFAIL\033[0m");
  return pass;
}

// =============================================================================
// main
// =============================================================================
int main()
{
  printf("============================================================\n");
  printf("   Fused Tril + Softmax Kernel — Correctness Tests\n");
  printf("   Threshold: 1e-5 (relaxed to 1e-4 for large dims)\n");
  printf("============================================================\n");

  int failures = 0;

  // ── Accuracy (GPU vs CPU reference) ──────────────────────────────────────
  printf("\n=== Accuracy Tests ===\n");

  // Basic single batch
  if (!test_accuracy("Single batch 4x4",         1,  4,  4,  0, NEG_INF)) ++failures;

  // Multiple batches – verifies local_row = global_row % H is correct
  if (!test_accuracy("4 batches of 4x4",          4,  4,  4,  0, NEG_INF)) ++failures;
  if (!test_accuracy("8 batches of 8x8",          8,  8,  8,  0, NEG_INF)) ++failures;

  // Positive diagonal: more elements kept per row
  if (!test_accuracy("DiagOffset +1 (B=2, 4x4)", 2,  4,  4,  1, NEG_INF)) ++failures;

  // Negative diagonal: local row 0 is fully masked → NaN row on both sides
  if (!test_accuracy("DiagOffset -1 (B=2, 4x4)", 2,  4,  4, -1, NEG_INF)) ++failures;

  // Larger square sub-matrices
  if (!test_accuracy("Square 16x16 (B=4)",        4, 16, 16,  0, NEG_INF)) ++failures;

  // Attention-like dims (relaxed threshold for float rounding in large reductions)
  if (!test_accuracy("Attn B=8 T=64",             8, 64, 64,  0, NEG_INF, 1e-4f)) ++failures;
  if (!test_accuracy("Attn B=8 T=128",            8,128,128,  0, NEG_INF, 1e-4f)) ++failures;

  // cols not a power of 2 – exercises block-stride boundary handling
  if (!test_accuracy("Non-pow2 W=48 (B=3)",       3, 48, 48,  0, NEG_INF)) ++failures;

  // All elements masked (trilDiag very negative) → every row is all-NaN
  if (!test_accuracy("All-masked (diag=-100)",    1,  4,  4,-100, NEG_INF)) ++failures;

  // ── Row-sum invariant ─────────────────────────────────────────────────────
  printf("\n=== Row Sum Invariant (softmax rows must sum to 1.0) ===\n");

  if (!test_row_sum("Basic B=1 H=8 W=8",          1,  8,  8,  0, NEG_INF)) ++failures;
  if (!test_row_sum("Batched B=4 H=32 W=32",      4, 32, 32,  0, NEG_INF)) ++failures;
  if (!test_row_sum("DiagOffset +1 B=2 H=8 W=8",  2,  8,  8,  1, NEG_INF)) ++failures;
  if (!test_row_sum("Attn B=8 T=64",              8, 64, 64,  0, NEG_INF, 1e-4f)) ++failures;

  // ── Spot checks ───────────────────────────────────────────────────────────
  printf("\n=== Spot Checks ===\n");

  // First row of every batch has only col-0 valid → softmax = 1.0 there
  if (!test_first_row_is_one(4,  8,  8)) ++failures;
  if (!test_first_row_is_one(8, 16, 16)) ++failures;

  // ── Summary ───────────────────────────────────────────────────────────────
  printf("\n============================================================\n");
  printf("  %s  (%d failure%s)\n",
         failures == 0 ? "\033[32mALL TESTS PASSED\033[0m"
                       : "\033[31mSOME TESTS FAILED\033[0m",
         failures, failures == 1 ? "" : "s");
  printf("============================================================\n");
  return failures > 0 ? 1 : 0;
}
