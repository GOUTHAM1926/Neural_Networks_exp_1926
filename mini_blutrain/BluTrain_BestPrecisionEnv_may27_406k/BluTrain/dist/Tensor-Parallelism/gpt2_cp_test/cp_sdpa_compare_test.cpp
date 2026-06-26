// =============================================================================
// CP vs Standard SDPA Numerical Comparison Test
//
// Runs on 2 GPUs with MPI. Computes attention via:
//   (A) Standard path: tril + softmax + matmul (non-CP, full [T x T])
//   (B) CP path: ring attention with SDPAMerger
//
// Compares forward outputs and backward gradients element-by-element
// to pinpoint where CP diverges from the reference.
//
// Build: make cp_sdpa_compare_test
// Run:   mpirun -np 2 ./cp_sdpa_compare_test_exec
// =============================================================================

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <mpi.h>
#include <cuda_runtime.h>

#include "TensorLib.h"
#include "autograd/AutogradOps.h"
#include "autograd/operations/ActivationOps.h"
#include "autograd/operations/MatrixOps.h"
#include "autograd/operations/ReshapeOps.h"
#include "autograd/operations/BinaryOps.h"
#include "ops/UnaryOps/Exponents.h"
#include "ops/UnaryOps/Reduction.h"
#include "ops/TensorOps.h"

// CP includes
#include "tensor/dtensor.h"
#include "tensor/device_mesh.h"
#include "communication/include/ProcessGroupNCCL.h"
#include "gpt2_cp_test/context_parallel/ContextParallel.h"

using namespace OwnTensor;

// Helper: compute max absolute difference between two CPU tensors
float max_abs_diff(const Tensor& a, const Tensor& b) {
    const float* pa = a.data<float>();
    const float* pb = b.data<float>();
    int64_t n = a.numel();
    float maxd = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        maxd = std::max(maxd, std::abs(pa[i] - pb[i]));
    }
    return maxd;
}

// Helper: compute mean absolute difference
float mean_abs_diff(const Tensor& a, const Tensor& b) {
    const float* pa = a.data<float>();
    const float* pb = b.data<float>();
    int64_t n = a.numel();
    float sum = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        sum += std::abs(pa[i] - pb[i]);
    }
    return sum / static_cast<float>(n);
}

// Helper: print first N elements of a CPU tensor
void print_first_n(const Tensor& t, int n, const char* label) {
    const float* p = t.data<float>();
    std::cout << "  " << label << " [first " << n << "]: ";
    for (int i = 0; i < std::min(n, static_cast<int>(t.numel())); ++i) {
        std::cout << std::fixed << std::setprecision(6) << p[i] << " ";
    }
    std::cout << std::endl;
}

// Standard (non-CP) attention forward
// Returns the attention output [B, H, T, D]
Tensor standard_sdpa_forward(
    Tensor& q, Tensor& k, Tensor& v,
    float scale)
{
    // scores = scale * Q @ K^T
    Shape scale_shape({{1}});
    TensorOptions scale_opts = TensorOptions()
        .with_dtype(q.dtype()).with_device(q.device());
    Tensor scale_tensor = Tensor::full(scale_shape, scale_opts, scale);

    Tensor q_scaled = autograd::mul(q, scale_tensor);
    Tensor k_t = autograd::transpose(k, -2, -1);
    Tensor scores = autograd::matmul(q_scaled, k_t);

    // causal mask + softmax (same as gpt2_attn_fixed.cpp)
    float neg_inf = -std::numeric_limits<float>::infinity();
    Tensor masked = autograd::tril(scores, 0, neg_inf);
    Tensor attn_probs = autograd::softmax(masked);

    // output = attn_probs @ v
    Tensor out = autograd::matmul(attn_probs, v);
    return out;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size != 2) {
        if (rank == 0) {
            std::cerr << "This test requires exactly 2 GPUs. Run: mpirun -np 2 ./cp_sdpa_compare_test_exec" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    cudaSetDevice(rank);

    // Setup ProcessGroup
    std::vector<int> ranks_vec = {0, 1};
    DeviceMesh mesh({2}, ranks_vec);
    auto pg = mesh.get_process_group(0);

    DeviceIndex device(Device::CUDA, rank);

    // =========================================================================
    // Test parameters — small enough to inspect, large enough to expose bugs
    // =========================================================================
    const int64_t B = 2, H = 2, T = 8, D = 4;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    Shape qkv_shape({{B, H, T, D}});
    TensorOptions opts = TensorOptions()
        .with_dtype(Dtype::Float32)
        .with_device(device)
        .with_req_grad(true);

    // Create identical Q, K, V on all ranks (same seed)
    Tensor q_full = Tensor::randn<float>(qkv_shape, opts, 100, 0.5f);
    Tensor k_full = Tensor::randn<float>(qkv_shape, opts, 200, 0.5f);
    Tensor v_full = Tensor::randn<float>(qkv_shape, opts, 300, 0.5f);

    if (rank == 0) {
        std::cout << "=== CP vs Standard SDPA Comparison Test ===" << std::endl;
        std::cout << "Config: B=" << B << " H=" << H << " T=" << T << " D=" << D
                  << " scale=" << scale << " world_size=" << world_size << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // =========================================================================
    // TEST 1: FORWARD COMPARISON
    // =========================================================================
    if (rank == 0) {
        std::cout << "\n--- TEST 1: Forward Output Comparison ---" << std::endl;
    }

    // --- (A) Standard SDPA on rank 0 ---
    Tensor std_out;
    if (rank == 0) {
        // Clone so we have separate autograd graphs
        Tensor q_std = q_full.clone(); q_std.set_requires_grad(true);
        Tensor k_std = k_full.clone(); k_std.set_requires_grad(true);
        Tensor v_std = v_full.clone(); v_std.set_requires_grad(true);

        std_out = standard_sdpa_forward(q_std, k_std, v_std, scale);
        std::cout << "[Rank 0] Standard SDPA output shape: ";
        std_out.print_meta();
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // --- (B) CP SDPA on both ranks ---
    Tensor q_cp = q_full.clone(); q_cp.set_requires_grad(true);
    Tensor k_cp = k_full.clone(); k_cp.set_requires_grad(true);
    Tensor v_cp = v_full.clone(); v_cp.set_requires_grad(true);

    ContextParallel cp(mesh, pg, scale, /*is_causal=*/true,
                       RotatorType::AlltoAll, /*load_balance=*/false);

    Tensor cp_out = cp.forward_cp(q_cp, k_cp, v_cp);

    if (rank == 0) {
        std::cout << "[Rank 0] CP SDPA output shape: ";
        cp_out.print_meta();
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // --- Compare forward outputs (on rank 0) ---
    if (rank == 0) {
        Tensor std_out_cpu = std_out.to_cpu();
        Tensor cp_out_cpu = cp_out.to_cpu();

        float fwd_max_diff = max_abs_diff(std_out_cpu, cp_out_cpu);
        float fwd_mean_diff = mean_abs_diff(std_out_cpu, cp_out_cpu);

        std::cout << "\n  Forward max  abs diff: " << std::scientific << fwd_max_diff << std::endl;
        std::cout << "  Forward mean abs diff: " << std::scientific << fwd_mean_diff << std::endl;

        if (fwd_max_diff < 1e-4f) {
            std::cout << "  >>> FORWARD PASS: MATCH <<<" << std::endl;
        } else {
            std::cout << "  >>> FORWARD PASS: MISMATCH <<<" << std::endl;
            print_first_n(std_out_cpu, 8, "std_out");
            print_first_n(cp_out_cpu, 8, "cp_out ");

            // Print per-position diff to find where divergence starts
            const float* s = std_out_cpu.data<float>();
            const float* c = cp_out_cpu.data<float>();
            std::cout << "\n  Per-element diff [first 16]:" << std::endl;
            for (int i = 0; i < std::min(16, (int)std_out_cpu.numel()); ++i) {
                float d = std::abs(s[i] - c[i]);
                if (d > 1e-6f) {
                    std::cout << "    [" << i << "] std=" << s[i] << " cp=" << c[i]
                              << " diff=" << d << std::endl;
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // =========================================================================
    // TEST 2: BACKWARD COMPARISON
    // =========================================================================
    if (rank == 0) {
        std::cout << "\n--- TEST 2: Backward Gradient Comparison ---" << std::endl;
    }

    // --- (A) Standard backward on rank 0 ---
    Tensor std_dq, std_dk, std_dv;
    if (rank == 0) {
        Tensor q_std2 = q_full.clone(); q_std2.set_requires_grad(true);
        Tensor k_std2 = k_full.clone(); k_std2.set_requires_grad(true);
        Tensor v_std2 = v_full.clone(); v_std2.set_requires_grad(true);

        Tensor out2 = standard_sdpa_forward(q_std2, k_std2, v_std2, scale);

        // Use ones-filled upstream gradient (equivalent to sum loss)
        Tensor ones_grad = Tensor::full(out2.shape(), out2.opts().with_req_grad(false), 1.0f);
        out2.backward(&ones_grad);

        if (!q_std2.has_grad()) {
            std::cerr << "[Rank 0] ERROR: Standard backward did not produce q gradient!" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        std_dq = q_std2.grad_view().to_cpu();
        std_dk = k_std2.grad_view().to_cpu();
        std_dv = v_std2.grad_view().to_cpu();

        std::cout << "[Rank 0] Standard backward completed." << std::endl;
        std::cout << "  dQ norm: " << std::sqrt(reduce_sum(std_dq * std_dq, {0,1,2,3}, false).data<float>()[0]) << std::endl;
        std::cout << "  dK norm: " << std::sqrt(reduce_sum(std_dk * std_dk, {0,1,2,3}, false).data<float>()[0]) << std::endl;
        std::cout << "  dV norm: " << std::sqrt(reduce_sum(std_dv * std_dv, {0,1,2,3}, false).data<float>()[0]) << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // --- (B) CP backward on all ranks ---
    {
        Tensor q_cp2 = q_full.clone(); q_cp2.set_requires_grad(true);
        Tensor k_cp2 = k_full.clone(); k_cp2.set_requires_grad(true);
        Tensor v_cp2 = v_full.clone(); v_cp2.set_requires_grad(true);

        ContextParallel cp2(mesh, pg, scale, /*is_causal=*/true,
                            RotatorType::AlltoAll, /*load_balance=*/false);

        Tensor out_cp2 = cp2.forward_cp(q_cp2, k_cp2, v_cp2);

        // Same ones-filled upstream gradient
        Tensor ones_cp = Tensor::full(out_cp2.shape(), out_cp2.opts().with_req_grad(false), 1.0f);
        out_cp2.backward(&ones_cp);

        if (rank == 0) {
            if (!q_cp2.has_grad()) {
                std::cerr << "[Rank 0] ERROR: CP backward did not produce q gradient!" << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            Tensor cp_dq = q_cp2.grad_view().to_cpu();
            Tensor cp_dk = k_cp2.grad_view().to_cpu();
            Tensor cp_dv = v_cp2.grad_view().to_cpu();

            std::cout << "\n[Rank 0] CP backward completed." << std::endl;
            std::cout << "  dQ norm: " << std::sqrt(reduce_sum(cp_dq * cp_dq, {0,1,2,3}, false).data<float>()[0]) << std::endl;
            std::cout << "  dK norm: " << std::sqrt(reduce_sum(cp_dk * cp_dk, {0,1,2,3}, false).data<float>()[0]) << std::endl;
            std::cout << "  dV norm: " << std::sqrt(reduce_sum(cp_dv * cp_dv, {0,1,2,3}, false).data<float>()[0]) << std::endl;

            // Compare against standard
            float dq_max = max_abs_diff(std_dq, cp_dq);
            float dk_max = max_abs_diff(std_dk, cp_dk);
            float dv_max = max_abs_diff(std_dv, cp_dv);
            float dq_mean = mean_abs_diff(std_dq, cp_dq);
            float dk_mean = mean_abs_diff(std_dk, cp_dk);
            float dv_mean = mean_abs_diff(std_dv, cp_dv);

            std::cout << "\n  Gradient comparison (std vs CP):" << std::endl;
            std::cout << "  dQ: max_diff=" << std::scientific << dq_max << " mean_diff=" << dq_mean << std::endl;
            std::cout << "  dK: max_diff=" << std::scientific << dk_max << " mean_diff=" << dk_mean << std::endl;
            std::cout << "  dV: max_diff=" << std::scientific << dv_max << " mean_diff=" << dv_mean << std::endl;

            float tol = 1e-3f;
            bool pass = (dq_max < tol && dk_max < tol && dv_max < tol);
            if (pass) {
                std::cout << "\n  >>> BACKWARD PASS: MATCH (tol=" << tol << ") <<<" << std::endl;
            } else {
                std::cout << "\n  >>> BACKWARD PASS: MISMATCH <<<" << std::endl;

                // Print which gradient diverges most
                if (dq_max >= tol) {
                    std::cout << "\n  dQ diverges — first mismatches:" << std::endl;
                    const float* sq = std_dq.data<float>();
                    const float* cq = cp_dq.data<float>();
                    int printed = 0;
                    for (int i = 0; i < (int)std_dq.numel() && printed < 10; ++i) {
                        float d = std::abs(sq[i] - cq[i]);
                        if (d > 1e-5f) {
                            // Decode index to [b, h, t, d]
                            int idx = i;
                            int d_dim = idx % D; idx /= D;
                            int t_dim = idx % T; idx /= T;
                            int h_dim = idx % H; idx /= H;
                            int b_dim = idx;
                            std::cout << "    [b=" << b_dim << ",h=" << h_dim
                                      << ",t=" << t_dim << ",d=" << d_dim
                                      << "] std=" << sq[i] << " cp=" << cq[i]
                                      << " diff=" << d << std::endl;
                            printed++;
                        }
                    }
                }
                if (dk_max >= tol) {
                    std::cout << "\n  dK diverges — first mismatches:" << std::endl;
                    const float* sk = std_dk.data<float>();
                    const float* ck = cp_dk.data<float>();
                    int printed = 0;
                    for (int i = 0; i < (int)std_dk.numel() && printed < 10; ++i) {
                        float d = std::abs(sk[i] - ck[i]);
                        if (d > 1e-5f) {
                            int idx = i;
                            int d_dim = idx % D; idx /= D;
                            int t_dim = idx % T; idx /= T;
                            int h_dim = idx % H; idx /= H;
                            int b_dim = idx;
                            std::cout << "    [b=" << b_dim << ",h=" << h_dim
                                      << ",t=" << t_dim << ",d=" << d_dim
                                      << "] std=" << sk[i] << " cp=" << ck[i]
                                      << " diff=" << d << std::endl;
                            printed++;
                        }
                    }
                }
                if (dv_max >= tol) {
                    std::cout << "\n  dV diverges — first mismatches:" << std::endl;
                    const float* sv = std_dv.data<float>();
                    const float* cv = cp_dv.data<float>();
                    int printed = 0;
                    for (int i = 0; i < (int)std_dv.numel() && printed < 10; ++i) {
                        float d = std::abs(sv[i] - cv[i]);
                        if (d > 1e-5f) {
                            int idx = i;
                            int d_dim = idx % D; idx /= D;
                            int t_dim = idx % T; idx /= T;
                            int h_dim = idx % H; idx /= H;
                            int b_dim = idx;
                            std::cout << "    [b=" << b_dim << ",h=" << h_dim
                                      << ",t=" << t_dim << ",d=" << d_dim
                                      << "] std=" << sv[i] << " cp=" << cv[i]
                                      << " diff=" << d << std::endl;
                            printed++;
                        }
                    }
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "\n=== Test Complete ===" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
