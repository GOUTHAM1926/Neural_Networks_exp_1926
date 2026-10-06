//======================================================================
// test_pg_watchdog.cpp — proves the ProcessGroupNCCL collective watchdog fires
// on a real rank desynchronisation instead of hanging forever (defect D3).
//
// Induces the exact production failure mode: the ranks disagree on how many
// collectives to issue, so rank 0's last all-reduce waits for a peer that never
// arrives. Before the watchdog this hung until the job's wall clock expired, with
// no output. Now rank 0 must print the diagnostic and exit 99 within roughly
// BLU_NCCL_TIMEOUT_MS.
//
// Run with a SHORT deadline, e.g.
//   BLU_NCCL_TIMEOUT_MS=8000 BLU_NCCL_POLL_MS=1000 mpirun -np 2 ./test_pg_watchdog
//
// Expected: a "NCCL COLLECTIVE WATCHDOG" block naming the stalled collective, and
// a non-zero job exit. This test PASSES BY ABORTING — a clean exit 0 means the
// watchdog did not fire and is a FAILURE.
//======================================================================

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <thread>
#include <chrono>

#include <mpi.h>
#include <cuda_runtime.h>

#include "TensorLib.h"
#include "ProcessGroupNCCL.h"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, W = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &W);

    if (W < 2) {
        if (rank == 0)
            std::printf("[test_pg_watchdog] SKIP: needs at least 2 ranks\n");
        MPI_Finalize();
        return 0;
    }

    int ndev = 0; cudaGetDeviceCount(&ndev);
    cudaSetDevice(rank % (ndev > 0 ? ndev : 1));

    auto pg = init_process_group(W, rank);
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        std::printf("[test_pg_watchdog] watchdog_active=%d — inducing a desync: rank 0 "
                    "issues 2 all-reduces, every other rank issues 1.\n",
                    pg->watchdog_active() ? 1 : 0);
        std::fflush(stdout);
    }

    float* buf = nullptr;
    cudaMalloc(&buf, 1024 * sizeof(float));
    cudaMemset(buf, 0, 1024 * sizeof(float));

    // Collective 1: matched on every rank, completes normally. This also proves the
    // watchdog does NOT false-positive on healthy traffic.
    pg->all_reduce(buf, buf, 1024, OwnTensor::Dtype::Float32, sum, /*sync=*/true);
    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        std::printf("[test_pg_watchdog] first (matched) all-reduce completed — no false "
                    "positive. Now issuing the unmatched one.\n");
        std::fflush(stdout);
    }

    if (rank == 0) {
        // Unmatched: no peer will ever join. The kernel spins on the comm stream, the
        // probe event never completes, and the watchdog must abort this rank.
        pg->all_reduce(buf, buf, 1024, OwnTensor::Dtype::Float32, sum, /*sync=*/false);
        // Idle without touching CUDA: the point is that the WATCHDOG notices, not the
        // main thread. If we reach the end of this loop the watchdog failed.
        for (int i = 0; i < 120; ++i) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        std::printf("[test_pg_watchdog] FAIL: 120 s elapsed and the watchdog never "
                    "fired.\n");
        std::fflush(stdout);
        MPI_Abort(MPI_COMM_WORLD, 1);
    } else {
        // Stay alive so rank 0 is genuinely waiting on a live-but-absent peer rather
        // than seeing a communicator torn down.
        std::this_thread::sleep_for(std::chrono::seconds(120));
    }

    MPI_Finalize();
    return 0;
}
