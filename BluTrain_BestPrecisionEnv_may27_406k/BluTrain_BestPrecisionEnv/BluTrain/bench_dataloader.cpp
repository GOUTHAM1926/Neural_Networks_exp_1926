// =============================================================================
// bench_dataloader.cpp — DataLoader access / H2D timing harness
// =============================================================================
//
// Mirrors the loader setup used by p0_gpt2_fmha_ddp.cpp (one DataLoader per
// rank, one GPU per rank) and reports three numbers per rank:
//
//   1. RAW MMAP READ        : pure CPU memcpy from the mmap'd shard region to
//                             a host buffer (B*T*sizeof(uint16_t)). Isolates
//                             "how fast can this rank pull batch-sized chunks
//                             of shard bytes through the page cache." Run
//                             cold + warm so OS page-cache effects show up.
//
//   2. next_batch() ISSUE   : host wall time of a DataLoader::next_batch()
//                             call. With the async double-buffered loader in
//                             DataLoader.h this is mmap→pinned memcpy plus an
//                             async cudaMemcpyAsync issue — returns before
//                             the H2D actually finishes. Tells us if CPU
//                             staging or shard-advance is stalling the host.
//
//   3. DATA-READY LATENCY   : next_batch() + cudaDeviceSynchronize(). Captures
//                             the async H2D landing on the device (the
//                             loader's cudaStreamWaitEvent has already been
//                             enqueued on stream 0, so syncing the default
//                             stream is enough). This is the number a
//                             consumer kernel would actually wait on.
//
// Reports per rank: count, mean, median (p50), p90, p99, max. Master then
// MPI_Gathers all per-rank means + p99s and prints a table so a slow GPU /
// slow shard / PCIe imbalance is obvious.
//
// Build (matches Makefile run-mpi target):
//     make run-mpi FILE=bench_dataloader.cpp NP=8
//
// Or manually (same flags p0_gpt2_fmha_ddp.cpp uses):
//     mpic++ -std=c++2a -O3 -DWITH_CUDA -ITensor-Implementations/include \
//            -I/usr/local/cuda/include \
//            -LTensor-Implementations/lib -L/usr/local/cuda/lib64 \
//            -ltensor -lcudart bench_dataloader.cpp -o bench_dataloader
//     mpirun -np 8 ./bench_dataloader
//
// Env knobs:
//   BENCH_DATA_ROOT        path to .bin shard dir (default matches the trainer)
//   BENCH_BATCHES          number of next_batch() calls timed per rank (def 200)
//   BENCH_WARMUP           warmup batches not included in stats (def 20)
//   BENCH_B / BENCH_T      micro-batch / seq len (def 16 / 1024 = trainer)
//   BENCH_SPLIT            "train" or "val" (def train)
// =============================================================================

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>

#include "TensorLib.h"
#include "DataLoader.h"

using clock_hr = std::chrono::high_resolution_clock;

static inline double us_since(const clock_hr::time_point& t0) {
    return std::chrono::duration<double, std::micro>(clock_hr::now() - t0).count();
}

static int env_int(const char* k, int def) {
    if (const char* v = std::getenv(k)) return std::atoi(v);
    return def;
}
static std::string env_str(const char* k, const char* def) {
    if (const char* v = std::getenv(k)) return std::string(v);
    return std::string(def);
}

struct Stats {
    double mean = 0, p50 = 0, p90 = 0, p99 = 0, max = 0, min = 0;
    size_t n = 0;
};

static Stats summarize(std::vector<double>& xs) {
    Stats s;
    s.n = xs.size();
    if (xs.empty()) return s;
    std::sort(xs.begin(), xs.end());
    double sum = 0;
    for (double x : xs) sum += x;
    s.mean = sum / xs.size();
    auto pick = [&](double q) {
        size_t i = static_cast<size_t>(q * (xs.size() - 1));
        return xs[i];
    };
    s.p50 = pick(0.50);
    s.p90 = pick(0.90);
    s.p99 = pick(0.99);
    s.max = xs.back();
    s.min = xs.front();
    return s;
}

static void print_stats(int rank, const char* label, const Stats& s, const char* unit) {
    std::cout << "[rank " << rank << "] " << std::setw(28) << std::left << label
              << " n=" << std::setw(5) << s.n
              << " mean=" << std::fixed << std::setprecision(2) << std::setw(8) << s.mean
              << " p50=" << std::setw(8) << s.p50
              << " p90=" << std::setw(8) << s.p90
              << " p99=" << std::setw(8) << s.p99
              << " max=" << std::setw(8) << s.max
              << " " << unit << std::endl;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    const bool is_master = (rank == 0);

    // One GPU per rank, same convention as the trainer.
    cudaSetDevice(rank);

    const int B           = env_int("BENCH_B", 16);
    const int T           = env_int("BENCH_T", 1024);
    const int n_batches   = env_int("BENCH_BATCHES", 200);
    const int warmup      = env_int("BENCH_WARMUP", 20);
    const std::string split     = env_str("BENCH_SPLIT", "train");
    const std::string data_root = env_str("BENCH_DATA_ROOT",
                                          "/mnt/volgrp03/3rd_floor/edu_fineweb10B_bin");

    if (is_master) {
        std::cout << "==== DataLoader bench ====\n"
                  << "  world_size : " << world << "\n"
                  << "  B          : " << B << "\n"
                  << "  T          : " << T << "\n"
                  << "  bytes/batch: " << (size_t)B * T * sizeof(u_int16_t)
                  << " (" << ((double)B * T * sizeof(u_int16_t) / 1024.0) << " KiB)\n"
                  << "  batches    : " << n_batches << " (after " << warmup << " warmup)\n"
                  << "  split      : " << split << "\n"
                  << "  data_root  : " << data_root << "\n"
                  << "==========================" << std::endl;
    }

    const size_t BT = static_cast<size_t>(B) * static_cast<size_t>(T);
    const size_t bytes = BT * sizeof(u_int16_t);

    // -----------------------------------------------------------------------
    // Phase 1: raw mmap → host memcpy. Open one shard directly via
    // UInt16ShardView (it's public in DataLoader.h) and time pure CPU reads.
    // Run cold first (forces page-ins) then warm (page cache). Cold gives the
    // worst-case bytes-from-disk number; warm tells us how fast the loader's
    // mmap memcpy can run once pages are resident.
    // -----------------------------------------------------------------------
    {
        std::vector<std::string> shards = list_shards(data_root, split, ".bin");
        if (shards.empty()) {
            if (is_master)
                std::cerr << "[FATAL] no .bin shards for split='" << split
                          << "' under " << data_root << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Each rank reads a different shard (round-robin) so the kernel cache
        // doesn't make every rank's "cold" pass actually warm.
        const std::string& path = shards[rank % shards.size()];
        UInt16ShardView view;
        view.open(path, 100000000);

        std::vector<u_int16_t> host_buf(BT);
        const u_int16_t* src = view.data_ptr();
        const size_t max_off = view.size_tokens() - BT - 1;

        // Stride through the shard so successive reads hit different pages.
        // Random would be ideal but cheap LCG is enough to defeat trivial
        // prefetch and matches the loader's batch_stride pattern.
        std::vector<double> cold_us, warm_us;
        cold_us.reserve(n_batches);
        warm_us.reserve(n_batches);

        // --- Cold pass: drop_caches isn't available without root, but striding
        // far enough between reads still surfaces page faults if the shard is
        // larger than RAM. On systems where everything fits in cache, cold ≈ warm
        // and that's a real signal too (means I/O isn't the bottleneck).
        size_t stride_tokens = BT * static_cast<size_t>(world);
        size_t off = (rank * BT) % max_off;
        for (int i = 0; i < n_batches; ++i) {
            auto t0 = clock_hr::now();
            std::memcpy(host_buf.data(), src + off, bytes);
            cold_us.push_back(us_since(t0));
            off += stride_tokens;
            if (off > max_off) off = (rank * BT) % max_off;
        }

        // --- Warm pass: reread the same region we just touched. Now strictly
        // from page cache (and CPU caches if BT is small enough), so this is
        // the *floor* on the loader's CPU-side cost.
        off = (rank * BT) % max_off;
        for (int i = 0; i < n_batches; ++i) {
            auto t0 = clock_hr::now();
            std::memcpy(host_buf.data(), src + off, bytes);
            warm_us.push_back(us_since(t0));
            off += stride_tokens;
            if (off > max_off) off = (rank * BT) % max_off;
        }

        // Drop warmup window from each pass.
        if ((int)cold_us.size() > warmup) cold_us.erase(cold_us.begin(), cold_us.begin() + warmup);
        if ((int)warm_us.size() > warmup) warm_us.erase(warm_us.begin(), warm_us.begin() + warmup);

        Stats cs = summarize(cold_us);
        Stats ws = summarize(warm_us);

        // Serialize per-rank prints so the table is readable under mpirun.
        for (int r = 0; r < world; ++r) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (r == rank) {
                print_stats(rank, "mmap CPU memcpy COLD",  cs, "us");
                print_stats(rank, "mmap CPU memcpy WARM",  ws, "us");
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (is_master) std::cout << std::endl;

    // -----------------------------------------------------------------------
    // Phase 2: the real DataLoader. Two timers per call:
    //   issue_us : host wall time of next_batch() (mmap→pinned + async H2D
    //              ENQUEUE; returns before the GPU has the data).
    //   ready_us : issue_us + cudaDeviceSynchronize. The bench can't peek at
    //              the loader's private copy_stream, but the loader records
    //              cudaStreamWaitEvent on the default stream before
    //              returning, so syncing the default stream waits on the
    //              H2D-done event. That gives a tight bound on data-ready
    //              latency from a consumer's POV.
    //
    // If issue_us << ready_us in steady state, the H2D is the long pole and
    // the prefetch is hiding it — good. If issue_us ≈ ready_us, the host is
    // spending real time inside next_batch() (likely mmap→pinned memcpy on
    // a cold shard, or a shard advance), and the prefetch isn't winning.
    // -----------------------------------------------------------------------
    {
        OwnTensor::DeviceIndex dev(OwnTensor::Device::CUDA, rank);
        DataLoaderLite loader(B, T, rank, world, split, data_root,
                              /*master_process=*/is_master,
                              /*max_tokens_per_shard=*/100000000,
                              /*device_idx=*/rank);

        std::vector<double> issue_us, ready_us;
        issue_us.reserve(n_batches);
        ready_us.reserve(n_batches);

        for (int i = 0; i < n_batches; ++i) {
            auto t0 = clock_hr::now();
            Batch b = loader.next_batch();
            double issue = us_since(t0);

            // Pop a small dependent kernel onto the default stream so the
            // event-wait actually has something to gate. Without this the
            // cudaStreamWaitEvent is a no-op from the consumer's perspective
            // and the timing degenerates to "issue_us + sync overhead".
            // We just need any default-stream work that reads b.input — a
            // tiny device→host pull of 1 element is enough.
            // (Skipped: the .to_cpu() round-trip would dominate. The
            // cudaDeviceSynchronize below is sufficient for ready-time since
            // it drains the H2D regardless.)
            cudaDeviceSynchronize();
            double ready = us_since(t0);

            issue_us.push_back(issue);
            ready_us.push_back(ready);
        }

        if ((int)issue_us.size() > warmup) issue_us.erase(issue_us.begin(), issue_us.begin() + warmup);
        if ((int)ready_us.size() > warmup) ready_us.erase(ready_us.begin(), ready_us.begin() + warmup);

        Stats is_ = summarize(issue_us);
        Stats rs = summarize(ready_us);

        for (int r = 0; r < world; ++r) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (r == rank) {
                print_stats(rank, "next_batch ISSUE (host)", is_, "us");
                print_stats(rank, "next_batch DATA-READY",   rs,  "us");
            }
        }

        // ---- Gather means + p99s for the cross-rank summary table. ----
        double my_issue_mean = is_.mean,  my_issue_p99 = is_.p99;
        double my_ready_mean = rs.mean,  my_ready_p99 = rs.p99;
        std::vector<double> all_issue_mean(world), all_issue_p99(world);
        std::vector<double> all_ready_mean(world), all_ready_p99(world);
        MPI_Gather(&my_issue_mean, 1, MPI_DOUBLE, all_issue_mean.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(&my_issue_p99,  1, MPI_DOUBLE, all_issue_p99.data(),  1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(&my_ready_mean, 1, MPI_DOUBLE, all_ready_mean.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(&my_ready_p99,  1, MPI_DOUBLE, all_ready_p99.data(),  1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (is_master) {
            std::cout << "\n==== Cross-rank summary (us) ====\n";
            std::cout << std::setw(6)  << "rank"
                      << std::setw(14) << "issue_mean"
                      << std::setw(14) << "issue_p99"
                      << std::setw(14) << "ready_mean"
                      << std::setw(14) << "ready_p99"
                      << std::setw(14) << "hidden(R-I)"
                      << std::endl;
            double min_ready = 1e18, max_ready = 0;
            for (int r = 0; r < world; ++r) {
                std::cout << std::setw(6) << r
                          << std::setw(14) << std::fixed << std::setprecision(2) << all_issue_mean[r]
                          << std::setw(14) << all_issue_p99[r]
                          << std::setw(14) << all_ready_mean[r]
                          << std::setw(14) << all_ready_p99[r]
                          << std::setw(14) << (all_ready_mean[r] - all_issue_mean[r])
                          << std::endl;
                min_ready = std::min(min_ready, all_ready_mean[r]);
                max_ready = std::max(max_ready, all_ready_mean[r]);
            }
            double skew_us  = max_ready - min_ready;
            double skew_pct = (min_ready > 0) ? (100.0 * skew_us / min_ready) : 0.0;
            std::cout << "\n  data-ready spread across ranks: "
                      << std::fixed << std::setprecision(2)
                      << skew_us << " us (" << skew_pct << "% of fastest)\n"
                      << "  hidden(R-I) ~= async H2D cost that the loader's "
                      << "prefetch is masking on the host.\n"
                      << "  If issue_mean dominates ready_mean - issue_mean, "
                      << "CPU staging is the long pole.\n"
                      << "  If a single rank's ready_mean is >>20% above the "
                      << "fastest, suspect that GPU's PCIe link / NUMA pin.\n";
        }
    }

    // -----------------------------------------------------------------------
    // Phase 3: bandwidth sanity. One large pinned→device cudaMemcpyAsync per
    // rank on a fresh stream, timed with cudaEvents — gives MB/s per PCIe
    // link without the loader in the picture, so any imbalance can be
    // attributed cleanly. Size = 32 batches' worth to amortise launch
    // overhead but stay small enough that the timing isn't dominated by
    // the GPU side of the bus.
    // -----------------------------------------------------------------------
    {
        const size_t bw_bytes = bytes * 32;
        void *pinned = nullptr, *dev = nullptr;
        cudaError_t e1 = cudaHostAlloc(&pinned, bw_bytes, cudaHostAllocDefault);
        cudaError_t e2 = cudaMalloc(&dev, bw_bytes);
        if (e1 != cudaSuccess || e2 != cudaSuccess) {
            if (is_master) std::cerr << "[bw] alloc failed, skipping\n";
        } else {
            std::memset(pinned, 0, bw_bytes);
            cudaStream_t s; cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
            cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);

            // Warmup
            for (int i = 0; i < 5; ++i) {
                cudaMemcpyAsync(dev, pinned, bw_bytes, cudaMemcpyHostToDevice, s);
            }
            cudaStreamSynchronize(s);

            const int reps = 50;
            cudaEventRecord(a, s);
            for (int i = 0; i < reps; ++i) {
                cudaMemcpyAsync(dev, pinned, bw_bytes, cudaMemcpyHostToDevice, s);
            }
            cudaEventRecord(b, s);
            cudaEventSynchronize(b);
            float ms = 0; cudaEventElapsedTime(&ms, a, b);
            double gb_per_s = ((double)bw_bytes * reps) / (ms / 1000.0) / 1e9;

            for (int r = 0; r < world; ++r) {
                MPI_Barrier(MPI_COMM_WORLD);
                if (r == rank) {
                    std::cout << "[rank " << rank << "] pinned->device H2D bw : "
                              << std::fixed << std::setprecision(2) << gb_per_s
                              << " GB/s  (" << bw_bytes / (1024 * 1024)
                              << " MiB x " << reps << " reps, "
                              << std::setprecision(3) << (ms / reps) << " ms/rep)"
                              << std::endl;
                }
            }

            cudaEventDestroy(a); cudaEventDestroy(b);
            cudaStreamDestroy(s);
            cudaFree(dev); cudaFreeHost(pinned);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
