#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <mpi.h>
#include <nccl.h>



#define CUDA_CHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCL_CHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

class NCCLBenchmark {
public:
    NCCLBenchmark() {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
        
        CUDA_CHECK(cudaSetDevice(rank_));
        
        ncclUniqueId id;
        if (rank_ == 0) ncclGetUniqueId(&id);
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
        
        NCCL_CHECK(ncclCommInitRank(&nccl_comm_, world_size_, id, rank_));
        
        CUDA_CHECK(cudaStreamCreate(&stream_));
    }
    
    ~NCCLBenchmark() {
        ncclCommDestroy(nccl_comm_);
        cudaStreamDestroy(stream_);
    }
    
    void run() {
        if (rank_ == 0) {
    
            std::cout << "NCCL Collectives Benchmark (" << world_size_ << " GPUs)\n";
 
        }
        

        std::vector<size_t> elmts = {
            // 1024 / 4,                // 1 KB
            // 4 * 1024 / 4,            // 4 KB
            // 16 * 1024 / 4,           // 16 KB
            // 64 * 1024 / 4,           // 64 KB
            // 256 * 1024 / 4,          // 256 KB
            // 1024 * 1024 / 4,         // 1 MB
            // 4 * 1024 * 1024 / 4,     // 4 MB
            // 16 * 1024 * 1024 / 4,    // 16 MB
            // 64 * 1024 * 1024 / 4,    // 64 MB
            // 256 * 1024 * 1024 / 4,   // 256 MB
            // 1024 * 1024 * 1024 / 4,  // 1 GB   
            // 1024 * 1024 * 1024 / 2,  // 2 GB
            // 768 * 1024 * 1024,       // 3 GB
            // 1024 * 1024 * 1024,      // 4 GB
            // 2048UL * 1024 * 1024,    // 8 GB
            2688UL * 1024 * 1024     // 10.5 GB

        };
        
    
        warmup();
        

        benchmarkAllReduce(elmts);
        benchmarkReduce(elmts);
        benchmarkBroadcast(elmts);
        benchmarkAllGather(elmts);
        benchmarkGather(elmts);
        benchmarkReduceScatter(elmts);
        benchmarkScatter(elmts);
        benchmarkAlltoAll(elmts);
        benchmarkRSplusAG(elmts);
        
        if (rank_ == 0) {
            std::cout << "Benchmark Complete\n";
        }
    }
    
private:
    int rank_;
    int world_size_;
    ncclComm_t nccl_comm_;
    cudaStream_t stream_;
    
    static constexpr int NUM_ITERATIONS = 10;
    static constexpr int WARMUP_ITERATIONS = 2;
    
    void warmup() {
        long unsigned int size = 1024;
        float* d_data;
        CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(float)));
        
        for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
            NCCL_CHECK(ncclAllReduce(d_data, d_data, size, ncclFloat, ncclSum, nccl_comm_, stream_));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        CUDA_CHECK(cudaFree(d_data));
    }
    
    
    template<typename OpFunc>

    double benchmarkOp(OpFunc op, long unsigned int count, int in_place) {
        float* d_data;
        if(in_place == 0)
            CUDA_CHECK(cudaMalloc(&d_data, count * sizeof(float)));
        else if (in_place == 1)
            CUDA_CHECK(cudaMalloc(&d_data, count * world_size_ *  sizeof(float)));
        // else if(in_place == 2){
        //     CUDA_CHECK(cudaMalloc(&d_recv, cnt * world_size_ * sizeof(float)));      
        //     CUDA_CHECK(cudaMalloc(&d_send, cnt * world_size_ * sizeof(float)));
        // }
        // Warm-up
        // if(in_place == 2){
        //     for (int i = 0; i < 5; ++i) {
        //         op(d_send, d_recv, count);
        //     }
        // }
        if(in_place == 0 || in_place == 1){
            for (int i = 0; i < 5; ++i) {
                op(d_data, count);
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        
        // Benchmark
      
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start,stream_);

        // if(in_place == 2){
        //     for (int i = 0; i < NUM_ITERATIONS; ++i) {
        //         op(d_send, d_recv, count);
        //     }
        // }
        if(in_place == 0 || in_place == 1){
            for (int i = 0; i < NUM_ITERATIONS; ++i) {
                op(d_data, count);
            }
        }
    
        CUDA_CHECK(cudaStreamSynchronize(stream_));
       
        cudaEventRecord(stop,stream_);
        // if( in_place == 2){
        //     CUDA_CHECK(cudaFree(d_send));
        //     CUDA_CHECK(cudaFree(d_recv));
        // }
        // else
        CUDA_CHECK(cudaFree(d_data));
        
        float duration;
        cudaEventElapsedTime(&duration, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return duration / NUM_ITERATIONS;
    }
    
    template<typename OpFunc>
    double benchmarkA2A(OpFunc op, long unsigned int count) {
        float* d_send,* d_recv;
        // if(in_place == 0)
        //     CUDA_CHECK(cudaMalloc(&d_data, count * sizeof(float)));
        // else if (in_place == 1)
        //     CUDA_CHECK(cudaMalloc(&d_data, count * world_size_ *  sizeof(float)));
        // else if(in_place == 2){
        CUDA_CHECK(cudaMalloc(&d_recv, count * world_size_ * sizeof(float)));      
        CUDA_CHECK(cudaMalloc(&d_send, count * world_size_ * sizeof(float)));
        // }
        // Warm-up
        for (int i = 0; i < 5; ++i) {
            op(d_send, d_recv, count);
        }
        // if(in_place == 2){
            
        // }
        // else if(in_place == 0 || in_place == 1){
        //     for (int i = 0; i < 5; ++i) {mbda(float*, float*, long unsigned int)
        //         op(d_data, count);
        //     }
        // }
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        
        // Benchmark
      
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start,stream_);

        for (int i = 0; i < NUM_ITERATIONS; ++i) {
            op(d_send, d_recv, count);
        }
        // if(in_place == 2){
            
        // }
        // else if(in_place == 0 || in_place == 1){
        //     for (int i = 0; i < NUM_ITERATIONS; ++i) {
        //         op(d_data, count);
        //     }
        // }
    
        CUDA_CHECK(cudaStreamSynchronize(stream_));
       
        cudaEventRecord(stop,stream_);
        CUDA_CHECK(cudaFree(d_send));
        CUDA_CHECK(cudaFree(d_recv));
        // if( in_place == 2){

        // }
        // else
        //     CUDA_CHECK(cudaFree(d_data));
        
        float duration;
        cudaEventElapsedTime(&duration, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return duration / NUM_ITERATIONS;
    }

    void benchmarkAllReduce(const std::vector<long unsigned int>& elmts) {
        if (rank_ == 0) {
            std::cout << "--- AllReduce ---\n";
        }
        
        for (long unsigned int count : elmts) {
            auto op = [&](float* d_data, long unsigned int cnt) {
                NCCL_CHECK(ncclAllReduce(d_data, d_data, cnt, ncclFloat, ncclSum, nccl_comm_, stream_));
            };
            
            double latency_ms = benchmarkOp(op, count, 0);
            double size_mb = (count * sizeof(float)) / (1024.0 * 1024.0);
            double bandwidth_gbps = (size_mb * 2 / 1024.0) / (latency_ms / 1000.0);
            
            if (rank_ == 0) {
                printResult(size_mb, latency_ms, bandwidth_gbps);
            }
        }
        
        if (rank_ == 0) std::cout << "\n";
    }
    
    void benchmarkAllGather(const std::vector<long unsigned int>& elmts) {
        if (rank_ == 0) {
            std::cout << "--- AllGather ---\n";
        }
        
        for (long unsigned int count : elmts) {

            // long unsigned int total_size = count * world_size_ * sizeof(float);
            // long unsigned int free_mem, total_mem;
            // cudaMemGetInfo(&free_mem, &total_mem);
            
            // if (total_size > free_mem * 0.8) {  // Use only 80% of free memory
            //     if (rank_ == 0) {
            //         double size_mb = (count * sizeof(float)) / (1024.0 * 1024.0);
            //         std::cout << "Size: " << std::fixed << std::setprecision(2) << std::setw(8) << size_mb 
            //                   << " MB | Skipped (would need " << (total_size / 1024.0 / 1024.0 / 1024.0) 
            //                   << " GB)\n";
            //     }
            //     continue;
            // }
            
            auto op = [&](float* d_data, long unsigned int cnt) {
                // float* d_recv;
                // CUDA_CHECK(cudaMalloc(&d_recv, cnt * ( world_size_ ) * sizeof(float)));
                NCCL_CHECK(ncclAllGather(d_data + rank_ * cnt, d_data, cnt, ncclFloat, nccl_comm_, stream_));
                // CUDA_CHECK(cudaFree(d_recv));
            };
            
            double latency_ms = benchmarkOp(op, count / 2, 1);
            double size_mb = (count / 2 * world_size_ * sizeof(float)) / (1024.0 * 1024.0);
            double bandwidth_gbps = (size_mb / 1024.0) / (latency_ms / 1000.0);
            
            if (rank_ == 0) {
                printResult(size_mb, latency_ms, bandwidth_gbps);
            }
        }
        
        if (rank_ == 0) std::cout << "\n";
    }
    
    void benchmarkReduceScatter(const std::vector<long unsigned int>& elmts) {
        if (rank_ == 0) {
            std::cout << "--- ReduceScatter ---\n";
        }
        
        for (long unsigned int count : elmts) {
            
            long unsigned int recv_count = count / world_size_;
            if (recv_count == 0) continue;  // skip if too smalll
            
            auto op = [&](float* d_data, long unsigned int cnt) {
                // float* d_recv;
                // CUDA_CHECK(cudaMalloc(&d_recv, recv_count * ( world_size_ - 2 ) *  sizeof(float)));
                NCCL_CHECK(ncclReduceScatter(d_data, d_data + rank_ * recv_count, recv_count, ncclFloat, ncclSum, nccl_comm_, stream_));
                // CUDA_CHECK(cudaFree(d_recv));
            };
            
            double latency_ms = benchmarkOp(op, count, 0);
            double size_mb = (count * sizeof(float)) / (1024.0 * 1024.0);
            double bandwidth_gbps = (size_mb / 1024.0) / (latency_ms / 1000.0);
            
            if (rank_ == 0) {
                printResult(size_mb, latency_ms, bandwidth_gbps);
            }
        }
        
        if (rank_ == 0) std::cout << "\n";
    }
    
    void benchmarkBroadcast(const std::vector<long unsigned int>& elmts) {
        if (rank_ == 0) {
            std::cout << "--- Broadcast ---\n";
        }
        
        for (long unsigned int count : elmts) {
            auto op = [&](float* d_data, long unsigned int cnt) {
                NCCL_CHECK(ncclBroadcast(d_data, d_data, cnt, ncclFloat, 0, nccl_comm_, stream_));
            };
            
            double latency_ms = benchmarkOp(op, count, 0);
            double size_mb = (count * sizeof(float)) / (1024.0 * 1024.0);
            double bandwidth_gbps = (size_mb / 1024.0) / (latency_ms / 1000.0);
            
            if (rank_ == 0) {
                printResult(size_mb, latency_ms, bandwidth_gbps);
            }
        }
        
        if (rank_ == 0) std::cout << "\n";
    }
    
    void benchmarkReduce(const std::vector<long unsigned int>& elmts) {
        if (rank_ == 0) {
            std::cout << "--- Reduce ---\n";
        }
        
        for (long unsigned int count : elmts) {
            auto op = [&](float* d_data, long unsigned int cnt) {
                NCCL_CHECK(ncclReduce(d_data, d_data, cnt, ncclFloat, ncclSum, 0, nccl_comm_, stream_));
            };
            
            double latency_ms = benchmarkOp(op, count, 0);
            double size_mb = (count * sizeof(float)) / (1024.0 * 1024.0);
            double bandwidth_gbps = (size_mb / 1024.0) / (latency_ms / 1000.0);
            
            if (rank_ == 0) {
                printResult(size_mb, latency_ms, bandwidth_gbps);
            }
        }
        
        if (rank_ == 0) std::cout << "\n";
    }
    
    // void benchmarkGather(const std::vector<long unsigned int>& elmts) {
    //     if (rank_ == 0) {
    //         std::cout << "--- Gather ---\n";
    //     }
        
    //     for (long unsigned int count : elmts) {
    //         long unsigned int total_size = count * world_size_ * sizeof(float);
    //         long unsigned int free_mem, total_mem;
    //         cudaMemGetInfo(&free_mem, &total_mem);
            
    //         if (total_size > free_mem * 0.8) {
    //             if (rank_ == 0) {
    //                 double size_mb = (count * sizeof(float)) / (1024.0 * 1024.0);
    //                 std::cout << "Size: " << std::fixed << std::setprecision(2) << std::setw(8) << size_mb 
    //                           << " MB | Skipped (would need " << (total_size / 1024.0 / 1024.0 / 1024.0) 
    //                           << " GB)\n";
    //             }
    //             continue;
    //         }
            
    //         auto op = [&](float* d_data, long unsigned int cnt) {
    //             if (rank_ == 0) {
    //                 float* d_recv;
    //                 CUDA_CHECK(cudaMalloc(&d_recv, cnt * world_size_ * sizeof(float)));
                
    //                 CUDA_CHECK(cudaFree(d_recv));
    //             } else {

    //             }
    //             NCCL_CHECK(ncclBroadcast(d_data, d_data, cnt, ncclFloat, 0, nccl_comm_, stream_));
    //         };
            
    //         double latency_ms = benchmarkOp(op, count);
    //         double size_mb = (count * sizeof(float)) / (1024.0 * 1024.0);
    //         double bandwidth_gbps = (size_mb / 1024.0) / (latency_ms / 1000.0);
            
    //         if (rank_ == 0) {
    //             printResult(size_mb, latency_ms, bandwidth_gbps);
    //         }
    //     }
        
    //     if (rank_ == 0) std::cout << "\n";
    // }
        
    void benchmarkGather(const std::vector<size_t>& elmts) {
        if (rank_ == 0) {
            std::cout << "--- Gather ---\n";
        }
        
        for (size_t count : elmts) {
            auto op = [&](float* d_data, size_t cnt) {
                
                    // if (rank_ == 0) {
                    // float* d_recv;
                    // CUDA_CHECK(cudaMalloc(&d_recv, cnt * (world_size_ - 2 ) * sizeof(float)));
                    NCCL_CHECK(ncclGather(d_data + rank_ * cnt, d_data, cnt, ncclFloat, 0, nccl_comm_, stream_));
                    // CUDA_CHECK(cudaFree(d_recv));
                    // }
            //     } else {
    
            //         NCCL_CHECK(ncclGather(d_data, d_data, cnt, ncclFloat, 0, nccl_comm_, stream_));
            //     }
            };
            
            double latency_ms = benchmarkOp(op, count / 2, 1);
            double size_mb = (count / 2 * world_size_ * sizeof(float)) / (1024.0 * 1024.0);
            double bandwidth_gbps = (size_mb / ( 2 * 1024.0) ) / (latency_ms / 1000.0);
            
            if (rank_ == 0) {
                printResult(size_mb, latency_ms, bandwidth_gbps);
            }
        }
        
        if (rank_ == 0) std::cout << "\n";
    }

    // void benchmarkScatter(const std::vector<size_t>& elmts) {
    //     if (rank_ == 0) {
    //         std::cout << "--- Scatter ---\n";
    //     }
        
    //     for (size_t count : elmts) {
    //         auto op = [&](float* d_data, size_t cnt) {
    //             // NCCL doesn't have direct Scatter, use Broadcast as proxy
    //             NCCL_CHECK(ncclBroadcast(d_data, d_data, cnt, ncclFloat, 0, nccl_comm_, stream_));
    //         };
            
    //         double latency_ms = benchmarkOp(op, count);
    //         double size_mb = (count * sizeof(float)) / (1024.0 * 1024.0);
    //         double bandwidth_gbps = (size_mb / 1024.0) / (latency_ms / 1000.0);
            
    //         if (rank_ == 0) {
    //             printResult(size_mb, latency_ms, bandwidth_gbps);
    //         }
    //     }
        
    //     if (rank_ == 0) std::cout << "\n";
    // }
    
    void benchmarkScatter(const std::vector<size_t>& elmts) {
        if (rank_ == 0) {
            std::cout << "--- Scatter ---\n";
        }
        
        for (size_t count : elmts) {
            size_t send_count = count / world_size_;
            auto op = [&](float* d_data, size_t cnt) {
                // if (rank_ == 0) {
                    
                    // float* d_send;
                    // CUDA_CHECK(cudaMalloc(&d_send, cnt * sizeof(float)));
                    NCCL_CHECK(ncclScatter(d_data, d_data + rank_ * send_count, send_count, ncclFloat, 0, nccl_comm_, stream_));
                    // CUDA_CHECK(cudaFree(d_send));
                // }
                // } else {
            
                //     NCCL_CHECK(ncclScatter(d_data, d_data, , ncclFloat, 0, nccl_comm_, stream_));
                // }
            };
            
            double latency_ms = benchmarkOp(op, count, 0);
            double size_mb = (count * sizeof(float)) / (1024.0 * 1024.0);
            double bandwidth_gbps = (size_mb / (2 * 1024.0)) / (latency_ms / 1000.0);
            
            if (rank_ == 0) {
                printResult(size_mb, latency_ms, bandwidth_gbps);
            }
        }
        
        if (rank_ == 0) std::cout << "\n";
    }

//     void benchmarkAlltoAll(const std::vector<size_t>& elmts) {
//         if (rank_ == 0) {
//             std::cout << "--- AlltoAll ---\n";
//         }
        
//         for (size_t count : elmts) {
//             // AlltoAll: each rank sends count/world_size to each other rank
//             size_t chunk_size = count / world_size_;
//             if (chunk_size == 0) continue;
            
//             auto op = [&](float* d_data, size_t cnt) {
//                 // NCCL doesn't have native AlltoAll
//                 // Use AllGather as approximation (similar pattern)
//                 float* d_recv;
//                 CUDA_CHECK(cudaMalloc(&d_recv, cnt * sizeof(float)));
//                 NCCL_CHECK(ncclAllGather(d_data, d_recv, chunk_size, ncclFloat, nccl_comm_, stream_));
//                 CUDA_CHECK(cudaFree(d_recv));
//             };
            
//             double latency_ms = benchmarkOp(op, count);
//             double size_mb = (count * sizeof(float)) / (1024.0 * 1024.0);
//             double bandwidth_gbps = (size_mb / 1024.0) / (latency_ms / 1000.0);
            
//             if (rank_ == 0) {
//                 printResult(size_mb, latency_ms, bandwidth_gbps);
//             }
//         }
        
//         if (rank_ == 0) std::cout << "\n";
//     }

    void benchmarkAlltoAll(const std::vector<size_t>& elmts) {
        if (rank_ == 0) {
            std::cout << "--- AlltoAll ---\n";
        }
        
        for (size_t count : elmts) {
            
            auto op = [&](float* d_send, float* d_recv, size_t cnt) {

                NCCL_CHECK(ncclAlltoAll(d_send, d_recv, cnt  , ncclFloat, nccl_comm_, stream_));
                
            };

            double latency_ms = benchmarkA2A(op, count / 4 );
            double size_mb = (count / 4 * 2 * world_size_ * sizeof(float)) / (1024.0 * 1024.0);
            double bandwidth_gbps = (size_mb / ( 2 * 1024.0) ) / (latency_ms / 1000.0);
            
            if (rank_ == 0) {
                printResult(size_mb, latency_ms, bandwidth_gbps);
            }
        }
        
        if (rank_ == 0) std::cout << "\n";

    }

    void benchmarkRSplusAG(const std::vector<long unsigned int>& elmts) {
        if (rank_ == 0) {
            std::cout << "--- ReduceScatter + AllGather ---\n";
        }
        
        for (long unsigned int count : elmts) {
            
            long unsigned int small_count = count / world_size_;
            if (small_count == 0) continue;  // skip if too smalll
            
            auto op = [&](float* d_data, long unsigned int count) {
         
                NCCL_CHECK(ncclReduceScatter(d_data, d_data + rank_ * small_count, small_count, ncclFloat, ncclSum, nccl_comm_, stream_));
          
                NCCL_CHECK(ncclAllGather(d_data + rank_ * small_count, d_data, small_count, ncclFloat, nccl_comm_, stream_));
              
            };
            
            double latency_ms = benchmarkOp(op, count, 0);
            double size_mb = (count * sizeof(float)) / (1024.0 * 1024.0);
            double bandwidth_gbps = (size_mb * 2 / 1024.0) / (latency_ms / 1000.0);
            
            if (rank_ == 0) {
                printResult(size_mb, latency_ms, bandwidth_gbps);
            }
        }
        
        if (rank_ == 0) std::cout << "\n";
    }
    
    void printResult(double size_mb, double latency_ms, double bandwidth_gbps) {
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Size: " << std::setw(8) << size_mb << " MB | ";
        std::cout << "Latency: " << std::setw(8) << latency_ms << " ms | ";
        std::cout << "Bandwidth: " << std::setw(8) << bandwidth_gbps << " GB/s\n";
    }

};


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    NCCLBenchmark benchmark;
    benchmark.run();
    
    MPI_Finalize();
    return 0;
}


