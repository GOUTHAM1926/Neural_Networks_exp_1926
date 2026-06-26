#include "tensor/dtensor.h"
#include <iostream>
#include <vector>
#include <nccl.h>
#include <nvtx3/nvtx3.hpp>
#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvToolsExtCuda.h>


#define CUDA_CHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    std::shared_ptr<Work> work_obj;
   
    DeviceMesh device_mesh ({2}, {0,1});

    auto pg = device_mesh.get_process_group(0);

    int num_devices = 0;
    cudaGetDeviceCount(&num_devices);
    if (num_devices > 0) {
        CUDA_CHECK(cudaSetDevice(rank % num_devices));
    }

    cudaStream_t comm_stream;
    cudaStreamCreate(&comm_stream);

    for (int t = 499; t < 500; t++){

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float duration;
    float avgduration;
    
    const int64_t B = 8;      // batch size
    const int64_t C = 2 * t;      // input features
    const int64_t F = 4 * t;      // token length
    
    // auto pg = device_mesh.get_process_group(0);

 

    // Allocate tensors BEFORE timing starts
    
    Layout w1_layout(device_mesh, { B , C , F }, 2);
    DTensor W1(device_mesh, pg, w1_layout);
    
    // Only root initializes with random data
    if (rank == 0) {
        W1.rand();
        // W1.display();
    }
    
    Layout W1_asS_layout(device_mesh,{ B , C , F/2 });
    DTensor W1_Shard(device_mesh, pg, W1_asS_layout);
    
    // Warmup run (not timed)
     for (int i = 0; i < 10; i++ ){
      // std::string message = "Start of iteration: ";
    // nvtxRangePush(message.c_str());  
    W1_Shard.shard_fused_transpose( 2 , 0 , W1 );
    // nvtxRangePop();
    } 

    cudaDeviceSynchronize();
    cudaEventRecord(start,comm_stream);
    
    for (int i = 0; i < 100; i++ ){
      W1_Shard.shard_fused_transpose( 2 , 0 , W1 );
    }
    
    CUDA_CHECK(cudaStreamSynchronize(comm_stream));
    cudaEventRecord(stop,comm_stream);

    CUDA_CHECK(cudaEventSynchronize(stop));
    
    // W1_Shard.display();

    cudaEventElapsedTime(&duration, start, stop);

    duration /= 100;

    cudaDeviceSynchronize();

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    int64_t size = ( B * C * F * 4 ) / ( 1024 * 1024 );
    int64_t throughput = size / (duration / 1000) ;

    if(rank == 0){
      std::cout<<"\n\n ========= BENCHMARKS "<< t <<" ========= \n\n"<<std::endl;

      std::cout<<" DURATION : "<< duration <<std::endl;
      std::cout<<" SIZE : "<< size <<std::endl;
      std::cout<<" THROUGHPUT : "<< throughput <<std::endl;    
      
      std::cout<<" B : "<< B <<std::endl;
      // std::cout<<" T : "<< T <<std::endl;
      std::cout<<" C : "<< C <<std::endl;
      std::cout<<" F : "<< F <<std::endl;
    }

  }
  MPI_Finalize();
  return 0;
}