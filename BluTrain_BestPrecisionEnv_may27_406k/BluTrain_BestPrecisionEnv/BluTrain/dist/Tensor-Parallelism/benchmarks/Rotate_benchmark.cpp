

#include "tensor/dtensor.h"
#include <iostream>
#include <vector>
#include <string>
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

    for(int t = 499 ; t < 500; t++) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float duration;

    
    // const int64_t B = 8;      // batch size
    // const int64_t C = 768;      // input features
    // const int64_t T = 1024;      // token length
    // const int64_t F = 768*4;     // hidden dim (will be sharded: F / P per GPU)

  
    const int64_t B = 8;      // batch size
    const int64_t C = 2 * t;      // input features
    // const int64_t T = 4 * t;      // token length
    const int64_t F = 4 * t;     // hidden dim (will be sharded: F / P per GPU

    // Allocate tensors BEFORE timing starts
    Layout w1_layout(device_mesh, {B ,C ,F }, 1);
    DTensor W1(device_mesh, pg, w1_layout);
    
  
    if(rank == 0 )W1.rand();  // Initialize once
    
    Layout W1_asS_layout(device_mesh,{B , C/2 , F});
    DTensor W1_Shard(device_mesh, pg, W1_asS_layout);
  
    // if(rank == 0) W1.display();
    for (int i = 0; i < 10; i++ ){
    //   // Use optimized shard_fused_transpose
      W1_Shard.shard_fused_transpose(2, 0, W1);  
    }

    // CUDA_CHECK(cudaEventSynchronize(stop));

    
    // nvtxRangePush("Start of rotation");
    cudaEventRecord(start,comm_stream);
    // for (int i = 0; i < 100; i++ ){
        // Use optimized shard_fused_transpose
        // std::string message = "Start of iteration: " + std::to_string(i);


    // std::string message = "Start of iteration: ";
    // nvtxRangePush(message.c_str());

    W1_Shard.shard_fused_transpose(1, 0, W1); 
        
    // nvtxRangePop(); 
    
    // }
    cudaEventRecord(stop,comm_stream);
    CUDA_CHECK(cudaEventSynchronize(stop));
    // CUDA_CHECK(cudaStreamSynchronize(comm_stream));
    // nvtxRangePop();
    
    cudaEventElapsedTime(&duration, start, stop);

    
    cudaDeviceSynchronize();
      
    // W1_Shard.display();
    
    
    // duration /= 100;
    

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
    // Layout grad_layout (device_mesh, {B, T, C });

    // DTensor grad_Y(device_mesh, pg, grad_layout);

    // grad_Y.rand();
    
    // if (rank == 0) {
        
    //     std::cout << "\n=== Before sync() ===" << std::endl;
    //     grad_Y.display();

    // }
    
    // grad_Y.sync(); // sum gradients
    
    // if (rank == 0) {
    //     std::cout << "\n=== After sync() - Gradients added ===" << std::endl;

    //        grad_Y.display();

    // }
    


    MPI_Finalize();
    return 0;
}


