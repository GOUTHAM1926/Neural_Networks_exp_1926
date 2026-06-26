#include<iostream>
#include<chrono>
#include<cuda_runtime.h>
#include<cstring>
void check_memory_type(void* ptr){
  cudaPointerAttributes attrs;
  cudaError_t err = cudaPointerGetAttributes(&attrs,ptr);
  std::cout<<"Memory Status : ";
  if (err != cudaSuccess || attrs.type == cudaMemoryTypeUnregistered){
    // If CUDA throws error OR says type=0 (Unregistered), it is standard Pageable.
    std::cout<<"[Pageable] (Reason: ";
    if(err!=cudaSuccess) std::cout << cudaGetErrorName(err);
    else std::cout << "Type=" << attrs.type << " Unregistered";
    std::cout << ")\n";
    
    cudaGetLastError(); //clear the error flag.
  }else{
      // If Type=1 (Host) or Type=2 (Device), it is Known/Pinned.
      std::cout<<"[Pinned/Registered] (CUDA attributes found! Type=" << attrs.type <<")\n";
  }
}
int main(){
//     std::cout<<"Memcpy tests\n";
//     size_t size=1024*1024*500; //500MiB
//     //1. Allocate source and destination memory on heap (pageable)
//     uint8_t* h_src = new uint8_t[size];
//     uint8_t* h_dst = new uint8_t[size];
//     //Initialize to force physical allocation(OS Logic)
//     std::memset(h_src,1,size);
//     std::memset(h_dst,0,size);
//     //checking by prinitng first 10 values.
//     for(int i=0;i<10;i++){
//       //std::cout<<+h_src[i]<<"\n";
//     std::cout<<+h_dst[i]<<"\n";   //+h_dst or +h_src is used to convert that char(uint8_t value is considered as char(ASCII)) into integer
//      }
// //2.Memcpy (CPU --->CPU)
// auto start = std::chrono::high_resolution_clock::now();
// //Standard  C++ memcpy 
// std::memcpy(h_dst,h_src,size);
// auto end = std::chrono::high_resolution_clock::now();
// // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
// //std::cout<<"CPU--->CPU Time(Memcpy 500 MiB) :" << duration.count() << "ms\n";
// double duration = std::chrono::duration<double,std::milli>(end-start).count();
// std::cout<<"CPU --->CPU (Memcpy - 500MiB) time :"<<duration<<"ms\n";
// //Bandwidth (GB/s)[ (size/duration)--->(Bytes/ms) --->convert to GiB/sec ]
// double bandwidth = ((size/(1024.0*1024.0*1024.0))/(duration/1000.0));// [GiB/sec]
// std::cout<<"Bandwidth(GiB/sec) :" <<bandwidth<<"\n";
// //checking whether they got copied correctly or not.
// for(int i=0;i<10;i++){
//   //std::cout<<+h_src[i]<<"\n";
//   std::cout<<+h_dst[i]<<"\n"; //+h_dst or +h_src is used to convert that char(uint8_t value is considered as char(ASCII)) into integer
     
// }
// check_memory_type(h_src);
// check_memory_type(h_dst);
// std::cin.get();
//delete[] h_src;
//delete[] h_dst;
//std::cout<<"pageable memory freed\n";
// void* src = nullptr;
// cudaError_t err = cudaMallocHost(&src,size);
// if (err!=cudaSuccess){
//   std::cout<<"Failed to allocate pinned memory : "<<cudaGetErrorString(err)<<"\n";

// }else{
//   std::cout<<"Success! Pinned Memory allocated at address : "<< src<<"\n";
//   check_memory_type(src);
//   std::cin.get();
//   //cudaFreeHost(src);
//   //std::cout<<"Memory freed\n";
// }
//cudaFreeHost(src);
// delete[] h_src;
// delete[] h_dst;
// std::cout<<"Memory Freed\n";




// --- CPU -> CPU VARIATIONS ---
 size_t copy_size = 1024 * 1024 * 500; // 500 MB (Renamed to avoid conflict)

// // Allocate 4 Buffers
// uint8_t* h_pg1 = new uint8_t[copy_size]; // Pageable 1
// uint8_t* h_pg2 = new uint8_t[copy_size]; // Pageable 2
// uint8_t* h_pin1 = nullptr; cudaMallocHost(&h_pin1, copy_size); // Pinned 1
// uint8_t* h_pin2 = nullptr; cudaMallocHost(&h_pin2, copy_size); // Pinned 2

// // Initialize
// std::memset(h_pg1, 1, copy_size);
// std::memset(h_pg2, 0, copy_size);
// std::memset(h_pin1, 2, copy_size);
// std::memset(h_pin2, 0, copy_size);

// // Lambda to benchmark (Fixed Syntax!)
// auto test_cpucpu = [&](const char* name, void* dst, void* src) {
//     auto s = std::chrono::high_resolution_clock::now();
//     std::memcpy(dst, src, copy_size);
//     auto e = std::chrono::high_resolution_clock::now();
    
//     double ms = std::chrono::duration<double, std::milli>(e - s).count();
//     double gb = (copy_size / (1024.0 * 1024.0 * 1024.0)) / (ms / 1000.0);
//     std::cout << name << ": " << ms << " ms | " << gb << " GiB/s\n";
// };

// std::cout << "\n--- CPU -> CPU Variations ---\n";
// test_cpucpu("1. Pageable -> Pageable", h_pg2, h_pg1);
// test_cpucpu("2. Pageable -> Pinned  ", h_pin1, h_pg1);
// test_cpucpu("3. Pinned   -> Pageable", h_pg2, h_pin2);
// test_cpucpu("4. Pinned   -> Pinned  ", h_pin1, h_pin2);

// // Cleanup
// delete[] h_pg1; delete[] h_pg2;
// cudaFreeHost(h_pin1); cudaFreeHost(h_pin2);

// CPU to  CPU (pageable to pinned)

// uint8_t* h_pg1= new uint8_t[copy_size];
// uint8_t* h_pin1=nullptr;
// cudaMallocHost(&h_pin1,copy_size);
// std::memset(h_pg1,1,copy_size);
// std::memset(h_pin1,0,copy_size);
// for(int i =0;i<10;i++){
//   //std::cout<<+h_pg1[i]<<"\n";
//   std::cout<<+h_pin1[i]<<"\n"; //+h_dst or +h_src is used to convert that char(uint8_t value is considered as char(ASCII)) into integer
// }
// check_memory_type(h_pg1);
// check_memory_type(h_pin1);
// auto start = std::chrono::high_resolution_clock::now();
// std::memcpy(h_pin1,h_pg1,copy_size);
// auto end= std::chrono::high_resolution_clock::now();
// double duration = std::chrono::duration<double,std::milli>(end-start).count();
// std::cout<<"CPU to CPU (pageable memory to pinned memory memcpy) time (500MiB) :" <<duration<<"ms\n";
// double bandwidth = ((copy_size/(1024.0*1024.0*1024.0))/(duration/1000.0));
// std::cout<<"Bandwidth(GiB/s) :"<<bandwidth<<"\n";
// //Checking whether the values are copied or not by printing first 10 bytes.
// for (int i=0;i<10;i++){
// //std::cout<<+h_pg1[i]<<"\n";
// std::cout<<+h_pin1[i]<<"\n"; //+h_dst or +h_src is used to convert that char(uint8_t value is considered as char(ASCII)) into integer
// }
// std::cin.get();
// delete[] h_pg1;
// cudaFreeHost(h_pin1);
// std::cin.get();


// // CPU to CPU (pinned to pageable)
// uint8_t* h_pin1=nullptr;
// cudaMallocHost(&h_pin1,copy_size);
// uint8_t* h_pg1 = new uint8_t[copy_size];
// std::memset(h_pin1,1,copy_size);
// std::memset(h_pg1,0,copy_size);
// for(int i=0;i<10;i++){
//   std::cout<<+h_pin1[i]<<"\n";
//   std::cout<<+h_pg1[i]<<"\n";  //+h_dst or +h_src is used to convert that char(uint8_t value is considered as char(ASCII)) into integer
// }
// check_memory_type(h_pin1);
// check_memory_type(h_pg1);
// auto start = std::chrono::high_resolution_clock::now();
// std::memcpy(h_pg1,h_pin1,copy_size);
// auto end= std::chrono::high_resolution_clock::now();
// double duration = std::chrono::duration<double,std::milli>(end-start).count();
// std::cout<<"CPU to CPU (pinned memory to pageable memory memcpy) time (500 MiB) :"<<duration<<"ms\n";
// double bandwidth =((copy_size/(1024.0*1024.0*1024.0))/(duration/1000.0));
// std::cout<<"Bandwidth(GiB/s) :"<<bandwidth<<"\n";
// for(int i=0;i<10;i++){
//   std::cout<<+h_pin1[i]<<"\n";
//   std::cout<<+h_pg1[i]<<"\n"; //+h_dst or +h_src is used to convert that char(uint8_t value is considered as char(ASCII)) into integer.
// }
// std::cin.get();
// cudaFreeHost(h_pin1);
// delete[] h_pg1;
// std::cin.get();


// // CPU to  CPU (pinned to pinned) 
// uint8_t* h_pin1=nullptr;
// cudaMallocHost(&h_pin1,copy_size);
// uint8_t* h_pin2=nullptr;
// cudaMallocHost(&h_pin2,copy_size);
// std::memset(h_pin1,0,copy_size);
// std::memset(h_pin2,1,copy_size);
// for(int i=0;i<10;i++){
//   std::cout<<+h_pin1[i]<<"\n";
//   std::cout<<+h_pin2[i]<<"\n"; //+h_dst or +h_src is used to convert that char(uint8_t value is considered as char(ASCII)) into integer.
// }
// check_memory_type(h_pin1);
// check_memory_type(h_pin2);
// auto start = std::chrono::high_resolution_clock::now();
// std::memcpy(h_pin2,h_pin1,copy_size);
// auto end = std::chrono::high_resolution_clock::now();
// double duration = std::chrono::duration<double,std::milli>(end-start).count();
// std::cout<< "CPU to CPU (pinned to pinned) time (500 MiB) :"<<duration<< "ms\n";
// double bandwidth = ((copy_size/(1024.0*1024.0*1024.0))/(duration/1000.0));
// std::cout<<"Bandwidth(GiB/s) :"<<bandwidth<<"\n";
// for(int i=0;i<10;i++){
//   std::cout<<+h_pin1[i]<<"\n";
//   std::cout<<+h_pin2[i]<<"\n"; //+h_dst or +h_src is used to convert that char(uint8_t value is considered as char(ASCII)) into integer.
// }
// std::cin.get();
// cudaFreeHost(h_pin1);
// cudaFreeHost(h_pin2);
// std::cin.get();


//CPU to GPU 
// //1. Pageable_ptr (CPU) to GPU (destination ptr) using cudaMemcpy (synchronous+2 copy)
// uint8_t* h_pg=new uint8_t[copy_size];
// uint8_t* d_ptr=nullptr;
// cudaMalloc(&d_ptr,copy_size);
// std::memset(h_pg,1,copy_size);
// cudaMemset(d_ptr,0,copy_size); //GPU memory via Driver 

// // for(int i=0;i<10;i++){
// //   std::cout<<+h_pg[i]<<"\n"; //+h_pg is used to convert that char(uint8_t value is considered as char(ASCII)) into integer.
// //   //to print the value of the GPU memory we need to copy it to the CPU memory and then print it.
// //    uint8_t temp_val=0;
// //    cudaMemcpy(&temp_val,d_ptr+i,1,cudaMemcpyDeviceToHost); //copies 1 byte as specified by 3rd parameter.
// //    std::cout<<+temp_val<<"\n"; 
// // }
// check_memory_type(h_pg);
// check_memory_type(d_ptr);
// auto start =std::chrono::high_resolution_clock::now();
// cudaMemcpy(d_ptr,h_pg,copy_size,cudaMemcpyHostToDevice);
// auto end = std::chrono::high_resolution_clock::now();
// double duration = std::chrono::duration<double,std::milli>(end-start).count();
// std::cout<<"CPU to GPU (pageable memory to GPU destination memory) time (500 MiB) :"<<duration<<"ms\n";
// double bandwidth = ((copy_size/(1024.0*1024.0*1024.0))/(duration/1000.0));
// std::cout<<"Bandwidth :" <<bandwidth<<" GiB/s is the speed of (CPU Memory copy + PCIe transfer+)"<<"\n";
// // for(int i=0;i<10;i++){
// //   std::cout<<+h_pg[i]<<"\n"; //+h_pg is used to convert that char(uint8_t value is considered as char(ASCII)) into integer.
// //   //to print the value of the GPU memory we need to copy it to the CPU memory and then print it.
// //    uint8_t temp_val=0;
// //    cudaMemcpy(&temp_val,d_ptr+i,1,cudaMemcpyDeviceToHost); //copies 1 byte as specified by 3rd parameter.
// //    std::cout<<+temp_val<<"\n"; 

// // }
// std::cin.get();
// delete[] h_pg;
// cudaFree(d_ptr);

//2.Pinned (CPU) to (GPU) using cudaMemcpy (synchronous+1 copy)(direct DMA ,no staging)
// uint8_t* h_pin=nullptr;
// cudaMallocHost(&h_pin,copy_size);
// uint8_t* d_ptr=nullptr;
// cudaMalloc(&d_ptr,copy_size);
// std::memset(h_pin,1,copy_size);
// cudaMemset(d_ptr,0,copy_size);
// check_memory_type(h_pin);
// check_memory_type(d_ptr);
// for(int i=0;i<10;i++){
//   std::cout<<+h_pin[i]<<"\n";
//   uint8_t temp_val=0;
//   cudaMemcpy(&temp_val,d_ptr+1,1,cudaMemcpyDeviceToHost);
//   std::cout<<+temp_val<<"\n";
// }
// auto start = std::chrono::high_resolution_clock::now();
// cudaMemcpy(d_ptr,h_pin,copy_size,cudaMemcpyHostToDevice);
// auto end = std::chrono::high_resolution_clock::now();
// double duration = std::chrono::duration<double,std::milli>(end-start).count();
// std::cout<<"CPU to GPU (pinned memory to GPU memory) time (500 MiB) :"<<duration<<"ms\n";
// double bandwidth = ((copy_size/(1024.0*1024.0*1024.0))/(duration/1000.0));
// std::cout<<"Bandwidth :" <<bandwidth<<" GiB/s is the speed of ( PCIe transfer+GPU copy)" <<"\n";
// for(int i=0;i<10;i++){
//   std::cout<<+h_pin[i]<<"\n";
//   uint8_t temp_val=0;
//   cudaMemcpy(&temp_val,d_ptr+i,1,cudaMemcpyDeviceToHost);
//   std::cout<<+temp_val<<"\n";
// }
// std::cin.get();
// cudaFreeHost(h_pin);
// cudaFree(d_ptr);

// //3.GPU to CPU (pageable memory to GPU destination memory) using cudaMemcpyAsync (still behaves like synchronous in this case)
// uint8_t* h_pg=new uint8_t[copy_size];
// uint8_t* d_ptr=nullptr;
// cudaMalloc(&d_ptr,copy_size);
// std::memset(h_pg,1,copy_size);
// cudaMemset(d_ptr,0,copy_size);
// check_memory_type(h_pg);
// check_memory_type(d_ptr);
// for(int i=0;i<10;i++){
//   std::cout<<+h_pg[i]<<"\n";
//   uint8_t temp_val=0;
//   cudaMemcpy(&temp_val,d_ptr+i,1,cudaMemcpyDeviceToHost);
//   std::cout<<+temp_val<<"\n";
// }
// auto start = std::chrono::high_resolution_clock::now();
// cudaMemcpyAsync(d_ptr,h_pg,copy_size,cudaMemcpyHostToDevice);
// //cudaDeviceSynchronize();  //to make sure that the copy is complete before the next copy ,so that confirming this async function is behaving synchronously when pageable memory is copied form cpu(host) to device(gpu). ,first run ihtout his line commenting this ,then run with this line,if time comes to be same and large like cudaMemcpy+pageable memory copy time,then its behaving synchronously.
// auto end = std::chrono::high_resolution_clock::now();
// double duration = std::chrono::duration<double,std::milli>(end-start).count();
// std::cout<<"GPU to CPU (pageable memory to GPU destination memory) time (500 MiB) :"<<duration<<"ms\n";
// double bandwidth = ((copy_size/(1024.0*1024.0*1024.0))/(duration/1000.0));
// std::cout<<"Bandwidth :" <<bandwidth<<" GiB/s is the speed of (CPU copies+ PCIe transfer+GPU copies)" <<"\n";
// for(int i=0;i<10;i++){
//   std::cout<<+h_pg[i]<<"\n";
//   uint8_t temp_val=0;
//   cudaMemcpy(&temp_val,d_ptr+i,1,cudaMemcpyDeviceToHost);
//   std::cout<<+temp_val<<"\n";
// }
// std::cin.get();
// delete[] h_pg;
// cudaFree(d_ptr);

// //4. CPU to GPU (pinned memory to GPU destination memory) using cudaMemcpyAsync (asynchronous)
// uint8_t* h_pin =nullptr;
// cudaMallocHost(&h_pin,copy_size);
// uint8_t* d_ptr =nullptr;
// cudaMalloc(&d_ptr,copy_size);
// std::memset(h_pin,1,copy_size);
// //cudaMemset(d_ptr,1,copy_size); 
// //cudaMemset(d_ptr,0,1);  // These two commented lines are for testing purposes.
// cudaMemset(d_ptr,0,copy_size);
// check_memory_type(h_pin);
// check_memory_type(d_ptr);
// for(int i=0;i<10;i++){
//   std::cout<<+h_pin[i]<<"\n";
//   uint8_t temp_val=0;
//   cudaMemcpy(&temp_val,d_ptr+i,1,cudaMemcpyDeviceToHost);
//   std::cout<<+temp_val<<"\n";
// }
// auto start = std::chrono::high_resolution_clock::now();
// cudaMemcpyAsync(d_ptr,h_pin,copy_size,cudaMemcpyHostToDevice);
// //cudaDeviceSynchronize(); //If you run with this line,async+sync --->sync,as the launch is asynchronous,but this cudaDeviceSynchronize() forcely stop the cpu till this process gets completed.So,its synchronous atlast.(Equals to cudaMemcpy+pinned memory from cpu to gpu).
// auto end = std::chrono::high_resolution_clock::now();
// double duration = std::chrono::duration<double,std::milli>(end-start).count();
// std::cout<<"CPU to GPU (pinned memory to GPU destination memory) time (500 MiB) :"<<duration<<"ms\n";
// double bandwidth = ((copy_size/(1024.0*1024.0*1024.0))/(duration/1000.0));
// std::cout<<"Bandwidth :" <<bandwidth<<" GiB/s is the speed of ( PCIe transfer+GPU copy)" <<"\n";
// for(int i=0;i<10;i++){
//   std::cout<<+h_pin[i]<<"\n";
//   uint8_t temp_val=0;
//   cudaMemcpy(&temp_val,d_ptr+i,1,cudaMemcpyDeviceToHost);
//   std::cout<<+temp_val<<"\n";
// }
// std::cin.get();
// cudaFreeHost(h_pin);
// cudaFree(d_ptr);

// //5.GPU to CPU (GPU source ptr to pageable destination ptr)
// uint8_t* d_src=nullptr;
// cudaMalloc(&d_src,copy_size);
// uint8_t* h_dst=new uint8_t[copy_size];
// cudaMemset(d_src,1,copy_size);
// std::memset(h_dst,0,copy_size);
// for(int i=0;i<10;i++){
//   uint8_t temp_val=0;
//   cudaMemcpy(&temp_val,d_src+i,1,cudaMemcpyDeviceToHost);
//   std::cout<<+temp_val<<"\n";
//   std::cout<<+h_dst[i]<<"\n";
// }
// check_memory_type(d_src);
// check_memory_type(h_dst);
// std::chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();
// cudaMemcpy(h_dst,d_src,copy_size,cudaMemcpyDeviceToHost);
// //cudaDeviceSynchronize();
// std::chrono::time_point<std::chrono::high_resolution_clock> end = std::chrono::high_resolution_clock::now();
// //std::chrono::duration<double,std::milli> duration = (end-start); // (end-start) is a duration object.
// //double time=duration.count();
// double duration = std::chrono::duration<double,std::milli>(end-start).count();
// std::cout<<"GPU to CPU (GPU source ptr to pageable ptr) time (500 MiB) :"<< duration<<"ms\n";
// double bandwidth = ((copy_size/(1024.0*1024.0*1024))/(duration/1000.0));
// std::cout<<"Bandwidth :"<<bandwidth<<" GiB/s\n";
// for(int i=0;i<10;i++){
//   uint8_t temp_val=0;
//   cudaMemcpy(&temp_val,d_src+i,1,cudaMemcpyDeviceToHost);
//   std::cout<<+temp_val<<"\n";
//   std::cout<<+h_dst[i]<<"\n";
// }
// std::cin.get();
// cudaFree(d_src);
// delete[] h_dst;

// //6.GPU to CPU (GPU source ptr to pageable destination ptr) using cudaMemcpyAsync (asynchronous) [with cudaDeviceSynchronize() (optional)]
// uint8_t* d_src=nullptr;
// cudaMalloc(&d_src,copy_size);
// uint8_t* h_dst=new uint8_t[copy_size];
// cudaMemset(d_src,1,copy_size);
// std::memset(h_dst,0,copy_size);
// for(int i=0;i<10;i++){
//   uint8_t temp_val=0;
//   cudaMemcpy(&temp_val,d_src+i,1,cudaMemcpyDeviceToHost);
//   std::cout<<+temp_val<<"\n";
//   std::cout<<+h_dst[i]<<"\n";
// }
// check_memory_type(d_src);
// check_memory_type(h_dst);
// std::chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();
// cudaMemcpyAsync(h_dst,d_src,copy_size,cudaMemcpyDeviceToHost);
// //cudaDeviceSynchronize();
// std::chrono::time_point<std::chrono::high_resolution_clock>end=std::chrono::high_resolution_clock::now();
// //std::chrono::duration<double,std::milli> duration = (end-start); //(end-start) is a duration object.
// //double time = duration.count();
// double duration = std::chrono::duration<double,std::milli>(end-start).count();
// std::cout<<"GPU to CPU (GPU source ptr to pageable ptr) time (500 MiB) :"<< duration<<"ms\n";
// double bandwidth = ((copy_size/(1024.0*1024.0*1024.0))/(duration/1000.0));
// std::cout<<"Bandwidth :"<<bandwidth<<" GiB/s\n";
// for(int i=0;i<10;i++){
//   uint8_t temp_val=0;
//   cudaMemcpy(&temp_val,d_src+i,1,cudaMemcpyDeviceToHost);
//   std::cout<<+temp_val<<"\n";
//   std::cout<<+h_dst[i]<<"\n";
// }
// std::cin.get();
// cudaFree(d_src);
// delete[] h_dst;

// //7.GPU to CPU (GPU source ptr to pinned destination ptr) using cudaMemcpy[ with cudaDeviceSynchronize() (optional)]
// uint8_t* d_src=nullptr;
// cudaMalloc(&d_src,copy_size);
// uint8_t* h_dst=nullptr;
// cudaMallocHost(&h_dst,copy_size);
// cudaMemset(d_src,1,copy_size);
// std::memset(h_dst,0,copy_size);
// for(int i =0;i<10;i++){
// uint8_t temp_val=0;
// cudaMemcpy(&temp_val,d_src+i,1,cudaMemcpyDeviceToHost);
// std::cout<<+temp_val<<"\n";
// std::cout<<+h_dst[i]<<"\n";
// }
// check_memory_type(d_src);
// check_memory_type(h_dst);
// std::chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();
// cudaMemcpy(h_dst,d_src,copy_size,cudaMemcpyDeviceToHost);
// cudaDeviceSynchronize();
// std::chrono::time_point<std::chrono::high_resolution_clock> end = std::chrono::high_resolution_clock::now();
// //std::chrono::duration<double,std::milli> duration = (end-start);
// //double time= duration.count();
// double duration = std::chrono::duration<double,std::milli> (end-start).count();
// std::cout<<"GPU to CPU (GPU source ptr to pinned ptr) time (500 MiB) : "<<duration<<"ms\n";
// double bandwidth = ((copy_size/(1024.0*1024.0*1024.0))/(duration/1000.0));
// std::cout<<"Bandwidth : "<<bandwidth<<" Gib/sec\n";
// for(int i=0;i<10;i++){
//   uint8_t temp_val=0;
//   cudaMemcpy(&temp_val,d_src+i,1,cudaMemcpyDeviceToHost);
//   std::cout<<+temp_val<<"\n";
//   std::cout<<+h_dst[i]<<"\n";
// }
// std::cin.get();
// cudaFree(d_src);
// cudaFreeHost(h_dst);

// //8. GPU to GPU (GPU source ptr to CPU pinned destination ptr using cudaMemcpyAsync) [cudaDeviceSynchronize() (optional)]
// uint8_t* d_src=nullptr;
// cudaMalloc(&d_src,copy_size);
// uint8_t* h_dst=nullptr;
// cudaMallocHost(&h_dst,copy_size);
// cudaMemset(d_src,1,copy_size);
// std::memset(h_dst,0,copy_size);
// for(int i=0;i<10;i++){
//   uint8_t temp_val=0;
//   cudaMemcpy(&temp_val,d_src+i,1,cudaMemcpyDeviceToHost);
//   std::cout<<+temp_val<<"\n";
//   std::cout<<+h_dst[i]<<"\n";
// } 
// check_memory_type(d_src);
// check_memory_type(h_dst);
// std::chrono::time_point<std::chrono::high_resolution_clock>start = std::chrono::high_resolution_clock::now();
// cudaMemcpyAsync(h_dst,d_src,copy_size,cudaMemcpyDeviceToHost);
// cudaDeviceSynchronize();
// std::chrono::time_point<std::chrono::high_resolution_clock>end = std::chrono::high_resolution_clock::now();
// //std::chrono::duration<double,std::milli> duration = (end-start);
// //double time = duration.count();
// double duration = std::chrono::duration<double,std::milli>(end-start).count();
// std::cout<<"GPU to GPU (GPU source pointer to CPU pinned destination pointer) time (500 MiB) : "<<duration<< "ms\n";
// double bandwidth = ((copy_size/(1024.0*1024.0*1024.0))/(duration/1000.0));
// std::cout<<"Bandwidth : "<<bandwidth<<" GiB/sec\n";
// for (int i=0;i<10;i++){
//   uint8_t temp_val=0;
//   cudaMemcpy(&temp_val,d_src+i,1,cudaMemcpyDeviceToHost);
//   std::cout<<+temp_val<<"\n";
//   std::cout<<+h_dst[i]<<"\n";
// }
// std::cin.get();
// cudaFree(d_src);
// cudaFreeHost(h_dst);

// //9.GPU to GPU (GPU source ptr to GPU destination ptr) using  cudaMemcpy [cudaDeviceSynchronize() (optional)]
// uint8_t* d_src=nullptr;
// cudaMalloc(&d_src,copy_size);
// uint8_t* d_dst=nullptr;
// cudaMalloc(&d_dst,copy_size);
// cudaMemset(d_src,1,copy_size);
// cudaMemset(d_dst,0,copy_size);
// for(int i=0;i<10;i++){
//   uint8_t temp_val;
//   cudaMemcpy(&temp_val,d_src+i,1,cudaMemcpyDeviceToHost);
//   std::cout<<+temp_val<<"\n";
//   uint8_t temp_val1;
//   cudaMemcpy(&temp_val1,d_dst+i,1,cudaMemcpyDeviceToHost);
//   std::cout<<+temp_val1<<"\n";
// }
// check_memory_type(d_src);
// check_memory_type(d_dst);
// std::chrono::time_point<std::chrono::high_resolution_clock>start = std::chrono::high_resolution_clock::now();
// cudaMemcpy(d_dst,d_src,copy_size,cudaMemcpyDeviceToDevice);
// //cudaDeviceSynchronize();
//  std::chrono::time_point<std::chrono::high_resolution_clock>end=std::chrono::high_resolution_clock::now();
// // std::chrono::duration<double,std::milli>duration=(end-start);
// //double time=duration.count();
// double duration = std::chrono::duration<double,std::milli>(end-start).count();
// std::cout<<"GPU to GPU (GPU source pointer to GPU destination pointer) time (500 MiB) : "<<duration<<"ms\n";
// double bandwidth = ((copy_size/(1024.0*1024.0*1024.0))/(duration/1000.0));
// std::cout<<"Bandwidth : "<<bandwidth<<" GiB/sec\n";
// for(int i=0;i<10;i++){
//   uint8_t temp_val;
//   cudaMemcpy(&temp_val,d_dst+i,1,cudaMemcpyDeviceToHost);
//   std::cout<<+temp_val<<"\n";
//   uint8_t temp_val1;
//   cudaMemcpy(&temp_val1,d_dst+i,1,cudaMemcpyDeviceToHost);
//   std::cout<<+temp_val1<<"\n";
// }
// std::cin.get();
// cudaFree(d_src);
// cudaFree(d_dst);


// //10.GPU to GPU (GPU source ptr to GPU destination ptr) using  cudaMemcpyAsync [cudaDeviceSynchronize() (optional)]
// uint8_t* d_src=nullptr;
// cudaMalloc(&d_src,copy_size);
// uint8_t* d_dst=nullptr;
// cudaMalloc(&d_dst,copy_size);
// cudaMemset(d_src,1,copy_size);
// cudaMemset(d_dst,0,copy_size);
// for(int i=0;i<10;i++){
//   uint8_t temp_val;
//   cudaMemcpy(&temp_val,d_src+i,1,cudaMemcpyDeviceToHost);
//   std::cout<<+temp_val<<"\n";
//   uint8_t temp_val1;
//   cudaMemcpy(&temp_val1,d_dst+i,1,cudaMemcpyDeviceToHost);
//   std::cout<<+temp_val1<<"\n";
// }
// check_memory_type(d_src);
// check_memory_type(d_dst);
// std::chrono::time_point<std::chrono::high_resolution_clock>start = std::chrono::high_resolution_clock::now();
// cudaMemcpyAsync(d_dst,d_src,copy_size,cudaMemcpyDeviceToDevice);
// cudaDeviceSynchronize();
//  std::chrono::time_point<std::chrono::high_resolution_clock>end=std::chrono::high_resolution_clock::now();
// // std::chrono::duration<double,std::milli>duration=(end-start);
// //double time=duration.count();
// double duration = std::chrono::duration<double,std::milli>(end-start).count();
// std::cout<<"GPU to GPU (GPU source pointer to GPU destination pointer) time (500 MiB) : "<<duration<<"ms\n";
// double bandwidth = ((copy_size/(1024.0*1024.0*1024.0))/(duration/1000.0));
// std::cout<<"Bandwidth : "<<bandwidth<<" GiB/sec\n";
// for(int i=0;i<10;i++){
//   uint8_t temp_val;
//   cudaMemcpy(&temp_val,d_dst+i,1,cudaMemcpyDeviceToHost);
//   std::cout<<+temp_val<<"\n";
//   uint8_t temp_val1;
//   cudaMemcpy(&temp_val1,d_dst+i,1,cudaMemcpyDeviceToHost);
//   std::cout<<+temp_val1<<"\n";
// }
// //std::cin.get();
// cudaFree(d_src);
// cudaFree(d_dst);

// // Memset (CPU) 
// uint8_t* h_ptr = new uint8_t[copy_size];
// for(int i=0;i<10;i++){
//   std::cout<<+h_ptr[i]<<"\n";
// }
// auto start = std::chrono::high_resolution_clock::now();
// std::memset(h_ptr,1,copy_size);
// auto end=std::chrono::high_resolution_clock::now();
// double duration = std::chrono::duration<double,std::milli>(end-start).count();
// std::cout<<"CPU memset time (500 MiB) : "<<duration<<"ms\n";
// double bandwidth = ((copy_size/(1024.0*1024.0*1024.0))/(duration/1000.0));
// std::cout<<"Bandwidth : "<<bandwidth<<" GiB/sec\n";
// for(int i=0;i<10;i++){
//   std::cout<<+h_ptr[i]<<"\n";
// }
// auto start1 = std::chrono::high_resolution_clock::now();
// std::memset(h_ptr,2,copy_size);
// auto end1=std::chrono::high_resolution_clock::now();
// double duration2 = std::chrono::duration<double,std::milli>(end1-start1).count();
// std::cout<<"CPU memset time (500 MiB) : "<<duration2<<"ms\n";
// double bandwidth2 = ((copy_size/(1024.0*1024.0*1024.0))/(duration2/1000.0));
// std::cout<<"Bandwidth : "<<bandwidth2<<" GiB/sec\n";
// for(int i=0;i<10;i++){
//   std::cout<<+h_ptr[i]<<"\n";
// }
// auto start2 = std::chrono::high_resolution_clock::now();
// std::memset(h_ptr,2,copy_size);
// auto end2=std::chrono::high_resolution_clock::now();
// double duration3 = std::chrono::duration<double,std::milli>(end2-start2).count();
// std::cout<<"CPU memset time (500 MiB) : "<<duration3<<"ms\n";
// double bandwidth3 = ((copy_size/(1024.0*1024.0*1024.0))/(duration3/1000.0));
// std::cout<<"Bandwidth : "<<bandwidth3<<" GiB/sec\n";
// for(int i=0;i<10;i++){
//   std::cout<<+h_ptr[i]<<"\n";
// }
// std::cin.get();
// delete[] h_ptr;



// Memset (GPU) 
uint8_t* h_ptr = nullptr;
cudaMalloc(&h_ptr,copy_size);
for(int i=0;i<10;i++){
uint8_t temp_val=0;
cudaMemcpy(&temp_val,h_ptr+i,1,cudaMemcpyDeviceToHost);
std::cout<<+temp_val<<"\n";
}
auto start = std::chrono::high_resolution_clock::now();
cudaMemset(h_ptr,1,copy_size);
cudaDeviceSynchronize();
auto end=std::chrono::high_resolution_clock::now();
double duration = std::chrono::duration<double,std::milli>(end-start).count();
std::cout<<"GPU memset time (500 MiB) : "<<duration<<"ms\n";
double bandwidth = ((copy_size/(1024.0*1024.0*1024.0))/(duration/1000.0));
std::cout<<"Bandwidth : "<<bandwidth<<" GiB/sec\n";
for(int i=0;i<10;i++){
  uint8_t temp_val2;
  cudaMemcpy(&temp_val2,h_ptr+i,1,cudaMemcpyDeviceToHost);
  std::cout<<+temp_val2<<"\n";
}
auto start1 = std::chrono::high_resolution_clock::now();
cudaMemset(h_ptr,2,copy_size);
cudaDeviceSynchronize();
auto end1=std::chrono::high_resolution_clock::now();
double duration2 = std::chrono::duration<double,std::milli>(end1-start1).count();
std::cout<<"GPU memset time (500 MiB) : "<<duration2<<"ms\n";
double bandwidth2 = ((copy_size/(1024.0*1024.0*1024.0))/(duration2/1000.0));
std::cout<<"Bandwidth : "<<bandwidth2<<" GiB/sec\n";
for(int i=0;i<10;i++){
  uint8_t temp_val3;
  cudaMemcpy(&temp_val3,h_ptr+i,1,cudaMemcpyDeviceToHost);
  std::cout<<+temp_val3<<"\n";
}
auto start2 = std::chrono::high_resolution_clock::now();
cudaMemset(h_ptr,3,copy_size);
cudaDeviceSynchronize();
auto end2=std::chrono::high_resolution_clock::now();
double duration3 = std::chrono::duration<double,std::milli>(end2-start2).count();
std::cout<<"GPU memset time (500 MiB) : "<<duration3<<"ms\n";
double bandwidth3 = ((copy_size/(1024.0*1024.0*1024.0))/(duration3/1000.0));
std::cout<<"Bandwidth : "<<bandwidth3<<" GiB/sec\n";
for(int i=0;i<10;i++){
  uint8_t temp_val4;
  cudaMemcpy(&temp_val4,h_ptr+i,1,cudaMemcpyDeviceToHost);
  std::cout<<+temp_val4<<"\n";
}
std::cin.get();
cudaFree(h_ptr);
}

