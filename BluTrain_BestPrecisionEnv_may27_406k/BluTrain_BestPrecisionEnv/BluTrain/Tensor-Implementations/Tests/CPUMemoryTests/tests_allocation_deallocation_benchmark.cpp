#include<iostream>
#include<chrono> //For measuring time.
#include<cstring> // For std::memset
// #include "device/AllocatorRegistry.h" //To get our allocator.
// #include "device/DeviceCore.h" //To set device.
#include<cuda_runtime.h> // for cuda runtime library methods.
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
int main() {
     size_t size = 1024 * 1024 * 200; // 200 MB

//     // CASE 1: Standard 'new' (Pageable) 
//     // 1. "new" puts data on the HEAP, not the stack. 
//     //    (Stack is small, only ~8MB. Heap is big, uses all RAM).
//     // 2. "new uint8_t[]" returns a "uint8_t*" (pointer to unsigned char).
//     uint8_t* ptr_pageable = new uint8_t[size];

//     // Why (void*)? 
//     // std::cout treats "char*" or "uint8_t*" as a STRING (text).
//     // It tries to print the letters at that address.
//     // By casting to (void*), we force it to print the HEX ADDRESS number.
//     /*The Problem: std::cout is designed to be "smart".
// If you give it an int*, it prints the address (e.g., 0x7ffee...).
// If you give it a char* (or uint8_t*), it thinks you are printing a String (like "Hello World").
// It will try to read characters from memory until it hits a \0 (null terminator).
// Since your memory is garbage/random, it might print weird symbols or crash.
// Packet Fix: By casting to 
// (void*)
// , we tell std::cout: "Forget the type! Just treat this as a generic memory address." 
// This forces it to print the hex number.
// */
//     std::cout << "100 MB created at pointer address: " << (void*)ptr_pageable << std::endl;
//     // --- Case 2: Verify MEMSET ---
//     // memset takes an 'int' (32-bit) but converts it to 'unsigned char' (8-bit).
//     // It grabs the lowest 8 bits. e.g., if you pass 257 (0x101), it takes 0x01.
//     // It then writes this byte to EVERY SINGLE BYTE in the range.
//     // std::memset(ptr_pageable, 1, size); // Fills 100MB with 0x01

//     // // Print first 10 bytes to verify
//     // std::cout << "First 10 bytes: ";
//     // for(int i=0; i<10; i++){
//     //     // We cast to (int) so cout prints "1" instead of the ASCII character for 1 (SOH)
//     //     std::cout << (int)ptr_pageable[i] << " ";
//     // }
//     // std::cout << std::endl;

//     // --- METRICS EXPLAINED ---
//     /*
//       Base-10 (Standard):
//       1 KB = 1000 bytes
//       1 MB = 1000 * 1000 bytes
//       1 GB = 1000 * 1000 * 1000 bytes
//       Used by: Hard Drive Manufacturers (to look bigger), Network speeds.

//       Base-2 (IEC Standard - "Bi" = Binary): 
//       1 KiB (Kibibyte) = 1024 bytes (2^10)
//       1 MiB (Mebibyte) = 1024 * 1024 bytes (2^20)
//       1 GiB (Gibibyte) = 1024 * 1024 * 1024 bytes (2^30)
//       Used by: Operating Systems (Linux/Windows RAM), Programmers.

//       Confusion:
//       When Windows says "You have 100MB free", it roughly means 100 MiB.
//       Most code uses size = 1024*1024 (MiB) but calls it "MB".
//       Here, we allocated 100 * 1024 * 1024 = 100 MiB.
//     */
// check_memory_type(ptr_pageable);
//     // --- CASE 2: The Upgrade (Pageable -> Pinned) ---
//     std::cout << "\n[UPGRADE] Attempting to Pin memory with cudaHostRegister...\n";
    
//     // This system call tells the OS Kernel to LOCK(page-lock/pin) the memory
//     cudaError_t status = cudaHostRegister(ptr_pageable, size, cudaHostRegisterDefault);
    
//     if (status != cudaSuccess) {
//         std::cerr << "Failed to pin memory: " << cudaGetErrorString(status) << "\n";
//     } else {
//         std::cout << "Success! Validating with Driver...\n";
//         check_memory_type(ptr_pageable); // Should now say PINNED
        
//         // IMPORTANT: Unregister before deleting
//         cudaHostUnregister(ptr_pageable);
//     }
    
//     std::cout << "\n(Program Paused. Press ENTER to finish)\n";
//     std::cin.get();
    
//     // cleanup
//     delete[] ptr_pageable;
// std::cout<<(void*)ptr_pageable<<"\n";
// Case 3 : Allocating pinned memory at start itself 
//1. Declare a pointer 
// uint8_t* ptr_pinned = nullptr;
// //2. Allocate memory using cudaMallocHost 
// //2.Allocate Pinned Memory directly
// //This does (malloc +mlock+ mapping in driver hash table in this prgm's memory plus in gpu's mmu lookup table)
// cudaError_t status = cudaMallocHost(&ptr_pinned,size);
// if(status != cudaSuccess){
//   std::cout<<"Failed: "<<cudaGetErrorString(status)<<"\n";
// }else {
//   std::cout << "Success! Created 100MB Pinned Memory at: " << (void*)ptr_pinned << "\n";
//         check_memory_type(ptr_pinned);
// }std::cin.get();
// cudaFreeHost(ptr_pinned);
// //Unpinning it using cudaHostUnregister and then deleting which gives segmentation fault.This allocated memory is managed by driver and should be freed by using cudaFreeHost only.
// // cudaHostUnregister(ptr_pinned);
// // delete[] ptr_pinned;
// std::cin.get();

//Case-4 : cudaHostAlloc (Mapped/Zero-copy)
// uint8_t* h_ptr = nullptr;
// void* d_ptr = nullptr;
// //Allow GPU to map this memory
// cudaError_t status = cudaHostAlloc((void**)&h_ptr,size,cudaHostAllocMapped);
// if(status != cudaSuccess){
//   std::cout<<"Alloc Failed : " << cudaGetErrorString(status) << "\n";

// }else{
//   std::cout << "Success! Mapped memory at : "<< (void*)h_ptr << "\n";
//   check_memory_type(h_ptr);
//  //Imp part : get GPU pointer 
//  //Ask Driver : " What address should the GPU use to see this CPU memory? "
//  cudaHostGetDevicePointer(&d_ptr,h_ptr,0);
//  std::cout << "Device Pointer (Map) " << d_ptr << "\n";
//  //Now we can access this memory from GPU directly
//  //We can also use cudaMemcpy(d_ptr,h_ptr,size,cudaMemcpyHostToDevice);
//  //But we dont need to copy it
// }
// std::cin.get();
// cudaFreeHost(h_ptr);
// std::cin.get();

//GPU ALlocation and deallocation (500 MiB)
//1)Using cudaMalloc and cudaFree
size_t size1 = 1024*1024*500;
/*Strictness: In C++, void* is a "Generic Pointer" (it can point to anything).
cudaMalloc is a generic function. It expects void**.
Laziness: If you declare uint8_t* ptr, you usually have to cast it: 
(void**)&ptr
.
If you declare void* ptr, you can just pass &ptr directly. It saves typing the cast 
(void**)
.*/
// uint8_t* d_ptr = nullptr;
// cudaError_t status =cudaMalloc((void**)& d_ptr,size1);
// if(status !=cudaSuccess){
//   std::cout <<"Allocation failed :"<<cudaGetErrorString(status)<<"\n";
// }else{
//   std::cout<<"Allocation Success :" <<(void*)d_ptr<<"\n";
// }
// std::cin.get();
// cudaError_t status1 =cudaFree(d_ptr);
// if(status1 != cudaSuccess){
//   std::cout<<"Deallocation failed :"<<cudaGetErrorString(status1)<<"\n";

// }else{
//   std::cout<<"Deallocation Success : "<<(void*)d_ptr<<"\n";
// }
// std::cin.get();
 //Time-Test/Speed-Test btn gpu allocation using cudaMalloc(synchronous) and cudaMallocAsync(asynchronous) 
using Clock = std::chrono::high_resolution_clock;
std::cout<<"Speed -Test\n";
//1.Measure cudaMalloc
void* p=nullptr;
auto start = Clock::now();
cudaMalloc(&p,size);
auto end = Clock::now();
auto duration = (end-start).count();
std::cout<<"cudaMalloc Time : "<<duration<<" ns\n";
auto start3 = Clock::now();
cudaFree(p);
//cudaDeviceSynchronize();
auto end3 = Clock::now();
auto duration3 = (end3-start3).count();
std::cout<<"cudaFree Time : "<<duration3<<" ns\n";
//2.Measure cudaMallocAsync
void* p1 = nullptr;
auto start1 = Clock::now();
cudaMallocAsync(&p1,size,0);
//cudaDeviceSynchronize();
auto end1 = Clock::now();
auto duration1 = (end1-start1).count();
std::cout<<"cudaMallocAsync Time : "<<duration1<<" ns\n";
auto start2 = Clock::now();
cudaFreeAsync(p1,0);
//cudaDeviceSynchronize();
auto end2 = Clock::now();
auto duration2 = (end2-start2).count();
std::cout<<"cudaFreeAsync Time : "<<duration2<<" ns\n";

void* p2 = nullptr;
auto start4 = Clock::now();
cudaMallocAsync(&p2,size,0);
//cudaDeviceSynchronize();
auto end4 = Clock::now();
auto duration4 = (end4-start4).count();
std::cout<<"cudaMallocAsync Time : "<<duration4<<" ns\n";
auto start5 = Clock::now();
cudaFreeAsync(p2,0);
//cudaDeviceSynchronize();
auto end5 = Clock::now();
auto duration5 = (end5-start5).count();
std::cout<<"cudaFreeAsync Time : "<<duration5<<" ns\n";
//cudaDeviceSynchronize();
void* p3 = nullptr;
auto start6 = Clock::now();
cudaMallocAsync(&p3,size,0);
//cudaDeviceSynchronize();
auto end6 = Clock::now();
auto duration6 = (end6-start6).count();
std::cout<<"cudaMallocAsync Time : "<<duration6<<" ns\n";
auto start7 = Clock::now();
cudaFreeAsync(p3,0);
//cudaDeviceSynchronize();
auto end7 = Clock::now();
auto duration7 = (end7-start7).count();
std::cout<<"cudaFreeAsync Time : "<<duration7<<" ns\n";
cudaDeviceSynchronize();
    return 0;
}
