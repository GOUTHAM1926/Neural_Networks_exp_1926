#include <chrono>
#include <cstring>
#include <iostream>
#include <omp.h>
using namespace std;
int main() {
  int M = 2000;
  int N = 2000;
  int K = 2000;
  int size_A =
      M * K * sizeof(float); // needed if we are allocating using malloc(size)
  int size_B =
      K * N * sizeof(float); // needed if we are allocating using malloc(size)
  int size_C =
      M * N * sizeof(float); // needed if we are allocating using malloc(size)
  // creaiting matrices
  // doing like this is wrong as this syntax (three lines down) is to create a
  // 2D array of float-pointers and not a 2D matrix of floats ,
  // float* A[M][K] = malloc(size_A);
  // float* B[K][N]= malloc(size_B);
  // float* C[M][N]= malloc(size_C);
  // In cpp , we have three choices to create matrices :
  // Option 1 : Flat 1D array (this is most used technique,as this will be
  // stored in heap and better for large matrices)
  // using malloc() to allocate memory in heap
  //  float* A = (float*)malloc(size_A); // so that u can access it as A[i*K+k]
  //  instead of A[i][k] , float* B = (float*)malloc(size_B); // in rhs we cast
  //  that to float  using (float*) becoz malloc returns void pointer , float* C
  //  = (float*)malloc(size_C);
  // using "new" command to allocate memory in heap
  float *A = new float[M * K]; // no cast and no size_of needed
  float *B = new float[K * N];
  float *C = new float[M * N];
  // std::memset(C, 0, size_C);

  // Option 2 :
  // stack 2D array (simpler but wont scale to  big N)
  // float A[M][K]; //no malloc needed , lives on the stack and cans ee A[i][k]
  // directly,but this breaks for large matrices(stack overflow) float B[K][N];
  // float C[M][N]; //in this case we can write C[i][j] instead of C[i*N+j]
  // but that option-A  (1-D stack is preferrable) as it works for any size and
  // will be contiguous in memory thats exactly how GPU(cuda) does it ---> no 2D
  // arrays on  the device
  // option 3 : 2D arrays on the heap (we can do this too  , but i requires
  // allocating an array of pointers , then each row separately) float** A = new
  // float*[M]; //array of M row-pointers for(int i=0;i<M;i++){
  //  A[i]=new float*[k]}; // each row is a separate allocation
  // and they are not contiguous in memory  - not cache friendly ,
  // here A[i][j] works  but gives bad performance becoz :
  // each row is a separate heap allocation ---> rows are scattered in memory
  // no guarantee rows are contiguous and not cache friendly
  // need M+1 separate delete[](if new() used)/free() (if malloc() used) calls
  // to free it ,
  // and pointer chasing is impossible in GPU VRAM (you can't chase pointers
  // from device code):
  // pointer chasing means following the pointers from one memory location to
  // another , which requires a separate memory access for each pointer :
  // Why float** (2D heap) is impossible on GPU — the pointer chasing problem :
  // What happens in memory with float** on CPU :
  //   CPU RAM :
  //   A (float**) → [ ptr_row0 | ptr_row1 | ptr_row2 ]   ← array of pointers
  //                      ↓           ↓           ↓
  //                  [1.0, 2.0]  [3.0, 4.0]  [5.0, 6.0]  ← actual data
  //                  (scattered!)
  // When you do A[i][j], the CPU does two memory reads :

  // First read: go to A[i] --->  get the pointer to row i
  // Second read: go to that pointer + j ---> get the actual float
  // This is called "pointer chasing" ---> you follow one pointer to find
  // another address
  // Now imagine you cudaMemcpy this to GPU :
  // cudaMemcpy(d_A, A, M * sizeof(float*), ...)
  // You just copied the array of pointers to GPU VRAM. But those pointers still
  // contain CPU RAM addresses , GPU VRAM:
  //   d_A → [ 0x7fff1000 | 0x7fff2000 | 0x7fff3000 ]  ← these are CPU
  //   addresses!
  //               ↓              ↓              ↓
  //            CRASH       CRASH        CRASH     ← GPU can't access CPU RAM!
  // /When the GPU thread does d_A[i], it gets 0x7fff1000 — a CPU address ,
  //  The GPU tries to read from that address → illegal memory access → kernel
  //  crash ,
  // The GPU physically cannot reach CPU RAM through those pointers ,
  // Could you fix it? Technically yes, but it's a nightmare :
  // You'd have to do this for EVERY row:
  // float* d_rows[M];
  // for (int i = 0; i < M; i++) {
  //     cudaMalloc(&d_rows[i], K * sizeof(float));     // allocate each row on
  //     GPU cudaMemcpy(d_rows[i], A[i], K * sizeof(float), ...);  // copy each
  //     row
  // }
  // // Then copy the pointer array itself to GPU:
  // float** d_A;
  // cudaMalloc(&d_A, M * sizeof(float*));
  // cudaMemcpy(d_A, d_rows, M * sizeof(float*), ...);

  // whats happening here in this hard path :
  // Step 1: Allocate NEW memory on GPU for each row :
  // cudaMalloc(&d_rows[0], ...);  → d_rows[0] = 0xGPU_ADDR_A  (GPU address!)
  // cudaMalloc(&d_rows[1], ...);  → d_rows[1] = 0xGPU_ADDR_B  (GPU address!)
  // cudaMalloc(&d_rows[2], ...);  → d_rows[2] = 0xGPU_ADDR_C  (GPU address!)
  // Step 2: Copy the data from each CPU row to the corresponding GPU row :
  // cudaMemcpy(d_rows[0], A[0], ...)  → copies floats from CPU row 0 → GPU row
  // 0 cudaMemcpy(d_rows[1], A[1], ...)  → copies floats from CPU row 1 → GPU
  // row 1 cudaMemcpy(d_rows[2], A[2], ...)  → copies floats from CPU row 2 →
  // GPU row 2 Step 3: Now d_rows[] is an array sitting on CPU that contains GPU
  // addresses : CPU side:
  //   d_rows = [ 0xGPU_ADDR_A | 0xGPU_ADDR_B | 0xGPU_ADDR_C ]
  //                   ↓               ↓               ↓
  // GPU VRAM:    [1.0, 2.0]      [3.0, 4.0]      [5.0, 6.0]   ← actual data, on
  // GPU
  // Step 4: Copy this pointer array itself to GPU :
  // cudaMalloc(&d_A, M * sizeof(float*));
  // cudaMemcpy(d_A, d_rows, M * sizeof(float*), ...);
  // Now on GPU:
  // GPU VRAM:
  //   d_A → [ 0xGPU_ADDR_A | 0xGPU_ADDR_B | 0xGPU_ADDR_C ]  ← GPU addresses
  //   this time!
  //                ↓               ↓               ↓
  //           [1.0, 2.0]      [3.0, 4.0]      [5.0, 6.0]     ← works!
  // So d_A[i][j] in the kernel would work — the pointer chase goes GPU→GPU.
  // That's M+1 separate allocations and M+1 separate memcpys.
  // Absolute nightmare. And the rows are still scattered in VRAM → no
  // coalescing.

  // setting values of A and B to 1.0f
  for (int i = 0; i < M * K; ++i) {
    A[i] = 1.0f;
  }
  for (int i = 0; i < K * N; ++i) {
    B[i] = 1.0f;
  }

  //   // printing A and B matrices :
  //   std::cout << "Matrix A : \n";
  //   for (int i = 0; i < M; ++i) {
  //     std::cout << "[";
  //     for (int k = 0; k < K; ++k) {
  //       std::cout << A[i * K + k] << " ";
  //     }

  //     std::cout << "]" << std::endl;
  //   }
  //   std::cout << "Matrix B : \n";
  //   for (int k = 0; k < K; ++k) {
  //     std::cout << "[";
  //     for (int j = 0; j < N; ++j) {
  //       std::cout << B[k * N + j] << " ";
  //     }
  //     std::cout << "]" << std::endl;
  //   }
  // for these printing logics , it will print like :
  //  Matrix A :
  //  [1 1 1 1 ]
  //  [1 1 1 1 ]
  //  [1 1 1 1 ]
  //  Matrix B :
  //  [1 1 1 ]
  //  [1 1 1 ]
  //  [1 1 1 ]
  //  [1 1 1 ]
  // as we are printing space after every element printed, after last element
  // too ,space has been printed , to avoid this we can use a conditional check
  // in two ways : Option A : Print space before each element, but skip it for
  // the first one :
  //  if (j > 0) cout << " ";    // space before element, but NOT before the
  //  first cout << value;
  // printing loop will be :
  //   std::cout << "Matrix-A:" << std::endl;
  //   for (int i = 0; i < M; ++i) {
  //     std::cout << "[";
  //     for (int k = 0; k < K; ++k) {
  //       if (k > 0)
  //         std::cout << " ";
  //       std::cout << A[i * K + k];
  //     }
  //     std::cout << "]" << "\n";
  //   }
  //   std::cout << "Matmul-B" << "\n";
  //   for (int i = 0; i < K; ++i) {
  //     std::cout << "[";
  //     for (int j = 0; j < N; ++j) {
  //       if (j > 0) {
  //         std::cout << " ";
  //       }
  //       std::cout << B[i * N + j];
  //     }
  //     std::cout << "]" << "\n";
  //   }
  // Result : (there will be no gap btn last element and closing bracket of that
  // row)
  //   Matrix-A:
  // [1 1 1 1]
  // [1 1 1 1]
  // [1 1 1 1]
  // Matmul-B
  // [1 1 1]
  // [1 1 1]
  // [1 1 1]
  // [1 1 1]
  // Option B : Print space after each element but skip if its the last one :
  //  cout << value;
  //  if (j < cols - 1) cout << " ";   // space after element, but NOT after
  // //Printing loop will be :
  //   std::cout << "Matrix--A" << "\n";
  //   for (int i = 0; i < M; ++i) {
  //     std::cout << "[";
  //     for (int j = 0; j < K; ++j) {
  //       std::cout << A[i * K + j];
  //       if (j < K - 1) {
  //         std::cout << " ";
  //       }
  //     }
  //     std::cout << "]" << "\n";
  //   }
  //   std::cout << "Matrix--B:\n";
  //   for (int i = 0; i < K; ++i) {
  //     std::cout << "[";
  //     for (int j = 0; j < N; ++j) {
  //       std::cout << B[i * N + j];
  //       if (j < N - 1) {
  //         std::cout << " ";
  //       }
  //     }
  //     std::cout << "]\n";
  //   }
  //   Result :
  //   Matrix--A
  // [1 1 1 1]
  // [1 1 1 1]
  // [1 1 1 1]
  // Matrix--B:
  // [1 1 1]
  // [1 1 1]
  // [1 1 1]
  // [1 1 1]
  //  ijk loop (looping order)and global accumulation
  double avg_time = 0.0L;
  int num_runs = 5;
  int warmup_runs = 3;
  // If wants to use explicit number of threads ,then to set those many threads
  // , can use this cpp function : "omp_set_num_threads(n)" where "n" is the
  // number of threads in which u want to run that prgm/application ,
  //  to use max threads in any given system ,there are two methods :
  //  1)omp_set_num_threads(omp_get_max_threads()) ; 2)dont use any of these omp
  //  functions(omp_set_max_threads(),omp_get_max_threads(), just that compiler
  //  directive " #pragma omp parallel for  is enough and it will take all the
  //  threads in that given system and parallelize over all threads")

  /*
  omp_get_num_procs() returns the number of physical + logical processors
  (cores) available to your program on the hardware. On an i7-14700K, both
  omp_get_max_threads() and omp_get_num_procs() will likely return 28 (your
  total logical cores). So in practice they are almost always the same number.

  The subtle difference:
  omp_get_num_procs() = "How many logical cores does this machine physically
  have?" omp_get_max_threads() = "How many threads will OpenMP actually use in
  the next parallel region?"

  They can differ in one situation: if you called omp_set_num_threads(4)
  earlier, then: omp_get_num_procs() still returns 28 (hardware doesn't change)
  omp_get_max_threads() now returns 4 (because you changed the thread limit)

  So omp_get_num_procs() always tells you the hardware truth, while
  omp_get_max_threads() tells you the current OpenMP configuration. For your
  experiments, omp_get_max_threads() is the one you want!
  */
  // omp_set_num_threads(20);
  //    warm up runs :
  for (int run = 0; run < warmup_runs; ++run) {
    std::memset(C, 0, size_C);
    // #pragma omp parallel for // openmp multiple threads parallelization ,
    for (int i = 0; i < M; ++i) {
      // #pragma omp parallel for
      for (int j = 0; j < N; ++j) {
#pragma omp parallel for
        for (int k = 0; k < K; ++k) {
          C[i * N + j] += A[i * K + k] * B[k * N + j];
        }
      }
    }
  }

  // actual calculation runs :
  for (int run = 0; run < num_runs; ++run) {
    std::memset(C, 0, size_C);
    auto start = std::chrono::high_resolution_clock::now();
    // #pragma omp parallel for // openmp multiple threads parallelization ,
    for (int i = 0; i < M; ++i) {
      // #pragma omp parallel for
      for (int j = 0; j < N; ++j) {
#pragma omp parallel for
        for (int k = 0; k < K; ++k) {
          C[i * N + j] += A[i * K + k] * B[k * N + j];
        }
      }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    // auto diff = (end - start).count();
    //   std::cout << "Time taken(naive-ijk loop with no optimizations): "
    //             << diff.count() << "secs" << std::endl;
    avg_time += diff.count();
  }
  avg_time /= num_runs;
  std::cout
      << "Time taken(naive-ijk loop with  openmp implementation(on j -loop  "
         ") in code and with "
         "-fopenmp flag and    "
         "with -O3 and -march=native  optimization flag   in "
         "compilation command)"
         " (M=2000,N=2000,K=2000)   : "
      << avg_time << "secs" << std::endl;
  // printing Matrix-C :
  //   std::cout << "Matrix C" << std::endl;
  //   for (int i = 0; i < M; i++) {
  //     std::cout << "[";
  //     for (int j = 0; j < N; j++) {
  //       std::cout << C[i * N + j] << " ";
  //     }
  //     std::cout << "]" << std::endl;
  //   }

  // free(A); //if used malloc , then free the memory using free() and if used
  // new() for allcoation instead of malloc, use delete[] for deallocation ,
  // free(B);
  // free(C);
  // std::cin.get();
  delete[] A;
  delete[] B;
  delete[] C;
}
