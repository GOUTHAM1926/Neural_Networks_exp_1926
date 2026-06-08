#include "core/delay_kernel.h"

__global__ void delay_kernel(long long cycles)
{
  long long start = clock64();
  while (clock64() - start < cycles)
  {

  }
}

void launch_delay(cudaStream_t stream)
{
  delay_kernel<<<1,1,0,stream>>>(4000000000ULL);
}

