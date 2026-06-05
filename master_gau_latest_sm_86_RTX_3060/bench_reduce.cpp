// Throughput benchmark for reduce_sum on GPU across the relevant shapes.
#include "core/Tensor.h"
#include "ops/UnaryOps/Reduction.h"
#include "dtype/Types.h"
#include "device/DeviceCore.h"
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

using namespace OwnTensor;

static void bench(const char* tag, std::vector<int> shape, int axis) {
    int n=1; for(int s:shape) n*=s;
    std::vector<float> data(n, 1.0f);
    std::vector<int64_t> sh(shape.begin(),shape.end());
    Tensor t(Shape{sh}, TensorOptions().with_dtype(Dtype::Float32));
    t.set_data(data);
    Tensor g = t.to_cuda(0);

    for(int i=0;i<20;i++){ Tensor r = reduce_sum(g,{(int64_t)axis},false); }
    cudaDeviceSynchronize();

    const int iters = 200;
    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for(int i=0;i<iters;i++){ Tensor r = reduce_sum(g,{(int64_t)axis},false); }
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms=0; cudaEventElapsedTime(&ms,a,b);
    double per = ms/iters;                       // ms per call
    double gbps = (double)n*4.0/(per*1e-3)/1e9;  // read-bound GB/s
    printf("%-26s shape[", tag);
    for(size_t i=0;i<shape.size();++i) printf("%s%d", i?"x":"", shape[i]);
    printf("] ax%d  %.3f ms/call  %.1f GB/s\n", axis, per, gbps);
}

int main(){
    printf("=== our GPU reduce_sum throughput (read-bound GB/s; RTX3060 ~360 GB/s peak) ===\n");
    bench("last-axis (row-sum)",   {16384,768}, 1);
    bench("leading-axis (biasgrad)",{16384,768}, 0);
    bench("middle-axis",           {32,512,768}, 1);
    bench("large last-axis",       {4096,4096}, 1);
    bench("large leading-axis",    {4096,4096}, 0);
    return 0;
}
