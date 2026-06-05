// Single-shape probe for the multi-CTA OuterContiguous failure:
// [1024,1024] reduce axis=0 (leading-axis sum) on GPU. One reduce_sum launch.
#include "core/Tensor.h"
#include "ops/UnaryOps/Reduction.h"
#include "dtype/Types.h"
#include <cstdio>
#include <vector>
#include <cmath>
using namespace OwnTensor;

int main() {
    const int R = 1024, C = 1024;
    std::vector<float> data((size_t)R * C);
    uint64_t s = 0xABCDEF;
    for (auto& v : data) { s = s * 6364136223846793005ULL + 1; v = (float)((double)(uint32_t)(s >> 32) / 2147483648.0 - 1.0); }
    Tensor t(Shape{{R, C}}, TensorOptions().with_dtype(Dtype::Float32));
    t.set_data(data);
    Tensor g = reduce_sum(t.to_cuda(0), {0}, false).to_cpu();   // GPU multi-CTA path
    // reference (column sums)
    std::vector<double> ref(C, 0.0);
    for (int r = 0; r < R; ++r) for (int c = 0; c < C; ++c) ref[c] += data[(size_t)r * C + c];
    double maxerr = 0; int bad = 0;
    for (int c = 0; c < C; ++c) { double e = std::fabs((double)g.data<float>()[c] - ref[c]); maxerr = std::max(maxerr, e); if (e > 1e-2 + 1e-3 * std::fabs(ref[c])) bad++; }
    printf("[1024,1024] axis=0 GPU multi-CTA: maxerr=%.4f  bad=%d/%d  gpu[430]=%.4f ref[430]=%.4f\n",
           maxerr, bad, C, g.data<float>()[430], ref[430]);
    return bad ? 1 : 0;
}
