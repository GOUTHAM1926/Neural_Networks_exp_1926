// Standalone driver: characterize reduction correctness across shapes/axes.
// Compares OURS-CPU and OURS-GPU against a brute-force reference.
#include "core/Tensor.h"
#include "ops/UnaryOps/Reduction.h"
#include "dtype/Types.h"
#include "device/DeviceCore.h"
#include <cstdio>
#include <vector>
#include <numeric>

using namespace OwnTensor;

// Brute-force reference reduction over `axis` of a row-major tensor.
static std::vector<float> reference(const std::vector<int>& shape, int axis,
                                    const std::vector<float>& data) {
    int ndim = (int)shape.size();
    std::vector<int> ostride(ndim), istride(ndim);
    int n_out = 1, n_in = 1;
    // input strides
    istride[ndim - 1] = 1;
    for (int d = ndim - 2; d >= 0; --d) istride[d] = istride[d + 1] * shape[d + 1];
    // output shape = shape with axis removed
    std::vector<int> oshape;
    for (int d = 0; d < ndim; ++d) if (d != axis) oshape.push_back(shape[d]);
    for (int s : oshape) n_out *= s;
    n_in = data.size();
    std::vector<float> out(n_out, 0.0f);
    for (int lin = 0; lin < n_in; ++lin) {
        // decompose lin into coords
        int rem = lin, ocoord = 0, omul = 1;
        std::vector<int> coord(ndim);
        for (int d = ndim - 1; d >= 0; --d) { coord[d] = rem % shape[d]; rem /= shape[d]; }
        // build output index skipping axis
        std::vector<int> oc;
        for (int d = 0; d < ndim; ++d) if (d != axis) oc.push_back(coord[d]);
        ocoord = 0; omul = 1;
        for (int d = (int)oshape.size() - 1; d >= 0; --d) { ocoord += oc[d] * omul; omul *= oshape[d]; }
        out[ocoord] += data[lin];
    }
    return out;
}

static bool eq(const float* p, const std::vector<float>& r, int n) {
    if (n != (int)r.size()) return false;
    for (int i = 0; i < n; ++i) if (p[i] != r[i]) return false;
    return true;
}

static void test(const std::vector<int>& shape, int axis) {
    int n = 1; for (int s : shape) n *= s;
    std::vector<float> data(n);
    std::iota(data.begin(), data.end(), 1.0f);   // 1,2,3,...

    auto ref = reference(shape, axis, data);

    std::vector<int64_t> sh64(shape.begin(), shape.end());
    Tensor cpu_t(Shape{sh64}, TensorOptions().with_dtype(Dtype::Float32));
    cpu_t.set_data(data);

    Tensor c = reduce_sum(cpu_t, {(int64_t)axis}, false);
    Tensor g = reduce_sum(cpu_t.to_cuda(0), {(int64_t)axis}, false).to_cpu();

    bool cok = eq(c.data<float>(), ref, (int)c.numel());
    bool gok = eq(g.data<float>(), ref, (int)g.numel());

    printf("shape[");
    for (size_t i = 0; i < shape.size(); ++i) printf("%s%d", i ? "," : "", shape[i]);
    printf("] axis=%d  reduced=%d outputs=%zu  CPU %-8s GPU %-8s\n",
           axis, shape[axis], ref.size(),
           cok ? "MATCH" : "MISMATCH", gok ? "MATCH" : "MISMATCH");
    if (!gok || !cok) {
        int n = std::min<int>(8, (int)ref.size());
        printf("      ref:"); for (int i=0;i<n;++i) printf(" %8.1f", ref[i]); printf("\n");
        printf("      cpu:"); for (int i=0;i<std::min<int>(8,(int)c.numel());++i) printf(" %8.1f", c.data<float>()[i]); printf("\n");
        printf("      gpu:"); for (int i=0;i<std::min<int>(8,(int)g.numel());++i) printf(" %8.1f", g.data<float>()[i]); printf("\n");
    }
}

int main() {
    printf("=== last-axis (InnerContiguous) — vary reduced_count ===\n");
    test({4, 2}, 1);
    test({4, 4}, 1);
    test({4, 8}, 1);
    test({4, 32}, 1);
    test({4, 256}, 1);
    test({8, 768}, 1);

    printf("\n=== HYPOTHESIS TEST: InnerContiguous with num_outputs==1 should WORK ===\n");
    test({1, 8}, 1);      // 1 output row  -> predict MATCH
    test({1, 768}, 1);    // 1 output row  -> predict MATCH
    test({768}, 0);       // full reduce, 1 output -> predict MATCH
    test({2, 768}, 1);    // 2 output rows -> predict MISMATCH
    test({3, 32}, 1);     // 3 output rows -> predict MISMATCH

    printf("\n=== first-axis (OuterContiguous, outer==1) ===\n");
    test({2, 3, 2}, 0);
    test({16, 64}, 0);

    printf("\n=== MIDDLE axis (the sandwich) — outer>1 ===\n");
    test({2, 3, 2}, 1);
    test({4, 8, 16}, 1);
    test({8, 50, 64}, 1);   // transformer-ish: reduce seq dim
    return 0;
}
