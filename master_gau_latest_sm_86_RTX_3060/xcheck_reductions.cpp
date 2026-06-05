// Comprehensive reduction validation: random data, many shapes/axes,
// CPU+GPU vs independent brute-force reference, plus a binary dump for a
// torch.sum cross-check (see xcheck_reductions.py).
#include "core/Tensor.h"
#include "ops/UnaryOps/Reduction.h"
#include "dtype/Types.h"
#include "device/DeviceCore.h"
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cmath>
#include <numeric>

using namespace OwnTensor;

static uint64_t rng_state = 0x12345678abcdULL;
static float frand() {                       // deterministic LCG -> [-1,1)
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    uint32_t x = (uint32_t)(rng_state >> 32);
    return (float)((double)x / 2147483648.0 - 1.0);
}

static FILE* dump = nullptr;
static int total = 0, failed = 0;

template<class W> static void wr(W v){ fwrite(&v, sizeof(v), 1, dump); }

// Brute-force reference: out index = input coords with `axes` removed.
static std::vector<double> reference(const std::vector<int>& shape,
                                     const std::vector<int>& axes,
                                     const std::vector<float>& data) {
    int nd = shape.size();
    std::vector<char> red(nd, 0); for (int a : axes) red[a] = 1;
    std::vector<int> oshape; for (int d=0; d<nd; ++d) if(!red[d]) oshape.push_back(shape[d]);
    int n_out = 1; for (int s : oshape) n_out *= s;
    std::vector<double> out(n_out, 0.0);
    int n_in = data.size();
    for (int lin=0; lin<n_in; ++lin) {
        int rem=lin; std::vector<int> coord(nd);
        for (int d=nd-1; d>=0; --d){ coord[d]=rem%shape[d]; rem/=shape[d]; }
        int oc=0; for (int d=0; d<nd; ++d) if(!red[d]) { int sz=shape[d]; oc=oc*sz+coord[d]; }
        out[oc] += data[lin];
    }
    return out;
}

static bool close(float a, double b, double atol, double rtol) {
    return std::fabs((double)a - b) <= atol + rtol * std::fabs(b);
}

static void test(const std::vector<int>& shape, const std::vector<int>& axes) {
    total++;
    int n=1; for(int s:shape) n*=s;
    std::vector<float> data(n); for(auto& v:data) v = frand();
    auto ref = reference(shape, axes, data);

    std::vector<int64_t> sh64(shape.begin(), shape.end()), ax64(axes.begin(), axes.end());
    Tensor t(Shape{sh64}, TensorOptions().with_dtype(Dtype::Float32));
    t.set_data(data);
    Tensor c = reduce_sum(t, ax64, false);
    Tensor g = reduce_sum(t.to_cuda(0), ax64, false).to_cpu();

    const double atol = 1e-3, rtol = 1e-3;
    bool cok = (int)c.numel()==(int)ref.size(), gok = (int)g.numel()==(int)ref.size();
    for (int i=0;i<(int)ref.size();++i){
        if (cok && !close(c.data<float>()[i], ref[i], atol, rtol)) cok=false;
        if (gok && !close(g.data<float>()[i], ref[i], atol, rtol)) gok=false;
    }
    if(!cok||!gok) failed++;
    printf("shape[");
    for(size_t i=0;i<shape.size();++i) printf("%s%d", i?"x":"", shape[i]);
    printf("] axes{"); for(size_t i=0;i<axes.size();++i) printf("%s%d", i?",":"", axes[i]);
    printf("} out=%zu  CPU %-8s GPU %-8s\n", ref.size(), cok?"MATCH":"FAIL", gok?"MATCH":"FAIL");

    // dump record for torch cross-check (single-axis only, to keep py simple)
    if (dump && axes.size()==1) {
        wr<int32_t>(shape.size());
        for(int s:shape) wr<int32_t>(s);
        wr<int32_t>(axes[0]);
        wr<int64_t>(n); fwrite(data.data(), sizeof(float), n, dump);
        wr<int64_t>(g.numel()); fwrite(g.data<float>(), sizeof(float), g.numel(), dump);
        fwrite(c.data<float>(), sizeof(float), c.numel(), dump);
    }
}

int main() {
    dump = fopen("/tmp/xcheck_dump.bin", "wb");

    printf("== last-axis / inner ==\n");
    for (int r : {2,3,7,8,16,31,32,33,64,256,768,1024}) test({5, r}, {1});
    printf("== leading-axis / outer ==\n");
    for (int o : {2,3,8,33,128}) test({o, 50}, {0});
    printf("== MIDDLE axis (3D) ==\n");
    for (auto sh : std::vector<std::vector<int>>{{2,3,2},{4,8,16},{8,50,64},{3,7,5},{16,128,64},{1,9,4},{7,1,5}})
        test(sh, {1});
    printf("== 4D middle/various single axes ==\n");
    test({2,3,4,5},{0}); test({2,3,4,5},{1}); test({2,3,4,5},{2}); test({2,3,4,5},{3});
    test({8,16,32,4},{1}); test({8,16,32,4},{2});
    printf("== multi-axis (Generic path) ==\n");
    test({4,8,16},{0,2}); test({2,3,4,5},{1,3}); test({4,8,16},{0,1}); test({4,8,16},{1,2});
    printf("== large ==\n");
    test({1024,1024},{1}); test({1024,1024},{0}); test({64,512,64},{1});

    fclose(dump);
    printf("\n==== %d/%d shapes MATCH on both CPU and GPU ====\n", total-failed, total);
    return failed ? 1 : 0;
}
