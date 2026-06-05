// Find smallest shapes that isolate the vec4 bug vs the multi-CTA bug.
#include "core/Tensor.h"
#include "ops/UnaryOps/Reduction.h"
#include "dtype/Types.h"
#include <cstdio>
#include <vector>
#include <cmath>
using namespace OwnTensor;

static void test(int R, int C) {  // [R,C] reduce axis=0 -> C column sums
    std::vector<float> d((size_t)R*C);
    uint64_t s=0x1234; for(auto&v:d){ s=s*6364136223846793005ULL+1; v=(float)((double)(uint32_t)(s>>32)/2147483648.0-1.0); }
    Tensor t(Shape{{R,C}}, TensorOptions().with_dtype(Dtype::Float32)); t.set_data(d);
    Tensor g = reduce_sum(t.to_cuda(0), {0}, false).to_cpu();
    std::vector<double> ref(C,0.0);
    for(int r=0;r<R;++r)for(int c=0;c<C;++c) ref[c]+=d[(size_t)r*C+c];
    int bad=0; double mx=0; for(int c=0;c<C;++c){double e=std::fabs((double)g.data<float>()[c]-ref[c]); mx=std::max(mx,e); if(e>1e-2+1e-3*std::fabs(ref[c]))bad++;}
    printf("[%d,%d] ax0  out=%d  bad=%d/%d  maxerr=%.3f  gpu[1..4]=%.3f %.3f %.3f %.3f  ref[1..4]=%.3f %.3f %.3f %.3f\n",
        R,C,C,bad,C,mx, g.data<float>()[0],g.data<float>()[1],g.data<float>()[2],g.data<float>()[3],
        ref[0],ref[1],ref[2],ref[3]);
}
int main(){
    printf("--- the known failure ---\n");
    test(1024,1024);
    printf("--- vary num_outputs (C), fixed large reduced R=1024 ---\n");
    test(1024,4); test(1024,64); test(1024,128); test(1024,256); test(1024,512);
    printf("--- vary reduced (R), fixed large num_outputs C=1024 ---\n");
    test(4,1024); test(64,1024); test(256,1024); test(512,1024);
    printf("--- both moderately large ---\n");
    test(256,256); test(512,512); test(2048,2048);
    return 0;
}
