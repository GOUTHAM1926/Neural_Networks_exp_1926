// CPU-only probe: confirm middle-axis reduction bug (#1) in BBP without GPU.
#include "core/Tensor.h"
#include "ops/UnaryOps/Reduction.h"
#include "dtype/Types.h"
#include <cstdio>
#include <vector>
using namespace OwnTensor;

int main() {
    // [2,3,2] reduce axis=1 (middle, outer=2>1), data = 1..12
    const int A=2,B=3,C=2; std::vector<float> d(A*B*C);
    for (int i=0;i<A*B*C;++i) d[i]=i+1;
    Tensor t(Shape{{A,B,C}}, TensorOptions().with_dtype(Dtype::Float32));
    t.set_data(d);
    Tensor r = reduce_sum(t, {1}, false);   // CPU path
    // reference
    float ref[4]={0};
    for(int a=0;a<A;++a)for(int c=0;c<C;++c)for(int b=0;b<B;++b) ref[a*C+c]+=d[a*B*C+b*C+c];
    printf("CPU middle-axis reduce_sum  [2,3,2] axis=1\n");
    printf("  ref: %.0f %.0f %.0f %.0f\n", ref[0],ref[1],ref[2],ref[3]);
    printf("  bbp: %.0f %.0f %.0f %.0f\n", r.data<float>()[0],r.data<float>()[1],r.data<float>()[2],r.data<float>()[3]);
    bool ok=true; for(int i=0;i<4;++i) ok &= (r.data<float>()[i]==ref[i]);
    printf("  -> %s\n", ok?"CORRECT":"*** BUG #1 PRESENT (middle-axis wrong on CPU) ***");
    return 0;
}
