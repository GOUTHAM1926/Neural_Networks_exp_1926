// Exercise the SHARED fixed paths (block_x_reduce + OuterContiguous remap) with
// non-sum ops (mean, max) and low-precision dtypes (fp16/bf16), CPU vs GPU vs
// an independent brute-force reference. Focuses on the previously-broken shapes.
#include "core/Tensor.h"
#include "ops/UnaryOps/Reduction.h"
#include "dtype/Types.h"
#include "device/DeviceCore.h"
#include <cstdio>
#include <vector>
#include <cmath>
#include <limits>

using namespace OwnTensor;

static uint64_t st = 0xDEADBEEFULL;
static float frand(){ st = st*6364136223846793005ULL + 1; return (float)((double)(uint32_t)(st>>32)/2147483648.0 - 1.0); }

static int total=0, failed=0;
enum Op { SUM, MEAN, MAX };

static std::vector<double> ref(const std::vector<int>& shape, int axis, const std::vector<float>& d, Op op){
    int nd=shape.size(); std::vector<char> red(nd,0); red[axis]=1;
    int n_out=1; for(int i=0;i<nd;++i) if(!red[i]) n_out*=shape[i];
    int reduced=shape[axis];
    std::vector<double> out(n_out, op==MAX? -1e300 : 0.0);
    int n_in=d.size();
    for(int lin=0; lin<n_in; ++lin){
        int rem=lin; std::vector<int> c(nd); for(int x=nd-1;x>=0;--x){c[x]=rem%shape[x];rem/=shape[x];}
        int oc=0; for(int x=0;x<nd;++x) if(!red[x]) oc=oc*shape[x]+c[x];
        if(op==MAX) out[oc]=std::max(out[oc],(double)d[lin]); else out[oc]+=d[lin];
    }
    if(op==MEAN) for(auto&v:out) v/=reduced;
    return out;
}

static void run(const std::vector<int>& shape, int axis, Op op, Dtype dt){
    total++;
    int n=1; for(int s:shape) n*=s;
    std::vector<float> data(n); for(auto&v:data) v=frand();
    auto r = ref(shape, axis, data, op);

    std::vector<int64_t> sh(shape.begin(),shape.end());
    Tensor t(Shape{sh}, TensorOptions().with_dtype(Dtype::Float32));
    t.set_data(data);
    if(dt!=Dtype::Float32) t = t.as_type(dt);

    auto call=[&](const Tensor& x){ return op==SUM?reduce_sum(x,{axis},false): op==MEAN?reduce_mean(x,{axis},false):reduce_max(x,{axis},false); };
    Tensor c = call(t);
    Tensor g = call(t.to_cuda(0)).to_cpu();
    if(dt!=Dtype::Float32){ c=c.as_type(Dtype::Float32); g=g.as_type(Dtype::Float32); }

    // fp16 has ~3 decimal digits; loosen tolerance for half/bf16 and for large sums.
    double atol = (dt==Dtype::Float32)?1e-3: (dt==Dtype::Float16?0.5:1.0);
    double rtol = (dt==Dtype::Float32)?1e-3: 2e-2;
    bool cok=(int)c.numel()==(int)r.size(), gok=(int)g.numel()==(int)r.size();
    double cmax=0,gmax=0;
    for(int i=0;i<(int)r.size();++i){
        double ce=std::fabs((double)c.data<float>()[i]-r[i]), ge=std::fabs((double)g.data<float>()[i]-r[i]);
        cmax=std::max(cmax,ce); gmax=std::max(gmax,ge);
        if(ce>atol+rtol*std::fabs(r[i])) cok=false;
        if(ge>atol+rtol*std::fabs(r[i])) gok=false;
    }
    if(!cok||!gok) failed++;
    const char* opn = op==SUM?"sum":op==MEAN?"mean":"max";
    const char* dn = dt==Dtype::Float32?"f32":dt==Dtype::Float16?"f16":"bf16";
    printf("%-4s %-4s shape[", opn, dn);
    for(size_t i=0;i<shape.size();++i) printf("%s%d", i?"x":"", shape[i]);
    printf("] ax%d  CPU %-5s GPU %-5s (err c=%.2e g=%.2e)\n", axis, cok?"OK":"FAIL", gok?"OK":"FAIL", cmax, gmax);
}

// quick multi-axis (Generic path) check for mean/max on GPU vs brute force
static void run_multi(const std::vector<int>& shape, std::vector<int> axes, Op op){
    total++;
    int nd=shape.size(); int n=1; for(int s:shape) n*=s;
    std::vector<float> data(n); for(auto&v:data) v=frand();
    std::vector<char> red(nd,0); for(int a:axes) red[a]=1;
    int n_out=1, reduced=1; for(int i=0;i<nd;++i){ if(red[i]) reduced*=shape[i]; else n_out*=shape[i]; }
    std::vector<double> r(n_out, op==MAX?-1e300:0.0);
    for(int lin=0;lin<n;++lin){ int rem=lin; std::vector<int> c(nd); for(int x=nd-1;x>=0;--x){c[x]=rem%shape[x];rem/=shape[x];}
        int oc=0; for(int x=0;x<nd;++x) if(!red[x]) oc=oc*shape[x]+c[x];
        if(op==MAX) r[oc]=std::max(r[oc],(double)data[lin]); else r[oc]+=data[lin]; }
    if(op==MEAN) for(auto&v:r) v/=reduced;
    std::vector<int64_t> sh(shape.begin(),shape.end()), ax(axes.begin(),axes.end());
    Tensor t(Shape{sh}, TensorOptions().with_dtype(Dtype::Float32)); t.set_data(data);
    auto call=[&](const Tensor& x){ return op==SUM?reduce_sum(x,ax,false):op==MEAN?reduce_mean(x,ax,false):reduce_max(x,ax,false); };
    Tensor g = call(t.to_cuda(0)).to_cpu();
    bool gok=(int)g.numel()==n_out; double gmax=0;
    for(int i=0;i<n_out&&gok;++i){ double e=std::fabs((double)g.data<float>()[i]-r[i]); gmax=std::max(gmax,e); if(e>1e-3+1e-3*std::fabs(r[i])) gok=false; }
    if(!gok) failed++;
    const char* opn=op==SUM?"sum":op==MEAN?"mean":"max";
    printf("%-4s multi shape[", opn); for(size_t i=0;i<shape.size();++i) printf("%s%d",i?"x":"",shape[i]);
    printf("] ax{"); for(size_t i=0;i<axes.size();++i) printf("%s%d",i?",":"",axes[i]); printf("} GPU %-5s (err %.2e)\n", gok?"OK":"FAIL", gmax);
}

int main(){
    std::vector<std::pair<std::vector<int>,int>> shapes = {
        {{5,768},1}, {{4,8,16},1}, {{8,50,64},1}, {{16,128,64},1}, {{128,50},0}
    };
    printf("=== mean / max (float32) on inner+middle+outer shapes ===\n");
    for(auto& s: shapes){ run(s.first,s.second,MEAN,Dtype::Float32); }
    for(auto& s: shapes){ run(s.first,s.second,MAX, Dtype::Float32); }
    printf("\n=== sum fp16 / bf16 ===\n");
    for(auto& s: shapes){ run(s.first,s.second,SUM,Dtype::Float16); }
    for(auto& s: shapes){ run(s.first,s.second,SUM,Dtype::Bfloat16); }
    printf("\n=== multi-axis Generic path: mean/max on GPU (bug #3 paths) ===\n");
    run_multi({4,8,16},{0,2},MEAN); run_multi({4,8,16},{0,2},MAX);
    run_multi({2,3,4,5},{1,3},MEAN); run_multi({2,3,4,5},{1,3},MAX);
    run_multi({8,16,32},{0,2},MEAN); run_multi({8,16,32},{0,2},MAX);

    printf("\n==== %d/%d op/dtype/shape combos OK on both CPU and GPU ====\n", total-failed, total);
    return failed?1:0;
}
