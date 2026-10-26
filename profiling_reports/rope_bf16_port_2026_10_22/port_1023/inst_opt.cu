// SASS reference for the port check: the kernels measured on 10-22 (opt/ round 1 bwd, opt/ round 2 reduce3),
// explicitly instantiated so their SASS can be compared with the library port.
#include "../opt/FusedRMSRoPE_bf16_opt.cu"
#include "../opt/FusedRMSRoPE_bf16_v2.cu"
namespace OwnTensor { namespace cuda {
template __global__ void ropt::bwd<true, 128, true>(const bf16*, const bf16*, const bf16*, const bf16*, const float*, const float*, const float*, const float*, const float*, bf16*, bf16*, float*, float*, int, int, int, int, const int32_t*);
template __global__ void ropt::bwd<false, 128, true>(const bf16*, const bf16*, const bf16*, const bf16*, const float*, const float*, const float*, const float*, const float*, bf16*, bf16*, float*, float*, int, int, int, int, const int32_t*);
template __global__ void ropt::bwd<true, 64, true>(const bf16*, const bf16*, const bf16*, const bf16*, const float*, const float*, const float*, const float*, const float*, bf16*, bf16*, float*, float*, int, int, int, int, const int32_t*);
template __global__ void ropt::bwd<false, 64, true>(const bf16*, const bf16*, const bf16*, const bf16*, const float*, const float*, const float*, const float*, const float*, bf16*, bf16*, float*, float*, int, int, int, int, const int32_t*);
template __global__ void ropt2::reduce3<128>(const float*, float*, int, const float*, float*, int, int);
}}
