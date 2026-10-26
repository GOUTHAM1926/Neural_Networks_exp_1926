// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different template instantiations to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

// 2026-10-19 MEASUREMENT BUILD (work doc 45.51): this translation unit (bf16, head dim 128, no split / paged / softcap /
// PackGQA = the dense benchmark's kernel) gets the per-step clock stamps of mainloop_fwd_sm90_tma_gmma_ws.hpp, level 3.
#define FA3_TL 3
#include "flash_fwd_launch_template.h"

#ifndef FLASHATTENTION_DISABLE_HDIM128
template void run_mha_fwd_<90, cutlass::bfloat16_t, 128, 128, false, false, false, false>(Flash_fwd_params &params, cudaStream_t stream);
#endif

#ifdef FA3_TL
// copy the stamps of the last launch to the host (Python: ctypes on flash_attn_3/_C*.so)
extern "C" __attribute__((visibility("default"))) int fa3_tl_read(void* host, size_t bytes) {
    return (int)cudaMemcpyFromSymbol(host, flash::g_fa3_tl, bytes);
}
extern "C" __attribute__((visibility("default"))) int fa3_tl_words() {
    return (int)(sizeof(flash::g_fa3_tl) / sizeof(unsigned int));
}
extern "C" __attribute__((visibility("default"))) int fa3_tl_level() { return FA3_TL; }
#endif
