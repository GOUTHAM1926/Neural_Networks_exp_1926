# kernel name -> short label (shared by bwd_breakdown.py)
import csv,sys,re
def short(n):
    rules=[("gqa_bwd_vj2_kernel<(bool)0>","OURS_bwd_main_dense"),("gqa_bwd_vj2_kernel<(bool)1>","OURS_bwd_main_packed"),
     ("bwd_prep_vj2_chats","ours_prep_chats"),("bwd_prep_vj2","ours_prep"),("gqa_dkdv_greduce_vj2","ours_dkdv_headsum"),("convert_dq_frag_vj2","ours_dq_convert"),
     ("gqa_v62_sm90","ours_fwd"),("k_fill_normal","fill_normal"),
     ("FlashAttnBwdSm90","FA3_bwd_main"),("FlashAttnFwdSm90","FA3_fwd"),("FlashAttnBwdPreprocess","fa3_prep"),
     ("FlashAttnBwdPostprocessConvertdQ<cute::tuple<cute::C<(int)64>","fa3_dq_convert"),("FlashAttnBwdPostprocessConvertdQ<cute::tuple<cute::C<(int)128>","fa3_dkdv_convert"),
     ("prepare_varlen_num_blocks","fa3_varlen_sched"),
     ("FlashAttentionBackwardSm90","FA4_bwd_main"),("FlashAttentionForwardSm90","FA4_fwd"),("FlashAttentionBackwardPreprocess","fa4_prep"),("FlashAttentionBackwardPostprocess","fa4_post"),
     ("flash_bprop","CUDNN_bwd_main"),("flash_fprop","CUDNN_fwd"),("compute_dot_do_o","cudnn_prep"),("convert_dq_to_16bits","cudnn_dq_convert"),("fmha_reduce_head","cudnn_dkdv_headsum"),("qkv_tma_setup","cudnn_tma_setup"),
     ("flash_bwd_dq_dk_dv","FA2_bwd_main"),("flash_fwd_kernel","FA2_fwd"),("flash_bwd_dot_do_o","fa2_prep"),("flash_bwd_convert_dq","fa2_dq_convert"),
     ("FillFunctor<float>","torch_zero_f32"),("FillFunctor","torch_fill"),("direct_copy","torch_copy"),("reduce_kernel","torch_reduce"),("[CUDA memset]","memset"),("[CUDA memcpy","memcpy")]
    for k,v in rules:
        if k in n: return v
    return "other:"+n[:50]
