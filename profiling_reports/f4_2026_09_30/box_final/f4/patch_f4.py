#!/usr/bin/env python3
"""F4 fix: at W == 1 the gradient no longer goes through the wire (send) buffer.

At one GPU nothing is sent, yet all three ZeRO optimizers turned every gradient into the wire
dtype (bf16 on a bf16 model) and back to fp32 before the norm and the update read it: one pointless
rounding of the whole gradient per step. With this patch, at W == 1:
  v1  ShardedOptimizer           copies each fp32 gradient straight into shard_grad_fp32_
  v2  SymmetricShardedOptimizer  the same
  mu  ShardedMuon                the norm and the update read each parameter's own gradient in place
W > 1 is untouched (every change sits behind W_ == 1). A forced wire dtype (BLU_BF16_REDUCE /
BLU_FP32_REDUCE) has no effect at W == 1 and v2/Muon now print one line saying so.

Base: BluTrain dist_side_merge_BLUCCL a3aba3e (md5s below). Run from the BluTrain directory:
  python3 patch_f4.py --check    # every anchor must match exactly once; writes nothing
  python3 patch_f4.py --apply    # apply (git keeps the originals: `git checkout -- <file>` undoes)
"""
import hashlib
import sys

BASE_MD5 = {
    "dist/zero/include/ShardedOptimizer.h":          "fda13c98ee1ccdcad8236a5a0bb41039",
    "dist/zero/include/SymmetricShardedOptimizer.h": "c265646dde562513ae65b25570285996",
    "dist/zero/include/ShardedMuon.h":               "5ef40d39d3baf51ca0999ca673b61094",
    "dist/zero/src/ShardedOptimizer.cpp":            "18d9acf2d354493167e9cf37a705f646",
    "dist/zero/src/SymmetricShardedOptimizer.cpp":   "313a52100c0d9b0ef2d312e9b33916d0",
    "dist/zero/src/ShardedMuon.cpp":                 "4601d379cd54c192ba2e470861a2c83e",
}

# ---------------------------------------------------------------------------------------------
# v1: ShardedOptimizer
# ---------------------------------------------------------------------------------------------
V1_H = "dist/zero/include/ShardedOptimizer.h"
V1_H_A = "    void cast_shard_grad_to_fp32();              // shard_grad_ -> shard_grad_fp32_\n"
V1_H_N = V1_H_A + "    void stage_grads_direct_fp32();              // W == 1 only (F4): grads -> shard_grad_fp32_\n"

V1_C = "dist/zero/src/ShardedOptimizer.cpp"
V1_FN_A = "void ShardedOptimizer::local_adam_step() {\n"
V1_FN_N = r'''// F4: at W == 1 nothing is sent, so the gradient does not go through the wire buffer.
// What: copy each parameter's gradient straight into shard_grad_fp32_, the buffer the unscale,
//       the norm, the clip and Adam read. Autograd keeps the gradient of a bf16/fp16 parameter
//       in fp32, so this is an fp32 -> fp32 copy (a lower-precision gradient is widened, which is
//       exact). At W == 1 the shard is the whole flat space (shard_begin_ == 0, S_ ==
//       padded_numel_), so a parameter's flat offset is also its shard offset.
// Was:  gather_grads_into_flat() (fp32 -> compute dtype into flat_stage_), a plain W == 1 copy
//       into shard_grad_ (reduce_scatter_grads), then cast_shard_grad_to_fp32() (back to fp32).
//       On a bf16 group that rounded the whole gradient to bf16 once per step although nothing
//       was sent.
// Why:  the optimizer sees the exact accumulated gradient, and in one pass instead of three.
void ShardedOptimizer::stage_grads_direct_fp32() {
    float* base = shard_grad_fp32_.data<float>();
    for (const auto& rec : records_) {
        float* dst = base + rec.offset;
        if (rec.param.has_grad()) {
            Tensor g = rec.param.grad_view();
            if (!g.is_contiguous()) g = g.contiguous();
            if (g.dtype() == Dtype::Float32) {
                d2d(dst, g.data(), rec.numel * sizeof(float), device_);
            } else {
                OwnTensor::convert_type_cuda_generic(g.data(), g.dtype(), dst, Dtype::Float32,
                                                     static_cast<int64_t>(rec.numel),
                                                     ocuda::getCurrentStream());
            }
        } else {
            // param requires grad but produced none this step (unused): gradient 0,
            // exactly as gather_grads_into_flat() treats it.
            zero_async(dst, rec.numel * sizeof(float));
        }
    }
    if (padded_numel_ > total_numel_) {
        zero_async(base + total_numel_, (padded_numel_ - total_numel_) * sizeof(float));
    }
}

''' + V1_FN_A

V1_RG_A = '''    } else {
        // Reference path: either overlap is off, or the caller did not arm this
        // pass — the (inert) hooks left grads untouched, so gather them normally.
        gather_grads_into_flat();
        reduce_scatter_grads();
    }
    cast_shard_grad_to_fp32();
'''
V1_RG_N = '''    } else if (W_ == 1) {
        // F4: at W == 1 nothing is sent, so the gradients skip the wire buffer and go
        // straight into shard_grad_fp32_ (see stage_grads_direct_fp32). Nothing to cast.
        stage_grads_direct_fp32();
    } else {
        // Reference path: either overlap is off, or the caller did not arm this
        // pass — the (inert) hooks left grads untouched, so gather them normally.
        gather_grads_into_flat();
        reduce_scatter_grads();
    }
    if (W_ > 1) cast_shard_grad_to_fp32();   // W == 1 wrote fp32 directly (F4)
'''

# ---------------------------------------------------------------------------------------------
# v2: SymmetricShardedOptimizer
# ---------------------------------------------------------------------------------------------
V2_H = "dist/zero/include/SymmetricShardedOptimizer.h"
V2_H_A = "    void cast_shard_grad_to_fp32();\n"
V2_H_N = V2_H_A + "    void stage_grads_direct_fp32();         // W == 1 only (F4): grads -> shard_grad_fp32_\n"

V2_C = "dist/zero/src/SymmetricShardedOptimizer.cpp"
V2_FN_A = ("// ======================================================================\n"
           "// One combined scale pass: 1/W (undo op=sum) * loss-scale inverse * clip coef\n")
V2_FN_N = r'''// F4: at W == 1 nothing is sent, so the gradient does not go through the wire buffer.
// What: copy each parameter's gradient straight into shard_grad_fp32_, the buffer the norm, the
//       scale pass and Adam read. Autograd keeps the gradient of a bf16/fp16 parameter in fp32,
//       so this is an fp32 -> fp32 copy (a lower-precision gradient is widened, which is exact).
//       At W == 1 the shard is the whole flat space (S_ == padded_numel_, and build_buckets()
//       gives every bucket shard_offset == begin), so a flat offset is also a shard offset.
// Was:  gather_grads_into_flat() (fp32 -> wire dtype into flat_grad_), then
//       cast_shard_grad_to_fp32() (wire dtype -> fp32). On a bf16 group that rounded the whole
//       gradient to bf16 once per step although nothing was sent -- and a forced wire dtype
//       (BLU_BF16_REDUCE) did the same to an fp32 group.
// Why:  the optimizer sees the exact accumulated gradient, and in one pass instead of two.
void SymmetricShardedOptimizer::stage_grads_direct_fp32() {
    float* base = shard_grad_fp32_.data<float>();
    for (const auto& rec : records_) {
        float* dst = base + rec.offset;
        if (rec.param.has_grad()) {
            Tensor g = rec.param.grad_view();
            if (!g.is_contiguous()) g = g.contiguous();
            if (g.dtype() == Dtype::Float32) {
                d2d(dst, g.data(), rec.numel * sizeof(float), device_);
            } else {
                OwnTensor::convert_type_cuda_generic(g.data(), g.dtype(), dst, Dtype::Float32,
                                                     static_cast<int64_t>(rec.numel),
                                                     ocuda::getCurrentStream());
            }
        } else {
            zero_async(dst, rec.numel * sizeof(float));   // unused this step: gradient 0
        }
    }
    if (padded_numel_ > total_numel_) {
        zero_async(base + total_numel_, (padded_numel_ - total_numel_) * sizeof(float));
    }
    sync_compute();
}

''' + V2_FN_A

V2_RG_A = '''void SymmetricShardedOptimizer::reduce_grads_now() {
    if (overlap_active_ && armed_) {
        finalize_overlap_reduce();
    } else {
        gather_grads_into_flat();
        reduce_scatter_grads();
    }
    cast_shard_grad_to_fp32();
    grads_reduced_ = true;
}
'''
V2_RG_N = '''void SymmetricShardedOptimizer::reduce_grads_now() {
    if (overlap_active_ && armed_) {
        finalize_overlap_reduce();
    } else if (W_ == 1) {
        // F4: at W == 1 nothing is sent, so the gradients skip the wire buffer and go
        // straight into shard_grad_fp32_ (see stage_grads_direct_fp32). Nothing to cast.
        stage_grads_direct_fp32();
    } else {
        gather_grads_into_flat();
        reduce_scatter_grads();
    }
    if (W_ > 1) cast_shard_grad_to_fp32();   // W == 1 wrote fp32 directly (F4)
    grads_reduced_ = true;
}
'''

V2_MSG_A = '                      << ", shard_grad_ " << shard_delta << " MB]" << std::endl;\n'
V2_MSG_N = V2_MSG_A + '''            // F4: at W == 1 stage_grads_direct_fp32() never touches the wire buffer, so the
            // forced dtype changes nothing. Say so, rather than let the flag look active.
            if (W_ == 1)
                std::cout << "[ZeRO-1 v2] that wire dtype has no effect at W=1: nothing is "
                             "sent, Adam reads the fp32 gradients directly (F4)" << std::endl;
'''

# ---------------------------------------------------------------------------------------------
# mu: ShardedMuon
# ---------------------------------------------------------------------------------------------
MU_H = "dist/zero/include/ShardedMuon.h"
MU_H_A1 = "    void stage_all_grads_reference();      // non-overlap path\n"
MU_H_N1 = MU_H_A1 + "    void prepare_direct_grads();           // W == 1 only (F4): readers use the model grads\n"
MU_H_A2 = "    std::vector<Tensor> reduce_keepalive_;\n"
MU_H_N2 = MU_H_A2 + '''    // W == 1 only (F4): each owned slot's gradient, contiguous, where autograd left it.
    // Read by accumulate_owned_grad_norm() and owned_grad_fp32(); invalid = no gradient.
    std::vector<Tensor> direct_grads_;
'''

MU_C = "dist/zero/src/ShardedMuon.cpp"
MU_RG_A = "void ShardedMuon::reduce_grads_now() {\n    if (rsag()) {\n"
MU_RG_N = '''void ShardedMuon::reduce_grads_now() {
    if (W_ == 1) {
        // F4: at W == 1 nothing is sent -- nothing is staged into the wire buffers; the norm
        // and the update read each parameter's own gradient (see prepare_direct_grads).
        prepare_direct_grads();
    } else if (rsag()) {
'''

MU_FN_A = "void ShardedMuon::stage_all_grads_reference() {\n"
MU_FN_N = r'''// F4: at W == 1 nothing is sent, so the gradients do not go through the wire buffers.
// What: keep a contiguous handle on each owned parameter's gradient, where autograd left it
//       (fp32 for a bf16/fp16 parameter: autograd keeps those gradients in fp32).
//       accumulate_owned_grad_norm() and owned_grad_fp32() read it from there, in its own
//       dtype. An invalid handle means "no gradient this step" and reads as zero.
// Was:  stage_all_grads_reference() / reduce_all_grads_reference() converted every gradient to
//       the wire dtype (flat_grad_ / flat_owned_grad_) and the readers widened it back: on a
//       bf16 model the whole gradient was rounded to bf16 once per step although nothing was
//       sent. Both collective modes (RS/AG and reduce/broadcast) take this path at W == 1, so
//       they still agree bit for bit (test_muon_rsag).
// Why:  the norm and the update see the exact accumulated gradient, with no staging pass.
void ShardedMuon::prepare_direct_grads() {
    direct_grads_.assign(owned_.size(), Tensor());
    for (size_t s = 0; s < owned_.size(); ++s) {
        const auto& rec = records_[owned_[s]];
        if (!rec.param.has_grad()) continue;
        Tensor g = rec.param.grad_view();
        direct_grads_[s] = g.is_contiguous() ? g : g.contiguous();
    }
}

''' + MU_FN_A

MU_OG_A = '''    Tensor dst = grad_scratch_.slice_inplace(0, rec.numel).view(rec.param.shape());
    const kernels::ElemKind kind = elem_kind(reduce_dtype_);
'''
MU_OG_N = '''    Tensor dst = grad_scratch_.slice_inplace(0, rec.numel).view(rec.param.shape());
    // F4: at W == 1 the gradient is read where autograd left it (prepare_direct_grads), in its
    // own dtype -- fp32 for a bf16/fp16 parameter -- instead of from the wire buffer.
    Dtype src_dtype = reduce_dtype_;
    if (W_ == 1) {
        const Tensor& g = direct_grads_.at(slot);
        if (!g.is_valid()) {   // no gradient this step: it is zero
            cudaMemsetAsync(dst.data(), 0, rec.numel * sizeof(float), ocuda::getCurrentStream());
            return dst;
        }
        src       = g.data();
        src_dtype = g.dtype();
    }
    const kernels::ElemKind kind = elem_kind(src_dtype);
'''
MU_OG2_A = "            src, reduce_dtype_, dst.data<float>(), Dtype::Float32,\n"
MU_OG2_N = "            src, src_dtype, dst.data<float>(), Dtype::Float32,\n"

MU_NORM_A = '''    const kernels::ElemKind kind = elem_kind(reduce_dtype_);
    if (kind == kernels::ElemKind::UNSUPPORTED) {
'''
MU_NORM_N = '''    if (W_ == 1) {
        // F4: read each owned gradient where autograd left it (prepare_direct_grads), in its own
        // dtype. One launch per owned tensor in slot order, as below, so it stays deterministic.
        for (size_t s = 0; s < owned_.size(); ++s) {
            const Tensor& g = direct_grads_.at(s);
            if (!g.is_valid()) continue;   // no gradient this step: adds 0
            const int64_t n = static_cast<int64_t>(records_[owned_[s]].numel);
            const kernels::ElemKind gk = elem_kind(g.dtype());
            if (gk == kernels::ElemKind::UNSUPPORTED) {
                // Generic fallback; the sync protects multi_tensor_*'s static descriptor buffer.
                Tensor f = owned_grad_fp32(s, /*apply_clip=*/false);
                std::vector<ocuda::TensorInfo> gi{ ocuda::TensorInfo{ f.data<float>(), n } };
                ocuda::multi_tensor_grad_norm_cuda(gi, d_out);
                sync_compute();
                continue;
            }
            kernels::sumsq_accumulate(g.data(), gk, n, grad_prescale_, d_out, d_sumsq_partials_,
                                      ocuda::getCurrentStream());
        }
        return;
    }

''' + MU_NORM_A

MU_MSG_A = "    reduce_dtype_ = resolve_reduce_dtype(cfg_.reduce_dtype, compute_dtype_);\n"
MU_MSG_N = MU_MSG_A + '''    // F4: at W == 1 nothing is sent (see prepare_direct_grads), so a forced wire dtype
    // (BLU_BF16_REDUCE / BLU_FP32_REDUCE) changes nothing. Say so, rather than let it look active.
    if (W_ == 1 && reduce_dtype_ != compute_dtype_ && cfg_.verbose && rank_ == 0)
        std::cout << "[ZeRO-1/Muon] reduce dtype " << OwnTensor::get_dtype_name(reduce_dtype_)
                  << " has no effect at W=1: nothing is sent, the gradients are read in their "
                     "own dtype (F4)" << std::endl;
'''

EDITS = [
    (V1_H, "v1 header: declare stage_grads_direct_fp32", V1_H_A, V1_H_N),
    (V1_C, "v1: stage_grads_direct_fp32()", V1_FN_A, V1_FN_N),
    (V1_C, "v1: reduce_grads_now W==1 branch", V1_RG_A, V1_RG_N),
    (V2_H, "v2 header: declare stage_grads_direct_fp32", V2_H_A, V2_H_N),
    (V2_C, "v2: stage_grads_direct_fp32()", V2_FN_A, V2_FN_N),
    (V2_C, "v2: reduce_grads_now W==1 branch", V2_RG_A, V2_RG_N),
    (V2_C, "v2: forced-dtype notice at W==1", V2_MSG_A, V2_MSG_N),
    (MU_H, "mu header: declare prepare_direct_grads", MU_H_A1, MU_H_N1),
    (MU_H, "mu header: direct_grads_ member", MU_H_A2, MU_H_N2),
    (MU_C, "mu: reduce_grads_now W==1 branch", MU_RG_A, MU_RG_N),
    (MU_C, "mu: prepare_direct_grads()", MU_FN_A, MU_FN_N),
    (MU_C, "mu: owned_grad_fp32 reads the direct grad", MU_OG_A, MU_OG_N),
    (MU_C, "mu: owned_grad_fp32 fallback uses src_dtype", MU_OG2_A, MU_OG2_N),
    (MU_C, "mu: accumulate_owned_grad_norm reads the direct grads", MU_NORM_A, MU_NORM_N),
    (MU_C, "mu: forced-dtype notice at W==1", MU_MSG_A, MU_MSG_N),
]


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode not in ("--check", "--apply"):
        sys.exit(__doc__)
    src = {}
    for f, want in BASE_MD5.items():
        try:
            text = open(f, encoding="utf-8").read()
        except FileNotFoundError:
            sys.exit(f"ABORT: {f} not found; run from the BluTrain directory")
        md5 = hashlib.md5(text.encode("utf-8")).hexdigest()
        ok = md5 == want
        print(f"{f}: md5 {md5} ({'= a3aba3e' if ok else 'DIFFERS from a3aba3e'})")
        if not ok:
            sys.exit("ABORT: a file differs from the base this patch was written against; nothing written")
        if "F4:" in text:
            sys.exit(f"ABORT: {f} already contains F4 changes; nothing written")
        src[f] = text
    out = dict(src)
    for f, label, anchor, new in EDITS:
        n = out[f].count(anchor)
        print(f"  {label}: anchor found {n} time(s)")
        if n != 1:
            sys.exit(f"ABORT: '{label}' anchor must match exactly once; nothing written")
        out[f] = out[f].replace(anchor, new)
    if mode == "--check":
        print(f"check OK: all {len(EDITS)} anchors match once. Nothing written.")
        return
    for f in BASE_MD5:
        open(f, "w", encoding="utf-8").write(out[f])
        print(f"wrote {f}: md5 {hashlib.md5(out[f].encode('utf-8')).hexdigest()}")
    print(f"applied: {len(EDITS)} edits in {len(BASE_MD5)} files")


if __name__ == "__main__":
    main()
