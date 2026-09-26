#!/usr/bin/env python3
"""F1 for ShardedMuon (BLU_OPTIMIZER=muon) and SymmetricShardedOptimizer (BLU_ZERO_V2=1).

Same fix v1 (ShardedOptimizer) got in a264586: a NaN/Inf gradient norm with no LossScaler
installed now raises ZeroNonFiniteGradError BEFORE any update, instead of being written into
the master weights. With a LossScaler: unchanged (skip + back off).

Run from the BluTrain root:  python3 patch_f1_muon_v2.py
Refuses to touch anything unless every file has the expected md5 (BluTrain e3633e8) and every
pattern matches exactly once. Writes nothing on any failure.
"""
import hashlib, sys

EXPECT = {
    "dist/zero/src/ShardedMuon.cpp":               "5b75c6e75590fdc31857d07c77737dbe",
    "dist/zero/src/SymmetricShardedOptimizer.cpp": "603c483e683a3a34348ddd2fa8373138",
    "bluscript_zero_ckpt.cpp":                     "a00f18e00cf649e3e53b24a8f8ddbe08",
}

# ---------------------------------------------------------------------------------------------
# shared helper text (file-local, anonymous namespace: no header / class-layout change)
# ---------------------------------------------------------------------------------------------
def helper(who):
    return f'''
// F1. Text for ZeroNonFiniteGradError: which optimizer and group, how big, which rank and
// step, so the log names the failing group. Same wording as v1's
// ShardedOptimizer::nonfinite_grad_msg. File-local on purpose: a new member function would
// change the header, and every trainer that includes it would need a rebuild.
std::string nonfinite_grad_msg(float nsq, Dtype dt, size_t numel, int rank, int64_t next_step) {{
    const char* g = dt == Dtype::Bfloat16 ? "bf16" : dt == Dtype::Float32 ? "fp32" : "other";
    return std::string("{who}: grad norm^2 = ") + std::to_string(nsq) + " in the " + g +
           " group (" + std::to_string(numel) + " params), rank " + std::to_string(rank) +
           ", optimizer step " + std::to_string(next_step) +
           "; no LossScaler installed, so the step cannot be skipped -- raised BEFORE any update";
}}
'''

EDITS = {
# =============================================================================================
"dist/zero/src/ShardedMuon.cpp": [
# ---- 1. helper ------------------------------------------------------------------------------
("""namespace {

inline size_t elem_size(Dtype d) { return Tensor::dtype_size(d); }
""",
"""namespace {

inline size_t elem_size(Dtype d) { return Tensor::dtype_size(d); }
""" + helper("ShardedMuon")),
# ---- 2. cached-norm path (BLU_GLOBAL_CLIP=1) -------------------------------------------------
("""        last_grad_norm_ = (nsq > 0.0f) ? std::sqrt(nsq) : 0.0f;
        if (sc && !std::isfinite(nsq)) { sc->update(true); last_step_skipped_ = true; return false; }
""",
"""        // F1 (cached-norm path, BLU_GLOBAL_CLIP=1). Same change as the guard in the
        // per-step path below; the full explanation is there.
        //   was: last_grad_norm_ = (nsq > 0) ? sqrt(nsq) : 0   -> a NaN norm was LOGGED AS 0
        //   was: if (sc && !isfinite(nsq)) skip               -> no scaler installed = never fires
        // reduce_and_measure() caches only a finite norm, so this guard cannot fire today; it is
        // kept in the same form as the one below so the two paths cannot drift apart.
        last_grad_norm_ = (nsq > 0.0f || std::isnan(nsq)) ? std::sqrt(nsq) : 0.0f;   // NaN logs as nan, not 0
        if (!std::isfinite(nsq)) {
            if (!sc) throw ZeroNonFiniteGradError(nonfinite_grad_msg(nsq, compute_dtype_, total_numel_,
                                                                     rank_, step_count_ + 1));
            sc->update(true); last_step_skipped_ = true; return false;
        }
"""),
# ---- 3. per-step path (the default) ---------------------------------------------------------
("""    last_grad_norm_ = (nsq > 0.0f) ? std::sqrt(nsq) : 0.0f;

    // --- global non-finite guard: atomic step skip (params untouched) ---
    if (sc && !std::isfinite(nsq)) {
        sc->update(true);
        last_step_skipped_ = true;
        return false;
    }
    }   // end of the "no cached norm" branch
""",
"""    last_grad_norm_ = (nsq > 0.0f || std::isnan(nsq)) ? std::sqrt(nsq) : 0.0f;   // NaN logs as nan, not 0

    // --- global non-finite guard: atomic step skip (params untouched) ---
    // F1 (the default path; same fix as ShardedOptimizer::step() got in v1).
    // Was:  `if (sc && !std::isfinite(nsq)) { skip }`. set_scaler() is called only from
    //       dist/zero/tests, so in training `sc` is null and this guard NEVER fired. A NaN/Inf
    //       gradient then went on into the update:
    //         1. nsq = NaN or Inf (multi_tensor_grad_norm keeps NaN/Inf); the old
    //            `(nsq > 0) ? sqrt(nsq) : 0` logged a NaN norm as 0
    //         2. Muon-path tensors (every 2-D matrix, i.e. the body) -- SILENT: update_one() applies
    //            weight decay, then muon_momentum_cuda writes the NaN/Inf into the momentum m IN
    //            PLACE, and only then does muon_has_nonfinite_cuda reject the Newton-Schulz output.
    //            The weights stay finite, but m keeps the NaN (m = mu*m + ..., so it never leaves):
    //            every LATER step of that matrix is rejected too, and it silently stops learning
    //            (weight decay only). A checkpoint saves that m, and auto-resume reloads it.
    //         3. AdamW-path tensors: NaN -> last_grad_norm_ = 0 -> clip_coef_ = min(1, max/1e-6) = 1,
    //            no clip, fused Adam writes NaN into master/m/v. Inf -> clip_coef_ = 0, and
    //            Inf * 0 = NaN, same result.
    //         4. a NaN weight makes the next forward's loss NaN -> the trainer aborts one step LATE;
    //            if this step was a checkpoint step, the NaN state was already saved.
    //       Measured before this fix (test_f1_nonfinite, W=1, bf16 and fp32, 256x192 + 128x256
    //       matrices): NaN or Inf in one matrix gradient -> no error, 0 NaN weights, the NaN kept
    //       in m, and that matrix's update rejected on this step AND on the next finite step (the
    //       finite control: 0 and 0 rejected); NaN in a vector -> NaN written into the weight.
    //       BLU_GLOBAL_CLIP=1 did not help: reduce_and_measure() returns -1 for a non-finite norm,
    //       the trainer then only drops the global coefficient and still calls step(), which
    //       arrives here (cached_nsq_ = -1), measures the same NaN again, and went on.
    // Why here: this one number is already computed and all-reduced every step (it is logged
    //       even with clipping off), so the check is free (one isfinite on a host float, once
    //       per group per step), it runs BEFORE update_owned() and before step_count_ advances,
    //       and every rank holds the same all-reduced value, so every rank takes the same branch
    //       (no rank is left waiting in a later collective).
    // Now:  with a LossScaler, unchanged (skip + back off). Without one, throw
    //       ZeroNonFiniteGradError; bluscript_zero_ckpt catches it and aborts with exit code 4,
    //       so no poisoned weights and no poisoned checkpoint. A finite norm takes exactly the
    //       old code path, so training is bit-identical.
    if (!std::isfinite(nsq)) {
        if (!sc) throw ZeroNonFiniteGradError(nonfinite_grad_msg(nsq, compute_dtype_, total_numel_,
                                                                 rank_, step_count_ + 1));
        sc->update(true);
        last_step_skipped_ = true;
        return false;
    }
    }   // end of the "no cached norm" branch
"""),
],
# =============================================================================================
"dist/zero/src/SymmetricShardedOptimizer.cpp": [
# ---- 1. helper ------------------------------------------------------------------------------
("""} // anonymous namespace
""",
helper("SymmetricShardedOptimizer") + """
} // anonymous namespace
"""),
# ---- 2. per-step path (the only guarded one) ------------------------------------------------
("""        if (sc && !std::isfinite(nsq)) {
            sc->update(true);
            last_step_skipped_ = true;
            return false;
        }
        last_grad_norm_ = nsq > 0.0f ? std::sqrt(nsq) : 0.0f;
""",
"""        // F1 (same fix as ShardedOptimizer::step() got in v1).
        // Was:  `if (sc && !std::isfinite(nsq)) { skip }`, and last_grad_norm_ set AFTER it.
        //       set_scaler() is called only from dist/zero/tests, so in training `sc` is null and
        //       this guard NEVER fired. A NaN/Inf gradient then went on into Adam:
        //         1. nsq = NaN or Inf (multi_tensor_grad_norm keeps NaN/Inf)
        //         2. NaN: the old `nsq > 0 ? sqrt(nsq) : 0` set last_grad_norm_ = 0 (logged as 0),
        //            so coef = max / (0 + 1e-6), clamped to 1 -> no clip, the NaN elements go on.
        //            Inf: last_grad_norm_ = Inf -> coef = 0 -> every gradient is scaled by 0, and
        //            Inf * 0 = NaN on the Inf elements.
        //         3. local_adam_step writes NaN into master/m/v for those elements
        //         4. a NaN weight makes the next forward's loss NaN -> the trainer aborts one step
        //            LATE; if this step was a checkpoint step, the NaN state was already saved and
        //            auto-resume reloads it.
        //       Measured before this fix (test_f1_nonfinite, W=1, bf16 and fp32): NaN or Inf in one
        //       gradient element -> no error, step_count advanced, that weight written as NaN.
        //       BLU_GLOBAL_CLIP=1 did not help: reduce_and_measure() returns -1 for a non-finite norm,
        //       the trainer then only drops the global coefficient and still calls step(), which
        //       arrives here (cached_nsq_ = -1), measures the same NaN again, and went on.
        // Why here: this norm is already computed and all-reduced for clipping, so the check is
        //       free (one isfinite on a host float, once per group per step), it runs BEFORE the
        //       scale and before local_adam_step / step_count_, and every rank holds the same
        //       all-reduced value, so every rank takes the same branch.
        // Now:  with a LossScaler, unchanged (skip + back off). Without one, throw
        //       ZeroNonFiniteGradError; bluscript_zero_ckpt catches it and aborts with exit code 4.
        //       last_grad_norm_ is now set before the guard, as in v1, so a skipped step logs the
        //       non-finite norm instead of the previous step's. A finite norm takes exactly the old
        //       code path, so training is bit-identical.
        // Not covered (same in v1): with max_grad_norm = 0 AND no scaler, need_norm is false and
        //       no norm is measured at all, so nothing can be checked. The trainers set
        //       max_grad_norm = cfg.grad_clip, default 1.0.
        last_grad_norm_ = (nsq > 0.0f || std::isnan(nsq)) ? std::sqrt(nsq) : 0.0f;   // NaN logs as nan, not 0
        if (!std::isfinite(nsq)) {
            if (!sc) throw ZeroNonFiniteGradError(nonfinite_grad_msg(nsq, compute_dtype_, total_numel_,
                                                                     rank_, step_count_ + 1));
            sc->update(true);
            last_step_skipped_ = true;
            return false;
        }
"""),
],
# =============================================================================================
"bluscript_zero_ckpt.cpp": [
# comment only: the catch already wraps step() of EVERY group, so Muon and v2 are covered as is
("""            // Why the check is NOT here in the trainer but in ShardedOptimizer::step():
""",
"""            // Which optimizers raise it: all three -- ShardedOptimizer (v1, the default),
            //   SymmetricShardedOptimizer (BLU_ZERO_V2=1) and ShardedMuon (BLU_OPTIMIZER=muon). This
            //   loop steps every group, so this one catch covers all of them.
            // Why the check is NOT here in the trainer but in each optimizer's step():
"""),
],
}

def main():
    out = {}
    for path, want in EXPECT.items():
        data = open(path, "rb").read()
        got = hashlib.md5(data).hexdigest()
        if got != want:
            sys.exit(f"ABORT: {path} md5 {got}, expected {want} (not BluTrain e3633e8?). Nothing written.")
        text = data.decode()
        for i, (old, new) in enumerate(EDITS[path], 1):
            n = text.count(old)
            if n != 1:
                sys.exit(f"ABORT: {path} edit {i}: pattern found {n} times (need exactly 1). Nothing written.")
            text = text.replace(old, new)
            if text.count(new) != 1:
                sys.exit(f"ABORT: {path} edit {i}: replacement not present exactly once. Nothing written.")
        out[path] = text
    for path, text in out.items():
        open(path, "w").write(text)
        print(f"patched {path}: {len(EDITS[path])} edit(s)")
    print("done: 3 files, 6 edits")

if __name__ == "__main__":
    main()
