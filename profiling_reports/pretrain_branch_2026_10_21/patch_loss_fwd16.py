# 2026-10-21: the sparse cross-entropy FORWARD with real 16-byte loads, as the backward got on 2026-10-17 (item 2, work doc 45.35).
# Both forwards (validation sparse_ce_forward_kernel_vec, training sparse_ce_forward_kernel_vec_save_stats) staged each 16-byte
# vector in shared memory with cp.async, WAITED for that one copy, and read it back as 8 two-byte loads: one load in flight per
# thread. New: loss_ld16 straight into registers, LOSS_FWD_UNROLL (default 4) loads in flight, the SAME online max / sum update
# in the SAME order (v = tid, tid + bdim, ... ; k = 0 .. VEC-1), so saved_max / saved_sum / the losses are bit-identical.
# LOSS_FWD_UNROLL=0 compiles the old path (for A/B). Anchored edits: each anchor must occur exactly the expected number of times.
# usage: python3 patch_loss_fwd16.py <path to LossKernels.cu>
import sys
p = sys.argv[1]; s = open(p).read()

HELPER_AFTER = 'static_assert(LOSS_BWD_UNROLL >= 1 && LOSS_BWD_UNROLL <= 16, "LOSS_BWD_UNROLL out of range");\n'
HELPER = HELPER_AFTER + '''
// 2026-10-21: the two sparse-CE FORWARDS, same idea. Both used to stage each 16-byte vector in shared memory with
// cp.async, wait for that single copy (cp.async.wait_group 0 right after it), and read it back as VEC two-byte shared
// loads: one load in flight per thread. loss_fwd_row_vec loads the vectors straight into registers with loss_ld16,
// LOSS_FWD_UNROLL of them back to back, and runs the SAME online max / sum update in the SAME order as before
// (vectors tid, tid + bdim, ... ; elements k = 0 .. VEC-1), so saved_max, saved_sum and the losses are bit-identical.
// 0 = the old cp.async path. Override with -DLOSS_FWD_UNROLL=n.
#ifndef LOSS_FWD_UNROLL
#define LOSS_FWD_UNROLL 4
#endif
static_assert(LOSS_FWD_UNROLL >= 0 && LOSS_FWD_UNROLL <= 16, "LOSS_FWD_UNROLL out of range");
#if LOSS_FWD_UNROLL > 0
template <typename T>
__device__ __forceinline__ void loss_fwd_row_vec(const T *row_logits, int64_t vec_count, int tid, int bdim,
                                                 float &local_max, float &local_sum) {
  using vec_t = LossVec<T>;
  constexpr int VEC = vec_t::WIDTH;
  constexpr int U = LOSS_FWD_UNROLL;
  const vec_t *logits_v = reinterpret_cast<const vec_t *>(row_logits);
  int64_t v = tid;
  for (; v + (U - 1) * static_cast<int64_t>(bdim) < vec_count; v += U * static_cast<int64_t>(bdim)) {
    vec_t in[U];
#pragma unroll
    for (int u = 0; u < U; ++u)
      in[u] = loss_ld16(&logits_v[v + u * bdim]);
#pragma unroll
    for (int u = 0; u < U; ++u) {
#pragma unroll
      for (int k = 0; k < VEC; ++k) {
        float val = static_cast<float>(in[u].v[k]);
        if (val > local_max) {
          local_sum = local_sum * expf(local_max - val) + 1.0f;
          local_max = val;
        } else {
          local_sum += expf(val - local_max);
        }
      }
    }
  }
  for (; v < vec_count; v += bdim) {   // the vectors left over after the unrolled part
    const vec_t in = loss_ld16(&logits_v[v]);
#pragma unroll
    for (int k = 0; k < VEC; ++k) {
      float val = static_cast<float>(in.v[k]);
      if (val > local_max) {
        local_sum = local_sum * expf(local_max - val) + 1.0f;
        local_max = val;
      } else {
        local_sum += expf(val - local_max);
      }
    }
  }
}
#endif
'''
assert s.count(HELPER_AFTER) == 1, "LOSS_BWD_UNROLL static_assert anchor"
s = s.replace(HELPER_AFTER, HELPER)

# validation forward: the aligned branch's loop -> the helper (old loop kept under LOSS_FWD_UNROLL == 0)
OLD_V = '''  if ((reinterpret_cast<uintptr_t>(row_logits) & 0xF) == 0) {
    //* v indexes FULL vectors only, so nothing is read past the row end and the
    //* tail loop below owns the remainder exactly once -- no double counting.
    for (int64_t v = tid; v < vec_count; v += bdim) {'''
NEW_V = '''  if ((reinterpret_cast<uintptr_t>(row_logits) & 0xF) == 0) {
#if LOSS_FWD_UNROLL > 0
    (void)s_stage;                         // 2026-10-21: no shared staging (loss_fwd_row_vec)
    loss_fwd_row_vec(row_logits, vec_count, tid, bdim, local_max, local_sum);
#else
    //* v indexes FULL vectors only, so nothing is read past the row end and the
    //* tail loop below owns the remainder exactly once -- no double counting.
    for (int64_t v = tid; v < vec_count; v += bdim) {'''
assert s.count(OLD_V) == 1, "validation forward anchor"
s = s.replace(OLD_V, NEW_V)
END_V = '''        float val = static_cast<float>(s_stage[tid * VEC + k]);
        if (val > local_max) {
          local_sum = local_sum * expf(local_max - val) + 1.0f;
          local_max = val;
        } else {
          local_sum += expf(val - local_max);
        }
      }
    }
  } else {
    //* Fallback to scalar for unaligned rows'''
assert s.count(END_V) == 1, "validation forward loop end anchor"
s = s.replace(END_V, END_V.replace("    }\n  } else {\n    //* Fallback", "    }\n#endif\n  } else {\n    //* Fallback"))

# training forward (save_stats)
OLD_T = '''  if ((reinterpret_cast<uintptr_t>(row_logits) & 0xF) == 0) {
    //* FULL vectors only -- the tail loop owns the remainder exactly once.
    for (int64_t v = tid; v < vec_count; v += bdim) {'''
NEW_T = '''  if ((reinterpret_cast<uintptr_t>(row_logits) & 0xF) == 0) {
#if LOSS_FWD_UNROLL > 0
    (void)s_stage;                         // 2026-10-21: no shared staging (loss_fwd_row_vec)
    loss_fwd_row_vec(row_logits, vec_count, tid, bdim, local_max, local_sum);
#else
    //* FULL vectors only -- the tail loop owns the remainder exactly once.
    for (int64_t v = tid; v < vec_count; v += bdim) {'''
assert s.count(OLD_T) == 1, "training forward anchor"
s = s.replace(OLD_T, NEW_T)
END_T = '''        float val = static_cast<float>(s_stage[tid * VEC + k]);
        if (val > local_max) {
          local_sum = local_sum * expf(local_max - val) + 1.0f;
          local_max = val;
        } else {
          local_sum += expf(val - local_max);
        }
      }
    }
  } else {
    for (int64_t j = threadIdx.x; j < vec_count * VEC; j += blockDim.x) {'''
assert s.count(END_T) == 1, "training forward loop end anchor"
s = s.replace(END_T, END_T.replace("    }\n  } else {\n    for (int64_t j", "    }\n#endif\n  } else {\n    for (int64_t j"))

open(p, "w").write(s)
print("patched", p)
