// =============================================================================
// train/production_zero/bluscript_sft.cpp — the SFT (supervised fine-tuning) trainer.
//
// This file = this directory's pretraining trainer, bluscript_zero_ckpt.cpp (same model, ZeRO-1 AdamW / Muon,
// DCP checkpoints, NCCL or BluCCL), plus the SFT parts. Without BLU_SFT_DATA it trains exactly like
// bluscript_zero_ckpt.cpp. With BLU_SFT_DATA=<dir> (train/ and val/ written by sft/prepare_sft_data.py):
//   1. Only the answers are trained. Targets come from labels.bin; label 65535 = no loss and no gradient
//      (prompts, role markers, the pad tail of a row).
//   2. One loss average per optimizer step. Every micro-step's loss is divided by the GLOBAL count of trained
//      tokens in the whole step (all micro-steps, all GPUs), so every trained token has the same weight, whatever
//      the length of its chat (torchtitan's global_valid_tokens). A per-micro-step mean would give short chats
//      more weight.
//   3. Packed rows. Several whole chats share one row of T tokens (far fewer rows than one chat per row + padding).
//      Attention gets the chat plan, so each chat attends only to itself, and RoPE gets per-chat positions that
//      restart at 0 for every chat. A chat inside a packed row then computes bit for bit what it computes alone
//      in its own row.
// Library pieces it uses (Tensor-Implementations): sparse_cross_entropy_loss(..., ignore_index, &denom),
// fused_rms_rope_bf16(..., positions), scaled_dot_product_attention(..., chats) and gqa_chat_plan_make.
// The full map of every change, its reason and its test results: sft/README.md.
//
// Where the SFT parts are (search for these strings):
//   "PackedRows g_packed", packed_rows_of()    positions + chat plan, built once per micro-batch from the loader
//   g_packed.positions / g_packed.chats        the RoPE and attention calls in the attention block
//   "SFT mode: BLU_SFT_DATA"                   settings, the two SftDataLoaders, the step count from the epochs
//   "BLU_N_LAYERS / BLU_ROPE_THETA"            build the model the starting checkpoint was trained as
//   "135M base"                                the other shape knobs + BLU_EMBED_SCALE (from preproduction-135M)
//   "initialise the weights from another run"  BLU_INIT_FROM: weights only, fresh optimizer, optional CRC32 check
//   "BLU_INIT_FROM_RAW"                        the same from sft/hf_to_blutrain.py output (a Hugging Face export)
//   "BLU_EXPORT"                               the final weights as a safetensors file (export_safetensors)
//   grad_scale                                 backward seed: W in SFT (each micro loss is already divided by the
//                                              global count and ZeRO averages over the W GPUs), else 1/GRAD_ACCUM
//   skip_micro_batches                         resume: skips exactly the micro-batches already trained
//   "SFT: the whole val set"                   validation by the same rule (one division by the set's count)
//   valid_tokens_ahead, sft_denom              the step's global trained-token count = the loss denominator
//   "struct SftCp", cp_make_group, cp_combine,  context parallelism (BLU_SFT_CP=U, 2026-11-06/07): U GPUs share each
//   cp_partition, cp_slice                     row; Adhitya's Ulysses all-to-alls around our attention (README 5.2)
//
// Build (BluTrain root, after `make -j$(nproc) profiler tensor dist SM_ARCH=90`): bash sft/build_sft_trainer.sh
// (-> ./bluscript_sft_bin; USE_FUSED_RMS_ROPE=1, which packed rows need). Run: sft/run_sft.sh, sft/README.md section 8.
//
// 2026-10-27: brought onto this branch from the SFT tree (where it was bluscript_sft.cpp at the repo root, on
// dist_side_merge) by a 3-way merge, base = the pretraining trainer it started from (909ab75). Adapted to this
// branch: the raw MPI calls became dist::comm::allreduce_scalar / abort_job (they work under mpirun AND BluRun), the
// SFT loaders put their batches on the LOCAL rank's GPU (a device ordinal is node-local), BLU_INIT_FROM loads over an
// NCCL group under BluCCL too, and the CSV keeps this branch's columns plus "trained_tokens". Everything below this
// block is the pretraining trainer's own header.
// =============================================================================
// =============================================================================
// train/production_zero/bluscript_zero_ckpt.cpp — bluscript_zero_ckpt.cpp driven by
// bluscript_zero_ckpt_config.yaml (+ its parallelism file, whose backend selects NCCL or BluCCL).
// bluscript.cpp — Llama-style decoder LM training (fp32 model, bf16 fused attn,
// ZeRO-1 mixed precision optimizer state sharding, WSD LR schedule)
//
// Architecture (per the production spec):
//   token embedding
//   -> N x TransformerBlock:
//        RMSNorm -> { QK-norm + RoPE + grouped-query causal attention }(fused
//                    kernel: gqa_fused_flash_attn_forward/backward) -> Wo -> +residual
//        RMSNorm -> SwiGLU FFN (gate_up -> swiglu -> down)           -> +residual
//   -> final RMSNorm
//   -> lm_head (weight-tied with the token embedding)
//   -> sparse cross-entropy loss
//   -> backward -> AdamW, or Muon+AdamW (BLU_OPTIMIZER=muon), with a WSD
//      (warmup / stable / cosine-decay) LR schedule
//
// The QK-norm scales (q_gamma / k_gamma) and the RoPE rotation are FUSED in one
// CUDA kernel (autograd::fused_rms_rope), and grouped-query causal attention is
// a second fused flash-attention kernel reached through
// autograd::scaled_dot_product_attention. Those ops run bf16 internally and
// bridge back to the fp32 model. Matmuls go through the library BLAS
// (nn::Linear / autograd::matmul).
//
// Optimizer: ZeRO-1 (owt::zero::ShardedOptimizer). FP32 master weights + Adam
// moments live only in each rank's optimizer shard, not the model. Designed for
// ZeRO-1 over 8 GPUs (one rank per GPU): each rank holds the full model params
// but only 1/world_size of the optimizer state; per step, grads reduce-scatter
// (avg) across ranks, the fused Adam update runs on each rank's FP32 shard, and
// updated params all-gather back — no separate DDP grad all-reduce.
//
// BLU_OPTIMIZER=muon switches the transformer body's 2-D weight matrices to
// Muon, keeping AdamW for the token embedding and every 1-D scale. ZeRO-1 STILL
// APPLIES to the Muon set, but it cannot be the same ZeRO-1: AdamW's update is
// elementwise, so any partition of the flat parameter buffer is valid, while
// Muon's Newton-Schulz orthogonalization is a whole-MATRIX operator that a
// contiguous flat slice cuts through the middle of. The Muon set therefore goes
// to owt::zero::ShardedMuon, which shards by WHOLE PARAMETER — one owner rank
// per matrix, holding that matrix's FP32 master + momentum, reducing its
// gradient to the owner and broadcasting the updated weight back. Same ~1/W
// state reduction, and cheaper than sharded AdamW (no second moment, no flat
// staging buffer). See dist/zero/include/ShardedMuon.h.t
//
// Launch:  make run-mpi FILE=bluscript.cpp NP=8
//          BLU_OPTIMIZER=muon mpirun -np 8 ./_runner_
// =============================================================================

#include <cstdint>
#include <cstdlib>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <random>
#include <map>
#include <cstring>        // std::memcmp (BLU_INIT_FROM_RAW exact check)
#include <sstream>
#include <set>
#include <functional>
#include <algorithm>
#include <cctype>
#include <spawn.h>
#include <unistd.h>

#include <mpi.h>
#include <cuda_runtime.h>

// Tensor library
#include "TensorLib.h"
#include "autograd/AutogradOps.h"               // matmul, add, reshape, transpose, swiglu, ...
#include "autograd/operations/AttentionOps.h"   // fused GQA scaled_dot_product_attention
#include "autograd/operations/RoPEOps.h"        // build_rope_cache
#include "autograd/operations/LossOps.h"        // sparse_cross_entropy_loss
#include "autograd/operations/RowGatherOps.h"   // gather_rows (the trained-rows switch, 2026-10-31)
#include "autograd/operations/ReshapeOps.h"
#include "autograd/operations/EmbeddingOps.h"
#include "nn/NN.h"
#include "nn/optimizer/Optim.h"
#include "checkpointing/GradMode.h"
#include "DistributedCheckpointing.h"   // DistCheckpoint::save / load / is_valid
#include "ZeroDcpBridge.h"              // ZeroDcpModule, ZeroDcpOptimizer
#include "Metadata.h"                   // CheckpointMetadata (BLU_INIT_VERIFY)
#include "Checksums.h"                  // CRC32 check of the weights BLU_INIT_FROM loaded
#include "device/CachingCudaAllocator.h"
#include "device/DeviceCore.h"                  // OwnTensor::cuda::getCurrentStream()

// Tracker + Memory
#include "device/SizeClass.h"
#include "profiler/AllocationTracker.h"

// Data + distributed
#include "DataLoader.h"
#include "MixedDataloader.h"      // MixedDataLoaderLite: weighted multi-domain mixing
#include "sft/SftDataLoader.h"    // SftDataLoader: SFT rows with masked labels (BLU_SFT_DATA)
#include "dist/distributed.h"
#include "LaunchEnv.h"                            // dist::comm::detect_launch — launcher-agnostic bring-up
#include "ProcessGroupBase.h"
#include "ProcessGroupComposite.h"                 // multi-node bluccl: BluCCL reduce-scatter + all-gather, NCCL for the rest
#include "BlucclFusedEmbedding.h"
#include "ConfigBridge.h"
#include "dist/zero/include/ShardedOptimizer.h"   // owt::zero::ShardedOptimizer (ZeRO-1)
#include "dist/zero/include/SymmetricShardedOptimizer.h" // owt::zero::SymmetricShardedOptimizer (ZeRO-1 v2)
#include "dist/zero/include/ShardedMuon.h"        // owt::zero::ShardedMuon (ZeRO-1 over whole tensors)
#include "dist/zero/include/ZeroOptimizerBase.h"        // common surface so v1/v2/muon are interchangeable
// 2026-11-06: context parallelism for long SFT rows (BLU_SFT_CP=U, off by default). Adhitya's Ulysses all-to-alls and
// their autograd nodes, used AS THEY ARE from the Context_Parallelism submodule (dist/Context_Parallelism, pinned
// cf1372d; header-only): cp::ulysses_combine / cp::ulysses_partition and UlyssesCombineBackward /
// UlyssesPartitionBackward. Our QK-norm + RoPE (per-chat positions) and our packed attention (chat plan) stay as they are.
#include "context_parallel/UlyssesAttention.h"
#include "context_parallel/HybridUlyssesBackward.h"
#include "sub_group.h"                                   // dist::comm::Sub_Group: the CP group (built in cp_make_group)

// Compile-time flag to toggle between fused RMSNorm+RoPE kernel (1) and sequential RMSNorm + RoPE (0)
#ifndef USE_FUSED_RMS_ROPE
#define USE_FUSED_RMS_ROPE 1
#endif

using namespace OwnTensor;

// =============================================================================
// bluscript_sft.cpp (2026-10-15): the SFT trainer. A copy of bluscript_zero_ckpt.cpp (pretraining, unchanged) with
// v2 PACKED rows added: several whole chats per row, each attending only to itself and with RoPE positions restarting
// at 0 for every chat. Packed data = BLU_SFT_DATA/<split>/seq_lengths.bin + seq_offsets.bin (the chats of every row as
// a list of lengths, TRL's seq_lengths; written by sft/prepare_sft_data.py --pack ffd); without them a split trains
// exactly as v1 (one chat per row).
//
// g_packed is set by the loop before every model.forward on an SFT batch (from SftDataLoader) and read by the
// attention block: positions [B,T] int32 = index of each token inside its chat (RoPE); chats = the per-chat attention
// plan (design A, the cu_seqlens way of FlashAttention varlen: OwnTensor::cuda::GqaChatPlan), built once per
// micro-batch from the loader's segment lists and shared by every layer's forward and backward. Empty = dense, exactly
// as before. There is no activation recompute, so it only has to stay valid until the micro-step's backward, which
// runs before the next batch sets it again (each attention backward node holds its own reference to the plan).
// (Design B, one seg_start value per token, was used here on the morning of 2026-10-15 and removed the same day.)
// =============================================================================
struct PackedRows { Tensor positions; OwnTensor::cuda::GqaChatPlanPtr chats; };
static PackedRows g_packed;
// The packed-row inputs of the batch `l` just returned: its RoPE positions, and the attention plan of its B rows for
// Hq query heads (one asynchronous upload; the host does not wait for the GPU). Empty when `l` holds v1 rows.
// 2026-11-13 (mixed units of the attention forward, work doc 45.194): the plan is built with the KV-head count Hkv of
// the attention call it is for, so that in a chat with an odd number of 64-row tiles two query heads of one KV head
// share a pair (forward -8.0% on rows of 32-160-token chats, -0.4..-1.05% on real rows, H200, heads 18/6; same results).
// BLU_GQA_MX=0 in the environment of the binary builds the plans of before.
// Old code (kept, not deleted):
// static PackedRows packed_rows_of(const SftDataLoader& l, int B, int T, int Hq) {
//     if (!l.packed()) return PackedRows{};
//     const std::vector<int32_t>& rows = l.last_segment_rows();
//     return PackedRows{l.last_positions(),
//                       OwnTensor::cuda::gqa_chat_plan_make(rows.data(), l.last_segment_starts().data(),
//                                                           l.last_segment_lengths().data(), static_cast<int>(rows.size()),
//                                                           B, T, Hq)};
// }
static PackedRows packed_rows_of(const SftDataLoader& l, int B, int T, int Hq, int Hkv) {
    if (!l.packed()) return PackedRows{};
    const std::vector<int32_t>& rows = l.last_segment_rows();
    return PackedRows{l.last_positions(),
                      OwnTensor::cuda::gqa_chat_plan_make(rows.data(), l.last_segment_starts().data(),
                                                          l.last_segment_lengths().data(), static_cast<int>(rows.size()),
                                                          B, T, Hq, Hkv)};
}

// =============================================================================
// Context parallelism (2026-11-06; BLU_SFT_CP=U, default 1 = off and every line below is skipped).
// U GPUs share one packed row: they load the SAME rows (the loader runs with rank / U of world / U) and each keeps
// token slice cp.rank of length T / U (input ids, labels, RoPE positions). Per layer, in the attention block:
//   our fused QK-norm + RoPE on the slice, with each token's true per-chat position;
//   combine (Adhitya's all-to-all): [B, H, T/U, hd] -> [B, H/U, T, hd], every GPU now holds the WHOLE row for H/U heads;
//   our packed attention with the full row's chat plan, exactly as on one GPU (the plan is built for H/U heads);
//   partition (his all-to-all back): [B, Hq/U, T, hd] -> [B, Hq, T/U, hd].
// Everything else (embedding, MLP, norms, lm_head, loss) runs on the slice with no communication. The loss rule is
// unchanged: trained-token sum of the slice / the step's trained count over all ranks (each CP group's rows counted
// once), gradients summed over all ranks by ZeRO as before.
// Needs Tensor-Implementations d59740b or later (2026-11-07): before it, autograd::contiguous() on an already-contiguous
// k_bf (output 1 of fused_rms_rope_bf16) dropped k's gradient and the k-norm gamma gradient in every layer (CP=2 step-0
// gradient norm -7.5% vs no CP on 2026-11-06). Tested 2026-11-07 on 2x H200 (records cp_2026_11_07_2x/results_2x.txt):
//   135M, 4k rows: CP=2 step-0 gradient norm 3.0473 = no CP 3.0473 (gammas 0.0458 = 0.0458); without that TI fix, same
//     box: 2.8367 (-6.9%); over 30 steps CP=2 differs from no CP by at most 0.000997 in loss (two identical no-CP runs:
//     0.000813);
//   135M, 64k / 128k rows: CP=2 within the run-to-run variation of CP=1 (step-0 norm -0.02% / +0.18%; two CP=1 runs
//     +0.56% / +0.24%); step 250.7 -> 158.2 ms / 492.9 -> 297.9 ms vs CP=1 on one GPU; peak per GPU 61.5 -> 36.3 GB /
//     111.0 -> 62.2 GB;
//   1.6B shape (random init), 4k: CP=2 step-0 norm +0.017% vs no CP; 64k: CP=2 fits (96.6 GB per GPU), one GPU runs out
//     of memory.
// Not tested yet: CP together with data parallelism (world > U, needs 4+ GPUs).
// =============================================================================
struct SftCp { int U = 1; int rank = 0; std::shared_ptr<ProcessGroupNCCL> pg, parent; };
static SftCp g_cp;
// The CP group: ncclCommSplit off a world NCCL group (collective over it: every rank calls it once), with a BLOCKING
// communication stream that the group owns -- the way Adhitya's DeviceMesh builds its CP sub-group
// (process_group/device_mesh.cpp: cudaStreamCreate, "blocking comm stream"). His all-to-alls launch on the group's
// stream without waiting on the compute stream; a blocking stream is ordered after the legacy default stream 0 that our
// kernels run on, so each all-to-all reads finished data. (dist::comm::sub_group_split gives a NON-blocking stream,
// which has no such ordering, so it is not used here.)
static std::shared_ptr<ProcessGroupNCCL> cp_make_group(ProcessGroupNCCL& world, int rank, int U) {
    ncclComm_t comm = nullptr;
    const ncclResult_t r = ncclCommSplit(world.get_comm(), /*color=*/rank / U, /*key=*/rank % U, &comm, nullptr);
    if (r != ncclSuccess || comm == nullptr)
        throw std::runtime_error(std::string("BLU_SFT_CP: ncclCommSplit failed: ") + ncclGetErrorString(r));
    cudaStream_t s = nullptr;
    if (cudaStreamCreate(&s) != cudaSuccess) throw std::runtime_error("BLU_SFT_CP: cudaStreamCreate failed");
    std::vector<int> ranks;
    for (int i = 0; i < U; ++i) ranks.push_back(rank / U * U + i);
    auto work = std::make_shared<Work>(s, /*comm=*/nullptr);
    auto pg = std::make_shared<dist::comm::Sub_Group>(ranks, rank % U, comm, work, s, /*root=*/0);
    pg->set_owns_stream(true);
    return pg;
}
// Adhitya's combine_ag / partition_ag (ContextParallel.h, private members there), the same lines around his public
// collectives and backward nodes: each all-to-all is a pure permutation, its gradient the inverse all-to-all.
static Tensor cp_combine(const Tensor& x) {
    Tensor out = OwnTensor::cp::ulysses_combine(g_cp.pg, x, g_cp.U);
    if (x.requires_grad()) {
        auto gf = std::make_shared<UlyssesCombineBackward>(g_cp.pg, g_cp.U);
        Tensor& xm = const_cast<Tensor&>(x);
        gf->set_next_edge(0, autograd::get_grad_edge(xm));
        out.set_grad_fn(gf);
        out.set_requires_grad(true);
    }
    return out;
}
static Tensor cp_partition(const Tensor& x) {
    Tensor out = OwnTensor::cp::ulysses_partition(g_cp.pg, x, g_cp.U);
    if (x.requires_grad()) {
        auto gf = std::make_shared<UlyssesPartitionBackward>(g_cp.pg, g_cp.U);
        Tensor& xm = const_cast<Tensor&>(x);
        gf->set_next_edge(0, autograd::get_grad_edge(xm));
        out.set_grad_fn(gf);
        out.set_requires_grad(true);
    }
    return out;
}
// This rank's token slice of a [B, T] tensor (ids, labels or positions; no gradient), contiguous.
static Tensor cp_slice(const Tensor& t) {
    if (g_cp.U == 1 || !t.is_valid()) return t;
    Tensor m = t;                                         // narrow() is non-const; a handle copy shares the storage
    const int64_t Tl = m.shape().dims[1] / g_cp.U;
    return m.narrow(1, static_cast<int64_t>(g_cp.rank) * Tl, Tl).contiguous();
}

using owt::zero::ShardedOptimizer;
using owt::zero::SymmetricShardedOptimizer;
using owt::zero::ShardedMuon;
using owt::zero::ZeroConfig;
using owt::zero::MuonConfig;
using owt::zero::AdamVariant;

// getenv_int lives in DataLoader.h; these two complete the set so the data
// mixture (roots and weights) is overridable from the launch script without a
// recompile, the same way every other knob in this file is.
static std::string getenv_str(const char* key, const std::string& def) {
    const char* v = std::getenv(key);
    return (v && *v) ? std::string(v) : def;
}
static float getenv_float(const char* key, float def) {
    const char* v = std::getenv(key);
    return (v && *v) ? std::strtof(v, nullptr) : def;
}


// =============================================================================
// Configuration — resolved from the bluscript.cpp template (integer-division
// placeholders evaluated to their intended values).
// =============================================================================
struct ModelConfig {
    // ---- model ----
    int64_t d_model        = 2304;
    int64_t n_layers       = 26;
    int64_t q_heads        = 18;
    int64_t kv_heads       = 6;                          // grouped-query: G = q/kv = 3
    int64_t head_dim       = 128;                   // = 64  (fused kernel needs 64 or 128)
    int64_t ffn_hidden     = 6144;              // 8/3 * d_model = 2048 (SwiGLU inner dim)
    int64_t context_length = 4096;                       // = T (multiple of 32, RoPE cache length)
    int     rope_theta     = 10000;
    int64_t vocab_size     = 50304;                      // 50257 (tiktoken) padded for GPU efficiency
    bool    weight_tying   = true;
    // BLU_EMBED_SCALE=1: multiply the token-embedding OUTPUT by sqrt(d_model)
    // before the first block (Vaswani et al. / Gemma). Input side only: under
    // weight tying the lm_head still reads the unscaled wte.weight.
    // Copied from preproduction-135M f37c616: the 135M base was trained with it on.
    bool    embed_scale    = false;

    float   rms_eps        = 1e-5f;                      // block RMSNorm epsilon
    float   qk_eps         = 1e-6f;                      // QK-norm (fused) epsilon
    bool    rope_interleaved = false;                    // false = NeoX half-split (Llama style)

    // ---- activation / parameter precision ----------------------------------
    // All false (the default) is the shipped mixed-precision layout and is
    // bit-identical to the pre-flag code: bf16 Linears + attention, fp32
    // residual stream / norms / embedding / logits, fp32 optimizer masters.
    // Each flag moves exactly ONE stage to bf16, so a regression bisects to a
    // single stage. Wired from BLU_ALL_BF16 (+ per-stage overrides) in main().
    //
    // Two things are deliberately NOT flags, because neither is reachable by
    // changing a dtype here:
    //
    //   * Optimizer masters and moments. shard_master_/shard_m_/shard_v_ are
    //     hardcoded Float32 (ShardedOptimizer.cpp allocate_buffers) and
    //     multi_tensor_adam_cuda takes float* only. They stay fp32 under every
    //     flag below, which is what keeps the update path safe: AdamW's
    //     decoupled decay is a (1 - lr*wd) = 1 - 6e-5 multiply, and bf16's
    //     relative grid is 2^-8 = 3.9e-3, so a bf16 master would round the
    //     entire weight-decay term away at any weight magnitude.
    //
    //   * QK-norm + RoPE. fused_rms_rope_forward reads q, k, q_gamma and
    //     k_gamma through data<float>() with no dtype dispatch, and
    //     autograd::rope rejects anything but fp32 outright. data<T>() is a
    //     bare reinterpret_cast, so handing that kernel a bf16 tensor would
    //     read garbage SILENTLY rather than throw. q_gamma/k_gamma therefore
    //     stay fp32 and the island is bracketed by explicit casts in
    //     CausalGQA::forward. See qk_dtype().
    //   * Logits. bf16_logits is wired end to end and the sparse-CE kernels now
    //     accept it: LossKernels.cu's vectorized paths key every chunk off
    //     16/sizeof(T) elements instead of a hard-coded 4, in the forward and in
    //     both backward variants (see VEC_ELEMS / Chunk16 there). Before that they
    //     staged 16-byte cp.async loads at a stride of 4 ELEMENTS — 8-byte aligned
    //     on bf16, a hard "misaligned address" fault — and
    //     `reinterpret_cast<float4*>(row_grad)` wrote 16 bytes per 8-byte logical
    //     span, overrunning the gradient buffer SILENTLY.
    //
    //     What is NOT fixed, and is not fixable in those kernels: LossOps keys the
    //     LOSS SCALAR's dtype off the logits dtype, so bf16_logits also makes the
    //     reported loss bf16. The per-row losses are summed in fp32 but stored back
    //     through a bf16 tensor before the divide by batch_size, and bf16 cannot
    //     represent a loss near 10.8 finer than one ULP = 0.0625 either way — so the
    //     logged/allreduced loss quantizes to ~0.06 steps. GRADIENTS ARE UNAFFECTED
    //     (backward() is seeded at 1.0, which is exact in bf16, and the softmax
    //     stats are fp32), so this costs the metric, not the run. Decoupling the
    //     loss dtype from the logits dtype needs a LossT template parameter through
    //     LossKernels/LossOps plus a grad_output dtype check in LossBackward, which
    //     reads it via data<LogitT>() with no dtype guard.
    bool bf16_resid  = false;   // residual stream + block RMSNorm scales
    bool bf16_logits = false;   // logits + loss scalar — see the note above
    bool bf16_embed  = false;   // wte (under weight tying, also the lm_head weight)

    Dtype resid_dtype() const { return bf16_resid  ? Dtype::Bfloat16 : Dtype::Float32; }
    Dtype logit_dtype() const { return bf16_logits ? Dtype::Bfloat16 : Dtype::Float32; }
    Dtype embed_dtype() const { return bf16_embed  ? Dtype::Bfloat16 : Dtype::Float32; }
    // Not a flag — a kernel constraint. See the note above.
    static Dtype qk_dtype() { return Dtype::Float32; }

    // ---- optimization / data ----
    int64_t B            = 8;                            // micro-batch per rank
    int64_t T            = 4096;                         // sequence length (== context_length)
    int64_t global_batch = 1048576;                       // tokens per optimizer step (all ranks)
    float   grad_clip    = 1.0f;
    float   weight_decay = 0.1f;

    // WSD schedule: linear warmup -> constant at max_lr -> cosine decay to min_lr.
    // Requires warmup_steps + stable_steps < max_steps (the remainder is the decay tail).
    float   max_lr       = 0.0012;
    float   min_lr       = 0.00012;                        // 10% of max_lr
    int     warmup_steps = 200; //715;
    int     max_steps    = 8000;  //19073;                        // ~10B tokens at 0.5M tokens/step
    int     stable_steps = 6200;                         // steps held at max_lr after warmup

    // ---- Muon (BLU_OPTIMIZER=muon) ----
    // Mirrors train_muon.py's config so the C++ and PyTorch arms are the same
    // experiment. Muon's group rides the SAME WSD curve as AdamW's — same
    // warmup_steps, same stable window, same cosine tail — via a second get_lr()
    // call with its own (max, min) pair, exactly as train_muon.py does at its
    // `muon_lr = get_lr(step, cfg, cfg.muon_lr, cfg.muon_min_lr)` line. There is
    // deliberately no separate muon_warmup_steps: both groups train the same
    // model on the same steps, so only the SCALE should differ.
    //
    // Under the default adjust_lr=match_rms_adamw the Newton-Schulz update's
    // entry RMS is lifted to 0.2 — what a converged AdamW update looks like —
    // which puts muon_lr on the SAME SCALE as max_lr. Setting them equal makes
    // the two arms differ only in update DIRECTION, which is the controlled
    // experiment worth running first. 8e-4 here is 4x max_lr, the post-sweep
    // value train_muon.py ships (Muon typically tolerates 2-4x more LR once it
    // is working).
    float   muon_lr           = 0.0012;
    float   muon_min_lr       = 0.00012;                   // 10% of muon_lr
    float   muon_momentum     = 0.95f;
    // Pinned to weight_decay by default so the decay is identical across both
    // groups and both arms — only meaningful because match_rms_adamw puts the
    // two LRs on one scale. train_muon.py pins it the same way.
    float   muon_weight_decay = 0.1f;
    int     muon_ns_steps     = 5;
    float   beta1             = 0.9f;
    float   beta2             = 0.95f;
    float   adam_eps          = 1e-8f;
    int     seed              = 1234;
    int     layers_per_stage  = 1;

    int     val_freq       = 250;
    int     val_loss_steps = 20;

    // ---- checkpointing ----
    // OFF by default so a throughput A/B against the PyTorch arms is apples to
    // apples: train_ds_zero1.py and train_zero1.py do not port checkpointing, so
    // at ckpt_freq=900 a 2000-step run would spend three DCP saves (steps
    // 899/1799/1999) writing 681 M params plus the sharded optimizer state that
    // the PyTorch side never pays for — the Muon master+momentum alone is
    // 2304 MB/GPU. BLU_CKPT=1 turns it back on; that is the setting for testing
    // DCP itself, and the only one under which resume is exercised.
    bool        checkpointing = true;
    int         ckpt_freq     = 1600;                   // save every 1800 optimizer steps
    int         ckpt_keep     = 4;                      // retain the most-recent N checkpoints
    std::string ckpt_dir      = "checkpoints_bluscript";
    std::string ckpt_prefix   = "blumodel";
};

// =============================================================================
// MFU — Model FLOPs Utilization
// =============================================================================
// The number frontier labs report next to throughput (PaLM 46.2%, Megatron-LM,
// MosaicML, Llama). It answers "what fraction of the GPUs' peak matmul rate is
// this run actually converting into MODEL math?" and is the only throughput
// metric comparable across model sizes, cluster sizes and codebases:
//
//     MFU = model_FLOPs_per_step / (step_wall_time * world_size * peak_FLOPS)
//
// Two conventions matter, and both are implemented here.
//
//  * MODEL FLOPs (this file's numerator) counts the arithmetic the MODEL
//    DEFINITION requires — the forward+backward matmuls of the architecture,
//    once. It is deliberately implementation-blind: activation recomputation,
//    padded GEMM tiles, redundant casts and communication do NOT count. That is
//    the point. A run that recomputes every layer does ~33% more hardware work
//    for the SAME model FLOPs, so its MFU correctly drops. HARDWARE FLOPs
//    utilization (HFU) counts what the silicon actually executed and is always
//    >= MFU. THIS LOOP USES NO ACTIVATION CHECKPOINTING (the micro loop calls
//    forward once and backward replays the stored graph), so here HFU == MFU
//    and only one number is reported.
//
//  * The ATTENTION-CORE count differs between papers. PaLM (Appendix B) and
//    Megatron-LM charge the full T^2 score matrix; DeepMind/Chinchilla and
//    MosaicML's llm-foundry halve it for causal masking. This kernel is
//    FlashAttention with is_causal=true, which genuinely SKIPS the masked
//    blocks — charging for them would credit work the GPU never did and inflate
//    MFU. So the causal-aware count is the headline number, and the PaLM-
//    convention count is printed alongside it so results stay comparable to
//    published figures (it is always the larger, i.e. more flattering, of the
//    two).
//
// Everything is derived from the architecture dims rather than from 6*N, so the
// count stays correct when weight_tying is turned off (an untied lm_head adds
// vocab*d_model PARAMETERS, but the embedding lookup is still a gather and not
// a matmul, so 6*N would over-count by 6*vocab*d_model). With tying ON the two
// agree exactly — the banner prints both as a self-check.
//
// Per convention, non-matmul work (RMSNorm, QK-norm, RoPE, SwiGLU's swish,
// softmax, and the optimizer's Adam / Newton-Schulz update) is NOT counted: it
// is a low-single-digit share of the FLOPs, it is memory-bandwidth bound rather
// than tensor-core bound, and every published MFU omits it. Counting it would
// make this run look better than the papers it is being compared against.
// =============================================================================
struct MFUMeter {
    // ---- per-token FLOPs of one forward pass (matmuls only) ----
    double fwd_per_token_causal = 0.0;   // FlashAttention is_causal (headline)
    double fwd_per_token_dense  = 0.0;   // PaLM / Megatron convention
    // ---- fwd + bwd = 3x forward (dX = dY@W^T and dW = X^T@dY each cost a fwd)
    double flops_per_token_causal = 0.0;
    double flops_per_token_dense  = 0.0;

    double      peak_flops_per_gpu = 0.0;   // dense BF16 tensor core, no sparsity
    std::string device_name        = "unknown";
    bool        peak_is_guess      = false; // no table entry matched

    int world_size   = 1;
    int warmup_steps = 5;                // steps excluded from the steady-state mean

    // Steady-state accumulators. Aggregate (total model FLOPs / total wall
    // time), not a mean of per-step ratios — one stalled step then costs
    // exactly its own wall time instead of dragging an average of ratios.
    double ss_tokens = 0.0;
    double ss_secs   = 0.0;
    int    ss_steps  = 0;
    int    seen      = 0;                // observe() calls, INCLUDING the warmup
    double best_mfu  = 0.0;

    // -------------------------------------------------------------------------
    // Peak dense BF16 tensor-core throughput per GPU, WITHOUT the 2:1
    // structured-sparsity doubling that NVIDIA's headline numbers carry. Dense
    // is the correct denominator: nothing in this step uses sparsity, so the
    // sparse peak would halve every MFU reported here.
    //
    // H200 SXM is H100 SXM's compute die paired with HBM3e, so its BF16 rate is
    // identical: 989.4 TFLOP/s dense (the spec sheet's "1,979 TFLOPS" is the
    // sparse figure). The PCIe / NVL parts clock lower and cannot be told apart
    // by name, so use BLU_PEAK_TFLOPS there.
    // -------------------------------------------------------------------------
    static double peak_bf16_tflops(const std::string& name, bool& matched) {
        struct Entry { const char* key; double tflops; };
        static const Entry table[] = {
            {"GB200", 2500.0},   // Blackwell superchip, per-GPU dense BF16
            {"B200",  2250.0},   // Blackwell SXM
            {"B100",  1800.0},
            {"H200",   989.4},   // SXM  (NVL/PCIe ~835 — set BLU_PEAK_TFLOPS)
            {"H100",   989.4},   // SXM  (PCIe ~756)
            {"H800",   989.4},
            {"L40S",   362.1},
            {"A100",   312.0},
            {"A800",   312.0},
        };
        for (const auto& e : table)
            if (name.find(e.key) != std::string::npos) { matched = true; return e.tflops; }
        matched = false;
        return 989.4;            // assume H200-class rather than divide by zero
    }

    static MFUMeter build(const ModelConfig& c, int world_size, int device_ordinal) {
        MFUMeter m;
        m.world_size = world_size;

        const double d   = static_cast<double>(c.d_model);
        const double L   = static_cast<double>(c.n_layers);
        const double hd  = static_cast<double>(c.head_dim);
        const double Nq  = static_cast<double>(c.q_heads);
        const double Nkv = static_cast<double>(c.kv_heads);
        const double F   = static_cast<double>(c.ffn_hidden);
        const double T   = static_cast<double>(c.T);
        const double V   = static_cast<double>(c.vocab_size);

        // A GEMM [m,k] x [k,n] costs 2*m*k*n FLOPs (one multiply + one add per
        // MAC). Per TOKEN every weight matmul is a [1,k] x [k,n] row, so each
        // costs exactly 2 * (its parameter count).
        const double qkv  = 2.0 * d * (Nq  * hd)      // wQ
                          + 2.0 * d * (Nkv * hd)      // wK   (GQA: Nkv < Nq)
                          + 2.0 * d * (Nkv * hd);     // wV
        const double proj = 2.0 * (Nq * hd) * d;      // wO
        const double ffn  = 2.0 * d * (2.0 * F)       // gate_up, emits [gate|up]
                          + 2.0 * F * d;              // down
        // Attention core — the two matmuls with NO parameters. They scale with
        // T, so a 6*N count is blind to them entirely:
        //   scores = Q @ K^T : per token, per q-head, [1,hd] x [hd,T] = 2*T*hd
        //   out    = P @ V   : per token, per q-head, [1,T]  x [T,hd] = 2*T*hd
        // GQA does not reduce this — the KV heads are broadcast across all Nq
        // queries, so the score matrix is still Nq deep.
        const double attn_dense  = 4.0 * T * hd * Nq;
        const double attn_causal = 0.5 * attn_dense;  // is_causal skips the mask

        // lm_head sits OUTSIDE the layer stack. The token embedding is a gather
        // (~0 FLOPs) and is correctly uncounted even though it holds the tied
        // weights.
        const double head = 2.0 * d * V;

        m.fwd_per_token_causal = L * (qkv + proj + ffn + attn_causal) + head;
        m.fwd_per_token_dense  = L * (qkv + proj + ffn + attn_dense)  + head;
        m.flops_per_token_causal = 3.0 * m.fwd_per_token_causal;
        m.flops_per_token_dense  = 3.0 * m.fwd_per_token_dense;

        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, device_ordinal) == cudaSuccess)
            m.device_name = prop.name;

        bool matched = false;
        double tflops = peak_bf16_tflops(m.device_name, matched);
        m.peak_is_guess = !matched;

        // BLU_PEAK_TFLOPS overrides the table: PCIe/NVL parts, a clock-capped
        // cluster, or an FP8 run (whose peak is 2x the BF16 figure).
        if (const char* v = std::getenv("BLU_PEAK_TFLOPS")) {
            const double ov = std::atof(v);
            if (ov > 0.0) { tflops = ov; m.peak_is_guess = false; }
        }
        m.peak_flops_per_gpu = tflops * 1e12;
        m.warmup_steps = getenv_int("BLU_MFU_WARMUP", 5);
        return m;
    }

    double cluster_peak() const {
        return peak_flops_per_gpu * static_cast<double>(world_size);
    }

    // Instantaneous MFU for one optimizer step. `tokens` is the GLOBAL token
    // count for the step (all ranks, all micro-batches), which is why the
    // denominator carries world_size: both sides scale together, so MFU is
    // invariant to cluster size and the MFU decay of a scaling run IS its
    // parallel-efficiency loss.
    double mfu(double dt, int64_t tokens) const {
        if (dt <= 0.0) return 0.0;
        return (flops_per_token_causal * static_cast<double>(tokens)) / (dt * cluster_peak());
    }
    double mfu_palm(double dt, int64_t tokens) const {
        if (dt <= 0.0) return 0.0;
        return (flops_per_token_dense * static_cast<double>(tokens)) / (dt * cluster_peak());
    }
    // Achieved model TFLOP/s on ONE GPU — the absolute number to hold against
    // the 989.4 ceiling when a ratio alone is ambiguous.
    double tflops_per_gpu(double dt, int64_t tokens) const {
        if (dt <= 0.0) return 0.0;
        return (flops_per_token_causal * static_cast<double>(tokens))
             / (dt * static_cast<double>(world_size) * 1e12);
    }

    // Fold one step into the steady-state aggregate. The first `warmup_steps`
    // are dropped: their dt is dominated by pipeline fill, the one-time ZeRO
    // shard / autograd-cache allocations and cuBLAS autotuning, and they drag
    // the reported figure well below what the run actually sustains.
    //
    // The window counts observe() CALLS, not the absolute step index: a run
    // resumed from a checkpoint starts the loop at start_step > 0, and an
    // absolute `step < warmup_steps` test would then skip the exclusion
    // entirely while its pipeline is filling exactly like a fresh run's.
    void observe(double dt, int64_t tokens) {
        if (seen++ < warmup_steps || dt <= 0.0) return;
        ss_tokens += static_cast<double>(tokens);
        ss_secs   += dt;
        ss_steps  += 1;
        best_mfu = std::max(best_mfu, mfu(dt, tokens));
    }
    double steady_mfu() const {
        if (ss_secs <= 0.0) return 0.0;
        return (flops_per_token_causal * ss_tokens) / (ss_secs * cluster_peak());
    }
    double steady_tok_per_sec() const { return ss_secs > 0.0 ? ss_tokens / ss_secs : 0.0; }

    void print_banner(int64_t num_params, int64_t tokens_per_step) const {
        const double G = 1e9;
        std::cout << "==================== MFU accounting ====================\n"
                  << std::fixed << std::setprecision(3)
                  << "  device            : " << device_name << " x " << world_size << "\n"
                  << "  peak BF16 (dense) : " << (peak_flops_per_gpu / 1e12) << " TFLOP/s per GPU"
                  << (peak_is_guess ? "   [!] name not in table - set BLU_PEAK_TFLOPS" : "") << "\n"
                  << "  cluster peak      : " << (cluster_peak() / 1e15) << " PFLOP/s\n"
                  << "  fwd FLOPs/token   : " << (fwd_per_token_causal   / G) << " GFLOP  (causal)\n"
                  << "  fwd+bwd /token    : " << (flops_per_token_causal / G) << " GFLOP  (causal, x3) <-- headline\n"
                  << "  fwd+bwd /token    : " << (flops_per_token_dense  / G) << " GFLOP  (PaLM convention)\n"
                  << "  6*N cross-check   : " << (6.0 * static_cast<double>(num_params) / G)
                  << " GFLOP  (weight matmuls only; excludes the T-dependent attention core)\n"
                  << "  model FLOPs/step  : "
                  << (flops_per_token_causal * static_cast<double>(tokens_per_step) / 1e15)
                  << " PFLOP  over " << tokens_per_step << " tokens\n"
                  << "  100% MFU would be : "
                  << (flops_per_token_causal * static_cast<double>(tokens_per_step)
                      / cluster_peak() * 1000.0) << " ms/step\n"
                  << "  note              : no activation checkpointing in this loop, so HFU == MFU\n"
                  << "========================================================" << std::endl;
    }
};

// =============================================================================
// Shared advancing RNG for custom weight init (mirrors gpt2_fmha_ddp.cpp).
// A single std::mt19937 is advanced across every parameter draw so the init is
// reproducible from one seed. Values are drawn on CPU and copy_'d into the
// (device or host) parameter so the registered tensor identity is preserved.
// =============================================================================
struct InitRng {
    std::mt19937 gen;
    explicit InitRng(uint64_t seed) : gen(static_cast<uint32_t>(seed)) {}

    Tensor normal(const Shape& shape, float std) {
        Tensor t(shape, TensorOptions().with_dtype(Dtype::Float32));
        std::normal_distribution<float> dist(0.0f, std);
        float* d = t.data<float>();
        const size_t n = t.numel();
        for (size_t i = 0; i < n; ++i) d[i] = dist(gen);
        return t;
    }
};

// copy_ requires the destination's requires_grad OFF while copying a fresh leaf,
// so toggle it around the copy (matches gpt2_fmha_ddp.cpp::init_linear_gpt2).
static void copy_into_param(Tensor& dst, const Tensor& src) {
    const bool was = dst.requires_grad();
    if (was) dst.set_requires_grad(false);
    dst.copy_(src.dtype() == dst.dtype() ? src : src.as_type(dst.dtype()));  // cast fp32 draw -> bf16
    if (was) dst.set_requires_grad(true);
}

static void init_linear(nn::Linear& layer, float std, InitRng& rng) {
    copy_into_param(layer.weight, rng.normal(layer.weight.shape(), std));
    if (layer.bias.is_valid()) {  // our Llama-style Linears use bias=false, so this is a no-op
        copy_into_param(layer.bias,
                        Tensor::zeros(layer.bias.shape(), TensorOptions().with_dtype(Dtype::Float32)));
    }
}

static void init_embedding(nn::Embedding& emb, float std, InitRng& rng) {
    copy_into_param(emb.weight, rng.normal(emb.weight.shape(), std));
}

// =============================================================================
// Transformer block: RMSNorm -> fused (QK-norm + RoPE + causal GQA) -> Wo -> +x
// =============================================================================
class CausalGQA : public nn::Module {
public:
    nn::RMSNorm norm;            // pre-attention RMSNorm over d_model
    nn::Linear  wQ, wK, wV;      // projections (no bias, Llama style)
    nn::Linear  wO;              // output projection
    Tensor      q_gamma, k_gamma;// [head_dim] fp32 QK-norm scales (learnable)
    Tensor      cos_sin_cache;   // shared [context_length, head_dim] fp32 RoPE cache (no grad)
    ModelConfig cfg;

    CausalGQA(const ModelConfig& c, DeviceIndex dev, const Tensor& cache, InitRng& rng)
        : norm(c.d_model, c.rms_eps, c.resid_dtype()),
          wQ(c.d_model, c.q_heads  * c.head_dim, /*bias=*/false, Dtype::Bfloat16),
          wK(c.d_model, c.kv_heads * c.head_dim, /*bias=*/false, Dtype::Bfloat16),
          wV(c.d_model, c.kv_heads * c.head_dim, /*bias=*/false, Dtype::Bfloat16),
          wO(c.q_heads * c.head_dim, c.d_model,  /*bias=*/false, Dtype::Bfloat16),
          cos_sin_cache(cache),
          cfg(c)
    {
        // GPT-2/Llama-style init: projections std=0.02, residual proj (Wo) scaled
        // by (2*n_layers)^-0.5 to keep residual-stream variance ~constant.
        const float resid_scale = 1.0f / std::sqrt(2.0f * static_cast<float>(c.n_layers));
        init_linear(wQ, 0.02f, rng);
        init_linear(wK, 0.02f, rng);
        init_linear(wV, 0.02f, rng);
        init_linear(wO, 0.02f * resid_scale, rng);

        // QK-norm scales initialized to 1.0 (identity RMS scale). Pinned to fp32
        // by the fused kernel, NOT by bf16_resid — see ModelConfig::qk_dtype().
        auto gopts = TensorOptions().with_dtype(ModelConfig::qk_dtype()).with_device(dev);
        q_gamma = Tensor::full(Shape{{c.head_dim}}, gopts, 1.0f);
        k_gamma = Tensor::full(Shape{{c.head_dim}}, gopts, 1.0f);
        q_gamma.set_requires_grad(true);
        k_gamma.set_requires_grad(true);

        norm.to(dev); wQ.to(dev); wK.to(dev); wV.to(dev); wO.to(dev);

        register_module(norm);
        register_module(wQ); register_module(wK); register_module(wV);
        register_module(wO);
        register_parameter(q_gamma);
        register_parameter(k_gamma);
    }

    Tensor forward(const Tensor& x) override {
        const int64_t B   = x.shape().dims[0];
        const int64_t T   = x.shape().dims[1];
        const int64_t hd  = cfg.head_dim;
        const int64_t Nq  = cfg.q_heads;
        const int64_t Nkv = cfg.kv_heads;

        Tensor h1 = norm.forward(x);                          // resid dtype [B,T,d_model]
        Tensor h  = autograd::to_dtype(h1, Dtype::Bfloat16);  // bf16 input for bf16 Linears
                                                              // (no-op when bf16_resid)

        // bf16 projections. Only q and k enter the fp32 QK-norm + RoPE island, and
        // that island is a kernel constraint, not a bf16_resid knob. v is NOT part
        // of it: it goes straight to scaled_dot_product_attention, which consumes
        // bf16, so widening it here and narrowing it again below was a pure
        // round-trip (~201 MiB/layer/micro-step fwd+bwd). See F2 in CKPT_PERF_AUDIT.md.
        //
        // q/k are left bf16 here too and widened further down, AFTER the head-major
        // layout change, so that copy moves 2 bytes/element instead of 4 (F3).
        const Dtype qk = ModelConfig::qk_dtype();
        Tensor q = wQ.forward(h);   // bf16 [B,T,Nq*hd]
        Tensor k = wK.forward(h);   // bf16 [B,T,Nkv*hd]
        Tensor v = wV.forward(h);   // bf16 [B,T,Nkv*hd]

        Tensor q_bf, k_bf;

#if USE_FUSED_RMS_ROPE
        // Fused QK-norm + RoPE kernel (expects 4D head-major layout [B, H, T, hd], fp32).
        //
        // The kernel needs fp32; the reshape/transpose/contiguous copy that feeds it
        // does not. Doing the copy in bf16 and widening once at the end hands the
        // kernel bit-identical values (bf16 -> fp32 is exact) for half the bytes.
        //
        // Keep to_dtype AFTER contiguous(). Tensor::as_type materializes a strided
        // source itself (the stride contract at the top of core/AsTypeTensor.cpp), so
        // the reverse order is correct -- but it pays for a bf16 copy AND an fp32 one
        // instead of just the bf16 copy, which is the opposite of the point here.
        // BLU_QKROPE_FP32=1 restores the fp32 head-major path above, kept for A/B checks against the bf16 kernel.
        // SFT packed rows (positions since 2026-10-10, in the bf16 kernel since 2026-10-22): both kernels take
        // g_packed.positions = each token's position INSIDE ITS OWN CHAT (0, 1, 2, ... restarting at every chat of
        // the row), forward and backward; an empty Tensor (pretraining, dense rows) means position t as before.
        // The bf16 kernel is bit-identical to the fp32 path, positions included (stage A + B tests,
        // profiling_reports/rope_bf16_port_2026_10_22), and its positions backward is the 2026-10-24 cp.async version.
        static const bool qkrope_fp32 = getenv_int("BLU_QKROPE_FP32", 0) != 0;
        if (!qkrope_fp32) {
            std::tie(q_bf, k_bf) = autograd::fused_rms_rope_bf16(
                q, k, Nq, Nkv, q_gamma, k_gamma, cos_sin_cache,
                /*pos_offset=*/0, cfg.qk_eps, /*interleaved=*/false, g_packed.positions);
        } else {
            Tensor q_4d = autograd::to_dtype(autograd::contiguous(
                autograd::transpose(autograd::reshape(q, Shape{{B, T, Nq,  hd}}), 1, 2)), qk);
            Tensor k_4d = autograd::to_dtype(autograd::contiguous(
                autograd::transpose(autograd::reshape(k, Shape{{B, T, Nkv, hd}}), 1, 2)), qk);

            auto [q_norm_rope, k_norm_rope] = autograd::fused_rms_rope(
                q_4d, k_4d, q_gamma, k_gamma, cos_sin_cache,
                /*pos_offset=*/0, cfg.qk_eps, /*interleaved=*/false, g_packed.positions);

            q_bf = autograd::to_dtype(q_norm_rope, Dtype::Bfloat16);
            k_bf = autograd::to_dtype(k_norm_rope, Dtype::Bfloat16);
        }
#else
        // Sequential QK-norm (RMSNorm) FIRST, then RoPE  (both fp32). The widen has
        // to happen here: rms_norm/rope are fp32-only. reshape is free either way,
        // so this is bandwidth-neutral against casting before the reshape.
        q = autograd::to_dtype(autograd::reshape(q, Shape{{B, T, Nq,  hd}}), qk);
        k = autograd::to_dtype(autograd::reshape(k, Shape{{B, T, Nkv, hd}}), qk);

        if (g_packed.chats)
            throw std::runtime_error("packed SFT rows need USE_FUSED_RMS_ROPE=1 (the sequential RoPE takes no "
                                     "per-chat positions)");
        q = autograd::rms_norm(q, q_gamma, static_cast<int>(hd), cfg.qk_eps);
        k = autograd::rms_norm(k, k_gamma, static_cast<int>(hd), cfg.qk_eps);
        q = autograd::rope(q, cos_sin_cache);
        k = autograd::rope(k, cos_sin_cache);

        q_bf = autograd::transpose(autograd::to_dtype(q, Dtype::Bfloat16), 1, 2);
        k_bf = autograd::transpose(autograd::to_dtype(k, Dtype::Bfloat16), 1, 2);
#endif

        // v never leaves bf16 (F2): wV already produced bf16 and SDPA consumes bf16.
        Tensor v_bf = autograd::transpose(autograd::reshape(v, Shape{{B, T, Nkv, hd}}), 1, 2);

        // 2026-11-06: the call below is unchanged; it now sits in the U == 1 branch (no context parallelism).
        Tensor attn;
        if (g_cp.U == 1) {
        attn = autograd::scaled_dot_product_attention(
            q_bf, k_bf, v_bf, static_cast<int>(Nq), static_cast<int>(Nkv),
            /*is_causal=*/true, /*dropout_p=*/0.0f, autograd::SDPBackend::FlashAttention,   // [B,Nq,T,hd] bf16
            g_packed.chats);   // packed rows: each chat attends only to itself; null = plain causal
        } else {
            // Context parallelism (BLU_SFT_CP=U, see SftCp above). Here T is this rank's slice (T_full / U); q_bf and
            // k_bf already carry RoPE at their true per-chat positions (g_packed.positions was sliced to match).
            // combine: [B, H, T, hd] -> [B, H/U, T*U, hd]; query head e uses KV head e / (Nq / Nkv), and the
            // all-to-all keeps that pairing (rank g receives query heads g*Nq/U.. and KV heads g*Nkv/U..).
            const int U = g_cp.U;
            Tensor qg = cp_combine(autograd::contiguous(q_bf));   // [B, Nq/U,  T_full, hd]
            Tensor kg = cp_combine(autograd::contiguous(k_bf));   // [B, Nkv/U, T_full, hd]
            Tensor vg = cp_combine(autograd::contiguous(v_bf));   // [B, Nkv/U, T_full, hd]
            Tensor attn_g = autograd::scaled_dot_product_attention(
                qg, kg, vg, static_cast<int>(Nq / U), static_cast<int>(Nkv / U),
                /*is_causal=*/true, /*dropout_p=*/0.0f, autograd::SDPBackend::FlashAttention,   // [B,Nq/U,T_full,hd]
                g_packed.chats);   // the FULL row's chat plan, built for Nq/U heads
            attn = cp_partition(autograd::contiguous(attn_g));     // [B, Nq, T, hd]: back to this rank's slice
        }

        Tensor merged = autograd::reshape(autograd::transpose(attn, 1, 2),
                                        Shape{{B, T, Nq * hd}});             // bf16
        Tensor proj = autograd::to_dtype(wO.forward(merged), cfg.resid_dtype());  // bf16 matmul -> resid
        return autograd::add(x, proj);                                           // residual
    }

};

// =============================================================================
// SwiGLU feed-forward: RMSNorm -> gate_up -> swiglu -> down -> +x
//   gate_up: d_model -> 2*ffn_hidden, output laid out [gate | up];
//   swiglu(z) = swish(gate) * up  -> ffn_hidden.
// =============================================================================
class SwiGLUMLP : public nn::Module {
public:
    nn::RMSNorm norm;      // resid dtype (scale dtype must match its input)
    nn::Linear  gate_up;   // d_model -> 2*ffn_hidden   (bf16)
    nn::Linear  down;      // ffn_hidden -> d_model      (bf16)
    Dtype       resid_dt;  // cached: the block does not otherwise keep cfg

    SwiGLUMLP(const ModelConfig& c, DeviceIndex dev, InitRng& rng)
        : norm(c.d_model, c.rms_eps, c.resid_dtype()),
          gate_up(c.d_model, 2 * c.ffn_hidden, /*bias=*/false, Dtype::Bfloat16),
          down(c.ffn_hidden, c.d_model,        /*bias=*/false, Dtype::Bfloat16),
          resid_dt(c.resid_dtype())
    {
        const float resid_scale = 1.0f / std::sqrt(2.0f * static_cast<float>(c.n_layers));
        init_linear(gate_up, 0.02f, rng);
        init_linear(down,    0.02f * resid_scale, rng);

        norm.to(dev); gate_up.to(dev); down.to(dev);
        register_module(norm);
        register_module(gate_up);
        register_module(down);
    }

    Tensor forward(const Tensor& x) override {
        Tensor h     = norm.forward(x);                                    // resid dtype [B,T,d_model]
        Tensor h_bf  = autograd::to_dtype(h, Dtype::Bfloat16);             // bf16 for gate_up
        Tensor gu    = gate_up.forward(h_bf);                              // bf16 [B,T,2*ffn]
        Tensor act   = autograd::swiglu(gu);                               // bf16 activation [B,T,ffn]
        Tensor out   = autograd::to_dtype(down.forward(act), resid_dt);    // bf16 matmul -> resid
        return autograd::add(x, out);                                      // residual
    }
};


// =============================================================================
// Full model
// =============================================================================
class BLU : public nn::Module {
public:
    ModelConfig cfg;
private:
    InitRng init_rng_;            // declared before members it seeds; advanced across all draws
public:
    nn::Embedding wte;            // token embedding [vocab, d_model]
    Tensor        cos_sin_cache;  // [context_length, head_dim] fp32 (shared, no grad)
    Tensor        embed_scale_t;  // [1] sqrt(d_model) in resid dtype (no grad); valid iff cfg.embed_scale
    std::vector<std::shared_ptr<CausalGQA>> attn;
    std::vector<std::shared_ptr<SwiGLUMLP>> mlp;
    nn::RMSNorm   norm_f;         // final RMSNorm
    std::shared_ptr<nn::Linear> lm_head;  // only allocated when weight_tying == false
    // 2026-10-31, the "lm_head on trained rows only" switch (BLU_SFT_TRAINED_ROWS=1, experimental, off by default;
    // SFT work doc 45.131): when set, forward() runs the lm_head only on these rows of [B*T] (Int32 [M], -1 = filler,
    // SftDataLoader::last_trained_rows) and returns logits [M, vocab]. nullptr (the default) = every row, as before.
    const Tensor* head_rows = nullptr;

    BLU(const ModelConfig& c, DeviceIndex dev, uint64_t seed = 1234)
        : cfg(c),
        init_rng_(seed),
        wte(static_cast<int>(c.vocab_size), static_cast<int>(c.d_model),
            /*padding_idx=*/-1, c.embed_dtype()),                                 // embed dtype
        cos_sin_cache(autograd::build_rope_cache(c.context_length, c.head_dim,
                                                static_cast<float>(c.rope_theta), dev)),
        norm_f(c.d_model, c.rms_eps, c.resid_dtype())                             // resid dtype
    {
        init_embedding(wte, 0.02f, init_rng_);
        wte.to(dev);
        norm_f.to(dev);
        if (c.embed_scale) {
            embed_scale_t = Tensor::full(Shape{{1}},
                TensorOptions().with_dtype(c.resid_dtype()).with_device(dev),
                std::sqrt(static_cast<float>(c.d_model)));
        }

        for (int i = 0; i < c.n_layers; ++i) {
            auto a = std::make_shared<CausalGQA>(c, dev, cos_sin_cache, init_rng_);
            auto m = std::make_shared<SwiGLUMLP>(c, dev, init_rng_);
            attn.push_back(a);
            mlp.push_back(m);
            register_module(a.get());
            register_module(m.get());
        }

        if (!c.weight_tying) {
            lm_head = std::make_shared<nn::Linear>(static_cast<int>(c.d_model),
                                                static_cast<int>(c.vocab_size),
                                                /*bias=*/false, Dtype::Bfloat16);  // bf16
            init_linear(*lm_head, 0.02f, init_rng_);
            lm_head->to(dev);
        }

        register_module(wte);
        register_module(norm_f);
        if (!c.weight_tying && lm_head) register_module(lm_head.get());
    }

// indices [B, T] (UInt16) -> logits [B, T, vocab] (logit dtype)
    Tensor forward(const Tensor& idx) override {
        // Embedding is a gather, so its output dtype is wte's. Cast to the
        // residual dtype: the two are independent flags, so bf16_embed without
        // bf16_resid has to widen here (and vice versa).
        Tensor x = autograd::to_dtype(wte.forward(idx), cfg.resid_dtype());  // [B,T,d_model]
        if (cfg.embed_scale) x = autograd::mul(x, embed_scale_t);           // x * sqrt(d_model)
        // Blocks are invoked through Module::operator(), not forward(), so any
        // registered module hook actually fires — a direct forward() call bypasses
        // hook dispatch entirely. The ZeRO-1 staged parameter refresh registers a
        // forward pre-hook on the block that begins each refresh stage (see
        // register_forward_pre_hook in main). With no hooks registered operator()
        // takes a fast path straight to forward(), so this costs nothing.
        for (int i = 0; i < cfg.n_layers; ++i) {
            x = (*attn[i])(x);
            x = (*mlp[i])(x);
        }
        x = norm_f.forward(x);                        // resid dtype

        // Output projection is a matmul -> always do it in bf16. Only the dtype the
        // logits are HANDED BACK in is a flag: fp32 costs a [B,T,vocab] fp32 tensor
        // (3.30 GB at B*T=16384, vocab=50304) purely to widen a bf16 product.
        Tensor x_bf = autograd::to_dtype(x, Dtype::Bfloat16);
        if (head_rows) x_bf = autograd::gather_rows(x_bf, *head_rows);   // [M, d_model]: trained rows only (switch)
        if (cfg.weight_tying) {
            // logits = x @ wte.weight^T ; keep the autograd edge back to wte.weight.
            //
            // Cast BEFORE the transpose (F4(a) in CKPT_PERF_AUDIT.md): the strided
            // copy is the expensive half and it does not need to run at 4
            // bytes/element. wte is [vocab, d_model] = 294.8 MiB fp32, rebuilt every
            // micro-step, so this drops ~30% of ~16.5 GiB/step. No-op when bf16_embed
            // is on, in which case the weight is already bf16.
            //
            // contiguous() stays. Dropping it (the bare-transpose variant that used to
            // sit here) hands matmul a strided B operand, and the forward GEMM wrapper
            // has no trans_b entry point to absorb it -- that needs F4(c). It also used
            // to feed to_dtype a transpose view, which as_type read in storage order
            // and scrambled; that hole is closed at the source now (the stride contract
            // in core/AsTypeTensor.cpp), but the GEMM side of it is not.
            //
            // Cast BEFORE transpose is safe for the BACKWARD too, but only because of
            // that same fix: TransposeBackward returns a strided view, and with the
            // cast now below the transpose in the graph, ToDtypeBackward is what
            // receives that view. Before the stride contract this ordering would have
            // silently permuted wte.weight's gradient under weight tying.
            Tensor w_bf   = autograd::to_dtype(wte.weight, Dtype::Bfloat16);        // [vocab, d_model]
            Tensor w_t_bf = autograd::contiguous(autograd::transpose(w_bf, 0, 1));  // [d_model, vocab]
            Tensor logits = autograd::matmul(x_bf, w_t_bf);                 // bf16 [B,T,vocab]
            return autograd::to_dtype(logits, cfg.logit_dtype());
        }
        return autograd::to_dtype(lm_head->forward(x_bf), cfg.logit_dtype());
    }

};

// =============================================================================
// WSD learning-rate schedule: linear Warmup -> Stable (constant max_lr) ->
// cosine Decay to min_lr over the remaining steps.
// =============================================================================
// BLU_SFT_SCHEDULE=linear (2026-10-28): the decay phase falls linearly instead of along the cosine.
static bool g_lr_linear = false;

static float get_lr(int step, float max_lr, float min_lr, int warmup, int max_steps, int stable_steps) {
    if (step < warmup) {
        return max_lr * static_cast<float>(step + 1) / static_cast<float>(warmup);
    }
    if ((step >= warmup) && (step < (stable_steps + warmup))) {
        return max_lr;
    }
    float decay_ratio = static_cast<float>(step - (stable_steps + warmup)) /
                        static_cast<float>((max_steps - (stable_steps + warmup)) - 1);
    assert(decay_ratio >= 0.0f && decay_ratio <= 1.0f);

    if (g_lr_linear) return min_lr + (max_lr - min_lr) * (1.0f - decay_ratio);

    // cosine decay
    const float coeff = 0.5f * (1.0f + std::cos(M_PI * decay_ratio));  // 1 -> 0
    return min_lr + coeff * (max_lr - min_lr);

    // linear decay
    //return min_lr + (max_lr-min_lr) * (1.0f - decay_ratio);
}

// =============================================================================
// Weight export (BLU_EXPORT=<path>.safetensors) — final model, rank 0 only. Copied from preproduction-135M
// f37c616 (2026-10-28) so an SFT run ends with a file sft/blutrain_to_hf.py turns into a Hugging Face model.
// Every tensor is written fp32 under a structural name, in this trainer's
// layout: Linear weights are [in, out] (y = x @ W), gate_up is [gate | up]
// along the out axis. The architecture rides in __metadata__ so a converter
// needs nothing else.
// =============================================================================
static void export_safetensors(BLU& model, const std::string& path, int steps) {
    const ModelConfig& c = model.cfg;
    std::vector<std::pair<std::string, Tensor>> named;
    named.push_back({"wte.weight", model.wte.weight});
    for (int i = 0; i < c.n_layers; ++i) {
        const std::string L = "layers." + std::to_string(i) + ".";
        named.push_back({L + "attn_norm.weight", model.attn[i]->norm.weight});
        named.push_back({L + "wq.weight",        model.attn[i]->wQ.weight});
        named.push_back({L + "wk.weight",        model.attn[i]->wK.weight});
        named.push_back({L + "wv.weight",        model.attn[i]->wV.weight});
        named.push_back({L + "wo.weight",        model.attn[i]->wO.weight});
        named.push_back({L + "q_gamma",          model.attn[i]->q_gamma});
        named.push_back({L + "k_gamma",          model.attn[i]->k_gamma});
        named.push_back({L + "mlp_norm.weight",  model.mlp[i]->norm.weight});
        named.push_back({L + "gate_up.weight",   model.mlp[i]->gate_up.weight});
        named.push_back({L + "down.weight",      model.mlp[i]->down.weight});
    }
    named.push_back({"norm_f.weight", model.norm_f.weight});
    if (!c.weight_tying && model.lm_head) named.push_back({"lm_head.weight", model.lm_head->weight});

    std::vector<Tensor> host;
    std::ostringstream hdr;
    hdr << "{\"__metadata__\":{"
        << "\"format\":\"blutrain\",\"linear_layout\":\"in_out\""
        << ",\"d_model\":\"" << c.d_model << "\",\"n_layers\":\"" << c.n_layers << "\""
        << ",\"q_heads\":\"" << c.q_heads << "\",\"kv_heads\":\"" << c.kv_heads << "\""
        << ",\"head_dim\":\"" << c.head_dim << "\",\"ffn_hidden\":\"" << c.ffn_hidden << "\""
        << ",\"vocab_size\":\"" << c.vocab_size << "\",\"context_length\":\"" << c.context_length << "\""
        << ",\"rope_theta\":\"" << c.rope_theta << "\",\"rms_eps\":\"" << c.rms_eps << "\""
        << ",\"qk_eps\":\"" << c.qk_eps << "\",\"embed_scale\":\"" << (c.embed_scale ? 1 : 0) << "\""
        << ",\"weight_tying\":\"" << (c.weight_tying ? 1 : 0) << "\",\"steps\":\"" << steps << "\"}";
    uint64_t off = 0;
    for (auto& [name, t] : named) {
        Tensor h = t.as_type(Dtype::Float32).to_cpu();
        const uint64_t nbytes = static_cast<uint64_t>(h.numel()) * sizeof(float);
        hdr << ",\"" << name << "\":{\"dtype\":\"F32\",\"shape\":[";
        const auto& dims = h.shape().dims;
        for (size_t d = 0; d < dims.size(); ++d) hdr << (d ? "," : "") << dims[d];
        hdr << "],\"data_offsets\":[" << off << "," << (off + nbytes) << "]}";
        off += nbytes;
        host.push_back(std::move(h));
    }
    hdr << "}";
    std::string header = hdr.str();
    while (header.size() % 8 != 0) header.push_back(' ');

    const std::string tmp = path + ".tmp";
    std::ofstream f(tmp, std::ios::binary);
    const uint64_t hlen = header.size();
    f.write(reinterpret_cast<const char*>(&hlen), sizeof hlen);
    f.write(header.data(), static_cast<std::streamsize>(header.size()));
    for (auto& h : host)
        f.write(reinterpret_cast<const char*>(h.data<float>()),
                static_cast<std::streamsize>(h.numel() * sizeof(float)));
    f.close();
    if (!f) throw std::runtime_error("export_safetensors: write failed: " + tmp);
    std::filesystem::rename(tmp, path);
    std::cout << "[export] wrote " << named.size() << " tensors (" << (off >> 20)
              << " MiB fp32) to " << path << std::endl;
}

// =============================================================================
// Training
// =============================================================================

static bool dir_has_checkpoints(const std::filesystem::path& dir, const std::string& prefix) {
    namespace sfs = std::filesystem;
    std::error_code ec;
    const std::string pre = prefix + "_step_";
    for (const auto& e : sfs::directory_iterator(dir, ec)) {
        const std::string d = e.path().filename().string();
        if (e.is_directory(ec) && d.size() > pre.size() && d.compare(0, pre.size(), pre) == 0 &&
            d.find_first_not_of("0123456789", pre.size()) == std::string::npos)
            return true;
    }
    return false;
}

// Runs of one base are <base>, <base>_001, <base>_002, ...; the highest is the latest.
// Fresh never reuses a run dir that has checkpoints, so it cannot load or prune one.
static std::string resolve_ckpt_run_dir(const std::string& base_in, const std::string& prefix,
                                        bool resume) {
    namespace sfs = std::filesystem;
    sfs::path base = sfs::path(base_in).lexically_normal();
    if (base.filename().empty()) base = base.parent_path();
    const sfs::path parent = base.parent_path().empty() ? sfs::path(".") : base.parent_path();
    const std::string stem = base.filename().string();

    int latest = -1;
    std::error_code ec;
    if (sfs::is_directory(base, ec)) latest = 0;
    for (const auto& e : sfs::directory_iterator(parent, ec)) {
        const std::string d = e.path().filename().string();
        if (!e.is_directory(ec) || d.size() < stem.size() + 4 ||
            d.compare(0, stem.size() + 1, stem + "_") != 0) continue;
        const std::string digits = d.substr(stem.size() + 1);
        if (digits.find_first_not_of("0123456789") != std::string::npos) continue;
        latest = std::max(latest, std::stoi(digits));
    }

    auto run_dir = [&](int i) {
        if (i == 0) return base;
        char suffix[16];
        std::snprintf(suffix, sizeof suffix, "_%03d", i);
        return base.parent_path() / (stem + suffix);
    };
    if (latest < 0) return base.string();
    if (resume) return run_dir(latest).string();
    if (!dir_has_checkpoints(run_dir(latest), prefix)) return run_dir(latest).string();
    return run_dir(latest + 1).string();
}

// Host-side deadline for the whole step: the NCCL watchdog cannot see hangs in MPI calls,
// the data loader, validation or checkpoint I/O. Exits non-zero so mpirun stops every rank.
class StepWatchdog {
public:
    void start(int timeout_s, int rank) {
        if (timeout_s <= 0) return;
        timeout_s_ = timeout_s;
        rank_ = rank;
        beat("startup", -1);
        thread_ = std::thread([this] { loop(); });
    }
    void beat(const char* phase, int step) {
        phase_.store(phase, std::memory_order_relaxed);
        step_.store(step, std::memory_order_relaxed);
        last_.store(now_s(), std::memory_order_relaxed);
    }
    void stop() {
        stop_.store(true);
        if (thread_.joinable()) thread_.join();
    }
    ~StepWatchdog() { stop(); }

private:
    static double now_s() {
        return std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    void loop() {
        while (!stop_.load()) {
            for (int i = 0; i < 20 && !stop_.load(); ++i)
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            const double idle = now_s() - last_.load(std::memory_order_relaxed);
            if (!stop_.load() && idle > timeout_s_) {
                std::fprintf(stderr,
                    "\n[step-watchdog] rank %d: no progress for %.0f s (limit %d s, "
                    "training.step_timeout_s) in phase '%s' at step %d. Stopping this rank "
                    "(exit 98) so mpirun stops the job.\n",
                    rank_, idle, timeout_s_, phase_.load(), step_.load());
                std::fflush(nullptr);
                std::_Exit(98);
            }
        }
    }
    std::atomic<const char*> phase_{"startup"};
    std::atomic<int>         step_{-1};
    std::atomic<double>      last_{0.0};
    std::atomic<bool>        stop_{false};
    std::thread              thread_;
    int                      timeout_s_ = 0;
    int                      rank_      = 0;
};

// =============================================================================
// Training
// =============================================================================
int main(int argc, char** argv) {
    zcfg::Report cfg_report;
    YAML::Node yaml_cfg;
    const bool print_config_only = zcfg::take_flag(&argc, &argv, "--print-config");
    try {
        yaml_cfg = zcfg::load_and_export(&argc, &argv, cfg_report);
    } catch (const std::exception& e) {
        std::cerr << "[config] FATAL: " << e.what() << std::endl;
        return 2;
    }
    // backend bluccl across nodes: BluCCL internode RS + AG (composite group), set before MPI/CUDA start. EAGER: a
    // lazily loaded kernel can wait on BluCCL's resident ones; NCCL_GIN_ENABLE=0: NCCL only all-reduces here.
    const int launch_nodes = [] {
        if (const char* n = std::getenv("BLUCCL_NUM_NODES"); n && std::atoi(n) > 0) return std::atoi(n);
        const char* w = std::getenv("OMPI_COMM_WORLD_SIZE");
        const char* l = std::getenv("OMPI_COMM_WORLD_LOCAL_SIZE");
        return (w && l && std::atoi(l) > 0) ? std::max(1, std::atoi(w) / std::atoi(l)) : 1;
    }();
    const bool bluccl_multi = getenv_str("BLU_BACKEND", "nccl") == "bluccl" && launch_nodes > 1;
    if (bluccl_multi)
        for (const auto& kv : std::vector<std::pair<const char*, const char*>>{
                 {"CUDA_MODULE_LOADING", "EAGER"}, {"BLU_IB_RS", "1"}, {"BLU_IB_RS_FLAT", "1"},
                 {"BLU_IB_AG_OVERLAP", "1"}, {"BLU_SYMM_IB", "1"}, {"NCCL_GIN_ENABLE", "0"}})
            ::setenv(kv.first, kv.second, /*overwrite=*/0);
    const bool user_set_fused_emb = std::getenv("BLUCCL_FUSED_EMB") != nullptr;
    if (const char* m = std::getenv("BLU_EMB_BWD"); m && *m) {
        const std::string mode = m;
        if (mode == "segmented") {
            ::setenv("BLUCCL_FUSED_EMB", "1", 0);
        } else if (mode == "atomic" || mode == "cooperative") {
            ::setenv("BLUTRAIN_EMB_BWD", m, 0);
            ::setenv("BLUCCL_FUSED_EMB", "0", 0);
        } else {
            std::cerr << "[config] FATAL: BLU_EMB_BWD / embedding.backward must be atomic, "
                         "cooperative or segmented (got '" << mode << "')" << std::endl;
            return 2;
        }
    }
    auto bind_config = [&](ModelConfig& c) {
        zcfg::bind_fields(yaml_cfg, {
            {"hyper-parameters", "context", &c.context_length}, {"hyper-parameters", "context", &c.T},
            {"hyper-parameters", "vocab", &c.vocab_size}, {"hyper-parameters", "n_embd", &c.d_model},
            {"hyper-parameters", "n_layers", &c.n_layers}, {"hyper-parameters", "n_heads", &c.q_heads},
            {"hyper-parameters", "kv_heads", &c.kv_heads}, {"hyper-parameters", "head_dim", &c.head_dim},
            {"hyper-parameters", "ffn_hidden", &c.ffn_hidden}, {"hyper-parameters", "rope_theta", &c.rope_theta},
            {"hyper-parameters", "qk_eps", &c.qk_eps}, {"hyper-parameters", "rms_eps", &c.rms_eps},
            {"hyper-parameters", "rope_interleaved", &c.rope_interleaved},
            {"hyper-parameters", "weight_tying", &c.weight_tying},
            {"hyper-parameters", "global_batch", &c.global_batch},
            {"hyper-parameters", "mini_batch", &c.B}, {"hyper-parameters", "seed", &c.seed},
            {"training", "total_steps", &c.max_steps}, {"training", "warmup_steps", &c.warmup_steps},
            {"training", "stable_steps", &c.stable_steps}, {"training", "validation_freq", &c.val_freq},
            {"training", "val_loss_steps", &c.val_loss_steps}, {"training", "grad_clip", &c.grad_clip},
            {"optimizer", "max_lr", &c.max_lr}, {"optimizer", "min_lr", &c.min_lr},
            {"optimizer", "beta1", &c.beta1}, {"optimizer", "beta2", &c.beta2},
            {"optimizer", "eps", &c.adam_eps}, {"optimizer", "weight_decay", &c.weight_decay},
            {"optimizer", "muon_lr", &c.muon_lr}, {"optimizer", "muon_min_lr", &c.muon_min_lr},
            {"optimizer", "muon_momentum", &c.muon_momentum},
            {"optimizer", "muon_weight_decay", &c.muon_weight_decay},
            {"optimizer", "muon_ns_steps", &c.muon_ns_steps},
            {"checkpointing", "prefix", &c.ckpt_prefix},
        }, cfg_report);
        // BLU_TEST_HD64=1 lets small GPUs (no causal 128 kernel) exercise the pipeline, e.g. logging and dashboards.
        const bool hd64_test = c.head_dim == 64 && getenv_int("BLU_TEST_HD64", 0) != 0;
        if (c.head_dim != 128 && !hd64_test)
            throw std::runtime_error("hyper-parameters.head_dim must be 128 (got " +
                                     std::to_string(c.head_dim) + "): only the 128 path has causal "
                                     "attention kernels on H100/H200; causal 64 silently runs non-causal");
        if (hd64_test)
            std::cerr << "WARNING: BLU_TEST_HD64=1: head_dim 64 attention is causal only on B300; on other GPUs it "
                         "runs NON-causal, so losses are not real results (pipeline/dashboard tests only)." << std::endl;
    };
    auto print_config = [&]() {
        std::cout << "[config] " << (cfg_report.path.empty() ? "no YAML (env and built-in defaults only)"
                                                             : cfg_report.path) << "\n";
        if (!cfg_report.parallel_path.empty())
            std::cout << "[config] parallelism: " << cfg_report.parallel_path << "\n";
        for (const auto& l : cfg_report.fields)   std::cout << "[config]   " << l << "\n";
        for (const auto& l : cfg_report.applied)  std::cout << "[config]   " << l << "\n";
        for (const auto& l : cfg_report.kept_env) std::cout << "[config]   " << l << "\n";
        std::cout << std::flush;
    };
    if (print_config_only) {
        ModelConfig c;
        try { bind_config(c); } catch (const std::exception& e) {
            std::cerr << "[config] FATAL: " << e.what() << std::endl;
            return 2;
        }
        print_config();
        return 0;
    }

    // Rank, world size, LOCAL rank and (under a non-MPI launcher) the
    // pre-minted ncclUniqueId, resolved from whichever launcher started us.
    // Initialises MPI when one is present. See dist/communication/include/
    // LaunchEnv.h for why every one of those four differs between mpirun and
    // BluRun, and why the local rank is the only legal device ordinal.
    const dist::comm::LaunchInfo launch = dist::comm::detect_launch(&argc, &argv);

    // Allocator size-class policy. MUST be set before the first allocation, so
    // it goes here rather than with the other env parsing further down.
    //
    // BLU_SIZE_CLASS=exact turns on generic tight rounding. The "default"
    // policy inflates every request in [1 MiB, 10 MiB) to a dedicated 20 MiB
    // block and every request below 1 MiB to a 2 MiB block, which on THIS
    // config (3.93 B) wastes 1,590 MiB on parameters plus FP32 gradients:
    //
    //   1,008 MiB   the 72 BF16 wK/wV, 6 MiB each -> 20 MiB each
    //     578 MiB   290 sub-1 MiB tensors (145 1-D scales + their FP32 grads),
    //               2.4 MiB of data in 580 MiB of blocks
    //       4 MiB   wte/lm_head rounding
    //
    // `exact` takes that to 1.5 MiB. Measured, see docs/ZERO1_MEMORY_AUDIT.md §8.
    //
    // ShardedMuon's per-tensor FP32 owned state costs NOTHING extra here — at
    // 3.93 B every owned matrix is 12/36/96/192 MiB FP32, above the 10 MiB
    // threshold and already 2 MiB-aligned. That was the second-largest item at
    // 681 M (2,016 MiB) purely because its owned shapes were 3 and 9 MiB.
    //
    // Left OFF by default: tight classes cut internal fragmentation but give up
    // block reuse across sizes, so judge the A/B on reserved_peak rather than on
    // active_current. Any mode name from SizeClassConfigDictionary works here.
    if (const char* sc = std::getenv("BLU_SIZE_CLASS"); sc && *sc) {
        OwnTensor::SizeClass::set_mode(sc);
    }

    const int  rank       = launch.rank;         // GLOBAL rank
    const int  world_size = launch.world_size;
    const int  local_rank = launch.local_rank;   // device ordinal on THIS node
    const bool is_master  = (rank == 0);

    // One GPU per rank, bound by the LOCAL rank. Binding by the global rank is
    // what makes a trainer single-node: on the second node of a 2x8 job, rank 8
    // asks for device 8 and the host has eight. detect_launch has already
    // validated that local_rank is inside this node's visible devices.
    CUDACHECK(cudaSetDevice(local_rank));
    DeviceIndex device(Device::CUDA, local_rank);

    if (is_master) {
        std::cout << "[launch] launcher=" << dist::comm::launcher_name(launch.launcher)
                  << " world_size=" << world_size
                  << " mpi_world_spans_job=" << (launch.mpi_world_matches ? "yes" : "no")
                  << std::endl;
    }

    ModelConfig cfg;
    try {
        bind_config(cfg);
    } catch (const std::exception& e) {
        if (is_master) std::cerr << "[config] FATAL: " << e.what() << std::endl;
        dist::comm::abort_job(launch, 2);
    }
    if (is_master) print_config();
    if (getenv_int("BLU_PARALLELISM", 1) == 0 && world_size > 1) {
        if (is_master)
            std::cerr << "ERROR: parallelism.enabled is false but the job has " << world_size
                      << " ranks. Enable it or launch with NP=1." << std::endl;
        dist::comm::abort_job(launch, 2);
    }

    // Model shape overrides (2026-10-22). A checkpoint stores the weights, not the config, so the run must build the
    // model the checkpoint was trained as. BLU_N_LAYERS / BLU_ROPE_THETA override ModelConfig / the YAML (on this
    // branch 26 / 10000; it was 26 / 500000 in the tree this SFT code came from, so always set both for a checkpoint);
    // e.g. navingv/fullchess has 28 layers (282 weights in its metadata.bin, 10 per layer + embedding + final norm).
    // A wrong layer count stops the BLU_INIT_FROM load ("[Architecture Mismatch]"). A wrong theta does NOT (no weight
    // depends on it): it must be taken from the run that made the checkpoint. Both are printed in the config block.
    cfg.n_layers   = getenv_int("BLU_N_LAYERS", static_cast<int>(cfg.n_layers));
    cfg.rope_theta = getenv_int("BLU_ROPE_THETA", cfg.rope_theta);
    // 135M base (2026-10-27): the other shape knobs and the embedding scale, with the names of preproduction-135M
    // f37c616 (BLU_VOCAB from its _135m_eqab trainer). Unset = ModelConfig / the YAML. All must be the values of the
    // run that made BLU_INIT_FROM: a wrong width, head count, ffn or vocab stops the load (shape mismatch), a wrong
    // BLU_EMBED_SCALE does NOT (no weight depends on it). The 135M base: 512 / 4 / 2 / 1408 / 49280, embed_scale on.
    cfg.d_model     = getenv_int("BLU_D_MODEL",    static_cast<int>(cfg.d_model));
    cfg.q_heads     = getenv_int("BLU_Q_HEADS",    static_cast<int>(cfg.q_heads));
    cfg.kv_heads    = getenv_int("BLU_KV_HEADS",   static_cast<int>(cfg.kv_heads));
    cfg.ffn_hidden  = getenv_int("BLU_FFN_HIDDEN", static_cast<int>(cfg.ffn_hidden));
    cfg.vocab_size  = getenv_int("BLU_VOCAB",      static_cast<int>(cfg.vocab_size));
    cfg.embed_scale = getenv_int("BLU_EMBED_SCALE", cfg.embed_scale ? 1 : 0) != 0;
    // BLU_SFT_T (2026-10-28, SFT at 8192): row length = RoPE cache length; must equal the data's meta.json "T" (the
    // loader checks). Unset = the YAML's context.
    cfg.T = cfg.context_length = getenv_int("BLU_SFT_T", static_cast<int>(cfg.T));
    if (cfg.kv_heads <= 0 || cfg.q_heads % cfg.kv_heads != 0 || cfg.d_model % 64 != 0 || cfg.ffn_hidden % 64 != 0
        || cfg.q_heads * cfg.head_dim != cfg.d_model) {
        if (is_master)
            std::cerr << "ERROR: bad shape override: q_heads must be a multiple of kv_heads, d_model and ffn_hidden "
                         "multiples of 64, and q_heads x head_dim = d_model" << std::endl;
        dist::comm::abort_job(launch, 1);
    }

    // OwnTensor::AllocationTracker::instance().init("4b_tracker.csv");

    
    // ---- activation / parameter precision ----------------------------------
    // BLU_ALL_BF16=1 turns on all three stages at once. Each stage then has its
    // own knob that overrides the master, so a regression bisects without
    // editing anything:
    //
    //   BLU_ALL_BF16=1                       all three (residual, logits, embed)
    //   BLU_ALL_BF16=1 BLU_BF16_EMBED=0      residual + logits only
    //   BLU_BF16_LOGITS=1                    logits only
    //   (unset)                              shipped mixed precision, unchanged
    //
    // The default is bit-identical to the pre-flag code: every to_dtype() this
    // touches resolves to Float32, and autograd::to_dtype returns its input
    // unchanged when the dtypes already match, so no cast node is even built.
    //
    // Optimizer masters/moments stay fp32 under all of these — that is a kernel
    // constraint, not an omission. See the ModelConfig precision block.
    {
        const int all = getenv_int("BLU_ALL_BF16", 0);
        cfg.bf16_resid  = getenv_int("BLU_BF16_RESID",  all) != 0;
        cfg.bf16_logits = getenv_int("BLU_BF16_LOGITS", all) != 0;
        cfg.bf16_embed  = getenv_int("BLU_BF16_EMBED",  all) != 0;

        // bf16 logits are now consumable: LossKernels.cu's vectorized forward and
        // both backward variants size every 16-byte chunk as 16/sizeof(T) elements
        // (VEC_ELEMS/Chunk16), so the cp.async source is 16-byte aligned on a 2-byte
        // dtype and the 128-bit gradient store no longer overruns the row. This is
        // the single largest memory item in the model: [B,T,vocab] fp32 is 3.07 GiB
        // retained for the CE backward plus a 3.07 GiB fp32 gradient, and the matmul
        // already produces bf16 — the widen existed only to feed the loss kernels.
        //
        // It is NOT free, and the cost is not numerical in the gradients: the loss
        // SCALAR comes back bf16 too (LossOps keys it off the logits dtype), which
        // quantizes the reported loss to ~0.06 at this batch size. See the
        // ModelConfig precision block. So warn every time rather than staying quiet,
        // because the affected number is the one an A/B is read from.
        if (cfg.bf16_logits && is_master) {
            std::cout << "[precision] bf16 logits ON — frees ~3 GiB ([B,T,vocab] fp32 "
                         "activation + gradient). Gradients are unaffected (fp32 softmax "
                         "stats, exact 1.0 backward seed), but the REPORTED loss is bf16 "
                         "and quantizes to ~0.06 steps: compare tokens/s and grad-norm "
                         "across an A/B, not the loss digits." << std::endl;
        }
    }

    // BLU_MAX_STEPS caps the run for benchmarking (the A/B harness in
    // run_zero_ab_8xh200.sh uses it). The WSD schedule divides by
    // (max_steps - warmup - stable - 1) and asserts the range below, so shrink
    // warmup/stable proportionally rather than leaving them at their full-run
    // values — otherwise a short benchmark aborts on "no decay tail".
    if (const int ms = getenv_int("BLU_MAX_STEPS", 0); ms > 0) {
        cfg.max_steps    = ms;
        cfg.warmup_steps = std::max(1, ms / 10);
        cfg.stable_steps = std::max(1, ms / 2);
        if (cfg.warmup_steps + cfg.stable_steps >= cfg.max_steps - 1) {
            cfg.stable_steps = std::max(1, cfg.max_steps - cfg.warmup_steps - 2);
        }
        if (is_master) {
            std::cout << "[bench] BLU_MAX_STEPS=" << ms
                      << " (warmup=" << cfg.warmup_steps
                      << " stable=" << cfg.stable_steps << ")" << std::endl;
        }
    }

    // ---- SFT mode: BLU_SFT_DATA=<dir> ----
    // <dir>/train and <dir>/val are written by sft/prepare_sft_data.py. In this mode the trainer
    //   * reads its batches through sft/SftDataLoader.h (targets from labels.bin, 65535 = not trained)
    //     instead of the pretraining mixture, which is never opened, so BLU_MIX_ROOT is not needed;
    //   * divides every micro-step's loss by the GLOBAL count of trained tokens in the step (all
    //     micro-steps, all ranks) and seeds backward with world_size instead of 1/GRAD_ACCUM, because
    //     the ZeRO optimizers average over ranks. Every trained token of a step then carries the same
    //     1/N, which is torchtitan's global_valid_tokens (loss_calculation_in_pretraining_and_sft.md);
    //   * takes its schedule from these knobs (defaults = the stage-0 plan):
    //       BLU_SFT_ROWS_PER_STEP  64    rows (conversations) per optimizer step, all ranks together
    //       BLU_SFT_B              8     rows per micro-batch per rank (bluscript_sft.cpp only, 2026-10-15);
    //                                    ROWS_PER_STEP must be a multiple of B x ranks. B=1 uses every row.
    //       BLU_SFT_EPOCHS         2     max_steps = epochs * micro-batches per epoch / GRAD_ACCUM
    //       BLU_SFT_LR             2e-5  AdamW peak LR; BLU_SFT_MIN_LR (default LR / 10) ends the cosine
    //       BLU_SFT_WARMUP         0.03  linear warmup as a fraction of max_steps, then cosine, no stable phase
    //       BLU_SFT_SCHEDULE       cosine  cosine | linear (Tulu 3) | constant (min = peak) | wsd (constant, then
    //                                    cosine over the last BLU_SFT_DECAY_FRAC = 0.2 of the steps, as pretraining)
    //       BLU_SFT_VAL_FREQ       10    validation every N steps, over the whole val set
    //       BLU_SFT_MUON_LR        -     required with BLU_OPTIMIZER=muon: the pretraining Muon LR would
    //                                    wreck the weights. BLU_SFT_MUON_MIN_LR defaults to it / 10
    //   * checkpoints into checkpoints_sft[_muon] unless BLU_CKPT_DIR is set, so it never auto-resumes a
    //     pretraining checkpoint left in the default directory.
    // BLU_MAX_STEPS is ignored in this mode: the schedule comes from the epochs.
    const std::string sft_dir = getenv_str("BLU_SFT_DATA", "");
    const bool sft = !sft_dir.empty();
    std::unique_ptr<SftDataLoader> sft_train, sft_val;
    bool trained_rows = false;                    // BLU_SFT_TRAINED_ROWS (2026-10-31, experimental, off by default)
    if (sft) {
        cfg.B = getenv_int("BLU_SFT_B", static_cast<int>(cfg.B));   // rows per micro-batch per rank (2026-10-15)
        if (cfg.B <= 0) {
            if (is_master) std::cerr << "ERROR: BLU_SFT_B must be positive" << std::endl;
            dist::comm::abort_job(launch, 1);
        }
        const int64_t rows_per_step = getenv_int("BLU_SFT_ROWS_PER_STEP", 64);
        const int64_t epochs        = getenv_int("BLU_SFT_EPOCHS", 2);
        const float   warmup_frac   = getenv_float("BLU_SFT_WARMUP", 0.03f);
        // 2026-11-06: context parallelism, BLU_SFT_CP=U (1 = off). The U ranks rank/U*U .. rank/U*U + U - 1 form one CP
        // group: same rows, one token slice each. So rows are picked by the DATA-parallel rank (rank / U of
        // world_size / U), and the row checks below count data-parallel ranks.
        g_cp.U = getenv_int("BLU_SFT_CP", 1);
        g_cp.rank = rank % std::max(1, g_cp.U);
        if (g_cp.U < 1 || world_size % g_cp.U != 0 || cfg.q_heads % g_cp.U != 0 || cfg.kv_heads % g_cp.U != 0 ||
            cfg.T % g_cp.U != 0) {
            if (is_master)
                std::cerr << "ERROR: BLU_SFT_CP (" << g_cp.U << ") must be >= 1 and divide world_size (" << world_size
                          << "), the query heads (" << cfg.q_heads << "), the KV heads (" << cfg.kv_heads
                          << ") and the row length T (" << cfg.T << ")" << std::endl;
            dist::comm::abort_job(launch, 1);
        }
        const int dp_rank = rank / g_cp.U, dp_world = world_size / g_cp.U;
        if (is_master && g_cp.U > 1)
            std::cout << "[sft] BLU_SFT_CP=" << g_cp.U << ": context parallelism, " << dp_world << " data-parallel x "
                      << g_cp.U << " CP ranks; each rank holds " << cfg.T / g_cp.U << " of the " << cfg.T
                      << " tokens of every row (Ulysses all-to-all around our attention)" << std::endl;
        // The previous line, kept for reference (= the line below at U == 1):
        // const int64_t rows_per_micro = cfg.B * static_cast<int64_t>(world_size);
        const int64_t rows_per_micro = cfg.B * static_cast<int64_t>(dp_world);
        if (rows_per_step <= 0 || rows_per_step % rows_per_micro != 0 || epochs <= 0) {
            if (is_master)
                std::cerr << "ERROR: BLU_SFT_ROWS_PER_STEP (" << rows_per_step << ") must be a positive multiple "
                          << "of B * world_size (" << rows_per_micro << "), and BLU_SFT_EPOCHS (" << epochs
                          << ") positive" << std::endl;
            dist::comm::abort_job(launch, 1);
        }
        // rank / world_size pick this rank's rows; local_rank is the GPU the batches are copied to (a device
        // ordinal is node-local: the global rank would ask for device 8 on the second node of a 2 x 8 job).
        // cfg.vocab_size: ids are checked against the model's rows (49,280 for the 135M base, 50,304 for the 1.7B).
        // 2026-11-06: rank / world_size -> dp_rank / dp_world (the same at U == 1). The previous lines, kept:
        // sft_train = std::make_unique<SftDataLoader>(sft_dir + "/train", static_cast<int>(cfg.B),
        //                                             static_cast<int>(cfg.T), rank, world_size, local_rank, is_master,
        //                                             cfg.vocab_size);
        // sft_val   = std::make_unique<SftDataLoader>(sft_dir + "/val", static_cast<int>(cfg.B),
        //                                             static_cast<int>(cfg.T), rank, world_size, local_rank, is_master,
        //                                             cfg.vocab_size);
        sft_train = std::make_unique<SftDataLoader>(sft_dir + "/train", static_cast<int>(cfg.B),
                                                    static_cast<int>(cfg.T), dp_rank, dp_world, local_rank, is_master,
                                                    cfg.vocab_size);
        sft_val   = std::make_unique<SftDataLoader>(sft_dir + "/val", static_cast<int>(cfg.B),
                                                    static_cast<int>(cfg.T), dp_rank, dp_world, local_rank, is_master,
                                                    cfg.vocab_size);
        // 2026-10-31: the "lm_head on trained rows only" switch, experimental, OFF by default (SFT work doc 45.131).
        trained_rows = getenv_int("BLU_SFT_TRAINED_ROWS", 0) != 0;
        if (trained_rows && g_cp.U > 1) {   // 2026-11-06: its row lists are for whole rows, not a rank's token slice
            if (is_master) std::cerr << "ERROR: BLU_SFT_TRAINED_ROWS=1 does not work with BLU_SFT_CP > 1" << std::endl;
            dist::comm::abort_job(launch, 1);
        }
        sft_train->set_trained_rows(trained_rows);
        if (is_master && trained_rows)
            std::cout << "[sft] BLU_SFT_TRAINED_ROWS=1: lm_head + loss on trained rows only (experimental)\n";
        const int64_t ga = rows_per_step / rows_per_micro;
        cfg.global_batch   = rows_per_step * cfg.T;           // so GRAD_ACCUM below comes out as ga
        cfg.max_steps      = static_cast<int>(epochs * sft_train->micro_batches_per_epoch() / ga);
        cfg.warmup_steps   = std::max(1, static_cast<int>(std::lround(warmup_frac * cfg.max_steps)));
        cfg.stable_steps   = 0;
        cfg.max_lr         = getenv_float("BLU_SFT_LR", 2e-5f);
        cfg.min_lr         = getenv_float("BLU_SFT_MIN_LR", cfg.max_lr / 10.0f);
        cfg.val_freq       = getenv_int("BLU_SFT_VAL_FREQ", 10);
        cfg.val_loss_steps = static_cast<int>(sft_val->micro_batches_per_epoch());
        cfg.ckpt_dir       = "checkpoints_sft";
        const char* opt = std::getenv("BLU_OPTIMIZER");
        if (opt && (std::string(opt) == "muon" || std::string(opt) == "MUON")) {
            cfg.muon_lr = getenv_float("BLU_SFT_MUON_LR", 0.0f);
            if (cfg.muon_lr <= 0.0f) {
                if (is_master)
                    std::cerr << "ERROR: BLU_OPTIMIZER=muon in SFT mode needs BLU_SFT_MUON_LR (the pretraining "
                                 "Muon LR would wreck the weights)" << std::endl;
                dist::comm::abort_job(launch, 1);
            }
            cfg.muon_min_lr = getenv_float("BLU_SFT_MUON_MIN_LR", cfg.muon_lr / 10.0f);
        }
        // Schedule (2026-10-28, stage 2 of the 135M SFT grid). Every decaying shape ends at the same min LR.
        const std::string sched = getenv_str("BLU_SFT_SCHEDULE", "cosine");
        if (sched == "linear") {
            g_lr_linear = true;
        } else if (sched == "constant") {
            cfg.min_lr = cfg.max_lr;
            cfg.muon_min_lr = cfg.muon_lr;
        } else if (sched == "wsd") {
            const float df = getenv_float("BLU_SFT_DECAY_FRAC", 0.2f);
            const int decay = std::max(2, static_cast<int>(std::lround(df * cfg.max_steps)));
            cfg.stable_steps = std::max(0, cfg.max_steps - cfg.warmup_steps - decay);
        } else if (sched != "cosine") {
            if (is_master) std::cerr << "ERROR: BLU_SFT_SCHEDULE must be cosine, linear, constant or wsd" << std::endl;
            dist::comm::abort_job(launch, 1);
        }
        if (is_master) {
            std::cout << "[sft] " << rows_per_step << " rows per step = B " << cfg.B << " x GRAD_ACCUM " << ga
                      << " x W " << world_size << " | " << epochs << " epochs = " << cfg.max_steps << " steps"
                      << " | lr " << cfg.max_lr << " -> " << cfg.min_lr << ", warmup " << cfg.warmup_steps
                      << " steps then " << sched << (sched == "wsd" ? " (stable " + std::to_string(cfg.stable_steps)
                      + " steps)" : std::string()) << " | val every " << cfg.val_freq << " steps ("
                      << cfg.val_loss_steps << " batches per rank)\n"
                      << "[sft] loss: label 65535 ignored, divided by the step's global trained-token count, "
                         "backward seed " << world_size << " (= W) | tok/s below counts all positions, "
                         "padding included" << std::endl;
            if (getenv_int("BLU_MAX_STEPS", 0) > 0)
                std::cout << "[sft] BLU_MAX_STEPS is ignored in SFT mode" << std::endl;
            if (!std::getenv("BLU_INIT_FROM"))
                std::cout << "[sft] WARNING: no BLU_INIT_FROM, so this run fine-tunes RANDOM weights" << std::endl;
        }
    }

    // Tokens consumed per optimizer step = B * T * grad_accum * world_size.
    // 2026-11-06: / g_cp.U -- the U ranks of a CP group share their rows (1 outside SFT context parallelism).
    // The previous line, kept for reference:
    // const int64_t tokens_per_micro = cfg.B * cfg.T * static_cast<int64_t>(world_size);
    const int64_t tokens_per_micro = cfg.B * cfg.T * static_cast<int64_t>(world_size / g_cp.U);
    if (cfg.global_batch % tokens_per_micro != 0) {
        if (is_master) {
            std::cerr << "ERROR: global_batch (" << cfg.global_batch
                      << ") not divisible by B*T*world_size (" << tokens_per_micro << ")\n";
        }
        dist::comm::abort_job(launch, 1);
    }
    const int GRAD_ACCUM = static_cast<int>(cfg.global_batch / tokens_per_micro);

    if (is_master) {
        std::cout << "================ Configuration ================\n"
                  << "  d_model         : " << cfg.d_model        << "\n"
                  << "  n_layers        : " << cfg.n_layers       << "\n"
                  << "  q_heads/kv_heads: " << cfg.q_heads << " / " << cfg.kv_heads
                  << "  (G=" << (cfg.q_heads / cfg.kv_heads) << ")\n"
                  << "  head_dim        : " << cfg.head_dim       << "\n"
                  << "  ffn_hidden      : " << cfg.ffn_hidden     << "  (SwiGLU)\n"
                  << "  context_length  : " << cfg.context_length << "\n"
                  << "  rope_theta      : " << cfg.rope_theta
                  << "  (interleaved="    << (cfg.rope_interleaved ? "GPT-J" : "NeoX") << ")\n"
                  << "  vocab_size      : " << cfg.vocab_size     << "\n"
                  << "  weight_tying    : " << (cfg.weight_tying ? "ON" : "OFF") << "\n"
                  << "  embed_scale     : " << (cfg.embed_scale ? "x sqrt(d_model)" : "off") << "\n"
                  << "  precision       : resid "
                  << OwnTensor::get_dtype_name(cfg.resid_dtype())
                  << " | logits " << OwnTensor::get_dtype_name(cfg.logit_dtype())
                  << " | embed "  << OwnTensor::get_dtype_name(cfg.embed_dtype())
                  << "\n"
                  << "                    (matmuls bf16, QK-norm+RoPE fp32, "
                     "optimizer masters fp32 — all fixed)\n"
                  << "  B x T           : " << cfg.B << " x " << cfg.T << "\n"
                  << "  world_size      : " << world_size         << "\n"
                  << "  global_batch    : " << cfg.global_batch
                  << "  (grad_accum=" << GRAD_ACCUM << ")\n"
                  << "  max_lr / min_lr : " << cfg.max_lr << " / " << cfg.min_lr << "\n"
                  << "  warmup / steps  : " << cfg.warmup_steps << " / " << cfg.max_steps << "\n"
                  << "  WSD stable_steps: " << cfg.stable_steps
                  << "  (decay tail=" << (cfg.max_steps - cfg.warmup_steps - cfg.stable_steps) << ")\n"
                  << "===============================================" << std::endl;
    }

    // The WSD schedule needs a non-empty decay tail; get_lr()'s decay_ratio
    // divides by (max_steps - warmup - stable - 1) and asserts on the range.
    if (cfg.warmup_steps + cfg.stable_steps >= cfg.max_steps - 1) {
        if (is_master) {
            std::cerr << "ERROR: warmup_steps + stable_steps ("
                      << (cfg.warmup_steps + cfg.stable_steps)
                      << ") leaves no decay tail before max_steps (" << cfg.max_steps << ")\n";
        }
        dist::comm::abort_job(launch, 1);
    }

    // ---- process group — only when multi-GPU. Used by ZeRO-1's
    // reduce-scatter/all-gather (there is no separate DDP process group). ----
    // BLU_ZERO_V2=1 selects the reduce-scatter/all-gather optimizer
    // (owt::zero::SymmetricShardedOptimizer) instead of v1's reduce/broadcast.
    // BLU_SYMMETRIC=1 additionally backs its wire buffers with NCCL symmetric
    // memory so the collectives can dispatch to the NVLS (NVLink SHARP) kernels.
    //
    // The memory mode is fixed at process-group creation, so it must be decided
    // HERE, before init_process_group.
    //
    // NOTE: symmetric memory is only useful with v2. NCCL 2.28 ships symmetric
    // device kernels for AllGather / AllReduce / ReduceScatter ONLY — there is no
    // symmetric ncclReduce and no symmetric ncclBroadcast, which are exactly the
    // two collectives v1 uses. Requesting it for v1 registers windows that
    // nothing can dispatch against, so we say so rather than let a benchmark
    // silently measure nothing.
    const std::string comm_backend = getenv_str("BLU_BACKEND", "nccl");
    if (comm_backend != "nccl" && comm_backend != "bluccl") {
        if (is_master)
            std::cerr << "ERROR: BLU_BACKEND / backend (parallelism config) must be nccl or bluccl (got '"
                      << comm_backend << "')" << std::endl;
        dist::comm::abort_job(launch, 1);
    }
    const bool use_bluccl = (comm_backend == "bluccl");
    if (is_master)
        std::cout << "[comm] backend " << comm_backend << " for the ZeRO collectives" << std::endl;
    // BluCCL has only all-reduce/reduce-scatter/all-gather: no ZeRO v1, no Muon reduce/broadcast.
    const bool use_zero_v2  = use_bluccl || getenv_int("BLU_ZERO_V2", 0) != 0;
    if (is_master && use_bluccl && getenv_int("BLU_ZERO_V2", 0) == 0)
        std::cout << "[BluCCL] zero1.variant forced to v2: ZeRO v1 needs reduce/broadcast, "
                     "which BluCCL does not implement." << std::endl;
    const bool want_symmetric = !use_bluccl && getenv_int("BLU_SYMMETRIC", 0) != 0;
    if (is_master && use_bluccl && getenv_int("BLU_SYMMETRIC", 0) != 0)
        std::cout << "[BluCCL] zero1.symmetric ignored: it selects NCCL symmetric memory; "
                     "BluCCL always uses NVLS and its own symmetric buffers." << std::endl;
    // Read here rather than at the optimizer block below: the memory mode is
    // fixed at process-group creation, so whether the Muon group can use
    // symmetric memory has to be known BEFORE init_process_group.
    const bool use_muon = [] {
        const char* v = std::getenv("BLU_OPTIMIZER");
        return v && (std::string(v) == "muon" || std::string(v) == "MUON");
    }();
    // ShardedMuon's default reduce-scatter/all-gather path dispatches to the same
    // symmetric kernels v2 does, so Muon benefits from symmetric memory
    // independently of BLU_ZERO_V2 (which only selects the AdamW remainder's
    // optimizer). Its BLU_MUON_RB=1 fallback does not — that path is
    // Reduce/Broadcast, which has no symmetric kernel.
    const bool muon_rb  = !use_bluccl && getenv_int("BLU_MUON_RB", 0) != 0;
    if (is_master && use_bluccl && getenv_int("BLU_MUON_RB", 0) != 0)
        std::cout << "[BluCCL] BLU_MUON_RB ignored: no BluCCL implementation; "
                     "Muon uses reduce-scatter/all-gather." << std::endl;
    const bool symmetric_useful = (use_zero_v2 || (use_muon && !muon_rb));
    if (is_master && want_symmetric && !symmetric_useful) {
        std::cout << "[ZeRO-1] WARNING: BLU_SYMMETRIC=1 has NO effect in this "
                     "configuration. NCCL has no symmetric kernel for ncclReduce "
                     "or ncclBroadcast (the collectives used by ZeRO v1 and by "
                     "BLU_MUON_RB=1); only AllGather/AllReduce/ReduceScatter have "
                     "one. Set BLU_ZERO_V2=1, or use Muon without BLU_MUON_RB, to "
                     "exercise NVLS." << std::endl;
    }

    // Bootstrap route follows the launcher: MPI_Bcast of the ncclUniqueId under
    // mpirun, the pre-minted NCCL_UNIQUE_ID_HEX under BluRun. Either way the
    // resolved local rank is handed in rather than re-derived, so the
    // communicator and the model land on the same device. nullptr at
    // world_size == 1.
    std::shared_ptr<ProcessGroupNCCL> nccl_pg;
    std::shared_ptr<dist::comm::ProcessGroupBase> pg;
    if (!use_bluccl) {
        nccl_pg = dist::comm::bring_up_process_group(launch,
                                                     (want_symmetric && symmetric_useful)
                                                         ? dist::comm::MemoryMode::Symmetric
                                                         : dist::comm::MemoryMode::Standard);
        pg = nccl_pg;
    } else if (bluccl_multi && world_size > 1) {
        // Across nodes: ZeRO's reduce-scatter and all-gather on BluCCL (internode RS / AG V2, built on NCCL's comm
        // stream so one stream orders both), all-reduce, broadcast and checkpoints on NCCL.
        nccl_pg = dist::comm::bring_up_process_group(launch, dist::comm::MemoryMode::Standard);
        const int ranks_per_node = world_size / launch_nodes;
        if (world_size % launch_nodes != 0) {
            if (is_master)
                std::cerr << "[BluCCL] FATAL: world " << world_size << " is not a multiple of " << launch_nodes
                          << " nodes" << std::endl;
            dist::comm::abort_job(launch, 1);
        }
        std::shared_ptr<dist::comm::ProcessGroupComposite> zpg;
        try {
            auto blu = std::make_shared<dist::comm::ProcessGroupBluCCL>(ranks_per_node, launch.local_rank,
                                                                        nccl_pg->get_stream());
            zpg = std::make_shared<dist::comm::ProcessGroupComposite>(blu, nccl_pg);
        } catch (const std::exception& e) {
            std::cerr << "[rank " << rank << "] [BluCCL] FATAL: could not create the multi-node BluCCL group: "
                      << e.what() << std::endl;
            dist::comm::abort_job(launch, 1);
        }
        pg = zpg;
        if (is_master)
            std::cout << "[BluCCL] multi-node group: " << launch_nodes << " node(s) x " << ranks_per_node
                      << "; reduce-scatter = " << (zpg->rs_via_nccl() ? "NCCL [BLU_COMPOSITE_RS=nccl]" : "BluCCL")
                      << ", all-gather = " << (zpg->ag_via_nccl() ? "NCCL [BLU_COMPOSITE_AG=nccl]" : "BluCCL")
                      << ", all-reduce/broadcast/checkpoint = NCCL" << std::endl;
    } else if (world_size > 1) {
        try {
            pg = init_process_group(world_size, rank, dist::comm::Backend::BluCCL, /*stream=*/0);
        } catch (const std::exception& e) {
            if (is_master)
                std::cerr << "[BluCCL] FATAL: could not create the BluCCL process group: "
                          << e.what() << "\n         BluCCL runs single-node only (any GPUs: NVLS, "
                             "P2P or host kernels). Use backend: nccl for multi-node."
                          << std::endl;
            dist::comm::abort_job(launch, 1);
        }
        if (is_master)
            std::cout << "[BluCCL] process group up: world=" << world_size
                      << " (algorithms picked for this machine; BLUCCL_ALGO_REPORT=1 prints them)" << std::endl;
    }

    // 2026-11-06: the context-parallel group (BLU_SFT_CP > 1; see SftCp / cp_make_group). It is split off the world
    // NCCL group; the single-node BluCCL backend has none, so one is brought up for the split (and kept alive).
    if (g_cp.U > 1) {
        g_cp.parent = nccl_pg ? nccl_pg : dist::comm::bring_up_process_group(launch, dist::comm::MemoryMode::Standard);
        try {
            g_cp.pg = cp_make_group(*g_cp.parent, rank, g_cp.U);
        } catch (const std::exception& e) {
            std::cerr << "[rank " << rank << "] [sft] FATAL: " << e.what() << std::endl;
            dist::comm::abort_job(launch, 1);
        }
        if (is_master)
            std::cout << "[sft] CP groups up: " << world_size / g_cp.U << " group(s) of " << g_cp.U
                      << " consecutive ranks (Ulysses all-to-all on a blocking stream)" << std::endl;
    }

    // Monitoring: every launch writes into its own <launch dir>/<logs_root>/<run name>_<mpirun|blurun>_<id>/
    // (see train/production_zero/monitoring/); the make wrapper pre-creates it for mpirun launches.
    const bool monitor = getenv_int("BLU_MONITOR", 0) != 0;
    const std::string launch_dir = getenv_str("BLU_LAUNCH_DIR", fs::current_path().string());
    std::string run_dir = monitor ? getenv_str("BLU_RUN_DIR", "") : "";
    if (monitor && run_dir.empty()) {
        const std::string name = getenv_str("BLU_RUN_NAME", "");
        const bool name_ok = !name.empty() && name.size() <= 64 &&
            std::all_of(name.begin(), name.end(), [](char c) { return std::isalnum((unsigned char)c) || c == '_' || c == '-'; });
        if (!name_ok) {
            if (is_master)
                std::cerr << "ERROR: monitoring is on, so a run name is required: pass RUN_NAME=<name> to make or set "
                             "run.name in the config (letters, digits, '_' or '-', at most 64; got '" << name << "')."
                          << std::endl;
            dist::comm::abort_job(launch, 2);
        }
        if (is_master) {
            const fs::path base = fs::absolute(fs::path(launch_dir) / getenv_str("BLU_LOGS_ROOT", "training_logs"));
            fs::create_directories(base);
            const std::string job_id = getenv_str("ORCHESTRATOR_JOB_ID", "");
            if (!job_id.empty()) {
                run_dir = (base / (name + "_blurun_" + job_id)).string();
                fs::create_directories(run_dir);
            } else {
                const std::string prefix = name + "_mpirun_";
                int n = 0;
                for (const auto& e : fs::directory_iterator(base)) {
                    const std::string f = e.path().filename().string();
                    if (f.rfind(prefix, 0) == 0 && f.size() > prefix.size() &&
                        std::all_of(f.begin() + prefix.size(), f.end(), ::isdigit))
                        n = std::max(n, std::stoi(f.substr(prefix.size())));
                }
                char id[16];
                do std::snprintf(id, sizeof(id), "%03d", ++n);
                while (!fs::create_directory(base / (prefix + id)));
                run_dir = (base / (prefix + id)).string();
            }
        }
    }
    if (is_master && !run_dir.empty())
        std::cout << "[monitor] run folder: " << run_dir << "  (live charts: run `make dashboard` in "
                  << launch_dir << ")" << std::endl;
    // BLU_COMM_PROFILE: per-collective CSVs (tools/rs_autotune.py, tools/rs_skew_report.py) go to <run folder>/commprof
    // unless BLU_COMM_PROFILE_DIR says otherwise; each node writes its own ranks' files.
    if (getenv_int("BLU_COMM_PROFILE", 0) != 0 && !std::getenv("BLU_COMM_PROFILE_DIR") && !run_dir.empty()) {
        const fs::path pdir = fs::path(run_dir) / "commprof";
        std::error_code ec;
        fs::create_directories(pdir, ec);
        ::setenv("BLU_COMM_PROFILE_DIR", pdir.string().c_str(), 1);
    }

    // An existing CSV is appended to on resume, replaced only with BLU_LOG_OVERWRITE=1, else refused.
    const bool csv_enabled = getenv_int("BLU_LOG_ENABLED", 1) != 0;
    const std::string csv_dir = run_dir.empty() ? getenv_str("BLU_LOG_DIR", "") : run_dir;
    const std::string csv_path = (csv_dir.empty() ? fs::path() : fs::path(csv_dir)) /
        (run_dir.empty() ? getenv_str("BLU_LOG_CSV", sft ? (use_muon ? "sft_muon.csv" : "sft_adamw.csv")
                                                         : (use_muon ? "mix_ratio_1.csv" : "adamw_noovl_contig.csv"))
                         : std::string("metrics.csv"));
    int csv_mode = 0;   // 0 = write new, 1 = append, -1 = refuse
    if (is_master && csv_enabled && fs::exists(csv_path) && getenv_int("BLU_LOG_OVERWRITE", 0) == 0)
        csv_mode = (!run_dir.empty() || getenv_int("BLU_LOG_APPEND", 0) != 0 ||
                    getenv_int("BLU_CKPT_LOAD", 0) != 0) ? 1 : -1;
    csv_mode = static_cast<int>(dist::comm::allreduce_scalar(launch, pg, static_cast<double>(csv_mode), ::sum));
    if (csv_mode < 0) {
        if (is_master)
            std::cerr << "ERROR: " << fs::absolute(csv_path).string() << " already exists. "
                      << "Move it away, set BLU_LOG_OVERWRITE=1 to replace it, "
                      << "or logs.append: true / checkpointing.load: true to append to it." << std::endl;
        dist::comm::abort_job(launch, 5);
    }

    if (is_master) std::cout << "Initializing model on CUDA device " << local_rank << "..." << std::endl;
    BLU model(cfg, device, /*seed=*/static_cast<uint64_t>(cfg.seed) /*same seed all ranks*/);

    // Parameter accounting.
    auto params = model.parameters();
    int64_t num_params = 0;
    for (auto& p : params) num_params += p.numel();
    if (is_master) std::cout << "Number of parameters: " << num_params << std::endl;

    // ---- SFT: initialise the weights from another run's checkpoint (weights only) ----
    // What:   BLU_INIT_FROM=<checkpoint dir> loads ONLY the model weights of that DCP checkpoint
    //         (e.g. blumodel_step_8000 of a 4-GPU Muon pretraining run) into this model. Its
    //         optimizer state, step, epoch and data position are NOT loaded: this run starts at
    //         step 0 with a fresh optimizer. If this run later resumes its OWN checkpoint
    //         (cfg.ckpt_dir, below), that load overwrites these weights together with the
    //         optimizer state, so a restarted SFT run continues where it stopped. Same contract
    //         as torchtitan's initial_load_path + initial_load_model_only=True.
    // Why here, before the optimizers: every ZeRO optimizer (ShardedOptimizer,
    //         SymmetricShardedOptimizer, ShardedMuon) copies the model weights into its fp32
    //         masters inside its constructor (init_master_from_params). Loaded after that, the
    //         masters would still hold the random init and the first step would write it back.
    // Needs:  weights are saved by position (param_NNNNNN), so the checkpoint must hold this
    //         model's weights with the same shape and dtype at every index. The dtypes follow the
    //         precision flags: a checkpoint saved with BLU_ALL_BF16=1 (bf16 embedding and norms)
    //         must be loaded with the same flag. A mismatch is not silent: DCP stops on a missing
    //         weight or a different shape ("[Architecture Mismatch]", LoadPlanning.cpp), and
    //         sync_back_to_model() stops on a different dtype (Tensor::copy_ refuses it).
    // Verify: BLU_INIT_VERIFY=1 also compares every loaded weight in GPU memory with the CRC32
    //         the checkpoint stored for it (a test switch; it copies every weight to the host).
    std::shared_ptr<ProcessGroupNCCL> init_pg;   // the NCCL group "dcp" made here when the trainer has none
                                                 // (W == 1, single-node BluCCL); the checkpointer reuses it
    if (const char* init_from = std::getenv("BLU_INIT_FROM"); init_from && *init_from) {
        const std::string init_dir = init_from;
        auto pname = [](size_t i) {
            char nm[32];
            std::snprintf(nm, sizeof nm, "param_%06zu", i);
            return std::string(nm);
        };
        // DCP loads over an NCCL group (it has no BluCCL backend), as the checkpointer below does:
        //   NCCL backend          -> the trainer's own group (nccl_pg);
        //   BluCCL, W > 1         -> a separate NCCL group "dcp" (collective: every rank gets here);
        //   W == 1                -> a single-rank group "dcp".
        // The last two are kept in init_pg and reused by the checkpointer, so "dcp" is created only once.
        std::shared_ptr<ProcessGroupNCCL> load_pg = nccl_pg;
        if (!load_pg) {
            init_pg = world_size > 1
                ? dist::comm::bring_up_process_group(launch, dist::comm::MemoryMode::Standard, "dcp")
                : init_process_group(1, 0, /*stream=*/0, /*group_name=*/"dcp", dist::comm::MemoryMode::Standard);
            load_pg = init_pg;
        }
        std::map<std::string, Tensor*> init_named;
        for (size_t i = 0; i < params.size(); ++i) init_named[pname(i)] = &params[i];
        owt::zero::ZeroDcpModule init_module(init_named);
        int   ck_epoch = 0, ck_step = 0;
        float ck_loss  = 0.0f;
        OwnTensor::DistCheckpoint::load(init_module, init_dir, load_pg, /*optimizer=*/nullptr,
                                        &ck_epoch, &ck_step, &ck_loss);
        init_module.sync_back_to_model();   // a no-op when the dtypes match

        if (getenv_int("BLU_INIT_VERIFY", 0) != 0) {
            OwnTensor::CheckpointMetadata init_meta;
            {
                std::ifstream mis(init_dir + "/metadata.bin", std::ios::binary);
                init_meta = OwnTensor::CheckpointMetadata::deserialize(mis);
            }
            size_t n_crc_ok = 0;
            std::vector<std::string> crc_bad;
            for (size_t i = 0; i < params.size(); ++i) {
                const auto& tm = init_meta.tensors.at(pname(i));
                Tensor host = params[i].to_cpu();
                if (tm.shards.size() == 1 && tm.shards[0].checksum_kind != 0 &&
                    OwnTensor::dcp::Checksum::verify(
                        static_cast<OwnTensor::dcp::Checksum::Kind>(tm.shards[0].checksum_kind),
                        host.data(), host.nbytes(), tm.shards[0].checksum_value))
                    ++n_crc_ok;
                else
                    crc_bad.push_back(pname(i));
            }
            if (is_master)
                std::cout << "[init] BLU_INIT_VERIFY: " << n_crc_ok << "/" << params.size()
                          << " weights match the checkpoint's CRC32"
                          << (crc_bad.empty() ? "" : "; first mismatch " + crc_bad[0]) << std::endl;
            if (!crc_bad.empty())
                throw std::runtime_error("BLU_INIT_FROM: loaded weights do not match the checkpoint");
        }
        if (is_master)
            std::cout << "[init] weights only from " << init_dir << " (its step " << ck_step << ", loss "
                      << std::fixed << std::setprecision(6) << ck_loss << "). Optimizer state, step "
                         "and data position start fresh." << std::endl;
    }

    // ---- SFT: initialise the weights from a converted Hugging Face export (BLU_INIT_FROM_RAW) ----
    // What:   BLU_INIT_FROM_RAW=<dir> written by sft/hf_to_blutrain.py: one fp32 file per weight (<name>.bin, BluTrain
    //         layout, the names of preproduction-135M's export_safetensors) + index.txt ("<name> <dims...>"). Weights
    //         only, like BLU_INIT_FROM: fresh optimizer, step 0. Made for the 135M base, which we have as Navin's
    //         Hugging Face file (2026-10-27), not as a complete DCP checkpoint.
    // Check:  every weight is read back from the GPU and must equal the file bit for bit (the files hold bf16-exact
    //         values, so the bf16 parameters store them exactly); the run stops otherwise. Every BluTrain weight must
    //         be in the folder and every file must belong to a weight: no silent partial load.
    if (const char* raw_dir_c = std::getenv("BLU_INIT_FROM_RAW"); raw_dir_c && *raw_dir_c) {
        const std::string raw_dir = raw_dir_c;
        if (std::getenv("BLU_INIT_FROM") && *std::getenv("BLU_INIT_FROM"))
            throw std::runtime_error("set BLU_INIT_FROM or BLU_INIT_FROM_RAW, not both");
        if (!cfg.weight_tying)
            throw std::runtime_error("BLU_INIT_FROM_RAW: the converter writes a tied model (no lm_head file)");
        std::vector<std::pair<std::string, Tensor*>> named;
        named.push_back({"wte.weight", &model.wte.weight});
        for (int i = 0; i < cfg.n_layers; ++i) {
            const std::string L = "layers." + std::to_string(i) + ".";
            named.push_back({L + "attn_norm.weight", &model.attn[i]->norm.weight});
            named.push_back({L + "wq.weight",        &model.attn[i]->wQ.weight});
            named.push_back({L + "wk.weight",        &model.attn[i]->wK.weight});
            named.push_back({L + "wv.weight",        &model.attn[i]->wV.weight});
            named.push_back({L + "wo.weight",        &model.attn[i]->wO.weight});
            named.push_back({L + "q_gamma",          &model.attn[i]->q_gamma});
            named.push_back({L + "k_gamma",          &model.attn[i]->k_gamma});
            named.push_back({L + "mlp_norm.weight",  &model.mlp[i]->norm.weight});
            named.push_back({L + "gate_up.weight",   &model.mlp[i]->gate_up.weight});
            named.push_back({L + "down.weight",      &model.mlp[i]->down.weight});
        }
        named.push_back({"norm_f.weight", &model.norm_f.weight});

        std::map<std::string, std::vector<int64_t>> file_dims;
        {
            std::ifstream idx(raw_dir + "/index.txt");
            if (!idx) throw std::runtime_error("BLU_INIT_FROM_RAW: no " + raw_dir + "/index.txt");
            std::string line;
            while (std::getline(idx, line)) {
                std::istringstream ls(line);
                std::string nm; int64_t d; std::vector<int64_t> dims;
                if (!(ls >> nm)) continue;
                while (ls >> d) dims.push_back(d);
                file_dims[nm] = dims;
            }
        }
        if (file_dims.size() != named.size())
            throw std::runtime_error("BLU_INIT_FROM_RAW: index.txt lists " + std::to_string(file_dims.size()) +
                                     " weights, this model has " + std::to_string(named.size()));
        size_t n_exact = 0;
        for (auto& [nm, p] : named) {
            auto it = file_dims.find(nm);
            if (it == file_dims.end()) throw std::runtime_error("BLU_INIT_FROM_RAW: no file for " + nm);
            if (it->second != p->shape().dims)
                throw std::runtime_error("BLU_INIT_FROM_RAW: " + nm + " has another shape in the file "
                                         "(wrong BLU_D_MODEL / heads / ffn / vocab / layers?)");
            Tensor host(p->shape(), TensorOptions().with_dtype(Dtype::Float32));
            const size_t bytes = static_cast<size_t>(host.numel()) * sizeof(float);
            std::ifstream f(raw_dir + "/" + nm + ".bin", std::ios::binary | std::ios::ate);
            if (!f || static_cast<size_t>(f.tellg()) != bytes)
                throw std::runtime_error("BLU_INIT_FROM_RAW: " + nm + ".bin is missing or not " +
                                         std::to_string(bytes) + " bytes");
            f.seekg(0);
            f.read(reinterpret_cast<char*>(host.data<float>()), static_cast<std::streamsize>(bytes));
            copy_into_param(*p, host.as_type(p->dtype()).to(p->device()));
            Tensor back = p->as_type(Dtype::Float32).to_cpu();          // the exact check
            if (std::memcmp(back.data<float>(), host.data<float>(), bytes) != 0)
                throw std::runtime_error("BLU_INIT_FROM_RAW: " + nm + " on the GPU differs from its file");
            ++n_exact;
        }
        if (is_master)
            std::cout << "[init] weights only from " << raw_dir << " (BLU_INIT_FROM_RAW): " << n_exact << "/"
                      << named.size() << " weights loaded and bit-equal to their files. Optimizer state, step and "
                         "data position start fresh." << std::endl;
    }

    // ---- MFU meter ----
    // Built here rather than at first use so its banner lands next to the
    // parameter count it cross-checks itself against (6*N), and so a bogus
    // BLU_PEAK_TFLOPS or an unrecognized GPU is reported BEFORE the run instead
    // of after 2000 steps of unusable numbers. `device_ordinal=local_rank`
    // because that is what this process bound above — a device ordinal is
    // node-local, so the global rank is the wrong thing to hand it on any job
    // spanning more than one node.
    MFUMeter mfu_meter = MFUMeter::build(cfg, world_size, local_rank);
    if (is_master) mfu_meter.print_banner(num_params, cfg.global_batch);

    const bool bluccl_fused_emb = getenv_int("BLU_BLUCCL_FUSED_EMB", 0) != 0;
    if (bluccl_fused_emb && use_bluccl && world_size > 1 && !user_set_fused_emb)
        ::setenv("BLUCCL_FUSED_EMB", "1", 1);
    else if (bluccl_fused_emb && is_master)
        std::cout << "[embedding] bluccl.fused_embedding_backward ignored: it needs ZeRO-1 on the "
                     "BluCCL backend with more than one GPU; using embedding.backward" << std::endl;
    // 2026-11-06: B * T / g_cp.U -- the segmented kernel's workspace is sized for EXACTLY this many tokens per call
    // and its hook ignores the real count, so under context parallelism it must be this rank's slice. Previous call:
    // const bool emb_segmented = dist::comm::install_fused_embedding_backward(
    //     cfg.B * cfg.T, cfg.d_model, cfg.vocab_size, /*padding_idx=*/-1, is_master);
    const bool emb_segmented = dist::comm::install_fused_embedding_backward(
        cfg.B * (cfg.T / g_cp.U), cfg.d_model, cfg.vocab_size, /*padding_idx=*/-1, is_master);
    if (is_master) {
        const char* m = std::getenv("BLUTRAIN_EMB_BWD");
        std::cout << "[embedding] backward: "
                  << (emb_segmented && bluccl_fused_emb && use_bluccl
                          ? "BluCCL fused embedding backward (bluccl.fused_embedding_backward; "
                            "sorted segments, fp32 sums, deterministic)"
                      : emb_segmented ? "segmented (sorted segments, fp32 sums, deterministic)"
                      : (m && std::string(m) == "cooperative") ? "cooperative (atomic-free, fp32 accumulation)"
                      : (m && std::string(m) == "atomic") ? "atomic (atomicAdd in the gradient dtype)"
                      : "built-in default (atomicAdd for bf16, cooperative for fp32)")
                  << std::endl;
    }
    if (is_master && !use_bluccl) {
        for (const char* f : {"BLU_FUSED_ADAMW", "BLU_FUSED_CLIP_ADAMW", "BLU_FUSED_MUON_PUBLISH"})
            if (getenv_int(f, 0) != 0)
                std::cout << "[comm] " << f << "=1 needs backend: bluccl; the optimizer will "
                             "decline it and use the unfused path." << std::endl;
    }
    if (is_master && getenv_int("BLU_FUSED_CLIP_ADAMW", 0) != 0 && getenv_int("BLU_FUSED_ADAMW", 0) == 0)
        std::cout << "[BluCCL] WARNING: bluccl.fused_clip_adamw has no effect without "
                     "bluccl.fused_adamw_allgather." << std::endl;

    // ---- optimizer: ZeRO-1 (optimizer-state sharding), mixed precision.
    // No DDP wrapper: ZeRO-1's reduce-scatter *replaces* DDP's gradient
    // all-reduce (running both would double-reduce the gradients), so
    // model.forward() is called directly below with no DataParallel wrapping.
    //
    // ShardedOptimizer requires every parameter it manages to share one dtype
    // (the fused Adam kernel + the reduce-scatter/all-gather wire format are
    // single-dtype). This model is deliberately mixed: the Linear projections
    // are BF16 (the matmuls run in bf16), while the token embedding, the
    // RMSNorm scales and the QK-norm scales (q_gamma/k_gamma) stay FP32 for
    // numerical stability. Group params by dtype and give each group its own
    // ShardedOptimizer instance; in practice that's exactly two groups.
    // Calling their step()s back-to-back (never concurrently) is safe: each
    // instance's local_adam_step() stream-syncs after every fused-Adam launch,
    // so the multi-tensor kernel's shared launch-metadata buffer is never
    // clobbered by the next instance's launch.
    //
    // BLU_OPTIMIZER=muon additionally splits off the Muon set BEFORE the dtype
    // grouping — see the ShardedMuon block below for why that set needs a
    // different sharding scheme entirely. (use_muon is resolved up at the
    // process-group block: the memory mode depends on it and is fixed there.)

    // ---- which parameters Muon may orthogonalize -----------------------------
    // The recipe is: Muon for the transformer body's 2-D weight matrices, AdamW
    // for everything else. "Everything else" is every 1-D parameter (the RMSNorm
    // scales and the QK-norm q_gamma/k_gamma — a vector has no second axis to
    // orthogonalize against) plus the two vocabulary matrices, the token
    // embedding and the output head. Those two are excluded because their rows
    // are near-independent per-token directions rather than a map between two
    // dense feature spaces, which is the structure that makes spectral
    // normalization the right prior in the first place.
    //
    // Under weight tying lm_head IS wte, so the single exclusion covers both.
    std::set<const void*> muon_excluded;
    for (auto& p : model.wte.parameters())          muon_excluded.insert(p.data());
    if (model.lm_head)
        for (auto& p : model.lm_head->parameters()) muon_excluded.insert(p.data());
    auto is_muon_param = [&](const Tensor& p) {
        return use_muon && p.ndim() >= 2 && !muon_excluded.count(p.data());
    };

    // Muon-eligible params grouped by dtype, and the AdamW remainder grouped by
    // dtype. Both optimizers are single-dtype (their fused kernels and wire
    // formats are), so the dtype split happens inside each. At the default
    // precision the partition is exactly: {bf16 body matrices} -> Muon,
    // {fp32 wte + all the 1-D scales} -> AdamW.
    //
    // The precision flags move parameters BETWEEN these dtype maps, they do not
    // change Muon eligibility (muon_excluded is keyed on pointer identity, so
    // wte stays out of Muon whatever dtype it is). Under BLU_ALL_BF16=1 the
    // AdamW side splits into a bf16 group (wte + the block RMSNorm scales) and a
    // residual fp32 group holding only q_gamma/k_gamma — 2*head_dim*n_layers =
    // 6144 elements, which the fused QK-norm+RoPE kernel pins to fp32. So "all
    // bf16" still leaves two AdamW dtype groups, and the one-group /
    // one-all-gather / one-clip-norm simplification does NOT materialize.
    std::map<Dtype, std::vector<Tensor>> muon_by_dtype;
    std::map<Dtype, std::vector<Tensor>> params_by_dtype;
    for (auto& p : params) {
        if (is_muon_param(p)) muon_by_dtype[p.dtype()].push_back(p);
        else                  params_by_dtype[p.dtype()].push_back(p);
    }

    // Keeping the embedding group on ELEMENTWISE ZeRO-1 is deliberate, not an
    // oversight. ShardedMuon owns whole tensors, and a group whose mass is one
    // 103M-element embedding plus ~145 tensors of a few thousand elements cannot
    // be balanced that way: one rank would hold 99.8% of the state. Elementwise
    // sharding splits that embedding evenly, and AdamW — being elementwise — is
    // free to be sharded that way. Muon is not. That asymmetry is the whole
    // reason the two sets take different optimizers.

    ZeroConfig zcfg;
    zcfg.variant                      = AdamVariant::ADAMW;
    zcfg.lr                           = cfg.max_lr;
    zcfg.beta1                        = cfg.beta1;
    zcfg.beta2                        = cfg.beta2;
    zcfg.eps                          = cfg.adam_eps;
    zcfg.weight_decay                 = cfg.weight_decay;
    zcfg.exclude_1d_from_weight_decay = true;   // GPT/Llama recipe: no wd on 1-D params
    zcfg.max_grad_norm                = cfg.grad_clip;
    // All ranks init weights from the same mt19937 seed (see GPT ctor, seed=1234
    // on every rank) via identical CPU draws, so the model is already replica-
    // consistent before the optimizer is built — the init broadcast would only
    // copy data that's already equal. Disable it (verify_consistency stays on as
    // the cheap startup layout check).
    zcfg.init_broadcast               = false;
    // Overlap the gradient reduce with the final micro-step's backward: grads are
    // bucketed and reduced-to-owner from autograd hooks as each bucket fills,
    // instead of one serial reduce-scatter after backward. Requires arming the
    // optimizer before the last backward (see arm_grad_sync below) and the
    // SEQUENTIAL autograd engine (the default). No-op at world_size==1.
    //
    // BLU_NO_OVERLAP=1 forces BOTH overlap features off, dropping the optimizer
    // onto its reference path: one monolithic gather + reduce_scatter after
    // backward, then a blocking all-gather + full-model scatter at the end of
    // step(). Slower, but it is the path the ZeRO-1 design is validated against
    // (bit-identical to single-process AdamW at W=1), so it is the bisection that
    // separates "the overlap machinery corrupts gradients or publishes bad
    // weights" from "the Adam/gradient math is wrong independently of overlap".
    // BISECTION (docs/ZERO1_OVERLAP_AUDIT.md). BLU_NO_OVERLAP conflated two
    // independent features, so a bad loss curve could not be attributed to either.
    // They are now separately switchable, and the STAGED REFRESH IS OFF BY DEFAULT:
    // this build keeps the hook-driven gradient reduce and publishes parameters
    // synchronously.
    //
    //   BLU_NO_OVERLAP=1        both off — the validated reference path
    //   BLU_NO_OVERLAP_BWD=1    suppress (A) only
    //   BLU_OVERLAP_REFRESH=1   restore (B) — the previous default
    //
    // Why (B) is the one dropped first: (A) is the untested half. Every test in
    // dist/zero/tests sets overlap_backward_comm=false and none of them calls
    // arm_grad_sync(), whereas the staged refresh has direct coverage in
    // test_zero_v2_staged.cpp and test_muon_rsag.cpp. Isolating the covered half
    // first is backwards; this leaves the UNCOVERED half running so a surviving
    // bad curve indicts it directly.
    // ---- BLU_GRAD_VIEW=1: gradients become views into the wire buffer --------
    //
    // WHAT IT REMOVES. Every backward hook currently copies a parameter's
    // gradient into the optimizer's flat reduce-scatter buffer. On this model's
    // Muon group that copy reads ~14.5 GB of fp32 gradients and writes ~7.25 GB
    // of bf16 EVERY step, on the compute stream, inside the last backward.
    //
    // But that copy is NOT where the cost turned out to be, and the measurement
    // is worth recording because it is counter-intuitive. AutogradMeta::
    // accumulate_grad stores a low-precision parameter's gradient UPCAST to
    // fp32, so with a bf16 wire buffer the dtypes disagree and every micro-step
    // after the first runs `*grad_ += update.as_type(Float32)` -- which
    // allocates and writes a full fp32 temporary. At grad_accum=8 that is seven
    // allocate-convert-add passes over the whole gradient set, per step. With a
    // view the slot is already the wire dtype, the dtypes match, and it collapses
    // to an in-place add.
    //
    // MEASURED on 8xH100, 3.93 B params, B=2 (grad_accum=8), 12 steps:
    //   step time   3515.6 -> 3399.8 ms   (-116 ms, -3.3%; run-to-run noise +-2 ms)
    //   peak memory 78444  -> 62828 MB/GPU (-15.6 GB, reproduces exactly)
    //
    // WHY IT IS OFF BY DEFAULT: the view carries the WIRE dtype, so on a bf16
    // group micro-step accumulation moves from fp32 to bf16 -- seven bf16 adds
    // instead of seven fp32 ones here. That is a numerical change, not a
    // scheduling one, and a 12-step loss comparison CANNOT see it (identical
    // reruns of this config differ by 0.017 at step 11). It needs a few-hundred-
    // step A/B against BLU_GRAD_VIEW=0 before it is trusted in production.
    //
    // MUTUALLY EXCLUSIVE with BLU_OVERLAP_REFRESH=1: flat_grad_ aliases
    // flat_stage_, and the staged refresh is still reading published parameters
    // out of flat_stage_ during the next forward, exactly when gradients would be
    // accumulating into the same bytes. Each optimizer checks this itself and
    // prints a DECLINED line rather than half-applying.
    //
    // BLU_FP32_REDUCE=1 keeps fp32 accumulation (the view is then fp32) at the
    // cost of doubling the wire bytes of every reduce-scatter.
    zcfg.grad_bucket_view = getenv_int("BLU_GRAD_VIEW", 0) != 0;
    if (is_master && zcfg.grad_bucket_view) {
        std::cout << "[ZeRO-1] BLU_GRAD_VIEW=1 requested: gradients accumulate "
                     "directly into the reduce-scatter buffer. Each group reports "
                     "below whether it engaged. NOTE this changes micro-step "
                     "gradient accumulation to the wire dtype -- A/B the loss curve."
                  << std::endl;
    }

    const bool no_overlap = getenv_int("BLU_NO_OVERLAP", 0) != 0;
    zcfg.overlap_backward_comm        = (world_size > 1) && !no_overlap
                                        && getenv_int("BLU_NO_OVERLAP_BWD", 0) == 0;
    // (B) Overlap the post-Adam parameter refresh with the NEXT forward pass:
    // instead of one blocking all-gather + full-model copy at the end of step(),
    // the updated parameters are broadcast in consumption-ordered stages and
    // installed just before the layers that read them (set_param_refresh_stages
    // below). OFF unless explicitly requested — see the bisection note above.
    //
    // CAVEAT for reading the result: turning this off does NOT merely remove the
    // staging. The synchronous publish it falls back to ends in a full
    // cudaDeviceSynchronize() of its own (ShardedMuon::all_gather_params_sync,
    // SymmetricShardedOptimizer::all_gather_params), so the host/device coupling
    // that the overlapped path removes comes back with it. A curve that recovers
    // here therefore means "(B) OR a race elsewhere that the sync masks", not
    // "(B)" alone. The line-1916 sync below becomes redundant, not load-bearing.
    zcfg.overlap_param_refresh        = (world_size > 1) && !no_overlap
                                        && getenv_int("BLU_OVERLAP_REFRESH", 0) != 0;
    // BLU_COMM_BUCKET_MB sets the target gradient-bucket size for the AdamW groups
    // (ZeroConfig::comm_bucket_bytes; Muon uses BLU_MUON_BUCKETS instead, which is a
    // bucket COUNT rather than a byte budget).
    //
    // This is the knob that trades overlap granularity against SYNCHRONISATION COUNT,
    // and the nsys trace says the second term dominates here. A bucket's
    // reduce-scatter is launched as soon as THIS rank has staged the bucket's
    // gradients, but the NCCL device kernel cannot retire until every rank has
    // launched its matching collective — so it sits on the GPU spinning, holding SMs,
    // until the slowest peer arrives. Measured on 8xH100 with the default 50 MB
    // (62 buckets on the 1.62 B bf16 group): the ReduceScatter kernels occupy 20x
    // more GPU time with overlap on than off, for identical bytes, and 41% of that
    // is exposed rather than hidden behind compute. The AllGather, whose launch
    // order is rank-independent, is unaffected (1.08x) — which is what identifies
    // rank skew rather than bandwidth as the cost.
    //
    // Larger buckets  -> fewer peer-arrival waits per step (200 MB gives ~16 buckets
    //                    instead of 62), but a bucket cannot fire until ALL of its
    //                    parameters have landed, so the overlap window shrinks.
    // Smaller buckets -> finer overlap, more spin points.
    //
    // 0 or unset keeps ZeroConfig's 50 MB default.
    if (const int mb = getenv_int("BLU_COMM_BUCKET_MB", 0); mb > 0) {
        zcfg.comm_bucket_bytes = static_cast<size_t>(mb) * 1024ull * 1024ull;
        if (is_master) {
            std::cout << "[ZeRO-1] BLU_COMM_BUCKET_MB=" << mb
                      << ": gradient bucket target " << mb << " MB (default 50)" << std::endl;
        }
    }
    // BLU_FP32_REDUCE=1 reduces gradients in FP32 instead of the model's compute
    // dtype. The bf16 group's gradients are otherwise reduce-scattered in bf16 on
    // BOTH the overlap and the reference path, so disabling overlap did not test
    // it. bf16 carries 8 mantissa bits and its rounding error scales with the
    // largest partial sum rather than the result, so a near-cancelling reduction
    // can emerge with a plausible norm and a badly wrong direction.
    //
    // BLU_BF16_REDUCE=1 is the mirror knob, for A/B testing the other direction:
    // it forces BF16 on the wire even for a group whose parameters are FP32 —
    // on this model that is the AdamW group (wte + all 1-D scales), which
    // SAME_AS_PARAM otherwise reduces in FP32 no matter how BLU_FP32_REDUCE is
    // set. It exists so "both optimizers reduce in the same precision" is
    // reachable from either end, not because it is a good idea:
    //
    //   - It does NOT save memory. flat_grad_ stops aliasing flat_stage_ and
    //     becomes its own [padded] BF16 buffer, so on this model's FP32 group it
    //     costs +147.6 MiB/GPU net (+196.8 flat_grad_, -49.2 shard_grad_) — the
    //     v2 banner prints the exact figure at startup. For scale, the same alias
    //     break in the other direction (BLU_FP32_REDUCE on the BF16 Muon group)
    //     costs +7020 MiB/GPU, which is what the nsys buffer fingerprints show.
    //   - It saves almost no time. That group's whole reduce is ~1.2 ms of a
    //     ~3.7 s step, and its buckets already sit near the latency floor
    //     (137 us min vs 154 us mean), so halving the bytes does not halve the
    //     time. Measured headroom is ~0.3 ms/step.
    //   - It costs precision exactly where the model chose not to spend it. wte
    //     is one of the two tensors deliberately kept in FP32, and under weight
    //     tying its gradient carries the lm_head projection as well; the RMSNorm
    //     scale gradients are accumulated over B*T in FP32 by reduce_dgamma
    //     before being handed over. Both would be truncated to 8 mantissa bits
    //     on the wire, and op=SUM means that happens at 4x the average's
    //     magnitude.
    //
    // Keep it for bisection, not for production. The two are mutually exclusive
    // because they name the same field.
    const bool want_fp32_reduce = getenv_int("BLU_FP32_REDUCE", 0) != 0;
    const bool want_bf16_reduce = getenv_int("BLU_BF16_REDUCE", 0) != 0;
    if (want_fp32_reduce && want_bf16_reduce) {
        if (is_master)
            std::cerr << "[ZeRO-1] BLU_FP32_REDUCE=1 and BLU_BF16_REDUCE=1 both set; "
                         "they set the same field to opposite values." << std::endl;
        dist::comm::abort_job(launch, 1);
    }
    if (want_fp32_reduce) {
        zcfg.reduce_dtype = owt::zero::ReduceDtype::FP32;
        if (is_master)
            std::cout << "[ZeRO-1] BLU_FP32_REDUCE=1: reducing gradients in FP32" << std::endl;
    } else if (want_bf16_reduce) {
        zcfg.reduce_dtype = owt::zero::ReduceDtype::BF16;
        if (is_master)
            std::cout << "[ZeRO-1] BLU_BF16_REDUCE=1: reducing gradients in BF16 "
                         "(diagnostic — costs memory on the FP32 group, saves ~0.3 ms/step)"
                      << std::endl;
    }
    if (is_master && getenv_int("BLU_GLOBAL_CLIP", 0) != 0)
        std::cout << "[ZeRO-1] BLU_GLOBAL_CLIP=1: clipping on the norm over ALL "
                     "parameter groups (DeepSpeed semantics) instead of per group"
                  << std::endl;
    zcfg.verbose                      = is_master;
    // Always printed, not only on the reference path: which halves of the overlap
    // are live is now a per-run choice, and every log has to record it or the A/B
    // is unreadable after the fact.
    if (is_master) {
        std::cout << "[ZeRO-1] overlap: reduce="
                  << (zcfg.overlap_backward_comm ? "ON (hook-driven, during backward)"
                                                 : "off (monolithic, after backward)")
                  << " | refresh="
                  << (zcfg.overlap_param_refresh ? "ON (staged into next forward)"
                                                 : "off (synchronous publish in step)");
        if (no_overlap) std::cout << "  [BLU_NO_OVERLAP=1]";
        std::cout << std::endl;
    }
    if (is_master) {
        std::cout << "[ZeRO-1] variant: " << (use_zero_v2 ? "v2 reduce-scatter/all-gather"
                                                          : "v1 reduce/broadcast")
                  << " | symmetric memory: " << ((want_symmetric && use_zero_v2) ? "requested"
                                                                                 : "off")
                  << "  (BLU_ZERO_V2=" << (use_zero_v2 ? 1 : 0)
                  << " BLU_SYMMETRIC=" << (want_symmetric ? 1 : 0) << ")" << std::endl;
    }

    // ---- ZeRO-1 for Muon --------------------------------------------------
    // Muon's update is NOT elementwise, so it cannot ride the flat contiguous
    // shard the AdamW groups use. Its core step is
    //
    //     U = NewtonSchulz(M)
    //
    // an iterated matrix polynomial (X <- aX + (bXX' + c(XX')^2)X) in which every
    // output entry depends on the WHOLE matrix. ShardedOptimizer gives rank r the
    // flat range [r*S,(r+1)*S) of every parameter concatenated end to end, which
    // lands mid-matrix: a rank ends up holding the last 300 rows of wQ, all of wK
    // and the first 900 rows of wV. Orthogonalizing those fragments separately is
    // a DIFFERENT operator, not an approximation of the right one — so reusing
    // the elementwise shard here would not be a slower or less accurate Muon, it
    // would not be Muon at all.
    //
    // ShardedMuon therefore shards by WHOLE PARAMETER: each matrix is assigned in
    // full to one owner rank, which holds its FP32 master + momentum and runs the
    // complete Newton-Schulz on it. Gradients reduce-to-owner, the owner updates,
    // and the result broadcasts back — so the state memory still drops by ~1/W,
    // which is what ZeRO-1 actually promises, while the math stays identical to
    // single-process Muon.
    //
    // It comes out CHEAPER than sharded AdamW on the same parameters: Muon keeps
    // no second moment for an orthogonalized param (8 B/elem of persistent state
    // against AdamW's 12), and whole-tensor ownership needs no flat staging
    // buffer at all, because a broadcast can write straight into the destination
    // tensor. That removes the unsharded [padded_numel] compute-dtype buffer that
    // dominates ShardedOptimizer's footprint on this model's BF16 group.
    // BLU_MUON_ADJUST_LR=original reproduces published modded-nanogpt numbers.
    // Note there is NO single lr that converts between the conventions: the ratio
    // is 0.2*sqrt(cols), which on this model spans 4.5x (wK/wV) to 21.2x
    // (gate_up). Switching the convention means re-sweeping muon_lr, so it is
    // reported at startup rather than left implicit.
    owt::zero::MuonAdjustLr muon_adjust = owt::zero::MuonAdjustLr::MATCH_RMS_ADAMW;
    if (const char* a = std::getenv("BLU_MUON_ADJUST_LR");
        a && std::string(a) == "original") {
        muon_adjust = owt::zero::MuonAdjustLr::ORIGINAL;
    }

    // Which collectives carry the Muon group. The default is reduce-scatter/
    // all-gather over a bin-packed flat buffer: whole tensors are packed into
    // per-rank contiguous sub-chunks so the shard boundary lands on tensor
    // boundaries, which keeps Newton-Schulz seeing whole matrices while dropping
    // ~2 collectives/parameter to ~2 collectives/bucket AND making the traffic
    // NVLS-eligible (NCCL ships symmetric kernels for ReduceScatter/AllGather but
    // none for Reduce or Broadcast, so the old path could never reach NVLink
    // SHARP however BLU_SYMMETRIC was set).
    //
    // BLU_MUON_RB=1 restores the original one-ncclReduce + one-ncclBroadcast per
    // parameter. Keep it for bisection: it is the formulation this optimizer was
    // first validated against, so a disagreement between the two isolates a bug
    // in the bin-packed layout from a fault in the Muon math itself. It also
    // needs no flat buffer, which matters if the ~3.3 GB the default costs on
    // this model's BF16 group is the difference between fitting and not.
    MuonConfig mcfg;
    mcfg.collective         = muon_rb
                                  ? owt::zero::MuonCollective::REDUCE_BROADCAST
                                  : owt::zero::MuonCollective::REDUCE_SCATTER_ALL_GATHER;
    mcfg.target_buckets     = getenv_int("BLU_MUON_BUCKETS", 6);
    mcfg.grad_bucket_view   = zcfg.grad_bucket_view;   // one flag for every group
    // BLU_MUON_PIPELINE=0 turns off the pipelined parameter publish (bucket k's
    // all-gather is issued as soon as this rank's owned tensors in bucket k are
    // updated, so it transfers underneath the Newton-Schulz of later buckets).
    //
    // Wired here because MuonConfig::pipeline_publish now DEFAULTS TO TRUE, so
    // without this line this script would silently get the new schedule with no
    // way to turn it off -- and the one thing it must stay is bisectable. It
    // changes only the launch order, not a single number; measured at -4 ms on
    // 8xH100 against a +-2 ms noise floor, i.e. no win, kept because it costs
    // nothing and helps more as the collective gets slower (multi-node).
    mcfg.pipeline_publish   = getenv_int("BLU_MUON_PIPELINE", 1) != 0;
    mcfg.fused_publish      = getenv_int("BLU_FUSED_MUON_PUBLISH", 0) != 0;
    mcfg.lr                 = cfg.muon_lr;
    mcfg.momentum           = cfg.muon_momentum;
    mcfg.weight_decay       = cfg.muon_weight_decay;
    mcfg.ns_steps           = cfg.muon_ns_steps;
    mcfg.nesterov           = true;
    mcfg.adjust_lr          = muon_adjust;
    // The aux AdamW path inside ShardedMuon. Setting adamw_lr to the AdamW peak
    // and lr to the Muon peak fixes the ratio set_lr() preserves, so feeding the
    // Muon-scaled schedule value below drives BOTH paths correctly from one
    // schedule. (On this model the Muon group is 100% matrices, so the aux path
    // is unused — it exists so an untied lm_head or a stray 1-D parameter landing
    // in a Muon dtype group is handled correctly rather than orthogonalized.)
    mcfg.adamw_lr           = cfg.max_lr;
    mcfg.adamw_beta1        = zcfg.beta1;
    mcfg.adamw_beta2        = zcfg.beta2;
    mcfg.adamw_eps          = zcfg.eps;
    mcfg.adamw_weight_decay = cfg.weight_decay;
    mcfg.exclude_1d_from_weight_decay = true;
    mcfg.max_grad_norm      = cfg.grad_clip;
    mcfg.reduce_dtype       = zcfg.reduce_dtype;
    mcfg.init_broadcast     = zcfg.init_broadcast;
    mcfg.verify_consistency = zcfg.verify_consistency;
    mcfg.overlap_backward_comm = zcfg.overlap_backward_comm;
    mcfg.overlap_param_refresh = zcfg.overlap_param_refresh;
    mcfg.verbose            = is_master;

    // Held as the shared base so v1, v2 and Muon are interchangeable. Every call
    // site below (arm_grad_sync, set_param_refresh_stages, refresh_param_stage,
    // drain_param_refresh, last_grad_norm, zero_grad) is virtual on
    // ZeroOptimizerBase, and the base still derives from OwnTensor::nn::Optimizer
    // so CheckpointManager::save/load_latest keep working unchanged.
    std::vector<std::unique_ptr<owt::zero::ZeroOptimizerBase>> optimizers;
    std::vector<std::string>          group_label;
    std::vector<std::vector<Tensor>>  group_params;   // parallel to `optimizers`
    // Each group's own (peak, floor) LR pair, fed through the ONE shared WSD
    // curve. This mirrors train_muon.py, which calls get_lr twice — once with
    // (max_lr, min_lr) for the AdamW group and once with (muon_lr, muon_min_lr)
    // for the Muon group — rather than scaling one curve by a constant. The two
    // are equivalent only while both floors are the same FRACTION of their peak;
    // carrying the pair keeps Muon's floor independently settable.
    std::vector<std::pair<float,float>> group_lr;

    // Muon groups first, then the AdamW remainder. Every rank builds this list in
    // the same order from the same deterministic partition, which is what keeps
    // the per-group NCCL collectives paired across the communicator.
    for (auto& kv : muon_by_dtype) {
        optimizers.push_back(std::make_unique<ShardedMuon>(kv.second, pg, mcfg));
        group_params.push_back(kv.second);
        // Same curve, Muon's own peak and floor: same warmup length, same stable
        // window, same cosine tail.
        group_lr.emplace_back(cfg.muon_lr, cfg.muon_min_lr);
        int64_t n = 0;
        for (auto& p : kv.second) n += p.numel();
        group_label.push_back("muon:" + OwnTensor::get_dtype_name(kv.first) + "/" +
                              std::to_string(n / 1000000) + "M");
    }
    for (auto& kv : params_by_dtype) {
        if (use_zero_v2) {
            optimizers.push_back(std::make_unique<SymmetricShardedOptimizer>(kv.second, pg, zcfg));
        } else {
            optimizers.push_back(std::make_unique<ShardedOptimizer>(kv.second, pg, zcfg));
        }
        group_params.push_back(kv.second);
        group_lr.emplace_back(cfg.max_lr, cfg.min_lr);
        // Label each group for the per-step gradient-norm report. The combined
        // sqrt(sum of squares) that used to be the only number reported cannot
        // distinguish "the 103M-param fp32 group (tied wte + all the RMSNorm/
        // QK-norm scales) blew up" from "the 1.62B-param bf16 group (every
        // attention/MLP weight) blew up" — and those point at completely different
        // subsystems. G0/G1 here are indices into `optimizers`, NOT step_order.
        int64_t n = 0;
        for (auto& p : kv.second) n += p.numel();
        group_label.push_back("adamw:" + OwnTensor::get_dtype_name(kv.first) + "/" +
                              std::to_string(n / 1000000) + "M");
    }
    // Multi-node bluccl: one reduce-scatter per bucket per step; all but the last BLU_RS_TAIL (default 2) run the
    // overlap kernel beside backward, the tail the standalone path. Other backends ignore the schedule.
    if (bluccl_multi && pg && world_size > 1) {
        int rs_total = 0;
        for (auto& o : optimizers) {
            if (auto* v2 = dynamic_cast<SymmetricShardedOptimizer*>(o.get())) rs_total += (int)v2->num_buckets();
            else if (auto* mu = dynamic_cast<ShardedMuon*>(o.get())) rs_total += (int)mu->num_buckets();
        }
        const int rs_tail = std::max(0, getenv_int("BLU_RS_TAIL", 2));
        pg->set_rs_schedule(rs_total, rs_tail);
        if (is_master)
            std::cout << "[BluCCL] reduce-scatters per step: " << rs_total << " -> overlap kernel for "
                      << std::max(0, rs_total - rs_tail) << ", standalone for the last " << std::min(rs_total, rs_tail)
                      << std::endl;
    }
    if (is_master) {
        std::cout << "[optimizer] " << (use_muon ? "MUON + AdamW" : "AdamW")
                  << "  (BLU_OPTIMIZER=" << (use_muon ? "muon" : "adamw") << ")";
        if (use_muon) {
            std::cout << " | muon_lr " << cfg.muon_lr
                      << " / " << cfg.muon_min_lr
                      << " momentum " << cfg.muon_momentum
                      << " wd " << cfg.muon_weight_decay
                      << " ns_steps " << cfg.muon_ns_steps
                      << " publish " << (mcfg.pipeline_publish
                                           ? "pipelined (AG per bucket, under NS)"
                                           : "sequential [BLU_MUON_PIPELINE=0]")
                      << " adjust_lr "
                      << (muon_adjust == owt::zero::MuonAdjustLr::MATCH_RMS_ADAMW
                              ? "match_rms_adamw" : "original");
        }
        std::cout << std::endl;
        for (size_t gi = 0; gi < optimizers.size(); ++gi) {
            std::cout << "  G" << gi << " " << group_label[gi]
                      << " : " << optimizers[gi]->variant_name()
                      << " | lr " << group_lr[gi].first
                      << " -> " << group_lr[gi].second << std::endl;
        }
    }

    // ---- what the gradient clip actually covers ------------------------------
    // The clip coefficient is computed from the norm over EVERY parameter, but it
    // is not applied to every parameter: ShardedMuon withholds the scale from the
    // orthogonalized tensors on purpose (Newton-Schulz renormalizes the momentum,
    // so a step-varying coefficient would only rotate the EMA). In the usual
    // partition that is every tensor it owns, which means `grad_clip` covers only
    // the embedding + 1-D group — a single-digit percentage of the model.
    //
    // Printed because the per-step `globalclip 0.0457` line otherwise reads as
    // "the model's gradients were scaled to 4.6%", which would be false. If this
    // says 6% coverage, `grad_clip` is not protecting the body and something else
    // has to.
    if (cfg.grad_clip > 0.0f) {
        int64_t covered = 0, total = 0;
        for (size_t gi = 0; gi < optimizers.size(); ++gi) {
            covered += static_cast<int64_t>(optimizers[gi]->clip_covered_numel());
            total   += optimizers[gi]->total_numel();
        }
        const double pct = total > 0 ? 100.0 * static_cast<double>(covered) / total : 0.0;
        if (is_master) {
            std::cout << "[ZeRO-1] grad_clip=" << cfg.grad_clip << " APPLIES TO "
                      << covered << " / " << total << " params ("
                      << std::fixed << std::setprecision(2) << pct << "%)"
                      << std::defaultfloat << std::endl;
            if (covered == 0) {
                std::cout << "[ZeRO-1] *** WARNING: grad_clip covers NO parameters. "
                             "Every group withholds the coefficient (Muon path). The "
                             "reported norm and globalclip are DIAGNOSTIC ONLY — no "
                             "gradient is being scaled. ***" << std::endl;
            } else if (pct < 25.0) {
                std::cout << "[ZeRO-1] *** WARNING: grad_clip leaves "
                          << std::fixed << std::setprecision(2) << (100.0 - pct)
                          << "% of parameters unclipped (the Muon-path body). ***"
                          << std::defaultfloat << std::endl;
            }
        }
    }

    // Order in which the dtype groups are STEPPED (distinct from their order in
    // `optimizers`, which the checkpointing code indexes as [0] == the group the
    // CheckpointManager API covers). The group carrying the bulk of the
    // parameters must step LAST: its staged refresh broadcasts are issued at the
    // very end of its step(), and any *other* group stepping afterwards would
    // enqueue a blocking norm all-reduce behind them on the same in-order comm
    // stream and then wait on it — head-of-line blocking that drains the
    // broadcasts and silently destroys the overlap. Ascending parameter count
    // puts the biggest group last.
    // Give every group (Muon whole-tensor groups AND AdamW flat-shard groups
    // alike) a distinct DCP scope. DCP names a state tensor
    // "optimizer.<state_type>.<param_fqn>" and param_fqn is derived from this
    // scope, so without it two groups would write the same names into the same
    // checkpoint and clobber each other.
    for (size_t g = 0; g < optimizers.size(); ++g) {
        optimizers[g]->set_dcp_scope("zero_g" + std::to_string(g));
    }

    std::vector<size_t> step_order(optimizers.size());
    for (size_t i = 0; i < step_order.size(); ++i) step_order[i] = i;
    {
        std::vector<int64_t> group_numel(optimizers.size(), 0);
        for (size_t gi = 0; gi < group_params.size(); ++gi)
            for (auto& p : group_params[gi]) group_numel[gi] += p.numel();
        std::stable_sort(step_order.begin(), step_order.end(),
                         [&](size_t a, size_t b) { return group_numel[a] < group_numel[b]; });
    }

    // ---- refresh stage granularity ------------------------------------------
    // Stage 0 holds ONLY wte. The embedding lookup is the very first op of the
    // forward, so wte's transfer has nothing to hide behind and is structurally
    // exposed however the stages are cut — but everything ELSE can be binned by
    // the layer that reads it, so layer i waits only on layer i's own pieces
    // while the later layers keep transferring underneath its compute.
    //
    // LAYERS_PER_STAGE trades pipelining granularity against collective count:
    //   1  = one stage per layer — finest. Each layer's attention compute covers
    //        the following layers' transfer many times over.
    //   >1 = coarser: fewer, larger broadcasts. Raise it when per-collective
    //        latency on the interconnect starts to dominate the bytes moved.
    const int LAYERS_PER_STAGE = std::max(1, cfg.layers_per_stage);
    if (zcfg.overlap_param_refresh) {
        const int n_layers       = static_cast<int>(cfg.n_layers);
        const int layer_stages   = (n_layers + LAYERS_PER_STAGE - 1) / LAYERS_PER_STAGE;
        const int REFRESH_STAGES = 1 + layer_stages;      // stage 0 == wte
        const int last_stage     = REFRESH_STAGES - 1;

        // Parameter -> stage, keyed by data pointer. set_param_refresh_stages
        // then maps it through input_index, so the caller never sees the flat
        // ordering.
        std::map<const void*, int> stage_by_ptr;
        for (auto& p : model.wte.parameters()) stage_by_ptr[p.data()] = 0;
        for (int i = 0; i < n_layers; ++i) {
            const int s = 1 + i / LAYERS_PER_STAGE;
            for (auto& p : model.attn[i]->parameters()) stage_by_ptr[p.data()] = s;
            for (auto& p : model.mlp[i]->parameters())  stage_by_ptr[p.data()] = s;
        }
        // norm_f and an untied lm_head are read AFTER every layer, so they ride
        // along with the final layer stage — installed by that layer's pre-hook,
        // which fires before either is reached. (Under weight tying the logits
        // matmul re-reads wte.weight, already installed as stage 0.)
        for (auto& p : model.norm_f.parameters()) stage_by_ptr[p.data()] = last_stage;
        if (model.lm_head)
            for (auto& p : model.lm_head->parameters()) stage_by_ptr[p.data()] = last_stage;

        int unmapped = 0;
        for (size_t oi = 0; oi < group_params.size(); ++oi) {
            std::vector<int> stage_of;
            stage_of.reserve(group_params[oi].size());
            for (auto& p : group_params[oi]) {
                auto it = stage_by_ptr.find(p.data());
                if (it != stage_by_ptr.end()) {
                    stage_of.push_back(it->second);
                } else {
                    // Unrecognised parameter (a module this mapping does not know
                    // about). Pin it to stage 0: always CORRECT, since stage 0 is
                    // installed before the forward starts — it only forgoes the
                    // overlap for that tensor instead of risking a read-before-
                    // install further down the forward.
                    stage_of.push_back(0);
                    ++unmapped;
                }
            }
            optimizers[oi]->set_param_refresh_stages(stage_of, REFRESH_STAGES);
        }
        if (is_master) {
            std::cout << "ZeRO-1 staged refresh: " << REFRESH_STAGES << " stages ("
                      << LAYERS_PER_STAGE << " layer(s)/stage)";
            if (unmapped > 0)
                std::cout << " | WARNING: " << unmapped << " parameter(s) unmatched, "
                             "pinned to stage 0 (correct, not overlapped)";
            std::cout << std::endl;
        }

        // Install each layer stage immediately before the first block that reads
        // it, via a forward pre-hook on that block. Stage 0 is already in place by
        // then (zero_grad installs it before the forward starts). Hooks fire in
        // ascending layer order, which is also ascending stage order — the order
        // refresh_param_stage() requires.
        //
        // Hooks fire ONLY through Module::operator(); GPT::forward invokes blocks
        // as (*attn[i])(x) precisely so this dispatch happens, and a direct
        // attn[i]->forward(x) call would bypass it entirely. refresh_param_stage()
        // early-outs once the step's stages are consumed, so the repeat calls from
        // the other GRAD_ACCUM-1 micro-steps and from the validation forward are
        // free.
        for (int i = 0; i < n_layers; i += LAYERS_PER_STAGE) {
            const int s = 1 + i / LAYERS_PER_STAGE;
            model.attn[i]->register_forward_pre_hook(
                [&optimizers, s](nn::Module&, const Tensor&) -> std::optional<Tensor> {
                    for (auto& opt : optimizers) opt->refresh_param_stage(s);
                    return std::nullopt;   // input unchanged
                });
        }
    }

    if (is_master) {
        int64_t n_decay = 0, n_nodecay = 0;
        for (const auto& p : params) {
            if (p.shape().dims.size() >= 2) n_decay += p.numel();
            else                            n_nodecay += p.numel();
        }
        std::cout << "decayed params: " << n_decay << ", non-decayed: " << n_nodecay
                  << " | ZeRO-1 optimizer groups: " << optimizers.size()
                  << " | world_size: " << world_size << std::endl;
    }

    // ---- data: weighted multi-domain mixture ----
    // Training draws from several domains at target ratios rather than from
    // one blended corpus, so the ratio is a knob we can move (and log) instead
    // of a property baked into the shards at tokenisation time.
    //
    // The weights are TARGETS, not probabilities: MixedDataLoaderLite's
    // error-tracking selection keeps the realised mixture within one sample of
    // them at every prefix of a phase — see pick_domain() in MixedDataloader.h.
    //
    // Every rank builds the SAME domain list and therefore picks the same
    // domain sequence; ranks differ only in their read offset within the picked
    // domain (B*T*rank, striding by B*T*world), which is the usual DDP split.
    // ---- mixture preset ------------------------------------------------
    // BLU_MIX_PRESET selects the recipe. The DEFAULT IS c0 — the original
    // static 70/20/10 over commoncrawl/code/math, with edufineweb absent from
    // the domain list entirely. Nothing below changes what an existing launch
    // script does; the newer recipes are opt-in.
    //
    //   c0  baseline    cc .70            code .20  math .10   (default)
    //   c1  rebalanced  cc .55  edu .15   code .12  math .18
    //   c2  anneal      two phases, switching at the WSD LR decay onset:
    //                     stable  cc .58  edu .05   code .22  math .15
    //                     anneal  cc .33  edu .25   code .22  math .20
    //
    // Why c1/c2 move weight the way they do — from the 1.7B static-70/20/10 run
    // at ~/Desktop/mix17b_8xh200 (4.5B tokens), the only multi-domain run to
    // date:
    //   * code was nearly saturated at 20% of the mix (val 0.940 -> 0.806 across
    //     the whole stable window) while math at 10% moved 2.086 -> 1.864. Per
    //     1% of mix that is -0.0067 nats for code vs -0.0222 for math, so both
    //     recipes shift budget code -> math.
    //   * every domain's val loss fell ~2.5x faster per step once the LR decayed
    //     (cc -0.026 -> -0.069 per 250 steps). That is the leverage c2's second
    //     phase exists to exploit: data shown while the weights are settling has
    //     outsized influence on the final model.
    //   * commoncrawl still had the most headroom and stays the largest slice.
    //
    // edufineweb is only 600M tokens (1/8 of the others) and is FILTERED web
    // rather than a separate domain, so c2 holds it low while the LR is high and
    // concentrates it into the anneal (~0.6 epochs). The 5x5 corpus study in
    // ~/Desktop/Datatest/ANALYSIS.md ranks FineWeb-Edu the best generalist of
    // five corpora (mean transfer gap 0.217), which is why it gets that slot
    // rather than more commoncrawl.
    //
    // Layout expected under BLU_MIX_ROOT:
    //   <root>/commoncrawl/{train_*.bin, val_*.bin}
    //   <root>/code/{...}   <root>/math/{...}
    //   <root>/edufineweb/{edufineweb_train_*.bin, edufineweb_val_*.bin}   (c1/c2 only)
    const std::string mix_root  = getenv_str("BLU_MIX_ROOT", "/data");
    const std::string mix_preset = getenv_str("BLU_MIX_PRESET", "c0");
    if (mix_preset != "c0" && mix_preset != "c1" && mix_preset != "c2") {
        if (is_master)
            std::cerr << "ERROR: BLU_MIX_PRESET must be one of c0|c1|c2 (got '"
                      << mix_preset << "')" << std::endl;
        dist::comm::abort_job(launch, 3);
    }
    const bool use_edu = (mix_preset != "c0");

    // Domain ORDER is load-bearing: the phase weight vectors are indexed
    // positionally against this list, as is the val blend below. edufineweb is
    // OMITTED for c0 rather than carried at weight 0 — a zero-weight domain can
    // still win pick_domain()'s argmax on a tie, and carrying it would make c0
    // require an edufineweb/ directory it never reads.
    std::vector<DomainConfig> domains;
    domains.push_back({"commoncrawl", mix_root + "/commoncrawl", "train", 0.0f,
                                      mix_root + "/commoncrawl", "val"});
    if (use_edu)
        domains.push_back({"edufineweb", mix_root + "/edufineweb", "train", 0.0f,
                                         mix_root + "/edufineweb", "val"});
    domains.push_back({"code", mix_root + "/code", "train", 0.0f,
                               mix_root + "/code", "val"});
    domains.push_back({"math", mix_root + "/math", "train", 0.0f,
                               mix_root + "/math", "val"});

    struct MixPhase {
        const char* name;
        int start_step;
        std::vector<float> w;   // positional, same order as `domains`
    };

    // c2's boundary is DERIVED from the LR schedule rather than hardcoded, so it
    // keeps tracking the decay onset when BLU_MAX_STEPS rewrites warmup_steps /
    // stable_steps (it makes them ms/10 and ms/2, moving the boundary from 80%
    // of the run to 60%).
    const int anneal_start = getenv_int("BLU_MIX_ANNEAL_START",
                                        cfg.warmup_steps + cfg.stable_steps);

    std::vector<MixPhase> mix_schedule;
    if (mix_preset == "c0") {
        mix_schedule.push_back({"c0-static", 0,
            {getenv_float("BLU_W_CC",   0.70f),
             getenv_float("BLU_W_CODE", 0.20f),
             getenv_float("BLU_W_MATH", 0.10f)}});
    } else if (mix_preset == "c1") {
        mix_schedule.push_back({"c1-static", 0,
            {getenv_float("BLU_W_CC",   0.55f),
             getenv_float("BLU_W_EDU",  0.15f),
             getenv_float("BLU_W_CODE", 0.12f),
             getenv_float("BLU_W_MATH", 0.18f)}});
    } else {  // c2
        mix_schedule.push_back({"stable", 0,
            {getenv_float("BLU_MIX_STABLE_CC",   0.58f),
             getenv_float("BLU_MIX_STABLE_EDU",  0.05f),
             getenv_float("BLU_MIX_STABLE_CODE", 0.22f),
             getenv_float("BLU_MIX_STABLE_MATH", 0.15f)}});
        mix_schedule.push_back({"anneal", anneal_start,
            {getenv_float("BLU_MIX_ANNEAL_CC",   0.33f),
             getenv_float("BLU_MIX_ANNEAL_EDU",  0.25f),
             getenv_float("BLU_MIX_ANNEAL_CODE", 0.22f),
             getenv_float("BLU_MIX_ANNEAL_MATH", 0.20f)}});
    }
    // Ablation handle: collapse any multi-phase recipe to its first phase, so
    // c2-without-the-anneal can be run from the same binary as the control for
    // the schedule itself (same domains, same weights, no transition).
    if (getenv_int("BLU_MIX_STATIC", 0) != 0 && mix_schedule.size() > 1)
        mix_schedule.resize(1);

    for (const auto& ph : mix_schedule) {
        if (ph.w.size() != domains.size()) {
            if (is_master)
                std::cerr << "ERROR: mix phase '" << ph.name << "' has " << ph.w.size()
                          << " weights for " << domains.size() << " domains" << std::endl;
            dist::comm::abort_job(launch, 3);
        }
    }

    // Which phase a given optimizer step belongs to. Phases are ordered by
    // start_step, so the last one whose start has been reached wins.
    auto phase_for = [&mix_schedule](int step) {
        int p = 0;
        for (size_t i = 0; i < mix_schedule.size(); ++i)
            if (step >= mix_schedule[i].start_step) p = static_cast<int>(i);
        return p;
    };

    if (is_master && !sft) {
        std::cout << "\n=== Data mixture: preset " << mix_preset
                  << (mix_schedule.size() > 1 ? " (phased)" : " (static)") << " ===\n";
        for (const auto& ph : mix_schedule) {
            std::cout << "  [" << std::left << std::setw(10) << ph.name << "] from step "
                      << std::right << std::setw(6) << ph.start_step << " :";
            for (size_t d = 0; d < domains.size(); ++d)
                std::cout << "  " << domains[d].name << "=" << std::fixed
                          << std::setprecision(3) << ph.w[d];
            std::cout << "\n";
        }
        std::cout << "=================================================" << std::endl;
    }

    // Seeded with phase 0 so construction has a valid weight vector; the step
    // loop re-applies the correct phase before the first batch is drawn.
    for (size_t d = 0; d < domains.size(); ++d) domains[d].weight = mix_schedule[0].w[d];

    // Not created in SFT mode: sft_train / sft_val above replace it, and opening it would need the
    // pretraining shards under BLU_MIX_ROOT. The last argument is the device ordinal, i.e. the LOCAL rank.
    std::unique_ptr<MixedDataLoaderLite> train_loader;
    if (!sft)
        train_loader = std::make_unique<MixedDataLoaderLite>(static_cast<int>(cfg.B), static_cast<int>(cfg.T),
                                                             rank, world_size, domains, is_master, 100000000,
                                                             local_rank);
    int cur_mix_phase = -1;   // forces set_weights() on the first step

    // One val loader per domain. A single blended val number can stay flat
    // while code and math move in opposite directions, which is precisely the
    // thing a mixture run needs to see; the headline `val_loss` below is these
    // recombined at the TRAINING weights, so it stays comparable to what the
    // single-source runs logged into the same CSV column. Empty in SFT mode (no pretraining loader).
    std::vector<std::pair<std::string, std::unique_ptr<DataLoaderLite>>> val_loaders;
    if (train_loader) val_loaders = train_loader->create_val_loaders();

    // Every rank must read the SAME shards: a partial copy on one node changes its
    // directory listing and nothing downstream notices. Fingerprint every domain's
    // train and val shard list (names + byte sizes) and refuse to start on a mismatch.
    // SFT mode opens no pretraining shard (its data is checked by SftDataLoader), so there is nothing to compare.
    if (world_size > 1 && !sft) {
        auto fingerprint = [](const std::vector<std::string>& shards) {
            uint64_t h = 1469598103934665603ull;
            auto mix = [&h](const void* p, size_t n) {
                const unsigned char* b = static_cast<const unsigned char*>(p);
                for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
            };
            const uint64_t n = shards.size();
            mix(&n, sizeof n);
            for (const auto& sh : shards) {
                mix(sh.data(), sh.size());
                std::error_code ec;
                uint64_t bytes = static_cast<uint64_t>(std::filesystem::file_size(sh, ec));
                if (ec) bytes = ~0ull;
                mix(&bytes, sizeof bytes);
            }
            return static_cast<double>(h & ((1ull << 52) - 1));
        };
        std::vector<std::pair<std::string, double>> fps;
        for (const auto& d : domains)
            fps.push_back({d.name + " train (" + d.data_root + ")",
                           fingerprint(list_shards_for_domain(d.data_root, d.split))});
        for (const auto& vl : val_loaders)
            fps.push_back({vl.first + " val", static_cast<double>(vl.second->shard_fingerprint())});

        const double n_lo = dist::comm::allreduce_scalar(launch, pg, static_cast<double>(fps.size()), op_min);
        const double n_hi = dist::comm::allreduce_scalar(launch, pg, static_cast<double>(fps.size()), op_max);
        if (n_lo != n_hi) {
            std::cerr << "[data] RANKS DISAGREE ON THE DATASET: rank " << rank << " has "
                      << val_loaders.size() << " val domain(s); ranks range from "
                      << n_lo - domains.size() << " to " << n_hi - domains.size()
                      << ". Every node needs an identical copy of the mix directory." << std::endl;
            dist::comm::abort_job(launch, 3);
        }
        for (const auto& [what, fp] : fps) {
            const double lo = dist::comm::allreduce_scalar(launch, pg, fp, op_min);
            const double hi = dist::comm::allreduce_scalar(launch, pg, fp, op_max);
            if (lo != hi) {
                std::cerr << "[data] RANKS DISAGREE ON THE DATASET: rank " << rank
                          << " sees fingerprint " << std::fixed << fp << " for " << what
                          << " (world min " << lo << ", max " << hi << "). Every node "
                             "needs an identical copy of the shard directory." << std::endl;
                dist::comm::abort_job(launch, 3);
            }
        }
        if (is_master)
            std::cout << "[data] shard lists agree across all " << world_size << " ranks ("
                      << fps.size() << " lists: " << domains.size() << " train, "
                      << val_loaders.size() << " val)" << std::endl;
    }
    std::vector<float> domain_val_loss(val_loaders.size(), -1.0f);

    // The headline val_loss is blended at a FIXED reference mixture, NOT at the
    // live phase weights. Blending at the live weights would make val_loss jump
    // discontinuously the moment the anneal phase starts — the metric would
    // change definition mid-curve, and the step-1600 drop would be an artefact
    // of reweighting rather than a real improvement. The stable-phase weights
    // are the reference because that phase is ~80% of the tokens. Every
    // per-domain loss is logged separately anyway, so any other blend can be
    // recovered from the CSV after the fact.
    std::vector<float> val_blend_w(val_loaders.size(), 0.0f);
    for (size_t d = 0; d < val_loaders.size() && d < mix_schedule[0].w.size(); ++d)
        val_blend_w[d] = mix_schedule[0].w[d];

    // ---- CSV log (master only) ----
    std::ofstream log_file;
    if (is_master && csv_enabled) {
        if (!csv_dir.empty()) fs::create_directories(csv_dir);
        log_file.open(csv_path, csv_mode == 1 ? std::ios::app : std::ios::out);
        if (csv_mode != 1) {
            // MFU columns are APPENDED, never inserted: run_zero_ab_8xh200.sh parses
            // this file positionally ($1..$8 in its awk block), so adding fields at
            // the end keeps the existing A/B reporting working unchanged.
            log_file << "step,train_loss,val_loss,tok_per_sec,grad_norm,dt_ms,lr,elapsed_min,"
                        "mfu,mfu_palm,tflops_per_gpu";
            // Per-domain val losses are APPENDED after the existing columns, for the
            // same reason the MFU columns were: run_zero_ab_8xh200.sh reads this file
            // positionally, so anything added at the end leaves it working unchanged.
            for (auto& vl : val_loaders) log_file << ",val_" << vl.first;
            log_file << ",tokens_seen,avg_tok_per_sec";
            if (train_loader)   // the pretraining mixture: one column per domain (none in SFT mode)
                for (int d = 0; d < train_loader->n_domains(); ++d) log_file << ",shard_" << train_loader->name_of(d);
            // SFT: the step's trained tokens (labels != 65535, all micro-batches, all ranks) = its loss denominator.
            if (sft) log_file << ",trained_tokens";
            log_file << "\n";
        }
        log_file << std::fixed << std::setprecision(6);
        log_file.flush();
    }
    if (is_master && csv_enabled && !run_dir.empty()) {
        std::vector<std::string> args = {
            getenv_str("BLU_MONITOR_PYTHON", "python3"),
            getenv_str("BLU_MONITOR_SCRIPT", launch_dir + "/train/production_zero/monitoring/blutrain_monitor.py"),
            "exporter", "--run-dir", run_dir, "--parent-pid", std::to_string(getpid()),
            "--world-size", std::to_string(world_size), "--nodes", std::to_string(launch_nodes),
            "--config", cfg_report.path};
        std::vector<char*> argv_c;
        for (auto& a : args) argv_c.push_back(a.data());
        argv_c.push_back(nullptr);
        posix_spawnattr_t attr;
        posix_spawnattr_init(&attr);
        posix_spawnattr_setflags(&attr, POSIX_SPAWN_SETSID);
        pid_t mon_pid = 0;
        const int spawn_rc = posix_spawnp(&mon_pid, argv_c[0], nullptr, &attr, argv_c.data(), environ);
        posix_spawnattr_destroy(&attr);
        if (spawn_rc != 0)
            std::cerr << "[monitor] could not start " << args[1] << "; training continues without live charts"
                      << std::endl;
    }

    // Reused GPU scalars (avoid per-step allocation).
    // The backward seed. Pretraining: each micro-step's loss is the mean over its own B*T tokens, so
    // 1/GRAD_ACCUM turns the sum of the micro-steps into their mean. SFT: each micro-step's loss is
    // already divided by the step's global trained-token count, so the micro-steps just add, and
    // world_size cancels the ZeRO average over ranks.
    Tensor grad_scale = Tensor::full(Shape{{1}}, TensorOptions().with_device(device),
                                     sft ? static_cast<float>(world_size)
                                         : 1.0f / static_cast<float>(GRAD_ACCUM));
    Tensor loss_accum_gpu = Tensor::zeros(Shape{{1}}, TensorOptions().with_device(device));
    Tensor sft_denom;          // SFT: the step's global trained-token count, refilled every step
    int64_t step_tokens = 0;   // SFT: that count as an integer, for the log

    float last_val_loss = -1.0f;  // carried across steps; refreshed every val_freq
    auto training_start = std::chrono::high_resolution_clock::now();

    // Spike diagnostics (BLU_DIAG=1). See the micro loop and the per-step report.
    const bool diag = getenv_int("BLU_DIAG", 0) != 0;
    std::vector<float> micro_loss;
    // (domain_index, shard_index, token_offset) — domain is what makes an
    // outlier actionable under mixing: "shard 2 offset 41M" names a different
    // file depending on which domain the batch was drawn from.
    struct MicroWhere { int domain; int shard; size_t offset; };
    std::vector<MicroWhere> micro_where;

    // Depth probe: the accumulated LOCAL gradient norm of one weight per probe
    // layer. The per-group norms show the spike is ~45x in the 1.62B bf16 body but
    // only ~6x in the embedding group, i.e. localized in the stack — this says at
    // WHICH DEPTH it starts. Compare the profile on a spiking step against a quiet
    // one: a blow-up that begins mid-stack and grows toward the loss points at the
    // forward, while a flat multiplicative lift across all depths points at the
    // optimizer/update. Read before opt->step(), so these are the pre-reduce
    // per-rank gradients (step() does not modify param .grad, and zero_grad only
    // runs at the top of the next step).
    const std::vector<int> probe_layers = {0, 8, 17, 26, 35};
    std::vector<std::pair<int, float>> layer_gnorm;   // (layer, ||grad(wQ.weight)||)
    std::vector<std::pair<int, float>> gamma_absmax;  // (layer, max|q_gamma|)
    std::vector<std::pair<int, float>> w_absmax;      // (layer, max|wQ.weight|)
    auto grad_l2 = [](const Tensor& p) -> float {
        if (!p.has_grad()) return -1.0f;
        Tensor h = p.grad_view().as_type(Dtype::Float32).to_cpu();
        const float* d = h.data<float>();
        const int64_t n = h.numel();
        double acc = 0.0;
        for (int64_t i = 0; i < n; ++i) acc += static_cast<double>(d[i]) * d[i];
        return static_cast<float>(std::sqrt(acc));
    };
    // Replica-consistency probe. After the staged refresh has been consumed, every
    // rank must hold BIT-IDENTICAL parameters: each byte of each parameter is
    // produced by exactly one owner's broadcast, so there is no averaging or
    // rounding path that could legitimately make two ranks differ. If they DO
    // differ, the ranks are training divergent models and averaging their
    // gradients is meaningless — which would show up exactly as this failure mode
    // (normal-looking gradient norm, destructive update, all micro-batches worse
    // the following step). Bit-exact comparison via min/max across ranks, so any
    // difference at all is reported rather than hidden under a tolerance.
    auto checksum = [](const Tensor& p) -> double {
        Tensor h = p.as_type(Dtype::Float32).to_cpu();
        const float* d = h.data<float>();
        const int64_t n = h.numel();
        double acc = 0.0;
        // Position-weighted so a permutation or a shifted piece is caught too, not
        // just a change in the multiset of values.
        for (int64_t i = 0; i < n; ++i) acc += static_cast<double>(d[i]) * (i % 8191 + 1);
        return acc;
    };
    auto abs_max = [](const Tensor& p) -> float {
        Tensor h = p.as_type(Dtype::Float32).to_cpu();
        const float* d = h.data<float>();
        const int64_t n = h.numel();
        float m = 0.0f;
        for (int64_t i = 0; i < n; ++i) m = std::max(m, std::fabs(d[i]));
        return m;
    };

    // ---- checkpointing: manager + auto-resume ----
    // save() writes one shared (non-sharded) path, so saves are master-only to
    // avoid an 8-rank write race; load_latest runs on EVERY rank (concurrent
    // reads are safe) so each rank restores its own optimizer m/v state and
    // fast-forwards its own data shard. Saves are synchronous (use_async=false).
    // NOTE: CheckpointManager's save/load API takes a single nn::Optimizer&,
    // so optimizers[0] rides the normal path together with the model's full
    // (unsharded) parameter tensors. Any further dtype group isn't covered by
    // that API, so its shard state round-trips through a small sidecar file
    // keyed off the same step number.
    // Muon and AdamW runs write to DIFFERENT directories. Their optimizer state
    // is not interchangeable — whole-tensor master+momentum against a flat
    // elementwise master/m/v shard — and optimizers[0] is a different class in
    // each mode, so a cross-mode resume would (correctly) abort on the layout
    // descriptor. Separating the directories means flipping BLU_OPTIMIZER starts
    // a clean run instead of failing against the other mode's checkpoints.
    // if (use_muon) cfg.ckpt_dir += "_muon";

    // ---- checkpointing: Distributed Checkpointing (DCP) + auto-resume ----
    // Every rank writes its OWN shard of every group's optimizer state (plus its
    // copy of the replicated parameters) into one step directory; DCP merges the
    // per-rank metadata and, on load, intersects each saved flat range against
    // the resuming rank's range. That is what makes the save correct at W>1 for
    // BOTH kinds of group here: ShardedOptimizer/SymmetricShardedOptimizer's flat
    // elementwise shard, and ShardedMuon's whole-tensor ownership (see
    // ShardedMuon::dcp_state_shard_specs) — a master-only save would persist only
    // one rank's slice of either. It also lets a run resume at a DIFFERENT world
    // size, including for the Muon groups: the new run derives its own owner
    // assignment and each record's checkpoint range is self-contained.
    //
    // Two adapters bridge this trainer to DCP's API (see ZeroDcpBridge.h): the
    // model is an nn::Module rather than a dnn::DModule, and there is one ZeRO
    // group (Muon or AdamW) per dtype where DCP takes a single optimizer pointer.
    //
    // Muon and AdamW runs still write to DIFFERENT directories. Their group
    // scopes are identical either way ("zero_g0", "zero_g1", ...), but which
    // CONCRETE class sits behind scope g (ShardedMuon vs. ShardedOptimizer /
    // SymmetricShardedOptimizer) depends on BLU_OPTIMIZER, so the same scope
    // name means a different on-disk layout in each mode; loading one under the
    // other surfaces as a loud DCP coverage failure rather than silent garbage,
    // but there is no reason to court it. Separating the directories means
    // flipping BLU_OPTIMIZER starts a clean run instead of failing against the
    // other mode's checkpoints.
    if (use_muon) cfg.ckpt_dir += "_muon";

    // BLU_CKPT_DIR overrides the whole thing, suffix included. Two reasons it
    // exists: a checkpoint pulled from elsewhere rarely lands in the directory
    // name this binary derives, and checkpointing.load picks resume vs a fresh
    // run folder (see resolve_ckpt_run_dir). BLU_CKPT_FREQ shortens
    // the save interval, which is what makes a save path testable without
    // waiting ckpt_freq steps for one.
    if (const char* d = std::getenv("BLU_CKPT_DIR"); d && *d) cfg.ckpt_dir = d;
    if (const int cf = getenv_int("BLU_CKPT_FREQ", 0); cf > 0) cfg.ckpt_freq = cf;
    if (const int ck = getenv_int("BLU_CKPT_KEEP", 0); ck > 0) cfg.ckpt_keep = ck;

    std::map<std::string, Tensor*> dcp_named;
    for (size_t i = 0; i < params.size(); ++i) {
        char nm[32];
        std::snprintf(nm, sizeof nm, "param_%06zu", i);
        dcp_named[nm] = &params[i];
    }
    owt::zero::ZeroDcpModule dcp_model(dcp_named);

    std::vector<owt::zero::ZeroOptimizerBase*> dcp_groups;
    for (auto& o : optimizers) dcp_groups.push_back(o.get());
    owt::zero::ZeroDcpOptimizer dcp_opt(dcp_groups);

    // DistCheckpoint::save/load dereference `pg` for rank/world_size, but the
    // process-group setup above leaves it NULL at world_size == 1 (ZeRO-1 needs
    // no collectives there). Give the checkpointer its own single-rank group so
    // one-GPU runs can still save and resume instead of faulting in save().
    std::shared_ptr<ProcessGroupNCCL> ckpt_pg = nccl_pg;
    // SFT: BLU_INIT_FROM (above) made the NCCL group "dcp" when the trainer had none (W == 1, or single-node BluCCL at
    // W > 1). Reuse it rather than create a second group with the same name; without BLU_INIT_FROM init_pg is null and
    // everything below runs exactly as in bluscript_zero_ckpt.cpp.
    if (init_pg) ckpt_pg = init_pg;
    bool ckpt_enabled = getenv_int("BLU_CKPT", cfg.checkpointing ? 1 : 0) != 0;
    // DCP is typed on ProcessGroupNCCL and has no BluCCL backend, so BluCCL runs checkpoint on its own NCCL group.
    if (ckpt_enabled && use_bluccl && world_size > 1 && !init_pg) {   // !init_pg: see ckpt_pg above
        try {
            ckpt_pg = dist::comm::bring_up_process_group(launch, dist::comm::MemoryMode::Standard, "dcp");
            if (is_master)
                std::cout << "[BluCCL] checkpointing on a separate NCCL group" << std::endl;
        } catch (const std::exception& e) {
            ckpt_enabled = false;
            if (is_master)
                std::cerr << "[WARN] checkpointing disabled: could not create the NCCL group "
                             "DCP requires: " << e.what() << std::endl;
        }
    }

    // DCP used to only exchange save plans over raw MPI_COMM_WORLD, which is a
    // world-of-one under BluRun's External launcher (no mpirun). DcpComm
    // (DcpComm.h) removes that restriction: it can drive the same byte
    // collectives over the trainer's own NCCL process group instead, so a
    // valid `pg` spanning the job is just as good as a real MPI world here —
    // only refuse when NEITHER exists.
    if (ckpt_enabled && !dist::comm::dcp_supported(launch) && !ckpt_pg) {
        if (is_master) {
            std::cerr << "[WARN] checkpointing disabled: "
                      << dist::comm::dcp_unsupported_reason(launch) << std::endl;
        }
        ckpt_enabled = false;
    }

    if (ckpt_enabled && !ckpt_pg) {
        try {
            ckpt_pg = init_process_group(1, 0, /*stream=*/0, /*group_name=*/"dcp",
                                         dist::comm::MemoryMode::Standard);
        } catch (const std::exception& e) {
            ckpt_enabled = false;
            std::cerr << "[WARN] checkpointing disabled: could not create a "
                         "single-rank process group for DCP: " << e.what() << std::endl;
        }
        if (!ckpt_pg) ckpt_enabled = false;
    }

    const bool ckpt_resume = getenv_int("BLU_CKPT_LOAD", 0) != 0;
    if (ckpt_enabled) {
        // Rank 0 decides: per-rank scans race on a new dir and differ without a shared FS.
        const OwnTensor::dcp::DcpComm ckpt_comm(ckpt_pg);
        std::string run_dir;
        if (ckpt_comm.rank() == 0)
            run_dir = resolve_ckpt_run_dir(cfg.ckpt_dir, cfg.ckpt_prefix, ckpt_resume);
        run_dir = ckpt_comm.allgatherv_bytes(run_dir)[0];
        if (is_master) {
            std::error_code ec;
            const auto abs_dir = fs::absolute(run_dir, ec);
            std::cout << "[ckpt] base=" << cfg.ckpt_dir
                      << " dir=" << (ec ? fs::path(run_dir) : abs_dir).string()
                      << " mode=" << (ckpt_resume ? "RESUME" : "FRESH")
                      << " freq=" << cfg.ckpt_freq << " keep=" << cfg.ckpt_keep << std::endl;
        }
        cfg.ckpt_dir = run_dir;
    }

    auto ckpt_path_for = [&](int s) {
        return (fs::path(cfg.ckpt_dir) /
                (cfg.ckpt_prefix + "_step_" + std::to_string(s))).string();
    };
    // All "<prefix>_step_<N>" directories under ckpt_dir, as (step, path).
    auto scan_checkpoints = [&]() {
        std::vector<std::pair<int, std::string>> found;
        std::error_code ec;
        if (!fs::exists(cfg.ckpt_dir, ec)) return found;
        const std::string pre = cfg.ckpt_prefix + "_step_";
        for (const auto& e : fs::directory_iterator(cfg.ckpt_dir, ec)) {
            if (!e.is_directory()) continue;
            const std::string d = e.path().filename().string();
            if (d.size() <= pre.size() || d.compare(0, pre.size(), pre) != 0) continue;
            const std::string digits = d.substr(pre.size());
            if (digits.find_first_not_of("0123456789") != std::string::npos) continue;
            found.push_back({std::stoi(digits), e.path().string()});
        }
        std::sort(found.begin(), found.end());
        return found;
    };
    auto prune_old_checkpoints = [&]() {
        if (cfg.ckpt_keep <= 0) return;
        auto found = scan_checkpoints();
        if (found.size() <= static_cast<size_t>(cfg.ckpt_keep)) return;
        const size_t to_delete = found.size() - static_cast<size_t>(cfg.ckpt_keep);
        for (size_t i = 0; i < to_delete; ++i) {
            std::error_code rm_ec;
            fs::remove_all(found[i].second, rm_ec);
        }
    };

    StepWatchdog step_watchdog;
    if (getenv_int("BLU_STEP_WATCHDOG", 0) != 0) {
        step_watchdog.start(getenv_int("BLU_STEP_TIMEOUT_S", 600), rank);
        if (is_master)
            std::cout << "[watchdog] step watchdog on, deadline "
                      << getenv_int("BLU_STEP_TIMEOUT_S", 600) << " s (training.step_timeout_s)" << std::endl;
    }
    step_watchdog.beat("checkpoint load", -1);

    int start_step = 0;
    if (ckpt_enabled && ckpt_resume) {
        // Newest checkpoint that actually completed (is_valid == metadata.bin
        // present), so a run killed mid-save falls back to the previous one.
        int         latest_step = -1;
        std::string latest_path;
        const auto all_ckpts = scan_checkpoints();
        for (auto it = all_ckpts.rbegin(); it != all_ckpts.rend(); ++it) {
            // Collective-safe overload (see is_valid(path, comm) in
            // DistributedCheckpointing.h): metadata.bin only exists on rank
            // 0's local disk with no shared filesystem between nodes, so the
            // plain is_valid() would diverge per rank -- some ranks find
            // "nothing to resume" while rank 0 proceeds into `load` below, a
            // collective the other ranks never enter. Deadlock, not a
            // skipped resume.
            if (OwnTensor::DistCheckpoint::is_valid(it->second, OwnTensor::dcp::DcpComm(ckpt_pg))) {
                latest_step = it->first;
                latest_path = it->second;
                break;
            }
        }
        if (latest_step < 0) {
            if (is_master)
                std::cerr << "ERROR: checkpointing.load is true but there is no valid checkpoint in "
                          << cfg.ckpt_dir << ". Set checkpointing.load: false to start a fresh run."
                          << std::endl;
            dist::comm::abort_job(launch, 4);
        }
        if (latest_step >= 0) {
            float resume_loss  = 0.0f;
            int   resume_epoch = 0;
            // Retire in-flight refreshes first: the load overwrites the very
            // shard buffers they publish from.
            dcp_opt.dcp_before_load();
            // Bridge (DcpCompat.cpp) derives DCP's comm from ckpt_pg automatically --
            // DcpComm(ckpt_pg) when it's a real job-spanning group (BluRun's case),
            // MPI_COMM_WORLD only as the world_size==1 fallback. No comm to pass here.
            OwnTensor::DistCheckpoint::load(dcp_model, latest_path, ckpt_pg, &dcp_opt,
                                            &resume_epoch, &start_step, &resume_loss);
            // Cover the dtype-change path, where DCP rebinds its staging tensor
            // instead of writing through to the model's storage.
            dcp_model.sync_back_to_model();
            // Republish the restored FP32 shards into the live model tensors --
            // the step the stream-based load_state() used to do internally.
            dcp_opt.dcp_after_load();
            // The validation loss is only recomputed every val_freq steps, so a
            // resume that does not land on one would print the -1 sentinel until
            // the next validation. Restore the value the checkpoint carried.
            {
                float v = -1.0f;
                if (dcp_opt.get_extra("last_val_loss", v)) last_val_loss = v;
            }
            if (is_master) {
                std::cout << "[resume] loaded DCP checkpoint at step " << start_step
                          << " (loss " << std::fixed << std::setprecision(6) << resume_loss
                          << ") from " << latest_path << std::endl;
            }
            // Skip the batches already consumed by steps [0, start_step).
            //
            // PHASE-AWARE, and it has to be: skip_batches() fast-forwards by
            // REPLAYING pick_domain(), so replaying the whole span under one set
            // of weights would land the cursors somewhere a continuous run never
            // visited. Walk the schedule instead, installing each phase's
            // weights before replaying the steps that belong to it.
            //
            // The resumed phase is deliberately NOT pinned here: cur_mix_phase
            // stays -1 so the step loop re-applies it on the first iteration.
            // That costs one extra rebase, which is harmless — the pre-resume
            // span was on target and the post-resume span is on target from a
            // zeroed accounting, so the run total is too.
            // SFT: rows are used in file order, GRAD_ACCUM micro-batches per step, so skip exactly those.
            if (sft) sft_train->skip_micro_batches(static_cast<int64_t>(start_step) * GRAD_ACCUM);
            for (size_t i = 0; !sft && i < mix_schedule.size(); ++i) {
                const int ph_start = mix_schedule[i].start_step;
                const int ph_end   = (i + 1 < mix_schedule.size())
                                   ? mix_schedule[i + 1].start_step : cfg.max_steps;
                const int lo = std::max(0,          ph_start);
                const int hi = std::min(start_step, ph_end);
                if (hi <= lo) continue;
                train_loader->set_weights(mix_schedule[i].w, mix_schedule[i].name);
                train_loader->skip_batches(static_cast<size_t>(hi - lo) *
                                           static_cast<size_t>(GRAD_ACCUM));
            }
        }
    }

    if (start_step >= cfg.max_steps) {
        if (is_master)
            std::cerr << "ERROR: the checkpoint in " << cfg.ckpt_dir << " is at step " << start_step
                      << ", already at training.total_steps " << cfg.max_steps << "; nothing to train. "
                      << "Set checkpointing.load: false for a fresh run." << std::endl;
        dist::comm::abort_job(launch, 4);
    }

    if (is_master) {
        std::cout << "\nStarting training"
                  << (start_step > 0 ? (" (resumed at step " + std::to_string(start_step) + ")") : "")
                  << "...\n" << std::endl;
    }

    for (int step = start_step; step < cfg.max_steps; ++step) {
        step_watchdog.beat("step start", step);
        // ---------- data-mixture phase ----------
        // Applied before validation so the [val] line and the step's batches
        // agree on which phase they belong to. set_weights() rebases the
        // selection accounting; see its comment for why that is mandatory.
        //
        // The loader prefetches NBUF batches ahead, so up to NBUF of the
        // GRAD_ACCUM micro-batches at the transition step were already picked
        // under the old weights. At GRAD_ACCUM=32 that is a 4-batch smear at
        // one step boundary — not worth draining the pipeline to avoid.
        if (const int ph = phase_for(step); !sft && ph != cur_mix_phase) {
            cur_mix_phase = ph;
            if (is_master) {
                std::cout << "\n[mix] step " << step << " entering phase '"
                          << mix_schedule[ph].name << "'" << std::endl;
            }
            train_loader->set_weights(mix_schedule[ph].w, mix_schedule[ph].name);
        }

        // ---------- periodic validation ----------
        // SFT: the whole val set by the training rule -- the token losses of every batch and rank are
        // added up and divided once by the set's trained-token count (counted up front, all-reduced).
        if (sft && (step % cfg.val_freq == 0 || step == cfg.max_steps - 1)) {
            autograd::NoGradGuard no_grad;
            for (auto& opt : optimizers) opt->refresh_param_stage(0);   // why: see the branch below
            sft_val->reset();
            const int64_t vb = sft_val->micro_batches_per_epoch();
            const int64_t local_n = sft_val->valid_tokens_ahead(vb);
            int64_t n = local_n;
            if (world_size > 1)   // a count, exact in a double (< 2^53)
                n = static_cast<int64_t>(dist::comm::allreduce_scalar(launch, pg, static_cast<double>(local_n), ::sum));
            n /= g_cp.U;   // 2026-11-06: the U ranks of a CP group counted the same rows (exact; 1 = off)
            Tensor vden = Tensor::full(Shape{{1}}, TensorOptions().with_device(device), static_cast<float>(n));
            Tensor vsum = Tensor::zeros(Shape{{1}}, TensorOptions().with_device(device));
            for (int64_t v = 0; v < vb; ++v) {
                step_watchdog.beat("validation", step);
                Batch b = sft_val->next_batch();
                // 2026-11-06: context parallelism -- the plan for this rank's Nq/U heads over the FULL row, then this
                // rank's token slice of the positions, ids and labels (cp_slice is a no-op at U == 1). Previous lines:
                // g_packed = packed_rows_of(*sft_val, static_cast<int>(cfg.B), static_cast<int>(cfg.T),
                //                           static_cast<int>(cfg.q_heads));
                // Tensor logits = model.forward(b.input);
                // vsum += autograd::to_dtype(autograd::sparse_cross_entropy_loss(
                //             logits, b.target, SftDataLoader::kIgnore, &vden), Dtype::Float32);
                // 2026-11-13: + this rank's KV-head count (kv_heads / U), for the attention plan's mixed units.
                // Old call (kept, not deleted):
                // g_packed = packed_rows_of(*sft_val, static_cast<int>(cfg.B), static_cast<int>(cfg.T),
                //                           static_cast<int>(cfg.q_heads / g_cp.U));
                g_packed = packed_rows_of(*sft_val, static_cast<int>(cfg.B), static_cast<int>(cfg.T),
                                          static_cast<int>(cfg.q_heads / g_cp.U), static_cast<int>(cfg.kv_heads / g_cp.U));
                g_packed.positions = cp_slice(g_packed.positions);
                Tensor logits = model.forward(cp_slice(b.input));
                vsum += autograd::to_dtype(autograd::sparse_cross_entropy_loss(
                            logits, cp_slice(b.target), SftDataLoader::kIgnore, &vden), Dtype::Float32);
            }
            const float local = vsum.to_cpu().data<float>()[0];
            float total = local;
            if (world_size > 1) total = dist::comm::allreduce_scalar(launch, pg, local, ::sum);
            last_val_loss = total;
            if (is_master)
                std::cout << "  [val] step " << step << " | val_loss " << std::fixed << std::setprecision(6)
                          << total << " | " << n << " trained tokens" << std::endl;
        } else if (step % cfg.val_freq == 0 || step == cfg.max_steps - 1) {
            autograd::NoGradGuard no_grad;
            // The validation forward reads the model parameters and runs BEFORE
            // zero_grad(), so the refresh still in flight from the previous step
            // has to be installed. Install ONLY stage 0 here and let the
            // validation forward's own pre-hooks consume stages 1..N as it walks
            // the layers — exactly as the training forward does. (drain_param_
            // refresh() here would pull all N stages into the prologue and run
            // the broadcasts against an idle compute stream, silently costing the
            // overlap on every validation step.)
            //
            // Safe under NoGradGuard: Module::operator() dispatches forward
            // pre-hooks unconditionally; only the backward sentinel nodes are
            // gated on GradMode (NN.cpp). And model.forward() reaches the blocks
            // as (*attn[i])(x), so the hooks do fire. If val_loss_steps == 0 the
            // remaining stages simply fall through to zero_grad() + the training
            // forward, which is still correct.
            for (auto& opt : optimizers) opt->refresh_param_stage(0);

            // Each domain gets its own val_loss_steps batches — the domains are
            // separate distributions, so splitting one budget across them would
            // just make all three estimates noisier rather than saving anything
            // that matters at val_freq=250.
            std::vector<float> local_val(val_loaders.size(), 0.0f);
            for (size_t dvi = 0; dvi < val_loaders.size(); ++dvi) {
                val_loaders[dvi].second->reset();
                float val_accum = 0.0f;
                for (int v = 0; v < cfg.val_loss_steps; ++v) {
                    step_watchdog.beat("validation", step);
                    Batch b = val_loaders[dvi].second->next_batch();
                    g_packed = PackedRows{};   // pretraining validation shards: plain causal rows
                    Tensor logits = model.forward(b.input);
                    Tensor loss = autograd::sparse_cross_entropy_loss(logits, b.target);
                    // Same fp32 bridge as the training loop — data<float>() on a bf16
                    // scalar reads garbage without it. Under NoGradGuard this is a
                    // plain cast; on the default path it is a no-op.
                    val_accum += autograd::to_dtype(loss, Dtype::Float32)
                                     .to_cpu().data<float>()[0] / static_cast<float>(cfg.val_loss_steps);
                }
                local_val[dvi] = val_accum;
            }

            // One allreduce per domain (allreduce_scalar is launcher-agnostic). These
            // are collectives, so every rank runs them in the same order regardless
            // of is_master.
            std::vector<float> global_val_vec(local_val.size(), 0.0f);
            if (world_size > 1) {
                for (size_t dvi = 0; dvi < local_val.size(); ++dvi)
                    global_val_vec[dvi] = dist::comm::allreduce_scalar(launch, pg, local_val[dvi], ::sum)
                                          / static_cast<float>(world_size);
            } else {
                global_val_vec = local_val;
            }

            // Recombine at the fixed reference mixture (see val_blend_w above).
            // create_val_loaders() emits loaders in domain order but SKIPS any
            // domain with no val_data_root, so the indices line up with the
            // weight vector only while every domain declares one; the weights
            // are re-normalised over the loaders actually present so a missing
            // val set degrades to a sensible blend instead of a silent misweight.
            float wsum = 0.0f, blended = 0.0f;
            for (size_t dvi = 0; dvi < global_val_vec.size(); ++dvi) {
                const float w = val_blend_w[dvi];
                blended += w * global_val_vec[dvi];
                wsum    += w;
                domain_val_loss[dvi] = global_val_vec[dvi];
            }
            const float global_val = (wsum > 0.0f) ? blended / wsum : 0.0f;

            last_val_loss = global_val;
            if (is_master) {
                std::cout << "  [val] step " << step << " | val_loss "
                          << std::fixed << std::setprecision(6) << global_val;
                for (size_t dvi = 0; dvi < global_val_vec.size(); ++dvi) {
                    std::cout << " | " << val_loaders[dvi].first << " "
                              << global_val_vec[dvi];
                }
                std::cout << std::endl;
            }
        }

        // ---------- training step ----------
        auto t0 = std::chrono::high_resolution_clock::now();

        for (auto& opt : optimizers) opt->zero_grad();
        loss_accum_gpu *= 0.0f;

        micro_loss.clear();
        micro_where.clear();
        if (sft) {
            // The step's global trained-token count: this rank's next GRAD_ACCUM micro-batches (read from
            // valid_counts.bin, nothing consumed), summed over ranks. Every micro-step divides by it.
            const int64_t local = sft_train->valid_tokens_ahead(GRAD_ACCUM);
            step_tokens = local;
            if (world_size > 1)   // a count, exact in a double (< 2^53)
                step_tokens = static_cast<int64_t>(
                    dist::comm::allreduce_scalar(launch, pg, static_cast<double>(local), ::sum));
            step_tokens /= g_cp.U;   // 2026-11-06: the U ranks of a CP group counted the same rows (exact; 1 = off)
            sft_denom = Tensor::full(Shape{{1}}, TensorOptions().with_device(device),
                                     static_cast<float>(step_tokens));
        }
        for (int micro = 0; micro < GRAD_ACCUM; ++micro) {
            step_watchdog.beat("micro-step: data, forward, backward", step);
            Batch batch = sft ? sft_train->next_batch() : train_loader->next_batch();
            // 2026-11-06: context parallelism -- the attention plan is for this rank's Nq/U heads over the FULL row
            // (q_heads / 1 at U == 1, as before). The previous lines, kept for reference:
            // g_packed = sft ? packed_rows_of(*sft_train, static_cast<int>(cfg.B), static_cast<int>(cfg.T),
            //                                 static_cast<int>(cfg.q_heads))
            //                : PackedRows{};
            // 2026-11-13: + this rank's KV-head count (kv_heads / U), for the attention plan's mixed units.
            // Old call (kept, not deleted):
            // g_packed = sft ? packed_rows_of(*sft_train, static_cast<int>(cfg.B), static_cast<int>(cfg.T),
            //                                 static_cast<int>(cfg.q_heads / g_cp.U))
            //                : PackedRows{};
            g_packed = sft ? packed_rows_of(*sft_train, static_cast<int>(cfg.B), static_cast<int>(cfg.T),
                                            static_cast<int>(cfg.q_heads / g_cp.U), static_cast<int>(cfg.kv_heads / g_cp.U))
                           : PackedRows{};
            g_packed.positions = cp_slice(g_packed.positions);   // this rank's tokens' RoPE positions (no-op at U 1)

            // The trained-rows switch (2026-10-31, off by default): lm_head + loss on the batch's trained rows only;
            // the loss sum and its denominator (sft_denom) are the same tokens as the full path.
            model.head_rows = trained_rows ? &sft_train->last_trained_rows() : nullptr;
            // 2026-11-06: cp_slice = this rank's token slice of the ids and the labels under context parallelism; it
            // returns its argument unchanged at U == 1 (and outside SFT). The previous lines, kept for reference:
            // Tensor logits = model.forward(batch.input);
            // const Tensor& sft_target = trained_rows ? sft_train->last_trained_labels() : batch.target;
            Tensor logits = model.forward(cp_slice(batch.input));
            model.head_rows = nullptr;                    // validation and anything else: every row
            const Tensor sft_target = trained_rows ? sft_train->last_trained_labels() : cp_slice(batch.target);
            // The loss backward writes the gradient over `logits` (always, since 2026-10-21; bit-identical
            // gradient, -3.1 GiB at B 8 x T 4096 bf16). `logits` feeds only this loss and is not read after backward().
            Tensor loss_raw = sft ? autograd::sparse_cross_entropy_loss(logits, sft_target,
                                                                         SftDataLoader::kIgnore, &sft_denom)
                                  : autograd::sparse_cross_entropy_loss(logits, batch.target, -1, nullptr);

            // LossOps keys the loss scalar's dtype off the logits, so with
            // bf16_logits it comes back bf16. Two things then break unless it is
            // bridged to fp32 here:
            //
            //   * Every host read below goes through data<float>(), which is a
            //     bare reinterpret_cast with no dtype check — reading a bf16
            //     scalar as float yields garbage SILENTLY, and loss_accum_gpu is
            //     an fp32 tensor besides.
            //   * backward() must be seeded at the dtype LossBackward reads
            //     grad_output at, which is the LOGITS dtype. Seeding the fp32
            //     view is what makes that correct: ToDtypeBackward recasts the
            //     incoming fp32 grad back to bf16 before the CE node sees it.
            //
            // On the default path to_dtype returns its input unchanged, so
            // loss32 IS loss_raw — same tensor, same graph, no extra node.
            Tensor loss32 = autograd::to_dtype(loss_raw, Dtype::Float32);

            loss_accum_gpu += loss32.detach();                     // running sum for logging

            // Per-micro-batch loss (BLU_DIAG=1). The reported train loss is the mean
            // over GRAD_ACCUM micro-steps, so a single pathological micro-batch is
            // diluted 16x while its gradient dominates the accumulated norm almost
            // undiluted — which is exactly how a modest loss bump can sit next to a
            // 40x norm spike. Printing the 16 values separates "one bad micro-batch"
            // from "the whole step is worse", and `micro_where` names the shard and
            // token offset so the responsible bytes can be decoded offline.
            //
            // Costs a device sync per micro-step, so it is off by default. It does
            // NOT perturb the result: the loss is already fully determined by this
            // point and the run is data-deterministic.
            if (diag) {
                micro_loss.push_back(loss32.to_cpu().data<float>()[0]);
                micro_where.push_back({batch.domain_index, batch.shard_index, batch.token_offset});
            }
            // No DDP all-reduce here: gradients just accumulate locally across
            // micro-steps; ZeRO-1's reduce-scatter inside optimizer->step()
            // (below) does the cross-rank averaging once, after accumulation.
            //
            // Overlap: arm the bucketed reduce on the LAST micro-step only, so the
            // hooks fire on the fully-accumulated gradient and reduce it to owner
            // ranks concurrently with the rest of this backward. Earlier micro-
            // steps just accumulate (the hooks are inert). No-op when overlap is
            // disabled (world_size==1).
            if (micro == GRAD_ACCUM - 1) {
                if (pg) pg->begin_rs_step();
                for (auto& opt : optimizers) opt->arm_grad_sync();
            }
            loss32.backward(&grad_scale);                          // seed 1/GRAD_ACCUM, or W in SFT
        }

        // Loss for logging. Pretraining: the mean over the micro-steps, then over the ranks. SFT: each
        // micro loss is already divided by the step's global count, so micro-steps and ranks just add up
        // (dividing again would make it GRAD_ACCUM * world_size times too small).
        float loss_accum;
        step_watchdog.beat("loss sync + all-reduce (waits for this step's GPU work)", step);
        {
            float local = loss_accum_gpu.to_cpu().data<float>()[0] /
                          (sft ? 1.0f : static_cast<float>(GRAD_ACCUM));
            if (world_size > 1) {
                loss_accum = dist::comm::allreduce_scalar(launch, pg, local, ::sum);
                if (!sft) loss_accum /= world_size;
            } else {
                loss_accum = local;
            }
        }

        // Depth probe (see declarations above). Before opt->step().
        layer_gnorm.clear();
        gamma_absmax.clear();
        w_absmax.clear();
        if (diag) {
            for (int L : probe_layers) {
                if (L >= static_cast<int>(cfg.n_layers)) continue;
                layer_gnorm.emplace_back(L, grad_l2(model.attn[L]->wQ.weight));
                gamma_absmax.emplace_back(L, abs_max(model.attn[L]->q_gamma));
                // Weight magnitude. On a spiking step the model's loss decouples
                // from the input (see the [micro] correlation), which is what
                // saturated activations look like. A weight that has run away is
                // the cheapest cause to rule in or out, and this reuses a copy the
                // checksum already needs.
                w_absmax.emplace_back(L, abs_max(model.attn[L]->wQ.weight));
            }

            // Replica consistency (see `checksum` above). Checked HERE, after the
            // micro loop: the staged refresh is consumed lazily by the forward's
            // per-layer pre-hooks, so this is the first point in the step at which
            // every stage is guaranteed installed. wQ.weight of layer L lives in
            // refresh stage 1+L, so the probe layers sample 5 different stages.
            if (world_size > 1) {
                double local_ck = 0.0;
                for (int L : probe_layers) {
                    if (L >= static_cast<int>(cfg.n_layers)) continue;
                    local_ck += checksum(model.attn[L]->wQ.weight);
                }
                const double ck_min = dist::comm::allreduce_scalar(launch, pg, local_ck, op_min);
                const double ck_max = dist::comm::allreduce_scalar(launch, pg, local_ck, op_max);
                if (is_master && ck_min != ck_max) {
                    const double denom = std::max(std::fabs(ck_max), 1e-30);
                    std::cout << "  [replica] *** RANKS DIVERGED *** min " << std::scientific
                              << std::setprecision(9) << ck_min << " max " << ck_max
                              << " reldiff " << ((ck_max - ck_min) / denom) << std::endl;
                } else if (is_master) {
                    std::cout << "  [replica] ranks bit-identical (ck " << std::scientific
                              << std::setprecision(9) << ck_max << ")" << std::endl;
                }
            }
        }

        if (std::isnan(loss_accum) || std::isinf(loss_accum)) {
            if (is_master) std::cerr << "ERROR: NaN/Inf loss at step " << step << std::endl;
            if (is_master) log_file.close();
            dist::comm::abort_job(launch, 2);
        }

        step_watchdog.beat("optimizer step", step);
        // WSD LR schedule, then step each ZeRO-1 dtype group. Each
        // ShardedOptimizer reduce-scatters its own grad shard (bandwidth-optimal
        // cross-rank averaging — this is what replaced the DDP all-reduce),
        // applies decoupled weight decay + clips to cfg.grad_clip internally
        // using the exact global norm of its shard (all-reduced across ranks),
        // runs the fused Adam step on its FP32 master shard, then all-gathers the
        // updated params back into the live model tensors.
        //
        // Combined grad norm across groups is exact (not approximate): the
        // per-group norms are each already the true global L2 norm of that
        // group's gradients, and the groups are disjoint parameter sets, so
        // sqrt(sum of squares) reconstructs the true norm over all params.
        // (Each group clips independently against its own norm rather than this
        // combined one.)
        // ONE WSD curve, evaluated once per group with that group's own (peak,
        // floor). Identical in form to train_muon.py's two get_lr calls: the
        // warmup length, the stable window and the cosine tail are shared, so the
        // groups stay phase-aligned and only their scale differs. `lr` is the
        // AdamW-arm value, kept for the CSV/console log.
        const float lr = get_lr(step, cfg.max_lr, cfg.min_lr, cfg.warmup_steps,
                                cfg.max_steps, cfg.stable_steps);

        // ---- cross-group global gradient clipping (BLU_GLOBAL_CLIP=1) ----
        // Each ZeRO instance owns a DISJOINT subset of the model and computes the
        // exact global norm of that subset, so clipping inside step() clips each
        // group INDEPENDENTLY. DeepSpeed / Megatron / torch.clip_grad_norm_ clip on
        // the norm over ALL parameters instead.
        //
        // CORRECTION (measured, 2 x 2000-step runs — see docs/ZERO1_CLIPPING_AUDIT.md).
        // This comment used to claim the FP32 group's norm runs ~29x the body's and
        // that global clipping therefore changes the body's effective step size by
        // ~2.6x. Both halves are wrong:
        //   * The ratio runs the OTHER way. G0 (body) / G1 (fp32) measured 4.4 at
        //     steps 0-199, 2.4 at 800-999, 16.2 at 1400-1599, 17.2 at 1800-1999.
        //     The body's norm is always the larger one.
        //   * There is no 2.6x step-size difference. AdamW is scale-invariant to a
        //     CONSTANT gradient factor; replaying the real coefficient series through
        //     an Adam update puts the distortion from its step-to-step VARIATION at
        //     ~5% RMS, not 160%.
        // And in this partition the body is not clipped at all (ShardedMuon skips
        // every use_muon record), so per-group vs global is a distinction without a
        // difference on 94% of the model: an A/B over 104 matched steps differs by
        // 0.005 mean loss, inside the 0.019 run-to-run noise floor.
        //
        // Phase 1 reduces every group's gradients and collects its ||g||^2; the
        // groups are disjoint so the squares simply add. Phase 2 hands the single
        // coefficient down and steps. step() detects that the reduction already
        // happened and does not repeat it.
        //
        // This also removes the head-of-line blocking `step_order` works around:
        // every norm all-reduce now happens in phase 1, before any group issues its
        // staged parameter transfers.
        //
        // Off by default — it changes training semantics, so it is opt-in and
        // A/B-able rather than a silent change.
        const bool global_clip = getenv_int("BLU_GLOBAL_CLIP", 0) != 0;
        float external_coef = -1.0f;
        if (global_clip && cfg.grad_clip > 0.0f) {
            float nsq_total = 0.0f;   // every group: the reported diagnostic norm
            float nsq_clip  = 0.0f;   // groups that actually APPLY the coefficient
            bool  skip      = false;
            for (size_t oi : step_order) {
                const float nsq = optimizers[oi]->reduce_and_measure();
                if (nsq >= 0.0f && optimizers[oi]->clip_covered_numel() > 0)
                    nsq_clip += nsq;
                // NOTE the test is `!(nsq >= 0)`, NOT `nsq < 0`. They differ on NaN:
                // `NaN < 0` is FALSE, so a NaN group norm used to slip through, make
                // nsq_total NaN, and then hit std::min(1.0f, NaN) — which returns
                // 1.0f (std::min is `b < a ? b : a`, and `NaN < 1.0f` is false).
                // The result was a NaN gradient with NO clip applied, written
                // straight into the parameters. reduce_and_measure() now returns -1
                // for any non-finite norm (scaler or not); this test catches both
                // that and a NaN that arrives some other way.
                if (!(nsq >= 0.0f)) skip = true;
                else                nsq_total += nsq;
            }
            // WHICH norm the coefficient is derived from.
            //
            // "Global" clipping means one coefficient over all parameters — correct
            // when the coefficient is then APPLIED to all parameters. Here it is
            // not: ShardedMuon withholds the scale from every orthogonalized tensor,
            // which in this partition is the whole body. Deriving the coefficient
            // from the full norm and applying it only to the embedding group means
            // the embedding is rescaled by a factor set by gradients it has no
            // relationship to. Measured over both 2000-step runs, the AdamW group
            // is only 7-11% of the total ||g||^2, so the coefficient tracked the
            // body: it drifted 0.52 -> 0.18 across the stable phase purely because
            // the body's norm grew from 2.0 to 4.5, damping the embedding harder and
            // harder for reasons internal to the body.
            //
            // Default: derive from the norm of the COVERED subset, so the clip is a
            // real clip of the thing it scales. The full norm is still what gets
            // reported. BLU_GLOBAL_CLIP_STRICT=1 restores the previous
            // full-norm-derived coefficient for A/B (and is the correct setting if
            // the Muon exemption is ever removed, since then coverage is 100%).
            const bool strict = getenv_int("BLU_GLOBAL_CLIP_STRICT", 0) != 0;
            const float total     = std::sqrt(nsq_total);
            const float clip_norm = strict ? total : std::sqrt(nsq_clip);
            if (!skip && std::isfinite(clip_norm) && std::isfinite(total)) {
                external_coef = std::min(1.0f, cfg.grad_clip / (clip_norm + 1e-6f));
                for (auto& opt : optimizers) opt->set_external_clip_coef(external_coef);
            } else if (is_master) {
                std::cerr << "WARNING: non-finite gradient norm at step " << step
                          << " — skipping the global clip; groups fall back to their "
                             "own per-group coefficient this step" << std::endl;
            }
        }

        float grad_norm_sq = 0.0f;
        std::vector<float> group_norm(optimizers.size(), 0.0f);
        for (size_t oi : step_order) {          // biggest group last — see step_order
            auto& opt = optimizers[oi];
            opt->set_lr(get_lr(step, group_lr[oi].first, group_lr[oi].second,
                               cfg.warmup_steps, cfg.max_steps, cfg.stable_steps));
            // F1, trainer side.
            // What: catch the optimizer's ZeroNonFiniteGradError and abort with one clear line
            //       and exit code 4 (the NaN-LOSS check earlier in this loop uses 2).
            // Which optimizers raise it: all three -- ShardedOptimizer (v1, the default),
            //   SymmetricShardedOptimizer (BLU_ZERO_V2=1) and ShardedMuon (BLU_OPTIMIZER=muon). This
            //   loop steps every group, so this one catch covers all of them.
            // Why the check is NOT here in the trainer but in each optimizer's step():
            //   - the loss check (NaN/Inf loss, earlier in this loop) runs before step() and only sees
            //     the loss; a finite loss can still carry a NaN/Inf gradient;
            //   - step() already has the all-reduced gradient norm (needed for clipping), so the
            //     check there is one isfinite() per group per step -- no GPU work, no memory, no
            //     change to finite steps -- and it fires BEFORE any weight update;
            //   - a per-element check inside the Adam kernel could not stop the step as a whole.
            //   Old path, for the record: NaN norm -> clip coef 1.0 (GradNormKernels.cu:190-193)
            //   -> scale skipped (MultiTensorKernels.cu:235-240) -> Adam wrote NaN into master/m/v
            //   (MultiTensorKernels.cu:321-334) -> run died one step later, and a checkpoint taken
            //   on that step would have saved the NaN state (auto-resume would reload it).
            // Why catch at all: without it the program still stops before the update and before
            //   the save (safe), but via std::terminate + MPI signal backtraces on every rank.
            try {
                opt->step();
            } catch (const owt::zero::ZeroNonFiniteGradError& e) {
                // step() raises this BEFORE touching master/m/v, and the checkpoint
                // save below is never reached, so no poisoned state is persisted.
                std::cerr << "[FATAL] rank " << rank << " step " << step << ": " << e.what()
                          << " -- stopped before the weight update; no checkpoint written"
                          << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 4);
            }
            const float g = opt->last_grad_norm();
            group_norm[oi] = g;
            grad_norm_sq += g * g;
        }
        const float grad_norm = std::sqrt(grad_norm_sq);

        // Timing / pipeline boundary.
        //
        // With the staged parameter refresh ON we deliberately do NOT synchronize
        // here. A cudaDeviceSynchronize() drains the compute stream and makes the
        // host wait for every kernel of this step before it may enqueue anything
        // else — so the next forward is not launched until the GPU has gone
        // completely idle. The refresh broadcasts, issued on the comm stream at
        // the end of step(), would then run to completion inside that idle window
        // and overlap nothing. Not a correctness bug, but it makes the staged
        // refresh pointless, because there is never any compute in flight for it
        // to hide behind.
        //
        // Dropping the sync lets the host run ahead and enqueue the next forward
        // while this step's Adam/cast tail is still executing, so the broadcasts
        // for stages 1..N transfer underneath layer 0's attention compute. Stage 0
        // (wte) stays exposed by construction: the embedding reads it first.
        //
        // dt then measures HOST wall-clock per iteration rather than drained-GPU
        // time. In steady state these agree: the GPU is near-saturated, so the
        // launch queue back-pressures cudaLaunchKernel and the host is throttled
        // to the GPU's rate. Only the first couple of steps read low while the
        // pipeline fills. The non-overlap path keeps its full device sync.
        //
        // With the staged refresh off (the current default) this is REDUNDANT, not
        // load-bearing: the synchronous publish inside step() already ends in a
        // cudaDeviceSynchronize(). It is kept because it also covers a step that
        // returned early (gradient overflow) and published nothing, and because it
        // is what restores drained-GPU timing for dt.
        if (!zcfg.overlap_param_refresh) cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        const double dt = std::chrono::duration<double>(t1 - t0).count();

        // 2026-11-06: world_size / g_cp.U -- a CP group's U ranks share their rows (1 = off). Previous line, kept:
        // const int64_t tokens = cfg.B * cfg.T * GRAD_ACCUM * static_cast<int64_t>(world_size);
        const int64_t tokens = cfg.B * cfg.T * GRAD_ACCUM * static_cast<int64_t>(world_size / g_cp.U);
        const double tok_per_sec = static_cast<double>(tokens) / dt;
        const double elapsed_min =
            std::chrono::duration<double>(t1 - training_start).count() / 60.0;

        // ---- MFU for this step ----
        // Uses the SAME dt as tok/s, so the caveat above applies here too: with
        // overlap_param_refresh on there is no device sync, and dt is host
        // wall-clock. In steady state the launch queue back-pressures the host
        // to the GPU's rate so the two agree, but the first few steps read fast
        // while the pipeline fills — which is exactly why observe() drops them
        // from the steady-state figure (BLU_MFU_WARMUP, default 5).
        const double step_mfu    = mfu_meter.mfu(dt, tokens);
        const double step_mfu_p  = mfu_meter.mfu_palm(dt, tokens);
        const double step_tflops = mfu_meter.tflops_per_gpu(dt, tokens);
        mfu_meter.observe(dt, tokens);

        // ---------- per-step report (every step) ----------
        if (is_master) {
            std::cout << "step " << std::setw(6) << step
                      << " | train " << std::fixed << std::setprecision(6) << loss_accum
                      << " | val "   << std::fixed << std::setprecision(6) << last_val_loss
                      << " | tok/s " << std::fixed << std::setprecision(2) << tok_per_sec
                      << " | norm "  << std::fixed << std::setprecision(4) << grad_norm
                      << " | dt "    << std::fixed << std::setprecision(2) << (dt * 1000.0) << " ms"
                      << " | mfu "   << std::fixed << std::setprecision(2) << (step_mfu * 100.0) << "%"
                      << " | TF/gpu " << std::fixed << std::setprecision(1) << step_tflops
                      << " | lr "    << std::scientific << std::setprecision(4) << lr;
            // Per-group norms: which dtype group actually spiked.
            for (size_t gi = 0; gi < group_norm.size(); ++gi)
                std::cout << " | G" << gi << "(" << group_label[gi] << ") "
                          << std::fixed << std::setprecision(4) << group_norm[gi];
            if (external_coef >= 0.0f)
                std::cout << " | globalclip " << std::fixed << std::setprecision(4)
                          << external_coef;
            // DIAGNOSTIC ONLY: Newton-Schulz health on the Muon path. `nsrej` is
            // how many owned tensors the divergence guard rejected THIS step,
            // `nsratio` the worst ||update||_F / sqrt(min(rows,cols)) among the
            // updates that passed it (healthy ~1.0). The guard only trips above
            // element magnitude 1000, so an update can be far too large and
            // still be applied — that shows up here and nowhere else.
            for (auto& opt : optimizers) {
                if (auto* sm = dynamic_cast<ShardedMuon*>(opt.get())) {
                    std::cout << " | nsrej " << sm->ns_rejected_last_step()
                              << "/" << sm->rejected_ns_updates()
                              << " | nsratio " << std::fixed << std::setprecision(3)
                              << sm->ns_max_update_ratio();
                    break;
                }
            }
            if (sft) std::cout << " | trained " << step_tokens;   // this step's global count N
            std::cout << std::endl;

            if (diag && !micro_loss.empty()) {
                // Flag micro-batches that stand out against this step's own median,
                // so the line is readable at a glance instead of 16 bare floats.
                std::vector<float> sorted = micro_loss;
                std::sort(sorted.begin(), sorted.end());
                const float med = sorted[sorted.size() / 2];
                std::cout << "  [micro] med " << std::fixed << std::setprecision(4) << med << " |";
                for (size_t m = 0; m < micro_loss.size(); ++m) {
                    const bool hot = micro_loss[m] > med + 0.5f;
                    std::cout << " " << (hot ? "*" : "") << std::setprecision(3) << micro_loss[m];
                }
                std::cout << std::endl;
                for (size_t m = 0; m < micro_loss.size(); ++m) {
                    if (micro_loss[m] > med + 0.5f)
                        std::cout << "  [micro] OUTLIER m=" << m
                                  << " loss " << std::setprecision(4) << micro_loss[m]
                                  << " (median " << med << ") from domain "
                                  << (micro_where[m].domain >= 0
                                          ? train_loader->name_of(micro_where[m].domain)
                                          : std::string("<none>"))
                                  << " shard " << micro_where[m].shard
                                  << " token offset " << micro_where[m].offset << std::endl;
                }
            }

            if (diag && !layer_gnorm.empty()) {
                std::cout << "  [depth] ||g(wQ)|| by layer:";
                for (auto& e : layer_gnorm)
                    std::cout << " L" << e.first << " " << std::scientific
                              << std::setprecision(3) << e.second;
                std::cout << " | max|q_gamma|:";
                for (auto& e : gamma_absmax)
                    std::cout << " L" << e.first << " " << std::fixed
                              << std::setprecision(4) << e.second;
                std::cout << " | max|wQ|:";
                for (auto& e : w_absmax)
                    std::cout << " L" << e.first << " " << std::fixed
                              << std::setprecision(4) << e.second;
                std::cout << std::endl;
            }

            log_file << step << ","
                     << loss_accum   << ","
                     << last_val_loss << ","
                     << tok_per_sec  << ","
                     << grad_norm    << ","
                     << (dt * 1000.0) << ","
                     << lr           << ","
                     << elapsed_min  << ","
                     << step_mfu     << ","
                     << step_mfu_p   << ","
                     << step_tflops;
            for (float dv : domain_val_loss) log_file << "," << dv;
            const double run_secs = elapsed_min * 60.0;
            log_file << "," << static_cast<int64_t>(step + 1) * tokens << ","
                     << (run_secs > 0 ? static_cast<double>(step + 1 - start_step) * tokens / run_secs : 0.0);
            if (train_loader)
                for (int d = 0; d < train_loader->n_domains(); ++d) log_file << "," << train_loader->current_shard_of(d);
            if (sft) log_file << "," << step_tokens;
            log_file << "\n";
            log_file.flush();
        }

        // ---------- periodic checkpoint (Distributed Checkpointing) ----------
        // Labeled step+1 (the next iteration to run) so the resume path picks up
        // exactly where this save left off — no duplicated or skipped iteration.
        //
        // COLLECTIVE, and deliberately NOT master-gated. Under ZeRO-1 each rank
        // owns a DIFFERENT slice of the optimizer state — a flat elementwise
        // slice for the AdamW groups, a disjoint set of whole tensors for the
        // Muon groups — so a master-only save would persist only one rank's
        // piece of either and every rank would then resume from rank 0's slice.
        // Here every rank writes its own shard into the step directory and DCP
        // merges the metadata.
        if (ckpt_enabled &&
            ((step + 1) % cfg.ckpt_freq == 0 || step == cfg.max_steps - 1)) {
            step_watchdog.beat("checkpoint save", step);
            try {
                // Install any refresh still in flight so the checkpoint holds
                // this step's weights rather than the previous step's.
                for (auto& opt : optimizers) opt->drain_param_refresh();
                // Carry the most recent validation loss into the checkpoint so a
                // resume reports it instead of the -1 sentinel.
                dcp_opt.set_extra("last_val_loss", last_val_loss);
                OwnTensor::DistCheckpoint::save(dcp_model, ckpt_path_for(step + 1),
                                                ckpt_pg, &dcp_opt, /*epoch=*/0,
                                                step + 1, loss_accum);
                if (is_master) prune_old_checkpoints();
            } catch (const std::exception& e) {
                std::cerr << "[WARN] checkpoint save at step " << (step + 1)
                          << " failed: " << e.what() << std::endl;
            }
        }

        // Reclaim the extra allocator high-water mark from step 0's one-time
        // warmup allocations (ZeRO shard buffers, autograd graph caches, the
        // step-0 validation pass, etc.).
        // 2026-11-04: at step 0 only (was step 0 and step 5). Measured on the 135M
        // SFT model (P032, work doc 45.145): the step-5 call freed nothing useful
        // (reserved_peak identical to the MiB: 21,558 MiB at B=4, 71,158 vs 71,156
        // at B=20) and cost one slow refill step after it (~6 s at B=4, ~26 s at
        // B=20); by step 5 the cache holds exactly what step 6 needs again. The
        // step-0 call is kept: without it, at B=20 the run held 3.4 GB more and
        // split cached blocks 76% more often. The allocator also trims its cache by
        // itself on an out-of-memory error, so neither call is needed to avoid OOM.
        // The old version (step 0 and step 5), kept for reference; to bring the step-5 call back:
        // if (step == 0 || step == 5) {
        //     CachingCUDAAllocator::instance().empty_cache();
        // }
        if (step == 0) {
            CachingCUDAAllocator::instance().empty_cache();
        }

        if (step == 5) {
            AllocationTracker::instance().shutdown();
        }
    }

    // The final step leaves its parameter refresh in flight (nothing consumed it,
    // since there is no next forward). Install it before teardown so the process
    // group is not destroyed with collectives outstanding.
    for (auto& opt : optimizers) opt->drain_param_refresh();
    cudaDeviceSynchronize();

    // BLU_EXPORT=<path>.safetensors: the final weights, fp32, BluTrain names (from preproduction-135M f37c616).
    if (const char* ex = std::getenv("BLU_EXPORT"); ex && *ex && is_master) {
        try {
            export_safetensors(model, ex, cfg.max_steps);
        } catch (const std::exception& e) {
            std::cerr << "[WARN] weight export failed: " << e.what() << std::endl;
        }
    }

    // Packed rows: g_packed is a global made before main, so it is destroyed AFTER the caching allocator (a static
    // made later), and freeing its two GPU tensors then hangs the process at exit (2026-10-15, e2e packed run:
    // "Training complete", then a futex wait with 57.8 GB still held; the v1 run, whose g_packed was empty, exited).
    // Free them here, while the allocator is alive.
    g_packed = PackedRows{};

    // Flush any in-flight async checkpoint before teardown (no-op for sync saves).
    // if (cfg.checkpointing) {
    //     try { ckpt.wait_for_completion(); }
    //     catch (const std::exception& e) {
    //         if (is_master) std::cerr << "[WARN] checkpoint flush at shutdown: " << e.what() << std::endl;
    //     }
    // }

    if (is_master) {
        log_file.close();

        // ---- end-of-run MFU summary ----
        // The steady-state figure is the one to quote: total model FLOPs over
        // total wall time across every step past the warmup window. It is an
        // aggregate rather than a mean of per-step ratios, so a stalled step
        // costs exactly its own wall clock instead of skewing an average, and
        // it is the same quantity a paper means by "we sustained X% MFU".
        // `best` is the ceiling this configuration touched at least once — the
        // gap between best and steady is jitter (checkpoints, validation,
        // straggler ranks), not a kernel problem.
        const double ss = mfu_meter.steady_mfu();
        std::cout << "\n==================== MFU summary ====================\n"
                  << std::fixed << std::setprecision(2)
                  << "  steady-state MFU  : " << (ss * 100.0) << "%"
                  << "   (" << mfu_meter.ss_steps << " steps, first "
                  << mfu_meter.warmup_steps << " excluded)\n"
                  << "  best-step MFU     : " << (mfu_meter.best_mfu * 100.0) << "%\n"
                  << "  model TFLOP/s/gpu : "
                  << (ss * mfu_meter.peak_flops_per_gpu / 1e12)
                  << "  of " << (mfu_meter.peak_flops_per_gpu / 1e12) << " peak\n"
                  << "  cluster PFLOP/s   : "
                  << (ss * mfu_meter.cluster_peak() / 1e15) << "\n"
                  << std::setprecision(0)
                  << "  steady tok/s      : " << mfu_meter.steady_tok_per_sec() << "\n"
                  << std::setprecision(3)
                  << "  total model FLOPs : "
                  << (mfu_meter.flops_per_token_causal * mfu_meter.ss_tokens / 1e18)
                  << " EFLOP over " << (mfu_meter.ss_secs / 60.0) << " min\n"
                  << "=====================================================" << std::endl;

        // What the mixture ACTUALLY delivered, next to what was asked for.
        // Worth reading even though the selection rule is exact: it also
        // reports how many times each domain wrapped its shard list, which is
        // the number that says whether the low-weight domains had enough data
        // or were repeated many times over.
        if (train_loader) train_loader->print_mixture_stats();

        std::cout << "\n=== Training complete ===" << std::endl;
        // Allocator memory stats, printed once at the end (no cost during training):
        //   active_peak = live peak (most memory in use at once); reserved_peak = most memory
        //   held from the GPU (what nvidia-smi shows, minus the CUDA context).
        { auto st = CachingCUDAAllocator::instance().get_stats(); std::cout << "[MEM] active_peak " << st.active_peak/1048576 << " MiB | allocated_peak " << st.allocated_peak/1048576 << " MiB | reserved_peak " << st.reserved_peak/1048576 << " MiB | cached_free " << st.cached_free_bytes/1048576 << " MiB" << std::endl; }
        { auto st = CachingCUDAAllocator::instance().get_stats(); std::cout << "[ALLOC] allocs " << st.num_allocs << " | frees " << st.num_frees << " | cache_hits " << st.num_cache_hits << " | cache_misses " << st.num_cache_misses << " | cuda_mallocs " << st.num_cuda_mallocs << " | splits " << st.num_splits << " | merges " << st.num_merges << std::endl; }
    }

    step_watchdog.stop();
    if (is_master && !run_dir.empty()) std::ofstream(fs::path(run_dir) / ".finished") << "ok\n";
    dist::comm::finalize_launch();
    return 0;
}