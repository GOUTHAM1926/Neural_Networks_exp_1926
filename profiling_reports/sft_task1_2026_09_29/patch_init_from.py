#!/usr/bin/env python3
"""SFT task 1: BLU_INIT_FROM=<checkpoint dir> -- load ONLY the model weights of another run's DCP checkpoint.

Edits bluscript_zero_ckpt.cpp (BluTrain, dist_side_merge) in 3 places:
  1. includes: Metadata.h (CheckpointMetadata) and Checksums.h (CRC32 check);
  2. right after "Number of parameters", i.e. BEFORE the optimizers are built: the weights-only load
     (BLU_INIT_VERIFY=1 adds a CRC32 check of every loaded weight, for testing);
  3. the checkpointing block reuses the single-rank DCP group created in (2) instead of making a second one.

Usage (from the BluTrain directory):
  python3 patch_init_from.py --check    # verify every anchor matches exactly once, write nothing
  python3 patch_init_from.py --apply    # apply; the original is kept as bluscript_zero_ckpt.cpp.orig_init
"""
import hashlib
import os
import sys

FILE = "bluscript_zero_ckpt.cpp"
EXPECTED_MD5 = "0a3ca9e4cd33caaa9a5d61a69578b52b"   # dist_side_merge 909ab75

A1 = '#include "ZeroDcpBridge.h"              // ZeroDcpModule, ZeroDcpOptimizer\n'
N1 = A1 + (
    '#include "Metadata.h"                   // CheckpointMetadata (BLU_INIT_VERIFY)\n'
    '#include "Checksums.h"                  // CRC32 check of the weights BLU_INIT_FROM loaded\n'
)

A2 = '    if (is_master) std::cout << "Number of parameters: " << num_params << std::endl;\n'
N2 = A2 + r'''
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
    std::shared_ptr<ProcessGroupNCCL> init_pg;   // single-rank DCP group at W == 1, reused below
    if (const char* init_from = std::getenv("BLU_INIT_FROM"); init_from && *init_from) {
        const std::string init_dir = init_from;
        auto pname = [](size_t i) {
            char nm[32];
            std::snprintf(nm, sizeof nm, "param_%06zu", i);
            return std::string(nm);
        };
        std::shared_ptr<ProcessGroupNCCL> load_pg = pg;
        if (!load_pg) {
            init_pg = init_process_group(1, 0, /*stream=*/0, /*group_name=*/"dcp",
                                         dist::comm::MemoryMode::Standard);
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
'''

A3 = '    std::shared_ptr<ProcessGroupNCCL> ckpt_pg = pg;\n'
N3 = ('    // At W == 1, reuse the single-rank group BLU_INIT_FROM created above (if any) instead of\n'
      '    // creating a second group with the same name.\n'
      '    std::shared_ptr<ProcessGroupNCCL> ckpt_pg = pg ? pg : init_pg;\n')

EDITS = [("includes", A1, N1), ("weights-only load", A2, N2), ("checkpoint group reuse", A3, N3)]


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode not in ("--check", "--apply"):
        sys.exit(__doc__)
    if not os.path.exists(FILE):
        sys.exit(f"ABORT: {FILE} not found; run from the BluTrain directory")
    src = open(FILE, encoding="utf-8").read()
    md5 = hashlib.md5(src.encode("utf-8")).hexdigest()
    print(f"{FILE} md5 {md5} ({'matches' if md5 == EXPECTED_MD5 else 'DIFFERS FROM'} dist_side_merge 909ab75)")
    if "BLU_INIT_FROM" in src:
        sys.exit("ABORT: BLU_INIT_FROM is already in the file; nothing written")
    out = src
    for label, anchor, new in EDITS:
        n = out.count(anchor)
        print(f"  {label}: anchor found {n} time(s)")
        if n != 1:
            sys.exit(f"ABORT: '{label}' anchor must match exactly once; nothing written")
        out = out.replace(anchor, new)
    if mode == "--check":
        print("check OK: all 3 anchors match once. Nothing written.")
        return
    open(FILE + ".orig_init", "w", encoding="utf-8").write(src)
    open(FILE, "w", encoding="utf-8").write(out)
    new_md5 = hashlib.md5(out.encode("utf-8")).hexdigest()
    print(f"applied: 3 edits. Backup {FILE}.orig_init. New md5 {new_md5}")


if __name__ == "__main__":
    main()
