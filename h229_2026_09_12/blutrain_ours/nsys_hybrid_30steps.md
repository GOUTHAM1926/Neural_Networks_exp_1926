
[OK] libprofiler is up to date

[BUILD] BluBridge-BLAS (sm_90)
/usr/bin/make -C Tensor-Implementations blas SM_ARCH=90
make[1]: Entering directory '/root/gautam_k/blublas/blutrain_ours/Tensor-Implementations'
\n\nlibblublas is up-to-date.
make[1]: Leaving directory '/root/gautam_k/blublas/blutrain_ours/Tensor-Implementations'

[BUILD] Tensor-Implementations (sm_90, backend=BluBridge-BLAS (libblublas.so))
/usr/bin/make -C Tensor-Implementations tensor WITH_BLUBLAS=1 SM_ARCH=90
make[1]: Entering directory '/root/gautam_k/blublas/blutrain_ours/Tensor-Implementations'
\n\nlibtensor is up-to-date.
make[1]: Leaving directory '/root/gautam_k/blublas/blutrain_ours/Tensor-Implementations'

[BUILD] dist (sm_90)
/usr/bin/make -C dist all SM_ARCH=90
make[1]: Entering directory '/root/gautam_k/blublas/blutrain_ours/dist'
\n[SUCCESS] dist build complete (sm_90):
   /root/gautam_k/blublas/blutrain_ours/dist/lib/libdist.a
   /root/gautam_k/blublas/blutrain_ours/dist/lib/libdist.so
make[1]: Leaving directory '/root/gautam_k/blublas/blutrain_ours/dist'

[SUCCESS] Full build complete.
  GPU arch:      sm_90
  BLAS:          BluBridge-BLAS (libblublas.so)
  Profiler libs: Profiler/lib/libprofiler.{a,so}
  Tensor libs:   Tensor-Implementations/lib/libtensor.{a,so}
  BluBLAS lib:   Tensor-Implementations/lib/libblublas.so
  Dist libs:     dist/lib/libdist.{a,so}

[COMPILE] bluscript_zero.cpp (BLAS: BluBridge-BLAS (libblublas.so))
/usr/bin/mpic++ -std=c++2a -fPIC -O3 -g -DWITH_CUDA -fopenmp -DUSE_FUSED_RMS_ROPE=1 -ITensor-Implementations/include -IProfiler/include -I/usr/local/cuda/include -Idist -Idist/FSDP -Idist/Tensor-Parallelism -Idist/Tensor-Parallelism/tensor -Idist/Tensor-Parallelism/dnn -Idist/Tensor-Parallelism/bridge -Idist/Tensor-Parallelism/memory -Idist/Data-Parallel/include -Idist/communication/include -Idist/zero/include -Idist/Distributed_Checkpointing/include -I/opt/nvidia/nsight-compute/2025.3.1/host/target-linux-x64/nvtx/include -o _runner_ bluscript_zero.cpp dist/FSDP/fsdp_init.cpp dist/FSDP/fsdp_comm.cpp dist/FSDP/fsdp_ddp.cpp dist/zero/src/ShardedOptimizer.cpp dist/zero/src/SymmetricShardedOptimizer.cpp -LTensor-Implementations/lib -LProfiler/lib -Ldist/lib -L/usr/local/cuda/lib64 -L/usr/lib/x86_64-linux-gnu -Wl,-rpath,/root/gautam_k/blublas/blutrain_ours/Tensor-Implementations/lib -Wl,-rpath,/root/gautam_k/blublas/blutrain_ours/Profiler/lib -Wl,-rpath,/root/gautam_k/blublas/blutrain_ours/dist/lib -Wl,-rpath,/usr/lib/x86_64-linux-gnu -ldist -ltensor -lprofiler -lblublas -lcuda -lcudart -lcublas -lcurand -lnccl -lmpi -lgomp -lpthread -ldl -lz
[RUN] mpirun -np 2 bluscript_zero.cpp

[bench] BLU_MAX_STEPS=30 (warmup=3 stable=15)
================ Configuration ================
  d_model         : 3072
  n_layers        : 36
  q_heads/kv_heads: 24 / 8  (G=3)
  head_dim        : 128
  ffn_hidden      : 8192  (SwiGLU)
  context_length  : 4096
  rope_theta      : 500000  (interleaved=NeoX)
  vocab_size      : 50304
  weight_tying    : OFF
  B x T           : 2 x 4096
  world_size      : 2
  global_batch    : 524288  (grad_accum=32)
  max_lr / min_lr : 0.0003 / 3e-05
  warmup / steps  : 3 / 30
  WSD stable_steps: 15  (decay tail=12)
===============================================
[watchdog] collective no-progress deadline 600 s, poll 2 s (BLU_NCCL_WATCHDOG=0 to disable)
Initializing model on CUDA device 0...
Number of parameters: 3933179904
MFU basis: 29.03 GFLOP/token | NVIDIA H200 x2 bf16 peak 1979.00 TFLOP/s
[ZeRO-1] variant: v2 reduce-scatter/all-gather | symmetric memory: requested  (BLU_ZERO_V2=1 BLU_SYMMETRIC=1)
[ZeRO-1 v2] build=v2.3  supports: rs+ag | can-symmetric-window | can-fp32-reduce | can-reshard-v1-ckpt
[ZeRO-1 v2] ACTIVE: reduce_dtype=bfloat16 (= param dtype)  symmetric=ON  overlap=bwd/refresh
[ZeRO-1 v2] zero1-v2 reduce-scatter/all-gather (symmetric/NVLS)
[ZeRO-1 v2] params=3778412544 padded=3778412544 W=2 shard=1889206272 | 145 buckets of 50.00 MB (13107200 elems/rank/bucket) | optimizer state/GPU: 21620.25 MB (vs 43240.50 MB unsharded)
[ZeRO-1 v2] build=v2.3  supports: rs+ag | can-symmetric-window | can-fp32-reduce | can-reshard-v1-ckpt
[ZeRO-1 v2] ACTIVE: reduce_dtype=float32 (= param dtype)  symmetric=ON  overlap=bwd/refresh
[ZeRO-1 v2] zero1-v2 reduce-scatter/all-gather (symmetric/NVLS)
[ZeRO-1 v2] params=154767360 padded=154767360 W=2 shard=77383680 | 12 buckets of 50.00 MB (6553600 elems/rank/bucket) | optimizer state/GPU: 885.59 MB (vs 1771.17 MB unsharded)
ZeRO-1 staged refresh: 37 stages (1 layer(s)/stage)
decayed params: 3932946432, non-decayed: 233472 | ZeRO-1 dtype groups: 2 | world_size: 2
found 1 shards for split train
found 1 shards for split val

Starting training...

[sm90] nn  ACCEPT  M=8192    N=3072    K=3072    b=1   a=1 be=0 lda=0       ldb=0       ldc=0       sA=0            sB=0            sC=25165824     want_sC=25165824
[BluBLAS] SM90 dispatcher: KOTA (hybrid) -- our kernels, his selection
[sm90] nn  ACCEPT  M=8192    N=3072    K=3072    b=1   a=1 be=0 lda=0       ldb=0       ldc=0       sA=0            sB=0            sC=25165824     want_sC=25165824
[BluBLAS] SM90 dispatcher: KOTA (hybrid) -- our kernels, his selection
[sm90] nn  ACCEPT  M=8192    N=1024    K=3072    b=1   a=1 be=0 lda=0       ldb=0       ldc=0       sA=0            sB=0            sC=8388608      want_sC=8388608
[sm90] nn  ACCEPT  M=8192    N=1024    K=3072    b=1   a=1 be=0 lda=0       ldb=0       ldc=0       sA=0            sB=0            sC=8388608      want_sC=8388608
[sm90] nn  ACCEPT  M=8192    N=16384   K=3072    b=1   a=1 be=0 lda=0       ldb=0       ldc=0       sA=0            sB=0            sC=134217728    want_sC=134217728
[sm90] nn  ACCEPT  M=8192    N=16384   K=3072    b=1   a=1 be=0 lda=0       ldb=0       ldc=0       sA=0            sB=0            sC=134217728    want_sC=134217728
[sm90] nn  ACCEPT  M=8192    N=3072    K=8192    b=1   a=1 be=0 lda=0       ldb=0       ldc=0       sA=0            sB=0            sC=25165824     want_sC=25165824
[sm90] nn  ACCEPT  M=8192    N=3072    K=8192    b=1   a=1 be=0 lda=0       ldb=0       ldc=0       sA=0            sB=0            sC=25165824     want_sC=25165824
[sm90] nn  ACCEPT  M=8192    N=50304   K=3072    b=1   a=1 be=0 lda=0       ldb=0       ldc=0       sA=0            sB=0            sC=412090368    want_sC=412090368
[sm90] nn  ACCEPT  M=8192    N=50304   K=3072    b=1   a=1 be=0 lda=0       ldb=0       ldc=0       sA=0            sB=0            sC=412090368    want_sC=412090368
  [val] step 0 | val_loss 11.437630
[sm90] nt  ACCEPT  M=8192    N=3072    K=50304   b=1   a=1 be=0 lda=0       ldb=0       ldc=0       sA=0            sB=0            sC=0            want_sC=25165824
[sm90] nt  ACCEPT  M=8192    N=3072    K=50304   b=1   a=1 be=0 lda=0       ldb=0       ldc=0       sA=0            sB=0            sC=0            want_sC=25165824
[sm90] tn  ACCEPT  M=3072    N=50304   K=8192    b=1   a=1 be=0 lda=0       ldb=0       ldc=0       sA=0            sB=0            sC=0            want_sC=154533888
[sm90] tn  ACCEPT  M=3072    N=50304   K=8192    b=1   a=1 be=0 lda=0       ldb=0       ldc=0       sA=0            sB=0            sC=0            want_sC=154533888
[sm90] nt  ACCEPT  M=8192    N=8192    K=3072    b=1   a=1 be=0 lda=0       ldb=0       ldc=0       sA=0            sB=0            sC=0            want_sC=67108864
[sm90] tn  ACCEPT  M=8192    N=3072    K=8192    b=1   a=1 be=0 lda=0       ldb=0       ldc=0       sA=0            sB=0            sC=0            want_sC=25165824
[sm90] nt  ACCEPT  M=8192    N=3072    K=16384   b=1   a=1 be=0 lda=0       ldb=0       ldc=0       sA=0            sB=0            sC=0            want_sC=25165824
[sm90] tn  ACCEPT  M=3072    N=16384   K=8192    b=1   a=1 be=0 lda=0       ldb=0       ldc=0       sA=0            sB=0            sC=0            want_sC=50331648
[sm90] nt  ACCEPT  M=8192    N=3072    K=3072    b=1   a=1 be=0 lda=0       ldb=0       ldc=0       sA=0            sB=0            sC=0            want_sC=25165824
[sm90] tn  ACCEPT  M=3072    N=3072    K=8192    b=1   a=1 be=0 lda=0       ldb=0       ldc=0       sA=0            sB=0            sC=0            want_sC=9437184
[sm90] nt  ACCEPT  M=8192    N=8192    K=3072    b=1   a=1 be=0 lda=0       ldb=0       ldc=0       sA=0            sB=0            sC=0            want_sC=67108864
[sm90] tn  ACCEPT  M=8192    N=3072    K=8192    b=1   a=1 be=0 lda=0       ldb=0       ldc=0       sA=0            sB=0            sC=0            want_sC=25165824
[sm90] nt  ACCEPT  M=8192    N=3072    K=16384   b=1   a=1 be=0 lda=0       ldb=0       ldc=0       sA=0            sB=0            sC=0            want_sC=25165824
[sm90] tn  ACCEPT  M=3072    N=16384   K=8192    b=1   a=1 be=0 lda=0       ldb=0       ldc=0       sA=0            sB=0            sC=0            want_sC=50331648
[sm90] nt  ACCEPT  M=8192    N=3072    K=3072    b=1   a=1 be=0 lda=0       ldb=0       ldc=0       sA=0            sB=0            sC=0            want_sC=25165824
[sm90] tn  ACCEPT  M=3072    N=3072    K=8192    b=1   a=1 be=0 lda=0       ldb=0       ldc=0       sA=0            sB=0            sC=0            want_sC=9437184
[sm90] nt  ACCEPT  M=8192    N=3072    K=1024    b=1   a=1 be=0 lda=0       ldb=0       ldc=0       sA=0            sB=0            sC=0            want_sC=25165824
[sm90] tn  ACCEPT  M=3072    N=1024    K=8192    b=1   a=1 be=0 lda=0       ldb=0       ldc=0       sA=0            sB=0            sC=0            want_sC=3145728
[sm90] nt  ACCEPT  M=8192    N=3072    K=1024    b=1   a=1 be=0 lda=0       ldb=0       ldc=0       sA=0            sB=0            sC=0            want_sC=25165824
[sm90] tn  ACCEPT  M=3072    N=1024    K=8192    b=1   a=1 be=0 lda=0       ldb=0       ldc=0       sA=0            sB=0            sC=0            want_sC=3145728
step      0 | train 11.436555 | val 11.437630 | tok/s 34771.32 | norm 22.3192 | dt 15078.17 ms | lr 1.0000e-04 | mfu 51.01% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 22.2538 | G1(float32/154M) 1.7063
step      1 | train 13.864193 | val 11.437630 | tok/s 36214.82 | norm 844.6065 | dt 14477.17 ms | lr 2.0000e-04 | mfu 53.13% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 844.4075 | G1(float32/154M) 18.3322
step      2 | train 16.725611 | val 11.437630 | tok/s 36099.76 | norm 277.0920 | dt 14523.31 ms | lr 3.0000e-04 | mfu 52.96% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 277.0326 | G1(float32/154M) 5.7356
step      3 | train 13.961983 | val 11.437630 | tok/s 35993.50 | norm 281.6245 | dt 14566.19 ms | lr 3.0000e-04 | mfu 52.81% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 281.1119 | G1(float32/154M) 16.9837
step      4 | train 13.347113 | val 11.437630 | tok/s 35811.65 | norm 7.3718 | dt 14640.15 ms | lr 3.0000e-04 | mfu 52.54% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 7.3699 | G1(float32/154M) 0.1659
step      5 | train 17.032223 | val 11.437630 | tok/s 35798.71 | norm 56.9356 | dt 14645.44 ms | lr 3.0000e-04 | mfu 52.52% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 56.9249 | G1(float32/154M) 1.1037
step      6 | train 13.108917 | val 11.437630 | tok/s 35863.70 | norm 41.1233 | dt 14618.90 ms | lr 3.0000e-04 | mfu 52.62% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 41.1145 | G1(float32/154M) 0.8512
step      7 | train 10.974763 | val 11.437630 | tok/s 35886.80 | norm 11.1767 | dt 14609.49 ms | lr 3.0000e-04 | mfu 52.65% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 11.1741 | G1(float32/154M) 0.2418
step      8 | train 10.776974 | val 11.437630 | tok/s 35922.64 | norm 4.4796 | dt 14594.92 ms | lr 3.0000e-04 | mfu 52.70% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 4.4788 | G1(float32/154M) 0.0877
step      9 | train 10.138756 | val 11.437630 | tok/s 35922.42 | norm 5.1249 | dt 14595.01 ms | lr 3.0000e-04 | mfu 52.70% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 5.1239 | G1(float32/154M) 0.1014
step     10 | train 9.410402 | val 11.437630 | tok/s 35955.36 | norm 4.5713 | dt 14581.64 ms | lr 3.0000e-04 | mfu 52.75% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 4.5705 | G1(float32/154M) 0.0831
step     11 | train 8.808736 | val 11.437630 | tok/s 35979.98 | norm 4.1127 | dt 14571.66 ms | lr 3.0000e-04 | mfu 52.79% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 4.1121 | G1(float32/154M) 0.0705
step     12 | train 8.258771 | val 11.437630 | tok/s 35961.61 | norm 3.7272 | dt 14579.10 ms | lr 3.0000e-04 | mfu 52.76% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 3.7267 | G1(float32/154M) 0.0641
step     13 | train 8.185595 | val 11.437630 | tok/s 35971.99 | norm 13.0561 | dt 14574.90 ms | lr 3.0000e-04 | mfu 52.78% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 13.0538 | G1(float32/154M) 0.2424
step     14 | train 8.359110 | val 11.437630 | tok/s 35992.17 | norm 15.4768 | dt 14566.72 ms | lr 3.0000e-04 | mfu 52.81% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 15.4740 | G1(float32/154M) 0.2937
step     15 | train 8.472660 | val 11.437630 | tok/s 36014.84 | norm 18.2142 | dt 14557.55 ms | lr 3.0000e-04 | mfu 52.84% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 18.2108 | G1(float32/154M) 0.3511
step     16 | train 8.524753 | val 11.437630 | tok/s 35974.25 | norm 20.5478 | dt 14573.98 ms | lr 3.0000e-04 | mfu 52.78% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 20.5443 | G1(float32/154M) 0.3792
step     17 | train 8.418642 | val 11.437630 | tok/s 36011.98 | norm 12.8016 | dt 14558.71 ms | lr 3.0000e-04 | mfu 52.83% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 12.7994 | G1(float32/154M) 0.2402
step     18 | train 8.294221 | val 11.437630 | tok/s 35923.40 | norm 3.7652 | dt 14594.61 ms | lr 3.0000e-04 | mfu 52.71% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 3.7642 | G1(float32/154M) 0.0871
step     19 | train 8.193657 | val 11.437630 | tok/s 35924.33 | norm 2.9829 | dt 14594.23 ms | lr 2.9453e-04 | mfu 52.71% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 2.9821 | G1(float32/154M) 0.0692
step     20 | train 8.124596 | val 11.437630 | tok/s 35946.85 | norm 4.7315 | dt 14585.09 ms | lr 2.7857e-04 | mfu 52.74% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 4.7306 | G1(float32/154M) 0.0907
step     21 | train 8.028868 | val 11.437630 | tok/s 35911.60 | norm 3.9804 | dt 14599.41 ms | lr 2.5341e-04 | mfu 52.69% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 3.9797 | G1(float32/154M) 0.0769
step     22 | train 8.006801 | val 11.437630 | tok/s 35916.98 | norm 3.8238 | dt 14597.22 ms | lr 2.2108e-04 | mfu 52.70% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 3.8230 | G1(float32/154M) 0.0753
step     23 | train 7.938396 | val 11.437630 | tok/s 35894.05 | norm 2.7285 | dt 14606.54 ms | lr 1.8421e-04 | mfu 52.66% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 2.7280 | G1(float32/154M) 0.0539
step     24 | train 7.899400 | val 11.437630 | tok/s 35909.07 | norm 1.9938 | dt 14600.43 ms | lr 1.4579e-04 | mfu 52.68% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 1.9934 | G1(float32/154M) 0.0390
step     25 | train 7.885095 | val 11.437630 | tok/s 35925.65 | norm 1.5469 | dt 14593.70 ms | lr 1.0892e-04 | mfu 52.71% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 1.5466 | G1(float32/154M) 0.0308
step     26 | train 7.852886 | val 11.437630 | tok/s 35940.91 | norm 1.7425 | dt 14587.50 ms | lr 7.6594e-05 | mfu 52.73% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 1.7421 | G1(float32/154M) 0.0346
step     27 | train 7.785094 | val 11.437630 | tok/s 35928.56 | norm 1.5894 | dt 14592.51 ms | lr 5.1431e-05 | mfu 52.71% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 1.5891 | G1(float32/154M) 0.0316
step     28 | train 7.819690 | val 11.437630 | tok/s 35927.32 | norm 1.3066 | dt 14593.02 ms | lr 3.5468e-05 | mfu 52.71% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 1.3063 | G1(float32/154M) 0.0271
  [val] step 29 | val_loss 7.976027
step     29 | train 7.824329 | val 7.976027 | tok/s 35940.12 | norm 1.1066 | dt 14587.82 ms | lr 3.0000e-05 | mfu 52.73% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 1.1063 | G1(float32/154M) 0.0232

==============================================================================
  STEADY-STATE SUMMARY — C++ ZeRO-1  world=2  fused_rms_rope=1  overlap=1
  (27 steps, first 3 excluded)
==============================================================================
  host  step time    :   14593.02 ms  (mean 14591.35)
  throughput         :    35927.3 tok/s   (35.9 Ktok/s)
  MFU                :      52.71 %
  tokens / step      :     524288
  peak mem allocated :     101.79 GiB
  peak mem reserved  :     103.18 GiB
==============================================================================


=== Training complete ===
Collecting data...
Generating '/tmp/nsys-report-e802.qdstrm'

[1/1] [0%                          ] h228_hybrid.nsys-rep
[1/1] [0%                          ] h228_hybrid.nsys-rep
[1/1] [5%                          ] h228_hybrid.nsys-rep
[1/1] [5%                          ] h228_hybrid.nsys-rep
[1/1] [6%                          ] h228_hybrid.nsys-rep
[1/1] [7%                          ] h228_hybrid.nsys-rep
[1/1] [6%                          ] h228_hybrid.nsys-rep
[1/1] [7%                          ] h228_hybrid.nsys-rep
[1/1] [6%                          ] h228_hybrid.nsys-rep
[1/1] [7%                          ] h228_hybrid.nsys-rep
[1/1] [6%                          ] h228_hybrid.nsys-rep
[1/1] [9%                          ] h228_hybrid.nsys-rep
[1/1] [10%                         ] h228_hybrid.nsys-rep
[1/1] [11%                         ] h228_hybrid.nsys-rep
[1/1] [10%                         ] h228_hybrid.nsys-rep
[1/1] [11%                         ] h228_hybrid.nsys-rep
[1/1] [10%                         ] h228_hybrid.nsys-rep
[1/1] [11%                         ] h228_hybrid.nsys-rep
[1/1] [10%                         ] h228_hybrid.nsys-rep
[1/1] [13%                         ] h228_hybrid.nsys-rep
[1/1] [12%                         ] h228_hybrid.nsys-rep
[1/1] [13%                         ] h228_hybrid.nsys-rep
[1/1] [12%                         ] h228_hybrid.nsys-rep
[1/1] [13%                         ] h228_hybrid.nsys-rep
[1/1] [12%                         ] h228_hybrid.nsys-rep
[1/1] [13%                         ] h228_hybrid.nsys-rep
[1/1] [=16%                        ] h228_hybrid.nsys-rep
[1/1] [=15%                        ] h228_hybrid.nsys-rep
[1/1] [=17%                        ] h228_hybrid.nsys-rep
[1/1] [=16%                        ] h228_hybrid.nsys-rep
[1/1] [==18%                       ] h228_hybrid.nsys-rep
[1/1] [==20%                       ] h228_hybrid.nsys-rep
[1/1] [==19%                       ] h228_hybrid.nsys-rep
[1/1] [==21%                       ] h228_hybrid.nsys-rep
[1/1] [==20%                       ] h228_hybrid.nsys-rep
[1/1] [==21%                       ] h228_hybrid.nsys-rep
[1/1] [===22%                      ] h228_hybrid.nsys-rep
[1/1] [===23%                      ] h228_hybrid.nsys-rep
[1/1] [===22%                      ] h228_hybrid.nsys-rep
[1/1] [===24%                      ] h228_hybrid.nsys-rep
[1/1] [===23%                      ] h228_hybrid.nsys-rep
[1/1] [====25%                     ] h228_hybrid.nsys-rep
[1/1] [===24%                      ] h228_hybrid.nsys-rep
[1/1] [====25%                     ] h228_hybrid.nsys-rep
[1/1] [====26%                     ] h228_hybrid.nsys-rep
[1/1] [====25%                     ] h228_hybrid.nsys-rep
[1/1] [====26%                     ] h228_hybrid.nsys-rep
[1/1] [====27%                     ] h228_hybrid.nsys-rep
[1/1] [====26%                     ] h228_hybrid.nsys-rep
[1/1] [====27%                     ] h228_hybrid.nsys-rep
[1/1] [====28%                     ] h228_hybrid.nsys-rep
[1/1] [====27%                     ] h228_hybrid.nsys-rep
[1/1] [=====29%                    ] h228_hybrid.nsys-rep
[1/1] [====28%                     ] h228_hybrid.nsys-rep
[1/1] [=====29%                    ] h228_hybrid.nsys-rep
[1/1] [====28%                     ] h228_hybrid.nsys-rep
[1/1] [=====29%                    ] h228_hybrid.nsys-rep
[1/1] [====28%                     ] h228_hybrid.nsys-rep
[1/1] [====27%                     ] h228_hybrid.nsys-rep
[1/1] [====26%                     ] h228_hybrid.nsys-rep
[1/1] [====25%                     ] h228_hybrid.nsys-rep
[1/1] [===24%                      ] h228_hybrid.nsys-rep
[1/1] [===23%                      ] h228_hybrid.nsys-rep
[1/1] [===22%                      ] h228_hybrid.nsys-rep
[1/1] [==21%                       ] h228_hybrid.nsys-rep
[1/1] [==20%                       ] h228_hybrid.nsys-rep
[1/1] [==19%                       ] h228_hybrid.nsys-rep
[1/1] [==18%                       ] h228_hybrid.nsys-rep
[1/1] [=17%                        ] h228_hybrid.nsys-rep
[1/1] [=16%                        ] h228_hybrid.nsys-rep
[1/1] [=15%                        ] h228_hybrid.nsys-rep
[1/1] [14%                         ] h228_hybrid.nsys-rep
[1/1] [13%                         ] h228_hybrid.nsys-rep
[1/1] [12%                         ] h228_hybrid.nsys-rep
[1/1] [11%                         ] h228_hybrid.nsys-rep
[1/1] [10%                         ] h228_hybrid.nsys-rep
[1/1] [9%                          ] h228_hybrid.nsys-rep
[1/1] [8%                          ] h228_hybrid.nsys-rep
[1/1] [7%                          ] h228_hybrid.nsys-rep
[1/1] [6%                          ] h228_hybrid.nsys-rep
[1/1] [5%                          ] h228_hybrid.nsys-rep
[1/1] [5%                          ] h228_hybrid.nsys-rep
[1/1] [6%                          ] h228_hybrid.nsys-rep
[1/1] [7%                          ] h228_hybrid.nsys-rep
[1/1] [8%                          ] h228_hybrid.nsys-rep
[1/1] [9%                          ] h228_hybrid.nsys-rep
[1/1] [10%                         ] h228_hybrid.nsys-rep
[1/1] [11%                         ] h228_hybrid.nsys-rep
[1/1] [12%                         ] h228_hybrid.nsys-rep
[1/1] [13%                         ] h228_hybrid.nsys-rep
[1/1] [14%                         ] h228_hybrid.nsys-rep
[1/1] [=15%                        ] h228_hybrid.nsys-rep
[1/1] [=16%                        ] h228_hybrid.nsys-rep
[1/1] [=17%                        ] h228_hybrid.nsys-rep
[1/1] [==18%                       ] h228_hybrid.nsys-rep
[1/1] [==19%                       ] h228_hybrid.nsys-rep
[1/1] [==20%                       ] h228_hybrid.nsys-rep
[1/1] [==21%                       ] h228_hybrid.nsys-rep
[1/1] [===22%                      ] h228_hybrid.nsys-rep
[1/1] [===23%                      ] h228_hybrid.nsys-rep
[1/1] [===24%                      ] h228_hybrid.nsys-rep
[1/1] [====25%                     ] h228_hybrid.nsys-rep
[1/1] [====26%                     ] h228_hybrid.nsys-rep
[1/1] [====27%                     ] h228_hybrid.nsys-rep
[1/1] [====28%                     ] h228_hybrid.nsys-rep
[1/1] [=====29%                    ] h228_hybrid.nsys-rep
[1/1] [=====30%                    ] h228_hybrid.nsys-rep
[1/1] [=====31%                    ] h228_hybrid.nsys-rep
[1/1] [=====32%                    ] h228_hybrid.nsys-rep
[1/1] [======33%                   ] h228_hybrid.nsys-rep
[1/1] [======34%                   ] h228_hybrid.nsys-rep
[1/1] [======35%                   ] h228_hybrid.nsys-rep
[1/1] [=======36%                  ] h228_hybrid.nsys-rep
[1/1] [=======37%                  ] h228_hybrid.nsys-rep
[1/1] [=======38%                  ] h228_hybrid.nsys-rep
[1/1] [=======39%                  ] h228_hybrid.nsys-rep
[1/1] [========40%                 ] h228_hybrid.nsys-rep
[1/1] [========41%                 ] h228_hybrid.nsys-rep
[1/1] [========42%                 ] h228_hybrid.nsys-rep
[1/1] [=========43%                ] h228_hybrid.nsys-rep
[1/1] [=========44%                ] h228_hybrid.nsys-rep
[1/1] [=========45%                ] h228_hybrid.nsys-rep
[1/1] [=========46%                ] h228_hybrid.nsys-rep
[1/1] [==========47%               ] h228_hybrid.nsys-rep
[1/1] [==========48%               ] h228_hybrid.nsys-rep
[1/1] [==========49%               ] h228_hybrid.nsys-rep
[1/1] [===========50%              ] h228_hybrid.nsys-rep
[1/1] [===========51%              ] h228_hybrid.nsys-rep
[1/1] [===========52%              ] h228_hybrid.nsys-rep
[1/1] [===========53%              ] h228_hybrid.nsys-rep
[1/1] [============54%             ] h228_hybrid.nsys-rep
[1/1] [============55%             ] h228_hybrid.nsys-rep
[1/1] [============56%             ] h228_hybrid.nsys-rep
[1/1] [============57%             ] h228_hybrid.nsys-rep
[1/1] [=============58%            ] h228_hybrid.nsys-rep
[1/1] [=============59%            ] h228_hybrid.nsys-rep
[1/1] [=============60%            ] h228_hybrid.nsys-rep
[1/1] [==============61%           ] h228_hybrid.nsys-rep
[1/1] [==============62%           ] h228_hybrid.nsys-rep
[1/1] [==============63%           ] h228_hybrid.nsys-rep
[1/1] [==============64%           ] h228_hybrid.nsys-rep
[1/1] [===============65%          ] h228_hybrid.nsys-rep
[1/1] [===============66%          ] h228_hybrid.nsys-rep
[1/1] [===============67%          ] h228_hybrid.nsys-rep
[1/1] [================68%         ] h228_hybrid.nsys-rep
[1/1] [================69%         ] h228_hybrid.nsys-rep
[1/1] [================70%         ] h228_hybrid.nsys-rep
[1/1] [================71%         ] h228_hybrid.nsys-rep
[1/1] [=================72%        ] h228_hybrid.nsys-rep
[1/1] [=================73%        ] h228_hybrid.nsys-rep
[1/1] [=================74%        ] h228_hybrid.nsys-rep
[1/1] [==================75%       ] h228_hybrid.nsys-rep
[1/1] [==================76%       ] h228_hybrid.nsys-rep
[1/1] [==================77%       ] h228_hybrid.nsys-rep
[1/1] [==================78%       ] h228_hybrid.nsys-rep
[1/1] [===================79%      ] h228_hybrid.nsys-rep
[1/1] [===================80%      ] h228_hybrid.nsys-rep
[1/1] [===================81%      ] h228_hybrid.nsys-rep
[1/1] [========================100%] h228_hybrid.nsys-rep
[1/1] [========================100%] h228_hybrid.nsys-rep
Generated:
	/root/gautam_k/blublas/h228_4b/h228_hybrid.nsys-rep
