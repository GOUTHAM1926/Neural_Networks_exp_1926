
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
[sm90] nt  ACCEPT  M=8192    N=8192    K=3072    b=1   a=1 be=0 lda=0       ldb=0       ldc=0       sA=0            sB=0            sC=0            want_sC=67108864
[sm90] tn  ACCEPT  M=8192    N=3072    K=8192    b=1   a=1 be=0 lda=0       ldb=0       ldc=0       sA=0            sB=0            sC=0            want_sC=25165824
[sm90] tn  ACCEPT  M=8192    N=3072    K=8192    b=1   a=1 be=0 lda=0       ldb=0       ldc=0       sA=0            sB=0            sC=0            want_sC=25165824
[sm90] nt  ACCEPT  M=8192    N=3072    K=16384   b=1   a=1 be=0 lda=0       ldb=0       ldc=0       sA=0            sB=0            sC=0            want_sC=25165824
[sm90] tn  ACCEPT  M=3072    N=16384   K=8192    b=1   a=1 be=0 lda=0       ldb=0       ldc=0       sA=0            sB=0            sC=0            want_sC=50331648
[sm90] nt  ACCEPT  M=8192    N=3072    K=16384   b=1   a=1 be=0 lda=0       ldb=0       ldc=0       sA=0            sB=0            sC=0            want_sC=25165824
[sm90] tn  ACCEPT  M=3072    N=16384   K=8192    b=1   a=1 be=0 lda=0       ldb=0       ldc=0       sA=0            sB=0            sC=0            want_sC=50331648
[sm90] nt  ACCEPT  M=8192    N=3072    K=3072    b=1   a=1 be=0 lda=0       ldb=0       ldc=0       sA=0            sB=0            sC=0            want_sC=25165824
[sm90] tn  ACCEPT  M=3072    N=3072    K=8192    b=1   a=1 be=0 lda=0       ldb=0       ldc=0       sA=0            sB=0            sC=0            want_sC=9437184
[sm90] nt  ACCEPT  M=8192    N=3072    K=3072    b=1   a=1 be=0 lda=0       ldb=0       ldc=0       sA=0            sB=0            sC=0            want_sC=25165824
[sm90] tn  ACCEPT  M=3072    N=3072    K=8192    b=1   a=1 be=0 lda=0       ldb=0       ldc=0       sA=0            sB=0            sC=0            want_sC=9437184
[sm90] nt  ACCEPT  M=8192    N=3072    K=1024    b=1   a=1 be=0 lda=0       ldb=0       ldc=0       sA=0            sB=0            sC=0            want_sC=25165824
[sm90] tn  ACCEPT  M=3072    N=1024    K=8192    b=1   a=1 be=0 lda=0       ldb=0       ldc=0       sA=0            sB=0            sC=0            want_sC=3145728
[sm90] nt  ACCEPT  M=8192    N=3072    K=1024    b=1   a=1 be=0 lda=0       ldb=0       ldc=0       sA=0            sB=0            sC=0            want_sC=25165824
[sm90] tn  ACCEPT  M=3072    N=1024    K=8192    b=1   a=1 be=0 lda=0       ldb=0       ldc=0       sA=0            sB=0            sC=0            want_sC=3145728
step      0 | train 11.436555 | val 11.437630 | tok/s 34580.40 | norm 22.3185 | dt 15161.42 ms | lr 1.0000e-04 | mfu 50.73% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 22.2532 | G1(float32/154M) 1.7064
step      1 | train 13.859459 | val 11.437630 | tok/s 36269.43 | norm 845.0493 | dt 14455.37 ms | lr 2.0000e-04 | mfu 53.21% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 844.8500 | G1(float32/154M) 18.3513
step      2 | train 16.728851 | val 11.437630 | tok/s 36214.70 | norm 276.4343 | dt 14477.21 ms | lr 3.0000e-04 | mfu 53.13% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 276.3752 | G1(float32/154M) 5.7149
step      3 | train 13.878107 | val 11.437630 | tok/s 36094.31 | norm 273.3144 | dt 14525.50 ms | lr 3.0000e-04 | mfu 52.96% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 272.8103 | G1(float32/154M) 16.5931
step      4 | train 13.320080 | val 11.437630 | tok/s 35995.50 | norm 7.3125 | dt 14565.38 ms | lr 3.0000e-04 | mfu 52.81% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 7.3102 | G1(float32/154M) 0.1804
step      5 | train 17.047878 | val 11.437630 | tok/s 35955.99 | norm 56.9357 | dt 14581.38 ms | lr 3.0000e-04 | mfu 52.75% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 56.9250 | G1(float32/154M) 1.1034
step      6 | train 13.160501 | val 11.437630 | tok/s 35919.85 | norm 41.6534 | dt 14596.05 ms | lr 3.0000e-04 | mfu 52.70% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 41.6445 | G1(float32/154M) 0.8613
step      7 | train 11.013187 | val 11.437630 | tok/s 35849.21 | norm 12.3330 | dt 14624.81 ms | lr 3.0000e-04 | mfu 52.60% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 12.3301 | G1(float32/154M) 0.2661
step      8 | train 10.799483 | val 11.437630 | tok/s 35872.21 | norm 4.4522 | dt 14615.44 ms | lr 3.0000e-04 | mfu 52.63% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 4.4514 | G1(float32/154M) 0.0877
step      9 | train 10.171822 | val 11.437630 | tok/s 35896.95 | norm 5.0281 | dt 14605.37 ms | lr 3.0000e-04 | mfu 52.67% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 5.0271 | G1(float32/154M) 0.0997
step     10 | train 9.450632 | val 11.437630 | tok/s 35922.10 | norm 4.5082 | dt 14595.14 ms | lr 3.0000e-04 | mfu 52.70% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 4.5075 | G1(float32/154M) 0.0823
step     11 | train 8.853624 | val 11.437630 | tok/s 35951.67 | norm 4.0755 | dt 14583.13 ms | lr 3.0000e-04 | mfu 52.75% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 4.0749 | G1(float32/154M) 0.0704
step     12 | train 8.308163 | val 11.437630 | tok/s 35955.26 | norm 3.7922 | dt 14581.68 ms | lr 3.0000e-04 | mfu 52.75% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 3.7916 | G1(float32/154M) 0.0656
step     13 | train 8.224754 | val 11.437630 | tok/s 35987.88 | norm 13.8267 | dt 14568.46 ms | lr 3.0000e-04 | mfu 52.80% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 13.8243 | G1(float32/154M) 0.2567
step     14 | train 8.255411 | val 11.437630 | tok/s 36015.97 | norm 15.0231 | dt 14557.10 ms | lr 3.0000e-04 | mfu 52.84% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 15.0205 | G1(float32/154M) 0.2827
step     15 | train 8.611368 | val 11.437630 | tok/s 36035.18 | norm 24.5589 | dt 14549.34 ms | lr 3.0000e-04 | mfu 52.87% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 24.5543 | G1(float32/154M) 0.4734
step     16 | train 8.639296 | val 11.437630 | tok/s 36000.14 | norm 19.3261 | dt 14563.50 ms | lr 3.0000e-04 | mfu 52.82% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 19.3227 | G1(float32/154M) 0.3634
step     17 | train 8.532705 | val 11.437630 | tok/s 36004.31 | norm 15.6376 | dt 14561.81 ms | lr 3.0000e-04 | mfu 52.82% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 15.6349 | G1(float32/154M) 0.2912
step     18 | train 8.344291 | val 11.437630 | tok/s 35943.73 | norm 4.0852 | dt 14586.35 ms | lr 3.0000e-04 | mfu 52.73% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 4.0842 | G1(float32/154M) 0.0913
step     19 | train 8.240036 | val 11.437630 | tok/s 35951.49 | norm 3.3387 | dt 14583.21 ms | lr 2.9453e-04 | mfu 52.75% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 3.3379 | G1(float32/154M) 0.0743
step     20 | train 8.135742 | val 11.437630 | tok/s 35944.64 | norm 2.7699 | dt 14585.99 ms | lr 2.7857e-04 | mfu 52.74% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 2.7692 | G1(float32/154M) 0.0614
step     21 | train 8.056861 | val 11.437630 | tok/s 35959.91 | norm 5.9208 | dt 14579.79 ms | lr 2.5341e-04 | mfu 52.76% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 5.9198 | G1(float32/154M) 0.1085
step     22 | train 8.027786 | val 11.437630 | tok/s 35951.67 | norm 4.3232 | dt 14583.13 ms | lr 2.2108e-04 | mfu 52.75% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 4.3225 | G1(float32/154M) 0.0818
step     23 | train 7.973903 | val 11.437630 | tok/s 35938.01 | norm 3.8843 | dt 14588.68 ms | lr 1.8421e-04 | mfu 52.73% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 3.8836 | G1(float32/154M) 0.0755
step     24 | train 7.932845 | val 11.437630 | tok/s 35943.82 | norm 2.6049 | dt 14586.32 ms | lr 1.4579e-04 | mfu 52.73% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 2.6044 | G1(float32/154M) 0.0510
step     25 | train 7.911107 | val 11.437630 | tok/s 35946.77 | norm 1.6619 | dt 14585.12 ms | lr 1.0892e-04 | mfu 52.74% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 1.6616 | G1(float32/154M) 0.0317
step     26 | train 7.869761 | val 11.437630 | tok/s 35923.54 | norm 1.4114 | dt 14594.55 ms | lr 7.6594e-05 | mfu 52.71% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 1.4112 | G1(float32/154M) 0.0269
step     27 | train 7.801728 | val 11.437630 | tok/s 35931.72 | norm 1.5274 | dt 14591.23 ms | lr 5.1431e-05 | mfu 52.72% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 1.5272 | G1(float32/154M) 0.0294
step     28 | train 7.834256 | val 11.437630 | tok/s 35941.59 | norm 1.3921 | dt 14587.22 ms | lr 3.5468e-05 | mfu 52.73% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 1.3919 | G1(float32/154M) 0.0280
  [val] step 29 | val_loss 7.989809
step     29 | train 7.840083 | val 7.989809 | tok/s 35947.23 | norm 1.4570 | dt 14584.93 ms | lr 3.0000e-05 | mfu 52.74% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 1.4567 | G1(float32/154M) 0.0293

==============================================================================
  STEADY-STATE SUMMARY — C++ ZeRO-1  world=2  fused_rms_rope=1  overlap=1
  (27 steps, first 3 excluded)
==============================================================================

host  step time    :   14584.93 ms  (mean 14581.87)
  throughput         :    35947.2 tok/s   (35.9 Ktok/s)
  MFU                :      52.74 %
  tokens / step      :     524288
  peak mem allocated :     101.79 GiB
  peak mem reserved  :     103.18 GiB
==============================================================================

=== Training complete ===
Collecting data...
Generating '/tmp/nsys-report-b2e3.qdstrm'

[1/6] [0%                          ] h228_hybrid.nsys-rep
[1/6] [0%                          ] h228_hybrid.nsys-rep
[1/6] [5%                          ] h228_hybrid.nsys-rep
[1/6] [6%                          ] h228_hybrid.nsys-rep
[1/6] [9%                          ] h228_hybrid.nsys-rep
[1/6] [8%                          ] h228_hybrid.nsys-rep
[1/6] [9%                          ] h228_hybrid.nsys-rep
[1/6] [10%                         ] h228_hybrid.nsys-rep
[1/6] [9%                          ] h228_hybrid.nsys-rep
[1/6] [12%                         ] h228_hybrid.nsys-rep
[1/6] [11%                         ] h228_hybrid.nsys-rep
[1/6] [12%                         ] h228_hybrid.nsys-rep
[1/6] [11%                         ] h228_hybrid.nsys-rep
[1/6] [14%                         ] h228_hybrid.nsys-rep
[1/6] [=16%                        ] h228_hybrid.nsys-rep
[1/6] [==18%                       ] h228_hybrid.nsys-rep
[1/6] [=17%                        ] h228_hybrid.nsys-rep
[1/6] [==19%                       ] h228_hybrid.nsys-rep
[1/6] [==18%                       ] h228_hybrid.nsys-rep
[1/6] [==19%                       ] h228_hybrid.nsys-rep
[1/6] [==20%                       ] h228_hybrid.nsys-rep
[1/6] [==21%                       ] h228_hybrid.nsys-rep
[1/6] [===22%                      ] h228_hybrid.nsys-rep
[1/6] [==21%                       ] h228_hybrid.nsys-rep
[1/6] [===23%                      ] h228_hybrid.nsys-rep
[1/6] [===22%                      ] h228_hybrid.nsys-rep
[1/6] [===24%                      ] h228_hybrid.nsys-rep
[1/6] [===23%                      ] h228_hybrid.nsys-rep
[1/6] [===24%                      ] h228_hybrid.nsys-rep
[1/6] [====25%                     ] h228_hybrid.nsys-rep
[1/6] [====26%                     ] h228_hybrid.nsys-rep
[1/6] [====25%                     ] h228_hybrid.nsys-rep
[1/6] [====27%                     ] h228_hybrid.nsys-rep
[1/6] [====26%                     ] h228_hybrid.nsys-rep
[1/6] [====27%                     ] h228_hybrid.nsys-rep
[1/6] [====28%                     ] h228_hybrid.nsys-rep
[1/6] [====27%                     ] h228_hybrid.nsys-rep
[1/6] [====28%                     ] h228_hybrid.nsys-rep
[1/6] [=====29%                    ] h228_hybrid.nsys-rep
[1/6] [====28%                     ] h228_hybrid.nsys-rep
[1/6] [=====29%                    ] h228_hybrid.nsys-rep
[1/6] [=====30%                    ] h228_hybrid.nsys-rep
[1/6] [=====29%                    ] h228_hybrid.nsys-rep
[1/6] [====28%                     ] h228_hybrid.nsys-rep
[1/6] [====27%                     ] h228_hybrid.nsys-rep
[1/6] [====26%                     ] h228_hybrid.nsys-rep
[1/6] [====25%                     ] h228_hybrid.nsys-rep
[1/6] [===24%                      ] h228_hybrid.nsys-rep
[1/6] [===23%                      ] h228_hybrid.nsys-rep
[1/6] [===22%                      ] h228_hybrid.nsys-rep
[1/6] [==21%                       ] h228_hybrid.nsys-rep
[1/6] [==20%                       ] h228_hybrid.nsys-rep
[1/6] [==19%                       ] h228_hybrid.nsys-rep
[1/6] [==18%                       ] h228_hybrid.nsys-rep
[1/6] [=17%                        ] h228_hybrid.nsys-rep
[1/6] [=16%                        ] h228_hybrid.nsys-rep
[1/6] [=15%                        ] h228_hybrid.nsys-rep
[1/6] [14%                         ] h228_hybrid.nsys-rep
[1/6] [13%                         ] h228_hybrid.nsys-rep
[1/6] [12%                         ] h228_hybrid.nsys-rep
[1/6] [11%                         ] h228_hybrid.nsys-rep
[1/6] [10%                         ] h228_hybrid.nsys-rep
[1/6] [9%                          ] h228_hybrid.nsys-rep
[1/6] [8%                          ] h228_hybrid.nsys-rep
[1/6] [7%                          ] h228_hybrid.nsys-rep
[1/6] [6%                          ] h228_hybrid.nsys-rep
[1/6] [5%                          ] h228_hybrid.nsys-rep
[1/6] [5%                          ] h228_hybrid.nsys-rep
[1/6] [6%                          ] h228_hybrid.nsys-rep
[1/6] [7%                          ] h228_hybrid.nsys-rep
[1/6] [8%                          ] h228_hybrid.nsys-rep
[1/6] [9%                          ] h228_hybrid.nsys-rep
[1/6] [10%                         ] h228_hybrid.nsys-rep
[1/6] [11%                         ] h228_hybrid.nsys-rep
[1/6] [12%                         ] h228_hybrid.nsys-rep
[1/6] [13%                         ] h228_hybrid.nsys-rep
[1/6] [14%                         ] h228_hybrid.nsys-rep
[1/6] [=15%                        ] h228_hybrid.nsys-rep
[1/6] [=16%                        ] h228_hybrid.nsys-rep
[1/6] [=17%                        ] h228_hybrid.nsys-rep
[1/6] [==18%                       ] h228_hybrid.nsys-rep
[1/6] [==19%                       ] h228_hybrid.nsys-rep
[1/6] [==20%                       ] h228_hybrid.nsys-rep
[1/6] [==21%                       ] h228_hybrid.nsys-rep
[1/6] [===22%                      ] h228_hybrid.nsys-rep
[1/6] [===23%                      ] h228_hybrid.nsys-rep
[1/6] [===24%                      ] h228_hybrid.nsys-rep
[1/6] [====25%                     ] h228_hybrid.nsys-rep
[1/6] [====26%                     ] h228_hybrid.nsys-rep
[1/6] [====27%                     ] h228_hybrid.nsys-rep
[1/6] [====28%                     ] h228_hybrid.nsys-rep
[1/6] [=====29%                    ] h228_hybrid.nsys-rep
[1/6] [=====30%                    ] h228_hybrid.nsys-rep
[1/6] [=====31%                    ] h228_hybrid.nsys-rep
[1/6] [=====32%                    ] h228_hybrid.nsys-rep
[1/6] [======33%                   ] h228_hybrid.nsys-rep
[1/6] [======34%                   ] h228_hybrid.nsys-rep
[1/6] [======35%                   ] h228_hybrid.nsys-rep
[1/6] [=======36%                  ] h228_hybrid.nsys-rep
[1/6] [=======37%                  ] h228_hybrid.nsys-rep
[1/6] [=======38%                  ] h228_hybrid.nsys-rep
[1/6] [=======39%                  ] h228_hybrid.nsys-rep
[1/6] [========40%                 ] h228_hybrid.nsys-rep
[1/6] [========41%                 ] h228_hybrid.nsys-rep
[1/6] [========42%                 ] h228_hybrid.nsys-rep
[1/6] [=========43%                ] h228_hybrid.nsys-rep
[1/6] [=========44%                ] h228_hybrid.nsys-rep
[1/6] [=========45%                ] h228_hybrid.nsys-rep
[1/6] [=========46%                ] h228_hybrid.nsys-rep
[1/6] [==========47%               ] h228_hybrid.nsys-rep
[1/6] [==========48%               ] h228_hybrid.nsys-rep
[1/6] [==========49%               ] h228_hybrid.nsys-rep
[1/6] [===========50%              ] h228_hybrid.nsys-rep
[1/6] [===========51%              ] h228_hybrid.nsys-rep
[1/6] [===========52%              ] h228_hybrid.nsys-rep
[1/6] [===========53%              ] h228_hybrid.nsys-rep
[1/6] [============54%             ] h228_hybrid.nsys-rep
[1/6] [============55%             ] h228_hybrid.nsys-rep
[1/6] [============56%             ] h228_hybrid.nsys-rep
[1/6] [============57%             ] h228_hybrid.nsys-rep
[1/6] [=============58%            ] h228_hybrid.nsys-rep
[1/6] [=============59%            ] h228_hybrid.nsys-rep
[1/6] [=============60%            ] h228_hybrid.nsys-rep
[1/6] [==============61%           ] h228_hybrid.nsys-rep
[1/6] [==============62%           ] h228_hybrid.nsys-rep
[1/6] [==============63%           ] h228_hybrid.nsys-rep
[1/6] [==============64%           ] h228_hybrid.nsys-rep
[1/6] [===============65%          ] h228_hybrid.nsys-rep
[1/6] [===============66%          ] h228_hybrid.nsys-rep
[1/6] [===============67%          ] h228_hybrid.nsys-rep
[1/6] [================68%         ] h228_hybrid.nsys-rep
[1/6] [================69%         ] h228_hybrid.nsys-rep
[1/6] [================70%         ] h228_hybrid.nsys-rep
[1/6] [================71%         ] h228_hybrid.nsys-rep
[1/6] [=================72%        ] h228_hybrid.nsys-rep
[1/6] [=================73%        ] h228_hybrid.nsys-rep
[1/6] [=================74%        ] h228_hybrid.nsys-rep
[1/6] [==================75%       ] h228_hybrid.nsys-rep
[1/6] [==================76%       ] h228_hybrid.nsys-rep
[1/6] [==================77%       ] h228_hybrid.nsys-rep
[1/6] [==================78%       ] h228_hybrid.nsys-rep
[1/6] [===================79%      ] h228_hybrid.nsys-rep
[1/6] [===================80%      ] h228_hybrid.nsys-rep
[1/6] [===================81%      ] h228_hybrid.nsys-rep
[1/6] [========================100%] h228_hybrid.nsys-rep
[1/6] [========================100%] h228_hybrid.nsys-rep

[2/6] [0%                          ] h228_hybrid.sqlite
[2/6] [1%                          ] h228_hybrid.sqlite
[2/6] [2%                          ] h228_hybrid.sqlite
[2/6] [3%                          ] h228_hybrid.sqlite
[2/6] [4%                          ] h228_hybrid.sqlite
[2/6] [5%                          ] h228_hybrid.sqlite
[2/6] [6%                          ] h228_hybrid.sqlite
[2/6] [7%                          ] h228_hybrid.sqlite
[2/6] [8%                          ] h228_hybrid.sqlite
[2/6] [9%                          ] h228_hybrid.sqlite
[2/6] [10%                         ] h228_hybrid.sqlite
[2/6] [11%                         ] h228_hybrid.sqlite
[2/6] [12%                         ] h228_hybrid.sqlite
[2/6] [13%                         ] h228_hybrid.sqlite
[2/6] [14%                         ] h228_hybrid.sqlite
[2/6] [=15%                        ] h228_hybrid.sqlite
[2/6] [=16%                        ] h228_hybrid.sqlite
[2/6] [=17%                        ] h228_hybrid.sqlite
[2/6] [==18%                       ] h228_hybrid.sqlite
[2/6] [==19%                       ] h228_hybrid.sqlite
[2/6] [==20%                       ] h228_hybrid.sqlite
[2/6] [==21%                       ] h228_hybrid.sqlite
[2/6] [===22%                      ] h228_hybrid.sqlite
[2/6] [===23%                      ] h228_hybrid.sqlite
[2/6] [===24%                      ] h228_hybrid.sqlite
[2/6] [====25%                     ] h228_hybrid.sqlite
[2/6] [====26%                     ] h228_hybrid.sqlite
[2/6] [====27%                     ] h228_hybrid.sqlite
[2/6] [====28%                     ] h228_hybrid.sqlite
[2/6] [=====29%                    ] h228_hybrid.sqlite
[2/6] [=====30%                    ] h228_hybrid.sqlite
[2/6] [=====31%                    ] h228_hybrid.sqlite
[2/6] [=====32%                    ] h228_hybrid.sqlite
[2/6] [======33%                   ] h228_hybrid.sqlite
[2/6] [======34%                   ] h228_hybrid.sqlite
[2/6] [======35%                   ] h228_hybrid.sqlite
[2/6] [=======36%                  ] h228_hybrid.sqlite
[2/6] [=======37%                  ] h228_hybrid.sqlite
[2/6] [=======38%                  ] h228_hybrid.sqlite
[2/6] [=======39%                  ] h228_hybrid.sqlite
[2/6] [========40%                 ] h228_hybrid.sqlite
[2/6] [========41%                 ] h228_hybrid.sqlite
[2/6] [========42%                 ] h228_hybrid.sqlite
[2/6] [=========43%                ] h228_hybrid.sqlite
[2/6] [=========44%                ] h228_hybrid.sqlite
[2/6] [=========45%                ] h228_hybrid.sqlite
[2/6] [=========46%                ] h228_hybrid.sqlite
[2/6] [==========47%               ] h228_hybrid.sqlite
[2/6] [==========48%               ] h228_hybrid.sqlite
[2/6] [==========49%               ] h228_hybrid.sqlite
[2/6] [===========50%              ] h228_hybrid.sqlite
[2/6] [===========51%              ] h228_hybrid.sqlite
[2/6] [===========52%              ] h228_hybrid.sqlite
[2/6] [===========53%              ] h228_hybrid.sqlite
[2/6] [============54%             ] h228_hybrid.sqlite
[2/6] [============55%             ] h228_hybrid.sqlite
[2/6] [============56%             ] h228_hybrid.sqlite
[2/6] [============57%             ] h228_hybrid.sqlite
[2/6] [=============58%            ] h228_hybrid.sqlite
[2/6] [=============59%            ] h228_hybrid.sqlite
[2/6] [=============60%            ] h228_hybrid.sqlite
[2/6] [==============61%           ] h228_hybrid.sqlite
[2/6] [==============62%           ] h228_hybrid.sqlite
[2/6] [==============63%           ] h228_hybrid.sqlite
[2/6] [==============64%           ] h228_hybrid.sqlite
[2/6] [===============65%          ] h228_hybrid.sqlite
[2/6] [===============66%          ] h228_hybrid.sqlite
[2/6] [===============67%          ] h228_hybrid.sqlite
[2/6] [================68%         ] h228_hybrid.sqlite
[2/6] [================69%         ] h228_hybrid.sqlite
[2/6] [================70%         ] h228_hybrid.sqlite
[2/6] [================71%         ] h228_hybrid.sqlite
[2/6] [=================72%        ] h228_hybrid.sqlite
[2/6] [=================73%        ] h228_hybrid.sqlite
[2/6] [=================74%        ] h228_hybrid.sqlite
[2/6] [==================75%       ] h228_hybrid.sqlite
[2/6] [==================76%       ] h228_hybrid.sqlite
[2/6] [==================77%       ] h228_hybrid.sqlite
[2/6] [==================78%       ] h228_hybrid.sqlite
[2/6] [===================79%      ] h228_hybrid.sqlite
[2/6] [===================80%      ] h228_hybrid.sqlite
[2/6] [===================81%      ] h228_hybrid.sqlite
[2/6] [===================82%      ] h228_hybrid.sqlite
[2/6] [====================83%     ] h228_hybrid.sqlite
[2/6] [====================84%     ] h228_hybrid.sqlite
[2/6] [====================85%     ] h228_hybrid.sqlite
[2/6] [=====================86%    ] h228_hybrid.sqlite
[2/6] [=====================87%    ] h228_hybrid.sqlite
[2/6] [=====================88%    ] h228_hybrid.sqlite
[2/6] [=====================89%    ] h228_hybrid.sqlite
[2/6] [======================90%   ] h228_hybrid.sqlite
[2/6] [======================91%   ] h228_hybrid.sqlite
[2/6] [======================92%   ] h228_hybrid.sqlite
[2/6] [=======================93%  ] h228_hybrid.sqlite
[2/6] [=======================94%  ] h228_hybrid.sqlite
[2/6] [=======================95%  ] h228_hybrid.sqlite
[2/6] [=======================96%  ] h228_hybrid.sqlite
[2/6] [========================97% ] h228_hybrid.sqlite
[2/6] [========================98% ] h228_hybrid.sqlite
[2/6] [========================99% ] h228_hybrid.sqlite
[2/6] [========================100%] h228_hybrid.sqlite
[2/6] [========================100%] h228_hybrid.sqlite
[3/6] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls   Avg (ns)    Med (ns)   Min (ns)  Max (ns)   StdDev (ns)                     Name
 --------  ---------------  ---------  ----------  ----------  --------  ---------  -----------  ------------------------------------------
     94.7     807286435358    6216556    129860.7     34730.0      2190  118291928     300815.8  cudaLaunchKernel                          
      2.5      21649483429      50898    425350.4      6090.0      1740  157515601    6844357.3  cudaMemcpyAsync                           
      1.1       9484813244      69122    137218.4     30820.0      3540    9797548     535276.1  cudaMemset                                
      0.9       7588031157       1480   5127048.1    247630.0       930  192854171   20880338.0  cudaMallocAsync_v11020                    
      0.2       1613102722        906   1780466.6    119605.0      2180   13288201    3418939.4  cudaStreamSynchronize                     
      0.1        513058318    6216556        82.5        70.0        50     569981        434.8  cuKernelGetName                           
      0.1        472974538        466   1014966.8     14355.0      1170  101606255    8308569.2  cuMemMap                                  
      0.1        460349446        354   1300422.2     28955.0     13120  106878929    8648443.4  cuMemCreate                               
      0.0        376256813        112   3359435.8    808440.5     36850  139940356   14693897.9  cuMemExportToShareableHandle              
      0.0        338469135      25370     13341.3      3190.0      1350    1391921      34409.9  cudaMemsetAsync                           
      0.0        329431588      70532      4670.7       690.0       380  107339439     566532.5  cudaEventRecord                           
      0.0        325595324    3784302        86.0        60.0        50     160170        360.8  cuTensorMapReplaceAddress                 
      0.0        300652300        478    628979.7     67170.0     37570  135643213    6267105.3  cuMemSetAccess                            
      0.0        289408028         22  13154910.4    294725.0     45130  125888945   35632551.6  cuLibraryLoadData                         
      0.0        245125944        112   2188624.5     68335.0     43080  136500224   14366183.2  cuMemImportFromShareableHandle            
      0.0        238660480         30   7955349.3      4235.0      1850  119040439   30178437.4  cudaMalloc                                
      0.0        109871227     699930       157.0        90.0        70      73500        466.8  cuTensorMapEncodeTiled                    
      0.0         93426067          4  23356516.8  22666193.5     11830   48081850   26979605.1  cuMulticastAddDevice                      
      0.0         93409308         16   5838081.8    229465.0     33370   45088628   15283552.1  cuMulticastBindMem                        
      0.0         81497455      18968      4296.6      2880.0      2260    7280156      72580.7  cuLaunchKernelEx                          
      0.0         69729073      70540       988.5       650.0       430     120830       1705.3  cudaStreamWaitEvent                       
      0.0         45008188         10   4500818.8    212395.0    110030   21719908    9012735.0  cudaDeviceSynchronize                     
      0.0         20529826         36    570272.9    158420.0     25350    3415353     881813.1  cuMemUnmap                                
      0.0         17979329       2018      8909.5      8540.0      1810     145770       4740.1  cudaEventSynchronize                      
      0.0         17445765         52    335495.5       260.0        70    3305853     817277.7  cuMemRelease                              
      0.0         11220958        798     14061.4       630.0       200     766420      70800.3  cudaEventCreateWithFlags                  
      0.0         10571182      37936       278.7       200.0       150      64310        615.9  cudaStreamGetCaptureInfo_v12030           
      0.0         10269918      18968       541.4       330.0       260      66340       1157.3  cudaGetFuncBySymbol_v11000                
      0.0          9902639      72031       137.5       110.0        70      39710        425.5  cudaGetDeviceProperties_v12000            
      0.0          8649508         38    227618.6      6770.0      3080    6040345    1039288.3  cudaStreamCreateWithFlags                 
      0.0          3422344        256     13368.5     12590.0      4660      43550       4463.0  cudaMemcpy                                
      0.0          3081383         58     53127.3      2235.0      1490    1625632     259763.7  cudaHostAlloc                             
      0.0          2985112          4    746278.0    378665.0      4430    2223352    1045735.8  cudaMallocFromPoolAsync_v11020            
      0.0          2636875        440      5992.9      2790.0       920      66080       8491.3  cuMemAddressReserve                       
      0.0          1715731        146     11751.6      6410.0      1870      42390      10519.0  cuMemsetD32Async                          
      0.0           678530         12     56544.2     47400.0     22410     142880      40595.1  cuMulticastUnbind                         
      0.0           536400         72      7450.0      7420.0      4890      16800       1419.9  cudaEventQuery                            
      0.0           492981        784       628.8       460.0       200       8210        526.7  cudaEventDestroy                          
      0.0           325540         32     10173.1      7740.0      5130      30900       5975.1  cudaStreamDestroy                         
      0.0           306700          4     76675.0     64265.0     43390     134780      41303.7  cudaMemPoolCreate_v11020                  
      0.0           264120        992       266.3       180.0       140      23840        776.3  cudaThreadExchangeStreamCaptureMode_v10010
      0.0           190020         36      5278.3      4065.0      2970      17010       3181.5  cudaFreeHost                              
      0.0           188700        914       206.5       160.0        60       2620        153.2  cuGetProcAddress_v2                       
      0.0            88200         56      1575.0       790.0       310      11180       2206.7  cuLibraryGetKernel                        
      0.0            80940         10      8094.0      5765.0      3800      16270       4800.9  cudaFree                                  
      0.0            79780        438       182.1       130.0        50       6150        357.7  cuMemGetAllocationGranularity             
      0.0            61820         12      5151.7      3570.0      2640      15090       4119.6  cuMemAddressFree                          
      0.0            38350          2     19175.0     19175.0     17560      20790       2284.0  cudaStreamCreateWithPriority              
      0.0            29010          2     14505.0     14505.0     13540      15470       1364.7  cuMulticastCreate                         
      0.0            24000         24      1000.0       780.0       490       2690        544.2  cuMemRetainAllocationHandle               
      0.0            14970         74       202.3       195.0       150        470         48.2  cudaGetDriverEntryPointByVersion_v12050   
      0.0             3720          2      1860.0      1860.0      1690       2030        240.4  cuInit                                    
      0.0             3550          2      1775.0      1775.0      1530       2020        346.5  cudaMemPoolSetAttribute_v11020            
      0.0             1860          4       465.0       460.0       100        840        415.9  cuModuleGetLoadingMode                    
      0.0              630          2       315.0       315.0       250        380         91.9  cuMulticastGetGranularity                 

[4/6] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)   StdDev (ns)                                                  Name
 --------  ---------------  ---------  ---------  ---------  --------  ---------  -----------  ----------------------------------------------------------------------------------------------------
     18.2     161104126362     432000   372926.2   213567.0     64416    2011196     365887.2  void blublas_ws::bgemm_sm90_kernel<blublas_ws::Sm90WarpspecConfig<(int)128, (int)256, (int)64, (int…
     13.9     123298660631     347520   354795.9   210367.0     68960    3724150     445712.5  void blublas_ws::bgemm_sm90_kernel<blublas_ws::Sm90WarpspecConfig<(int)128, (int)256, (int)64, (int…
     13.0     114911671199     138240   831247.6  1002173.5    500381    2108058     283359.1  void blublas_ws::bgemm_sm90_kernel<blublas_ws::Sm90WarpspecConfig<(int)128, (int)256, (int)64, (int…
     12.2     107741532453      69120  1558760.6  1557820.0   1264756    2024850      49819.8  void OwnTensor::cuda::gqa_bwd_vj1d::gqa_bwd_vj1d<(int)64, (int)64, (int)128>(CUtensorMap_st, CUtens…
      6.0      53062206234    1267630    41859.4    36832.0      3968    3004847      54717.5  OwnTensor::sm90::vec4_convert_to_fp32(const __nv_bfloat16 *, float *, unsigned long)                
      4.6      40816059915     863720    47256.1    36256.0      3584    2580059      56431.6  OwnTensor::sm90::vec8_convert_to_bf16(const float *, __nv_bfloat16 *, unsigned long)                
      4.4      38643539524      69120   559079.0   552990.0    508315    1100957      47905.5  void blublas_ws::bgemm_sm90_kernel<blublas_ws::Sm90WarpspecConfig<(int)192, (int)128, (int)64, (int…
      3.5      30917231608      72000   429406.0   428863.0    356095     749055      17527.9  OwnTensor::cuda::gqa62::gqa_v62_sm90(CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, const __nv_bfl…
      3.3      29217541347     138240   211353.7   209695.0    185695     349215      11872.9  void blublas_ws::bgemm_sm90_kernel<blublas_ws::Sm90WarpspecConfig<(int)192, (int)192, (int)64, (int…
      2.4      21539227476      69120   311620.8   311295.0    309662     336350       1642.0  void OwnTensor::cuda::swiglu_bwd_v22_vec_kernel_sm90<float>(const T1 *, const T1 *, T1 *, int)      
      2.3      20088450829     763510    26310.7     8384.0       896     107167      28975.2  void OwnTensor::add_inplace_kernel_sm90<float, (int)2>(T1 *, const T1 *, unsigned long)             
      1.8      15608777715       8700  1794112.4   296959.0     32128  231067954   11123032.4  ncclSymkDevKernel_ReduceScatter_LDMC_sum_bf16(ncclSymkDevWorkArgs4K)                                
      1.5      13403060568      72000   186153.6   186208.0    183296     209279       1568.8  void OwnTensor::cuda::swiglu_fwd_vec_kernel_sm90<float>(const T1 *, T1 *, int)                      
      1.3      11285191124     140160    80516.5    80416.0     76416     347487       3874.4  void OwnTensor::cuda::rms_bwd_fused_block_sm90<float, (int)4, (int)2>(const T1 *, const T1 *, const…
      1.3      11245562882      72884   154294.0   139104.0    137728     442591      62970.9  void OwnTensor::add_inplace_kernel_flat_sm90<float, (bool)0>(T1 *, const T1 *, unsigned long, unsig…
      1.2      10599338736     144000    73606.5    73760.0     71423     106976       1173.0  void OwnTensor::add_kernel<float, (int)2>(const T1 *, const T1 *, T1 *, unsigned long)              
      1.2      10245758469     138240    74115.7    73248.0     63296     152992       6855.7  void blublas_ws::bgemm_sm90_kernel<blublas_ws::Sm90WarpspecConfig<(int)192, (int)128, (int)64, (int…
      1.0       9192104931      69120   132987.6   132736.0    127616     449631       5798.4  void OwnTensor::cuda::fused_qk_norm_rope_bwd_kernel_sm90<(bool)1, (int)128>(const float *, const fl…
      1.0       9021145705     146000    61788.7    61632.0     58175     151456       2317.9  void OwnTensor::cuda::rms_norm_fwd_staged_kernel_sm90<float>(const T1 *, const T1 *, T1 *, float *,…
      0.8       6686017098       2000  3343008.5  3337256.0   3142562    4108596      84973.8  void blublas_ws::bgemm_sm90_kernel<blublas_ws::Sm90WarpspecConfig<(int)256, (int)128, (int)64, (int…
      0.7       6310172155       1920  3286548.0  3277049.5   3002666    3820613      88004.7  void blublas_ws::bgemm_sm90_kernel<blublas_ws::Sm90WarpspecConfig<(int)256, (int)128, (int)64, (int…
      0.6       5294209367      72000    73530.7    73536.0     70880     134208       1998.6  void OwnTensor::cuda::fused_qknorm_rope_forward_t_sm90<(bool)1, (int)128, (int)4, (int)8>(const flo…
      0.6       5198089684     138240    37601.9    37408.0     35807      59232        933.0  void OwnTensor::add_inplace_kernel_sm90<OwnTensor::bfloat16_t, (int)2>(T1 *, const T1 *, unsigned l…
      0.6       4881363011     282240    17295.1    21648.0      7296      36960       8616.0  void OwnTensor::sm90::strided_inner_vec_copy_kernel<uint4, unsigned short, (int)8, (int)3, unsigned…
      0.6       4871163911     138240    35237.0    39840.0     17759      65024      16659.5  void OwnTensor::sm90::strided_inner_vec_copy_kernel<float4, float, (int)4, (int)3, unsigned int>(co…
      0.3       2881945603     138240    20847.4    23408.0      9408      54944       9826.9  OwnTensor::cuda::reduce_dgamma_kernel(const float *, float *, int, int)                             
      0.3       2836915062     138240    20521.7    20896.0     18624      34688       1168.9  OwnTensor::cuda::gqa_bwd_vj1d::gqa_dkdv_greduce(const __nv_bfloat16 *, __nv_bfloat16 *, int, int, i…
      0.3       2755038167       1920  1434915.7  1434924.0   1342838    1619483      24324.3  OwnTensor::cuda::embedding_backward_kernel_cooperative(const unsigned short *, const float *, float…
      0.3       2572972838      69120    37224.7    36800.0     35775      48191        757.8  OwnTensor::cuda::gqa_bwd_vj1d::convert_dq_accum_to_bf16_v6(const float4 *, uint2 *, long)           
      0.3       2532944794      69120    36645.6    36320.0     34208      82400        900.3  OwnTensor::cuda::gqa_bwd_vj1d::compute_drowsum_v22(const __nv_bfloat16 *, const __nv_bfloat16 *, fl…
      0.3       2419506573       9420   256847.8   174079.0     24128    1393756     213483.0  ncclSymkDevKernel_AllGather_STMC(ncclSymkDevWorkArgs4K)                                             
      0.2       1592873711       1920   829621.7   829662.0    823134     836381       1947.3  void OwnTensor::cuda::sparseCENormalize_from_stats<float, unsigned short>(const T1 *, const T2 *, T…
      0.1        827234494        150  5514896.6   549678.5      3488   13281831    6329722.8  OwnTensor::cuda::multi_tensor_adamw_sm90_kernel(const OwnTensor::cuda::AdamTensorMeta *, const int …
      0.1        763669130       1920   397744.3   397631.0    390461     410687       1698.9  void OwnTensor::cuda::sparse_ce_forward_kernel_vec_save_stats<float, unsigned short>(const T1 *, co…
      0.1        484089016     140160     3453.8     3424.0      2912       5632        186.7  OwnTensor::cuda::rms_bwd_gamma_finalize_sm90(const float *, float *, int, int)                      
      0.0        348820173        120  2906834.8  2892760.5    207998    5644628    2707936.3  OwnTensor::cuda::multi_tensor_scale_kernel(const OwnTensor::cuda::TensorInfo *, const long *, int, …
      0.0        184376244        720   256078.1   163648.0    133055    2608761     465633.3  ncclSymkDevKernel_ReduceScatter_LDMC_sum_f32(ncclSymkDevWorkArgs4K)                                 
      0.0        106273347        120   885611.2   882349.5     78815    1697661     808139.2  OwnTensor::cuda::multi_tensor_grad_norm_kernel(const OwnTensor::cuda::TensorInfo *, const long *, i…
      0.0         72808626       2000    36404.3    36224.0     31232      46720       1708.2  OwnTensor::cuda::embedding_forward_kernel_vectorized(const unsigned short *, const float *, float *…
      0.0         31790548         80   397381.8   396335.0    393437     434463       5029.6  void OwnTensor::cuda::sparse_ce_forward_kernel_vec<float, unsigned short>(const T1 *, const T2 *, T…
      0.0          9254777        128    72302.9     6768.0      5536     760606     143591.2  ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)              
      0.0          6926797       4000     1731.7     1760.0      1120       3200        316.4  void OwnTensor::cuda::sum_reduction_kernel<float>(const T1 *, T1 *, long)                           
      0.0          2860022       2000     1430.0     1472.0      1088       2368        147.6  void OwnTensor::scalar_div_copy<float, float>(const T1 *, double, T2 *, unsigned long)              
      0.0           276351        120     2302.9     2336.0      2080       2624        170.4  OwnTensor::cuda::grad_norm_cleanup_kernel(const float *, int, float *)                              
      0.0           109184         60     1819.7     1600.0      1184      10336       1231.2  void OwnTensor::scalar_mul_inplace<float>(T1 *, double, unsigned long)                              
      0.0             6912          2     3456.0     3456.0      3456       3456          0.0  OwnTensor::cuda::rope_build_cache_kernel(float *, long, long, float)                                

[5/6] Executing 'cuda_gpu_mem_time_sum' stats report

 Time (%)  Total Time (ns)  Count  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)            Operation
 --------  ---------------  -----  --------  --------  --------  --------  -----------  ------------------------------
     59.9       2233838221  94638   23604.0   23360.0       704   1838254      27315.5  [CUDA memset]                 
     25.0        933447055   6166  151386.2    4864.0       704  36455151    1023094.2  [CUDA memcpy Host-to-Device]  
     15.0        560160644  44720   12526.0    4512.0       640    335007      23799.7  [CUDA memcpy Device-to-Device]
      0.0           669567    268    2498.4    2240.0      2048      6912        762.3  [CUDA memcpy Device-to-Host]  

[6/6] Executing 'cuda_gpu_mem_size_sum' stats report

 Total (MB)   Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)            Operation
 -----------  -----  --------  --------  --------  --------  -----------  ------------------------------
 9152812.422  94638    96.714   100.663     0.000  7556.825      118.798  [CUDA memset]
 1050336.018  44720    23.487     6.291     0.001   618.136       48.459  [CUDA memcpy Device-to-Device]
   16417.961   6166     2.663     0.016     0.000   618.136       17.465  [CUDA memcpy Host-to-Device]  
       0.001    268     0.000     0.000     0.000     0.000        0.000  [CUDA memcpy Device-to-Host]  

Generated:
 /root/gautam_k/blublas/h228_4b/h228_hybrid.nsys-rep
 /root/gautam_k/blublas/h228_4b/h228_hybrid.sqlite
