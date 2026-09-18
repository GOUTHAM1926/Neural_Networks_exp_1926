
[OK] libprofiler is up to date

[BUILD] BluBridge-BLAS (sm_90)
/usr/bin/make -C Tensor-Implementations blas SM_ARCH=90
make[1]: Entering directory '/root/gautam_k/blublas/blutrain_ours/Tensor-Implementations'
\n\nlibblublas is up-to-date.
make[1]: Leaving directory '/root/gautam_k/blublas/blutrain_ours/Tensor-Implementations'

[BUILD] Tensor-Implementations (sm_90, backend=cuBLAS)
/usr/bin/make -C Tensor-Implementations tensor SM_ARCH=90
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
  BLAS:          cuBLAS
  Profiler libs: Profiler/lib/libprofiler.{a,so}
  Tensor libs:   Tensor-Implementations/lib/libtensor.{a,so}
  BluBLAS lib:   Tensor-Implementations/lib/libblublas.so
  Dist libs:     dist/lib/libdist.{a,so}

[COMPILE] bluscript_zero.cpp (BLAS: cuBLAS)
/usr/bin/mpic++ -std=c++2a -fPIC -O3 -g -DWITH_CUDA -fopenmp -DUSE_FUSED_RMS_ROPE=1 -ITensor-Implementations/include -IProfiler/include -I/usr/local/cuda/include -Idist -Idist/FSDP -Idist/Tensor-Parallelism -Idist/Tensor-Parallelism/tensor -Idist/Tensor-Parallelism/dnn -Idist/Tensor-Parallelism/bridge -Idist/Tensor-Parallelism/memory -Idist/Data-Parallel/include -Idist/communication/include -Idist/zero/include -Idist/Distributed_Checkpointing/include -I/opt/nvidia/nsight-compute/2025.3.1/host/target-linux-x64/nvtx/include -o _runner_ bluscript_zero.cpp dist/FSDP/fsdp_init.cpp dist/FSDP/fsdp_comm.cpp dist/FSDP/fsdp_ddp.cpp dist/zero/src/ShardedOptimizer.cpp dist/zero/src/SymmetricShardedOptimizer.cpp -LTensor-Implementations/lib -LProfiler/lib -Ldist/lib -L/usr/local/cuda/lib64 -L/usr/lib/x86_64-linux-gnu -Wl,-rpath,/root/gautam_k/blublas/blutrain_ours/Tensor-Implementations/lib -Wl,-rpath,/root/gautam_k/blublas/blutrain_ours/Profiler/lib -Wl,-rpath,/root/gautam_k/blublas/blutrain_ours/dist/lib -Wl,-rpath,/usr/lib/x86_64-linux-gnu -ldist -ltensor -lprofiler  -lcuda -lcudart -lcublas -lcurand -lnccl -lmpi -lgomp -lpthread -ldl -lz
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

  [val] step 0 | val_loss 11.437630
step      0 | train 11.436555 | val 11.437630 | tok/s 35677.88 | norm 22.3192 | dt 14695.04 ms | lr 1.0000e-04 | mfu 52.34% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 22.2539 | G1(float32/154M) 1.7064
step      1 | train 13.860121 | val 11.437630 | tok/s 36274.43 | norm 844.6154 | dt 14453.38 ms | lr 2.0000e-04 | mfu 53.22% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 844.4162 | G1(float32/154M) 18.3451
step      2 | train 16.731327 | val 11.437630 | tok/s 36393.48 | norm 276.8495 | dt 14406.10 ms | lr 3.0000e-04 | mfu 53.39% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 276.7904 | G1(float32/154M) 5.7187
step      3 | train 13.910307 | val 11.437630 | tok/s 36413.01 | norm 276.8095 | dt 14398.37 ms | lr 3.0000e-04 | mfu 53.42% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 276.3020 | G1(float32/154M) 16.7539
step      4 | train 13.329878 | val 11.437630 | tok/s 36281.79 | norm 7.3388 | dt 14450.44 ms | lr 3.0000e-04 | mfu 53.23% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 7.3368 | G1(float32/154M) 0.1717
step      5 | train 17.050310 | val 11.437630 | tok/s 36378.72 | norm 56.9346 | dt 14411.94 ms | lr 3.0000e-04 | mfu 53.37% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 56.9239 | G1(float32/154M) 1.1033
step      6 | train 13.151763 | val 11.437630 | tok/s 36348.72 | norm 41.3994 | dt 14423.84 ms | lr 3.0000e-04 | mfu 53.33% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 41.3905 | G1(float32/154M) 0.8565
step      7 | train 10.987839 | val 11.437630 | tok/s 36326.63 | norm 11.6377 | dt 14432.61 ms | lr 3.0000e-04 | mfu 53.30% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 11.6350 | G1(float32/154M) 0.2512
step      8 | train 10.784580 | val 11.437630 | tok/s 36299.20 | norm 4.4775 | dt 14443.51 ms | lr 3.0000e-04 | mfu 53.26% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 4.4767 | G1(float32/154M) 0.0876
step      9 | train 10.151375 | val 11.437630 | tok/s 36334.41 | norm 5.2489 | dt 14429.52 ms | lr 3.0000e-04 | mfu 53.31% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 5.2479 | G1(float32/154M) 0.1042
step     10 | train 9.425695 | val 11.437630 | tok/s 36376.71 | norm 4.5753 | dt 14412.74 ms | lr 3.0000e-04 | mfu 53.37% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 4.5746 | G1(float32/154M) 0.0832
step     11 | train 8.826727 | val 11.437630 | tok/s 36411.81 | norm 4.1332 | dt 14398.84 ms | lr 3.0000e-04 | mfu 53.42% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 4.1326 | G1(float32/154M) 0.0710
step     12 | train 8.274923 | val 11.437630 | tok/s 36462.75 | norm 3.7611 | dt 14378.73 ms | lr 3.0000e-04 | mfu 53.50% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 3.7606 | G1(float32/154M) 0.0646
step     13 | train 8.164083 | val 11.437630 | tok/s 36516.95 | norm 11.8795 | dt 14357.39 ms | lr 3.0000e-04 | mfu 53.58% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 11.8774 | G1(float32/154M) 0.2212
step     14 | train 8.303514 | val 11.437630 | tok/s 36519.76 | norm 14.4657 | dt 14356.28 ms | lr 3.0000e-04 | mfu 53.58% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 14.4631 | G1(float32/154M) 0.2745
step     15 | train 8.439135 | val 11.437630 | tok/s 36556.74 | norm 18.0140 | dt 14341.76 ms | lr 3.0000e-04 | mfu 53.63% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 18.0106 | G1(float32/154M) 0.3474
step     16 | train 8.496618 | val 11.437630 | tok/s 36503.90 | norm 19.9983 | dt 14362.52 ms | lr 3.0000e-04 | mfu 53.56% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 19.9949 | G1(float32/154M) 0.3693
step     17 | train 8.411270 | val 11.437630 | tok/s 36478.56 | norm 12.9631 | dt 14372.50 ms | lr 3.0000e-04 | mfu 53.52% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 12.9609 | G1(float32/154M) 0.2431
step     18 | train 8.278131 | val 11.437630 | tok/s 36447.24 | norm 3.7257 | dt 14384.85 ms | lr 3.0000e-04 | mfu 53.47% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 3.7247 | G1(float32/154M) 0.0860
step     19 | train 8.180685 | val 11.437630 | tok/s 36433.81 | norm 2.9511 | dt 14390.15 ms | lr 2.9453e-04 | mfu 53.45% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 2.9503 | G1(float32/154M) 0.0683
step     20 | train 8.112746 | val 11.437630 | tok/s 36444.17 | norm 4.7814 | dt 14386.06 ms | lr 2.7857e-04 | mfu 53.47% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 4.7806 | G1(float32/154M) 0.0912
step     21 | train 8.022301 | val 11.437630 | tok/s 36445.32 | norm 4.0611 | dt 14385.60 ms | lr 2.5341e-04 | mfu 53.47% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 4.0603 | G1(float32/154M) 0.0782
step     22 | train 8.000442 | val 11.437630 | tok/s 36437.85 | norm 3.6514 | dt 14388.56 ms | lr 2.2108e-04 | mfu 53.46% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 3.6507 | G1(float32/154M) 0.0722
step     23 | train 7.932590 | val 11.437630 | tok/s 36409.67 | norm 2.6866 | dt 14399.69 ms | lr 1.8421e-04 | mfu 53.42% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 2.6861 | G1(float32/154M) 0.0532
step     24 | train 7.894207 | val 11.437630 | tok/s 36401.60 | norm 1.9012 | dt 14402.88 ms | lr 1.4579e-04 | mfu 53.41% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 1.9009 | G1(float32/154M) 0.0372
step     25 | train 7.881781 | val 11.437630 | tok/s 36410.72 | norm 1.5621 | dt 14399.28 ms | lr 1.0892e-04 | mfu 53.42% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 1.5617 | G1(float32/154M) 0.0311
step     26 | train 7.851205 | val 11.437630 | tok/s 36404.06 | norm 1.7892 | dt 14401.91 ms | lr 7.6594e-05 | mfu 53.41% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 1.7889 | G1(float32/154M) 0.0355
step     27 | train 7.783315 | val 11.437630 | tok/s 36425.28 | norm 1.6136 | dt 14393.52 ms | lr 5.1431e-05 | mfu 53.44% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 1.6133 | G1(float32/154M) 0.0321
step     28 | train 7.817814 | val 11.437630 | tok/s 36413.51 | norm 1.3018 | dt 14398.17 ms | lr 3.5468e-05 | mfu 53.42% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 1.3015 | G1(float32/154M) 0.0269
  [val] step 29 | val_loss 7.973344
step     29 | train 7.822111 | val 7.973344 | tok/s 36451.08 | norm 1.0806 | dt 14383.33 ms | lr 3.0000e-05 | mfu 53.48% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 1.0804 | G1(float32/154M) 0.0227

==============================================================================
  STEADY-STATE SUMMARY — C++ ZeRO-1  world=2  fused_rms_rope=1  overlap=1
  (27 steps, first 3 excluded)
==============================================================================
  host  step time    :   14398.17 ms  (mean 14395.74)
  throughput         :    36413.5 tok/s   (36.4 Ktok/s)
  MFU                :      53.42 %
  tokens / step      :     524288
  peak mem allocated :     101.79 GiB
  peak mem reserved  :     103.18 GiB
==============================================================================


=== Training complete ===
                                                                                                                       
[1/6] [0%                          ] h228_hybrid.nsys-rep
[1/6] [0%                          ] h228_hybrid.nsys-rep
[1/6] [6%                          ] h228_hybrid.nsys-rep
[1/6] [5%                          ] h228_hybrid.nsys-rep
[1/6] [6%                          ] h228_hybrid.nsys-rep
[1/6] [5%                          ] h228_hybrid.nsys-rep
[1/6] [6%                          ] h228_hybrid.nsys-rep
[1/6] [5%                          ] h228_hybrid.nsys-rep
[1/6] [6%                          ] h228_hybrid.nsys-rep
[1/6] [9%                          ] h228_hybrid.nsys-rep
[1/6] [8%                          ] h228_hybrid.nsys-rep
[1/6] [9%                          ] h228_hybrid.nsys-rep
[1/6] [10%                         ] h228_hybrid.nsys-rep
[1/6] [9%                          ] h228_hybrid.nsys-rep
[1/6] [11%                         ] h228_hybrid.nsys-rep
[1/6] [12%                         ] h228_hybrid.nsys-rep
[1/6] [11%                         ] h228_hybrid.nsys-rep
[1/6] [10%                         ] h228_hybrid.nsys-rep
[1/6] [11%                         ] h228_hybrid.nsys-rep
[1/6] [10%                         ] h228_hybrid.nsys-rep
[1/6] [13%                         ] h228_hybrid.nsys-rep
[1/6] [12%                         ] h228_hybrid.nsys-rep
[1/6] [=15%                        ] h228_hybrid.nsys-rep
[1/6] [14%                         ] h228_hybrid.nsys-rep
[1/6] [=16%                        ] h228_hybrid.nsys-rep
[1/6] [=15%                        ] h228_hybrid.nsys-rep
[1/6] [==18%                       ] h228_hybrid.nsys-rep
[1/6] [=17%                        ] h228_hybrid.nsys-rep
[1/6] [==19%                       ] h228_hybrid.nsys-rep
[1/6] [==18%                       ] h228_hybrid.nsys-rep
[1/6] [==20%                       ] h228_hybrid.nsys-rep
[1/6] [==19%                       ] h228_hybrid.nsys-rep
[1/6] [==21%                       ] h228_hybrid.nsys-rep
[1/6] [==20%                       ] h228_hybrid.nsys-rep
[1/6] [===22%                      ] h228_hybrid.nsys-rep
[1/6] [==21%                       ] h228_hybrid.nsys-rep
[1/6] [===23%                      ] h228_hybrid.nsys-rep
[1/6] [===22%                      ] h228_hybrid.nsys-rep
[1/6] [===24%                      ] h228_hybrid.nsys-rep
[1/6] [===23%                      ] h228_hybrid.nsys-rep
[1/6] [===24%                      ] h228_hybrid.nsys-rep
[1/6] [====25%                     ] h228_hybrid.nsys-rep
[1/6] [===24%                      ] h228_hybrid.nsys-rep
[1/6] [====26%                     ] h228_hybrid.nsys-rep
[1/6] [====25%                     ] h228_hybrid.nsys-rep
[1/6] [====26%                     ] h228_hybrid.nsys-rep
[1/6] [====27%                     ] h228_hybrid.nsys-rep
[1/6] [====26%                     ] h228_hybrid.nsys-rep
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
[2/6] [=========[3/6] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls   Avg (ns)    Med (ns)   Min (ns)  Max (ns)   StdDev (ns)                        Name                      
 --------  ---------------  ---------  ----------  ----------  --------  ---------  -----------  ------------------------------------------------
     53.7     449709518075    4949276     90863.7     27610.0      2170   55186446     231719.9  cudaLaunchKernel                                
     31.3     261934436612    1286248    203642.2     36370.0      2270    8373357     403709.1  cuLaunchKernelEx                                
      9.6      80681186500     872170     92506.3     18270.0      1330    9378568     266212.7  cudaMemsetAsync                                 
      2.4      20196429820      50898    396802.0      7555.0      1720  149304794    6241311.2  cudaMemcpyAsync                                 
      1.3      10609121133       1480   7168325.1    286300.0       890  172953504   23711381.3  cudaMallocAsync_v11020                          
      0.8       6371008060      69122     92170.5     64110.0      3270    7564056     210747.6  cudaMemset                                      
      0.2       1611545787        906   1778748.1    138426.0      2230   13273991    3420941.3  cudaStreamSynchronize                           
      0.1       1163428467     917337      1268.3       980.0       380   45071678      48343.3  cudaEventRecord                                 
      0.1        840443890        466   1803527.7     16750.0      1250  738026765   34287421.3  cuMemMap                                        
      0.1        677156930        354   1912872.7     26445.0     14040  100808484   10400619.2  cuMemCreate                                     
      0.1        662666329        478   1386331.2     77480.0     41300   68937387    8217060.1  cuMemSetAccess                                  
      0.1        481357947        112   4297838.8    750786.0     44490  139553256   17103944.8  cuMemExportToShareableHandle                    
      0.1        453379604    4234000       107.1        90.0        60    1646252        899.8  cuStreamGetCaptureInfo_v2                       
      0.1        435179330    4949276        87.9        70.0        50    2432762       1155.0  cuKernelGetName                                 
      0.0        357859469         58   6169990.8    105070.0     17520  120553891   22197553.6  cuLibraryLoadData                               
      0.0        265611142         38   6989766.9      7365.0      3250  134185202   30020419.7  cudaStreamCreateWithFlags                       
      0.0        258541450    2534560       102.0        90.0        50     503041        522.3  cuKernelGetAttribute                            
      0.0        216449349         36   6012481.9    948015.5     39820  141463498   23460566.4  cuMemUnmap                                      
      0.0        215585809         32   6737056.5     18430.0       450  107943580   25833409.1  cuKernelSetAttribute                            
      0.0        156796750         12  13066395.8    612750.5     23640  136972934   39126154.1  cuMulticastUnbind                               
      0.0        136426915         42   3248259.9      4200.0      1860   67973357   14590192.2  cudaMalloc                                      
      0.0        111351014     699840       159.1        90.0        70      86880        450.0  cuTensorMapEncodeTiled                          
      0.0        104666276        112    934520.3     52905.0     33250   68081967    6440167.7  cuMemImportFromShareableHandle                  
      0.0         88704826         16   5544051.6    190180.0     34460   44398977   14706685.9  cuMulticastBindMem                              
      0.0         85239082          4  21309770.5  20278782.0     11790   44669728   24649906.9  cuMulticastAddDevice                            
      0.0         69780627      70540       989.2       640.0       430     236840       1821.6  cudaStreamWaitEvent                             
      0.0         44831635         10   4483163.5    169430.0     98990   21405637    8901277.4  cudaDeviceSynchronize                           
      0.0         34845130         52    670098.7       355.0       110   17445754    2522416.1  cuMemRelease                                    
      0.0         17862728       2018      8851.7      8500.0      1810      84420       4243.8  cudaEventSynchronize                            
      0.0         10360481      37936       273.1       200.0       150     157610        997.0  cudaStreamGetCaptureInfo_v12030                 
      0.0         10191133        870     11713.9       580.0       200     681951      62877.9  cudaEventCreateWithFlags                        
      0.0          9987906      72007       138.7       120.0        70      69180        518.6  cudaGetDeviceProperties_v12000                  
      0.0          9906771      18968       522.3       350.0       270      29051        599.1  cudaGetFuncBySymbol_v11000                      
      0.0          6658406          4   1664601.5    112210.5      5000    6428985    3177832.1  cudaMallocFromPoolAsync_v11020                  
      0.0          4378824         58     75497.0      2570.0      1430    2891912     407828.3  cudaHostAlloc                                   
      0.0          3596255        256     14047.9     12750.0      4700     129040       8668.6  cudaMemcpy                                      
      0.0          2665871        440      6058.8      2690.0       790      68550       8171.1  cuMemAddressReserve                             
      0.0          1843660        146     12627.8     11770.0      1910      85290      12921.7  cuMemsetD32Async                                
      0.0           550650         81      6798.1      6270.0      4900      25820       2907.5  cudaEventQuery                                  
      0.0           549222        784       700.5       500.0       210      27150       1066.9  cudaEventDestroy                                
      0.0           499721       2674       186.9       130.0        60      20010        444.2  cuGetProcAddress_v2                             
      0.0           489671          4    122417.8     93150.0     47890     255481      97307.6  cudaMemPoolCreate_v11020                        
      0.0           323300         32     10103.1      8155.0      5820      26820       4664.8  cudaStreamDestroy                               
      0.0           242680        992       244.6       180.0       140       9890        461.7  cudaThreadExchangeStreamCaptureMode_v10010      
      0.0           215851         36      5995.9      4150.0      3150      19200       4298.9  cudaFreeHost                                    
      0.0           166190         92      1806.4       610.0       170      11320       2642.2  cuLibraryGetKernel                              
      0.0           158170         14     11297.9      7520.0      1220      28040       9245.8  cudaFree                                        
      0.0            92840         12      7736.7      4285.0      3100      24410       7022.8  cuMemAddressFree                                
      0.0            77290        438       176.5       120.0        60       6050        373.4  cuMemGetAllocationGranularity                   
      0.0            42730         80       534.1       200.0       150      23980       2657.4  cudaGetDriverEntryPointByVersion_v12050         
      0.0            42090          2     21045.0     21045.0     18950      23140       2962.8  cudaStreamCreateWithPriority                    
      0.0            27150          2     13575.0     13575.0     11810      15340       2496.1  cuMulticastCreate                               
      0.0            27030         60       450.5       390.0       360       1930        249.9  cudaOccupancyMaxActiveClusters_v11070           
      0.0            24470         24      1019.6       880.0       480       2290        464.4  cuMemRetainAllocationHandle                     
      0.0            20630         10      2063.0      1800.0      1250       4720        994.8  cuInit                                          
      0.0             3870          2      1935.0      1935.0      1650       2220        403.1  cudaMemPoolSetAttribute_v11020                  
      0.0             3700          2      1850.0      1850.0      1670       2030        254.6  cudaOccupancyAvailableDynamicSMemPerBlock_v10200
      0.0             3420          8       427.5       115.0        80       1600        612.4  cuModuleGetLoadingMode                          
      0.0              530          2       265.0       265.0       240        290         35.4  cuMulticastGetGranularity                       

[4/6] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  ---------  ---------  --------  ---------  -----------  ----------------------------------------------------------------------------------------------------
     13.8     120683593069     347520   347270.9   209631.0     69248    3909822     426327.5  nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT                                                      
     12.9     112111037606     138240   810988.4   884715.5    495356    1344987     268323.9  nvjet_sm90_tst_128x256_64x4_2x1_v_bz_coopA_NTN                                                      
     12.1     105238064523      69120  1522541.4  1527261.0   1264151    1940795      40012.0  void OwnTensor::cuda::gqa_bwd_vj1d::gqa_bwd_vj1d<(int)64, (int)64, (int)128>(CUtensorMap_st, CUtens…
      9.2      79973104831      72000  1110737.6  1107708.5   1041078    1429340      35268.7  nvjet_sm90_tst_192x192_64x3_1x2_h_bz_coopB_NNN                                                      
      8.9      77680360382     360000   215778.8   201343.0     61504     676222     165638.4  nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_NNT                                                      
      6.1      52915374392    1267630    41743.5    36703.0      3968    3006064      54552.7  OwnTensor::sm90::vec4_convert_to_fp32(const __nv_bfloat16 *, float *, unsigned long)                
      4.7      40791983151     863720    47228.2    36287.0      3647    2585752      56417.7  OwnTensor::sm90::vec8_convert_to_bf16(const float *, __nv_bfloat16 *, unsigned long)                
      4.5      39641425531      69120   573516.0   574494.0    521436     714558      14762.7  nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT                                                      
      4.0      34823599375     140160   248456.0   206111.0    183679    3900286     337090.5  nvjet_sm90_tst_192x192_64x3_1x2_h_bz_coopB_NTN                                                      
      3.4      29493558152      72000   409632.8   407903.0    357663     751870      17401.3  OwnTensor::cuda::gqa62::gqa_v62_sm90(CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, const __nv_bfl…
      2.5      21536855350      69120   311586.4   311232.0    309599     334719       1609.7  void OwnTensor::cuda::swiglu_bwd_v22_vec_kernel_sm90<float>(const T1 *, const T1 *, T1 *, int)      
      2.3      20099725175     763510    26325.4     8352.0       928     103872      29017.9  void OwnTensor::add_inplace_kernel_sm90<float, (int)2>(T1 *, const T1 *, unsigned long)             
      1.8      15907284555       8700  1828423.5   313935.0     34432  226537237   11350918.6  ncclSymkDevKernel_ReduceScatter_LDMC_sum_bf16(ncclSymkDevWorkArgs4K)                                
      1.5      13400230640      72000   186114.3   186208.0    183231     209024       1373.0  void OwnTensor::cuda::swiglu_fwd_vec_kernel_sm90<float>(const T1 *, T1 *, int)                      
      1.3      11266792399     140160    80385.2    80256.0     76608     374622       3962.8  void OwnTensor::cuda::rms_bwd_fused_block_sm90<float, (int)4, (int)2>(const T1 *, const T1 *, const…
      1.3      11247485411      72884   154320.4   139071.0    137760     443965      63011.4  void OwnTensor::add_inplace_kernel_flat_sm90<float, (bool)0>(T1 *, const T1 *, unsigned long, unsig…
      1.2      10597096775     144000    73590.9    73759.0     71488      99039       1267.5  void OwnTensor::add_kernel<float, (int)2>(const T1 *, const T1 *, T1 *, unsigned long)              
      1.2      10114203282     138240    73164.1    72448.0     64864     149503       7658.0  nvjet_sm90_tst_128x192_64x5_2x1_v_bz_coopB_NTN                                                      
      1.0       9148728070      69120   132360.1   132096.0    128064     630686       4901.6  void OwnTensor::cuda::fused_qk_norm_rope_bwd_kernel_sm90<(bool)1, (int)128>(const float *, const fl…
      1.0       8954335898     146000    61331.1    61215.0     58400     175199       2252.1  void OwnTensor::cuda::rms_norm_fwd_staged_kernel_sm90<float>(const T1 *, const T1 *, T1 *, float *,…
      0.7       6504989129       2000  3252494.6  3235494.5   3138602    3971012      79250.0  nvjet_sm90_tst_192x192_64x3_2x1_v_bz_coopB_NNN                                                      
      0.6       5270911958      72000    73207.1    73216.0     70880     122720       1760.9  void OwnTensor::cuda::fused_qknorm_rope_forward_t_sm90<(bool)1, (int)128, (int)4, (int)8>(const flo…
      0.6       5192012994     138240    37558.0    37376.0     35775      57088        920.1  void OwnTensor::add_inplace_kernel_sm90<OwnTensor::bfloat16_t, (int)2>(T1 *, const T1 *, unsigned l…
      0.6       4861396693     138240    35166.4    42111.5     17760      88352      16652.5  void OwnTensor::sm90::strided_inner_vec_copy_kernel<float4, float, (int)4, (int)3, unsigned int>(co…
      0.6       4855018826     282240    17201.7    22192.0      7264      40991       8585.7  void OwnTensor::sm90::strided_inner_vec_copy_kernel<uint4, unsigned short, (int)8, (int)3, unsigned…
      0.3       2833143522     138240    20494.4    20928.0     18624      42368       1199.2  OwnTensor::cuda::gqa_bwd_vj1d::gqa_dkdv_greduce(const __nv_bfloat16 *, __nv_bfloat16 *, int, int, i…
      0.3       2740537256       1920  1427363.2  1426444.5   1363804    1623771      23548.5  OwnTensor::cuda::embedding_backward_kernel_cooperative(const unsigned short *, const float *, float…
      0.3       2736864881     138240    19797.9    22464.0      9184      56352       9396.0  OwnTensor::cuda::reduce_dgamma_kernel(const float *, float *, int, int)                             
      0.3       2562305232      69120    37070.4    36640.0     35616      52480        766.6  OwnTensor::cuda::gqa_bwd_vj1d::convert_dq_accum_to_bf16_v6(const float4 *, uint2 *, long)           
      0.3       2513600655      69120    36365.8    36000.0     34304      91456        902.3  OwnTensor::cuda::gqa_bwd_vj1d::compute_drowsum_v22(const __nv_bfloat16 *, const __nv_bfloat16 *, fl…
      0.3       2315995655       9420   245859.4   173663.0     24160    1330684     188294.7  ncclSymkDevKernel_AllGather_STMC(ncclSymkDevWorkArgs4K)                                             
      0.2       1590372081       1920   828318.8   828286.0    822494     862776       2127.8  void OwnTensor::cuda::sparseCENormalize_from_stats<float, unsigned short>(const T1 *, const T2 *, T…
      0.1        827253339        150  5515022.3   549950.0      3488   13267362    6329320.5  OwnTensor::cuda::multi_tensor_adamw_sm90_kernel(const OwnTensor::cuda::AdamTensorMeta *, const int …
      0.1        758415142       1920   395007.9   394911.0    391103     429596       1670.6  void OwnTensor::cuda::sparse_ce_forward_kernel_vec_save_stats<float, unsigned short>(const T1 *, co…
      0.1        477770797     140160     3408.8     3392.0      2912       5376        189.8  OwnTensor::cuda::rms_bwd_gamma_finalize_sm90(const float *, float *, int, int)                      
      0.0        348561830        120  2904681.9  2889095.5    207583    5635701    2706032.7  OwnTensor::cuda::multi_tensor_scale_kernel(const OwnTensor::cuda::TensorInfo *, const long *, int, …
      0.0        188064554        720   261200.8   163680.0    133183    2588825     479521.8  ncclSymkDevKernel_ReduceScatter_LDMC_sum_f32(ncclSymkDevWorkArgs4K)                                 
      0.0        106211766        120   885098.1   882541.5     78496    1697756     807935.0  OwnTensor::cuda::multi_tensor_grad_norm_kernel(const OwnTensor::cuda::TensorInfo *, const long *, i…
      0.0         72746318       2000    36373.2    36160.0     31296      46048       1718.5  OwnTensor::cuda::embedding_forward_kernel_vectorized(const unsigned short *, const float *, float *…
      0.0         31621297         80   395266.2   394685.5    389788     410205       3371.6  void OwnTensor::cuda::sparse_ce_forward_kernel_vec<float, unsigned short>(const T1 *, const T2 *, T…
      0.0         10988306        128    85846.1     6912.0      5440    1154421     180822.1  ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)              
      0.0          6568019       4000     1642.0     1664.0      1152       3040        285.7  void OwnTensor::cuda::sum_reduction_kernel<float>(const T1 *, T1 *, long)                           
      0.0          2745690       2000     1372.8     1376.0      1120       2592        131.0  void OwnTensor::scalar_div_copy<float, float>(const T1 *, double, T2 *, unsigned long)              
      0.0           276575        120     2304.8     2320.0      2080       2880        182.1  OwnTensor::cuda::grad_norm_cleanup_kernel(const float *, int, float *)                              
      0.0           104544         60     1742.4     1616.0      1152       8160        887.7  void OwnTensor::scalar_mul_inplace<float>(T1 *, double, unsigned long)                              
      0.0             6976          2     3488.0     3488.0      3488       3488          0.0  OwnTensor::cuda::rope_build_cache_kernel(float *, long, long, float)                                

[5/6] Executing 'cuda_gpu_mem_time_sum' stats report

 Time (%)  Total Time (ns)  Count   Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)            Operation           
 --------  ---------------  ------  --------  --------  --------  --------  -----------  ------------------------------
     66.3       3020158982  941438    3208.0     896.0       736   1831634      11018.1  [CUDA memset]                 
     21.3        968567363    6166  157082.0    4864.0       704  43134979    1104886.9  [CUDA memcpy Host-to-Device]  
     12.4        564979458   44720   12633.7    4480.0       671    338975      23906.8  [CUDA memcpy Device-to-Device]
      0.0           677948     268    2529.7    2240.0      2144      6848        829.9  [CUDA memcpy Device-to-Host]  

[6/6] Executing 'cuda_gpu_mem_size_sum' stats report

 Total (MB)   Count   Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)            Operation           
 -----------  ------  --------  --------  --------  --------  -----------  ------------------------------
 9152815.809  941438     9.722     0.000     0.000  7556.825       47.586  [CUDA memset]                 
 1050336.018   44720    23.487     6.291     0.001   618.136       48.459  [CUDA memcpy Device-to-Device]
   16417.961    6166     2.663     0.016     0.000   618.136       17.465  [CUDA memcpy Host-to-Device]  
       0.001     268     0.000     0.000     0.000     0.000        0.000  [CUDA memcpy Device-to-Host]  

Generated:
	/root/gautam_k/blublas/h228_4b/h228_hybrid.nsys-rep
	/root/gautam_k/blublas/h228_4b/h228_hybrid.sqlite
