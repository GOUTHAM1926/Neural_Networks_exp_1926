
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
step      0 | train 11.436555 | val 11.437630 | tok/s 35291.19 | norm 22.3186 | dt 14856.06 ms | lr 1.0000e-04 | mfu 51.78% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 22.2532 | G1(float32/154M) 1.7063
step      1 | train 13.864397 | val 11.437630 | tok/s 36391.09 | norm 844.8801 | dt 14407.04 ms | lr 2.0000e-04 | mfu 53.39% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 844.6809 | G1(float32/154M) 18.3479
step      2 | train 16.731552 | val 11.437630 | tok/s 36354.72 | norm 276.9637 | dt 14421.46 ms | lr 3.0000e-04 | mfu 53.34% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 276.9044 | G1(float32/154M) 5.7304
step      3 | train 13.960423 | val 11.437630 | tok/s 36356.41 | norm 281.4689 | dt 14420.78 ms | lr 3.0000e-04 | mfu 53.34% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 280.9580 | G1(float32/154M) 16.9511
step      4 | train 13.340696 | val 11.437630 | tok/s 36268.13 | norm 7.3717 | dt 14455.89 ms | lr 3.0000e-04 | mfu 53.21% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 7.3699 | G1(float32/154M) 0.1654
step      5 | train 17.043840 | val 11.437630 | tok/s 36367.89 | norm 56.9546 | dt 14416.24 ms | lr 3.0000e-04 | mfu 53.36% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 56.9439 | G1(float32/154M) 1.1039
step      6 | train 13.118275 | val 11.437630 | tok/s 36323.75 | norm 40.8258 | dt 14433.75 ms | lr 3.0000e-04 | mfu 53.29% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 40.8170 | G1(float32/154M) 0.8458
step      7 | train 10.952916 | val 11.437630 | tok/s 36298.57 | norm 10.7907 | dt 14443.77 ms | lr 3.0000e-04 | mfu 53.26% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 10.7882 | G1(float32/154M) 0.2326
step      8 | train 10.757318 | val 11.437630 | tok/s 36290.65 | norm 4.5012 | dt 14446.92 ms | lr 3.0000e-04 | mfu 53.24% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 4.5004 | G1(float32/154M) 0.0876
step      9 | train 10.111529 | val 11.437630 | tok/s 36318.80 | norm 5.3033 | dt 14435.72 ms | lr 3.0000e-04 | mfu 53.29% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 5.3022 | G1(float32/154M) 0.1048
step     10 | train 9.376067 | val 11.437630 | tok/s 36364.91 | norm 4.6301 | dt 14417.42 ms | lr 3.0000e-04 | mfu 53.35% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 4.6294 | G1(float32/154M) 0.0838
step     11 | train 8.771112 | val 11.437630 | tok/s 36377.80 | norm 4.1650 | dt 14412.31 ms | lr 3.0000e-04 | mfu 53.37% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 4.1644 | G1(float32/154M) 0.0709
step     12 | train 8.215862 | val 11.437630 | tok/s 36324.47 | norm 3.5990 | dt 14433.46 ms | lr 3.0000e-04 | mfu 53.29% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 3.5985 | G1(float32/154M) 0.0611
step     13 | train 8.200764 | val 11.437630 | tok/s 36514.45 | norm 12.4506 | dt 14358.37 ms | lr 3.0000e-04 | mfu 53.57% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 12.4484 | G1(float32/154M) 0.2312
step     14 | train 8.479906 | val 11.437630 | tok/s 36511.39 | norm 17.9118 | dt 14359.57 ms | lr 3.0000e-04 | mfu 53.57% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 17.9085 | G1(float32/154M) 0.3431
step     15 | train 8.399956 | val 11.437630 | tok/s 36465.70 | norm 12.8619 | dt 14377.57 ms | lr 3.0000e-04 | mfu 53.50% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 12.8595 | G1(float32/154M) 0.2473
step     16 | train 8.462440 | val 11.437630 | tok/s 36477.85 | norm 18.4793 | dt 14372.78 ms | lr 3.0000e-04 | mfu 53.52% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 18.4761 | G1(float32/154M) 0.3417
step     17 | train 8.394936 | val 11.437630 | tok/s 36441.74 | norm 11.2106 | dt 14387.02 ms | lr 3.0000e-04 | mfu 53.47% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 11.2086 | G1(float32/154M) 0.2129
step     18 | train 8.288635 | val 11.437630 | tok/s 36391.63 | norm 3.7762 | dt 14406.83 ms | lr 3.0000e-04 | mfu 53.39% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 3.7752 | G1(float32/154M) 0.0877
step     19 | train 8.201155 | val 11.437630 | tok/s 36406.94 | norm 3.1941 | dt 14400.77 ms | lr 2.9453e-04 | mfu 53.41% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 3.1932 | G1(float32/154M) 0.0721
step     20 | train 8.110281 | val 11.437630 | tok/s 36405.54 | norm 3.1064 | dt 14401.33 ms | lr 2.7857e-04 | mfu 53.41% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 3.1058 | G1(float32/154M) 0.0632
step     21 | train 8.012595 | val 11.437630 | tok/s 36416.96 | norm 4.4288 | dt 14396.81 ms | lr 2.5341e-04 | mfu 53.43% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 4.4280 | G1(float32/154M) 0.0869
step     22 | train 7.976705 | val 11.437630 | tok/s 36382.60 | norm 2.9491 | dt 14410.41 ms | lr 2.2108e-04 | mfu 53.38% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 2.9485 | G1(float32/154M) 0.0590
step     23 | train 7.922859 | val 11.437630 | tok/s 36374.03 | norm 3.1341 | dt 14413.80 ms | lr 1.8421e-04 | mfu 53.37% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 3.1335 | G1(float32/154M) 0.0622
step     24 | train 7.904049 | val 11.437630 | tok/s 36398.53 | norm 2.4117 | dt 14404.10 ms | lr 1.4579e-04 | mfu 53.40% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 2.4112 | G1(float32/154M) 0.0487
step     25 | train 7.898573 | val 11.437630 | tok/s 36397.99 | norm 2.1427 | dt 14404.31 ms | lr 1.0892e-04 | mfu 53.40% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 2.1422 | G1(float32/154M) 0.0426
step     26 | train 7.865810 | val 11.437630 | tok/s 36396.42 | norm 1.9497 | dt 14404.94 ms | lr 7.6594e-05 | mfu 53.40% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 1.9493 | G1(float32/154M) 0.0388
step     27 | train 7.794263 | val 11.437630 | tok/s 36388.87 | norm 1.4540 | dt 14407.92 ms | lr 5.1431e-05 | mfu 53.39% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 1.4537 | G1(float32/154M) 0.0292
step     28 | train 7.826425 | val 11.437630 | tok/s 36380.93 | norm 1.0810 | dt 14411.06 ms | lr 3.5468e-05 | mfu 53.38% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 1.0808 | G1(float32/154M) 0.0228
  [val] step 29 | val_loss 7.981514
step     29 | train 7.831325 | val 7.981514 | tok/s 36409.56 | norm 1.1135 | dt 14399.74 ms | lr 3.0000e-05 | mfu 53.42% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 1.1132 | G1(float32/154M) 0.0226

==============================================================================
  STEADY-STATE SUMMARY — C++ ZeRO-1  world=2  fused_rms_rope=1  overlap=1
  (27 steps, first 3 excluded)
==============================================================================
  host  step time    :   14407.92 ms  (mean 14408.65)
  throughput         :    36388.9 tok/s   (36.4 Ktok/s)
  MFU                :      53.39 %
  tokens / step      :     524288
  peak mem allocated :     101.79 GiB
  peak mem reserved  :     103.18 GiB
==============================================================================


=== Training complete ===
Collecting data...
Generating '/tmp/nsys-report-ecda.qdstrm'

[1/6] [0%                          ] h228_hybrid.nsys-rep
[1/6] [0%                          ] h228_hybrid.nsys-rep
[1/6] [8%                          ] h228_hybrid.nsys-rep
[1/6] [7%                          ] h228_hybrid.nsys-rep
[1/6] [8%                          ] h228_hybrid.nsys-rep
[1/6] [7%                          ] h228_hybrid.nsys-rep
[1/6] [10%                         ] h228_hybrid.nsys-rep
[1/6] [9%                          ] h228_hybrid.nsys-rep
[1/6] [10%                         ] h228_hybrid.nsys-rep
[1/6] [9%                          ] h228_hybrid.nsys-rep
[1/6] [12%                         ] h228_hybrid.nsys-rep
[1/6] [11%                         ] h228_hybrid.nsys-rep
[1/6] [12%                         ] h228_hybrid.nsys-rep
[1/6] [14%                         ] h228_hybrid.nsys-rep
[1/6] [=15%                        ] h228_hybrid.nsys-rep
[1/6] [=17%                        ] h228_hybrid.nsys-rep
[1/6] [=16%                        ] h228_hybrid.nsys-rep
[1/6] [==18%                       ] h228_hybrid.nsys-rep
[1/6] [==19%                       ] h228_hybrid.nsys-rep
[1/6] [==20%                       ] h228_hybrid.nsys-rep
[1/6] [==21%                       ] h228_hybrid.nsys-rep
[1/6] [===22%                      ] h228_hybrid.nsys-rep
[1/6] [===23%                      ] h228_hybrid.nsys-rep
[1/6] [===22%                      ] h228_hybrid.nsys-rep
[1/6] [===24%                      ] h228_hybrid.nsys-rep
[1/6] [===23%                      ] h228_hybrid.nsys-rep
[1/6] [====25%                     ] h228_hybrid.nsys-rep
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
[1/6] [=====30%                    ] h228_hybrid.nsys-rep
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

 Time (%)  Total Time (ns)  Num Calls   Avg (ns)    Med (ns)   Min (ns)  Max (ns)   StdDev (ns)                        Name                      
 --------  ---------------  ---------  ----------  ----------  --------  ---------  -----------  ------------------------------------------------
     53.7     449766782268    4949276     90875.3     27660.0      2180   54622135     230686.1  cudaLaunchKernel                                
     31.3     262183900631    1286248    203836.2     36900.0      2270    9547588     403359.2  cuLaunchKernelEx                                
      9.6      80813208668     872170     92657.6     18230.0      1340   19858277     268330.9  cudaMemsetAsync                                 
      2.4      20229645158      50898    397454.6      7880.0      1770  149450255    6245131.3  cudaMemcpyAsync                                 
      1.3      11091638831       1480   7494350.6    309625.0       930  192761040   24266287.6  cudaMallocAsync_v11020                          
      0.7       6276074241      69122     90797.1     64170.0      3260    7212826     198512.6  cudaMemset                                      
      0.2       1655555773        906   1827324.3    101305.0      2160   36482040    3624439.2  cudaStreamSynchronize                           
      0.1       1253109957     917335      1366.0      1000.0       380  126204765     134081.9  cudaEventRecord                                 
      0.1        777375882        354   2195977.1     24840.0     14260  128766818   12763794.6  cuMemCreate                                     
      0.1        451146711    4234000       106.6        80.0        60     539260        505.5  cuStreamGetCaptureInfo_v2                       
      0.1        442040222        478    924770.3     79530.0     41990   68785687    6197627.1  cuMemSetAccess                                  
      0.0        415547991    4949276        84.0        70.0        50    2950543       1440.5  cuKernelGetName                                 
      0.0        362526671        112   3236845.3    788580.0     45180  141514338   14807955.0  cuMemExportToShareableHandle                    
      0.0        356645269         58   6149056.4     81280.0     17890  118678819   22302308.3  cuLibraryLoadData                               
      0.0        270598046        466    580682.5     16840.0      1250  101306004    6222894.9  cuMemMap                                        
      0.0        248545855    2534560        98.1        80.0        50     569240        551.9  cuKernelGetAttribute                            
      0.0        236141946         32   7379435.8     18515.0       480  116022097   28475287.1  cuKernelSetAttribute                            
      0.0        135411674         38   3563465.1      7260.0      3140  131414920   21309239.9  cudaStreamCreateWithFlags                       
      0.0        116971428     699840       167.1        90.0        70      88250        546.0  cuTensorMapEncodeTiled                          
      0.0         93075737         16   5817233.6    126735.0     33240   45516888   15452572.2  cuMulticastBindMem                              
      0.0         89327155          4  22331788.8  22271228.5     12360   44772338   25752071.7  cuMulticastAddDevice                            
      0.0         70474377      70540       999.1       650.0       420     314040       2181.0  cudaStreamWaitEvent                             
      0.0         66536045         12   5544670.4     69385.5     30630   32966067   12796251.7  cuMulticastUnbind                               
      0.0         53590824        112    478489.5     76920.0     33690   11367380    1388822.3  cuMemImportFromShareableHandle                  
      0.0         43089544         10   4308954.4    214305.0    109590   21082257    8694840.0  cudaDeviceSynchronize                           
      0.0         41393295         36   1149813.8    231080.5     33410    6886726    1835530.5  cuMemUnmap                                      
      0.0         26039361         52    500756.9       325.0        90    6897036    1359312.3  cuMemRelease                                    
      0.0         18290359       2018      9063.6      8690.0      1770     104470       4447.9  cudaEventSynchronize                            
      0.0         11376074        870     13075.9       580.0       200     631920      69289.3  cudaEventCreateWithFlags                        
      0.0         10432246      18968       550.0       330.0       270      61680        895.6  cudaGetFuncBySymbol_v11000                      
      0.0         10251016      37936       270.2       200.0       150      62520        619.9  cudaStreamGetCaptureInfo_v12030                 
      0.0         10119716      72007       140.5       120.0        80      75370        548.3  cudaGetDeviceProperties_v12000                  
      0.0          3797393         58     65472.3      2365.0      1430    2333222     339998.0  cudaHostAlloc                                   
      0.0          3578443        256     13978.3     12580.0      4600      74541       6322.8  cudaMemcpy                                      
      0.0          2622841        440      5961.0      2745.0       930      57730       8030.6  cuMemAddressReserve                             
      0.0          1834142        146     12562.6      6470.0      1900      60440      11971.2  cuMemsetD32Async                                
      0.0          1620601         42     38585.7      4890.0      1970     263100      71483.7  cudaMalloc                                      
      0.0           581700       2674       217.5       140.0        60      24180        540.6  cuGetProcAddress_v2                             
      0.0           553631        784       706.2       590.0       230       2510        438.9  cudaEventDestroy                                
      0.0           544591         77      7072.6      6460.0      4260      28890       3410.2  cudaEventQuery                                  
      0.0           374540          4     93635.0     93675.0      4120     183070     103115.0  cudaMallocFromPoolAsync_v11020                  
      0.0           348890          4     87222.5     86955.0     46470     128510      44857.7  cudaMemPoolCreate_v11020                        
      0.0           336910         32     10528.4      7950.0      5780      27640       5692.2  cudaStreamDestroy                               
      0.0           240500        992       242.4       180.0       140      10870        388.7  cudaThreadExchangeStreamCaptureMode_v10010      
      0.0           206260         36      5729.4      4125.0      3030      17310       3522.6  cudaFreeHost                                    
      0.0           181390         92      1971.6       555.0       170      17460       3403.8  cuLibraryGetKernel                              
      0.0           130140         14      9295.7      8275.0      1030      23370       6646.8  cudaFree                                        
      0.0            99030         12      8252.5      4295.0      2880      29290       8176.8  cuMemAddressFree                                
      0.0            89290        438       203.9       130.0        60      10580        669.0  cuMemGetAllocationGranularity                   
      0.0            76820          2     38410.0     38410.0     33860      42960       6434.7  cudaStreamCreateWithPriority                    
      0.0            44950          2     22475.0     22475.0     15820      29130       9411.6  cuMulticastCreate                               
      0.0            26750         60       445.8       380.0       350       1890        246.4  cudaOccupancyMaxActiveClusters_v11070           
      0.0            25970         80       324.6       375.0       150        950        148.2  cudaGetDriverEntryPointByVersion_v12050         
      0.0            20490         24       853.8       825.0       370       1720        309.9  cuMemRetainAllocationHandle                     
      0.0            19040         10      1904.0      1675.0      1220       4290        889.5  cuInit                                          
      0.0             3440          2      1720.0      1720.0      1570       1870        212.1  cudaOccupancyAvailableDynamicSMemPerBlock_v10200
      0.0             3230          2      1615.0      1615.0      1460       1770        219.2  cudaMemPoolSetAttribute_v11020                  
      0.0             2350          8       293.8       115.0        70        770        304.1  cuModuleGetLoadingMode                          
      0.0              590          2       295.0       295.0       160        430        190.9  cuMulticastGetGranularity                       

[4/6] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  ---------  ---------  --------  ---------  -----------  ----------------------------------------------------------------------------------------------------
     13.8     120721440547     347520   347379.8   209695.0     69183    3819173     426484.6  nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT                                                      
     12.8     112140196476     138240   811199.3   884876.5    496604    1356188     268394.3  nvjet_sm90_tst_128x256_64x4_2x1_v_bz_coopA_NTN                                                      
     12.1     105262292638      69120  1522892.0  1527067.0   1268949    1925498      40306.4  void OwnTensor::cuda::gqa_bwd_vj1d::gqa_bwd_vj1d<(int)64, (int)64, (int)128>(CUtensorMap_st, CUtens…
      9.2      80017715148      72000  1111357.2  1108829.0   1039798    1440275      35195.9  nvjet_sm90_tst_192x192_64x3_1x2_h_bz_coopB_NNN                                                      
      8.9      77734849884     360000   215930.1   201440.0     61376     775230     165757.3  nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_NNT                                                      
      6.1      52913928637    1267630    41742.4    36703.0      4064    2998960      54545.2  OwnTensor::sm90::vec4_convert_to_fp32(const __nv_bfloat16 *, float *, unsigned long)                
      4.7      40791620907     863720    47227.8    36256.0      3616    2583387      56414.2  OwnTensor::sm90::vec8_convert_to_bf16(const float *, __nv_bfloat16 *, unsigned long)                
      4.5      39648675707      69120   573620.9   574431.0    521691     721054      14812.7  nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT                                                      
      4.0      34830661563     140160   248506.4   206079.0    183711    3807174     337250.9  nvjet_sm90_tst_192x192_64x3_1x2_h_bz_coopB_NTN                                                      
      3.4      29511287358      72000   409879.0   408223.0    357918     752542      17521.8  OwnTensor::cuda::gqa62::gqa_v62_sm90(CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, const __nv_bfl…
      2.5      21536503202      69120   311581.4   311231.0    309662     335679       1605.1  void OwnTensor::cuda::swiglu_bwd_v22_vec_kernel_sm90<float>(const T1 *, const T1 *, T1 *, int)      
      2.3      20099528877     763510    26325.2     8352.0       896     106368      29018.1  void OwnTensor::add_inplace_kernel_sm90<float, (int)2>(T1 *, const T1 *, unsigned long)             
      1.8      16103699631       8700  1851000.0   312031.0     34688  225909571   11604862.3  ncclSymkDevKernel_ReduceScatter_LDMC_sum_bf16(ncclSymkDevWorkArgs4K)                                
      1.5      13400484093      72000   186117.8   186207.0    183296     318911       1468.9  void OwnTensor::cuda::swiglu_fwd_vec_kernel_sm90<float>(const T1 *, T1 *, int)                      
      1.3      11265328946     140160    80374.8    80256.0     76672     377342       3437.2  void OwnTensor::cuda::rms_bwd_fused_block_sm90<float, (int)4, (int)2>(const T1 *, const T1 *, const…
      1.3      11247295325      72884   154317.8   139071.0    137663     448189      63017.0  void OwnTensor::add_inplace_kernel_flat_sm90<float, (bool)0>(T1 *, const T1 *, unsigned long, unsig…
      1.2      10597274569     144000    73592.2    73759.0     71456      99999       1270.3  void OwnTensor::add_kernel<float, (int)2>(const T1 *, const T1 *, T1 *, unsigned long)              
      1.2      10115422631     138240    73172.9    72416.0     65247     149919       7640.5  nvjet_sm90_tst_128x192_64x5_2x1_v_bz_coopB_NTN                                                      
      1.0       9150314647      69120   132383.0   132127.0    128224     640828       5271.8  void OwnTensor::cuda::fused_qk_norm_rope_bwd_kernel_sm90<(bool)1, (int)128>(const float *, const fl…
      1.0       8955297635     146000    61337.7    61215.0     58207     172096       2299.4  void OwnTensor::cuda::rms_norm_fwd_staged_kernel_sm90<float>(const T1 *, const T1 *, T1 *, float *,…
      0.7       6505109885       2000  3252554.9  3235576.5   3140521    4060516      76539.1  nvjet_sm90_tst_192x192_64x3_2x1_v_bz_coopB_NNN                                                      
      0.6       5270725288      72000    73204.5    73216.0     70816     129024       1801.7  void OwnTensor::cuda::fused_qknorm_rope_forward_t_sm90<(bool)1, (int)128, (int)4, (int)8>(const flo…
      0.6       5192468163     138240    37561.3    37376.0     35776      58720        925.8  void OwnTensor::add_inplace_kernel_sm90<OwnTensor::bfloat16_t, (int)2>(T1 *, const T1 *, unsigned l…
      0.6       4861744764     138240    35168.9    40208.0     17535     117312      16650.1  void OwnTensor::sm90::strided_inner_vec_copy_kernel<float4, float, (int)4, (int)3, unsigned int>(co…
      0.6       4854274382     282240    17199.1    22400.0      7328      39584       8584.5  void OwnTensor::sm90::strided_inner_vec_copy_kernel<uint4, unsigned short, (int)8, (int)3, unsigned…
      0.3       2833165187     138240    20494.5    20928.0     18623      36992       1193.3  OwnTensor::cuda::gqa_bwd_vj1d::gqa_dkdv_greduce(const __nv_bfloat16 *, __nv_bfloat16 *, int, int, i…
      0.3       2740350837       1920  1427266.1  1425948.0   1362620    1614490      24100.3  OwnTensor::cuda::embedding_backward_kernel_cooperative(const unsigned short *, const float *, float…
      0.3       2737001031     138240    19798.9    21936.0      9184      53407       9396.4  OwnTensor::cuda::reduce_dgamma_kernel(const float *, float *, int, int)                             
      0.3       2561971464      69120    37065.6    36640.0     35583      51520        766.4  OwnTensor::cuda::gqa_bwd_vj1d::convert_dq_accum_to_bf16_v6(const float4 *, uint2 *, long)           
      0.3       2513639030      69120    36366.3    36000.0     34304      58272        874.7  OwnTensor::cuda::gqa_bwd_vj1d::compute_drowsum_v22(const __nv_bfloat16 *, const __nv_bfloat16 *, fl…
      0.3       2403266019       9420   255123.8   174143.0     24224   35492407     413001.5  ncclSymkDevKernel_AllGather_STMC(ncclSymkDevWorkArgs4K)                                             
      0.2       1590127239       1920   828191.3   828158.0    821246     865626       2058.9  void OwnTensor::cuda::sparseCENormalize_from_stats<float, unsigned short>(const T1 *, const T2 *, T…
      0.1        827229563        150  5514863.8   548750.5      3456   13267456    6329949.3  OwnTensor::cuda::multi_tensor_adamw_sm90_kernel(const OwnTensor::cuda::AdamTensorMeta *, const int …
      0.1        758326802       1920   394961.9   394911.0    390911     426973       1481.7  void OwnTensor::cuda::sparse_ce_forward_kernel_vec_save_stats<float, unsigned short>(const T1 *, co…
      0.1        477605564     140160     3407.6     3392.0      2912       5312        186.1  OwnTensor::cuda::rms_bwd_gamma_finalize_sm90(const float *, float *, int, int)                      
      0.0        348628622        120  2905238.5  2892437.0    207135    5644978    2706434.5  OwnTensor::cuda::multi_tensor_scale_kernel(const OwnTensor::cuda::TensorInfo *, const long *, int, …
      0.0        188110629        720   261264.8   163776.0    133183    2581624     479454.5  ncclSymkDevKernel_ReduceScatter_LDMC_sum_f32(ncclSymkDevWorkArgs4K)                                 
      0.0        106214855        120   885123.8   882237.5     79008    1697916     807648.8  OwnTensor::cuda::multi_tensor_grad_norm_kernel(const OwnTensor::cuda::TensorInfo *, const long *, i…
      0.0         72666869       2000    36333.4    36128.0     31167      46399       1702.7  OwnTensor::cuda::embedding_forward_kernel_vectorized(const unsigned short *, const float *, float *…
      0.0         31636118         80   395451.5   394958.0    389917     409567       3102.5  void OwnTensor::cuda::sparse_ce_forward_kernel_vec<float, unsigned short>(const T1 *, const T2 *, T…
      0.0         13137484        128   102636.6     6800.0      5472    3035031     320323.9  ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)              
      0.0          6444591       4000     1611.1     1696.0      1120       3232        327.9  void OwnTensor::cuda::sum_reduction_kernel<float>(const T1 *, T1 *, long)                           
      0.0          2608825       2000     1304.4     1280.0      1120       2528        122.2  void OwnTensor::scalar_div_copy<float, float>(const T1 *, double, T2 *, unsigned long)              
      0.0           276288        120     2302.4     2304.0      2080       2688        166.4  OwnTensor::cuda::grad_norm_cleanup_kernel(const float *, int, float *)                              
      0.0           110207         60     1836.8     1664.0      1312       9952       1098.8  void OwnTensor::scalar_mul_inplace<float>(T1 *, double, unsigned long)                              
      0.0             6944          2     3472.0     3472.0      3456       3488         22.6  OwnTensor::cuda::rope_build_cache_kernel(float *, long, long, float)                                

[5/6] Executing 'cuda_gpu_mem_time_sum' stats report

 Time (%)  Total Time (ns)  Count   Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)            Operation           
 --------  ---------------  ------  --------  --------  --------  --------  -----------  ------------------------------
     67.1       3020073457  941438    3207.9     896.0       704   1836207      11034.3  [CUDA memset]                 
     20.3        912846372    6166  148045.1    4832.0       704  30330469     950151.1  [CUDA memcpy Host-to-Device]  
     12.5        564338997   44720   12619.4    4512.0       672    343199      23897.7  [CUDA memcpy Device-to-Device]
      0.0           668959     268    2496.1    2240.0      1984      5696        707.2  [CUDA memcpy Device-to-Host]  

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
