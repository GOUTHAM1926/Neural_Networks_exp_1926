
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
  warmup / steps  : 200 / 2000
  WSD stable_steps: 1600  (decay tail=200)
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
step      0 | train 11.436555 | val 11.437630 | tok/s 34924.05 | norm 22.3188 | dt 15012.24 ms | lr 1.5000e-06 | mfu 51.24% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 22.2535 | G1(float32/154M) 1.7064
step      1 | train 11.356039 | val 11.437630 | tok/s 36104.02 | norm 20.3567 | dt 14521.59 ms | lr 3.0000e-06 | mfu 52.97% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 20.2961 | G1(float32/154M) 1.5703
step      2 | train 10.583187 | val 11.437630 | tok/s 35976.78 | norm 29.5352 | dt 14572.95 ms | lr 4.5000e-06 | mfu 52.78% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 29.4307 | G1(float32/154M) 2.4831
step      3 | train 11.277708 | val 11.437630 | tok/s 35810.63 | norm 197.9275 | dt 14640.57 ms | lr 6.0000e-06 | mfu 52.54% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 197.4990 | G1(float32/154M) 13.0161
step      4 | train 11.134930 | val 11.437630 | tok/s 35650.67 | norm 171.7284 | dt 14706.26 ms | lr 7.5000e-06 | mfu 52.30% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 171.4328 | G1(float32/154M) 10.0706
step      5 | train 10.355480 | val 11.437630 | tok/s 35617.65 | norm 35.7145 | dt 14719.89 ms | lr 9.0000e-06 | mfu 52.26% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 35.6669 | G1(float32/154M) 1.8433
step      6 | train 10.818175 | val 11.437630 | tok/s 35635.67 | norm 39.1796 | dt 14712.45 ms | lr 1.0500e-05 | mfu 52.28% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 39.1270 | G1(float32/154M) 2.0292
step      7 | train 10.381914 | val 11.437630 | tok/s 35741.70 | norm 133.3448 | dt 14668.80 ms | lr 1.2000e-05 | mfu 52.44% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 133.2371 | G1(float32/154M) 5.3570
step      8 | train 9.623417 | val 11.437630 | tok/s 35770.56 | norm 26.2727 | dt 14656.97 ms | lr 1.3500e-05 | mfu 52.48% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 26.2507 | G1(float32/154M) 1.0751
step      9 | train 10.034176 | val 11.437630 | tok/s 35808.55 | norm 76.1910 | dt 14641.42 ms | lr 1.5000e-05 | mfu 52.54% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 76.1350 | G1(float32/154M) 2.9205
step     10 | train 9.630663 | val 11.437630 | tok/s 35842.54 | norm 36.0914 | dt 14627.53 ms | lr 1.6500e-05 | mfu 52.59% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 36.0669 | G1(float32/154M) 1.3297
step     11 | train 9.436483 | val 11.437630 | tok/s 35848.33 | norm 15.9692 | dt 14625.17 ms | lr 1.8000e-05 | mfu 52.59% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 15.9572 | G1(float32/154M) 0.6193
step     12 | train 9.229979 | val 11.437630 | tok/s 35846.12 | norm 30.4553 | dt 14626.07 ms | lr 1.9500e-05 | mfu 52.59% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 30.4433 | G1(float32/154M) 0.8555
step     13 | train 9.304943 | val 11.437630 | tok/s 35830.94 | norm 36.3145 | dt 14632.27 ms | lr 2.1000e-05 | mfu 52.57% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 36.2958 | G1(float32/154M) 1.1648
step     14 | train 9.114820 | val 11.437630 | tok/s 35847.39 | norm 13.7275 | dt 14625.56 ms | lr 2.2500e-05 | mfu 52.59% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 13.7212 | G1(float32/154M) 0.4166
step     15 | train 9.029736 | val 11.437630 | tok/s 35846.63 | norm 13.4367 | dt 14625.86 ms | lr 2.4000e-05 | mfu 52.59% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 13.4282 | G1(float32/154M) 0.4773
step     16 | train 8.770510 | val 11.437630 | tok/s 35812.50 | norm 12.9906 | dt 14639.80 ms | lr 2.5500e-05 | mfu 52.54% | mem 101.79/103.18 GiB | G0(bfloat16/3778M) 12.9722 | G1(float32/154M) 0.6911
make: *** [Makefile:312: run-mpi] Terminated
The target application terminated. One or more process it created re-parented.
Waiting for termination of re-parented processes.
Use the `--wait` option to modify this behavior.
Collecting data...
Generating '/tmp/nsys-report-2474.qdstrm'

[1/6] [0%                          ] h228_hybrid.nsys-rep
[1/6] [0%                          ] h228_hybrid.nsys-rep
[1/6] [0%                          ] h228_hybrid.nsys-rep
[1/6] [6%                          ] h228_hybrid.nsys-rep
[1/6] [8%                          ] h228_hybrid.nsys-rep
[1/6] [7%                          ] h228_hybrid.nsys-rep
[1/6] [8%                          ] h228_hybrid.nsys-rep
[1/6] [7%                          ] h228_hybrid.nsys-rep
[1/6] [6%                          ] h228_hybrid.nsys-rep
[1/6] [10%                         ] h228_hybrid.nsys-rep
[1/6] [11%                         ] h228_hybrid.nsys-rep
[1/6] [10%                         ] h228_hybrid.nsys-rep
[1/6] [11%                         ] h228_hybrid.nsys-rep
[1/6] [10%                         ] h228_hybrid.nsys-rep
[1/6] [13%                         ] h228_hybrid.nsys-rep
[1/6] [14%                         ] h228_hybrid.nsys-rep
[1/6] [=15%                        ] h228_hybrid.nsys-rep
[1/6] [14%                         ] h228_hybrid.nsys-rep
[1/6] [=15%                        ] h228_hybrid.nsys-rep
[1/6] [==18%                       ] h228_hybrid.nsys-rep
[1/6] [=17%                        ] h228_hybrid.nsys-rep
[1/6] [==20%                       ] h228_hybrid.nsys-rep
[1/6] [==19%                       ] h228_hybrid.nsys-rep
[1/6] [==18%                       ] h228_hybrid.nsys-rep
[1/6] [==21%                       ] h228_hybrid.nsys-rep
[1/6] [==20%                       ] h228_hybrid.nsys-rep
[1/6] [===23%                      ] h228_hybrid.nsys-rep
[1/6] [===22%                      ] h228_hybrid.nsys-rep
[1/6] [==21%                       ] h228_hybrid.nsys-rep
[1/6] [===23%                      ] h228_hybrid.nsys-rep
[1/6] [===22%                      ] h228_hybrid.nsys-rep
[1/6] [===24%                      ] h228_hybrid.nsys-rep
[1/6] [===23%                      ] h228_hybrid.nsys-rep
[1/6] [====25%                     ] h228_hybrid.nsys-rep
[1/6] [===24%                      ] h228_hybrid.nsys-rep
[1/6] [====26%                     ] h228_hybrid.nsys-rep
[1/6] [====25%                     ] h228_hybrid.nsys-rep
[1/6] [====27%                     ] h228_hybrid.nsys-rep
[1/6] [====26%                     ] h228_hybrid.nsys-rep
[1/6] [====27%                     ] h228_hybrid.nsys-rep
[1/6] [====28%                     ] h228_hybrid.nsys-rep
[1/6] [====27%                     ] h228_hybrid.nsys-rep
[1/6] [=====29%                    ] h228_hybrid.nsys-rep
[1/6] [====28%                     ] h228_hybrid.nsys-rep
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
     53.3     263252694594    2844192     92558.0     27950.0      2180   54948546     237721.3  cudaLaunchKernel                                
     31.0     153353853383     738992    207517.6     37310.5      2260   13347501     410522.4  cuLaunchKernelEx                                
      9.6      47344904160     501539     94399.2     18790.0      1350    8131787     273318.1  cudaMemsetAsync                                 
      2.3      11258318367       1480   7606971.9    319890.5       880  192376520   24453365.1  cudaMallocAsync_v11020                          
      2.3      11255242806      29846    377110.6      7770.0      1770  140932407    5933807.6  cudaMemcpyAsync                                 
      0.8       3740524689      39781     94027.9     65430.0      3190    6568075     208527.8  cudaMemset                                      
      0.2        923610597        563   1640516.2     82690.0      2200   13283981    3303798.0  cudaStreamSynchronize                           
      0.1        631271720     526581      1198.8       980.0       380    3672083      12848.5  cudaEventRecord                                 
      0.1        370453850        382    969774.5     83235.0     39450   69765898    6143806.7  cuMemSetAccess                                  
      0.1        318298347         58   5487902.5     98060.0     17480  119085949   21846022.7  cuLibraryLoadData                               
      0.1        253024310    2433075       104.0        80.0        60     570240        562.2  cuStreamGetCaptureInfo_v2                       
      0.0        240971677    2844194        84.7        70.0        50     134310        404.2  cuKernelGetName                                 
      0.0        236651526         32   7395360.2     18320.0       440  115864626   28427160.0  cuKernelSetAttribute                            
      0.0        176760695        370    477731.6     14350.0      1330  100276174    6023595.8  cuMemMap                                        
      0.0        159428816         16   9964301.0    268825.0     37000   66721326   21394856.4  cuMulticastBindMem                              
      0.0        153668029        258    595612.5     22055.5     13960  105478668    6611425.0  cuMemCreate                                     
      0.0        145195886    1456480        99.7        80.0        50     158161        450.7  cuKernelGetAttribute                            
      0.0         87547822          4  21886955.5  21178972.5     12440   45177437   25263874.8  cuMulticastAddDevice                            
      0.0         67513703     402221       167.9        90.0        70      92930        553.2  cuTensorMapEncodeTiled                          
      0.0         40110377      40052      1001.5       650.0       430     226321       2094.4  cudaStreamWaitEvent                             
      0.0         34898373        112    311592.6     58920.0     33510   11450929    1112187.4  cuMemImportFromShareableHandle                  
      0.0         11437379         26    439899.2     12545.0      3240    6401635    1529102.5  cudaStreamCreateWithFlags                       
      0.0         10528059       1158      9091.6      8490.0      1740      70770       4503.5  cudaEventSynchronize                            
      0.0          6835383        818      8356.2       585.0       200     598511      51659.1  cudaEventCreateWithFlags                        
      0.0          6079005      10752       565.4       330.0       270      67320       1321.7  cudaGetFuncBySymbol_v11000                      
      0.0          5769152      21504       268.3       200.0       150      32450        519.4  cudaStreamGetCaptureInfo_v12030                 
      0.0          5490942      41263       133.1       110.0        70      30370        340.9  cudaGetDeviceProperties_v12000                  
      0.0          2896513          4    724128.3    633050.5      9790    1620622     834183.8  cudaMallocFromPoolAsync_v11020                  
      0.0          2766583         58     47699.7      3910.0      1430    1238091     219936.5  cudaHostAlloc                                   
      0.0          2514113          8    314264.1    141310.5    100220    1222531     388689.7  cudaDeviceSynchronize                           
      0.0          2362170        152     15540.6     12555.0      5150     134400      14124.8  cudaMemcpy                                      
      0.0          1781900        344      5179.9      2410.0       900      56970       7885.4  cuMemAddressReserve                             
      0.0          1723434        146     11804.3     10330.0      1920      99710      11993.4  cuMemsetD32Async                                
      0.0          1717584         42     40894.9      5110.0      1850     243000      70870.9  cudaMalloc                                      
      0.0           721310       2674       269.7       220.0        60      41370       1049.3  cuGetProcAddress_v2                             
      0.0           321140          4     80285.0     69035.0     42220     140850      44386.2  cudaMemPoolCreate_v11020                        
      0.0           237481        992       239.4       190.0       140       3460        200.1  cudaThreadExchangeStreamCaptureMode_v10010      
      0.0           194490         14     13892.1     14655.0      7600      24040       5502.7  cudaStreamDestroy                               
      0.0           160230         92      1741.6       560.0       160       9730       2558.9  cuLibraryGetKernel                              
      0.0           148900         16      9306.3       585.0       240      77460      24110.9  cuMemRelease                                    
      0.0           117850         76      1550.7      1400.0       590       8770       1065.4  cudaEventDestroy                                
      0.0            61320        342       179.3       120.0        70       5660        389.9  cuMemGetAllocationGranularity                   
      0.0            50370          6      8395.0      4225.0      1450      22260       8833.4  cudaFree                                        
      0.0            45920         60       765.3       890.0       350       3340        482.3  cudaOccupancyMaxActiveClusters_v11070           
      0.0            40591          2     20295.5     20295.5     18770      21821       2157.4  cudaStreamCreateWithPriority                    
      0.0            37030          4      9257.5      9130.0      5300      13470       3978.5  cudaEventQuery                                  
      0.0            26190         12      2182.5      2010.0      1000       4170        888.3  cuMemRetainAllocationHandle                     
      0.0            24880          2     12440.0     12440.0     11740      13140        989.9  cuMulticastCreate                               
      0.0            24450         10      2445.0      2165.0      1260       3730        939.9  cuInit                                          
      0.0            18230         80       227.9       190.0       150       1080        135.5  cudaGetDriverEntryPointByVersion_v12050         
      0.0             5480          2      2740.0      2740.0      2200       3280        763.7  cudaOccupancyAvailableDynamicSMemPerBlock_v10200
      0.0             3380          2      1690.0      1690.0      1590       1790        141.4  cudaMemPoolSetAttribute_v11020                  
      0.0             3010          8       376.3       155.0        80       1090        435.5  cuModuleGetLoadingMode                          
      0.0              530          2       265.0       265.0       160        370        148.5  cuMulticastGetGranularity                       

[4/6] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  ---------  ---------  --------  ---------  -----------  ----------------------------------------------------------------------------------------------------
     13.9      70612458254     199999   353064.1   212351.0     69631    3851423     437145.7  nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT                                                      
     12.9      65479614170      79560   823021.8   893834.0    495643    1363356     272749.0  nvjet_sm90_tst_128x256_64x4_2x1_v_bz_coopA_NTN                                                      
     12.1      61419232399      39779  1544011.5  1543453.0   1267894    1932090      43546.2  void OwnTensor::cuda::gqa_bwd_vj1d::gqa_bwd_vj1d<(int)64, (int)64, (int)128>(CUtensorMap_st, CUtens…
      9.3      47253872497      41256  1145381.8  1145180.0   1043990    1405020      34045.1  nvjet_sm90_tst_192x192_64x3_1x2_h_bz_coopB_NNN                                                      
      9.1      46228920224     206280   224107.6   209983.0     64415     667326     171954.7  nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_NNT                                                      
      6.0      30427920833     728563    41764.3    36736.0      4096    3000305      54509.3  OwnTensor::sm90::vec4_convert_to_fp32(const __nv_bfloat16 *, float *, unsigned long)                
      4.6      23437259971     495873    47264.6    36288.0      3616    2578739      56404.2  OwnTensor::sm90::vec8_convert_to_bf16(const float *, __nv_bfloat16 *, unsigned long)                
      4.5      23138138180      39780   581652.5   580734.0    522139     722494      15749.7  nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT                                                      
      4.0      20446200473      80663   253476.8   208415.0    183231    3841855     355094.7  nvjet_sm90_tst_192x192_64x3_1x2_h_bz_coopB_NTN                                                      
      3.4      17496111120      41256   424086.5   423839.0    358109     724956      17018.1  OwnTensor::cuda::gqa62::gqa_v62_sm90(CUtensorMap_st, CUtensorMap_st, CUtensorMap_st, const __nv_bfl…
      2.4      12395682602      39780   311605.9   311262.0    309630     334752       1672.7  void OwnTensor::cuda::swiglu_bwd_v22_vec_kernel_sm90<float>(const T1 *, const T1 *, T1 *, int)      
      2.3      11567837810     439125    26342.9     8352.0       896     117791      29020.2  void OwnTensor::add_inplace_kernel_sm90<float, (int)2>(T1 *, const T1 *, unsigned long)             
      1.7       8791567852       4930  1783279.5   318847.0     34687  256382490   10801477.0  ncclSymkDevKernel_ReduceScatter_LDMC_sum_bf16(ncclSymkDevWorkArgs4K)                                
      1.5       7678352947      41256   186114.8   186208.0    183136     207200       1308.5  void OwnTensor::cuda::swiglu_fwd_vec_kernel_sm90<float>(const T1 *, T1 *, int)                      
      1.3       6493532720      80663    80502.0    80384.0     76607     365790       3256.5  void OwnTensor::cuda::rms_bwd_fused_block_sm90<float, (int)4, (int)2>(const T1 *, const T1 *, const…
      1.3       6468971517      41914   154339.2   139072.0    137791     444063      63000.8  void OwnTensor::add_inplace_kernel_flat_sm90<float, (bool)0>(T1 *, const T1 *, unsigned long, unsig…
      1.2       6074815288      82512    73623.4    73792.0     71519      98912       1202.5  void OwnTensor::add_kernel<float, (int)2>(const T1 *, const T1 *, T1 *, unsigned long)              
      1.2       5905688079      79556    74233.1    73248.0     65248     151167       7657.7  nvjet_sm90_tst_128x192_64x5_2x1_v_bz_coopB_NTN                                                      
      1.0       5278051463      39778   132687.7   132384.0    128287     442975       5822.0  void OwnTensor::cuda::fused_qk_norm_rope_bwd_kernel_sm90<(bool)1, (int)128>(const float *, const fl…
      1.0       5158067106      83658    61656.6    61535.0     58400     163104       2284.6  void OwnTensor::cuda::rms_norm_fwd_staged_kernel_sm90<float>(const T1 *, const T1 *, T1 *, float *,…
      0.8       3894354263       1146  3398214.9  3413784.5   3145764    4019805      88587.3  nvjet_sm90_tst_192x192_64x3_2x1_v_bz_coopB_NNN                                                      
      0.6       3029783360      41256    73438.6    73472.0     70944     111904       1677.0  void OwnTensor::cuda::fused_qknorm_rope_forward_t_sm90<(bool)1, (int)128, (int)4, (int)8>(const flo…
      0.6       2988704665      79556    37567.3    37376.0     35744      56480        917.5  void OwnTensor::add_inplace_kernel_sm90<OwnTensor::bfloat16_t, (int)2>(T1 *, const T1 *, unsigned l…
      0.6       2801058202      79556    35208.6    41055.5     17760      61920      16668.9  void OwnTensor::sm90::strided_inner_vec_copy_kernel<float4, float, (int)4, (int)3, unsigned int>(co…
      0.5       2796535736     162069    17255.2    23744.0      7328      40640       8602.3  void OwnTensor::sm90::strided_inner_vec_copy_kernel<uint4, unsigned short, (int)8, (int)3, unsigned…
      0.3       1633464590      79556    20532.3    20992.0     18687      36960       1201.3  OwnTensor::cuda::gqa_bwd_vj1d::gqa_dkdv_greduce(const __nv_bfloat16 *, __nv_bfloat16 *, int, int, i…
      0.3       1589219568      79556    19976.1    21216.0      9184      46144       9479.2  OwnTensor::cuda::reduce_dgamma_kernel(const float *, float *, int, int)                             
      0.3       1581149504       1104  1432200.6  1431324.0   1359837    1620603      25248.0  OwnTensor::cuda::embedding_backward_kernel_cooperative(const unsigned short *, const float *, float…
      0.3       1477462139      39778    37142.7    36735.0     35552      48672        768.8  OwnTensor::cuda::gqa_bwd_vj1d::convert_dq_accum_to_bf16_v6(const float4 *, uint2 *, long)           
      0.3       1453425507      39779    36537.5    36192.0     34207      85632        922.4  OwnTensor::cuda::gqa_bwd_vj1d::compute_drowsum_v22(const __nv_bfloat16 *, const __nv_bfloat16 *, fl…
      0.3       1310498068       5338   245503.6   172704.0     24288    1257015     190199.9  ncclSymkDevKernel_AllGather_STMC(ncclSymkDevWorkArgs4K)                                             
      0.2        916286105       1106   828468.4   828350.0    823005     862873       2338.5  void OwnTensor::cuda::sparseCENormalize_from_stats<float, unsigned short>(const T1 *, const T2 *, T…
      0.1        468699077         85  5514106.8   549727.0      3520   13277698    6345188.4  OwnTensor::cuda::multi_tensor_adamw_sm90_kernel(const OwnTensor::cuda::AdamTensorMeta *, const int …
      0.1        439418993       1106   397304.7   397246.5    391101     426748       1929.5  void OwnTensor::cuda::sparse_ce_forward_kernel_vec_save_stats<float, unsigned short>(const T1 *, co…
      0.1        277137463      80663     3435.7     3424.0      2912       5440        184.5  OwnTensor::cuda::rms_bwd_gamma_finalize_sm90(const float *, float *, int, int)                      
      0.0        197673500         68  2906963.2  2892422.0    207648    5655187    2717091.5  OwnTensor::cuda::multi_tensor_scale_kernel(const OwnTensor::cuda::TensorInfo *, const long *, int, …
      0.0        105306951        408   258105.3   163663.5    133760    2577657     468746.9  ncclSymkDevKernel_ReduceScatter_LDMC_sum_f32(ncclSymkDevWorkArgs4K)                                 
      0.0         60225988         68   885676.3   882925.5     79008    1697437     810747.9  OwnTensor::cuda::multi_tensor_grad_norm_kernel(const OwnTensor::cuda::TensorInfo *, const long *, i…
      0.0         41709808       1146    36396.0    36192.0     31200      45152       1693.8  OwnTensor::cuda::embedding_forward_kernel_vectorized(const unsigned short *, const float *, float *…
      0.0         15860445         40   396511.1   395277.0    392157     420413       4649.2  void OwnTensor::cuda::sparse_ce_forward_kernel_vec<float, unsigned short>(const T1 *, const T2 *, T…
      0.0          6961979         76    91605.0     6704.0      5536    1226772     205735.6  ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>)              
      0.0          3882391       2292     1693.9     1728.0      1120       3104        303.2  void OwnTensor::cuda::sum_reduction_kernel<float>(const T1 *, T1 *, long)                           
      0.0          1613015       1146     1407.5     1440.0      1120       2304        138.7  void OwnTensor::scalar_div_copy<float, float>(const T1 *, double, T2 *, unsigned long)              
      0.0           156448         68     2300.7     2352.0      2080       2624        164.3  OwnTensor::cuda::grad_norm_cleanup_kernel(const float *, int, float *)                              
      0.0            59392         36     1649.8     1648.0      1152       2528        330.3  void OwnTensor::scalar_mul_inplace<float>(T1 *, double, unsigned long)                              
      0.0             6944          2     3472.0     3472.0      3456       3488         22.6  OwnTensor::cuda::rope_build_cache_kernel(float *, long, long, float)                                

[5/6] Executing 'cuda_gpu_mem_time_sum' stats report

 Time (%)  Total Time (ns)  Count   Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)            Operation           
 --------  ---------------  ------  --------  --------  --------  --------  -----------  ------------------------------
     56.7       1756173764  541466    3243.4     896.0       704   1837422      12177.8  [CUDA memset]                 
     32.7       1013056442    4056  249767.4    4768.0       704  30264966    1245159.6  [CUDA memcpy Host-to-Device]  
     10.5        325611186   25792   12624.5    4544.0       672    333663      23778.4  [CUDA memcpy Device-to-Device]
      0.0           380735     150    2538.2    2240.0      2207      5984        746.0  [CUDA memcpy Device-to-Host]  

[6/6] Executing 'cuda_gpu_mem_size_sum' stats report

 Total (MB)   Count   Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)            Operation           
 -----------  ------  --------  --------  --------  --------  -----------  ------------------------------
 5317268.715  541466     9.820     0.000     0.000  7556.825       52.539  [CUDA memset]                 
  609093.698   25792    23.616     6.291     0.001   618.136       48.348  [CUDA memcpy Device-to-Device]
   16389.774    4056     4.041     0.016     0.000   618.136       21.405  [CUDA memcpy Host-to-Device]  
       0.001     150     0.000     0.000     0.000     0.000        0.000  [CUDA memcpy Device-to-Host]  

Generated:
	/root/gautam_k/blublas/h228_4b/h228_hybrid.nsys-rep
	/root/gautam_k/blublas/h228_4b/h228_hybrid.sqlite
