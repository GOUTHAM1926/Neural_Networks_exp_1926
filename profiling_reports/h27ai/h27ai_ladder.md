# §H27ai — instruction-cut ladder, V1 pin
shape 2048x512x16384  layout tn  gpu 1  commit 595adc5
date 2026-08-26T15:06:12+00:00

## binary check
  present  128x128x64_2x1_v_sk2_dh_rot2_pdl_d6
  present  128x128x64_2x1_v_sk2_dh_rot2_pdl_d6_cs
  present  128x128x64_2x1_v_sk2_dh_rot2_pdl_d6_dw
  present  128x128x64_2x1_v_sk2_dh_rot2_pdl_d6_dw_uad
  present  128x128x64_2x1_v_sk2_dh_rot2_pdl_d6_dw_uad_cs

## PHASE: locked clocks
  lock requested 1400 ; reports: 1980 MHz, 1980 MHz

### locked :: 128x128x64_2x1_v_sk2_dh_rot2_pdl_d6
Device : NVIDIA H200 (SM 9.0) SMs:132
matrix : 1 configs x 1 shapes -> 1 runs (after layout/config filters)   tol=0.050
config                     layout shape            status    maxRelErr  maxAbsErr    TFLOPS    cuBLAS     %cuB
128x128x64_2x1_v_sk2_dh_rot2_pdl_d6 tn     2048x512x16384   PASS        0.00000   0.000000    601.87    571.60   105.3%
1 PASS, 0 FAIL, 0 CRASH, 1/1 cases run (1 configs x 1 shapes)
ALL PASS - full SM90 feature x variant matrix validated vs cuBLAS
  achieved clock: mean 1410 MHz  (min 1410 max 1410)   power mean 168 W   n=3

### locked :: 128x128x64_2x1_v_sk2_dh_rot2_pdl_d6_cs
Device : NVIDIA H200 (SM 9.0) SMs:132
matrix : 1 configs x 1 shapes -> 1 runs (after layout/config filters)   tol=0.050
config                     layout shape            status    maxRelErr  maxAbsErr    TFLOPS    cuBLAS     %cuB
128x128x64_2x1_v_sk2_dh_rot2_pdl_d6_cs tn     2048x512x16384   PASS        0.00000   0.000000    601.68    569.25   105.7%
1 PASS, 0 FAIL, 0 CRASH, 1/1 cases run (1 configs x 1 shapes)
ALL PASS - full SM90 feature x variant matrix validated vs cuBLAS
  achieved clock: mean 1410 MHz  (min 1410 max 1410)   power mean 142 W   n=4

### locked :: 128x128x64_2x1_v_sk2_dh_rot2_pdl_d6_dw
Device : NVIDIA H200 (SM 9.0) SMs:132
matrix : 1 configs x 1 shapes -> 1 runs (after layout/config filters)   tol=0.050
config                     layout shape            status    maxRelErr  maxAbsErr    TFLOPS    cuBLAS     %cuB
128x128x64_2x1_v_sk2_dh_rot2_pdl_d6_dw tn     2048x512x16384   PASS        0.00000   0.000000    599.93    574.13   104.5%
1 PASS, 0 FAIL, 0 CRASH, 1/1 cases run (1 configs x 1 shapes)
ALL PASS - full SM90 feature x variant matrix validated vs cuBLAS
  achieved clock: mean 1410 MHz  (min 1410 max 1410)   power mean 163 W   n=3

### locked :: 128x128x64_2x1_v_sk2_dh_rot2_pdl_d6_dw_uad
Device : NVIDIA H200 (SM 9.0) SMs:132
matrix : 1 configs x 1 shapes -> 1 runs (after layout/config filters)   tol=0.050
config                     layout shape            status    maxRelErr  maxAbsErr    TFLOPS    cuBLAS     %cuB
128x128x64_2x1_v_sk2_dh_rot2_pdl_d6_dw_uad tn     2048x512x16384   PASS        0.00000   0.000000    558.52    572.82    97.5%
1 PASS, 0 FAIL, 0 CRASH, 1/1 cases run (1 configs x 1 shapes)
ALL PASS - full SM90 feature x variant matrix validated vs cuBLAS
  achieved clock: mean 1410 MHz  (min 1410 max 1410)   power mean 167 W   n=2

### locked :: 128x128x64_2x1_v_sk2_dh_rot2_pdl_d6_dw_uad_cs
Device : NVIDIA H200 (SM 9.0) SMs:132
matrix : 1 configs x 1 shapes -> 1 runs (after layout/config filters)   tol=0.050
config                     layout shape            status    maxRelErr  maxAbsErr    TFLOPS    cuBLAS     %cuB
128x128x64_2x1_v_sk2_dh_rot2_pdl_d6_dw_uad_cs tn     2048x512x16384   PASS        0.00000   0.000000    602.81    574.51   104.9%
1 PASS, 0 FAIL, 0 CRASH, 1/1 cases run (1 configs x 1 shapes)
ALL PASS - full SM90 feature x variant matrix validated vs cuBLAS
  achieved clock: mean 1410 MHz  (min 1410 max 1410)   power mean 132 W   n=4

## PHASE: free clocks
  clocks reset to free

### free :: 128x128x64_2x1_v_sk2_dh_rot2_pdl_d6
Device : NVIDIA H200 (SM 9.0) SMs:132
matrix : 1 configs x 1 shapes -> 1 runs (after layout/config filters)   tol=0.050
config                     layout shape            status    maxRelErr  maxAbsErr    TFLOPS    cuBLAS     %cuB
128x128x64_2x1_v_sk2_dh_rot2_pdl_d6 tn     2048x512x16384   PASS        0.00000   0.000000    614.79    642.65    95.7%
1 PASS, 0 FAIL, 0 CRASH, 1/1 cases run (1 configs x 1 shapes)
ALL PASS - full SM90 feature x variant matrix validated vs cuBLAS
  achieved clock: mean 1977 MHz  (min 1965 max 1980)   power mean 173 W   n=5

### free :: 128x128x64_2x1_v_sk2_dh_rot2_pdl_d6_cs
Device : NVIDIA H200 (SM 9.0) SMs:132
matrix : 1 configs x 1 shapes -> 1 runs (after layout/config filters)   tol=0.050
config                     layout shape            status    maxRelErr  maxAbsErr    TFLOPS    cuBLAS     %cuB
128x128x64_2x1_v_sk2_dh_rot2_pdl_d6_cs tn     2048x512x16384   PASS        0.00000   0.000000    622.21    639.51    97.3%
1 PASS, 0 FAIL, 0 CRASH, 1/1 cases run (1 configs x 1 shapes)
ALL PASS - full SM90 feature x variant matrix validated vs cuBLAS
  achieved clock: mean 1965 MHz  (min 1920 max 1980)   power mean 171 W   n=5

### free :: 128x128x64_2x1_v_sk2_dh_rot2_pdl_d6_dw
Device : NVIDIA H200 (SM 9.0) SMs:132
matrix : 1 configs x 1 shapes -> 1 runs (after layout/config filters)   tol=0.050
config                     layout shape            status    maxRelErr  maxAbsErr    TFLOPS    cuBLAS     %cuB
128x128x64_2x1_v_sk2_dh_rot2_pdl_d6_dw tn     2048x512x16384   PASS        0.00000   0.000000    618.60    655.43    94.4%
1 PASS, 0 FAIL, 0 CRASH, 1/1 cases run (1 configs x 1 shapes)
ALL PASS - full SM90 feature x variant matrix validated vs cuBLAS
  achieved clock: mean 1806 MHz  (min 795 max 1980)   power mean 174 W   n=7

### free :: 128x128x64_2x1_v_sk2_dh_rot2_pdl_d6_dw_uad
Device : NVIDIA H200 (SM 9.0) SMs:132
matrix : 1 configs x 1 shapes -> 1 runs (after layout/config filters)   tol=0.050
config                     layout shape            status    maxRelErr  maxAbsErr    TFLOPS    cuBLAS     %cuB
128x128x64_2x1_v_sk2_dh_rot2_pdl_d6_dw_uad tn     2048x512x16384   PASS        0.00000   0.000000    612.90    657.64    93.2%
1 PASS, 0 FAIL, 0 CRASH, 1/1 cases run (1 configs x 1 shapes)
ALL PASS - full SM90 feature x variant matrix validated vs cuBLAS
  achieved clock: mean 1932 MHz  (min 1740 max 1980)   power mean 179 W   n=5

### free :: 128x128x64_2x1_v_sk2_dh_rot2_pdl_d6_dw_uad_cs
Device : NVIDIA H200 (SM 9.0) SMs:132
matrix : 1 configs x 1 shapes -> 1 runs (after layout/config filters)   tol=0.050
config                     layout shape            status    maxRelErr  maxAbsErr    TFLOPS    cuBLAS     %cuB
128x128x64_2x1_v_sk2_dh_rot2_pdl_d6_dw_uad_cs tn     2048x512x16384   PASS        0.00000   0.000000    642.58    645.22    99.6%
1 PASS, 0 FAIL, 0 CRASH, 1/1 cases run (1 configs x 1 shapes)
ALL PASS - full SM90 feature x variant matrix validated vs cuBLAS
  achieved clock: mean 1980 MHz  (min 1980 max 1980)   power mean 189 W   n=2


## SUMMARY  (cfg | layout | shape | pass | maxrelerr | abs | ourTF | cubTF | %cuB)
```
128x128x64_2x1_v_sk2_dh_rot2_pdl_d6 tn 2048x512x16384 PASS 0.00000 0.000000 601.87 571.60 105.3%
128x128x64_2x1_v_sk2_dh_rot2_pdl_d6_cs tn 2048x512x16384 PASS 0.00000 0.000000 601.68 569.25 105.7%
128x128x64_2x1_v_sk2_dh_rot2_pdl_d6_dw tn 2048x512x16384 PASS 0.00000 0.000000 599.93 574.13 104.5%
128x128x64_2x1_v_sk2_dh_rot2_pdl_d6_dw_uad tn 2048x512x16384 PASS 0.00000 0.000000 558.52 572.82 97.5%
128x128x64_2x1_v_sk2_dh_rot2_pdl_d6_dw_uad_cs tn 2048x512x16384 PASS 0.00000 0.000000 602.81 574.51 104.9%
128x128x64_2x1_v_sk2_dh_rot2_pdl_d6 tn 2048x512x16384 PASS 0.00000 0.000000 614.79 642.65 95.7%
128x128x64_2x1_v_sk2_dh_rot2_pdl_d6_cs tn 2048x512x16384 PASS 0.00000 0.000000 622.21 639.51 97.3%
128x128x64_2x1_v_sk2_dh_rot2_pdl_d6_dw tn 2048x512x16384 PASS 0.00000 0.000000 618.60 655.43 94.4%
128x128x64_2x1_v_sk2_dh_rot2_pdl_d6_dw_uad tn 2048x512x16384 PASS 0.00000 0.000000 612.90 657.64 93.2%
128x128x64_2x1_v_sk2_dh_rot2_pdl_d6_dw_uad_cs tn 2048x512x16384 PASS 0.00000 0.000000 642.58 645.22 99.6%
```

## ACHIEVED CLOCKS
```
  achieved clock: mean 1410 MHz  (min 1410 max 1410)   power mean 168 W   n=3
  achieved clock: mean 1410 MHz  (min 1410 max 1410)   power mean 142 W   n=4
  achieved clock: mean 1410 MHz  (min 1410 max 1410)   power mean 163 W   n=3
  achieved clock: mean 1410 MHz  (min 1410 max 1410)   power mean 167 W   n=2
  achieved clock: mean 1410 MHz  (min 1410 max 1410)   power mean 132 W   n=4
  achieved clock: mean 1977 MHz  (min 1965 max 1980)   power mean 173 W   n=5
  achieved clock: mean 1965 MHz  (min 1920 max 1980)   power mean 171 W   n=5
  achieved clock: mean 1806 MHz  (min 795 max 1980)   power mean 174 W   n=7
  achieved clock: mean 1932 MHz  (min 1740 max 1980)   power mean 179 W   n=5
  achieved clock: mean 1980 MHz  (min 1980 max 1980)   power mean 189 W   n=2
```
DONE 2026-08-26T15:07:02+00:00
