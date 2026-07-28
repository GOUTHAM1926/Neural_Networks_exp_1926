=========================================================================
SM103 Cluster GEMM Unified Template Test Suite
Device : NVIDIA B300 SXM6 AC  (SM 10.3)
=========================================================================

=== Testing Layout: NN | Config: 128x128x64 (cta_group::2, horizontal wave, clusterN=1, mcastA=0, dyn=0) ===
Shape                  M     N     K Batch alpha  beta  Template   cuBLAS     %cuBLAS  MaxRelErr   Status  wall(ms)
--------------------------------------------------------------------------------------------------------------------

[  1/ 84] tiny                 256   128    64     1   1.0   0.0      0.50 T      1.12 T   44.9%   0.00e+00  PASS       168.6
[  2/ 84] small                256   256   128     1   1.0   0.0      2.02 T      4.45 T   45.4%   0.00e+00  PASS         2.5
[  3/ 84] medium               512   512   128     1   1.0   0.0      8.02 T     17.56 T   45.7%   0.00e+00  PASS         4.1
[  4/ 84] large               1024  1024   256     1   1.0   0.0     63.86 T    135.74 T   47.0%   0.00e+00  PASS        30.7
[  5/ 84] batched-4            512   512   128     4   1.0   0.0     32.34 T     61.25 T   52.8%   0.00e+00  PASS        10.3
[  6/ 84] batched-8            256   256   128     8   1.0   0.0     16.15 T     35.05 T   46.1%   0.00e+00  PASS         9.2
[  7/ 84] alpha-scaling        512   512   128     1   1.5   0.0      8.11 T     16.95 T   47.9%   0.00e+00  PASS         2.7
[  8/ 84] beta-accumulate      256   256   128     1   1.0   0.5      1.16 T      4.36 T   26.6%   0.00e+00  PASS         3.6
[  9/ 84] alpha-beta-batch     512   512   128     2   1.2   0.8      9.27 T     32.94 T   28.1%   6.85e-03  PASS        12.5
[ 10/ 84] cmp-square4096      4096  4096  4096     1   1.0   0.0   1287.69 T   1322.12 T   97.4%   0.00e+00  PASS       452.1
[ 11/ 84] cmp-wideN-gate      4096 12288  4096     1   1.0   0.0   1287.58 T   1523.32 T   84.5%   0.00e+00  PASS       948.4
[ 12/ 84] cmp-skinny-K        4096  1024  4096     1   1.0   0.0    921.75 T   1161.62 T   79.4%   0.00e+00  PASS       269.6
Result: 12 / 12 PASSED   (suite wall time: 1914.2 ms)

=== Testing Layout: NN | Config: 128x256x64 (cta_group::2, horizontal wave, clusterN=1, mcastA=0, dyn=0) ===
Shape                  M     N     K Batch alpha  beta  Template   cuBLAS     %cuBLAS  MaxRelErr   Status  wall(ms)
--------------------------------------------------------------------------------------------------------------------

[ 13/ 84] tiny-256             512   256   128     1   1.0   0.0      3.51 T      8.86 T   39.6%   0.00e+00  PASS         3.3
[ 14/ 84] small-256            512   512   128     1   1.0   0.0      6.54 T     17.59 T   37.2%   0.00e+00  PASS         3.1
[ 15/ 84] medium-256          1024  1024   256     1   1.0   0.0     51.52 T    136.23 T   37.8%   0.00e+00  PASS        29.9
[ 16/ 84] batched-4-256        512   512   128     4   1.0   0.0     26.13 T     70.02 T   37.3%   0.00e+00  PASS        13.7
[ 17/ 84] alpha-beta-256       512   512   128     1   1.5   0.5      2.96 T     17.67 T   16.8%   1.91e-06  PASS         7.4
[ 18/ 84] cmp-square4096      4096  4096  4096     1   1.0   0.0   1229.31 T   1329.98 T   92.4%   0.00e+00  PASS       473.4
[ 19/ 84] cmp-wideN-gate      4096 12288  4096     1   1.0   0.0   1405.87 T   1524.93 T   92.2%   0.00e+00  PASS       940.8
[ 20/ 84] cmp-skinny-K        4096  1024  4096     1   1.0   0.0    922.78 T   1157.67 T   79.7%   0.00e+00  PASS       280.5
Result: 8 / 8 PASSED   (suite wall time: 1752.0 ms)

=== Testing Layout: NT | Config: 128x128x64 (cta_group::2, horizontal wave, clusterN=1, mcastA=0, dyn=0) ===
Shape                  M     N     K Batch alpha  beta  Template   cuBLAS     %cuBLAS  MaxRelErr   Status  wall(ms)
--------------------------------------------------------------------------------------------------------------------

[ 21/ 84] tiny                 256   128    64     1   1.0   0.0      0.50 T      1.11 T   45.4%   0.00e+00  PASS         2.0
[ 22/ 84] small                256   256   128     1   1.0   0.0      2.02 T      4.46 T   45.3%   0.00e+00  PASS         2.3
[ 23/ 84] medium               512   512   128     1   1.0   0.0      8.04 T     16.43 T   48.9%   0.00e+00  PASS         3.9
[ 24/ 84] large               1024  1024   256     1   1.0   0.0     64.68 T    133.15 T   48.6%   0.00e+00  PASS        18.2
[ 25/ 84] batched-4            512   512   128     4   1.0   0.0     32.36 T     67.95 T   47.6%   0.00e+00  PASS        10.2
[ 26/ 84] batched-8            256   256   128     8   1.0   0.0     16.10 T     36.47 T   44.2%   0.00e+00  PASS        12.6
[ 27/ 84] alpha-scaling        512   512   128     1   1.5   0.0      8.08 T     18.72 T   43.2%   0.00e+00  PASS         2.7
[ 28/ 84] beta-accumulate      256   256   128     1   1.0   0.5      1.16 T      4.57 T   25.3%   0.00e+00  PASS         3.6
[ 29/ 84] alpha-beta-batch     512   512   128     2   1.2   0.8      9.29 T     31.87 T   29.1%   6.99e-03  PASS        20.8
[ 30/ 84] cmp-square4096      4096  4096  4096     1   1.0   0.0   1291.21 T   1329.57 T   97.1%   0.00e+00  PASS       448.4
[ 31/ 84] cmp-wideN-gate      4096 12288  4096     1   1.0   0.0   1271.59 T   1520.58 T   83.6%   0.00e+00  PASS       968.1
[ 32/ 84] cmp-skinny-K        4096  1024  4096     1   1.0   0.0    924.52 T   1176.12 T   78.6%   0.00e+00  PASS       273.7
Result: 12 / 12 PASSED   (suite wall time: 1766.5 ms)

=== Testing Layout: NT | Config: 128x256x64 (cta_group::2, horizontal wave, clusterN=1, mcastA=0, dyn=0) ===
Shape                  M     N     K Batch alpha  beta  Template   cuBLAS     %cuBLAS  MaxRelErr   Status  wall(ms)
--------------------------------------------------------------------------------------------------------------------

[ 33/ 84] tiny-256             512   256   128     1   1.0   0.0      3.67 T      9.18 T   39.9%   0.00e+00  PASS         3.2
[ 34/ 84] small-256            512   512   128     1   1.0   0.0      6.49 T     18.47 T   35.1%   0.00e+00  PASS         3.0
[ 35/ 84] medium-256          1024  1024   256     1   1.0   0.0     51.80 T    143.64 T   36.1%   0.00e+00  PASS        11.3
[ 36/ 84] batched-4-256        512   512   128     4   1.0   0.0     25.85 T     72.69 T   35.6%   0.00e+00  PASS        38.6
[ 37/ 84] alpha-beta-256       512   512   128     1   1.5   0.5      2.96 T     13.95 T   21.2%   0.00e+00  PASS         7.8
[ 38/ 84] cmp-square4096      4096  4096  4096     1   1.0   0.0   1227.98 T   1322.95 T   92.8%   0.00e+00  PASS       477.5
[ 39/ 84] cmp-wideN-gate      4096 12288  4096     1   1.0   0.0   1386.64 T   1526.01 T   90.9%   0.00e+00  PASS       942.8
[ 40/ 84] cmp-skinny-K        4096  1024  4096     1   1.0   0.0    925.40 T   1162.75 T   79.6%   0.00e+00  PASS       276.9
Result: 8 / 8 PASSED   (suite wall time: 1761.0 ms)

=== Testing Layout: TN | Config: 128x128x64 (cta_group::2, horizontal wave, clusterN=1, mcastA=0, dyn=0) ===
Shape                  M     N     K Batch alpha  beta  Template   cuBLAS     %cuBLAS  MaxRelErr   Status  wall(ms)
--------------------------------------------------------------------------------------------------------------------

[ 41/ 84] tiny                 256   128    64     1   1.0   0.0      0.50 T      1.05 T   48.1%   0.00e+00  PASS         2.0
[ 42/ 84] small                256   256   128     1   1.0   0.0      2.03 T      4.67 T   43.4%   0.00e+00  PASS         1.5
[ 43/ 84] medium               512   512   128     1   1.0   0.0      8.10 T     18.46 T   43.9%   0.00e+00  PASS         3.9
[ 44/ 84] large               1024  1024   256     1   1.0   0.0     63.72 T    145.13 T   43.9%   0.00e+00  PASS        33.6
[ 45/ 84] batched-4            512   512   128     4   1.0   0.0     32.27 T     71.61 T   45.1%   0.00e+00  PASS        10.4
[ 46/ 84] batched-8            256   256   128     8   1.0   0.0     16.12 T     37.45 T   43.0%   0.00e+00  PASS        32.3
[ 47/ 84] alpha-scaling        512   512   128     1   1.5   0.0      8.05 T     18.17 T   44.3%   0.00e+00  PASS         3.0
[ 48/ 84] beta-accumulate      256   256   128     1   1.0   0.5      1.16 T      4.55 T   25.5%   0.00e+00  PASS         3.8
[ 49/ 84] alpha-beta-batch     512   512   128     2   1.2   0.8      9.25 T     37.17 T   24.9%   6.37e-03  PASS        35.2
[ 50/ 84] cmp-square4096      4096  4096  4096     1   1.0   0.0   1288.45 T   1336.29 T   96.4%   0.00e+00  PASS       449.1
[ 51/ 84] cmp-wideN-gate      4096 12288  4096     1   1.0   0.0   1287.81 T   1521.57 T   84.6%   0.00e+00  PASS       936.1
[ 52/ 84] cmp-skinny-K        4096  1024  4096     1   1.0   0.0    927.36 T   1159.80 T   80.0%   0.00e+00  PASS       252.1
Result: 12 / 12 PASSED   (suite wall time: 1763.1 ms)

=== Testing Layout: TN | Config: 128x256x64 (cta_group::2, horizontal wave, clusterN=1, mcastA=0, dyn=0) ===
Shape                  M     N     K Batch alpha  beta  Template   cuBLAS     %cuBLAS  MaxRelErr   Status  wall(ms)
--------------------------------------------------------------------------------------------------------------------

[ 53/ 84] tiny-256             512   256   128     1   1.0   0.0      3.45 T      8.64 T   39.9%   0.00e+00  PASS         2.3
[ 54/ 84] small-256            512   512   128     1   1.0   0.0      6.50 T     18.43 T   35.3%   0.00e+00  PASS         3.0
[ 55/ 84] medium-256          1024  1024   256     1   1.0   0.0     51.91 T    143.76 T   36.1%   0.00e+00  PASS        18.7
[ 56/ 84] batched-4-256        512   512   128     4   1.0   0.0     25.92 T     71.82 T   36.1%   0.00e+00  PASS        10.3
[ 57/ 84] alpha-beta-256       512   512   128     1   1.5   0.5      2.96 T     18.09 T   16.4%   0.00e+00  PASS         7.4
[ 58/ 84] cmp-square4096      4096  4096  4096     1   1.0   0.0   1230.90 T   1332.97 T   92.3%   0.00e+00  PASS       448.4
[ 59/ 84] cmp-wideN-gate      4096 12288  4096     1   1.0   0.0   1409.57 T   1522.93 T   92.6%   0.00e+00  PASS       947.9
[ 60/ 84] cmp-skinny-K        4096  1024  4096     1   1.0   0.0    929.97 T   1155.56 T   80.5%   0.00e+00  PASS       255.7
Result: 8 / 8 PASSED   (suite wall time: 1693.8 ms)

########## P1: 128x256 tile, VERTICAL wave (2x1) ##########

=== Testing Layout: NN | Config: 128x256x64 (cta_group::2, vertical wave, clusterN=1, mcastA=0, dyn=0) ===
Shape                  M     N     K Batch alpha  beta  Template   cuBLAS     %cuBLAS  MaxRelErr   Status  wall(ms)
--------------------------------------------------------------------------------------------------------------------

[ 61/ 84] tiny-256             512   256   128     1   1.0   0.0      3.36 T      9.15 T   36.8%   0.00e+00  PASS         2.2
[ 62/ 84] small-256            512   512   128     1   1.0   0.0      6.48 T     18.59 T   34.8%   0.00e+00  PASS         3.0
[ 63/ 84] medium-256          1024  1024   256     1   1.0   0.0     51.81 T    142.66 T   36.3%   0.00e+00  PASS        19.3
[ 64/ 84] batched-4-256        512   512   128     4   1.0   0.0     25.99 T     71.79 T   36.2%   0.00e+00  PASS        10.2
[ 65/ 84] alpha-beta-256       512   512   128     1   1.5   0.5      2.96 T     17.98 T   16.5%   1.91e-06  PASS         5.8
[ 66/ 84] cmp-square4096      4096  4096  4096     1   1.0   0.0   1226.28 T   1332.52 T   92.0%   0.00e+00  PASS       449.1
[ 67/ 84] cmp-wideN-gate      4096 12288  4096     1   1.0   0.0   1480.41 T   1524.47 T   97.1%   0.00e+00  PASS       957.0
[ 68/ 84] cmp-skinny-K        4096  1024  4096     1   1.0   0.0    935.72 T   1169.08 T   80.0%   0.00e+00  PASS       275.6
Result: 8 / 8 PASSED   (suite wall time: 1722.2 ms)

=== Testing Layout: NT | Config: 128x256x64 (cta_group::2, vertical wave, clusterN=1, mcastA=0, dyn=0) ===
Shape                  M     N     K Batch alpha  beta  Template   cuBLAS     %cuBLAS  MaxRelErr   Status  wall(ms)
--------------------------------------------------------------------------------------------------------------------

[ 69/ 84] tiny-256             512   256   128     1   1.0   0.0      3.68 T      6.28 T   58.6%   0.00e+00  PASS         2.6
[ 70/ 84] small-256            512   512   128     1   1.0   0.0      6.53 T     18.20 T   35.9%   0.00e+00  PASS         3.4
[ 71/ 84] medium-256          1024  1024   256     1   1.0   0.0     51.45 T    115.35 T   44.6%   0.00e+00  PASS        35.8
[ 72/ 84] batched-4-256        512   512   128     4   1.0   0.0     25.99 T     72.19 T   36.0%   0.00e+00  PASS        10.8
[ 73/ 84] alpha-beta-256       512   512   128     1   1.5   0.5      2.97 T     18.27 T   16.2%   0.00e+00  PASS         5.9
[ 74/ 84] cmp-square4096      4096  4096  4096     1   1.0   0.0   1224.82 T   1326.16 T   92.4%   0.00e+00  PASS       449.2
[ 75/ 84] cmp-wideN-gate      4096 12288  4096     1   1.0   0.0   1478.93 T   1525.34 T   97.0%   0.00e+00  PASS       957.6
[ 76/ 84] cmp-skinny-K        4096  1024  4096     1   1.0   0.0    932.59 T   1155.68 T   80.7%   0.00e+00  PASS       276.3
Result: 8 / 8 PASSED   (suite wall time: 1741.5 ms)

=== Testing Layout: TN | Config: 128x256x64 (cta_group::2, vertical wave, clusterN=1, mcastA=0, dyn=0) ===
Shape                  M     N     K Batch alpha  beta  Template   cuBLAS     %cuBLAS  MaxRelErr   Status  wall(ms)
--------------------------------------------------------------------------------------------------------------------

[ 77/ 84] tiny-256             512   256   128     1   1.0   0.0      3.23 T      9.22 T   35.0%   0.00e+00  PASS         2.3
[ 78/ 84] small-256            512   512   128     1   1.0   0.0      6.49 T     17.92 T   36.2%   0.00e+00  PASS         3.0
[ 79/ 84] medium-256          1024  1024   256     1   1.0   0.0     51.81 T    142.12 T   36.5%   0.00e+00  PASS        10.5
[ 80/ 84] batched-4-256        512   512   128     4   1.0   0.0     25.92 T     73.62 T   35.2%   0.00e+00  PASS        33.7
[ 81/ 84] alpha-beta-256       512   512   128     1   1.5   0.5      2.97 T     18.16 T   16.4%   0.00e+00  PASS         6.2
[ 82/ 84] cmp-square4096      4096  4096  4096     1   1.0   0.0   1230.91 T   1334.11 T   92.3%   0.00e+00  PASS       436.9
[ 83/ 84] cmp-wideN-gate      4096 12288  4096     1   1.0   0.0   1479.82 T   1521.93 T   97.2%   0.00e+00  PASS       933.6
[ 84/ 84] cmp-skinny-K        4096  1024  4096     1   1.0   0.0    927.32 T   1151.96 T   80.5%   0.00e+00  PASS       268.0
Result: 8 / 8 PASSED   (suite wall time: 1694.3 ms)

########## P2 2x2 (4-CTA) SUITES SKIPPED (set RUN_2X2=1 to include) ##########

########## P3 2x2f (dynamic cluster) SKIPPED — RUN_2X2F=1 to include ##########

########## P4 1x2 (cta_group::1) SKIPPED — RUN_1X2=1 to include ##########

########## P5 swap-AB SKIPPED — RUN_1X2B=1 to include ##########

########## P6 B2 (160x192) SKIPPED — RUN_1X2C=1 to include ##########

########## P7 composite-M 256x256_2x2 SKIPPED — RUN_CM256=1 to include ##########

########## P8 mixed composite 192x128 SKIPPED — RUN_CM192=1 to include ##########

########## P9 masked 160x128 SKIPPED — RUN_CM160=1 to include ##########

########## P10 small tiles (64x128/64x64) SKIPPED — RUN_CM64=1 to include ##########

########## B5-MIRROR: kernel-for-kernel (42 cases) ##########

=== B5-MIRROR: kernel-for-kernel — each shape runs OUR clone of the nvjet kernel cuBLAS picked (§16.3) ===

preheating: 60s of q3-8b-LMhead ...
preheat done.

Shape                nvjet kernel           our clone        ours(T) cuBLAS(T)    %cuB
------------------------------------------------------------------------------------------

q25-0.5b-Q           160x192_2x2 (unbuilt)  128x256_2x2*356.3T    517.2T   68.9%
q25-0.5b-K           64x64_2x2              64x64_2x2          89.1T    188.2T   47.3%
q25-0.5b-Gate        128x192_2x1_v          192x128_v         750.6T    944.2T   79.5%
q25-0.5b-Down        256x128_2x2            128x256_2x2       784.5T    897.6T   87.4%
q25-0.5b-LMhead      128x256_2x1_v          128x256_v        1520.7T   1436.7T  105.8%
q25-1.5b-Q           128x192_2x2f           192x128_2x2f      622.0T    871.7T   71.3%
q25-1.5b-K           128x64_2x2             64x128_2x2        220.8T    414.6T   53.3%
q25-1.5b-Gate        128x256_2x1_v          128x256_v        1137.2T   1248.2T   91.1%
q25-1.5b-Down        128x192_2x2f           192x128_2x2f      962.9T   1137.7T   84.6%
q25-1.5b-LMhead      128x256_2x1_v          128x256_v        1413.5T   1475.9T   95.8%
q25-3b-Q             160x192_1x2            swap160x192       504.1T    993.5T   50.7%
q25-3b-K             128x64_2x2             64x128_2x2        265.6T    473.8T   56.1%
q25-3b-Gate          128x256_2x1_v          128x256_v        1220.4T   1289.5T   94.6%
q25-3b-Down          128x256_2x2            128x256_2x2      1140.3T   1190.0T   95.8%
q25-3b-LMhead        128x256_2x1_v          128x256_v        1449.1T   1542.4T   94.0%
q25-7b-Q             128x160_2x1_v          160x128m_v        835.1T   1162.4T   71.8%
q25-7b-K             128x128_2x2            128x128_2x2       606.4T    838.0T   72.4%
q25-7b-Gate          128x256_2x1_v          128x256_v        1369.8T   1408.0T   97.3%
q25-7b-Down          320x192_2x2f (unbuilt) 128x128_v*       1298.1T   1215.5T  106.8%
q25-7b-LMhead        128x256_2x1_v          128x256_v        1487.9T   1561.3T   95.3%
q3-0.6b-Q            160x192_1x2            swap160x192       397.1T    830.0T   47.8%
q3-0.6b-K            128x256_2x2            128x256_2x2       459.6T    717.1T   64.1%
q3-0.6b-O            128x256_2x2            128x256_2x2       683.2T    893.6T   76.5%
q3-0.6b-Gate         128x256_2x2f           128x256_2x2f      777.5T    994.4T   78.2%
q3-0.6b-Down         128x256_2x2            128x256_2x2       781.3T    987.8T   79.1%
q3-0.6b-LMhead       128x256_2x1_v          128x256_v        1414.7T   1475.6T   95.9%
q3-1.7b-Q            160x192_1x2            swap160x192       492.0T   1014.5T   48.5%
q3-1.7b-K            128x256_2x2            128x256_2x2       640.2T    878.2T   72.9%
q3-1.7b-Gate         128x256_2x2f           128x256_2x2f     1044.9T   1146.5T   91.1%
q3-1.7b-Down         128x256_2x2            128x256_2x2      1065.0T   1151.8T   92.5%
q3-1.7b-LMhead       128x256_2x1_v          128x256_v        1444.8T   1476.2T   97.9%
q3-4b-Q              128x256_2x2f           128x256_2x2f     1016.0T   1127.0T   90.1%
q3-4b-K              128x256_2x2            128x256_2x2       733.8T    938.7T   78.2%
q3-4b-O              128x192_2x1_v          192x128_v        1001.2T   1212.0T   82.6%
q3-4b-Gate           128x256_2x1_v          128x256_v        1248.7T   1279.1T   97.6%
q3-4b-Down           128x192_2x1_v          192x128_v        1146.3T   1331.1T   86.1%
q3-4b-LMhead         128x256_2x1_v          128x256_v        1433.0T   1484.2T   96.6%
q3-8b-Q              128x256_2x2f           128x256_2x2f     1079.4T   1166.1T   92.6%
q3-8b-K              128x256_2x2            128x256_2x2       842.7T   1031.7T   81.7%
q3-8b-Gate           128x256_2x1_v          128x256_v        1329.6T   1358.1T   97.9%
q3-8b-Down           256x256_2x2            256x256_2x2      1191.7T   1217.8T   97.9%
q3-8b-LMhead         128x256_2x1_v          128x256_v        1475.9T   1533.6T   96.2%
------------------------------------------------------------------------------------------

MIRROR avg: 82.4% of cuBLAS   kernel-for-kernel wins: 2 / 42   (* = best-alt for the 2 unbuilt sw32 tiles)

=========================================================================
ALL TESTS PASSED SUCCESSFULLY
=========================================================================

## SUMMARY
