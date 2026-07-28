=========================================================================
SM103 Cluster GEMM Unified Template Test Suite
Device : NVIDIA B300 SXM6 AC  (SM 10.3)
=========================================================================

=== Testing Layout: NN | Config: 128x128x64 (cta_group::2, horizontal wave, clusterN=1, mcastA=0, dyn=0) ===
Shape                  M     N     K Batch alpha  beta  Template   cuBLAS     %cuBLAS  MaxRelErr   Status  wall(ms)
--------------------------------------------------------------------------------------------------------------------

[  1/406] tiny                 256   128    64     1   1.0   0.0      0.66 T      1.12 T   59.1%   0.00e+00  PASS       167.0
[  2/406] small                256   256   128     1   1.0   0.0      2.67 T      4.56 T   58.6%   0.00e+00  PASS         2.4
[  3/406] medium               512   512   128     1   1.0   0.0     10.70 T     15.97 T   67.0%   0.00e+00  PASS         4.1
[  4/406] large               1024  1024   256     1   1.0   0.0     82.81 T    137.97 T   60.0%   0.00e+00  PASS        26.3
[  5/406] batched-4            512   512   128     4   1.0   0.0     42.88 T     70.02 T   61.2%   0.00e+00  PASS        10.4
[  6/406] batched-8            256   256   128     8   1.0   0.0     21.54 T     32.33 T   66.6%   0.00e+00  PASS         9.3
[  7/406] alpha-scaling        512   512   128     1   1.5   0.0     10.78 T     17.54 T   61.4%   0.00e+00  PASS         2.7
[  8/406] beta-accumulate      256   256   128     1   1.0   0.5      1.61 T      4.28 T   37.6%   0.00e+00  PASS         3.6
[  9/406] alpha-beta-batch     512   512   128     2   1.2   0.8     11.47 T     35.16 T   32.6%   6.85e-03  PASS        18.7
[ 10/406] cmp-square4096      4096  4096  4096     1   1.0   0.0   1629.04 T   1738.25 T   93.7%   0.00e+00  PASS       499.3
[ 11/406] cmp-wideN-gate      4096 12288  4096     1   1.0   0.0   1421.77 T   1823.86 T   78.0%   0.00e+00  PASS       968.0
[ 12/406] cmp-skinny-K        4096  1024  4096     1   1.0   0.0   1195.04 T   1574.75 T   75.9%   0.00e+00  PASS       268.6
Result: 12 / 12 PASSED   (suite wall time: 1980.5 ms)

=== Testing Layout: NN | Config: 128x256x64 (cta_group::2, horizontal wave, clusterN=1, mcastA=0, dyn=0) ===
Shape                  M     N     K Batch alpha  beta  Template   cuBLAS     %cuBLAS  MaxRelErr   Status  wall(ms)
--------------------------------------------------------------------------------------------------------------------

[ 13/406] tiny-256             512   256   128     1   1.0   0.0      4.59 T      8.97 T   51.1%   0.00e+00  PASS         3.4
[ 14/406] small-256            512   512   128     1   1.0   0.0      8.08 T     18.02 T   44.8%   0.00e+00  PASS         3.0
[ 15/406] medium-256          1024  1024   256     1   1.0   0.0     64.26 T    141.22 T   45.5%   0.00e+00  PASS        32.2
[ 16/406] batched-4-256        512   512   128     4   1.0   0.0     32.34 T     68.56 T   47.2%   0.00e+00  PASS        10.3
[ 17/406] alpha-beta-256       512   512   128     1   1.5   0.5      4.06 T     16.35 T   24.8%   1.91e-06  PASS         7.3
[ 18/406] cmp-square4096      4096  4096  4096     1   1.0   0.0   1599.32 T   1741.25 T   91.8%   0.00e+00  PASS       492.6
[ 19/406] cmp-wideN-gate      4096 12288  4096     1   1.0   0.0   1569.08 T   1823.34 T   86.1%   0.00e+00  PASS       977.2
[ 20/406] cmp-skinny-K        4096  1024  4096     1   1.0   0.0   1285.53 T   1578.68 T   81.4%   0.00e+00  PASS       273.5
Result: 8 / 8 PASSED   (suite wall time: 1799.6 ms)

=== Testing Layout: NT | Config: 128x128x64 (cta_group::2, horizontal wave, clusterN=1, mcastA=0, dyn=0) ===
Shape                  M     N     K Batch alpha  beta  Template   cuBLAS     %cuBLAS  MaxRelErr   Status  wall(ms)
--------------------------------------------------------------------------------------------------------------------

[ 21/406] tiny                 256   128    64     1   1.0   0.0      0.66 T      1.12 T   59.0%   0.00e+00  PASS         2.0
[ 22/406] small                256   256   128     1   1.0   0.0      2.65 T      4.47 T   59.3%   0.00e+00  PASS         2.3
[ 23/406] medium               512   512   128     1   1.0   0.0     10.71 T     17.73 T   60.4%   0.00e+00  PASS         4.0
[ 24/406] large               1024  1024   256     1   1.0   0.0     85.53 T    140.10 T   61.1%   0.00e+00  PASS        18.1
[ 25/406] batched-4            512   512   128     4   1.0   0.0     43.01 T     69.64 T   61.8%   0.00e+00  PASS        31.6
[ 26/406] batched-8            256   256   128     8   1.0   0.0     21.45 T     33.27 T   64.5%   0.00e+00  PASS        11.6
[ 27/406] alpha-scaling        512   512   128     1   1.5   0.0     10.75 T     18.77 T   57.3%   0.00e+00  PASS         2.6
[ 28/406] beta-accumulate      256   256   128     1   1.0   0.5      1.61 T      4.67 T   34.5%   0.00e+00  PASS         3.5
[ 29/406] alpha-beta-batch     512   512   128     2   1.2   0.8     12.11 T     37.28 T   32.5%   6.99e-03  PASS        24.3
[ 30/406] cmp-square4096      4096  4096  4096     1   1.0   0.0   1631.95 T   1736.18 T   94.0%   0.00e+00  PASS       473.2
[ 31/406] cmp-wideN-gate      4096 12288  4096     1   1.0   0.0   1385.14 T   1817.37 T   76.2%   0.00e+00  PASS       974.9
[ 32/406] cmp-skinny-K        4096  1024  4096     1   1.0   0.0   1192.32 T   1576.48 T   75.6%   0.00e+00  PASS       269.2
Result: 12 / 12 PASSED   (suite wall time: 1817.4 ms)

=== Testing Layout: NT | Config: 128x256x64 (cta_group::2, horizontal wave, clusterN=1, mcastA=0, dyn=0) ===
Shape                  M     N     K Batch alpha  beta  Template   cuBLAS     %cuBLAS  MaxRelErr   Status  wall(ms)
--------------------------------------------------------------------------------------------------------------------

[ 33/406] tiny-256             512   256   128     1   1.0   0.0      4.66 T      9.25 T   50.3%   0.00e+00  PASS         3.2
[ 34/406] small-256            512   512   128     1   1.0   0.0      8.85 T     17.06 T   51.8%   0.00e+00  PASS         2.9
[ 35/406] medium-256          1024  1024   256     1   1.0   0.0     64.40 T    148.47 T   43.4%   0.00e+00  PASS        18.0
[ 36/406] batched-4-256        512   512   128     4   1.0   0.0     32.38 T     73.55 T   44.0%   0.00e+00  PASS        34.5
[ 37/406] alpha-beta-256       512   512   128     1   1.5   0.5      4.07 T     18.64 T   21.8%   0.00e+00  PASS         7.4
[ 38/406] cmp-square4096      4096  4096  4096     1   1.0   0.0   1608.36 T   1737.17 T   92.6%   0.00e+00  PASS       465.8
[ 39/406] cmp-wideN-gate      4096 12288  4096     1   1.0   0.0   1543.54 T   1821.01 T   84.8%   0.00e+00  PASS       973.4
[ 40/406] cmp-skinny-K        4096  1024  4096     1   1.0   0.0   1287.77 T   1583.34 T   81.3%   0.00e+00  PASS       260.1
Result: 8 / 8 PASSED   (suite wall time: 1765.3 ms)

=== Testing Layout: TN | Config: 128x128x64 (cta_group::2, horizontal wave, clusterN=1, mcastA=0, dyn=0) ===
Shape                  M     N     K Batch alpha  beta  Template   cuBLAS     %cuBLAS  MaxRelErr   Status  wall(ms)
--------------------------------------------------------------------------------------------------------------------

[ 41/406] tiny                 256   128    64     1   1.0   0.0      0.67 T      1.21 T   55.3%   0.00e+00  PASS         2.0
[ 42/406] small                256   256   128     1   1.0   0.0      2.69 T      4.19 T   64.2%   0.00e+00  PASS         1.4
[ 43/406] medium               512   512   128     1   1.0   0.0     10.58 T     18.62 T   56.8%   0.00e+00  PASS         3.9
[ 44/406] large               1024  1024   256     1   1.0   0.0     82.93 T    147.69 T   56.2%   0.00e+00  PASS        18.5
[ 45/406] batched-4            512   512   128     4   1.0   0.0     42.33 T     72.60 T   58.3%   0.00e+00  PASS        32.1
[ 46/406] batched-8            256   256   128     8   1.0   0.0     21.43 T     37.35 T   57.4%   0.00e+00  PASS         9.2
[ 47/406] alpha-scaling        512   512   128     1   1.5   0.0     10.77 T     18.64 T   57.8%   0.00e+00  PASS         2.6
[ 48/406] beta-accumulate      256   256   128     1   1.0   0.5      1.62 T      4.57 T   35.4%   0.00e+00  PASS         3.7
[ 49/406] alpha-beta-batch     512   512   128     2   1.2   0.8     11.79 T     37.02 T   31.8%   6.37e-03  PASS        12.6
[ 50/406] cmp-square4096      4096  4096  4096     1   1.0   0.0   1641.49 T   1738.54 T   94.4%   0.00e+00  PASS       478.1
[ 51/406] cmp-wideN-gate      4096 12288  4096     1   1.0   0.0   1418.76 T   1827.11 T   77.7%   0.00e+00  PASS       959.6
[ 52/406] cmp-skinny-K        4096  1024  4096     1   1.0   0.0   1195.17 T   1570.49 T   76.1%   0.00e+00  PASS       276.8
Result: 12 / 12 PASSED   (suite wall time: 1800.5 ms)

=== Testing Layout: TN | Config: 128x256x64 (cta_group::2, horizontal wave, clusterN=1, mcastA=0, dyn=0) ===
Shape                  M     N     K Batch alpha  beta  Template   cuBLAS     %cuBLAS  MaxRelErr   Status  wall(ms)
--------------------------------------------------------------------------------------------------------------------

[ 53/406] tiny-256             512   256   128     1   1.0   0.0      4.59 T      9.19 T   50.0%   0.00e+00  PASS         2.3
[ 54/406] small-256            512   512   128     1   1.0   0.0      8.18 T     18.49 T   44.3%   0.00e+00  PASS         3.0
[ 55/406] medium-256          1024  1024   256     1   1.0   0.0     64.63 T    147.88 T   43.7%   0.00e+00  PASS        10.6
[ 56/406] batched-4-256        512   512   128     4   1.0   0.0     32.51 T     73.07 T   44.5%   0.00e+00  PASS        10.2
[ 57/406] alpha-beta-256       512   512   128     1   1.5   0.5      4.06 T     18.86 T   21.5%   0.00e+00  PASS         7.3
[ 58/406] cmp-square4096      4096  4096  4096     1   1.0   0.0   1608.63 T   1737.97 T   92.6%   0.00e+00  PASS       464.5
[ 59/406] cmp-wideN-gate      4096 12288  4096     1   1.0   0.0   1573.03 T   1827.08 T   86.1%   0.00e+00  PASS       955.7
[ 60/406] cmp-skinny-K        4096  1024  4096     1   1.0   0.0   1282.69 T   1577.64 T   81.3%   0.00e+00  PASS       275.3
Result: 8 / 8 PASSED   (suite wall time: 1729.0 ms)

########## P1: 128x256 tile, VERTICAL wave (2x1) ##########

=== Testing Layout: NN | Config: 128x256x64 (cta_group::2, vertical wave, clusterN=1, mcastA=0, dyn=0) ===
Shape                  M     N     K Batch alpha  beta  Template   cuBLAS     %cuBLAS  MaxRelErr   Status  wall(ms)
--------------------------------------------------------------------------------------------------------------------

[ 61/406] tiny-256             512   256   128     1   1.0   0.0      4.31 T      9.19 T   46.9%   0.00e+00  PASS         2.2
[ 62/406] small-256            512   512   128     1   1.0   0.0      8.11 T     18.25 T   44.5%   0.00e+00  PASS         3.0
[ 63/406] medium-256          1024  1024   256     1   1.0   0.0     64.55 T    134.76 T   47.9%   0.00e+00  PASS        10.5
[ 64/406] batched-4-256        512   512   128     4   1.0   0.0     32.67 T     72.88 T   44.8%   0.00e+00  PASS        10.3
[ 65/406] alpha-beta-256       512   512   128     1   1.5   0.5      4.07 T     18.28 T   22.3%   1.91e-06  PASS         5.8
[ 66/406] cmp-square4096      4096  4096  4096     1   1.0   0.0   1603.89 T   1740.69 T   92.1%   0.00e+00  PASS       471.7
[ 67/406] cmp-wideN-gate      4096 12288  4096     1   1.0   0.0   1813.60 T   1822.39 T   99.5%   0.00e+00  PASS       949.4
[ 68/406] cmp-skinny-K        4096  1024  4096     1   1.0   0.0   1286.23 T   1579.03 T   81.5%   0.00e+00  PASS       274.4
Result: 8 / 8 PASSED   (suite wall time: 1727.3 ms)

=== Testing Layout: NT | Config: 128x256x64 (cta_group::2, vertical wave, clusterN=1, mcastA=0, dyn=0) ===
Shape                  M     N     K Batch alpha  beta  Template   cuBLAS     %cuBLAS  MaxRelErr   Status  wall(ms)
--------------------------------------------------------------------------------------------------------------------

[ 69/406] tiny-256             512   256   128     1   1.0   0.0      4.58 T      9.25 T   49.5%   0.00e+00  PASS         2.2
[ 70/406] small-256            512   512   128     1   1.0   0.0      8.51 T     18.45 T   46.1%   0.00e+00  PASS         3.1
[ 71/406] medium-256          1024  1024   256     1   1.0   0.0     64.74 T    141.10 T   45.9%   0.00e+00  PASS        10.6
[ 72/406] batched-4-256        512   512   128     4   1.0   0.0     32.44 T     73.14 T   44.4%   0.00e+00  PASS        10.3
[ 73/406] alpha-beta-256       512   512   128     1   1.5   0.5      4.05 T     18.66 T   21.7%   0.00e+00  PASS         5.7
[ 74/406] cmp-square4096      4096  4096  4096     1   1.0   0.0   1604.67 T   1739.28 T   92.3%   0.00e+00  PASS       466.2
[ 75/406] cmp-wideN-gate      4096 12288  4096     1   1.0   0.0   1809.77 T   1823.25 T   99.3%   0.00e+00  PASS       959.1
[ 76/406] cmp-skinny-K        4096  1024  4096     1   1.0   0.0   1288.00 T   1582.52 T   81.4%   0.00e+00  PASS       277.0
Result: 8 / 8 PASSED   (suite wall time: 1734.1 ms)

=== Testing Layout: TN | Config: 128x256x64 (cta_group::2, vertical wave, clusterN=1, mcastA=0, dyn=0) ===
Shape                  M     N     K Batch alpha  beta  Template   cuBLAS     %cuBLAS  MaxRelErr   Status  wall(ms)
--------------------------------------------------------------------------------------------------------------------

[ 77/406] tiny-256             512   256   128     1   1.0   0.0      4.19 T      9.27 T   45.2%   0.00e+00  PASS         2.2
[ 78/406] small-256            512   512   128     1   1.0   0.0      8.09 T     18.46 T   43.8%   0.00e+00  PASS         3.0
[ 79/406] medium-256          1024  1024   256     1   1.0   0.0     64.54 T    145.95 T   44.2%   0.00e+00  PASS        11.3
[ 80/406] batched-4-256        512   512   128     4   1.0   0.0     32.25 T     69.27 T   46.6%   0.00e+00  PASS        37.3
[ 81/406] alpha-beta-256       512   512   128     1   1.5   0.5      4.06 T     18.01 T   22.6%   0.00e+00  PASS         5.8
[ 82/406] cmp-square4096      4096  4096  4096     1   1.0   0.0   1599.17 T   1742.49 T   91.8%   0.00e+00  PASS       443.9
[ 83/406] cmp-wideN-gate      4096 12288  4096     1   1.0   0.0   1813.75 T   1824.08 T   99.4%   0.00e+00  PASS       958.8
[ 84/406] cmp-skinny-K        4096  1024  4096     1   1.0   0.0   1287.77 T   1571.87 T   81.9%   0.00e+00  PASS       278.1
Result: 8 / 8 PASSED   (suite wall time: 1740.4 ms)

########## P2: 128x256 tile, 2x2 cluster (4-CTA) ##########

=== Testing Layout: NN | Config: 128x256x64 (cta_group::2, horizontal wave, clusterN=2, mcastA=0, dyn=0) ===
Shape                  M     N     K Batch alpha  beta  Template   cuBLAS     %cuBLAS  MaxRelErr   Status  wall(ms)
--------------------------------------------------------------------------------------------------------------------

[ 85/406] 2x2-small            512   512   128     1   1.0   0.0      8.05 T     18.66 T   43.1%   0.00e+00  PASS         3.2
[ 86/406] 2x2-medium          1024  1024   256     1   1.0   0.0     64.53 T    148.47 T   43.5%   0.00e+00  PASS        16.8
[ 87/406] 2x2-large           2048  2048   512     1   1.0   0.0    413.10 T    753.40 T   54.8%   0.00e+00  PASS        38.8
[ 88/406] 2x2-batched          512   512   128     4   1.0   0.0     32.34 T     74.20 T   43.6%   0.00e+00  PASS        18.4
[ 89/406] 2x2-alpha-beta       512   512   128     1   1.5   0.5      4.06 T     18.62 T   21.8%   1.91e-06  PASS         5.6
[ 90/406] 2x2-square4096      4096  4096  4096     1   1.0   0.0   1631.98 T   1738.85 T   93.9%   0.00e+00  PASS       457.6
[ 91/406] 2x2-wideN           4096 12288  4096     1   1.0   0.0   1498.47 T   1821.87 T   82.2%   0.00e+00  PASS       952.4
Result: 7 / 7 PASSED   (suite wall time: 1492.8 ms)

=== Testing Layout: NT | Config: 128x256x64 (cta_group::2, horizontal wave, clusterN=2, mcastA=0, dyn=0) ===
Shape                  M     N     K Batch alpha  beta  Template   cuBLAS     %cuBLAS  MaxRelErr   Status  wall(ms)
--------------------------------------------------------------------------------------------------------------------

[ 92/406] 2x2-small            512   512   128     1   1.0   0.0      8.13 T     18.40 T   44.1%   0.00e+00  PASS         3.0
[ 93/406] 2x2-medium          1024  1024   256     1   1.0   0.0     64.38 T    148.01 T   43.5%   0.00e+00  PASS         9.7
[ 94/406] 2x2-large           2048  2048   512     1   1.0   0.0    412.22 T    751.08 T   54.9%   0.00e+00  PASS        76.6
[ 95/406] 2x2-batched          512   512   128     4   1.0   0.0     32.47 T     73.42 T   44.2%   0.00e+00  PASS        11.9
[ 96/406] 2x2-alpha-beta       512   512   128     1   1.5   0.5      4.06 T     18.41 T   22.0%   0.00e+00  PASS         5.7
[ 97/406] 2x2-square4096      4096  4096  4096     1   1.0   0.0   1623.35 T   1731.98 T   93.7%   0.00e+00  PASS       440.9
[ 98/406] 2x2-wideN           4096 12288  4096     1   1.0   0.0   1450.87 T   1814.56 T   80.0%   0.00e+00  PASS       967.0
Result: 7 / 7 PASSED   (suite wall time: 1514.8 ms)

=== Testing Layout: TN | Config: 128x256x64 (cta_group::2, horizontal wave, clusterN=2, mcastA=0, dyn=0) ===
Shape                  M     N     K Batch alpha  beta  Template   cuBLAS     %cuBLAS  MaxRelErr   Status  wall(ms)
--------------------------------------------------------------------------------------------------------------------

[ 99/406] 2x2-small            512   512   128     1   1.0   0.0      8.06 T     15.98 T   50.5%   0.00e+00  PASS         3.0
[100/406] 2x2-medium          1024  1024   256     1   1.0   0.0     64.18 T    148.21 T   43.3%   0.00e+00  PASS        22.2
[101/406] 2x2-large           2048  2048   512     1   1.0   0.0    416.11 T    745.24 T   55.8%   0.00e+00  PASS        49.7
[102/406] 2x2-batched          512   512   128     4   1.0   0.0     32.28 T     50.67 T   63.7%   0.00e+00  PASS        11.5
[103/406] 2x2-alpha-beta       512   512   128     1   1.5   0.5      4.06 T     18.35 T   22.1%   0.00e+00  PASS         6.2
[104/406] 2x2-square4096      4096  4096  4096     1   1.0   0.0   1631.98 T   1741.00 T   93.7%   0.00e+00  PASS       479.0
[105/406] 2x2-wideN           4096 12288  4096     1   1.0   0.0   1499.54 T   1828.58 T   82.0%   0.00e+00  PASS       951.0
Result: 7 / 7 PASSED   (suite wall time: 1522.7 ms)

########## P2-M2: 128x256 2x2 cluster, A-MULTICAST ON ##########

=== Testing Layout: NN | Config: 128x256x64 (cta_group::2, horizontal wave, clusterN=2, mcastA=1, dyn=0) ===
Shape                  M     N     K Batch alpha  beta  Template   cuBLAS     %cuBLAS  MaxRelErr   Status  wall(ms)
--------------------------------------------------------------------------------------------------------------------

[106/406] 2x2-small            512   512   128     1   1.0   0.0      8.03 T     18.58 T   43.2%   0.00e+00  PASS         3.0
[107/406] 2x2-medium          1024  1024   256     1   1.0   0.0     63.12 T    147.95 T   42.7%   0.00e+00  PASS         9.9
[108/406] 2x2-large           2048  2048   512     1   1.0   0.0    412.34 T    747.73 T   55.1%   0.00e+00  PASS        57.6
[109/406] 2x2-batched          512   512   128     4   1.0   0.0     32.21 T     68.01 T   47.4%   0.00e+00  PASS        27.1
[110/406] 2x2-alpha-beta       512   512   128     1   1.5   0.5      4.06 T     18.70 T   21.7%   1.91e-06  PASS         5.7
[111/406] 2x2-square4096      4096  4096  4096     1   1.0   0.0   1624.85 T   1738.54 T   93.5%   0.00e+00  PASS       445.6
[112/406] 2x2-wideN           4096 12288  4096     1   1.0   0.0   1495.15 T   1820.07 T   82.1%   0.00e+00  PASS       956.6
Result: 7 / 7 PASSED   (suite wall time: 1505.6 ms)

=== Testing Layout: NT | Config: 128x256x64 (cta_group::2, horizontal wave, clusterN=2, mcastA=1, dyn=0) ===
Shape                  M     N     K Batch alpha  beta  Template   cuBLAS     %cuBLAS  MaxRelErr   Status  wall(ms)
--------------------------------------------------------------------------------------------------------------------

[113/406] 2x2-small            512   512   128     1   1.0   0.0      8.06 T     13.47 T   59.8%   0.00e+00  PASS         3.5
[114/406] 2x2-medium          1024  1024   256     1   1.0   0.0     64.27 T    146.53 T   43.9%   0.00e+00  PASS        30.7
[115/406] 2x2-large           2048  2048   512     1   1.0   0.0    412.85 T    754.67 T   54.7%   0.00e+00  PASS        51.8
[116/406] 2x2-batched          512   512   128     4   1.0   0.0     32.23 T     74.93 T   43.0%   0.00e+00  PASS        36.3
[117/406] 2x2-alpha-beta       512   512   128     1   1.5   0.5      4.07 T     18.44 T   22.1%   0.00e+00  PASS         6.0
[118/406] 2x2-square4096      4096  4096  4096     1   1.0   0.0   1616.32 T   1736.67 T   93.1%   0.00e+00  PASS       437.9
[119/406] 2x2-wideN           4096 12288  4096     1   1.0   0.0   1458.26 T   1821.03 T   80.1%   0.00e+00  PASS       961.3
Result: 7 / 7 PASSED   (suite wall time: 1527.4 ms)

=== Testing Layout: TN | Config: 128x256x64 (cta_group::2, horizontal wave, clusterN=2, mcastA=1, dyn=0) ===
Shape                  M     N     K Batch alpha  beta  Template   cuBLAS     %cuBLAS  MaxRelErr   Status  wall(ms)
--------------------------------------------------------------------------------------------------------------------

[120/406] 2x2-small            512   512   128     1   1.0   0.0      8.05 T     18.81 T   42.8%   0.00e+00  PASS         3.0
[121/406] 2x2-medium          1024  1024   256     1   1.0   0.0     64.38 T    148.60 T   43.3%   0.00e+00  PASS        33.5
[122/406] 2x2-large           2048  2048   512     1   1.0   0.0    412.79 T    748.98 T   55.1%   0.00e+00  PASS        57.2
[123/406] 2x2-batched          512   512   128     4   1.0   0.0     32.06 T     67.30 T   47.6%   0.00e+00  PASS        37.1
[124/406] 2x2-alpha-beta       512   512   128     1   1.5   0.5      4.06 T     18.35 T   22.1%   0.00e+00  PASS         5.7
[125/406] 2x2-square4096      4096  4096  4096     1   1.0   0.0   1633.56 T   1737.06 T   94.0%   0.00e+00  PASS       442.6
[126/406] 2x2-wideN           4096 12288  4096     1   1.0   0.0   1499.74 T   1827.35 T   82.1%   0.00e+00  PASS       950.1
Result: 7 / 7 PASSED   (suite wall time: 1529.2 ms)

########## P3: 128x256 2x2f (dynamic cluster, pref 4 / fb 2) ##########

=== Testing Layout: NN | Config: 128x256x64 (cta_group::2, horizontal wave, clusterN=2, mcastA=0, dyn=1) ===
Shape                  M     N     K Batch alpha  beta  Template   cuBLAS     %cuBLAS  MaxRelErr   Status  wall(ms)
--------------------------------------------------------------------------------------------------------------------

[127/406] 2x2-small            512   512   128     1   1.0   0.0      8.07 T     13.64 T   59.2%   0.00e+00  PASS         3.5
[128/406] 2x2-medium          1024  1024   256     1   1.0   0.0     64.27 T    147.23 T   43.7%   0.00e+00  PASS        10.5
[129/406] 2x2-large           2048  2048   512     1   1.0   0.0    415.66 T    747.52 T   55.6%   0.00e+00  PASS        53.1
[130/406] 2x2-batched          512   512   128     4   1.0   0.0     32.34 T     72.19 T   44.8%   0.00e+00  PASS        25.3
[131/406] 2x2-alpha-beta       512   512   128     1   1.5   0.5      4.05 T     18.33 T   22.1%   1.91e-06  PASS         5.6
[132/406] 2x2-square4096      4096  4096  4096     1   1.0   0.0   1601.85 T   1737.90 T   92.2%   0.00e+00  PASS       438.7
[133/406] 2x2-wideN           4096 12288  4096     1   1.0   0.0   1571.60 T   1820.03 T   86.4%   0.00e+00  PASS       930.5
Result: 7 / 7 PASSED   (suite wall time: 1467.2 ms)

=== Testing Layout: NT | Config: 128x256x64 (cta_group::2, horizontal wave, clusterN=2, mcastA=0, dyn=1) ===
Shape                  M     N     K Batch alpha  beta  Template   cuBLAS     %cuBLAS  MaxRelErr   Status  wall(ms)
--------------------------------------------------------------------------------------------------------------------

[134/406] 2x2-small            512   512   128     1   1.0   0.0      8.08 T     18.65 T   43.3%   0.00e+00  PASS         3.0
[135/406] 2x2-medium          1024  1024   256     1   1.0   0.0     64.49 T    146.65 T   44.0%   0.00e+00  PASS        26.9
[136/406] 2x2-large           2048  2048   512     1   1.0   0.0    412.47 T    745.65 T   55.3%   0.00e+00  PASS        49.2
[137/406] 2x2-batched          512   512   128     4   1.0   0.0     32.13 T     73.81 T   43.5%   0.00e+00  PASS        11.1
[138/406] 2x2-alpha-beta       512   512   128     1   1.5   0.5      4.07 T     18.44 T   22.1%   0.00e+00  PASS         5.6
[139/406] 2x2-square4096      4096  4096  4096     1   1.0   0.0   1600.33 T   1736.64 T   92.2%   0.00e+00  PASS       465.8
[140/406] 2x2-wideN           4096 12288  4096     1   1.0   0.0   1539.18 T   1821.26 T   84.5%   0.00e+00  PASS       964.0
Result: 7 / 7 PASSED   (suite wall time: 1525.6 ms)

=== Testing Layout: TN | Config: 128x256x64 (cta_group::2, horizontal wave, clusterN=2, mcastA=0, dyn=1) ===
Shape                  M     N     K Batch alpha  beta  Template   cuBLAS     %cuBLAS  MaxRelErr   Status  wall(ms)
--------------------------------------------------------------------------------------------------------------------

[141/406] 2x2-small            512   512   128     1   1.0   0.0      8.07 T     18.80 T   42.9%   0.00e+00  PASS         3.0
[142/406] 2x2-medium          1024  1024   256     1   1.0   0.0     64.39 T    148.08 T   43.5%   0.00e+00  PASS        30.5
[143/406] 2x2-large           2048  2048   512     1   1.0   0.0    412.15 T    754.03 T   54.7%   0.00e+00  PASS        51.8
[144/406] 2x2-batched          512   512   128     4   1.0   0.0     32.18 T     67.14 T   47.9%   0.00e+00  PASS        11.1
[145/406] 2x2-alpha-beta       512   512   128     1   1.5   0.5      4.05 T     18.62 T   21.8%   0.00e+00  PASS         5.6
[146/406] 2x2-square4096      4096  4096  4096     1   1.0   0.0   1602.18 T   1740.61 T   92.0%   0.00e+00  PASS       446.6
[147/406] 2x2-wideN           4096 12288  4096     1   1.0   0.0   1575.35 T   1825.88 T   86.3%   0.00e+00  PASS       938.7
Result: 7 / 7 PASSED   (suite wall time: 1487.3 ms)

########## P3-M2: 2x2f + A-MULTICAST (adaptive) ##########

=== Testing Layout: NN | Config: 128x256x64 (cta_group::2, horizontal wave, clusterN=2, mcastA=1, dyn=1) ===
Shape                  M     N     K Batch alpha  beta  Template   cuBLAS     %cuBLAS  MaxRelErr   Status  wall(ms)
--------------------------------------------------------------------------------------------------------------------

[148/406] 2x2-small            512   512   128     1   1.0   0.0      8.06 T     18.43 T   43.7%   0.00e+00  PASS         3.0
[149/406] 2x2-medium          1024  1024   256     1   1.0   0.0     64.32 T    135.57 T   47.4%   0.00e+00  PASS        31.9
[150/406] 2x2-large           2048  2048   512     1   1.0   0.0    412.79 T    748.57 T   55.1%   0.00e+00  PASS        71.0
[151/406] 2x2-batched          512   512   128     4   1.0   0.0     32.23 T     69.62 T   46.3%   0.00e+00  PASS        27.0
[152/406] 2x2-alpha-beta       512   512   128     1   1.5   0.5      4.05 T     18.11 T   22.3%   1.91e-06  PASS         5.8
[153/406] 2x2-square4096      4096  4096  4096     1   1.0   0.0   1589.29 T   1740.30 T   91.3%   0.00e+00  PASS       473.5
[154/406] 2x2-wideN           4096 12288  4096     1   1.0   0.0   1572.80 T   1818.42 T   86.5%   0.00e+00  PASS       965.9
Result: 7 / 7 PASSED   (suite wall time: 1578.2 ms)

=== Testing Layout: NT | Config: 128x256x64 (cta_group::2, horizontal wave, clusterN=2, mcastA=1, dyn=1) ===
Shape                  M     N     K Batch alpha  beta  Template   cuBLAS     %cuBLAS  MaxRelErr   Status  wall(ms)
--------------------------------------------------------------------------------------------------------------------

[155/406] 2x2-small            512   512   128     1   1.0   0.0      8.14 T     18.48 T   44.1%   0.00e+00  PASS         3.1
[156/406] 2x2-medium          1024  1024   256     1   1.0   0.0     64.44 T    146.72 T   43.9%   0.00e+00  PASS        37.7
[157/406] 2x2-large           2048  2048   512     1   1.0   0.0    413.87 T    756.80 T   54.7%   0.00e+00  PASS        64.3
[158/406] 2x2-batched          512   512   128     4   1.0   0.0     32.18 T     74.63 T   43.1%   0.00e+00  PASS        26.9
[159/406] 2x2-alpha-beta       512   512   128     1   1.5   0.5      4.05 T     18.09 T   22.4%   0.00e+00  PASS         5.7
[160/406] 2x2-square4096      4096  4096  4096     1   1.0   0.0   1600.72 T   1738.64 T   92.1%   0.00e+00  PASS       435.1
[161/406] 2x2-wideN           4096 12288  4096     1   1.0   0.0   1546.23 T   1819.95 T   85.0%   0.00e+00  PASS       930.2
Result: 7 / 7 PASSED   (suite wall time: 1502.9 ms)

=== Testing Layout: TN | Config: 128x256x64 (cta_group::2, horizontal wave, clusterN=2, mcastA=1, dyn=1) ===
Shape                  M     N     K Batch alpha  beta  Template   cuBLAS     %cuBLAS  MaxRelErr   Status  wall(ms)
--------------------------------------------------------------------------------------------------------------------

[162/406] 2x2-small            512   512   128     1   1.0   0.0      8.07 T     18.68 T   43.2%   0.00e+00  PASS         3.0
[163/406] 2x2-medium          1024  1024   256     1   1.0   0.0     64.47 T    148.41 T   43.4%   0.00e+00  PASS        26.6
[164/406] 2x2-large           2048  2048   512     1   1.0   0.0    414.96 T    755.09 T   55.0%   0.00e+00  PASS        49.4
[165/406] 2x2-batched          512   512   128     4   1.0   0.0     32.24 T     67.46 T   47.8%   0.00e+00  PASS        11.8
[166/406] 2x2-alpha-beta       512   512   128     1   1.5   0.5      4.07 T     18.58 T   21.9%   0.00e+00  PASS         5.7
[167/406] 2x2-square4096      4096  4096  4096     1   1.0   0.0   1602.00 T   1736.95 T   92.2%   0.00e+00  PASS       462.9
[168/406] 2x2-wideN           4096 12288  4096     1   1.0   0.0   1576.14 T   1823.00 T   86.5%   0.00e+00  PASS       930.0
Result: 7 / 7 PASSED   (suite wall time: 1489.3 ms)

########## P4: 1x2 cluster (cta_group::1), 128x192 bring-up ##########

=== Testing Layout: NN | Config: 128x192x64 (cta_group::2, horizontal wave, clusterN=2, mcastA=0, dyn=0) ===
Shape                  M     N     K Batch alpha  beta  Template   cuBLAS     %cuBLAS  MaxRelErr   Status  wall(ms)
--------------------------------------------------------------------------------------------------------------------

[169/406] 1x2-tiny             128   384   128     1   1.0   0.0      1.98 T      3.47 T   57.0%   0.00e+00  PASS         1.6
[170/406] 1x2-small            512   768   256     1   1.0   0.0     24.19 T     56.05 T   43.2%   0.00e+00  PASS        19.7
[171/406] 1x2-medium          1024  1152   512     1   1.0   0.0    145.22 T    330.69 T   43.9%   0.00e+00  PASS        47.3
[172/406] 1x2-batched          512   768   128     4   1.0   0.0     60.25 T    109.18 T   55.2%   0.00e+00  PASS        15.3
[173/406] 1x2-alpha-beta       512   768   128     1   1.5   0.5      6.97 T     27.75 T   25.1%   0.00e+00  PASS         9.7
[174/406] 1x2-qwen-q06        4096  2048  1024     1   1.0   0.0    831.84 T   1317.80 T   63.1%   0.00e+00  PASS       116.7
[175/406] 1x2-qwen-q3b        4096  2048  2048     1   1.0   0.0   1056.94 T   1544.84 T   68.4%   0.00e+00  PASS       198.8
Result: 7 / 7 PASSED   (suite wall time: 409.1 ms)

=== Testing Layout: NT | Config: 128x192x64 (cta_group::2, horizontal wave, clusterN=2, mcastA=0, dyn=0) ===
Shape                  M     N     K Batch alpha  beta  Template   cuBLAS     %cuBLAS  MaxRelErr   Status  wall(ms)
--------------------------------------------------------------------------------------------------------------------

[176/406] 1x2-tiny             128   384   128     1   1.0   0.0      2.00 T      3.47 T   57.6%   0.00e+00  PASS         1.6
[177/406] 1x2-small            512   768   256     1   1.0   0.0     24.17 T     55.16 T   43.8%   0.00e+00  PASS         6.8
[178/406] 1x2-medium          1024  1152   512     1   1.0   0.0    145.19 T    332.00 T   43.7%   0.00e+00  PASS        39.5
[179/406] 1x2-batched          512   768   128     4   1.0   0.0     61.94 T    111.25 T   55.7%   0.00e+00  PASS        38.8
[180/406] 1x2-alpha-beta       512   768   128     1   1.5   0.5      6.96 T     27.68 T   25.2%   0.00e+00  PASS         9.9
[181/406] 1x2-qwen-q06        4096  2048  1024     1   1.0   0.0    831.46 T   1358.48 T   61.2%   0.00e+00  PASS       134.6
[182/406] 1x2-qwen-q3b        4096  2048  2048     1   1.0   0.0   1053.00 T   1589.90 T   66.2%   0.00e+00  PASS       180.5
Result: 7 / 7 PASSED   (suite wall time: 411.6 ms)

=== Testing Layout: TN | Config: 128x192x64 (cta_group::2, horizontal wave, clusterN=2, mcastA=0, dyn=0) ===
Shape                  M     N     K Batch alpha  beta  Template   cuBLAS     %cuBLAS  MaxRelErr   Status  wall(ms)
--------------------------------------------------------------------------------------------------------------------

[183/406] 1x2-tiny             128   384   128     1   1.0   0.0      2.00 T      3.42 T   58.4%   0.00e+00  PASS         1.5
[184/406] 1x2-small            512   768   256     1   1.0   0.0     24.30 T     55.24 T   44.0%   0.00e+00  PASS        19.9
[185/406] 1x2-medium          1024  1152   512     1   1.0   0.0    145.86 T    334.95 T   43.5%   0.00e+00  PASS        42.7
[186/406] 1x2-batched          512   768   128     4   1.0   0.0     62.00 T    111.60 T   55.6%   0.00e+00  PASS        18.0
[187/406] 1x2-alpha-beta       512   768   128     1   1.5   0.5      6.97 T     28.05 T   24.9%   0.00e+00  PASS         9.4
[188/406] 1x2-qwen-q06        4096  2048  1024     1   1.0   0.0    828.18 T   1297.89 T   63.8%   0.00e+00  PASS       120.4
[189/406] 1x2-qwen-q3b        4096  2048  2048     1   1.0   0.0   1052.64 T   1539.86 T   68.4%   0.00e+00  PASS       186.4
Result: 7 / 7 PASSED   (suite wall time: 398.4 ms)

########## P4-M2: 1x2 + A-MULTICAST ##########

=== Testing Layout: NN | Config: 128x192x64 (cta_group::2, horizontal wave, clusterN=2, mcastA=1, dyn=0) ===
Shape                  M     N     K Batch alpha  beta  Template   cuBLAS     %cuBLAS  MaxRelErr   Status  wall(ms)
--------------------------------------------------------------------------------------------------------------------

[190/406] 1x2-tiny             128   384   128     1   1.0   0.0      2.01 T      3.37 T   59.5%   0.00e+00  PASS         1.5
[191/406] 1x2-small            512   768   256     1   1.0   0.0     24.23 T     54.85 T   44.2%   0.00e+00  PASS         5.8
[192/406] 1x2-medium          1024  1152   512     1   1.0   0.0    145.10 T    332.30 T   43.7%   0.00e+00  PASS        42.2
[193/406] 1x2-batched          512   768   128     4   1.0   0.0     59.24 T    108.94 T   54.4%   0.00e+00  PASS        14.0
[194/406] 1x2-alpha-beta       512   768   128     1   1.5   0.5      6.96 T     27.64 T   25.2%   0.00e+00  PASS         7.9
[195/406] 1x2-qwen-q06        4096  2048  1024     1   1.0   0.0    831.07 T   1311.04 T   63.4%   0.00e+00  PASS       123.3
[196/406] 1x2-qwen-q3b        4096  2048  2048     1   1.0   0.0   1052.43 T   1546.51 T   68.1%   0.00e+00  PASS       177.3
Result: 7 / 7 PASSED   (suite wall time: 371.9 ms)

=== Testing Layout: NT | Config: 128x192x64 (cta_group::2, horizontal wave, clusterN=2, mcastA=1, dyn=0) ===
Shape                  M     N     K Batch alpha  beta  Template   cuBLAS     %cuBLAS  MaxRelErr   Status  wall(ms)
--------------------------------------------------------------------------------------------------------------------

[197/406] 1x2-tiny             128   384   128     1   1.0   0.0      2.01 T      3.46 T   58.1%   0.00e+00  PASS         1.4
[198/406] 1x2-small            512   768   256     1   1.0   0.0     24.08 T     55.75 T   43.2%   0.00e+00  PASS        20.8
[199/406] 1x2-medium          1024  1152   512     1   1.0   0.0    145.02 T    333.47 T   43.5%   0.00e+00  PASS        39.5
[200/406] 1x2-batched          512   768   128     4   1.0   0.0     58.61 T    110.86 T   52.9%   0.00e+00  PASS        13.8
[201/406] 1x2-alpha-beta       512   768   128     1   1.5   0.5      6.96 T     27.89 T   25.0%   0.00e+00  PASS         7.9
[202/406] 1x2-qwen-q06        4096  2048  1024     1   1.0   0.0    831.65 T   1370.97 T   60.7%   0.00e+00  PASS       120.5
[203/406] 1x2-qwen-q3b        4096  2048  2048     1   1.0   0.0   1050.27 T   1585.80 T   66.2%   0.00e+00  PASS       193.3
Result: 7 / 7 PASSED   (suite wall time: 397.2 ms)

=== Testing Layout: TN | Config: 128x192x64 (cta_group::2, horizontal wave, clusterN=2, mcastA=1, dyn=0) ===
Shape                  M     N     K Batch alpha  beta  Template   cuBLAS     %cuBLAS  MaxRelErr   Status  wall(ms)
--------------------------------------------------------------------------------------------------------------------

[204/406] 1x2-tiny             128   384   128     1   1.0   0.0      1.98 T      3.47 T   57.0%   0.00e+00  PASS         1.4
[205/406] 1x2-small            512   768   256     1   1.0   0.0     24.27 T     56.10 T   43.3%   0.00e+00  PASS         5.9
[206/406] 1x2-medium          1024  1152   512     1   1.0   0.0    145.52 T    338.86 T   42.9%   0.00e+00  PASS        24.2
[207/406] 1x2-batched          512   768   128     4   1.0   0.0     59.37 T    108.80 T   54.6%   0.00e+00  PASS        29.2
[208/406] 1x2-alpha-beta       512   768   128     1   1.5   0.5      6.96 T     27.90 T   24.9%   0.00e+00  PASS         7.8
[209/406] 1x2-qwen-q06        4096  2048  1024     1   1.0   0.0    832.16 T   1296.16 T   64.2%   0.00e+00  PASS       107.6
[210/406] 1x2-qwen-q3b        4096  2048  2048     1   1.0   0.0   1044.14 T   1541.51 T   67.7%   0.00e+00  PASS       170.3
Result: 7 / 7 PASSED   (suite wall time: 346.4 ms)

########## P5: 1x2 swap-AB B1 (C^T, composite M'=192) ##########

=== Testing Layout: TN | Config: 128x192x64 (cta_group::2, horizontal wave, clusterN=2, mcastA=0, dyn=0) ===
Shape                  M     N     K Batch alpha  beta  Template   cuBLAS     %cuBLAS  MaxRelErr   Status  wall(ms)
--------------------------------------------------------------------------------------------------------------------

[211/406] 1x2b-tiny            128   384   128     1   1.0   0.0      1.22 T      3.49 T   34.8%   0.00e+00  PASS         1.5
[212/406] 1x2b-small           512   768   256     1   1.0   0.0     19.42 T     56.86 T   34.2%   0.00e+00  PASS        10.6
[213/406] 1x2b-medium         1024  1152   512     1   1.0   0.0     97.71 T    338.10 T   28.9%   0.00e+00  PASS        31.6
[214/406] 1x2b-batched         512   768   128     4   1.0   0.0     38.66 T    112.05 T   34.5%   0.00e+00  PASS        36.2
[215/406] 1x2b-alpha-beta      512   768   128     1   1.5   0.5      5.43 T     25.68 T   21.1%   0.00e+00  PASS         7.9
[216/406] 1x2b-qwen-q06       4096  2048  1024     1   1.0   0.0    694.84 T   1296.48 T   53.6%   0.00e+00  PASS       138.1
[217/406] 1x2b-qwen-q3b       4096  2048  2048     1   1.0   0.0    887.68 T   1543.29 T   57.5%   0.00e+00  PASS       169.0
Result: 7 / 7 PASSED   (suite wall time: 395.0 ms)

########## P5-M2: swap-AB + A-MULTICAST ##########

=== Testing Layout: TN | Config: 128x192x64 (cta_group::2, horizontal wave, clusterN=2, mcastA=1, dyn=0) ===
Shape                  M     N     K Batch alpha  beta  Template   cuBLAS     %cuBLAS  MaxRelErr   Status  wall(ms)
--------------------------------------------------------------------------------------------------------------------

[218/406] 1x2b-tiny            128   384   128     1   1.0   0.0      1.21 T      3.45 T   35.2%   0.00e+00  PASS         1.5
[219/406] 1x2b-small           512   768   256     1   1.0   0.0     19.38 T     56.10 T   34.5%   0.00e+00  PASS        11.0
[220/406] 1x2b-medium         1024  1152   512     1   1.0   0.0     97.37 T    333.18 T   29.2%   0.00e+00  PASS        27.8
[221/406] 1x2b-batched         512   768   128     4   1.0   0.0     38.96 T    111.75 T   34.9%   0.00e+00  PASS        33.9
[222/406] 1x2b-alpha-beta      512   768   128     1   1.5   0.5      5.40 T     27.63 T   19.5%   0.00e+00  PASS         7.9
[223/406] 1x2b-qwen-q06       4096  2048  1024     1   1.0   0.0    674.76 T   1292.73 T   52.2%   0.00e+00  PASS       101.5
[224/406] 1x2b-qwen-q3b       4096  2048  2048     1   1.0   0.0    879.04 T   1540.08 T   57.1%   0.00e+00  PASS       183.2
Result: 7 / 7 PASSED   (suite wall time: 366.7 ms)

########## P6: 1x2 swap-AB B2 (160x192, 64B-swizzle operand) ##########

=== Testing Layout: TN | Config: 160x192x64 (cta_group::2, horizontal wave, clusterN=2, mcastA=0, dyn=0) ===
Shape                  M     N     K Batch alpha  beta  Template   cuBLAS     %cuBLAS  MaxRelErr   Status  wall(ms)
--------------------------------------------------------------------------------------------------------------------

[225/406] 1x2c-tiny            160   384   128     1   1.0   0.0      1.50 T      4.30 T   34.9%   0.00e+00  PASS         2.8
[226/406] 1x2c-small           320   768   256     1   1.0   0.0     11.18 T     33.74 T   33.1%   0.00e+00  PASS         4.5
[227/406] 1x2c-oobM            192   384   128     1   1.0   0.0      1.83 T      5.32 T   34.4%   0.00e+00  PASS         1.6
[228/406] 1x2c-batched         480   768   128     4   1.0   0.0     36.49 T     96.22 T   37.9%   0.00e+00  PASS        33.8
[229/406] 1x2c-alpha-beta      320   768   128     1   1.5   0.5      2.78 T     17.26 T   16.1%   0.00e+00  PASS         5.8
[230/406] 1x2c-qwen-q06       4096  2048  1024     1   1.0   0.0    595.93 T   1291.80 T   46.1%   0.00e+00  PASS       132.3
[231/406] 1x2c-qwen-q3b       4096  2048  2048     1   1.0   0.0    759.82 T   1544.84 T   49.2%   0.00e+00  PASS       169.8
Result: 7 / 7 PASSED   (suite wall time: 350.6 ms)

########## P6-M2: B2 + A-MULTICAST ##########

=== Testing Layout: TN | Config: 160x192x64 (cta_group::2, horizontal wave, clusterN=2, mcastA=1, dyn=0) ===
Shape                  M     N     K Batch alpha  beta  Template   cuBLAS     %cuBLAS  MaxRelErr   Status  wall(ms)
--------------------------------------------------------------------------------------------------------------------

[232/406] 1x2c-tiny            160   384   128     1   1.0   0.0      1.52 T      4.29 T   35.4%   0.00e+00  PASS         1.6
[233/406] 1x2c-small           320   768   256     1   1.0   0.0     10.58 T     24.79 T   42.7%   0.00e+00  PASS         4.4
[234/406] 1x2c-oobM            192   384   128     1   1.0   0.0      1.83 T      5.24 T   34.9%   0.00e+00  PASS         1.6
[235/406] 1x2c-batched         480   768   128     4   1.0   0.0     36.40 T    104.81 T   34.7%   0.00e+00  PASS        33.8
[236/406] 1x2c-alpha-beta      320   768   128     1   1.5   0.5      2.78 T     17.25 T   16.1%   0.00e+00  PASS         5.7
[237/406] 1x2c-qwen-q06       4096  2048  1024     1   1.0   0.0    596.42 T   1293.66 T   46.1%   0.00e+00  PASS       101.0
[238/406] 1x2c-qwen-q3b       4096  2048  2048     1   1.0   0.0    759.23 T   1539.86 T   49.3%   0.00e+00  PASS       186.1
Result: 7 / 7 PASSED   (suite wall time: 334.2 ms)

########## Qwen 42-shape sweep (P1 impact) ##########

=== Qwen 42-shape sweep (TN, 128x256 tile): horizontal vs vertical vs cuBLAS ===
Shape                wave         N      K     TFLOPS   %cuB     cuBLAS  Status  wall(ms)  best
------------------------------------------------------------------------------------------------

[239/406] q25-0.5b-Q           horiz      896    896     552.5T  65.3%     845.6T  ok          86.6
[240/406] q25-0.5b-Q           vert       896    896     570.0T  67.3%     847.3T  ok          69.1  best=vert
[241/406] q25-0.5b-K           horiz      128    896      90.6T  34.6%     261.4T  ok          65.6
[242/406] q25-0.5b-K           vert       128    896      90.5T  36.0%     251.7T  ok          66.1  best=horiz
[243/406] q25-0.5b-Gate        horiz     4864    896    1241.4T  86.0%    1443.3T  ok         185.4
[244/406] q25-0.5b-Gate        vert      4864    896    1241.7T  85.9%    1446.3T  ok         185.6  best=vert
[245/406] q25-0.5b-Down        horiz      896   4864    1195.7T 101.8%    1175.0T  ok         298.0
[246/406] q25-0.5b-Down        vert       896   4864    1217.3T 100.8%    1207.9T  ok         287.2  best=vert
[247/406] q25-0.5b-LMhead      horiz   151936    896    1288.0T  76.1%    1691.6T  ok        3868.3
[248/406] q25-0.5b-LMhead      vert    151936    896    1682.2T  99.4%    1692.2T  ok        3917.5  best=vert
[249/406] q25-1.5b-Q           horiz     1536   1536     908.2T  67.3%    1349.4T  ok         119.9
[250/406] q25-1.5b-Q           vert      1536   1536     889.0T  65.8%    1350.1T  ok         137.9  best=horiz
[251/406] q25-1.5b-K           horiz      256   1536     227.1T  36.6%     621.2T  ok          80.1
[252/406] q25-1.5b-K           vert       256   1536     222.5T  35.8%     620.6T  ok          96.6  best=horiz
[253/406] q25-1.5b-Gate        horiz     8960   1536    1624.2T  93.5%    1737.8T  ok         352.2
[254/406] q25-1.5b-Gate        vert      8960   1536    1614.6T  93.2%    1732.9T  ok         382.4  best=horiz
[255/406] q25-1.5b-Down        horiz     1536   8960    1253.1T  79.2%    1582.3T  ok         612.7
[256/406] q25-1.5b-Down        vert      1536   8960    1248.3T  79.2%    1575.4T  ok         602.0  best=horiz
[257/406] q25-1.5b-LMhead      horiz   151936   1536    1381.9T  79.9%    1730.5T  ok        4940.8
[258/406] q25-1.5b-LMhead      vert    151936   1536    1792.1T 102.3%    1751.5T  ok        4925.9  best=vert
[259/406] q25-3b-Q             horiz     2048   2048    1285.0T  84.2%    1526.9T  ok         173.5
[260/406] q25-3b-Q             vert      2048   2048    1277.7T  83.2%    1536.0T  ok         185.2  best=horiz
[261/406] q25-3b-K             horiz      256   2048     259.6T  35.9%     722.4T  ok         105.5
[262/406] q25-3b-K             vert       256   2048     259.8T  35.9%     723.7T  ok         100.7  best=vert
[263/406] q25-3b-Gate          horiz    11008   2048    1695.0T  96.0%    1765.1T  ok         526.5
[264/406] q25-3b-Gate          vert     11008   2048    1691.6T  95.6%    1769.3T  ok         530.9  best=horiz
[265/406] q25-3b-Down          horiz     2048  11008    1622.4T  95.7%    1694.6T  ok         796.7
[266/406] q25-3b-Down          vert      2048  11008    1619.5T  95.8%    1690.3T  ok         790.7  best=horiz
[267/406] q25-3b-LMhead        horiz   151936   2048    1415.3T 103.7%    1364.7T  ok        5820.6
[268/406] q25-3b-LMhead        vert    151936   2048    1832.7T 104.4%    1755.0T  ok        5870.8  best=vert
[269/406] q25-7b-Q             horiz     3584   3584    1422.5T  86.3%    1649.1T  ok         378.4
[270/406] q25-7b-Q             vert      3584   3584    1424.5T  86.0%    1655.5T  ok         390.9  best=vert
[271/406] q25-7b-K             horiz      512   3584     662.3T  48.8%    1356.9T  ok         195.3
[272/406] q25-7b-K             vert       512   3584     663.3T  48.5%    1367.6T  ok         186.0  best=vert
[273/406] q25-7b-Gate          horiz    18944   3584    1446.7T  77.7%    1861.9T  ok        1214.5
[274/406] q25-7b-Gate          vert     18944   3584    1868.4T 100.5%    1859.6T  ok        1223.4  best=vert
[275/406] q25-7b-Down          horiz     3584  18944    1497.0T  90.2%    1659.4T  ok        1709.3
[276/406] q25-7b-Down          vert      3584  18944    1493.8T  90.2%    1656.7T  ok        1727.6  best=horiz
[277/406] q25-7b-LMhead        horiz   152064   3584    1362.8T  97.6%    1396.3T  ok        8535.9
[278/406] q25-7b-LMhead        vert    152064   3584    1873.5T 124.0%    1511.0T  ok        8494.1  best=vert
[279/406] q3-0.6b-Q            horiz     2048   1024    1036.2T  80.0%    1295.4T  ok         110.8
[280/406] q3-0.6b-Q            vert      2048   1024    1039.0T  80.0%    1298.7T  ok         145.1  best=vert
[281/406] q3-0.6b-K            horiz     1024   1024     689.1T  64.9%    1062.3T  ok          97.1
[282/406] q3-0.6b-K            vert      1024   1024     688.7T  64.1%    1074.0T  ok          84.4  best=horiz
[283/406] q3-0.6b-O            horiz     1024   2048    1038.3T  75.6%    1373.9T  ok         144.6
[284/406] q3-0.6b-O            vert      1024   2048    1038.4T  75.9%    1369.0T  ok         145.3  best=vert
[285/406] q3-0.6b-Gate         horiz     3072   1024    1139.3T  76.6%    1487.2T  ok         148.6
[286/406] q3-0.6b-Gate         vert      3072   1024    1136.3T  76.3%    1489.0T  ok         143.3  best=horiz
[287/406] q3-0.6b-Down         horiz     1024   3072    1135.9T  75.3%    1508.6T  ok         212.2
[288/406] q3-0.6b-Down         vert      1024   3072    1120.5T  73.9%    1515.6T  ok         193.0  best=horiz
[289/406] q3-0.6b-LMhead       horiz   151936   1024    1314.1T  76.2%    1723.7T  ok        4078.0
[290/406] q3-0.6b-LMhead       vert    151936   1024    1715.0T  98.8%    1735.1T  ok        4127.7  best=vert
[291/406] q3-1.7b-Q            horiz     2048   2048    1281.8T  83.3%    1539.6T  ok         168.6
[292/406] q3-1.7b-Q            vert      2048   2048    1284.4T  83.6%    1536.2T  ok         170.0  best=vert
[293/406] q3-1.7b-K            horiz     1024   2048    1037.0T  75.7%    1370.6T  ok         141.9
[294/406] q3-1.7b-K            vert      1024   2048    1037.3T  75.8%    1368.9T  ok         144.8  best=vert
[295/406] q3-1.7b-Gate         horiz     6144   2048    1539.7T  90.6%    1699.6T  ok         334.6
[296/406] q3-1.7b-Gate         vert      6144   2048    1529.3T  90.2%    1696.3T  ok         335.6  best=horiz
[297/406] q3-1.7b-Down         horiz     2048   6144    1559.5T  91.4%    1705.6T  ok         469.9
[298/406] q3-1.7b-Down         vert      2048   6144    1558.7T  91.4%    1704.8T  ok         464.7  best=horiz
[299/406] q3-1.7b-LMhead       horiz   151936   2048    1415.0T 104.4%    1356.0T  ok        5779.9
[300/406] q3-1.7b-LMhead       vert    151936   2048    1835.7T 103.2%    1778.9T  ok        5793.6  best=vert
[301/406] q3-4b-Q              horiz     4096   2560    1534.0T  89.8%    1708.3T  ok         314.7
[302/406] q3-4b-Q              vert      4096   2560    1526.4T  89.4%    1708.3T  ok         290.7  best=horiz
[303/406] q3-4b-K              horiz     1024   2560    1054.9T  72.9%    1447.4T  ok         162.7
[304/406] q3-4b-K              vert      1024   2560    1107.5T  76.9%    1439.8T  ok         167.4  best=vert
[305/406] q3-4b-O              horiz     2560   4096    1339.0T  77.1%    1735.9T  ok         341.1
[306/406] q3-4b-O              vert      2560   4096    1326.4T  76.1%    1743.1T  ok         379.7  best=horiz
[307/406] q3-4b-Gate           horiz     9728   2560    1689.1T  95.4%    1770.7T  ok         548.6
[308/406] q3-4b-Gate           vert      9728   2560    1707.9T  96.3%    1772.9T  ok         541.3  best=vert
[309/406] q3-4b-Down           horiz     2560   9728    1416.3T  84.6%    1673.5T  ok         791.2
[310/406] q3-4b-Down           vert      2560   9728    1405.4T  83.8%    1677.6T  ok         758.8  best=horiz
[311/406] q3-4b-LMhead         horiz   151936   2560    1417.0T 103.1%    1374.2T  ok        6693.5
[312/406] q3-4b-LMhead         vert    151936   2560    1863.3T 125.5%    1485.2T  ok        6674.2  best=vert
[313/406] q3-8b-Q              horiz     4096   4096    1611.9T  92.8%    1737.8T  ok         429.4
[314/406] q3-8b-Q              vert      4096   4096    1606.3T  92.3%    1740.5T  ok         433.1  best=horiz
[315/406] q3-8b-K              horiz     1024   4096    1285.7T  81.5%    1577.9T  ok         254.7
[316/406] q3-8b-K              vert      1024   4096    1286.5T  81.9%    1570.5T  ok         266.9  best=vert
[317/406] q3-8b-Gate           horiz    12288   4096    1572.3T  86.1%    1826.5T  ok         926.3
[318/406] q3-8b-Gate           vert     12288   4096    1812.2T  99.5%    1821.7T  ok         954.9  best=vert
[319/406] q3-8b-Down           horiz     4096  12288    1652.6T  92.6%    1784.9T  ok        1210.8
[320/406] q3-8b-Down           vert      4096  12288    1653.2T  92.6%    1784.7T  ok        1193.2  best=vert
[321/406] q3-8b-LMhead         horiz   151936   4096    1342.3T  93.5%    1436.1T  ok        9383.8
[322/406] q3-8b-LMhead         vert    151936   4096    1717.4T 112.2%    1530.5T  ok        9367.2  best=vert
------------------------------------------------------------------------------------------------

vertical wins: 24 / 42   avg% of cuBLAS: horiz 80.9%, vert 85.0%, best-of-2 85.2%   (sweep wall time: 125.8 s)

=== Qwen 42-shape sweep (TN, 128x128 tile): horizontal vs vertical vs cuBLAS ===
Shape                wave         N      K     TFLOPS   %cuB     cuBLAS  Status  wall(ms)  best
------------------------------------------------------------------------------------------------

[323/406] q25-0.5b-Q           horiz      896    896     528.9T  63.0%     839.7T  ok          65.1
[324/406] q25-0.5b-Q           vert       896    896     555.9T  65.2%     852.8T  ok          66.8  best=vert
[325/406] q25-0.5b-K           horiz      128    896     112.8T  44.7%     252.5T  ok          67.1
[326/406] q25-0.5b-K           vert       128    896     112.8T  44.5%     253.5T  ok          57.3  best=vert
[327/406] q25-0.5b-Gate        horiz     4864    896    1158.3T  80.2%    1443.9T  ok         167.6
[328/406] q25-0.5b-Gate        vert      4864    896    1157.7T  80.3%    1440.9T  ok         181.5  best=horiz
[329/406] q25-0.5b-Down        horiz      896   4864    1111.4T  93.2%    1193.1T  ok         284.7
[330/406] q25-0.5b-Down        vert       896   4864    1155.6T  96.1%    1202.0T  ok         285.9  best=vert
[331/406] q25-0.5b-LMhead      horiz   151936    896    1114.3T  65.2%    1709.9T  ok        3881.4
[332/406] q25-0.5b-LMhead      vert    151936    896    1423.0T  83.7%    1699.7T  ok        3850.4  best=vert
[333/406] q25-1.5b-Q           horiz     1536   1536    1012.1T  74.5%    1358.0T  ok         119.7
[334/406] q25-1.5b-Q           vert      1536   1536     975.7T  72.1%    1352.5T  ok         129.7  best=horiz
[335/406] q25-1.5b-K           horiz      256   1536     309.8T  49.8%     622.7T  ok          82.6
[336/406] q25-1.5b-K           vert       256   1536     309.8T  49.5%     625.6T  ok         101.5  best=vert
[337/406] q25-1.5b-Gate        horiz     8960   1536    1457.8T  84.1%    1733.0T  ok         347.6
[338/406] q25-1.5b-Gate        vert      8960   1536    1404.8T  81.3%    1728.0T  ok         378.6  best=horiz
[339/406] q25-1.5b-Down        horiz     1536   8960    1462.4T  93.1%    1570.7T  ok         623.4
[340/406] q25-1.5b-Down        vert      1536   8960    1367.8T  87.0%    1572.8T  ok         600.2  best=horiz
[341/406] q25-1.5b-LMhead      horiz   151936   1536    1209.6T  68.7%    1760.3T  ok        4912.7
[342/406] q25-1.5b-LMhead      vert    151936   1536    1557.5T  86.4%    1803.0T  ok        4876.3  best=vert
[343/406] q25-3b-Q             horiz     2048   2048    1187.6T  77.2%    1539.0T  ok         179.6
[344/406] q25-3b-Q             vert      2048   2048    1192.4T  77.7%    1535.3T  ok         186.6  best=vert
[345/406] q25-3b-K             horiz      256   2048     412.5T  57.3%     719.7T  ok         104.6
[346/406] q25-3b-K             vert       256   2048     413.4T  57.8%     715.1T  ok         100.3  best=vert
[347/406] q25-3b-Gate          horiz    11008   2048    1560.1T  88.2%    1769.5T  ok         521.1
[348/406] q25-3b-Gate          vert     11008   2048    1538.3T  87.0%    1768.2T  ok         535.2  best=horiz
[349/406] q25-3b-Down          horiz     2048  11008    1489.8T  87.7%    1699.2T  ok         806.8
[350/406] q25-3b-Down          vert      2048  11008    1411.0T  83.2%    1696.3T  ok         812.0  best=horiz
[351/406] q25-3b-LMhead        horiz   151936   2048    1237.8T  77.4%    1598.3T  ok        5778.9
[352/406] q25-3b-LMhead        vert    151936   2048    1604.9T 103.3%    1553.0T  ok        5765.4  best=vert
[353/406] q25-7b-Q             horiz     3584   3584    1443.9T  88.9%    1623.6T  ok         370.2
[354/406] q25-7b-Q             vert      3584   3584    1423.3T  86.3%    1648.8T  ok         387.3  best=horiz
[355/406] q25-7b-K             horiz      512   3584    1001.3T  73.5%    1362.6T  ok         192.3
[356/406] q25-7b-K             vert       512   3584    1031.7T  76.3%    1352.0T  ok         185.3  best=vert
[357/406] q25-7b-Gate          horiz    18944   3584    1273.6T  68.3%    1863.4T  ok        1224.1
[358/406] q25-7b-Gate          vert     18944   3584    1651.6T  88.7%    1862.6T  ok        1198.0  best=vert
[359/406] q25-7b-Down          horiz     3584  18944    1457.7T  87.9%    1658.5T  ok        1699.1
[360/406] q25-7b-Down          vert      3584  18944    1451.8T  87.6%    1657.8T  ok        1710.1  best=horiz
[361/406] q25-7b-LMhead        horiz   152064   3584    1247.2T  88.0%    1417.2T  ok        8470.3
[362/406] q25-7b-LMhead        vert    152064   3584    1652.9T 111.0%    1489.0T  ok        8424.4  best=vert
[363/406] q3-0.6b-Q            horiz     2048   1024     924.7T  70.8%    1305.1T  ok         115.4
[364/406] q3-0.6b-Q            vert      2048   1024     923.3T  71.1%    1298.0T  ok         102.9  best=horiz
[365/406] q3-0.6b-K            horiz     1024   1024     689.5T  64.2%    1074.2T  ok          74.2
[366/406] q3-0.6b-K            vert      1024   1024     690.2T  64.6%    1068.0T  ok          84.2  best=vert
[367/406] q3-0.6b-O            horiz     1024   2048     932.8T  68.5%    1362.1T  ok         144.0
[368/406] q3-0.6b-O            vert      1024   2048    1039.8T  75.9%    1369.9T  ok         144.1  best=vert
[369/406] q3-0.6b-Gate         horiz     3072   1024    1045.7T  70.3%    1486.6T  ok         128.0
[370/406] q3-0.6b-Gate         vert      3072   1024    1048.0T  70.6%    1484.0T  ok         146.4  best=vert
[371/406] q3-0.6b-Down         horiz     1024   3072    1134.2T  75.1%    1510.8T  ok         197.8
[372/406] q3-0.6b-Down         vert      1024   3072    1137.4T  75.4%    1508.8T  ok         199.7  best=vert
[373/406] q3-0.6b-LMhead       horiz   151936   1024    1168.3T  67.1%    1740.9T  ok        4092.3
[374/406] q3-0.6b-LMhead       vert    151936   1024    1468.3T  84.5%    1737.6T  ok        4051.6  best=vert
[375/406] q3-1.7b-Q            horiz     2048   2048    1191.1T  77.2%    1542.6T  ok         187.0
[376/406] q3-1.7b-Q            vert      2048   2048    1190.8T  77.4%    1538.3T  ok         181.6  best=horiz
[377/406] q3-1.7b-K            horiz     1024   2048     930.5T  68.2%    1364.2T  ok         144.6
[378/406] q3-1.7b-K            vert      1024   2048    1032.6T  75.4%    1369.7T  ok         144.3  best=vert
[379/406] q3-1.7b-Gate         horiz     6144   2048    1480.9T  86.9%    1703.6T  ok         327.7
[380/406] q3-1.7b-Gate         vert      6144   2048    1438.3T  84.5%    1702.4T  ok         324.2  best=horiz
[381/406] q3-1.7b-Down         horiz     2048   6144    1466.5T  85.8%    1708.8T  ok         462.9
[382/406] q3-1.7b-Down         vert      2048   6144    1443.4T  84.6%    1706.7T  ok         475.0  best=horiz
[383/406] q3-1.7b-LMhead       horiz   151936   2048    1237.0T  74.9%    1651.9T  ok        5787.2
[384/406] q3-1.7b-LMhead       vert    151936   2048    1604.4T  91.4%    1755.5T  ok        5761.6  best=vert
[385/406] q3-4b-Q              horiz     4096   2560    1528.9T  89.3%    1712.6T  ok         298.7
[386/406] q3-4b-Q              vert      4096   2560    1496.8T  87.2%    1716.2T  ok         289.3  best=horiz
[387/406] q3-4b-K              horiz     1024   2560    1041.7T  72.0%    1446.2T  ok         175.8
[388/406] q3-4b-K              vert      1024   2560    1042.4T  72.4%    1440.3T  ok         160.0  best=vert
[389/406] q3-4b-O              horiz     2560   4096    1437.1T  82.4%    1744.3T  ok         367.0
[390/406] q3-4b-O              vert      2560   4096    1410.0T  80.8%    1744.8T  ok         377.9  best=horiz
[391/406] q3-4b-Gate           horiz     9728   2560    1589.8T  89.7%    1771.4T  ok         548.4
[392/406] q3-4b-Gate           vert      9728   2560    1567.1T  88.4%    1773.1T  ok         561.7  best=horiz
[393/406] q3-4b-Down           horiz     2560   9728    1486.6T  88.7%    1676.7T  ok         763.0
[394/406] q3-4b-Down           vert      2560   9728    1415.7T  84.5%    1675.8T  ok         774.8  best=horiz
[395/406] q3-4b-LMhead         horiz   151936   2560    1258.6T  87.5%    1438.8T  ok        6666.1
[396/406] q3-4b-LMhead         vert    151936   2560    1626.6T 100.5%    1617.9T  ok        6693.4  best=vert
[397/406] q3-8b-Q              horiz     4096   4096    1631.5T  93.9%    1736.9T  ok         447.8
[398/406] q3-8b-Q              vert      4096   4096    1581.1T  91.0%    1736.9T  ok         439.1  best=horiz
[399/406] q3-8b-K              horiz     1024   4096    1197.0T  75.9%    1577.3T  ok         276.8
[400/406] q3-8b-K              vert      1024   4096    1200.1T  76.4%    1570.3T  ok         262.9  best=vert
[401/406] q3-8b-Gate           horiz    12288   4096    1417.6T  77.6%    1826.5T  ok         934.3
[402/406] q3-8b-Gate           vert     12288   4096    1658.3T  90.9%    1824.4T  ok         932.2  best=vert
[403/406] q3-8b-Down           horiz     4096  12288    1541.0T  86.2%    1787.3T  ok        1210.2
[404/406] q3-8b-Down           vert      4096  12288    1558.1T  87.1%    1788.4T  ok        1196.3  best=vert
[405/406] q3-8b-LMhead         horiz   151936   4096    1235.8T  86.6%    1427.3T  ok        9355.2
[406/406] q3-8b-LMhead         vert    151936   4096    1630.1T 108.0%    1509.9T  ok        9345.9  best=vert
------------------------------------------------------------------------------------------------

vertical wins: 25 / 42   avg% of cuBLAS: horiz 77.4%, vert 81.6%, best-of-2 82.4%   (sweep wall time: 125.1 s)

=========================================================================
ALL TESTS PASSED SUCCESSFULLY
=========================================================================

## SUMMARY
