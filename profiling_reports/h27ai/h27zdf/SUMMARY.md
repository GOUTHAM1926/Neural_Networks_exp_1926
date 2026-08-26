# H27zdf paired + null control  GPU 1  loud-tree-grows-fin-03  20260826T150830Z  reps: 21  window: 400ms
commit : 595adc5 H27ai: cut the k-loop integer excess - R2UR 8->1, ULEA 7->1, 24.4% fewer instructions per wgmma
clocks : 1410 MHz, 1980 MHz

  21 complete reps of 6 arms

  arm            mean      min      max   raw (unpaired, as before)   
  PIN_A        105.08   104.30   105.80
  NULL_B       105.01   104.20   105.70
  CS           104.98   104.50   105.50
  DW           105.10   104.30   105.70
  DW_UAD        97.75    97.20    98.20
  DW_UAD_CS    104.86   104.40   105.40

  PAIRED against arm 0 (PIN_A) -- per-rep difference, so drift common to a rep cancels
  arm            mean_d       sd_d      min_d      max_d   verdict
  NULL_B          -0.07       0.65      -1.30       1.10   <- NULL CONTROL: identical binary. This is the METHOD ERROR.
  CS              -0.10       0.55      -1.20       0.90   NOT separable from the null
  DW               0.02       0.62      -0.90       1.40   NOT separable from the null
  DW_UAD          -7.33       0.45      -8.50      -6.70   believable
  DW_UAD_CS       -0.22       0.46      -0.90       1.00   NOT separable from the null

  RESOLUTION FLOOR = the null control's |mean| plus its SD = 0.72 points.
  Nothing smaller than that can be claimed on this box with this method, no matter how
  many reps are averaged, because it is bias and not noise.
  SKEW was adopted on +1.62 measured as two SEQUENTIAL BLOCKS. Compare that to the floor.

RESULTS IN: test_reports/h27zdf   summary: test_reports/h27zdf/SUMMARY.md


====================================================================================================
# H27zdf paired + null control  GPU 1  loud-tree-grows-fin-03  20260826T151747Z  reps: 21  window: 400ms
commit : 595adc5 H27ai: cut the k-loop integer excess - R2UR 8->1, ULEA 7->1, 24.4% fewer instructions per wgmma
clocks : 345 MHz, 1980 MHz

  21 complete reps of 6 arms

  arm            mean      min      max   raw (unpaired, as before)   
  PIN_A         97.33    92.90   102.10
  NULL_B        97.80    92.20   100.90
  CS            96.50    91.70    99.60
  DW            97.02    92.30   101.90
  DW_UAD        92.89    87.30    96.30
  DW_UAD_CS     96.63    93.10   100.60

  PAIRED against arm 0 (PIN_A) -- per-rep difference, so drift common to a rep cancels
  arm            mean_d       sd_d      min_d      max_d   verdict
  NULL_B           0.48       2.68      -4.00       6.70   <- NULL CONTROL: identical binary. This is the METHOD ERROR.
  CS              -0.83       2.66      -6.60       2.60   NOT separable from the null
  DW              -0.31       3.31      -5.00       9.00   NOT separable from the null
  DW_UAD          -4.44       2.93     -13.10       0.70   believable
  DW_UAD_CS       -0.70       2.80      -5.40       5.00   NOT separable from the null

  RESOLUTION FLOOR = the null control's |mean| plus its SD = 3.16 points.
  Nothing smaller than that can be claimed on this box with this method, no matter how
  many reps are averaged, because it is bias and not noise.
  SKEW was adopted on +1.62 measured as two SEQUENTIAL BLOCKS. Compare that to the floor.

RESULTS IN: test_reports/h27zdf   summary: test_reports/h27zdf/SUMMARY.md
