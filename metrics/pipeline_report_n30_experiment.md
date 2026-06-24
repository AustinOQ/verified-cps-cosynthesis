# Pipeline Report

Generated: 2026-06-24T16:24:54-04:00
Models: thermostat, cruise, mixing
Seeds: 42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71
CPU execution mode: aggressive

Confidence intervals are 95% CI half-widths computed across the seed rows generated for this pipeline call. For n=1, the CI half-width is reported as 0.

## SMV Verification

| system | IC3 proven | IC3 failed | IC3 runtime ms | IC3 peak MB | BMC proven | BMC failed | BMC runtime ms | BMC peak MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| thermostat | 3 | 0 | 103 | 40.0 | 3 | 0 | 10 | 36.0 |
| cruise | 5 | 0 | 102 | 41.9 | 5 | 0 | 13 | 36.1 |
| mixing | 1 | 0 | 4204 | 90.2 | 1 | 0 | 160 | 49.8 |
| TOTAL | 9 | 0 | - | - | 9 | 0 | - | - |

## Trained Model Results

| system | seeds | selected safe rate | final test success | final test safety violation rate | final test override rate | avg steps to completion | train time s | train peak RSS MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| thermostat | 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71 | 1.0000 +/- 0.0000 (n=30) | 1.0000 +/- 0.0000 (n=30) | 0.0000 +/- 0.0000 (n=30) | 0.00176 +/- 0.00002 (n=30) | 515.73 +/- 9.50 (n=30) | 1282.3 +/- 22.2 (n=30) | 240.0 +/- 2.7 (n=30) |
| cruise | 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71 | 1.0000 +/- 0.0000 (n=30) | 1.0000 +/- 0.0000 (n=30) | 0.0000 +/- 0.0000 (n=30) | 0.02337 +/- 0.00021 (n=30) | 42.82 +/- 0.39 (n=30) | 128.6 +/- 2.1 (n=30) | 92.2 +/- 0.2 (n=30) |
| mixing | 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71 | 1.0000 +/- 0.0000 (n=30) | 1.0000 +/- 0.0000 (n=30) | 0.0000 +/- 0.0000 (n=30) | 0.13464 +/- 0.01942 (n=30) | 33.24 +/- 0.37 (n=30) | 147.9 +/- 16.5 (n=30) | 96.5 +/- 0.2 (n=30) |

## Runtime Verification

| system | seeds | runtime success | runtime safety violation rate | runtime override rate | avg steps to completion | inference peak RSS MB | policy us mean | shield us mean | total us mean | test wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| thermostat | 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71 | 1.0000 +/- 0.0000 (n=30) | 0.0000 +/- 0.0000 (n=30) | 0.00176 +/- 0.00002 (n=30) | 515.73 +/- 9.50 (n=30) | 98.4 +/- 0.5 (n=30) | 51.98 +/- 0.60 (n=30) | 11.77 +/- 0.14 (n=30) | 72.03 +/- 0.84 (n=30) | 21.26 +/- 0.43 (n=30) |
| cruise | 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71 | 1.0000 +/- 0.0000 (n=30) | 0.0000 +/- 0.0000 (n=30) | 0.02337 +/- 0.00021 (n=30) | 42.82 +/- 0.39 (n=30) | 76.9 +/- 0.2 (n=30) | 55.90 +/- 1.02 (n=30) | 16.48 +/- 0.19 (n=30) | 82.04 +/- 1.52 (n=30) | 2.55 +/- 0.05 (n=30) |
| mixing | 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71 | 1.0000 +/- 0.0000 (n=30) | 0.0000 +/- 0.0000 (n=30) | 0.13464 +/- 0.01942 (n=30) | 33.24 +/- 0.37 (n=30) | 79.4 +/- 0.2 (n=30) | 58.64 +/- 2.79 (n=30) | 18.79 +/- 1.17 (n=30) | 88.15 +/- 4.30 (n=30) | 3.88 +/- 0.19 (n=30) |

## Runtime Latency Details

| system | seeds | policy us p95 | policy us p99 | shield us p95 | shield us p99 | total us p95 | total us p99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| thermostat | 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71 | 56.04 +/- 0.71 (n=30) | 64.74 +/- 1.93 (n=30) | 12.35 +/- 0.14 (n=30) | 14.50 +/- 0.20 (n=30) | 77.05 +/- 0.97 (n=30) | 91.09 +/- 2.49 (n=30) |
| cruise | 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71 | 70.08 +/- 3.64 (n=30) | 121.19 +/- 15.74 (n=30) | 18.29 +/- 0.49 (n=30) | 62.24 +/- 13.32 (n=30) | 105.04 +/- 4.46 (n=30) | 197.55 +/- 14.08 (n=30) |
| mixing | 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71 | 79.71 +/- 1.40 (n=30) | 88.25 +/- 1.87 (n=30) | 56.31 +/- 5.73 (n=30) | 68.43 +/- 1.68 (n=30) | 134.71 +/- 5.72 (n=30) | 161.11 +/- 5.74 (n=30) |

## Selected Checkpoints

| system | seed | source | selected episode | selected success | selected override | selected safety violation | final safe |
|---|---:|---|---:|---:|---:|---:|---|
| thermostat | 42 | safe_eval | 500 | 1.0000 | 0.00160 | 0.00000 | True |
| thermostat | 43 | safe_eval | 1100 | 1.0000 | 0.00161 | 0.00000 | True |
| thermostat | 44 | safe_eval | 300 | 1.0000 | 0.00150 | 0.00000 | True |
| thermostat | 45 | safe_eval | 400 | 1.0000 | 0.00159 | 0.00000 | True |
| thermostat | 46 | safe_eval | 1800 | 1.0000 | 0.00164 | 0.00000 | True |
| thermostat | 47 | safe_eval | 2000 | 1.0000 | 0.00160 | 0.00000 | True |
| thermostat | 48 | safe_eval | 400 | 1.0000 | 0.00160 | 0.00000 | True |
| thermostat | 49 | safe_eval | 1700 | 1.0000 | 0.00147 | 0.00000 | True |
| thermostat | 50 | safe_eval | 1400 | 1.0000 | 0.00163 | 0.00000 | True |
| thermostat | 51 | safe_eval | 1300 | 1.0000 | 0.00155 | 0.00000 | True |
| thermostat | 52 | safe_eval | 200 | 1.0000 | 0.00161 | 0.00000 | True |
| thermostat | 53 | safe_eval | 700 | 1.0000 | 0.00149 | 0.00000 | True |
| thermostat | 54 | safe_eval | 100 | 1.0000 | 0.00165 | 0.00000 | True |
| thermostat | 55 | safe_eval | 500 | 1.0000 | 0.00157 | 0.00000 | True |
| thermostat | 56 | safe_eval | 1700 | 1.0000 | 0.00161 | 0.00000 | True |
| thermostat | 57 | safe_eval | 2000 | 1.0000 | 0.00161 | 0.00000 | True |
| thermostat | 58 | safe_eval | 1700 | 1.0000 | 0.00146 | 0.00000 | True |
| thermostat | 59 | safe_eval | 300 | 1.0000 | 0.00158 | 0.00000 | True |
| thermostat | 60 | safe_eval | 1000 | 1.0000 | 0.00185 | 0.00000 | True |
| thermostat | 61 | safe_eval | 1200 | 1.0000 | 0.00157 | 0.00000 | True |
| thermostat | 62 | safe_eval | 1600 | 1.0000 | 0.00164 | 0.00000 | True |
| thermostat | 63 | safe_eval | 300 | 1.0000 | 0.00158 | 0.00000 | True |
| thermostat | 64 | safe_eval | 1200 | 1.0000 | 0.00166 | 0.00000 | True |
| thermostat | 65 | safe_eval | 1300 | 1.0000 | 0.00163 | 0.00000 | True |
| thermostat | 66 | safe_eval | 700 | 1.0000 | 0.00154 | 0.00000 | True |
| thermostat | 67 | safe_eval | 2000 | 1.0000 | 0.00157 | 0.00000 | True |
| thermostat | 68 | safe_eval | 100 | 1.0000 | 0.00157 | 0.00000 | True |
| thermostat | 69 | safe_eval | 700 | 1.0000 | 0.00159 | 0.00000 | True |
| thermostat | 70 | safe_eval | 700 | 1.0000 | 0.00162 | 0.00000 | True |
| thermostat | 71 | safe_eval | 1000 | 1.0000 | 0.00164 | 0.00000 | True |
| cruise | 42 | safe_eval | 1200 | 1.0000 | 0.02195 | 0.00000 | True |
| cruise | 43 | safe_eval | 1200 | 1.0000 | 0.02203 | 0.00000 | True |
| cruise | 44 | safe_eval | 1700 | 1.0000 | 0.02073 | 0.00000 | True |
| cruise | 45 | safe_eval | 1900 | 1.0000 | 0.02191 | 0.00000 | True |
| cruise | 46 | safe_eval | 300 | 1.0000 | 0.02214 | 0.00000 | True |
| cruise | 47 | safe_eval | 1100 | 1.0000 | 0.02206 | 0.00000 | True |
| cruise | 48 | safe_eval | 800 | 1.0000 | 0.02173 | 0.00000 | True |
| cruise | 49 | safe_eval | 1000 | 1.0000 | 0.02179 | 0.00000 | True |
| cruise | 50 | safe_eval | 1700 | 1.0000 | 0.02241 | 0.00000 | True |
| cruise | 51 | safe_eval | 900 | 1.0000 | 0.02138 | 0.00000 | True |
| cruise | 52 | safe_eval | 800 | 1.0000 | 0.02180 | 0.00000 | True |
| cruise | 53 | safe_eval | 1300 | 1.0000 | 0.02224 | 0.00000 | True |
| cruise | 54 | safe_eval | 1500 | 1.0000 | 0.02207 | 0.00000 | True |
| cruise | 55 | safe_eval | 700 | 1.0000 | 0.02251 | 0.00000 | True |
| cruise | 56 | safe_eval | 300 | 1.0000 | 0.02193 | 0.00000 | True |
| cruise | 57 | safe_eval | 1600 | 1.0000 | 0.02154 | 0.00000 | True |
| cruise | 58 | safe_eval | 1000 | 1.0000 | 0.02217 | 0.00000 | True |
| cruise | 59 | safe_eval | 200 | 1.0000 | 0.02183 | 0.00000 | True |
| cruise | 60 | safe_eval | 700 | 1.0000 | 0.02081 | 0.00000 | True |
| cruise | 61 | safe_eval | 1000 | 1.0000 | 0.02240 | 0.00000 | True |
| cruise | 62 | safe_eval | 1300 | 1.0000 | 0.02151 | 0.00000 | True |
| cruise | 63 | safe_eval | 1200 | 1.0000 | 0.02153 | 0.00000 | True |
| cruise | 64 | safe_eval | 1100 | 1.0000 | 0.02152 | 0.00000 | True |
| cruise | 65 | safe_eval | 600 | 1.0000 | 0.02204 | 0.00000 | True |
| cruise | 66 | safe_eval | 400 | 1.0000 | 0.02239 | 0.00000 | True |
| cruise | 67 | safe_eval | 1400 | 1.0000 | 0.02136 | 0.00000 | True |
| cruise | 68 | safe_eval | 1200 | 1.0000 | 0.02219 | 0.00000 | True |
| cruise | 69 | safe_eval | 600 | 1.0000 | 0.02169 | 0.00000 | True |
| cruise | 70 | safe_eval | 100 | 1.0000 | 0.02182 | 0.00000 | True |
| cruise | 71 | safe_eval | 900 | 1.0000 | 0.02079 | 0.00000 | True |
| mixing | 42 | safe_eval | 100 | 1.0000 | 0.16813 | 0.00000 | True |
| mixing | 43 | safe_eval | 100 | 1.0000 | 0.11804 | 0.00000 | True |
| mixing | 44 | safe_eval | 100 | 1.0000 | 0.18432 | 0.00000 | True |
| mixing | 45 | safe_eval | 100 | 1.0000 | 0.12484 | 0.00000 | True |
| mixing | 46 | safe_eval | 100 | 1.0000 | 0.14860 | 0.00000 | True |
| mixing | 47 | safe_eval | 100 | 1.0000 | 0.16284 | 0.00000 | True |
| mixing | 48 | safe_eval | 100 | 1.0000 | 0.17027 | 0.00000 | True |
| mixing | 49 | safe_eval | 100 | 1.0000 | 0.10353 | 0.00000 | True |
| mixing | 50 | safe_eval | 100 | 1.0000 | 0.09155 | 0.00000 | True |
| mixing | 51 | safe_eval | 100 | 1.0000 | 0.18602 | 0.00000 | True |
| mixing | 52 | safe_eval | 100 | 1.0000 | 0.11027 | 0.00000 | True |
| mixing | 53 | safe_eval | 100 | 1.0000 | 0.20133 | 0.00000 | True |
| mixing | 54 | safe_eval | 100 | 1.0000 | 0.15632 | 0.00000 | True |
| mixing | 55 | safe_eval | 100 | 1.0000 | 0.21164 | 0.00000 | True |
| mixing | 56 | safe_eval | 100 | 1.0000 | 0.13883 | 0.00000 | True |
| mixing | 57 | safe_eval | 100 | 1.0000 | 0.14517 | 0.00000 | True |
| mixing | 58 | safe_eval | 100 | 1.0000 | 0.14202 | 0.00000 | True |
| mixing | 59 | safe_eval | 100 | 1.0000 | 0.01852 | 0.00000 | True |
| mixing | 60 | safe_eval | 100 | 1.0000 | 0.11456 | 0.00000 | True |
| mixing | 61 | safe_eval | 200 | 1.0000 | 0.02310 | 0.00000 | True |
| mixing | 62 | safe_eval | 100 | 1.0000 | 0.10725 | 0.00000 | True |
| mixing | 63 | safe_eval | 100 | 1.0000 | 0.16976 | 0.00000 | True |
| mixing | 64 | safe_eval | 100 | 1.0000 | 0.05894 | 0.00000 | True |
| mixing | 65 | safe_eval | 100 | 1.0000 | 0.01952 | 0.00000 | True |
| mixing | 66 | safe_eval | 100 | 1.0000 | 0.21557 | 0.00000 | True |
| mixing | 67 | safe_eval | 100 | 1.0000 | 0.09924 | 0.00000 | True |
| mixing | 68 | safe_eval | 100 | 1.0000 | 0.08845 | 0.00000 | True |
| mixing | 69 | safe_eval | 100 | 1.0000 | 0.09967 | 0.00000 | True |
| mixing | 70 | safe_eval | 100 | 1.0000 | 0.20453 | 0.00000 | True |
| mixing | 71 | safe_eval | 100 | 1.0000 | 0.21264 | 0.00000 | True |
