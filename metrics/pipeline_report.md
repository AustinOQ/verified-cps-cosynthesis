# Pipeline Report

Generated: 2026-06-23T18:17:19-04:00
Models: thermostat, cruise, mixing
Seeds: 42

Confidence intervals are 95% CI half-widths computed across the seed rows generated for this pipeline call. For n=1, the CI half-width is reported as 0.

## SMV Verification

| system | IC3 proven | IC3 failed | IC3 runtime ms | IC3 peak MB | BMC proven | BMC failed | BMC runtime ms | BMC peak MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| thermostat | 3 | 0 | 103 | 40.1 | 3 | 0 | 10 | 36.3 |
| cruise | 5 | 0 | 103 | 42.5 | 5 | 0 | 11 | 36.4 |
| mixing | 1 | 0 | 4405 | 88.1 | 1 | 0 | 156 | 49.2 |
| TOTAL | 9 | 0 | - | - | 9 | 0 | - | - |

## Trained Model Results

| system | seeds | selected safe rate | final test success | final test safety violation rate | final test override rate | avg steps to completion | train time s | train peak RSS MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| thermostat | 42 | 1.0000 +/- 0.0000 (n=1) | 1.0000 +/- 0.0000 (n=1) | 0.0000 +/- 0.0000 (n=1) | 0.00175 +/- 0.00000 (n=1) | 530.10 +/- 0.00 (n=1) | 694.5 +/- 0.0 (n=1) | 236.2 +/- 0.0 (n=1) |
| cruise | 42 | 1.0000 +/- 0.0000 (n=1) | 1.0000 +/- 0.0000 (n=1) | 0.0000 +/- 0.0000 (n=1) | 0.02256 +/- 0.00000 (n=1) | 44.33 +/- 0.00 (n=1) | 65.4 +/- 0.0 (n=1) | 94.1 +/- 0.0 (n=1) |
| mixing | 42 | 1.0000 +/- 0.0000 (n=1) | 1.0000 +/- 0.0000 (n=1) | 0.0000 +/- 0.0000 (n=1) | 0.18576 +/- 0.00000 (n=1) | 34.70 +/- 0.00 (n=1) | 66.6 +/- 0.0 (n=1) | 97.1 +/- 0.0 (n=1) |

## Runtime Verification

| system | seeds | runtime success | runtime safety violation rate | runtime override rate | avg steps to completion | inference peak RSS MB | policy us mean | shield us mean | total us mean | test wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| thermostat | 42 | 1.0000 +/- 0.0000 (n=1) | 0.0000 +/- 0.0000 (n=1) | 0.00175 +/- 0.00000 (n=1) | 530.10 +/- 0.00 (n=1) | 99.2 +/- 0.0 (n=1) | 26.54 +/- 0.00 (n=1) | 5.87 +/- 0.00 (n=1) | 36.59 +/- 0.00 (n=1) | 11.19 +/- 0.00 (n=1) |
| cruise | 42 | 1.0000 +/- 0.0000 (n=1) | 0.0000 +/- 0.0000 (n=1) | 0.02256 +/- 0.00000 (n=1) | 44.33 +/- 0.00 (n=1) | 77.6 +/- 0.0 (n=1) | 26.30 +/- 0.00 (n=1) | 7.88 +/- 0.00 (n=1) | 38.35 +/- 0.00 (n=1) | 1.23 +/- 0.00 (n=1) |
| mixing | 42 | 1.0000 +/- 0.0000 (n=1) | 0.0000 +/- 0.0000 (n=1) | 0.18576 +/- 0.00000 (n=1) | 34.70 +/- 0.00 (n=1) | 81.3 +/- 0.0 (n=1) | 27.14 +/- 0.00 (n=1) | 11.25 +/- 0.00 (n=1) | 42.93 +/- 0.00 (n=1) | 2.22 +/- 0.00 (n=1) |

## Runtime Latency Details

| system | seeds | policy us p95 | policy us p99 | shield us p95 | shield us p99 | total us p95 | total us p99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| thermostat | 42 | 28.60 +/- 0.00 (n=1) | 35.87 +/- 0.00 (n=1) | 6.48 +/- 0.00 (n=1) | 8.17 +/- 0.00 (n=1) | 39.80 +/- 0.00 (n=1) | 49.60 +/- 0.00 (n=1) |
| cruise | 42 | 29.15 +/- 0.00 (n=1) | 31.81 +/- 0.00 (n=1) | 8.93 +/- 0.00 (n=1) | 16.16 +/- 0.00 (n=1) | 44.36 +/- 0.00 (n=1) | 46.78 +/- 0.00 (n=1) |
| mixing | 42 | 30.99 +/- 0.00 (n=1) | 39.01 +/- 0.00 (n=1) | 33.07 +/- 0.00 (n=1) | 35.22 +/- 0.00 (n=1) | 64.82 +/- 0.00 (n=1) | 71.75 +/- 0.00 (n=1) |

## Selected Checkpoints

| system | seed | source | selected episode | selected success | selected override | selected safety violation | final safe |
|---|---:|---|---:|---:|---:|---:|---|
| thermostat | 42 | safe_eval | 500 | 1.0000 | 0.00160 | 0.00000 | True |
| cruise | 42 | safe_eval | 1200 | 1.0000 | 0.02195 | 0.00000 | True |
| mixing | 42 | safe_eval | 100 | 1.0000 | 0.16813 | 0.00000 | True |
