# Bounded weighting decision for Experiment 2

Source run: `batch_gt_fixed_8gpu_20260921_064611` (284 completed batches).
The 113,231 logged valid tokens contain 11,446 top-10% selections.

## Observed `g_t` distribution

| Population | min | p10 | p50 | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| All valid tokens | 4.61e-11 | 0.0174 | 1.681 | 24.321 | 44.244 | 209.560 | 502.959 |
| Selected top-10% | 8.883 | 19.654 | 43.551 | 209.089 | 243.147 | 291.327 | 502.959 |

Within the selected population, 26.13% of tokens have `g_t > 100`, 11.68%
have `g_t > 200`, and the maximum is 11.55 times the median. The raw score is
therefore heavy-tailed enough that it should not be used directly as the OPD
loss weight.

## Candidate within-batch weighting transforms

Each candidate below is normalized by its selected-token mean in each batch,
matching the normalization performed by the training loss.

| Transform | min | p50 | p90 | p99 | max | observed max/min |
|---|---:|---:|---:|---:|---:|---:|
| Raw `g_t` | 0.149 | 0.624 | 2.206 | 4.461 | 9.019 | 60.62x |
| `log1p(g_t)` | 0.658 | 0.954 | 1.263 | 1.513 | 1.783 | 2.71x |
| Winsorized p10/p90 -> `[0.5, 1.5]` | 0.436 | 0.815 | 1.730 | 2.073 | 2.208 | 5.07x |
| Stable rank -> `[0.5, 1.5]` | 0.500 | 1.000 | 1.411 | 1.500 | 1.500 | 3.00x |

The recommended transform is stable rank -> `[0.5, 1.5]`. It preserves the
ordering induced by `g_t`, makes the selected-token mean exactly 1, and places
a deterministic 3x ceiling on relative weights independent of score scale or
outliers. Tokens outside the top-10% retain weight 0.

The previous run already used binary weights (1 for selected, 0 otherwise),
not raw `g_t` weights. Consequently, raw-score outliers did not explain that
run's failure against uniform OPD. This new bounded-rank mode tests a separate
hypothesis: whether mild prioritization *within* the selected set helps. It
does not address the observed tendency of `g_t` selection itself to favor chat
and reasoning-wrapper tokens.
