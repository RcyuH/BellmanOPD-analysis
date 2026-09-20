# CMT analysis pipeline

This folder provides two paths:

1. **Existing runs, no training:** read `token_score_stats/` from a CMT run and plot the distributions and paired sampled relationships among `g`, `X`, `D`, and `w`.
2. **Future runs, opt-in instrumentation:** collect bounded per-token records and immediate before/after optimizer-update probes. The base training config keeps `analysis.enabled: false`, so ordinary training does not execute any analysis hook.

The analysis tools do not launch training. The commands below for analysis-enabled training document an optional future run; they are not needed to inspect existing logs.

## Exact quantities and observation levels

The CMT selector in `b200_experiment/selectors/cmt_selector.py` defines:

| Stored field | Exact source and interpretation |
|---|---|
| `g` | `diagnostics["gain"]`: local support-matched PGT gain, `Var_{p_U}(log q_U - log p_U)`. |
| `x` | `diagnostics["successor_excess"]`: `R_(t+1) - g_t M_(t+1)`, the baseline-subtracted successor opportunity on the **observed** rollout. It is not `R`, `M`, `V`, or `H`. |
| `d` | `diagnostics["sequential_gain"]`: `lambda * gamma * marginal_flux * successor_excess`, the one-rollout estimator of the sequential derivative. It may be negative. With canonical `lambda=1`, `g_plus_d` equals `learning_value` within floating-point precision. |
| `g_plus_x`, `g_plus_d` | Direct sums of the preceding stored fields. |
| `training_weight` | Actual groupwise CMT Gibbs weight applied to the token in `_opd_train_step`, after global PPO-group allocation. |
| `opd_ppo_loss_before` | Actual per-position clipped OPD PPO loss calculated by the training forward immediately before backward. This is distinct from sampled-action negative log probability. |
| `student_token_nll` | Negative log probability of the sampled response token from the rollout-time student score. |
| `student_target_logprob`, `teacher_target_logprob` | Full-vocabulary sampled-action log probabilities already returned by the student/teacher scorers. Their difference is also logged as `teacher_student_sampled_logprob_gap`. |
| `kl_support_reverse` | `support_reverse_kl = KL(p_U || q_U)` on the union of student/teacher Top-K actions. This is a **conditional support KL**, not full-vocabulary KL and not TA's `D = KL(q_U || p_U)`. |
| `student_entropy`, `teacher_entropy` | Full-vocabulary entropy reductions returned by the existing scorers. |
| `sequence_length`, `response_position`, `normalized_position` | Valid response-token count, zero-based response position, and `(position + 1) / sequence_length`. |

`tokens/step-*.jsonl` is a **sample of valid response tokens at rollout scoring time**. `score_model_step` identifies the weights used to compute `g/X/D`; `scoring_step` is the rollout's final optimizer-step label. `optimizer_step` identifies the update using that token, if reached; it can be `null` when training stops partway through a rollout. `sample_id`, `dataset_index`, rank, token ID, actual loss, weight, and gradient norm permit joins with `metrics.jsonl`. The file contains only compact scalar records. `save_full_logits: true` is explicitly rejected.

`progress/step-*.jsonl` is a **sample measured immediately before and after one complete optimizer update**. Both forwards evaluate the same sampled response prefix under the same fixed union support and frozen teacher probabilities. It stores `student_token_nll_before/after`, `kl_support_reverse_before/after`, `delta_nll`, `delta_kl`, the actual training weight/loss, and the original rollout's `g/X/D`. It also stores `delta_future_kl` and `delta_future_nll`: mean improvement over up to the next `downstream_window_tokens` positions on that **same sampled continuation**. They are null at a terminal token. A positive delta means the measured loss/KL decreased. This is the effect of the *whole PPO minibatch update*, not an intervention on one isolated token or a measurement of newly reachable branches. For a later PPO minibatch in a rollout, `g/X/D` were computed at `score_model_step`, before earlier updates in that rollout; the probe's `*_before` values are refreshed immediately before its own update.

There is no reasoning/content mask in the current rollout structure; only a contiguous valid-response mask is available. The logger preserves `valid_token_mask: true` for sampled positions and never emits padding/filler as an observation. No full-vocabulary token KL is computed by default; it would require an additional large reduction.

## Existing runs: no training

From the repository root:

```bash
python -m analysis.run_analysis \
  --input /path/to/existing/cmt_run \
  --output /path/to/existing/cmt_run/analysis_figures
```

You can also run `python run_analysis.py --input ... --output ...` from the
`analysis/` directory.

If the run contains only the original `token_score_stats/step-*.json`, this creates `score_distributions.png/.pdf` and `score_correlations.csv`. It pairs `gain`, `successor_excess`, `sequential_gain`, and optional `w` samples only when their counts equal `metrics.jsonl`'s `num_valid_tokens` for that step. This verifies that finite-value filtering has not shifted the shared sample indices. Original score-stat files do **not** retain token identity, position, or before/after measurements, so the command does not invent learning-progress plots.

## Future analysis-enabled training (optional)

To enable bounded instrumentation during a **new or continuing** CMT training run:

```bash
bash scripts/train_cmt_b200.sh \
  --overlay analysis/configs/cmt.yaml
```

The overlay can be tuned with `--set analysis.learning_progress_every_n_steps=200`, `--set analysis.num_sequences_to_track=1`, etc. The base config is disabled; applying the overlay to an already completed run does not retroactively create progress measurements. `max_probe_response_position` bounds sampled probe positions to the first 512 response tokens; the probe forward extends at most another `downstream_window_tokens` tokens (default 32). Thus the before/after sample is deliberately not uniform over the whole sequence. Per-token score snapshots can sample positions over the whole response. For distributed training, every rank performs the same scheduled probe forwards, and one bounded `all_gather_object` per logged rollout sends sampled CPU records to rank 0. Rank 0 alone writes raw files and TensorBoard; all other ranks write none. TensorBoard means and histograms describe the pooled **sample**, not all tokens in a rollout.

Raw output under `<experiment.output_dir>/analysis/`:

```text
manifest.json
tokens/step-*.jsonl
progress/step-*.jsonl
tensorboard/events.out.tfevents.*
```

TensorBoard:

```bash
tensorboard --logdir /path/to/cmt_run/analysis/tensorboard
```

It shows sampled means/std/min/max and histograms for `g`, `x`, `d`, combined scores, NLL, support KL, entropy, and realized deltas, plus Pearson/Spearman correlations between each predictor and each delta. Existing training TensorBoard logs remain separate.

## Offline plots for an instrumented run

```bash
python -m analysis.run_analysis \
  --input /path/to/cmt_run/analysis \
  --output /path/to/cmt_run/analysis_figures \
  --outcome delta_kl --bins 10
```

The command writes PNG/PDF figures for score distributions, normalized-position profiles, `g` and `D` quantiles versus realized progress, all five predictor curves (`g`, `X`, `D`, `g+X`, `g+D`), low/high `D` comparisons within `g` bins, and score/progress density. Companion CSV files contain the plotted aggregates and Pearson/Spearman correlations. `--outcome delta_nll`, `--outcome delta_future_kl`, or `--outcome delta_future_nll` repeat the analysis for those outcomes. The conditional comparison is observational; matching on `g` does not identify a causal effect of `D`.

## Scope and overhead

- Only CMT is supported, because other training methods do not define this exact `g/X/D` triplet. Enabling analysis for another method raises an error.
- Scoring and token sampling are detached. The actual OPD objective, Gibbs allocator, backward pass, optimizer, and checkpoints are unchanged. The extra model forwards run only on the configured progress interval and never backpropagate.
- Resume rewinds analysis token/progress files after the resume checkpoint step, matching the run's training history. Raw files are bounded by the configured number of sequences and positions.
- This pipeline does not establish that teaching a specific token caused downstream access to change. That claim requires a separate intervention or branching-inference experiment; a full-budget performance sweep also requires already trained comparison runs or additional training.
