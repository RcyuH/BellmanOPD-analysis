# Analysis pipelines

## Experiment 1: exact prompt-token OPD interventions

`run_prompt_token_influence.py` implements a separate causal branching
experiment for every token `x_t` in every fully rendered Competition-MATH
training prompt. At optimizer step `s`, all branches start from exactly the
same `theta_s`:

```text
token branch t: theta_(s,t) = theta_s - learning_rate * grad L_OPD,t
uniform branch: theta_(s,u) = theta_s - learning_rate * grad mean_t L_OPD,t
```

After each virtual update it measures the token-weighted, full-vocabulary
`KL(teacher || student)` over **every rendered prompt token of every problem in
the Competition-MATH test split**. Positive `distance_improvement` means the
student moved closer to the teacher. Parameters are restored from an exact
device-side snapshot before the next token branch, so token branches at a step
never contaminate one another. The uniform branch is retained as the real SGD
update and training proceeds through the full shuffled Competition-MATH train
split. This is plain SGD by design and exactly matches the requested update;
there is no hidden Adam state, gradient clipping, or weight decay.

Position `t` means the teacher/student next-token distributions immediately
after consuming prompt token `x_t`. Consequently an N-token rendered prompt
has N interventions, including chat-template/special tokens and the final
assistant-prefix token. Each row stores raw ID, tokenizer piece, whitespace-
preserving decoded text, a terminal-visible rendering, UTF-8 bytes, and the
special-token flag. The full rendered prompt and complete ordered token-ID list
are stored once in the matching `prompts/step-*.json` file.

Run on eight B200s (paths inherit the normal repository config):

```bash
bash analysis/run_prompt_token_influence.sh \
  --set prompt_token_influence.output_dir=outputs/token_exp_01
```

The launcher uses `torchrun` with eight replicated workers by default. Each
GPU holds one complete frozen teacher, student, parameter snapshot, and current
gradient. Token position `t` is owned by rank `t mod 8`, so the eight GPUs
evaluate eight independent counterfactual branches concurrently. Baseline and
uniform test distances shard Competition-MATH test examples across ranks and
sum their KL numerator/token counts with NCCL. Rank zero computes the uniform
gradient and SGD update once, then broadcasts the updated parameters to the
other seven replicas; a parameter-signature equality guard verifies the
result. Only rank zero writes logs and checkpoints.

One-GPU smoke test (explicitly override the declared world size):

```bash
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 \
bash analysis/run_prompt_token_influence.sh \
  --set prompt_token_influence.expected_world_size=1 \
  --set prompt_token_influence.max_steps=1 \
  --set prompt_token_influence.output_dir=outputs/token_exp_smoke
```

Outputs:

```text
manifest.json
resolved_config.yaml
steps/step-*.jsonl
prompts/step-*.json
checkpoints/step-*/
summary.json
```

Rank the most improving tokens globally and compare them with the uniform
branch at each step:

```bash
python -m analysis.summarize_prompt_token_influence \
  --input outputs/token_exp_01 \
  --output outputs/token_exp_01/ranking
```

This exact design is intentionally expensive: one N-token training prompt
requires N+2 complete passes over the full test split (baseline, N token
branches, uniform). `max_steps` exists only for smoke tests; leaving it `null`
is the declared full-train experiment and no train/test sampling or truncation
is performed.

## CMT observational analysis

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
qualitative/step-*.jsonl
tensorboard/events.out.tfevents.*
```

TensorBoard:

```bash
tensorboard --logdir /path/to/cmt_run/analysis/tensorboard
```

It shows sampled means/std/min/max and histograms for `g`, `x`, `d`, combined scores, NLL, support KL, entropy, and realized deltas, plus Pearson/Spearman correlations between each predictor and each delta. Existing training TensorBoard logs remain separate.

With `analysis.qualitative.enabled: true`, TensorBoard also receives a small
text sample under `Analysis/examples/high_D`,
`Analysis/examples/same_g_different_D`,
`Analysis/examples/g_vs_gD_ranking_change`, and
`Analysis/examples/failure_cases`. The JSONL and HTML report remain the main
interfaces because TensorBoard text is intentionally bounded.

## Qualitative token report

The supplied CMT overlay enables qualitative logging. It decodes tokens with
the exact tokenizer loaded by training and retains both the raw token ID and
tokenizer piece. Whitespace is left untouched in JSON; the HTML renders spaces,
newlines, and tabs with visible markers.

For each sampled before/after probe, `qualitative/step-*.jsonl` stores:

| Field | Meaning |
|---|---|
| `prefix_tail_text`, `context_window`, `current_token` | Up to `prefix_tokens` tokens before the position, a ±`context_tokens` response window, and the selected token. Special-token status and raw IDs are retained. |
| `reference_text` | Dataset answer/solution when one of the repository's known answer fields exists. It is evidence from the dataset, not a generated teacher continuation. |
| `student_top_k_before`, `student_top_k_after` | Actual full-vocabulary student Top-K at each probe instant. These lists can differ after the update. |
| `teacher_top_k` | Teacher-leading tokens inside the rollout-time selector union; probabilities are reconstructed from its conditional probabilities and retained support mass. |
| `candidate_tokens` | Informative paired candidates from the fixed rollout-time union of student/teacher Top-K actions plus the sampled target. Each has full-vocabulary student probability/log-probability before and after, reconstructed teacher probability, conditional-on-union probabilities, within-set ranks, `delta_p`, and `delta_rank`. A token that enters only the after-update full-vocabulary Top-K appears in `student_top_k_after`, but has no invented before/after pair outside this fixed set. |
| `teacher_preferred_token` | Highest teacher-probability token inside the stored candidate set. This need not be the global teacher argmax if the selector support was configured differently. |
| `target_logprob_gain` | `log p_after(y_t) - log p_before(y_t)`; positive means the sampled token became more likely. It equals `delta_nll`. |
| `local_reverse_kl_gain` / `delta_kl` | `KL(p_U || q_U)_before - KL(p_U || q_U)_after`; positive means the student moved closer under this fixed-support reverse KL. This is not `KL(q || p)` over the full vocabulary. |
| `teacher_preferred_probability_gain` | Change in student probability assigned to the teacher's highest-probability stored candidate. |
| `delta_future_kl_h1/h4/h8/h16` | Mean fixed-support reverse-KL decrease over the next observed 1/4/8/16 valid tokens. A terminal position has `null`. |
| `rank_g`, `rank_g_plus_x`, `rank_g_plus_d` | One-based rank of this position among valid positions in the same response under each counterfactual score. `top_positions_*` stores the configured leading positions. These are diagnostic rankings; they do not rerun allocation or optimization. |
| `observed_update_scope` | Reminder that before/after changes follow the whole PPO minibatch update, not an isolated update at the displayed token. |

Generate the self-contained report after training or after a resumed segment:

```bash
python -m analysis.visualize_token_examples \
  --input /path/to/cmt_run \
  --output /path/to/cmt_run/analysis/token_report \
  --examples-per-category 5
```

Open `token_report/token_examples.html`. The page includes a switchable
`g`/`X`/`D`/`g+X`/`g+D` token heatmap, before/after candidate tables,
counterfactual sequence rankings, dataset references, local outcomes, and
future outcomes. It mines high `g`, high `D`, high-`g`/low-`D`,
moderate-`g`/high-`D`, ranking changes, positive and negative downstream
changes, failures, and pairs with similar `g` (or similar `g` and `X`) but
different `D`. Thresholds are configurable:

```bash
python -m analysis.visualize_token_examples \
  --input /path/to/cmt_run/analysis \
  --output /path/to/report.html \
  --similar-g-relative-tolerance 0.05 \
  --similar-g-absolute-tolerance 1e-6 \
  --minimum-d-gap 0.1
```

The command also writes `mined_examples.json`. If only older `progress/` logs
exist, it creates a scalar-only report and explicitly marks decoded context,
candidate distributions, and sequence ranks as unavailable; these values
cannot be reconstructed from the old compact records. It joins a token ID from
`tokens/` when the independently sampled coordinates happen to match. Pass
`--tokenizer /local/path/to/the/training-tokenizer` to decode those matched IDs;
this still cannot recover the missing prefix or before/after candidate set.

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
- Qualitative mode adds no backward pass and saves no full-vocabulary tensor. It reuses the scheduled before/after probe and stores at most `max_candidates` decoded candidates per selected position. Candidate ranks therefore refer to the retained support, not the entire vocabulary.
- The report's “good/worse” wording uses only measured teacher-alignment signals: local/future fixed-support KL change, sampled-target log-probability gain, and teacher-preferred-candidate probability gain. It does not judge prose by a text heuristic.
- Short before/after generations are not emitted. Producing a true “before” continuation after the optimizer step would require retaining or cloning a model state and would materially raise storage/runtime. Candidate probability movement gives a deterministic local comparison without that intervention.
- This pipeline does not establish that teaching a specific token caused downstream access to change. That claim requires a separate intervention or branching-inference experiment; a full-budget performance sweep also requires already trained comparison runs or additional training.
