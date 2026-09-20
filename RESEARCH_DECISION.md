# Research decision: GO for Projected-Gradient Teachability (PGT)

**Historical baseline note:** this file documents the earlier PGT hypothesis. The
current trajectory-aware research decision and implementation are in
[`CMT_RESEARCH_DECISION.md`](CMT_RESEARCH_DECISION.md).
The current code additionally conditionalizes both student and teacher on the
literal union so that PGT/CMT training geometry matches the executable support;
see [`CMT_REFINEMENT_DECISION.md`](CMT_REFINEMENT_DECISION.md).

Ngày 2026-09-06, sau khi đọc lại toàn bộ implementation của `TA-OPD-B200`,
đối chiếu literature về OPD/token selection/sequential credit, và chạy các test
không cần CUDA, quyết định là **GO có điều kiện**: triển khai PGT để kiểm định
một research hypothesis đủ rõ và có thể bị bác bỏ. Đây chưa phải claim rằng PGT
đã cải thiện accuracy; repository này là implementation + protocol để thực hiện
phép kiểm định đó.

## 1. Research question và gap thực sự

Tại một state/prefix do student sinh ra, supervision của teacher đáng được ưu
tiên nếu local update của student có thể tạo ra **policy improvement hữu ích với
chi phí trust-region cố định**, không chỉ vì teacher và student bất đồng, có
entropy cao, hay teacher đặt mass ngoài Top-K của student.

Audit literature cho thấy các tín hiệu gần nhất giải quyết các phần khác nhau:

* TA-OPD dùng local forward-KL trên hợp của hai Top-K và nhân với teacher mass
  trên student Top-K ([TA-OPD](https://arxiv.org/abs/2605.26844)). Đây là compatibility
  heuristic, không phải độ lớn của update policy khả dụng; teacher-only actions
  còn bị loại khỏi loss chính.
* TrOPD dùng trust probability kiểu `min(q/p, 1)` để giảm outlier
  ([TrOPD](https://arxiv.org/html/2606.01249)). Nó là reliability correction,
  không phải state-wise learning value.
* TIP xây oracle token weight dựa trên signal/curvature và dùng proxy entropy/
  disagreement cho training ([TIP](https://arxiv.org/html/2604.14084)); nó không
  cung cấp selector không-critic tính đúng projected gain của local OPD objective.
* vOPD cung cấp closed-form value baseline và phân tích baseline tối ưu
  ([vOPD](https://arxiv.org/html/2605.07865)); đó là variance/value-baseline
  correction, không phải một state selector cùng support với loss.
* OPRD đo signal-to-noise ở mức distillation update
  ([OPRD](https://arxiv.org/html/2606.06021)); đây là global/representation-level
  diagnostic, không phải local teachability.
* IW-OPD đặt OPD trong một trust-region projection ở mức prefix và suy ra
  importance weight từ teacher/student prefix ratio
  ([IW-OPD](https://yannx1e.github.io/IW-OPD/)). Đây là một competing sequential
  allocation rule, không phải bằng chứng cho Bellman suffix recursion.
* TOPD dùng short-horizon continuation/trajectory alignment để nhận diện các
  reasoning forks ([TOPD](https://arxiv.org/abs/2606.00305)); phương pháp đó có
  near-future computation mà PGT chủ động không đưa vào training.
* Fisher-Projected OPD cho vision-language models chiếu teacher correction vào
  tangent space được ước lượng bằng visual perturbations
  ([FP-OPD](https://arxiv.org/abs/2608.01263)). Đây là related-but-different và
  quan trọng đối với novelty: FP-OPD thay đổi target để phù hợp capacity, còn
  PGT chỉ dùng categorical Fisher để định lượng scalar token allocation, không
  dùng perturbation/counterfactual probe. Nếu PGT không vượt TA/IW-OPD hoặc
  capacity mismatch chi phối kết quả, novelty claim phải hạ xuống hoặc bỏ.
* MiniLLM truyền return-to-go của log-ratio về token
  ([MiniLLM](https://arxiv.org/html/2306.08543)); Czarnecki et al. phân tích
  policy-distillation correction ([Policy Distillation](https://proceedings.mlr.press/v89/czarnecki19a.html)).
  Cả hai không cho thấy Bellman suffix average là causal learning value cho
  stop-gradient local OPD.
* CROP dùng counterfactual relevance ([CROP](https://arxiv.org/html/2608.13387),
  không phù hợp constraint “không counterfactual rollout trong training”).

Vì vậy gap được đặt hẹp và falsifiable như sau:

> Có thể dùng chính local OPD gradient và local student Fisher metric để xếp hạng
> state/token, không learned critic, không counterfactual rollout, không Bellman
> assumption, và score đó có dự đoán tốt hơn các heuristic TA/RAC về one-step
> distillation improvement hay không?

Đây là một gap phương pháp có thể kiểm định, không phải tuyên bố rằng mọi
gradient norm hoặc mọi natural-gradient quantity đều mới. PGT chỉ đáng giữ nếu
diagnostics ở §7 xác nhận score dự đoán useful learning signal; nếu không, phải
quay về baseline/NO-GO.

## 2. Reconstruction chính xác của baseline hiện tại

### OPD objective

Trong `TA-OPD-B200/b200_experiment/opd_core.py`, objective được pin theo upstream
commit `ac26e38d6f1572eb027597b48a9f4e01f6915ef8`: student-generated rollout,
student Top-K (`K=16`),

\[
 p_j=\operatorname{softmax}(\ell^S_j),\qquad
 r_j=\log q_j-\log p_j,\qquad
 a_j=p_jr_j .
\]

PPO-style candidate loss dùng `a_j` detached và token-mean toàn response. Teacher
không được rollout; teacher chỉ score cùng student token IDs.

### TA-OPD

TA score cả student Top-K và teacher Top-K, union/renormalize local support, tính
`D = KL(q_union || p_union)` và `C = teacher mass on student Top-K`, robust global
quantile-normalize rồi dùng `s_TA=D_norm*C_norm`. Hard global Top-`rho` chọn token
(mặc định `rho=0.10`), nhưng loss vẫn chỉ train student Top-K. `C` vì thế vừa là
reliability heuristic vừa có thể triệt tiêu đúng các token mà teacher muốn mở
rộng support.

### Bellman-RAC

Implementation hiện tại đặt `g=s_TA`,
`a=min(q(sampled)/p(sampled),1)`, rồi scan suffix:

\[
 R_t=g_t+\gamma a_tR_{t+1},\quad
 M_t=1+\gamma a_tM_{t+1},\quad V_t=R_t/M_t .
\]

Đây là normalized suffix average. Deterministic autoregressive transition không
tự biến nó thành Bellman value: Bellman evaluation cần reward/cost và một
continuation policy cố định; trên một rollout, suffix là return của action đã
sample, không phải causal effect của token. Nếu loss là state-local KL và
rollout/teacher score stop-gradient, truyền KL của tương lai ngược về token hiện
tại còn thay đổi objective (và thiếu occupancy-gradient correction). Vì vậy PGT
**bỏ recurrence, `gamma`, sampled-ratio alignment và Bellman normalization**.

## 3. Formulation đã chọn: PGT

Gọi state/prefix là `c`. Với mỗi state, lấy literal support union

\[
 U(c)=\operatorname{TopK}(p_c)\cup\operatorname{TopK}(q_c),
\]

và dùng đúng support này cho cả score lẫn OPD loss. Trên `U`, student được
renormalize:

\[
 p_j=\frac{\exp \ell^S_j}{\sum_{k\in U}\exp\ell^S_k},\quad
 r_j=\log q_j-\log p_j,\quad
 \bar r=\sum_{j\in U}p_jr_j .
\]

Với student logits `z`, expected local OPD update có hướng

\[
 \mu_j=\frac{\partial}{\partial z_j}\sum_kp_kr_k
       =p_j(r_j-\bar r).
\]

Categorical Fisher là `F = diag(p) - p p^T`. Với một local KL/trust-region
budget, maximum first-order improvement là

\[
 s_{PGT}(c)=\mu^\top F^+\mu
           =r^\top Fr
           =\sum_{j\in U}p_j(r_j-\bar r)^2
           =\operatorname{Var}_{p}[\log q-\log p].
\]

Đây là natural-gradient energy của **chính OPD update local**, không phải KL
đơn thuần. Nó có ba tính chất hữu ích:

1. invariant với additive logit/log-probability shift;
2. bằng zero nếu mọi `log q - log p` trên support là hằng số (không có policy
   direction để học dù divergence có thể lớn);
3. teacher-only actions được đưa vào support và loss, thay vì bị `C` loại bỏ
   trước khi gradient xuất hiện.

`teacher_tail_mass = 1 - sum_{j in U} q_j` được log riêng. Khi tail mass lớn,
PGT score được coi là truncated-support estimate và ablation phải kiểm tra độ
nhạy theo `K`; không được diễn giải nó như full-vocabulary optimum.

Assumption quan trọng cần kiểm chứng: categorical logits tại một prefix được
xem như local policy coordinates. Neural parameter sharing khiến student không
thể điều chỉnh từng logit độc lập; do đó `s_PGT` là upper-level local utility,
không phải exact parameter-space improvement. FP-OPD cho thấy capacity/tangent
space có thể làm assumption này sai. Đây là lý do one-step parameter update và
capacity/support strata trong §7 là bắt buộc, không phải optional visualization.

### Loss và allocation

PGT vẫn dùng upstream PPO/clipped machinery, nhưng candidate support là `U` và
`p` được normalize trên `U`; score không backprop qua selector. Hard global
Top-`rho` trên `s_PGT` quyết định state/token nào nhận supervision. Không có
critic, rollout bổ sung, reward model, recurrence, discount hay teacher sampling.

Pseudo-code:

```text
for student rollout batch:
    score student and teacher on the same sampled token states
    S <- student TopK IDs/logprobs; T <- teacher TopK IDs/logprobs
    U <- stable unique(S union T)
    p <- softmax(student logprobs on U)
    r <- teacher_logprob(U) - student_logprob(U)
    score <- sum(p * (r - sum(p*r))^2)       # s_PGT
    gather score and diagnostics across ranks
    selected <- global_top_rho(score, valid_response_tokens)
    train OPD/PPO loss on U, masked to selected tokens
    log support width, teacher tail mass, reverse-KL proxy, and selected fraction
```

### Compute, hyperparameters, numerical behavior

* Existing joint student/teacher cross-scoring is reused; no additional model
  forward beyond the TA union path.
* Selector cost is `O(B*T*K)` after scoring. Candidate width is at most `2K`
  (default 32 rather than 16), so candidate gather/backward memory is the main
  overhead; full-vocabulary logits are never materialized by PGT.
* Defaults: `top_k=16`, `rho=0.10`, `pgt_vocab_chunk_tokens=2048`, global hard
  budget and FP32 selector arithmetic. `PGT_RHO` overrides `TA_RHO` in the
  launcher.
* Duplicate IDs are masked deterministically (first occurrence wins); invalid
  rollout padding has zero score/weight. `-inf` is used only before softmax on
  unsupported candidates; all detached diagnostics are finite.
* `restricted_reverse_kl`, union masses and teacher tail mass are diagnostics,
  not extra weighting terms. This avoids silently turning PGT into another
  weighted recombination heuristic.

## 4. Candidate comparison and rejected directions

| Candidate | Core assumption | Main failure mode | Decision |
|---|---|---|---|
| TA `D*C` | disagreement × teacher compatibility predicts value | suppresses teacher-only support; scale depends on global quantiles | baseline only |
| Bellman-RAC | suffix local scores approximate causal future credit | no specified reward/continuation policy; return is realized suffix | retain for ablation, not theory |
| return-to-go / MiniLLM-like | future log-ratio is credit for current token | same sampled continuation is not counterfactual causal effect | no |
| vOPD/SNR-style | variance/value baseline fixes noisy policy gradient | useful correction but not a distinct state selector; close prior overlap | no as main contribution |
| entropy/disagreement/TIP proxies | uncertainty/disagreement tracks utility | high disagreement can be unlearnable or gradient-irrelevant | diagnostic baselines |
| counterfactual relevance | alternate continuation estimates usefulness | violates training compute constraint | diagnostic-only, if ever |
| **PGT** | local Fisher-projected OPD improvement is learning value | truncated support and parameter coupling can break proxy | **main method** |

PGT is deliberately not advertised as “the optimal learning value”. It is the
exact local natural-gradient quantity for the finite-support, stop-gradient OPD
surrogate in categorical policy coordinates. Its empirical advantage over
TA/IW-OPD/other equal-budget controls is a hypothesis, not a result; if the
parameter-sharing or support-truncation diagnostics fail, the correct outcome is
to withdraw the contribution.

## 5. What was implemented

The new sibling directory is `/mnt/hdd/nhatminh/OPD/BellmanOPD/`; the original
`TA-OPD-B200/` tree was not modified. PGT-specific changes are:

* `b200_experiment/selectors/pgt_selector.py`: union support, `s_PGT`, and
  diagnostics;
* `b200_experiment/opd_core.py`: optional support mask so score and loss use the
  same union;
* `trainer.py`: PGT branch, global budget, checkpoint/metrics/TensorBoard metadata;
* `configs/qwen3_b200_pgt.yaml` and `scripts/train_pgt_b200.sh`;
* `scripts/eval_pgt_b200.sh`, checkpoint re-evaluation and plotting hooks;
* `tests/test_pgt_selector.py` plus compatibility updates to selector/logging/
  checkpoint tests.

Inherited unchanged in spirit: vLLM rollout, shared no-think tokenizer protocol,
FSDP/gradient checkpointing, global distributed normalization, resume/checkpoint
format, evaluation cache, TensorBoard, JSONL/CSV logging and the pinned OPD
objective.

## 6. Minimal experiment matrix

At fixed seed/data/model/batch/optimizer, run:

1. upstream OPD (all valid tokens);
2. TA-OPD (current hard top-`rho` selector);
3. Bellman-RAC (current sequential baseline);
4. PGT-union (main);
5. PGT-student-only (same score but no teacher-only candidates);
6. PGT-union with shuffled scores / random equal-budget selection;
7. PGT with `K ∈ {8,16,32}` and `rho ∈ {0.05,0.10,0.20}`.

All comparisons need equal optimizer steps, selected-token budget, rollout
protocol, and evaluation checkpoints. Report accuracy and training efficiency,
but do not infer task correctness from teacher agreement alone.

## 7. Diagnostics that can falsify PGT

These are required before making a contribution claim:

* **One-step holdout:** freeze a rollout batch, apply one optimizer step using
  each token score, then measure reduction in exact finite-support reverse KL and
  teacher cross-entropy on a fresh score batch. Spearman correlation between
  pre-step `s_PGT` and per-token reduction must beat `D`, `C`, `D*C`, entropy and
  `|r_sampled|` baselines.
* **Gradient check:** compare `s_PGT` with the exact finite-support Fisher
  natural-gradient energy computed from dense logits on a small vocabulary; this
  validates the selector implementation, not downstream usefulness.
* **Support strata:** report low-overlap/high-gain tokens, teacher-tail-mass
  bins, support width, sampled-token rank, and selected-token fraction. The key
  prediction is a gain in low-overlap states where TA's `C` suppresses useful
  teacher-only actions.
* **Budget/selection controls:** random, uniform, TA, and PGT with the same
  selected count; compare score calibration and effective sample size.
* **Sequential falsification:** compare PGT with RAC and with RAC `gamma=0`.
  If future suffix propagation consistently improves the one-step local metric
  after controlling budget, the “no sequential signal needed” decision is false
  for that setting and should be revisited.
* **Stability:** monitor non-finite scores, teacher tail mass, union width,
  gradient norm, clipping fraction, score concentration/Gini, and sensitivity to
  `K`, `rho`, temperature, and candidate dtype.

Fail PGT if it does not beat equal-budget controls on the local holdout, if gains
vanish after controlling support tail mass, if teacher-only support increases
instability without useful improvement, or if a simpler TA/entropy baseline is
equally predictive. In that case the correct paper decision is NO-GO, not another
normalization/discount variant.

## 8. Running

```bash
cd /mnt/hdd/nhatminh/OPD/BellmanOPD
bash scripts/train_pgt_b200.sh
bash scripts/eval_pgt_b200.sh
PLOT_METHODS='opd ta rac pgt' bash scripts/plot_training_progress.sh
REEVAL_METHODS='pgt' bash scripts/reeval_all_checkpoints_b200.sh
```

The final evaluation command in `eval_all_b200.sh` keeps PGT opt-in via
`RUN_PGT_EVAL=true`, so existing OPD/TA/RAC workflows remain compatible.
