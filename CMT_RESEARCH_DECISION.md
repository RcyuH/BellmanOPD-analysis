# Research decision: conditional GO for CMT-OPD

**Superseded formulation note:** this document records the first cumulative/full-
vocabulary CMT prototype. The support-consistent, length-robust, bounded
truncated-kernel formulation now used by the code is documented in
[`CMT_REFINEMENT_DECISION.md`](CMT_REFINEMENT_DECISION.md).

Ngày 2026-09-06, sau khi audit implementation của `TA-OPD-B200`, kiểm tra
literature, và stress-test các formulation tuần tự, quyết định là **GO có điều
kiện** cho một method mới tên **Coupled Marginal Teachability (CMT-OPD)**.

Đây là GO để chạy một hypothesis có thể bị bác bỏ, không phải claim rằng CMT đã
tăng accuracy. CMT chỉ đáng trở thành contribution nếu các diagnostic ở §11 cho
thấy score dự đoán learning signal tốt hơn các baseline với cùng budget. Nếu
không, quyết định đúng về mặt nghiên cứu là NO-GO; repository vẫn giữ code như
một protocol để kiểm định hypothesis đó.

## 1. Research question và ranh giới claim

Tại prefix `s` do student tự sinh, teacher supervision nên được nhấn mạnh bao
nhiêu? Câu hỏi không phải là “teacher và student bất đồng bao nhiêu”, mà là:

> Với một local update khả dụng của student, supervision tại `s` tạo ra bao
> nhiêu cải thiện objective distillation ở hiện tại, và bao nhiêu cơ hội học
> downstream còn có thể được mở ra bởi action hiện tại?

Phần “downstream” chỉ được đưa vào khi có một continuation operator cụ thể. CMT
không gọi score là task value, causal credit, teacher-policy value, hay exact
shared-parameter update value. Nó là một **local model-space opportunity score**
cho một surrogate finite-horizon đã định nghĩa rõ.

Các constraint được giữ nguyên:

* teacher và student chỉ được score trên student-generated states;
* không learned critic, reward model, counterfactual rollout hay tree search
  trong training;
* score được tính từ hai logits đã cần cho OPD/TA, với một số reduction full
  vocabulary bị giới hạn bộ nhớ;
* score detached và chỉ dùng để phân bổ/select OPD supervision;
* giữ nguyên rollout, checkpoint, FSDP/vLLM, evaluation và logging protocol của
  `TA-OPD-B200` tối đa có thể.

Claim bị cấm trong paper nếu không có thêm evidence: “CMT đo learning value thật”,
“Bellman value chính xác”, “global maximal trajectory agreement”, “causal token
importance”, hoặc “giải quyết gradient interference”.

## 2. Audit chính xác implementation hiện tại

### 2.1 Shared OPD path

Student sinh response on-policy. Teacher không sinh response; cả hai model được
forward trên cùng các prefix/token IDs. Với mỗi position, code giữ student
Top-K, teacher Top-K và các log-probability trên support cần cho loss. Upstream
OPD objective được pin theo commit `ac26e38d6f1572eb027597b48a9f4e01f6915ef8`:

```text
alpha_k = softmax(student logits on optimization support)
A_k     = alpha_k * (log q_k - log p_old_k)       # detached
loss_t  = PPOClip(log p_theta,k / log p_old,k, A_k)
L       = sum_t w_t * loss_t / sum_t w_t
```

Rollout token vẫn là student token; teacher chỉ cung cấp target distribution.
Padding/terminal reset theo từng response; mọi global normalization được gather
qua các FSDP rank.

### 2.2 TA-OPD

TA tạo literal union `U=TopK(p) ∪ TopK(q)`, renormalize hai distribution trên
`U`, tính local forward KL `D=KL(q_U||p_U)` và teacher mass trên student Top-K
`C`. Hai scalar được robust global quantile-normalize rồi score
`Norm(D)*Norm(C)`. Hard global top-`rho` token được chọn; loss chính vẫn chỉ có
support đã được selector cho phép.

Điều này giải quyết một vấn đề có thật: ưu tiên vị trí disagreement nhưng tránh
những vị trí teacher không đáng tin trên student support. Tuy nhiên `C` vừa là
reliability correction vừa có thể dập tắt đúng teacher-only actions mà student
cần học. Quantile scale phụ thuộc batch và không đại diện cho update magnitude.

### 2.3 Bellman-RAC hiện tại

RAC đặt `g_t=s_TA(t)` và dùng sampled action `y_t`:

```text
a_t = min(1, q(y_t|s_t) / p(y_t|s_t))
R_t = g_t + gamma * a_t * R_{t+1}
M_t = 1   + gamma * a_t * M_{t+1}
V_t = R_t / (M_t + eps)
w_t = w_min + (1-w_min) * robust_quantile(V_t)^beta
```

Đây là normalized suffix average trên realized trajectory. Autoregressive
transition deterministic không đủ để biến nó thành Bellman value: Bellman
evaluation còn cần immediate reward/cost và continuation policy cố định. Trên
một rollout, `R_{t+1}` là return của action đã sample, không phải causal effect
của token `y_t`. Nếu loss là state-local KL và scoring/rollout stop-gradient,
đưa future KL về trước còn thay đổi objective và thiếu occupancy-gradient term.

Vì vậy CMT không mặc định giữ `V=R/M` làm training score. Code vẫn log `R`, `M`,
`V` để kiểm tra hypothesis; score chính dùng một derivative của successor
opportunity được định nghĩa ở §6.

### 2.4 PGT là baseline lịch sử, không phải claim cuối cùng

`BellmanOPD/RESEARCH_DECISION.md` mô tả PGT (Projected-Gradient Teachability):
trên union support, `s_PGT=Var_p(log q-log p)`. PGT là một baseline quan trọng
vì nó trả lời local question tốt hơn raw KL: nó đo natural-gradient energy của
OPD direction dưới categorical Fisher. CMT tái sử dụng PGT support/loss và thêm
một term sequential có semantics cụ thể; `g` của CMT là full-vocabulary Fisher
gain, còn `s_PGT` của baseline hiện tại là union-support gain. Vì vậy PGT-union,
full-vocabulary `lambda=0` và PGT-student-only đều là các ablation bắt buộc;
không được gọi `lambda=0` là bit-for-bit cùng implementation với PGT-union.

## 3. Literature và novelty audit

Không có bằng chứng rằng một trong các mảnh riêng lẻ dưới đây là mới:

| Mảnh | Prior gần nhất | Kết luận novelty |
|---|---|---|
| reverse-KL/local disagreement | TA-OPD, MiniLLM, TIP | không mới |
| `Var_p(log q-log p)`/Fisher energy | natural-gradient và gradient-allocation literature | không claim mới riêng lẻ |
| `min(1,q/p)` | speculative decoding; maximal coupling | không mới |
| return-to-go/suffix propagation | MiniLLM, OPPO, KETCHUP, Bellman Distill, TOPD | không mới |
| prefix importance / future branches | IW-OPD, TOPD, FutureBridge | competing mechanisms |
| KL-constrained Gibbs allocation | REPS và entropy-regularized allocation | allocator không mới |
| **đạo hàm successor common-mass dưới local teacher-directed path, ước lượng bằng đúng một student rollout, rồi dùng như state allocation score** | chưa tìm thấy targeted OPD work dùng đúng operator này | **novelty candidate, phải kiểm chứng thực nghiệm** |

Các nguồn cần đặt cạnh method khi viết paper: [TA-OPD](https://arxiv.org/abs/2605.26844),
[MiniLLM](https://arxiv.org/abs/2306.08543), [TIP](https://arxiv.org/abs/2604.14084),
[IW-OPD](https://arxiv.org/abs/2606.22600), [TOPD](https://arxiv.org/abs/2606.00305),
[FutureBridge](https://arxiv.org/abs/2608.01953), [OPPO](https://arxiv.org/abs/2605.21851),
[KETCHUP](https://aclanthology.org/2026.findings-eacl.39/), [vOPD](https://arxiv.org/abs/2605.07865),
và [REPS](https://ojs.aaai.org/index.php/AAAI/article/view/7727).

Primitive `min(p,q)` là common mass của maximal coupling; trong speculative
decoding nó xuất hiện như accept/residual rule ([Speculative Decoding,
ICML 2023](https://proceedings.mlr.press/v202/leviathan23a.html)). Maximal coupling
trên từng step không đồng nghĩa global maximal path agreement; MEXIT chỉ ra
lookahead có thể thắng greedy stepwise coupling ([MEXIT](https://doi.org/10.1016/j.spa.2018.03.001)).
Do đó contribution ở đây không phải “phát minh coupling”, mà là định nghĩa một
surrogate learning-opportunity operator và đạo hàm local của nó trong OPD.

## 4. First-principles: state nào đáng nhận supervision?

Ba tín hiệu thường bị trộn lẫn nhưng không tương đương:

1. **Disagreement**: `p` và `q` khác nhau. Có thể lớn nhưng student không có
   hướng cập nhật hữu ích, hoặc support bị truncation.
2. **Accessibility**: student có thể chuyển mass về hướng teacher không, dưới
   một update budget cố định? Đây là learnability/local policy geometry.
3. **Downstream opportunity**: action hiện tại đưa rollout vào những prefix nơi
   còn nhiều local learning opportunity hay không?

Một score có ý nghĩa cần tách (1) khỏi (2), và chỉ thêm (3) nếu transition
operator nói rõ được action nào còn accessible. CMT dùng (2) là PGT local gain,
và (3) là common-mass successor opportunity. Nó không giả định teacher policy là
continuation policy, không dùng task reward, và không giả định neural parameter
coordinates có thể thay đổi độc lập.

## 5. Candidate comparison và rejected directions

| Candidate | Assumption | Failure mode | Quyết định |
|---|---|---|---|
| TA `D*C` | disagreement × compatibility là useful value | teacher-only suppression, batch-scale dependence | baseline |
| PGT | categorical Fisher local OPD direction xấp xỉ update utility | truncation, parameter sharing | baseline mạnh / local term |
| RAC/Bellman return | suffix score là causal credit | realized suffix không phải counterfactual, reward chưa định nghĩa | ablation, không làm theory |
| MiniLLM-like return-to-go | future log-ratio đáng credit cho prefix trước | same sampled continuation; không có alternative action | reject as main |
| entropy/disagreement/TIP proxy | uncertainty/disagreement tương quan learning value | unlearnable hoặc gradient-irrelevant state | diagnostics |
| critic/value model | học continuation value | vi phạm constraint, critic bias | reject |
| counterfactual/tree rollout | đo causal downstream effect | compute vượt constraint | diagnostic-only |
| parameter-gradient transfer | score theo held-out gradient alignment | noisy, optimizer/parameterization-dependent, overlap với GradMatch/GLISTER | reject as main |
| **CMT** | common-mass successor là accessible continuation opportunity | local greedy coupling không global; frozen descendants; noisy ratio | **main candidate, conditional GO** |

CMT được chọn vì nó thêm một mechanism chứ không chỉ thay exponent/discount:
future opportunity chỉ truyền qua **common probability mass** và chỉ qua hướng
teacher có thể tăng dưới local path. Nếu tắt derivative hoặc thay `min(p,q)` bằng
`a=1`, method suy biến thành các return-to-go baseline.

## 6. CMT formulation

### 6.1 Local quantities

Tại state/prefix `s`, viết `p_a=p(a|s)`, `q_a=q(a|s)`,

\[
r_a=\log q_a-\log p_a,\qquad \bar r=\mathbb E_p[r].
\]

Xét local teacher-directed mirror path

\[
p_\eta(a)=\frac{p_a\exp(\eta r_a)}{\sum_b p_b\exp(\eta r_b)}.
\]

Khi đó `dot p_0(a)=p_a(r_a-bar r)` và local reverse-KL improvement là

\[
g(s)=-\left.\frac{d}{d\eta}KL(p_\eta\|q)\right|_{0}
    =\operatorname{Var}_{p}[r].
\]

`g` chính là PGT full-vocabulary gain. Nó không phải teacher disagreement đơn
thuần: constant `r` có thể tạo KL nhưng có zero policy direction.

### 6.2 Common-mass successor operator

Định nghĩa sub-Markov operator trên student-generated successor states:

\[
Kf(s)=\sum_a \min(p_a,q_a) f(sa).
\]

Với `Y~p`,

\[
a(s,Y)=\min(1,q_Y/p_Y),\qquad
\mathbb E_p[a(s,Y)f(sY)]=Kf(s).
\]

`E[a]=sum_a min(p_a,q_a)=1-TV(p,q)` là xác suất sống của one-step maximal
coupling. Đây là **accessibility** chứ không phải policy value hay task success.

Cho finite response horizon `H`,

\[
R_t=g_t+\gamma K R_{t+1},\qquad R_H=0.
\]

`gamma=1` là default vì response đã finite-horizon; `gamma<1` chỉ là explicit
time preference/geometric killing, không được giới thiệu như variance fix.

### 6.3 Marginal successor opportunity

Giữ descendants `R(sa)` cố định khi lấy local derivative. Future opportunity tại
state `s` là `C(s)=K R(s)`. Vì đạo hàm của `min(p_a,q_a)` chỉ khác 0 khi
`p_a<q_a`, ta có

\[
L_\lambda(s)=g(s)+\lambda\gamma
\sum_{a:p_a<q_a}p_a(r_a-\bar r)R(sa).
\]

Term thứ hai chỉ thưởng cho mass mà local teacher-directed update có thể tăng và
đồng thời successor còn có `R`. Nó không truyền future score qua actions mà
teacher đã không yêu cầu (các `p_a>q_a` đang bị killed common mass).

Trên một student rollout, estimator dùng trong code là

\[
\widehat L_t=g_t+\lambda\gamma\,
  1[r(y_t)>0](r(y_t)-\bar r_t)\widehat R_{t+1},
\]

với

\[
\widehat R_t=g_t+\gamma a_t\widehat R_{t+1}.
\]

Điều kiện `r(y)>0` tương đương `p_y<q_y`. Với descendants cố định và finite
horizon, estimator là unbiased cho surrogate `L_lambda` bằng induction. Nó không
unbiased cho true shared-neural-network parameter update, vì parameter sharing
và descendant logits thay đổi đồng thời bị bỏ qua.

### 6.4 Allocation policy

Thay hard top-`rho` mặc định, CMT giải allocation trên global valid token set:

\[
\max_{w_i\ge0}\sum_i \frac{w_i}{N}L_i
\quad\text{s.t.}\quad
KL(w/N\|u_N)\le\epsilon,
\]

với `u_N` uniform. Nghiệm là Gibbs tilt

\[
w_i=N\frac{\exp(L_i/\tau)}{\sum_j\exp(L_j/\tau)},
\]

`tau` được tìm bằng bisection để đạt budget `epsilon`. Mean weight bằng một,
mọi valid token vẫn có support, và allocation không phụ thuộc affine scale của
`L` khi `epsilon` cố định. Đây là một allocator chuẩn (REPS-like), không phải
novelty; novelty candidate nằm ở `L`, không ở Gibbs.

Trong training, `L` và `w` detached. Gibbs của noisy `L_hat` không phải Gibbs của
true `L` (nonlinear bias); vì vậy code log variance/replicate diagnostics và
không claim exact optimal allocation.

### 6.5 Sequential choice và failure boundary

CMT có thể tắt sequential term bằng `cmt_successor_lambda=0`, khi đó nó trở về
full-vocabulary PGT gain. `cmt_gamma=1` là primary. `R/M/V` được log để so sánh
với RAC nhưng `V` không dùng làm main score. Nếu experiments cho thấy
`lambda>0` không cải thiện held-out useful-learning prediction, phải report PGT
hoặc NO-GO thay vì giữ Bellman vì tên method.

## 7. Pseudocode

```text
for student rollout batch:
    score student and teacher logits on the same prefixes
    U_t <- stable union(StudentTopK_t, TeacherTopK_t)
    p_U, q_U <- normalized probabilities on U_t       # differentiable OPD support
    g_t <- Var_{full p_t}[log q_t - log p_t]           # exact reduction
    m_t <- sum_vocab min(p_t, q_t)                     # common mass
    r_t <- log q_t[y_t] - log p_t[y_t]
    a_t <- exp(min(r_t, 0))
    R_t <- g_t + gamma * a_t * R_{t+1}                 # suffix estimator
    flux_t <- 1[r_t > 0] * (r_t - E_p[r_t])
    L_t <- g_t + lambda * gamma * flux_t * R_{t+1}

    gather L over all distributed valid tokens
    w <- Gibbs allocation under KL(w || uniform) <= epsilon
    train the existing OPD loss on U_t with global weighted-token mean w_t
    log g, m, a, R, M, V, flux, sequential_gain, L and w
```

No extra model forward is needed beyond the joint student/teacher score already
required by TA/PGT. Full-vocabulary quantities are reductions while both bounded
micro-batch logit views coexist; the differentiable OPD support remains the
memory-efficient Top-K union.

## 8. Implementation map

The new sibling directory is `/mnt/hdd/nhatminh/OPD/BellmanOPD/`; the original
`TA-OPD-B200/` tree remains untouched.

* `b200_experiment/scoring.py`: exact FP32 reductions of `E_p[r]`, `Var_p[r]`
  and `sum min(p,q)` in bounded chunks; no full-vocabulary tensor is retained.
* `b200_experiment/selectors/cmt_selector.py`: recurrence, marginal flux,
  `CMTSelector`, and KL-constrained allocation.
* `b200_experiment/trainer.py`: CMT dispatch, strict on-policy sampling check,
  global allocation, detached selector diagnostics and shared OPD loss.
* `configs/qwen3_b200_cmt.yaml`, `scripts/train_cmt_b200.sh`: production launch
  path inheriting base model/data/distributed/checkpoint settings.
* `scripts/eval_cmt_b200.sh`, re-evaluation/plotting/checkpoint hooks and
  TensorBoard/JSONL logging: CMT is a first-class method identifier.
* `tests/test_cmt_selector.py` and compatibility tests: recurrence, finite
  difference derivative, padding boundary, allocation budget and logging.

The existing PGT files and `RESEARCH_DECISION.md` are intentionally preserved as
the local baseline and ablation documentation.

## 9. Complexity, hyperparameters and numerical safety

Let `B` be rollout trajectories, `T` response length, and `V` vocabulary size.
Joint scoring has the same two model forwards as the optimized TA/PGT path; CMT
adds `O(BTV)` arithmetic reductions in chunks, not another forward. Selector scan
is `O(BT)` (the existing affine suffix scan), and global allocation is `O(N)` per
bisection iteration (`N` valid tokens, 64 iterations by default). There is no
counterfactual generation and no critic.

Defaults:

```text
top_k = 16
cmt_gamma = 1.0
cmt_successor_lambda = 1.0
cmt_allocation_kl = 0.5
score_chunk_steps = 128
pgt_vocab_chunk_tokens = 2048
```

The rollout distribution must equal the scored student distribution for the
one-rollout estimator: CMT rejects `top_p != 1` and non-positive temperature.
All ratio/common-mass reductions use FP32; `logsumexp` is used for normalization;
`exp(min(r,0))` prevents ratio explosion; invalid/padded positions are zeroed;
finite checks fail fast. The full-vocabulary score is exact for the scored logits,
but the training loss still uses Top-K union; `teacher_tail_mass` is logged and
must be stratified in analysis.

The sequential term is always non-negative under this local path (`bar r<=0` and
`r>0`), so CMT is not a negative-transfer/interference score. `lambda` therefore
controls downstream emphasis, not signed correction. Near `p=q`, the local term is
second-order while the successor derivative can be higher-order; a large default
lambda must not be assumed universally optimal.

## 10. What counts as useful learning signal?

Correlation with training progress alone is insufficient. The primary validation
target is **held-out immediate distillation improvement per equal update budget**:

1. Freeze a rollout batch and compute each candidate score.
2. Partition tokens into score quantiles/equal-budget selections.
3. Apply a small, equal-norm local OPD update (or compute its exact categorical
   local update for a tiny-vocabulary sanity case).
4. Re-score the same prefixes and a held-out set of prefixes after the update.
5. Measure reduction in full reverse KL and improvement in held-out OPD objective,
   not only teacher agreement or task accuracy.

For a more realistic audit, apply the same shared-parameter micro-update to each
equal-budget subset in isolated copies/checkpoints and report rank correlation
between pre-update score and post-update held-out KL reduction. This diagnostic
may be expensive and is not part of CMT training; it is required to distinguish a
heuristic progress correlate from a useful-learning predictor.

## 11. Minimal falsification/ablation matrix

At fixed seeds, data order, rollout protocol, optimizer steps, selected-token
budget and evaluation checkpoints, run:

1. OPD uniform; TA-OPD; Bellman-RAC; PGT-union; PGT-student-only; CMT.
2. CMT with `lambda=0` (PGT control), `gamma=0`, and shuffled successor `R`.
3. CMT with `a=1` and with unclipped `q/p` only as diagnostic controls; these
   should test whether common-mass killing matters.
4. `epsilon in {0.0, 0.1, 0.5, 1.0}` and hard top-`rho` equal-budget allocator.
5. `K in {8,16,32}` and tail-mass strata; report score/loss support mismatch.
6. Student-vs-teacher disagreement, entropy, reverse-KL and random equal-budget
   selectors as non-method baselines.
7. Tiny-vocabulary exact enumeration: compare `R`, `K R`, and the finite-
   difference derivative of `C(p_eta)` against the one-rollout estimator.
8. Parameter-sharing audit: compare categorical `g` rank with actual shared-model
   one-step held-out KL reduction. A low rank correlation falsifies the strong
   interpretation of PGT/CMT, even if final task accuracy improves.

Required diagnostics include Spearman/Pearson rank correlation, top-budget lift,
calibration by score quantile, variance across rollout seeds, selected-weight KL,
teacher tail mass, common mass, `R/M/V`, sequential-vs-local contribution, and
actual full reverse-KL reduction. A result that only improves a noisy progress
curve but fails held-out update prediction is not evidence for learning value.

## 12. Decision rule

The implementation is a **conditional GO** because the common-mass marginal
operator is a real methodological distinction from TA/PGT/RAC and is practical
under the requested constraints. It becomes a publishable contribution only if:

* CMT predicts held-out useful learning signal better than PGT/TA/RAC at equal
  budget;
* the improvement survives `lambda=0`, shuffled-suffix, and tail/support
  controls;
* the gain is not solely due to Gibbs reweighting or changed total gradient
  mass;
* failure boundaries and parameter-sharing mismatch are reported honestly.

If these conditions fail, the correct final paper decision is **NO-GO** rather than
renaming a weighting/normalization variant. The code is intentionally structured
so that this falsification is cheap: setting `lambda=0` gives the full-vocabulary
PGT-style control, while the existing union-support PGT baseline remains available,
and all intermediate quantities are logged.

## 13. Idea-evaluator audit (Higher/Faster/Stronger/Cheaper/Broader)

| Dimension | Assessment | Evidence required |
|---|---|---|
| Higher | Có một semantics rõ hơn `D*C`: marginal improvement của local OPD cộng accessible successor opportunity | held-out KL reduction và calibration theo score quantile |
| Faster | Không thêm generation/critic/forward; nhưng full-vocabulary reductions có thể chậm hơn PGT | B200 wall-clock, peak memory và tokens/s; không assume speedup |
| Stronger | `min(p,q)` có probabilistic meaning và derivative finite-difference kiểm được; không phải causal value | exact tiny-vocab check, shuffled-suffix and parameter-sharing audit |
| Cheaper | Rẻ hơn critic/tree methods, nhưng đắt hơn Top-K-only do `O(BTV)` reductions | score-time breakdown và ablation `lambda=0` |
| Broader | Áp dụng cho categorical on-policy distillation beyond Qwen3/Competition-MATH | ít nhất một model/data transfer test nếu paper claim generality |

Lifecycle hiện tại là **validation-stage idea**, chưa phải empirical contribution.
Fatal-flaw audit: (i) common mass/acceptance đã có trong speculative decoding và
maximal coupling; (ii) local greedy coupling không global path-optimal; (iii)
future term không causal nếu không có alternative continuation; (iv) shared neural
parameters phá exactness của categorical derivative; (v) Gibbs allocation của
noisy score bị nonlinear bias; (vi) positive-only successor term không biểu diễn
negative transfer. Nếu một flaw trong số này quyết định kết quả, phải hạ claim
hoặc NO-GO chứ không thêm normalization/discount để che lấp.

## 14. Verification status

Passed in the available CPU environment:

* CMT selector, finite-difference and allocation tests;
* scoring and PGT compatibility tests;
* selector logging, TensorBoard tag, launcher and checkpoint-evaluation tests;
* Python bytecode compilation and `bash -n` for all shell scripts;
* CMT YAML loading and strict method/config wiring.

Full B200 training and matplotlib/pandas-dependent plotting were not run here:
the available host has an incompatible NVIDIA driver and the lightweight test
environment lacks those optional plotting dependencies. These are execution
limitations, not evidence of training quality. No accuracy or speedup claim is
made until the B200 matrix in §11 is run.
