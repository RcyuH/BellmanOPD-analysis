# CMT-OPD refinement: support-matched bounded truncated-kernel formulation

> **Implementation note (2026-09-18):** production now separates two supports.
> CMT score/accessibility uses
> `U_t = TopK(student_t) union TopK(teacher_t)`, as derived below, while the
> differentiable OPD loss always uses Student Top-16. Teacher-only union tokens
> can affect learning-value allocation but never become policy-loss candidates.

Ngày 2026-09-06, tôi tiếp tục audit CMT v1 theo các failure mode mới được nêu:
geometry mismatch, length bias, variance của one-rollout estimator, coupling
semantics, frozen descendants, Gibbs allocation và inverse-coverage variance.
Kết luận là **giữ research direction nhưng thay sequential kernel**. CMT v1/v3
không còn là formulation chính.

## 1. Final decision

**GO có điều kiện cho CMT-OPD v4**, với năm thay đổi bắt buộc:

1. mọi local quantity của selector dùng cùng conditional Top-K-union simplex,
   còn OPD loss dùng Student Top-16 theo upstream `only_stu`;
2. local PGT/CMT geometry dùng conditional union simplex, còn sequential
   accessibility dùng original probability mass trên union;
3. rollout ngoài union bị coi là killed transition trong raw truncated kernel;
   không dùng inverse-coverage correction;
4. sequential value dùng local-baseline excess opportunity, không dùng batch
   reward-centering trong episodic trajectories;
5. `R/M/V` chỉ là occupancy diagnostics; training dùng đạo hàm marginal của
   excess opportunity.

GO ở đây nghĩa là có một surrogate nhất quán để chạy falsification experiments,
không phải accuracy claim. Nếu offline micro-update diagnostics không cho thấy
score mới dự đoán useful learning signal hơn PGT/TA/RAC, paper-level verdict phải
là NO-GO.

## 2. Những vấn đề thực sự của CMT v1

### Geometry mismatch

CMT v1 dùng `Var_p(log q-log p)` và common mass full vocabulary, trong khi loss
chỉ tối ưu literal `TopK(student) ∪ TopK(teacher)`. Ngoài ra PGT cũ chỉ
conditionalize student weights, còn teacher log-probability vẫn là global log
probability. Vì vậy selector không đo đúng update mà OPD core thực sự có thể
thực hiện.

### Length bias và episodic reward-centering

Với `g_t=g` và `a_t=1`, cumulative recursion cho `R_t=(H-t)g`. Token đầu được
ưu tiên chỉ vì còn nhiều token, dù mọi state có cùng opportunity. `M_t` là expected
accessible occupancy và `R_t/M_t` có semantics ratio-of-expectations, nhưng
one-rollout `R_hat/M_hat` là biased self-normalization và vẫn không phải causal
credit. V2 thay bằng batch-centered return `H=(g-rho_B)+KH`. Audit literature
cho thấy average-reward centering là canonical cho continuing problems, nhưng
reward shifts có thể đổi policy ordering trong episodic problems có termination
([De Asis et al., RLC 2026](https://arxiv.org/abs/2605.04368)). Vì vậy `rho_B`
không đủ để gọi đây là một episodic average-reward objective.

### Conditional-rollout mismatch và variance

Student rollout vẫn sample từ full `p`, còn common-mass transition của CMT v1
ngầm dùng distribution đã conditionalize trên union. Nếu token sampled nằm ngoài
union, estimator cũ không nói rõ đang ước lượng kernel nào. Nếu union coverage là
`m_p`, việc bỏ qua event này tạo bias cho conditional kernel; nếu sửa bằng inverse
coverage, variance tăng theo `1/m_p`. Đây là trade-off bắt buộc, không sửa bằng
arbitrary clipping.

### Frozen descendants

CMT chỉ là đạo hàm của local categorical surrogate khi giữ descendant
`R(sa), M(sa)` cố định.
Shared Transformer parameters làm logits ở các descendant prefix thay đổi cùng
nhau trong một update thật. Không có critic, counterfactual rollout, per-token
full gradients hay global Fisher nào cho phép sửa exact effect trong budget này.
Vì vậy claim cuối cùng phải là local surrogate opportunity, và cần offline
shared-parameter micro-update diagnostic.

## 3. Support-matched action geometry

Tại state `s`, đặt

\[
U(s)=\operatorname{TopK}(p_s)\cup\operatorname{TopK}(q_s),
\quad m_p=\sum_{a\in U}p_a,
\quad m_q=\sum_{a\in U}q_a.
\]

Action simplex mà differentiable OPD loss thực sự tối ưu là conditional simplex

\[
p_U(a)=p_a/m_p,\qquad q_U(a)=q_a/m_q,
\qquad a\in U.
\]

The tail is not silently discarded: `m_p`, `m_q`, `1-m_p`, `1-m_q`, and tail
coverage are logged. A deterministic coarse-graining map
`v -> v` for `v in U`, `v -> tail` otherwise would produce a valid
`U ∪ {tail}` distribution, but the current OPD optimizer has no trainable tail
logit/action. Including that category in the selector would therefore violate
the desired selector/optimizer geometry. Conditional projection onto `U` is the
correct executable surrogate; the tail is an explicit coverage diagnostic and a
sampling-censoring event.

The two uses of mass are intentionally different but compatible. The local
OPD update is a categorical update **within** the executable simplex, so its
coordinates are `p_U,q_U`. The sequential process asks for the probability that
the original student/teacher policies both keep a transition on an executable
action, so it uses

\[
p^{\mathrm{raw}}_a=m_p p_U(a),\qquad
q^{\mathrm{raw}}_a=m_q q_U(a),\qquad a\in U.
\]

The tail is therefore not renormalized into the trajectory process: leaving the
materialized union is an intentional killed transition. This is a product-space
factorization (local conditional policy geometry plus raw occupancy), not a claim
that the conditional and raw distributions are the same object.

Information-geometric data processing is a warning here, not a claim that Top-K
conditioning preserves the full policy geometry: Markov coarse-graining cannot
increase Fisher distinguishability ([Ay, Jost, Lê & Schwachhöfer](https://arxiv.org/abs/1207.6736);
see also the survey on [information geometry of Markov kernels](https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2023.1195562/full)).
We therefore report the tail explicitly and interpret the conditional simplex as
the geometry of the update actually executed by the masked OPD core, not as a
geometry-preserving approximation to the full vocabulary.

Define

\[
r_U(a)=\log q_U(a)-\log p_U(a),\qquad
\bar r_U=\mathbb E_{p_U}[r_U].
\]

The teacher-directed mirror path on the same simplex is

\[
p_{U,\eta}(a)=
\frac{p_U(a)\exp(\eta r_U(a))}
{\sum_{b\in U}p_U(b)\exp(\eta r_U(b))}.
\]

Its tangent is `dot p_U=p_U(r_U-bar r_U)`. The local projected-gradient gain is

\[
g_U(s)=
-\left.\frac{d}{d\eta}KL(p_{U,\eta}\|q_U)\right|_{0}
=\operatorname{Var}_{p_U}[r_U].
\]

The conditional categorical Fisher metric gives the same result:
`mu^T F^+ mu = r_U^T F r_U`. The code now stores conditional student and
teacher log-probabilities on `U`, so `g_U`, the frozen OPD advantages, and the
current differentiable candidate log-probabilities all use the same simplex.

Full-vocabulary `E_p[r]`, variance and common mass remain optional diagnostics
only (`cmt_full_vocab_diagnostics=false` by default); they no longer affect the
training score.

## 4. Bounded Top-K truncated coupling

For sequential accessibility, retain original mass on the materialized union:

\[
\widetilde K_U f(s)=
\sum_{a\in U}\min(p(a\mid s),q(a\mid s))f(sa).
\]

This is a sub-Markov kernel: missing student/teacher mass and actions outside
`U` are killed rather than renormalized away. With `Y~p`,

\[
\widehat{\widetilde K_U f}(s)=
\mathbf 1\{Y\in U\}
\min\left(1,\frac{q(Y\mid s)}{p(Y\mid s)}\right)f(sY)
\]

is exactly unbiased because

\[
\mathbb E_p[\widehat{\widetilde K_U f}]
=\sum_{a\in U}p(a)\min(1,q(a)/p(a))f(sa)
=\widetilde K_U f(s).
\]

The transition factor is always in `[0,1]`; no inverse-coverage factor appears.
The kernel's one-step survival is the **truncated raw common mass**
`\tilde\kappa_U=\sum_{a\in U}\min(p(a),q(a))`, not
`1-TV(p_U,q_U)`. The latter remains a conditional-support diagnostic only.

This remains a one-step accessibility statement, not global maximal trajectory
coupling or causal token credit. The acceptance rule itself is established in
speculative decoding ([Leviathan et al., ICML 2023](https://proceedings.mlr.press/v202/leviathan23a.html));
the use here is a Top-K truncated OPD successor operator.

The exactness statement assumes the rollout samples the scored student policy
`p`. If vLLM uses top-p truncation, the estimator remains bounded but is no
longer unbiased for this raw kernel (the code deliberately does not pretend to
know or correct the changed sampling law); the implementation warns instead of
silently applying an invalid correction.

## 5. Length-robust sequential value: local-baseline excess

Let `\widetilde K_U` denote the raw truncated common-mass sub-Markov kernel. For finite
response horizon, `gamma=1` is canonical; discounting is not needed to make the
quantity finite. The raw cumulative diagnostics are

\[
R_t=g_t+\gamma\widetilde K_U R_{t+1},\qquad
M_t=1+\gamma\widetilde K_U M_{t+1},\qquad
V_t=R_t/M_t.
\]

`R` is expected cumulative accessible local opportunity, `M` expected accessible
occupancy, and `V=R/M` a ratio-of-expectations diagnostic. This is the standard
reward/occupancy distinction; see [Dayan's successor representation](https://doi.org/10.1162/neco.1993.5.4.613)
and [Momennejad's review](https://pmc.ncbi.nlm.nih.gov/articles/PMC6941356/).

For a root state with local opportunity `g_t`, define the frozen-descendant
excess objective

\[
E_t = R_t-g_tM_t
    = \gamma \widetilde K_{U,t}
      \bigl(R_{t+1}-g_tM_{t+1}\bigr).
\]

The baseline is not a batch statistic: it is the current state's own
support-matched local opportunity. Hold `m_p,m_q` fixed while the local
conditional student distribution follows the mirror path. Then

\[
\dot{\widetilde K}_{U,t}f
=\sum_{a:m_pp_U(a)<m_qq_U(a)}
m_p p_U(a)(r_U(a)-\bar r_U)f(sa),
\]

so the sequential marginal derivative is

\[
D_t=\gamma\sum_{a:m_pp_U(a)<m_qq_U(a)}
m_p p_U(a)(r_U(a)-\bar r_U)
\bigl[R_{t+1}(sa)-g_tM_{t+1}(sa)\bigr].
\]

The production score is

\[
L_{CMT}(s_t)=g_t+\lambda D_t,
\qquad \lambda=1 \text{ canonical}.
\]

For `lambda=1`, this is not an arbitrary sum of two normalized signals. Define
the frozen-descendant local objective

\[
\mathcal J_t(\eta)=
 KL(p_{U,t}\|q_{U,t})-KL(p_{U,t,\eta}\|q_{U,t})
 +E_t(p_{U,t,\eta}),
\]

where the descendant returns/masses and the root baseline `g_t` are held fixed
inside `E_t`. Then

\[
\left.\frac{d}{d\eta}\mathcal J_t(\eta)\right|_{0}=g_t+D_t.
\]

Thus `lambda=1` is the derivative of one finite-horizon local categorical
surrogate; `lambda=0` is an explicitly named local-only ablation. This is not an
episodic average-reward claim.

`lambda=1` is canonical because both terms are derivatives in the same
conditional policy geometry; `lambda=0` is a named local-only ablation, not a
production tuning knob.

### Limiting cases

1. If `g_t=g` and `a_t=1`, then `R_t=gM_t`, hence `E_t=D_t=0` and `L_t=g`.
   Earlier tokens are not rewarded only because their suffix is longer.
2. If `g_t=g` and `a_t=a<1`, the same cancellation holds pathwise; survival
   changes occupancy but not relative opportunity.
3. If one future state has high `g` amid low-g states, the successor contrast is
   positive before that state when it exceeds the current local baseline. A
   teacher-deficit action receives extra emphasis only when its observed
   continuation carries excess opportunity.
4. Invalid padding/termination resets the scan. Adding valid low-opportunity
   tokens can matter only through a genuine excess relative to the current
   state, not through a batch-length statistic.

The old `R/M/V` remain logged for direct comparison to RAC and for offline
diagnostics. `H=R-g_tM_t` and the resulting marginal term are the production
sequential quantities.

## 6. One-rollout estimator and variance

For a sampled full-policy action `Y~p`, define

\[
\hat k_t=\mathbf 1\{Y_t\in U_t\}
\min\left(1,\frac{q_t(Y_t)}{p_t(Y_t)}\right).
\]

Recursively estimate `R` and `M` with the same one-rollout scan. The production
marginal term estimator is

\[
\widehat{D}_t=
\gamma\mathbf 1\{Y_t\in U_t,\,p_t(Y_t)<q_t(Y_t)\}
(r_{U,t}(Y_t)-\bar r_{U,t})
\bigl[\hat R_{t+1}-g_t\hat M_{t+1}\bigr].
\]

Because `R` and `M` enter linearly, this bounded one-rollout estimator is
unbiased for the **raw truncated** local excess derivative under frozen
descendants and `Y~p`. It is not an unbiased estimator of the old conditional
kernel; that is an intentional estimand change. It is also not unbiased for a
shared-parameter Transformer update.

Conditionally on the successor second moment, its second moment is

\[
\mathbb E[\widehat D_t^2\mid s]
=\sum_{a\in U}p(a)\mathbf1_{p(a)<q(a)}
(r_U(a)-\bar r_U)^2
\mathbb E\left[(R_{t+1}(sa)-g_tM_{t+1}(sa))^2\right].
\]

There is no `1/m_p` amplification and every transition factor is bounded by one;
future-return variance can still grow with a long or highly variable suffix. The
implementation logs `support_coverage`, `conditional_support_common_mass`,
`support_common_mass`, `transition_weight`, `R`, `M`, `H`, `successor_excess`,
and `sequential_gain`; it does not silently clip them.

### Old versus new estimator

The previous estimator targeted a different operator:

\[
K_U f=\sum_{a\in U}\min(p_U(a),q_U(a))f(sa),\qquad
\widehat K_U f=\frac{\mathbf1\{Y\in U\}}{m_p}
\min(1,q_U(Y)/p_U(Y))f(sY).
\]

It was unbiased for `K_U`, but its conditional second moment contains
`1/m_p`, and an individual transition can exceed one. The new estimator is
unbiased for `\widetilde K_U`, not `K_U`:

\[
\mathbb E[\widehat K_U^2 f^2\mid s]
=\frac1{m_p}\sum_{a\in U}p_U(a)a_U(a)^2f(sa)^2,
\]

whereas

\[
\mathbb E[\widehat{\widetilde K}_U^2 f^2\mid s]
=m_p\sum_{a\in U}p_U(a)\widetilde a(a)^2f(sa)^2,
\quad
\widetilde a(a)=\min\left(1,\frac{m_qq_U(a)}{m_pp_U(a)}\right).
\]

There is no universal variance ordering because the estimands differ. The new
construction removes the inverse-coverage amplification and bounds every
transition product; its price is that omitted tail mass is real killing, so the
sequential score can be smaller when `m_p` or `m_q` is small. That is an
interpretive change, not a hidden normalization.

### Rao–Blackwellization boundary

The local gain `g_U`, raw/conditional support common masses, and common-mass derivative

\[
\phi_U=\sum_{a:m_pp_U(a)<m_qq_U(a)}m_pp_U(a)(r_U(a)-\bar r_U)
\]

are summed exactly over `U`. This is the available action-space
Rao–Blackwellization and costs only `O(|U|)`. The future term contains
action-specific successor excess values. Summing them over `U` would require
evaluating those counterfactual successors or a learned Q/critic, both
disallowed. The directly available control variate is the root state's own
local gain:

\[
\dot{\widetilde K}_U\bigl(R-g_tM\bigr),
\]

No additional state-only baseline is introduced, because doing so would require
a new estimator/model or reintroduce an arbitrary batch statistic. The current
version keeps the exact local Rao–Blackwellization and exposes residual variance
for offline replicate diagnostics.

This is consistent with the discrete Rao–Blackwell literature: exact summation
over available categories reduces variance, while action-dependent future values
still require their values to be evaluated ([Rao-Blackwellized stochastic
gradients](https://proceedings.mlr.press/v97/liu19c/liu19c.pdf); [all-action policy
gradient estimators](https://optrl2019.github.io/assets/accepted_papers/72.pdf)).

## 7. Allocation under score noise

The final allocator remains

\[
\max_{w\ge0}\mathbb E_w[L]
\quad\text{s.t.}\quad KL(w\|u_N)\le\epsilon,
\]

whose solution is the Gibbs tilt with mean-one weights. `epsilon` is an explicit
information budget in nats relative to uniform, not a hidden temperature.

Because `L_hat` is noisy, Gibbs is not the optimizer for the unobserved true
`L`; exponentiation can amplify outliers. We do not claim otherwise. The current
principled default is to keep the allocator simple and falsifiable, while logging
score quantiles, effective sample size, coverage, and replicate variance. A
future uncertainty-aware extension would require an independently estimated
state/action uncertainty and a stated risk objective; it should not be added as
an arbitrary clipping rule.

## 8. Frozen-descendant claim boundary

CMT's sequential term is exactly the right derivative of a finite-horizon
**local categorical excess-opportunity surrogate**:

* the current `p_U` changes along the teacher-directed mirror path;
* successor values `R(sa), M(sa)` are held fixed;
* the common-mass transition is the raw Top-K truncated killed kernel
  `\widetilde K_U`;
* the root local baseline `g_t` is held fixed during the derivative.

It is not the derivative of the full neural training trajectory. Parameter sharing,
optimizer momentum, PPO clipping, changing hidden states, and occupancy changes
are omitted. There is no practical correction satisfying all constraints without
extra model evaluations or gradient estimators. The correct test is therefore an
offline shared-parameter micro-update experiment, not a stronger verbal claim.

## 9. Pseudocode

```text
for student on-policy rollout:
    score p and q on the same prefixes
    U <- literal union(student Top-K, teacher Top-K)
    mp <- sum_U p; mq <- sum_U q
    pU <- p / mp; qU <- q / mq
    rU <- log(qU) - log(pU)
    g  <- Var_pU(rU)
    # Y is still sampled from full student p; never score the vocabulary tail.
    inU <- (Y in U)
    if inU:
        rY_raw <- log(mq * qU[Y]) - log(mp * pU[Y])
        accept <- min(1, exp(rY_raw))
    else:
        rY_raw <- -infinity
        accept <- 0
    k_hat <- accept
    R, M <- reverse_scan(g, k_hat), reverse_scan(1, k_hat)
    excess_next <- R[next] - g * M[next]
    flux_hat <- inU * 1[rY_raw > 0] * (rU[Y] - E_pU[rU])
    L <- g + lambda * gamma * flux_hat * excess_next

    gather L globally
    w <- Gibbs allocation under KL(w || uniform) <= epsilon
    optimize existing OPD loss on the same conditional U support with w
```

## 10. Compute and numerical behavior

* No additional model forward or rollout is needed by the production CMT path.
* Selector cost is `O(B*T*|U|)`, with `|U| <= 2K`; the old full-vocabulary
  reduction is disabled by default.
* The conditional current log-probability gather uses only candidate logits and
  a support-masked `logsumexp`, so selector and loss share the same simplex.
* Full-vocabulary diagnostics remain opt-in and isolated from training score.
* `gamma=1`, `lambda=1`, `top_k=16`, and `epsilon=0.5` remain canonical defaults.
* FP32 log-sum-exp, finite checks, invalid-mask resets and no arbitrary ratio
  clipping are used. The production transition factor is bounded in `[0,1]`;
  `coverage_correction` is retained only as a compatibility diagnostic and is
  one on every in-support transition.

## 11. Required falsification matrix

### Geometry

Compare full-vocabulary diagnostic score, old union PGT, and support-matched PGT/
CMT against realized one-step OPD improvement. Stratify by `m_p`, `m_q`, teacher
tail mass, and union width.

### Sequential information

Run local-only (`lambda=0`), final CMT, suffix-shuffled `H`, transition-factor
shuffled `k_hat`, `a=1`, reversed suffix, and length-matched swapped suffix. A
non-shuffled CMT advantage must survive equal-budget comparison; otherwise the
sequential hypothesis is rejected.

### Length

Report rank/correlation of `L`, raw `R`, `H`, and sequential gain with remaining
length, normalized position, `M`, and support coverage. Constant-g synthetic
trajectories must yield flat `L` regardless of length.

### Variance

On a small frozen subset, repeat student rollouts with independent seeds. Compare
the empirical variance and bias of one-rollout `L_hat` with exact tiny-vocabulary
enumeration, the old inverse-coverage conditional estimator, and the new raw
truncated estimator. Report variance versus `m_p`, transition bounds, and
response length; compare estimands explicitly rather than treating raw and
conditional values as interchangeable.

### Actual learning value

For each score quantile and equal token budget:

1. freeze a checkpoint and selected prefixes;
2. apply an equal-norm small OPD micro-update in isolated model copies;
3. measure fixed-context conditional-KL reduction;
4. measure held-out teacher-transfer reduction;
5. optionally measure correctness on a small diagnostic benchmark.

The useful-learning target is not training-progress correlation alone.

## 12. Implementation status

Implemented in the sibling repository:

* `pgt_selector.py`: conditionalizes both student and teacher on literal union;
* `opd_core.py` and `trainer.py`: support-masked current log-probability gather,
  so the differentiable loss uses that same conditional simplex;
* `cmt_selector.py`: bounded raw Top-K truncated coupling, local-baseline excess
  derivative `H=R-gM`, conditional/raw common-mass diagnostics, and raw `R/M/V`
  diagnostics;
* `scoring.py`: full-vocabulary reductions are optional diagnostics and disabled
  by default for CMT training;
* TensorBoard, selector JSON, config, launcher, checkpoint/evaluation and smoke
  hooks expose support coverage, bounded transition weight, conditional/raw
  common mass, `H`, successor excess and sequential gain.

The original `TA-OPD-B200/` tree is unchanged. The old cumulative CMT report is
kept as historical v1; this document is authoritative for the current code.

## 13. Strongest defensible novelty claim

Do not claim novel Fisher geometry, maximal coupling, Gibbs allocation, or Bellman
recursion individually. The strongest defensible claim is narrower:

> CMT-OPD is a support-matched OPD allocation method that derives a local-baseline
> excess successor teachability term from a raw Top-K truncated common-mass
> kernel, while local learning geometry remains on the same conditional Top-K
> simplex optimized by OPD; its strict student rollout estimator is bounded and
> uses no inverse-coverage correction, critic, or counterfactual rollout.

The construction is new only at this full methodology/research-question level
until a targeted prior search proves otherwise. Its benefit over PGT/TA/RAC remains
an empirical hypothesis.

This claim deliberately excludes support projection and Bellman/TD recursion as
novel by themselves. Recent Bellman-distillation work already combines a reduced
action support with soft Bellman/TD targets ([AAAI 2026, *Language Model
Distillation: A Temporal Difference Imitation Learning Perspective*](https://ojs.aaai.org/index.php/AAAI/article/view/40750)).
The differentiating hypothesis here is the derivative of a raw truncated
common-mass kernel, with a local-baseline excess contrast and a bounded
strict-student estimator; its superiority is not established by the
implementation or by the cited prior work.

## 14. Verification status

After this refinement, the following must pass before a B200 run is trusted:

* support conditionalization sums to one for both `p_U` and `q_U`;
* finite-difference derivative on a finite support;
* raw truncated-kernel estimator identity on synthetic distributions;
* local-baseline excess derivative finite-difference and full-action enumeration;
* constant-g length invariance and padding boundary;
* allocation KL budget and TensorBoard/selector logging;
* full non-plotting test suite, Python compilation and shell syntax checks.

Current host verification: `106 passed, 1 skipped, 3 warnings, 7 subtests passed`
with the non-plotting suite; `compileall` and every `scripts/*.sh` syntax check
and `ruff check b200_experiment tests` also pass. The two optional collection groups requiring `pandas` and
`matplotlib` were not runnable in the available environment.

The available host still cannot run full B200 training because its NVIDIA driver
is older than the installed PyTorch CUDA build; missing `pandas`/`matplotlib`
remain optional test-environment limitations. No accuracy/speedup claim is made
until the offline falsification matrix and controlled B200 runs are executed.
