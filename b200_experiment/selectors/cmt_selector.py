from __future__ import annotations

import math

import torch

from .pgt_selector import PGTOutput
from .rac_selector import bellman_parallel_scan


@torch.no_grad()
def kl_constrained_allocation(
    values: torch.Tensor,
    epsilon: float,
    *,
    iterations: int = 64,
    tolerance: float = 1e-7,
) -> tuple[torch.Tensor, float, float]:
    """Solve max_w E_w[value] with KL(w || uniform) <= epsilon."""
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("KL allocation expects a non-empty one-dimensional tensor")
    if epsilon < 0.0:
        raise ValueError("KL allocation epsilon must be non-negative")
    scores = values.detach().float()
    if not torch.isfinite(scores).all():
        raise FloatingPointError("KL allocation received non-finite values")
    count = scores.numel()
    if count == 1 or epsilon <= tolerance or bool(scores.eq(scores[0]).all()):
        return torch.ones_like(scores), 0.0, 0.0

    centered = scores - scores.max()
    maxima = centered.eq(0).sum().item()
    maximum_kl = math.log(count / max(int(maxima), 1))
    target = min(float(epsilon), maximum_kl)
    if target >= maximum_kl - tolerance:
        probabilities = centered.eq(0).float() / float(maxima)
        return probabilities * count, math.inf, maximum_kl

    log_count = math.log(count)

    def distribution_and_kl(inverse_temperature: float):
        logits = centered * float(inverse_temperature)
        log_normalizer = torch.logsumexp(logits, dim=0)
        probabilities = torch.exp(logits - log_normalizer)
        kl = (
            float(inverse_temperature) * (probabilities * centered).sum()
            - log_normalizer
            + log_count
        )
        return probabilities, float(kl.item())

    low, high = 0.0, 1.0
    _, high_kl = distribution_and_kl(high)
    while high_kl < target and high < 1e12:
        high *= 2.0
        _, high_kl = distribution_and_kl(high)
    for _ in range(max(1, int(iterations))):
        midpoint = 0.5 * (low + high)
        _, midpoint_kl = distribution_and_kl(midpoint)
        if midpoint_kl < target:
            low = midpoint
        else:
            high = midpoint
    inverse_temperature = 0.5 * (low + high)
    probabilities, achieved_kl = distribution_and_kl(inverse_temperature)
    return probabilities * count, inverse_temperature, achieved_kl


class CMTSelector:
    """Support-matched, local-excess Coupled Marginal Teachability.

    The local scoring action space is the student/teacher Top-K union U. Student and
    teacher probabilities are conditionalized on U by ``PGTSelector`` for the
    local PGT/CMT geometry. Sequential accessibility instead uses the original
    probability mass on U, yielding the truncated sub-Markov kernel

        K_tilde f(s) = sum_{a in U} min(p(a|s), q(a|s)) f(sa).

    The strict full-policy student rollout therefore needs no inverse-coverage
    correction: for ``Y in U``, ``min(1, m_q*q_U(Y)/(m_p*p_U(Y)))`` is an
    unbiased one-sample transition factor; for ``Y`` outside ``U`` it is zero.
    The factor is bounded by one and requires no tail probability lookup.

    The sequential quantity is a local-baseline excess opportunity, not an
    episodic average-reward return.  For a root state with local opportunity
    ``g_t``, the frozen-descendant quantity is

        E_t = R_t - g_t M_t
            = gamma * K_tilde (R_{t+1} - g_t M_{t+1}),

    and the current-action derivative uses the successor contrast
    ``R_{t+1} - g_t M_{t+1}``.  This baseline is the current state's own
    support-matched local value, so constant-gain suffixes cancel exactly while
    the one-rollout derivative remains unbiased under frozen descendants.
    With ``successor_lambda=1``, ``g_t + D_t`` is the derivative of a single
    frozen-descendant surrogate consisting of local reverse-KL improvement plus
    the baseline-subtracted successor opportunity; lambda=0 is only a named
    local-only ablation.
    Descendant values are frozen in this categorical surrogate; no causal
    shared-neural-network claim is made. This union never becomes the OPD loss
    support: optimization is separately restricted to Student Top-16.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        successor_lambda: float = 1.0,
        ablation_arm: str = "canonical",
    ):
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("CMT gamma must be in [0, 1]")
        if successor_lambda < 0.0:
            raise ValueError("CMT successor lambda must be non-negative")
        arm = str(ablation_arm).strip().lower()
        aliases = {
            "canonical": "canonical",
            "cmt": "canonical",
            "g": "g",
            "g_x": "g_x",
            "gx": "g_x",
            "g_d": "g_d",
            "gd": "g_d",
        }
        if arm not in aliases:
            raise ValueError(
                "CMT ablation_arm must be one of canonical, g, g_x, or g_d; "
                f"got {ablation_arm!r}"
            )
        self.gamma = float(gamma)
        self.successor_lambda = float(successor_lambda)
        self.ablation_arm = aliases[arm]

    @torch.no_grad()
    def compute_scores(
        self,
        pgt_support: PGTOutput,
        sampled_token_ids: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> PGTOutput:
        """Compute CMT scores from local conditional and raw transition quantities.

        ``sampled_token_ids`` are drawn from the full student policy.  Local
        learning geometry remains conditional on U, while the successor kernel
        retains the original masses p(a), q(a) on U.  Tokens outside U are
        intentionally killed rather than reweighted into the conditional
        simplex.
        """
        expected_shape = valid_mask.shape
        if valid_mask.ndim != 2:
            raise ValueError("CMT valid_mask must have shape [batch, time]")
        if sampled_token_ids.shape != expected_shape:
            raise ValueError("CMT sampled token IDs must align with valid_mask")
        if pgt_support.candidate_ids.shape[:2] != expected_shape:
            raise ValueError("CMT support must align with valid_mask")

        valid = valid_mask.bool()
        candidate_ids = pgt_support.candidate_ids
        support_mask = pgt_support.support_mask.bool()
        student_cond = pgt_support.student_candidate_log_probs.detach().float()
        teacher_cond = pgt_support.teacher_candidate_log_probs.detach().float()
        support_p = torch.where(
            support_mask, student_cond.exp(), torch.zeros_like(student_cond)
        )
        support_q = torch.where(
            support_mask, teacher_cond.exp(), torch.zeros_like(teacher_cond)
        )
        support_r = torch.where(
            support_mask, teacher_cond - student_cond, torch.zeros_like(student_cond)
        )
        mean_r = (support_p * support_r).sum(dim=-1)
        g = pgt_support.diagnostics["gain"].detach().float().clamp_min(0.0)
        g = torch.where(valid, g, torch.zeros_like(g))

        student_mass = pgt_support.diagnostics["student_support_mass"].detach().float()
        teacher_mass = pgt_support.diagnostics["teacher_support_mass"].detach().float()
        tiny = torch.finfo(torch.float32).tiny
        student_mass = student_mass.clamp_min(tiny)
        teacher_mass = teacher_mass.clamp_min(tiny)
        log_mass_ratio = torch.log(teacher_mass) - torch.log(student_mass)
        original_p = support_p * student_mass.unsqueeze(-1)
        original_q = support_q * teacher_mass.unsqueeze(-1)
        original_r = support_r + log_mass_ratio.unsqueeze(-1)

        sampled_ids = sampled_token_ids.long().unsqueeze(-1)
        in_support_matrix = candidate_ids.eq(sampled_ids) & support_mask
        in_support = in_support_matrix.any(dim=-1) & valid
        # IDs are unique on support, so summing selects exactly one slot.
        sampled_cond_student = torch.where(
            in_support_matrix, student_cond, torch.zeros_like(student_cond)
        ).sum(dim=-1)
        sampled_cond_teacher = torch.where(
            in_support_matrix, teacher_cond, torch.zeros_like(teacher_cond)
        ).sum(dim=-1)
        sampled_cond_r = sampled_cond_teacher - sampled_cond_student
        sampled_r = sampled_cond_r + torch.where(
            in_support, log_mass_ratio, torch.zeros_like(log_mass_ratio)
        )
        acceptance = torch.where(
            in_support,
            torch.exp(sampled_r.clamp(max=0.0)),
            torch.zeros_like(sampled_r),
        )
        # The raw truncated kernel is estimated directly under Y~p.  There is
        # deliberately no 1 / P_student(U) importance factor.
        transition_weight = acceptance
        # Kept as a compatibility diagnostic for older JSON consumers.  It is
        # identically one on an in-support transition and is never applied as
        # an inverse-coverage correction.
        coverage_correction = torch.where(
            in_support, torch.ones_like(student_mass), torch.zeros_like(student_mass)
        )
        # The active branch of min(p(a), q(a)) is determined by original mass,
        # not by conditional p_U/q_U.  The tangent itself remains the local
        # conditional mirror tangent p_U * (r_U - E_pU[r_U]).
        teacher_deficit = in_support & sampled_r.gt(0.0)
        marginal_flux = torch.where(
            teacher_deficit,
            sampled_cond_r - mean_r,
            torch.zeros_like(sampled_cond_r),
        )

        # This is the part that can be Rao-Blackwellized with no successor
        # evaluations.  The action-conditioned future term cannot be summed
        # over U without evaluating those counterfactual successors.
        support_common_mass = torch.minimum(original_p, original_q).sum(dim=-1)
        conditional_support_common_mass = torch.minimum(support_p, support_q).sum(dim=-1)
        common_mass_derivative = torch.where(
            support_mask & original_p.lt(original_q),
            original_p * (support_r - mean_r.unsqueeze(-1)),
            torch.zeros_like(original_p),
        ).sum(dim=-1)

        # Raw cumulative return is retained only as a diagnostic.  The score
        # uses the local-baseline excess derivative below to remove
        # constant-opportunity length bias without episodic reward centering.
        cumulative_return, masses, cumulative_value = bellman_parallel_scan(
            g, transition_weight, valid, gamma=self.gamma
        )
        # E_t = R_t - g_t M_t is the expected suffix opportunity in excess of
        # the current state's own local opportunity.  It is deliberately not
        # implemented as reward centering by a batch average: that construction
        # is canonical for continuing average-reward problems, but can change
        # policy ordering in episodic problems with termination.
        local_excess = torch.where(
            valid,
            cumulative_return - g * masses,
            torch.zeros_like(cumulative_return),
        )
        successor_return = torch.zeros_like(cumulative_return)
        successor_mass = torch.zeros_like(masses)
        if cumulative_return.shape[1] > 1:
            successor_return[:, :-1] = torch.where(
                valid[:, 1:],
                cumulative_return[:, 1:],
                torch.zeros_like(cumulative_return[:, 1:]),
            )
            successor_mass[:, :-1] = torch.where(
                valid[:, 1:],
                masses[:, 1:],
                torch.zeros_like(masses[:, 1:]),
            )
        successor_excess = successor_return - g * successor_mass
        sequential_gain = (
            self.successor_lambda
            * self.gamma
            * marginal_flux
            * successor_excess
        )
        canonical_learning_value = torch.where(
            valid, g + sequential_gain, torch.zeros_like(g)
        )
        # Ablation arms deliberately reuse the exact CMT intermediates and the
        # same downstream KL/Gibbs allocator and weighted OPD objective.  The
        # default is byte-for-byte equivalent to the canonical score.  X is
        # the semantic successor excess (not R/M/V/H); D is the canonical
        # sequential marginal gain.
        if self.ablation_arm == "g":
            learning_value = g
            score_definition = "ablation_local_gain_g"
        elif self.ablation_arm == "g_x":
            learning_value = torch.where(
                valid, g + successor_excess, torch.zeros_like(g)
            )
            score_definition = "ablation_local_gain_plus_successor_excess_g_x"
        else:
            learning_value = canonical_learning_value
            score_definition = (
                "canonical_cmt_g_plus_sequential_gain"
                if self.ablation_arm == "g_d"
                else "support_matched_pgt_plus_local_baseline_excess_successor_derivative"
            )
        diagnostics = dict(pgt_support.diagnostics)
        diagnostics.update(
            gain=g,
            s_PGT=g,
            support_reverse_kl=(-mean_r).clamp_min(0.0),
            support_common_mass=torch.where(
                valid, support_common_mass, torch.zeros_like(support_common_mass)
            ),
            sampled_log_ratio=torch.where(
                in_support, sampled_r, torch.zeros_like(sampled_r)
            ),
            sampled_conditional_log_ratio=torch.where(
                in_support, sampled_cond_r, torch.zeros_like(sampled_cond_r)
            ),
            alignment=acceptance,
            transition_weight=transition_weight,
            support_coverage=torch.where(
                valid, student_mass, torch.zeros_like(student_mass)
            ),
            coverage_correction=coverage_correction,
            teacher_support_mass=torch.where(
                valid, teacher_mass, torch.zeros_like(teacher_mass)
            ),
            conditional_support_common_mass=torch.where(
                valid,
                conditional_support_common_mass,
                torch.zeros_like(conditional_support_common_mass),
            ),
            teacher_deficit=teacher_deficit.float(),
            marginal_flux=marginal_flux,
            common_mass_derivative=common_mass_derivative,
            R=cumulative_return,
            M=masses,
            V=cumulative_value,
            H=local_excess,
            successor_excess=successor_excess,
            # Compatibility alias retained for older selector JSON readers;
            # this is no longer raw successor R, but the baseline-subtracted
            # successor excess used by the production derivative.
            successor_R=successor_excess,
            sequential_gain=sequential_gain,
            learning_value=learning_value,
            s_CMT=learning_value,
            score_definition=score_definition,
            ablation_arm=self.ablation_arm,
            ablation_score=learning_value,
            transition_definition=(
                "truncated_original_union_common_mass_without_inverse_coverage"
            ),
            gamma=self.gamma,
            successor_lambda=self.successor_lambda,
        )
        for value in (
            g,
            mean_r,
            support_p,
            original_p,
            original_q,
            original_r,
            acceptance,
            transition_weight,
            marginal_flux,
            cumulative_return,
            masses,
            cumulative_value,
            local_excess,
            successor_return,
            successor_mass,
            successor_excess,
            sequential_gain,
            learning_value,
        ):
            if value.requires_grad or value.grad_fn is not None:
                raise AssertionError("CMT statistics must be detached")
        return PGTOutput(
            learning_value,
            diagnostics,
            pgt_support.candidate_ids,
            pgt_support.student_candidate_log_probs,
            pgt_support.teacher_candidate_log_probs,
            pgt_support.support_mask,
        )
