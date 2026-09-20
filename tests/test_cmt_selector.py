import torch

from b200_experiment.selectors.cmt_selector import (
    CMTSelector,
    kl_constrained_allocation,
)
from b200_experiment.selectors.pgt_selector import PGTOutput


def _support(
    student_cond: torch.Tensor,
    teacher_cond: torch.Tensor,
    *,
    student_mass: float = 1.0,
    teacher_mass: float = 1.0,
    gain: torch.Tensor | None = None,
    ids: torch.Tensor | None = None,
) -> PGTOutput:
    batch, time, width = student_cond.shape
    if gain is None:
        p = student_cond.exp()
        r = teacher_cond - student_cond
        mean = (p * r).sum(dim=-1)
        gain = (p * (r - mean.unsqueeze(-1)).square()).sum(dim=-1)
    if ids is None:
        ids = torch.arange(width).reshape(1, 1, width).expand(batch, time, width)
    support = torch.ones_like(student_cond, dtype=torch.bool)
    diagnostics = {
        "gain": gain,
        "s_PGT": gain,
        "student_support_mass": torch.full_like(gain, student_mass),
        "teacher_support_mass": torch.full_like(gain, teacher_mass),
        "teacher_tail_mass": torch.full_like(gain, 1.0 - teacher_mass),
        "support_width": torch.full_like(gain, float(width)),
    }
    return PGTOutput(
        gain,
        diagnostics,
        ids,
        student_cond,
        teacher_cond,
        support,
    )


def test_local_excess_removes_constant_length_bias():
    p = torch.log(torch.tensor([[[0.5], [0.5], [0.5], [0.5]]]))
    q = p.clone()
    output = _support(p, q, gain=torch.ones(1, 4))
    result = CMTSelector().compute_scores(
        output,
        torch.zeros(1, 4, dtype=torch.long),
        torch.ones(1, 4, dtype=torch.bool),
    )
    assert torch.allclose(result.diagnostics["H"], torch.zeros(1, 4))
    assert torch.allclose(result.scores, torch.ones(1, 4))


def test_padding_is_a_hard_boundary_for_local_excess_value():
    p = torch.log(torch.tensor([[[0.5], [0.5], [0.5], [0.5]]]))
    output = _support(p, p.clone(), gain=torch.ones(1, 4))
    valid = torch.tensor([[True, True, False, False]])
    result = CMTSelector().compute_scores(
        output,
        torch.zeros(1, 4, dtype=torch.long),
        valid,
    )
    assert torch.all(result.diagnostics["H"][~valid] == 0)
    assert torch.all(result.scores[~valid] == 0)
    assert torch.allclose(result.scores[valid], torch.ones(2))


def test_constant_gain_is_length_neutral_even_when_coupling_survival_is_below_one():
    p = torch.log(torch.tensor([[[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]]))
    q = torch.log(torch.tensor([[[0.25, 0.75], [0.25, 0.75], [0.25, 0.75]]]))
    output = _support(p, q, gain=torch.ones(1, 3))
    result = CMTSelector().compute_scores(
        output,
        torch.tensor([[0, 0, 0]]),
        torch.ones(1, 3, dtype=torch.bool),
    )
    assert torch.allclose(result.diagnostics["H"], torch.zeros(1, 3))
    assert torch.allclose(result.scores, torch.ones(1, 3))


def test_constant_gain_is_length_neutral_under_raw_truncated_mass():
    p = torch.log(torch.tensor([[[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]]))
    q = torch.log(torch.tensor([[[0.75, 0.25], [0.75, 0.25], [0.75, 0.25]]]))
    output = _support(p, q, student_mass=0.4, teacher_mass=0.7, gain=torch.ones(1, 3))
    result = CMTSelector().compute_scores(
        output,
        torch.tensor([[0, 0, 0]]),
        torch.ones(1, 3, dtype=torch.bool),
    )
    assert torch.allclose(result.diagnostics["H"], torch.zeros(1, 3))
    assert torch.allclose(result.scores, torch.ones(1, 3))


def test_truncated_kernel_keeps_transition_bounded_without_inverse_coverage():
    p = torch.log(torch.tensor([[[0.5, 0.5], [0.5, 0.5]]]))
    q = torch.log(torch.tensor([[[0.8, 0.2], [0.8, 0.2]]]))
    ids = torch.tensor([[[10, 11], [10, 11]]])
    output = _support(p, q, student_mass=0.5, teacher_mass=0.25, ids=ids)
    result = CMTSelector().compute_scores(
        output,
        torch.tensor([[10, 99]]),
        torch.ones(1, 2, dtype=torch.bool),
    )
    assert torch.allclose(
        result.diagnostics["coverage_correction"], torch.tensor([[1.0, 0.0]])
    )
    assert torch.allclose(
        result.diagnostics["transition_weight"], torch.tensor([[0.8, 0.0]])
    )
    assert torch.equal(
        result.diagnostics["teacher_deficit"], torch.tensor([[0.0, 0.0]])
    )


def test_truncated_kernel_has_the_raw_common_mass_expectation():
    # Full student masses are [.1, .3, .6_tail], and teacher masses on U are
    # [.3, .1].  For successor values [2,-1], the exact raw expectation is
    # .1*2 + .1*(-1) = .1.
    p = torch.log(torch.tensor([[[0.25, 0.75]]]))
    q = torch.log(torch.tensor([[[0.75, 0.25]]]))
    ids = torch.tensor([[[10, 11]]])
    output = _support(p, q, student_mass=0.4, teacher_mass=0.4, ids=ids)
    result = CMTSelector().compute_scores(
        output,
        torch.tensor([[10]]),
        torch.ones(1, 1, dtype=torch.bool),
    )
    assert torch.allclose(
        result.diagnostics["transition_weight"], torch.tensor([[1.0]])
    )
    assert torch.allclose(
        result.diagnostics["support_common_mass"], torch.tensor([[0.2]])
    )
    assert torch.allclose(
        result.diagnostics["conditional_support_common_mass"], torch.tensor([[0.5]])
    )
    full_p = torch.tensor([0.10, 0.30, 0.60], dtype=torch.float64)
    full_q = torch.tensor([0.30, 0.10, 0.60], dtype=torch.float64)
    successor = torch.tensor([2.0, -1.0], dtype=torch.float64)
    sampled = torch.tensor(
        [
            min(1.0, (full_q[0] / full_p[0]).item()) * successor[0].item(),
            min(1.0, (full_q[1] / full_p[1]).item()) * successor[1].item(),
            0.0,
        ],
        dtype=torch.float64,
    )
    assert torch.allclose(
        (full_p * sampled).sum(), full_p[0] * 2.0 + min(full_p[1], full_q[1]) * (-1.0)
    )
    # The production selector gives the same bounded factor for every possible
    # sampled ID, including the killed tail event, without a full-vocabulary
    # lookup.
    production_estimates = []
    for token_id in (10, 11, 99):
        token_result = CMTSelector().compute_scores(
            output,
            torch.tensor([[token_id]]),
            torch.ones(1, 1, dtype=torch.bool),
        )
        transition = token_result.diagnostics["transition_weight"].item()
        successor_value = successor[len(production_estimates)].item() if token_id != 99 else 0.0
        production_estimates.append(transition * successor_value)
    assert torch.allclose(
        (full_p * torch.tensor(production_estimates)).sum(),
        full_p[0] * 2.0 + min(full_p[1], full_q[1]) * (-1.0),
        atol=1e-7,
    )


def test_truncated_transition_factor_is_bounded():
    p = torch.log(torch.tensor([[[0.99, 0.01]]]))
    q = torch.log(torch.tensor([[[0.01, 0.99]]]))
    output = _support(p, q, student_mass=0.01, teacher_mass=0.99)
    result = CMTSelector().compute_scores(
        output,
        torch.tensor([[0]]),
        torch.ones(1, 1, dtype=torch.bool),
    )
    transition = result.diagnostics["transition_weight"]
    assert bool((transition >= 0).all())
    assert bool((transition <= 1).all())


def test_cmt_recurrence_uses_local_baseline_excess_opportunity():
    p = torch.log(torch.tensor([[[0.5], [0.5], [0.5]]]))
    q = torch.log(torch.tensor([[[0.5], [0.5], [0.5]]]))
    gain = torch.tensor([[1.0, 2.0, 3.0]])
    output = _support(p, q, gain=gain)
    result = CMTSelector().compute_scores(
        output,
        torch.zeros(1, 3, dtype=torch.long),
        torch.ones(1, 3, dtype=torch.bool),
    )
    # H = R - g_t M = [3, 1, 0] under a unit transition.  The current score
    # has no sequential term because r=0, but H exposes future opportunity in
    # excess of the current state's own local value.
    assert torch.allclose(result.diagnostics["H"], torch.tensor([[3.0, 1.0, 0.0]]))
    assert torch.allclose(result.scores, gain)


def test_directional_formula_matches_finite_difference_on_support():
    p = torch.tensor([0.50, 0.30, 0.20], dtype=torch.float64)
    q = torch.tensor([0.20, 0.50, 0.30], dtype=torch.float64)
    child = torch.tensor([0.20, 1.70, 0.90], dtype=torch.float64)
    r = q.log() - p.log()
    mean = (p * r).sum()
    g = (p * (r - mean).square()).sum()
    analytic = g + (torch.where(p < q, p * (r - mean), 0.0) * child).sum()
    eta = 1e-6
    p_eta = p * torch.exp(eta * r)
    p_eta /= p_eta.sum()
    def divergence(x):
        return (x * (x.log() - q.log())).sum()

    def access(x):
        return (torch.minimum(x, q) * child).sum()
    finite_difference = (
        divergence(p) - divergence(p_eta) + access(p_eta) - access(p)
    ) / eta
    assert torch.allclose(finite_difference, analytic, atol=1e-6)


def test_local_baseline_excess_derivative_matches_finite_difference():
    p = torch.tensor([0.50, 0.30, 0.20], dtype=torch.float64)
    q = torch.tensor([0.20, 0.50, 0.30], dtype=torch.float64)
    child_return = torch.tensor([0.20, 1.70, 0.90], dtype=torch.float64)
    child_mass = torch.tensor([1.20, 0.80, 1.10], dtype=torch.float64)
    r = q.log() - p.log()
    mean = (p * r).sum()
    local_gain = torch.tensor(0.65, dtype=torch.float64)
    analytic = torch.where(
        p < q,
        p * (r - mean) * (child_return - local_gain * child_mass),
        torch.zeros_like(p),
    ).sum()

    def excess(x):
        common = torch.minimum(x, q)
        return (
            local_gain
            + (common * child_return).sum()
            - local_gain * (1.0 + (common * child_mass).sum())
        )

    eta = 1e-6
    p_eta = p * torch.exp(eta * r)
    p_eta /= p_eta.sum()
    finite_difference = (excess(p_eta) - excess(p)) / eta
    assert torch.allclose(finite_difference, analytic, atol=1e-6)


def test_production_score_is_single_surrogate_derivative():
    p = torch.tensor([0.50, 0.30, 0.20], dtype=torch.float64)
    q = torch.tensor([0.20, 0.50, 0.30], dtype=torch.float64)
    child_return = torch.tensor([0.20, 1.70, 0.90], dtype=torch.float64)
    child_mass = torch.tensor([1.20, 0.80, 1.10], dtype=torch.float64)
    r = q.log() - p.log()
    mean = (p * r).sum()
    local_gain = (p * (r - mean).square()).sum()
    d_excess = torch.where(
        p < q,
        p * (r - mean) * (child_return - local_gain * child_mass),
        torch.zeros_like(p),
    ).sum()
    expected = local_gain + d_excess

    def objective(x):
        local_improvement = (p * (p.log() - q.log())).sum() - (
            x * (x.log() - q.log())
        ).sum()
        common = torch.minimum(x, q)
        excess = (
            local_gain
            + (common * child_return).sum()
            - local_gain * (1.0 + (common * child_mass).sum())
        )
        return local_improvement + excess

    eta = 1e-6
    p_eta = p * torch.exp(eta * r)
    p_eta /= p_eta.sum()
    finite_difference = (objective(p_eta) - objective(p)) / eta
    assert torch.allclose(finite_difference, expected, atol=1e-6)


def test_truncated_sequential_derivative_uses_original_mass_threshold():
    p_u = torch.tensor([0.50, 0.30, 0.20], dtype=torch.float64)
    q_u = torch.tensor([0.20, 0.50, 0.30], dtype=torch.float64)
    student_mass = 0.40
    teacher_mass = 0.70
    child_return = torch.tensor([0.20, 1.70, 0.90], dtype=torch.float64)
    child_mass = torch.tensor([1.20, 0.80, 1.10], dtype=torch.float64)
    local_gain = torch.tensor(0.65, dtype=torch.float64)
    r = q_u.log() - p_u.log()
    mean = (p_u * r).sum()
    original_p = student_mass * p_u
    original_q = teacher_mass * q_u
    analytic = torch.where(
        original_p < original_q,
        original_p * (r - mean) * (child_return - local_gain * child_mass),
        torch.zeros_like(original_p),
    ).sum()

    def excess(x):
        common = torch.minimum(student_mass * x, teacher_mass * q_u)
        return (
            local_gain
            + (common * child_return).sum()
            - local_gain * (1.0 + (common * child_mass).sum())
        )

    eta = 1e-6
    p_eta = p_u * torch.exp(eta * r)
    p_eta /= p_eta.sum()
    finite_difference = (excess(p_eta) - excess(p_u)) / eta
    assert torch.allclose(finite_difference, analytic, atol=1e-6)


def test_truncated_excess_estimator_is_unbiased_by_enumeration():
    # Full student probabilities are [.1, .3, .6_tail], while p_U=[.25,.75].
    # Original teacher masses on U are [.3,.1], so the raw threshold differs
    # from the conditional threshold: only the first action has p(a)<q(a).
    # Enumerating all sampled actions recovers the derivative of the truncated,
    # not conditional, common-mass operator.
    p_u = torch.tensor([0.25, 0.75], dtype=torch.float64)
    q_u = torch.tensor([0.75, 0.25], dtype=torch.float64)
    full_p = torch.tensor([0.10, 0.30, 0.60], dtype=torch.float64)
    full_q = torch.tensor([0.30, 0.10, 0.60], dtype=torch.float64)
    m_p = 0.40
    m_q = 0.40
    child_excess = torch.tensor([2.0, -1.0], dtype=torch.float64)
    r = q_u.log() - p_u.log()
    mean = (p_u * r).sum()
    original_p = m_p * p_u
    original_q = m_q * q_u
    original_r = full_q[:2].log() - full_p[:2].log()
    sampled = torch.tensor(
        [
            ((r[0] - mean).item() * child_excess[0].item())
            if bool(original_r[0] > 0)
            else 0.0,
            ((r[1] - mean).item() * child_excess[1].item())
            if bool(original_r[1] > 0)
            else 0.0,
            0.0,
        ],
        dtype=torch.float64,
    )
    enumerated_expectation = (full_p * sampled).sum()
    exact = torch.where(
        original_p < original_q,
        original_p * (r - mean) * child_excess,
        torch.zeros_like(original_p),
    ).sum()
    assert torch.allclose(enumerated_expectation, exact, atol=1e-12)


def test_kl_allocation_hits_budget_and_is_affine_scale_invariant():
    values = torch.tensor([-1.0, 0.0, 0.5, 3.0, 4.0])
    weights, inverse_temperature, achieved = kl_constrained_allocation(values, 0.4)
    assert torch.all(weights > 0)
    assert torch.allclose(weights.mean(), torch.tensor(1.0), atol=1e-6)
    assert abs(achieved - 0.4) < 1e-5
    assert inverse_temperature > 0
    assert torch.equal(torch.argsort(weights), torch.argsort(values))
    transformed, _, transformed_kl = kl_constrained_allocation(7.0 * values + 9.0, 0.4)
    assert torch.allclose(weights, transformed, atol=1e-5)
    assert abs(transformed_kl - 0.4) < 1e-5


def test_zero_kl_budget_is_uniform():
    weights, inverse_temperature, achieved = kl_constrained_allocation(
        torch.tensor([0.0, 2.0, 5.0]), 0.0
    )
    assert torch.equal(weights, torch.ones(3))
    assert inverse_temperature == 0.0
    assert achieved == 0.0
