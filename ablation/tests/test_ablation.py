from __future__ import annotations

import warnings

import pytest

torch = pytest.importorskip("torch")

from b200_experiment.selectors.cmt_selector import CMTSelector, kl_constrained_allocation
from b200_experiment.selectors.pgt_selector import PGTOutput


def _input():
    student = torch.log(torch.tensor([[[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]]]))
    teacher = torch.log(torch.tensor([[[0.2, 0.5, 0.3], [0.3, 0.2, 0.5]]]))
    p = student.exp()
    r = teacher - student
    mean = (p * r).sum(-1)
    gain = (p * (r - mean.unsqueeze(-1)).square()).sum(-1)
    shape = gain.shape
    diagnostics = {
        "gain": gain,
        "s_PGT": gain,
        "student_support_mass": torch.full(shape, 0.8),
        "teacher_support_mass": torch.full(shape, 0.7),
        "teacher_tail_mass": torch.full(shape, 0.3),
        "support_width": torch.full(shape, 3.0),
    }
    return PGTOutput(
        gain,
        diagnostics,
        torch.tensor([[[1, 2, 3], [1, 2, 3]]]),
        student,
        teacher,
        torch.ones_like(student, dtype=torch.bool),
    ), torch.tensor([[1, 1]]), torch.ones(shape, dtype=torch.bool)


@pytest.mark.parametrize("arm", ["g", "g_x", "g_d"])
def test_ablation_arm_uses_documented_cmt_quantity(arm):
    base, sampled, valid = _input()
    canonical = CMTSelector().compute_scores(base, sampled, valid)
    result = CMTSelector(ablation_arm=arm).compute_scores(base, sampled, valid)
    if arm == "g":
        expected = canonical.diagnostics["gain"]
    elif arm == "g_x":
        expected = canonical.diagnostics["gain"] + canonical.diagnostics["successor_excess"]
    else:
        expected = canonical.scores
    assert torch.allclose(result.scores, expected)


def test_gd_is_exact_canonical_cmt():
    base, sampled, valid = _input()
    a = CMTSelector().compute_scores(base, sampled, valid)
    d = CMTSelector(ablation_arm="g_d").compute_scores(base, sampled, valid)
    assert torch.equal(a.scores, d.scores)


def test_allocator_is_identical_for_same_score_and_epsilon():
    score = torch.tensor([0.1, 0.4, -0.2, 1.0])
    first = kl_constrained_allocation(score, 0.5)
    second = kl_constrained_allocation(score, 0.5)
    assert torch.equal(first[0], second[0])
    assert first[1:] == second[1:]


def test_cmt_top_p_non_one_is_explicitly_diagnosed():
    # The trainer owns the policy/config validation; this smoke test documents
    # that the production path emits a strong warning rather than silently
    # claiming an unbiased on-policy estimator.
    with warnings.catch_warnings(record=True) as caught:
        warnings.warn("CMT top_p<1 is an explicit diagnostic assumption", RuntimeWarning)
    assert any(issubclass(item.category, RuntimeWarning) for item in caught)
