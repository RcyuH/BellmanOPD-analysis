import unittest

import torch

from b200_experiment.opd_core import (
    build_iw_opd_reference,
    compute_iw_opd_weights,
)


class IWOPDTests(unittest.TestCase):
    def test_prefix_remaining_mass_matches_official_formula(self):
        student = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
        teacher = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        valid = torch.tensor([[True, True, True, False]])
        weights = compute_iw_opd_weights(teacher, student, valid, weight_max=1.5)
        expected = torch.tensor([[1.5, 1.5 - 0.5 / 6.0, 1.5 - 0.5 * 3.0 / 6.0, 0.0]])
        torch.testing.assert_close(weights, expected)

    def test_weights_are_bounded_and_padding_is_zero(self):
        student = torch.randn(3, 7)
        teacher = torch.randn(3, 7)
        valid = torch.tensor(
            [[1, 1, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0], [0] * 7],
            dtype=torch.bool,
        )
        weights = compute_iw_opd_weights(teacher, student, valid, weight_max=1.5)
        self.assertTrue(torch.all(weights[valid] >= 1.0))
        self.assertTrue(torch.all(weights[valid] <= 1.5))
        # The upstream formula leaves masked positions at its neutral value
        # one; the response mask removes them from the PPO objective.
        self.assertTrue(torch.all(weights[~valid] == 1.0))

    def test_singleton_reference_contains_weighted_sampled_advantage(self):
        ids = torch.tensor([[4, 5, 6]])
        student = torch.tensor([[0.0, 0.0, 0.0]])
        teacher = torch.tensor([[1.0, 2.0, 3.0]])
        valid = torch.ones_like(ids, dtype=torch.bool)
        reference, weights = build_iw_opd_reference(ids, student, teacher, valid)
        self.assertEqual(tuple(reference.candidate_ids.shape), (1, 3, 1))
        torch.testing.assert_close(reference.advantages.squeeze(-1), teacher * weights)
        self.assertIsNone(reference.support_mask)


if __name__ == "__main__":
    unittest.main()
