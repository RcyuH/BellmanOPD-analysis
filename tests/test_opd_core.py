from __future__ import annotations

import unittest

import torch

from b200_experiment.opd_core import (
    OPD_LOSS_TOP_K,
    TopKOPDReference,
    build_topk_opd_reference,
    build_student_topk_opd_reference,
    gather_candidate_log_probs,
    topk_candidate_ppo_loss,
    topk_reference_from_logits,
    weighted_token_sums,
)


class TopKOPDCoreTests(unittest.TestCase):
    def test_all_distillation_methods_share_student_top16_loss_support(self):
        student_ids = torch.arange(OPD_LOSS_TOP_K).reshape(1, 1, -1)
        student = torch.log_softmax(torch.randn(1, 1, OPD_LOSS_TOP_K), dim=-1)
        teacher_on_student = torch.log_softmax(
            torch.randn(1, 1, OPD_LOSS_TOP_K), dim=-1
        )
        valid = torch.tensor([[True]])
        for method in ("opd", "ta", "cmt"):
            with self.subTest(method=method):
                reference = build_student_topk_opd_reference(
                    student_ids, student, teacher_on_student, valid
                )
                self.assertTrue(torch.equal(reference.candidate_ids, student_ids))
                self.assertEqual(reference.candidate_ids.shape[-1], 16)

    def test_union_width_is_rejected_by_policy_loss_builder(self):
        union_width = 2 * OPD_LOSS_TOP_K
        with self.assertRaisesRegex(ValueError, "exactly Student Top-16"):
            build_student_topk_opd_reference(
                torch.arange(union_width).reshape(1, 1, -1),
                torch.zeros(1, 1, union_width),
                torch.zeros(1, 1, union_width),
                torch.tensor([[True]]),
            )

    def test_inference_mode_scores_are_materialized_before_backward(self):
        with torch.inference_mode():
            candidate_ids = torch.tensor([[[0, 2], [1, 3]]], dtype=torch.long)
            student = torch.log_softmax(torch.randn(1, 2, 2), dim=-1)
            teacher = torch.log_softmax(torch.randn(1, 2, 2), dim=-1)
            valid = torch.ones(1, 2, dtype=torch.bool)
            # The core must remain safe even if a future caller constructs the
            # frozen reference before leaving its inference-mode scope.
            reference = build_topk_opd_reference(candidate_ids, student, teacher, valid)

        self.assertFalse(torch.is_inference(reference.candidate_ids))
        self.assertFalse(torch.is_inference(reference.old_student_log_probs))
        self.assertFalse(torch.is_inference(reference.teacher_log_probs))
        self.assertFalse(torch.is_inference(reference.student_weights))
        self.assertFalse(torch.is_inference(reference.advantages))

        logits = torch.randn(1, 2, 5, requires_grad=True)
        current = gather_candidate_log_probs(
            logits,
            reference.candidate_ids,
            temperature=1.0,
            chunk_steps=1,
        )
        loss = topk_candidate_ppo_loss(
            current, reference, clip_low=0.2, clip_high=0.28, dual_clip=3.0
        )
        loss.sum().backward()
        self.assertIsNotNone(logits.grad)

    def test_only_stu_student_p_matches_upstream_formula_with_k16(self):
        torch.manual_seed(7)
        student_logits = torch.randn(2, 3, 29)
        teacher_logits = torch.randn(2, 3, 29)
        valid = torch.tensor([[True, True, False], [True, True, True]])

        reference = topk_reference_from_logits(
            student_logits, teacher_logits, valid, top_k=16
        )

        student_all = torch.log_softmax(student_logits.float(), dim=-1)
        teacher_all = torch.log_softmax(teacher_logits.float(), dim=-1)
        ids = torch.topk(student_all, k=16, dim=-1).indices
        student_on_s = student_all.gather(-1, ids)
        teacher_on_s = teacher_all.gather(-1, ids)
        upstream_weights = torch.softmax(student_on_s, dim=-1)
        upstream_reward = -(student_on_s - teacher_on_s) * upstream_weights
        upstream_reward = upstream_reward * valid.unsqueeze(-1)

        self.assertEqual(reference.candidate_ids.shape[-1], 16)
        self.assertTrue(torch.equal(reference.candidate_ids, ids))
        self.assertTrue(
            torch.allclose(reference.advantages, upstream_reward, atol=1e-6)
        )

    def test_opd_gathers_teacher_only_on_student_topk_ids(self):
        student_logits = torch.tensor([[[9.0, 8.0, 1.0, 0.0]]])
        teacher_logits = torch.tensor([[[0.0, 1.0, 8.0, 9.0]]])
        valid = torch.tensor([[True]])
        reference = topk_reference_from_logits(
            student_logits, teacher_logits, valid, top_k=2
        )
        expected_ids = torch.tensor([[[0, 1]]])
        teacher_all = torch.log_softmax(teacher_logits.float(), dim=-1)
        self.assertTrue(torch.equal(reference.candidate_ids, expected_ids))
        self.assertTrue(
            torch.allclose(
                reference.teacher_log_probs,
                teacher_all.gather(-1, expected_ids),
            )
        )
        self.assertFalse(bool(reference.candidate_ids.eq(2).any()))
        self.assertFalse(bool(reference.candidate_ids.eq(3).any()))

    def test_policy_loss_uses_all_candidates_not_only_sampled_token(self):
        student = torch.log_softmax(torch.arange(16, dtype=torch.float32), dim=0)
        teacher = torch.log_softmax(
            torch.arange(15, -1, -1, dtype=torch.float32), dim=0
        )
        student = student.reshape(1, 1, 16)
        teacher = teacher.reshape(1, 1, 16)
        valid = torch.ones(1, 1, dtype=torch.bool)
        reference = build_topk_opd_reference(
            torch.arange(16).reshape(1, 1, 16), student, teacher, valid
        )
        current = student.clone().requires_grad_(True)

        loss = topk_candidate_ppo_loss(
            current, reference, clip_low=0.2, clip_high=0.28, dual_clip=3.0
        )
        loss.sum().backward()

        self.assertEqual(current.grad.shape[-1], 16)
        self.assertEqual(int(current.grad.ne(0).sum()), 16)
        expected = -reference.advantages.sum(dim=-1)
        self.assertTrue(torch.allclose(loss.detach(), expected, atol=1e-6))

    def test_trajectory_advantages_broadcast_over_response_tokens(self):
        """GRPO's one advantage per trajectory must cover every token."""
        current = torch.zeros(2, 3, 1, requires_grad=True)
        reference = TopKOPDReference(
            candidate_ids=torch.zeros(2, 3, 1, dtype=torch.long),
            old_student_log_probs=torch.zeros(2, 3, 1),
            teacher_log_probs=torch.zeros(2, 3, 1),
            student_weights=torch.ones(2, 3, 1),
            advantages=torch.tensor([1.0, -1.0]),
        )
        loss = topk_candidate_ppo_loss(
            current, reference, clip_low=0.2, clip_high=0.2, dual_clip=None
        )
        self.assertEqual(tuple(loss.shape), (2, 3))
        self.assertTrue(torch.allclose(loss[0], torch.full((3,), -1.0)))
        self.assertTrue(torch.allclose(loss[1], torch.full((3,), 1.0)))
        loss.sum().backward()
        self.assertIsNotNone(current.grad)

    def test_conditional_candidate_log_probs_match_executable_support(self):
        logits = torch.tensor([[[2.0, 1.0, 0.0, -1.0]]], requires_grad=True)
        ids = torch.tensor([[[0, 2, 0]]])
        support = torch.tensor([[[True, True, False]]])
        actual = gather_candidate_log_probs(
            logits,
            ids,
            temperature=1.0,
            chunk_steps=1,
            support_mask=support,
        )
        expected = torch.log_softmax(torch.tensor([2.0, 0.0]), dim=0)
        self.assertTrue(torch.allclose(actual[0, 0, :2], expected, atol=1e-6))
        self.assertTrue(torch.isfinite(actual).all())
        actual[0, 0, 0].backward()
        self.assertEqual(float(logits.grad[0, 0, 1]), 0.0)
        self.assertEqual(float(logits.grad[0, 0, 3]), 0.0)

    def test_all_methods_share_signal_and_only_position_weights_change(self):
        per_position = torch.tensor([[1.0, 2.0, 4.0], [8.0, 16.0, 32.0]])
        valid = torch.tensor([[True, True, False], [True, True, True]])
        allocations = {
            "opd": torch.ones_like(per_position),
            "ta": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            "rac": torch.tensor([[0.1, 0.2, 0.0], [0.4, 0.8, 1.0]]),
        }
        for weights in allocations.values():
            numerator, denominator = weighted_token_sums(per_position, weights, valid)
            expected_weights = weights * valid
            self.assertTrue(
                torch.allclose(
                    numerator / denominator,
                    (per_position * expected_weights).sum() / expected_weights.sum(),
                )
            )

    def test_global_token_mean_and_ddp_shards_match_single_process(self):
        losses = torch.tensor([[1.0, 3.0, 5.0], [7.0, 11.0, 13.0]])
        valid = torch.tensor([[True, True, False], [True, True, True]])
        cases = (
            torch.ones_like(losses),
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]),
            torch.tensor([[0.2, 0.4, 0.0], [0.3, 0.7, 1.0]]),
        )
        for weights in cases:
            full_num, full_den = weighted_token_sums(losses, weights, valid)
            shard0 = weighted_token_sums(losses[:1], weights[:1], valid[:1])
            shard1 = weighted_token_sums(losses[1:], weights[1:], valid[1:])
            ddp_value = (shard0[0] + shard1[0]) / (shard0[1] + shard1[1])
            self.assertTrue(torch.allclose(ddp_value, full_num / full_den))


if __name__ == "__main__":
    unittest.main()
