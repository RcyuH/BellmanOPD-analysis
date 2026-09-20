from __future__ import annotations

import unittest

import torch

from b200_experiment.scoring import RolloutBatch
from b200_experiment.trainer import _grpo_group_advantages


class _Tokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return {1: "42", 2: "0", 3: "7", 4: "8"}[int(ids[0])] if ids else ""


class GRPOTests(unittest.TestCase):
    def _rollout(self):
        response_ids = torch.tensor([[1], [2], [3], [4]], dtype=torch.long)
        valid = torch.ones_like(response_ids, dtype=torch.bool)
        return RolloutBatch(
            input_ids=torch.tensor([[9, 1], [9, 2], [9, 3], [9, 4]]),
            attention_mask=torch.ones(4, 2, dtype=torch.long),
            response_ids=response_ids,
            valid_mask=valid,
            rollout_log_probs=torch.zeros(4, 1),
            prompt_width=1,
        )

    def test_group_normalization_is_relative_and_bounded(self):
        rollout = self._rollout()
        records = [
            {"problem": "p", "answer": "42"},
            {"problem": "p", "answer": "0"},
            {"problem": "q", "answer": "7"},
            {"problem": "q", "answer": "8"},
        ]
        _, advantages, stats = _grpo_group_advantages(
            rollout,
            _Tokenizer(),
            records,
            torch.ones(4, dtype=torch.bool),
            [0, 1, 0, 1],
            2,
            torch.device("cpu"),
        )
        self.assertTrue(torch.allclose(advantages, torch.zeros_like(advantages)))
        self.assertEqual(stats["reward_mean"], 1.0)

    def test_group_with_one_success_has_plus_minus_one_advantage(self):
        rollout = self._rollout()
        records = [
            {"problem": "p", "answer": "42"},
            {"problem": "p", "answer": "0"},
            {"problem": "q", "answer": "7"},
            {"problem": "q", "answer": "0"},
        ]
        _, advantages, _ = _grpo_group_advantages(
            rollout,
            _Tokenizer(),
            records,
            torch.ones(4, dtype=torch.bool),
            [0, 1, 0, 1],
            2,
            torch.device("cpu"),
        )
        self.assertTrue(torch.allclose(advantages, torch.tensor([1.0, -1.0, 1.0, -1.0])))

    def test_dapo_solution_field_is_used_for_math_reward(self):
        rollout = self._rollout()
        records = [
            {"prompt": "p", "solution": "42"},
            {"prompt": "p", "solution": "0"},
            {"prompt": "q", "solution": "7"},
            {"prompt": "q", "solution": "0"},
        ]
        _, advantages, stats = _grpo_group_advantages(
            rollout,
            _Tokenizer(),
            records,
            torch.ones(4, dtype=torch.bool),
            [0, 1, 0, 1],
            2,
            torch.device("cpu"),
            answer_key="answer",
            benchmark="Competition-MATH",
        )
        self.assertTrue(torch.allclose(advantages, torch.tensor([1.0, -1.0, 1.0, -1.0])))
        self.assertEqual(stats["reward_mean"], 0.5)

    def test_dapo_nested_reward_model_ground_truth_is_fallback(self):
        rollout = self._rollout()
        records = [
            {"prompt": "p", "reward_model": {"ground_truth": "42"}},
            {"prompt": "p", "reward_model": {"ground_truth": "0"}},
            {"prompt": "q", "reward_model": {"ground_truth": "7"}},
            {"prompt": "q", "reward_model": {"ground_truth": "0"}},
        ]
        _, advantages, _ = _grpo_group_advantages(
            rollout,
            _Tokenizer(),
            records,
            torch.ones(4, dtype=torch.bool),
            [0, 1, 0, 1],
            2,
            torch.device("cpu"),
        )
        self.assertTrue(torch.allclose(advantages, torch.tensor([1.0, -1.0, 1.0, -1.0])))


if __name__ == "__main__":
    unittest.main()
