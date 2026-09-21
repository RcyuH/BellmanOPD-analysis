import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from analysis.batch_gt_comparison import (
    build_prompt_pgt_reference,
    normalized_effective_weights,
    top_gt_weights,
    train_prompt_batch,
    weighted_loss,
)
from analysis.run_batch_gt_comparison import _distributed_pair_teacher_distance
from analysis.summarize_batch_gt_comparison import summarize


def test_top_gt_weights_select_exact_batch_global_ceiling_with_stable_ties():
    scores = torch.tensor([[9.0, 8.0, 8.0, 0.0], [7.0, 6.0, 5.0, 4.0]])
    valid = torch.tensor([[True, True, True, False], [True, True, True, True]])
    weights = top_gt_weights(scores, valid, fraction=0.30)
    assert int(weights.sum()) == math.ceil(0.30 * int(valid.sum())) == 3
    # Stable flatten-index tie breaking keeps position 1 before position 2.
    assert weights.tolist() == [[1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]


def test_effective_weight_and_weighted_loss_match_selected_token_mean():
    valid = torch.tensor([[True, True, True, False]])
    weights = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    losses = torch.tensor([[2.0, 100.0, 4.0, 1000.0]])
    effective = normalized_effective_weights(weights, valid)
    assert torch.equal(effective, torch.tensor([[0.5, 0.0, 0.5, 0.0]]))
    assert weighted_loss(losses, weights, valid).item() == 3.0


class _TinyLM(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.values = torch.nn.Parameter(
            torch.as_tensor(logits, dtype=torch.float32).detach().clone()
        )

    def forward(self, input_ids, attention_mask, **kwargs):
        batch, width = input_ids.shape
        return SimpleNamespace(logits=self.values.reshape(1, 1, -1).expand(batch, width, -1))


def test_pgt_weights_can_enter_real_autograd_update_after_inference_scoring():
    student = _TinyLM(torch.linspace(-1.0, 1.0, 20))
    teacher = _TinyLM(torch.linspace(1.0, -1.0, 20))
    teacher.requires_grad_(False)
    ids = torch.tensor([[1, 2, 3]])
    mask = torch.ones_like(ids)
    reference, scores, _ = build_prompt_pgt_reference(student, teacher, ids, mask)
    weights = top_gt_weights(scores, mask, fraction=0.34)
    assert not weights.is_inference()
    before = student.values.detach().clone()
    result = train_prompt_batch(
        student,
        torch.optim.SGD(student.parameters(), lr=0.1),
        ids,
        mask,
        reference,
        weights,
        student_temperature=1.0,
        clip_low=0.2,
        clip_high=0.28,
        dual_clip=3.0,
        chunk_steps=8,
        max_grad_norm=10.0,
    )
    assert result["selected_tokens"] == 2
    assert not torch.equal(before, student.values)


class _SingleRankContext:
    rank = 0
    world_size = 1
    device = torch.device("cpu")

    @staticmethod
    def sum_float(value):
        return float(value)

    @staticmethod
    def sum_int(value):
        return int(value)

    @staticmethod
    def max_int(value):
        return int(value)


class _EvalModel:
    def __init__(self, value):
        self.value = float(value)
        self.training = True

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def __call__(self, input_ids, **kwargs):
        batch, width = input_ids.shape
        logits = torch.full((batch, width, 2), self.value)
        return SimpleNamespace(logits=logits)


class _EvalTokenizer:
    def __call__(self, prompt, **kwargs):
        length = len(prompt)
        return {
            "input_ids": torch.ones((1, length), dtype=torch.long),
            "attention_mask": torch.ones((1, length), dtype=torch.long),
        }


def test_distributed_pair_distance_reduces_exact_sums_and_shares_counts(monkeypatch):
    def fake_kl(student_logits, teacher_logits, valid_mask, **kwargs):
        count = valid_mask.sum()
        value = student_logits[0, 0, 0]
        return value.double() * count, count

    monkeypatch.setattr(
        "analysis.run_batch_gt_comparison.full_vocab_forward_kl_from_logits",
        fake_kl,
    )
    top, uniform = _distributed_pair_teacher_distance(
        _EvalModel(2.0),
        _EvalModel(3.0),
        _EvalModel(0.0),
        _EvalTokenizer(),
        ["a", "bb"],
        distributed=_SingleRankContext(),
        distance_kwargs={
            "max_prompt_tokens": 10,
            "student_temperature": 1.0,
            "teacher_temperature": 1.0,
            "vocab_chunk_positions": 4,
        },
    )
    assert top["value"] == 2.0
    assert top["sum"] == 6.0
    assert uniform["value"] == 3.0
    assert uniform["sum"] == 9.0
    assert top["num_tokens"] == uniform["num_tokens"] == 3
    assert top["num_problems"] == uniform["num_problems"] == 2
    assert top["maximum_prompt_tokens"] == uniform["maximum_prompt_tokens"] == 2


def test_summary_uses_paired_batch_improvement_advantage(tmp_path: Path):
    root = tmp_path / "run"
    steps = root / "steps"
    steps.mkdir(parents=True)
    advantages = [0.2, 0.1, 0.3]
    top_before = 2.0
    uniform_before = 2.0
    for step, advantage in enumerate(advantages):
        top_gain = 0.4 + advantage
        uniform_gain = 0.4
        payload = {
            "optimizer_step_before": step,
            "valid_tokens": 10,
            "top_gt_selected_tokens": 1,
            "top_gt_selected_fraction": 0.1,
            "top_gt": {
                "distance_before": top_before,
                "distance_after": top_before - top_gain,
                "distance_improvement": top_gain,
            },
            "uniform": {
                "distance_before": uniform_before,
                "distance_after": uniform_before - uniform_gain,
                "distance_improvement": uniform_gain,
            },
            "improvement_advantage_top_gt_minus_uniform": advantage,
            "top_gt_wins_this_batch": True,
            "student_weight_l2_distance_after_batch": float(step + 1),
        }
        (steps / f"step-{step:06d}.json").write_text(json.dumps(payload))
        top_before -= top_gain
        uniform_before -= uniform_gain
    result = summarize(root, tmp_path / "report")
    assert result["conclusion"] == "supports_top_gt"
    assert result["top_gt_batch_win_fraction"] == 1.0
    assert math.isclose(
        result["mean_improvement_advantage_top_gt_minus_uniform"], 0.2
    )
    assert (tmp_path / "report" / "batch_comparison.csv").is_file()
