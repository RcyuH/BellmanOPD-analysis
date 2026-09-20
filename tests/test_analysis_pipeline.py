from __future__ import annotations

import json
import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

from analysis.metrics import conditional_d_comparison, correlations, quantile_curve
from analysis.pipeline import AnalysisLogger
from analysis.run_analysis import _legacy_score_samples
from b200_experiment.scoring import RolloutBatch
from b200_experiment.distributed import DistributedContext
from b200_experiment.opd_core import topk_reference_from_logits
from b200_experiment.trainer import _opd_train_step


class _SingleRank:
    rank = 0
    is_main = True

    def barrier(self):
        pass

    def all_gather_objects(self, value):
        return [value]

    def broadcast_object(self, value):
        return value


class _ToyStudent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(9, 9)
        self.projection = torch.nn.Linear(9, 9)
        self.config = SimpleNamespace(use_cache=False)

    def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=False, return_dict=True):
        return SimpleNamespace(logits=self.projection(self.embedding(input_ids)))


def test_analysis_keeps_token_identity_and_measures_same_prefix_before_after():
    torch.manual_seed(5)
    rollout = RolloutBatch(
        input_ids=torch.tensor([[1, 2, 3, 4, 5]]),
        attention_mask=torch.ones(1, 5, dtype=torch.long),
        response_ids=torch.tensor([[3, 4, 5]]),
        valid_mask=torch.ones(1, 3, dtype=torch.bool),
        rollout_log_probs=torch.zeros(1, 3),
        prompt_width=2,
    )
    diagnostics = {
        "gain": torch.tensor([[.2, .3, .4]]),
        "successor_excess": torch.tensor([[.5, .6, .7]]),
        "sequential_gain": torch.tensor([[.1, -.1, .2]]),
        "learning_value": torch.tensor([[.3, .2, .6]]),
        "support_reverse_kl": torch.tensor([[.8, .9, 1.0]]),
        "transition_weight": torch.ones(1, 3),
        "support_common_mass": torch.full((1, 3), .7),
    }
    selector = SimpleNamespace(
        diagnostics=diagnostics,
        candidate_ids=torch.tensor([[[1, 2], [1, 2], [1, 2]]]),
        support_mask=torch.ones(1, 3, 2, dtype=torch.bool),
        teacher_candidate_log_probs=torch.log(torch.tensor([[[.4, .6], [.4, .6], [.4, .6]]])),
    )
    student_scores = SimpleNamespace(sampled_log_probs=torch.full((1, 3), -1.0), entropies=torch.full((1, 3), 1.2))
    teacher_scores = SimpleNamespace(sampled_log_probs=torch.full((1, 3), -.8), entropies=torch.full((1, 3), 1.0))
    model = _ToyStudent()
    model.train()
    with tempfile.TemporaryDirectory() as temporary:
        logger = AnalysisLogger(Path(temporary), {
            "tensorboard": False,
            "log_every_n_steps": 1,
            "save_every_n_steps": 1,
            "measure_learning_progress": True,
            "learning_progress_every_n_steps": 1,
            "num_sequences_to_track": 1,
            "num_tokens_per_sequence": 3,
            "max_probe_response_position": 3,
        }, _SingleRank())
        session = logger.begin_rollout(
            scoring_step=1,
            first_optimizer_step=1,
            rollout=rollout,
            selector=selector,
            student_scores=student_scores,
            teacher_scores=teacher_scores,
            objective_valid=rollout.valid_mask,
            sample_ids=["problem-1::response-0"],
            dataset_indices=[12],
            temperature=1.0,
        )
        session.before_optimizer_step(1, model, torch.tensor([0]), rollout.valid_mask, torch.tensor([[.5, 1., 1.5]]))
        assert model.training
        session.on_training_chunk(1, torch.tensor([0]), torch.tensor([[1., 2., 3.]]), torch.tensor([[.5, 1., 1.5]]), rollout.valid_mask)
        with torch.no_grad():
            model.projection.bias[3] += 0.4
        session.after_optimizer_step(1, model, 2.5)
        assert model.training
        logger.finish_rollout(session)
        logger.close()
        token_path = Path(temporary) / "analysis/tokens/step-000001.jsonl"
        progress_path = Path(temporary) / "analysis/progress/step-000001.jsonl"
        tokens = [json.loads(line) for line in token_path.read_text().splitlines()]
        progress = [json.loads(line) for line in progress_path.read_text().splitlines()]
        assert len(tokens) == len(progress) == 3
        assert all(row["sample_id"] == "problem-1::response-0" for row in tokens)
        assert [row["training_weight"] for row in tokens] == [.5, 1., 1.5]
        assert [row["opd_ppo_loss_before"] for row in tokens] == [1., 2., 3.]
        assert all(row["gradient_norm"] == 2.5 for row in tokens)
        assert all(abs(row["g_plus_d"] - row["learning_value"]) < 1e-6 for row in tokens)
        assert any(abs(row["delta_nll"]) > 1e-5 for row in progress)
        assert [row["opd_ppo_loss_before"] for row in progress] == [1., 2., 3.]
        assert [row["training_weight"] for row in progress] == [.5, 1., 1.5]
        assert all(abs(row["delta_kl"] - (row["kl_support_reverse_before"] - row["kl_support_reverse_after"])) < 1e-7 for row in progress)
        assert sum(row["delta_future_kl"] is not None for row in progress) == 2
        assert next(row for row in progress if row["response_position"] == 2)["delta_future_kl"] is None


def test_analysis_statistics_are_paired_and_conditional():
    assert correlations([1, 2, 3], [2, 4, 6])["spearman"] == 1.0
    assert len(quantile_curve([{"g": value, "delta_kl": value / 2} for value in range(8)], "g", "delta_kl", 4)) == 4
    rows = [{"g": index // 8, "d": index % 8, "delta_kl": index % 8} for index in range(16)]
    groups = conditional_d_comparison(rows, "delta_kl", g_bins=2)
    assert len(groups) == 2
    assert all(row["outcome_high_mean"] > row["outcome_low_mean"] for row in groups)


def test_legacy_samples_require_full_valid_token_alignment():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        score_dir = root / "token_score_stats"
        score_dir.mkdir()
        payload = {
            "step": 100,
            "scores": {
                name: {"count": 3, "sample": values}
                for name, values in {
                    "gain": [.1, .2, .3],
                    "successor_excess": [.4, .5, .6],
                    "sequential_gain": [.01, .02, .03],
                }.items()
            },
        }
        (score_dir / "step-000100.json").write_text(json.dumps(payload))
        (root / "metrics.jsonl").write_text(json.dumps({"step": 100, "num_valid_tokens": 3}) + "\n")
        rows = _legacy_score_samples(root)
        assert len(rows) == 3
        assert abs(rows[0]["g_plus_d"] - .11) < 1e-7
        (root / "metrics.jsonl").write_text(json.dumps({"step": 100, "num_valid_tokens": 4}) + "\n")
        assert _legacy_score_samples(root) == []


def test_analysis_hooks_do_not_change_optimizer_update():
    torch.manual_seed(17)
    initial = _ToyStudent()
    baseline, instrumented = copy.deepcopy(initial), copy.deepcopy(initial)
    rollout = RolloutBatch(
        input_ids=torch.tensor([[1, 2, 3, 4, 5]]),
        attention_mask=torch.ones(1, 5, dtype=torch.long),
        response_ids=torch.tensor([[3, 4, 5]]),
        valid_mask=torch.ones(1, 3, dtype=torch.bool),
        rollout_log_probs=torch.zeros(1, 3),
        prompt_width=2,
    )
    with torch.no_grad():
        logits = initial(rollout.input_ids).logits[:, 1:4]
        reference = topk_reference_from_logits(
            logits,
            logits + torch.linspace(-.2, .2, 9),
            rollout.valid_mask,
            top_k=3,
        )
    diagnostics = {
        "gain": torch.tensor([[.2, .3, .4]]),
        "successor_excess": torch.tensor([[.5, .6, .7]]),
        "sequential_gain": torch.tensor([[.1, -.1, .2]]),
        "learning_value": torch.tensor([[.3, .2, .6]]),
        "support_reverse_kl": torch.tensor([[.8, .9, 1.0]]),
        "transition_weight": torch.ones(1, 3),
        "support_common_mass": torch.full((1, 3), .7),
    }
    selector = SimpleNamespace(
        diagnostics=diagnostics,
        candidate_ids=torch.tensor([[[1, 2], [1, 2], [1, 2]]]),
        support_mask=torch.ones(1, 3, 2, dtype=torch.bool),
        teacher_candidate_log_probs=torch.log(torch.tensor([[[.4, .6], [.4, .6], [.4, .6]]])),
    )
    sampled = SimpleNamespace(sampled_log_probs=torch.full((1, 3), -1.0), entropies=torch.ones(1, 3))
    config = {
        "experiment": {"method": "cmt"},
        "rollout": {"temperature": 1.0},
        "selector": {"score_chunk_steps": 2},
        "training": {"micro_batch_size_per_gpu": 1, "ppo_mini_batch_size": 1, "max_grad_norm": 10.0},
    }
    context = DistributedContext(0, 0, 1, torch.device("cpu"))
    baseline_metrics = _opd_train_step(
        baseline, torch.optim.SGD(baseline.parameters(), lr=.01), rollout,
        rollout.valid_mask.float(), reference, config, torch.device("cpu"), context,
        gibbs_scores=diagnostics["learning_value"], gibbs_epsilon=.1,
    )
    with tempfile.TemporaryDirectory() as temporary:
        logger = AnalysisLogger(Path(temporary), {
            "tensorboard": False, "log_every_n_steps": 1, "save_every_n_steps": 1,
            "measure_learning_progress": True, "learning_progress_every_n_steps": 1,
            "num_sequences_to_track": 1, "num_tokens_per_sequence": 3,
            "max_probe_response_position": 3,
        }, context)
        session = logger.begin_rollout(
            scoring_step=1, first_optimizer_step=1, rollout=rollout,
            selector=selector, student_scores=sampled, teacher_scores=sampled,
            objective_valid=rollout.valid_mask, sample_ids=["sample::response-0"],
            dataset_indices=[0], temperature=1.0,
        )
        instrumented_metrics = _opd_train_step(
            instrumented, torch.optim.SGD(instrumented.parameters(), lr=.01), rollout,
            rollout.valid_mask.float(), reference, config, torch.device("cpu"), context,
            gibbs_scores=diagnostics["learning_value"], gibbs_epsilon=.1,
            analysis_session=session,
        )
        logger.finish_rollout(session)
        logger.close()
        progress = list((Path(temporary) / "analysis/progress").glob("*.jsonl"))
        assert len(progress) == 1
        assert len(progress[0].read_text().splitlines()) == 3
    for original, with_analysis in zip(baseline.parameters(), instrumented.parameters()):
        assert torch.allclose(original, with_analysis, atol=1e-7, rtol=1e-6)
    assert abs(baseline_metrics["loss"] - instrumented_metrics["loss"]) < 1e-7
