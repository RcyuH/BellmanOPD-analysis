import json
from pathlib import Path
from types import SimpleNamespace

import torch

from analysis.prompt_token_influence import (
    apply_sgd_update,
    atomic_jsonl,
    clone_parameters,
    decode_token,
    full_vocab_forward_kl_from_logits,
    improvement_record,
    restore_parameters,
)
from analysis.summarize_prompt_token_influence import summarize
from analysis.run_prompt_token_influence import (
    DistributedRuntime,
    _fast_step_rows,
    _refresh_fast_direction,
    assigned_prompt_positions,
)
from analysis.fast_prompt_token_influence import (
    SparseHeadDirection,
    directional_logprob_derivative,
    local_teacher_topk_head_gradient,
    prompt_head_first_order_scores,
)
from analysis.prompt_token_influence import (
    build_prompt_opd_reference,
    prompt_opd_losses,
)


class _Tokenizer:
    all_special_ids = [9]

    def convert_ids_to_tokens(self, token_id):
        return {3: "▁hello", 9: "<eos>"}[token_id]

    def decode(self, token_ids, **kwargs):
        return {3: " hello", 9: "<eos>"}[token_ids[0]]


class _ToyTokenizer:
    all_special_ids = []

    def __call__(self, text, **kwargs):
        ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def convert_ids_to_tokens(self, token_id):
        return str(token_id)

    def decode(self, token_ids, **kwargs):
        return str(token_ids[0])


class _ToyLM(torch.nn.Module):
    def __init__(self, vocab=17, hidden=4):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab, hidden)
        self.head = torch.nn.Linear(hidden, vocab, bias=False)

    def get_output_embeddings(self):
        return self.head

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, attention_mask, **kwargs):
        hidden = self.embedding(input_ids)
        logits = self.head(hidden)
        return SimpleNamespace(logits=logits, hidden_states=(hidden,))


def test_decode_token_keeps_whitespace_and_visible_form():
    row = decode_token(_Tokenizer(), 3)
    assert row["token_piece"] == "▁hello"
    assert row["decoded_text"] == " hello"
    assert row["visible_text"] == "·hello"
    assert bytes.fromhex(row["decoded_utf8_hex"]).decode() == " hello"
    assert row["is_special_token"] is False


def test_virtual_sgd_restore_is_exact():
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    snapshot = clone_parameters([parameter])
    apply_sgd_update([parameter], [torch.tensor([0.5, -1.0])], 0.2)
    assert torch.allclose(parameter, torch.tensor([0.9, -1.8]))
    restore_parameters([parameter], snapshot)
    assert torch.equal(parameter, snapshot[0])


def test_eight_gpu_token_partition_is_disjoint_and_complete():
    shards = [assigned_prompt_positions(35, rank, 8) for rank in range(8)]
    assert sorted(position for shard in shards for position in shard) == list(range(35))
    assert sum(len(shard) for shard in shards) == 35
    assert shards[0] == [0, 8, 16, 24, 32]
    assert shards[7] == [7, 15, 23, 31]


def test_full_vocab_teacher_to_student_kl_and_mask():
    student = torch.tensor([[[0.0, 0.0], [3.0, -1.0]]])
    teacher = torch.tensor([[[1.0, -1.0], [-2.0, 2.0]]])
    mask = torch.tensor([[True, False]])
    total, count = full_vocab_forward_kl_from_logits(student, teacher, mask)
    q = torch.softmax(teacher[0, 0], dim=-1)
    expected = (q * (torch.log_softmax(teacher[0, 0], -1) - torch.log_softmax(student[0, 0], -1))).sum()
    assert count.item() == 1
    assert torch.allclose(total, expected.double())


def test_sparse_head_logprob_direction_matches_finite_difference():
    torch.manual_seed(4)
    logits = torch.randn(1, 2, 5)
    hidden = torch.randn(1, 2, 3)
    candidates = torch.tensor([[[0, 1, 3], [2, 3, 4]]])
    active = torch.tensor([1, 3])
    direction_weight = torch.randn(2, 3)
    direction_bias = torch.randn(2)
    analytical = directional_logprob_derivative(
        logits,
        hidden,
        candidates,
        active_token_ids=active,
        direction_weight=direction_weight,
        direction_bias=direction_bias,
        temperature=1.0,
    )
    delta = torch.zeros_like(logits)
    delta[:, :, active] = hidden @ direction_weight.T + direction_bias
    epsilon = 1.0e-3
    plus = torch.log_softmax(logits + epsilon * delta, dim=-1).gather(-1, candidates)
    minus = torch.log_softmax(logits - epsilon * delta, dim=-1).gather(-1, candidates)
    numerical = (plus - minus) / (2.0 * epsilon)
    torch.testing.assert_close(analytical, numerical, rtol=2e-3, atol=2e-4)


def test_fast_test_gradient_matches_head_autograd():
    torch.manual_seed(7)
    student, teacher = _ToyLM(), _ToyLM()
    weight, bias, distance_sum, token_count, problem_count, _ = (
        local_teacher_topk_head_gradient(
            student,
            teacher,
            _ToyTokenizer(),
            ["toy"],
            device=torch.device("cpu"),
            max_prompt_tokens=4,
            support_top_k=3,
            student_temperature=1.0,
            teacher_temperature=1.0,
        )
    )
    assert bias is None
    assert (token_count, problem_count) == (3, 1)
    ids = torch.tensor([[1, 2, 3]])
    student_logits = student(ids, torch.ones_like(ids)).logits
    teacher_logits = teacher(ids, torch.ones_like(ids)).logits.detach()
    candidates = teacher_logits.topk(3, dim=-1).indices
    q = torch.log_softmax(teacher_logits.gather(-1, candidates), dim=-1)
    p = torch.log_softmax(student_logits.gather(-1, candidates), dim=-1)
    expected = (q.exp() * (q - p)).sum(dim=-1).sum()
    expected_grad = torch.autograd.grad(expected, student.head.weight)[0]
    torch.testing.assert_close(weight, expected_grad, rtol=1e-5, atol=1e-6)
    assert abs(distance_sum - float(expected.item())) < 1e-6


def test_fast_direction_refresh_normalizes_inference_tensor():
    torch.manual_seed(9)
    student, teacher = _ToyLM(), _ToyLM()
    runtime = DistributedRuntime(0, 0, 1, torch.device("cpu"), False)
    direction = _refresh_fast_direction(
        student,
        teacher,
        _ToyTokenizer(),
        ["toy"],
        settings={
            "max_eval_prompt_tokens": 4,
            "fast": {"support_top_k": 3, "position_chunk": 2},
        },
        runtime=runtime,
        step=0,
    )
    assert direction.num_test_tokens == 3
    assert direction.num_test_problems == 1
    assert direction.weight.is_inference() is False


def test_fast_token_scores_match_per_token_head_gradients():
    torch.manual_seed(8)
    student, teacher = _ToyLM(), _ToyLM()
    ids = torch.tensor([[1, 2, 3]])
    mask = torch.ones_like(ids)
    reference = build_prompt_opd_reference(
        student, teacher, ids, mask, top_k=16
    )
    active = torch.tensor([0, 3, 7])
    weight = torch.randn(3, 4)
    direction = SparseHeadDirection(
        active_token_ids=active,
        weight=weight,
        bias=None,
        gradient_l2_norm=1.0,
        distance_value=0.1,
        num_test_tokens=3,
        num_test_problems=1,
        refresh_step=0,
        support_top_k=3,
    )
    fast_losses, fast_scores = prompt_head_first_order_scores(
        student,
        ids,
        mask,
        reference,
        direction,
        student_temperature=1.0,
        clip_low=0.2,
        clip_high=0.28,
        dual_clip=3.0,
        chunk_steps=2,
    )
    losses = prompt_opd_losses(
        student,
        ids,
        mask,
        reference,
        student_temperature=1.0,
        clip_low=0.2,
        clip_high=0.28,
        dual_clip=3.0,
        chunk_steps=2,
    )
    torch.testing.assert_close(fast_losses, losses.detach())
    full_direction = torch.zeros_like(student.head.weight)
    full_direction[active] = weight
    for position in range(ids.shape[1]):
        gradient = torch.autograd.grad(
            losses[0, position], student.head.weight, retain_graph=True
        )[0]
        expected_score = (gradient * full_direction).sum()
        torch.testing.assert_close(
            fast_scores[0, position], expected_score, rtol=1e-5, atol=1e-6
        )


def test_fast_step_scores_all_tokens_and_trains_uniformly():
    torch.manual_seed(11)
    student, teacher = _ToyLM(), _ToyLM()
    tokenizer = _ToyTokenizer()
    ids = torch.tensor([[1, 2, 3]])
    mask = torch.ones_like(ids)
    reference = build_prompt_opd_reference(student, teacher, ids, mask, top_k=16)
    original = student.head.weight.detach().clone()
    runtime = DistributedRuntime(0, 0, 1, torch.device("cpu"), False)
    direction = SparseHeadDirection(
        active_token_ids=torch.tensor([0, 1, 2]),
        weight=torch.randn(3, 4),
        bias=None,
        gradient_l2_norm=1.0,
        distance_value=0.2,
        num_test_tokens=3,
        num_test_problems=1,
        refresh_step=0,
        support_top_k=3,
    )
    rows, _ = _fast_step_rows(
        model=student,
        tokenizer=tokenizer,
        encoded={"input_ids": ids, "attention_mask": mask},
        reference=reference,
        direction=direction,
        parameters=list(student.parameters()),
        config={
            "training": {
                "learning_rate": 1e-3,
                "ppo_clip_low": 0.2,
                "ppo_clip_high": 0.28,
                "ppo_dual_clip": 3.0,
            },
            "rollout": {"temperature": 1.0},
            "opd": {"teacher_temperature": 1.0},
            "selector": {"score_chunk_steps": 2},
        },
        settings={"learning_rate": 1e-3},
        runtime=runtime,
        seed=42,
        step=0,
        epoch=0,
        dataset_index=0,
        sample_id="toy",
    )
    assert len(rows) == 4
    assert [row["prompt_position"] for row in rows[:3]] == [0, 1, 2]
    assert all(row["distance_after"] is None for row in rows)
    assert all(row["is_exact_intervention"] is False for row in rows)
    assert rows[-1]["intervention"] == "uniform_mean_all_prompt_tokens_first_order"
    assert not torch.equal(student.head.weight, original)


def test_improvement_sign_is_positive_when_distance_falls():
    result = improvement_record(2.0, 1.5)
    assert result["distance_improvement"] == 0.5
    assert result["relative_distance_improvement"] == 0.25


def test_summarizer_ranks_full_token_text(tmp_path: Path):
    root = tmp_path / "run"
    rows = [
        {
            "intervention": "single_token",
            "optimizer_step_before": 0,
            "epoch": 0,
            "dataset_index": 4,
            "sample_id": "a",
            "prompt_position": 0,
            "token_id": 3,
            "token_piece": "▁hello",
            "decoded_text": " hello",
            "visible_text": "·hello",
            "is_special_token": False,
            "opd_loss": 1.0,
            "gradient_l2_norm": 2.0,
            "distance_before": 2.0,
            "distance_after": 1.7,
            "distance_improvement": 0.3,
            "relative_distance_improvement": 0.15,
        },
        {
            "intervention": "uniform_mean_all_prompt_tokens",
            "optimizer_step_before": 0,
            "distance_improvement": 0.1,
        },
    ]
    atomic_jsonl(root / "steps" / "step-000000.jsonl", rows)
    result = summarize(root, tmp_path / "summary", 10)
    assert result["token_interventions"] == 1
    csv_text = (tmp_path / "summary" / "top_tokens.csv").read_text()
    assert " hello" in csv_text
    payload = json.loads((tmp_path / "summary" / "summary.json").read_text())
    assert payload["completed_steps_found"] == 1
    assert payload["ranking_kind"] == "measured_exact_intervention"


def test_summarizer_marks_first_order_score_as_prediction(tmp_path: Path):
    root = tmp_path / "run"
    rows = [
        {
            "intervention": "single_token_first_order",
            "optimizer_step_before": 0,
            "prompt_position": 0,
            "token_id": 3,
            "decoded_text": " hello",
            "is_exact_intervention": False,
            "predicted_distance_improvement": 0.03,
            "distance_before": None,
            "distance_after": None,
            "distance_improvement": None,
        },
        {
            "intervention": "uniform_mean_all_prompt_tokens_first_order",
            "optimizer_step_before": 0,
            "is_exact_intervention": False,
            "predicted_distance_improvement": 0.01,
            "distance_improvement": None,
        },
    ]
    atomic_jsonl(root / "steps" / "step-000000.jsonl", rows)
    result = summarize(root, tmp_path / "summary", 10)
    assert result["ranking_kind"] == "predicted_first_order_output_head"
    assert result["completed_steps_found"] == 1
    assert "0.03" in (tmp_path / "summary" / "step_summary.csv").read_text()
