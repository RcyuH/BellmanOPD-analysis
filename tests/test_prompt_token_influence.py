import json
from pathlib import Path

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


class _Tokenizer:
    all_special_ids = [9]

    def convert_ids_to_tokens(self, token_id):
        return {3: "▁hello", 9: "<eos>"}[token_id]

    def decode(self, token_ids, **kwargs):
        return {3: " hello", 9: "<eos>"}[token_ids[0]]


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


def test_full_vocab_teacher_to_student_kl_and_mask():
    student = torch.tensor([[[0.0, 0.0], [3.0, -1.0]]])
    teacher = torch.tensor([[[1.0, -1.0], [-2.0, 2.0]]])
    mask = torch.tensor([[True, False]])
    total, count = full_vocab_forward_kl_from_logits(student, teacher, mask)
    q = torch.softmax(teacher[0, 0], dim=-1)
    expected = (q * (torch.log_softmax(teacher[0, 0], -1) - torch.log_softmax(student[0, 0], -1))).sum()
    assert count.item() == 1
    assert torch.allclose(total, expected.double())


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
    assert json.loads((tmp_path / "summary" / "summary.json").read_text())["completed_steps_found"] == 1

