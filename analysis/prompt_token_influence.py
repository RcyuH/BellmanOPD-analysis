"""Exact one-token SGD interventions for prompt-level OPD analysis.

This module contains the small, testable pieces used by
``analysis.run_prompt_token_influence``.  The experiment deliberately keeps
the intervention optimizer separate from the production PPO trainer: every
counterfactual starts from the exact same parameter snapshot and applies the
update requested by the experiment,

    theta_token = theta - learning_rate * grad(OPD_loss_at_token).

The observable called "distance" is a behavioural distance because the
student and teacher need not have compatible parameterizations.  It is the
token-weighted full-vocabulary ``KL(teacher || student)`` on rendered
Competition-MATH test prompts.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from b200_experiment.opd_core import (
    OPD_LOSS_TOP_K,
    TopKOPDReference,
    build_student_topk_opd_reference,
    gather_candidate_log_probs,
    topk_candidate_ppo_loss,
)


SCHEMA_VERSION = 1
DISTANCE_NAME = "full_vocab_teacher_to_student_kl"
TOKEN_SEMANTICS = "teacher_distribution_after_consuming_prompt_token"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def decode_token(tokenizer, token_id: int) -> dict[str, Any]:
    """Return lossless/auditable representations of one tokenizer ID."""
    token_id = int(token_id)
    try:
        piece = tokenizer.convert_ids_to_tokens(token_id)
    except (AttributeError, TypeError, ValueError):
        piece = None
    try:
        decoded = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        decoded = tokenizer.decode([token_id], skip_special_tokens=False)
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    # visible_text is only a convenience for terminals/CSVs. decoded_text is
    # the authoritative value and JSON preserves its whitespace verbatim.
    visible = (
        str(decoded)
        .replace(" ", "·")
        .replace("\n", "\\n")
        .replace("\t", "\\t")
        .replace("\r", "\\r")
    )
    return {
        "token_id": token_id,
        "token_piece": None if piece is None else str(piece),
        "decoded_text": str(decoded),
        "visible_text": visible,
        "decoded_utf8_hex": str(decoded).encode("utf-8").hex(),
        "is_special_token": token_id in special_ids,
    }


def trainable_parameters(model) -> list[torch.nn.Parameter]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("Student has no trainable parameters")
    return parameters


@torch.no_grad()
def clone_parameters(parameters: Sequence[torch.nn.Parameter]) -> list[torch.Tensor]:
    """Keep an exact device-side snapshot for drift-free intervention restore."""
    return [parameter.detach().clone() for parameter in parameters]


@torch.no_grad()
def restore_parameters(
    parameters: Sequence[torch.nn.Parameter], snapshot: Sequence[torch.Tensor]
) -> None:
    if len(parameters) != len(snapshot):
        raise ValueError("Parameter snapshot does not match the model")
    for parameter, original in zip(parameters, snapshot):
        parameter.copy_(original)


@torch.no_grad()
def apply_sgd_update(
    parameters: Sequence[torch.nn.Parameter],
    gradients: Sequence[torch.Tensor | None],
    learning_rate: float,
) -> None:
    if len(parameters) != len(gradients):
        raise ValueError("Gradient list does not match the model")
    if not math.isfinite(float(learning_rate)) or float(learning_rate) <= 0:
        raise ValueError("learning_rate must be finite and positive")
    for parameter, gradient in zip(parameters, gradients):
        if gradient is not None:
            parameter.add_(gradient, alpha=-float(learning_rate))


def gradient_l2_norm(gradients: Sequence[torch.Tensor | None]) -> float:
    total = torch.zeros((), dtype=torch.float64)
    for gradient in gradients:
        if gradient is not None:
            total += gradient.detach().double().square().sum().cpu()
    return math.sqrt(float(total.item()))


def _forward_logits(model, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    return output.logits


@torch.inference_mode()
def build_prompt_opd_reference(
    student,
    teacher,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    top_k: int = OPD_LOSS_TOP_K,
    student_temperature: float = 1.0,
    teacher_temperature: float = 1.0,
) -> TopKOPDReference:
    """Freeze the ordinary Student-Top-16 OPD reference at every prompt state.

    Position ``t`` uses the causal logits emitted after the model has consumed
    prompt token ``x_t``.  Thus an ``N``-token prompt produces exactly ``N``
    interventions, including the final state that predicts the first answer
    token.
    """
    if int(top_k) != OPD_LOSS_TOP_K:
        raise ValueError(
            f"Prompt intervention uses the repository OPD invariant K={OPD_LOSS_TOP_K}; "
            f"got {top_k}"
        )
    if student_temperature <= 0 or teacher_temperature <= 0:
        raise ValueError("OPD temperatures must be positive")
    student_was_training = student.training
    student.eval()
    teacher.eval()
    student_logits = _forward_logits(student, input_ids, attention_mask)
    teacher_logits = _forward_logits(teacher, input_ids, attention_mask)
    candidate_ids = torch.topk(
        student_logits.float() / float(student_temperature),
        k=OPD_LOSS_TOP_K,
        dim=-1,
    ).indices
    student_log_probs = gather_candidate_log_probs(
        student_logits,
        candidate_ids,
        temperature=student_temperature,
        chunk_steps=128,
    )
    teacher_log_probs = gather_candidate_log_probs(
        teacher_logits,
        candidate_ids,
        temperature=teacher_temperature,
        chunk_steps=128,
    )
    reference = build_student_topk_opd_reference(
        candidate_ids,
        student_log_probs,
        teacher_log_probs,
        attention_mask.bool(),
    )
    del student_logits, teacher_logits, candidate_ids, student_log_probs, teacher_log_probs
    if student_was_training:
        student.train()
    return reference


def prompt_opd_losses(
    student,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    reference: TopKOPDReference,
    *,
    student_temperature: float,
    clip_low: float,
    clip_high: float,
    dual_clip: float,
    chunk_steps: int,
) -> torch.Tensor:
    """Return differentiable per-prompt-token OPD losses with shape ``[B,L]``."""
    logits = _forward_logits(student, input_ids, attention_mask)
    current = gather_candidate_log_probs(
        logits,
        reference.candidate_ids,
        temperature=student_temperature,
        chunk_steps=chunk_steps,
    )
    losses = topk_candidate_ppo_loss(
        current,
        reference,
        clip_low=clip_low,
        clip_high=clip_high,
        dual_clip=dual_clip,
    )
    if losses.shape != attention_mask.shape:
        raise AssertionError("Prompt OPD losses do not align with prompt tokens")
    return losses


def full_vocab_forward_kl_from_logits(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    student_temperature: float = 1.0,
    teacher_temperature: float = 1.0,
    vocab_chunk_positions: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(KL sum, valid position count)`` without retaining FP32 copies.

    The reduction is ``KL(q_teacher || p_student)`` over the complete
    vocabulary, then a token-weighted sum over valid prompt states.
    """
    if student_logits.shape != teacher_logits.shape or student_logits.ndim != 3:
        raise ValueError("Student and teacher logits must share shape [B,L,V]")
    if student_logits.shape[:2] != valid_mask.shape:
        raise ValueError("valid_mask must align with logits")
    if student_temperature <= 0 or teacher_temperature <= 0:
        raise ValueError("Distance temperatures must be positive")
    position_chunk = max(1, int(vocab_chunk_positions))
    total = torch.zeros((), dtype=torch.float64, device=student_logits.device)
    count = torch.zeros((), dtype=torch.long, device=student_logits.device)
    width = student_logits.shape[1]
    for begin in range(0, width, position_chunk):
        end = min(width, begin + position_chunk)
        mask = valid_mask[:, begin:end].bool()
        if not bool(mask.any()):
            continue
        student_logp = torch.log_softmax(
            student_logits[:, begin:end].float() / float(student_temperature), dim=-1
        )
        teacher_logp = torch.log_softmax(
            teacher_logits[:, begin:end].float() / float(teacher_temperature), dim=-1
        )
        per_position = (
            teacher_logp.exp() * (teacher_logp - student_logp)
        ).sum(dim=-1)
        total += per_position[mask].double().sum()
        count += mask.sum()
        del student_logp, teacher_logp, per_position
    return total, count


@torch.inference_mode()
def competition_math_teacher_distance(
    student,
    teacher,
    tokenizer,
    rendered_test_prompts: Sequence[str],
    *,
    device: torch.device,
    max_prompt_tokens: int,
    student_temperature: float = 1.0,
    teacher_temperature: float = 1.0,
    vocab_chunk_positions: int = 32,
) -> dict[str, float | int | str]:
    """Measure full-test behavioural distance, with no sampling or truncation."""
    if not rendered_test_prompts:
        raise ValueError("Competition-MATH test split is empty")
    student_was_training = student.training
    student.eval()
    teacher.eval()
    total = 0.0
    token_count = 0
    maximum_observed = 0
    for prompt_index, prompt in enumerate(rendered_test_prompts):
        encoded = tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
            truncation=False,
        )
        length = int(encoded["attention_mask"].sum().item())
        maximum_observed = max(maximum_observed, length)
        if length > int(max_prompt_tokens):
            raise ValueError(
                f"Competition-MATH test prompt {prompt_index} has {length} tokens, "
                f"above max_eval_prompt_tokens={max_prompt_tokens}; refusing to "
                "truncate or silently exclude a test example"
            )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        teacher_logits = _forward_logits(teacher, input_ids, attention_mask)
        student_logits = _forward_logits(student, input_ids, attention_mask)
        kl_sum, count = full_vocab_forward_kl_from_logits(
            student_logits,
            teacher_logits,
            attention_mask.bool(),
            student_temperature=student_temperature,
            teacher_temperature=teacher_temperature,
            vocab_chunk_positions=vocab_chunk_positions,
        )
        total += float(kl_sum.item())
        token_count += int(count.item())
        del input_ids, attention_mask, teacher_logits, student_logits, kl_sum, count
    if student_was_training:
        student.train()
    return {
        "name": DISTANCE_NAME,
        "value": total / token_count,
        "sum": total,
        "num_tokens": token_count,
        "num_problems": len(rendered_test_prompts),
        "maximum_prompt_tokens": maximum_observed,
    }


def improvement_record(before: float, after: float) -> dict[str, float]:
    before = float(before)
    after = float(after)
    absolute = before - after
    return {
        "distance_before": before,
        "distance_after": after,
        "distance_improvement": absolute,
        "relative_distance_improvement": absolute / before if before != 0.0 else 0.0,
    }

