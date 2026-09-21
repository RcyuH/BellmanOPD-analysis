"""Core utilities for the batchwise top-g_t versus uniform OPD experiment."""

from __future__ import annotations

import math
from typing import Any

import torch

from b200_experiment.opd_core import (
    OPD_LOSS_TOP_K,
    TopKOPDReference,
    build_student_topk_opd_reference,
    gather_candidate_log_probs,
)
from b200_experiment.selectors import PGTSelector, top_budget_mask

from .prompt_token_influence import prompt_opd_losses


GT_DEFINITION = "Var_pU(log(q_U)-log(p_U))"
GT_SUPPORT = "conditional_student_teacher_distributions_on_union_topk"


@torch.inference_mode()
def build_prompt_pgt_reference(
    student,
    teacher,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    student_temperature: float = 1.0,
    teacher_temperature: float = 1.0,
    token_chunk_size: int = 2048,
) -> tuple[TopKOPDReference, torch.Tensor, dict[str, torch.Tensor]]:
    """Compute production PGT ``g_t`` and the separate Student-Top-16 loss ref.

    As in the production PGT trainer, ``g_t`` is evaluated on the literal
    union of student and teacher Top-K sets, while the OPD loss itself remains
    restricted to Student Top-16 IDs.
    """
    if student_temperature <= 0 or teacher_temperature <= 0:
        raise ValueError("OPD temperatures must be positive")
    student_was_training = student.training
    student.eval()
    teacher.eval()
    student_logits = student(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    ).logits
    teacher_logits = teacher(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    ).logits
    student_ids = torch.topk(
        student_logits.float() / float(student_temperature),
        k=OPD_LOSS_TOP_K,
        dim=-1,
    ).indices
    teacher_ids = torch.topk(
        teacher_logits.float() / float(teacher_temperature),
        k=OPD_LOSS_TOP_K,
        dim=-1,
    ).indices
    student_top = gather_candidate_log_probs(
        student_logits,
        student_ids,
        temperature=student_temperature,
        chunk_steps=128,
    )
    teacher_on_student = gather_candidate_log_probs(
        teacher_logits,
        student_ids,
        temperature=teacher_temperature,
        chunk_steps=128,
    )
    teacher_top = gather_candidate_log_probs(
        teacher_logits,
        teacher_ids,
        temperature=teacher_temperature,
        chunk_steps=128,
    )
    student_on_teacher = gather_candidate_log_probs(
        student_logits,
        teacher_ids,
        temperature=student_temperature,
        chunk_steps=128,
    )
    valid = attention_mask.bool()
    pgt = PGTSelector().compute_scores_from_topk(
        student_ids,
        teacher_ids,
        student_top,
        teacher_on_student,
        teacher_top,
        student_on_teacher,
        valid,
        token_chunk_size=token_chunk_size,
    )
    reference = build_student_topk_opd_reference(
        student_ids,
        student_top,
        teacher_on_student,
        valid,
    )
    diagnostics = {
        name: value.detach().clone()
        for name, value in pgt.diagnostics.items()
        if torch.is_tensor(value) and value.shape == valid.shape
    }
    scores = pgt.scores.detach().clone()
    del (
        student_logits,
        teacher_logits,
        student_ids,
        teacher_ids,
        student_top,
        teacher_on_student,
        teacher_top,
        student_on_teacher,
        pgt,
    )
    if student_was_training:
        student.train()
    return reference, scores, diagnostics


def top_gt_weights(
    scores: torch.Tensor, valid_mask: torch.Tensor, fraction: float = 0.10
) -> torch.Tensor:
    """Return the exact stable top-ceil(fraction*N) binary batch mask."""
    # PGT scoring runs under inference_mode. Materialize an ordinary tensor
    # before it participates in a differentiable weighted loss.
    with torch.inference_mode(False):
        return top_budget_mask(
            scores, valid_mask.bool(), float(fraction)
        ).detach().clone().float()


def uniform_weights(valid_mask: torch.Tensor) -> torch.Tensor:
    return valid_mask.bool().float()


def normalized_effective_weights(
    weights: torch.Tensor, valid_mask: torch.Tensor
) -> torch.Tensor:
    if weights.shape != valid_mask.shape:
        raise ValueError("weights and valid_mask must have the same shape")
    active = weights.float() * valid_mask.bool().float()
    mass = active.sum()
    if float(mass.item()) <= 0:
        raise ValueError("At least one valid token must have positive weight")
    return active / mass


def weighted_loss(
    per_token_loss: torch.Tensor,
    weights: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    effective = normalized_effective_weights(weights, valid_mask)
    return (per_token_loss * effective).sum()


def train_prompt_batch(
    model,
    optimizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    reference: TopKOPDReference,
    weights: torch.Tensor,
    *,
    student_temperature: float,
    clip_low: float,
    clip_high: float,
    dual_clip: float,
    chunk_steps: int,
    max_grad_norm: float,
) -> dict[str, Any]:
    """Apply one real optimizer update and return auditable batch scalars."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = prompt_opd_losses(
        model,
        input_ids,
        attention_mask,
        reference,
        student_temperature=student_temperature,
        clip_low=clip_low,
        clip_high=clip_high,
        dual_clip=dual_clip,
        chunk_steps=chunk_steps,
    )
    loss = weighted_loss(losses, weights, attention_mask.bool())
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Non-finite weighted OPD batch loss")
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        float(max_grad_norm),
    )
    optimizer.step()
    result = {
        "loss": float(loss.detach().float().item()),
        "gradient_l2_norm_before_clip": float(gradient_norm.detach().float().item()),
        "selected_tokens": int(((weights > 0) & attention_mask.bool()).sum().item()),
        "valid_tokens": int(attention_mask.sum().item()),
        "per_token_loss": losses.detach().float().cpu(),
    }
    optimizer.zero_grad(set_to_none=True)
    del losses, loss, gradient_norm
    return result


@torch.no_grad()
def model_parameter_l2_distance(left, right) -> float:
    """L2 distance between the two trained student weight vectors."""
    left_parameters = list(left.parameters())
    right_parameters = list(right.parameters())
    if len(left_parameters) != len(right_parameters):
        raise ValueError("Student models have different parameter structures")
    square_sum = 0.0
    for left_parameter, right_parameter in zip(left_parameters, right_parameters):
        if left_parameter.shape != right_parameter.shape:
            raise ValueError("Student models have different parameter shapes")
        difference = left_parameter.detach().float() - right_parameter.detach().float()
        square_sum += float(difference.square().sum().item())
    return math.sqrt(square_sum)
