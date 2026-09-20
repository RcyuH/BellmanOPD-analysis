from __future__ import annotations

from dataclasses import dataclass

import torch


# Reference used for the main experiment.  Keep this visible in run metadata and
# tests so a future upstream change cannot silently alter the objective.
UPSTREAM_OPD_COMMIT = "ac26e38d6f1572eb027597b48a9f4e01f6915ef8"
UPSTREAM_TOP_K_STRATEGY = "only_stu"
UPSTREAM_REWARD_WEIGHT_MODE = "student_p"
UPSTREAM_ADV_ESTIMATOR = "token_reward_direct"
UPSTREAM_LOSS_AGG_MODE = "token-mean"
OPD_LOSS_TOP_K = 16


def compute_iw_opd_weights(
    teacher_sampled_log_probs: torch.Tensor,
    student_sampled_log_probs: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    weight_max: float = 1.5,
    use_abs: bool = True,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Compute the official IW-OPD prefix remaining-mass multiplier.

    The returned tensor is detached and has the same ``[B, T]`` shape as the
    sampled-token log-probabilities.  It is *not* an allocator: valid tokens
    retain the ordinary OPD token mean, while this multiplier is applied to
    the sampled-token OPD advantage before the PPO clipping operation.  This
    is the ``adv_mass_weights`` construction in YannX1e/IW-OPD
    (https://github.com/YannX1e/Importance-Weighted-On-Policy-Distillation).
    """
    if teacher_sampled_log_probs.shape != student_sampled_log_probs.shape:
        raise ValueError("IW teacher and student sampled log-probs must match")
    if teacher_sampled_log_probs.shape != valid_mask.shape:
        raise ValueError("IW sampled log-probs must align with valid_mask")
    if teacher_sampled_log_probs.ndim != 2:
        raise ValueError("IW sampled log-probs must have shape [batch, time]")
    if float(weight_max) < 1.0:
        raise ValueError("iw_opd.weight_max must be at least 1")
    if float(eps) <= 0.0:
        raise ValueError("iw_opd.eps must be positive")
    with torch.no_grad():
        mask = valid_mask.to(dtype=torch.float32)
        gap = teacher_sampled_log_probs.float() - student_sampled_log_probs.float()
        discrepancy = gap.abs() if use_abs else gap
        discrepancy = discrepancy * mask
        total_mass = discrepancy.sum(dim=-1, keepdim=True)
        prefix_mass = torch.cumsum(discrepancy, dim=-1) - discrepancy
        fraction_before = prefix_mass / (total_mass + float(eps))
        remaining_mass = (1.0 - fraction_before).clamp(0.0, 1.0) * mask
        weights = 1.0 + (float(weight_max) - 1.0) * remaining_mass
        return weights.detach()


def build_iw_opd_reference(
    sampled_ids: torch.Tensor,
    student_sampled_log_probs: torch.Tensor,
    teacher_sampled_log_probs: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    weight_max: float = 1.5,
    use_abs: bool = True,
    eps: float = 1.0e-8,
) -> tuple[TopKOPDReference, torch.Tensor]:
    """Build a sampled-action (K=1) reference matching official IW-OPD.

    Unlike the historical Top-K OPD objective, IW-OPD's advantage is defined
    only for the actually sampled response token.  Keeping a singleton
    candidate support makes the existing differentiable PPO path execute the
    same clipped surrogate without changing any other method.
    """
    if sampled_ids.shape != valid_mask.shape:
        raise ValueError("IW sampled token IDs must align with valid_mask")
    weights = compute_iw_opd_weights(
        teacher_sampled_log_probs,
        student_sampled_log_probs,
        valid_mask,
        weight_max=weight_max,
        use_abs=use_abs,
        eps=eps,
    )
    with torch.inference_mode(False):
        ids = sampled_ids.detach().clone().long().unsqueeze(-1)
        student = student_sampled_log_probs.detach().clone().float().unsqueeze(-1)
        teacher = teacher_sampled_log_probs.detach().clone().float().unsqueeze(-1)
        valid = valid_mask.detach().clone().bool()
        advantages = ((teacher - student) * weights.unsqueeze(-1)).where(
            valid.unsqueeze(-1), torch.zeros_like(teacher)
        )
        reference = TopKOPDReference(
            candidate_ids=ids,
            old_student_log_probs=student,
            teacher_log_probs=teacher,
            student_weights=valid.unsqueeze(-1).float(),
            advantages=advantages,
            # ``None`` is deliberate: the singleton candidate identifies the
            # sampled action, but the PPO log-prob must still be normalized over
            # the full model vocabulary.  Conditionalizing on K=1 would make
            # every current sampled log-prob equal to zero and change the
            # official PPO ratio.
            support_mask=None,
        )
    return reference, weights


@dataclass(frozen=True)
class TopKOPDReference:
    """Frozen on-policy Top-K support and rewards for one rollout batch."""

    candidate_ids: torch.Tensor
    old_student_log_probs: torch.Tensor
    teacher_log_probs: torch.Tensor
    student_weights: torch.Tensor
    advantages: torch.Tensor
    # ``None`` preserves the pinned upstream full-vocabulary log-probability
    # objective on Student Top-K candidate IDs. Selector supports (for example
    # the TA/CMT student/teacher union) must never be passed into this field.
    support_mask: torch.Tensor | None = None


def _validate_topk_tensors(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    valid_mask: torch.Tensor,
    support_mask: torch.Tensor | None = None,
) -> None:
    if student_log_probs.ndim != 3:
        raise ValueError("Top-K log-probabilities must have shape [batch, time, K]")
    if student_log_probs.shape != teacher_log_probs.shape:
        raise ValueError("Student and teacher Top-K log-probabilities must match")
    if student_log_probs.shape[:2] != valid_mask.shape:
        raise ValueError("Top-K log-probabilities must align with valid_mask")
    if student_log_probs.shape[-1] <= 0:
        raise ValueError("Top-K support cannot be empty")
    if support_mask is not None and support_mask.shape != student_log_probs.shape:
        raise ValueError("support_mask must align with Top-K log-probabilities")


def build_topk_opd_reference(
    candidate_ids: torch.Tensor,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    valid_mask: torch.Tensor,
    support_mask: torch.Tensor | None = None,
) -> TopKOPDReference:
    """Port thunlp/OPD's only_stu + student_p token reward exactly.

    Upstream computes ``rm_scores = -(S_logp - T_on_S) * softmax(S_logp)``
    across K, then uses ``token_reward_direct`` so these rewards become the
    detached candidate-wise advantages. Callers pass full-vocabulary
    log-probabilities gathered only at Student Top-K IDs.
    """
    _validate_topk_tensors(
        student_log_probs, teacher_log_probs, valid_mask, support_mask
    )
    if candidate_ids.shape != student_log_probs.shape:
        raise ValueError("Student candidate IDs and Top-K log-probabilities must match")
    # ``score_original_rollout`` deliberately runs under inference_mode.  A
    # dtype conversion is allowed to be a no-op (FP32 -> FP32, int64 -> int64),
    # so ``detach().float()``/``detach().long()`` can otherwise leak inference
    # tensors into the differentiable PPO gather below.  Clone with inference
    # mode explicitly disabled to materialize ordinary frozen tensors that
    # autograd is allowed to save for backward.
    with torch.inference_mode(False):
        candidate_ids = candidate_ids.detach().clone().to(dtype=torch.long)
        student = student_log_probs.detach().clone().to(dtype=torch.float32)
        teacher = teacher_log_probs.detach().clone().to(dtype=torch.float32)
        valid = valid_mask.detach().clone().to(dtype=torch.bool)
        support = (
            torch.ones_like(student, dtype=torch.bool)
            if support_mask is None
            else support_mask.detach().clone().to(dtype=torch.bool)
        )
        masked_student = student.masked_fill(~support, -torch.inf)
        weights = torch.softmax(masked_student, dim=-1)
        advantages = (teacher - student) * weights
        position_mask = valid.unsqueeze(-1) & support
        weights = torch.where(position_mask, weights, torch.zeros_like(weights))
        advantages = torch.where(
            position_mask, advantages, torch.zeros_like(advantages)
        )
    return TopKOPDReference(
        candidate_ids=candidate_ids,
        old_student_log_probs=student,
        teacher_log_probs=teacher,
        student_weights=weights,
        advantages=advantages,
        support_mask=support,
    )


def build_student_topk_opd_reference(
    student_top_k_ids: torch.Tensor,
    student_top_k_log_probs: torch.Tensor,
    teacher_on_student_log_probs: torch.Tensor,
    valid_mask: torch.Tensor,
) -> TopKOPDReference:
    """Build the invariant Student-Top-16 policy-loss reference.

    TA/CMT may use a larger student/teacher union to compute selector scores,
    but that union is intentionally absent from this API. This makes it
    impossible for teacher-only Top-K IDs to enter the differentiable OPD
    candidate loss through the production reference builder.
    """
    if student_top_k_ids.shape[-1] != OPD_LOSS_TOP_K:
        raise ValueError(
            "OPD loss support must contain exactly Student Top-16 IDs; "
            f"got K={student_top_k_ids.shape[-1]}"
        )
    return build_topk_opd_reference(
        student_top_k_ids,
        student_top_k_log_probs,
        teacher_on_student_log_probs,
        valid_mask,
        support_mask=None,
    )


def topk_reference_from_logits(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    top_k: int,
    student_temperature: float = 1.0,
    teacher_temperature: float = 1.0,
) -> TopKOPDReference:
    """Correctness/reference implementation used by synthetic unit tests."""
    if student_logits.shape != teacher_logits.shape or student_logits.ndim != 3:
        raise ValueError("Student/teacher logits must share shape [batch, time, vocab]")
    if student_logits.shape[:2] != valid_mask.shape:
        raise ValueError("Logits must align with valid_mask")
    if student_temperature <= 0 or teacher_temperature <= 0:
        raise ValueError("OPD scoring temperatures must be positive")
    k = min(int(top_k), student_logits.shape[-1])
    if k <= 0:
        raise ValueError("top_k must be positive")
    student_all = torch.log_softmax(
        student_logits.float() / float(student_temperature), dim=-1
    )
    teacher_all = torch.log_softmax(
        teacher_logits.float() / float(teacher_temperature), dim=-1
    )
    candidate_ids = torch.topk(student_all, k=k, dim=-1).indices
    return build_topk_opd_reference(
        candidate_ids,
        student_all.gather(-1, candidate_ids),
        teacher_all.gather(-1, candidate_ids),
        valid_mask,
    )


def gather_candidate_log_probs(
    response_logits: torch.Tensor,
    candidate_ids: torch.Tensor,
    *,
    temperature: float,
    chunk_steps: int,
    support_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gather differentiable FP32 log-probs without retaining full FP32 vocab.

    When ``support_mask`` is supplied, probabilities are conditionalized on
    the executable candidate support.  This is the geometry used by CMT/PGT:
    the optimizer's categorical simplex is exactly the student Top-K rather than
    the unmaterialized vocabulary tail.
    """
    if response_logits.ndim != 3 or candidate_ids.ndim != 3:
        raise ValueError("Expected logits [B,T,V] and candidate IDs [B,T,K]")
    if response_logits.shape[:2] != candidate_ids.shape[:2]:
        raise ValueError("Candidate IDs do not align with response logits")
    if support_mask is not None:
        if support_mask.shape != candidate_ids.shape:
            raise ValueError("support_mask must align with candidate IDs")
        support_mask = support_mask.to(dtype=torch.bool, device=candidate_ids.device)
    if temperature <= 0:
        raise ValueError("OPD scoring temperature must be positive")
    width = response_logits.shape[1]
    chunks = []
    for begin in range(0, width, max(1, int(chunk_steps))):
        end = min(begin + max(1, int(chunk_steps)), width)
        logits = response_logits[:, begin:end].float() / float(temperature)
        selected = logits.gather(-1, candidate_ids[:, begin:end])
        if support_mask is None:
            normalizer = torch.logsumexp(logits, dim=-1, keepdim=True)
        else:
            selected_mask = support_mask[:, begin:end]
            normalizer = torch.logsumexp(
                selected.masked_fill(~selected_mask, -torch.inf),
                dim=-1,
                keepdim=True,
            )
        chunks.append(selected - normalizer)
    return torch.cat(chunks, dim=1)


def topk_candidate_ppo_loss(
    current_log_probs: torch.Tensor,
    reference: TopKOPDReference,
    *,
    clip_low: float,
    clip_high: float,
    dual_clip: float | None = 3.0,
) -> torch.Tensor:
    """Upstream candidate-wise PPO loss, summed over K to one loss per position."""
    if current_log_probs.shape != reference.old_student_log_probs.shape:
        raise ValueError("Current and frozen Top-K log-probabilities must match")
    if dual_clip is not None and dual_clip <= 1.0:
        raise ValueError("dual_clip must be greater than 1")
    log_ratio = (current_log_probs - reference.old_student_log_probs).clamp(
        min=-20.0, max=20.0
    )
    ratio = torch.exp(log_ratio)
    advantages = reference.advantages
    # OPD/PGT references store candidate-wise advantages as [B,T,K], while
    # GRPO naturally supplies one outcome advantage per sampled trajectory
    # ([B]) or one value broadcast over the response ([B,1,1]).  PyTorch does
    # not broadcast [B] against [B,T,K] as intended: it aligns the leading B
    # with K and can produce the opaque ``B vs T`` error seen at the first
    # GRPO optimizer step.  Canonicalize the trajectory-level forms here so
    # the loss remains reusable and the invariant is enforced at the common
    # objective boundary.
    if advantages.ndim == 1:
        if advantages.shape[0] != ratio.shape[0]:
            raise ValueError(
                "Trajectory advantages must have one value per PPO row"
            )
        advantages = advantages.reshape(
            advantages.shape[0], *([1] * (ratio.ndim - 1))
        )
    elif advantages.ndim == 2 and advantages.shape == ratio.shape[:2]:
        advantages = advantages.unsqueeze(-1)
    elif (
        advantages.ndim == 2
        and advantages.shape[0] == ratio.shape[0]
        and advantages.shape[1] == 1
    ):
        advantages = advantages.reshape(
            advantages.shape[0], *([1] * (ratio.ndim - 1))
        )
    elif advantages.ndim != ratio.ndim:
        raise ValueError(
            "Advantages must be [B], [B,T], or candidate-wise [B,T,K]; "
            f"got {tuple(advantages.shape)} for ratio {tuple(ratio.shape)}"
        )
    if advantages.shape != ratio.shape:
        try:
            torch.broadcast_shapes(advantages.shape, ratio.shape)
        except RuntimeError as error:
            raise ValueError(
                "Advantages are not broadcastable over candidate PPO ratios: "
                f"{tuple(advantages.shape)} vs {tuple(ratio.shape)}"
            ) from error
    loss_unclipped = -advantages * ratio
    loss_clipped = -advantages * ratio.clamp(
        1.0 - float(clip_low), 1.0 + float(clip_high)
    )
    upper_clipped = torch.maximum(loss_unclipped, loss_clipped)
    if dual_clip is None:
        # Canonical GRPO uses the clipped PPO surrogate without the optional
        # dual-clip negative-advantage extension used by the OPD recipe.
        candidate_loss = upper_clipped
    else:
        dual_clipped = torch.minimum(-advantages * float(dual_clip), upper_clipped)
        candidate_loss = torch.where(advantages < 0, dual_clipped, upper_clipped)
    return candidate_loss.sum(dim=-1)


def weighted_token_sums(
    per_position_loss: torch.Tensor,
    position_weights: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return numerator/denominator for the common global token-mean."""
    if (
        per_position_loss.shape != valid_mask.shape
        or position_weights.shape != valid_mask.shape
    ):
        raise ValueError("Loss, position weights, and valid mask must have one shape")
    weights = torch.where(
        valid_mask.bool(),
        position_weights.detach().float(),
        torch.zeros_like(position_weights, dtype=torch.float32),
    )
    if bool((weights < 0).any()):
        raise ValueError("Position weights cannot be negative")
    return (per_position_loss * weights).sum(), weights.sum()


def topk_overlap_fraction(
    student_ids: torch.Tensor, teacher_ids: torch.Tensor, valid_mask: torch.Tensor
) -> torch.Tensor:
    """Mean set-overlap fraction |S_student ∩ S_teacher| / K per position."""
    if (
        student_ids.shape != teacher_ids.shape
        or student_ids.shape[:2] != valid_mask.shape
    ):
        raise ValueError("Top-K ID tensors must match each other and valid_mask")
    overlap = (
        student_ids.unsqueeze(-1)
        .eq(teacher_ids.unsqueeze(-2))
        .any(dim=-1)
        .float()
        .mean(dim=-1)
    )
    return overlap[valid_mask.bool()]
