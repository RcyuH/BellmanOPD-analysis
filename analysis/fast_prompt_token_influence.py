"""Fast first-order prompt-token influence in the output-head subspace.

For a fixed test-distance gradient ``v = grad D_test`` restricted to the
student output projection, every token score is

    predicted_improvement_t = learning_rate * <v, grad L_t>.

All token inner products are computed together from one prompt forward.  The
test gradient uses a conditional KL on teacher Top-K actions, which makes it
sparse in vocabulary rows and avoids a full-model backward through every test
prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from b200_experiment.opd_core import (
    TopKOPDReference,
    gather_candidate_log_probs,
    topk_candidate_ppo_loss,
)


FAST_DISTANCE_NAME = "teacher_topk_conditional_forward_kl"


@dataclass
class SparseHeadDirection:
    active_token_ids: torch.Tensor
    weight: torch.Tensor
    bias: torch.Tensor | None
    gradient_l2_norm: float
    distance_value: float
    num_test_tokens: int
    num_test_problems: int
    refresh_step: int
    support_top_k: int


@torch.inference_mode()
def local_teacher_topk_head_gradient(
    student,
    teacher,
    tokenizer,
    rendered_prompts: Sequence[str],
    *,
    device: torch.device,
    max_prompt_tokens: int,
    support_top_k: int,
    student_temperature: float,
    teacher_temperature: float,
    position_chunk: int = 32,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, float, int, int, int]:
    """Return an unnormalized local output-head gradient and KL statistics."""
    if not rendered_prompts:
        raise ValueError("Each distributed rank needs at least one calibration prompt")
    if support_top_k <= 0:
        raise ValueError("support_top_k must be positive")
    if student_temperature <= 0 or teacher_temperature <= 0:
        raise ValueError("Temperatures must be positive")
    head = student.get_output_embeddings()
    if not isinstance(head, torch.nn.Linear) or not head.weight.requires_grad:
        raise ValueError(
            "Fast mode requires a trainable linear student output projection"
        )
    input_embeddings = student.get_input_embeddings()
    if (
        input_embeddings is not None
        and input_embeddings.weight.data_ptr() == head.weight.data_ptr()
    ):
        raise ValueError(
            "Fast output-head influence requires untied input/output weights; "
            "a tied head would also change hidden states through prompt tokens"
        )
    gradient_weight = torch.zeros_like(head.weight, dtype=torch.float32, device=device)
    gradient_bias = (
        torch.zeros_like(head.bias, dtype=torch.float32, device=device)
        if getattr(head, "bias", None) is not None
        else None
    )
    distance_sum = 0.0
    token_count = 0
    maximum_observed = 0
    chunk_size = max(1, int(position_chunk))
    student_was_training = student.training
    student.eval()
    teacher.eval()
    for prompt_index, prompt in enumerate(rendered_prompts):
        encoded = tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
            truncation=False,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        length = int(attention_mask.sum().item())
        maximum_observed = max(maximum_observed, length)
        if length > int(max_prompt_tokens):
            raise ValueError(
                f"Calibration prompt {prompt_index} has {length} tokens, above "
                f"max_eval_prompt_tokens={max_prompt_tokens}"
            )
        student_output = student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
            output_hidden_states=True,
        )
        teacher_output = teacher(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        student_logits = student_output.logits[0, :length]
        teacher_logits = teacher_output.logits[0, :length]
        hidden = student_output.hidden_states[-1][0, :length]
        for begin in range(0, length, chunk_size):
            end = min(length, begin + chunk_size)
            teacher_scaled = (
                teacher_logits[begin:end].float() / float(teacher_temperature)
            )
            k = min(int(support_top_k), teacher_scaled.shape[-1])
            teacher_values, candidate_ids = torch.topk(teacher_scaled, k=k, dim=-1)
            teacher_logp = torch.log_softmax(teacher_values, dim=-1)
            student_values = (
                student_logits[begin:end].float().gather(-1, candidate_ids)
                / float(student_temperature)
            )
            student_logp = torch.log_softmax(student_values, dim=-1)
            teacher_probability = teacher_logp.exp()
            student_probability = student_logp.exp()
            per_position_kl = (
                teacher_probability * (teacher_logp - student_logp)
            ).sum(dim=-1)
            distance_sum += float(per_position_kl.double().sum().item())
            token_count += end - begin

            # d KL(q || p) / d raw_student_logit on the conditional support.
            selected_logit_gradient = (
                student_probability - teacher_probability
            ) / float(student_temperature)
            chunk_hidden = hidden[begin:end].float()
            repeated_hidden = chunk_hidden.repeat_interleave(k, dim=0)
            contributions = selected_logit_gradient.reshape(-1, 1) * repeated_hidden
            flat_ids = candidate_ids.reshape(-1)
            gradient_weight.index_add_(0, flat_ids, contributions)
            if gradient_bias is not None:
                gradient_bias.index_add_(0, flat_ids, selected_logit_gradient.reshape(-1))
            del (
                teacher_scaled,
                teacher_values,
                candidate_ids,
                teacher_logp,
                student_values,
                student_logp,
                teacher_probability,
                student_probability,
                per_position_kl,
                selected_logit_gradient,
                repeated_hidden,
                contributions,
                flat_ids,
            )
        del (
            encoded,
            input_ids,
            attention_mask,
            student_output,
            teacher_output,
            student_logits,
            teacher_logits,
            hidden,
        )
        if progress_callback is not None:
            progress_callback(prompt_index + 1, len(rendered_prompts))
    if student_was_training:
        student.train()
    return (
        gradient_weight,
        gradient_bias,
        distance_sum,
        token_count,
        len(rendered_prompts),
        maximum_observed,
    )


def directional_logprob_derivative(
    logits: torch.Tensor,
    hidden: torch.Tensor,
    candidate_ids: torch.Tensor,
    *,
    active_token_ids: torch.Tensor,
    direction_weight: torch.Tensor,
    direction_bias: torch.Tensor | None,
    temperature: float,
) -> torch.Tensor:
    """Derivative of candidate log-probabilities along a sparse head direction."""
    if logits.ndim != 3 or hidden.ndim != 3 or candidate_ids.ndim != 3:
        raise ValueError("Expected logits/hidden/candidates with batch and time axes")
    if logits.shape[:2] != hidden.shape[:2] or logits.shape[:2] != candidate_ids.shape[:2]:
        raise ValueError("Prompt tensors do not align")
    if direction_weight.shape[0] != active_token_ids.numel():
        raise ValueError("Sparse direction IDs and rows do not align")
    if direction_weight.shape[1] != hidden.shape[-1]:
        raise ValueError("Sparse direction hidden width does not match model")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    active_ids = active_token_ids.to(device=logits.device, dtype=torch.long)
    active_logits = logits.float().index_select(-1, active_ids)
    log_normalizer = torch.logsumexp(
        logits.float() / float(temperature), dim=-1, keepdim=True
    )
    active_probability = torch.exp(
        active_logits / float(temperature) - log_normalizer
    )
    active_delta_logits = torch.matmul(
        hidden.float(), direction_weight.to(device=hidden.device, dtype=torch.float32).T
    )
    if direction_bias is not None:
        active_delta_logits = active_delta_logits + direction_bias.to(
            device=hidden.device, dtype=torch.float32
        )
    expected_delta = (active_probability * active_delta_logits).sum(
        dim=-1, keepdim=True
    )
    vocab_to_active = torch.full(
        (logits.shape[-1],), -1, dtype=torch.long, device=logits.device
    )
    vocab_to_active[active_ids] = torch.arange(active_ids.numel(), device=logits.device)
    candidate_active_index = vocab_to_active[candidate_ids]
    candidate_is_active = candidate_active_index.ge(0)
    safe_index = candidate_active_index.clamp_min(0)
    candidate_delta = active_delta_logits.gather(-1, safe_index)
    candidate_delta = torch.where(
        candidate_is_active, candidate_delta, torch.zeros_like(candidate_delta)
    )
    return (candidate_delta - expected_delta) / float(temperature)


def prompt_head_first_order_scores(
    student,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    reference: TopKOPDReference,
    direction: SparseHeadDirection,
    *,
    student_temperature: float,
    clip_low: float,
    clip_high: float,
    dual_clip: float,
    chunk_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-token OPD loss and ``<grad D, grad L_t>`` for all positions."""
    # This is a score-only pass. Avoid allocating an all-layer backward graph:
    # the only derivative needed is the small [batch, time, K] PPO derivative.
    with torch.no_grad():
        output = student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
            output_hidden_states=True,
        )
        logits = output.logits
        hidden = output.hidden_states[-1]
        frozen_current = gather_candidate_log_probs(
            logits,
            reference.candidate_ids,
            temperature=student_temperature,
            chunk_steps=chunk_steps,
        )
    current = frozen_current.detach().clone().requires_grad_(True)
    losses = topk_candidate_ppo_loss(
        current,
        reference,
        clip_low=clip_low,
        clip_high=clip_high,
        dual_clip=dual_clip,
    )
    loss_gradient_on_candidate_logp = torch.autograd.grad(
        losses.sum(), current, retain_graph=False, create_graph=False
    )[0]
    with torch.no_grad():
        directional_logp = directional_logprob_derivative(
            logits,
            hidden,
            reference.candidate_ids,
            active_token_ids=direction.active_token_ids,
            direction_weight=direction.weight,
            direction_bias=direction.bias,
            temperature=student_temperature,
        )
        inner_products = (
            loss_gradient_on_candidate_logp * directional_logp
        ).sum(dim=-1)
    return losses.detach(), inner_products.detach()
