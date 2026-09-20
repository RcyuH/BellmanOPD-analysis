from __future__ import annotations

from dataclasses import dataclass

import torch

from .base import SelectorOutput


@dataclass
class PGTOutput(SelectorOutput):
    """PGT/CMT selector output plus its local scoring support."""

    candidate_ids: torch.Tensor
    student_candidate_log_probs: torch.Tensor
    teacher_candidate_log_probs: torch.Tensor
    support_mask: torch.Tensor


class PGTSelector:
    """Projected-Gradient Teachability (PGT).

    PGT scores a state by natural-gradient energy on a local categorical
    surrogate. Let p and q be the **conditional** student/teacher
    distributions on the finite union of their Top-K supports and
    r=log q-log p on that same action space.  The expected logit gradient is
    ``mu_j=p_j (r_j-E_p[r])``.  With the categorical Fisher
    ``F=diag(p)-p p^T``, the largest first-order improvement under a local
    KL/trust-region budget is proportional to

        mu^T F^+ mu = Var_p[r].

    This is not a divergence/overlap heuristic: a constant log-ratio has zero
    policy gradient and therefore zero PGT value. The returned union is a
    selector/CMT scoring support only. Production OPD optimization separately
    uses Student Top-16, so teacher-only union actions can change teachability
    without becoming policy-loss candidates. All outputs are detached.
    """

    @torch.inference_mode()
    def compute_scores_from_topk(
        self,
        student_top_k_ids: torch.Tensor,
        teacher_top_k_ids: torch.Tensor,
        student_top_k_log_probs: torch.Tensor,
        teacher_on_student_log_probs: torch.Tensor,
        teacher_top_k_log_probs: torch.Tensor,
        student_on_teacher_log_probs: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        token_chunk_size: int = 2048,
    ) -> PGTOutput:
        tensors = (
            student_top_k_ids,
            teacher_top_k_ids,
            student_top_k_log_probs,
            teacher_on_student_log_probs,
            teacher_top_k_log_probs,
            student_on_teacher_log_probs,
        )
        if any(value.ndim != 3 for value in tensors):
            raise ValueError("Compact PGT inputs must have shape [batch, time, K]")
        if any(value.shape != tensors[0].shape for value in tensors[1:]):
            raise ValueError("Compact PGT Top-K inputs must share one shape")
        if tensors[0].shape[:2] != valid_mask.shape:
            raise ValueError("Compact PGT inputs must align with valid_mask")

        batch, time, k = student_top_k_ids.shape
        flat_valid = valid_mask.reshape(-1)
        valid_indices = flat_valid.nonzero(as_tuple=False).squeeze(-1)
        union_width = 2 * k
        flat_inputs = [value.reshape(-1, k) for value in tensors]

        candidate_ids = torch.zeros(
            (batch * time, union_width), dtype=torch.long,
            device=student_top_k_ids.device,
        )
        student_support = torch.zeros(
            (batch * time, union_width), dtype=torch.float32,
            device=student_top_k_ids.device,
        )
        teacher_support = torch.zeros_like(student_support)
        support = torch.zeros(
            (batch * time, union_width), dtype=torch.bool,
            device=student_top_k_ids.device,
        )
        gain = torch.zeros(batch * time, dtype=torch.float32, device=student_top_k_ids.device)
        euclidean_gain = torch.zeros_like(gain)
        restricted_kl = torch.zeros_like(gain)
        student_support_mass = torch.zeros_like(gain)
        teacher_support_mass = torch.zeros_like(gain)

        chunk_size = max(1, int(token_chunk_size))
        for begin in range(0, valid_indices.numel(), chunk_size):
            indices = valid_indices[begin : begin + chunk_size]
            stu_ids, tea_ids, stu_logp, tea_on_stu, tea_logp, stu_on_tea = [
                value.index_select(0, indices) for value in flat_inputs
            ]
            ids = torch.cat((stu_ids, tea_ids), dim=-1)
            equal = ids.unsqueeze(-1).eq(ids.unsqueeze(-2))
            unique = ~torch.tril(equal, diagonal=-1).any(dim=-1)
            stu = torch.cat((stu_logp, stu_on_tea), dim=-1).float()
            tea = torch.cat((tea_on_stu, tea_logp), dim=-1).float()
            p_support_logz = torch.logsumexp(
                stu.masked_fill(~unique, -torch.inf), dim=-1, keepdim=True
            )
            q_support_logz = torch.logsumexp(
                tea.masked_fill(~unique, -torch.inf), dim=-1, keepdim=True
            )
            p_cond = torch.where(
                unique, stu - p_support_logz, torch.zeros_like(stu)
            )
            q_cond = torch.where(
                unique, tea - q_support_logz, torch.zeros_like(tea)
            )
            p = torch.where(unique, p_cond.exp(), torch.zeros_like(p_cond))
            r = torch.where(unique, q_cond - p_cond, torch.zeros_like(stu))
            mean_r = (p * r).sum(dim=-1, keepdim=True)
            centered = r - mean_r
            local_gain = (p * centered.square()).sum(dim=-1)
            local_euclidean = (p.square() * centered.square()).sum(dim=-1)
            local_kl = -(p * r).sum(dim=-1)
            local_student_mass = torch.where(
                unique, stu.exp(), torch.zeros_like(stu)
            ).sum(dim=-1)
            local_teacher_mass = torch.where(
                unique, tea.exp(), torch.zeros_like(tea)
            ).sum(dim=-1)
            candidate_ids.index_copy_(
                0, indices, torch.where(unique, ids, torch.zeros_like(ids))
            )
            student_support.index_copy_(0, indices, p_cond)
            teacher_support.index_copy_(0, indices, q_cond)
            support.index_copy_(0, indices, unique)
            gain.index_copy_(0, indices, local_gain)
            euclidean_gain.index_copy_(0, indices, local_euclidean)
            restricted_kl.index_copy_(0, indices, local_kl)
            student_support_mass.index_copy_(0, indices, local_student_mass)
            teacher_support_mass.index_copy_(0, indices, local_teacher_mass)

        shape = (batch, time)
        diagnostics = {
            "gain": gain.reshape(shape),
            "s_PGT": gain.reshape(shape),
            "euclidean_gain": euclidean_gain.reshape(shape),
            "restricted_reverse_kl": restricted_kl.reshape(shape),
            "student_support_mass": student_support_mass.reshape(shape),
            "teacher_support_mass": teacher_support_mass.reshape(shape),
            "teacher_tail_mass": (1.0 - teacher_support_mass).clamp_min(0.0).reshape(shape),
            "support_width": support.float().sum(dim=-1).reshape(shape),
            "score_definition": "natural_gradient_energy_var_p_log_teacher_minus_student",
            "support_definition": "literal_union_student_topk_teacher_topk",
            "support_geometry": "conditional_student_teacher_distributions_on_union",
        }
        # Keep the score explicitly zero on invalid rollout padding.
        diagnostics["gain"] = torch.where(valid_mask, diagnostics["gain"], torch.zeros_like(diagnostics["gain"]))
        diagnostics["s_PGT"] = diagnostics["gain"]
        return PGTOutput(
            diagnostics["s_PGT"],
            diagnostics,
            candidate_ids.reshape(batch, time, union_width),
            student_support.reshape(batch, time, union_width),
            teacher_support.reshape(batch, time, union_width),
            support.reshape(batch, time, union_width),
        )
