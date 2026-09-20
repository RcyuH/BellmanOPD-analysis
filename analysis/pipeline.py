"""Bounded CMT diagnostics collected alongside, but outside, the training graph.

The hooks in ``trainer.py`` are deliberately narrow: one at the actual PPO
token loss, one immediately before an optimizer update, and one immediately
after it.  All retained values are detached CPU scalars.  No analysis tensor
contributes to the optimizer objective.
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from b200_experiment.scoring import RolloutBatch, score_original_rollout

from .metrics import PREDICTORS, correlations


def _as_cpu_values(tensor: torch.Tensor, coords: list[tuple[int, int]]) -> list[float]:
    rows = torch.tensor([row for row, _ in coords], device=tensor.device)
    positions = torch.tensor([position for _, position in coords], device=tensor.device)
    return tensor.detach()[rows, positions].float().cpu().tolist()


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _rank_descending(values: list[float], index: int) -> int:
    """One-based competition rank; equal scores receive the same rank."""
    return 1 + sum(value > values[index] for value in values)


class RolloutAnalysis:
    """One rollout's sampled records; instantiated on every distributed rank."""

    def __init__(
        self,
        logger: "AnalysisLogger",
        *,
        scoring_step: int,
        first_optimizer_step: int,
        rollout: RolloutBatch,
        selector,
        student_scores,
        teacher_scores,
        objective_valid: torch.Tensor,
        sample_ids: list[str],
        dataset_indices: list[int],
        reference_texts: list[str | None] | None = None,
        temperature: float,
    ) -> None:
        self.logger = logger
        self.scoring_step = int(scoring_step)
        self.first_optimizer_step = int(first_optimizer_step)
        self.score_model_step = int(first_optimizer_step) - 1
        self.rollout = rollout
        self.selector = selector
        self.objective_valid = objective_valid
        self.sample_ids = sample_ids
        self.dataset_indices = dataset_indices
        self.reference_texts = reference_texts or [None] * len(sample_ids)
        if len(self.reference_texts) != len(sample_ids):
            raise ValueError("reference_texts must align with sample_ids")
        self.teacher_scores = teacher_scores
        self.temperature = float(temperature)
        self.progress_rows: list[dict[str, Any]] = []
        self.qualitative_rows: list[dict[str, Any]] = []
        self._qualitative_row_cache: dict[int, dict[str, Any]] = {}
        self._pending: dict[str, Any] | None = None
        self.save_tokens = logger.settings["save_token_statistics"] and any(
            step % logger.settings["save_every_n_steps"] == 0
            for step in range(first_optimizer_step, scoring_step + 1)
        )
        self.summarize_tokens = any(
            step % logger.settings["log_every_n_steps"] == 0
            for step in range(first_optimizer_step, scoring_step + 1)
        )
        self.expect_progress = logger.settings["measure_learning_progress"] and any(
            step % logger.settings["learning_progress_every_n_steps"] == 0
            for step in range(first_optimizer_step, scoring_step + 1)
        )
        self.token_rows: list[dict[str, Any]] = []
        self._token_lookup: dict[tuple[int, int], dict[str, Any]] = {}
        if self.save_tokens or self.summarize_tokens:
            coords = self._sample_coordinates(
                objective_valid,
                list(range(objective_valid.shape[0])),
                scoring_step,
            )
            self.token_rows = self._base_rows(
                coords, student_scores, teacher_scores, scoring_step
            )
            self._token_lookup = dict(zip(coords, self.token_rows))

    def _sample_coordinates(
        self, mask: torch.Tensor, possible_rows: list[int], step: int
    ) -> list[tuple[int, int]]:
        settings = self.logger.settings
        generator = random.Random(settings["seed"] + 1000003 * step + self.logger.rank)
        active = [row for row in possible_rows if bool(mask[row].any())]
        selected_rows = sorted(
            generator.sample(active, min(len(active), settings["num_sequences_to_track"]))
        )
        result: list[tuple[int, int]] = []
        for row in selected_rows:
            positions = mask[row].nonzero(as_tuple=False).flatten().cpu().tolist()
            chosen = sorted(
                generator.sample(positions, min(len(positions), settings["num_tokens_per_sequence"]))
            )
            result.extend((row, int(position)) for position in chosen)
        return result

    def _base_rows(self, coords, student_scores, teacher_scores, step) -> list[dict]:
        if not coords:
            return []
        diagnostics = self.selector.diagnostics
        names = {
            "g": "gain",
            "x": "successor_excess",
            "d": "sequential_gain",
            "learning_value": "learning_value",
            "kl_support_reverse": "support_reverse_kl",
            "transition_weight": "transition_weight",
            "teacher_student_common_mass": "support_common_mass",
        }
        columns = {name: _as_cpu_values(diagnostics[source], coords) for name, source in names.items()}
        columns.update(
            student_target_logprob=_as_cpu_values(student_scores.sampled_log_probs, coords),
            teacher_target_logprob=_as_cpu_values(teacher_scores.sampled_log_probs, coords),
            student_entropy=_as_cpu_values(student_scores.entropies, coords),
            teacher_entropy=_as_cpu_values(teacher_scores.entropies, coords),
            token_id=_as_cpu_values(self.rollout.response_ids, coords),
        )
        lengths = self.objective_valid.long().sum(dim=-1).cpu().tolist()
        rows: list[dict] = []
        for index, (row, position) in enumerate(coords):
            item = {name: values[index] for name, values in columns.items()}
            item["token_id"] = int(item["token_id"])
            item.update(
                schema_version=1,
                scoring_step=int(step),
                score_model_step=self.score_model_step,
                optimizer_step=None,
                rank=self.logger.rank,
                sample_id=self.sample_ids[row],
                dataset_index=int(self.dataset_indices[row]),
                local_sequence_index=int(row),
                response_position=int(position),
                sequence_length=int(lengths[row]),
                normalized_position=(position + 1) / max(int(lengths[row]), 1),
                valid_token_mask=True,
                g_plus_x=item["g"] + item["x"],
                g_plus_d=item["g"] + item["d"],
                student_token_nll=-item["student_target_logprob"],
                teacher_student_sampled_logprob_gap=(
                    item["teacher_target_logprob"] - item["student_target_logprob"]
                ),
            )
            rows.append(item)
        return rows

    def _decode(self, token_ids: list[int]) -> str | None:
        tokenizer = self.logger.tokenizer
        if tokenizer is None:
            return None
        try:
            return tokenizer.decode(
                token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            return tokenizer.decode(token_ids, skip_special_tokens=False)

    def _token_identity(self, token_id: int) -> dict[str, Any]:
        tokenizer = self.logger.tokenizer
        piece = None
        special = False
        if tokenizer is not None:
            try:
                piece = tokenizer.convert_ids_to_tokens(int(token_id))
            except (AttributeError, TypeError, ValueError):
                piece = None
            special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
            special = int(token_id) in special_ids
        return {
            "token_id": int(token_id),
            "token_piece": piece,
            "decoded_text": self._decode([int(token_id)]),
            "is_special_token": special,
        }

    def _qualitative_row_snapshot(self, row: int) -> dict[str, Any]:
        cached = self._qualitative_row_cache.get(row)
        if cached is not None:
            return cached
        response_length = int(self.objective_valid[row].sum())
        prompt_width = int(self.rollout.prompt_width)
        prompt_ids = self.rollout.input_ids[row, :prompt_width]
        prompt_mask = self.rollout.attention_mask[row, :prompt_width].bool()
        diagnostics = self.selector.diagnostics
        gain = diagnostics["gain"][row, :response_length].detach().float().cpu().tolist()
        successor = diagnostics["successor_excess"][row, :response_length].detach().float().cpu().tolist()
        sequential = diagnostics["sequential_gain"][row, :response_length].detach().float().cpu().tolist()
        cached = {
            "prompt_ids": prompt_ids[prompt_mask].detach().cpu().tolist(),
            "response_ids": self.rollout.response_ids[row, :response_length].detach().cpu().tolist(),
            "valid_positions": self.objective_valid[row, :response_length].nonzero(as_tuple=False).flatten().cpu().tolist(),
            "g": gain,
            "x": successor,
            "d": sequential,
            "g_plus_x": [g + x for g, x in zip(gain, successor)],
            "g_plus_d": [g + d for g, d in zip(gain, sequential)],
        }
        self._qualitative_row_cache[row] = cached
        return cached

    def _context_record(self, row: int, position: int) -> dict[str, Any]:
        settings = self.logger.settings
        snapshot = self._qualitative_row_snapshot(row)
        response_ids = snapshot["response_ids"]
        response_length = len(response_ids)
        prefix_ids = snapshot["prompt_ids"] + response_ids[:position]
        prefix_limit = settings["qualitative_prefix_tokens"]
        prefix_tail = prefix_ids[-prefix_limit:]
        radius = settings["qualitative_context_tokens"]
        begin = max(0, position - radius)
        end = min(response_length, position + radius + 1)
        window = []
        for response_position in range(begin, end):
            token_id = int(response_ids[response_position])
            token = self._token_identity(token_id)
            token.update(
                response_position=response_position,
                selected=response_position == position,
                g=snapshot["g"][response_position],
                x=snapshot["x"][response_position],
                d=snapshot["d"][response_position],
            )
            token["g_plus_x"] = token["g"] + token["x"]
            token["g_plus_d"] = token["g"] + token["d"]
            window.append(token)
        reference = self.reference_texts[row]
        reference_limit = settings["qualitative_reference_chars"]
        reference_text = None if reference is None else str(reference)
        reference_truncated = bool(reference_text and len(reference_text) > reference_limit)
        if reference_text is not None:
            reference_text = reference_text[:reference_limit]
        return {
            "prefix_tail_token_ids": [int(value) for value in prefix_tail],
            "prefix_tail_text": self._decode(prefix_tail),
            "prefix_truncated": len(prefix_ids) > len(prefix_tail),
            "current_token": self._token_identity(response_ids[position]),
            "context_window": window,
            "context_window_text": self._decode(response_ids[begin:end]),
            "context_window_start": begin,
            "reference_text": reference_text,
            "reference_truncated": reference_truncated,
        }

    def _counterfactual_ranking(self, row: int, position: int) -> dict[str, Any]:
        snapshot = self._qualitative_row_snapshot(row)
        valid_positions = snapshot["valid_positions"]
        values = {
            name: [snapshot[name][pos] for pos in valid_positions]
            for name in ("g", "g_plus_x", "g_plus_d")
        }
        selected_index = valid_positions.index(position)
        response_ids = snapshot["response_ids"]
        top_count = self.logger.settings["qualitative_rank_top_k"]
        result: dict[str, Any] = {}
        for name, scores in values.items():
            result[f"rank_{name}"] = _rank_descending(scores, selected_index)
            ordered = sorted(range(len(scores)), key=lambda index: (-scores[index], valid_positions[index]))
            result[f"top_positions_{name}"] = [
                {
                    "response_position": int(valid_positions[index]),
                    "score": float(scores[index]),
                    **self._token_identity(int(response_ids[valid_positions[index]])),
                }
                for index in ordered[:top_count]
            ]
        result["rank_shift_g_to_g_plus_d"] = result["rank_g"] - result["rank_g_plus_d"]
        result["rank_shift_g_plus_x_to_g_plus_d"] = (
            result["rank_g_plus_x"] - result["rank_g_plus_d"]
        )
        return result

    def _candidate_direction(
        self,
        before: list[dict[str, Any]],
        after: list[dict[str, Any]],
        target_token_id: int,
    ) -> list[dict[str, Any]]:
        before_by_id = {int(item["token_id"]): item for item in before}
        after_by_id = {int(item["token_id"]): item for item in after}
        token_ids = sorted(set(before_by_id) | set(after_by_id))
        rows = []
        for token_id in token_ids:
            old = before_by_id.get(token_id, {})
            new = after_by_id.get(token_id, {})
            p_before = old.get("student_probability")
            p_after = new.get("student_probability")
            rows.append({
                **self._token_identity(token_id),
                "is_target_token": token_id == target_token_id,
                "p_student_before": p_before,
                "p_student_after": p_after,
                "delta_p": (
                    p_after - p_before
                    if p_before is not None and p_after is not None else None
                ),
                "logp_student_before": old.get("student_logprob"),
                "logp_student_after": new.get("student_logprob"),
                "p_student_conditional_before": old.get("student_conditional_probability"),
                "p_student_conditional_after": new.get("student_conditional_probability"),
                "p_teacher": old.get("teacher_probability", new.get("teacher_probability")),
                "p_teacher_conditional": old.get(
                    "teacher_conditional_probability",
                    new.get("teacher_conditional_probability"),
                ),
            })
        for key, rank_key in (
            ("p_student_before", "rank_before"),
            ("p_student_after", "rank_after"),
        ):
            available = [index for index, row in enumerate(rows) if row[key] is not None]
            ordered = sorted(available, key=lambda index: (-rows[index][key], rows[index]["token_id"]))
            for rank, index in enumerate(ordered, 1):
                rows[index][rank_key] = rank
        for row in rows:
            if row.get("rank_before") is not None and row.get("rank_after") is not None:
                row["delta_rank"] = row["rank_before"] - row["rank_after"]
        top_k = self.logger.settings["qualitative_candidate_top_k"]
        keep: set[int] = {index for index, row in enumerate(rows) if row["is_target_token"]}
        for key in ("p_student_before", "p_student_after", "p_teacher"):
            ranked = sorted(
                (index for index, row in enumerate(rows) if row.get(key) is not None),
                key=lambda index: (-rows[index][key], rows[index]["token_id"]),
            )
            keep.update(ranked[:top_k])
        changed = sorted(
            (index for index, row in enumerate(rows) if row.get("delta_p") is not None),
            key=lambda index: (-abs(rows[index]["delta_p"]), rows[index]["token_id"]),
        )
        keep.update(changed[:top_k])
        maximum = self.logger.settings["qualitative_max_candidates"]
        if len(keep) > maximum:
            target_indices = [index for index in keep if rows[index]["is_target_token"]]
            remaining = sorted(
                (index for index in keep if index not in target_indices),
                key=lambda index: (-abs(rows[index].get("delta_p") or 0.0), rows[index]["token_id"]),
            )
            keep = set((target_indices + remaining)[:maximum])
        selected = [row for index, row in enumerate(rows) if index in keep]
        return sorted(
            selected,
            key=lambda row: (-(abs(row.get("delta_p") or 0.0)), row["token_id"]),
        )

    def on_training_chunk(
        self,
        step: int,
        chunk_indices: torch.Tensor,
        per_position_loss: torch.Tensor,
        chunk_weights: torch.Tensor,
        valid_chunk: torch.Tensor,
    ) -> None:
        if self._pending is not None:
            for chunk_row, original_index in enumerate(chunk_indices.detach().cpu().tolist()):
                for row, position in self._pending["coords"]:
                    if row == original_index and position < valid_chunk.shape[1] and bool(valid_chunk[chunk_row, position]):
                        self._pending["losses"][(row, position)] = float(per_position_loss[chunk_row, position].detach())
        if not self._token_lookup:
            return
        for chunk_row, original_index in enumerate(chunk_indices.detach().cpu().tolist()):
            for (row, position), record in self._token_lookup.items():
                if row != original_index or position >= valid_chunk.shape[1]:
                    continue
                if bool(valid_chunk[chunk_row, position]):
                    record["optimizer_step"] = int(step)
                    record["opd_ppo_loss_before"] = float(per_position_loss[chunk_row, position].detach())
                    record["training_weight"] = float(chunk_weights[chunk_row, position].detach())

    def _probe_scores(self, model, coords: list[tuple[int, int]]) -> list[dict[str, Any]]:
        """Score the same fixed prefixes/support with model weights at this instant."""
        rows = sorted({row for row, _ in coords})
        row_map = {row: index for index, row in enumerate(rows)}
        index = torch.tensor(rows, dtype=torch.long, device=self.rollout.input_ids.device)
        downstream_width = max(
            self.logger.settings["downstream_window_tokens"],
            max(self.logger.settings["qualitative_horizons"], default=0),
        )
        width = min(
            self.rollout.response_ids.shape[1],
            max(position for _, position in coords) + 1
            + downstream_width,
        )
        sampled = RolloutBatch(
            input_ids=self.rollout.input_ids.index_select(0, index),
            attention_mask=self.rollout.attention_mask.index_select(0, index),
            response_ids=self.rollout.response_ids.index_select(0, index)[:, :width],
            valid_mask=self.rollout.valid_mask.index_select(0, index)[:, :width],
            rollout_log_probs=self.rollout.rollout_log_probs.index_select(0, index)[:, :width],
            prompt_width=self.rollout.prompt_width,
        )
        candidate_ids = self.selector.candidate_ids.index_select(0, index)[:, :width]
        support = self.selector.support_mask.index_select(0, index)[:, :width]
        teacher_logp = self.selector.teacher_candidate_log_probs.index_select(0, index)[:, :width]
        support_width = candidate_ids.shape[-1]
        probe_candidate_ids = candidate_ids
        if self.logger.settings["qualitative_enabled"]:
            # The sampled target is appended so its full-vocabulary probability
            # can be compared even when it falls outside the Top-K union.
            probe_candidate_ids = torch.cat(
                (candidate_ids, sampled.response_ids.unsqueeze(-1)), dim=-1
            )
        was_training = model.training
        try:
            with torch.inference_mode():
                scores = score_original_rollout(
                    model,
                    sampled,
                    retain_response_logits=False,
                    top_k=(
                        self.logger.settings["qualitative_candidate_top_k"]
                        if self.logger.settings["qualitative_enabled"] else 0
                    ),
                    candidate_ids=probe_candidate_ids,
                    temperature=self.temperature,
                    micro_batch_size=None,
                    length_bucketed=False,
                )
                if scores.candidate_log_probs is None:
                    raise AssertionError("Probe did not return union-support log probabilities")
                union_student_logp = scores.candidate_log_probs[..., :support_width]
                p_log = torch.log_softmax(
                    union_student_logp.float().masked_fill(~support, -torch.inf), dim=-1
                )
                q_log = torch.log_softmax(
                    teacher_logp.float().masked_fill(~support, -torch.inf), dim=-1
                )
                p = p_log.exp()
                kl = torch.where(support, p * (p_log - q_log), torch.zeros_like(p)).sum(dim=-1)
                result = []
                for row, position in coords:
                    local = row_map[row]
                    sequence_end = int(sampled.valid_mask[local].sum())
                    future_end = min(
                        sequence_end,
                        position + 1 + self.logger.settings["downstream_window_tokens"],
                    )
                    future = slice(position + 1, future_end)
                    item: dict[str, Any] = {
                        "student_target_logprob": float(scores.sampled_log_probs[local, position]),
                        "student_token_nll": -float(scores.sampled_log_probs[local, position]),
                        "student_entropy": float(scores.entropies[local, position]),
                        "kl_support_reverse": float(kl[local, position]),
                        "future_kl_mean": (
                            float(kl[local, future].mean()) if future_end > position + 1 else None
                        ),
                        "future_nll_mean": (
                            -float(scores.sampled_log_probs[local, future].mean())
                            if future_end > position + 1 else None
                        ),
                    }
                    for horizon in self.logger.settings["qualitative_horizons"]:
                        horizon_end = min(sequence_end, position + 1 + horizon)
                        horizon_slice = slice(position + 1, horizon_end)
                        item[f"future_kl_mean_h{horizon}"] = (
                            float(kl[local, horizon_slice].mean())
                            if horizon_end > position + 1 else None
                        )
                        item[f"future_nll_mean_h{horizon}"] = (
                            -float(scores.sampled_log_probs[local, horizon_slice].mean())
                            if horizon_end > position + 1 else None
                        )
                    if self.logger.settings["qualitative_enabled"]:
                        if scores.top_k_ids is None or scores.top_k_log_probs is None:
                            raise AssertionError("Qualitative probe did not return student Top-K")
                        item["student_top_k"] = [
                            {
                                "token_id": int(token_id),
                                "student_logprob": float(log_probability),
                                "student_probability": math.exp(float(log_probability)),
                            }
                            for token_id, log_probability in zip(
                                scores.top_k_ids[local, position].detach().cpu().tolist(),
                                scores.top_k_log_probs[local, position].detach().float().cpu().tolist(),
                            )
                        ]
                        teacher_mass_tensor = self.selector.diagnostics.get("teacher_support_mass")
                        teacher_mass = (
                            float(teacher_mass_tensor[row, position].detach())
                            if torch.is_tensor(teacher_mass_tensor) else 1.0
                        )
                        target_id = int(self.rollout.response_ids[row, position].detach())
                        target_teacher_logp = float(
                            self.teacher_scores.sampled_log_probs[row, position].detach()
                        )
                        candidates: dict[int, dict[str, Any]] = {}
                        for candidate_index in range(support_width):
                            if not bool(support[local, position, candidate_index]):
                                continue
                            token_id = int(candidate_ids[local, position, candidate_index].detach())
                            conditional_teacher = float(q_log[local, position, candidate_index].exp())
                            entry = {
                                "token_id": token_id,
                                "student_logprob": float(
                                    union_student_logp[local, position, candidate_index]
                                ),
                                "student_probability": float(
                                    union_student_logp[local, position, candidate_index].exp()
                                ),
                                "student_conditional_probability": float(
                                    p_log[local, position, candidate_index].exp()
                                ),
                                "teacher_conditional_probability": conditional_teacher,
                                "teacher_probability": conditional_teacher * teacher_mass,
                            }
                            candidates[token_id] = entry
                        target_logp = float(scores.sampled_log_probs[local, position])
                        target_entry = candidates.setdefault(target_id, {"token_id": target_id})
                        target_entry.update(
                            student_logprob=target_logp,
                            student_probability=math.exp(target_logp),
                            teacher_probability=math.exp(target_teacher_logp),
                        )
                        item["candidate_distribution"] = list(candidates.values())
                    result.append(item)
                return result
        finally:
            model.train(was_training)

    def before_optimizer_step(
        self, step: int, model, indices: torch.Tensor, ppo_valid: torch.Tensor,
        ppo_weights: torch.Tensor,
    ) -> None:
        if not self.logger.settings["measure_learning_progress"]:
            return
        if step % self.logger.settings["learning_progress_every_n_steps"]:
            return
        original_rows = indices.detach().cpu().tolist()
        local_mask = torch.zeros_like(self.objective_valid)
        possible = []
        position_cap = self.logger.settings["max_probe_response_position"]
        allowed_positions = torch.arange(ppo_valid.shape[1], device=ppo_valid.device) < position_cap
        for slot, row in enumerate(original_rows):
            eligible = ppo_valid[slot] & allowed_positions
            if bool(eligible.any()):
                local_mask[row] |= eligible
                if row not in possible:
                    possible.append(row)
        coords = self._sample_coordinates(local_mask, possible, step)
        if not coords:
            raise AssertionError("Every rank must have valid tokens in a PPO minibatch")
        slot_for_row = {
            row: slot for slot, row in enumerate(original_rows)
            if bool(ppo_valid[slot].any())
        }
        weights = {
            (row, position): float(ppo_weights[slot_for_row[row], position].detach())
            for row, position in coords
        }
        before = self._probe_scores(model, coords)
        self._pending = {"step": step, "coords": coords, "before": before, "weights": weights, "losses": {}}

    def after_optimizer_step(self, step: int, model, gradient_norm: float) -> None:
        for record in self.token_rows:
            if record.get("optimizer_step") == step:
                record["gradient_norm"] = float(gradient_norm)
        if self._pending is None:
            return
        if self._pending["step"] != step:
            raise AssertionError("Analysis before/after optimizer steps are misaligned")
        coords = self._pending["coords"]
        after = self._probe_scores(model, coords)
        for (row, position), before_item, after_item in zip(coords, self._pending["before"], after):
            diagnostics = self.selector.diagnostics
            g = float(diagnostics["gain"][row, position].detach())
            x = float(diagnostics["successor_excess"][row, position].detach())
            d = float(diagnostics["sequential_gain"][row, position].detach())
            before_scalars = {
                key: value for key, value in before_item.items()
                if key not in {"candidate_distribution", "student_top_k"}
            }
            after_scalars = {
                key: value for key, value in after_item.items()
                if key not in {"candidate_distribution", "student_top_k"}
            }
            record = {
                "schema_version": 1,
                "optimizer_step": int(step),
                "scoring_step": self.scoring_step,
                "score_model_step": self.score_model_step,
                "rank": self.logger.rank,
                "sample_id": self.sample_ids[row],
                "dataset_index": int(self.dataset_indices[row]),
                "response_position": int(position),
                "sequence_length": int(self.objective_valid[row].sum()),
                "normalized_position": (position + 1) / max(int(self.objective_valid[row].sum()), 1),
                "valid_token_mask": True,
                "g": g,
                "x": x,
                "d": d,
                "g_plus_x": g + x,
                "g_plus_d": g + d,
                "gradient_norm": float(gradient_norm),
                "training_weight": self._pending["weights"][(row, position)],
                "opd_ppo_loss_before": self._pending["losses"].get((row, position)),
                "teacher_target_logprob": float(self.teacher_scores.sampled_log_probs[row, position].detach()),
                "teacher_entropy": float(self.teacher_scores.entropies[row, position].detach()),
                **{f"{key}_before": value for key, value in before_scalars.items()},
                **{f"{key}_after": value for key, value in after_scalars.items()},
                "delta_nll": before_item["student_token_nll"] - after_item["student_token_nll"],
                "delta_kl": before_item["kl_support_reverse"] - after_item["kl_support_reverse"],
                "delta_future_kl": (
                    before_item["future_kl_mean"] - after_item["future_kl_mean"]
                    if before_item["future_kl_mean"] is not None else None
                ),
                "delta_future_nll": (
                    before_item["future_nll_mean"] - after_item["future_nll_mean"]
                    if before_item["future_nll_mean"] is not None else None
                ),
            }
            record["teacher_student_sampled_logprob_gap_before"] = (
                record["teacher_target_logprob"] - before_item["student_target_logprob"]
            )
            record["teacher_student_sampled_logprob_gap_after"] = (
                record["teacher_target_logprob"] - after_item["student_target_logprob"]
            )
            for horizon in self.logger.settings["qualitative_horizons"]:
                before_value = before_item[f"future_kl_mean_h{horizon}"]
                after_value = after_item[f"future_kl_mean_h{horizon}"]
                record[f"delta_future_kl_h{horizon}"] = (
                    before_value - after_value if before_value is not None else None
                )
                before_nll = before_item[f"future_nll_mean_h{horizon}"]
                after_nll = after_item[f"future_nll_mean_h{horizon}"]
                record[f"delta_future_nll_h{horizon}"] = (
                    before_nll - after_nll if before_nll is not None else None
                )
            self.progress_rows.append(record)
            if self.logger.settings["qualitative_enabled"]:
                candidates = self._candidate_direction(
                    before_item.get("candidate_distribution", []),
                    after_item.get("candidate_distribution", []),
                    int(self.rollout.response_ids[row, position].detach()),
                )
                teacher_preferred = max(
                    (candidate for candidate in candidates if candidate.get("p_teacher") is not None),
                    key=lambda candidate: candidate["p_teacher"],
                    default=None,
                )
                preferred_gain = (
                    teacher_preferred["delta_p"]
                    if teacher_preferred is not None else None
                )
                def decoded_top_k(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
                    return [
                        {**item, **self._token_identity(int(item["token_id"]))}
                        for item in items
                    ]
                teacher_top_k = sorted(
                    (
                        candidate for candidate in candidates
                        if candidate.get("p_teacher") is not None
                    ),
                    key=lambda candidate: candidate["p_teacher"],
                    reverse=True,
                )[:self.logger.settings["qualitative_candidate_top_k"]]
                qualitative = {
                    **record,
                    "schema_version": 2,
                    "candidate_scope": (
                        "fixed rollout-time union of student/teacher Top-K plus sampled target; "
                        "ranks are within this stored candidate set"
                    ),
                    "candidate_tokens": candidates,
                    "student_top_k_before": decoded_top_k(before_item.get("student_top_k", [])),
                    "student_top_k_after": decoded_top_k(after_item.get("student_top_k", [])),
                    "teacher_top_k": teacher_top_k,
                    "target_logprob_gain": record["delta_nll"],
                    "local_reverse_kl_gain": record["delta_kl"],
                    "teacher_preferred_token": teacher_preferred,
                    "teacher_preferred_probability_gain": preferred_gain,
                    "local_direction_label": self.logger.direction_label(record["delta_kl"]),
                    "observed_update_scope": "whole PPO minibatch optimizer update",
                    **self._context_record(row, position),
                    **self._counterfactual_ranking(row, position),
                }
                for horizon in self.logger.settings["qualitative_horizons"]:
                    qualitative[f"future_direction_label_h{horizon}"] = self.logger.direction_label(
                        qualitative[f"delta_future_kl_h{horizon}"]
                    )
                self.qualitative_rows.append(qualitative)
        self._pending = None


class AnalysisLogger:
    """One rank-0 raw/TensorBoard writer with one bounded gather per rollout."""

    def __init__(
        self,
        run_dir: Path,
        config: dict[str, Any],
        distributed,
        *,
        tokenizer=None,
        resume_step: int = 0,
    ):
        qualitative = dict(config.get("qualitative", {}))
        horizons = [int(value) for value in qualitative.get("future_horizons", [1, 4, 8, 16])]
        self.settings = {
            "seed": int(config.get("seed", 0)),
            "log_every_n_steps": int(config.get("log_every_n_steps", 10)),
            "save_every_n_steps": int(config.get("save_every_n_steps", 100)),
            "learning_progress_every_n_steps": int(config.get("learning_progress_every_n_steps", 100)),
            "num_sequences_to_track": int(config.get("num_sequences_to_track", 2)),
            "num_tokens_per_sequence": int(config.get("num_tokens_per_sequence", 8)),
            "max_probe_response_position": int(config.get("max_probe_response_position", 512)),
            "downstream_window_tokens": int(config.get("downstream_window_tokens", 32)),
            "save_token_statistics": bool(config.get("save_token_statistics", True)),
            "measure_learning_progress": bool(config.get("measure_learning_progress", False)),
            "tensorboard": bool(config.get("tensorboard", True)),
            "qualitative_enabled": bool(qualitative.get("enabled", False)),
            "qualitative_candidate_top_k": int(qualitative.get("candidate_top_k", 8)),
            "qualitative_max_candidates": int(qualitative.get("max_candidates", 40)),
            "qualitative_context_tokens": int(qualitative.get("context_tokens", 20)),
            "qualitative_prefix_tokens": int(qualitative.get("prefix_tokens", 128)),
            "qualitative_reference_chars": int(qualitative.get("reference_chars", 4000)),
            "qualitative_rank_top_k": int(qualitative.get("rank_top_k", 5)),
            "qualitative_tensorboard_examples": int(
                qualitative.get("tensorboard_examples", 4)
            ),
            "qualitative_direction_tolerance": float(
                qualitative.get("direction_tolerance", 1e-6)
            ),
            "qualitative_horizons": sorted(set(horizons)),
        }
        if any(value <= 0 for key, value in self.settings.items() if key in {
            "log_every_n_steps", "save_every_n_steps", "learning_progress_every_n_steps",
            "num_sequences_to_track", "num_tokens_per_sequence", "max_probe_response_position",
            "downstream_window_tokens", "qualitative_candidate_top_k",
            "qualitative_max_candidates", "qualitative_context_tokens",
            "qualitative_prefix_tokens", "qualitative_rank_top_k",
            "qualitative_reference_chars", "qualitative_tensorboard_examples",
        }):
            raise ValueError("Analysis intervals and sample sizes must be positive")
        if any(value <= 0 for value in self.settings["qualitative_horizons"]):
            raise ValueError("analysis.qualitative.future_horizons must contain positive integers")
        if self.settings["qualitative_direction_tolerance"] < 0:
            raise ValueError("analysis.qualitative.direction_tolerance must be non-negative")
        if self.settings["qualitative_enabled"] and not self.settings["measure_learning_progress"]:
            raise ValueError("Qualitative logging requires analysis.measure_learning_progress=true")
        if self.settings["qualitative_enabled"] and tokenizer is None:
            raise ValueError("Qualitative logging requires the training tokenizer")
        if bool(config.get("save_full_logits", False)):
            raise ValueError("analysis.save_full_logits is unsupported: compact derived statistics only")
        self.distributed = distributed
        self.rank = distributed.rank
        self.tokenizer = tokenizer
        configured = Path(config.get("output_dir", "analysis"))
        self.root = (configured if configured.is_absolute() else run_dir / configured).resolve()
        self.writer = None
        setup_error = None
        if distributed.is_main:
            try:
                self.root.mkdir(parents=True, exist_ok=True)
                manifest = {
                    "schema_version": 2,
                    "method": "cmt",
                    "score_definitions": {
                        "g": "selector.diagnostics.gain: Var_pU(log qU - log pU)",
                        "x": "selector.diagnostics.successor_excess: R_(t+1) - g_t M_(t+1)",
                        "d": "selector.diagnostics.sequential_gain: lambda gamma marginal_flux X; one-rollout estimate",
                        "kl_support_reverse": "KL(p_U || q_U) on the fixed rollout-time union support",
                        "opd_ppo_loss_before": "actual per-position clipped OPD PPO loss before optimizer update",
                        "delta_kl": "fixed-support reverse KL immediately before minus after one optimizer step",
                        "delta_future_kl": "mean fixed-support reverse KL on the next W tokens, before minus after one optimizer step; same sampled trajectory",
                        "delta_future_kl_hH": "mean fixed-support reverse KL over the next H observed tokens, before minus after",
                        "target_logprob_gain": "log p_after(y_t) - log p_before(y_t); equal to delta_nll",
                        "teacher_preferred_probability_gain": "change in student probability for the teacher-highest token inside the stored candidate set",
                    },
                    "sampling": "deterministic per rank/step; bounded sequences and valid response positions",
                    "settings": self.settings,
                }
                (self.root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
                for subdir in ("tokens", "progress", "qualitative"):
                    destination = self.root / subdir
                    destination.mkdir(exist_ok=True)
                    if resume_step:
                        for stale in destination.glob("step-*.jsonl"):
                            if int(stale.stem.removeprefix("step-")) > resume_step:
                                stale.unlink()
                if self.settings["tensorboard"]:
                    from torch.utils.tensorboard import SummaryWriter
                    self.writer = SummaryWriter(str(self.root / "tensorboard"), purge_step=resume_step + 1 if resume_step else None)
            except Exception as error:
                setup_error = f"{type(error).__name__}: {error}"
        setup_error = distributed.broadcast_object(setup_error)
        if setup_error is not None:
            raise RuntimeError(f"Analysis logger setup failed: {setup_error}")

    def begin_rollout(self, **kwargs) -> RolloutAnalysis:
        return RolloutAnalysis(self, **kwargs)

    def direction_label(self, gain: float | None) -> str:
        if gain is None or not math.isfinite(float(gain)):
            return "unavailable"
        tolerance = self.settings["qualitative_direction_tolerance"]
        if gain > tolerance:
            return "teacher_alignment_improved"
        if gain < -tolerance:
            return "teacher_alignment_worsened"
        return "approximately_unchanged"

    def finish_rollout(self, session: RolloutAnalysis) -> None:
        if session._pending is not None:
            raise AssertionError("An analysis probe was not completed")
        if not (session.save_tokens or session.summarize_tokens or session.expect_progress):
            return
        gathered = self.distributed.all_gather_objects({
            "tokens": session.token_rows,
            "progress": session.progress_rows,
            "qualitative": session.qualitative_rows,
        })
        write_error = None
        if self.distributed.is_main:
            try:
                tokens = [row for part in gathered for row in part["tokens"]]
                progress = [row for part in gathered for row in part["progress"]]
                qualitative = [row for part in gathered for row in part["qualitative"]]
                if session.save_tokens and tokens:
                    _atomic_jsonl(self.root / "tokens" / f"step-{session.scoring_step:06d}.jsonl", tokens)
                if self.writer is not None and session.summarize_tokens:
                    self._write_tensorboard(session.scoring_step, tokens, prefix="tokens")
                for step in sorted({int(row["optimizer_step"]) for row in progress}):
                    subset = [row for row in progress if int(row["optimizer_step"]) == step]
                    _atomic_jsonl(self.root / "progress" / f"step-{step:06d}.jsonl", subset)
                    qualitative_subset = [
                        row for row in qualitative if int(row["optimizer_step"]) == step
                    ]
                    if qualitative_subset:
                        _atomic_jsonl(
                            self.root / "qualitative" / f"step-{step:06d}.jsonl",
                            qualitative_subset,
                        )
                    if self.writer is not None:
                        self._write_tensorboard(step, subset, prefix="progress")
                        if qualitative_subset:
                            self._write_qualitative_tensorboard(step, qualitative_subset)
                        for predictor in PREDICTORS:
                            for outcome in ("delta_nll", "delta_kl", "delta_future_nll", "delta_future_kl"):
                                pair = correlations(
                                    [row[predictor] for row in subset],
                                    [row[outcome] for row in subset],
                                )
                                for name in ("pearson", "spearman"):
                                    if math.isfinite(float(pair[name])):
                                        self.writer.add_scalar(f"analysis/corr_{predictor}_{outcome}_{name}", pair[name], step)
            except Exception as error:
                write_error = f"{type(error).__name__}: {error}"
        write_error = self.distributed.broadcast_object(write_error)
        if write_error is not None:
            raise RuntimeError(f"Analysis logger write failed: {write_error}")

    def _write_tensorboard(self, step: int, rows: list[dict], *, prefix: str) -> None:
        if not rows or self.writer is None:
            return
        keys = ("g", "x", "d", "g_plus_x", "g_plus_d")
        keys += (("student_token_nll", "opd_ppo_loss_before", "kl_support_reverse", "student_entropy", "teacher_entropy") if prefix == "tokens" else ("delta_nll", "delta_kl", "delta_future_nll", "delta_future_kl"))
        for key in keys:
            values = np.asarray([row[key] for row in rows if row.get(key) is not None], dtype=np.float64)
            values = values[np.isfinite(values)]
            if not values.size:
                continue
            for statistic, value in (
                ("mean", values.mean()), ("std", values.std()),
                ("min", values.min()), ("max", values.max()),
            ):
                self.writer.add_scalar(f"analysis/{prefix}/{key}_{statistic}", float(value), step)
            self.writer.add_histogram(f"analysis/{prefix}/{key}", values, step)
        self.writer.add_scalar(f"analysis/{prefix}/count", len(rows), step)

    @staticmethod
    def _qualitative_markdown(row: dict[str, Any]) -> str:
        token = row.get("current_token", {})
        decoded = token.get("decoded_text")
        if decoded is None:
            decoded = f"token_id={token.get('token_id')}"
        decoded = str(decoded).replace("|", "\\|").replace("\n", "↵")
        context = str(row.get("context_window_text") or "[decode unavailable]")
        context = context.replace("|", "\\|").replace("\n", "↵")
        return (
            f"**sample:** `{row.get('sample_id')}`  \n"
            f"**position:** {row.get('response_position')} / {row.get('sequence_length')}  \n"
            f"**token:** `{decoded}`  \n"
            f"**context:** `{context}`  \n\n"
            "| g | X | D | g+D | Δlocal reverse-KL | Δfuture-KL | rank g → g+D |\n"
            "|---:|---:|---:|---:|---:|---:|---:|\n"
            f"| {row.get('g', 0):.5g} | {row.get('x', 0):.5g} | "
            f"{row.get('d', 0):.5g} | {row.get('g_plus_d', 0):.5g} | "
            f"{row.get('delta_kl', float('nan')):.5g} | "
            f"{row.get('delta_future_kl', float('nan')) if row.get('delta_future_kl') is not None else 'n/a'} | "
            f"{row.get('rank_g')} → {row.get('rank_g_plus_d')} |"
        )

    def _write_qualitative_tensorboard(self, step: int, rows: list[dict[str, Any]]) -> None:
        if self.writer is None or not rows:
            return
        limit = self.settings["qualitative_tensorboard_examples"]
        categories: dict[str, list[dict[str, Any]]] = {
            "high_D": sorted(rows, key=lambda row: row["d"], reverse=True)[:limit],
            "g_vs_gD_ranking_change": sorted(
                rows,
                key=lambda row: abs(row.get("rank_shift_g_to_g_plus_d", 0)),
                reverse=True,
            )[:limit],
            "failure_cases": sorted(
                [
                    row for row in rows
                    if row.get("delta_future_kl") is not None
                    and row["d"] * row["delta_future_kl"] < 0
                ],
                key=lambda row: abs(row["d"] * row["delta_future_kl"]),
                reverse=True,
            )[:limit],
        }
        # Closest-g pair with the largest D gap in this optimizer update.
        if len(rows) >= 2:
            scale = max(max(row["g"] for row in rows) - min(row["g"] for row in rows), 1e-12)
            pairs = [
                (abs(left["d"] - right["d"]), left, right)
                for index, left in enumerate(rows)
                for right in rows[index + 1:]
                if abs(left["g"] - right["g"]) / scale <= 0.05
            ]
            if pairs:
                _, left, right = max(pairs, key=lambda item: item[0])
                categories["same_g_different_D"] = [left, right]
        for category, selected in categories.items():
            for index, row in enumerate(selected):
                self.writer.add_text(
                    f"Analysis/examples/{category}/{index + 1}",
                    self._qualitative_markdown(row),
                    step,
                )

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
