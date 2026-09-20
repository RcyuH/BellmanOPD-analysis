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
        self.teacher_scores = teacher_scores
        self.temperature = float(temperature)
        self.progress_rows: list[dict[str, Any]] = []
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

    def _probe_scores(self, model, coords: list[tuple[int, int]]) -> list[dict[str, float]]:
        """Score the same fixed prefixes/support with model weights at this instant."""
        rows = sorted({row for row, _ in coords})
        row_map = {row: index for index, row in enumerate(rows)}
        index = torch.tensor(rows, dtype=torch.long, device=self.rollout.input_ids.device)
        width = min(
            self.rollout.response_ids.shape[1],
            max(position for _, position in coords) + 1
            + self.logger.settings["downstream_window_tokens"],
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
        was_training = model.training
        try:
            with torch.inference_mode():
                scores = score_original_rollout(
                    model,
                    sampled,
                    retain_response_logits=False,
                    candidate_ids=candidate_ids,
                    temperature=self.temperature,
                    micro_batch_size=None,
                    length_bucketed=False,
                )
                if scores.candidate_log_probs is None:
                    raise AssertionError("Probe did not return union-support log probabilities")
                p_log = torch.log_softmax(
                    scores.candidate_log_probs.float().masked_fill(~support, -torch.inf), dim=-1
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
                    result.append({
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
                    })
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
                **{f"{key}_before": value for key, value in before_item.items()},
                **{f"{key}_after": value for key, value in after_item.items()},
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
            self.progress_rows.append(record)
        self._pending = None


class AnalysisLogger:
    """One rank-0 raw/TensorBoard writer with one bounded gather per rollout."""

    def __init__(self, run_dir: Path, config: dict[str, Any], distributed, *, resume_step: int = 0):
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
        }
        if any(value <= 0 for key, value in self.settings.items() if key in {
            "log_every_n_steps", "save_every_n_steps", "learning_progress_every_n_steps",
            "num_sequences_to_track", "num_tokens_per_sequence", "max_probe_response_position",
            "downstream_window_tokens"
        }):
            raise ValueError("Analysis intervals and sample sizes must be positive")
        if bool(config.get("save_full_logits", False)):
            raise ValueError("analysis.save_full_logits is unsupported: compact derived statistics only")
        self.distributed = distributed
        self.rank = distributed.rank
        configured = Path(config.get("output_dir", "analysis"))
        self.root = (configured if configured.is_absolute() else run_dir / configured).resolve()
        self.writer = None
        setup_error = None
        if distributed.is_main:
            try:
                self.root.mkdir(parents=True, exist_ok=True)
                manifest = {
                    "schema_version": 1,
                    "method": "cmt",
                    "score_definitions": {
                        "g": "selector.diagnostics.gain: Var_pU(log qU - log pU)",
                        "x": "selector.diagnostics.successor_excess: R_(t+1) - g_t M_(t+1)",
                        "d": "selector.diagnostics.sequential_gain: lambda gamma marginal_flux X; one-rollout estimate",
                        "kl_support_reverse": "KL(p_U || q_U) on the fixed rollout-time union support",
                        "opd_ppo_loss_before": "actual per-position clipped OPD PPO loss before optimizer update",
                        "delta_kl": "fixed-support reverse KL immediately before minus after one optimizer step",
                        "delta_future_kl": "mean fixed-support reverse KL on the next W tokens, before minus after one optimizer step; same sampled trajectory",
                    },
                    "sampling": "deterministic per rank/step; bounded sequences and valid response positions",
                    "settings": self.settings,
                }
                (self.root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
                for subdir in ("tokens", "progress"):
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

    def finish_rollout(self, session: RolloutAnalysis) -> None:
        if session._pending is not None:
            raise AssertionError("An analysis probe was not completed")
        if not (session.save_tokens or session.summarize_tokens or session.expect_progress):
            return
        gathered = self.distributed.all_gather_objects({
            "tokens": session.token_rows,
            "progress": session.progress_rows,
        })
        write_error = None
        if self.distributed.is_main:
            try:
                tokens = [row for part in gathered for row in part["tokens"]]
                progress = [row for part in gathered for row in part["progress"]]
                if session.save_tokens and tokens:
                    _atomic_jsonl(self.root / "tokens" / f"step-{session.scoring_step:06d}.jsonl", tokens)
                if self.writer is not None and session.summarize_tokens:
                    self._write_tensorboard(session.scoring_step, tokens, prefix="tokens")
                for step in sorted({int(row["optimizer_step"]) for row in progress}):
                    subset = [row for row in progress if int(row["optimizer_step"]) == step]
                    _atomic_jsonl(self.root / "progress" / f"step-{step:06d}.jsonl", subset)
                    if self.writer is not None:
                        self._write_tensorboard(step, subset, prefix="progress")
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

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
