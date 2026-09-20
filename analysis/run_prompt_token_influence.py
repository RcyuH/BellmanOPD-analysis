"""Train prompt OPD and analyze each token with fast influence or exact probes.

Every distributed worker owns a complete student and teacher. Fast mode is
the default; exact full-test interventions remain available for validation.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import yaml

from b200_experiment.config import (
    apply_overrides,
    load_config,
    resolve_runtime_paths,
    save_config,
)
from b200_experiment.data import (
    epoch_batch_indices,
    filter_overlong_prompt_records,
    read_records,
    stable_sample_id,
    tokenize_prompts,
    validate_prompt_records,
)
from b200_experiment.evaluation import load_benchmark, render_evaluation_prompt
from b200_experiment.models import load_models
from b200_experiment.opd_core import OPD_LOSS_TOP_K

from .prompt_token_influence import (
    DISTANCE_NAME,
    SCHEMA_VERSION,
    TOKEN_SEMANTICS,
    apply_sgd_update,
    atomic_json,
    atomic_jsonl,
    build_prompt_opd_reference,
    clone_parameters,
    competition_math_teacher_distance,
    decode_token,
    gradient_l2_norm,
    improvement_record,
    prompt_opd_losses,
    restore_parameters,
    trainable_parameters,
)
from .fast_prompt_token_influence import (
    FAST_DISTANCE_NAME,
    SparseHeadDirection,
    local_teacher_topk_head_gradient,
    prompt_head_first_order_scores,
)


@dataclass(frozen=True)
class DistributedRuntime:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    initialized_here: bool

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @classmethod
    def initialize(cls) -> "DistributedRuntime":
        if not torch.cuda.is_available():
            raise RuntimeError("The exact intervention experiment requires CUDA")
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank < 0 or local_rank >= torch.cuda.device_count():
            raise ValueError(
                f"LOCAL_RANK={local_rank} is invalid for "
                f"{torch.cuda.device_count()} visible CUDA devices"
            )
        torch.cuda.set_device(local_rank)
        initialized_here = False
        if world_size > 1 and not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
            initialized_here = True
        if world_size > 1 and (
            not dist.is_initialized()
            or dist.get_world_size() != world_size
            or dist.get_rank() != rank
        ):
            raise RuntimeError("torchrun environment and process group disagree")
        return cls(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            device=torch.device("cuda", local_rank),
            initialized_here=initialized_here,
        )

    def barrier(self) -> None:
        if self.world_size > 1:
            dist.barrier()

    def close(self) -> None:
        if self.initialized_here and dist.is_initialized():
            dist.destroy_process_group()


def assigned_prompt_positions(
    num_tokens: int, rank: int, world_size: int
) -> list[int]:
    """Return one deterministic, disjoint round-robin token shard."""
    if num_tokens < 0:
        raise ValueError("num_tokens cannot be negative")
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("rank/world_size are invalid")
    return list(range(rank, num_tokens, world_size))


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _settings(config: dict[str, Any]) -> dict[str, Any]:
    settings = dict(config.get("prompt_token_influence", {}))
    if not settings.get("enabled", True):
        raise ValueError("prompt_token_influence.enabled must be true")
    if int(settings.get("train_batch_size", 1)) != 1:
        raise ValueError(
            "Exact per-prompt attribution requires train_batch_size=1 so the "
            "uniform baseline is exactly x_1+...+x_N for one prompt"
        )
    if str(settings.get("actual_update", "uniform_mean")) != "uniform_mean":
        raise ValueError("actual_update must be 'uniform_mean'")
    if int(config.get("selector", {}).get("top_k", OPD_LOSS_TOP_K)) != OPD_LOSS_TOP_K:
        raise ValueError(
            f"The repository OPD objective requires selector.top_k={OPD_LOSS_TOP_K}"
        )
    if settings.get("student_parameters_float32", True) is not True:
        raise ValueError(
            "student_parameters_float32 must be true: BF16 can round the "
            "requested small SGD updates to zero"
        )
    mode = str(settings.get("mode", "fast_first_order"))
    if mode not in {"fast_first_order", "exact"}:
        raise ValueError("prompt_token_influence.mode must be fast_first_order or exact")
    distance_name = (
        FAST_DISTANCE_NAME if mode == "fast_first_order" else DISTANCE_NAME
    )
    if str(settings.get("distance", distance_name)) != distance_name:
        raise ValueError(f"prompt_token_influence.distance must be {distance_name!r}")
    return settings


def _output_root(settings: dict[str, Any]) -> Path:
    root = Path(str(settings.get("output_dir", "outputs/prompt_token_influence")))
    return root.expanduser().resolve()


def _prepare_output(
    output: Path, config: dict[str, Any], runtime: DistributedRuntime
) -> None:
    """Let rank zero create output and broadcast startup failures to peers."""
    status: list[str | None] = [None]
    if runtime.is_main:
        try:
            if output.exists() and any(output.iterdir()):
                raise FileExistsError(
                    f"Output directory is not empty: {output}. Use a new "
                    "output_dir so branches from different runs cannot mix."
                )
            output.mkdir(parents=True, exist_ok=True)
            save_config(config, output / "resolved_config.yaml")
        except Exception as error:
            status[0] = f"{type(error).__name__}: {error}"
    if runtime.world_size > 1:
        dist.broadcast_object_list(status, src=0)
    if status[0] is not None:
        raise RuntimeError(status[0])
    runtime.barrier()


def _distance_kwargs(
    settings: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    return {
        "device": device,
        "max_prompt_tokens": int(settings.get("max_eval_prompt_tokens", 2048)),
        "student_temperature": float(
            settings.get("distance_student_temperature", 1.0)
        ),
        "teacher_temperature": float(
            settings.get("distance_teacher_temperature", 1.0)
        ),
        "vocab_chunk_positions": int(
            settings.get("distance_vocab_chunk_positions", 32)
        ),
    }


def _distributed_teacher_distance(
    student,
    teacher,
    tokenizer,
    test_prompts: list[str],
    *,
    settings: dict[str, Any],
    runtime: DistributedRuntime,
) -> dict[str, float | int | str]:
    """Evaluate disjoint test shards, then reduce to the exact global mean."""
    local_prompts = test_prompts[runtime.rank :: runtime.world_size]
    if not local_prompts:
        raise ValueError(
            "Competition-MATH test split must contain at least world_size prompts"
        )
    local = competition_math_teacher_distance(
        student,
        teacher,
        tokenizer,
        local_prompts,
        **_distance_kwargs(settings, runtime.device),
    )
    sums = torch.tensor(
        [
            float(local["sum"]),
            float(local["num_tokens"]),
            float(local["num_problems"]),
        ],
        dtype=torch.float64,
        device=runtime.device,
    )
    maximum = torch.tensor(
        float(local["maximum_prompt_tokens"]),
        dtype=torch.float64,
        device=runtime.device,
    )
    if runtime.world_size > 1:
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    total, token_count, problem_count = sums.tolist()
    return {
        "name": DISTANCE_NAME,
        "value": total / token_count,
        "sum": total,
        "num_tokens": int(token_count),
        "num_problems": int(problem_count),
        "maximum_prompt_tokens": int(maximum.item()),
    }


@torch.no_grad()
def _replica_signature(
    parameters: list[torch.nn.Parameter], runtime: DistributedRuntime
) -> list[float]:
    """Cheap cross-rank guard against accidental student-state divergence."""
    samples: list[torch.Tensor] = []
    for parameter in parameters:
        flat = parameter.detach().reshape(-1)
        if flat.numel():
            samples.extend(
                (flat[0].float(), flat[flat.numel() // 2].float(), flat[-1].float())
            )
    values = torch.stack(samples)
    signature = torch.stack(
        (values.double().sum(), values.double().square().sum())
    )
    if runtime.world_size > 1:
        minimum = signature.clone()
        maximum = signature.clone()
        dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        if not torch.equal(minimum, maximum):
            raise RuntimeError(
                "Replicated student parameters diverged across ranks after the "
                "uniform update; token branches would no longer share theta"
            )
    return [float(value) for value in signature.cpu().tolist()]


@torch.no_grad()
def _broadcast_student_parameters(
    parameters: list[torch.nn.Parameter], runtime: DistributedRuntime
) -> None:
    """Make rank zero's uniform update the exact next state on every worker."""
    if runtime.world_size <= 1:
        return
    for parameter in parameters:
        dist.broadcast(parameter.data, src=0)


def _refresh_fast_direction(
    student,
    teacher,
    tokenizer,
    calibration_prompts: list[str],
    *,
    settings: dict[str, Any],
    runtime: DistributedRuntime,
    step: int,
) -> SparseHeadDirection:
    fast = dict(settings.get("fast", {}))
    local_prompts = calibration_prompts[runtime.rank :: runtime.world_size]

    def progress(completed: int, total: int) -> None:
        if runtime.is_main and (
            completed == total or completed % max(1, total // 4) == 0
        ):
            print(
                json.dumps(
                    {
                        "event": "test_gradient_refresh_progress",
                        "step": step,
                        "rank": runtime.rank,
                        "completed_local_problems": completed,
                        "total_local_problems": total,
                    }
                ),
                flush=True,
            )

    (
        gradient_weight,
        gradient_bias,
        distance_sum,
        token_count,
        problem_count,
        maximum_observed,
    ) = local_teacher_topk_head_gradient(
        student,
        teacher,
        tokenizer,
        local_prompts,
        device=runtime.device,
        max_prompt_tokens=int(settings.get("max_eval_prompt_tokens", 2048)),
        support_top_k=int(fast.get("support_top_k", 16)),
        student_temperature=float(
            settings.get("distance_student_temperature", 1.0)
        ),
        teacher_temperature=float(
            settings.get("distance_teacher_temperature", 1.0)
        ),
        position_chunk=int(fast.get("position_chunk", 32)),
        progress_callback=progress,
    )
    # The scorer runs under inference_mode; NCCL reduction and normalization
    # mutate these tensors after it returns, so materialize ordinary tensors.
    with torch.inference_mode(False):
        gradient_weight = gradient_weight.clone()
        if gradient_bias is not None:
            gradient_bias = gradient_bias.clone()
    statistics = torch.tensor(
        [distance_sum, token_count, problem_count],
        dtype=torch.float64,
        device=runtime.device,
    )
    maximum = torch.tensor(
        maximum_observed, dtype=torch.int64, device=runtime.device
    )
    if runtime.world_size > 1:
        dist.all_reduce(gradient_weight, op=dist.ReduceOp.SUM)
        if gradient_bias is not None:
            dist.all_reduce(gradient_bias, op=dist.ReduceOp.SUM)
        dist.all_reduce(statistics, op=dist.ReduceOp.SUM)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    global_distance_sum, global_token_count, global_problem_count = statistics.tolist()
    if global_token_count <= 0:
        raise ValueError("Fast calibration set contains no prompt tokens")
    gradient_weight.div_(global_token_count)
    if gradient_bias is not None:
        gradient_bias.div_(global_token_count)
    active_mask = gradient_weight.abs().sum(dim=-1).gt(0)
    if gradient_bias is not None:
        active_mask |= gradient_bias.ne(0)
    active_ids = active_mask.nonzero(as_tuple=False).flatten()
    active_weight = gradient_weight.index_select(0, active_ids).contiguous()
    active_bias = (
        gradient_bias.index_select(0, active_ids).contiguous()
        if gradient_bias is not None
        else None
    )
    squared_norm = active_weight.double().square().sum()
    if active_bias is not None:
        squared_norm += active_bias.double().square().sum()
    direction = SparseHeadDirection(
        active_token_ids=active_ids,
        weight=active_weight,
        bias=active_bias,
        gradient_l2_norm=float(squared_norm.sqrt().item()),
        distance_value=float(global_distance_sum / global_token_count),
        num_test_tokens=int(global_token_count),
        num_test_problems=int(global_problem_count),
        refresh_step=int(step),
        support_top_k=int(fast.get("support_top_k", 16)),
    )
    del gradient_weight, gradient_bias, active_mask, statistics, maximum
    return direction


def _gather_rows(
    local_rows: list[dict[str, Any]], runtime: DistributedRuntime
) -> list[dict[str, Any]]:
    if runtime.world_size == 1:
        return local_rows
    gathered: list[list[dict[str, Any]] | None] = [None] * runtime.world_size
    dist.all_gather_object(gathered, local_rows)
    if not runtime.is_main:
        return []
    rows = [row for rank_rows in gathered if rank_rows for row in rank_rows]
    rows.sort(
        key=lambda row: (
            row["intervention"] != "single_token",
            int(row["prompt_position"] or 0),
        )
    )
    return rows


def _exact_step_rows(
    *,
    model,
    teacher,
    tokenizer,
    encoded: dict[str, torch.Tensor],
    reference,
    test_prompts: list[str],
    parameters: list[torch.nn.Parameter],
    snapshot: list[torch.Tensor],
    config: dict[str, Any],
    settings: dict[str, Any],
    runtime: DistributedRuntime,
    seed: int,
    step: int,
    epoch: int,
    dataset_index: int,
    sample_id: str,
) -> tuple[list[dict[str, Any]], list[float]]:
    training = config["training"]
    learning_rate = float(settings.get("learning_rate", training["learning_rate"]))
    student_temperature = float(config.get("rollout", {}).get("temperature", 1.0))
    teacher_temperature = float(config.get("opd", {}).get("teacher_temperature", 1.0))
    baseline = _distributed_teacher_distance(
        model,
        teacher,
        tokenizer,
        test_prompts,
        settings=settings,
        runtime=runtime,
    )
    baseline_value = float(baseline["value"])
    valid_coordinates = encoded["attention_mask"].bool().nonzero(as_tuple=False).tolist()
    token_ids = encoded["input_ids"]
    common = {
        "schema_version": SCHEMA_VERSION,
        "optimizer_step_before": int(step),
        "optimizer_step_after": int(step + 1),
        "epoch": int(epoch),
        "dataset_index": int(dataset_index),
        "sample_id": str(sample_id),
        "prompt_num_tokens": len(valid_coordinates),
        "token_semantics": TOKEN_SEMANTICS,
        "distance_name": DISTANCE_NAME,
        "distance_num_test_problems": int(baseline["num_problems"]),
        "distance_num_test_tokens": int(baseline["num_tokens"]),
        "learning_rate": learning_rate,
        "opd_top_k": OPD_LOSS_TOP_K,
        "opd_student_temperature": student_temperature,
        "opd_teacher_temperature": teacher_temperature,
        "distributed_world_size": runtime.world_size,
    }
    local_rows: list[dict[str, Any]] = []
    model.train()
    branch_seed = seed + 1_000_003 * step
    for prompt_position in assigned_prompt_positions(
        len(valid_coordinates), runtime.rank, runtime.world_size
    ):
        batch_index, padded_position = valid_coordinates[prompt_position]
        restore_parameters(parameters, snapshot)
        # Equal RNG state gives every token branch the same dropout masks.
        _seed_everything(branch_seed)
        losses = prompt_opd_losses(
            model,
            encoded["input_ids"],
            encoded["attention_mask"],
            reference,
            student_temperature=student_temperature,
            clip_low=float(training.get("ppo_clip_low", 0.2)),
            clip_high=float(training.get("ppo_clip_high", 0.28)),
            dual_clip=float(training.get("ppo_dual_clip", 3.0)),
            chunk_steps=int(config.get("selector", {}).get("score_chunk_steps", 128)),
        )
        loss = losses[batch_index, padded_position]
        gradients = torch.autograd.grad(
            loss, parameters, allow_unused=True, materialize_grads=False
        )
        loss_value = float(loss.detach().float().item())
        grad_norm = gradient_l2_norm(gradients)
        apply_sgd_update(parameters, gradients, learning_rate)
        # The virtual update is now materialized in the parameters. Release
        # the full prompt autograd graph and full-model gradient before the
        # long Competition-MATH test pass.
        del losses, loss, gradients
        # The owning rank evaluates its branch on the complete test split.
        after = competition_math_teacher_distance(
            model,
            teacher,
            tokenizer,
            test_prompts,
            **_distance_kwargs(settings, runtime.device),
        )
        identity = decode_token(tokenizer, int(token_ids[batch_index, padded_position]))
        local_rows.append(
            {
                **common,
                "worker_rank": runtime.rank,
                "intervention": "single_token",
                "prompt_position": int(prompt_position),
                "padded_tensor_position": int(padded_position),
                **identity,
                "opd_loss": loss_value,
                "gradient_l2_norm": grad_norm,
                **improvement_record(baseline_value, float(after["value"])),
            }
        )
        del after

    # All replicas restore theta_s. Rank zero alone computes the real uniform
    # update, then broadcasts its exact parameters. This is more robust than
    # relying on nominally identical kernels/dropout across eight processes.
    restore_parameters(parameters, snapshot)
    uniform_loss_value = 0.0
    uniform_grad_norm = 0.0
    if runtime.is_main:
        _seed_everything(branch_seed)
        losses = prompt_opd_losses(
            model,
            encoded["input_ids"],
            encoded["attention_mask"],
            reference,
            student_temperature=student_temperature,
            clip_low=float(training.get("ppo_clip_low", 0.2)),
            clip_high=float(training.get("ppo_clip_high", 0.28)),
            dual_clip=float(training.get("ppo_dual_clip", 3.0)),
            chunk_steps=int(
                config.get("selector", {}).get("score_chunk_steps", 128)
            ),
        )
        valid_mask = encoded["attention_mask"].bool()
        uniform_loss = losses[valid_mask].mean()
        uniform_gradients = list(
            torch.autograd.grad(
                uniform_loss,
                parameters,
                allow_unused=True,
                materialize_grads=False,
            )
        )
        uniform_loss_value = float(uniform_loss.detach().float().item())
        uniform_grad_norm = gradient_l2_norm(uniform_gradients)
        apply_sgd_update(parameters, uniform_gradients, learning_rate)
        del losses, uniform_loss, uniform_gradients
    _broadcast_student_parameters(parameters, runtime)
    uniform_metrics = torch.tensor(
        [uniform_loss_value, uniform_grad_norm],
        dtype=torch.float64,
        device=runtime.device,
    )
    if runtime.world_size > 1:
        dist.broadcast(uniform_metrics, src=0)
    uniform_loss_value, uniform_grad_norm = uniform_metrics.tolist()
    uniform_after = _distributed_teacher_distance(
        model,
        teacher,
        tokenizer,
        test_prompts,
        settings=settings,
        runtime=runtime,
    )
    signature = _replica_signature(parameters, runtime)
    if runtime.is_main:
        local_rows.append(
            {
                **common,
                "worker_rank": None,
                "intervention": "uniform_mean_all_prompt_tokens",
                "prompt_position": None,
                "padded_tensor_position": None,
                "token_id": None,
                "token_piece": None,
                "decoded_text": None,
                "visible_text": "UNIFORM(x_1,...,x_N)",
                "decoded_utf8_hex": None,
                "is_special_token": None,
                "opd_loss": uniform_loss_value,
                "gradient_l2_norm": uniform_grad_norm,
                "replica_signature_after": signature,
                **improvement_record(baseline_value, float(uniform_after["value"])),
            }
        )
    del uniform_after
    return local_rows, signature


def _fast_step_rows(
    *,
    model,
    tokenizer,
    encoded: dict[str, torch.Tensor],
    reference,
    direction: SparseHeadDirection,
    parameters: list[torch.nn.Parameter],
    config: dict[str, Any],
    settings: dict[str, Any],
    runtime: DistributedRuntime,
    seed: int,
    step: int,
    epoch: int,
    dataset_index: int,
    sample_id: str,
) -> tuple[list[dict[str, Any]], list[float]]:
    """Score every token vectorially, then apply one real uniform SGD update."""
    training = config["training"]
    learning_rate = float(settings.get("learning_rate", training["learning_rate"]))
    student_temperature = float(config.get("rollout", {}).get("temperature", 1.0))
    teacher_temperature = float(config.get("opd", {}).get("teacher_temperature", 1.0))
    valid_coordinates = encoded["attention_mask"].bool().nonzero(as_tuple=False).tolist()
    token_ids = encoded["input_ids"]
    common = {
        "schema_version": SCHEMA_VERSION,
        "optimizer_step_before": int(step),
        "optimizer_step_after": int(step + 1),
        "epoch": int(epoch),
        "dataset_index": int(dataset_index),
        "sample_id": str(sample_id),
        "prompt_num_tokens": len(valid_coordinates),
        "token_semantics": TOKEN_SEMANTICS,
        "influence_method": "first_order_output_head",
        "is_exact_intervention": False,
        "distance_name": FAST_DISTANCE_NAME,
        "distance_scope": "fixed_competition_math_calibration_subset",
        "distance_gradient_refresh_step": direction.refresh_step,
        "distance_num_test_problems": direction.num_test_problems,
        "distance_num_test_tokens": direction.num_test_tokens,
        "distance_support_top_k": direction.support_top_k,
        "distance_gradient_l2_norm": direction.gradient_l2_norm,
        "learning_rate": learning_rate,
        "opd_top_k": OPD_LOSS_TOP_K,
        "opd_student_temperature": student_temperature,
        "opd_teacher_temperature": teacher_temperature,
        "distributed_world_size": runtime.world_size,
    }
    rows: list[dict[str, Any]] = []
    branch_seed = seed + 1_000_003 * step
    model.train()
    if runtime.is_main:
        _seed_everything(branch_seed)
        token_losses, inner_products = prompt_head_first_order_scores(
            model,
            encoded["input_ids"],
            encoded["attention_mask"],
            reference,
            direction,
            student_temperature=student_temperature,
            clip_low=float(training.get("ppo_clip_low", 0.2)),
            clip_high=float(training.get("ppo_clip_high", 0.28)),
            dual_clip=float(training.get("ppo_dual_clip", 3.0)),
            chunk_steps=int(config.get("selector", {}).get("score_chunk_steps", 128)),
        )
        for prompt_position, (batch_index, padded_position) in enumerate(
            valid_coordinates
        ):
            predicted = learning_rate * float(
                inner_products[batch_index, padded_position].float().item()
            )
            identity = decode_token(
                tokenizer, int(token_ids[batch_index, padded_position])
            )
            rows.append(
                {
                    **common,
                    "worker_rank": 0,
                    "intervention": "single_token_first_order",
                    "prompt_position": int(prompt_position),
                    "padded_tensor_position": int(padded_position),
                    **identity,
                    "opd_loss": float(
                        token_losses[batch_index, padded_position].float().item()
                    ),
                    "gradient_l2_norm": None,
                    "test_train_gradient_inner_product": float(
                        inner_products[batch_index, padded_position].float().item()
                    ),
                    "predicted_distance_improvement": predicted,
                    "calibration_distance_at_refresh": direction.distance_value,
                    "distance_before": None,
                    "distance_after": None,
                    "distance_improvement": None,
                    "relative_distance_improvement": None,
                }
            )
        uniform_predicted = learning_rate * float(
            inner_products[encoded["attention_mask"].bool()].float().mean().item()
        )
        del token_losses, inner_products
    else:
        uniform_predicted = 0.0

    # Rank zero performs the real all-parameter uniform update. This training
    # path is unchanged by the output-head approximation used only for ranking.
    uniform_loss_value = 0.0
    uniform_grad_norm = 0.0
    if runtime.is_main:
        _seed_everything(branch_seed)
        losses = prompt_opd_losses(
            model,
            encoded["input_ids"],
            encoded["attention_mask"],
            reference,
            student_temperature=student_temperature,
            clip_low=float(training.get("ppo_clip_low", 0.2)),
            clip_high=float(training.get("ppo_clip_high", 0.28)),
            dual_clip=float(training.get("ppo_dual_clip", 3.0)),
            chunk_steps=int(
                config.get("selector", {}).get("score_chunk_steps", 128)
            ),
        )
        uniform_loss = losses[encoded["attention_mask"].bool()].mean()
        uniform_gradients = list(
            torch.autograd.grad(
                uniform_loss,
                parameters,
                allow_unused=True,
                materialize_grads=False,
            )
        )
        uniform_loss_value = float(uniform_loss.detach().float().item())
        uniform_grad_norm = gradient_l2_norm(uniform_gradients)
        apply_sgd_update(parameters, uniform_gradients, learning_rate)
        del losses, uniform_loss, uniform_gradients
    _broadcast_student_parameters(parameters, runtime)
    metrics = torch.tensor(
        [uniform_loss_value, uniform_grad_norm, uniform_predicted],
        dtype=torch.float64,
        device=runtime.device,
    )
    if runtime.world_size > 1:
        dist.broadcast(metrics, src=0)
    uniform_loss_value, uniform_grad_norm, uniform_predicted = metrics.tolist()
    signature = _replica_signature(parameters, runtime)
    if runtime.is_main:
        rows.append(
            {
                **common,
                "worker_rank": None,
                "intervention": "uniform_mean_all_prompt_tokens_first_order",
                "prompt_position": None,
                "padded_tensor_position": None,
                "token_id": None,
                "token_piece": None,
                "decoded_text": None,
                "visible_text": "UNIFORM(x_1,...,x_N)",
                "decoded_utf8_hex": None,
                "is_special_token": None,
                "opd_loss": uniform_loss_value,
                "gradient_l2_norm": uniform_grad_norm,
                "test_train_gradient_inner_product": (
                    uniform_predicted / learning_rate
                ),
                "predicted_distance_improvement": uniform_predicted,
                "replica_signature_after": signature,
                "calibration_distance_at_refresh": direction.distance_value,
                "distance_before": None,
                "distance_after": None,
                "distance_improvement": None,
                "relative_distance_improvement": None,
            }
        )
    return rows, signature


def _run(config: dict[str, Any], runtime: DistributedRuntime) -> dict[str, Any]:
    settings = _settings(config)
    mode = str(settings.get("mode", "fast_first_order"))
    expected_world_size = settings.get("expected_world_size")
    if expected_world_size is not None and runtime.world_size != int(expected_world_size):
        raise ValueError(
            f"Config expects {expected_world_size} workers, but torchrun started "
            f"WORLD_SIZE={runtime.world_size}"
        )
    seed = int(config.get("experiment", {}).get("seed", 42))
    _seed_everything(seed)
    output = _output_root(settings)
    _prepare_output(output, config, runtime)

    student, teacher, tokenizer, model_metadata = load_models(config, runtime.device)
    student.float()
    model_metadata["experiment_student_parameter_dtype"] = "float32"
    model_metadata["experiment_teacher_parameter_dtype"] = str(
        next(teacher.parameters()).dtype
    )
    parameters = trainable_parameters(student)
    initial_signature = _replica_signature(parameters, runtime)
    train_records, train_files = read_records(
        config["data"]["path"], split=config["data"].get("split")
    )
    validate_prompt_records(train_records, config["data"])
    train_records, train_filter = filter_overlong_prompt_records(
        train_records, tokenizer, config["data"]
    )
    benchmark = str(settings.get("benchmark", "Competition-MATH"))
    if benchmark != "Competition-MATH":
        raise ValueError("This experiment is fixed to benchmark='Competition-MATH'")
    test_records, test_schema = load_benchmark(
        benchmark, config["evaluation"]["benchmarks"][benchmark]
    )
    test_prompts = [
        render_evaluation_prompt(tokenizer, record, config) for record in test_records
    ]
    calibration_prompts: list[str] = []
    calibration_indices: list[int] = []
    fast = dict(settings.get("fast", {}))
    if mode == "fast_first_order":
        requested_limit = int(fast.get("calibration_test_problems", 128))
        if requested_limit < runtime.world_size:
            raise ValueError(
                "fast.calibration_test_problems must be at least world_size"
            )
        calibration_count = min(requested_limit, len(test_prompts))
        calibration_indices = sorted(
            random.Random(seed + 8_000_009).sample(
                range(len(test_prompts)), calibration_count
            )
        )
        calibration_prompts = [test_prompts[index] for index in calibration_indices]
    epochs = int(settings.get("epochs", 1))
    if epochs <= 0:
        raise ValueError("prompt_token_influence.epochs must be positive")
    total_steps = len(train_records) * epochs
    configured_max = settings.get("max_steps")
    if configured_max is not None:
        total_steps = min(total_steps, int(configured_max))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "prompt_token_opd_influence",
        "mode": mode,
        "token_semantics": TOKEN_SEMANTICS,
        "distributed": {
            "strategy": (
                "replicated_models_token_branch_parallelism"
                if mode == "exact"
                else "replicated_models_sharded_test_gradient"
            ),
            "world_size": runtime.world_size,
            "token_assignment": (
                "prompt_position modulo world_size"
                if mode == "exact"
                else "all-token vectorized first-order scoring on rank zero"
            ),
            "test_gradient_evaluation": "calibration-test-sharded all-reduce",
            "student_update": "uniform SGD on rank zero, then exact parameter broadcast",
            "replica_guard": "sampled parameter signature equality after every step",
            "initial_replica_signature": initial_signature,
        },
        "distance": (
            {
                "name": DISTANCE_NAME,
                "direction": "KL(teacher || student)",
                "vocabulary": "full",
                "reduction": "mean over every rendered test-prompt token",
                "benchmark": benchmark,
                "num_test_problems": len(test_records),
                "sampling": False,
                "truncation": False,
            }
            if mode == "exact"
            else {
                "name": FAST_DISTANCE_NAME,
                "direction": "conditional KL(teacher || student)",
                "parameter_scope": "student output projection",
                "support_top_k": int(fast.get("support_top_k", 16)),
                "benchmark": benchmark,
                "calibration_test_problems": len(calibration_prompts),
                "calibration_indices": calibration_indices,
                "gradient_refresh_interval": int(
                    fast.get("test_gradient_interval", 50)
                ),
                "token_scoring": "learning_rate * <grad D, grad L_t>",
                "exact": False,
            }
        ),
        "interventions": (
            {
                "single_token": "theta - learning_rate * grad(L_t)",
                "uniform": "theta - learning_rate * grad(mean_t L_t)",
                "branching": "all branches at a step start from identical theta",
                "actual_training_update": "uniform branch",
            }
            if mode == "exact"
            else {
                "single_token": "first-order prediction only; no virtual model update",
                "formula": "learning_rate * <grad D_test, grad L_t>",
                "parameter_scope": "student output projection",
                "all_tokens_scored": True,
                "actual_training_update": "all-parameter uniform SGD branch",
            }
        ),
        "train": {
            "files": [str(path) for path in train_files],
            "num_records_after_filter": len(train_records),
            "filter": train_filter,
            "epochs": epochs,
            "planned_steps": total_steps,
            "shuffle_seed": seed,
            "uses_every_token": True,
            "token_sampling": False,
        },
        "test_schema": test_schema,
        "model": model_metadata,
        "outputs": {
            "token_interventions": "steps/step-XXXXXX.jsonl",
            "full_prompts": "prompts/step-XXXXXX.json",
            "checkpoints": "checkpoints/step-XXXXXX",
        },
    }
    if runtime.is_main:
        atomic_json(output / "manifest.json", manifest)
    runtime.barrier()

    started = time.time()
    checkpoint_interval = int(settings.get("checkpoint_interval", 0))
    fast_direction: SparseHeadDirection | None = None
    for step in range(total_steps):
        if mode == "fast_first_order" and (
            fast_direction is None
            or step % max(1, int(fast.get("test_gradient_interval", 50))) == 0
        ):
            refresh_started = time.time()
            if runtime.is_main:
                print(
                    json.dumps(
                        {
                            "event": "test_gradient_refresh_started",
                            "step": step,
                            "calibration_problems": len(calibration_prompts),
                            "world_size": runtime.world_size,
                        }
                    ),
                    flush=True,
                )
            fast_direction = _refresh_fast_direction(
                student,
                teacher,
                tokenizer,
                calibration_prompts,
                settings=settings,
                runtime=runtime,
                step=step,
            )
            if runtime.is_main:
                print(
                    json.dumps(
                        {
                            "event": "test_gradient_refreshed",
                            "step": step,
                            "calibration_problems": fast_direction.num_test_problems,
                            "calibration_tokens": fast_direction.num_test_tokens,
                            "active_vocabulary_rows": int(
                                fast_direction.active_token_ids.numel()
                            ),
                            "distance": fast_direction.distance_value,
                            "gradient_l2_norm": fast_direction.gradient_l2_norm,
                            "elapsed_seconds": time.time() - refresh_started,
                        }
                    ),
                    flush=True,
                )
        dataset_index = epoch_batch_indices(len(train_records), 1, step, seed)[0]
        epoch = step // len(train_records)
        record = train_records[dataset_index]
        encoded, rendered = tokenize_prompts(
            [record], tokenizer, config["data"], runtime.device
        )
        if runtime.is_main:
            print(
                json.dumps(
                    {
                        "event": "training_step_started",
                        "step": step,
                        "mode": mode,
                        "prompt_tokens": int(encoded["attention_mask"].sum().item()),
                    }
                ),
                flush=True,
            )
        rendered_prompt = rendered[0]
        sample_id = stable_sample_id(record, dataset_index)
        reference = build_prompt_opd_reference(
            student,
            teacher,
            encoded["input_ids"],
            encoded["attention_mask"],
            top_k=OPD_LOSS_TOP_K,
            student_temperature=float(config.get("rollout", {}).get("temperature", 1.0)),
            teacher_temperature=float(config.get("opd", {}).get("teacher_temperature", 1.0)),
        )
        step_started = time.time()
        if mode == "exact":
            snapshot = clone_parameters(parameters)
            local_rows, signature = _exact_step_rows(
                model=student,
                teacher=teacher,
                tokenizer=tokenizer,
                encoded=encoded,
                reference=reference,
                test_prompts=test_prompts,
                parameters=parameters,
                snapshot=snapshot,
                config=config,
                settings=settings,
                runtime=runtime,
                seed=seed,
                step=step,
                epoch=epoch,
                dataset_index=dataset_index,
                sample_id=sample_id,
            )
            rows = _gather_rows(local_rows, runtime)
            del snapshot, local_rows
        else:
            if fast_direction is None:
                raise AssertionError("Fast influence direction was not initialized")
            rows, signature = _fast_step_rows(
                model=student,
                tokenizer=tokenizer,
                encoded=encoded,
                reference=reference,
                direction=fast_direction,
                parameters=parameters,
                config=config,
                settings=settings,
                runtime=runtime,
                seed=seed,
                step=step,
                epoch=epoch,
                dataset_index=dataset_index,
                sample_id=sample_id,
            )
        del reference
        if runtime.is_main:
            atomic_jsonl(output / "steps" / f"step-{step:06d}.jsonl", rows)
            token_ids = (
                encoded["input_ids"][0][encoded["attention_mask"][0].bool()]
                .detach()
                .cpu()
                .tolist()
            )
            atomic_json(
                output / "prompts" / f"step-{step:06d}.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "optimizer_step_before": step,
                    "epoch": epoch,
                    "dataset_index": dataset_index,
                    "sample_id": sample_id,
                    "rendered_prompt": rendered_prompt,
                    "token_ids": token_ids,
                    "num_tokens": len(token_ids),
                    "world_size": runtime.world_size,
                    "token_assignment": (
                        {
                            str(rank): assigned_prompt_positions(
                                len(token_ids), rank, runtime.world_size
                            )
                            for rank in range(runtime.world_size)
                        }
                        if mode == "exact"
                        else {"0": "all positions, vectorized first-order"}
                    ),
                    "replica_signature_after": signature,
                    "elapsed_seconds": time.time() - step_started,
                },
            )
            if checkpoint_interval > 0 and (
                (step + 1) % checkpoint_interval == 0 or step + 1 == total_steps
            ):
                checkpoint = output / "checkpoints" / f"step-{step + 1:06d}"
                checkpoint.mkdir(parents=True, exist_ok=False)
                student.save_pretrained(checkpoint, safe_serialization=True)
                tokenizer.save_pretrained(checkpoint)
            token_rows = [
                row
                for row in rows
                if row["intervention"].startswith("single_token")
            ]
            uniform_row = next(
                row
                for row in rows
                if row["intervention"].startswith("uniform_mean_all_prompt_tokens")
            )
            print(
                json.dumps(
                    {
                        "completed_step": step + 1,
                        "total_steps": total_steps,
                        "dataset_index": dataset_index,
                        "tokens": len(token_rows),
                        "world_size": runtime.world_size,
                        "mode": mode,
                        "best_token_improvement": max(
                            row[
                                "distance_improvement"
                                if mode == "exact"
                                else "predicted_distance_improvement"
                            ]
                            for row in token_rows
                        ),
                        "uniform_improvement": uniform_row[
                            "distance_improvement"
                            if mode == "exact"
                            else "predicted_distance_improvement"
                        ],
                        "elapsed_seconds": time.time() - step_started,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        runtime.barrier()
        del encoded, rows

    summary = {
        "schema_version": SCHEMA_VERSION,
        "completed_steps": total_steps,
        "completed_full_train_split": total_steps == len(train_records) * epochs,
        "world_size": runtime.world_size,
        "elapsed_seconds": time.time() - started,
        "output_dir": str(output),
    }
    if runtime.is_main:
        atomic_json(output / "summary.json", summary)
    runtime.barrier()
    return summary


def run(config: dict[str, Any]) -> dict[str, Any] | None:
    runtime = DistributedRuntime.initialize()
    try:
        summary = _run(config, runtime)
        return summary if runtime.is_main else None
    finally:
        runtime.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "configs"
            / "prompt_token_influence.yaml"
        ),
    )
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    config = apply_overrides(load_config(args.config), args.set)
    config = resolve_runtime_paths(config)
    summary = run(config)
    if summary is not None:
        print(yaml.safe_dump(summary, sort_keys=False, allow_unicode=True))


if __name__ == "__main__":
    main()
