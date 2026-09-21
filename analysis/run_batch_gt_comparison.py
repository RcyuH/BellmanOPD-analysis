"""Train and compare top-10%-g_t OPD against uniform OPD batch by batch."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import yaml

from b200_experiment.config import apply_overrides, load_config, resolve_runtime_paths, save_config
from b200_experiment.data import (
    epoch_batch_indices,
    filter_overlong_prompt_records,
    read_records,
    stable_sample_id,
    tokenize_prompts,
    validate_prompt_records,
)
from b200_experiment.distributed import (
    DistributedContext,
    contiguous_partition,
    initialize_distributed,
)
from b200_experiment.evaluation import load_benchmark, render_evaluation_prompt
from b200_experiment.models import load_models, load_student_model

from .batch_gt_comparison import (
    GT_DEFINITION,
    GT_SUPPORT,
    build_prompt_pgt_reference,
    model_parameter_l2_distance,
    normalized_effective_weights,
    top_gt_weights,
    train_prompt_batch,
    uniform_weights,
)
from .prompt_token_influence import (
    DISTANCE_NAME,
    SCHEMA_VERSION,
    atomic_json,
    atomic_jsonl,
    build_prompt_opd_reference,
    decode_token,
    full_vocab_forward_kl_from_logits,
    improvement_record,
)
from .summarize_batch_gt_comparison import summarize as summarize_comparison


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _settings(config: dict[str, Any]) -> dict[str, Any]:
    settings = dict(config.get("batch_gt_comparison", {}))
    if not settings.get("enabled", True):
        raise ValueError("batch_gt_comparison.enabled must be true")
    fraction = float(settings.get("token_fraction", 0.10))
    if not 0.0 < fraction <= 1.0:
        raise ValueError("batch_gt_comparison.token_fraction must be in (0,1]")
    weighting = str(settings.get("selected_token_weighting", "binary"))
    if weighting not in {"binary", "bounded_rank"}:
        raise ValueError(
            "batch_gt_comparison.selected_token_weighting must be "
            "'binary' or 'bounded_rank'"
        )
    minimum = float(settings.get("selected_weight_min", 0.5))
    maximum = float(settings.get("selected_weight_max", 1.5))
    if not math.isfinite(minimum) or not math.isfinite(maximum):
        raise ValueError("selected token weight bounds must be finite")
    if minimum <= 0 or maximum < minimum:
        raise ValueError(
            "selected token weights require 0 < selected_weight_min <= "
            "selected_weight_max"
        )
    if str(settings.get("benchmark", "Competition-MATH")) != "Competition-MATH":
        raise ValueError("Experiment 2 is fixed to Competition-MATH")
    if str(settings.get("distance", DISTANCE_NAME)) != DISTANCE_NAME:
        raise ValueError(f"distance must be {DISTANCE_NAME!r}")
    return settings


def _adamw(model, config: dict[str, Any]):
    training = config["training"]
    kwargs = {
        "lr": float(training["learning_rate"]),
        "betas": tuple(float(value) for value in training.get("adam_betas", (0.9, 0.95))),
        "weight_decay": float(training.get("weight_decay", 0.0)),
    }
    if bool(training.get("fused_optimizer", True)) and torch.cuda.is_available():
        kwargs["fused"] = True
    return torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        **kwargs,
    )


def _distance_kwargs(settings: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "device": device,
        "max_prompt_tokens": int(settings.get("max_eval_prompt_tokens", 2048)),
        "student_temperature": float(settings.get("distance_student_temperature", 1.0)),
        "teacher_temperature": float(settings.get("distance_teacher_temperature", 1.0)),
        "vocab_chunk_positions": int(settings.get("distance_vocab_chunk_positions", 32)),
    }


@torch.inference_mode()
def _distributed_pair_teacher_distance(
    top_gt_student,
    uniform_student,
    teacher,
    tokenizer,
    rendered_test_prompts: list[str],
    *,
    distributed: DistributedContext,
    distance_kwargs: dict[str, Any],
) -> tuple[dict[str, float | int | str], dict[str, float | int | str]]:
    """Evaluate both students on one test shard while sharing teacher forwards."""
    begin, end = contiguous_partition(
        len(rendered_test_prompts), distributed.rank, distributed.world_size
    )
    local_prompts = rendered_test_prompts[begin:end]
    top_was_training = top_gt_student.training
    uniform_was_training = uniform_student.training
    top_gt_student.eval()
    uniform_student.eval()
    teacher.eval()
    local_top_sum = 0.0
    local_uniform_sum = 0.0
    local_tokens = 0
    local_maximum = 0
    max_prompt_tokens = int(distance_kwargs["max_prompt_tokens"])
    for local_index, prompt in enumerate(local_prompts):
        encoded = tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
            truncation=False,
        )
        length = int(encoded["attention_mask"].sum().item())
        local_maximum = max(local_maximum, length)
        if length > max_prompt_tokens:
            global_index = begin + local_index
            raise ValueError(
                f"Competition-MATH test prompt {global_index} has {length} tokens, "
                f"above max_eval_prompt_tokens={max_prompt_tokens}; refusing to "
                "truncate or silently exclude a test example"
            )
        input_ids = encoded["input_ids"].to(distributed.device)
        attention_mask = encoded["attention_mask"].to(distributed.device)
        forward_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
            "return_dict": True,
        }
        teacher_logits = teacher(**forward_kwargs).logits
        top_logits = top_gt_student(**forward_kwargs).logits
        kl_kwargs = {
            "student_temperature": float(distance_kwargs["student_temperature"]),
            "teacher_temperature": float(distance_kwargs["teacher_temperature"]),
            "vocab_chunk_positions": int(distance_kwargs["vocab_chunk_positions"]),
        }
        top_sum, top_count = full_vocab_forward_kl_from_logits(
            top_logits,
            teacher_logits,
            attention_mask.bool(),
            **kl_kwargs,
        )
        del top_logits
        uniform_logits = uniform_student(**forward_kwargs).logits
        uniform_sum, uniform_count = full_vocab_forward_kl_from_logits(
            uniform_logits,
            teacher_logits,
            attention_mask.bool(),
            **kl_kwargs,
        )
        if int(top_count.item()) != int(uniform_count.item()):
            raise AssertionError("The two KL branches counted different test tokens")
        local_top_sum += float(top_sum.item())
        local_uniform_sum += float(uniform_sum.item())
        local_tokens += int(top_count.item())
        del (
            input_ids,
            attention_mask,
            teacher_logits,
            uniform_logits,
            top_sum,
            uniform_sum,
            top_count,
            uniform_count,
        )
    if top_was_training:
        top_gt_student.train()
    if uniform_was_training:
        uniform_student.train()
    top_total = distributed.sum_float(local_top_sum)
    uniform_total = distributed.sum_float(local_uniform_sum)
    token_count = distributed.sum_int(local_tokens)
    problem_count = distributed.sum_int(len(local_prompts))
    maximum = distributed.max_int(local_maximum)
    if token_count <= 0:
        raise ValueError("Competition-MATH test split has no rendered tokens")
    common = {
        "name": DISTANCE_NAME,
        "num_tokens": token_count,
        "num_problems": problem_count,
        "maximum_prompt_tokens": maximum,
    }
    return (
        {**common, "value": top_total / token_count, "sum": top_total},
        {**common, "value": uniform_total / token_count, "sum": uniform_total},
    )


@torch.no_grad()
def _broadcast_model_state(model, distributed: DistributedContext) -> None:
    """Make rank 0 the sole weight authority before sharded evaluation.

    Running independent fused-AdamW updates on nominally identical replicas can
    drift across devices after several steps. Broadcasting every parameter and
    buffer makes the evaluated model mathematically unambiguous: all test
    shards always see rank 0's exact post-update weights.
    """
    if not distributed.enabled:
        return
    for parameter in model.parameters():
        if parameter.device != distributed.device:
            raise RuntimeError("Student parameters must be on the local CUDA device")
        dist.broadcast(parameter.data, src=0)
    for buffer in model.buffers():
        if buffer.device != distributed.device:
            raise RuntimeError("Student buffers must be on the local CUDA device")
        dist.broadcast(buffer.data, src=0)


def _save_pair(g_model, uniform_model, tokenizer, root: Path, step: int, final: bool) -> None:
    name = "final" if final else f"step-{step:06d}"
    for method, model in (("top_gt", g_model), ("uniform", uniform_model)):
        destination = root / "checkpoints" / method / name
        destination.mkdir(parents=True, exist_ok=False)
        model.save_pretrained(destination, safe_serialization=True)
        tokenizer.save_pretrained(destination)


@torch.no_grad()
def _assert_identical_initial_students(left, right) -> None:
    left_named = list(left.named_parameters())
    right_named = list(right.named_parameters())
    if len(left_named) != len(right_named):
        raise AssertionError("The two student trajectories have different structures")
    for (left_name, left_parameter), (right_name, right_parameter) in zip(
        left_named, right_named
    ):
        if left_name != right_name or not torch.equal(left_parameter, right_parameter):
            raise AssertionError(
                "The top-g_t and uniform trajectories did not start from identical "
                f"weights; first mismatch: {left_name!r} vs {right_name!r}"
            )


def _token_rows(
    tokenizer,
    encoded: dict[str, torch.Tensor],
    batch_indices: list[int],
    sample_ids: list[str],
    g_scores: torch.Tensor,
    g_weights: torch.Tensor,
    uniform: torch.Tensor,
    g_losses: torch.Tensor,
    uniform_losses: torch.Tensor,
    step: int,
) -> list[dict[str, Any]]:
    valid = encoded["attention_mask"].bool()
    g_effective = normalized_effective_weights(g_weights, valid).cpu()
    uniform_effective = normalized_effective_weights(uniform, valid).cpu()
    scores = g_scores.detach().float().cpu()
    raw_g = g_weights.detach().float().cpu()
    raw_uniform = uniform.detach().float().cpu()
    input_ids = encoded["input_ids"].detach().cpu()
    valid_cpu = valid.cpu()
    valid_scores = scores[valid_cpu]
    order = torch.argsort(valid_scores, descending=True, stable=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(1, order.numel() + 1)
    flat_rank_index = 0
    rows: list[dict[str, Any]] = []
    for local_row in range(input_ids.shape[0]):
        prompt_position = 0
        for padded_position in range(input_ids.shape[1]):
            if not bool(valid_cpu[local_row, padded_position]):
                continue
            token_id = int(input_ids[local_row, padded_position])
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "optimizer_step_before": int(step),
                    "local_batch_row": local_row,
                    "dataset_index": int(batch_indices[local_row]),
                    "sample_id": sample_ids[local_row],
                    "prompt_position": prompt_position,
                    "padded_tensor_position": padded_position,
                    **decode_token(tokenizer, token_id),
                    "g_t": float(scores[local_row, padded_position]),
                    "g_rank_in_batch": int(ranks[flat_rank_index]),
                    "top_gt_selected": bool(raw_g[local_row, padded_position]),
                    "top_gt_raw_weight": float(raw_g[local_row, padded_position]),
                    "top_gt_effective_weight": float(g_effective[local_row, padded_position]),
                    "uniform_raw_weight": float(raw_uniform[local_row, padded_position]),
                    "uniform_effective_weight": float(uniform_effective[local_row, padded_position]),
                    "top_gt_opd_loss": float(g_losses[local_row, padded_position]),
                    "uniform_opd_loss": float(uniform_losses[local_row, padded_position]),
                }
            )
            prompt_position += 1
            flat_rank_index += 1
    return rows


def run(config: dict[str, Any]) -> dict[str, Any]:
    settings = _settings(config)
    distributed = initialize_distributed(
        str(config.get("distributed", {}).get("backend", "nccl"))
    )
    device = distributed.device
    seed = int(config.get("experiment", {}).get("seed", 42))
    # Rank 0 owns both optimizers and is the sole weight authority. The
    # expensive held-out KL calculation, which dominates this experiment, is
    # sharded across every rank after exact weight broadcasts.
    _seed(seed)
    output = Path(settings.get("output_dir", "outputs/batch_gt_comparison")).expanduser().resolve()
    output_is_nonempty = output.exists() and any(output.iterdir())
    if distributed.any(output_is_nonempty):
        raise FileExistsError(f"Output directory is not empty: {output}")
    if distributed.is_main:
        output.mkdir(parents=True, exist_ok=True)
        save_config(config, output / "resolved_config.yaml")
    distributed.barrier()

    g_model, teacher, tokenizer, model_metadata = load_models(config, device)
    uniform_model, _uniform_tokenizer, _uniform_metadata = load_student_model(config, device)
    del _uniform_tokenizer, _uniform_metadata
    _assert_identical_initial_students(g_model, uniform_model)
    _broadcast_model_state(g_model, distributed)
    _broadcast_model_state(uniform_model, distributed)
    g_optimizer = _adamw(g_model, config) if distributed.is_main else None
    uniform_optimizer = _adamw(uniform_model, config) if distributed.is_main else None

    records, train_files = read_records(config["data"]["path"], config["data"].get("split"))
    validate_prompt_records(records, config["data"])
    records, train_filter = filter_overlong_prompt_records(records, tokenizer, config["data"])
    test_records, test_schema = load_benchmark(
        "Competition-MATH",
        config["evaluation"]["benchmarks"]["Competition-MATH"],
    )
    test_prompts = [render_evaluation_prompt(tokenizer, row, config) for row in test_records]
    batch_size = int(settings.get("train_batch_size", 4))
    if batch_size <= 0:
        raise ValueError("train_batch_size must be positive")
    epochs = int(settings.get("epochs", 1))
    if epochs <= 0:
        raise ValueError("batch_gt_comparison.epochs must be positive")
    steps_per_epoch = math.ceil(len(records) / batch_size)
    total_steps = steps_per_epoch * epochs
    if settings.get("max_steps") is not None:
        total_steps = min(total_steps, int(settings["max_steps"]))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "batchwise_top_10_percent_gt_vs_uniform_opd",
        "paired_design": (
            "Both trajectories start from the identical checkpoint and consume identical "
            "batches; after step 1 they evolve independently. Each improvement is measured "
            "against that trajectory's own pre-batch theta."
        ),
        "g_t": {
            "definition": GT_DEFINITION,
            "support": GT_SUPPORT,
            "selection": "stable descending batch-global top ceil(rho*N_valid)",
            "rho": float(settings.get("token_fraction", 0.10)),
            "selected_token_weighting": str(
                settings.get("selected_token_weighting", "binary")
            ),
            "selected_weight_bounds": [
                float(settings.get("selected_weight_min", 0.5)),
                float(settings.get("selected_weight_max", 1.5)),
            ],
            "weighting_note": (
                "bounded_rank uses g_t magnitude only for top-rho selection and "
                "rank order; selected weights are linear in rank and normalized "
                "to unit total mass by the loss"
            ),
        },
        "loss": {
            "top_gt": "normalized weighted OPD loss over selected tokens only",
            "uniform": "mean OPD loss over all valid tokens",
            "opd_support": "student Top-16",
        },
        "distance": {
            "name": DISTANCE_NAME,
            "direction": "KL(teacher || student)",
            "vocabulary": "full",
            "benchmark": "Competition-MATH",
            "num_test_problems": len(test_records),
            "sampling": False,
            "truncation": False,
        },
        "train": {
            "files": [str(path) for path in train_files],
            "records": len(records),
            "filter": train_filter,
            "batch_size": batch_size,
            "epochs": epochs,
            "planned_steps": total_steps,
            "same_batch_order": True,
            "matched_training_rng_per_batch": True,
        },
        "distributed": {
            "world_size": distributed.world_size,
            "training": (
                "rank 0 owns both AdamW trajectories; exact post-update student "
                "parameters and buffers are broadcast before every sharded evaluation"
            ),
            "test_kl": (
                "contiguous Competition-MATH shards reduced across ranks; both "
                "students share each teacher forward"
            ),
            "writer_rank": 0,
            "weight_authority_rank": 0,
            "before_distance_reuse": (
                "step t KL_before equals cached step t-1 KL_after because no update "
                "occurs between batches"
            ),
        },
        "models": model_metadata,
        "test_schema": test_schema,
    }
    if distributed.is_main:
        atomic_json(output / "manifest.json", manifest)
    distributed.barrier()
    training = config["training"]
    distance_kwargs = _distance_kwargs(settings, device)
    started = time.time()
    checkpoint_interval = int(settings.get("checkpoint_interval", 0))
    g_before_cache: dict[str, float | int | str] | None = None
    uniform_before_cache: dict[str, float | int | str] | None = None
    for step in range(total_steps):
        batch_indices = epoch_batch_indices(len(records), batch_size, step, seed)
        batch_records = [records[index] for index in batch_indices]
        sample_ids = [stable_sample_id(row, index) for row, index in zip(batch_records, batch_indices)]
        encoded, rendered_prompts = tokenize_prompts(batch_records, tokenizer, config["data"], device)
        step_started = time.time()
        if g_before_cache is None or uniform_before_cache is None:
            g_before, uniform_before = _distributed_pair_teacher_distance(
                g_model,
                uniform_model,
                teacher,
                tokenizer,
                test_prompts,
                distributed=distributed,
                distance_kwargs=distance_kwargs,
            )
            if not math.isclose(
                float(g_before["value"]),
                float(uniform_before["value"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise AssertionError(
                    "Identical initial students produced different test KL values"
                )
        else:
            g_before = g_before_cache
            uniform_before = uniform_before_cache

        if distributed.is_main:
            if g_optimizer is None or uniform_optimizer is None:
                raise AssertionError("Rank 0 must own both optimizers")
            g_reference, g_scores, g_diagnostics = build_prompt_pgt_reference(
                g_model,
                teacher,
                encoded["input_ids"],
                encoded["attention_mask"],
                student_temperature=float(
                    config.get("rollout", {}).get("temperature", 1.0)
                ),
                teacher_temperature=float(
                    config.get("opd", {}).get("teacher_temperature", 1.0)
                ),
                token_chunk_size=int(
                    config.get("selector", {}).get("pgt_vocab_chunk_tokens", 2048)
                ),
            )
            g_token_weights = top_gt_weights(
                g_scores,
                encoded["attention_mask"],
                float(settings.get("token_fraction", 0.10)),
                weighting=str(
                    settings.get("selected_token_weighting", "binary")
                ),
                minimum=float(settings.get("selected_weight_min", 0.5)),
                maximum=float(settings.get("selected_weight_max", 1.5)),
            )
            uniform_token_weights = uniform_weights(encoded["attention_mask"])
            uniform_reference = build_prompt_opd_reference(
                uniform_model,
                teacher,
                encoded["input_ids"],
                encoded["attention_mask"],
                student_temperature=float(
                    config.get("rollout", {}).get("temperature", 1.0)
                ),
                teacher_temperature=float(
                    config.get("opd", {}).get("teacher_temperature", 1.0)
                ),
            )
            common_train_kwargs = {
                "student_temperature": float(
                    config.get("rollout", {}).get("temperature", 1.0)
                ),
                "clip_low": float(training.get("ppo_clip_low", 0.2)),
                "clip_high": float(training.get("ppo_clip_high", 0.28)),
                "dual_clip": float(training.get("ppo_dual_clip", 3.0)),
                "chunk_steps": int(
                    config.get("selector", {}).get("score_chunk_steps", 128)
                ),
                "max_grad_norm": float(training.get("max_grad_norm", 1.0)),
            }
            # Match dropout/stochastic-layer randomness across both updates.
            cpu_rng_state = torch.get_rng_state()
            cuda_rng_state = torch.cuda.get_rng_state(device)
            g_train = train_prompt_batch(
                g_model,
                g_optimizer,
                encoded["input_ids"],
                encoded["attention_mask"],
                g_reference,
                g_token_weights,
                **common_train_kwargs,
            )
            torch.set_rng_state(cpu_rng_state)
            torch.cuda.set_rng_state(cuda_rng_state, device)
            uniform_train = train_prompt_batch(
                uniform_model,
                uniform_optimizer,
                encoded["input_ids"],
                encoded["attention_mask"],
                uniform_reference,
                uniform_token_weights,
                **common_train_kwargs,
            )
            expected_selected = math.ceil(
                float(settings.get("token_fraction", 0.10))
                * g_train["valid_tokens"]
            )
            if g_train["selected_tokens"] != expected_selected:
                raise AssertionError(
                    f"Expected exactly {expected_selected} top-g_t tokens, got "
                    f"{g_train['selected_tokens']}"
                )
            if uniform_train["selected_tokens"] != uniform_train["valid_tokens"]:
                raise AssertionError(
                    "Uniform OPD did not train on every valid batch token"
                )

        # Only rank 0 mutates weights. Every evaluation worker receives those
        # exact parameters before it scores its disjoint Competition-MATH shard.
        _broadcast_model_state(g_model, distributed)
        _broadcast_model_state(uniform_model, distributed)
        g_after, uniform_after = _distributed_pair_teacher_distance(
            g_model,
            uniform_model,
            teacher,
            tokenizer,
            test_prompts,
            distributed=distributed,
            distance_kwargs=distance_kwargs,
        )
        g_before_cache = g_after
        uniform_before_cache = uniform_after
        if distributed.is_main:
            g_change = improvement_record(
                float(g_before["value"]), float(g_after["value"])
            )
            uniform_change = improvement_record(
                float(uniform_before["value"]), float(uniform_after["value"])
            )
            summary = {
                "schema_version": SCHEMA_VERSION,
                "optimizer_step_before": step,
                "optimizer_step_after": step + 1,
                "epoch": step // steps_per_epoch,
                "batch_indices": batch_indices,
                "sample_ids": sample_ids,
                "valid_tokens": g_train["valid_tokens"],
                "top_gt_selected_tokens": g_train["selected_tokens"],
                "top_gt_selected_fraction": (
                    g_train["selected_tokens"] / g_train["valid_tokens"]
                ),
                "top_gt_selected_token_weighting": str(
                    settings.get("selected_token_weighting", "binary")
                ),
                "top_gt_selected_raw_weight_min": float(
                    g_token_weights[g_token_weights > 0].min().item()
                ),
                "top_gt_selected_raw_weight_mean": float(
                    g_token_weights[g_token_weights > 0].mean().item()
                ),
                "top_gt_selected_raw_weight_max": float(
                    g_token_weights[g_token_weights > 0].max().item()
                ),
                "top_gt_loss": g_train["loss"],
                "uniform_loss": uniform_train["loss"],
                "top_gt_gradient_l2_norm_before_clip": g_train[
                    "gradient_l2_norm_before_clip"
                ],
                "uniform_gradient_l2_norm_before_clip": uniform_train[
                    "gradient_l2_norm_before_clip"
                ],
                "top_gt": g_change,
                "uniform": uniform_change,
                "improvement_advantage_top_gt_minus_uniform": (
                    g_change["distance_improvement"]
                    - uniform_change["distance_improvement"]
                ),
                "top_gt_wins_this_batch": (
                    g_change["distance_improvement"]
                    > uniform_change["distance_improvement"]
                ),
                "student_weight_l2_distance_after_batch": (
                    model_parameter_l2_distance(g_model, uniform_model)
                ),
                "distance_num_test_problems": int(g_before["num_problems"]),
                "distance_num_test_tokens": int(g_before["num_tokens"]),
                "elapsed_seconds": time.time() - step_started,
            }
            g_per_token_loss = g_train.pop("per_token_loss")
            uniform_per_token_loss = uniform_train.pop("per_token_loss")
            token_rows = _token_rows(
                tokenizer,
                encoded,
                batch_indices,
                sample_ids,
                g_scores,
                g_token_weights,
                uniform_token_weights,
                g_per_token_loss,
                uniform_per_token_loss,
                step,
            )
            atomic_json(output / "steps" / f"step-{step:06d}.json", summary)
            atomic_jsonl(output / "weights" / f"step-{step:06d}.jsonl", token_rows)
            atomic_json(
                output / "prompts" / f"step-{step:06d}.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "optimizer_step_before": step,
                    "dataset_indices": batch_indices,
                    "sample_ids": sample_ids,
                    "rendered_prompts": rendered_prompts,
                },
            )
            print(json.dumps(summary, ensure_ascii=False), flush=True)
            del (
                g_reference,
                uniform_reference,
                g_scores,
                g_diagnostics,
                g_token_weights,
                uniform_token_weights,
                token_rows,
                g_per_token_loss,
                uniform_per_token_loss,
            )
        if checkpoint_interval > 0 and (step + 1) % checkpoint_interval == 0:
            if distributed.is_main:
                _save_pair(g_model, uniform_model, tokenizer, output, step + 1, False)
            distributed.barrier()
        del encoded
    if distributed.is_main:
        _save_pair(g_model, uniform_model, tokenizer, output, total_steps, True)
    distributed.barrier()
    result = None
    if distributed.is_main:
        comparison = summarize_comparison(output, output / "comparison")
        result = {
            "schema_version": SCHEMA_VERSION,
            "completed_steps": total_steps,
            "completed_full_train_split": total_steps == steps_per_epoch * epochs,
            "distributed_world_size": distributed.world_size,
            "top_gt_final_checkpoint": str(
                (output / "checkpoints" / "top_gt" / "final").resolve()
            ),
            "uniform_final_checkpoint": str(
                (output / "checkpoints" / "uniform" / "final").resolve()
            ),
            "comparison_summary": str(
                (output / "comparison" / "comparison_summary.json").resolve()
            ),
            "comparison_conclusion": comparison["conclusion"],
            "elapsed_seconds": time.time() - started,
        }
        atomic_json(output / "summary.json", result)
    result = distributed.broadcast_object(result, source=0)
    if result is None:
        raise AssertionError("Rank 0 did not produce an experiment summary")
    distributed.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "configs" / "batch_gt_comparison.yaml",
    )
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    config = resolve_runtime_paths(apply_overrides(load_config(args.config), args.set))
    try:
        result = run(config)
        if int(os.environ.get("RANK", "0")) == 0:
            print(yaml.safe_dump(result, sort_keys=False, allow_unicode=True))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
