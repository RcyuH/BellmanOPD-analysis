"""Train and compare top-10%-g_t OPD against uniform OPD batch by batch."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
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
    competition_math_teacher_distance,
    decode_token,
    improvement_record,
)


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


def _save_pair(g_model, uniform_model, tokenizer, root: Path, step: int, final: bool) -> None:
    name = "final" if final else f"step-{step:06d}"
    for method, model in (("top_gt", g_model), ("uniform", uniform_model)):
        destination = root / "checkpoints" / method / name
        destination.mkdir(parents=True, exist_ok=False)
        model.save_pretrained(destination, safe_serialization=True)
        tokenizer.save_pretrained(destination)


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
    if not torch.cuda.is_available():
        raise RuntimeError("Experiment 2 requires a CUDA GPU")
    device = torch.device("cuda", 0)
    seed = int(config.get("experiment", {}).get("seed", 42))
    _seed(seed)
    output = Path(settings.get("output_dir", "outputs/batch_gt_comparison")).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    save_config(config, output / "resolved_config.yaml")

    g_model, teacher, tokenizer, model_metadata = load_models(config, device)
    uniform_model, _uniform_tokenizer, _uniform_metadata = load_student_model(config, device)
    del _uniform_tokenizer, _uniform_metadata
    g_optimizer = _adamw(g_model, config)
    uniform_optimizer = _adamw(uniform_model, config)

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
        },
        "loss": {
            "top_gt": "mean OPD loss over selected tokens only",
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
        },
        "models": model_metadata,
        "test_schema": test_schema,
    }
    atomic_json(output / "manifest.json", manifest)
    training = config["training"]
    distance_kwargs = _distance_kwargs(settings, device)
    started = time.time()
    checkpoint_interval = int(settings.get("checkpoint_interval", 0))
    for step in range(total_steps):
        batch_indices = epoch_batch_indices(len(records), batch_size, step, seed)
        batch_records = [records[index] for index in batch_indices]
        sample_ids = [stable_sample_id(row, index) for row, index in zip(batch_records, batch_indices)]
        encoded, rendered_prompts = tokenize_prompts(batch_records, tokenizer, config["data"], device)
        step_started = time.time()
        g_before = competition_math_teacher_distance(
            g_model, teacher, tokenizer, test_prompts, **distance_kwargs
        )
        uniform_before = competition_math_teacher_distance(
            uniform_model, teacher, tokenizer, test_prompts, **distance_kwargs
        )

        g_reference, g_scores, g_diagnostics = build_prompt_pgt_reference(
            g_model,
            teacher,
            encoded["input_ids"],
            encoded["attention_mask"],
            student_temperature=float(config.get("rollout", {}).get("temperature", 1.0)),
            teacher_temperature=float(config.get("opd", {}).get("teacher_temperature", 1.0)),
            token_chunk_size=int(config.get("selector", {}).get("pgt_vocab_chunk_tokens", 2048)),
        )
        g_token_weights = top_gt_weights(
            g_scores,
            encoded["attention_mask"],
            float(settings.get("token_fraction", 0.10)),
        )
        uniform_token_weights = uniform_weights(encoded["attention_mask"])
        uniform_reference = build_prompt_opd_reference(
            uniform_model,
            teacher,
            encoded["input_ids"],
            encoded["attention_mask"],
            student_temperature=float(config.get("rollout", {}).get("temperature", 1.0)),
            teacher_temperature=float(config.get("opd", {}).get("teacher_temperature", 1.0)),
        )
        common_train_kwargs = {
            "student_temperature": float(config.get("rollout", {}).get("temperature", 1.0)),
            "clip_low": float(training.get("ppo_clip_low", 0.2)),
            "clip_high": float(training.get("ppo_clip_high", 0.28)),
            "dual_clip": float(training.get("ppo_dual_clip", 3.0)),
            "chunk_steps": int(config.get("selector", {}).get("score_chunk_steps", 128)),
            "max_grad_norm": float(training.get("max_grad_norm", 1.0)),
        }
        g_train = train_prompt_batch(
            g_model,
            g_optimizer,
            encoded["input_ids"],
            encoded["attention_mask"],
            g_reference,
            g_token_weights,
            **common_train_kwargs,
        )
        uniform_train = train_prompt_batch(
            uniform_model,
            uniform_optimizer,
            encoded["input_ids"],
            encoded["attention_mask"],
            uniform_reference,
            uniform_token_weights,
            **common_train_kwargs,
        )
        g_after = competition_math_teacher_distance(
            g_model, teacher, tokenizer, test_prompts, **distance_kwargs
        )
        uniform_after = competition_math_teacher_distance(
            uniform_model, teacher, tokenizer, test_prompts, **distance_kwargs
        )
        g_change = improvement_record(float(g_before["value"]), float(g_after["value"]))
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
            "top_gt_selected_fraction": g_train["selected_tokens"] / g_train["valid_tokens"],
            "top_gt_loss": g_train["loss"],
            "uniform_loss": uniform_train["loss"],
            "top_gt_gradient_l2_norm_before_clip": g_train["gradient_l2_norm_before_clip"],
            "uniform_gradient_l2_norm_before_clip": uniform_train["gradient_l2_norm_before_clip"],
            "top_gt": g_change,
            "uniform": uniform_change,
            "improvement_advantage_top_gt_minus_uniform": (
                g_change["distance_improvement"] - uniform_change["distance_improvement"]
            ),
            "top_gt_wins_this_batch": (
                g_change["distance_improvement"] > uniform_change["distance_improvement"]
            ),
            "student_weight_l2_distance_after_batch": model_parameter_l2_distance(
                g_model, uniform_model
            ),
            "distance_num_test_problems": int(g_before["num_problems"]),
            "distance_num_test_tokens": int(g_before["num_tokens"]),
            "elapsed_seconds": time.time() - step_started,
        }
        token_rows = _token_rows(
            tokenizer,
            encoded,
            batch_indices,
            sample_ids,
            g_scores,
            g_token_weights,
            uniform_token_weights,
            g_train.pop("per_token_loss"),
            uniform_train.pop("per_token_loss"),
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
        if checkpoint_interval > 0 and (step + 1) % checkpoint_interval == 0:
            _save_pair(g_model, uniform_model, tokenizer, output, step + 1, False)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        del (
            encoded,
            g_reference,
            uniform_reference,
            g_scores,
            g_diagnostics,
            g_token_weights,
            uniform_token_weights,
            token_rows,
        )
    _save_pair(g_model, uniform_model, tokenizer, output, total_steps, True)
    result = {
        "schema_version": SCHEMA_VERSION,
        "completed_steps": total_steps,
        "completed_full_train_split": total_steps == steps_per_epoch * epochs,
        "top_gt_final_checkpoint": str((output / "checkpoints" / "top_gt" / "final").resolve()),
        "uniform_final_checkpoint": str((output / "checkpoints" / "uniform" / "final").resolve()),
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(output / "summary.json", result)
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
    print(yaml.safe_dump(run(config), sort_keys=False, allow_unicode=True))


if __name__ == "__main__":
    main()

