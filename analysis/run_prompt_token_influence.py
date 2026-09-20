"""Run exact prompt-token OPD interventions on Competition-MATH.

Example:

    python -m analysis.run_prompt_token_influence \
      --config analysis/configs/prompt_token_influence.yaml
"""

from __future__ import annotations

import argparse
import json
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
    render_record_prompt,
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
            "Exact per-prompt attribution currently requires train_batch_size=1; "
            "this keeps each uniform baseline equal to x_1+...+x_N for one prompt"
        )
    if str(settings.get("actual_update", "uniform_mean")) != "uniform_mean":
        raise ValueError("actual_update must be 'uniform_mean'")
    if str(settings.get("distance", DISTANCE_NAME)) != DISTANCE_NAME:
        raise ValueError(f"distance must be {DISTANCE_NAME!r}")
    if int(config.get("selector", {}).get("top_k", OPD_LOSS_TOP_K)) != OPD_LOSS_TOP_K:
        raise ValueError(f"The repository OPD objective requires selector.top_k={OPD_LOSS_TOP_K}")
    return settings


def _output_root(config: dict[str, Any], settings: dict[str, Any]) -> Path:
    root = Path(str(settings.get("output_dir", "outputs/prompt_token_influence")))
    return root.expanduser().resolve()


def _step_rows(
    *,
    model,
    teacher,
    tokenizer,
    encoded: dict[str, torch.Tensor],
    rendered_prompt: str,
    reference,
    test_prompts: list[str],
    parameters,
    snapshot,
    config: dict[str, Any],
    settings: dict[str, Any],
    device: torch.device,
    step: int,
    epoch: int,
    dataset_index: int,
    sample_id: str,
) -> tuple[list[dict[str, Any]], list[torch.Tensor | None]]:
    training = config["training"]
    learning_rate = float(settings.get("learning_rate", training["learning_rate"]))
    student_temperature = float(config.get("rollout", {}).get("temperature", 1.0))
    teacher_temperature = float(config.get("opd", {}).get("teacher_temperature", 1.0))
    distance_kwargs = {
        "device": device,
        "max_prompt_tokens": int(settings.get("max_eval_prompt_tokens", 2048)),
        "student_temperature": float(settings.get("distance_student_temperature", 1.0)),
        "teacher_temperature": float(settings.get("distance_teacher_temperature", 1.0)),
        "vocab_chunk_positions": int(settings.get("distance_vocab_chunk_positions", 32)),
    }
    baseline = competition_math_teacher_distance(
        model, teacher, tokenizer, test_prompts, **distance_kwargs
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
    }
    rows: list[dict[str, Any]] = []
    model.train()
    for prompt_position, (batch_index, padded_position) in enumerate(valid_coordinates):
        restore_parameters(parameters, snapshot)
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
        grad_norm = gradient_l2_norm(gradients)
        apply_sgd_update(parameters, gradients, learning_rate)
        after = competition_math_teacher_distance(
            model, teacher, tokenizer, test_prompts, **distance_kwargs
        )
        identity = decode_token(tokenizer, int(token_ids[batch_index, padded_position]))
        rows.append(
            {
                **common,
                "intervention": "single_token",
                "prompt_position": int(prompt_position),
                "padded_tensor_position": int(padded_position),
                **identity,
                "opd_loss": float(loss.detach().float().item()),
                "gradient_l2_norm": grad_norm,
                **improvement_record(baseline_value, float(after["value"])),
            }
        )
        del losses, loss, gradients, after

    # The uniform branch is evaluated from the same theta as every token branch.
    restore_parameters(parameters, snapshot)
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
    valid_mask = encoded["attention_mask"].bool()
    uniform_loss = losses[valid_mask].mean()
    uniform_gradients = list(
        torch.autograd.grad(
            uniform_loss, parameters, allow_unused=True, materialize_grads=False
        )
    )
    uniform_grad_norm = gradient_l2_norm(uniform_gradients)
    apply_sgd_update(parameters, uniform_gradients, learning_rate)
    uniform_after = competition_math_teacher_distance(
        model, teacher, tokenizer, test_prompts, **distance_kwargs
    )
    rows.append(
        {
            **common,
            "intervention": "uniform_mean_all_prompt_tokens",
            "prompt_position": None,
            "padded_tensor_position": None,
            "token_id": None,
            "token_piece": None,
            "decoded_text": None,
            "visible_text": "UNIFORM(x_1,...,x_N)",
            "decoded_utf8_hex": None,
            "is_special_token": None,
            "opd_loss": float(uniform_loss.detach().float().item()),
            "gradient_l2_norm": uniform_grad_norm,
            **improvement_record(baseline_value, float(uniform_after["value"])),
        }
    )
    # Keep the uniform branch as the real step. No Adam state, clipping, or
    # weight decay is hidden here: this is exactly theta <- theta - eta*g_mean.
    del losses, uniform_loss, uniform_after
    return rows, uniform_gradients


def run(config: dict[str, Any]) -> dict[str, Any]:
    settings = _settings(config)
    if not torch.cuda.is_available():
        raise RuntimeError("The exact intervention experiment requires a CUDA GPU")
    device = torch.device("cuda", 0)
    seed = int(config.get("experiment", {}).get("seed", 42))
    _seed_everything(seed)
    output = _output_root(config, settings)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output}. Use a new output_dir so "
            "step branches cannot be mixed across runs."
        )
    output.mkdir(parents=True, exist_ok=True)
    save_config(config, output / "resolved_config.yaml")

    student, teacher, tokenizer, model_metadata = load_models(config, device)
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
    parameters = trainable_parameters(student)
    epochs = int(settings.get("epochs", 1))
    if epochs <= 0:
        raise ValueError("prompt_token_influence.epochs must be positive")
    total_steps = len(train_records) * epochs
    configured_max = settings.get("max_steps")
    if configured_max is not None:
        total_steps = min(total_steps, int(configured_max))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment": "prompt_token_opd_exact_sgd_intervention",
        "token_semantics": TOKEN_SEMANTICS,
        "distance": {
            "name": DISTANCE_NAME,
            "direction": "KL(teacher || student)",
            "vocabulary": "full",
            "reduction": "mean over every rendered test-prompt token",
            "benchmark": benchmark,
            "num_test_problems": len(test_records),
            "sampling": False,
            "truncation": False,
        },
        "interventions": {
            "single_token": "theta - learning_rate * grad(L_t)",
            "uniform": "theta - learning_rate * grad(mean_t L_t)",
            "branching": "all branches at a step start from identical theta",
            "actual_training_update": "uniform branch",
        },
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
    atomic_json(output / "manifest.json", manifest)
    started = time.time()
    checkpoint_interval = int(settings.get("checkpoint_interval", 0))
    for step in range(total_steps):
        dataset_index = epoch_batch_indices(len(train_records), 1, step, seed)[0]
        epoch = step // len(train_records)
        record = train_records[dataset_index]
        encoded, rendered = tokenize_prompts([record], tokenizer, config["data"], device)
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
        snapshot = clone_parameters(parameters)
        step_started = time.time()
        rows, uniform_gradients = _step_rows(
            model=student,
            teacher=teacher,
            tokenizer=tokenizer,
            encoded=encoded,
            rendered_prompt=rendered_prompt,
            reference=reference,
            test_prompts=test_prompts,
            parameters=parameters,
            snapshot=snapshot,
            config=config,
            settings=settings,
            device=device,
            step=step,
            epoch=epoch,
            dataset_index=dataset_index,
            sample_id=sample_id,
        )
        # _step_rows leaves the student at the evaluated uniform update. This is
        # the actual state for the next step; the returned gradients are only
        # retained until here to make that contract explicit.
        del uniform_gradients, snapshot, reference
        atomic_jsonl(output / "steps" / f"step-{step:06d}.jsonl", rows)
        atomic_json(
            output / "prompts" / f"step-{step:06d}.json",
            {
                "schema_version": SCHEMA_VERSION,
                "optimizer_step_before": step,
                "epoch": epoch,
                "dataset_index": dataset_index,
                "sample_id": sample_id,
                "rendered_prompt": rendered_prompt,
                "token_ids": encoded["input_ids"][0][encoded["attention_mask"][0].bool()]
                .detach()
                .cpu()
                .tolist(),
                "num_tokens": int(encoded["attention_mask"].sum().item()),
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
        print(
            json.dumps(
                {
                    "completed_step": step + 1,
                    "total_steps": total_steps,
                    "dataset_index": dataset_index,
                    "tokens": len(rows) - 1,
                    "best_token_improvement": max(
                        row["distance_improvement"]
                        for row in rows
                        if row["intervention"] == "single_token"
                    ),
                    "uniform_improvement": rows[-1]["distance_improvement"],
                    "elapsed_seconds": time.time() - step_started,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        del encoded, rows
    summary = {
        "schema_version": SCHEMA_VERSION,
        "completed_steps": total_steps,
        "completed_full_train_split": total_steps == len(train_records) * epochs,
        "elapsed_seconds": time.time() - started,
        "output_dir": str(output),
    }
    atomic_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "configs" / "prompt_token_influence.yaml",
    )
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    config = apply_overrides(load_config(args.config), args.set)
    config = resolve_runtime_paths(config)
    print(yaml.safe_dump(run(config), sort_keys=False, allow_unicode=True))


if __name__ == "__main__":
    main()

