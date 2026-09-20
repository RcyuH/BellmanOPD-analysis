from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import load_config, resolve_runtime_paths
from .evaluation import (
    configured_benchmark_names,
    ensure_extended_benchmark_specs,
    evaluation_metric_name,
)
from .evaluation_cache import evaluate_or_reuse_base
from .vllm_evaluation import evaluate_vllm_distributed


METHODS = (
    ("opd", "OPD", "opd"),
    ("ta", "TA-OPD", "ta_opd"),
    ("rac", "Bellman-RAC", "rac_opd"),
    ("pgt", "PGT", "pgt_opd"),
    ("cmt", "CMT-OPD", "cmt_opd"),
    ("grpo", "GRPO", "grpo"),
    ("iw", "IW-OPD", "iw"),
)
_CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")
_STEP_DIRECTORY_PATTERN = re.compile(r"^step-(\d+)$")


@dataclass(frozen=True)
class EvaluationTarget:
    step: int
    model_path: Path
    model_name: str
    model_role: str


def _require_model_snapshot(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_dir() or not (path / "config.json").is_file():
        raise FileNotFoundError(
            f"{description} is not a complete model snapshot (missing config.json): "
            f"{path}"
        )
    return path


def _last_jsonl_row(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    last = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last = json.loads(line)
    return last


def _infer_final_step(run_output: Path) -> int:
    latest_path = run_output / "latest.json"
    if latest_path.is_file():
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        if str(latest.get("checkpoint", "")).rstrip("/") == "final":
            return int(latest["step"])
    summary_path = run_output / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if "steps" in summary:
            return int(summary["steps"])
    metric = _last_jsonl_row(run_output / "metrics.jsonl")
    if metric is not None and "step" in metric:
        return int(metric["step"])
    raise ValueError(
        f"Cannot infer the optimizer step represented by {run_output / 'final'}; "
        "expected latest.json, summary.json, or metrics.jsonl"
    )


def discover_evaluation_targets(
    run_output: str | Path,
    method: str,
    display_name: str,
    *,
    include_base: bool = True,
    allow_unmatched_eval_directories: bool = False,
) -> tuple[Path, dict[str, Any], list[EvaluationTarget], int]:
    """Discover step 0, numbered checkpoints, and the saved final snapshot."""
    run_output = Path(run_output).expanduser().resolve()
    if not run_output.is_dir():
        raise FileNotFoundError(
            f"Missing {display_name} output directory: {run_output}"
        )
    config_path = run_output / "resolved_config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing resolved training config: {config_path}")
    config = ensure_extended_benchmark_specs(
        resolve_runtime_paths(load_config(config_path))
    )
    configured_method = str(config.get("experiment", {}).get("method", "")).lower()
    if configured_method != method:
        raise ValueError(
            f"Expected method={method!r} in {config_path}, got {configured_method!r}"
        )

    targets_by_step: dict[int, EvaluationTarget] = {}
    if include_base:
        base_path = _require_model_snapshot(
            Path(config["models"]["student_path"]), "Base student"
        )
        targets_by_step[0] = EvaluationTarget(
            step=0,
            model_path=base_path,
            model_name="Base student",
            model_role="base_student",
        )

    for checkpoint in sorted(run_output.glob("checkpoint-*")):
        if not checkpoint.is_dir():
            continue
        match = _CHECKPOINT_PATTERN.fullmatch(checkpoint.name)
        if match is None:
            continue
        step = int(match.group(1))
        targets_by_step[step] = EvaluationTarget(
            step=step,
            model_path=_require_model_snapshot(
                checkpoint, f"{display_name} checkpoint at step {step}"
            ),
            model_name=f"{display_name} step {step}",
            model_role=method,
        )

    final_path = run_output / "final"
    if final_path.is_dir():
        final_step = _infer_final_step(run_output)
        targets_by_step[final_step] = EvaluationTarget(
            step=final_step,
            model_path=_require_model_snapshot(
                final_path, f"{display_name} final checkpoint"
            ),
            model_name=f"{display_name} step {final_step}",
            model_role=method,
        )

    trained_targets = [target for step, target in targets_by_step.items() if step > 0]
    if not trained_targets:
        raise FileNotFoundError(
            f"No saved checkpoint-* or final model was found under {run_output}"
        )

    existing_eval_steps = set()
    eval_root = run_output / str(
        config.get("training_evaluation", {}).get("output_subdir", "training_eval")
    )
    if eval_root.is_dir():
        for path in eval_root.iterdir():
            match = _STEP_DIRECTORY_PATTERN.fullmatch(path.name)
            if path.is_dir() and match is not None:
                existing_eval_steps.add(int(match.group(1)))
    unmatched = sorted(existing_eval_steps - set(targets_by_step))
    if unmatched and not allow_unmatched_eval_directories:
        raise ValueError(
            f"{display_name} has old evaluation directories without a matching saved "
            f"checkpoint: {unmatched}. Restore those checkpoints, or explicitly use "
            "--allow-unmatched-eval-directories to leave those directories untouched "
            "and omit them from the new history."
        )

    summary_path = run_output / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        max_steps = int(summary.get("steps", max(targets_by_step)))
    else:
        metric = _last_jsonl_row(run_output / "metrics.jsonl")
        max_steps = int(metric["step"]) if metric is not None else max(targets_by_step)
    max_steps = max(max_steps, max(targets_by_step))
    return (
        config_path,
        config,
        sorted(targets_by_step.values(), key=lambda item: item.step),
        max_steps,
    )


def _atomic_replace_directory(staged: Path, destination: Path) -> None:
    """Replace one exact step directory, restoring the old one if commit fails."""
    backup = destination.with_name(f".{destination.name}.pre-reeval-{uuid.uuid4().hex}")
    had_destination = destination.exists()
    if had_destination:
        os.replace(destination, backup)
    try:
        os.replace(staged, destination)
    except BaseException:
        if had_destination and backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    else:
        if backup.exists():
            shutil.rmtree(backup)


def _metric_artifact_slug(metric_name: str) -> str:
    """Filesystem-safe, stable suffix for a re-evaluation metric."""
    return metric_name.replace("@", "_at_").replace("-", "_")


def _history_metric(path: Path) -> str | None:
    """Return the single metric recorded by a history file, if unambiguous."""
    if not path.is_file():
        return None
    metrics: set[str] = set()
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                parameters = row.get("parameters", {})
                if parameters.get("metric"):
                    metrics.add(str(parameters["metric"]))
                for benchmark in row.get("benchmarks", {}).values():
                    if benchmark.get("metric"):
                        metrics.add(str(benchmark["metric"]))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return next(iter(metrics)) if len(metrics) == 1 else None


def _reevaluation_artifact_paths(
    run_output: Path, output_subdir: str, metric_name: str
) -> tuple[Path, Path, Path]:
    """Choose metric-isolated artifacts without clobbering another metric."""
    canonical_root = run_output / output_subdir
    canonical_history = run_output / "eval_history.jsonl"
    if metric_name not in {"avg@8", "pass@8"}:
        return canonical_root, canonical_history, run_output / "eval_metrics.csv"
    slug = _metric_artifact_slug(metric_name)
    metric_root = run_output / f"{output_subdir}_{slug}"
    metric_history = run_output / f"eval_history_{slug}.jsonl"
    metric_csv = run_output / f"eval_metrics_{slug}.csv"
    if metric_history.exists() or _history_metric(canonical_history) != metric_name:
        return metric_root, metric_history, metric_csv
    return canonical_root, canonical_history, run_output / "eval_metrics.csv"


def _write_history_atomically(
    run_output: Path,
    history: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    *,
    history_path: Path | None = None,
    metrics_path: Path | None = None,
) -> None:
    history_path = history_path or (run_output / "eval_history.jsonl")
    metrics_path = metrics_path or (run_output / "eval_metrics.csv")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    history_temp = run_output / f".{history_path.name}.reeval-{uuid.uuid4().hex}.tmp"
    with history_temp.open("w", encoding="utf-8") as handle:
        for row in history:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    metrics_temp = run_output / f".{metrics_path.name}.reeval-{uuid.uuid4().hex}.tmp"
    # Keep the canonical avg@8/pass@8 schema first, but preserve legacy columns
    # found in an existing CSV.  Older runs may contain ``avg_at_16``; partial
    # re-evaluation must not fail (or silently discard that historical value)
    # merely because the new evaluation protocol uses avg@8.  Dynamic extras
    # also make this writer forward-compatible with future metric fields.
    canonical_fieldnames = (
        "step",
        "method",
        "backend",
        "benchmark",
        "correct",
        "total",
        "accuracy",
        "avg_at_n",
        "avg_at_8",
        "pass_at_k",
        "pass_at_8",
        "problems",
        "samples_per_problem",
        "metric",
        "evaluation_time_sec",
    )
    canonical_set = set(canonical_fieldnames)
    extra_fieldnames = tuple(
        sorted(
            {
                str(key)
                for row in metric_rows
                for key in row
                if str(key) not in canonical_set
            }
        )
    )
    fieldnames = (*canonical_fieldnames, *extra_fieldnames)
    with metrics_temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(metric_rows)

    history_backup = run_output / f".{history_path.name}.pre-reeval-{uuid.uuid4().hex}"
    metrics_backup = run_output / f".{metrics_path.name}.pre-reeval-{uuid.uuid4().hex}"
    history_existed = history_path.exists()
    metrics_existed = metrics_path.exists()
    history_backed_up = metrics_backed_up = False
    history_committed = metrics_committed = False
    try:
        if history_existed:
            os.replace(history_path, history_backup)
            history_backed_up = True
        if metrics_existed:
            os.replace(metrics_path, metrics_backup)
            metrics_backed_up = True
        os.replace(history_temp, history_path)
        history_committed = True
        os.replace(metrics_temp, metrics_path)
        metrics_committed = True
    except BaseException:
        if history_committed:
            history_path.unlink(missing_ok=True)
        if metrics_committed:
            metrics_path.unlink(missing_ok=True)
        if history_backed_up and history_backup.exists():
            os.replace(history_backup, history_path)
        if metrics_backed_up and metrics_backup.exists():
            os.replace(metrics_backup, metrics_path)
        raise
    finally:
        history_temp.unlink(missing_ok=True)
        metrics_temp.unlink(missing_ok=True)
    history_backup.unlink(missing_ok=True)
    metrics_backup.unlink(missing_ok=True)


def _read_history_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_metric_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _merge_partial_history(
    existing: list[dict[str, Any]], new_rows: list[dict[str, Any]], method: str
) -> list[dict[str, Any]]:
    """Merge re-evaluated benchmark dictionaries by (step, method)."""
    merged = [dict(row) for row in existing]
    for replacement in new_rows:
        step = int(replacement["step"])
        found = False
        for index, row in enumerate(merged):
            if int(row.get("step", -1)) != step or str(row.get("method")) != method:
                continue
            if not isinstance(row.get("benchmarks"), dict):
                merged[index] = replacement
                found = True
                break
            row_copy = dict(row)
            row_copy["benchmarks"] = {
                **dict(row.get("benchmarks", {})),
                **dict(replacement.get("benchmarks", {})),
            }
            for key in (
                "max_steps", "backend", "evaluation_time", "base_cache_status",
                "parameters", "details", "model_role",
            ):
                if key in replacement:
                    row_copy[key] = replacement[key]
            merged[index] = row_copy
            found = True
            break
        if not found:
            merged.append(replacement)
    return sorted(merged, key=lambda row: (int(row.get("step", 0)), str(row.get("method", ""))))


def _merge_partial_metrics(
    existing: list[dict[str, Any]], new_rows: list[dict[str, Any]], method: str
) -> list[dict[str, Any]]:
    keys = {
        (int(row.get("step", -1)), str(row.get("method", "")), str(row.get("benchmark", "")))
        for row in new_rows
    }
    output = [
        row
        for row in existing
        if (
            int(row.get("step", -1)),
            str(row.get("method", "")),
            str(row.get("benchmark", "")),
        )
        not in keys
    ]
    output.extend(new_rows)
    return sorted(output, key=lambda row: (int(row.get("step", 0)), str(row.get("method", "")), str(row.get("benchmark", ""))))


def _merge_step_artifacts(
    staged: Path, destination: Path, suite: dict[str, Any]
) -> dict[str, Any]:
    """Preserve old benchmark prediction files when a step is partially re-run."""
    if not destination.is_dir() or not (destination / "summary.json").is_file():
        return suite
    try:
        old_suite = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return suite
    new_names = set(suite.get("benchmarks", {}))
    # Copy predictions for benchmarks not present in this partial suite. Their
    # paths already point at the destination step directory and remain valid.
    for name, result in old_suite.get("benchmarks", {}).items():
        if name in new_names:
            continue
        prediction_value = result.get("predictions")
        if not prediction_value:
            continue
        prediction = destination / Path(prediction_value).name
        if prediction.is_file():
            shutil.copy2(prediction, staged / prediction.name)

    old_detailed_value = old_suite.get("detailed_outputs")
    new_detailed_value = suite.get("detailed_outputs")
    if not old_detailed_value or not new_detailed_value:
        old_detailed = new_detailed = None
    else:
        old_detailed = destination / Path(old_detailed_value).name
        new_detailed = staged / Path(new_detailed_value).name
    if old_detailed is not None and new_detailed is not None and old_detailed.is_file() and new_detailed.is_file():
        # Keep old detailed rows for untouched benchmarks and replace rows for
        # benchmarks that were explicitly re-evaluated.
        import gzip

        old_rows: list[dict[str, Any]] = []
        with gzip.open(old_detailed, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if row.get("benchmark") not in new_names:
                        old_rows.append(row)
        with gzip.open(new_detailed, "rt", encoding="utf-8") as handle:
            new_rows = [json.loads(line) for line in handle if line.strip()]
        with gzip.open(new_detailed, "wt", encoding="utf-8") as handle:
            for row in (*old_rows, *new_rows):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    merged = dict(suite)
    merged["benchmarks"] = {
        **dict(old_suite.get("benchmarks", {})),
        **dict(suite.get("benchmarks", {})),
    }
    return merged


def _runtime_settings(
    config: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    training_evaluation = config.get("training_evaluation", {})
    vllm_settings = dict(training_evaluation.get("vllm", {}))
    overrides = {
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "gpu_headroom_gib": args.gpu_headroom_gib,
        "gpu_workspace_headroom_gib": args.gpu_workspace_headroom_gib,
        "max_num_seqs": args.max_num_seqs,
        "max_model_len": args.max_model_len,
        "seed": args.seed,
    }
    vllm_settings.update(
        {key: value for key, value in overrides.items() if value is not None}
    )
    # Old resolved configs predate these vLLM 0.17.1 throughput controls.  They
    # only affect scheduling/throughput, not the requested sampling protocol.
    vllm_settings.setdefault("enable_chunked_prefill", True)
    vllm_settings.setdefault("performance_mode", "throughput")
    vllm_settings.setdefault("async_scheduling", True)
    return {
        "backend": "vllm",
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "num_responses": int(args.num_responses),
        "metric": args.metric,
        "max_new_tokens": int(
            args.max_new_tokens
            if args.max_new_tokens is not None
            else training_evaluation.get(
                "max_new_tokens", config["evaluation"].get("max_new_tokens", 2048)
            )
        ),
        "limit": None,
        "benchmark_names": list(
            configured_benchmark_names(
                config,
                getattr(args, "benchmarks", None)
                or training_evaluation.get("benchmark_names"),
            )
        ),
        "vllm": vllm_settings,
    }


def _reevaluate_method(
    run_output: Path,
    method: str,
    display_name: str,
    config_path: Path,
    config: dict[str, Any],
    targets: list[EvaluationTarget],
    max_steps: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    settings = _runtime_settings(config, args)
    output_subdir = str(
        config.get("training_evaluation", {}).get("output_subdir", "training_eval")
    )
    requested_metric = evaluation_metric_name(
        settings["num_responses"], settings.get("metric")
    )
    eval_root, history_path, metrics_path = _reevaluation_artifact_paths(
        run_output, output_subdir, requested_metric
    )
    manifest_path = run_output / "checkpoint_reevaluation_manifest.json"
    if history_path.name != "eval_history.jsonl":
        manifest_path = run_output / (
            f"checkpoint_reevaluation_manifest_{_metric_artifact_slug(requested_metric)}.json"
        )
    evaluated = []
    if args.dry_run:
        return {
            "method": method,
            "display_name": display_name,
            "output": str(run_output),
            "temperature": settings["temperature"],
            "metric": requested_metric,
            "history": str(history_path.resolve()),
            "metrics": str(metrics_path.resolve()),
            "manifest": str(manifest_path.resolve()),
            "targets": [
                {"step": target.step, "model_path": str(target.model_path)}
                for target in targets
            ],
            "dry_run": True,
        }

    eval_root.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for target in targets:
        step_dir = eval_root / f"step-{target.step:06d}"
        staged = Path(
            tempfile.mkdtemp(prefix=f".reeval-step-{target.step:06d}-", dir=eval_root)
        )
        started = time.perf_counter()
        print(
            f"[{display_name}] re-evaluating step {target.step} at "
            f"temperature={settings['temperature']}: {target.model_path}",
            flush=True,
        )

        def run_evaluator() -> dict[str, Any]:
            return evaluate_vllm_distributed(
                target.model_name,
                target.model_path,
                config,
                staged,
                settings,
                config_path,
                world_size=args.world_size,
            )

        try:
            cache_status = "disabled"
            if target.step == 0 and not args.no_base_cache:
                suite, cache_status = evaluate_or_reuse_base(
                    config=config,
                    runtime_settings=settings,
                    model_path=target.model_path,
                    model_name=target.model_name,
                    destination=staged,
                    evaluator=run_evaluator,
                    cache_dir=args.base_cache_dir,
                    # Re-eval must replace the requested output. It may reuse a
                    # verified shared result, never the old destination itself.
                    reuse_destination=False,
                )
                if cache_status == "shared":
                    skipped_responses = sum(
                        int(result["total"]) for result in suite["benchmarks"].values()
                    )
                    print(
                        f"[{display_name}] reused identical cached base evaluation; "
                        f"skipped {skipped_responses:,} generations.",
                        flush=True,
                    )
            else:
                suite = run_evaluator()
            elapsed = time.perf_counter() - started
            summary_path = staged / "summary.json"
            actual_temperature = float(suite["parameters"]["temperature"])
            if actual_temperature != float(settings["temperature"]):
                raise AssertionError(
                    f"Evaluator reported temperature={actual_temperature}, expected "
                    f"{settings['temperature']}"
                )
            if float(suite["parameters"]["top_p"]) != float(settings["top_p"]):
                raise AssertionError("Evaluator did not honor requested top_p")
            if int(suite["parameters"]["num_responses"]) != int(
                settings["num_responses"]
            ):
                raise AssertionError("Evaluator did not honor requested response count")
            expected_benchmarks = tuple(settings["benchmark_names"])
            if tuple(suite["benchmarks"]) != expected_benchmarks:
                raise AssertionError(
                    f"Evaluator returned benchmarks {tuple(suite['benchmarks'])}, "
                    f"expected {expected_benchmarks}"
                )
            for benchmark, result in suite["benchmarks"].items():
                prediction = Path(result["predictions"])
                if not prediction.is_file():
                    raise FileNotFoundError(
                        f"Missing staged predictions for {benchmark}: {prediction}"
                    )
                result["predictions"] = str(step_dir / prediction.name)
            detailed_outputs = Path(suite["detailed_outputs"])
            if not detailed_outputs.is_file():
                raise FileNotFoundError(
                    f"Missing staged detailed outputs: {detailed_outputs}"
                )
            suite["detailed_outputs"] = str(step_dir / detailed_outputs.name)
            suite = _merge_step_artifacts(staged, step_dir, suite)
            summary_path.write_text(
                json.dumps(suite, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            _atomic_replace_directory(staged, step_dir)
        finally:
            if staged.exists():
                shutil.rmtree(staged)

        samples_per_problem = int(settings["num_responses"])
        metric_name = str(
            suite.get("parameters", {}).get(
                "metric", evaluation_metric_name(samples_per_problem)
            )
        )
        history_entry = {
            "step": target.step,
            "max_steps": max_steps,
            "method": method,
            "model_role": target.model_role,
            "backend": "vllm",
            "evaluation_time": elapsed,
            "base_cache_status": cache_status if target.step == 0 else None,
            "benchmarks": {
                name: {
                    "correct": result["correct"],
                    "total": result["total"],
                    "accuracy": result["accuracy"],
                    "avg_at_n": result.get("avg_at_n", result["accuracy"]),
                    **(
                        {"avg_at_8": result["avg_at_8"]}
                        if "avg_at_8" in result
                        else {}
                    ),
                    "problems": result.get("problems"),
                    **(
                        {"pass_at_k": result["pass_at_k"]}
                        if "pass_at_k" in result
                        else {}
                    ),
                    **(
                        {"pass_at_8": result["pass_at_8"]}
                        if "pass_at_8" in result
                        else {}
                    ),
                    "samples_per_problem": result.get(
                        "samples_per_problem", samples_per_problem
                    ),
                    "metric": metric_name,
                }
                for name, result in suite["benchmarks"].items()
            },
            "parameters": suite["parameters"],
            "details": str((step_dir / "summary.json").resolve()),
        }
        history.append(history_entry)
        for benchmark, result in history_entry["benchmarks"].items():
            metric_rows.append(
                {
                    "step": target.step,
                    "method": method,
                    "backend": "vllm",
                    "benchmark": benchmark,
                    **result,
                    "evaluation_time_sec": elapsed,
                }
            )
        evaluated.append(
            {
                "step": target.step,
                "model_path": str(target.model_path),
                "details": history_entry["details"],
            }
        )

    # Partial benchmark runs must augment, rather than erase, an existing
    # checkpoint row. This is what lets GPQA/AMC23 be added to histories that
    # already contain the original four datasets.
    history = _merge_partial_history(
        _read_history_rows(history_path), history, method
    )
    metric_rows = _merge_partial_metrics(
        _read_metric_rows(metrics_path), metric_rows, method
    )
    _write_history_atomically(
        run_output,
        history,
        metric_rows,
        history_path=history_path,
        metrics_path=metrics_path,
    )
    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "display_name": display_name,
        "temperature": settings["temperature"],
        "top_p": settings["top_p"],
        "num_responses": settings["num_responses"],
        "metric": evaluation_metric_name(
            settings["num_responses"], settings.get("metric")
        ),
        "backend": "vllm",
        "full_benchmarks": list(settings["benchmark_names"]),
        "history": str(history_path.resolve()),
        "metrics": str(metrics_path.resolve()),
        "artifact_root": str(eval_root.resolve()),
        "evaluated": evaluated,
    }
    manifest_temp = run_output / f".{manifest_path.name}.{uuid.uuid4().hex}.tmp"
    manifest_temp.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(manifest_temp, manifest_path)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Re-evaluate every saved checkpoint for selected OPD/TA-OPD/RAC/PGT "
            "methods and replace their periodic-evaluation histories"
        )
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=[method for method, _display_name, _output_slug in METHODS],
        default=["opd", "ta", "rac"],
        help="Methods to re-evaluate; defaults to the legacy three (PGT is opt-in)",
    )
    parser.add_argument("--opd-output")
    parser.add_argument("--ta-output")
    parser.add_argument("--rac-output")
    parser.add_argument("--pgt-output")
    parser.add_argument("--cmt-output")
    parser.add_argument("--grpo-output")
    parser.add_argument("--iw-output")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--num-responses", type=int, default=8)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        help=(
            "Optional canonical benchmark subset. Existing history rows are merged "
            "per benchmark, so omitted datasets are preserved."
        ),
    )
    parser.add_argument(
        "--metric",
        default=None,
        help="Evaluation metric, e.g. avg@8 or pass@8 (both use 8 samples).",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        help="Number of independent TP=1 vLLM evaluators; defaults to visible GPU count.",
    )
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument("--gpu-memory-utilization")
    parser.add_argument("--gpu-headroom-gib", type=float)
    parser.add_argument("--gpu-workspace-headroom-gib", type=float)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--base-cache-dir")
    parser.add_argument("--no-base-cache", action="store_true")
    parser.add_argument("--skip-base", action="store_true")
    parser.add_argument("--allow-unmatched-eval-directories", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.temperature <= 0:
        raise ValueError("Checkpoint re-evaluation requires temperature > 0")
    if not 0.0 < args.top_p <= 1.0:
        raise ValueError("Checkpoint re-evaluation top_p must be in (0, 1]")
    if args.num_responses <= 0:
        raise ValueError("Checkpoint re-evaluation num_responses must be positive")
    evaluation_metric_name(args.num_responses, args.metric)
    selected_methods = tuple(dict.fromkeys(args.methods))
    requested = {
        "opd": args.opd_output,
        "ta": args.ta_output,
        "rac": args.rac_output,
        "pgt": args.pgt_output,
        "cmt": args.cmt_output,
        "grpo": args.grpo_output,
        "iw": args.iw_output,
    }
    missing = [method for method in selected_methods if requested[method] is None]
    if missing:
        required_flags = ", ".join(f"--{method}-output" for method in missing)
        raise ValueError(
            f"Missing output path for selected method(s): {', '.join(missing)}; "
            f"provide {required_flags}"
        )
    results = []
    for method, display_name, _output_slug in METHODS:
        if method not in selected_methods:
            continue
        run_output = Path(requested[method]).expanduser().resolve()
        config_path, config, targets, max_steps = discover_evaluation_targets(
            run_output,
            method,
            display_name,
            include_base=not args.skip_base,
            allow_unmatched_eval_directories=args.allow_unmatched_eval_directories,
        )
        results.append(
            _reevaluate_method(
                run_output,
                method,
                display_name,
                config_path,
                config,
                targets,
                max_steps,
                args,
            )
        )
    print(json.dumps({"methods": results}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
