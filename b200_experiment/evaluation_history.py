"""Utilities for recording standalone checkpoint evaluations in a run output.

Training-time evaluation already stores one row per ``(step, method)`` in the
run's ``eval_history.jsonl``.  Standalone evaluation should use the same schema
so plotting and resume tooling can consume it without a separate results
directory.  The helpers here deliberately have no model/CUDA dependencies;
they only transform a completed evaluator suite into atomically-updated
history artifacts.
"""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any


EVAL_METRIC_FIELDS = (
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


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.standalone-{os.getpid()}-{time.time_ns()}.tmp")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _same_key(row: dict[str, Any], step: int, method: str) -> bool:
    try:
        row_step = int(row.get("step"))
    except (TypeError, ValueError):
        return False
    return row_step == int(step) and str(row.get("method", "")) == method


def _upsert_rows(
    rows: list[dict[str, Any]],
    replacement: dict[str, Any],
    *,
    step: int,
    method: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    replaced = False
    for row in rows:
        if _same_key(row, step, method):
            if not replaced:
                output.append(replacement)
                replaced = True
        else:
            output.append(row)
    if not replaced:
        output.append(replacement)
    return output


def _merge_history_entry(
    rows: list[dict[str, Any]],
    replacement: dict[str, Any],
    *,
    step: int,
    method: str,
) -> list[dict[str, Any]]:
    """Upsert one checkpoint while preserving benchmarks absent from a partial run."""
    output: list[dict[str, Any]] = []
    replaced = False
    for row in rows:
        if not _same_key(row, step, method):
            output.append(row)
            continue
        if replaced:
            continue
        if not isinstance(row.get("benchmarks"), dict):
            output.append(replacement)
            replaced = True
            continue
        merged = dict(row)
        merged["benchmarks"] = {
            **dict(row.get("benchmarks", {})),
            **dict(replacement.get("benchmarks", {})),
        }
        # Runtime metadata describes the protocol used for this write. The
        # per-benchmark metric remains authoritative when a suite is merged.
        for key in (
            "max_steps", "backend", "evaluation_time", "base_cache_status",
            "parameters", "details", "model_role",
        ):
            if key in replacement:
                merged[key] = replacement[key]
        output.append(merged)
        replaced = True
    if not replaced:
        output.append(replacement)
    return output


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=EVAL_METRIC_FIELDS,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _history_entry(
    suite: dict[str, Any],
    *,
    method: str,
    step: int,
    max_steps: int,
    details_path: Path,
    evaluation_time: float | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    parameters = dict(suite.get("parameters", {}))
    backend = str(parameters.get("backend", "unknown"))
    samples_per_problem = int(parameters.get("num_responses", 1))
    metric_name = str(parameters.get("metric") or "")
    if not metric_name:
        # Avoid importing the model evaluator in this lightweight persistence
        # module.  All production evaluator suites include ``metric``; this
        # fallback keeps hand-built/offline suites usable in tests.
        metric_name = (
            f"avg@{samples_per_problem}"
            if samples_per_problem > 1
            else "accuracy"
        )

    benchmarks: dict[str, dict[str, Any]] = {}
    metric_rows: list[dict[str, Any]] = []
    for name, result in suite.get("benchmarks", {}).items():
        accuracy = result.get("accuracy", result.get("avg_at_n"))
        if accuracy is None:
            raise ValueError(f"Evaluation benchmark {name!r} has no accuracy")
        entry: dict[str, Any] = {
            "correct": result.get("correct"),
            "total": result.get("total"),
            "accuracy": accuracy,
            "avg_at_n": result.get("avg_at_n", accuracy),
            "problems": result.get("problems"),
            "samples_per_problem": result.get(
                "samples_per_problem", samples_per_problem
            ),
            "metric": result.get("metric", metric_name),
        }
        for key in ("avg_at_8", "pass_at_k", "pass_at_8"):
            if key in result:
                entry[key] = result[key]
        benchmarks[str(name)] = entry
        metric_rows.append(
            {
                "step": int(step),
                "method": method,
                "backend": backend,
                "benchmark": str(name),
                **entry,
                "evaluation_time_sec": evaluation_time,
            }
        )

    return {
        "step": int(step),
        "max_steps": int(max_steps),
        "method": method,
        "model_role": "base_student" if int(step) == 0 else method,
        "backend": backend,
        "evaluation_time": evaluation_time,
        "base_cache_status": None,
        "benchmarks": benchmarks,
        "parameters": parameters,
        "details": str(details_path.resolve()),
    }, metric_rows


def record_checkpoint_evaluation(
    run_output: str | Path,
    suite: dict[str, Any],
    *,
    method: str,
    step: int,
    max_steps: int | None = None,
    details_path: str | Path | None = None,
    evaluation_time: float | None = None,
) -> dict[str, str]:
    """Upsert one standalone evaluation into a method run output.

    Existing rows are keyed by ``(step, method)``.  Re-evaluating the same
    checkpoint therefore replaces exactly that row (and its benchmark rows in
    ``eval_metrics.csv``), while evaluations of other checkpoints remain intact.
    The two files are written atomically and no model/runtime dependencies are
    imported here.
    """

    run_output = Path(run_output).expanduser().resolve()
    run_output.mkdir(parents=True, exist_ok=True)
    step = int(step)
    if step < 0:
        raise ValueError(f"Evaluation step must be non-negative, got {step}")
    if max_steps is None:
        max_steps = step
    max_steps = max(int(max_steps), step)
    details = Path(details_path or (run_output / "summary.json")).expanduser().resolve()
    entry, metric_rows = _history_entry(
        suite,
        method=method,
        step=step,
        max_steps=max_steps,
        details_path=details,
        evaluation_time=evaluation_time,
    )

    history_path = run_output / "eval_history.jsonl"
    history_rows = _merge_history_entry(
        _read_jsonl(history_path), entry, step=step, method=method
    )
    _atomic_write_jsonl(history_path, history_rows)

    metrics_path = run_output / "eval_metrics.csv"
    metrics_rows = _read_csv(metrics_path)
    for metric_row in metric_rows:
        metrics_rows = [
            row
            for row in metrics_rows
            if not (
                str(row.get("step", "")) == str(step)
                and str(row.get("method", "")) == method
                and str(row.get("benchmark", "")) == str(metric_row["benchmark"])
            )
        ]
    metrics_rows.extend(metric_rows)
    _atomic_write_csv(metrics_path, metrics_rows)
    return {
        "history": str(history_path),
        "metrics": str(metrics_path),
        "details": str(details),
    }
