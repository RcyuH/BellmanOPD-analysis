from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import gzip
import json
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from .config import load_config
from .distributed import isolate_distributed_subprocess_environment
from .evaluation import (
    configured_benchmark_names,
    detailed_model_output_record,
    evaluation_metric_name,
    ensure_extended_benchmark_specs,
    grade_evaluation_response,
    load_benchmark,
    metric_problem_score,
    render_evaluation_prompt,
)


def _terminate_process_group(process: subprocess.Popen, timeout: float = 10.0) -> None:
    """Terminate a vLLM launcher and all of its engine worker descendants.

    vLLM creates EngineCore/worker processes below the Python launcher. Calling
    ``Popen.terminate`` on the launcher alone can orphan those children after a
    failed evaluation, leaving most of a 180-GiB GPU allocated and making the
    next re-evaluation fail before vLLM starts. Distributed evaluators are
    launched in their own sessions, so killing the process group is scoped to
    exactly one evaluation worker.
    """
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (AttributeError, ProcessLookupError, PermissionError):
        if process.poll() is None:
            process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (AttributeError, ProcessLookupError, PermissionError):
            if process.poll() is None:
                process.kill()
        process.wait()


@contextmanager
def _exclusive_evaluation_gpu_locks(
    devices: list[str], timeout: float = 24 * 60 * 60
):
    """Serialize independent re-evaluations that target the same GPU.

    Separate shell commands commonly inherit the launcher default
    ``CUDA_VISIBLE_DEVICES=0``. Without a lock, the first vLLM process reserves
    the GPU and every other command races into the same device. Locks are
    filesystem based, released automatically if a process crashes, and
    acquired in sorted device order to avoid deadlocks for multi-GPU masks.
    """
    normalized = sorted(
        {str(device).strip() for device in devices if str(device).strip()},
        key=lambda value: (0, int(value)) if value.isdigit() else (1, value),
    )
    if not normalized:
        yield
        return
    try:
        import fcntl
    except ImportError:  # pragma: no cover - Linux production path
        yield
        return

    lock_root = Path(
        os.environ.get(
            "BELLMANOPD_EVAL_LOCK_DIR",
            f"/tmp/bellmanopd-vllm-eval-locks-{os.getuid()}",
        )
    )
    lock_root.mkdir(parents=True, exist_ok=True)
    handles = []
    started = time.monotonic()
    announced = False
    try:
        for device in normalized:
            path = lock_root / f"gpu-{device}.lock"
            handle = path.open("a+")
            while True:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if not announced:
                        print(
                            "Another evaluation is using GPU "
                            f"{device}; waiting for its vLLM process to exit...",
                            flush=True,
                        )
                        announced = True
                    if time.monotonic() - started >= timeout:
                        raise TimeoutError(
                            f"Timed out after {timeout:.0f}s waiting for evaluation "
                            f"GPU lock(s) on {','.join(normalized)}"
                        )
                    time.sleep(1.0)
            handles.append(handle)
        yield
    finally:
        for handle in reversed(handles):
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()


def _resolve_gpu_memory_utilization(vllm_settings: dict[str, Any]) -> float:
    requested = vllm_settings.get("gpu_memory_utilization", "auto")
    if str(requested).lower() != "auto":
        utilization = float(requested)
        if not 0.0 < utilization <= 1.0:
            raise ValueError("vLLM gpu_memory_utilization must be in (0, 1]")
        return utilization
    if not torch.cuda.is_available():
        raise RuntimeError("Automatic vLLM memory sizing requires CUDA")
    headroom_gib = float(vllm_settings.get("gpu_headroom_gib", 4))
    if headroom_gib < 0:
        raise ValueError("vLLM gpu_headroom_gib must be non-negative")
    # vLLM's reservation covers the engine/KV cache, but generation can still
    # allocate CUDA-graph, attention and allocator workspace.  ``mem_get_info``
    # cannot see that future peak, so reserve it explicitly in addition to the
    # configured device headroom.  The default is intentional for old resolved
    # configs that predate this setting.
    workspace_headroom_gib = float(
        vllm_settings.get("gpu_workspace_headroom_gib", 2)
    )
    if workspace_headroom_gib < 0:
        raise ValueError(
            "vLLM gpu_workspace_headroom_gib must be non-negative"
        )
    effective_headroom_gib = headroom_gib + workspace_headroom_gib
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    usable_bytes = free_bytes - int(effective_headroom_gib * 2**30)
    if usable_bytes <= 0:
        # This check runs before constructing ``vllm.LLM``.  Failing here is
        # intentional: overriding the utilization fraction cannot make a
        # model fit when the device has less free VRAM than the safety
        # headroom.  Include the logical device and visibility mask so that a
        # standalone re-evaluation launched from a busy training node is
        # diagnosable instead of looking like a generic RAM failure.
        try:
            device_index = torch.cuda.current_device()
        except Exception:  # pragma: no cover - defensive for unusual CUDA setups
            device_index = "unknown"
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<all>")
        raise RuntimeError(
            f"Only {free_bytes / 2**30:.1f} GiB VRAM is free of "
            f"{total_bytes / 2**30:.1f} GiB total, below the "
            f"effective {effective_headroom_gib:.1f} GiB headroom "
            f"({headroom_gib:.1f} GiB device + "
            f"{workspace_headroom_gib:.1f} GiB vLLM workspace) "
            f"(CUDA device {device_index}, CUDA_VISIBLE_DEVICES={visible!r}). "
            "This is usually an existing training/vLLM process or the wrong "
            "GPU. Run `nvidia-smi`, stop only stale processes you own, or "
            "select a free GPU. Setting a numeric gpu_memory_utilization "
            "bypasses this safety check but will still OOM if the model does "
            "not fit in the available VRAM."
        )
    # Keep the effective reserve available when this is the only process too.
    return min(0.98, usable_bytes / total_bytes)


def _prompt(tokenizer, problem: str, config: dict[str, Any]) -> str:
    # Kept for callers that use the historical helper directly. New code
    # renders the complete normalized row so GPQA/AMC23 metadata is retained.
    return render_evaluation_prompt(tokenizer, {"problem": problem, "metadata": {}}, config)


def _release_vllm_engine(engine: Any) -> None:
    """Stop a vLLM engine and release its CUDA allocations synchronously.

    Re-evaluation of many checkpoints happens in one Python process when
    ``world_size=1``.  Relying on process exit for cleanup therefore leaks the
    previous ``LLM`` instance into the next checkpoint.  vLLM has exposed
    different shutdown hooks across releases, so call the available hooks
    defensively, then drop references and flush the local CUDA allocator.
    """
    if engine is None:
        return
    candidates = [engine, getattr(engine, "llm_engine", None)]
    called: set[tuple[int, str]] = set()
    for candidate in candidates:
        if candidate is None:
            continue
        for method_name in ("shutdown_background_loop", "shutdown", "close"):
            method = getattr(candidate, method_name, None)
            marker = (id(candidate), method_name)
            if not callable(method) or marker in called:
                continue
            called.add(marker)
            try:
                method()
            except Exception:
                # Cleanup must not mask the original evaluation exception;
                # object destruction below remains a valid fallback.
                pass
    del candidates
    del engine
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _evaluate_vllm_suite_impl(
    model_name: str,
    model_path: str | Path,
    config: dict[str, Any],
    output_dir: str | Path,
    runtime_settings: dict[str, Any],
) -> dict[str, Any]:
    # Import here so unit tests and the Hugging Face fallback do not require an
    # initialized vLLM/CUDA runtime.
    from vllm import LLM, SamplingParams

    ensure_extended_benchmark_specs(config)
    model_path = Path(model_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    max_new_tokens = int(runtime_settings.get("max_new_tokens", 2048))
    temperature = float(runtime_settings.get("temperature", 0.7))
    top_p = float(runtime_settings.get("top_p", 0.95))
    samples_per_problem = int(runtime_settings.get("num_responses", 8))
    metric_name = evaluation_metric_name(
        samples_per_problem, runtime_settings.get("metric")
    )
    if temperature <= 0:
        raise ValueError("Sampled evaluation requires positive temperature")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("Evaluation top_p must be in (0, 1]")
    if samples_per_problem <= 0:
        raise ValueError("Evaluation num_responses must be positive")
    limit = runtime_settings.get("limit")
    shard_rank = int(runtime_settings.get("_shard_rank", 0))
    shard_world_size = int(runtime_settings.get("_shard_world_size", 1))
    if shard_world_size <= 0 or not 0 <= shard_rank < shard_world_size:
        raise ValueError(
            "Evaluation shard must satisfy 0 <= rank < positive world size: "
            f"rank={shard_rank}, world_size={shard_world_size}"
        )
    benchmark_names = configured_benchmark_names(
        config, runtime_settings.get("benchmark_names")
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    loaded: dict[str, tuple[list[dict[str, str]], dict[str, Any]]] = {}
    prompts: list[str] = []
    prompt_rows: list[tuple[str, dict[str, Any], str]] = []
    for benchmark in benchmark_names:
        records, schema = load_benchmark(
            benchmark, config["evaluation"]["benchmarks"][benchmark]
        )
        if limit is not None:
            records = records[: int(limit)]
        if shard_world_size > 1:
            records = [
                row
                for index, row in enumerate(records)
                if index % shard_world_size == shard_rank
            ]
        loaded[benchmark] = (records, schema)
        for row in records:
            rendered_prompt = render_evaluation_prompt(tokenizer, row, config)
            prompts.append(rendered_prompt)
            prompt_rows.append((benchmark, row, rendered_prompt))

    if prompts:
        tqdm.write(f"Fully rendered EVAL prompt:\n{prompts[0]}")

    vllm_settings = dict(runtime_settings.get("vllm", {}))
    engine_kwargs: dict[str, Any] = {
        "model": str(model_path),
        "tokenizer": str(model_path),
        "dtype": config["models"].get("dtype", "bfloat16"),
        "tensor_parallel_size": int(vllm_settings.get("tensor_parallel_size", 1)),
        "gpu_memory_utilization": _resolve_gpu_memory_utilization(vllm_settings),
        "max_num_seqs": int(vllm_settings.get("max_num_seqs", 256)),
        "enable_prefix_caching": bool(vllm_settings.get("enable_prefix_caching", True)),
        "disable_log_stats": True,
        "seed": int(vllm_settings.get("seed", 1234)),
        "trust_remote_code": False,
        "generation_config": "vllm",
    }
    try:
        checkpoint_config = json.loads(
            (model_path / "config.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError, TypeError):
        checkpoint_config = {}
    if str(checkpoint_config.get("model_type", "")) == "qwen3_5":
        # Evaluation is text-only.  Qwen3.5 snapshots deliberately preserve
        # the official composite config for vLLM 0.17 compatibility, so prune
        # the vision tower rather than allocating unused dummy/missing weights.
        engine_kwargs["language_model_only"] = True
    max_model_len = vllm_settings.get("max_model_len", 4096)
    if max_model_len is not None:
        engine_kwargs["max_model_len"] = int(max_model_len)
    max_num_batched_tokens = vllm_settings.get("max_num_batched_tokens")
    if max_num_batched_tokens is not None:
        engine_kwargs["max_num_batched_tokens"] = int(max_num_batched_tokens)
    for boolean_setting in ("enable_chunked_prefill", "async_scheduling"):
        if vllm_settings.get(boolean_setting) is not None:
            engine_kwargs[boolean_setting] = bool(vllm_settings[boolean_setting])
    performance_mode = vllm_settings.get("performance_mode")
    if performance_mode not in (None, ""):
        engine_kwargs["performance_mode"] = str(performance_mode)

    sample_scope = (
        "full-dataset"
        if shard_world_size == 1
        else f"rank {shard_rank}/{shard_world_size} shard"
    )
    tqdm.write(
        f"Eval {model_name}: loading vLLM engine for "
        f"{len(prompts)} {sample_scope} samples..."
    )
    engine = None
    try:
        engine = LLM(**engine_kwargs)
        sampling = SamplingParams(
            n=samples_per_problem,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
        # One call produces one tqdm progress bar for all configured benchmarks.
        generated = (
            engine.generate(prompts, sampling, use_tqdm=True) if prompts else []
        )
    finally:
        # This is essential for sequential checkpoint re-evaluation in one
        # process. Training-time evaluation already gets process-level cleanup,
        # but the world_size=1 re-eval path does not.
        _release_vllm_engine(engine)
        engine = None
    if len(generated) != len(prompt_rows):
        raise RuntimeError(
            f"vLLM returned {len(generated)} outputs for {len(prompt_rows)} prompts"
        )

    suite: dict[str, Any] = {
        "model": model_name,
        "model_path": str(model_path),
        "benchmarks": {},
        "parameters": {
            "backend": "vllm",
            "max_new_tokens": max_new_tokens,
            "limit": limit,
            "do_sample": temperature > 0,
            "temperature": temperature,
            "top_p": top_p,
            "num_responses": samples_per_problem,
            "metric": metric_name,
            "gpu_headroom_gib": float(vllm_settings.get("gpu_headroom_gib", 4)),
            "gpu_workspace_headroom_gib": float(
                vllm_settings.get("gpu_workspace_headroom_gib", 2)
            ),
            **{key: value for key, value in engine_kwargs.items() if key != "model"},
        },
    }
    grouped: dict[str, list[tuple[dict[str, str], str, list[str]]]] = {
        name: [] for name in benchmark_names
    }
    for (benchmark, row, rendered_prompt), request_output in zip(
        prompt_rows, generated
    ):
        if len(request_output.outputs) != samples_per_problem:
            raise RuntimeError(
                f"vLLM returned {len(request_output.outputs)} responses for one "
                f"problem, expected {samples_per_problem}"
            )
        responses = [item.text for item in request_output.outputs]
        grouped[benchmark].append((row, rendered_prompt, responses))

    grade_progress = tqdm(
        total=len(prompt_rows),
        desc=f"Grade {model_name}",
        unit="sample",
        dynamic_ncols=True,
        leave=True,
        disable=False,
        mininterval=0.2,
    )
    detailed_output_path = output_dir / "model_outputs_detailed.jsonl.gz"
    try:
        with gzip.open(detailed_output_path, "wt", encoding="utf-8") as detailed:
            for benchmark in benchmark_names:
                records, schema = loaded[benchmark]
                prediction_path = output_dir / (
                    f"{benchmark.lower().replace('-', '_')}_predictions.jsonl.gz"
                )
                correct = graded = problems = 0
                problem_score_sum = 0.0
                with gzip.open(prediction_path, "wt", encoding="utf-8") as handle:
                    for row, rendered_prompt, responses in grouped[benchmark]:
                        correctness = [
                            grade_evaluation_response(response, row, benchmark)
                            for response in responses
                        ]
                        correct += sum(map(int, correctness))
                        graded += samples_per_problem
                        problems += 1
                        raw_problem_score = sum(correctness) / samples_per_problem
                        selected_problem_score = metric_problem_score(
                            correctness, metric_name
                        )
                        problem_score_sum += selected_problem_score
                        detailed.write(
                            json.dumps(
                                detailed_model_output_record(
                                    model_name=model_name,
                                    model_path=model_path,
                                    backend="vllm",
                                    benchmark=benchmark,
                                    row=row,
                                    rendered_prompt=rendered_prompt,
                                    responses=responses,
                                    correctness=correctness,
                                    temperature=temperature,
                                    top_p=top_p,
                                    max_new_tokens=max_new_tokens,
                                    seed=int(engine_kwargs["seed"]),
                                ),
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        handle.write(
                            json.dumps(
                                {
                                    **row,
                                    "responses": responses,
                                    "correct": correctness,
                                    "problem_score": raw_problem_score,
                                    "metric_score": selected_problem_score,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        grade_progress.set_postfix_str(
                            f"{benchmark} {metric_name}={problem_score_sum / max(problems, 1):.3f}",
                            refresh=False,
                        )
                        grade_progress.update(1)
                selected_score = problem_score_sum / max(problems, 1)
                benchmark_result = {
                    "correct": correct,
                    "total": graded,
                    "problems": len(records),
                    "samples_per_problem": samples_per_problem,
                    "avg_at_n": selected_score,
                    "accuracy": selected_score,
                    "predictions": str(prediction_path),
                    "schema": schema,
                }
                if metric_name.startswith("pass@"):
                    benchmark_result["pass_at_k"] = selected_score
                    if metric_name == "pass@8":
                        benchmark_result["pass_at_8"] = selected_score
                elif samples_per_problem == 8:
                    benchmark_result["avg_at_8"] = selected_score
                suite["benchmarks"][benchmark] = benchmark_result
    finally:
        grade_progress.close()

    suite["detailed_outputs"] = str(detailed_output_path.resolve())

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(suite, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return suite


def merge_vllm_evaluation_shards(
    model_name: str,
    model_path: str | Path,
    config: dict[str, Any],
    destination: str | Path,
    shard_dirs: list[str | Path],
    runtime_settings: dict[str, Any],
) -> dict[str, Any]:
    """Merge deterministic per-rank vLLM evaluation artifacts.

    Each evaluator rank receives rows ``index % world_size == rank``.  Merging
    by the inverse round-robin restores the original benchmark order, so the
    resulting summary/prediction/detailed-output schema is identical to a
    single evaluator run.  No distributed collective is involved here; this
    function is intentionally filesystem-only and is called by rank 0 after
    every rank has published a successful shard status.
    """
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    shard_paths = [Path(path).resolve() for path in shard_dirs]
    if not shard_paths:
        raise ValueError("At least one evaluation shard is required")
    benchmark_names = configured_benchmark_names(
        config, runtime_settings.get("benchmark_names")
    )
    loaded: dict[str, tuple[list[dict[str, str]], dict[str, Any]]] = {}
    for benchmark in benchmark_names:
        records, schema = load_benchmark(
            benchmark, config["evaluation"]["benchmarks"][benchmark]
        )
        if runtime_settings.get("limit") is not None:
            records = records[: int(runtime_settings["limit"])]
        loaded[benchmark] = (records, schema)

    suites = []
    for shard in shard_paths:
        summary_path = shard / "summary.json"
        if not summary_path.is_file():
            raise RuntimeError(f"Missing evaluation shard summary: {summary_path}")
        suite = json.loads(summary_path.read_text(encoding="utf-8"))
        if tuple(suite.get("benchmarks", {})) != benchmark_names:
            raise RuntimeError(
                f"Evaluation shard has unexpected benchmark order: {summary_path}"
            )
        suites.append(suite)
    expected_parameters = {
        "backend": "vllm",
        "max_new_tokens": int(runtime_settings["max_new_tokens"]),
        "limit": runtime_settings.get("limit"),
        "temperature": float(runtime_settings["temperature"]),
        "top_p": float(runtime_settings["top_p"]),
        "num_responses": int(runtime_settings["num_responses"]),
        "metric": evaluation_metric_name(
            int(runtime_settings["num_responses"]), runtime_settings.get("metric")
        ),
    }
    for suite in suites:
        parameters = suite.get("parameters", {})
        for key, expected in expected_parameters.items():
            actual = parameters.get(key)
            if isinstance(expected, float):
                try:
                    matches = float(actual) == expected
                except (TypeError, ValueError):
                    matches = False
            else:
                matches = actual == expected
            if not matches:
                raise RuntimeError(
                    f"Evaluation shard parameter mismatch for {key}: "
                    f"expected {expected!r}, got {actual!r}"
                )

    def read_prediction_rows(path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            raise RuntimeError(f"Missing evaluation prediction shard: {path}")
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def interleave(rows_by_rank: list[list[Any]]) -> list[Any]:
        rows: list[Any] = []
        for local_index in range(max((len(rows) for rows in rows_by_rank), default=0)):
            for rank_rows in rows_by_rank:
                if local_index < len(rank_rows):
                    rows.append(rank_rows[local_index])
        return rows

    merged_benchmarks: dict[str, dict[str, Any]] = {}
    for benchmark in benchmark_names:
        records, schema = loaded[benchmark]
        rows_by_rank = [
            read_prediction_rows(
                shard
                / Path(suite["benchmarks"][benchmark]["predictions"]).name
            )
            for shard, suite in zip(shard_paths, suites)
        ]
        merged_rows = interleave(rows_by_rank)
        expected_rows = [(row["id"], row["problem"], row["answer"]) for row in records]
        actual_rows = [
            (row.get("id"), row.get("problem"), row.get("answer"))
            for row in merged_rows
        ]
        if actual_rows != expected_rows:
            raise RuntimeError(
                f"Evaluation shard merge changed {benchmark} problem ordering or "
                f"dropped/duplicated problems: expected {len(expected_rows)}, "
                f"got {len(actual_rows)}"
            )
        samples_per_problem = int(runtime_settings["num_responses"])
        for row in merged_rows:
            if len(row.get("correct", [])) != samples_per_problem:
                raise RuntimeError(
                    f"Evaluation shard has an incomplete response set for {benchmark} "
                    f"problem {row.get('id')!r}: expected {samples_per_problem}"
                )
        prediction_path = destination / (
            f"{benchmark.lower().replace('-', '_')}_predictions.jsonl.gz"
        )
        correct = sum(
            int(is_correct)
            for row in merged_rows
            for is_correct in row.get("correct", [])
        )
        total = sum(len(row.get("correct", [])) for row in merged_rows)
        # Recompute from per-response grades instead of trusting a shard's
        # aggregate field; this is numerically identical to the single-GPU
        # evaluator and makes the merged metric auditable.
        problem_score_sum = sum(
            metric_problem_score(
                list(map(bool, row["correct"])),
                evaluation_metric_name(
                    samples_per_problem, runtime_settings.get("metric")
                ),
            )
            for row in merged_rows
        )
        with gzip.open(prediction_path, "wt", encoding="utf-8") as handle:
            for row in merged_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        result = {
            "correct": correct,
            "total": total,
            "problems": len(merged_rows),
            "samples_per_problem": samples_per_problem,
            "avg_at_n": problem_score_sum / max(len(merged_rows), 1),
            "accuracy": problem_score_sum / max(len(merged_rows), 1),
            "predictions": str(prediction_path.resolve()),
            "schema": schema,
        }
        if expected_parameters["metric"].startswith("pass@"):
            result["pass_at_k"] = result["accuracy"]
            if expected_parameters["metric"] == "pass@8":
                result["pass_at_8"] = result["accuracy"]
        elif samples_per_problem == 8:
            result["avg_at_8"] = result["avg_at_n"]
        merged_benchmarks[benchmark] = result

    detailed_rows_by_rank: list[list[dict[str, Any]]] = []
    for shard, suite in zip(shard_paths, suites):
        path = shard / Path(suite["detailed_outputs"]).name
        if not path.is_file():
            raise RuntimeError(f"Missing detailed evaluation shard: {path}")
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            detailed_rows_by_rank.append(
                [json.loads(line) for line in handle if line.strip()]
            )
    detailed_by_benchmark: dict[str, list[list[dict[str, Any]]]] = {
        benchmark: [] for benchmark in benchmark_names
    }
    for rank_rows in detailed_rows_by_rank:
        for benchmark in benchmark_names:
            detailed_by_benchmark[benchmark].append(
                [row for row in rank_rows if row.get("benchmark") == benchmark]
            )
    detailed_output_path = destination / "model_outputs_detailed.jsonl.gz"
    with gzip.open(detailed_output_path, "wt", encoding="utf-8") as handle:
        for benchmark in benchmark_names:
            merged_detailed = interleave(detailed_by_benchmark[benchmark])
            if len(merged_detailed) != merged_benchmarks[benchmark]["problems"]:
                raise RuntimeError(
                    f"Detailed output count mismatch for {benchmark}: "
                    f"{len(merged_detailed)} vs {merged_benchmarks[benchmark]['problems']}"
                )
            expected_ids = [row["id"] for row in loaded[benchmark][0]]
            actual_ids = [row.get("problem_id") for row in merged_detailed]
            if actual_ids != expected_ids:
                raise RuntimeError(
                    f"Detailed evaluation shard merge changed {benchmark} problem "
                    "ordering or dropped/duplicated detailed rows"
                )
            for row in merged_detailed:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    first_parameters = dict(suites[0]["parameters"])
    first_parameters["tensor_parallel_size"] = 1
    suite = {
        "model": model_name,
        "model_path": str(Path(model_path).resolve()),
        "benchmarks": merged_benchmarks,
        "parameters": first_parameters,
        "detailed_outputs": str(detailed_output_path.resolve()),
    }
    with (destination / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(suite, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return suite


def _evaluate_vllm_distributed_impl(
    model_name: str,
    model_path: str | Path,
    config: dict[str, Any],
    output_dir: str | Path,
    runtime_settings: dict[str, Any],
    resolved_config_path: str | Path,
    *,
    world_size: int | None = None,
) -> dict[str, Any]:
    """Evaluate with one TP=1 vLLM child per visible GPU and merge the shards."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    devices = [item.strip() for item in visible.split(",") if item.strip()]
    if not devices:
        devices = [str(index) for index in range(torch.cuda.device_count())]
    requested_world = int((len(devices) or 1) if world_size is None else world_size)
    if requested_world <= 0:
        raise ValueError("world_size must be a positive integer")
    if requested_world > len(devices):
        raise ValueError(
            f"Requested {requested_world} evaluation workers but only "
            f"{len(devices)} GPUs are visible ({visible!r})"
        )

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    coordination = Path(
        tempfile.mkdtemp(prefix=".distributed-eval-", dir=output_dir.parent)
    )
    shard_dirs = [coordination / f"rank-{rank:05d}" for rank in range(requested_world)]
    processes: list[subprocess.Popen] = []
    commands: list[list[str]] = []
    try:
        repo_root = Path(__file__).resolve().parents[1]
        for rank, shard_dir in enumerate(shard_dirs):
            environment = isolate_distributed_subprocess_environment()
            environment["VLLM_LOGGING_LEVEL"] = environment.get(
                "VLLM_LOGGING_LEVEL", "WARNING"
            )
            environment["PYTHONUNBUFFERED"] = "1"
            # The parent owns the per-GPU lock for the lifetime of this
            # distributed evaluation. Child evaluators must not try to lock
            # the same device again after their inherited file descriptors
            # are closed by Popen.
            environment["BELLMANOPD_EVAL_LOCK_HELD"] = "1"
            environment["PYTHONPATH"] = os.pathsep.join(
                filter(None, (str(repo_root), environment.get("PYTHONPATH", "")))
            )
            environment["CUDA_VISIBLE_DEVICES"] = devices[rank]
            child_settings = {
                **runtime_settings,
                "_shard_rank": rank,
                "_shard_world_size": requested_world,
                "vllm": {
                    **dict(runtime_settings.get("vllm", {})),
                    "tensor_parallel_size": 1,
                },
            }
            command = [
                sys.executable,
                "-m",
                "b200_experiment.vllm_evaluation",
                "--config",
                str(Path(resolved_config_path).resolve()),
                "--model",
                str(Path(model_path).resolve()),
                "--name",
                model_name,
                "--output",
                str(shard_dir.resolve()),
                "--settings-json",
                json.dumps(child_settings),
            ]
            commands.append(command)
            processes.append(
                subprocess.Popen(
                    command,
                    cwd=repo_root,
                    env=environment,
                    # Keep vLLM EngineCore/worker descendants in a dedicated
                    # process group so failure cleanup cannot orphan GPU users.
                    start_new_session=True,
                )
            )
        timeout = float(runtime_settings.get("sync_timeout_sec", 24 * 60 * 60))
        if timeout <= 0:
            raise ValueError("sync_timeout_sec must be positive")
        deadline = time.monotonic() + timeout
        while processes:
            for index, process in enumerate(processes):
                if process.poll() is not None and process.returncode != 0:
                    raise subprocess.CalledProcessError(
                        process.returncode, commands[index]
                    )
            if all(process.poll() is not None for process in processes):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Distributed vLLM evaluation exceeded {timeout:.1f}s"
                )
            time.sleep(0.25)
        # A launcher can exit after writing summary.json while an EngineCore
        # worker is still shutting down. Reap/terminate the dedicated process
        # groups before releasing the GPU locks, otherwise the next independent
        # eval can observe stale VRAM usage for a short window.
        for process in processes:
            _terminate_process_group(process)
        return merge_vllm_evaluation_shards(
            model_name,
            model_path,
            config,
            output_dir,
            shard_dirs,
            runtime_settings,
        )
    except BaseException:
        for process in processes:
            # Also attempt cleanup for launchers that already exited with an
            # error: their vLLM EngineCore descendants can outlive the parent.
            _terminate_process_group(process)
        raise
    finally:
        shutil.rmtree(coordination, ignore_errors=True)


def evaluate_vllm_distributed(
    model_name: str,
    model_path: str | Path,
    config: dict[str, Any],
    output_dir: str | Path,
    runtime_settings: dict[str, Any],
    resolved_config_path: str | Path,
    *,
    world_size: int | None = None,
) -> dict[str, Any]:
    """Run evaluation with exclusive GPU ownership and clean failure teardown.

    The implementation remains one TP=1 child per selected GPU.  The
    filesystem lock only prevents independent re-evaluation commands from
    concurrently placing two vLLM engines on the same device; it does not
    change sampling, sharding, or the evaluation result.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    devices = [item.strip() for item in visible.split(",") if item.strip()]
    if not devices:
        devices = [str(index) for index in range(torch.cuda.device_count())]
    requested_world = int((len(devices) or 1) if world_size is None else world_size)
    if requested_world <= 0:
        raise ValueError("world_size must be a positive integer")
    if requested_world > len(devices):
        raise ValueError(
            f"Requested {requested_world} evaluation workers but only "
            f"{len(devices)} GPUs are visible ({visible!r})"
        )
    lock_timeout = float(
        runtime_settings.get(
            "gpu_lock_timeout_sec",
            runtime_settings.get("vllm", {}).get(
                "gpu_lock_timeout_sec", 24 * 60 * 60
            ),
        )
    )
    if lock_timeout <= 0:
        raise ValueError("gpu_lock_timeout_sec must be positive")
    with _exclusive_evaluation_gpu_locks(devices[:requested_world], lock_timeout):
        return _evaluate_vllm_distributed_impl(
            model_name,
            model_path,
            config,
            output_dir,
            runtime_settings,
            resolved_config_path,
            world_size=requested_world,
        )


def evaluate_vllm_suite(
    model_name: str,
    model_path: str | Path,
    config: dict[str, Any],
    output_dir: str | Path,
    runtime_settings: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate one vLLM engine while serializing access to its GPU.

    Distributed parent evaluators set ``BELLMANOPD_EVAL_LOCK_HELD`` before
    spawning their children. Direct single-GPU evaluation and training-time
    evaluator children acquire the same lock here, preventing independent
    commands from racing on one device.
    """
    if os.environ.get("BELLMANOPD_EVAL_LOCK_HELD") == "1":
        return _evaluate_vllm_suite_impl(
            model_name, model_path, config, output_dir, runtime_settings
        )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    devices = [item.strip() for item in visible.split(",") if item.strip()]
    if not devices:
        devices = [str(index) for index in range(torch.cuda.device_count())]
    lock_timeout = float(
        runtime_settings.get(
            "gpu_lock_timeout_sec",
            runtime_settings.get("vllm", {}).get(
                "gpu_lock_timeout_sec", 24 * 60 * 60
            ),
        )
    )
    if lock_timeout <= 0:
        raise ValueError("gpu_lock_timeout_sec must be positive")
    with _exclusive_evaluation_gpu_locks(devices[:1] or devices, lock_timeout):
        return _evaluate_vllm_suite_impl(
            model_name, model_path, config, output_dir, runtime_settings
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="vLLM periodic evaluator")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--settings-json", required=True)
    return parser


def main() -> int:
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    args = build_parser().parse_args()
    evaluate_vllm_suite(
        args.name,
        args.model,
        load_config(args.config),
        args.output,
        json.loads(args.settings_json),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
