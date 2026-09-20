from __future__ import annotations

import gzip
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from b200_experiment.trainer import (
    _run_distributed_vllm_evaluation,
    _run_training_evaluation,
)


class _FilesystemDistributedContext:
    def __init__(self, rank: int, world_size: int = 2):
        self.rank = rank
        self.world_size = world_size

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def _config(root: Path) -> dict:
    benchmark = root / "benchmark.jsonl"
    benchmark.write_text(
        "".join(
            json.dumps({"problem": f"problem-{index}", "answer": "1"}) + "\n"
            for index in range(5)
        ),
        encoding="utf-8",
    )
    return {
        "models": {"student_path": str(root), "dtype": "bfloat16"},
        "data": {"chat_template_kwargs": {}},
        "training_evaluation": {
            "output_subdir": "training_eval",
            "reuse_base_evaluation": False,
            "sync_timeout_sec": 10,
        },
        "evaluation": {
            "benchmarks": {"MATH-500": {"path": str(benchmark)}}
        },
    }


def _fake_evaluator(
    _model,
    _tokenizer,
    model_name,
    _model_path,
    _step,
    _config,
    _resolved_config_path,
    output_dir,
    runtime_settings,
    **kwargs,
):
    rank = int(kwargs.get("distributed_rank", 0))
    world_size = int(kwargs.get("distributed_world_size", 1))
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = [
        {"id": str(index), "problem": f"problem-{index}", "answer": "1"}
        for index in range(5)
    ]
    rows = [row for index, row in enumerate(all_rows) if index % world_size == rank]
    prediction_path = output_dir / "math_500_predictions.jsonl.gz"
    with gzip.open(prediction_path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    {
                        **row,
                        "responses": ["1"],
                        "correct": [True],
                        "problem_score": 1.0,
                    }
                )
                + "\n"
            )
    detailed_path = output_dir / "model_outputs_detailed.jsonl.gz"
    with gzip.open(detailed_path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    {
                        "benchmark": "MATH-500",
                        "problem_id": row["id"],
                        "model_name": model_name,
                    }
                )
                + "\n"
            )
    suite = {
        "model": model_name,
        "model_path": str(Path(_model_path).resolve()),
        "benchmarks": {
            "MATH-500": {
                "correct": len(rows),
                "total": len(rows),
                "problems": len(rows),
                "samples_per_problem": 1,
                "avg_at_n": 1.0,
                "accuracy": 1.0,
                "predictions": str(prediction_path.resolve()),
                "schema": {"benchmark": "MATH-500"},
            }
        },
        "parameters": {
            "backend": "vllm",
            "max_new_tokens": int(runtime_settings["max_new_tokens"]),
            "limit": None,
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 1.0,
            "num_responses": 1,
            "metric": "accuracy",
            "tensor_parallel_size": 1,
        },
        "detailed_outputs": str(detailed_path.resolve()),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(suite) + "\n", encoding="utf-8"
    )
    return suite


class DistributedTrainingEvaluationTests(unittest.TestCase):
    def test_single_gpu_path_keeps_root_evaluation_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _config(root)
            output_dir = root / "single-run"
            settings = {
                "backend": "vllm",
                "temperature": 1.0,
                "top_p": 1.0,
                "num_responses": 1,
                "batch_size": 1,
                "max_new_tokens": 8,
                "limit": None,
                "benchmark_names": ["MATH-500"],
                "vllm": {"tensor_parallel_size": 1},
            }
            with patch(
                "b200_experiment.trainer._evaluate_vllm_subprocess",
                side_effect=_fake_evaluator,
            ):
                history = _run_training_evaluation(
                    None,
                    None,
                    "opd",
                    1,
                    1,
                    {**config, "training_evaluation": {**config["training_evaluation"], **settings}},
                    output_dir,
                    root / "resolved.yaml",
                    checkpoint=root,
                )
            self.assertEqual(history["benchmarks"]["MATH-500"]["problems"], 5)
            self.assertTrue((output_dir / "training_eval/step-000001/summary.json").is_file())
            self.assertTrue((output_dir / "eval_history.jsonl").is_file())
            self.assertFalse(list((output_dir / "training_eval/step-000001").glob(".rank-*")))

    def test_cached_base_evaluation_skips_all_rank_generators(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _config(root)
            model_dir = root / "model"
            model_dir.mkdir()
            config["models"]["student_path"] = str(model_dir)
            config["training_evaluation"]["reuse_base_evaluation"] = True
            # Keep the cache outside the model directory so its files do not
            # alter the model signature used by the cache key.
            config["training_evaluation"]["base_cache_dir"] = str(root / "cache")
            settings = {
                "backend": "vllm",
                "temperature": 1.0,
                "top_p": 1.0,
                "num_responses": 1,
                "batch_size": 1,
                "max_new_tokens": 8,
                "limit": None,
                "benchmark_names": ["MATH-500"],
                "vllm": {"tensor_parallel_size": 1},
            }

            def run_pair(output_dir: Path, evaluator):
                errors = []

                def worker(rank: int):
                    try:
                        _run_distributed_vllm_evaluation(
                            model=None,
                            tokenizer=None,
                            method="opd",
                            step=0,
                            config=config,
                            output_dir=output_dir,
                            resolved_config_path=root / "resolved.yaml",
                            checkpoint=None,
                            runtime_settings=settings,
                            distributed=_FilesystemDistributedContext(rank),
                        )
                    except Exception as exc:
                        errors.append(exc)

                with patch(
                    "b200_experiment.trainer._evaluate_vllm_subprocess",
                    side_effect=evaluator,
                ):
                    threads = [
                        threading.Thread(target=worker, args=(rank,))
                        for rank in range(2)
                    ]
                    for thread in threads:
                        thread.start()
                    for thread in threads:
                        thread.join(timeout=10)
                self.assertFalse(errors)

            run_pair(root / "first", _fake_evaluator)
            with patch(
                "b200_experiment.trainer._evaluate_vllm_subprocess",
                side_effect=AssertionError("cache should avoid generation"),
            ) as evaluator:
                # The cache-hit path is coordinated by files alone; neither
                # rank should invoke the evaluator subprocess.
                errors = []

                def cached_worker(rank: int):
                    try:
                        _run_distributed_vllm_evaluation(
                            model=None,
                            tokenizer=None,
                            method="opd",
                            step=0,
                            config=config,
                            output_dir=root / "second",
                            resolved_config_path=root / "resolved.yaml",
                            checkpoint=None,
                            runtime_settings=settings,
                            distributed=_FilesystemDistributedContext(rank),
                        )
                    except Exception as exc:
                        errors.append(exc)

                threads = [
                    threading.Thread(target=cached_worker, args=(rank,))
                    for rank in range(2)
                ]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=10)
                self.assertFalse(errors)
                evaluator.assert_not_called()

    def test_two_rank_filesystem_merge_has_no_collective_during_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _config(root)
            output_dir = root / "run"
            settings = {
                "backend": "vllm",
                "temperature": 1.0,
                "top_p": 1.0,
                "num_responses": 1,
                "batch_size": 1,
                "max_new_tokens": 8,
                "limit": None,
                "benchmark_names": ["MATH-500"],
                "vllm": {"tensor_parallel_size": 1},
            }
            results = {}
            errors = []

            def worker(rank: int):
                try:
                    results[rank] = _run_distributed_vllm_evaluation(
                        model=None,
                        tokenizer=None,
                        method="opd",
                        step=1,
                        config=config,
                        output_dir=output_dir,
                        resolved_config_path=root / "resolved.yaml",
                        checkpoint=root,
                        runtime_settings=settings,
                        distributed=_FilesystemDistributedContext(rank),
                    )
                except Exception as exc:  # pragma: no cover - asserted below
                    errors.append(exc)

            with patch(
                "b200_experiment.trainer._evaluate_vllm_subprocess",
                side_effect=_fake_evaluator,
            ):
                threads = [threading.Thread(target=worker, args=(rank,)) for rank in range(2)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=10)

            self.assertFalse(errors)
            self.assertEqual(set(results), {0, 1})
            self.assertEqual(results[0][0]["benchmarks"]["MATH-500"]["problems"], 5)
            self.assertEqual(results[1][0]["benchmarks"]["MATH-500"]["accuracy"], 1.0)
            self.assertTrue((output_dir / "training_eval/step-000001/summary.json").is_file())
            self.assertFalse((output_dir / "training_eval/step-000001/.rank-00000").exists())
            self.assertFalse((output_dir / "training_eval/.distributed-step-000001").exists())

    def test_rank_failure_is_published_without_waiting_for_a_collective(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _config(root)
            config["training_evaluation"]["sync_timeout_sec"] = 3
            output_dir = root / "run"
            settings = {
                "backend": "vllm",
                "temperature": 1.0,
                "top_p": 1.0,
                "num_responses": 1,
                "batch_size": 1,
                "max_new_tokens": 8,
                "limit": None,
                "benchmark_names": ["MATH-500"],
                "vllm": {"tensor_parallel_size": 1},
            }

            def failing_evaluator(*args, **kwargs):
                if int(kwargs["distributed_rank"]) == 1:
                    raise RuntimeError("synthetic rank failure")
                time.sleep(0.2)
                return _fake_evaluator(*args, **kwargs)

            errors = []

            def worker(rank: int):
                try:
                    _run_distributed_vllm_evaluation(
                        model=None,
                        tokenizer=None,
                        method="opd",
                        step=1,
                        config=config,
                        output_dir=output_dir,
                        resolved_config_path=root / "resolved.yaml",
                        checkpoint=root,
                        runtime_settings=settings,
                        distributed=_FilesystemDistributedContext(rank),
                    )
                except Exception as exc:
                    errors.append(str(exc))

            with patch(
                "b200_experiment.trainer._evaluate_vllm_subprocess",
                side_effect=failing_evaluator,
            ):
                threads = [threading.Thread(target=worker, args=(rank,)) for rank in range(2)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=5)

            self.assertEqual(len(errors), 2)
            self.assertTrue(any("synthetic rank failure" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
