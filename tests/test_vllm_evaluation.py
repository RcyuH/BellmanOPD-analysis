from __future__ import annotations

import gzip
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from b200_experiment.evaluation import metric_problem_score
from b200_experiment.vllm_evaluation import (
    _resolve_gpu_memory_utilization,
    _release_vllm_engine,
    evaluate_vllm_suite,
    merge_vllm_evaluation_shards,
)


class _Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]


class _SamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _LLM:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.generate_calls = []
        self.__class__.instances.append(self)

    def generate(self, prompts, sampling, use_tqdm):
        self.generate_calls.append((prompts, sampling, use_tqdm))
        return [
            types.SimpleNamespace(
                outputs=[
                    types.SimpleNamespace(text="1")
                    for _ in range(sampling.kwargs.get("n", 1))
                ]
            )
            for _ in prompts
        ]


class _ClosableEngine:
    def __init__(self):
        self.shutdown_calls = 0

    def shutdown(self):
        self.shutdown_calls += 1


class VllmEvaluationTests(unittest.TestCase):
    def test_engine_cleanup_calls_shutdown_hook(self):
        engine = _ClosableEngine()
        with patch("torch.cuda.is_available", return_value=False):
            _release_vllm_engine(engine)
        self.assertEqual(engine.shutdown_calls, 1)

    def test_pass_at_k_uses_unbiased_without_replacement_estimator(self):
        self.assertAlmostEqual(
            metric_problem_score([True] + [False] * 15, "pass@8"), 0.5
        )

    def test_pass_at_8_is_reported_and_uses_problem_level_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            benchmark_path = root / "math.jsonl"
            benchmark_path.write_text(
                json.dumps({"problem": "1?", "answer": "1"}) + "\n",
                encoding="utf-8",
            )
            config = {
                "models": {"dtype": "bfloat16"},
                "data": {"chat_template_kwargs": {}},
                "evaluation": {
                    "benchmarks": {"MATH-500": {"path": str(benchmark_path)}}
                },
            }
            settings = {
                "max_new_tokens": 8,
                "temperature": 1.0,
                "top_p": 1.0,
                "num_responses": 8,
                "metric": "pass@8",
                "benchmark_names": ["MATH-500"],
                "vllm": {"gpu_memory_utilization": 0.4},
            }
            fake_vllm = types.SimpleNamespace(LLM=_LLM, SamplingParams=_SamplingParams)
            with (
                patch.dict(sys.modules, {"vllm": fake_vllm}),
                patch(
                    "b200_experiment.vllm_evaluation.AutoTokenizer.from_pretrained",
                    return_value=_Tokenizer(),
                ),
            ):
                suite = evaluate_vllm_suite(
                    "student", root, config, root / "results", settings
                )
            result = suite["benchmarks"]["MATH-500"]
            self.assertEqual(suite["parameters"]["metric"], "pass@8")
            self.assertEqual(result["pass_at_k"], 1.0)
            self.assertEqual(result["pass_at_8"], 1.0)
            self.assertEqual(result["accuracy"], 1.0)

    def test_distributed_shard_only_generates_its_round_robin_problems(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            benchmark_path = root / "math.jsonl"
            benchmark_path.write_text(
                "".join(
                    json.dumps({"problem": f"{i}?", "answer": str(i)}) + "\n"
                    for i in range(4)
                ),
                encoding="utf-8",
            )
            config = {
                "models": {"dtype": "bfloat16"},
                "data": {"chat_template_kwargs": {}},
                "evaluation": {
                    "benchmarks": {"MATH-500": {"path": str(benchmark_path)}}
                },
            }
            settings = {
                "max_new_tokens": 8,
                "temperature": 1.0,
                "top_p": 1.0,
                "num_responses": 1,
                "benchmark_names": ["MATH-500"],
                "_shard_rank": 1,
                "_shard_world_size": 2,
                "vllm": {"gpu_memory_utilization": 0.4},
            }
            fake_vllm = types.SimpleNamespace(LLM=_LLM, SamplingParams=_SamplingParams)
            _LLM.instances.clear()
            with (
                patch.dict(sys.modules, {"vllm": fake_vllm}),
                patch(
                    "b200_experiment.vllm_evaluation.AutoTokenizer.from_pretrained",
                    return_value=_Tokenizer(),
                ),
            ):
                suite = evaluate_vllm_suite(
                    "student", root, config, root / "results", settings
                )
            self.assertEqual(len(_LLM.instances[-1].generate_calls[0][0]), 2)
            self.assertEqual(suite["benchmarks"]["MATH-500"]["problems"], 2)

    def test_one_response_reports_accuracy_without_avg16_label(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            benchmark_path = root / "math.jsonl"
            benchmark_path.write_text(
                json.dumps({"problem": "1?", "answer": "1"}) + "\n",
                encoding="utf-8",
            )
            config = {
                "models": {"dtype": "bfloat16"},
                "data": {"chat_template_kwargs": {}},
                "evaluation": {
                    "benchmarks": {"MATH-500": {"path": str(benchmark_path)}}
                },
            }
            settings = {
                "max_new_tokens": 8,
                "temperature": 1.0,
                "top_p": 1.0,
                "num_responses": 1,
                "benchmark_names": ["MATH-500"],
                "vllm": {"gpu_memory_utilization": 0.4},
            }
            fake_vllm = types.SimpleNamespace(LLM=_LLM, SamplingParams=_SamplingParams)
            _LLM.instances.clear()
            with (
                patch.dict(sys.modules, {"vllm": fake_vllm}),
                patch(
                    "b200_experiment.vllm_evaluation.AutoTokenizer.from_pretrained",
                    return_value=_Tokenizer(),
                ),
            ):
                suite = evaluate_vllm_suite(
                    "student", root, config, root / "results", settings
                )

            result = suite["benchmarks"]["MATH-500"]
            self.assertEqual(suite["parameters"]["metric"], "accuracy")
            self.assertEqual(result["total"], 1)
            self.assertEqual(result["avg_at_n"], 1.0)
            self.assertNotIn("avg_at_8", result)
            detailed_path = Path(suite["detailed_outputs"])
            self.assertEqual(detailed_path.name, "model_outputs_detailed.jsonl.gz")
            with gzip.open(detailed_path, "rt", encoding="utf-8") as handle:
                detailed_rows = [json.loads(line) for line in handle]
            self.assertEqual(len(detailed_rows), 1)
            detailed = detailed_rows[0]
            self.assertEqual(detailed["model_name"], "student")
            self.assertEqual(detailed["backend"], "vllm")
            self.assertEqual(detailed["benchmark"], "MATH-500")
            self.assertEqual(
                detailed["rendered_prompt"],
                "1? Let's think step by step and output the final answer within \\boxed{}.",
            )
            self.assertEqual(detailed["samples_per_problem"], 1)
            self.assertEqual(detailed["outputs"][0]["response"], "1")
            self.assertTrue(detailed["outputs"][0]["correct"])

    def test_auto_memory_uses_all_free_vram_except_headroom(self):
        gib = 2**30
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.mem_get_info", return_value=(160 * gib, 180 * gib)),
        ):
            utilization = _resolve_gpu_memory_utilization(
                {"gpu_memory_utilization": "auto", "gpu_headroom_gib": 4}
            )
        # Auto sizing reserves 4 GiB device headroom plus the default 2 GiB
        # transient vLLM workspace reserve.
        self.assertAlmostEqual(utilization, 154 / 180)

    def test_auto_memory_workspace_headroom_can_be_disabled_explicitly(self):
        gib = 2**30
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.mem_get_info", return_value=(160 * gib, 180 * gib)),
        ):
            utilization = _resolve_gpu_memory_utilization(
                {
                    "gpu_memory_utilization": "auto",
                    "gpu_headroom_gib": 4,
                    "gpu_workspace_headroom_gib": 0,
                }
            )
        self.assertAlmostEqual(utilization, 156 / 180)

    def test_auto_memory_error_identifies_vram_device_and_visibility(self):
        gib = 2**30
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.mem_get_info", return_value=(int(0.7 * gib), 24 * gib)),
            patch("torch.cuda.current_device", return_value=2),
            patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "2"}, clear=False),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                r"Only 0\.7 GiB VRAM is free.*effective 6\.0 GiB headroom.*CUDA device 2.*CUDA_VISIBLE_DEVICES='2'.*nvidia-smi",
            ):
                _resolve_gpu_memory_utilization(
                    {"gpu_memory_utilization": "auto", "gpu_headroom_gib": 4}
                )

    def test_all_benchmarks_share_one_generate_progress_bar(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            benchmarks = {}
            for name in ("MATH-500", "AIME24", "AIME25"):
                path = root / f"{name}.jsonl"
                path.write_text(
                    json.dumps({"problem": f"{name}: 1?", "answer": "1"}) + "\n",
                    encoding="utf-8",
                )
                benchmarks[name] = {"path": str(path)}
            config = {
                "models": {"dtype": "bfloat16"},
                "data": {"chat_template_kwargs": {"enable_thinking": True}},
                "evaluation": {"benchmarks": benchmarks},
            }
            settings = {
                "max_new_tokens": 32,
                "temperature": 1.0,
                "top_p": 0.95,
                "num_responses": 8,
                "benchmark_names": list(benchmarks),
                "vllm": {
                    "max_num_seqs": 64,
                    "max_num_batched_tokens": 4096,
                    "gpu_memory_utilization": 0.4,
                    "enable_chunked_prefill": True,
                    "async_scheduling": True,
                    "performance_mode": "throughput",
                },
            }
            fake_vllm = types.SimpleNamespace(LLM=_LLM, SamplingParams=_SamplingParams)
            _LLM.instances.clear()
            with (
                patch.dict(sys.modules, {"vllm": fake_vllm}),
                patch(
                    "b200_experiment.vllm_evaluation.AutoTokenizer.from_pretrained",
                    return_value=_Tokenizer(),
                ),
            ):
                suite = evaluate_vllm_suite(
                    "student", root, config, root / "results", settings
                )

            self.assertEqual(len(_LLM.instances), 1)
            calls = _LLM.instances[0].generate_calls
            self.assertEqual(len(calls), 1)
            self.assertEqual(len(calls[0][0]), 3)
            self.assertTrue(calls[0][2])
            self.assertEqual(_LLM.instances[0].kwargs["max_num_seqs"], 64)
            self.assertEqual(_LLM.instances[0].kwargs["max_num_batched_tokens"], 4096)
            self.assertTrue(_LLM.instances[0].kwargs["enable_chunked_prefill"])
            self.assertTrue(_LLM.instances[0].kwargs["async_scheduling"])
            self.assertEqual(_LLM.instances[0].kwargs["performance_mode"], "throughput")
            self.assertEqual(calls[0][1].kwargs["temperature"], 1.0)
            self.assertEqual(calls[0][1].kwargs["top_p"], 0.95)
            self.assertEqual(calls[0][1].kwargs["n"], 8)
            self.assertTrue(suite["parameters"]["do_sample"])
            self.assertEqual(
                [suite["benchmarks"][name]["accuracy"] for name in benchmarks],
                [1.0, 1.0, 1.0],
            )
            self.assertEqual(suite["benchmarks"]["MATH-500"]["total"], 8)
            self.assertEqual(suite["benchmarks"]["MATH-500"]["problems"], 1)

    def test_two_rank_shards_merge_to_single_gpu_metrics_and_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            benchmark_path = root / "math.jsonl"
            benchmark_path.write_text(
                "".join(
                    json.dumps({"problem": f"{index}?", "answer": "1"}) + "\n"
                    for index in range(5)
                ),
                encoding="utf-8",
            )
            config = {
                "models": {"dtype": "bfloat16"},
                "data": {"chat_template_kwargs": {}},
                "evaluation": {
                    "benchmarks": {"MATH-500": {"path": str(benchmark_path)}}
                },
            }
            settings = {
                "max_new_tokens": 8,
                "temperature": 1.0,
                "top_p": 1.0,
                "num_responses": 1,
                "benchmark_names": ["MATH-500"],
                "vllm": {"gpu_memory_utilization": 0.4},
            }
            fake_vllm = types.SimpleNamespace(LLM=_LLM, SamplingParams=_SamplingParams)
            _LLM.instances.clear()
            with (
                patch.dict(sys.modules, {"vllm": fake_vllm}),
                patch(
                    "b200_experiment.vllm_evaluation.AutoTokenizer.from_pretrained",
                    return_value=_Tokenizer(),
                ),
            ):
                single = evaluate_vllm_suite(
                    "student", root, config, root / "single", settings
                )
                for rank in range(2):
                    evaluate_vllm_suite(
                        "student",
                        root,
                        config,
                        root / f"rank-{rank}",
                        {
                            **settings,
                            "_shard_rank": rank,
                            "_shard_world_size": 2,
                        },
                    )
                merged = merge_vllm_evaluation_shards(
                    "student",
                    root,
                    config,
                    root / "merged",
                    [root / "rank-0", root / "rank-1"],
                    settings,
                )

            single_result = single["benchmarks"]["MATH-500"]
            merged_result = merged["benchmarks"]["MATH-500"]
            for key in ("correct", "total", "problems", "samples_per_problem", "accuracy"):
                self.assertEqual(merged_result[key], single_result[key])
            with gzip.open(
                merged_result["predictions"], "rt", encoding="utf-8"
            ) as handle:
                merged_rows = [json.loads(line) for line in handle]
            self.assertEqual([row["id"] for row in merged_rows], [str(i) for i in range(5)])
            self.assertTrue(all(instance.kwargs["tensor_parallel_size"] == 1 for instance in _LLM.instances))
            self.assertEqual(
                [len(instance.generate_calls[0][0]) for instance in _LLM.instances],
                [5, 3, 2],
            )


if __name__ == "__main__":
    unittest.main()
