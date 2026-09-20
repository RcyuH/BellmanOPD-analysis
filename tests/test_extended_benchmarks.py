from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from b200_experiment.evaluation import (
    configured_benchmark_names,
    ensure_extended_benchmark_specs,
    grade_evaluation_response,
    load_benchmark,
)


class ExtendedBenchmarkTests(unittest.TestCase):
    def test_gpqa_prompt_ground_truth_export_and_old_config_are_migrated(self):
        """The downloaded GPQA file has id/prompt/ground_truth columns."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gpqa_diamond.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "id": 7,
                        "prompt": (
                            "Which option is correct?\n\n"
                            "A. first answer\n"
                            "B. second answer\n"
                            "C. third answer\n"
                            "D. fourth answer"
                        ),
                        "ground_truth": "D",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            # Simulate a checkpoint's historical resolved config.  Missing
            # Question/Correct Answer/Incorrect Answer columns should fall
            # back to the prompt/ground_truth schema automatically.
            rows, schema = load_benchmark(
                "GPQA-Diamond",
                {
                    "task_type": "gpqa",
                    "path": str(path),
                    "question_key": "Question",
                    "answer_key": "Correct Answer",
                    "choice_keys": [
                        "Correct Answer",
                        "Incorrect Answer 1",
                        "Incorrect Answer 2",
                        "Incorrect Answer 3",
                    ],
                },
            )
            self.assertEqual(schema["question_key"], "prompt")
            self.assertEqual(schema["answer_key"], "ground_truth")
            self.assertEqual(rows[0]["id"], "7")
            self.assertEqual(rows[0]["metadata"]["choices"], [
                "first answer",
                "second answer",
                "third answer",
                "fourth answer",
            ])
            self.assertEqual(rows[0]["metadata"]["correct_choice"], 3)
            self.assertTrue(rows[0]["metadata"]["choices_embedded"])

    def test_gpqa_jsonl_normalizes_choices_and_grades_letters(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gpqa_diamond.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "Question": "Which is correct?",
                        "Correct Answer": "Correct",
                        "Incorrect Answer 1": "Wrong one",
                        "Incorrect Answer 2": "Wrong two",
                        "Incorrect Answer 3": "Wrong three",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            rows, schema = load_benchmark(
                "GPQA-Diamond",
                {
                    "task_type": "gpqa",
                    "path": str(path),
                    "question_key": "Question",
                    "answer_key": "Correct Answer",
                    "choice_keys": [
                        "Correct Answer",
                        "Incorrect Answer 1",
                        "Incorrect Answer 2",
                        "Incorrect Answer 3",
                    ],
                    "choice_shuffle_seed": 1234,
                },
            )
            self.assertEqual(schema["task_type"], "gpqa")
            self.assertEqual(len(rows[0]["metadata"]["choices"]), 4)
            correct = rows[0]["metadata"]["correct_choice"]
            self.assertTrue(
                grade_evaluation_response(
                    "The answer is " + "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[correct],
                    rows[0],
                    "GPQA-Diamond",
                )
            )

    def test_amc23_parquet_uses_standard_math_schema(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "test-00000-of-00001.parquet"
            pd.DataFrame(
                [{"id": 7, "question": "What is 2+2?", "answer": "4"}]
            ).to_parquet(path)
            rows, schema = load_benchmark(
                "AMC23",
                {
                    "task_type": "math",
                    "path": str(path),
                    "question_key": "question",
                    "answer_key": "answer",
                    "id_key": "id",
                },
            )
            self.assertEqual(schema["task_type"], "math")
            self.assertEqual(
                rows[0], {"id": "7", "problem": "What is 2+2?", "answer": "4"}
            )

    def test_configured_order_includes_new_benchmarks_and_allows_subset(self):
        config = {
            "evaluation": {
                "benchmarks": {
                    name: {"path": "/tmp/unused"}
                    for name in (
                        "Competition-MATH",
                        "MATH-500",
                        "AIME24",
                        "AIME25",
                        "GPQA-Diamond",
                        "AMC23",
                    )
                }
            }
        }
        self.assertEqual(
            configured_benchmark_names(config, ["GPQA-Diamond", "AMC23"]),
            ("GPQA-Diamond", "AMC23"),
        )

    def test_old_ifeval_config_is_migrated_to_amc23_in_memory(self):
        config = {
            "paths": {"storage_root": "/workspace/storage-shared"},
            "evaluation": {
                "benchmarks": {},
            },
            "training_evaluation": {
                "benchmark_names": ["MATH-500", "IFEval"],
            },
        }
        ensure_extended_benchmark_specs(config)
        self.assertEqual(
            config["training_evaluation"]["benchmark_names"],
            ["MATH-500", "AMC23"],
        )
        self.assertEqual(
            config["evaluation"]["benchmarks"]["AMC23"]["path"],
            "/workspace/storage-shared/nlp/minhpn19/data/amc23/test-00000-of-00001.parquet",
        )


if __name__ == "__main__":
    unittest.main()
