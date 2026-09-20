from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from b200_experiment.evaluation_history import record_checkpoint_evaluation


def _suite(accuracy: float, *, metric: str = "avg@8") -> dict:
    return {
        "parameters": {
            "backend": "vllm",
            "num_responses": 8,
            "metric": metric,
            "temperature": 0.7,
            "top_p": 0.95,
        },
        "benchmarks": {
            "MATH-500": {
                "correct": 4,
                "total": 8,
                "problems": 1,
                "accuracy": accuracy,
                "avg_at_n": accuracy,
                "samples_per_problem": 8,
                "pass_at_k": accuracy,
                "pass_at_8": accuracy,
            }
        },
    }


class EvaluationHistoryTests(unittest.TestCase):
    def test_standalone_evaluation_appends_and_replaces_by_step_and_method(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "eval_history.jsonl").write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        {"step": 1, "method": "opd", "old": True},
                        {"step": 2, "method": "opd", "old": True},
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            with (output / "eval_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=("step", "method", "benchmark", "accuracy"))
                writer.writeheader()
                writer.writerow({"step": 2, "method": "opd", "benchmark": "MATH-500", "accuracy": 0.5})

            record_checkpoint_evaluation(
                output,
                _suite(0.75),
                method="opd",
                step=1,
                max_steps=10,
                details_path=output / "checkpoint_eval" / "step-000001" / "summary.json",
            )
            record_checkpoint_evaluation(
                output,
                _suite(0.875),
                method="opd",
                step=3,
                max_steps=10,
            )

            history = [
                json.loads(line)
                for line in (output / "eval_history.jsonl").read_text().splitlines()
            ]
            self.assertEqual([(row["step"], row["method"]) for row in history], [(1, "opd"), (2, "opd"), (3, "opd")])
            self.assertAlmostEqual(history[0]["benchmarks"]["MATH-500"]["accuracy"], 0.75)
            self.assertNotIn("old", history[0])
            self.assertAlmostEqual(history[-1]["benchmarks"]["MATH-500"]["accuracy"], 0.875)

            with (output / "eval_metrics.csv").open(newline="", encoding="utf-8") as handle:
                metrics = list(csv.DictReader(handle))
            self.assertEqual(
                {(row["step"], row["method"]) for row in metrics},
                {("1", "opd"), ("2", "opd"), ("3", "opd")},
            )
            self.assertIn("avg@8", {row["metric"] for row in metrics})

    def test_partial_benchmark_eval_preserves_existing_benchmarks(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            existing = {
                "step": 4,
                "method": "cmt",
                "benchmarks": {
                    "MATH-500": {"accuracy": 0.51, "metric": "avg@8"},
                    "AIME24": {"accuracy": 0.22, "metric": "avg@8"},
                },
                "parameters": {"metric": "avg@8"},
            }
            (output / "eval_history.jsonl").write_text(
                json.dumps(existing) + "\n", encoding="utf-8"
            )
            suite = {
                "parameters": {"backend": "vllm", "num_responses": 8, "metric": "avg@8"},
                "benchmarks": {
                    "GPQA-Diamond": {
                        "correct": 3,
                        "total": 4,
                        "problems": 4,
                        "accuracy": 0.75,
                        "avg_at_n": 0.75,
                    },
                    "AMC23": {
                        "correct": 2,
                        "total": 4,
                        "problems": 4,
                        "accuracy": 0.5,
                        "avg_at_n": 0.5,
                    },
                },
            }
            record_checkpoint_evaluation(output, suite, method="cmt", step=4)
            row = json.loads((output / "eval_history.jsonl").read_text())
            self.assertEqual(
                set(row["benchmarks"]),
                {"MATH-500", "AIME24", "GPQA-Diamond", "AMC23"},
            )
            self.assertEqual(row["benchmarks"]["MATH-500"]["accuracy"], 0.51)


if __name__ == "__main__":
    unittest.main()
