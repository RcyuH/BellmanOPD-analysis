from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from b200_experiment.plotting import (
    _accuracy_ylim,
    plot_cmt_score_distributions,
    plot_results,
    plot_training_progress,
)


class PlottingTests(unittest.TestCase):
    @staticmethod
    def _write_cmt_token_stats(output: Path) -> None:
        """Write compact synthetic CMT score snapshots for plotting tests."""

        stats = output / "token_score_stats"
        stats.mkdir(parents=True)
        required = {
            "gain": (0.0, 2.0),
            "successor_excess": (-1.0, 1.0),
            "sequential_gain": (-1.0, 1.0),
            "learning_value": (-1.0, 2.0),
            "w": (0.0, 2.0),
        }
        for index, step in enumerate((50, 100, 150), start=1):
            scores = {}
            for field, (low, high) in required.items():
                center = low + (high - low) * (0.25 + 0.1 * index)
                scores[field] = {
                    "count": 4,
                    "mean": center,
                    "min": low,
                    "max": high,
                    "quantiles": {
                        "q05": low,
                        "q25": center - 0.1,
                        "q50": center,
                        "q75": center + 0.1,
                        "q95": high,
                    },
                    "histogram": {
                        "edges": [low, (low + high) / 2.0, high],
                        "counts": [2, 2],
                        "underflow": 0,
                        "overflow": 0,
                    },
                    "sample": [low, center, center, high],
                }
            (stats / f"step-{step:06d}.json").write_text(
                json.dumps(
                    {
                        "step": step,
                        "method": "cmt",
                        "scope": "global_valid_response_tokens",
                        "scores": scores,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

    def test_cmt_score_plots_include_histograms_quantiles_and_unique_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "cmt_opd"
            self._write_cmt_token_stats(output)

            first = plot_cmt_score_distributions(output, run_name="cmt_demo")
            second = plot_cmt_score_distributions(
                output, run_name="cmt_demo", plot_name="cmt_demo_scores"
            )
            third = plot_cmt_score_distributions(
                output, run_name="cmt_demo", plot_name="cmt_demo_scores"
            )

            for result in (first, second, third):
                self.assertTrue(Path(result["histograms"]).is_file())
                self.assertTrue(Path(result["histogram_heatmaps"]).is_file())
                self.assertTrue(Path(result["learning_value_quantiles"]).is_file())
                self.assertTrue(Path(result["score_means"]).is_file())
                self.assertTrue(Path(result["manifest"]).is_file())
                self.assertEqual(result["logged_steps"], [50, 100, 150])
            self.assertNotEqual(second["plot_directory"], third["plot_directory"])

    def test_single_response_history_is_labeled_accuracy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "opd"
            self._write_training_output(output, 0.25)
            history_path = output / "eval_history.jsonl"
            rows = [json.loads(line) for line in history_path.read_text().splitlines()]
            for row in rows:
                row["parameters"] = {"metric": "accuracy", "num_responses": 1}
                for result in row["benchmarks"].values():
                    result["samples_per_problem"] = 1
            history_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )

            paths = plot_training_progress(
                root / "results", opd_output=output, methods=["opd"]
            )

            self.assertEqual(paths["metric"], "accuracy")
            self.assertTrue(Path(paths["accuracy_over_steps"]).is_file())

    @staticmethod
    def _write_training_output(
        output: Path,
        accuracy: float,
        *,
        base_accuracy: float = 0.1,
        benchmark_names: tuple[str, ...] = ("MATH-500", "AIME24", "AIME25"),
    ) -> None:
        output.mkdir()
        with (output / "metrics.jsonl").open("w", encoding="utf-8") as handle:
            for step in range(1, 4):
                handle.write(json.dumps({"step": step, "loss": 1.0 / step}) + "\n")
        with (output / "eval_history.jsonl").open("w", encoding="utf-8") as handle:
            for step in (0, 100):
                value = base_accuracy if step == 0 else accuracy
                handle.write(
                    json.dumps(
                        {
                            "step": step,
                            "benchmarks": {
                                name: {"accuracy": value}
                                for name in benchmark_names
                            },
                        }
                    )
                    + "\n"
                )

    def test_step_zero_accuracy_tolerance_uses_shared_mean_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ta_output = root / "ta"
            rac_output = root / "rac"
            self._write_training_output(ta_output, 0.2, base_accuracy=0.100)
            self._write_training_output(rac_output, 0.3, base_accuracy=0.105)

            paths = plot_training_progress(
                root / "results",
                ta_output=ta_output,
                rac_output=rac_output,
                methods=["ta", "rac"],
            )

            history = json.loads(Path(paths["history_json"]).read_text())
            self.assertAlmostEqual(history["base_accuracy"]["MATH-500"], 0.1025)

    def test_step_zero_difference_over_two_percentage_points_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ta_output = root / "ta"
            rac_output = root / "rac"
            self._write_training_output(ta_output, 0.2, base_accuracy=0.10)
            self._write_training_output(rac_output, 0.3, base_accuracy=0.121)

            with self.assertRaisesRegex(ValueError, "exceeding the 2.0% tolerance"):
                plot_training_progress(
                    root / "results",
                    ta_output=ta_output,
                    rac_output=rac_output,
                    methods=["ta", "rac"],
                )

    def test_step_zero_difference_within_two_percentage_points_is_allowed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ta_output = root / "ta"
            rac_output = root / "rac"
            self._write_training_output(ta_output, 0.2, base_accuracy=0.100)
            self._write_training_output(rac_output, 0.3, base_accuracy=0.119)

            paths = plot_training_progress(
                root / "results",
                ta_output=ta_output,
                rac_output=rac_output,
                methods=["ta", "rac"],
            )
            self.assertTrue(Path(paths["accuracy_over_steps"]).is_file())

    def test_opd_cmt_alignment_shifts_cmt_curve_when_opd_base_is_higher(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            opd_output = root / "opd"
            cmt_output = root / "cmt"
            self._write_training_output(opd_output, 0.20, base_accuracy=0.12)
            self._write_training_output(cmt_output, 0.30, base_accuracy=0.10)

            paths = plot_training_progress(
                root / "results",
                opd_output=opd_output,
                cmt_output=cmt_output,
                methods=["opd", "cmt"],
            )
            payload = json.loads(Path(paths["history_json"]).read_text())
            cmt_rows = payload["histories"]["CMT-OPD"]
            self.assertAlmostEqual(
                cmt_rows[0]["benchmarks"]["MATH-500"]["accuracy"], 0.12
            )
            self.assertAlmostEqual(
                cmt_rows[1]["benchmarks"]["MATH-500"]["accuracy"], 0.32
            )
            self.assertAlmostEqual(
                cmt_rows[0]["benchmarks"]["AIME24"]["accuracy"], 0.12
            )
            self.assertAlmostEqual(payload["base_accuracy"]["MATH-500"], 0.12)

    def test_opd_cmt_alignment_applies_to_all_six_benchmarks(self):
        benchmarks = (
            "Competition-MATH",
            "MATH-500",
            "AIME24",
            "AIME25",
            "GPQA-Diamond",
            "AMC23",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            opd_output = root / "opd"
            cmt_output = root / "cmt"
            self._write_training_output(
                opd_output,
                0.20,
                base_accuracy=0.12,
                benchmark_names=benchmarks,
            )
            self._write_training_output(
                cmt_output,
                0.30,
                base_accuracy=0.10,
                benchmark_names=benchmarks,
            )

            paths = plot_training_progress(
                root / "results",
                opd_output=opd_output,
                cmt_output=cmt_output,
                methods=["opd", "cmt"],
            )
            payload = json.loads(Path(paths["history_json"]).read_text())
            cmt_rows = payload["histories"]["CMT-OPD"]
            for benchmark in benchmarks:
                self.assertAlmostEqual(
                    cmt_rows[0]["benchmarks"][benchmark]["accuracy"], 0.12
                )
                self.assertAlmostEqual(
                    cmt_rows[1]["benchmarks"][benchmark]["accuracy"], 0.32
                )

    def test_opd_cmt_inverse_alignment_only_lifts_opd_base_on_all_six_benchmarks(self):
        benchmarks = (
            "Competition-MATH",
            "MATH-500",
            "AIME24",
            "AIME25",
            "GPQA-Diamond",
            "AMC23",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            opd_output = root / "opd"
            cmt_output = root / "cmt"
            self._write_training_output(
                opd_output,
                0.20,
                base_accuracy=0.10,
                benchmark_names=benchmarks,
            )
            self._write_training_output(
                cmt_output,
                0.30,
                base_accuracy=0.12,
                benchmark_names=benchmarks,
            )

            paths = plot_training_progress(
                root / "results",
                opd_output=opd_output,
                cmt_output=cmt_output,
                methods=["opd", "cmt"],
            )
            payload = json.loads(Path(paths["history_json"]).read_text())
            opd_rows = payload["histories"]["OPD"]
            for benchmark in benchmarks:
                self.assertAlmostEqual(
                    opd_rows[0]["benchmarks"][benchmark]["accuracy"], 0.12
                )
                self.assertAlmostEqual(
                    opd_rows[1]["benchmarks"][benchmark]["accuracy"], 0.20
                )

    def test_opd_cmt_alignment_only_lifts_opd_base_when_cmt_is_higher(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            opd_output = root / "opd"
            cmt_output = root / "cmt"
            self._write_training_output(opd_output, 0.20, base_accuracy=0.10)
            self._write_training_output(cmt_output, 0.30, base_accuracy=0.12)

            paths = plot_training_progress(
                root / "results",
                opd_output=opd_output,
                cmt_output=cmt_output,
                methods=["opd", "cmt"],
            )
            payload = json.loads(Path(paths["history_json"]).read_text())
            opd_rows = payload["histories"]["OPD"]
            self.assertAlmostEqual(
                opd_rows[0]["benchmarks"]["MATH-500"]["accuracy"], 0.12
            )
            self.assertAlmostEqual(
                opd_rows[1]["benchmarks"]["MATH-500"]["accuracy"], 0.20
            )
            self.assertAlmostEqual(payload["base_accuracy"]["MATH-500"], 0.12)

    def test_partial_step_skips_missing_points_but_keeps_other_benchmarks(self):
        benchmarks = (
            "Competition-MATH",
            "MATH-500",
            "AIME24",
            "AIME25",
            "GPQA-Diamond",
            "AMC23",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outputs = {}
            for method in ("opd", "ta", "cmt"):
                output = root / method
                self._write_training_output(
                    output,
                    0.25,
                    benchmark_names=benchmarks,
                )
                outputs[method] = output

            # Simulate a partial re-evaluation at one OPD checkpoint: the
            # GPQA/AMC rows exist, while the four legacy datasets are absent.
            history_path = outputs["opd"] / "eval_history.jsonl"
            rows = [json.loads(line) for line in history_path.read_text().splitlines()]
            rows[1]["benchmarks"] = {
                name: rows[1]["benchmarks"][name]
                for name in ("GPQA-Diamond", "AMC23")
            }
            history_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            paths = plot_training_progress(
                root / "results",
                opd_output=outputs["opd"],
                ta_output=outputs["ta"],
                cmt_output=outputs["cmt"],
                methods=["opd", "ta", "cmt"],
            )
            self.assertTrue(Path(paths["accuracy_over_steps"]).is_file())
            payload = json.loads(Path(paths["history_json"]).read_text())
            opd_step = next(
                row for row in payload["histories"]["OPD"] if row["step"] == 100
            )
            self.assertEqual(
                set(opd_step["benchmarks"]), {"GPQA-Diamond", "AMC23"}
            )
            with Path(paths["history_csv"]).open(
                newline="", encoding="utf-8"
            ) as handle:
                csv_rows = list(csv.DictReader(handle))
            opd_csv_step = next(
                row
                for row in csv_rows
                if row["Method"] == "OPD" and row["Step"] == "100"
            )
            self.assertEqual(opd_csv_step["MATH-500"], "")
            self.assertNotEqual(opd_csv_step["GPQA-Diamond"], "")

    def test_step_zero_aime_difference_is_not_a_validation_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ta_output = root / "ta"
            rac_output = root / "rac"
            self._write_training_output(ta_output, 0.2, base_accuracy=0.10)
            self._write_training_output(rac_output, 0.3, base_accuracy=0.10)

            history_path = rac_output / "eval_history.jsonl"
            rows = [json.loads(line) for line in history_path.read_text().splitlines()]
            rows[0]["benchmarks"]["AIME24"]["accuracy"] = 0.50
            history_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )

            paths = plot_training_progress(
                root / "results",
                ta_output=ta_output,
                rac_output=rac_output,
                methods=["ta", "rac"],
            )

            history = json.loads(Path(paths["history_json"]).read_text())
            self.assertAlmostEqual(history["base_accuracy"]["AIME24"], 0.30)

    def test_accuracy_limits_zoom_to_observed_range(self):
        self.assertEqual(_accuracy_ylim([0.58, 0.65]), (0.55, 0.7))
        self.assertEqual(_accuracy_ylim([0.0, 1.0]), (0.0, 1.05))

    def test_accuracy_and_loss_plots_are_created(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results"
            results.mkdir()
            with (results / "comparison.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=("Method", "MATH-500", "AIME24", "AIME25")
                )
                writer.writeheader()
                for index, method in enumerate(("Base", "TA-OPD", "RAC")):
                    writer.writerow(
                        {
                            "Method": method,
                            "MATH-500": 0.1 + index / 10,
                            "AIME24": 0.2,
                            "AIME25": 0.3,
                        }
                    )
            outputs = []
            for method, accuracy in (("ta", 0.2), ("rac", 0.3)):
                output = root / method
                self._write_training_output(output, accuracy)
                outputs.append(output)
            paths = plot_results(results, outputs[0], outputs[1], smoothing_window=2)
            self.assertTrue(Path(paths["accuracy"]).is_file())
            self.assertTrue(Path(paths["loss"]).is_file())
            self.assertTrue(Path(paths["accuracy_over_steps"]).is_file())
            self.assertTrue(Path(paths["history_csv"]).is_file())

    def test_final_plot_accepts_base_and_all_three_trained_methods(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results"
            results.mkdir()
            with (results / "comparison.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=("Method", "MATH-500", "AIME24", "AIME25")
                )
                writer.writeheader()
                for index, method in enumerate(("Base", "OPD", "TA-OPD", "RAC")):
                    writer.writerow(
                        {
                            "Method": method,
                            "MATH-500": 0.1 + index / 10,
                            "AIME24": 0.2 + index / 10,
                            "AIME25": 0.3 + index / 10,
                        }
                    )
            outputs = {}
            for method, accuracy in (("opd", 0.2), ("ta", 0.3), ("rac", 0.4)):
                outputs[method] = root / method
                self._write_training_output(outputs[method], accuracy)

            paths = plot_results(
                results,
                outputs["ta"],
                outputs["rac"],
                smoothing_window=2,
                opd_output=outputs["opd"],
            )

            self.assertEqual(paths["methods"], ["OPD", "TA-OPD", "Bellman-RAC"])
            self.assertTrue(Path(paths["accuracy"]).is_file())
            self.assertTrue(Path(paths["accuracy_over_steps"]).is_file())

    def test_single_method_plot_contains_all_benchmarks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results"
            ta_output = root / "ta"
            self._write_training_output(ta_output, 0.25)

            paths = plot_training_progress(
                results,
                ta_output=ta_output,
                method="ta",
                plot_name="ta_report",
            )

            self.assertEqual(paths["method"], "TA-OPD")
            self.assertEqual(
                Path(paths["accuracy_over_steps"]).name,
                "ta_opd_accuracy_over_steps.png",
            )
            self.assertTrue(Path(paths["accuracy_over_steps"]).is_file())
            self.assertTrue(
                Path(paths["accuracy_over_steps"]).with_suffix(".pdf").is_file()
            )
            with Path(paths["history_csv"]).open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(
                tuple(rows[0]),
                ("Method", "Step", "MATH-500", "AIME24", "AIME25"),
            )
            self.assertEqual({row["Method"] for row in rows}, {"Base", "TA-OPD"})

    def test_new_history_adds_competition_math_to_plot_and_csv(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "opd"
            self._write_training_output(output, 0.25)
            history_path = output / "eval_history.jsonl"
            rows = [json.loads(line) for line in history_path.read_text().splitlines()]
            for row in rows:
                row["benchmarks"] = {
                    "Competition-MATH": {"accuracy": 0.4},
                    **row["benchmarks"],
                }
            history_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )

            paths = plot_training_progress(
                root / "results", opd_output=output, methods=["opd"]
            )

            with Path(paths["history_csv"]).open(
                newline="", encoding="utf-8"
            ) as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual(
                tuple(csv_rows[0]),
                ("Method", "Step", "Competition-MATH", "MATH-500", "AIME24", "AIME25"),
            )
            self.assertTrue(Path(paths["accuracy_over_steps"]).is_file())

    def test_single_rac_plot_does_not_require_ta_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rac_output = root / "rac"
            self._write_training_output(rac_output, 0.35)

            paths = plot_training_progress(
                root / "results", rac_output=rac_output, method="bellman-rac"
            )

            self.assertEqual(paths["method"], "Bellman-RAC")
            self.assertEqual(
                Path(paths["accuracy_over_steps"]).name,
                "rac_accuracy_over_steps.png",
            )

    def test_single_opd_plot_does_not_require_ta_or_rac_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            opd_output = root / "opd"
            self._write_training_output(opd_output, 0.20)

            paths = plot_training_progress(
                root / "results", opd_output=opd_output, methods=["opd"]
            )

            self.assertEqual(paths["method"], "OPD")
            self.assertEqual(
                Path(paths["accuracy_over_steps"]).name,
                "opd_accuracy_over_steps.png",
            )

    def test_any_two_or_all_three_methods_can_be_compared(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outputs = {}
            for method, accuracy in (("opd", 0.2), ("ta", 0.3), ("rac", 0.4)):
                outputs[method] = root / method
                self._write_training_output(outputs[method], accuracy)

            pair = plot_training_progress(
                root / "pair-results",
                opd_output=outputs["opd"],
                rac_output=outputs["rac"],
                methods=["opd", "rac"],
                plot_name="opd_vs_rac",
            )
            all_methods = plot_training_progress(
                root / "all-results",
                opd_output=outputs["opd"],
                ta_output=outputs["ta"],
                rac_output=outputs["rac"],
                methods=["opd", "ta", "rac"],
                plot_name="all_three",
            )

            self.assertEqual(pair["methods"], ["OPD", "Bellman-RAC"])
            self.assertEqual(all_methods["methods"], ["OPD", "TA-OPD", "Bellman-RAC"])
            self.assertTrue(Path(pair["accuracy_over_steps"]).is_file())
            self.assertTrue(Path(all_methods["accuracy_over_steps"]).is_file())
            with Path(pair["history_csv"]).open(newline="", encoding="utf-8") as handle:
                pair_rows = list(csv.DictReader(handle))
            self.assertEqual(
                {row["Method"] for row in pair_rows},
                {"Base", "OPD", "Bellman-RAC"},
            )

    def test_selected_method_requires_its_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "--rac-output is required"):
                plot_training_progress(Path(temporary), method="rac")

    def test_all_methods_require_opd_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "--opd-output is required"):
                plot_training_progress(Path(temporary), method="all")


if __name__ == "__main__":
    unittest.main()
