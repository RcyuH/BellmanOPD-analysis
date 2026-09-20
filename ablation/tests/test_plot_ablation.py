from __future__ import annotations

import json
import sys
from pathlib import Path

from ablation.plots.plot_ablation import (
    BENCHMARK_ORDER,
    _align_arm_bases,
    _resolve_benchmarks,
    main,
)


def _write_run(root: Path, arm: str) -> None:
    output = root / f"run_{arm}"
    output.mkdir(parents=True)
    (output / "ablation_spec.json").write_text(
        json.dumps({"arm": arm}) + "\n", encoding="utf-8"
    )
    rows = []
    for step in (0, 10):
        rows.append(
            {
                "step": step,
                "benchmarks": {
                    benchmark: {"accuracy": 0.50 + 0.01 * step / 10}
                    for benchmark in BENCHMARK_ORDER
                },
            }
        )
    (output / "eval_history.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_default_arm_plot_uses_all_six_benchmarks(tmp_path, monkeypatch):
    for arm in ("g", "g_x", "g_d"):
        _write_run(tmp_path, arm)
    output = tmp_path / "figures"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_ablation.py",
            "--input-root",
            str(tmp_path),
            "--output-dir",
            str(output),
        ],
    )
    assert main() == 0
    assert (output / "ablation_curves.png").is_file()
    assert (output / "ablation_final.png").is_file()
    assert _resolve_benchmarks(None, None) == BENCHMARK_ORDER


def test_arm_plot_accepts_benchmark_subset(tmp_path, monkeypatch):
    _write_run(tmp_path, "g")
    output = tmp_path / "figures"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_ablation.py",
            "--input-root",
            str(tmp_path),
            "--output-dir",
            str(output),
            "--benchmarks",
            "MATH-500,GPQA-Diamond",
        ],
    )
    assert main() == 0
    assert (output / "ablation_curves.png").is_file()


def test_arm_plot_can_reuse_production_cmt_as_g_d(tmp_path, monkeypatch):
    _write_run(tmp_path, "g")
    _write_run(tmp_path, "g_x")
    cmt_output = tmp_path / "outputs" / "cmt_existing" / "cmt_opd"
    cmt_output.mkdir(parents=True)
    rows = [
        {
            "step": step,
            "benchmarks": {
                benchmark: {"accuracy": 0.60 + 0.01 * step / 10}
                for benchmark in BENCHMARK_ORDER
            },
        }
        for step in (0, 10)
    ]
    (cmt_output / "eval_history.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    output = tmp_path / "figures"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_ablation.py",
            "--input-root",
            str(tmp_path),
            "--output-dir",
            str(output),
            "--run-name",
            "run_g",
            "--run-name",
            "run_g_x",
            "--g-d-output",
            str(cmt_output),
            "--g-d-run-name",
            "cmt_existing",
        ],
    )
    assert main() == 0
    assert (output / "ablation_curves.png").is_file()
    assert (output / "ablation_final.png").is_file()


def test_base_alignment_shifts_only_gd_trajectory_when_gd_is_low():
    grouped = {
        "MATH-500": {
            "g": [(0, 0.50), (10, 0.60)],
            "g_x": [(0, 0.40), (10, 0.70)],
            "g_d": [(0, 0.30), (10, 0.35)],
        }
    }
    aligned = _align_arm_bases(grouped)["MATH-500"]

    # g is the highest original base, so all three displayed bases coincide
    # at 0.50.  The complete g_d curve receives the +0.20 shift, whereas g_x
    # receives a base-only replacement and keeps its later point unchanged.
    assert aligned["g"][0] == (0, 0.50)
    assert aligned["g_x"][0] == (0, 0.50)
    assert aligned["g_x"][1] == (10, 0.70)
    assert aligned["g_d"] == [(0, 0.50), (10, 0.55)]


def test_base_alignment_only_changes_g_and_gx_bases_when_gd_is_high():
    grouped = {
        "MATH-500": {
            "g": [(0, 0.40), (10, 0.60)],
            "g_x": [(0, 0.45), (10, 0.70)],
            "g_d": [(0, 0.60), (10, 0.65)],
        }
    }
    aligned = _align_arm_bases(grouped)["MATH-500"]

    assert aligned["g"][0] == (0, 0.60)
    assert aligned["g"][1] == (10, 0.60)
    assert aligned["g_x"][0] == (0, 0.60)
    assert aligned["g_x"][1] == (10, 0.70)
    assert aligned["g_d"] == grouped["MATH-500"]["g_d"]
