"""Summarize paired per-batch KL improvements for Experiment 2."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_steps(root: Path) -> list[dict]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / "steps").glob("step-*.json"))
    ]


def _bootstrap_mean_interval(
    values: np.ndarray, *, seed: int = 42, samples: int = 10000
) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        value = float(values[0])
        return value, value
    generator = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    # Chunk bootstrap indices to avoid an unnecessarily large n_bootstrap x n
    # matrix for long full-dataset runs.
    chunk = 256
    written = 0
    while written < samples:
        count = min(chunk, samples - written)
        indices = generator.integers(0, values.size, size=(count, values.size))
        means[written : written + count] = values[indices].mean(axis=1)
        written += count
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def summarize(root: Path, output: Path) -> dict:
    rows = load_steps(root)
    if not rows:
        raise FileNotFoundError(f"No steps/step-*.json found under {root}")
    output.mkdir(parents=True, exist_ok=True)
    table = []
    for row in rows:
        top = row["top_gt"]
        uniform = row["uniform"]
        table.append(
            {
                "optimizer_step_before": row["optimizer_step_before"],
                "valid_tokens": row["valid_tokens"],
                "top_gt_selected_tokens": row["top_gt_selected_tokens"],
                "top_gt_selected_fraction": row["top_gt_selected_fraction"],
                "top_gt_distance_before": top["distance_before"],
                "top_gt_distance_after": top["distance_after"],
                "top_gt_improvement": top["distance_improvement"],
                "uniform_distance_before": uniform["distance_before"],
                "uniform_distance_after": uniform["distance_after"],
                "uniform_improvement": uniform["distance_improvement"],
                "top_gt_minus_uniform": row[
                    "improvement_advantage_top_gt_minus_uniform"
                ],
                "top_gt_wins": row["top_gt_wins_this_batch"],
                "student_weight_l2_distance_after_batch": row[
                    "student_weight_l2_distance_after_batch"
                ],
            }
        )
    with (output / "batch_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    top_improvement = np.asarray(
        [row["top_gt_improvement"] for row in table], dtype=np.float64
    )
    uniform_improvement = np.asarray(
        [row["uniform_improvement"] for row in table], dtype=np.float64
    )
    advantage = top_improvement - uniform_improvement
    ci_low, ci_high = _bootstrap_mean_interval(advantage)
    if ci_low > 0:
        conclusion = "supports_top_gt"
    elif ci_high < 0:
        conclusion = "supports_uniform"
    else:
        conclusion = "inconclusive"
    result = {
        "schema_version": 1,
        "input": str(root.resolve()),
        "num_completed_batches": len(table),
        "mean_top_gt_improvement": float(top_improvement.mean()),
        "mean_uniform_improvement": float(uniform_improvement.mean()),
        "mean_improvement_advantage_top_gt_minus_uniform": float(advantage.mean()),
        "median_improvement_advantage_top_gt_minus_uniform": float(
            np.median(advantage)
        ),
        "top_gt_batch_win_fraction": float((advantage > 0).mean()),
        "paired_bootstrap_95_percent_ci_for_mean_advantage": [ci_low, ci_high],
        "trajectory_total_kl_reduction": {
            "top_gt": float(
                table[0]["top_gt_distance_before"]
                - table[-1]["top_gt_distance_after"]
            ),
            "uniform": float(
                table[0]["uniform_distance_before"]
                - table[-1]["uniform_distance_after"]
            ),
        },
        "final_test_kl": {
            "top_gt": float(table[-1]["top_gt_distance_after"]),
            "uniform": float(table[-1]["uniform_distance_after"]),
        },
        "conclusion": conclusion,
        "conclusion_rule": (
            "supports_top_gt iff paired bootstrap CI of mean per-batch KL-improvement "
            "advantage is entirely above zero; supports_uniform iff entirely below zero"
        ),
        "caveat": (
            "After the first update the two trajectories have different theta and AdamW "
            "state; each per-batch improvement is relative to its own pre-batch state."
        ),
    }
    (output / "comparison_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.input, args.output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
