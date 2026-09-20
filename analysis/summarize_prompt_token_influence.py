"""Rank prompt-token influence, keeping predicted and measured gains distinct."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_rows(root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted((root / "steps").glob("step-*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def _score(row: dict) -> float:
    key = (
        "predicted_distance_improvement"
        if row.get("predicted_distance_improvement") is not None
        else "distance_improvement"
    )
    value = row.get(key)
    if value is None:
        raise ValueError(f"Row has neither predicted nor measured improvement: {row}")
    return float(value)


def summarize(root: Path, output: Path, top_k: int = 1000) -> dict:
    rows = load_rows(root)
    tokens = [
        row
        for row in rows
        if str(row.get("intervention", "")).startswith("single_token")
    ]
    uniform = [
        row
        for row in rows
        if str(row.get("intervention", "")).startswith(
            "uniform_mean_all_prompt_tokens"
        )
    ]
    modes = {
        bool(row.get("is_exact_intervention", True)) for row in tokens
    }
    if len(modes) > 1:
        raise ValueError("Cannot rank exact and first-order runs together")
    tokens.sort(
        key=lambda row: (
            -_score(row),
            int(row["optimizer_step_before"]),
            int(row["prompt_position"]),
        )
    )
    output.mkdir(parents=True, exist_ok=True)
    columns = (
        "global_rank",
        "optimizer_step_before",
        "epoch",
        "dataset_index",
        "sample_id",
        "worker_rank",
        "distributed_world_size",
        "prompt_position",
        "token_id",
        "token_piece",
        "decoded_text",
        "visible_text",
        "is_special_token",
        "opd_loss",
        "gradient_l2_norm",
        "distance_before",
        "distance_after",
        "distance_improvement",
        "relative_distance_improvement",
        "predicted_distance_improvement",
        "influence_method",
        "is_exact_intervention",
        "distance_gradient_refresh_step",
        "calibration_distance_at_refresh",
        "ranking_score",
    )
    selected = tokens[: max(0, int(top_k))]
    with (output / "top_tokens.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for rank, row in enumerate(selected, 1):
            writer.writerow({"global_rank": rank, **row, "ranking_score": _score(row)})
    by_step = []
    step_ids = sorted({int(row["optimizer_step_before"]) for row in rows})
    for step in step_ids:
        step_tokens = [row for row in tokens if int(row["optimizer_step_before"]) == step]
        step_uniform = [row for row in uniform if int(row["optimizer_step_before"]) == step]
        if not step_tokens or len(step_uniform) != 1:
            continue
        best = max(step_tokens, key=_score)
        by_step.append(
            {
                "optimizer_step_before": step,
                "num_tokens": len(step_tokens),
                "best_prompt_position": best["prompt_position"],
                "best_token_id": best["token_id"],
                "best_token_text": best["decoded_text"],
                "best_token_improvement": _score(best),
                "mean_token_improvement": sum(_score(row) for row in step_tokens) / len(step_tokens),
                "positive_token_fraction": sum(_score(row) > 0 for row in step_tokens) / len(step_tokens),
                "uniform_improvement": _score(step_uniform[0]),
            }
        )
    if by_step:
        with (output / "step_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(by_step[0]))
            writer.writeheader()
            writer.writerows(by_step)
    summary = {
        "input": str(root.resolve()),
        "token_interventions": len(tokens),
        "uniform_interventions": len(uniform),
        "completed_steps_found": len(by_step),
        "ranking_kind": (
            "measured_exact_intervention"
            if modes == {True}
            else "predicted_first_order_output_head"
        ),
        "positive_token_interventions": sum(_score(row) > 0 for row in tokens),
        "top_tokens_csv": str((output / "top_tokens.csv").resolve()),
        "step_summary_csv": str((output / "step_summary.csv").resolve()),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=1000)
    args = parser.parse_args()
    print(json.dumps(summarize(args.input, args.output, args.top_k), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
