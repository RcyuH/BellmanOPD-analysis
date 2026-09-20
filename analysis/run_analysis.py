"""Offline, read-only CMT analysis of sampled token and update records."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

if __package__:
    from .metrics import PREDICTORS, conditional_d_comparison, correlations, quantile_curve
else:
    # Also support `python run_analysis.py` from inside the analysis directory.
    from metrics import PREDICTORS, conditional_d_comparison, correlations, quantile_curve


def _read_rows(directory: Path) -> list[dict]:
    result: list[dict] = []
    for path in sorted(directory.glob("step-*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            result.extend(json.loads(line) for line in handle if line.strip())
    return result


def _legacy_score_samples(root: Path) -> list[dict]:
    """Read existing CMT score samples without inventing token identities/outcomes."""
    result = []
    valid_by_step = {}
    metrics_path = root / "metrics.jsonl"
    if metrics_path.is_file():
        with metrics_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    item = json.loads(line)
                    if item.get("step") is not None and item.get("num_valid_tokens") is not None:
                        valid_by_step[int(item["step"])] = int(item["num_valid_tokens"])
    for path in sorted((root / "token_score_stats").glob("step-*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        scores = payload.get("scores", {})
        names = {"g": "gain", "x": "successor_excess", "d": "sequential_gain"}
        if any(name not in scores for name in names.values()):
            continue
        if "w" in scores:
            names["w"] = "w"
        counts = {int(scores[name]["count"]) for name in names.values()}
        samples = {key: scores[name].get("sample", []) for key, name in names.items()}
        lengths = {len(values) for values in samples.values()}
        if len(counts) != 1 or len(lengths) != 1 or valid_by_step.get(int(payload["step"])) != next(iter(counts)):
            continue
        for index in range(next(iter(lengths))):
            values = {key: float(sample[index]) for key, sample in samples.items()}
            result.append({
                "scoring_step": int(payload["step"]),
                "legacy_sample_index": index,
                **values,
                "g_plus_x": values["g"] + values["x"],
                "g_plus_d": values["g"] + values["d"],
            })
    return result


def _csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save(fig, output: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(output / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")


def _position_profile(rows: list[dict], *, bins: int = 20) -> list[dict]:
    result = []
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        selected = [
            row for row in rows
            if low < float(row["normalized_position"]) <= high
            and all(math.isfinite(float(row[key])) for key in ("g", "x", "d"))
        ]
        if not selected:
            continue
        item = {"position_bin": index + 1, "position_midpoint": (low + high) / 2, "n": len(selected)}
        for key in ("g", "x", "d"):
            values = np.asarray([row[key] for row in selected], dtype=float)
            item[f"{key}_mean"] = float(values.mean())
            item[f"{key}_se"] = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else math.nan
        result.append(item)
    return result


def run(input_dir: Path, output_dir: Path, *, outcome: str = "delta_kl", bins: int = 10) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = input_dir.expanduser().resolve()
    if not (root / "manifest.json").is_file() and (root / "analysis" / "manifest.json").is_file():
        root = root / "analysis"
    legacy = not (root / "manifest.json").is_file()
    if legacy and not (root / "token_score_stats").is_dir():
        raise FileNotFoundError(f"Neither analysis/manifest.json nor token_score_stats found under {input_dir}")
    if outcome not in {"delta_kl", "delta_nll", "delta_future_kl", "delta_future_nll"}:
        raise ValueError("outcome must be delta_kl, delta_nll, delta_future_kl, or delta_future_nll")
    if bins <= 0:
        raise ValueError("bins must be positive")
    output = output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    tokens = _legacy_score_samples(root) if legacy else _read_rows(root / "tokens")
    progress = [] if legacy else _read_rows(root / "progress")
    result = {
        "input": str(root), "token_rows": len(tokens), "progress_rows": len(progress),
        "outcome": outcome, "figures": [], "legacy_score_samples": legacy,
        "interpretation": "Associations on sampled training states; before/after is one full optimizer update, not a token-only intervention.",
    }
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 120})

    if tokens:
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
        for axis, key in zip(axes, ("g", "x", "d")):
            values = np.asarray([row[key] for row in tokens], dtype=float)
            axis.hist(values[np.isfinite(values)], bins=40, color="#355c7d", alpha=.85)
            axis.set(xlabel=key, ylabel="Sampled token count", title=f"Distribution of {key}")
            axis.grid(alpha=.2)
        _save(fig, output, "score_distributions")
        plt.close(fig)
        result["figures"].append("score_distributions")
        static = []
        for left in PREDICTORS:
            for right in ("g", "x", "d", "w"):
                if left == right or not all(left in row and right in row for row in tokens):
                    continue
                static.append({"left": left, "right": right, **correlations(
                    [row[left] for row in tokens], [row[right] for row in tokens]
                )})
        _csv(output / "score_correlations.csv", static)
        position = _position_profile(tokens) if all("normalized_position" in row for row in tokens) else []
        _csv(output / "position_profile.csv", position)
        if position:
            fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharex=True)
            for axis, key, label in zip(axes, ("g", "x", "d"), ("Local gain $g_t$", "Successor excess $X_t$", "Sequential gain $\\hat D_t$")):
                x = np.asarray([row["position_midpoint"] for row in position])
                mean = np.asarray([row[f"{key}_mean"] for row in position])
                se = np.asarray([row[f"{key}_se"] for row in position])
                axis.plot(x, mean, marker="o", markersize=3)
                axis.fill_between(x, mean - 1.96 * se, mean + 1.96 * se, alpha=.18)
                axis.set_title(label)
                axis.set_xlabel("Normalized response position")
                axis.set_ylabel("Mean score")
                axis.grid(alpha=.2)
            _save(fig, output, "position_distributions")
            plt.close(fig)
            result["figures"].append("position_distributions")

    if progress:
        if not any(row.get(outcome) is not None and math.isfinite(float(row[outcome])) for row in progress):
            raise ValueError(f"No finite {outcome} observations in progress records")
        curves = []
        correlation_rows = []
        for predictor in PREDICTORS:
            curve = quantile_curve(progress, predictor, outcome, bins)
            curves.extend({"predictor": predictor, "outcome": outcome, **row} for row in curve)
            pair = correlations([row[predictor] for row in progress], [row[outcome] for row in progress])
            correlation_rows.append({"predictor": predictor, "outcome": outcome, **pair})
        _csv(output / "quantile_curves.csv", curves)
        _csv(output / "correlations.csv", correlation_rows)
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
        for axis, predictor in zip(axes, ("g", "d")):
            selected = [row for row in curves if row["predictor"] == predictor]
            axis.errorbar(
                [row["bin"] for row in selected],
                [row["outcome_mean"] for row in selected],
                yerr=[1.96 * row["outcome_se"] for row in selected],
                marker="o", capsize=2,
            )
            axis.axhline(0, color="black", linewidth=.7, alpha=.5)
            axis.set(title=f"{predictor} quantile vs realized {outcome}", xlabel=f"{predictor} quantile (low to high)", ylabel=f"Mean {outcome}")
            axis.grid(alpha=.2)
        _save(fig, output, "g_d_quantile_progress")
        plt.close(fig)
        result["figures"].append("g_d_quantile_progress")

        fig, axis = plt.subplots(figsize=(6.2, 4.1))
        for predictor in PREDICTORS:
            selected = [row for row in curves if row["predictor"] == predictor]
            axis.plot([row["bin"] for row in selected], [row["outcome_mean"] for row in selected], marker="o", label=predictor)
        axis.axhline(0, color="black", linewidth=.7, alpha=.5)
        axis.set(xlabel="Predictor quantile (low to high)", ylabel=f"Mean {outcome}", title="Predictive relationship on the same sampled updates")
        axis.legend(frameon=False)
        axis.grid(alpha=.2)
        _save(fig, output, "predictor_comparison")
        plt.close(fig)
        result["figures"].append("predictor_comparison")

        conditional = conditional_d_comparison(progress, outcome, g_bins=bins)
        _csv(output / "conditional_d_within_g.csv", conditional)
        if conditional:
            fig, axis = plt.subplots(figsize=(6.2, 4.1))
            x = np.asarray([row["g_bin"] for row in conditional])
            axis.plot(x, [row["outcome_low_mean"] for row in conditional], marker="o", label="Low D within g bin")
            axis.plot(x, [row["outcome_high_mean"] for row in conditional], marker="o", label="High D within g bin")
            axis.set(xlabel="Matched g quantile", ylabel=f"Mean {outcome}", title="Conditional comparison (observational)")
            axis.legend(frameon=False)
            axis.grid(alpha=.2)
            _save(fig, output, "conditional_d_within_g")
            plt.close(fig)
            result["figures"].append("conditional_d_within_g")

        fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
        for axis, predictor in zip(axes, ("g", "d", "g_plus_d")):
            x = np.asarray([row[predictor] for row in progress], dtype=float)
            y = np.asarray([row[outcome] for row in progress], dtype=float)
            finite = np.isfinite(x) & np.isfinite(y)
            axis.hexbin(x[finite], y[finite], gridsize=30, mincnt=1, cmap="viridis")
            axis.set(xlabel=predictor, ylabel=outcome)
        _save(fig, output, "score_progress_density")
        plt.close(fig)
        result["figures"].append("score_progress_density")
    else:
        result["missing_progress_reason"] = (
            "No before/after update records. Existing score logs cannot reconstruct an optimizer-step outcome; "
            "future runs can enable analysis.measure_learning_progress."
        )
    if legacy:
        result["legacy_limit"] = (
            "Original token_score_stats samples have no token IDs, sequence positions, or realized progress. "
            "Paired score samples are used only when each score count equals metrics.jsonl num_valid_tokens, "
            "which confirms no finite-value filtering changed their shared index order."
        )
    (output / "analysis_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Run output or its analysis directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outcome", choices=("delta_kl", "delta_nll", "delta_future_kl", "delta_future_nll"), default="delta_kl")
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output, outcome=args.outcome, bins=args.bins), indent=2))


if __name__ == "__main__":
    main()
