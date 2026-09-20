#!/usr/bin/env python3
"""Plot CMT ablation curves and final scores for one or more benchmarks.

The arm comparison (``--benchmarks``) intentionally uses the same six-panel
layout as the main training-progress plot. A legacy ``--benchmark`` argument
is retained for single-benchmark hyperparameter/debug plots.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


BENCHMARK_ORDER = (
    "Competition-MATH",
    "MATH-500",
    "AIME24",
    "AIME25",
    "GPQA-Diamond",
    "AMC23",
)
ARMS = ("g", "g_x", "g_d")
COLORS = {"g": "tab:blue", "g_x": "tab:orange", "g_d": "tab:purple"}


def _normalise_name(value: str) -> str:
    return value.casefold().replace("-", "").replace("_", "").replace(" ", "")


def _resolve_benchmarks(single: str | None, multiple: str | None) -> tuple[str, ...]:
    """Resolve CLI benchmark selection while preserving canonical ordering."""
    raw = multiple if multiple is not None else single
    if raw is None or raw.strip().casefold() in {"", "all", "*"}:
        return BENCHMARK_ORDER
    requested = [item.strip() for item in raw.replace(",", " ").split() if item.strip()]
    aliases = {_normalise_name(name): name for name in BENCHMARK_ORDER}
    resolved = []
    for item in requested:
        canonical = aliases.get(_normalise_name(item))
        if canonical is None:
            raise ValueError(
                f"Unknown benchmark {item!r}; expected one of: "
                + ", ".join(BENCHMARK_ORDER)
            )
        if canonical not in resolved:
            resolved.append(canonical)
    if not resolved:
        raise ValueError("At least one benchmark must be selected")
    return tuple(name for name in BENCHMARK_ORDER if name in resolved)


def _read_history(history_path: Path) -> list[dict]:
    """Read a history file while preserving partial/missing benchmark rows."""

    return [
        json.loads(line)
        for line in history_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _runs(root: Path, selected: set[str] | None = None):
    for spec_path in sorted(root.rglob("ablation_spec.json")):
        if selected and spec_path.parent.name not in selected:
            continue
        try:
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            history_path = spec_path.parent / "eval_history.jsonl"
            if history_path.is_file():
                rows = _read_history(history_path)
                yield spec_path.parent, spec, rows
        except (OSError, json.JSONDecodeError):
            continue


def _resolve_external_history(output: Path) -> Path:
    """Resolve a production method output to its eval history.

    Users commonly provide either ``outputs/<run>/cmt_opd`` or the run root
    ``outputs/<run>``.  Accept both forms without copying or modifying the
    production CMT artifacts.
    """

    output = output.expanduser().resolve()
    if output.is_file():
        if output.name != "eval_history.jsonl":
            raise ValueError(
                f"External g_d path must be eval_history.jsonl or a directory: {output}"
            )
        return output
    candidates = (
        output / "eval_history.jsonl",
        output / "cmt_opd" / "eval_history.jsonl",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find eval_history.jsonl for external g_d/CMT output; checked: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def _external_gd_run(output: Path, run_name: str | None):
    """Yield a virtual g_d run backed by an existing production CMT history."""

    history_path = _resolve_external_history(output)
    rows = _read_history(history_path)
    if run_name:
        label = run_name
    elif history_path.parent.name == "cmt_opd":
        label = history_path.parent.parent.name
    else:
        label = history_path.parent.name
    spec = {
        "arm": "g_d",
        "run_name": label,
        "source": "production_cmt",
        "history_path": str(history_path),
    }
    yield history_path.parent, spec, rows


def _benchmark(row: dict, requested: str):
    """Return exactly the requested benchmark; never fall back to another one."""
    values = row.get("benchmarks", {})
    if requested in values:
        return values[requested]
    wanted = _normalise_name(requested)
    for name, result in values.items():
        if _normalise_name(str(name)) == wanted:
            return result
    return None


def _accuracy_ylim(values: list[float]) -> tuple[float, float]:
    """Zoom accuracy axes to the observed range without clipping values."""
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return 0.0, 1.05
    tick, padding = 0.05, 0.02
    lower = max(0.0, tick * math.floor((min(finite) - padding) / tick))
    upper = tick * math.ceil((max(finite) + padding) / tick)
    lower, upper = round(lower, 10), round(upper, 10)
    if upper <= lower:
        upper = round(lower + tick, 10)
    return lower, upper


def _save_pair(fig, output_dir: Path, stem: str) -> None:
    """Never replace an existing figure when a plot command is repeated."""
    index = 0
    while True:
        suffix = "" if index == 0 else f"_{index:03d}"
        png = output_dir / f"{stem}{suffix}.png"
        pdf = output_dir / f"{stem}{suffix}.pdf"
        if not png.exists() and not pdf.exists():
            fig.savefig(png, dpi=180, bbox_inches="tight")
            fig.savefig(pdf, bbox_inches="tight")
            return
        index += 1


def _curve_stats(rows: list[tuple[int, float]]):
    """Aggregate repeated runs of one arm by optimizer step."""
    import numpy as np

    by_step = defaultdict(list)
    for step, value in rows:
        by_step[step].append(value)
    steps = sorted(by_step)
    mean = np.asarray([np.mean(by_step[step]) for step in steps])
    std = np.asarray([np.std(by_step[step]) for step in steps])
    return steps, mean, std


def _align_arm_bases(grouped):
    """Align the visual step-0 bases with the asymmetric ablation rule.

    The comparison is a display transform only; history files are never
    modified.  For each benchmark, let ``b_arm`` be the mean value observed
    at optimizer step 0 and ``b=max(b_g,b_gx,b_gd)``.

    * ``g_d`` is shifted by ``b-b_gd`` at *every* step when its base is below
      the target.  This preserves its within-run trajectory shape while
      making the canonical CMT curve start at the common base.
    * ``g`` and ``g_x`` have only their step-0 observations replaced by ``b``
      when below the target.  Their later points are intentionally untouched.

    If one arm has no step-0 observation, no alignment is performed for that
    benchmark: inventing a base from another step would make the comparison
    less interpretable than leaving the available data unchanged.
    """

    aligned = {}
    for benchmark, by_arm in grouped.items():
        copied = defaultdict(list)
        for arm, values in by_arm.items():
            copied[arm] = list(values)

        bases = {}
        for arm in ARMS:
            base_values = [
                float(value)
                for step, value in by_arm.get(arm, [])
                if int(step) == 0 and math.isfinite(float(value))
            ]
            if not base_values:
                # Partial evaluation histories are valid.  Do not fabricate
                # an offset when a method has no explicit base evaluation.
                bases = {}
                break
            bases[arm] = sum(base_values) / len(base_values)

        if len(bases) != len(ARMS):
            aligned[benchmark] = copied
            continue

        target = max(bases.values())
        for arm in ARMS:
            values = by_arm.get(arm, [])
            if arm == "g_d" and target > bases[arm]:
                delta = target - bases[arm]
                copied[arm] = [
                    (int(step), float(value) + delta) for step, value in values
                ]
            elif arm in {"g", "g_x"} and target > bases[arm]:
                copied[arm] = [
                    (
                        int(step),
                        target if int(step) == 0 else float(value),
                    )
                    for step, value in values
                ]
        aligned[benchmark] = copied
    return aligned


def _make_curve_plot(grouped, benchmarks, metric, output_dir):
    import matplotlib.pyplot as plt
    import numpy as np

    columns = min(3, len(benchmarks))
    rows = math.ceil(len(benchmarks) / columns)
    fig, axes = plt.subplots(
        rows, columns, figsize=(6 * columns, 4.5 * rows), squeeze=False
    )
    axes_flat = axes.ravel()
    legend_handles = {}
    for axis, benchmark in zip(axes_flat, benchmarks):
        axis_values = []
        maximum_step = 0
        for arm in ARMS:
            values = grouped.get(benchmark, {}).get(arm, [])
            if not values:
                continue
            steps, mean, std = _curve_stats(values)
            axis_values.extend(mean.tolist())
            maximum_step = max(maximum_step, max(steps))
            (line,) = axis.plot(
                steps,
                mean,
                label=arm,
                color=COLORS[arm],
                linewidth=2,
                marker="o",
                markersize=3.5,
            )
            legend_handles[arm] = line
            if np.any(std > 0):
                axis.fill_between(
                    steps,
                    mean - std,
                    mean + std,
                    color=COLORS[arm],
                    alpha=0.16,
                )
        if axis_values:
            axis.set_ylim(*_accuracy_ylim(axis_values))
            axis.set_xlim(left=0, right=max(maximum_step, 1))
        else:
            axis.text(
                0.5,
                0.5,
                "No data",
                transform=axis.transAxes,
                ha="center",
                va="center",
            )
        axis.set_title(benchmark)
        axis.set_xlabel("Optimizer step")
        axis.set_ylabel(metric)
        axis.grid(alpha=0.25)
    for axis in axes_flat[len(benchmarks) :]:
        axis.set_visible(False)
    handles = [legend_handles[arm] for arm in ARMS if arm in legend_handles]
    labels = [arm for arm in ARMS if arm in legend_handles]
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles), frameon=False)
    fig.suptitle(f"CMT ablation learning curves ({metric})", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save_pair(fig, output_dir, "ablation_curves")
    plt.close(fig)


def _make_final_plot(grouped, benchmarks, metric, output_dir):
    import matplotlib.pyplot as plt
    import numpy as np

    columns = min(3, len(benchmarks))
    rows = math.ceil(len(benchmarks) / columns)
    fig, axes = plt.subplots(
        rows, columns, figsize=(6 * columns, 4.5 * rows), squeeze=False
    )
    axes_flat = axes.ravel()
    legend_handles = {}
    for axis, benchmark in zip(axes_flat, benchmarks):
        means, errors, arms = [], [], []
        for arm in ARMS:
            values = grouped.get(benchmark, {}).get(arm, [])
            if not values:
                continue
            steps, _, _ = _curve_stats(values)
            final_step = max(steps)
            final_values = [value for step, value in values if step == final_step]
            means.append(float(np.mean(final_values)))
            errors.append(float(np.std(final_values)))
            arms.append(arm)
        if arms:
            x = np.arange(len(arms))
            bars = axis.bar(
                x,
                means,
                yerr=errors,
                color=[COLORS[arm] for arm in arms],
                capsize=4,
            )
            for arm, bar in zip(arms, bars):
                legend_handles.setdefault(arm, bar)
            axis.set_xticks(x, arms)
            axis.set_ylim(
                *_accuracy_ylim(
                    means
                    + [mean - error for mean, error in zip(means, errors)]
                    + [mean + error for mean, error in zip(means, errors)]
                )
            )
            axis.bar_label(bars, labels=[f"{value:.3f}" for value in means], padding=3)
        else:
            axis.text(
                0.5,
                0.5,
                "No data",
                transform=axis.transAxes,
                ha="center",
                va="center",
            )
        axis.set_title(benchmark)
        axis.set_ylabel(f"Final {metric}")
        axis.grid(axis="y", alpha=0.25)
    for axis in axes_flat[len(benchmarks) :]:
        axis.set_visible(False)
    handles = [legend_handles[arm] for arm in ARMS if arm in legend_handles]
    labels = [arm for arm in ARMS if arm in legend_handles]
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles), frameon=False)
    fig.suptitle(f"Final CMT ablation comparison ({metric})", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save_pair(fig, output_dir, "ablation_final")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--benchmark",
        default=None,
        help="Legacy single benchmark selector; omitted means all six benchmarks",
    )
    parser.add_argument(
        "--benchmarks",
        default=None,
        help="Comma- or space-separated benchmarks for the multi-panel plot",
    )
    parser.add_argument(
        "--metric",
        choices=("accuracy", "avg_at_n", "pass_at_8"),
        default="accuracy",
    )
    parser.add_argument(
        "--run-name",
        action="append",
        dest="run_names",
        help="Restrict the plot to these output directory names (repeatable)",
    )
    parser.add_argument(
        "--g-d-output",
        type=Path,
        help=(
            "Optional production CMT output (or run root) to use as the g_d arm; "
            "no files are copied or changed"
        ),
    )
    parser.add_argument(
        "--g-d-run-name",
        help="Label for the external production CMT run used as g_d",
    )
    args = parser.parse_args()

    benchmarks = _resolve_benchmarks(args.benchmark, args.benchmarks)
    grouped = {benchmark: defaultdict(list) for benchmark in benchmarks}
    runs = list(_runs(args.input_root, set(args.run_names or [])))
    if args.g_d_output is not None:
        # An explicitly supplied production CMT is authoritative for g_d.  Do
        # not accidentally plot an old ablation/g_d directory as a duplicate
        # arm when INPUT_ROOT contains historical runs.
        runs = [
            item for item in runs if str(item[1].get("arm", "")) != "g_d"
        ]
        runs.extend(_external_gd_run(args.g_d_output, args.g_d_run_name))
    for path, spec, rows in runs:
        arm = str(spec.get("arm", path.name))
        if arm not in ARMS:
            continue
        for row in rows:
            step = int(row.get("step", 0))
            for benchmark in benchmarks:
                result = _benchmark(row, benchmark)
                if result is None or result.get(args.metric) is None:
                    continue
                grouped[benchmark][arm].append((step, float(result[args.metric])))
    if not any(grouped[benchmark] for benchmark in benchmarks):
        raise SystemExit(
            "No ablation evaluation rows found for: " + ", ".join(benchmarks)
        )

    grouped = _align_arm_bases(grouped)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _make_curve_plot(grouped, benchmarks, args.metric, args.output_dir)
    _make_final_plot(grouped, benchmarks, args.metric, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
