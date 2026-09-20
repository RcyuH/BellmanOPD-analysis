from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .evaluation import BENCHMARK_ORDER, MODEL_ORDER, evaluation_metric_name


_PROGRESS_METHOD_ALIASES = {
    "all": "all",
    "three": "all",
    "both": "both",
    "opd": "opd",
    "pure-opd": "opd",
    "ta": "ta",
    "ta-opd": "ta",
    "rac": "rac",
    "bellman-rac": "rac",
    "pgt": "pgt",
    "cmt": "cmt",
    "grpo": "grpo",
    "iw": "iw",
    "iw-opd": "iw",
}
_PROGRESS_METHODS = {
    "opd": {
        "label": "OPD",
        "slug": "opd",
        "color": "tab:green",
        "output_argument": "--opd-output",
    },
    "ta": {
        "label": "TA-OPD",
        "slug": "ta_opd",
        "color": "tab:blue",
        "output_argument": "--ta-output",
    },
    "rac": {
        "label": "Bellman-RAC",
        "slug": "rac",
        "color": "tab:orange",
        "output_argument": "--rac-output",
    },
    "pgt": {
        "label": "PGT",
        "slug": "pgt_opd",
        "color": "tab:red",
        "output_argument": "--pgt-output",
    },
    "cmt": {
        "label": "CMT-OPD",
        "slug": "cmt_opd",
        "color": "tab:purple",
        "output_argument": "--cmt-output",
    },
    "grpo": {
        "label": "GRPO",
        "slug": "grpo",
        "color": "tab:brown",
        "output_argument": "--grpo-output",
    },
    "iw": {
        "label": "IW-OPD",
        "slug": "iw_opd",
        "color": "tab:cyan",
        "output_argument": "--iw-output",
    },
}

# Accuracy is stored as a fraction in [0, 1].  Evaluation at step 0 can differ
# slightly between method runs because of sampling/evaluation nondeterminism;
# Two percentage points is the accepted Step-0 sampling/evaluation tolerance.
# This only guards the comparable supervised math benchmarks; it is not used
# to alter the source evaluation histories.
_STEP_ZERO_ACCURACY_TOLERANCE = 0.02
_STEP_ZERO_BASE_CHECK_BENCHMARKS = frozenset({"Competition-MATH", "MATH-500"})


def _accuracy_ylim(values: list[float] | tuple[float, ...]) -> tuple[float, float]:
    """Return readable, data-dependent limits for accuracy plots.

    The old plots always used ``[0, 1.05]``, which compresses the differences
    when all methods occupy a narrow band (for example, 0.58--0.65).  We keep
    a small two-point (percentage-point) visual margin and round the limits to
    five-point ticks, yielding 0.55--0.70 for that example.  Limits are never
    allowed to clip an observed value; the lower limit is clamped at zero
    because accuracy itself is non-negative.
    """

    finite = [float(value) for value in values if np.isfinite(float(value))]
    if not finite:
        return 0.0, 1.05

    tick = 0.05
    padding = 0.02
    lower = max(0.0, tick * math.floor((min(finite) - padding) / tick))
    upper = tick * math.ceil((max(finite) + padding) / tick)
    # Floating-point round-off can produce e.g. 0.7000000000000001.  This is
    # only a presentation helper, so rounding keeps serialized/tested limits
    # stable without changing any metric values.
    lower = round(lower, 10)
    upper = round(upper, 10)
    if upper <= lower:
        upper = round(lower + tick, 10)
    return lower, upper


def _plot_directory(results_dir: Path, plot_name: str | None) -> Path:
    name = plot_name or f"plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not name.replace("-", "").replace(".", "").replace("_", "").isalnum():
        raise ValueError(f"Invalid plot output name: {name!r}")
    path = results_dir / "plots" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_figure(fig, path: Path) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _normalize_progress_method(method: str) -> str:
    normalized = method.strip().lower()
    if normalized not in _PROGRESS_METHOD_ALIASES:
        choices = ", ".join(_PROGRESS_METHOD_ALIASES)
        raise ValueError(f"Unknown plot method {method!r}; expected one of: {choices}")
    return _PROGRESS_METHOD_ALIASES[normalized]


def _normalize_progress_methods(
    method: str = "both", methods: list[str] | tuple[str, ...] | None = None
) -> tuple[str, ...]:
    requested = list(methods) if methods is not None else [method]
    selected = []
    for item in requested:
        normalized = _normalize_progress_method(item)
        expanded = {
            # Keep the historical ``all`` alias stable.  GRPO is opt-in via
            # ``--methods ... grpo`` so old plotting commands do not suddenly
            # require a GRPO run that may not exist.
            "all": ("opd", "ta", "rac", "pgt", "cmt"),
            "both": ("ta", "rac"),
        }.get(normalized, (normalized,))
        for candidate in expanded:
            if candidate not in selected:
                selected.append(candidate)
    if not selected:
        raise ValueError("At least one training method must be selected for plotting")
    return tuple(selected)


def _read_eval_history(output: str | Path | None, method: str) -> list[dict]:
    spec = _PROGRESS_METHODS[method]
    if output is None:
        raise ValueError(
            f"{spec['output_argument']} is required when plotting {spec['label']}"
        )
    history_path = Path(output).resolve() / "eval_history.jsonl"
    if not history_path.is_file():
        raise FileNotFoundError(
            f"Missing {spec['label']} evaluation history: {history_path}"
        )
    rows = _read_jsonl(history_path)
    if not rows:
        raise ValueError(f"{spec['label']} evaluation history is empty: {history_path}")
    rows.sort(key=lambda row: int(row["step"]))
    for row in rows:
        reported = row.get("benchmarks", {})
        # Histories produced by the previous release can still contain an
        # IFEval result.  It is intentionally ignored (never relabelled as
        # AMC23) so replacing the benchmark does not make old plots unreadable.
        unknown = set(reported) - set(BENCHMARK_ORDER) - {"IFEval"}
        benchmark_names = tuple(
            benchmark for benchmark in BENCHMARK_ORDER if benchmark in reported
        )
        if unknown or not benchmark_names:
            raise ValueError(
                f"{spec['label']} evaluation at step {row.get('step')} has invalid "
                f"benchmarks: {tuple(reported)}"
            )
        missing = [
            benchmark
            for benchmark in benchmark_names
            if "accuracy" not in reported[benchmark]
        ]
        if missing:
            raise ValueError(
                f"{spec['label']} evaluation at step {row.get('step')} is missing "
                f"accuracy for: {', '.join(missing)}"
            )
    return rows


def _shared_history_benchmarks(histories: dict[str, list[dict]]) -> tuple[str, ...]:
    """Return benchmarks present in every method at least once.

    A checkpoint evaluation can be partial (for example, a re-evaluation of
    only GPQA/AMC) and therefore one history row may contain fewer benchmarks
    than the other rows.  That must remove only the missing point, not the
    entire benchmark from the comparison.
    """
    available_sets: list[set[str]] = []
    for method, rows in histories.items():
        available: set[str] = set()
        for row in rows:
            available.update(row.get("benchmarks", {}))
        if not available:
            raise ValueError(f"{method} has no evaluation benchmarks")
        available_sets.append(available)
    common = set.intersection(*available_sets) if available_sets else set()
    shared = tuple(name for name in BENCHMARK_ORDER if name in common)
    if not shared:
        raise ValueError("No common evaluation benchmarks were found")
    return shared


def _step_zero_accuracy(rows: list[dict], benchmark: str) -> float | None:
    """Return a method's Step-0 accuracy for one benchmark, if present."""
    for row in rows:
        if int(row.get("step", -1)) == 0 and benchmark in row.get("benchmarks", {}):
            return float(row["benchmarks"][benchmark]["accuracy"])
    return None


def _align_opd_cmt_for_plot(
    histories: dict[str, list[dict]],
    benchmark_names: tuple[str, ...],
) -> tuple[dict[str, list[dict]], dict[str, float]]:
    """Align OPD/CMT Step-0 points using the requested asymmetric rule.

    The raw ``eval_history.jsonl`` files are never modified.  For plotting,
    this rule is applied independently to every benchmark shared by OPD/CMT
    (including GPQA-Diamond and AMC23): when OPD starts above CMT, the OPD-CMT
    gap is added to every CMT point.  If CMT starts above OPD, only OPD's
    Step-0 point is lifted.  The shared Base reference is the higher of the two
    initial values, so the two initial points coincide without inventing a
    correction for the whole OPD curve in the second case.  This is separate
    from the Step-0 consistency gate, which intentionally checks only
    Competition-MATH and MATH-500.
    """
    opd_label = _PROGRESS_METHODS["opd"]["label"]
    cmt_label = _PROGRESS_METHODS["cmt"]["label"]
    if opd_label not in histories or cmt_label not in histories:
        return histories, {}

    adjusted: dict[str, list[dict]] = dict(histories)
    alignment: dict[str, float] = {}
    for benchmark in benchmark_names:
        opd_base = _step_zero_accuracy(histories[opd_label], benchmark)
        cmt_base = _step_zero_accuracy(histories[cmt_label], benchmark)
        if opd_base is None or cmt_base is None:
            continue
        target = max(opd_base, cmt_base)
        alignment[benchmark] = target
        if opd_base == cmt_base:
            continue

        if opd_base > cmt_base:
            method_to_shift = cmt_label
            delta = opd_base - cmt_base
            shift_all_steps = True
        else:
            method_to_shift = opd_label
            delta = cmt_base - opd_base
            shift_all_steps = False

        transformed_rows = []
        # Start from the already transformed rows so adjustments for multiple
        # benchmarks accumulate instead of overwriting one another.
        for row in adjusted[method_to_shift]:
            copied = dict(row)
            copied_benchmarks = dict(row.get("benchmarks", {}))
            result = copied_benchmarks.get(benchmark)
            should_shift = shift_all_steps or int(row.get("step", -1)) == 0
            if result is not None and should_shift:
                copied_result = dict(result)
                copied_result["accuracy"] = float(result["accuracy"]) + delta
                copied_benchmarks[benchmark] = copied_result
            copied["benchmarks"] = copied_benchmarks
            transformed_rows.append(copied)
        adjusted[method_to_shift] = transformed_rows

    return adjusted, alignment


def _history_metric_name(histories: dict[str, list[dict]]) -> str:
    """Require one evaluation protocol and return its display metric."""
    metrics = set()
    for rows in histories.values():
        for row in rows:
            configured = row.get("parameters", {}).get("metric")
            if configured:
                metrics.add(str(configured))
                continue
            samples = {
                int(result.get("samples_per_problem", 8))
                for result in row["benchmarks"].values()
            }
            if len(samples) != 1:
                raise ValueError(
                    f"Evaluation step {row.get('step')} mixes response counts: {samples}"
                )
            metrics.add(evaluation_metric_name(samples.pop()))
    if len(metrics) != 1:
        raise ValueError(
            "Cannot compare histories generated with different evaluation metrics: "
            f"{sorted(metrics)}"
        )
    return metrics.pop()


def _moving_average(values: list[float], window: int) -> np.ndarray:
    values_array = np.asarray(values, dtype=float)
    if window <= 1:
        return values_array
    result = np.empty_like(values_array)
    for index in range(len(values_array)):
        result[index] = values_array[max(0, index - window + 1) : index + 1].mean()
    return result


def _plot_loss_comparison(
    plots_dir: Path,
    outputs: dict[str, str | Path],
    smoothing_window: int,
) -> Path:
    fig, axis = plt.subplots(figsize=(9, 5.5))
    for method, output in outputs.items():
        spec = _PROGRESS_METHODS[method]
        label, color = spec["label"], spec["color"]
        metrics = _read_jsonl(Path(output).resolve() / "metrics.jsonl")
        if not metrics:
            raise ValueError(f"{label} training metrics are empty")
        steps = [int(row["step"]) for row in metrics]
        losses = [float(row["loss"]) for row in metrics]
        axis.plot(
            steps, losses, color=color, alpha=0.25, linewidth=1, label=f"{label} raw"
        )
        axis.plot(
            steps,
            _moving_average(losses, smoothing_window),
            color=color,
            linewidth=2,
            label=f"{label} MA({smoothing_window})",
        )
        if method != "opd" and any("unweighted_opd_loss" in row for row in metrics):
            unweighted = [
                float(row.get("unweighted_opd_loss", row["loss"])) for row in metrics
            ]
            axis.plot(
                steps,
                _moving_average(unweighted, smoothing_window),
                color=color,
                linewidth=1.2,
                linestyle=":",
                label=f"{label} unweighted",
            )
    axis.set_xlabel("Optimizer step")
    axis.set_ylabel("OPD loss")
    axis.set_title(" vs ".join(_PROGRESS_METHODS[item]["label"] for item in outputs))
    axis.set_title(axis.get_title() + " training loss")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    path = plots_dir / "loss_comparison.png"
    _save_figure(fig, path)
    plt.close(fig)
    return path


def _read_token_stats(output: Path) -> list[dict]:
    root = output / "token_score_stats"
    if not root.is_dir():
        return []
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(root.glob("step-*.json"))
    ]
    return sorted(rows, key=lambda row: int(row["step"]))


def _snapshot_rows(rows: list[dict]) -> list[dict]:
    if not rows:
        return []
    indices = sorted({0, len(rows) // 2, len(rows) - 1})
    return [rows[index] for index in indices]


def _plot_token_score_distributions(
    plots_dir: Path, outputs: dict[str, str | Path]
) -> dict[str, str]:
    result: dict[str, str] = {}
    ta_rows = (
        _read_token_stats(Path(outputs["ta"]).resolve()) if "ta" in outputs else []
    )
    rac_rows = (
        _read_token_stats(Path(outputs["rac"]).resolve()) if "rac" in outputs else []
    )
    if ta_rows:
        fig, axis = plt.subplots(figsize=(8.5, 5.2))
        for row in _snapshot_rows(ta_rows):
            histogram = row["scores"]["s_TA"]["histogram"]
            edges = np.asarray(histogram["edges"], dtype=float)
            counts = np.asarray(histogram["counts"], dtype=float)
            counts /= max(counts.sum(), 1.0)
            axis.stairs(counts, edges, linewidth=1.8, label=f"step {row['step']}")
        axis.set(xlabel="TA local teachability s_teach", ylabel="Token fraction")
        axis.set_title("TA-OPD token-score distribution")
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        path = plots_dir / "ta_token_score_distribution.png"
        _save_figure(fig, path)
        plt.close(fig)
        result["ta_token_scores"] = str(path)
    if rac_rows:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
        for axis, key, title in zip(
            axes,
            ("g", "V", "w"),
            ("Local g", "Bellman V", "Soft weight w"),
        ):
            for row in _snapshot_rows(rac_rows):
                histogram = row["scores"][key]["histogram"]
                edges = np.asarray(histogram["edges"], dtype=float)
                counts = np.asarray(histogram["counts"], dtype=float)
                counts /= max(counts.sum(), 1.0)
                axis.stairs(counts, edges, linewidth=1.7, label=f"step {row['step']}")
            axis.set_title(title)
            axis.set_xlabel("Score")
            axis.grid(alpha=0.25)
        axes[0].set_ylabel("Token fraction")
        axes[-1].legend()
        fig.suptitle("Bellman-RAC token-score distributions")
        fig.tight_layout()
        path = plots_dir / "rac_token_score_distributions.png"
        _save_figure(fig, path)
        plt.close(fig)
        result["rac_token_scores"] = str(path)

        fig, axis = plt.subplots(figsize=(8.5, 5.2))
        steps = [int(row["step"]) for row in rac_rows]
        for key, label in (
            ("alignment", "mean alignment"),
            ("V", "mean V"),
            ("w", "mean weight"),
        ):
            axis.plot(
                steps,
                [float(row["scores"][key]["mean"]) for row in rac_rows],
                marker="o",
                linewidth=1.8,
                label=label,
            )
        axis.set(xlabel="Optimizer step", ylabel="Mean over valid tokens")
        axis.set_ylim(0.0, 1.05)
        axis.set_title("Bellman-RAC score means")
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        path = plots_dir / "rac_score_means.png"
        _save_figure(fig, path)
        plt.close(fig)
        result["rac_score_means"] = str(path)
    return result


# CMT keeps a richer set of token-score diagnostics than the legacy
# comparison plot needs.  Keep this mapping explicit so the notation used in
# the CMT derivation is visible in the generated figure and does not depend on
# internal compatibility aliases (for example ``gain`` rather than ``g``).
_CMT_SCORE_PLOT_SPECS = (
    ("gain", r"$g_t$ (local PGT gain)"),
    ("successor_excess", r"$X_t=R_{t+1}-g_tM_{t+1}$"),
    ("sequential_gain", r"$D_t$ (sequential marginal gain)"),
    ("learning_value", r"$L_t$ (CMT learning value)"),
    ("w", r"$w_t$ (allocated supervision weight)"),
)


def _safe_plot_identifier(value: str) -> str:
    """Turn a run name into a filesystem-safe plot identifier."""

    normalized = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in str(value)
    ).strip("._-")
    return normalized or "cmt_run"


def _unique_diagnostic_plot_directory(
    output_root: Path, requested_name: str
) -> Path:
    """Create a non-destructive, unique directory for a diagnostic render."""

    if not requested_name.replace("-", "").replace(".", "").replace("_", "").isalnum():
        raise ValueError(f"Invalid CMT diagnostic plot name: {requested_name!r}")
    root = output_root.resolve() / "plots"
    root.mkdir(parents=True, exist_ok=True)
    candidate = root / requested_name
    suffix = 2
    while True:
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            candidate = root / f"{requested_name}_{suffix:02d}"
            suffix += 1


def _cmt_score_payload(row: dict, key: str) -> dict | None:
    scores = row.get("scores", {})
    value = scores.get(key)
    if isinstance(value, dict) and isinstance(value.get("histogram"), dict):
        return value
    return None


def plot_cmt_score_distributions(
    cmt_output: str | Path,
    *,
    run_name: str | None = None,
    plot_name: str | None = None,
    output_root: str | Path | None = None,
) -> dict[str, str | int | list[str]]:
    """Plot CMT token-score distributions and learning-value evolution.

    ``token_score_stats/step-*.json`` contains exact binned counts over all
    valid tokens in the logged global rollout batch.  This function only
    consumes those compact artifacts; it never reruns scoring or model
    inference.  Every invocation gets a fresh ``plots/<name>`` directory,
    even when the caller reuses ``plot_name``.
    """

    source = Path(cmt_output).expanduser().resolve()
    rows = _read_token_stats(source)
    if not rows:
        raise FileNotFoundError(
            "No CMT token-score statistics found under "
            f"{source / 'token_score_stats'}; enable "
            "logging.token_score_stats_enabled during training."
        )

    run_tag = _safe_plot_identifier(run_name or source.parent.name or source.name)
    requested_name = plot_name or (
        f"cmt_scores_{run_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    )
    plots_dir = _unique_diagnostic_plot_directory(
        Path(output_root).expanduser().resolve() if output_root is not None else source,
        _safe_plot_identifier(requested_name),
    )
    prefix = f"cmt_{run_tag}"

    missing = [
        key
        for key, _ in _CMT_SCORE_PLOT_SPECS
        if not any(_cmt_score_payload(row, key) is not None for row in rows)
    ]
    if missing:
        raise ValueError(
            "CMT token-score statistics are missing required fields: "
            + ", ".join(missing)
        )

    # One compact figure compares the distribution at the first, middle and
    # final logged training snapshots.  Histogram counts already include the
    # logger's underflow/overflow values through edge clipping; the explicit
    # counters remain available in the JSON artifact for auditing.
    snapshots = _snapshot_rows(rows)
    figure, axes = plt.subplots(2, 3, figsize=(17, 9), squeeze=False)
    axes_flat = axes.reshape(-1)
    for axis, (key, label) in zip(axes_flat, _CMT_SCORE_PLOT_SPECS):
        for row in snapshots:
            payload = _cmt_score_payload(row, key)
            if payload is None:
                continue
            histogram = payload["histogram"]
            edges = np.asarray(histogram["edges"], dtype=float)
            counts = np.asarray(histogram["counts"], dtype=float)
            counts /= max(float(counts.sum()), 1.0)
            axis.stairs(
                counts,
                edges,
                linewidth=1.8,
                label=f"step {int(row['step'])}",
            )
        axis.set_title(label)
        axis.set_xlabel("Score")
        axis.set_ylabel("Token fraction")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    axes_flat[-1].axis("off")
    figure.suptitle(f"CMT token-score distributions — {run_tag}")
    figure.tight_layout()
    histogram_path = plots_dir / f"{prefix}_token_score_histograms.png"
    _save_figure(figure, histogram_path)
    plt.close(figure)

    # The snapshot figure is readable for a report, while this companion
    # heatmap retains every logged training step.  Each row is normalized by
    # its own token count, so a changing rollout length does not appear as a
    # spurious score-density change.
    figure, axes = plt.subplots(2, 3, figsize=(17, 9), squeeze=False)
    axes_flat = axes.reshape(-1)
    for axis, (key, label) in zip(axes_flat, _CMT_SCORE_PLOT_SPECS):
        field_rows = [
            (int(row["step"]), _cmt_score_payload(row, key)) for row in rows
        ]
        field_rows = [item for item in field_rows if item[1] is not None]
        if not field_rows:
            axis.axis("off")
            continue
        first_histogram = field_rows[0][1]["histogram"]
        edges = np.asarray(first_histogram["edges"], dtype=float)
        matrix = []
        step_values = []
        for step, payload in field_rows:
            histogram = payload["histogram"]
            current_edges = np.asarray(histogram["edges"], dtype=float)
            counts = np.asarray(histogram["counts"], dtype=float)
            # All artifacts produced by one logger share a range; fail loudly
            # instead of silently drawing a misleading heatmap if a file was
            # manually mixed from another configuration.
            if current_edges.shape != edges.shape or not np.allclose(
                current_edges, edges
            ):
                raise ValueError(
                    f"Inconsistent histogram bins for CMT field {key!r}"
                )
            matrix.append(counts / max(float(counts.sum()), 1.0))
            step_values.append(step)
        image = axis.imshow(
            np.asarray(matrix),
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=(edges[0], edges[-1], step_values[0], step_values[-1]),
        )
        axis.set_title(label)
        axis.set_xlabel("Score")
        axis.set_ylabel("Logged optimizer step")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Token fraction")
    axes_flat[-1].axis("off")
    figure.suptitle(f"CMT score histograms across training steps — {run_tag}")
    figure.tight_layout()
    heatmap_path = plots_dir / f"{prefix}_token_score_histogram_heatmaps.png"
    _save_figure(figure, heatmap_path)
    plt.close(figure)

    # Learning-value trajectory: retain both central quantiles and the mean so
    # a changing tail cannot be mistaken for a shift in the typical token.
    learning_rows = [
        (int(row["step"]), _cmt_score_payload(row, "learning_value"))
        for row in rows
    ]
    learning_rows = [item for item in learning_rows if item[1] is not None]
    steps = np.asarray([item[0] for item in learning_rows], dtype=float)
    mean = np.asarray([float(item[1]["mean"]) for item in learning_rows])
    q05 = np.asarray([float(item[1]["quantiles"]["q05"]) for item in learning_rows])
    q25 = np.asarray([float(item[1]["quantiles"]["q25"]) for item in learning_rows])
    q50 = np.asarray([float(item[1]["quantiles"]["q50"]) for item in learning_rows])
    q75 = np.asarray([float(item[1]["quantiles"]["q75"]) for item in learning_rows])
    q95 = np.asarray([float(item[1]["quantiles"]["q95"]) for item in learning_rows])
    figure, axis = plt.subplots(figsize=(10, 5.8))
    axis.fill_between(steps, q05, q95, alpha=0.16, label="q05–q95")
    axis.fill_between(steps, q25, q75, alpha=0.28, label="q25–q75")
    axis.plot(steps, mean, linewidth=2.2, marker="o", markersize=3.5, label="mean")
    axis.plot(steps, q50, linewidth=1.5, linestyle="--", label="median (q50)")
    axis.set_xlabel("Optimizer step (logged rollout endpoint)")
    axis.set_ylabel("CMT learning value")
    axis.set_title(f"CMT learning-value distribution over training — {run_tag}")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    learning_path = plots_dir / f"{prefix}_learning_value_quantiles.png"
    _save_figure(figure, learning_path)
    plt.close(figure)

    # A scalar-mean companion makes it easy to compare the five logged fields
    # without opening the JSON files individually.
    figure, axis = plt.subplots(figsize=(10, 5.8))
    for key, label in _CMT_SCORE_PLOT_SPECS:
        values = []
        score_steps = []
        for row in rows:
            payload = _cmt_score_payload(row, key)
            if payload is not None:
                score_steps.append(int(row["step"]))
                values.append(float(payload["mean"]))
        axis.plot(score_steps, values, marker="o", markersize=3, label=label)
    axis.set_xlabel("Optimizer step (logged rollout endpoint)")
    axis.set_ylabel("Mean over valid response tokens")
    axis.set_title(f"CMT token-score means over training — {run_tag}")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    means_path = plots_dir / f"{prefix}_score_means.png"
    _save_figure(figure, means_path)
    plt.close(figure)

    manifest = {
        "run_name": run_tag,
        "source": str(source),
        "plot_directory": str(plots_dir),
        "logged_steps": [int(row["step"]) for row in rows],
        "score_fields": [key for key, _ in _CMT_SCORE_PLOT_SPECS],
        "histograms": str(histogram_path),
        "histogram_heatmaps": str(heatmap_path),
        "learning_value_quantiles": str(learning_path),
        "score_means": str(means_path),
    }
    manifest_path = plots_dir / f"{prefix}_plot_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "run_name": run_tag,
        "plot_directory": str(plots_dir),
        "histograms": str(histogram_path),
        "histogram_heatmaps": str(heatmap_path),
        "learning_value_quantiles": str(learning_path),
        "score_means": str(means_path),
        "manifest": str(manifest_path),
        "logged_steps": manifest["logged_steps"],
    }


def _plot_single_method_accuracy(
    plots_dir: Path,
    method: str,
    rows: list[dict],
    metric_name: str,
    benchmark_names: tuple[str, ...],
) -> Path:
    spec = _PROGRESS_METHODS[method]
    colors = {
        "Competition-MATH": "tab:purple",
        "MATH-500": "tab:blue",
        "AIME24": "tab:orange",
        "AIME25": "tab:green",
        "GPQA-Diamond": "tab:brown",
        "AMC23": "tab:pink",
    }
    line_styles = {
        "Competition-MATH": ":",
        "MATH-500": "-",
        "AIME24": "--",
        "AIME25": "-.",
        "GPQA-Diamond": (0, (3, 1, 1, 1)),
        "AMC23": (0, (5, 2)),
    }
    fig, axis = plt.subplots(figsize=(9, 5.5))
    accuracy_values: list[float] = []
    all_steps = sorted({int(row["step"]) for row in rows})
    for benchmark in benchmark_names:
        available = {
            int(row["step"]): row
            for row in rows
            if benchmark in row.get("benchmarks", {})
        }
        if not available:
            continue
        benchmark_steps = all_steps
        values = [
            float(available[step]["benchmarks"][benchmark]["accuracy"])
            if step in available
            else float("nan")
            for step in benchmark_steps
        ]
        accuracy_values.extend(value for value in values if np.isfinite(value))
        axis.plot(
            benchmark_steps,
            values,
            color=colors[benchmark],
            linestyle=line_styles[benchmark],
            marker="o",
            markersize=4.5,
            linewidth=2,
            label=benchmark,
        )
    axis.set_xlabel("Optimizer step")
    axis.set_ylabel(metric_name)
    axis.set_ylim(*_accuracy_ylim(accuracy_values))
    axis.set_title(f"{spec['label']} evaluation {metric_name} during training")
    axis.grid(alpha=0.25)
    axis.legend(title="Dataset")
    fig.tight_layout()
    path = plots_dir / f"{spec['slug']}_accuracy_over_steps.png"
    _save_figure(fig, path)
    plt.close(fig)
    return path


def _write_training_history(
    results_dir: Path,
    histories: dict[str, list[dict]],
    base_accuracy: dict[str, float],
    benchmark_names: tuple[str, ...],
    filename_prefix: str = "",
) -> tuple[Path, Path]:
    combined_rows = []
    if all(benchmark in base_accuracy for benchmark in benchmark_names):
        combined_rows.append(
            {
                "Method": "Base",
                "Step": 0,
                **{
                    benchmark: base_accuracy[benchmark]
                    for benchmark in benchmark_names
                },
            }
        )
    for method, rows in histories.items():
        for row in rows:
            row_benchmarks = row.get("benchmarks", {})
            combined_rows.append(
                {
                    "Method": method,
                    "Step": int(row["step"]),
                    **{
                        benchmark: float(row["benchmarks"][benchmark]["accuracy"])
                        for benchmark in benchmark_names
                        if benchmark in row_benchmarks
                    },
                }
            )
    csv_path = results_dir / f"{filename_prefix}training_eval_history.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("Method", "Step", *benchmark_names)
        )
        writer.writeheader()
        writer.writerows(combined_rows)
    json_path = results_dir / f"{filename_prefix}training_eval_history.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {"base_accuracy": base_accuracy, "histories": histories},
            handle,
            indent=2,
            ensure_ascii=False,
        )
        handle.write("\n")
    return csv_path, json_path


def plot_training_progress(
    results_dir: str | Path,
    ta_output: str | Path | None = None,
    rac_output: str | Path | None = None,
    smoothing_window: int = 10,
    plot_name: str | None = None,
    _plots_dir: Path | None = None,
    method: str = "both",
    opd_output: str | Path | None = None,
    methods: list[str] | tuple[str, ...] | None = None,
    pgt_output: str | Path | None = None,
    cmt_output: str | Path | None = None,
    grpo_output: str | Path | None = None,
    iw_output: str | Path | None = None,
):
    """Plot the configured evaluation metric for any selected methods."""
    results_dir = Path(results_dir).resolve()
    plots_dir = _plots_dir or _plot_directory(results_dir, plot_name)
    selected_methods = _normalize_progress_methods(method, methods)
    outputs = {
        "opd": opd_output,
        "ta": ta_output,
        "rac": rac_output,
        "pgt": pgt_output,
        "cmt": cmt_output,
        "grpo": grpo_output,
        "iw": iw_output,
    }
    histories = {
        _PROGRESS_METHODS[item]["label"]: _read_eval_history(outputs[item], item)
        for item in selected_methods
    }
    benchmark_names = _shared_history_benchmarks(histories)
    metric_name = _history_metric_name(histories)

    base_accuracy: dict[str, float] = {}
    for benchmark in benchmark_names:
        candidates = []
        for rows in histories.values():
            value = _step_zero_accuracy(rows, benchmark)
            if value is not None:
                candidates.append(value)
        if candidates:
            spread = max(candidates) - min(candidates)
            # Competition-MATH and MATH-500 are the comparable base checks.
            # AIME evaluations are intentionally not used as a gate because
            # their small/problem-specific sample counts make Step-0 noise
            # substantially less informative for this consistency check.
            if (
                benchmark in _STEP_ZERO_BASE_CHECK_BENCHMARKS
                and len(candidates) > 1
                and spread > _STEP_ZERO_ACCURACY_TOLERANCE
            ):
                raise ValueError(
                    f"Step-0 base accuracy differs between methods for "
                    f"{benchmark} by {spread:.3%}, exceeding the "
                    f"{_STEP_ZERO_ACCURACY_TOLERANCE:.1%} tolerance: {candidates}"
                )
            # A small discrepancy is expected when each run evaluates its
            # initial checkpoint independently.  Use the mean as the shared
            # reference line rather than privileging whichever method happens
            # to appear first; each method's own step-0 point remains visible
            # in its plotted series and in the exported history.
            base_accuracy[benchmark] = float(np.mean(candidates))

    histories_for_plot, aligned_bases = _align_opd_cmt_for_plot(
        histories, benchmark_names
    )
    # When the asymmetric OPD/CMT correction applies, the Base guide line must
    # use the same common target as the aligned Step-0 points.
    base_accuracy.update(aligned_bases)

    if len(selected_methods) == 1:
        selected_method = selected_methods[0]
        rows = next(iter(histories_for_plot.values()))
        progress_path = _plot_single_method_accuracy(
            plots_dir, selected_method, rows, metric_name, benchmark_names
        )
        prefix = f"{_PROGRESS_METHODS[selected_method]['slug']}_"
        history_csv, history_json = _write_training_history(
            results_dir,
            histories_for_plot,
            base_accuracy,
            benchmark_names,
            filename_prefix=prefix,
        )
        return {
            "method": _PROGRESS_METHODS[selected_method]["label"],
            "metric": metric_name,
            "accuracy_over_steps": str(progress_path),
            "history_csv": str(history_csv),
            "history_json": str(history_json),
        }

    fig, axes = plt.subplots(
        1,
        len(benchmark_names),
        figsize=(5 * len(benchmark_names), 4.8),
        # Each benchmark gets its own readable accuracy range.  Sharing a
        # 0--100% axis across, for example, MATH and AIME would let the lower
        # scoring benchmark flatten the curves of the other one.
        sharey=False,
    )
    axes = np.atleast_1d(axes)
    colors = {
        spec["label"]: spec["color"]
        for item, spec in _PROGRESS_METHODS.items()
        if item in selected_methods
    }
    for axis, benchmark in zip(axes, benchmark_names):
        maximum_step = 0
        benchmark_accuracy_values: list[float] = []
        benchmark_steps = sorted(
            {
                int(row["step"])
                for rows in histories_for_plot.values()
                for row in rows
            }
        )
        for method, rows in histories_for_plot.items():
            available = {
                int(row["step"]): row
                for row in rows
                if benchmark in row.get("benchmarks", {})
            }
            if not available:
                continue
            steps = benchmark_steps
            values = [
                (
                    float(available[step]["benchmarks"][benchmark]["accuracy"])
                    if step in available
                    else float("nan")
                )
                for step in steps
            ]
            finite_values = [value for value in values if np.isfinite(value)]
            benchmark_accuracy_values.extend(finite_values)
            maximum_step = max(maximum_step, max(steps))
            axis.plot(
                steps,
                values,
                color=colors[method],
                marker="o",
                linewidth=2,
                label=method,
            )
            for step, value in zip(steps, values):
                if step == 0 or not np.isfinite(value):
                    continue
                axis.annotate(
                    f"{value:.3f}",
                    (step, value),
                    textcoords="offset points",
                    xytext=(0, 7),
                    ha="center",
                    fontsize=8,
                    color=colors[method],
                )
        if benchmark in base_accuracy:
            benchmark_accuracy_values.append(base_accuracy[benchmark])
            axis.hlines(
                base_accuracy[benchmark],
                0,
                maximum_step,
                colors="black",
                linestyles="--",
                linewidth=1.5,
                label="Base student",
            )
        axis.set_title(benchmark)
        axis.set_xlabel("Optimizer step")
        axis.set_ylim(*_accuracy_ylim(benchmark_accuracy_values))
        axis.grid(alpha=0.25)
    axes[0].set_ylabel(metric_name)
    legend_by_label = {}
    for axis in axes:
        handles, labels = axis.get_legend_handles_labels()
        legend_by_label.update(zip(labels, handles))
    fig.legend(
        list(legend_by_label.values()),
        list(legend_by_label),
        loc="upper center",
        ncol=len(selected_methods) + 1,
        frameon=False,
    )
    compared_names = ", ".join(
        _PROGRESS_METHODS[item]["label"] for item in selected_methods
    )
    fig.suptitle(f"Evaluation {metric_name} during {compared_names} training", y=1.02)
    fig.tight_layout()
    progress_path = plots_dir / "accuracy_over_steps.png"
    _save_figure(fig, progress_path)
    plt.close(fig)

    history_csv, history_json = _write_training_history(
        results_dir, histories_for_plot, base_accuracy, benchmark_names
    )
    selected_outputs = {
        item: outputs[item] for item in selected_methods if outputs[item] is not None
    }
    loss_path = _plot_loss_comparison(plots_dir, selected_outputs, smoothing_window)
    token_plots = _plot_token_score_distributions(plots_dir, selected_outputs)
    return {
        "methods": [_PROGRESS_METHODS[item]["label"] for item in selected_methods],
        "metric": metric_name,
        "accuracy_over_steps": str(progress_path),
        "loss": str(loss_path),
        "history_csv": str(history_csv),
        "history_json": str(history_json),
        **token_plots,
    }


def plot_results(
    results_dir: str | Path,
    ta_output: str | Path,
    rac_output: str | Path | None = None,
    smoothing_window: int = 10,
    plot_name: str | None = None,
    opd_output: str | Path | None = None,
    pgt_output: str | Path | None = None,
    cmt_output: str | Path | None = None,
    grpo_output: str | Path | None = None,
    iw_output: str | Path | None = None,
):
    results_dir = Path(results_dir).resolve()
    plots_dir = _plot_directory(results_dir, plot_name)
    comparison_path = results_dir / "comparison.csv"
    with comparison_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"comparison.csv is empty: {comparison_path}")
    benchmark_names = tuple(
        benchmark for benchmark in BENCHMARK_ORDER if benchmark in rows[0]
    )
    if not benchmark_names:
        raise ValueError(
            f"comparison.csv contains none of the supported benchmarks: {BENCHMARK_ORDER}"
        )
    for row in rows:
        missing = [name for name in benchmark_names if row.get(name) in (None, "")]
        if missing:
            raise ValueError(
                f"comparison.csv row {row.get('Method')} is missing: {missing}"
            )
    by_method = {row["Method"]: row for row in rows}
    plotted_models = tuple(by_method)
    expected_models = tuple(item for item in MODEL_ORDER if item in by_method)
    if (
        not plotted_models
        or plotted_models[0] != "Base"
        or plotted_models != expected_models
    ):
        raise ValueError(
            f"comparison.csv must contain Base followed by an ordered subset of "
            f"{MODEL_ORDER[1:]}; got {plotted_models}"
        )

    metric_name = "avg@8"
    comparison_json = results_dir / "comparison.json"
    if comparison_json.is_file():
        comparison_payload = json.loads(comparison_json.read_text(encoding="utf-8"))
        configured_metrics = {
            str(detail.get("parameters", {}).get("metric"))
            for detail in comparison_payload.get("details", {}).values()
            if detail.get("parameters", {}).get("metric")
        }
        if len(configured_metrics) > 1:
            raise ValueError(
                "Cannot plot final evaluations with different metrics: "
                f"{sorted(configured_metrics)}"
            )
        if configured_metrics:
            metric_name = configured_metrics.pop()

    x = np.arange(len(benchmark_names))
    width = min(0.8 / len(plotted_models), 0.24)
    fig, axis = plt.subplots(figsize=(max(9, 2.4 * len(benchmark_names)), 5.5))
    center = (len(plotted_models) - 1) / 2
    all_accuracy_values: list[float] = []
    for offset, method in enumerate(plotted_models):
        values = [float(by_method[method][benchmark]) for benchmark in benchmark_names]
        all_accuracy_values.extend(values)
        bars = axis.bar(x + (offset - center) * width, values, width, label=method)
        axis.bar_label(
            bars, labels=[f"{value:.3f}" for value in values], padding=3, fontsize=9
        )
    axis.set_xticks(x, benchmark_names)
    axis.set_ylim(*_accuracy_ylim(all_accuracy_values))
    axis.set_ylabel(metric_name)
    axis.set_title("Final evaluation: " + " vs ".join(plotted_models))
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    accuracy_path = plots_dir / "accuracy_comparison.png"
    _save_figure(fig, accuracy_path)
    plt.close(fig)

    training_outputs: dict[str, str | Path] = {"ta": ta_output}
    if rac_output is not None:
        training_outputs["rac"] = rac_output
    if pgt_output is not None:
        training_outputs["pgt"] = pgt_output
    if cmt_output is not None:
        training_outputs["cmt"] = cmt_output
    if grpo_output is not None:
        training_outputs["grpo"] = grpo_output
    if iw_output is not None:
        training_outputs["iw"] = iw_output
    if opd_output is not None:
        training_outputs = {"opd": opd_output, **training_outputs}
    loss_path = _plot_loss_comparison(plots_dir, training_outputs, smoothing_window)
    result = {
        "metric": metric_name,
        "accuracy": str(accuracy_path),
        "loss": str(loss_path),
    }
    opd_history = (
        Path(opd_output).resolve() / "eval_history.jsonl"
        if opd_output is not None
        else None
    )
    ta_history = Path(ta_output).resolve() / "eval_history.jsonl"
    rac_history = (
        Path(rac_output).resolve() / "eval_history.jsonl"
        if rac_output is not None
        else None
    )
    pgt_history = (
        Path(pgt_output).resolve() / "eval_history.jsonl"
        if pgt_output is not None
        else None
    )
    cmt_history = (
        Path(cmt_output).resolve() / "eval_history.jsonl"
        if cmt_output is not None
        else None
    )
    grpo_history = (
        Path(grpo_output).resolve() / "eval_history.jsonl"
        if grpo_output is not None
        else None
    )
    iw_history = (
        Path(iw_output).resolve() / "eval_history.jsonl"
        if iw_output is not None
        else None
    )
    if ta_history.is_file():
        progress_methods = ["ta"]
        if rac_history is not None and rac_history.is_file():
            progress_methods.append("rac")
        if opd_history is not None and opd_history.is_file():
            progress_methods.insert(0, "opd")
        if pgt_history is not None and pgt_history.is_file():
            progress_methods.append("pgt")
        if cmt_history is not None and cmt_history.is_file():
            progress_methods.append("cmt")
        if grpo_history is not None and grpo_history.is_file():
            progress_methods.append("grpo")
        if iw_history is not None and iw_history.is_file():
            progress_methods.append("iw")
        result.update(
            plot_training_progress(
                results_dir,
                ta_output,
                rac_output,
                smoothing_window,
                plot_name=plot_name,
                _plots_dir=plots_dir,
                opd_output=opd_output,
                pgt_output=pgt_output,
                cmt_output=cmt_output,
                grpo_output=grpo_output,
                iw_output=iw_output,
                methods=progress_methods,
            )
        )
    return result
