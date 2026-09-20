from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from .autotune import run_batch_autotune
from .config import apply_overrides, load_with_overlays, resolve_runtime_paths, save_config
from .evaluation import (
    aggregate_evaluations,
    configured_benchmark_names,
    ensure_extended_benchmark_specs,
    evaluate_suite,
)
from .evaluation_history import record_checkpoint_evaluation
from .plotting import (
    plot_cmt_score_distributions,
    plot_results,
    plot_training_progress,
)
from .preflight import run_preflight
from .trainer import run_training


def _configured(args):
    return resolve_runtime_paths(
        apply_overrides(
            load_with_overlays(args.config, getattr(args, "overlay", [])),
            getattr(args, "overrides", []),
        )
    )


def _visible_gpu_count() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible.strip():
        return len([item for item in visible.split(",") if item.strip()])
    try:
        import torch

        return int(torch.cuda.device_count())
    except Exception:
        return 1


def _infer_checkpoint_step(model_path: str | Path, run_output: Path) -> int:
    """Infer the optimizer step represented by a saved checkpoint directory."""
    model_path = Path(model_path).expanduser().resolve()
    name = model_path.name
    if name.startswith("checkpoint-"):
        suffix = name.removeprefix("checkpoint-")
        if suffix.isdigit():
            return int(suffix)
    if name != "final":
        raise ValueError(
            "Cannot record standalone evaluation history for checkpoint path "
            f"{model_path}: expected a final/ checkpoint-N directory or pass "
            "--history-step explicitly"
        )

    latest_path = run_output / "latest.json"
    if latest_path.is_file():
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        if str(latest.get("checkpoint", "")).rstrip("/") == "final":
            if latest.get("step") is not None:
                return int(latest["step"])
    summary_path = run_output / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("steps") is not None:
            return int(summary["steps"])
    metrics_path = run_output / "metrics.jsonl"
    if metrics_path.is_file():
        last_step = None
        with metrics_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    last_step = json.loads(line).get("step")
        if last_step is not None:
            return int(last_step)
    raise ValueError(
        f"Cannot infer the optimizer step represented by {model_path}; "
        "run output needs latest.json, summary.json, or metrics.jsonl"
    )


def _infer_max_steps(run_output: Path, step: int) -> int:
    summary_path = run_output / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("steps") is not None:
            return max(int(summary["steps"]), step)
    return step


def _default_history_method(name: str, config: dict) -> str:
    configured = str(config.get("experiment", {}).get("method", "")).strip().lower()
    if configured and configured != "base":
        return configured
    return {
        "OPD": "opd",
        "TA-OPD": "ta",
        "RAC": "rac",
        "PGT": "pgt",
        "CMT-OPD": "cmt",
        "IW-OPD": "iw",
    }.get(name, name.lower().replace("-", "_"))


def _record_standalone_history(
    args,
    config: dict,
    suite: dict,
    elapsed: float,
) -> dict:
    run_output = getattr(args, "history_run_output", None)
    if not run_output:
        return suite
    run_output = Path(run_output).expanduser().resolve()
    step = (
        int(args.history_step)
        if getattr(args, "history_step", None) is not None
        else _infer_checkpoint_step(args.model, run_output)
    )
    max_steps = (
        int(args.history_max_steps)
        if getattr(args, "history_max_steps", None) is not None
        else _infer_max_steps(run_output, step)
    )
    artifacts = record_checkpoint_evaluation(
        run_output,
        suite,
        method=getattr(args, "history_method", None)
        or _default_history_method(args.name, config),
        step=step,
        max_steps=max_steps,
        details_path=Path(args.output).expanduser().resolve() / "summary.json",
        evaluation_time=elapsed,
    )
    suite["recorded_history"] = artifacts
    return suite


def _evaluate_checkpoint(args) -> dict:
    config = ensure_extended_benchmark_specs(_configured(args))
    if getattr(args, "benchmarks", None):
        # A partial re-evaluation (for example only GPQA-Diamond + AMC23)
        # changes the requested suite without mutating the resolved training
        # configuration on disk.
        config.setdefault("evaluation", {})["benchmark_names"] = list(args.benchmarks)
    started = time.perf_counter()
    if str(config.get("evaluation", {}).get("backend", "hf")).lower() != "vllm":
        suite = evaluate_suite(args.name, args.model, config, args.output)
    else:
        requested_world = int(os.environ.get("EVAL_WORLD_SIZE", _visible_gpu_count()))
        if requested_world <= 1:
            suite = evaluate_suite(args.name, args.model, config, args.output)
        else:
            from .vllm_evaluation import evaluate_vllm_distributed

            output = Path(args.output).expanduser().resolve()
            output.mkdir(parents=True, exist_ok=True)
            resolved = output / ".resolved_eval_config.yaml"
            save_config(config, resolved)
            evaluation = config["evaluation"]
            settings = {
                "backend": "vllm",
                "temperature": evaluation.get("temperature", 0.7),
                "top_p": evaluation.get("top_p", 0.95),
                "num_responses": evaluation.get("num_responses", 8),
                "metric": evaluation.get("metric"),
                "max_new_tokens": evaluation.get("max_new_tokens", 2048),
                "limit": evaluation.get("limit"),
                "benchmark_names": list(configured_benchmark_names(config)),
                "vllm": evaluation.get("vllm", {}),
            }
            try:
                suite = evaluate_vllm_distributed(
                    args.name,
                    args.model,
                    config,
                    output,
                    settings,
                    resolved,
                    world_size=requested_world,
                )
            finally:
                resolved.unlink(missing_ok=True)
    return _record_standalone_history(args, config, suite, time.perf_counter() - started)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone B200 OPD, TA-OPD, CMT-OPD, GRPO, IW-OPD, and legacy baselines"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--overlay", action="append", default=[])
    train.add_argument(
        "--set", action="append", default=[], dest="overrides", metavar="KEY=VALUE"
    )

    preflight = commands.add_parser("preflight")
    preflight.add_argument("--config", required=True)
    preflight.add_argument("--overlay", action="append", default=[])
    preflight.add_argument(
        "--set", action="append", default=[], dest="overrides", metavar="KEY=VALUE"
    )
    preflight.add_argument("--output", required=True)

    tune = commands.add_parser("autotune-batch")
    tune.add_argument("--opd-config")
    tune.add_argument("--ta-config", required=True)
    tune.add_argument("--rac-config", required=True)
    tune.add_argument("--output", required=True)
    tune.add_argument("--generated-config", required=True)
    tune.add_argument("--candidates", nargs="+", type=int)
    tune.add_argument(
        "--set", action="append", default=[], dest="overrides", metavar="KEY=VALUE"
    )

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--overlay", action="append", default=[])
    evaluate.add_argument(
        "--set", action="append", default=[], dest="overrides", metavar="KEY=VALUE"
    )
    evaluate.add_argument(
        "--name",
        required=True,
        choices=("Base", "OPD", "TA-OPD", "RAC", "PGT", "CMT-OPD", "GRPO", "IW-OPD"),
    )
    evaluate.add_argument("--model", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument(
        "--benchmarks",
        nargs="+",
        help=(
            "Optional canonical benchmark subset. Useful for partial re-evaluation; "
            "the selected results are merged into the existing history."
        ),
    )
    evaluate.add_argument(
        "--history-run-output",
        help=(
            "Method run output directory whose eval_history.jsonl should receive "
            "this standalone checkpoint result"
        ),
    )
    evaluate.add_argument(
        "--history-method",
        help="Method slug for the history row (for example opd, ta, rac, or cmt)",
    )
    evaluate.add_argument(
        "--history-step",
        type=int,
        help="Optimizer step to record; inferred from checkpoint name when omitted",
    )
    evaluate.add_argument(
        "--history-max-steps",
        type=int,
        help="Total training steps stored in the history row; inferred when omitted",
    )

    aggregate = commands.add_parser("aggregate-eval")
    aggregate.add_argument("--base-dir", required=True)
    aggregate.add_argument("--opd-dir")
    aggregate.add_argument("--ta-dir", required=True)
    aggregate.add_argument("--rac-dir")
    aggregate.add_argument("--pgt-dir")
    aggregate.add_argument("--cmt-dir")
    aggregate.add_argument("--grpo-dir")
    aggregate.add_argument("--iw-dir")
    aggregate.add_argument("--output", required=True)

    plot = commands.add_parser("plot")
    plot.add_argument("--results", required=True)
    plot.add_argument("--opd-output")
    plot.add_argument("--ta-output", required=True)
    plot.add_argument("--rac-output")
    plot.add_argument("--pgt-output")
    plot.add_argument("--cmt-output")
    plot.add_argument("--grpo-output")
    plot.add_argument("--iw-output")
    plot.add_argument("--smoothing-window", type=int, default=10)
    plot.add_argument("--plot-name")

    progress_plot = commands.add_parser("plot-training-progress")
    progress_plot.add_argument("--results", required=True)
    progress_plot.add_argument(
        "--method",
        default="both",
        choices=(
            "all",
            "both",
            "opd",
            "pure-opd",
            "ta",
            "ta-opd",
            "rac",
            "bellman-rac",
            "pgt",
            "cmt",
            "grpo",
            "iw",
            "iw-opd",
        ),
        help="Legacy single selector: all, both=TA+RAC, or one method",
    )
    progress_plot.add_argument(
        "--methods",
        nargs="+",
        choices=(
            "opd", "pure-opd", "ta", "ta-opd", "rac", "bellman-rac", "pgt", "cmt", "grpo", "iw", "iw-opd"
        ),
        help="One or more methods to plot in the requested order",
    )
    progress_plot.add_argument("--opd-output")
    progress_plot.add_argument("--ta-output")
    progress_plot.add_argument("--rac-output")
    progress_plot.add_argument("--pgt-output")
    progress_plot.add_argument("--cmt-output")
    progress_plot.add_argument("--grpo-output")
    progress_plot.add_argument("--iw-output")
    progress_plot.add_argument("--smoothing-window", type=int, default=10)
    progress_plot.add_argument("--plot-name")

    cmt_scores = commands.add_parser(
        "plot-cmt-scores",
        help="Plot CMT token-score histograms and learning-value quantiles",
    )
    cmt_scores.add_argument(
        "--cmt-output",
        required=True,
        help="CMT training output containing token_score_stats/",
    )
    cmt_scores.add_argument(
        "--run-name",
        help="Run label used in output directory and figure filenames",
    )
    cmt_scores.add_argument(
        "--output-dir",
        help="Optional base directory; a unique plots/<name> child is created",
    )
    cmt_scores.add_argument(
        "--plot-name",
        help="Optional plot directory name; an unused suffix is added on collision",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        run_training(_configured(args), command_line=sys.argv)
        return 0
    if args.command == "preflight":
        result = run_preflight(_configured(args), args.output)
    elif args.command == "autotune-batch":
        result = run_batch_autotune(
            args.ta_config,
            args.rac_config,
            args.output,
            args.generated_config,
            args.candidates,
            opd_config=args.opd_config,
            overrides=args.overrides,
        )
    elif args.command == "evaluate":
        result = _evaluate_checkpoint(args)
    elif args.command == "aggregate-eval":
        model_dirs = {"Base": args.base_dir}
        if args.opd_dir:
            model_dirs["OPD"] = args.opd_dir
        model_dirs["TA-OPD"] = args.ta_dir
        if args.rac_dir:
            model_dirs["RAC"] = args.rac_dir
        if args.pgt_dir:
            model_dirs["PGT"] = args.pgt_dir
        if args.cmt_dir:
            model_dirs["CMT-OPD"] = args.cmt_dir
        if args.grpo_dir:
            model_dirs["GRPO"] = args.grpo_dir
        if args.iw_dir:
            model_dirs["IW-OPD"] = args.iw_dir
        result = aggregate_evaluations(
            model_dirs,
            args.output,
        )
    elif args.command == "plot":
        result = plot_results(
            args.results,
            args.ta_output,
            args.rac_output,
            args.smoothing_window,
            args.plot_name,
            opd_output=args.opd_output,
            pgt_output=args.pgt_output,
            cmt_output=args.cmt_output,
            grpo_output=args.grpo_output,
            iw_output=args.iw_output,
        )
    elif args.command == "plot-training-progress":
        result = plot_training_progress(
            args.results,
            args.ta_output,
            args.rac_output,
            args.smoothing_window,
            args.plot_name,
            method=args.method,
            opd_output=args.opd_output,
            methods=args.methods,
            pgt_output=args.pgt_output,
            cmt_output=args.cmt_output,
            grpo_output=args.grpo_output,
            iw_output=args.iw_output,
        )
    elif args.command == "plot-cmt-scores":
        result = plot_cmt_score_distributions(
            args.cmt_output,
            run_name=args.run_name,
            plot_name=args.plot_name,
            output_root=args.output_dir,
        )
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
