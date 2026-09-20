#!/usr/bin/env python3
"""Plot final score or learning-curve AUC against an ablation parameter."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _save_pair(fig, output_dir: Path, stem: str) -> None:
    index = 0
    while True:
        suffix = "" if index == 0 else f"_{index:03d}"
        png = output_dir / f"{stem}{suffix}.png"
        pdf = output_dir / f"{stem}{suffix}.pdf"
        if not png.exists() and not pdf.exists():
            fig.savefig(png, dpi=180)
            fig.savefig(pdf)
            return
        index += 1


def _result(row, benchmark, metric):
    values = row.get("benchmarks", {})
    item = values.get(benchmark)
    if item is None:
        for name, candidate in values.items():
            if name.lower().replace("-", "") == benchmark.lower().replace("-", ""):
                item = candidate
                break
    return None if item is None else item.get(metric)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameter", choices=("epsilon", "gamma", "cmt_gamma", "top_k", "lr", "learning_rate"), required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--benchmark", default="MATH-500")
    parser.add_argument("--metric", default="accuracy")
    parser.add_argument("--aggregate", choices=("final", "auc"), default="final")
    parser.add_argument("--run-name", action="append", dest="run_names", help="Restrict the plot to these output directory names (repeatable)")
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    key = {"epsilon": "cmt_allocation_kl", "gamma": "cmt_gamma", "cmt_gamma": "cmt_gamma", "top_k": "top_k", "lr": "learning_rate", "learning_rate": "learning_rate"}[args.parameter]
    points = []
    for spec_path in sorted(args.input_root.rglob("ablation_spec.json")):
        if args.run_names and spec_path.parent.name not in set(args.run_names):
            continue
        try:
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            value = spec.get(key)
            if value is None:
                continue
            rows = [json.loads(line) for line in (spec_path.parent / "eval_history.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            curve = [(int(row.get("step", 0)), _result(row, args.benchmark, args.metric)) for row in rows]
            curve = sorted((s, float(v)) for s, v in curve if v is not None)
            if not curve:
                continue
            score = curve[-1][1] if args.aggregate == "final" else float(np.trapz([v for _, v in curve], [s for s, _ in curve]))
            points.append((float(value), score))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    if not points:
        raise SystemExit("No matching ablation runs found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    points.sort()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x, y = zip(*points)
    ax.plot(x, y, "o-", linewidth=2)
    ax.set_xlabel(args.parameter)
    ax.set_ylabel(f"{args.aggregate} {args.metric} ({args.benchmark})")
    ax.set_title(f"CMT hyperparameter analysis: {args.parameter}")
    if args.parameter in {"lr", "learning_rate"}:
        ax.set_xscale("log")
    ax.grid(alpha=.25)
    fig.tight_layout()
    stem = f"hparam_{args.parameter}_{args.aggregate}"
    _save_pair(fig, args.output_dir, stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
