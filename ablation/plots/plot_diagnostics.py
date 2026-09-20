#!/usr/bin/env python3
"""Plot detached CMT score/weight summaries emitted by the trainer."""
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--run-name", action="append", dest="run_names")
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = {"gain": [], "successor_excess": [], "sequential_gain": [], "learning_value": [], "w": []}
    for path in args.input_root.rglob("token_score_stats/step-*.json"):
        if args.run_names and path.parent.parent.name not in set(args.run_names):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if args.step is not None and int(payload.get("step", -1)) != args.step:
            continue
        for key in samples:
            item = payload.get("scores", {}).get(key, {})
            samples[key].extend(item.get("sample", []))
    available = [(key, values) for key, values in samples.items() if values]
    if not available:
        raise SystemExit("No token_score_stats samples found")
    fig, axes = plt.subplots(1, len(available), figsize=(4 * len(available), 3.5), squeeze=False)
    for axis, (key, values) in zip(axes[0], available):
        axis.hist(values, bins=40, alpha=.8)
        axis.set_title(key)
        axis.set_xlabel("value")
        axis.set_ylabel("count")
        axis.grid(alpha=.2)
    fig.tight_layout()
    _save_pair(fig, args.output_dir, "allocation_diagnostics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
