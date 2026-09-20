#!/usr/bin/env python3
"""Write a small reproducibility record before an ablation run starts."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    out = Path(sys.argv[1]).resolve()
    arm = sys.argv[2]
    repo_root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except Exception:
        commit = "unavailable"
    payload = {
        "arm": arm,
        "seed": int(os.environ.get("SEED", "42")),
        "rollout_seed": int(os.environ.get("ROLLOUT_SEED", "42")),
        "top_k": int(os.environ.get("TOP_K", "16")),
        "cmt_allocation_kl": float(os.environ.get("CMT_ALLOCATION_KL", "0.5")),
        "cmt_gamma": float(os.environ.get("CMT_GAMMA", "1.0")),
        "cmt_successor_lambda": float(os.environ.get("CMT_SUCCESSOR_LAMBDA", "1.0")),
        "learning_rate": float(os.environ.get("LR", os.environ.get("LEARNING_RATE", "1e-6"))),
        "train_max_new_tokens": int(os.environ.get("MAX_RESPONSE_LEN", "4096")),
        "eval_max_new_tokens": int(os.environ.get("TRAIN_EVAL_MAX_NEW_TOKENS", "7168")),
        "eval_seed": int(os.environ.get("TRAIN_EVAL_SEED", "1234")),
        "eval_benchmarks": [
            item.strip()
            for item in os.environ.get(
                "TRAIN_EVAL_BENCHMARKS",
                "Competition-MATH,MATH-500,AIME24,AIME25,GPQA-Diamond,AMC23",
            ).replace(",", " ").split()
            if item.strip()
        ],
        "rollout_temperature": float(os.environ.get("ROLLOUT_TEMPERATURE", "1.0")),
        "rollout_top_p": float(os.environ.get("ROLLOUT_TOP_P", "1.0")),
        "student_model": os.environ.get("STUDENT_MODEL", ""),
        "teacher_model": os.environ.get("TEACHER_MODEL", ""),
        "train_dataset": os.environ.get("TRAIN_DATASET", ""),
        "git_commit": commit,
        "command": os.environ.get("ABLATION_COMMAND", ""),
    }
    out.mkdir(parents=True, exist_ok=True)
    destination = out / "ablation_spec.json"
    # A resume must retain the original run identity and tuned values.
    if not destination.exists():
        destination.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
