"""Small numerical helpers shared by online summaries and offline plots."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


PREDICTORS = ("g", "x", "d", "g_plus_x", "g_plus_d")


def finite_pairs(x: Iterable[float], y: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(list(x), dtype=np.float64)
    right = np.asarray(list(y), dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("Paired metrics must have equal lengths")
    mask = np.isfinite(left) & np.isfinite(right)
    return left[mask], right[mask]


def _ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks for ties, without requiring scipy."""
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def correlations(x: Iterable[float], y: Iterable[float]) -> dict[str, float | int]:
    left, right = finite_pairs(x, y)
    result: dict[str, float | int] = {"n": int(left.size), "pearson": math.nan, "spearman": math.nan}
    if left.size < 2:
        return result
    if np.std(left) > 0 and np.std(right) > 0:
        result["pearson"] = float(np.corrcoef(left, right)[0, 1])
    left_rank, right_rank = _ranks(left), _ranks(right)
    if np.std(left_rank) > 0 and np.std(right_rank) > 0:
        result["spearman"] = float(np.corrcoef(left_rank, right_rank)[0, 1])
    return result


def quantile_curve(rows: list[dict], score: str, outcome: str, bins: int = 10) -> list[dict]:
    """Equal-count bins on one score, retaining only paired finite outcomes."""
    pairs = [
        (float(row[score]), float(row[outcome]))
        for row in rows
        if row.get(score) is not None and row.get(outcome) is not None
    ]
    pairs = [(x, y) for x, y in pairs if math.isfinite(x) and math.isfinite(y)]
    if not pairs:
        return []
    pairs.sort(key=lambda pair: pair[0])
    result = []
    for index, positions in enumerate(np.array_split(np.arange(len(pairs)), min(bins, len(pairs)))):
        selected = [pairs[int(position)] for position in positions]
        result.append({
            "bin": index + 1,
            "n": len(selected),
            "score_mean": float(np.mean([item[0] for item in selected])),
            "outcome_mean": float(np.mean([item[1] for item in selected])),
            "outcome_se": (
                float(np.std([item[1] for item in selected], ddof=1) / np.sqrt(len(selected)))
                if len(selected) > 1 else math.nan
            ),
        })
    return result


def conditional_d_comparison(rows: list[dict], outcome: str, g_bins: int = 10) -> list[dict]:
    """Compare high/low D within equal-count g strata; no causal claim."""
    valid = [
        row for row in rows
        if all(row.get(key) is not None and math.isfinite(float(row[key])) for key in ("g", "d", outcome))
    ]
    valid.sort(key=lambda row: float(row["g"]))
    result = []
    if len(valid) < 8:
        return result
    for index, positions in enumerate(np.array_split(np.arange(len(valid)), min(g_bins, len(valid) // 8))):
        stratum = sorted((valid[int(position)] for position in positions), key=lambda row: float(row["d"]))
        quarter = len(stratum) // 4
        if quarter < 2:
            continue
        low, high = stratum[:quarter], stratum[-quarter:]
        result.append({
            "g_bin": index + 1,
            "n_low": len(low),
            "n_high": len(high),
            "g_low_mean": float(np.mean([row["g"] for row in low])),
            "g_high_mean": float(np.mean([row["g"] for row in high])),
            "d_low_mean": float(np.mean([row["d"] for row in low])),
            "d_high_mean": float(np.mean([row["d"] for row in high])),
            "outcome_low_mean": float(np.mean([row[outcome] for row in low])),
            "outcome_high_mean": float(np.mean([row[outcome] for row in high])),
        })
    return result
