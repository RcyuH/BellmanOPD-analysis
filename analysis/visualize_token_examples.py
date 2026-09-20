"""Mine CMT token examples and render a self-contained qualitative HTML report."""

from __future__ import annotations

import argparse
import html
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


PREDICTOR_LABELS = {
    "g": "g",
    "x": "X",
    "d": "D",
    "g_plus_x": "g + X",
    "g_plus_d": "g + D",
}


def _jsonl_rows(directory: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not directory.exists():
        return rows
    for path in sorted(directory.glob("step-*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def find_analysis_root(input_path: Path) -> Path:
    candidates = (input_path, input_path / "analysis")
    for candidate in candidates:
        if (candidate / "manifest.json").exists() or (candidate / "progress").exists():
            return candidate
    return input_path


def load_records(input_path: Path) -> tuple[list[dict[str, Any]], Path, bool]:
    root = find_analysis_root(input_path)
    qualitative = _jsonl_rows(root / "qualitative")
    if qualitative:
        return qualitative, root, False
    progress = _jsonl_rows(root / "progress")
    tokens = _jsonl_rows(root / "tokens")
    token_lookup = {
        (
            row.get("scoring_step"), row.get("sample_id"),
            row.get("response_position"), row.get("rank"),
        ): row
        for row in tokens
        if row.get("token_id") is not None
    }
    for row in progress:
        token_row = token_lookup.get((
            row.get("scoring_step"), row.get("sample_id"),
            row.get("response_position"), row.get("rank"),
        ))
        if token_row is not None:
            row.setdefault("token_id", token_row.get("token_id"))
        row.setdefault("current_token", {
            "token_id": row.get("token_id"),
            "decoded_text": None,
            "token_piece": None,
        })
        row["legacy_record"] = True
    return progress, root, True


def decode_legacy_token_ids(rows: list[dict[str, Any]], tokenizer_path: Path | str) -> int:
    """Decode joinable legacy token IDs with the same local HF tokenizer family."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    decoded = 0
    special_ids = set(tokenizer.all_special_ids or [])
    for row in rows:
        token = row.get("current_token") or {}
        token_id = token.get("token_id", row.get("token_id"))
        if token_id is None or token.get("decoded_text") is not None:
            continue
        token["token_id"] = int(token_id)
        token["decoded_text"] = tokenizer.decode(
            [int(token_id)], skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        token["token_piece"] = tokenizer.convert_ids_to_tokens(int(token_id))
        token["is_special_token"] = int(token_id) in special_ids
        row["current_token"] = token
        decoded += 1
    return decoded


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    index = (len(ordered) - 1) * fraction
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - index) + ordered[high] * (index - low)


def _future_key(rows: list[dict[str, Any]]) -> str | None:
    for key in ("delta_future_kl_h8", "delta_future_kl_h16", "delta_future_kl_h4", "delta_future_kl"):
        if any(_finite(row.get(key)) for row in rows):
            return key
    return None


def _unique(rows: Iterable[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    result = []
    seen = set()
    for row in rows:
        identity = (row.get("optimizer_step"), row.get("sample_id"), row.get("response_position"))
        if identity in seen:
            continue
        seen.add(identity)
        result.append(row)
        if len(result) >= limit:
            break
    return result


def _percentile_ranks(rows: list[dict[str, Any]], key: str) -> dict[int, float]:
    available = [(index, float(row[key])) for index, row in enumerate(rows) if _finite(row.get(key))]
    ordered = sorted(available, key=lambda item: item[1])
    denominator = max(len(ordered) - 1, 1)
    return {index: rank / denominator for rank, (index, _) in enumerate(ordered)}


def _similar_pairs(
    rows: list[dict[str, Any]],
    *,
    include_x: bool,
    relative_tolerance: float,
    absolute_tolerance: float,
    minimum_d_gap: float,
    limit: int,
) -> list[dict[str, Any]]:
    if len(rows) < 2:
        return []
    g_values = [float(row["g"]) for row in rows if _finite(row.get("g"))]
    x_values = [float(row["x"]) for row in rows if _finite(row.get("x"))]
    g_scale = max(_quantile(g_values, .9) - _quantile(g_values, .1), 1e-12)
    x_scale = max(_quantile(x_values, .9) - _quantile(x_values, .1), 1e-12) if x_values else 1.0
    g_tolerance = max(absolute_tolerance, relative_tolerance * g_scale, 1e-12)
    x_tolerance = max(absolute_tolerance, relative_tolerance * x_scale, 1e-12)
    # Only D extremes in each small score cell can form a maximally separated
    # pair. This keeps mining linear in the number of records instead of O(n²).
    cells: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not all(_finite(row.get(key)) for key in ("g", "d")):
            continue
        if include_x and not _finite(row.get("x")):
            continue
        cell = (
            math.floor(float(row["g"]) / g_tolerance),
            math.floor(float(row.get("x", 0.0)) / x_tolerance) if include_x else 0,
        )
        cells[cell].append(row)
    extremes: dict[tuple[int, int], list[dict[str, Any]]] = {}
    retain = max(2, limit)
    for cell, cell_rows in cells.items():
        ordered = sorted(cell_rows, key=lambda row: float(row["d"]))
        extremes[cell] = _unique(ordered[:retain] + ordered[-retain:], 2 * retain)
    pairs = []
    seen = set()
    x_offsets = (-1, 0, 1) if include_x else (0,)
    for cell, left_rows in extremes.items():
        for g_offset in (-1, 0, 1):
            for x_offset in x_offsets:
                neighbor = (cell[0] + g_offset, cell[1] + x_offset)
                for left in left_rows:
                    for right in extremes.get(neighbor, []):
                        identity = tuple(sorted((id(left), id(right))))
                        if identity in seen or left is right:
                            continue
                        seen.add(identity)
                        g_gap = abs(float(left["g"]) - float(right["g"]))
                        x_gap = abs(float(left.get("x", 0.0)) - float(right.get("x", 0.0)))
                        if g_gap > g_tolerance or (include_x and x_gap > x_tolerance):
                            continue
                        d_gap = abs(float(left["d"]) - float(right["d"]))
                        if d_gap < minimum_d_gap:
                            continue
                        pairs.append({
                            "left": left,
                            "right": right,
                            "g_gap": g_gap,
                            "x_gap": x_gap,
                            "d_gap": d_gap,
                        })
    return sorted(pairs, key=lambda pair: (-pair["d_gap"], pair["g_gap"]))[:limit]


def mine_examples(
    rows: list[dict[str, Any]],
    *,
    examples_per_category: int = 5,
    similar_g_relative_tolerance: float = .05,
    similar_g_absolute_tolerance: float = 1e-6,
    minimum_d_gap: float = 0.0,
) -> dict[str, Any]:
    if not rows:
        return {"categories": {}, "pairs": {}, "future_outcome": None}
    finite_g = [float(row["g"]) for row in rows if _finite(row.get("g"))]
    finite_d = [float(row["d"]) for row in rows if _finite(row.get("d"))]
    g_hi, g_mid = _quantile(finite_g, .8), _quantile(finite_g, .4)
    d_hi, d_lo = _quantile(finite_d, .8), _quantile(finite_d, .2)
    future = _future_key(rows)
    categories: dict[str, list[dict[str, Any]]] = {
        "highest_g": _unique(sorted(rows, key=lambda row: float(row.get("g", -math.inf)), reverse=True), examples_per_category),
        "highest_D": _unique(sorted(rows, key=lambda row: float(row.get("d", -math.inf)), reverse=True), examples_per_category),
        "high_g_low_D": _unique((row for row in sorted(rows, key=lambda row: float(row.get("g", 0)), reverse=True) if _finite(row.get("g")) and _finite(row.get("d")) and row["g"] >= g_hi and row["d"] <= d_lo), examples_per_category),
        "moderate_g_high_D": _unique((row for row in sorted(rows, key=lambda row: float(row.get("d", 0)), reverse=True) if _finite(row.get("g")) and _finite(row.get("d")) and g_mid <= row["g"] < g_hi and row["d"] >= d_hi), examples_per_category),
        "g_vs_gD_ranking_change": _unique(
            sorted(
                (row for row in rows if _finite(row.get("rank_shift_g_to_g_plus_d"))),
                key=lambda row: abs(float(row["rank_shift_g_to_g_plus_d"])),
                reverse=True,
            ),
            examples_per_category,
        ),
    }
    if future:
        with_future = [row for row in rows if _finite(row.get(future))]
        categories["largest_positive_downstream"] = _unique(sorted(with_future, key=lambda row: row[future], reverse=True), examples_per_category)
        categories["largest_negative_downstream"] = _unique(sorted(with_future, key=lambda row: row[future]), examples_per_category)
        categories["failure_cases"] = _unique(sorted((row for row in with_future if _finite(row.get("d")) and row["d"] * row[future] < 0), key=lambda row: abs(row["d"] * row[future]), reverse=True), examples_per_category)
        outcome_rank = _percentile_ranks(rows, future)
        gx_rank = _percentile_ranks(rows, "g_plus_x")
        gd_rank = _percentile_ranks(rows, "g_plus_d")
        gd_better = []
        for index, row in enumerate(rows):
            if index not in outcome_rank or index not in gx_rank or index not in gd_rank:
                continue
            gx_error = abs(gx_rank[index] - outcome_rank[index])
            gd_error = abs(gd_rank[index] - outcome_rank[index])
            if gd_error < gx_error:
                enriched = dict(row)
                enriched["gd_vs_gx_observed_rank_advantage"] = gx_error - gd_error
                enriched["observed_outcome_key"] = future
                gd_better.append(enriched)
        categories["gD_tracks_observed_better_than_gX"] = _unique(sorted(gd_better, key=lambda row: row["gd_vs_gx_observed_rank_advantage"], reverse=True), examples_per_category)
    pairs = {
        "similar_g_different_D": _similar_pairs(
            rows,
            include_x=False,
            relative_tolerance=similar_g_relative_tolerance,
            absolute_tolerance=similar_g_absolute_tolerance,
            minimum_d_gap=minimum_d_gap,
            limit=examples_per_category,
        ),
        "similar_gX_different_D": _similar_pairs(
            rows,
            include_x=True,
            relative_tolerance=similar_g_relative_tolerance,
            absolute_tolerance=similar_g_absolute_tolerance,
            minimum_d_gap=minimum_d_gap,
            limit=examples_per_category,
        ),
    }
    return {
        "categories": {name: selected for name, selected in categories.items() if selected},
        "pairs": {name: selected for name, selected in pairs.items() if selected},
        "future_outcome": future,
        "thresholds": {"g_high": g_hi, "g_moderate": g_mid, "D_high": d_hi, "D_low": d_lo},
    }


def _fmt(value: Any, digits: int = 5) -> str:
    if not _finite(value):
        return "n/a"
    return f"{float(value):.{digits}g}"


def _visible_token(token: dict[str, Any] | None) -> str:
    token = token or {}
    decoded = token.get("decoded_text")
    piece = token.get("token_piece")
    value = decoded if decoded is not None else piece
    if value is None:
        value = f"[id {token.get('token_id', '?')}]"
    value = str(value).replace(" ", "␠").replace("\n", "↵").replace("\t", "⇥")
    return html.escape(value)


def _badge(label: str, value: Any, css: str = "") -> str:
    return f'<span class="badge {css}"><b>{html.escape(label)}</b> {_fmt(value)}</span>'


def _direction_text(row: dict[str, Any]) -> str:
    local = row.get("delta_kl")
    preferred = row.get("teacher_preferred_probability_gain")
    future_key = next((key for key in ("delta_future_kl_h8", "delta_future_kl_h16", "delta_future_kl_h4", "delta_future_kl") if _finite(row.get(key))), None)
    statements = []
    if _finite(local):
        if local > 0:
            statements.append("fixed-support KL(p∥q) decreased locally")
        elif local < 0:
            statements.append("fixed-support KL(p∥q) increased locally")
        else:
            statements.append("fixed-support KL(p∥q) was unchanged locally")
    if _finite(preferred):
        verb = "increased" if preferred > 0 else "decreased" if preferred < 0 else "did not change"
        statements.append(f"student probability of the teacher-preferred stored candidate {verb}")
    if future_key:
        value = row[future_key]
        verb = "decreased" if value > 0 else "increased" if value < 0 else "was unchanged"
        statements.append(f"mean future fixed-support KL {verb} ({future_key})")
    return "; ".join(statements) + "." if statements else "Objective direction signals are unavailable for this record."


def _heatmap(row: dict[str, Any]) -> str:
    window = row.get("context_window") or []
    if not window:
        context = html.escape(str(row.get("context_window_text") or row.get("prefix_tail_text") or "Decoded context unavailable in this log."))
        return f'<div class="context unavailable">{context}</div>'
    tokens = []
    for token in window:
        attrs = " ".join(
            f'data-{key.replace("_", "-")}="{float(token.get(key, 0.0))}"'
            for key in PREDICTOR_LABELS
        )
        selected = " selected" if token.get("selected") else ""
        title = f"position {token.get('response_position')} · id {token.get('token_id')}"
        tokens.append(
            f'<span class="heat-token{selected}" {attrs} title="{html.escape(title)}">'
            f'{_visible_token(token)}<small>{token.get("response_position")}</small></span>'
        )
    return '<div class="heatmap">' + "".join(tokens) + "</div>"


def _candidate_table(row: dict[str, Any]) -> str:
    candidates = row.get("candidate_tokens") or []
    if not candidates:
        return '<p class="unavailable">Candidate distributions were not retained in this log.</p>'
    body = []
    for candidate in candidates:
        delta = candidate.get("delta_p")
        direction = "up" if _finite(delta) and delta > 0 else "down" if _finite(delta) and delta < 0 else ""
        flags = []
        if candidate.get("is_target_token"):
            flags.append("target")
        preferred = row.get("teacher_preferred_token") or {}
        if candidate.get("token_id") == preferred.get("token_id"):
            flags.append("teacher top")
        body.append(
            "<tr>"
            f'<td><code>{_visible_token(candidate)}</code><small> id {candidate.get("token_id")}</small></td>'
            f'<td>{html.escape(", ".join(flags))}</td>'
            f'<td>{_fmt(candidate.get("p_student_before"))}</td>'
            f'<td>{_fmt(candidate.get("p_student_after"))}</td>'
            f'<td class="{direction}">{_fmt(delta)}</td>'
            f'<td>{_fmt(candidate.get("logp_student_before"))}</td>'
            f'<td>{_fmt(candidate.get("logp_student_after"))}</td>'
            f'<td>{_fmt(candidate.get("p_teacher"))}</td>'
            f'<td>{candidate.get("rank_before", "n/a")}</td>'
            f'<td>{candidate.get("rank_after", "n/a")}</td>'
            f'<td>{candidate.get("delta_rank", "n/a")}</td>'
            "</tr>"
        )
    return (
        '<div class="table-wrap"><table><thead><tr><th>candidate</th><th>role</th>'
        '<th>p before</th><th>p after</th><th>Δp</th><th>logp before</th>'
        '<th>logp after</th><th>p teacher</th><th>rank before</th><th>rank after</th><th>Δrank</th>'
        '</tr></thead><tbody>' + "".join(body) + "</tbody></table></div>"
    )


def _top_k_panels(row: dict[str, Any]) -> str:
    panels = []
    for field, title, probability_key in (
        ("student_top_k_before", "Student before · full-vocab Top-K", "student_probability"),
        ("student_top_k_after", "Student after · full-vocab Top-K", "student_probability"),
        ("teacher_top_k", "Teacher · stored-support Top-K", "p_teacher"),
    ):
        items = row.get(field) or []
        if not items:
            continue
        content = "".join(
            f'<li><b>#{index}</b> <code>{_visible_token(item)}</code> '
            f'<span>p={_fmt(item.get(probability_key))}</span></li>'
            for index, item in enumerate(items, 1)
        )
        panels.append(f'<div><h5>{html.escape(title)}</h5><ol>{content}</ol></div>')
    return '<div class="ranking">' + "".join(panels) + "</div>" if panels else ""


def _ranking(row: dict[str, Any]) -> str:
    panels = []
    for key in ("g", "g_plus_x", "g_plus_d"):
        positions = row.get(f"top_positions_{key}") or []
        if not positions:
            continue
        items = "".join(
            f'<li><b>#{index}</b> t={item.get("response_position")} '
            f'<code>{_visible_token(item)}</code> <span>{_fmt(item.get("score"))}</span></li>'
            for index, item in enumerate(positions, 1)
        )
        panels.append(f'<div><h5>{PREDICTOR_LABELS[key]}</h5><ol>{items}</ol></div>')
    if not panels:
        return '<p class="unavailable">Sequence ranking summaries were not retained in this log.</p>'
    return '<div class="ranking">' + "".join(panels) + "</div>"


def _record_card(row: dict[str, Any], category: str, index: int) -> str:
    token = row.get("current_token") or {"token_id": row.get("token_id")}
    prefix = row.get("prefix_tail_text")
    reference = row.get("reference_text")
    horizons = "".join(
        _badge(f"Δfuture KL H={h}", row.get(f"delta_future_kl_h{h}"), "future")
        for h in (1, 4, 8, 16)
        if f"delta_future_kl_h{h}" in row
    )
    return f'''
    <article class="card" data-category="{html.escape(category)}">
      <header><div><span class="category">{html.escape(category)}</span>
      <h3>{html.escape(str(row.get("sample_id", "unknown sample")))} · t={row.get("response_position", "?")}</h3></div>
      <div class="token-focus"><span>selected token</span><code>{_visible_token(token)}</code><small>id {token.get("token_id", "?")}</small></div></header>
      <div class="badges">
        {_badge("g", row.get("g"))}{_badge("X", row.get("x"))}{_badge("D", row.get("d"))}
        {_badge("g+X", row.get("g_plus_x"))}{_badge("g+D", row.get("g_plus_d"))}
        {_badge("weight", row.get("training_weight"))}{_badge("loss", row.get("opd_ppo_loss_before"))}
      </div>
      <p class="meta">optimizer step {row.get("optimizer_step", "?")} · normalized position {_fmt(row.get("normalized_position"))} · sequence length {row.get("sequence_length", "?")}</p>
      <section><h4>Prefix tail</h4><pre>{html.escape(str(prefix)) if prefix is not None else "Decode unavailable for this record."}</pre></section>
      <section><h4>Token heatmap</h4>{_heatmap(row)}</section>
      <section class="signals"><h4>Observed direction of the whole minibatch update</h4>
        <div class="badges">{_badge("target logp gain", row.get("target_logprob_gain", row.get("delta_nll")), "local")}
        {_badge("Δlocal KL(p∥q)", row.get("delta_kl"), "local")}{horizons}</div>
        <p>{html.escape(_direction_text(row))}</p>
      </section>
      <details open><summary>Candidate probability movement</summary>{_top_k_panels(row)}{_candidate_table(row)}</details>
      <details><summary>Counterfactual top positions within this response</summary>{_ranking(row)}</details>
      <details><summary>Teacher / reference signal</summary><pre>{html.escape(str(reference)) if reference is not None else "Reference text unavailable; teacher signal is represented by probabilities above."}</pre></details>
    </article>'''


def _pair_card(pair: dict[str, Any], category: str, index: int) -> str:
    left, right = pair["left"], pair["right"]
    def compact(row: dict[str, Any], label: str) -> str:
        token = row.get("current_token") or {"token_id": row.get("token_id")}
        return f'''<div class="pair-side"><h4>{label}</h4><p><code>{_visible_token(token)}</code> at t={row.get("response_position")}</p>
        <p>{html.escape(str(row.get("context_window_text") or row.get("prefix_tail_text") or "decode unavailable"))}</p>
        <div class="badges">{_badge("g", row.get("g"))}{_badge("X", row.get("x"))}{_badge("D", row.get("d"))}{_badge("g+D", row.get("g_plus_d"))}</div>
        <p>rank g → g+D: {row.get("rank_g", "n/a")} → {row.get("rank_g_plus_d", "n/a")}</p>
        <p>{html.escape(_direction_text(row))}</p></div>'''
    return f'''<article class="card pair-card" data-category="{html.escape(category)}"><span class="category">{html.escape(category)}</span>
    <h3>Similar local score, different D · pair {index}</h3>
    <p class="meta">|Δg|={_fmt(pair.get("g_gap"))} · |ΔX|={_fmt(pair.get("x_gap"))} · |ΔD|={_fmt(pair.get("d_gap"))}</p>
    <div class="pair">{compact(left, "Example A")}{compact(right, "Example B")}</div></article>'''


CSS = r'''
:root { color-scheme: light; --ink:#17202a; --muted:#65707e; --paper:#f5f6f8; --card:#fff; --accent:#5b4bdb; }
* { box-sizing:border-box } body { margin:0; font:14px/1.5 ui-sans-serif,system-ui,-apple-system,sans-serif; color:var(--ink); background:var(--paper) }
.hero { background:linear-gradient(130deg,#161b33,#333065 65%,#5545ae); color:white; padding:38px max(24px,calc((100vw - 1250px)/2)) }
.hero h1 { margin:0 0 8px; font-size:30px }.hero p { max-width:950px; color:#d8d8ef }
.controls { position:sticky; top:0; z-index:5; display:flex; gap:20px; align-items:center; padding:12px max(24px,calc((100vw - 1250px)/2)); background:#ffffffed; backdrop-filter:blur(8px); border-bottom:1px solid #ddd }
main { max-width:1250px; margin:22px auto; padding:0 22px 60px }.card { background:var(--card); margin:18px 0; padding:20px; border:1px solid #dfe2e7; border-radius:14px; box-shadow:0 5px 18px #1820330c }
.card header { display:flex; justify-content:space-between; gap:20px }.card h3 { margin:5px 0 0 }.category { text-transform:uppercase; letter-spacing:.08em; font-size:11px; color:var(--accent); font-weight:750 }.token-focus { display:grid; text-align:right }.token-focus code { font-size:19px }.meta,.unavailable { color:var(--muted) }.badges { display:flex; gap:7px; flex-wrap:wrap; margin:12px 0 }.badge { background:#eef0f4; padding:4px 8px; border-radius:7px; font-variant-numeric:tabular-nums }.badge.local { background:#e9f6f0 }.badge.future { background:#edf2ff }
pre { white-space:pre-wrap; word-break:break-word; padding:12px; background:#f6f7f9; border-radius:8px; max-height:230px; overflow:auto }.heatmap { display:flex; flex-wrap:wrap; gap:4px; padding:14px; background:#f7f8fa; border-radius:9px }.heat-token { --v:0; position:relative; padding:6px 5px 14px; min-width:24px; text-align:center; border-radius:6px; background:rgba(60,90,210,calc(.08 + var(--v)*.65)); border:1px solid transparent }.heat-token.negative { background:rgba(220,70,80,calc(.08 + var(--v)*.65)) }.heat-token.selected { border:2px solid #131722; transform:translateY(-2px) }.heat-token small { position:absolute; bottom:0; left:0; right:0; color:#6d7480; font-size:8px }
.table-wrap { overflow:auto } table { border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums } th,td { padding:7px 8px; border-bottom:1px solid #e6e8ec; text-align:right; white-space:nowrap } th:first-child,td:first-child,th:nth-child(2),td:nth-child(2) { text-align:left }.up { color:#08783e;font-weight:700 }.down { color:#bc2f3b;font-weight:700 } details { margin-top:15px } summary { cursor:pointer;font-weight:700 }.ranking { display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px }.ranking>div { background:#f7f8fa;padding:10px;border-radius:8px }.ranking h5 { margin:0 }.ranking ol { padding-left:25px }.ranking li span { float:right }.pair { display:grid;grid-template-columns:1fr 1fr;gap:14px }.pair-side { background:#f7f8fa;padding:14px;border-radius:9px }.notice { padding:14px;border-left:4px solid #d39a26;background:#fff6df }.method { background:#eef1ff;padding:14px;border-radius:9px }
@media(max-width:760px){.pair,.ranking{grid-template-columns:1fr}.card header{display:block}.token-focus{text-align:left}.controls{flex-wrap:wrap}}
'''

JS = r'''
function updateHeatmaps() {
  const metric = document.getElementById('metric').value;
  document.querySelectorAll('.heatmap').forEach(map => {
    const tokens = [...map.querySelectorAll('.heat-token')];
    const attr = 'data-' + metric.replaceAll('_','-');
    const values = tokens.map(t => Number(t.getAttribute(attr) || 0));
    const scale = Math.max(...values.map(Math.abs), 1e-12);
    tokens.forEach((token,index) => {
      const value = values[index]; token.style.setProperty('--v', Math.min(Math.abs(value)/scale,1));
      token.classList.toggle('negative', value < 0);
      token.title = token.title.split(' · ')[0] + ` · ${metric}=${value.toPrecision(4)}`;
    });
  });
}
function filterCards() {
  const selected = document.getElementById('category').value;
  document.querySelectorAll('.card').forEach(card => card.hidden = selected !== 'all' && card.dataset.category !== selected);
}
document.getElementById('metric').addEventListener('change', updateHeatmaps);
document.getElementById('category').addEventListener('change', filterCards);
updateHeatmaps();
'''


def render_report(rows: list[dict[str, Any]], mined: dict[str, Any], *, source: Path, legacy: bool) -> str:
    categories = list(mined["categories"]) + list(mined["pairs"])
    options = ''.join(f'<option value="{html.escape(name)}">{html.escape(name)}</option>' for name in categories)
    cards = []
    for category, selected in mined["categories"].items():
        cards.extend(_record_card(row, category, index) for index, row in enumerate(selected, 1))
    for category, pairs in mined["pairs"].items():
        cards.extend(_pair_card(pair, category, index) for index, pair in enumerate(pairs, 1))
    legacy_notice = (
        '<p class="notice"><b>Legacy log:</b> these progress records predate qualitative logging. '
        'Token text, context, candidate probabilities, and counterfactual sequence rankings cannot be reconstructed. '
        'Use the qualitative training overlay for future or resumed steps.</p>' if legacy else ''
    )
    future = mined.get("future_outcome") or "unavailable"
    return f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CMT token examples</title><style>{CSS}</style></head>
<body><div class="hero"><h1>CMT qualitative token analysis</h1><p>Concrete before/after evidence for g, X, D and the g+D ranking. Positive KL gain means the measured fixed-support reverse KL decreased after the whole PPO minibatch update.</p></div>
<div class="controls"><label>Category <select id="category"><option value="all">all</option>{options}</select></label><label>Heatmap metric <select id="metric">{''.join(f'<option value="{key}">{label}</option>' for key,label in PREDICTOR_LABELS.items())}</select></label><span>{len(rows)} input records</span></div>
<main>{legacy_notice}<section class="method"><b>Measurement scope.</b> Source: <code>{html.escape(str(source))}</code>. Future mining outcome: <code>{html.escape(future)}</code>. Candidate probabilities and ranks are limited to the stored rollout-time student/teacher Top-K union plus the sampled target. The local KL is KL(p<sub>student,U</sub>∥q<sub>teacher,U</sub>), conditioned on that fixed union. Each change is caused by the complete minibatch optimizer update, so it is not an isolated-token intervention.</section>{''.join(cards) if cards else '<p class="notice">No usable records were found.</p>'}</main><script>{JS}</script></body></html>'''


def _portable_mined(mined: dict[str, Any]) -> dict[str, Any]:
    return mined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Run directory or its analysis directory")
    parser.add_argument("--output", required=True, type=Path, help="HTML path or output directory")
    parser.add_argument("--examples-per-category", type=int, default=5)
    parser.add_argument("--similar-g-relative-tolerance", type=float, default=.05)
    parser.add_argument("--similar-g-absolute-tolerance", type=float, default=1e-6)
    parser.add_argument("--minimum-d-gap", type=float, default=0.0)
    parser.add_argument(
        "--tokenizer", type=Path,
        help="Optional local Hugging Face tokenizer for joinable legacy token IDs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.examples_per_category <= 0:
        raise SystemExit("--examples-per-category must be positive")
    rows, root, legacy = load_records(args.input.resolve())
    decoded_legacy = decode_legacy_token_ids(rows, args.tokenizer) if args.tokenizer else 0
    mined = mine_examples(
        rows,
        examples_per_category=args.examples_per_category,
        similar_g_relative_tolerance=args.similar_g_relative_tolerance,
        similar_g_absolute_tolerance=args.similar_g_absolute_tolerance,
        minimum_d_gap=args.minimum_d_gap,
    )
    if args.output.suffix.lower() == ".html":
        html_path = args.output.resolve()
        output_dir = html_path.parent
    else:
        output_dir = args.output.resolve()
        html_path = output_dir / "token_examples.html"
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path.write_text(render_report(rows, mined, source=root, legacy=legacy), encoding="utf-8")
    (output_dir / "mined_examples.json").write_text(
        json.dumps(_portable_mined(mined), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {html_path}")
    print(f"Wrote {output_dir / 'mined_examples.json'}")
    if legacy:
        print("Warning: no qualitative records found; generated a scalar-only legacy report.")
        if args.tokenizer:
            print(f"Decoded {decoded_legacy} legacy token IDs; prefix context remains unavailable.")


if __name__ == "__main__":
    main()
