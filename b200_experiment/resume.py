from __future__ import annotations

import csv
import gzip
import json
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from .fsdp import is_fsdp_model, scatter_full_optimizer_state_dict


_CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")
_SELECTOR_CHUNK_PATTERN = re.compile(
    r"^selected_steps_(\d+)_(\d+)(?:_rank-\d+)?\.jsonl\.gz$"
)
_STEP_JSON_PATTERN = re.compile(r"^step-(\d+)\.json$")
_STEP_DIRECTORY_PATTERN = re.compile(r"^step-(\d+)$")


@dataclass(frozen=True)
class ResumeState:
    checkpoint: Path
    optimizer_path: Path
    step: int


def resolve_resume_checkpoint(
    value: str | Path | None, output_dir: str | Path | None = None
) -> Path | None:
    if value is None or not str(value).strip():
        return None
    if str(value).strip().lower() == "auto":
        if output_dir is None:
            raise ValueError("RESUME=auto requires an existing run output directory")
        root = Path(output_dir).expanduser().resolve()
        latest = root / "latest.json"
        checkpoint = None
        if latest.is_file():
            payload = json.loads(latest.read_text(encoding="utf-8"))
            candidate = Path(payload["checkpoint"])
            checkpoint = candidate if candidate.is_absolute() else root / candidate
        if checkpoint is None or not checkpoint.is_dir():
            candidates = sorted(
                (
                    path
                    for path in root.glob("checkpoint-*")
                    if path.is_dir() and (path / "optimizer.pt").is_file()
                ),
                key=lambda path: int(path.name.rsplit("-", 1)[-1]),
            )
            if (root / "final/optimizer.pt").is_file():
                candidates.append(root / "final")
            if not candidates:
                raise FileNotFoundError(
                    f"RESUME=auto found no complete checkpoint under {root}"
                )
            checkpoint = candidates[-1]
        checkpoint = checkpoint.resolve()
    else:
        checkpoint = Path(value).expanduser().resolve()
    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(
            f"Resume checkpoint is missing config.json: {checkpoint}"
        )
    if not (checkpoint / "optimizer.pt").is_file():
        raise FileNotFoundError(
            f"True resume requires optimizer.pt, but it is missing from {checkpoint}"
        )
    return checkpoint


def _torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _cpu_byte_rng_state(value: Any, name: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise ValueError(f"Invalid {name}: expected a tensor")
    state = value.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    if state.ndim != 1:
        raise ValueError(f"Invalid {name}: expected a one-dimensional tensor")
    return state


def _restore_rng_states(payload: dict[str, Any], device: torch.device) -> None:
    if "torch_rng_state" in payload:
        torch.set_rng_state(
            _cpu_byte_rng_state(payload["torch_rng_state"], "torch_rng_state")
        )
    if device.type != "cuda" or "cuda_rng_state_all" not in payload:
        return

    saved = payload["cuda_rng_state_all"]
    # Accept the list produced by torch.cuda.get_rng_state_all() as well as a
    # tensor from older single-GPU checkpoints.
    if torch.is_tensor(saved):
        saved_states = [saved]
    elif isinstance(saved, (list, tuple)):
        saved_states = list(saved)
    else:
        raise ValueError(
            "Invalid cuda_rng_state_all: expected a tensor or a list of tensors"
        )
    if not saved_states:
        raise ValueError("Invalid cuda_rng_state_all: no CUDA RNG states were saved")

    target_index = device.index if device.index is not None else 0
    # A checkpoint can be resumed with fewer/more visible GPUs. Restore the
    # matching logical device when present, otherwise use the sole/main state.
    source_index = target_index if target_index < len(saved_states) else 0
    state = _cpu_byte_rng_state(
        saved_states[source_index], f"cuda_rng_state_all[{source_index}]"
    )
    torch.cuda.set_rng_state(state, device=device)


def _canonical_parameter_name(name: str) -> str:
    """Normalize wrapper prefixes used by DDP/FSDP to the HF name."""
    normalized = str(name)
    while normalized.startswith("module."):
        normalized = normalized[len("module.") :]
    return normalized.replace("_fsdp_wrapped_module.", "")


def _model_parameter_name_by_identity(model) -> dict[int, str]:
    if model is None:
        raise ValueError(
            "Optimizer format conversion requires the current model so parameter "
            "IDs can be matched to HF parameter names"
        )
    names: dict[int, str] = {}
    canonical_names: dict[str, int] = {}
    for name, parameter in model.named_parameters():
        canonical = _canonical_parameter_name(name)
        identity = id(parameter)
        # Tied/shared parameters may be exposed by more than one name. Keep
        # the first FQN, matching FSDP's canonical FQN selection, while still
        # rejecting two distinct parameters that collide after unwrapping.
        if identity in names:
            continue
        previous_identity = canonical_names.get(canonical)
        if previous_identity is not None and previous_identity != identity:
            raise ValueError(
                "Current model exposes duplicate canonical parameter name: "
                f"{canonical!r}"
            )
        names[identity] = canonical
        canonical_names[canonical] = identity
    return names


def _optimizer_parameter_id_to_name(optimizer, model) -> dict[Any, str]:
    """Map PyTorch optimizer IDs to current HF parameter names.

    Integer IDs have no persistent meaning by themselves. PyTorch assigns them
    in param-group order, so we use the current optimizer's group topology and
    validate it rather than silently pairing incompatible parameters.
    """
    current_state = optimizer.state_dict()
    current_groups = current_state.get("param_groups")
    actual_groups = getattr(optimizer, "param_groups", None)
    if not isinstance(current_groups, list) or not isinstance(actual_groups, list):
        raise ValueError("Current optimizer has no usable param_groups")
    if len(current_groups) != len(actual_groups):
        raise ValueError(
            "Optimizer param-group count changed since the checkpoint was saved: "
            f"current={len(current_groups)}, actual={len(actual_groups)}"
        )
    names_by_identity = _model_parameter_name_by_identity(model)
    result: dict[Any, str] = {}
    for group_index, (saved_group, actual_group) in enumerate(
        zip(current_groups, actual_groups)
    ):
        saved_ids = saved_group.get("params")
        actual_parameters = actual_group.get("params")
        if not isinstance(saved_ids, list) or not isinstance(actual_parameters, list):
            raise ValueError(
                f"Optimizer param_group[{group_index}] has invalid params list"
            )
        if len(saved_ids) != len(actual_parameters):
            raise ValueError(
                "Optimizer parameter count changed in param group "
                f"{group_index}: current={len(saved_ids)}, "
                f"actual={len(actual_parameters)}"
            )
        for parameter_id, parameter in zip(saved_ids, actual_parameters):
            name = names_by_identity.get(id(parameter))
            if name is None:
                raise ValueError(
                    "Current optimizer contains a parameter absent from the model; "
                    f"cannot map optimizer ID {parameter_id!r}"
                )
            previous = result.get(parameter_id)
            if previous is not None and previous != name:
                raise ValueError(
                    f"Optimizer ID {parameter_id!r} maps to both {previous!r} and "
                    f"{name!r}"
                )
            result[parameter_id] = name
    return result


def _optimizer_checkpoint_format(payload: dict[str, Any]) -> str:
    """Return and validate the on-disk optimizer format.

    Older checkpoints did not carry ``optimizer_format``. Infer those files
    from state/group key types so they remain resumable.
    """
    optimizer_state = payload.get("optimizer")
    if not isinstance(optimizer_state, dict):
        raise ValueError("Optimizer checkpoint has no dictionary optimizer state")
    state = optimizer_state.get("state")
    groups = optimizer_state.get("param_groups")
    if not isinstance(state, dict) or not isinstance(groups, list):
        raise ValueError(
            "Optimizer checkpoint state must contain dict 'state' and list "
            "'param_groups'"
        )
    keys = list(state)
    group_params = [
        parameter_id
        for group in groups
        if isinstance(group, dict) and isinstance(group.get("params"), list)
        for parameter_id in group["params"]
    ]
    observed = keys + group_params
    inferred = None
    if observed and all(isinstance(item, str) for item in observed):
        inferred = "fsdp_full_v1"
    elif observed and all(isinstance(item, int) for item in observed):
        inferred = "standard"
    elif observed:
        raise ValueError(
            "Optimizer checkpoint mixes string parameter names and integer "
            "parameter IDs"
        )
    requested = payload.get("optimizer_format")
    if requested is None:
        return inferred or "standard"
    requested = str(requested)
    if requested not in {"standard", "fsdp_full_v1"}:
        raise ValueError(f"Unsupported optimizer checkpoint format: {requested!r}")
    if inferred is not None and requested != inferred:
        raise ValueError(
            "Optimizer checkpoint format metadata disagrees with its parameter "
            f"keys: metadata={requested!r}, observed={inferred!r}"
        )
    return requested


def _validate_source_group_structure(
    source_groups: Any, optimizer, format_name: str
) -> None:
    actual_groups = getattr(optimizer, "param_groups", None)
    if not isinstance(source_groups, list) or not isinstance(actual_groups, list):
        raise ValueError(f"Invalid {format_name} optimizer param_groups")
    if len(source_groups) != len(actual_groups):
        raise ValueError(
            "Optimizer param-group count changed since the checkpoint was saved: "
            f"checkpoint={len(source_groups)}, current={len(actual_groups)}"
        )
    for group_index, (source_group, actual_group) in enumerate(
        zip(source_groups, actual_groups)
    ):
        source_params = (
            source_group.get("params") if isinstance(source_group, dict) else None
        )
        actual_params = (
            actual_group.get("params") if isinstance(actual_group, dict) else None
        )
        if not isinstance(source_params, list) or not isinstance(actual_params, list):
            raise ValueError(
                f"Invalid {format_name} optimizer param_group[{group_index}]"
            )
        if len(source_params) != len(actual_params):
            raise ValueError(
                "Optimizer parameter count changed in param group "
                f"{group_index}: checkpoint={len(source_params)}, "
                f"current={len(actual_params)}"
            )


def _standard_optimizer_to_full(
    optimizer_state: dict[str, Any], optimizer, model
) -> dict[str, Any]:
    """Convert PyTorch integer-ID state to FSDP full name-keyed state."""
    id_to_name = _optimizer_parameter_id_to_name(optimizer, model)
    source_state = optimizer_state.get("state")
    source_groups = optimizer_state.get("param_groups")
    if not isinstance(source_state, dict) or not isinstance(source_groups, list):
        raise ValueError("Invalid standard optimizer state")
    _validate_source_group_structure(source_groups, optimizer, "standard")

    full_state: dict[str, Any] = {}
    for parameter_id, value in source_state.items():
        name = id_to_name.get(parameter_id)
        if name is None:
            raise ValueError(
                "Standard optimizer state contains an unknown parameter ID: "
                f"{parameter_id!r}"
            )
        full_state[name] = value

    full_groups: list[dict[str, Any]] = []
    for group_index, source_group in enumerate(source_groups):
        if not isinstance(source_group, dict) or not isinstance(
            source_group.get("params"), list
        ):
            raise ValueError(f"Invalid standard optimizer param_group[{group_index}]")
        converted = dict(source_group)
        converted["params"] = []
        for parameter_id in source_group["params"]:
            name = id_to_name.get(parameter_id)
            if name is None:
                raise ValueError(
                    "Standard optimizer param_group contains an unknown "
                    f"parameter ID: {parameter_id!r}"
                )
            converted["params"].append(name)
        full_groups.append(converted)
    return {"state": full_state, "param_groups": full_groups}


def _full_optimizer_to_standard(
    optimizer_state: dict[str, Any], optimizer, model
) -> dict[str, Any]:
    """Convert FSDP full name-keyed state to integer-ID state."""
    id_to_name = _optimizer_parameter_id_to_name(optimizer, model)
    name_to_id = {name: parameter_id for parameter_id, name in id_to_name.items()}
    source_state = optimizer_state.get("state")
    source_groups = optimizer_state.get("param_groups")
    if not isinstance(source_state, dict) or not isinstance(source_groups, list):
        raise ValueError("Invalid FSDP full optimizer state")
    _validate_source_group_structure(source_groups, optimizer, "FSDP full")

    standard_state: dict[Any, Any] = {}
    for raw_name, value in source_state.items():
        if not isinstance(raw_name, str):
            raise ValueError(
                "FSDP full optimizer state must use string parameter names; "
                f"got {raw_name!r}"
            )
        name = _canonical_parameter_name(raw_name)
        parameter_id = name_to_id.get(name)
        if parameter_id is None:
            raise ValueError(
                "FSDP optimizer state contains a parameter absent from the current "
                f"model: {raw_name!r}"
            )
        standard_state[parameter_id] = value

    standard_groups: list[dict[str, Any]] = []
    target_groups = optimizer.state_dict().get("param_groups")
    if not isinstance(target_groups, list):
        raise ValueError("Current optimizer has no usable param_groups")
    for group_index, source_group in enumerate(source_groups):
        if not isinstance(source_group, dict) or not isinstance(
            source_group.get("params"), list
        ):
            raise ValueError(f"Invalid FSDP optimizer param_group[{group_index}]")
        target_group = target_groups[group_index]
        target_ids = target_group["params"]
        source_names = []
        for raw_name in source_group["params"]:
            if not isinstance(raw_name, str):
                raise ValueError(
                    "FSDP full optimizer param_groups must use string parameter "
                    f"names; got {raw_name!r}"
                )
            name = _canonical_parameter_name(raw_name)
            if name not in name_to_id:
                raise ValueError(
                    "FSDP optimizer param_group contains a parameter absent from "
                    f"the current model: {raw_name!r}"
                )
            source_names.append(name)
        target_names = []
        for parameter_id in target_ids:
            name = id_to_name.get(parameter_id)
            if name is None:
                raise ValueError(
                    "Current optimizer contains an unknown parameter ID: "
                    f"{parameter_id!r}"
                )
            target_names.append(name)
        if sorted(source_names) != sorted(target_names):
            raise ValueError(
                "FSDP optimizer parameter names do not match current optimizer "
                f"group {group_index}: checkpoint={sorted(source_names)}, "
                f"current={sorted(target_names)}"
            )
        converted = dict(source_group)
        # Optimizer.load_state_dict() zips each loaded group's ``params`` with
        # the current group's Parameter objects by position. Use the current
        # order here; preserving FSDP's name-sorted order would attach moments
        # to the wrong parameters even though the IDs look valid.
        converted["params"] = list(target_ids)
        standard_groups.append(converted)
    return {"state": standard_state, "param_groups": standard_groups}


def restore_optimizer(
    optimizer,
    checkpoint: str | Path,
    device: torch.device,
    *,
    model=None,
    distributed=None,
) -> ResumeState:
    checkpoint = Path(checkpoint).resolve()
    optimizer_path = checkpoint / "optimizer.pt"
    if model is not None and is_fsdp_model(model):
        if distributed is None:
            raise ValueError("FSDP optimizer restore requires distributed context")
        payload = None
        load_error = None
        if distributed.is_main:
            try:
                payload = _torch_load(optimizer_path, torch.device("cpu"))
            except Exception as error:
                load_error = f"{type(error).__name__}: {error}"
        if load_error is not None:
            metadata = {
                "optimizer_format": None,
                "optimizer_format_error": (
                    f"could not load optimizer.pt: {load_error}"
                ),
            }
        elif payload is not None and not isinstance(payload, dict):
            metadata = {
                "optimizer_format": None,
                "optimizer_format_error": (
                    "checkpoint payload must be a dictionary, got "
                    f"{type(payload).__name__}"
                ),
            }
        else:
            metadata = (
                {key: value for key, value in payload.items() if key != "optimizer"}
                if payload is not None
                else None
            )
        if metadata is not None and isinstance(payload, dict):
            try:
                metadata["optimizer_format"] = _optimizer_checkpoint_format(payload)
            except Exception as error:
                # Do not raise before the metadata broadcast: non-main FSDP
                # ranks would otherwise wait forever at that collective.
                metadata["optimizer_format"] = None
                metadata["optimizer_format_error"] = str(error)
        metadata = distributed.broadcast_object(metadata)
        if not isinstance(metadata, dict):
            raise ValueError(f"Invalid FSDP optimizer checkpoint {optimizer_path}")
        if metadata.get("optimizer_format_error"):
            raise ValueError(
                f"Invalid optimizer checkpoint {optimizer_path}: "
                f"{metadata['optimizer_format_error']}"
            )
        if "step" not in metadata:
            raise ValueError(f"Invalid FSDP optimizer checkpoint {optimizer_path}")
        source_optimizer = payload["optimizer"] if payload is not None else None
        conversion_error = None
        if metadata["optimizer_format"] == "standard" and payload is not None:
            try:
                source_optimizer = _standard_optimizer_to_full(
                    source_optimizer, optimizer, model
                )
            except Exception as error:
                conversion_error = str(error)
        conversion_error = distributed.broadcast_object(conversion_error)
        if conversion_error:
            raise ValueError(
                f"Cannot convert optimizer checkpoint {optimizer_path} for FSDP: "
                f"{conversion_error}"
            )
        sharded_state = scatter_full_optimizer_state_dict(
            source_optimizer,
            model,
            optimizer,
        )
        optimizer.load_state_dict(sharded_state)
        step = int(metadata["step"])
        if step < 0:
            raise ValueError(f"Resume step must be non-negative, got {step}")
        rng_states = metadata.get("rng_states")
        if isinstance(rng_states, list) and rng_states:
            rank_state = rng_states[
                distributed.rank if distributed.rank < len(rng_states) else 0
            ]
            _restore_rng_states(
                {
                    "torch_rng_state": rank_state["torch_rng_state"],
                    "cuda_rng_state_all": [rank_state["cuda_rng_state"]],
                },
                device,
            )
        match = _CHECKPOINT_PATTERN.match(checkpoint.name)
        if match is not None and int(match.group(1)) != step:
            raise ValueError(
                f"Checkpoint directory says step {int(match.group(1))}, but "
                f"optimizer.pt says step {step}"
            )
        return ResumeState(checkpoint, optimizer_path, step)

    payload = _torch_load(optimizer_path, device)
    if (
        not isinstance(payload, dict)
        or "step" not in payload
        or "optimizer" not in payload
    ):
        raise ValueError(
            f"Invalid optimizer checkpoint {optimizer_path}; expected step and optimizer"
        )
    step = int(payload["step"])
    if step < 0:
        raise ValueError(f"Resume step must be non-negative, got {step}")
    match = _CHECKPOINT_PATTERN.match(checkpoint.name)
    if match is not None and int(match.group(1)) != step:
        raise ValueError(
            f"Checkpoint directory says step {int(match.group(1))}, but "
            f"optimizer.pt says step {step}"
        )
    optimizer_state = payload["optimizer"]
    if _optimizer_checkpoint_format(payload) == "fsdp_full_v1":
        optimizer_state = _full_optimizer_to_standard(optimizer_state, optimizer, model)
    optimizer.load_state_dict(optimizer_state)
    # map_location normally handles this. The explicit walk also supports
    # optimizer states saved by older PyTorch versions on cuda:0.
    for state in optimizer.state.values():
        for key, item in state.items():
            if torch.is_tensor(item):
                state[key] = item.to(device)
    _restore_rng_states(payload, device)
    return ResumeState(checkpoint, optimizer_path, step)


def _temporary_sibling(path: Path) -> Path:
    return path.with_name(f".{path.name}.resume-rewind-{uuid.uuid4().hex}.tmp")


def _stage_jsonl_rewind(
    path: Path, resume_step: int, step_field: str
) -> tuple[Path | None, int | None, int]:
    if not path.is_file():
        return None, None, 0
    temporary = _temporary_sibling(path)
    retained_last_step = None
    removed_rows = 0
    try:
        with path.open(encoding="utf-8") as source, temporary.open(
            "x", encoding="utf-8"
        ) as target:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    target.write(line)
                    continue
                try:
                    step = int(json.loads(line)[step_field])
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"Cannot resume safely: malformed {path} line {line_number}"
                    ) from error
                if step <= resume_step:
                    target.write(line)
                    if not line.endswith("\n"):
                        target.write("\n")
                    retained_last_step = (
                        step
                        if retained_last_step is None
                        else max(retained_last_step, step)
                    )
                else:
                    removed_rows += 1
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    if removed_rows == 0:
        temporary.unlink()
        return None, retained_last_step, 0
    return temporary, retained_last_step, removed_rows


def _stage_csv_rewind(
    path: Path, resume_step: int
) -> tuple[Path | None, int | None, int]:
    if not path.is_file():
        return None, None, 0
    temporary = _temporary_sibling(path)
    retained_last_step = None
    removed_rows = 0
    try:
        with path.open(newline="", encoding="utf-8") as source, temporary.open(
            "x", newline="", encoding="utf-8"
        ) as target:
            reader = csv.DictReader(source)
            if not reader.fieldnames or "step" not in reader.fieldnames:
                raise ValueError(f"Cannot resume safely: {path} has no step column")
            writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
            writer.writeheader()
            for line_number, row in enumerate(reader, start=2):
                try:
                    step = int(row["step"])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"Cannot resume safely: malformed {path} line {line_number}"
                    ) from error
                if step <= resume_step:
                    writer.writerow(row)
                    retained_last_step = (
                        step
                        if retained_last_step is None
                        else max(retained_last_step, step)
                    )
                else:
                    removed_rows += 1
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    if removed_rows == 0:
        temporary.unlink()
        return None, retained_last_step, 0
    return temporary, retained_last_step, removed_rows


def _stage_selector_rewind(
    path: Path, resume_step: int
) -> tuple[Path | None, bool, int]:
    temporary = _temporary_sibling(path)
    retained_rows = 0
    removed_rows = 0
    try:
        with gzip.open(path, "rt", encoding="utf-8") as source, gzip.open(
            temporary, "xt", encoding="utf-8", compresslevel=6
        ) as target:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    step = int(json.loads(line)["training_step"])
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"Cannot resume safely: malformed {path} line {line_number}"
                    ) from error
                if step <= resume_step:
                    target.write(line)
                    if not line.endswith("\n"):
                        target.write("\n")
                    retained_rows += 1
                else:
                    removed_rows += 1
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    if removed_rows == 0:
        temporary.unlink()
        return None, False, 0
    if retained_rows == 0:
        temporary.unlink()
        return None, True, removed_rows
    return temporary, False, removed_rows


def _step_paths_after(root: Path, pattern: re.Pattern, resume_step: int) -> list[Path]:
    if not root.is_dir():
        return []
    stale = []
    for path in root.iterdir():
        match = pattern.match(path.name)
        if match is not None and int(match.group(1)) > resume_step:
            stale.append(path)
    return sorted(stale)


def validate_append_history(
    output_dir: str | Path,
    resume_step: int,
    resume_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Atomically rewind append-only outputs to the selected checkpoint step."""
    if resume_step < 0:
        raise ValueError(f"Resume step must be non-negative, got {resume_step}")
    output_dir = Path(output_dir).resolve()
    replacements: list[tuple[Path, Path]] = []
    deletions: set[Path] = set()
    removed_rows: dict[str, int] = {}
    metrics_step = None
    evaluation_step = None
    selector_removed_rows = 0

    try:
        for filename, kind in (
            ("metrics.jsonl", "jsonl"),
            ("eval_history.jsonl", "jsonl"),
            ("train_metrics.csv", "csv"),
            ("eval_metrics.csv", "csv"),
        ):
            path = output_dir / filename
            if kind == "jsonl":
                temporary, retained_step, removed = _stage_jsonl_rewind(
                    path, resume_step, "step"
                )
            else:
                temporary, retained_step, removed = _stage_csv_rewind(path, resume_step)
            if filename == "metrics.jsonl":
                metrics_step = retained_step
            elif filename == "eval_history.jsonl":
                evaluation_step = retained_step
            if temporary is not None:
                replacements.append((temporary, path))
            if removed:
                removed_rows[filename] = removed

        selector_root = output_dir / "selector_scores"
        if selector_root.is_dir():
            for path in sorted(selector_root.glob("selected_steps_*.jsonl.gz")):
                if _SELECTOR_CHUNK_PATTERN.match(path.name) is None:
                    continue
                temporary, delete_path, removed = _stage_selector_rewind(
                    path, resume_step
                )
                if temporary is not None:
                    replacements.append((temporary, path))
                if delete_path:
                    deletions.add(path)
                selector_removed_rows += removed

        deletions.update(
            _step_paths_after(
                output_dir / "token_score_stats", _STEP_JSON_PATTERN, resume_step
            )
        )
        deletions.update(
            _step_paths_after(
                output_dir / "training_eval", _STEP_DIRECTORY_PATTERN, resume_step
            )
        )
        deletions.update(
            _step_paths_after(output_dir, _CHECKPOINT_PATTERN, resume_step)
        )

        checkpoint = (
            Path(resume_checkpoint).expanduser().resolve()
            if resume_checkpoint is not None
            else None
        )
        final_checkpoint = output_dir / "final"
        if final_checkpoint.is_dir() and checkpoint != final_checkpoint.resolve():
            deletions.add(final_checkpoint)
        summary = output_dir / "summary.json"
        if summary.is_file():
            deletions.add(summary)

        if checkpoint is not None:
            try:
                checkpoint_value = str(checkpoint.relative_to(output_dir))
            except ValueError:
                checkpoint_value = str(checkpoint)
            latest = output_dir / "latest.json"
            latest_temporary = _temporary_sibling(latest)
            latest_temporary.write_text(
                json.dumps(
                    {
                        "step": resume_step,
                        "checkpoint": checkpoint_value,
                        "final": checkpoint.name == "final",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            replacements.append((latest_temporary, latest))

        for temporary, destination in replacements:
            temporary.replace(destination)
        for path in sorted(deletions, key=lambda item: len(item.parts), reverse=True):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
    except BaseException:
        for temporary, _ in replacements:
            temporary.unlink(missing_ok=True)
        raise

    removed_paths = [str(path.relative_to(output_dir)) for path in sorted(deletions)]
    return {
        "metrics_last_step": metrics_step,
        "evaluation_last_step": evaluation_step,
        "selector_logs_checked": True,
        "rewound": bool(removed_rows or selector_removed_rows or removed_paths),
        "resume_step": resume_step,
        "removed_rows": removed_rows,
        "selector_rows_removed": selector_removed_rows,
        "removed_paths": removed_paths,
    }


def _get(config: dict[str, Any], dotted: str):
    value: Any = config
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def validate_resume_config(
    checkpoint: str | Path,
    current: dict[str, Any],
    *,
    allow_mismatch: bool = False,
) -> dict[str, Any]:
    """Check scientific settings while allowing GPU count/eval/save changes."""
    source_path = Path(checkpoint).resolve().parent / "resolved_config.yaml"
    if not source_path.is_file():
        return {"source_config": None, "checked": False, "mismatches": {}}
    source = yaml.safe_load(source_path.read_text(encoding="utf-8")) or {}
    keys = (
        "experiment.method",
        "experiment.seed",
        "distributed.strategy",
        "models.student_path",
        "models.teacher_path",
        "models.teacher_no_think",
        "data.path",
        "data.split",
        "data.prompt_key",
        "data.prefer_source_prompt",
        "data.chat_template_kwargs",
        "rollout.backend",
        "rollout.batch_size",
        "rollout.num_responses",
        "rollout.max_new_tokens",
        "rollout.temperature",
        "rollout.top_p",
        "rollout.seed",
        "selector.top_k",
        "opd.adv_estimator",
        "opd.top_k_strategy",
        "opd.reward_weight_mode",
        "opd.loss_agg_mode",
        "opd.teacher_temperature",
        "selector.rac_gamma",
        "selector.rac_w_min",
        "selector.rac_beta",
        "selector.cmt_allocation_kl",
        "selector.cmt_gamma",
        "selector.cmt_successor_lambda",
        "selector.cmt_full_vocab_diagnostics",
        "token_budget.rho",
        "training.learning_rate",
        "training.adam_betas",
        "training.weight_decay",
        "training.ppo_clip_low",
        "training.ppo_clip_high",
        "training.ppo_dual_clip",
    )
    mismatches = {
        key: {"checkpoint": _get(source, key), "current": _get(current, key)}
        for key in keys
        if _get(source, key) != _get(current, key)
    }
    if mismatches and not allow_mismatch:
        formatted = ", ".join(
            f"{key}={values['checkpoint']!r}->{values['current']!r}"
            for key, values in mismatches.items()
        )
        raise ValueError(
            "Resume would change controlled training settings: "
            + formatted
            + ". Set RESUME_ALLOW_CONFIG_MISMATCH=true only if intentional."
        )
    return {
        "source_config": str(source_path),
        "checked": True,
        "mismatches": mismatches,
    }
