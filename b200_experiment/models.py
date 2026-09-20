from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as transformers_logging

transformers_logging.disable_progress_bar()


def effective_text_config(model_config):
    """Return the decoder config used by ``AutoModelForCausalLM``.

    Text-only checkpoints (Qwen3 and Llama) expose the decoder fields at the
    top level.  Qwen3.5 is distributed as a composite vision-language config,
    while ``AutoModelForCausalLM`` deliberately extracts ``text_config`` and
    builds ``Qwen3_5ForCausalLM``.  Keeping this resolution in one place avoids
    reading a missing top-level ``vocab_size`` and records the architecture we
    actually train rather than the outer checkpoint container.
    """
    get_text_config = getattr(model_config, "get_text_config", None)
    if callable(get_text_config):
        text_config = get_text_config()
        if text_config is not None:
            return text_config
    return getattr(model_config, "text_config", None) or model_config


def model_vocab_size(model_config) -> int:
    text_config = effective_text_config(model_config)
    vocab_size = getattr(text_config, "vocab_size", None)
    if vocab_size is None:
        raise ValueError(
            "Model config does not expose a text vocabulary size at either "
            "config.vocab_size or config.text_config.vocab_size"
        )
    return int(vocab_size)


def is_qwen35_composite_text_model(model) -> bool:
    """Whether a text-only HF model originated from a composite Qwen3.5 repo."""
    source_config = getattr(model, "_b200_source_config", None)
    return (
        str(getattr(source_config, "model_type", "")) == "qwen3_5"
        and str(getattr(getattr(model, "config", None), "model_type", ""))
        == "qwen3_5_text"
    )


def qwen35_composite_weight_name(name: str) -> str:
    """Map a HF Qwen3.5 text-only key to the official composite namespace.

    vLLM <=0.26 registers the official Qwen3.5 conditional architecture but
    not its text-only architecture.  Its conditional loader understands the
    official ``model.language_model.*`` keys, so rollout IPC and exported
    checkpoints use that namespace.  The trainable HF module itself remains
    text-only and therefore does not allocate or optimize the vision tower.
    """
    if name.startswith("model."):
        return "model.language_model." + name[len("model.") :]
    return name


def _dtype(name: str):
    values = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    if name not in values:
        raise ValueError(f"Unsupported dtype {name!r}")
    return values[name]


def _vocab_digest(tokenizer) -> str:
    payload = json.dumps(
        tokenizer.get_vocab(), sort_keys=True, ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def assert_tokenizer_compatibility(
    student_tokenizer, teacher_tokenizer, student_config, teacher_config
):
    # OPD shares the student's integer token IDs with the teacher.  Therefore
    # the token-to-ID mapping must be exact.  The semantic roles assigned to
    # those IDs (special_tokens_map) may legitimately differ between a Base
    # student and an Instruct teacher and are not used for teacher scoring.
    fatal_checks: dict[str, Any] = {
        "vocab_mapping": student_tokenizer.get_vocab() == teacher_tokenizer.get_vocab(),
        "added_vocab": student_tokenizer.get_added_vocab()
        == teacher_tokenizer.get_added_vocab(),
        "model_vocab_size": model_vocab_size(student_config)
        == model_vocab_size(teacher_config),
        "tokenizer_length": len(student_tokenizer) == len(teacher_tokenizer),
    }
    if not all(fatal_checks.values()):
        failed = [name for name, passed in fatal_checks.items() if not passed]
        raise ValueError(
            "Student/teacher token IDs are incompatible: " + ", ".join(failed)
        )
    checks = dict(fatal_checks)
    checks["special_tokens_map_equal"] = (
        student_tokenizer.special_tokens_map == teacher_tokenizer.special_tokens_map
    )
    checks["vocab_sha256"] = _vocab_digest(student_tokenizer)
    return checks


def validate_shared_tokenizer_protocol(config: dict[str, Any]) -> dict[str, Any]:
    """Validate the fixed student-tokenizer/no-think teacher protocol."""
    if config.get("models", {}).get("teacher_no_think") is not True:
        raise ValueError(
            "models.teacher_no_think must be true: the teacher is always run "
            "with the shared no-think prompt protocol"
        )
    enable_thinking = (
        config.get("data", {}).get("chat_template_kwargs", {}).get("enable_thinking")
    )
    if enable_thinking is not False:
        raise ValueError(
            "data.chat_template_kwargs.enable_thinking must be false because the "
            "student-rendered prompt IDs are shared directly with the no-think teacher"
        )
    return {
        "tokenizer_source": "student",
        "teacher_input": "shared_student_token_ids",
        "teacher_retokenization": False,
        "teacher_no_think": True,
    }


def choose_attention_implementation(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        from transformers.utils import is_flash_attn_2_available

        if is_flash_attn_2_available():
            return "flash_attention_2"
    except (ImportError, AttributeError):
        pass
    return "sdpa"


def model_dtype_kwargs(dtype: torch.dtype) -> dict[str, torch.dtype]:
    """Transformers 4.x calls this torch_dtype; 5.x renamed it to dtype."""
    major = int(transformers.__version__.split(".", 1)[0])
    return {"dtype" if major >= 5 else "torch_dtype": dtype}


def inspect_model_assets(config: dict[str, Any]) -> dict[str, Any]:
    model_cfg = config["models"]
    tokenizer_protocol = validate_shared_tokenizer_protocol(config)
    student_path = Path(model_cfg["student_path"]).resolve()
    teacher_path = Path(model_cfg["teacher_path"]).resolve()
    for role, path in (("student", student_path), ("teacher", teacher_path)):
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"{role} checkpoint is missing config.json: {path}")
    student_tokenizer = AutoTokenizer.from_pretrained(
        student_path, local_files_only=True
    )
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        teacher_path, local_files_only=True
    )
    student_config = AutoConfig.from_pretrained(student_path, local_files_only=True)
    teacher_config = AutoConfig.from_pretrained(teacher_path, local_files_only=True)
    student_text_config = effective_text_config(student_config)
    teacher_text_config = effective_text_config(teacher_config)
    compatibility = assert_tokenizer_compatibility(
        student_tokenizer, teacher_tokenizer, student_config, teacher_config
    )
    return {
        "student_path": str(student_path),
        "teacher_path": str(teacher_path),
        "student_model_type": student_config.model_type,
        "teacher_model_type": teacher_config.model_type,
        "student_text_model_type": student_text_config.model_type,
        "teacher_text_model_type": teacher_text_config.model_type,
        "student_parameters_declared": getattr(student_config, "num_parameters", None),
        "teacher_parameters_declared": getattr(teacher_config, "num_parameters", None),
        "compatibility": compatibility,
        "tokenizer_protocol": tokenizer_protocol,
        "attention_implementation": choose_attention_implementation(
            model_cfg.get("attention_implementation", "auto")
        ),
    }


def load_models(config: dict[str, Any], device: torch.device):
    model_cfg = config["models"]
    student_path = Path(model_cfg["student_path"]).resolve()
    teacher_path = Path(model_cfg["teacher_path"]).resolve()
    assets = inspect_model_assets(config)
    # This is intentionally the only runtime tokenizer.  Rollout response IDs
    # and the full prompt+response IDs are passed straight to teacher.forward;
    # teacher text is never decoded and re-tokenized.
    tokenizer = AutoTokenizer.from_pretrained(student_path, local_files_only=True)
    dtype = _dtype(model_cfg.get("dtype", "bfloat16"))
    attention = assets["attention_implementation"]
    common = {
        "local_files_only": True,
        "low_cpu_mem_usage": True,
        "attn_implementation": attention,
        **model_dtype_kwargs(dtype),
    }
    student = AutoModelForCausalLM.from_pretrained(student_path, **common).to(device)
    teacher = AutoModelForCausalLM.from_pretrained(teacher_path, **common).to(device)
    # Preserve the outer checkpoint configs.  This matters for Qwen3.5: the HF
    # training modules are text-only, but vLLM 0.17 loads the official
    # composite architecture.  Snapshot/IPC code uses this immutable metadata
    # to translate only the serialization namespace; training stays text-only.
    student._b200_source_config = AutoConfig.from_pretrained(
        student_path, local_files_only=True
    )
    teacher._b200_source_config = AutoConfig.from_pretrained(
        teacher_path, local_files_only=True
    )
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    teacher.eval()

    training = config.get("training", {})
    if training.get("use_lora", False):
        from peft import LoraConfig, get_peft_model

        lora = training.get("lora", {})
        student = get_peft_model(
            student,
            LoraConfig(
                r=int(lora.get("rank", 8)),
                lora_alpha=int(lora.get("alpha", 16)),
                lora_dropout=float(lora.get("dropout", 0.0)),
                target_modules=list(lora.get("target_modules", ["q_proj", "v_proj"])),
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
    if training.get("gradient_checkpointing", False):
        student.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(student, "enable_input_require_grads"):
            student.enable_input_require_grads()
    student.config.use_cache = False
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    assets.update(
        dtype=str(dtype),
        trainable_parameters=sum(
            p.numel() for p in student.parameters() if p.requires_grad
        ),
        student_parameters=sum(p.numel() for p in student.parameters()),
        teacher_parameters=sum(p.numel() for p in teacher.parameters()),
    )
    return student, teacher, tokenizer, assets


def load_student_model(config: dict[str, Any], device: torch.device):
    """Load only the trainable student for teacher-free baselines.

    GRPO does not use a teacher or a reference network.  Keeping this loader
    separate from ``load_models`` is important on B200: constructing an
    otherwise-unused teacher would both change the memory/throughput baseline
    and make GRPO fail for checkpoints where no teacher is available.
    """
    model_cfg = config["models"]
    student_path = Path(model_cfg["student_path"]).resolve()
    if not (student_path / "config.json").is_file():
        raise FileNotFoundError(
            f"student checkpoint is missing config.json: {student_path}"
        )
    student_config = AutoConfig.from_pretrained(student_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(student_path, local_files_only=True)
    dtype = _dtype(model_cfg.get("dtype", "bfloat16"))
    attention = choose_attention_implementation(
        model_cfg.get("attention_implementation", "auto")
    )
    common = {
        "local_files_only": True,
        "low_cpu_mem_usage": True,
        "attn_implementation": attention,
        **model_dtype_kwargs(dtype),
    }
    student = AutoModelForCausalLM.from_pretrained(student_path, **common).to(device)
    student._b200_source_config = student_config
    training = config.get("training", {})
    if training.get("use_lora", False):
        from peft import LoraConfig, get_peft_model

        lora = training.get("lora", {})
        student = get_peft_model(
            student,
            LoraConfig(
                r=int(lora.get("rank", 8)),
                lora_alpha=int(lora.get("alpha", 16)),
                lora_dropout=float(lora.get("dropout", 0.0)),
                target_modules=list(lora.get("target_modules", ["q_proj", "v_proj"])),
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
    if training.get("gradient_checkpointing", False):
        student.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(student, "enable_input_require_grads"):
            student.enable_input_require_grads()
    student.config.use_cache = False
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    metadata = {
        "student_path": str(student_path),
        "teacher_path": None,
        "student_model_type": student_config.model_type,
        "student_text_model_type": effective_text_config(student_config).model_type,
        "teacher_model_type": None,
        "student_parameters_declared": getattr(student_config, "num_parameters", None),
        "teacher_parameters_declared": None,
        "compatibility": None,
        "tokenizer_protocol": {
            "tokenizer_source": "student",
            "teacher_used": False,
        },
        "attention_implementation": attention,
        "dtype": str(dtype),
        "trainable_parameters": sum(
            p.numel() for p in student.parameters() if p.requires_grad
        ),
        "student_parameters": sum(p.numel() for p in student.parameters()),
        "teacher_parameters": 0,
    }
    return student, tokenizer, metadata


def load_student_tokenizer(config: dict[str, Any]):
    """Load the sole runtime tokenizer without materializing either model.

    Training uses this lightweight path to validate the fully rendered prompt
    lengths before starting vLLM or loading multi-billion-parameter models.
    """
    student_path = Path(config["models"]["student_path"]).resolve()
    tokenizer = AutoTokenizer.from_pretrained(student_path, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
