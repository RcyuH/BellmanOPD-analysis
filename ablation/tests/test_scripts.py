from pathlib import Path
import os
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def test_launchers_resolve_paths_from_script_directory():
    for path in (ROOT / "scripts").glob("*.sh"):
        text = path.read_text(encoding="utf-8")
        assert "SCRIPT_DIR" in text
    train = (ROOT / "scripts/train.sh").read_text(encoding="utf-8")
    assert "MAX_RESPONSE_LEN" in train and "TRAIN_EVAL_MAX_NEW_TOKENS" in train
    assert "TRAIN_EVAL_SEED" in train
    assert "TRAIN_EVAL_BENCHMARKS" in train
    for benchmark in (
        "Competition-MATH",
        "MATH-500",
        "AIME24",
        "AIME25",
        "GPQA-Diamond",
        "AMC23",
    ):
        assert benchmark in train
    assert "selector.cmt_ablation_arm" in train


def test_common_defaults_are_canonical():
    text = (ROOT / "configs/common.yaml").read_text(encoding="utf-8")
    for expected in ("cmt_allocation_kl: 0.5", "cmt_gamma: 1.0", "cmt_successor_lambda: 1.0", "top_k: 16", "max_new_tokens: 4096"):
        assert expected in text
    gamma_script = (ROOT / "scripts/sweep_gamma.sh").read_text(encoding="utf-8")
    assert "CMT_GAMMA" in gamma_script
    assert "train.sh\" g_d" in gamma_script


def test_train_launcher_is_cwd_independent_and_keeps_token_budgets():
    with tempfile.TemporaryDirectory() as temp:
        output = Path(temp) / "run"
        env = dict(os.environ, ABLATION_DRY_RUN="true", OUTPUT_DIR=str(output))
        result = subprocess.run(
            ["bash", str(ROOT / "scripts/train.sh"), "g_x"],
            cwd="/tmp",
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "train_max_new_tokens=4096 eval_max_new_tokens=7168" in result.stdout
        assert (output / "ablation_spec.json").is_file()


def test_plot_launcher_uses_a_fresh_named_directory():
    text = (ROOT / "scripts/plot_ablation.sh").read_text(encoding="utf-8")
    assert "FIGURE_ROOT" in text
    assert "PLOT_TAG" in text
    assert "BENCHMARKS" in text
    assert "--benchmarks" in text
    assert "date +%Y%m%d_%H%M%S_%N" in text
    assert "GD_CMT_RUN_NAME" in text
    assert "--g-d-output" in text
    assert "--g-d-run-name" in text
