from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from b200_experiment.config import load_config


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPTS = (
    "train_opd_b200.sh",
    "train_ta_b200.sh",
    "train_rac_b200.sh",
    "train_cmt_b200.sh",
    "train_grpo_b200.sh",
    "train_iw_b200.sh",
)


class TrainingLauncherTests(unittest.TestCase):
    def test_opd_then_rac_workflow_has_exact_sequential_stages(self):
        path = REPO_ROOT / "scripts/train_opd_reeval_then_rac_reeval_b200.sh"
        content = path.read_text(encoding="utf-8")
        stages = (
            'bash "${SCRIPT_DIR}/train_opd_b200.sh"',
            'bash "${SCRIPT_DIR}/reeval_method_checkpoints_b200.sh" opd',
            'bash "${SCRIPT_DIR}/train_rac_b200.sh"',
            'bash "${SCRIPT_DIR}/reeval_method_checkpoints_b200.sh" rac',
        )
        positions = [content.index(stage) for stage in stages]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("export TRAIN_EVAL_ENABLED=false", content)
        self.assertIn('export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"', content)
        self.assertIn(
            'export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"', content
        )
        self.assertIn('export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"', content)
        self.assertIn("models/Qwen3-8B", content)
        self.assertIn("Qwen3-1.7B-Base", content)
        self.assertIn("competition_math", content)

    def test_training_launchers_do_not_require_preflight(self):
        for name in TRAIN_SCRIPTS:
            content = (REPO_ROOT / "scripts" / name).read_text(encoding="utf-8")
            with self.subTest(script=name):
                self.assertIn("USER CONFIG", content)
                self.assertNotIn("require_b200_validation", content)
                for variable in (
                    "STUDENT_MODEL",
                    "TEACHER_MODEL",
                    "TRAIN_DATA",
                    "PROMPT_KEY",
                    "GLOBAL_BATCH_SIZE",
                    "PPO_MINI_BATCH_SIZE",
                    "MICRO_BATCH_SIZE_PER_GPU",
                    "NUM_RESPONSES",
                    "NUM_EPOCHS",
                    "MAX_STEPS",
                    "LR",
                    "MAX_PROMPT_LEN",
                    "OVERLONG_PROMPT_POLICY",
                    "MAX_RESPONSE_LEN",
                    "TOP_K",
                    "SAVE_INTERVAL",
                    "EVAL_INTERVAL",
                ):
                    self.assertIn(f"export {variable}=", content)

    def test_opd_ta_cmt_use_requested_shared_defaults(self):
        for name in (
            "train_opd_b200.sh",
            "train_ta_b200.sh",
            "train_cmt_b200.sh",
        ):
            content = (REPO_ROOT / "scripts" / name).read_text(encoding="utf-8")
            with self.subTest(script=name):
                self.assertIn("LEARNING_RATE:-5.0e-6", content)
                self.assertIn("MAX_NEW_TOKENS:-4096", content)
                self.assertIn('NUM_RESPONSES="${NUM_RESPONSES:-4}"', content)
                self.assertIn(
                    'PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"', content
                )
                self.assertIn("MICRO_BATCH_SIZE:-16", content)
                self.assertIn('SAVE_INTERVAL="${SAVE_INTERVAL:-150}"', content)
                self.assertIn('EVAL_INTERVAL="${EVAL_INTERVAL:-150}"', content)
                self.assertIn("ROLLOUT_VLLM_GPU_MEMORY_UTILIZATION:-0.60", content)
                self.assertIn("ROLLOUT_VLLM_MAX_MODEL_LEN:-5200", content)
                self.assertIn('_DEFAULT_NUM_EPOCHS="3"', content)
                self.assertIn('_DEFAULT_NUM_EPOCHS="2"', content)
                self.assertIn('TOP_K="${TOP_K:-16}"', content)
        for method in ("opd", "ta", "cmt"):
            config = load_config(REPO_ROOT / "configs" / f"qwen3_b200_{method}.yaml")
            with self.subTest(config=method):
                self.assertEqual(config["selector"]["top_k"], 16)
                self.assertEqual(config["training"]["learning_rate"], 5.0e-6)
                self.assertEqual(config["rollout"]["max_new_tokens"], 4096)
                self.assertEqual(config["rollout"]["num_responses"], 4)
                self.assertEqual(config["training"]["ppo_mini_batch_size"], 64)
                self.assertEqual(config["training"]["micro_batch_size_per_gpu"], 16)
                self.assertEqual(config["training"]["epochs"], 3)
                self.assertEqual(config["training"]["save_interval"], 150)
                self.assertEqual(config["rollout"]["vllm"]["max_model_len"], 5200)
                self.assertEqual(
                    config["rollout"]["vllm"]["gpu_memory_utilization"], 0.60
                )
                self.assertEqual(config["training_evaluation"]["interval_steps"], 150)

    def test_grpo_launcher_uses_memory_safe_default_microbatch(self):
        content = (REPO_ROOT / "scripts" / "train_grpo_b200.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "MICRO_BATCH_SIZE_PER_GPU:-${MICRO_BATCH_SIZE:-1}",
            content,
        )
        self.assertIn("micro-batch only changes gradient accumulation", content)

    def test_iw_launcher_has_shared_dataset_presets_and_requested_defaults(self):
        content = (REPO_ROOT / "scripts" / "train_iw_b200.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("dapo_math|dapo-math|dapo)", content)
        self.assertIn("competition_math|competition-math|math)", content)
        self.assertIn(
            'MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-8}"', content
        )
        self.assertIn(
            'ROLLOUT_VLLM_GPU_MEMORY_UTILIZATION="${ROLLOUT_VLLM_GPU_MEMORY_UTILIZATION:-0.60}"',
            content,
        )

    def test_training_progress_launcher_dispatches_grpo(self):
        content = (REPO_ROOT / "scripts" / "plot_training_progress.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("grpo) add_plot_method grpo ;;", content)
        self.assertIn("iw|iw-opd) add_plot_method iw ;;", content)

    def test_common_config_accepts_new_model_and_data_aliases(self):
        environment = dict(os.environ)
        environment.update(
            {
                "PYTHON_BIN": sys.executable,
                "STORAGE_ROOT": "/storage",
                "STUDENT_MODEL": "/models/student",
                "TEACHER_MODEL": "/models/teacher",
                "TRAIN_DATA": "/datasets/train.parquet",
                "PROMPT_KEY": "question_text",
                "CUDA_VISIBLE_DEVICES": "0",
                "TMPDIR": str(REPO_ROOT.parent),
            }
        )
        for legacy in (
            "STUDENT_MODEL_PATH",
            "TEACHER_MODEL_PATH",
            "TRAIN_DATA_PATH",
            "TRAIN_PROMPT_KEY",
        ):
            environment.pop(legacy, None)
        completed = subprocess.run(
            [
                "bash",
                "-c",
                "source scripts/common_b200.sh; "
                "printf '%s|%s|%s|%s' \"$STUDENT_MODEL_PATH\" "
                '"$TEACHER_MODEL_PATH" "$TRAIN_DATA_PATH" '
                '"$TRAIN_PROMPT_KEY"',
            ],
            cwd=REPO_ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(
            completed.stdout,
            "/models/student|/models/teacher|/datasets/train.parquet|question_text",
        )

    def test_train_all_forwards_one_exact_shared_asset_selection(self):
        content = (REPO_ROOT / "scripts/train_all_b200.sh").read_text(encoding="utf-8")
        self.assertLess(content.index("USER CONFIG"), content.index("source "))
        with tempfile.TemporaryDirectory(dir=REPO_ROOT.parent) as temporary:
            temporary_path = Path(temporary)
            capture = temporary_path / "children.txt"
            fake_bash = temporary_path / "bash"
            fake_bash.write_text(
                "#!/bin/sh\n"
                'printf \'%s|%s|%s|%s|%s\\n\' "$1" "$STUDENT_MODEL" '
                '"$TEACHER_MODEL" "$TRAIN_DATA" "$PROMPT_KEY" '
                '>> "$CAPTURE_FILE"\n',
                encoding="utf-8",
            )
            fake_bash.chmod(0o755)
            environment = dict(os.environ)
            environment.update(
                {
                    "PATH": f"{temporary}:{environment['PATH']}",
                    "CAPTURE_FILE": str(capture),
                    "PYTHON_BIN": sys.executable,
                    "CUDA_VISIBLE_DEVICES": "0",
                    "STUDENT_MODEL": "/new/student",
                    "TEACHER_MODEL": "/new/teacher",
                    "TRAIN_DATA": "/new/train.parquet",
                    "PROMPT_KEY": "question",
                    # Stale legacy values must not win over the USER CONFIG.
                    "STUDENT_MODEL_PATH": "/stale/student",
                    "TEACHER_MODEL_PATH": "/stale/teacher",
                    "TRAIN_DATA_PATH": "/stale/train.parquet",
                    "TRAIN_PROMPT_KEY": "stale_prompt",
                }
            )
            subprocess.run(
                ["/bin/bash", "scripts/train_all_b200.sh"],
                cwd=REPO_ROOT,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            lines = capture.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 3)
        self.assertEqual(
            [Path(line.split("|", 1)[0]).name for line in lines],
            ["train_opd_b200.sh", "train_ta_b200.sh", "train_rac_b200.sh"],
        )
        for line in lines:
            self.assertEqual(
                line.split("|")[1:],
                [
                    "/new/student",
                    "/new/teacher",
                    "/new/train.parquet",
                    "question",
                ],
            )


if __name__ == "__main__":
    unittest.main()
