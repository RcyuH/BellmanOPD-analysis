from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

import torch

from b200_experiment.selector_logging import SelectedTokenLogger, TokenScoreStatsLogger


class FakeTokenizer:
    @staticmethod
    def convert_ids_to_tokens(ids):
        return [f"tok-{item}" for item in ids]


class SelectorLoggingTests(unittest.TestCase):
    def test_ta_selected_tokens_are_incrementally_gzipped(self):
        with tempfile.TemporaryDirectory() as temporary:
            mask = torch.tensor([[True, False], [False, True]])
            diagnostics = {
                key: torch.arange(4, dtype=torch.float32).reshape(2, 2)
                for key in ("D", "C", "D_norm", "C_norm", "s_TA")
            }
            logger = SelectedTokenLogger(
                temporary, FakeTokenizer(), "ta", chunk_steps=2
            )
            count = logger.write(
                step=1,
                dataset_indices=[10, 11],
                sample_ids=["a", "b"],
                response_ids=torch.tensor([[5, 6], [7, 8]]),
                selected_mask=mask,
                diagnostics=diagnostics,
            )
            self.assertEqual(count, 2)
            path = next((Path(temporary) / "selector_scores").glob("*.jsonl.gz"))
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle]
            self.assertEqual(rows[0]["token_text"], "tok-5")
            self.assertEqual(rows[1]["sample_id"], "b")
            self.assertIn("s_TA", rows[0])

    def test_distributed_ranks_use_independent_gzip_members(self):
        with tempfile.TemporaryDirectory() as temporary:
            mask = torch.tensor([[True]])
            diagnostics = {
                key: torch.ones(1, 1) for key in ("D", "C", "D_norm", "C_norm", "s_TA")
            }
            for rank in (0, 1):
                logger = SelectedTokenLogger(
                    temporary,
                    FakeTokenizer(),
                    "ta",
                    rank=rank,
                    world_size=2,
                )
                logger.write(
                    step=1,
                    dataset_indices=[10 + rank],
                    sample_ids=[str(rank)],
                    response_ids=torch.tensor([[5 + rank]]),
                    selected_mask=mask,
                    diagnostics=diagnostics,
                    batch_index_offset=rank,
                )
            paths = sorted((Path(temporary) / "selector_scores").glob("*.jsonl.gz"))
            self.assertEqual(len(paths), 2)
            self.assertIn("rank-00000", paths[0].name)
            self.assertIn("rank-00001", paths[1].name)

    def test_cmt_selected_tokens_include_bounded_kernel_diagnostics(self):
        with tempfile.TemporaryDirectory() as temporary:
            mask = torch.tensor([[True, False]])
            keys = (
                "gain",
                "support_common_mass",
                "conditional_support_common_mass",
                "alignment",
                "transition_weight",
                "support_coverage",
                "teacher_deficit",
                "marginal_flux",
                "successor_excess",
                "sequential_gain",
                "learning_value",
                "s_CMT",
            )
            diagnostics = {key: torch.ones(1, 2) for key in keys}
            logger = SelectedTokenLogger(
                temporary, FakeTokenizer(), "cmt", chunk_steps=2
            )
            count = logger.write(
                step=1,
                dataset_indices=[10],
                sample_ids=["cmt"],
                response_ids=torch.tensor([[5, 6]]),
                selected_mask=mask,
                diagnostics=diagnostics,
            )
            self.assertEqual(count, 1)
            path = next((Path(temporary) / "selector_scores").glob("*.jsonl.gz"))
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                row = json.loads(next(handle))
            self.assertEqual(row["token_text"], "tok-5")
            self.assertEqual(row["transition_weight"], 1.0)
            self.assertEqual(row["conditional_support_common_mass"], 1.0)

    def test_compact_rac_stats_cover_all_values_and_bound_raw_sample(self):
        with tempfile.TemporaryDirectory() as temporary:
            diagnostics = {
                key: torch.linspace(0, 1, 100)
                for key in ("g", "alignment", "V", "z", "w")
            }
            logger = TokenScoreStatsLogger(
                temporary, "rac", interval=50, bins=10, raw_sample_size=7
            )
            path = logger.write(1, 100, diagnostics)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["scores"]["V"]["count"], 100)
            self.assertEqual(sum(payload["scores"]["V"]["histogram"]["counts"]), 100)
            self.assertEqual(len(payload["scores"]["V"]["sample"]), 7)
            self.assertIsNone(logger.write(2, 100, diagnostics))

    def test_compact_opd_stats_record_uniform_all_token_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            logger = TokenScoreStatsLogger(temporary, "opd", interval=50, bins=10)
            path = logger.write(1, 100, {"w": torch.ones(17)})
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(tuple(payload["scores"]), ("w",))
            self.assertEqual(payload["scores"]["w"]["count"], 17)
            self.assertEqual(payload["scores"]["w"]["mean"], 1.0)

    def test_compact_cmt_stats_accept_all_selector_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            keys = (
                "s_CMT", "gain", "support_reverse_kl", "support_common_mass",
                "alignment", "transition_weight", "support_coverage",
                "coverage_correction", "teacher_deficit", "marginal_flux",
                "common_mass_derivative", "R", "M", "V", "H",
                "sequential_gain", "learning_value", "w",
            )
            diagnostics = {key: torch.linspace(0, 1, 9) for key in keys}
            logger = TokenScoreStatsLogger(temporary, "cmt", interval=10, bins=5)
            path = logger.write(1, 10, diagnostics)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(set(payload["scores"]), set(keys))
            self.assertEqual(payload["scores"]["learning_value"]["count"], 9)


if __name__ == "__main__":
    unittest.main()
