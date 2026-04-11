# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from collections import deque

import torch
from parameterized import parameterized
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.ppo.metric_utils import (
    KEY_NUM_SEQS_CORRECTION_FACTOR,
    KEY_NUM_TOKENS_CORRECTION_FACTOR,
    KEY_ORIGINAL_BATCH_SIZE_PER_DP_GROUP,
    ZERO_ADV_EPS,
    aggregate_zero_adv_window,
    ceildiv,
    compute_gen_batch_multiplier,
    filter_zero_adv_batch,
    maybe_add_corrected_mfu,
    pull_and_merge_gen_batches,
)

EXPECTED_METRIC_KEYS = (
    "actor/filter_zero_adv/kept_ratio",
    "actor/filter_zero_adv/num_kept",
    "actor/filter_zero_adv/num_nonzero",
    "actor/filter_zero_adv/num_padded",
    "actor/filter_zero_adv/num_total",
)


def _make_simple_batch(batch_size, seq_len=4, label=""):
    """Helper to create a simple DataProto batch for pull_and_merge tests."""
    td = TensorDict(
        {
            "input_ids": torch.arange(batch_size * seq_len).reshape(batch_size, seq_len).float(),
            "attention_mask": torch.ones(batch_size, seq_len),
        },
        batch_size=(batch_size,),
    )
    return DataProto(batch=td)


def _make_batch_dicts(num_batches, batch_size=4, seq_len=4):
    """Create a list of batch_dicts (as a dataloader would yield)."""
    dicts = []
    for i in range(num_batches):
        td = TensorDict(
            {
                "input_ids": torch.full((batch_size, seq_len), float(i + 1)),
                "attention_mask": torch.ones(batch_size, seq_len),
            },
            batch_size=(batch_size,),
        )
        dicts.append(td)
    return dicts


def _make_batch(num_nonzero, num_zero, seq_len, attention_lengths=None):
    """Helper to construct a DataProto batch for filter_zero_adv_batch tests.

    Args:
        num_nonzero: Number of sequences with nonzero advantage.
        num_zero: Number of sequences with zero advantage.
        seq_len: Sequence length for all tensors.
        attention_lengths: Optional tuple of attention lengths per sequence.
            If None, all sequences have full attention.
    """
    bs = num_nonzero + num_zero
    advantages = torch.zeros(bs, seq_len)
    if num_nonzero > 0:
        advantages[:num_nonzero] = torch.randn(num_nonzero, seq_len).abs() + 1.0
    response_mask = torch.ones(bs, seq_len)

    if attention_lengths is not None:
        attention_mask = torch.zeros(bs, seq_len)
        for i, length in enumerate(attention_lengths):
            attention_mask[i, :length] = 1.0
    else:
        attention_mask = torch.ones(bs, seq_len)

    td = TensorDict(
        {
            "advantages": advantages,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
        },
        batch_size=(bs,),
    )
    return DataProto(batch=td)


class TestCeildiv(unittest.TestCase):
    @parameterized.expand(
        (
            (1, 1, 1),
            (4, 2, 2),
            (5, 2, 3),
            (10, 3, 4),
            (10, 4, 3),
            (7, 1, 7),
            (0, 5, 0),
            (1, 5, 1),
            (128, 128, 1),
            (129, 128, 2),
        )
    )
    def test_ceildiv(self, a, b, expected):
        self.assertEqual(ceildiv(a, b), expected)


class TestMaybeAddCorrectedMfu(unittest.TestCase):
    """Tests for maybe_add_corrected_mfu in metric_utils.py."""

    @parameterized.expand(
        (
            ("absent", {}),
            ("explicit_none", {KEY_NUM_TOKENS_CORRECTION_FACTOR: None}),
        )
    )
    def test_no_correction(self, _name, meta_info):
        """When correction factor is absent or None, no corrected metric is added."""
        metrics = {"perf/mfu/actor": 0.25, "perf/mfu/critic": 0.10}
        maybe_add_corrected_mfu(metrics, meta_info)
        self.assertEqual(sorted(metrics.keys()), ["perf/mfu/actor", "perf/mfu/critic"])
        self.assertAlmostEqual(metrics["perf/mfu/actor"], 0.25)
        self.assertAlmostEqual(metrics["perf/mfu/critic"], 0.10)

    @parameterized.expand(
        (
            ("full_batch", 1.0, 0.25, 0.25),
            ("half_filtered", 0.5, 0.30, 0.15),
            ("quarter_filtered", 0.25, 0.40, 0.10),
            ("slight_filter", 0.8, 0.20, 0.16),
            ("zero_mfu", 0.5, 0.0, 0.0),
        )
    )
    def test_correction_applied(self, _name, token_correction, mfu, expected_corrected):
        """Corrected MFU = original MFU * token_correction_factor; other keys unchanged."""
        metrics = {"perf/mfu/actor": mfu, "perf/mfu/critic": 0.10, "loss": 1.5}
        meta_info = {KEY_NUM_TOKENS_CORRECTION_FACTOR: token_correction}
        maybe_add_corrected_mfu(metrics, meta_info)
        self.assertEqual(
            sorted(metrics.keys()), ["loss", "perf/mfu/actor", "perf/mfu/actor_corrected", "perf/mfu/critic"]
        )
        self.assertAlmostEqual(metrics["perf/mfu/actor_corrected"], expected_corrected, places=6)
        self.assertAlmostEqual(metrics["perf/mfu/actor"], mfu)
        self.assertAlmostEqual(metrics["perf/mfu/critic"], 0.10)
        self.assertAlmostEqual(metrics["loss"], 1.5)

    def test_correction_overwrites_existing(self):
        """When perf/mfu/actor_corrected already exists, it is overwritten."""
        metrics = {"perf/mfu/actor": 0.30, "perf/mfu/actor_corrected": 999.0}
        meta_info = {KEY_NUM_TOKENS_CORRECTION_FACTOR: 0.5}
        maybe_add_corrected_mfu(metrics, meta_info)
        self.assertEqual(sorted(metrics.keys()), ["perf/mfu/actor", "perf/mfu/actor_corrected"])
        self.assertAlmostEqual(metrics["perf/mfu/actor_corrected"], 0.15, places=6)
        self.assertAlmostEqual(metrics["perf/mfu/actor"], 0.30)


class TestComputeGenBatchMultiplier(unittest.TestCase):
    """Tests for aggregate_zero_adv_window and compute_gen_batch_multiplier."""

    @parameterized.expand(
        (
            ("empty", [], "median", 0.0),
            ("min_basic", [0.3, 0.5, 0.7], "min", 0.3),
            ("mean_basic", [0.2, 0.4, 0.6], "mean", 0.4),
            ("median_odd", [0.3, 0.9, 0.5], "median", 0.5),
            ("median_even", [0.3, 0.5, 0.7, 0.9], "median", 0.6),
        )
    )
    def test_aggregate_zero_adv_window(self, _name, window_values, stats, expected):
        self.assertAlmostEqual(aggregate_zero_adv_window(deque(window_values), stats), expected)

    @parameterized.expand(("max", "invalid", "sum"))
    def test_invalid_stats_raises(self, stats):
        with self.assertRaises(ValueError):
            aggregate_zero_adv_window(deque([0.5]), stats)

    @parameterized.expand(
        (
            # (name, window_values, max_num_gen_batches, stats, expected_multiplier)
            ("empty_window", [], 3, "min", 1),
            ("all_zeros", [0.0, 0.0, 0.0, 0.0, 0.0], 3, "min", 1),
            ("min_p_0.3_cap3", [0.5, 0.4, 0.3, 0.6, 0.7], 3, "min", 1),
            ("min_p_0.5_cap3", [0.6, 0.5, 0.7, 0.6, 0.55], 3, "min", 2),
            ("min_p_0.6_cap3", [0.7, 0.8, 0.6, 0.65, 0.75], 3, "min", 2),
            ("min_p_0.7_cap3", [0.8, 0.75, 0.85, 0.7, 0.9], 3, "min", 3),
            ("min_p_0.7_cap5", [0.8, 0.75, 0.85, 0.7, 0.9], 5, "min", 3),
            ("min_p_0.8_cap3", [0.85, 0.9, 0.8, 0.82, 0.88], 3, "min", 3),
            ("min_p_0.8_cap5", [0.85, 0.9, 0.82, 0.8, 0.88], 5, "min", 5),
            ("min_p_0.9_cap5", [0.92, 0.95, 0.9, 0.91, 0.93], 5, "min", 5),
            ("min_p_1.0_cap3", [1.0, 1.0, 1.0], 3, "min", 3),
            ("min_single_value", [0.6], 4, "min", 2),
            ("min_mixed_with_zero", [0.5, 0.0, 0.6], 3, "min", 1),
            ("min_warmup_zeros_then_real", [0.0, 0.0, 0.0, 0.6, 0.7], 3, "min", 1),
            ("median_p_0.7_cap3", [0.7, 0.8, 0.6, 0.65, 0.75], 3, "median", 3),
            ("median_p_0.8_cap5", [0.85, 0.9, 0.82, 0.8, 0.88], 5, "median", 5),
            ("median_mixed_with_zero", [0.5, 0.0, 0.6], 3, "median", 2),
            ("median_warmup_zeros_then_real", [0.0, 0.0, 0.0, 0.6, 0.7], 3, "median", 1),
            ("median_single", [0.8], 5, "median", 5),
            ("mean_p_0.8_cap5", [0.85, 0.9, 0.82, 0.8, 0.88], 5, "mean", 5),
            ("mean_mixed_with_zero", [0.5, 0.0, 0.6], 3, "mean", 1),
            ("mean_warmup_zeros_then_real", [0.0, 0.0, 0.0, 0.6, 0.7], 3, "mean", 1),
        )
    )
    def test_compute_gen_batch_multiplier(self, _name, window_values, max_cap, stats, expected):
        self.assertEqual(compute_gen_batch_multiplier(deque(window_values), max_cap, stats), expected)


class TestPullAndMergeGenBatches(unittest.TestCase):
    """Tests for pull_and_merge_gen_batches in metric_utils.py."""

    # ------------------------------------------------------------------ #
    #  No-op cases: original batch returned as-is
    # ------------------------------------------------------------------ #

    @parameterized.expand(
        (
            ("num_extra_zero", 4, 0),
            ("num_extra_negative", 4, -1),
            ("epoch_boundary_empty", 4, 2),
        )
    )
    def test_no_op(self, _name, first_bs, num_extra):
        """Original batch returned unchanged when no extra batches pulled."""
        first = _make_simple_batch(first_bs)
        result, num_pulled = pull_and_merge_gen_batches(first, iter([]), num_extra)
        self.assertIs(result, first)
        self.assertEqual(num_pulled, 0)

    # ------------------------------------------------------------------ #
    #  Pull cases: extra batches merged
    # ------------------------------------------------------------------ #

    @parameterized.expand(
        (
            # (name, first_bs, extra_bs, num_avail, num_extra, expected_pulled, remaining, expected_ids)
            (
                "pull_one",
                4,
                4,
                1,
                1,
                1,
                0,
                torch.cat(
                    [
                        torch.arange(16).reshape(4, 4).float(),
                        torch.full((4, 4), 1.0),
                    ]
                ),
            ),
            (
                "pull_two",
                4,
                4,
                3,
                2,
                2,
                1,
                torch.cat(
                    [
                        torch.arange(16).reshape(4, 4).float(),
                        torch.full((4, 4), 1.0),
                        torch.full((4, 4), 2.0),
                    ]
                ),
            ),
            (
                "epoch_boundary_partial",
                4,
                4,
                1,
                3,
                1,
                0,
                torch.cat(
                    [
                        torch.arange(16).reshape(4, 4).float(),
                        torch.full((4, 4), 1.0),
                    ]
                ),
            ),
            (
                "different_batch_sizes",
                3,
                5,
                1,
                1,
                1,
                0,
                torch.cat(
                    [
                        torch.arange(12).reshape(3, 4).float(),
                        torch.full((5, 4), 1.0),
                    ]
                ),
            ),
        )
    )
    def test_pull(self, _name, first_bs, extra_bs, num_avail, num_extra, expected_pulled, remaining, expected_ids):
        seq_len = 4
        first = _make_simple_batch(first_bs, seq_len=seq_len)
        extra_dicts = _make_batch_dicts(num_avail, batch_size=extra_bs, seq_len=seq_len)
        dl_iter = iter(extra_dicts)
        result, num_pulled = pull_and_merge_gen_batches(first, dl_iter, num_extra)
        self.assertEqual(num_pulled, expected_pulled)
        self.assertEqual(len(list(dl_iter)), remaining)
        torch.testing.assert_close(result.batch["input_ids"], expected_ids)

    def test_values_preserved(self):
        """Verify tensor values are preserved after merge with explicit values."""
        first = _make_simple_batch(2, seq_len=3)
        first.batch["input_ids"] = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        extra_dicts = _make_batch_dicts(1, batch_size=2, seq_len=3)
        extra_dicts[0]["input_ids"] = torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
        result, num_pulled = pull_and_merge_gen_batches(first, iter(extra_dicts), 1)
        self.assertEqual(num_pulled, 1)
        self.assertEqual(result.batch.batch_size[0], 4)
        torch.testing.assert_close(
            result.batch["input_ids"],
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]),
        )


class TestFilterZeroAdvBatch(unittest.TestCase):
    """Tests for filter_zero_adv_batch in metric_utils.py."""

    # ------------------------------------------------------------------ #
    #  No-op: batch returned unchanged (select logic)
    # ------------------------------------------------------------------ #

    @parameterized.expand(
        (
            ("all_nonzero_dp2", 8, 0, 4, 2),
            ("all_nonzero_dp4", 8, 0, 4, 4),
            ("all_nonzero_dp8", 16, 0, 8, 8),
        )
    )
    def test_filter_no_op_all_nonzero(self, _name, num_nonzero, num_zero, seq_len, dp_size):
        """When all sequences have nonzero advantage, batch is returned unchanged."""
        batch = _make_batch(num_nonzero, num_zero, seq_len)
        filtered, metrics = filter_zero_adv_batch(batch, dp_size)

        bs = num_nonzero + num_zero
        self.assertIs(filtered, batch)
        self.assertEqual(sorted(metrics.keys()), list(EXPECTED_METRIC_KEYS))
        self.assertEqual(metrics["actor/filter_zero_adv/num_total"], bs)
        self.assertEqual(metrics["actor/filter_zero_adv/num_kept"], bs)
        self.assertAlmostEqual(metrics["actor/filter_zero_adv/kept_ratio"], 1.0)
        self.assertNotIn(KEY_ORIGINAL_BATCH_SIZE_PER_DP_GROUP, filtered.meta_info)

    @parameterized.expand(
        (
            ("need_3_pad_have_1", 9, 1, 4, 4),
            ("need_3_pad_have_2", 13, 2, 4, 8),
            ("need_3_pad_have_3", 13, 3, 4, 8),
            ("need_6_pad_have_6", 10, 6, 4, 8),
            ("no_zeros_unaligned", 7, 0, 4, 4),
        )
    )
    def test_filter_no_op_not_enough_zeros_to_pad(self, _name, num_nonzero, num_zero, seq_len, dp_size):
        """When there aren't enough zero-adv samples for dp_size alignment, skip filtering."""
        batch = _make_batch(num_nonzero, num_zero, seq_len)
        filtered, metrics = filter_zero_adv_batch(batch, dp_size)

        bs = num_nonzero + num_zero
        self.assertIs(filtered, batch)
        self.assertEqual(sorted(metrics.keys()), list(EXPECTED_METRIC_KEYS))
        self.assertEqual(metrics["actor/filter_zero_adv/num_total"], bs)
        self.assertEqual(metrics["actor/filter_zero_adv/num_kept"], bs)
        self.assertAlmostEqual(metrics["actor/filter_zero_adv/kept_ratio"], 1.0)
        self.assertNotIn(KEY_ORIGINAL_BATCH_SIZE_PER_DP_GROUP, filtered.meta_info)

    def test_filter_no_op_empty_response_mask(self):
        """When original_num_tokens == 0, skip filtering."""
        bs, seq_len, dp_size = 8, 4, 2
        td = TensorDict(
            {
                "advantages": torch.zeros(bs, seq_len),
                "attention_mask": torch.ones(bs, seq_len),
                "response_mask": torch.zeros(bs, seq_len),
            },
            batch_size=(bs,),
        )
        batch = DataProto(batch=td)
        filtered, metrics = filter_zero_adv_batch(batch, dp_size)

        self.assertIs(filtered, batch)
        self.assertEqual(sorted(metrics.keys()), list(EXPECTED_METRIC_KEYS))
        self.assertEqual(metrics["actor/filter_zero_adv/num_total"], bs)
        self.assertEqual(metrics["actor/filter_zero_adv/num_kept"], bs)
        self.assertAlmostEqual(metrics["actor/filter_zero_adv/kept_ratio"], 1.0)
        self.assertNotIn(KEY_ORIGINAL_BATCH_SIZE_PER_DP_GROUP, filtered.meta_info)

    # ------------------------------------------------------------------ #
    #  Filtered: sequences are actually removed
    # ------------------------------------------------------------------ #

    @parameterized.expand(
        (
            # (name, num_nz, num_z, seq, dp, kept, padded, orig_bs_per_dp, seq_corr, token_corr)
            ("6nz_4z_dp4", 6, 4, 4, 4, 8, 2, 3, 0.8, 0.8),
            ("8nz_4z_dp4_aligned", 8, 4, 4, 4, 8, 0, 3, 2 / 3, 2 / 3),
            ("3nz_7z_dp1", 3, 7, 4, 1, 3, 0, 10, 0.3, 0.3),
            ("5nz_5z_dp2", 5, 5, 4, 2, 6, 1, 5, 0.6, 0.6),
            ("4nz_12z_dp4", 4, 12, 8, 4, 4, 0, 4, 0.25, 0.25),
            ("1nz_15z_dp4", 1, 15, 4, 4, 4, 3, 4, 0.25, 0.25),
        )
    )
    def test_filtered_mixed(
        self,
        _name,
        num_nonzero,
        num_zero,
        seq_len,
        dp_size,
        expected_kept,
        expected_padded,
        expected_original_bs_per_dp,
        expected_seq_correction,
        expected_token_correction,
    ):
        """Nonzero-adv sequences are kept, zero-adv removed (with dp_size padding)."""
        batch = _make_batch(num_nonzero, num_zero, seq_len)
        filtered, metrics = filter_zero_adv_batch(batch, dp_size)

        bs = num_nonzero + num_zero
        # Metrics
        self.assertEqual(sorted(metrics.keys()), list(EXPECTED_METRIC_KEYS))
        self.assertEqual(metrics["actor/filter_zero_adv/num_total"], bs)
        self.assertEqual(metrics["actor/filter_zero_adv/num_nonzero"], num_nonzero)
        self.assertEqual(metrics["actor/filter_zero_adv/num_kept"], expected_kept)
        self.assertEqual(metrics["actor/filter_zero_adv/num_padded"], expected_padded)
        self.assertAlmostEqual(metrics["actor/filter_zero_adv/kept_ratio"], expected_kept / bs)
        self.assertEqual(filtered.batch["advantages"].shape[0], expected_kept)
        # meta_info
        self.assertEqual(filtered.meta_info[KEY_ORIGINAL_BATCH_SIZE_PER_DP_GROUP], expected_original_bs_per_dp)
        self.assertAlmostEqual(filtered.meta_info[KEY_NUM_SEQS_CORRECTION_FACTOR], expected_seq_correction)
        self.assertAlmostEqual(
            filtered.meta_info[KEY_NUM_TOKENS_CORRECTION_FACTOR], expected_token_correction, places=6
        )

    @parameterized.expand(
        (
            # (name, bs, seq, dp, kept, orig_bs_per_dp, seq_corr, token_corr)
            ("16x8_dp4", 16, 8, 4, 4, 4, 0.25, 0.25),
            ("8x4_dp2", 8, 4, 2, 2, 4, 0.25, 0.25),
            ("32x4_dp8", 32, 4, 8, 8, 4, 0.25, 0.25),
        )
    )
    def test_filtered_all_zero_keeps_dp_size(
        self,
        _name,
        bs,
        seq_len,
        dp_size,
        expected_kept,
        expected_original_bs_per_dp,
        expected_seq_correction,
        expected_token_correction,
    ):
        """When all sequences have zero advantage, keep dp_size shortest samples."""
        attention_lengths = tuple(range(1, bs + 1))
        batch = _make_batch(0, bs, seq_len, attention_lengths=attention_lengths)

        filtered, metrics = filter_zero_adv_batch(batch, dp_size)

        # Metrics
        self.assertEqual(sorted(metrics.keys()), list(EXPECTED_METRIC_KEYS))
        self.assertEqual(metrics["actor/filter_zero_adv/num_total"], bs)
        self.assertEqual(metrics["actor/filter_zero_adv/num_nonzero"], 0)
        self.assertEqual(metrics["actor/filter_zero_adv/num_kept"], expected_kept)
        self.assertAlmostEqual(metrics["actor/filter_zero_adv/kept_ratio"], expected_kept / bs)
        self.assertEqual(filtered.batch["advantages"].shape[0], expected_kept)
        kept_lengths = filtered.batch["attention_mask"].sum(dim=-1)
        self.assertTrue((kept_lengths <= dp_size).all())
        # meta_info
        self.assertEqual(filtered.meta_info[KEY_ORIGINAL_BATCH_SIZE_PER_DP_GROUP], expected_original_bs_per_dp)
        self.assertAlmostEqual(filtered.meta_info[KEY_NUM_SEQS_CORRECTION_FACTOR], expected_seq_correction)
        self.assertAlmostEqual(
            filtered.meta_info[KEY_NUM_TOKENS_CORRECTION_FACTOR], expected_token_correction, places=6
        )

    # ------------------------------------------------------------------ #
    #  With ppo_mini_batch_size: align to dp_size * K
    # ------------------------------------------------------------------ #

    @parameterized.expand(
        (
            # (name, num_nz, num_z, seq, dp, mini_bs, kept, padded, orig_bs_per_dp)
            ("128_dp8_mini32_k1", 90, 38, 4, 8, 32, 96, 6, 16),
            ("128_dp8_mini4_k4", 90, 38, 4, 8, 4, 96, 6, 16),
            ("64_dp4_mini8_k2", 40, 24, 4, 4, 8, 40, 0, 16),
            ("32_dp4_mini4_k2", 20, 12, 4, 4, 4, 24, 4, 8),
            ("32_dp4_mini2_few_nz", 3, 29, 4, 4, 2, 4, 1, 8),
        )
    )
    def test_filtered_with_ppo_mini_batch_size(
        self,
        _name,
        num_nonzero,
        num_zero,
        seq_len,
        dp_size,
        ppo_mini_batch_size,
        expected_kept,
        expected_padded,
        expected_original_bs_per_dp,
    ):
        """With ppo_mini_batch_size, align to dp_size * K for even mini-batch distribution."""
        batch = _make_batch(num_nonzero, num_zero, seq_len)
        filtered, metrics = filter_zero_adv_batch(batch, dp_size, ppo_mini_batch_size=ppo_mini_batch_size)

        bs = num_nonzero + num_zero
        self.assertEqual(metrics["actor/filter_zero_adv/num_total"], bs)
        self.assertEqual(metrics["actor/filter_zero_adv/num_nonzero"], num_nonzero)
        self.assertEqual(metrics["actor/filter_zero_adv/num_kept"], expected_kept)
        self.assertEqual(metrics["actor/filter_zero_adv/num_padded"], expected_padded)
        self.assertEqual(filtered.batch["advantages"].shape[0], expected_kept)
        self.assertEqual(filtered.meta_info[KEY_ORIGINAL_BATCH_SIZE_PER_DP_GROUP], expected_original_bs_per_dp)
        # Verify alignment: kept must be divisible by dp_size * align_opt_steps
        bs_per_dp = ceildiv(bs, dp_size)
        k_original = ceildiv(bs_per_dp, ppo_mini_batch_size)
        align_opt_steps = min(k_original, max(1, ceildiv(num_nonzero, dp_size)))
        self.assertEqual(expected_kept % (dp_size * align_opt_steps), 0)

    @parameterized.expand(
        (
            # All zero with ppo_mini_batch_size: align to dp_size * K
            # 32 total, dp=4, mini_bs=4 → bs_per_dp=8, K=2
            # 0 nz → align_opt_steps=min(2, max(1, ceil(0/4)))=min(2,1)=1, align=4
            ("32_dp4_mini4_all_zero", 32, 4, 4, 4),
            # 16 total, dp=2, mini_bs=4 → bs_per_dp=8, K=2
            # 0 nz → align_opt_steps=1, align=2
            ("16_dp2_mini4_all_zero", 16, 4, 2, 4),
        )
    )
    def test_filtered_all_zero_with_ppo_mini_batch_size(self, _name, bs, seq_len, dp_size, ppo_mini_batch_size):
        """All-zero with ppo_mini_batch_size: align_opt_steps capped at 1, keep dp_size samples."""
        attention_lengths = tuple(range(1, bs + 1))
        batch = _make_batch(0, bs, seq_len, attention_lengths=attention_lengths)

        filtered, metrics = filter_zero_adv_batch(batch, dp_size, ppo_mini_batch_size=ppo_mini_batch_size)

        # align_opt_steps = min(K, max(1, ceil(0/dp))) = 1, so align = dp_size
        self.assertEqual(metrics["actor/filter_zero_adv/num_kept"], dp_size)
        self.assertEqual(filtered.batch["advantages"].shape[0], dp_size)

    # ------------------------------------------------------------------ #
    #  Edge cases: eps threshold, response_mask masking
    # ------------------------------------------------------------------ #

    @parameterized.expand(
        (
            ("half_eps", ZERO_ADV_EPS * 0.5, 0),
            ("just_below_eps", ZERO_ADV_EPS * 0.99, 0),
            ("at_eps", ZERO_ADV_EPS, 8),  # >= check: exactly eps is nonzero
            ("double_eps", ZERO_ADV_EPS * 2.0, 8),
            ("well_above", 1e-4, 8),
        )
    )
    def test_advantage_eps_threshold(self, _name, adv_value, expected_nonzero):
        """Advantages are classified as zero/nonzero based on ZERO_ADV_EPS threshold."""
        bs, seq_len, dp_size = 8, 4, 2
        td = TensorDict(
            {
                "advantages": torch.full((bs, seq_len), adv_value),
                "attention_mask": torch.ones(bs, seq_len),
                "response_mask": torch.ones(bs, seq_len),
            },
            batch_size=(bs,),
        )
        batch = DataProto(batch=td)
        _, metrics = filter_zero_adv_batch(batch, dp_size)

        self.assertEqual(metrics["actor/filter_zero_adv/num_nonzero"], expected_nonzero)

    def test_response_mask_determines_zero_adv(self):
        """Only response tokens matter for zero-adv detection (advantages * response_mask)."""
        bs, seq_len, dp_size = 4, 8, 2
        response_mask = torch.zeros(bs, seq_len)
        response_mask[2:, 4:] = 1.0  # only last 2 sequences have response tokens
        td = TensorDict(
            {
                "advantages": torch.ones(bs, seq_len) * 10.0,
                "attention_mask": torch.ones(bs, seq_len),
                "response_mask": response_mask,
            },
            batch_size=(bs,),
        )
        batch = DataProto(batch=td)
        _, metrics = filter_zero_adv_batch(batch, dp_size)

        self.assertEqual(metrics["actor/filter_zero_adv/num_nonzero"], 2)

    # ------------------------------------------------------------------ #
    #  Padding selects shortest
    # ------------------------------------------------------------------ #

    def test_padding_selects_shortest_zero_adv(self):
        """When padding is needed, the shortest zero-adv samples (by attention_mask) are chosen."""
        seq_len, dp_size = 8, 4
        # 5 nonzero (len=8 each) + 5 zero with varying attention lengths
        zero_adv_lengths = (2, 6, 1, 4, 3)
        attention_lengths = (8, 8, 8, 8, 8) + zero_adv_lengths
        batch = _make_batch(5, 5, seq_len, attention_lengths=attention_lengths)

        filtered, metrics = filter_zero_adv_batch(batch, dp_size)

        # 5 → next multiple of 4 = 8, so 3 pads needed (shortest: len=1, 2, 3)
        self.assertEqual(metrics["actor/filter_zero_adv/num_kept"], 8)
        kept_attn_lengths = sorted(filtered.batch["attention_mask"].sum(dim=-1).tolist())
        # 5 nonzero (len=8 each) + 3 shortest zero-adv pads (len=1, 2, 3)
        self.assertEqual(kept_attn_lengths, [1.0, 2.0, 3.0, 8.0, 8.0, 8.0, 8.0, 8.0])


if __name__ == "__main__":
    unittest.main()
