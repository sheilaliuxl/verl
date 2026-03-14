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
"""Tests for SFTTrainer utility methods: _compute_ray_rpc_train_steps, _interleave_batches."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from parameterized import parameterized
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData

from verl.trainer.sft_trainer_ray import SFTTrainer

# Reusable batch dicts for interleave tests (batch_size=4, seq_len=1).
_BATCH_00 = {"x": ((0,), (1,), (2,), (3,))}
_BATCH_01 = {"x": ((10,), (11,), (12,), (13,))}
_BATCH_02 = {"x": ((20,), (21,), (22,), (23,))}


def _make_trainer(**overrides):
    """Create a minimal SFTTrainer mock with just the fields needed for util methods."""
    defaults = dict(
        total_training_steps=256,
        steps_per_epoch=256,
        save_freq=-1,
        test_freq=-1,
        start_profile_step=-1,
        end_profile_step=-1,
        config=SimpleNamespace(trainer=SimpleNamespace(ray_rpc_train_steps=10)),
    )
    defaults.update(overrides)
    trainer = MagicMock(spec=SFTTrainer)
    for key, val in defaults.items():
        setattr(trainer, key, val)
    # Bind the real methods
    trainer._compute_ray_rpc_train_steps = SFTTrainer._compute_ray_rpc_train_steps.__get__(trainer)
    trainer._interleave_batches = SFTTrainer._interleave_batches.__get__(trainer)
    return trainer


class TestComputeRayRpcTrainSteps(unittest.TestCase):
    """Tests for SFTTrainer._compute_ray_rpc_train_steps."""

    @parameterized.expand(
        (
            # (name, trainer_overrides, global_step, expected)
            (
                "default_returns_1_step0",
                dict(config=SimpleNamespace(trainer=SimpleNamespace(ray_rpc_train_steps=1))),
                0,
                1,
            ),
            (
                "default_returns_1_step100",
                dict(config=SimpleNamespace(trainer=SimpleNamespace(ray_rpc_train_steps=1))),
                100,
                1,
            ),
            ("basic_multi_steps", dict(), 0, 10),
            ("bounded_by_remaining_6", dict(total_training_steps=256), 250, 6),
            ("bounded_by_remaining_1", dict(total_training_steps=256), 255, 1),
            ("epoch_used", dict(steps_per_epoch=100), 95, 5),
            ("epoch_no_op_at_boundary", dict(steps_per_epoch=100), 100, 10),
            ("epoch_no_op_mid_epoch", dict(steps_per_epoch=100), 50, 10),
            ("save_freq_no_op", dict(save_freq=50), 0, 10),
            ("save_freq_used", dict(save_freq=50), 45, 5),
            ("save_freq_no_op_at_boundary", dict(save_freq=50), 50, 10),
            ("test_freq_used", dict(test_freq=20), 15, 5),
            ("test_freq_no_op_at_boundary", dict(test_freq=20), 20, 10),
            ("both_freqs_test_used", dict(save_freq=50, test_freq=30), 25, 5),
            ("both_freqs_save_used", dict(save_freq=50, test_freq=30), 42, 8),
            ("profiler_start_used", dict(start_profile_step=100, end_profile_step=110), 95, 5),
            ("profiler_start_no_op", dict(start_profile_step=100, end_profile_step=110), 100, 10),
            ("profiler_end_used", dict(start_profile_step=100, end_profile_step=105), 102, 3),
            ("negative_freqs_ignored", dict(save_freq=-1, test_freq=-1), 0, 10),
            (
                "all_constraints",
                dict(
                    total_training_steps=103,
                    steps_per_epoch=50,
                    save_freq=50,
                    test_freq=30,
                    start_profile_step=100,
                    end_profile_step=110,
                ),
                97,
                3,
            ),
            ("returns_at_least_1", dict(total_training_steps=1), 0, 1),
        )
    )
    def test_compute_ray_rpc_train_steps(self, _name, trainer_overrides, global_step, expected_steps):
        trainer = _make_trainer(**trainer_overrides)
        self.assertEqual(trainer._compute_ray_rpc_train_steps(global_step), expected_steps)


class TestInterleaveBatches(unittest.TestCase):
    """Tests for SFTTrainer._interleave_batches."""

    def setUp(self):
        self.trainer = _make_trainer()

    def test_single_batch_passthrough(self):
        """Single batch is returned as-is."""
        batch = TensorDict({"input_ids": torch.arange(8).reshape(4, 2)}, batch_size=(4,))
        result = self.trainer._interleave_batches((batch,))
        self.assertIs(result, batch)

    @parameterized.expand(
        (
            # (name, batch_dicts, batch_size, expected_dict, expected_batch_size)
            (
                "two_batches",
                (_BATCH_00, _BATCH_01),
                4,
                {"x": ((0,), (10,), (1,), (11,), (2,), (12,), (3,), (13,))},
                8,
            ),
            (
                "three_batches",
                (_BATCH_00, _BATCH_01, _BATCH_02),
                4,
                {"x": ((0,), (10,), (20,), (1,), (11,), (21,), (2,), (12,), (22,), (3,), (13,), (23,))},
                12,
            ),
            (
                "multiple_fields",
                (
                    {"input_ids": ((1, 2), (3, 4)), "labels": ((10, 20), (30, 40))},
                    {"input_ids": ((5, 6), (7, 8)), "labels": ((50, 60), (70, 80))},
                ),
                2,
                {
                    "input_ids": ((1, 2), (5, 6), (3, 4), (7, 8)),
                    "labels": ((10, 20), (50, 60), (30, 40), (70, 80)),
                },
                4,
            ),
        )
    )
    def test_interleave(self, _name, batch_dicts, batch_size, expected_dict, expected_batch_size):
        batches = tuple(
            TensorDict(
                {k: torch.tensor(v) for k, v in d.items()},
                batch_size=(batch_size,),
            )
            for d in batch_dicts
        )
        result = self.trainer._interleave_batches(batches)

        self.assertEqual(result.batch_size, torch.Size((expected_batch_size,)))
        self.assertEqual(set(result.keys()), set(expected_dict.keys()))
        for key, expected_list in expected_dict.items():
            torch.testing.assert_close(result[key], torch.tensor(expected_list))

    def test_dp_chunking_correctness(self):
        """After interleave + DP chunk, each worker gets the correct slice per step."""
        b1 = TensorDict({"x": torch.tensor(_BATCH_00["x"])}, batch_size=(4,))
        b1["pad_mode"] = NonTensorData("no_padding")
        b1["max_token_len"] = NonTensorData(4096)
        b2 = TensorDict({"x": torch.tensor(_BATCH_01["x"])}, batch_size=(4,))
        b2["pad_mode"] = NonTensorData("no_padding")
        b2["max_token_len"] = NonTensorData(8192)
        result = self.trainer._interleave_batches((b1, b2))

        # NonTensorData copied from first batch (b2's different max_token_len is ignored)
        self.assertEqual(result["pad_mode"], "no_padding")
        self.assertEqual(result["max_token_len"], 4096)

        # Simulate 4 DP workers
        dp_size = 4
        num_steps = 2
        chunks = result["x"].chunk(dp_size)
        self.assertEqual(len(chunks), dp_size)
        for worker_id in range(dp_size):
            worker_data = chunks[worker_id]
            sub_batches = worker_data.chunk(num_steps)
            self.assertEqual(len(sub_batches), num_steps)
            # Worker should get its own sample from each batch
            self.assertEqual(sub_batches[0].item(), worker_id)  # from b1
            self.assertEqual(sub_batches[1].item(), worker_id + 10)  # from b2

    def test_single_nested_batch_passthrough(self):
        """Single nested tensor batch is returned as-is."""
        batch = TensorDict(
            {
                "x": torch.nested.as_nested_tensor(
                    [torch.tensor([1, 2]), torch.tensor([3, 4, 5])],
                    layout=torch.jagged,
                )
            },
            batch_size=(2,),
        )
        result = self.trainer._interleave_batches((batch,))
        self.assertIs(result, batch)

    def test_interleave_nested_tensors(self):
        """Interleave works with nested tensors and copies NonTensorData (meta_info)."""
        b1 = TensorDict(
            {
                "x": torch.nested.as_nested_tensor(
                    [torch.tensor([1, 2, 3]), torch.tensor([4, 5])],
                    layout=torch.jagged,
                )
            },
            batch_size=(2,),
        )
        b1["pad_mode"] = NonTensorData("no_padding")
        b1["max_token_len"] = NonTensorData(4096)
        b2 = TensorDict(
            {
                "x": torch.nested.as_nested_tensor(
                    [torch.tensor([10, 20]), torch.tensor([30, 40, 50, 60])],
                    layout=torch.jagged,
                )
            },
            batch_size=(2,),
        )
        b2["pad_mode"] = NonTensorData("no_padding")
        b2["max_token_len"] = NonTensorData(8192)
        result = self.trainer._interleave_batches((b1, b2))

        self.assertEqual(result.batch_size, torch.Size((4,)))
        self.assertTrue(result["x"].is_nested)
        # Interleaved order: b1[0], b2[0], b1[1], b2[1]
        torch.testing.assert_close(result["x"][0], torch.tensor([1, 2, 3]))
        torch.testing.assert_close(result["x"][1], torch.tensor([10, 20]))
        torch.testing.assert_close(result["x"][2], torch.tensor([4, 5]))
        torch.testing.assert_close(result["x"][3], torch.tensor([30, 40, 50, 60]))
        # NonTensorData copied from first batch
        self.assertEqual(result["pad_mode"], "no_padding")
        self.assertEqual(result["max_token_len"], 4096)


if __name__ == "__main__":
    unittest.main()
