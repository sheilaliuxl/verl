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

"""Unit tests for per-prompt history tracking and historical injection."""

import unittest

import numpy as np
import torch
from parameterized import parameterized
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.ppo.prompt_history import (
    KEY_PROMPT_ID,
    STAT_INJECT_COUNT,
    STAT_MISS_COUNT,
    group_prompt_by_uid,
    inject_historical_responses,
    new_prompt_entry,
    update_prompt_history,
)


def _make_batch(
    n_prompts: int,
    n_per_prompt: int,
    prompt_len: int,
    response_len: int,
    scores: tuple[tuple[float, ...], ...],
    prompt_ids: tuple[int, ...] | None = None,
):
    """Build a DataProto batch mimicking the structure after balance_batch.

    Args:
        n_prompts: Number of unique prompts.
        n_per_prompt: Number of responses per prompt (n in GRPO).
        prompt_len: Length of prompt portion.
        response_len: Length of response portion.
        scores: Per-prompt tuple of per-response scores, shape [n_prompts][n_per_prompt].
        prompt_ids: Optional explicit prompt IDs. Defaults to 0..n_prompts-1.
    """
    bs = n_prompts * n_per_prompt
    seq_len = prompt_len + response_len

    if prompt_ids is None:
        prompt_ids = tuple(range(n_prompts))

    # Build uid and index arrays (scattered order like after balance_batch)
    uids = []
    indices = []
    for p in range(n_prompts):
        uid = f"uid_{prompt_ids[p]}"
        for _ in range(n_per_prompt):
            uids.append(uid)
            indices.append(prompt_ids[p])

    # Responses: fill with token IDs 10+i so we can verify replacement
    responses = torch.zeros(bs, response_len, dtype=torch.long)
    for i in range(bs):
        responses[i, :response_len] = torch.arange(10 + i, 10 + i + response_len)

    # Prompts
    prompts = torch.ones(bs, prompt_len, dtype=torch.long) * 5

    # Attention mask: 1 for prompt + response, 0 for nothing (no padding initially)
    attention_mask = torch.ones(bs, seq_len)

    # Response mask: all 1s (no padding initially)
    response_mask = torch.ones(bs, response_len)

    # rm_scores: place score at last response token (token_level_scores style)
    rm_scores = torch.zeros(bs, response_len)
    flat_idx = 0
    for p in range(n_prompts):
        for r in range(n_per_prompt):
            rm_scores[flat_idx, response_len - 1] = scores[p][r]
            flat_idx += 1

    # token_level_scores (same as rm_scores for these tests)
    token_level_scores = rm_scores.clone()

    td = TensorDict(
        {
            "responses": responses,
            "prompts": prompts,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "rm_scores": rm_scores,
            "token_level_scores": token_level_scores,
        },
        batch_size=(bs,),
    )

    return DataProto(
        batch=td,
        non_tensor_batch={
            "uid": np.array(uids, dtype=object),
            KEY_PROMPT_ID: np.array(indices, dtype=object),
        },
    )


class TestGroupByUid(unittest.TestCase):
    @parameterized.expand(
        (
            # (name, uids, expected_groups)
            ("basic", ("a", "a", "a", "b", "b", "b"), {"a": [0, 1, 2], "b": [3, 4, 5]}),
            ("scattered", ("a", "b", "a", "b", "a", "b"), {"a": [0, 2, 4], "b": [1, 3, 5]}),
            ("single", ("x", "x", "x"), {"x": [0, 1, 2]}),
            ("unequal", ("a", "a", "b", "b", "b", "c"), {"a": [0, 1], "b": [2, 3, 4], "c": [5]}),
            ("unequal_scattered", ("a", "b", "b", "a", "c"), {"a": [0, 3], "b": [1, 2], "c": [4]}),
        )
    )
    def test_grouping(self, _name, uids, expected):
        bs = len(uids)
        td = TensorDict({"responses": torch.zeros(bs, 4)}, batch_size=(bs,))
        batch = DataProto(batch=td, non_tensor_batch={"uid": np.array(uids, dtype=object)})
        self.assertEqual(group_prompt_by_uid(batch), expected)


class TestNewPromptEntry(unittest.TestCase):
    def test_defaults(self):
        entry = new_prompt_entry()
        self.assertEqual(
            entry,
            {
                "consecutive_all_0": 0,
                "consecutive_all_1": 0,
                "correct_responses": (),
                "last_epoch": -1,
                "num_correct": 0,
                "response_epoch": -1,
            },
        )


class TestUpdatePromptHistory(unittest.TestCase):
    @parameterized.expand(
        (
            # (name, scores, expected_history_dict)
            (
                "all_zero",
                (0, 0, 0, 0),
                {
                    "consecutive_all_0": 1,
                    "consecutive_all_1": 0,
                    "correct_responses": (),
                    "last_epoch": 0,
                    "num_correct": 0,
                    "response_epoch": -1,
                },
            ),
            (
                "all_one",
                (1, 1, 1, 1),
                {
                    "consecutive_all_0": 0,
                    "consecutive_all_1": 1,
                    "correct_responses": (),
                    "last_epoch": 0,
                    "num_correct": 4,
                    "response_epoch": -1,
                },
            ),
            (
                "mixed_half",
                (1, 0, 1, 0),
                {
                    "consecutive_all_0": 0,
                    "consecutive_all_1": 0,
                    "correct_responses": (),
                    "last_epoch": 0,
                    "num_correct": 2,
                    "response_epoch": -1,
                },
            ),
        )
    )
    def test_streak_and_count(self, _name, scores, expected):
        batch = _make_batch(1, 4, 4, 8, scores=(scores,))
        history = {}
        update_prompt_history(batch, history, epoch=0)

        # correct_responses contains tensors; compare separately
        hist = history[0]
        self.assertEqual(hist["correct_responses"], expected["correct_responses"])
        hist_without_resp = {k: v for k, v in hist.items() if k != "correct_responses"}
        expected_without_resp = {k: v for k, v in expected.items() if k != "correct_responses"}
        self.assertEqual(hist_without_resp, expected_without_resp)

    def test_mixed_resets_streaks(self):
        """Mixed group should reset both streaks built up previously."""
        batch = _make_batch(1, 4, 4, 8, scores=((1, 0, 1, 0),))
        history = {}
        update_prompt_history(batch, history, epoch=0)
        # Manually set a streak to verify reset
        history[0]["consecutive_all_1"] = 3
        history[0]["consecutive_all_0"] = 0
        history[0]["last_epoch"] = 0

        update_prompt_history(batch, history, epoch=1)
        self.assertEqual(history[0]["consecutive_all_1"], 0)
        self.assertEqual(history[0]["consecutive_all_0"], 0)

    def test_streak_across_epochs(self):
        """Consecutive all-1 across multiple epochs should accumulate."""
        batch = _make_batch(1, 4, 4, 8, scores=((1, 1, 1, 1),))
        history = {}

        for epoch in range(4):
            update_prompt_history(batch, history, epoch=epoch)

        self.assertEqual(history[0]["consecutive_all_1"], 4)
        self.assertEqual(history[0]["last_epoch"], 3)

    def test_same_epoch_no_double_count(self):
        """Multiple calls within same epoch should not double-count streaks."""
        batch = _make_batch(1, 4, 4, 8, scores=((1, 1, 1, 1),))
        history = {}

        update_prompt_history(batch, history, epoch=0)
        update_prompt_history(batch, history, epoch=0)  # same epoch again

        self.assertEqual(history[0]["consecutive_all_1"], 1)

    def test_multiple_prompts(self):
        """Each prompt tracked independently."""
        batch = _make_batch(
            3,
            2,
            4,
            8,
            scores=((1, 1), (0, 0), (1, 0)),
        )
        history = {}
        update_prompt_history(batch, history, epoch=0)

        self.assertEqual(history[0]["consecutive_all_1"], 1)
        self.assertEqual(history[1]["consecutive_all_0"], 1)
        # prompt 2 is mixed
        self.assertEqual(history[2]["consecutive_all_1"], 0)
        self.assertEqual(history[2]["consecutive_all_0"], 0)

    @parameterized.expand(
        (
            # (name, scores, expected_stored_count)
            ("minority_correct", (0, 1, 0, 0), 1),
            ("majority_correct", (0, 1, 1, 0), 0),
            ("all_correct", (1, 1, 1, 1), 0),
            ("all_wrong", (0, 0, 0, 0), 0),
        )
    )
    def test_storage_policy(self, _name, scores, exp_stored):
        batch = _make_batch(1, 4, 4, 8, scores=(scores,))
        history = {}
        update_prompt_history(batch, history, epoch=0)
        self.assertEqual(len(history[0]["correct_responses"]), exp_stored)

    def test_clears_stale_cache_when_majority_correct(self):
        """Should clear stored responses when prompt becomes majority correct."""
        history = {0: new_prompt_entry()}
        history[0]["correct_responses"] = ((torch.arange(100, 108, dtype=torch.long), 8, 1.0),)
        batch = _make_batch(1, 4, 4, 8, scores=((1, 1, 1, 0),))
        update_prompt_history(batch, history, epoch=1)

        self.assertEqual(history[0]["num_correct"], 3)
        self.assertEqual(history[0]["correct_responses"], ())


class TestInjectHistoricalResponses(unittest.TestCase):
    def _make_history_with_correct(self, prompt_id, response_len, score=1.0, num_stored=1):
        """Create a history entry with stored correct response(s)."""
        entry = new_prompt_entry()
        entry["correct_responses"] = tuple(
            (torch.arange(100 + k * response_len, 100 + (k + 1) * response_len, dtype=torch.long), response_len, score)
            for k in range(num_stored)
        )
        entry["consecutive_all_0"] = 1
        entry["last_epoch"] = 0
        return {prompt_id: entry}

    @parameterized.expand(
        (
            # (name, epoch, scores, has_history, exp_inject, exp_miss)
            ("epoch_0", 0, (0, 0, 0, 0), True, 0, 0),
            ("non_zero_group", 1, (1, 0, 0, 0), True, 0, 0),
            ("no_history", 1, (0, 0, 0, 0), False, 0, 1),
            ("inject", 1, (0, 0, 0, 0), True, 1, 0),
        )
    )
    def test_injection_gating(self, _name, epoch, scores, has_history, exp_inject, exp_miss):
        response_len = 8
        batch = _make_batch(1, 4, 4, response_len, scores=(scores,))
        history = self._make_history_with_correct(0, response_len) if has_history else {}

        stats = inject_historical_responses(batch, history, epoch=epoch)
        self.assertEqual(stats[STAT_INJECT_COUNT], exp_inject)
        self.assertEqual(stats[STAT_MISS_COUNT], exp_miss)

    def test_injection_replaces_longest_response(self):
        """All-0 group with history should have longest wrong response replaced."""
        response_len = 8
        batch = _make_batch(1, 4, 4, response_len, scores=((0, 0, 0, 0),))
        history = self._make_history_with_correct(0, response_len, score=1.0)

        stats = inject_historical_responses(batch, history, epoch=1)
        self.assertEqual(stats[STAT_INJECT_COUNT], 1)

        # Longest wrong response replaced first (all equal length, so first position)
        replaced = batch.batch["responses"][0]
        expected = torch.arange(100, 100 + response_len, dtype=torch.long)
        self.assertTrue(torch.equal(replaced, expected))

        # rm_scores should have score at last actual token
        self.assertAlmostEqual(batch.batch["rm_scores"][0, response_len - 1].item(), 1.0)

        # Other responses should be unchanged
        for i in range(1, 4):
            self.assertFalse(torch.equal(batch.batch["responses"][i], expected))

    def test_multiple_prompts_selective_injection(self):
        """Only all-0 groups with history get injected; others unchanged."""
        response_len = 8
        batch = _make_batch(
            3,
            2,
            4,
            response_len,
            scores=((0, 0), (0, 0), (1, 0)),
        )

        history = self._make_history_with_correct(0, response_len)

        stats = inject_historical_responses(batch, history, epoch=1)
        self.assertEqual(stats[STAT_INJECT_COUNT], 1)

        expected = torch.arange(100, 100 + response_len, dtype=torch.long)
        self.assertTrue(torch.equal(batch.batch["responses"][0], expected))

    def test_rm_scores_updated(self):
        """rm_scores should be updated for injected position."""
        response_len = 8
        batch = _make_batch(1, 4, 4, response_len, scores=((0, 0, 0, 0),))
        history = self._make_history_with_correct(0, response_len, score=0.75)

        inject_historical_responses(batch, history, epoch=1)

        self.assertAlmostEqual(batch.batch["rm_scores"][0, response_len - 1].item(), 0.75)
        self.assertAlmostEqual(batch.batch["rm_scores"][1].sum().item(), 0.0)

    @parameterized.expand(
        (
            # (name, n, max_num_injected, num_stored, expected_replaced)
            ("inject_2", 4, 2, 2, 2),
            ("capped_by_group_size", 3, 5, 5, 2),  # min(5, 5, 3-1) = 2
            ("capped_by_stored", 4, 3, 1, 1),  # min(3, 1, 4-1) = 1
        )
    )
    def test_multi_inject(self, _name, n, max_num_injected, num_stored, expected_replaced):
        response_len = 8
        batch = _make_batch(1, n, 4, response_len, scores=((0,) * n,))
        history = self._make_history_with_correct(0, response_len, score=1.0, num_stored=num_stored)

        stats = inject_historical_responses(
            batch,
            history,
            epoch=1,
            max_num_injected=max_num_injected,
        )
        self.assertEqual(stats[STAT_INJECT_COUNT], 1)

        # Verify exactly expected_replaced positions were replaced
        replaced_count = 0
        for i in range(n):
            for k in range(num_stored):
                expected = torch.arange(
                    100 + k * response_len,
                    100 + (k + 1) * response_len,
                    dtype=torch.long,
                )
                if torch.equal(batch.batch["responses"][i], expected):
                    replaced_count += 1
                    break
        self.assertEqual(replaced_count, expected_replaced)


class TestInjectValidation(unittest.TestCase):
    @parameterized.expand(
        (
            # stored content length > batch response_len
            ("stored_too_long", 6, 8),
            # stored tensor shape != batch response_len
            ("shape_mismatch", 10, 8),
        )
    )
    def test_raises_on_length_mismatch(self, _name, batch_resp_len, stored_resp_len):
        batch = _make_batch(1, 4, 4, batch_resp_len, scores=((0, 0, 0, 0),))
        entry = new_prompt_entry()
        entry["correct_responses"] = ((torch.zeros(stored_resp_len, dtype=torch.long), stored_resp_len, 1.0),)
        entry["last_epoch"] = 0
        history = {0: entry}

        with self.assertRaises(ValueError):
            inject_historical_responses(batch, history, epoch=1)


class TestEndToEnd(unittest.TestCase):
    """Test the full cycle: update history in epoch 0, inject in epoch 1."""

    def test_update_then_inject(self):
        response_len = 8
        prompt_len = 4
        n = 4

        # Epoch 0: prompt 0 has mixed results (some correct)
        batch_e0 = _make_batch(1, n, prompt_len, response_len, scores=((0, 1, 0, 0),))
        history = {}
        update_prompt_history(batch_e0, history, epoch=0)

        # Should have stored the correct response (index 1)
        self.assertEqual(len(history[0]["correct_responses"]), 1)
        stored_resp, stored_len, stored_score = history[0]["correct_responses"][0]
        self.assertEqual(stored_score, 1.0)

        # Epoch 1: same prompt, now all-0
        batch_e1 = _make_batch(1, n, prompt_len, response_len, scores=((0, 0, 0, 0),))

        stats = inject_historical_responses(batch_e1, history, epoch=1)
        self.assertEqual(stats[STAT_INJECT_COUNT], 1)

        # The injected response should be the one stored from epoch 0
        replaced = batch_e1.batch["responses"][0]
        self.assertTrue(torch.equal(replaced, stored_resp.to(replaced.device)))

        # The group now has mixed scores -> non-zero advantage possible
        scores_after = batch_e1.batch["rm_scores"].sum(-1)
        self.assertGreater(scores_after[0].item(), 0)
        for i in range(1, n):
            self.assertEqual(scores_after[i].item(), 0)


if __name__ == "__main__":
    unittest.main()
