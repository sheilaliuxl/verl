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

"""Per-prompt history tracking and historical injection.

Tracks per-prompt statistics (all-0/all-1 streaks, stored correct responses)
across epochs, and injects historical correct responses into all-0 groups
to break zero-advantage symmetry in GRPO.
"""

from collections import defaultdict

from verl import DataProto
from verl.trainer.ppo.metric_utils import KEY_ATTENTION_MASK, KEY_RESPONSE_MASK

# Persistent prompt identifier key in non_tensor_batch. Currently uses dataset "index"
# from extra_info; can be switched to "uid" if a globally-unique ID is added later.
KEY_PROMPT_ID = "index"

# Per-prompt history entry field names.
KEY_CONSECUTIVE_ALL_0 = "consecutive_all_0"
KEY_CONSECUTIVE_ALL_1 = "consecutive_all_1"
KEY_CORRECT_RESPONSES = "correct_responses"
KEY_LAST_EPOCH = "last_epoch"
KEY_NUM_CORRECT = "num_correct"
KEY_RESPONSE_EPOCH = "response_epoch"

# Injection stats keys (returned by inject_historical_responses).
STAT_INJECT_COUNT = "inject_count"
STAT_INJECT_STALE_COUNT = "inject_stale_count"
STAT_MAX_NUM_CORRECT_MISSED = "max_num_correct_missed"
STAT_MAX_NUM_CORRECT_RETRIEVED = "max_num_correct_retrieved"
STAT_MAX_STALE_EPOCHS = "max_stale_epochs"
STAT_MISS_COUNT = "miss_count"
STAT_MISS_RATIO = "miss_ratio"


def group_prompt_by_uid(batch: DataProto) -> dict[str, list[int]]:
    """Group response indices by UID, handling scattered order after balance_batch."""
    uid_to_positions: dict[str, list[int]] = defaultdict(list)
    for i, uid in enumerate(batch.non_tensor_batch["uid"]):
        uid_to_positions[uid].append(i)
    return dict(uid_to_positions)


def new_prompt_entry() -> dict:
    """Create a fresh prompt history entry."""
    return {
        KEY_CONSECUTIVE_ALL_0: 0,
        KEY_CONSECUTIVE_ALL_1: 0,
        KEY_CORRECT_RESPONSES: (),  # tuple of (response_tensor, length, score)
        KEY_LAST_EPOCH: -1,
        KEY_NUM_CORRECT: 0,
        KEY_RESPONSE_EPOCH: -1,
    }


def update_prompt_history(
    batch: DataProto,
    prompt_history: dict,
    epoch: int,
    score_key: str = "token_level_scores",
    negative_score_threshold_le: float = 0.0,
    max_num_stored: int = 1,
) -> dict:
    """Update per-prompt history after each step.

    Tracks consecutive all-0/all-1 streaks and stores correct responses
    for future injection. Prefers shortest correct responses (more likely
    to fit in future batches without tensor expansion).

    Args:
        batch: The training batch (read-only).
        prompt_history: Per-prompt history dict (modified in-place).
        epoch: Current epoch number.
        score_key: Batch key containing per-token scores.
        negative_score_threshold_le: Scores <= this are negative/wrong. Default 0.
        max_num_stored: Max correct responses to store per prompt. Default 1.

    Returns:
        Dict with per-step num_correct/num_incorrect distribution:
            num_correct_XX: count of prompts with XX correct responses.
            num_incorrect_XX: count of prompts with XX incorrect responses.
    """
    uid_to_positions = group_prompt_by_uid(batch)
    prompt_ids = batch.non_tensor_batch[KEY_PROMPT_ID]
    scores = batch.batch[score_key].sum(-1).cpu().tolist()

    correct_counts: dict[int, int] = defaultdict(int)
    incorrect_counts: dict[int, int] = defaultdict(int)

    for uid, positions in uid_to_positions.items():
        prompt_id = prompt_ids[positions[0]]
        group_scores = [scores[i] for i in positions]

        hist = prompt_history.get(prompt_id)
        if hist is None:
            hist = new_prompt_entry()

        # Track num_correct; only store response tensor when minority correct
        # (groups with many correct rarely regress to all-0, so skip the tensor).
        num_correct = sum(s_i > negative_score_threshold_le for s_i in group_scores)
        is_all_0 = num_correct == 0
        is_all_1 = num_correct == len(positions)

        correct_counts[num_correct] += 1
        incorrect_counts[len(positions) - num_correct] += 1

        # Only update streaks on epoch transitions (each prompt seen once per epoch)
        if epoch != hist[KEY_LAST_EPOCH]:
            if is_all_1:
                hist[KEY_CONSECUTIVE_ALL_1] += 1
                hist[KEY_CONSECUTIVE_ALL_0] = 0
            elif is_all_0:
                hist[KEY_CONSECUTIVE_ALL_0] += 1
                hist[KEY_CONSECUTIVE_ALL_1] = 0
            else:  # mixed
                hist[KEY_CONSECUTIVE_ALL_1] = 0
                hist[KEY_CONSECUTIVE_ALL_0] = 0
            hist[KEY_LAST_EPOCH] = epoch

        hist[KEY_NUM_CORRECT] = num_correct

        # Store responses when minority correct; clear when majority correct.
        if 0 < num_correct < len(positions) / 2:
            collected = []
            for i in positions:
                if scores[i] > negative_score_threshold_le:
                    resp_tokens = batch.batch["responses"][i].cpu()
                    if KEY_RESPONSE_MASK in batch.batch:
                        resp_len = int(batch.batch[KEY_RESPONSE_MASK][i].sum().item())
                    else:
                        resp_len = resp_tokens.shape[0]
                    collected.append((resp_tokens, resp_len, scores[i]))
            # Prefer shortest responses (more likely to fit without expansion)
            collected.sort(key=lambda x: x[1])
            hist[KEY_CORRECT_RESPONSES] = tuple(collected[:max_num_stored])
            hist[KEY_RESPONSE_EPOCH] = epoch
        elif num_correct >= len(positions) / 2:
            hist[KEY_CORRECT_RESPONSES] = ()
            hist[KEY_RESPONSE_EPOCH] = -1

        prompt_history[prompt_id] = hist

    stats = {f"num_correct_{k:02d}": v for k, v in sorted(correct_counts.items())}
    stats.update({f"num_incorrect_{k:02d}": v for k, v in sorted(incorrect_counts.items())})
    return stats


def inject_historical_responses(
    batch: DataProto,
    prompt_history: dict,
    epoch: int,
    max_num_injected: int = 1,
    negative_score_threshold_le: float = 0.0,
) -> dict:
    """Inject stored correct responses into all-0 groups to break zero-advantage symmetry.

    For all-0 groups where a correct response was stored from a previous epoch,
    replace up to ``max_num_injected`` wrong responses with the historical correct
    one. This creates a mixed-reward group with non-zero advantage, providing
    gradient signal.

    Replaces the longest wrong responses first (most wasted compute).

    Must be called after reward model scoring (rm_scores available) and before
    extract_reward, so that extract_reward produces a correct reward_tensor
    from the updated rm_scores.

    Args:
        batch: The training batch (modified in-place).
        prompt_history: Per-prompt history dict (KEY_PROMPT_ID -> entry).
        epoch: Current epoch number.
        max_num_injected: Maximum number of responses to replace per all-0 group.
            Must be < group size (n) to keep at least one wrong response. Default 1.
        negative_score_threshold_le: Scores <= this are negative/wrong. Default 0,
            works for both {-1, 1} and {0, 1} reward schemes.

    Returns:
        Dict with injection statistics.
    """
    stats = {
        STAT_INJECT_COUNT: 0,
        STAT_INJECT_STALE_COUNT: 0,
        STAT_MISS_COUNT: 0,
        STAT_MAX_NUM_CORRECT_RETRIEVED: 0,
        STAT_MAX_NUM_CORRECT_MISSED: 0,
        STAT_MAX_STALE_EPOCHS: 0,
    }
    if epoch == 0:
        return stats

    uid_to_positions = group_prompt_by_uid(batch)
    prompt_ids = batch.non_tensor_batch[KEY_PROMPT_ID]
    scores = batch.batch["rm_scores"].sum(-1).cpu().tolist()

    response_len = batch.batch["responses"].shape[1]
    prompt_len = batch.batch["prompts"].shape[1]

    # Pass 1: identify all-0 groups with stored correct responses.
    groups_to_inject = []
    for uid, positions in uid_to_positions.items():
        group_scores = [scores[i] for i in positions]
        if max(group_scores) > negative_score_threshold_le:
            continue

        prompt_id = prompt_ids[positions[0]]
        hist = prompt_history.get(prompt_id)
        stored = hist.get(KEY_CORRECT_RESPONSES, ()) if hist else ()
        if not stored:
            # num_correct > 0 but no vectors means stale: prompt was majority
            # correct in a prior epoch (vectors cleared) then regressed to all-0.
            stats[STAT_MISS_COUNT] += 1
            if hist is not None:
                stats[STAT_MAX_NUM_CORRECT_MISSED] = max(
                    stats[STAT_MAX_NUM_CORRECT_MISSED],
                    hist.get(KEY_NUM_CORRECT, 0),
                )
            continue

        num_to_inject = min(max_num_injected, len(stored), len(positions) - 1)
        for j in range(num_to_inject):
            if stored[j][1] > response_len:
                raise ValueError(f"Stored response len {stored[j][1]} > {response_len}")
        groups_to_inject.append((positions, stored, hist, num_to_inject))

    # Pass 2: inject stored correct responses.
    for positions, stored, hist, num_to_inject in groups_to_inject:
        # Sort positions by actual content length descending so we replace
        # the longest wrong responses first (most wasted compute).
        if KEY_RESPONSE_MASK in batch.batch:
            positions_by_len = sorted(
                positions,
                key=lambda i: batch.batch[KEY_RESPONSE_MASK][i].sum().item(),
                reverse=True,
            )
        else:
            positions_by_len = positions

        for j in range(num_to_inject):
            replace_pos = positions_by_len[j]
            stored_resp, stored_len, stored_score = stored[j]

            device = batch.batch["responses"].device
            if stored_resp.shape[0] != response_len:
                raise ValueError(f"Stored response shape {stored_resp.shape[0]} != {response_len}")

            # 1. Overwrite response tokens
            batch.batch["responses"][replace_pos] = stored_resp.clone().to(device)

            # 2. Overwrite response portion of attention_mask (prompt portion unchanged)
            batch.batch[KEY_ATTENTION_MASK][replace_pos, prompt_len : prompt_len + stored_len] = 1
            batch.batch[KEY_ATTENTION_MASK][replace_pos, prompt_len + stored_len :] = 0

            # 3. Overwrite response_mask
            if KEY_RESPONSE_MASK in batch.batch:
                batch.batch[KEY_RESPONSE_MASK][replace_pos] = 0
                batch.batch[KEY_RESPONSE_MASK][replace_pos, :stored_len] = 1

            # 4. Overwrite rm_scores (score at last actual response token)
            batch.batch["rm_scores"][replace_pos] = 0
            batch.batch["rm_scores"][replace_pos, stored_len - 1] = stored_score

        response_epoch = hist.get(KEY_RESPONSE_EPOCH, -1)
        if response_epoch < epoch - 1:
            stale_epochs = epoch - response_epoch
            stats[STAT_INJECT_STALE_COUNT] += 1
            stats[STAT_MAX_STALE_EPOCHS] = max(
                stats[STAT_MAX_STALE_EPOCHS],
                stale_epochs,
            )
        stats[STAT_MAX_NUM_CORRECT_RETRIEVED] = max(
            stats[STAT_MAX_NUM_CORRECT_RETRIEVED],
            hist.get(KEY_NUM_CORRECT, 0),
        )
        stats[STAT_INJECT_COUNT] += 1

    all_0_count = stats[STAT_INJECT_COUNT] + stats[STAT_MISS_COUNT]
    stats[STAT_MISS_RATIO] = stats[STAT_MISS_COUNT] / all_0_count if all_0_count > 0 else 0.0
    return stats
