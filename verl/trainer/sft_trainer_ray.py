# Copyright 2024 Bytedance Ltd. and/or its affiliates
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


import os
import time
from functools import partial

from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import logging

import hydra
import ray
import torch
import torch.distributed
from omegaconf import OmegaConf
from torch.utils.data import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint import CheckpointHandler, OrchestrationMode
from verl.utils.dataset.dataset_utils import SFTTensorCollator
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.device import auto_set_device, get_device_name
from verl.utils.logger import log_with_rank
from verl.utils.tracking import Tracking
from verl.workers.engine_workers import TrainingWorker

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))


class SFTTrainer:
    def __init__(
        self,
        config,
    ):
        self.config = config

        self._build_config()
        self._build_dataset()
        self._build_dataloader()

        self._build_engine()
        self._build_ckpt_handler()

        # Initialize resume-related variables
        self.resume_global_step = self.ckpt_handler.load_checkpoint()

        self.device_name = self.config.trainer.device

        print(self.config)

    def _build_ckpt_handler(self):
        resume_mode = getattr(self.config.trainer, "resume_mode", "auto")
        resume_from_path = getattr(self.config.trainer, "resume_from_path", None)
        max_ckpt_to_keep = getattr(self.config.trainer, "max_ckpt_to_keep", None)
        default_hdfs_dir = getattr(self.config.trainer, "default_hdfs_dir", None)

        self.ckpt_handler = CheckpointHandler(
            engine=self.training_client,
            train_dataloader=self.train_dataloader,
            default_local_dir=self.config.trainer.default_local_dir,
            max_ckpt_to_keep=max_ckpt_to_keep,
            default_hdfs_dir=default_hdfs_dir,
            resume_mode=resume_mode,
            resume_from_path=resume_from_path,
            mode=OrchestrationMode.RAY,
        )

    def _build_config(self):
        from verl.utils.config import omega_conf_to_dataclass

        self.model_config = omega_conf_to_dataclass(self.config.model)
        self.engine_config = omega_conf_to_dataclass(self.config.engine)
        self.optimizer_config = omega_conf_to_dataclass(self.config.optim)
        self.checkpoint_config = omega_conf_to_dataclass(self.config.checkpoint)
        self.profiler_config = omega_conf_to_dataclass(self.config.profiler)

        # check profile interval
        self.profiler_interval = self.config.trainer.profile_interval
        self._validate_profiler_interval()

    def _validate_profiler_interval(self):
        assert len(self.profiler_interval) == 2
        self.start_profile_step = self.profiler_interval[0]
        self.end_profile_step = self.profiler_interval[1]
        assert self.end_profile_step >= self.start_profile_step
        if self.start_profile_step < 0:
            assert self.end_profile_step < 0

    def _build_engine(self):
        from verl.workers.engine_workers import TrainingWorkerConfig
        from verl.workers.utils.losses import sft_loss

        self.loss_fn = partial(sft_loss, config=None)

        config = TrainingWorkerConfig(
            model_type="language_model",
            model_config=self.model_config,
            engine_config=self.engine_config,
            optimizer_config=self.optimizer_config,
            checkpoint_config=self.checkpoint_config,
            profiler_config=self.profiler_config,
        )

        wg_kwargs = {}
        if self.start_profile_step != -1:
            wg_kwargs["profile_steps"] = list(range(self.start_profile_step, self.end_profile_step + 1))
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.profiler, "tool") == "nsys":
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )

        # create resource pool and worker group
        from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup

        n_gpus_per_node = self.config.trainer.n_gpus_per_node
        nnodes = self.config.trainer.nnodes
        self.resource_pool = RayResourcePool(process_on_nodes=[n_gpus_per_node] * nnodes)
        ray_cls_with_init = RayClassWithInitArgs(ray.remote(TrainingWorker), config=config)
        self.training_client = RayWorkerGroup(
            resource_pool=self.resource_pool,
            ray_cls_with_init=ray_cls_with_init,
            device_name=self.config.trainer.device,
            **wg_kwargs,
        )
        self.training_client.set_loss_fn(loss_fn=self.loss_fn)
        self.training_client.reset()

    def _build_dataset(self):
        config = self.config
        tokenizer = self.model_config.tokenizer
        processor = self.model_config.processor
        train_dataset = create_sft_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor=processor,
            max_samples=config.data.get("train_max_samples", -1),
        )
        if config.data.val_files:
            val_dataset = create_sft_dataset(
                config.data.val_files,
                config.data,
                tokenizer,
                processor=processor,
                max_samples=config.data.get("val_max_samples", -1),
            )
        else:
            val_dataset = None

        self.train_dataset, self.val_dataset = train_dataset, val_dataset

    def _build_dataloader(self):
        # build dataset
        config = self.config
        # build dataloader
        # Use data parallel rank and size instead of global rank and world size

        # Set pin_memory_device when pin_memory is enabled.
        device_name = get_device_name()

        dp_rank = 0
        dp_size = 1

        self.train_sampler = DistributedSampler(
            self.train_dataset, shuffle=True, num_replicas=dp_size, rank=dp_rank, drop_last=True
        )

        self.global_batch_size = config.data.train_batch_size
        self.train_batch_size_per_dp = self.global_batch_size // dp_size
        self.collate_fn = SFTTensorCollator(config.data.pad_mode)

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size_per_dp,
            sampler=self.train_sampler,
            collate_fn=self.collate_fn,
            num_workers=8,
            pin_memory=False,
            drop_last=True,
            pin_memory_device=device_name,
        )

        if self.val_dataset:
            self.val_sampler = DistributedSampler(
                self.val_dataset, shuffle=False, num_replicas=dp_size, rank=dp_rank, drop_last=True
            )
            self.val_dataloader = StatefulDataLoader(
                dataset=self.val_dataset,
                batch_size=self.train_batch_size_per_dp,
                sampler=self.val_sampler,
                collate_fn=self.collate_fn,
                num_workers=8,
                pin_memory=False,
                drop_last=True,
                pin_memory_device=device_name,
            )
        else:
            self.val_dataloader = None

        # update
        if self.config.trainer.total_training_steps is not None:
            self.total_training_steps = self.config.trainer.total_training_steps
        else:
            self.total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        self.optimizer_config.total_training_steps = self.total_training_steps

        self.steps_per_epoch = len(self.train_dataloader)

        # manage save and test frequency
        self.save_freq = self.config.trainer.save_freq
        if self.save_freq == "after_each_epoch":
            self.save_freq = self.steps_per_epoch

        self.test_freq = self.config.trainer.test_freq
        if self.test_freq == "after_each_epoch":
            self.test_freq = self.steps_per_epoch

    def _get_batch_seqlens(self, data):
        # mean over dp group
        is_nested = data["input_ids"].is_nested
        if is_nested:
            batch_seqlens: torch.Tensor = data["input_ids"].offsets().diff()
        else:
            batch_seqlens: torch.Tensor = data["attention_mask"].sum(dim=-1)
        return batch_seqlens

    def _compute_ray_rpc_train_steps(self, global_step):
        """Compute how many steps to batch into a single RPC call.

        Bounded by epoch boundary, save_freq, test_freq, profiler boundaries,
        total_training_steps, and max_ray_rpc_train_steps config.
        """
        max_ray_rpc_train_steps = int(getattr(self.config.trainer, "ray_rpc_train_steps", 1))
        if max_ray_rpc_train_steps <= 1:
            return 1

        remaining = self.total_training_steps - global_step
        num_steps = min(max_ray_rpc_train_steps, remaining)

        # Break at epoch boundary to avoid partial dataloader fetches
        steps_to_epoch_end = self.steps_per_epoch - (global_step % self.steps_per_epoch)
        num_steps = min(num_steps, steps_to_epoch_end)

        # Break at save/test boundaries
        for freq in (self.save_freq, self.test_freq):
            if freq > 0:
                steps_to_boundary = freq - (global_step % freq)
                num_steps = min(num_steps, steps_to_boundary)

        # Break at profiler boundaries
        if self.start_profile_step >= 0:
            for profile_step in (self.start_profile_step, self.end_profile_step):
                if global_step < profile_step:
                    num_steps = min(num_steps, profile_step - global_step)

        return max(1, num_steps)

    def _interleave_batches(self, batches):
        """Interleave N TensorDicts so DP chunking gives each worker the correct slice per step.

        Input: N TensorDicts each with batch_size [B, ...]
        Output: single TensorDict with batch_size [B*N, ...] where samples are interleaved
                as [b0[0], b1[0], ..., bN[0], b0[1], b1[1], ..., bN[1], ...]
        """
        num_batches = len(batches)
        if num_batches == 1:
            return batches[0]

        batch_size = batches[0].batch_size[0]
        result = {}
        for key, val in batches[0].items():
            if not isinstance(val, torch.Tensor):
                continue
            if val.is_nested:
                # Element-wise interleave for nested tensors (variable-length sequences).
                elements = [batch[key][b] for b in range(batch_size) for batch in batches]
                result[key] = torch.nested.as_nested_tensor(elements, layout=torch.jagged)
            else:
                # Stack/transpose/reshape for regular tensors.
                stacked = torch.stack([b[key] for b in batches], dim=0)  # [N, B, ...]
                interleaved = stacked.transpose(0, 1).reshape(-1, *stacked.shape[2:])  # [B*N, ...]
                result[key] = interleaved

        packed = TensorDict(result, batch_size=[batch_size * num_batches])
        # Copy NonTensorData fields (meta_info) from the first batch.
        # Re-wrap to avoid batch_size mismatch with packed.
        for key, val in batches[0].items():
            if not isinstance(val, torch.Tensor):
                packed[key] = NonTensorData(val.data)
        return packed

    def fit(self):
        tracking = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        global_step = self.resume_global_step  # Start from resumed step
        last_valid_metric = None

        log_with_rank(
            f"Total training steps: {self.total_training_steps},",
            logger=logger,
            rank=0,
            log_only_rank_0=True,
        )

        # With StatefulDataLoader, we don't need to manually calculate epochs and steps
        # The dataloader will automatically resume from where it left off
        if global_step > 0:
            log_with_rank(
                f"StatefulDataLoader will automatically resume from global step: {global_step}",
                logger=logger,
                rank=0,
                log_only_rank_0=True,
            )

        # Calculate which epoch we're starting from for sampler.set_epoch()
        start_epoch = global_step // self.steps_per_epoch

        meta_info = {
            "use_remove_padding": self.config.model.use_remove_padding,
            "use_dynamic_bsz": self.config.data.use_dynamic_bsz,
            "max_token_len_per_gpu": self.config.data.max_token_len_per_gpu,
            "micro_batch_size_per_gpu": self.config.data.micro_batch_size_per_gpu,
            "temperature": 1.0,
            "global_batch_size": self.global_batch_size,
            "pad_mode": self.config.data.pad_mode,
            "pad_token_id": self.model_config.tokenizer.pad_token_id,
        }

        # ray_rpc_train_steps = 1 (default) uses train_batch directly without packing overhead;
        # ray_rpc_train_steps > 1 (including 1.5) uses train_multi_batch.
        ray_rpc_single_batch_mode = getattr(self.config.trainer, "ray_rpc_train_steps", 1) <= 1

        total_driver_time = 0
        total_worker_time = 0
        total_tokens = 0
        for epoch in range(start_epoch, self.config.trainer.total_epochs):
            self.train_sampler.set_epoch(epoch=epoch)

            dataloader_iter = iter(self.train_dataloader)
            progress_bar = tqdm(
                total=self.steps_per_epoch,
                initial=global_step % self.steps_per_epoch if epoch == start_epoch else 0,
                desc=f"Epoch {epoch + 1}/{self.config.trainer.total_epochs}",
            )

            while global_step < self.total_training_steps:
                driver_start = time.perf_counter()

                if ray_rpc_single_batch_mode:
                    # Direct path: no packing/interleave overhead.
                    try:
                        data = next(dataloader_iter)
                    except StopIteration:
                        break
                    data = tu.get_tensordict(tensor_dict=data, non_tensor_dict=meta_info)
                    all_batch_seqlens = [self._get_batch_seqlens(data=data).tolist()]
                    # NonTensorData wrapping is necessary. Otherwise, it is interpreted as NonTensorStack.
                    tu.assign_non_tensor(
                        data,
                        update_lr_scheduler=True,
                        global_token_num=NonTensorData(all_batch_seqlens[0]),
                    )

                    if self.start_profile_step == global_step:
                        self.training_client.start_profile()
                    worker_start = time.perf_counter()
                    output = self.training_client.train_batch(data).get()
                    worker_time = time.perf_counter() - worker_start

                    actual_steps = 1

                    all_metrics = [tu.get(output, "metrics")]
                else:
                    ray_rpc_train_steps = self._compute_ray_rpc_train_steps(global_step)

                    # Prefetch N batches
                    batches = []
                    all_batch_seqlens = []
                    for _ in range(ray_rpc_train_steps):
                        try:
                            data = next(dataloader_iter)
                        except StopIteration:
                            break
                        data = tu.get_tensordict(tensor_dict=data, non_tensor_dict=meta_info)
                        batch_seqlens = self._get_batch_seqlens(data=data).tolist()
                        batches.append(data)
                        all_batch_seqlens.append(batch_seqlens)

                    if not batches:
                        break

                    actual_steps = len(batches)

                    # Interleave batches and send in one RPC (works for N=1 too)
                    packed = self._interleave_batches(batches)
                    # NonTensorData wrapping is necessary. Otherwise, it is interpreted as NonTensorStack.
                    tu.assign_non_tensor(
                        packed,
                        update_lr_scheduler=True,
                        num_steps=actual_steps,
                        all_global_token_nums=NonTensorData(all_batch_seqlens),
                    )

                    # _compute_ray_rpc_train_steps breaks at profiler boundaries, so
                    # global_step == start_profile_step at the start of the profiled RPC.
                    if self.start_profile_step == global_step:
                        self.training_client.start_profile()
                    worker_start = time.perf_counter()
                    output = self.training_client.train_multi_batch(packed).get()
                    worker_time = time.perf_counter() - worker_start

                    all_metrics = tu.get(output, "all_metrics")

                if self.end_profile_step == global_step + actual_steps:
                    self.training_client.stop_profile()

                # Log metrics for each step (shared across both paths).
                worker_time_per_step = worker_time / actual_steps
                for i in range(actual_steps):
                    global_step += 1
                    metrics = all_metrics[i]
                    metrics["train/loss"] = metrics.pop("loss")
                    metrics["train/grad_norm"] = metrics.pop("grad_norm")
                    metrics["train/lr"] = metrics.pop("lr")
                    metrics["train/mfu"] = metrics.pop("mfu")
                    metrics["train/time(s)"] = worker_time_per_step
                    metrics["train/global_tokens"] = torch.sum(
                        torch.tensor(all_batch_seqlens[i], device=self.device_name)
                    ).item()
                    total_tokens += metrics["train/global_tokens"]
                    metrics["train/total_tokens(B)"] = total_tokens / 1e9
                    tracking.log(data=metrics, step=global_step)

                progress_bar.update(actual_steps)

                is_last_step = global_step >= self.total_training_steps
                is_eval_step = self.test_freq > 0 and global_step % self.test_freq == 0
                is_eval_step = self.val_dataloader is not None and (is_eval_step or is_last_step)
                is_save_step = self.save_freq > 0 and global_step % self.save_freq == 0

                if is_eval_step:
                    val_losses = []
                    for val_data in self.val_dataloader:
                        val_data = tu.get_tensordict(tensor_dict=val_data, non_tensor_dict=meta_info)
                        output = self.training_client.infer_batch(val_data)
                        output = output.get()
                        metrics = tu.get(output, "metrics")
                        val_losses.append(metrics["loss"])

                    val_loss = torch.mean(torch.tensor(val_losses, device=self.device_name))

                    metric = {"val/loss": val_loss.detach().item()}
                    tracking.log(data=metric, step=global_step)
                    last_valid_metric = metric

                if is_last_step or is_save_step:
                    self.ckpt_handler.save_checkpoint(step=global_step)

                driver_time = time.perf_counter() - driver_start
                total_driver_time += driver_time
                total_worker_time += worker_time

                if is_last_step:
                    percent = total_worker_time / total_driver_time * 100 if total_driver_time > 0 else 0
                    print(
                        f"Total time for train steps: driver={total_driver_time:.2f}s,"
                        f" worker={total_worker_time:.2f}s ({percent:.2f}%)."
                    )
                    print(f"Final validation metrics: {last_valid_metric}")
                    progress_bar.close()
                    return

            progress_bar.close()


def run_sft(config):
    ray.init()
    trainer = SFTTrainer(config=config)
    trainer.fit()


@hydra.main(config_path="config", config_name="sft_trainer_engine", version_base=None)
def main(config):
    # Automatically set `config.trainer.device = npu` when running on Ascend NPU.
    auto_set_device(config)
    run_sft(config)


def create_sft_dataset(data_paths, data_config, tokenizer, processor, max_samples=-1):
    """Create a dataset."""
    # build dataset
    # First check if a custom dataset class is specified
    if data_config.custom_cls.get("path", None):
        from verl.utils.import_utils import load_extern_type

        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
    else:
        # Default to multi-turn dataset
        dataset_cls = MultiTurnSFTDataset

    # Create datasets based on the selected class
    dataset = dataset_cls(
        parquet_files=data_paths, tokenizer=tokenizer, config=data_config, processor=processor, max_samples=max_samples
    )
    return dataset


if __name__ == "__main__":
    main()
