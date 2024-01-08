import inspect
import logging
import math
import os
from functools import partial
from logging import StreamHandler
from typing import Optional, Union

import evaluate
import torch
import torch.distributed as dist
import wandb
from arguments.inference_args import InferenceArguments
from datasets import Dataset, concatenate_datasets, load_dataset
from setproctitle import setproctitle
from simple_parsing import ArgumentParser
from torch.cuda.amp import autocast
from torch.distributed.fsdp import CPUOffload
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.utils.data import DistributedSampler
from trainer.fsdp import Trainer
from transformers import AutoTokenizer, AutoConfig
from transformers.models.roberta.modeling_roberta import RobertaForSequenceClassification, RobertaEncoder
from utils.comfy import (
    apply_to_collection,
    bfloat_support,
    dataclass_to_namespace,
    json_to_dict,
    seed_everything,
    tensor_dict_to_device,
)
from utils.data.custom_sampler import DistributedLengthGroupedSampler
from utils.FSDP import mixed_precision
from utils.FSDP.wrapping import get_transformers_wrapper
from utils.model_checkpointing.fsdp_handler import load_model_checkpoint

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)8s] %(message)s")

timeFileHandler = StreamHandler()
timeFileHandler.setFormatter(formatter)

logger.addHandler(timeFileHandler)

SHARDING_STRATEGY = {
    "FULL_SHARD": ShardingStrategy.FULL_SHARD,
    "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
    "NO_SHARD": ShardingStrategy.NO_SHARD,
    "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
    "_HYBRID_SHARD_ZERO2": ShardingStrategy._HYBRID_SHARD_ZERO2,
}


# TODO(User): override training_step and eval_loop for your style
class FSDPTrainer(Trainer):
    def __init__(
        self, device_id, criterion, eval_metric, precision: str = "fp32", cmd_logger=None, metric_on_cpu: bool = False
    ):
        super().__init__(device_id, criterion, eval_metric, precision, cmd_logger, metric_on_cpu)

    def test_loop(
        self,
        model,
        test_loader: Optional[torch.utils.data.DataLoader],
        **kwargs,
    ):
        """The test loop ruunning a single test epoch.

        Args:
            model: model
            test_loader: The dataloader yielding the test batches.
            limit_batches: Limits the batches during this test epoch.
                If greater than the number of batches in the ``test_loader``, this has no effect.

        """
        # no test if test_loader wasn't passed
        if test_loader is None:
            return

        def on_start_test(model):
            model.eval()
            # requires_grad = True, but loss.backward() raised error
            # because grad_fn is None
            torch.set_grad_enabled(False)

        on_start_test(model)

        def on_test_epoch_start():
            pass

        if self.device_id == 0:
            iterable = self.progbar_wrapper(test_loader, total=len(test_loader), desc="test")
            pbar = enumerate(iterable)
        else:
            pbar = enumerate(test_loader)

        eval_step = 0
        tot_batch_logits = list()
        tot_batch_labels = list()
        tot_batch_loss = list()

        if self.metric_on_cpu:
            metric_on_device = torch.device("cpu")
        else:
            metric_on_device = self.device

        for batch_idx, batch in pbar:
            # I tried to output the most accurate LOSS to WANDB with ALL_GATHER for all LOSS sections,
            # but it was not much different from outputting the value of GPU 0.
            # Therefore, all sections except EVAL EPOCH END only output the value of rank 0.
            tensor_dict_to_device(batch, self.device, non_blocking=self.non_blocking)
            # I use distributed dataloader and wandb log only rank:0, and epoch loss all gather

            def on_test_batch_start(batch, batch_idx):
                pass

            on_test_batch_start(batch, batch_idx)

            # TODO(User): fit the input and output for your model architecture!
            # Model is float32 but can calculate fp16 safety!
            with autocast(enabled=self.mixed_precision, dtype=self.precision):
                output = model(**batch)

            # avoid gradients in stored/accumulated values -> prevents potential OOM
            self._current_val_return = apply_to_collection(
                output, torch.Tensor, lambda x: x.detach().to(metric_on_device)
            )

            loss = self._current_val_return["loss"]

            # TODO(User): what do you want to log items every epoch end?
            tot_batch_logits.append(output["logits"].to(metric_on_device))
            tot_batch_labels.append(batch["labels"].to(metric_on_device))
            tot_batch_loss.append(loss.to(metric_on_device))

            def on_test_batch_end(eval_out, batch, batch_idx):
                pass

            on_test_batch_end(self._current_val_return, batch, batch_idx)

            cmd_output = {"loss": self._current_val_return["loss"]}
            if self.device_id == 0:
                self._format_iterable(iterable, cmd_output, "val")
            eval_step += 1

        device_total_loss = torch.mean(
            torch.tensor(tot_batch_loss, dtype=tot_batch_loss[0].dtype, device=metric_on_device)
        )

        # TODO(User): Create any form you want to output to wandb!
        def on_test_epoch_end(tot_batch_logits, tot_batch_labels, device_total_loss, metric_device):
            tot_batch_logits = torch.cat(tot_batch_logits, dim=0)
            tot_batch_labels = torch.cat(tot_batch_labels, dim=0)

            # all_gather` requires a `fixed length tensor` as input.
            # Since the length of the data on each GPU may be different, the length should be passed to `all_gather` first.
            local_size = torch.tensor([tot_batch_logits.size(0)], dtype=torch.long, device=metric_device)
            size_list = [
                torch.tensor([0], dtype=torch.long, device=metric_device) for _ in range(dist.get_world_size())
            ]
            loss_list = [
                torch.tensor(0, dtype=device_total_loss.dtype, device=metric_device)
                for _ in range(dist.get_world_size())
            ]
            if metric_device == torch.device("cpu"):
                dist.all_gather_object(size_list, local_size)
                dist.all_gather_object(loss_list, device_total_loss)
            else:
                dist.all_gather(size_list, local_size)
                dist.all_gather(loss_list, device_total_loss)

            total_loss = torch.tensor(loss_list).mean()

            # Create a fixed length tensor with the length of `all_gather`.
            logits_gathered_data = [
                torch.zeros(
                    (size.item(), tot_batch_logits.size(-1)), dtype=tot_batch_logits.dtype, device=metric_device
                )
                for size in size_list
            ]
            labels_gathered_data = [
                torch.zeros(size.item(), dtype=tot_batch_labels.dtype, device=metric_device) for size in size_list
            ]

            if metric_device == torch.device("cpu"):
                # Collect and match data from all GPUs.
                dist.all_gather_object(logits_gathered_data, tot_batch_logits)
                dist.all_gather_object(labels_gathered_data, tot_batch_labels)
            else:
                dist.all_gather(logits_gathered_data, tot_batch_logits)
                dist.all_gather(labels_gathered_data, tot_batch_labels)

            if self.device_id == 0:
                # example 4 gpus : [gpu0[tensor],gpu1[tensor],gpu2[tensor],gpu3[tensor]]
                logits_gathered_data = torch.cat(logits_gathered_data, dim=0)
                predictions = torch.argmax(logits_gathered_data, axis=-1)
                references = torch.cat(labels_gathered_data, dim=0)

                total_acc = self.eval_metric.compute(predictions=predictions, references=references)

                self.logger.info(f"Total Loss: {total_loss}")
                self.logger.info(f"Total Metric: {total_acc}")

        on_test_epoch_end(tot_batch_logits, tot_batch_labels, device_total_loss, metric_on_device)


def main(hparams: InferenceArguments):
    # reference: https://github.com/pytorch/examples/blob/main/distributed/FSDP/T5_training.py
    setproctitle("fsdp_HF_infer")
    seed_everything(hparams.seed)
    seed_everything(hparams.seed)
    dist.init_process_group("nccl")
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    rank = int(os.environ.get("RANK", -1))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    logger.info(
        f"Start running basic deepspeed example on total {world_size} computers, {rank}'s process on {local_rank}'s gpu."
    )

    assert world_size > -1 and rank > -1 and local_rank > -1, "Your distributed environ is wrong, plz check and retry!"

    torch.cuda.set_device(local_rank)

    config = AutoConfig.from_pretrained(hparams.transformers_model_name)
    config.num_labels = 2
    tokenizer = AutoTokenizer.from_pretrained(hparams.transformers_model_name)
    model = RobertaForSequenceClassification(config)

    load_model_checkpoint(model, local_rank, hparams.model_path, logger)

    fsdp_config = json_to_dict(hparams.fsdp_config)
    if hparams.model_dtype:
        logger.warning(f"your argument input precision is passed, will use fsdp_config file {hparams.fsdp_config}")

    mixed_precision_policy = None
    if fsdp_config["mixed_precision"]:
        if not fsdp_config["use_fp16"]:
            assert bfloat_support(), "Your machine is not supported bf16"
            mixed_precision_policy = mixed_precision.bfSixteen
            if rank == 0:
                logger.info("bFloat16 enabled for mixed precision - using bfSixteen policy")
            if hparams.model_dtype not in ["bf16", "bfloat16"]:
                logger.warning(
                    f"model will be bfloat16 mixed_precision, but your model_dtype input {hparams.model_dtype}"
                )
        elif fsdp_config["use_fp16"]:
            mixed_precision_policy = mixed_precision.fpSixteen
            if rank == 0:
                logger.info("FP16 enabled.")
        else:
            logger.info("bFloat16 support not present. Will use FP32, and not mixed precision")

    # TODO(User): you have to input your fsdp model layer
    wrapping_policy = get_transformers_wrapper(RobertaEncoder)

    imdb = load_dataset(hparams.data_path)

    def preprocess(examples):
        tokenized_text = tokenizer(examples["text"], return_token_type_ids=False, return_tensors="pt")
        examples["input_ids"] = tokenized_text["input_ids"][0]
        examples["attention_mask"] = tokenized_text["attention_mask"][0]
        examples["labels"] = int(examples["label"])

        return examples

    def filter_and_min_sample(datasets: Dataset, max_length: int = 512, min_sample_count: dict[str, int] = None):
        datasets = datasets.filter(lambda x: len(x["input_ids"]) <= max_length)
        true_datasets = datasets.filter(lambda x: x["labels"] == 1)
        false_datasets = datasets.filter(lambda x: x["labels"] == 0)
        if min_sample_count:
            sampling_count = min(len(true_datasets), len(false_datasets), min_sample_count["all"])
        else:
            sampling_count = min(len(true_datasets), len(false_datasets))
        sampling_true = Dataset.from_dict(true_datasets.shuffle()[:sampling_count])
        sampling_false = Dataset.from_dict(false_datasets.shuffle()[:sampling_count])
        result = concatenate_datasets([sampling_true, sampling_false])
        assert len(result) % 2 == 0, "`split=all` sampling error check plz"
        return result

    test_dataset = imdb["test"].map(preprocess, remove_columns=imdb["test"].column_names)
    test_dataset = filter_and_min_sample(test_dataset, tokenizer.model_max_length)

    from transformers import DataCollatorWithPadding

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    if hparams.group_by_length:
        # TODO(User): model_input_name is changed by your dataset's lengths column!
        custom_test_sampler = DistributedLengthGroupedSampler(
            batch_size=hparams.per_device_test_batch_size,
            dataset=test_dataset,
            rank=rank,
            seed=hparams.seed,
            shuffle=False,
            model_input_name="input_ids",
        )
    else:
        custom_test_sampler = DistributedSampler(test_dataset, seed=hparams.seed, rank=rank, shuffle=False)

    test_kwargs = {"batch_size": hparams.per_device_test_batch_size, "sampler": custom_test_sampler}
    cuda_kwargs = {"num_workers": hparams.num_workers, "pin_memory": True, "shuffle": False}
    test_kwargs.update(cuda_kwargs)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, collate_fn=data_collator, **test_kwargs)

    fsdp_model = FSDP(
        model,
        auto_wrap_policy=wrapping_policy,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=SHARDING_STRATEGY[fsdp_config["sharding_strategy"]],
        device_id=local_rank,
        limit_all_gathers=fsdp_config["limit_all_gathers"],
        cpu_offload=CPUOffload(offload_params=fsdp_config["offload_params"]),
    )

    # TODO(User): input your eval_metric
    eval_metric = evaluate.load("accuracy")
    trainer = FSDPTrainer(
        device_id=local_rank,
        criterion=None,
        eval_metric=eval_metric,
        precision=hparams.model_dtype,
        cmd_logger=logger,
        max_epochs=1,
        grad_accum_steps=hparams.accumulate_grad_batches,
        metric_on_cpu=hparams.metric_on_cpu,
    )

    trainer.test_loop(model=fsdp_model, test_loader=test_dataloader)

    dist.destroy_process_group()


if __name__ == "__main__":
    assert torch.distributed.is_available(), "DDP is only multi gpu!! check plz!"
    assert torch.cuda.is_available(), "CPU training is not allowed."
    parser = ArgumentParser()
    parser.add_arguments(InferenceArguments, dest="training_args")
    args = parser.parse_args()
    args = dataclass_to_namespace(args, "training_args")

    main(args)
