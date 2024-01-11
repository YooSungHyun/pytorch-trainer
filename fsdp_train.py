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
from arguments.training_args import TrainingArguments
from datasets import Dataset, concatenate_datasets, load_dataset
from setproctitle import setproctitle
from simple_parsing import ArgumentParser
from torch.cuda.amp import autocast
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import CPUOffload
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.utils.data import DistributedSampler
from trainer.fsdp import Trainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.models.roberta.modeling_roberta import RobertaEncoder
from utils.comfy import (
    apply_to_collection,
    bfloat_support,
    dataclass_to_namespace,
    json_to_dict,
    seed_everything,
    tensor_dict_to_device,
    web_log_every_n,
)
from utils.data.custom_sampler import DistributedLengthGroupedSampler
from utils.FSDP import mixed_precision
from utils.FSDP.wrapping import get_transformers_wrapper

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
        self,
        device_id,
        criterion,
        eval_metric,
        precision: str = "fp32",
        cmd_logger=None,
        web_logger=None,
        max_epochs: Optional[int] = 1000,
        max_steps: Optional[int] = None,
        grad_accum_steps: int = 1,
        limit_train_batches: Union[int, float] = float("inf"),
        limit_val_batches: Union[int, float] = float("inf"),
        validation_frequency: int = 1,
        checkpoint_dir: str = "./checkpoints",
        checkpoint_frequency: int = 1,
        chk_addr_dict: dict = None,
        non_blocking: bool = True,
        log_every_n: int = 1,
        max_norm: float = 0.0,
        metric_on_cpu: bool = False,
    ):
        super().__init__(
            device_id,
            criterion,
            eval_metric,
            precision,
            cmd_logger,
            web_logger,
            max_epochs,
            max_steps,
            grad_accum_steps,
            limit_train_batches,
            limit_val_batches,
            validation_frequency,
            checkpoint_dir,
            checkpoint_frequency,
            chk_addr_dict,
            non_blocking,
            log_every_n,
            max_norm,
            metric_on_cpu,
        )

    def training_step(self, model, batch, batch_idx):
        # TODO(User): fit the input and output for your model architecture!
        with autocast(enabled=self.mixed_precision, dtype=self.precision):
            output = model(**batch)
            loss = output["loss"]

        def on_before_backward(loss):
            pass

        on_before_backward(loss)

        self.grad_scaler.scale(loss).backward()

        def on_after_backward():
            pass

        on_after_backward()

        log_output = {"loss": loss}
        # avoid gradients in stored/accumulated values -> prevents potential OOM
        self._current_train_return = apply_to_collection(log_output, dtype=torch.Tensor, function=lambda x: x.detach())

        web_log_every_n(
            self.web_logger,
            {
                "train/loss": self._current_train_return["loss"],
                "train/step": self.step,
                "train/global_step": self.global_step,
                "train/epoch": self.current_epoch,
            },
            self.step,
            self.log_every_n,
            self.device_id,
        )
        return loss

    def eval_loop(
        self,
        model,
        val_loader: Optional[torch.utils.data.DataLoader],
        limit_batches: Union[int, float] = float("inf"),
    ):
        """The validation loop ruunning a single validation epoch.

        Args:
            model: model
            val_loader: The dataloader yielding the validation batches.
            limit_batches: Limits the batches during this validation epoch.
                If greater than the number of batches in the ``val_loader``, this has no effect.

        """
        # no validation if val_loader wasn't passed
        if val_loader is None:
            return

        def on_start_eval(model):
            model.eval()
            # requires_grad = True, but loss.backward() raised error
            # because grad_fn is None
            torch.set_grad_enabled(False)

        on_start_eval(model)

        def on_validation_epoch_start():
            pass

        if self.device_id == 0:
            iterable = self.progbar_wrapper(val_loader, total=min(len(val_loader), limit_batches), desc="Validation")
            pbar = enumerate(iterable)
        else:
            pbar = enumerate(val_loader)

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

            # end epoch if stopping training completely or max batches for this epoch reached
            if self.should_stop or batch_idx >= limit_batches:
                break

            def on_validation_batch_start(batch, batch_idx):
                pass

            on_validation_batch_start(batch, batch_idx)

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

            def on_validation_batch_end(eval_out, batch, batch_idx):
                pass

            on_validation_batch_end(self._current_val_return, batch, batch_idx)

            web_log_every_n(
                self.web_logger,
                {
                    "eval_step/loss": self._current_val_return["loss"],
                    "eval_step/step": eval_step,
                    "eval_step/global_step": self.global_step,
                    "eval_step/epoch": self.current_epoch,
                },
                eval_step,
                self.log_every_n,
                self.device_id,
            )
            cmd_output = {"loss": self._current_val_return["loss"]}
            if self.device_id == 0:
                self._format_iterable(iterable, cmd_output, "val")
            eval_step += 1

        device_total_loss = torch.mean(torch.tensor(tot_batch_loss, dtype=loss.dtype, device=metric_on_device))

        # TODO(User): Create any form you want to output to wandb!
        def on_validation_epoch_end(tot_batch_logits, tot_batch_labels, device_total_loss, metric_device):
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

            # example 4 gpus : [gpu0[tensor],gpu1[tensor],gpu2[tensor],gpu3[tensor]]
            logits_gathered_data = torch.cat(logits_gathered_data, dim=0)
            predictions = torch.argmax(logits_gathered_data, axis=-1)
            references = torch.cat(labels_gathered_data, dim=0)

            total_acc = self.eval_metric.compute(predictions=predictions, references=references)
            # epoch monitoring is must doing every epoch
            web_log_every_n(
                self.web_logger,
                {"eval/loss": total_loss, "eval/acc": total_acc, "eval/epoch": self.current_epoch},
                self.current_epoch,
                1,
                self.device_id,
            )

        on_validation_epoch_end(tot_batch_logits, tot_batch_labels, device_total_loss, metric_on_device)

        def on_validation_model_train(model):
            torch.set_grad_enabled(True)
            model.train()

        on_validation_model_train(model)


def main(hparams: TrainingArguments):
    # reference: https://github.com/pytorch/examples/blob/main/distributed/FSDP/T5_training.py
    setproctitle(os.environ.get("WANDB_PROJECT", "torch-trainer"))
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

    web_logger = None
    if local_rank == 0:
        web_logger = wandb.init(config=hparams)

    model = AutoModelForSequenceClassification.from_pretrained(hparams.transformers_model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(hparams.transformers_model_name)

    imdb = load_dataset("imdb")

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

    train_dataset = imdb["train"].map(preprocess, remove_columns=imdb["train"].column_names)
    train_dataset = filter_and_min_sample(train_dataset, tokenizer.model_max_length)
    eval_dataset = imdb["test"].map(preprocess, remove_columns=imdb["test"].column_names)
    eval_dataset = filter_and_min_sample(eval_dataset, tokenizer.model_max_length)

    from transformers import DataCollatorWithPadding

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    if local_rank == 0:
        web_logger.watch(model, log_freq=hparams.log_every_n)

    if hparams.group_by_length:
        # TODO(User): model_input_name is changed by your dataset's lengths column!
        custom_train_sampler = DistributedLengthGroupedSampler(
            batch_size=hparams.per_device_train_batch_size,
            dataset=train_dataset,
            rank=rank,
            seed=hparams.seed,
            model_input_name="input_ids",
        )
        custom_eval_sampler = DistributedLengthGroupedSampler(
            batch_size=hparams.per_device_eval_batch_size,
            dataset=eval_dataset,
            rank=rank,
            seed=hparams.seed,
            model_input_name="input_ids",
        )
    else:
        # DistributedSampler's shuffle: each device get random indices data segment in every epoch
        custom_train_sampler = DistributedSampler(
            train_dataset, seed=hparams.seed, rank=rank, shuffle=hparams.sampler_shuffle
        )
        custom_eval_sampler = DistributedSampler(eval_dataset, seed=hparams.seed, rank=rank, shuffle=False)

    train_kwargs = {"batch_size": hparams.per_device_train_batch_size, "sampler": custom_train_sampler}
    test_kwargs = {"batch_size": hparams.per_device_eval_batch_size, "sampler": custom_eval_sampler}
    cuda_kwargs = {"num_workers": hparams.num_workers, "pin_memory": True, "shuffle": False}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, collate_fn=data_collator, **train_kwargs)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, collate_fn=data_collator, **test_kwargs)

    fsdp_model = FSDP(
        model,
        auto_wrap_policy=wrapping_policy,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=SHARDING_STRATEGY[fsdp_config["sharding_strategy"]],
        device_id=local_rank,
        limit_all_gathers=fsdp_config["limit_all_gathers"],
        cpu_offload=CPUOffload(offload_params=fsdp_config["offload_params"]),
    )

    # optimizer have to initialize after FSDP
    optimizer = torch.optim.AdamW(
        fsdp_model.parameters(),
        lr=hparams.learning_rate,
        eps=hparams.optim_eps,
        betas=(hparams.optim_beta1, hparams.optim_beta2),
        weight_decay=hparams.weight_decay,
    )

    if fsdp_config["fsdp_activation_checkpointing"]:
        non_reentrant_wrapper = partial(
            checkpoint_wrapper,
            offload_to_cpu=fsdp_config["offload_params"],
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        # apply activation checkpointing to model returns None as model is updated directly
        # TODO(User): Input your fsdp model layer in check_fn!
        apply_activation_checkpointing(
            fsdp_model,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=lambda submodule: isinstance(submodule, RobertaEncoder),
        )

        def is_forward_signature_columns(module, key):
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(module.forward)
            input_columns = list(signature.parameters.keys())
            return key in input_columns

        def remove_columns(module, args, kwargs):
            for key in list(kwargs.keys()):
                if not is_forward_signature_columns(module, key):
                    kwargs.pop(key)
            return (args, kwargs)

        """ TODO(User): check your model architecture, and change this line
            When using activation_checkpointing in the FSDP module, we found that the transformers model input kwargs contained unnecessary keys (ex. `offload_to_cpu`),
            so we used a hook in nn.module to force the values to be cleaned up before input.
            Since the structure of your model and the section to be checkpointed may be different, please check and modify it before using it.
        """
        fsdp_model.roberta.encoder._fsdp_wrapped_module._checkpoint_wrapped_module.register_forward_pre_hook(
            remove_columns, with_kwargs=True
        )
        # for i in range(len(fsdp_model.decoder.block)):
        #     fsdp_model.decoder.block[i]._fsdp_wrapped_module._checkpoint_wrapped_module.register_forward_pre_hook(
        #         remove_columns, with_kwargs=True
        #     )

    # dataloader already calculate len(total_data) / (batch_size * dist.get_world_size())
    # accumulation is always floor
    steps_per_epoch = math.floor(len(train_dataloader) / hparams.accumulate_grad_batches)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=hparams.learning_rate,
        pct_start=hparams.warmup_ratio,
        epochs=hparams.max_epochs,
        final_div_factor=hparams.final_div_factor,
        steps_per_epoch=steps_per_epoch,
    )

    # monitor: ReduceLROnPlateau scheduler is stepped using loss, so monitor input train or val loss
    lr_scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1, "monitor": None}
    assert id(scheduler) == id(lr_scheduler["scheduler"])

    # this example is not needs criterion and trainable_loss
    criterion = None
    trainable_loss = None

    # I think some addr is same into trainer init&fit respectfully
    chk_addr_dict = {
        "train_dataloader": id(train_dataloader),
        "eval_dataloader": id(eval_dataloader),
        "model": id(model),
        "optimizer": id(optimizer),
        "criterion": id(criterion),
        "scheduler_cfg": id(lr_scheduler),
        "scheduler_cfg[scheduler]": id(lr_scheduler["scheduler"]),
        "trainable_loss": id(trainable_loss),
    }

    log_str = f"""\n##########################################
    train_dataloader addr: {chk_addr_dict["train_dataloader"]}
    eval_dataloader addr: {chk_addr_dict["eval_dataloader"]}
    model addr: {chk_addr_dict["model"]}
    optimizer addr: {chk_addr_dict["optimizer"]}
    criterion addr: {chk_addr_dict["criterion"]}
    scheduler_cfg addr: {chk_addr_dict["scheduler_cfg"]}
    scheduler addr: {chk_addr_dict["scheduler_cfg[scheduler]"]}
    ##########################################
    """
    logger.debug(log_str)
    # TODO(User): input your eval_metric
    eval_metric = evaluate.load("accuracy")
    trainer = FSDPTrainer(
        device_id=local_rank,
        criterion=criterion,
        eval_metric=eval_metric,
        precision=hparams.model_dtype,
        cmd_logger=logger,
        web_logger=web_logger,
        max_epochs=hparams.max_epochs,
        grad_accum_steps=hparams.accumulate_grad_batches,
        chk_addr_dict=chk_addr_dict,
        checkpoint_dir=hparams.output_dir,
        log_every_n=hparams.log_every_n,
        max_norm=hparams.max_norm,
        metric_on_cpu=hparams.metric_on_cpu,
    )

    trainer.fit(
        model=fsdp_model,
        optimizer=optimizer,
        scheduler_cfg=lr_scheduler,
        train_loader=train_dataloader,
        val_loader=eval_dataloader,
        ckpt_path=hparams.output_dir,
        trainable_loss=trainable_loss,
    )

    from utils.model_checkpointing.fsdp_handler import save_model_checkpoint

    if local_rank == 0:
        web_logger.finish(exit_code=0)
    dist.destroy_process_group()


if __name__ == "__main__":
    assert torch.distributed.is_available(), "DDP is only multi gpu!! check plz!"
    assert torch.cuda.is_available(), "CPU training is not allowed."
    parser = ArgumentParser()
    parser.add_arguments(TrainingArguments, dest="training_args")
    args = parser.parse_args()
    args = dataclass_to_namespace(args, "training_args")

    main(args)
