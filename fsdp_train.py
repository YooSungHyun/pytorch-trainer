import logging
import math
import os
from logging import StreamHandler

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
from arguments.training_args import TrainingArguments

from trainer.fsdp import Trainer
from transformers import AutoTokenizer, GPT2TokenizerFast
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5Block
from datasets import load_dataset
from setproctitle import setproctitle
from simple_parsing import ArgumentParser
from sklearn.preprocessing import MinMaxScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, random_split
from utils.comfy import dataclass_to_namespace, seed_everything, bfloat_support, json_to_dict
from utils.data.summarization_dataset import wikihow
from utils.data.custom_dataloader import CustomDataLoader
from utils.data.custom_sampler import DistributedLengthGroupedSampler

from utils.fsdp import mixed_precision
from utils.fsdp.wrapping import get_transformers_wrapper
from utils.fsdp.activation_checkpointing_functions import apply_fsdp_checkpointing
import inspect


from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)

from functools import partial
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)


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

CHECKPOINT_TYPE = {
    "FULL_STATE_DICT": StateDictType.FULL_STATE_DICT,
    "LOCAL_STATE_DICT": StateDictType.LOCAL_STATE_DICT,
    "SHARDED_STATE_DICT": StateDictType.SHARDED_STATE_DICT,
}


def main(hparams: TrainingArguments):
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

    wrapping_policy = get_transformers_wrapper(T5Block)

    # reference: https://github.com/pytorch/examples/blob/main/distributed/FSDP/T5_training.py
    web_logger = None
    if local_rank == 0:
        web_logger = wandb.init(config=hparams)

    os.makedirs(hparams.output_dir, exist_ok=True)

    model = T5ForConditionalGeneration.from_pretrained(hparams.transformers_model_name)
    tokenizer = T5Tokenizer.from_pretrained(hparams.transformers_model_name)

    train_dataset = wikihow(tokenizer, "train", "./raw_data/", 1500, 512, 150, False)
    eval_dataset = wikihow(tokenizer, "validation", "./raw_data/", 300, 512, 150, False)

    if local_rank == 0:
        web_logger.watch(model, log_freq=hparams.log_every_n)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hparams.learning_rate,
        eps=hparams.optim_eps,
        betas=(hparams.optim_beta1, hparams.optim_beta2),
        weight_decay=hparams.weight_decay,
    )

    if hparams.group_by_length:
        custom_train_sampler = DistributedLengthGroupedSampler(
            batch_size=hparams.per_device_train_batch_size,
            dataset=train_dataset,
            rank=rank,
            seed=hparams.seed,
            model_input_name="source_id",
        )
        custom_eval_sampler = DistributedLengthGroupedSampler(
            batch_size=hparams.per_device_eval_batch_size,
            dataset=eval_dataset,
            rank=rank,
            seed=hparams.seed,
            model_input_name="source_id",
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

    train_dataloader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, **test_kwargs)
    # DataLoader's shuffle: one device get random indices dataset in every epoch
    # example np_dataset is already set (feature)7:1(label), so, it can be all shuffle `True` between sampler and dataloader
    # TODO(user): If you not needs, just make #comment#
    # train_dataloader = CustomDataLoader(
    #     dataset=train_dataset,
    #     feature_column_name=hparams.feature_column_name,
    #     labels_column_name=hparams.labels_column_name,
    #     batch_size=hparams.per_device_train_batch_size,
    #     sampler=custom_train_sampler,
    #     num_workers=hparams.num_workers,
    #     drop_last=hparams.dataloader_drop_last,
    #     pin_memory=True,
    #     persistent_workers=True,
    # )

    # eval_dataloader = CustomDataLoader(
    #     dataset=eval_dataset,
    #     feature_column_name=hparams.feature_column_name,
    #     labels_column_name=hparams.labels_column_name,
    #     batch_size=hparams.per_device_eval_batch_size,
    #     sampler=custom_eval_sampler,
    #     num_workers=hparams.num_workers,
    #     drop_last=hparams.dataloader_drop_last,
    #     pin_memory=True,
    #     persistent_workers=True,
    # )

    fsdp_model = FSDP(
        model,
        auto_wrap_policy=wrapping_policy,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=SHARDING_STRATEGY[fsdp_config["sharding_strategy"]],
        device_id=local_rank,
        limit_all_gathers=fsdp_config["limit_all_gathers"],
        cpu_offload=CPUOffload(offload_params=fsdp_config["offload_params"]),
    )

    if fsdp_config["fsdp_activation_checkpointing"]:
        non_reentrant_wrapper = partial(
            checkpoint_wrapper,
            offload_to_cpu=fsdp_config["offload_params"],
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        # apply activation checkpointing to model returns None as model is updated directly
        apply_activation_checkpointing(
            fsdp_model,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=lambda submodule: isinstance(submodule, T5Block),
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
        for i in range(len(fsdp_model.encoder.block)):
            fsdp_model.encoder.block[i]._fsdp_wrapped_module._checkpoint_wrapped_module.register_forward_pre_hook(
                remove_columns, with_kwargs=True
            )
        for i in range(len(fsdp_model.decoder.block)):
            fsdp_model.decoder.block[i]._fsdp_wrapped_module._checkpoint_wrapped_module.register_forward_pre_hook(
                remove_columns, with_kwargs=True
            )

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

    trainer = Trainer(
        device_id=local_rank,
        precision=hparams.model_dtype,
        cmd_logger=logger,
        web_logger=web_logger,
        max_epochs=hparams.max_epochs,
        grad_accum_steps=hparams.accumulate_grad_batches,
        chk_addr_dict=chk_addr_dict,
        checkpoint_dir=hparams.output_dir,
        log_every_n=hparams.log_every_n,
        max_norm=hparams.max_norm,
        checkpoint_type=CHECKPOINT_TYPE[fsdp_config["checkpoint_type"]],
    )

    trainer.fit(
        model=fsdp_model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler_cfg=lr_scheduler,
        train_loader=train_dataloader,
        val_loader=eval_dataloader,
        ckpt_path=hparams.output_dir,
        trainable_loss=trainable_loss,
    )

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
