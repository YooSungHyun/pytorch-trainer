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

from setproctitle import setproctitle
from simple_parsing import ArgumentParser
from sklearn.preprocessing import MinMaxScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, random_split
from utils.comfy import dataclass_to_namespace, seed_everything, bfloat_support
from utils.data.custom_dataloader import CustomDataLoader
from utils.data.custom_sampler import DistributedLengthGroupedSampler

import policies
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)8s] %(message)s")

timeFileHandler = StreamHandler()
timeFileHandler.setFormatter(formatter)

logger.addHandler(timeFileHandler)


def get_policies(cfg, rank):
    """establish current policies for mixed precision and fsdp wrapping"""

    mixed_precision_policy = None
    wrapping_policy = None

    # mixed precision -----
    if cfg.mixed_precision:
        bfloat_available = bfloat_support()
        if bfloat_available and not cfg.use_fp16:
            mixed_precision_policy = policies.bfSixteen
            if rank == 0:
                print("bFloat16 enabled for mixed precision - using bfSixteen policy")
        elif cfg.use_fp16:
            mixed_precision_policy = policies.fpSixteen
            if rank == 0:
                print("FP16 enabled.")
        else:
            # mixed_precision_policy = policies.fpSixteen
            print("bFloat16 support not present. Will use FP32, and not mixed precision")

    wrapping_policy = policies.get_t5_wrapper()

    return mixed_precision_policy, wrapping_policy


def main(hparams: TrainingArguments):
    dist.init_process_group("nccl")
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    rank = int(os.environ.get("RANK", -1))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    logger.info(
        f"Start running basic deepspeed example on total {world_size} computers, {rank}'s process on {local_rank}'s gpu."
    )

    assert world_size > -1 and rank > -1 and local_rank > -1, "Your distributed environ is wrong, plz check and retry!"

    torch.cuda.set_device(local_rank)
    fsdp_config = 
    if hparams.model_dtype in ["bf16","bfloat16","fp16","float16"]:

    mixed_precision_policy, t5_auto_wrap_policy = get_policies(train_config, rank)
    # reference: https://www.kaggle.com/code/anitarostami/lstm-multivariate-forecasting
    setproctitle(os.environ.get("WANDB_PROJECT", "torch-trainer"))
    web_logger = None
    if local_rank == 0:
        web_logger = wandb.init(config=hparams)
    seed_everything(hparams.seed)
    os.makedirs(hparams.output_dir, exist_ok=True)

    model = T5ForConditionalGeneration.from_pretrained(hparams.transformers_model_name)
    tokenizer = T5Tokenizer.from_pretrained(hparams.transformers_model_name)

    df_train = pd.read_csv(hparams.train_datasets_path, header=0, encoding="utf-8")
    # Kaggle author Test Final RMSE: 0.06539
    df_eval = pd.read_csv(hparams.eval_datasets_path, header=0, encoding="utf-8")

    df_train_scaled = df_train.copy()
    df_test_scaled = df_eval.copy()

    # Define the mapping dictionary
    mapping = {"NE": 0, "SE": 1, "NW": 2, "cv": 3}

    # Replace the string values with numerical values
    df_train_scaled["wnd_dir"] = df_train_scaled["wnd_dir"].map(mapping)
    df_test_scaled["wnd_dir"] = df_test_scaled["wnd_dir"].map(mapping)

    df_train_scaled["date"] = pd.to_datetime(df_train_scaled["date"])
    # Resetting the index
    df_train_scaled.set_index("date", inplace=True)
    logger.info(df_train_scaled.head())

    scaler = MinMaxScaler()

    # Define the columns to scale
    columns = ["pollution", "dew", "temp", "press", "wnd_dir", "wnd_spd", "snow", "rain"]

    df_test_scaled = df_test_scaled[columns]

    # Scale the selected columns to the range 0-1
    df_train_scaled[columns] = scaler.fit_transform(df_train_scaled[columns])
    df_test_scaled[columns] = scaler.transform(df_test_scaled[columns])

    # Show the scaled data
    logger.info(df_train_scaled.head())

    df_train_scaled = np.array(df_train_scaled)
    df_test_scaled = np.array(df_test_scaled)

    x = []
    y = []
    n_future = 1
    n_past = 11

    #  Train Sets
    for i in range(n_past, len(df_train_scaled) - n_future + 1):
        x.append(df_train_scaled[i - n_past : i, 1 : df_train_scaled.shape[1]])
        y.append(df_train_scaled[i + n_future - 1 : i + n_future, 0])
    x_train, y_train = np.array(x), np.array(y)

    #  Test Sets
    x = []
    y = []
    for i in range(n_past, len(df_test_scaled) - n_future + 1):
        x.append(df_test_scaled[i - n_past : i, 1 : df_test_scaled.shape[1]])
        y.append(df_test_scaled[i + n_future - 1 : i + n_future, 0])
    x_test, y_test = np.array(x), np.array(y)

    logger.info(
        "X_train shape : {}   y_train shape : {} \n"
        "X_test shape : {}      y_test shape : {} ".format(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    )

    train_dataset = NumpyDataset(
        x_train,
        y_train,
        feature_column_name=hparams.feature_column_name,
        labels_column_name=hparams.labels_column_name,
    )
    eval_dataset = NumpyDataset(
        x_test,
        y_test,
        feature_column_name=hparams.feature_column_name,
        labels_column_name=hparams.labels_column_name,
    )
    # if hparams.eval_datasets_path:
    #     eval_df = pd.read_csv(hparams.eval_datasets_path)
    #     train_dataset = PandasDataset(train_df, length_column_name=hparams.length_column_name)
    # else:
    #     # it is just for lstm example
    #     train_df = train_df[::-1]
    #     train_size = int(len(train_df) * hparams.train_data_ratio)
    #     splited_train_df = train_df[0:train_size]
    #     eval_df = train_df[train_size - seq_length :]
    #     train_dataset = PandasDataset(splited_train_df, length_column_name=hparams.length_column_name)
    #     # if you use another one, plz check here
    #     # train_size = int(hparams.train_data_ratio * len(train_df))
    #     # eval_size = len(train_df) - train_size
    #     # train_dataset = PandasDataset(hparams.train_datasets_path)
    #     # train_dataset, eval_dataset = random_split(train_dataset, [train_size, eval_size])
    # eval_dataset = PandasDataset(eval_df, length_column_name=hparams.length_column_name)

    # Instantiate objects
    model = Net().cuda(device_id)
    ddp_model = DDP(model, device_ids=[device_id], find_unused_parameters=True)

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
            model_input_name=train_dataset.length_column_name,
        )
        custom_eval_sampler = DistributedLengthGroupedSampler(
            batch_size=hparams.per_device_eval_batch_size,
            dataset=eval_dataset,
            rank=rank,
            seed=hparams.seed,
            model_input_name=eval_dataset.length_column_name,
        )
    else:
        # DistributedSampler's shuffle: each device get random indices data segment in every epoch
        custom_train_sampler = DistributedSampler(
            train_dataset, seed=hparams.seed, rank=rank, shuffle=hparams.sampler_shuffle
        )
        custom_eval_sampler = DistributedSampler(eval_dataset, seed=hparams.seed, rank=rank, shuffle=False)

    # DataLoader's shuffle: one device get random indices dataset in every epoch
    # example np_dataset is already set (feature)7:1(label), so, it can be all shuffle `True` between sampler and dataloader
    train_dataloader = CustomDataLoader(
        dataset=train_dataset,
        feature_column_name=hparams.feature_column_name,
        labels_column_name=hparams.labels_column_name,
        batch_size=hparams.per_device_train_batch_size,
        sampler=custom_train_sampler,
        num_workers=hparams.num_workers,
        drop_last=hparams.dataloader_drop_last,
        pin_memory=True,
        persistent_workers=True,
    )

    eval_dataloader = CustomDataLoader(
        dataset=eval_dataset,
        feature_column_name=hparams.feature_column_name,
        labels_column_name=hparams.labels_column_name,
        batch_size=hparams.per_device_eval_batch_size,
        sampler=custom_eval_sampler,
        num_workers=hparams.num_workers,
        drop_last=hparams.dataloader_drop_last,
        pin_memory=True,
        persistent_workers=True,
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
    criterion = torch.nn.MSELoss()
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
    )

    trainer.fit(
        model=ddp_model,
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