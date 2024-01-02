import logging
import math
import os
from logging import StreamHandler

import deepspeed
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import wandb
from arguments.training_args import TrainingArguments
from networks.models import Net
from setproctitle import setproctitle
from simple_parsing import ArgumentParser
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DistributedSampler, random_split
from trainer.deepspeed import Trainer
from utils.comfy import dataclass_to_namespace, seed_everything, json_to_dict, update_auto_nested_dict
from utils.data.custom_dataloader import CustomDataLoader
from utils.data.custom_sampler import DistributedLengthGroupedSampler
from utils.data.np_dataset import NumpyDataset
from utils.data.pd_dataset import PandasDataset
from torch_optimizer import Adafactor


logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)8s] %(message)s")

timeFileHandler = StreamHandler()
timeFileHandler.setFormatter(formatter)

logger.addHandler(timeFileHandler)


def main(hparams: TrainingArguments):
    world_size = os.environ.get("WORLD_SIZE", None)
    if world_size:
        world_size = int(world_size)
    else:
        world_size = torch.cuda.device_count()

    rank = os.environ.get("LOCAL_RANK", None)
    if rank:
        rank = int(rank)
    else:
        rank = dist.get_rank()
    # logger.info(f"Start running basic DDP example on rank {rank}.")

    device_id = rank % world_size
    torch.cuda.set_device(device_id)
    deepspeed.init_distributed("nccl")

    # reference: https://www.kaggle.com/code/anitarostami/lstm-multivariate-forecasting
    setproctitle(os.environ.get("WANDB_PROJECT", "torch-trainer"))
    web_logger = None
    if device_id == 0:
        web_logger = wandb.init(config=hparams)
        web_logger.log_code(hparams.deepspeed_config)
    seed_everything(hparams.seed)
    os.makedirs(hparams.output_dir, exist_ok=True)

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

    if hparams.group_by_length:
        custom_train_sampler = DistributedLengthGroupedSampler(
            batch_size=hparams.per_device_train_batch_size,
            dataset=train_dataset,
            rank=device_id,
            seed=hparams.seed,
            model_input_name=train_dataset.length_column_name,
        )
        custom_eval_sampler = DistributedLengthGroupedSampler(
            batch_size=hparams.per_device_eval_batch_size,
            dataset=eval_dataset,
            rank=device_id,
            seed=hparams.seed,
            model_input_name=eval_dataset.length_column_name,
        )
    else:
        # DistributedSampler's shuffle: each device get random indices data segment in every epoch
        custom_train_sampler = DistributedSampler(
            train_dataset, seed=hparams.seed, rank=device_id, shuffle=hparams.sampler_shuffle
        )
        custom_eval_sampler = DistributedSampler(eval_dataset, seed=hparams.seed, rank=device_id, shuffle=False)

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

    # Instantiate objects
    model = Net().cuda(device_id)

    if device_id == 0:
        web_logger.watch(model, log_freq=hparams.log_every_n)

    optimizer = None
    initial_lr = hparams.learning_rate / hparams.div_factor

    # if using OneCycleLR, optim lr is not important,
    # torch native, lr update to start_lr automatically
    # deepspeed, lr update to end_lr automatically
    # optimizer = Adafactor(
    #     model.parameters(),
    #     lr=hparams.learning_rate,
    #     beta1=hparams.optim_beta1,
    #     weight_decay=hparams.weight_decay,
    # )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hparams.learning_rate,
        eps=hparams.optim_eps,
        betas=(hparams.optim_beta1, hparams.optim_beta2),
        weight_decay=hparams.weight_decay,
    )

    cycle_momentum = True
    if isinstance(optimizer, Adafactor):
        cycle_momentum = False

    # TODO(user): If you want to using deepspeed lr_scheduler, change this code line
    # max_lr = hparams.learning_rate
    # initial_lr = hparams.learning_rate / hparams.div_factor
    # min_lr = hparams.learning_rate / hparams.final_div_factor
    scheduler = None
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=hparams.learning_rate,
        pct_start=hparams.warmup_ratio,
        epochs=hparams.max_epochs,
        div_factor=hparams.div_factor,
        final_div_factor=hparams.final_div_factor,
        steps_per_epoch=steps_per_epoch,
        cycle_momentum=cycle_momentum,
    )

    # If you want to using own optimizer and scheduler,
    # check config `zero_allow_untested_optimizer` and `zero_force_ds_cpu_optimizer`
    ds_config = json_to_dict(hparams.deepspeed_config)
    update_auto_nested_dict(ds_config, "lr", initial_lr)
    update_auto_nested_dict(ds_config, "train_micro_batch_size_per_gpu", hparams.per_device_train_batch_size)
    update_auto_nested_dict(ds_config, "gradient_accumulation_steps", hparams.accumulate_grad_batches)
    if "fp16" in ds_config.keys() and ds_config["fp16"]["enabled"]:
        hparams.model_dtype = "fp16"
    elif "bfp16" in ds_config.keys() and ds_config["bf16"]["enabled"]:
        hparams.model_dtype = "bf16"
    else:
        hparams.model_dtype = "fp32"

    # Since the deepspeed lr scheduler is, after all, just a generic object-inherited custom scheduler, Only authorize the use of torch scheduler.
    # Also, the ZeroOptimizer.param_groups address is the same as the torch scheduler.optimizer.param_groups address.
    # Therefore, there is absolutely no reason to use the lr_scheduler provided by Deepspeed.
    assert (
        scheduler is not None or "scheduler" not in ds_config.keys()
    ), "Don't use Deepspeed Scheduler!!!!, It is so confused. Plz implement something!"

    if optimizer is not None:
        from deepspeed.runtime.zero.utils import is_zero_supported_optimizer

        if not is_zero_supported_optimizer(optimizer):
            ds_config.update({"zero_allow_untested_optimizer": True})
        if "zero_optimization" in ds_config.keys():
            if "offload_optimizer" in ds_config["zero_optimization"].keys():
                # custom optimizer and using cpu offload
                ds_config.update({"zero_force_ds_cpu_optimizer": False})

    # 0: model, 1: optimizer, 2: dataloader, 3: lr scheduler가 나온다
    # dataloader는 deepspeed에서 권장하는 세팅이지만, 어짜피 distributedsampler 적용된 놈이 나온다.
    # if optimizer and scheduler is None, it is initialized by ds_config
    model, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=None,
        dist_init_required=True,
        config=ds_config,
    )

    # monitor: ReduceLROnPlateau scheduler is stepped using loss, so monitor input train or val loss
    lr_scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1, "monitor": None}
    assert id(scheduler) == id(lr_scheduler["scheduler"]), "scheduler mismatch! plz check!!!!"
    criterion = torch.nn.MSELoss()
    trainable_loss = None

    # I think some addr is same into trainer init&fit respectfully
    chk_addr_dict = {
        "train_dataloader": id(train_dataloader),
        "eval_dataloader": id(eval_dataloader),
        "model": id(model),
        "optimizer": id(optimizer.param_groups),
        "criterion": id(criterion),
        "scheduler_cfg": id(lr_scheduler),
        "scheduler_cfg[scheduler]": id(lr_scheduler["scheduler"]),
        "scheduler_cfg[scheduler].optimizer.param_groups": id(lr_scheduler["scheduler"].optimizer.param_groups),
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
    scheduler's optimizer value addr: {chk_addr_dict["scheduler_cfg[scheduler].optimizer.param_groups"]}
    ##########################################
    """
    logger.debug(log_str)

    trainer = Trainer(
        device_id=device_id,
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
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler_cfg=lr_scheduler,
        train_loader=train_dataloader,
        val_loader=eval_dataloader,
        ckpt_path=hparams.output_dir,
        trainable_loss=trainable_loss,
    )

    if device_id == 0:
        web_logger.finish(exit_code=0)


if __name__ == "__main__":
    assert torch.distributed.is_available(), "DDP is only multi gpu!! check plz!"
    assert torch.cuda.is_available(), "CPU training is not allowed."
    parser = ArgumentParser()
    parser.add_arguments(TrainingArguments, dest="training_args")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    args = dataclass_to_namespace(args, "training_args")

    main(args)
