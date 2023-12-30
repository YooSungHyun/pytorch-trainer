import math
import os
from datetime import timedelta

import pandas as pd
import torch
from arguments.training_args import TrainingArguments
from custom_dataloader import CustomDataLoader
from custom_sampler import DistributedLengthGroupedSampler, LengthGroupedSampler
from networks.models import Net
from pd_dataset import PandasDataset
from np_dataset import NumpyDataset
from setproctitle import setproctitle
from simple_parsing import ArgumentParser
from torch.utils.data import random_split, DistributedSampler, RandomSampler, SequentialSampler
from utils.comfy import dataclass_to_namespace, seed_everything
from cpu_trainer import Trainer
import logging
from logging.handlers import TimedRotatingFileHandler
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import wandb

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)8s] %(message)s")

log_file_name = "./logs/output.log"

timeFileHandler = TimedRotatingFileHandler(filename=log_file_name, when="midnight", interval=1, encoding="utf-8")
timeFileHandler.setFormatter(formatter)

logger.addHandler(timeFileHandler)


def main(hparams: TrainingArguments):
    setproctitle(os.environ.get("WANDB_PROJECT", "lightning_logs"))
    web_logger = wandb.init(config=hparams)
    seed_everything(hparams.seed)
    os.makedirs(hparams.output_dir, exist_ok=True)

    train_df = pd.read_csv(hparams.train_datasets_path, encoding="utf-8")
    # 7일간의 데이터가 입력으로 들어가고 batch size는 임의로 지정
    seq_length = 7

    train_df = train_df[::-1]
    train_size = int(len(train_df) * 0.7)
    train_set = train_df[0:train_size]
    test_set = train_df[train_size - seq_length :]

    # Input scale
    scaler_x = MinMaxScaler()
    scaler_x.fit(train_set.iloc[:, :-1])

    train_set.iloc[:, :-1] = scaler_x.transform(train_set.iloc[:, :-1])
    test_set.iloc[:, :-1] = scaler_x.transform(test_set.iloc[:, :-1])

    # Output scale
    scaler_y = MinMaxScaler()
    scaler_y.fit(train_set.iloc[:, [-1]])

    train_set.iloc[:, -1] = scaler_y.transform(train_set.iloc[:, [-1]])
    test_set.iloc[:, -1] = scaler_y.transform(test_set.iloc[:, [-1]])

    # 데이터셋 생성 함수
    def build_dataset(time_series, seq_length):
        dataX = []
        dataY = []
        for i in range(0, len(time_series) - seq_length):
            _x = time_series[i : i + seq_length, :]
            _y = time_series[i + seq_length, [-1]]
            # print(_x, "-->",_y)
            dataX.append(_x)
            dataY.append(_y)

        return np.array(dataX), np.array(dataY)

    trainX, trainY = build_dataset(np.array(train_set), seq_length)
    testX, testY = build_dataset(np.array(test_set), seq_length)

    train_dataset = NumpyDataset(trainX, trainY)
    eval_dataset = NumpyDataset(testX, testY)
    # if hparams.eval_datasets_path:
    #     eval_df = pd.read_csv(hparams.eval_datasets_path)
    #     train_dataset = PandasDataset(train_df, length_column=hparams.length_column)
    # else:
    #     # it is just for lstm example
    #     train_df = train_df[::-1]
    #     train_size = int(len(train_df) * hparams.train_data_ratio)
    #     splited_train_df = train_df[0:train_size]
    #     eval_df = train_df[train_size - seq_length :]
    #     train_dataset = PandasDataset(splited_train_df, length_column=hparams.length_column)
    #     # if you use another one, plz check here
    #     # train_size = int(hparams.train_data_ratio * len(train_df))
    #     # eval_size = len(train_df) - train_size
    #     # train_dataset = PandasDataset(hparams.train_datasets_path)
    #     # train_dataset, eval_dataset = random_split(train_dataset, [train_size, eval_size])
    # eval_dataset = PandasDataset(eval_df, length_column=hparams.length_column)

    # Instantiate objects
    model = Net(input_dim=5, hidden_dim=10, seq_len=seq_length, output_dim=1, layers=1)
    wandb.watch(model, log_freq=hparams.log_every_n_steps)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hparams.learning_rate,
        eps=hparams.optim_eps,
        betas=(hparams.optim_beta1, hparams.optim_beta2),
        weight_decay=hparams.weight_decay,
    )
    if hparams.group_by_length:
        custom_train_sampler = LengthGroupedSampler(
            batch_size=hparams.batch_size, dataset=train_dataset, model_input_name=train_dataset.length_column
        )
        custom_eval_sampler = LengthGroupedSampler(
            batch_size=hparams.batch_size, dataset=eval_dataset, model_input_name=eval_dataset.length_column
        )
    else:
        custom_train_sampler = SequentialSampler(train_dataset)
        custom_eval_sampler = SequentialSampler(eval_dataset)
        # custom_train_sampler = RandomSampler(train_dataset)
        # custom_eval_sampler = RandomSampler(eval_dataset)

    train_dataloader = CustomDataLoader(
        dataset=train_dataset,
        key_to_inputs=["inputs"],
        key_to_labels=["labels"],
        batch_size=hparams.per_device_train_batch_size,
        num_workers=hparams.num_workers,
        drop_last=True,
        shuffle=True,
    )
    eval_dataloader = CustomDataLoader(
        dataset=eval_dataset,
        key_to_inputs=["inputs"],
        key_to_labels=["labels"],
        batch_size=hparams.per_device_eval_batch_size,
        num_workers=hparams.num_workers,
        drop_last=True,
        shuffle=True,
    )

    steps_per_epoch = math.ceil(len(train_dataloader) // (1 * hparams.accumulate_grad_batches))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=hparams.learning_rate,
        pct_start=hparams.warmup_ratio,
        epochs=hparams.max_epochs,
        final_div_factor=hparams.final_div_factor,
        steps_per_epoch=steps_per_epoch,
    )
    lr_scheduler = {"interval": "step", "scheduler": scheduler, "name": "AdamW", "frequency": 1, "monitor": None}
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

    log_str = f"""##########################################
    train_dataloader addr: {chk_addr_dict["train_dataloader"]}
    eval_dataloader addr: {chk_addr_dict["eval_dataloader"]}
    model addr: {chk_addr_dict["model"]}
    optimizer addr: {chk_addr_dict["optimizer"]}
    criterion addr: {chk_addr_dict["criterion"]}
    scheduler_cfg addr: {chk_addr_dict["scheduler_cfg"]}
    scheduler addr: {chk_addr_dict["scheduler_cfg[scheduler]"]}
    ##########################################
    """
    logger.info(log_str)

    trainer = Trainer(
        cmd_logger=logger,
        web_logger=web_logger,
        max_epochs=hparams.max_epochs,
        grad_accum_steps=hparams.accumulate_grad_batches,
        total_global_step=steps_per_epoch,
        chk_addr_dict=chk_addr_dict,
        checkpoint_dir=hparams.output_dir,
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
        wandb_upload_wait=10,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(TrainingArguments, dest="training_args")
    args = parser.parse_args()
    args = dataclass_to_namespace(args, "training_args")

    main(args)
