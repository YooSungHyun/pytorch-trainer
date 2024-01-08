import logging
import math
import os
from logging import StreamHandler
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
import wandb
from arguments.training_args import TrainingArguments
from networks.models import Net
from setproctitle import setproctitle
from simple_parsing import ArgumentParser
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import RandomSampler, SequentialSampler, random_split
from trainer.cpu import Trainer
from utils.comfy import (
    apply_to_collection,
    dataclass_to_namespace,
    json_to_dict,
    seed_everything,
    tensor_dict_to_device,
    update_auto_nested_dict,
    web_log_every_n,
)
from utils.data.custom_dataloader import CustomDataLoader
from utils.data.custom_sampler import LengthGroupedSampler
from utils.data.np_dataset import NumpyDataset

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)8s] %(message)s")

timeFileHandler = StreamHandler()
timeFileHandler.setFormatter(formatter)

logger.addHandler(timeFileHandler)


# TODO(User): override training_step and eval_loop for your style
class CPUTrainer(Trainer):
    def __init__(
        self,
        criterion,
        eval_metric=None,
        precision="fp32",
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
    ):
        super().__init__(
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
        )

    def training_step(self, model, batch, batch_idx) -> torch.Tensor:
        """A single training step, running forward and backward. The optimizer step is called separately, as this is
        given as a closure to the optimizer step.

        Args:
            model: model to train
            batch: the batch to run the forward on
            batch_idx: index of the current batch w.r.t the current epoch

        """
        # TODO(User): fit the input and output for your model architecture!
        labels = batch.pop("labels")

        outputs = model(**batch)
        loss = self.criterion(outputs, labels)

        def on_before_backward(loss):
            pass

        on_before_backward(loss)
        loss.backward()

        def on_after_backward():
            pass

        on_after_backward()

        outputs = {"loss": loss}
        # avoid gradients in stored/accumulated values -> prevents potential OOM
        self._current_train_return = apply_to_collection(outputs, dtype=torch.Tensor, function=lambda x: x.detach())

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

        iterable = self.progbar_wrapper(val_loader, total=min(len(val_loader), limit_batches), desc="Validation")
        eval_step = 0
        tot_batch_logits = list()
        tot_batch_labels = list()
        for batch_idx, batch in enumerate(iterable):
            tensor_dict_to_device(batch, "cpu", non_blocking=self.non_blocking)
            # end epoch if stopping training completely or max batches for this epoch reached
            if self.should_stop or batch_idx >= limit_batches:
                break

            def on_validation_batch_start(batch, batch_idx):
                pass

            on_validation_batch_start(batch, batch_idx)

            # TODO(User): fit the input and output for your model architecture!
            label = batch.pop("labels")

            output = model(**batch)

            loss = self.criterion(output, label)

            # TODO(User): what do you want to log items every epoch end?
            tot_batch_logits.append(output)
            tot_batch_labels.append(label)

            log_output = {"loss": loss}
            # avoid gradients in stored/accumulated values -> prevents potential OOM
            self._current_val_return = apply_to_collection(log_output, torch.Tensor, lambda x: x.detach())

            def on_validation_batch_end(eval_out, batch, batch_idx):
                pass

            on_validation_batch_end(output, batch, batch_idx)

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
            )
            self._format_iterable(iterable, self._current_val_return, "val")
            eval_step += 1

        # TODO(User): Create any form you want to output to wandb!
        def on_validation_epoch_end(tot_batch_logits, tot_batch_labels):
            tot_batch_logits = torch.cat(tot_batch_logits, dim=0)
            tot_batch_labels = torch.cat(tot_batch_labels, dim=0)
            epoch_loss = self.criterion(tot_batch_logits, tot_batch_labels)
            epoch_rmse = torch.sqrt(epoch_loss)
            # epoch monitoring is must doing every epoch
            web_log_every_n(
                self.web_logger, {"eval/loss": epoch_rmse, "eval/epoch": self.current_epoch}, self.current_epoch, 1
            )

        on_validation_epoch_end(tot_batch_logits, tot_batch_labels)

        def on_validation_model_train(model):
            torch.set_grad_enabled(True)
            model.train()

        on_validation_model_train(model)


def main(hparams: TrainingArguments):
    # reference: https://www.kaggle.com/code/anitarostami/lstm-multivariate-forecasting
    setproctitle(os.environ.get("WANDB_PROJECT", "torch-trainer"))
    web_logger = wandb.init(config=hparams)
    seed_everything(hparams.seed)

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

    # Instantiate objects
    model = Net()
    web_logger.watch(model, log_freq=hparams.log_every_n)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hparams.learning_rate,
        eps=hparams.optim_eps,
        betas=(hparams.optim_beta1, hparams.optim_beta2),
        weight_decay=hparams.weight_decay,
    )

    generator = None
    if hparams.sampler_shuffle:
        generator = torch.Generator()
        generator.manual_seed(hparams.seed)
    if hparams.group_by_length:
        custom_train_sampler = LengthGroupedSampler(
            batch_size=hparams.per_device_train_batch_size,
            dataset=train_dataset,
            model_input_name=train_dataset.length_column_name,
            generator=generator,
        )
        custom_eval_sampler = LengthGroupedSampler(
            batch_size=hparams.per_device_eval_batch_size,
            dataset=eval_dataset,
            model_input_name=eval_dataset.length_column_name,
        )
    else:
        # custom_train_sampler = SequentialSampler(train_dataset)
        custom_eval_sampler = SequentialSampler(eval_dataset)
        custom_train_sampler = RandomSampler(train_dataset, generator=generator)
        # custom_eval_sampler = RandomSampler(eval_dataset, generator=generator)

    # If 1 device for training, sampler suffle True and dataloader shuffle True is same meaning
    train_dataloader = CustomDataLoader(
        dataset=train_dataset,
        feature_column_name=hparams.feature_column_name,
        labels_column_name=hparams.labels_column_name,
        batch_size=hparams.per_device_train_batch_size,
        sampler=custom_train_sampler,
        num_workers=hparams.num_workers,
        drop_last=hparams.dataloader_drop_last,
    )

    eval_dataloader = CustomDataLoader(
        dataset=eval_dataset,
        feature_column_name=hparams.feature_column_name,
        labels_column_name=hparams.labels_column_name,
        batch_size=hparams.per_device_eval_batch_size,
        sampler=custom_eval_sampler,
        num_workers=hparams.num_workers,
        drop_last=hparams.dataloader_drop_last,
    )

    # dataloader already calculate total_data / batch_size
    # accumulation is always floor
    train_steps_per_epoch = math.floor(len(train_dataloader) / (hparams.accumulate_grad_batches))

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=hparams.learning_rate,
        pct_start=hparams.warmup_ratio,
        epochs=hparams.max_epochs,
        final_div_factor=hparams.final_div_factor,
        steps_per_epoch=train_steps_per_epoch,
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
    # TODO(User): input your eval_metric
    eval_metric = None
    trainer = CPUTrainer(
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
    )

    trainer.fit(
        model=model,
        optimizer=optimizer,
        scheduler_cfg=lr_scheduler,
        train_loader=train_dataloader,
        val_loader=eval_dataloader,
        ckpt_path=hparams.output_dir,
        trainable_loss=trainable_loss,
    )

    web_logger.finish(exit_code=0)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(TrainingArguments, dest="training_args")
    args = parser.parse_args()
    args = dataclass_to_namespace(args, "training_args")

    main(args)
