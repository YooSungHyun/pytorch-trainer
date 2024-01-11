import logging
import math
import os
from logging import StreamHandler
from typing import Optional, Union

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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from trainer.ddp import Trainer
from utils.comfy import (
    dataclass_to_namespace,
    seed_everything,
    apply_to_collection,
    web_log_every_n,
    tensor_dict_to_device,
)
from utils.data.custom_dataloader import CustomDataLoader
from utils.data.custom_sampler import DistributedLengthGroupedSampler
from utils.data.np_dataset import NumpyDataset
from torch.cuda.amp import autocast

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)8s] %(message)s")

timeFileHandler = StreamHandler()
timeFileHandler.setFormatter(formatter)

logger.addHandler(timeFileHandler)


# TODO(User): override training_step and eval_loop for your style
class DDPTrainer(Trainer):
    def __init__(
        self,
        device_id,
        criterion,
        eval_metric,
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
        max_norm: float = 0.0,
        metric_on_cpu: bool = False,
    ) -> None:
        """Exemplary Trainer with Fabric. This is a very simple trainer focused on readablity but with reduced
        featureset. As a trainer with more included features, we recommend using the

        Args:
            precision: fp16, bf16, fp32

            loggers: A single logger or a list of loggers.

            max_epochs: The maximum number of epochs to train
            max_steps: The maximum number of (optimizer) steps to train
            grad_accum_steps: How many batches to process before each optimizer step
            limit_train_batches: Limits the number of train batches per epoch
                If greater than number of batches in the dataloader, this has no effect.
            limit_val_batches: Limits the number of validation batches per epoch.
                If greater than number of batches in the dataloader, this has no effect.
            validation_frequency: How many epochs to run before each validation epoch.
            checkpoint_dir: Directory to store checkpoints to.
            checkpoint_frequency: How many epochs to run before each checkpoint is written.
            non_blocking: async data transfer cpu to gpu or reverse. (if ddp, true is recommanded)
        """
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

    def training_step(self, model, batch, batch_idx) -> torch.Tensor:
        """A single training step, running forward and backward. The optimizer step is called separately, as this is
        given as a closure to the optimizer step.

        Args:
            model: model to train
            batch: the batch to run the forward on
            batch_idx: index of the current batch w.r.t the current epoch

        """
        # TODO(User): fit the input and output for your model architecture!
        with autocast(enabled=self.mixed_precision, dtype=self.precision):
            labels = batch.pop("labels")

            outputs = model(**batch)
            loss = self.criterion(outputs, labels)

        def on_before_backward(loss):
            pass

        on_before_backward(loss)
        self.grad_scaler.scale(loss).backward()

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
            with autocast(enabled=self.mixed_precision, dtype=self.precision):
                labels = batch.pop("labels")

                outputs = model(**batch)
                loss = self.criterion(outputs, labels)
            # TODO(User): what do you want to log items every epoch end?
            tot_batch_logits.append(outputs.to(metric_on_device))
            tot_batch_labels.append(labels.to(metric_on_device))

            outputs = {"loss": loss}
            # avoid gradients in stored/accumulated values -> prevents potential OOM
            self._current_val_return = apply_to_collection(
                outputs, torch.Tensor, lambda x: x.detach().to(metric_on_device)
            )

            def on_validation_batch_end(eval_out, batch, batch_idx):
                pass

            on_validation_batch_end(outputs, batch, batch_idx)

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
            if self.device_id == 0:
                self._format_iterable(iterable, self._current_val_return, "val")
            eval_step += 1

        # TODO(User): Create any form you want to output to wandb!
        def on_validation_epoch_end(tot_batch_logits, tot_batch_labels, metric_device):
            tot_batch_logits = torch.cat(tot_batch_logits, dim=0)
            tot_batch_labels = torch.cat(tot_batch_labels, dim=0)

            # all_gather` requires a `fixed length tensor` as input.
            # Since the length of the data on each GPU may be different, the length should be passed to `all_gather` first.
            local_size = torch.tensor([tot_batch_logits.size(0)], dtype=torch.long, device=metric_device)
            size_list = [
                torch.tensor([0], dtype=torch.long, device=metric_device) for _ in range(dist.get_world_size())
            ]
            if metric_device == torch.device("cpu"):
                dist.all_gather_object(size_list, local_size)
            else:
                dist.all_gather(size_list, local_size)

            # Create a fixed length tensor with the length of `all_gather`.
            logits_gathered_data = [
                torch.zeros(
                    (size.item(), tot_batch_logits.size(-1)), dtype=tot_batch_logits.dtype, device=metric_device
                )
                for size in size_list
            ]
            labels_gathered_data = [
                torch.zeros(
                    (size.item(), tot_batch_labels.size(-1)), dtype=tot_batch_labels.dtype, device=metric_device
                )
                for size in size_list
            ]

            # Collect and match data from all GPUs.
            if metric_device == torch.device("cpu"):
                # Collect and match data from all GPUs.
                dist.all_gather_object(logits_gathered_data, tot_batch_logits)
                dist.all_gather_object(labels_gathered_data, tot_batch_labels)
            else:
                dist.all_gather(logits_gathered_data, tot_batch_logits)
                dist.all_gather(labels_gathered_data, tot_batch_labels)

            # example 4 gpus : [gpu0[tensor],gpu1[tensor],gpu2[tensor],gpu3[tensor]]
            logits_gathered_data = torch.cat(logits_gathered_data, dim=0)
            labels_gathered_data = torch.cat(labels_gathered_data, dim=0)
            epoch_loss = self.criterion(logits_gathered_data, labels_gathered_data)
            epoch_rmse = torch.sqrt(epoch_loss)
            # epoch monitoring is must doing every epoch
            web_log_every_n(
                self.web_logger,
                {"eval/loss": epoch_rmse, "eval/epoch": self.current_epoch},
                self.current_epoch,
                1,
                self.device_id,
            )

        on_validation_epoch_end(tot_batch_logits, tot_batch_labels, metric_on_device)

        def on_validation_model_train(model):
            torch.set_grad_enabled(True)
            model.train()

        on_validation_model_train(model)


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

    # reference: https://www.kaggle.com/code/anitarostami/lstm-multivariate-forecasting
    setproctitle(os.environ.get("WANDB_PROJECT", "torch-trainer"))
    web_logger = None
    if local_rank == 0:
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
    model = Net().cuda(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

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
    # TODO(User): input your eval_metric
    eval_metric = None
    trainer = DDPTrainer(
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
        model=ddp_model,
        optimizer=optimizer,
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
