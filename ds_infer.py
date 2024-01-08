import logging
import math
import os
from logging import StreamHandler
from typing import Optional, Union

import deepspeed
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from arguments.inference_args import InferenceArguments
from networks.models import Net
from setproctitle import setproctitle
from simple_parsing import ArgumentParser
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DistributedSampler
from trainer.deepspeed import Trainer
from utils.comfy import dataclass_to_namespace, seed_everything, apply_to_collection, tensor_dict_to_device
from utils.data.custom_dataloader import CustomDataLoader
from utils.data.custom_sampler import DistributedLengthGroupedSampler
from utils.data.np_dataset import NumpyDataset
from torch.cuda.amp import autocast

from utils.model_checkpointing.ds_handler import load_checkpoint_for_infer

# it is only lstm example.
torch.backends.cudnn.enabled = False

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)8s] %(message)s")

timeFileHandler = StreamHandler()
timeFileHandler.setFormatter(formatter)

logger.addHandler(timeFileHandler)


# TODO(User): override training_step and eval_loop for your style
class DSTrainer(Trainer):
    def __init__(
        self, device_id, criterion, eval_metric=None, precision="fp32", cmd_logger=None, metric_on_cpu: bool = False
    ):
        super().__init__(device_id, criterion, eval_metric, precision, cmd_logger, metric_on_cpu)

    def test_loop(self, model, test_loader: Optional[torch.utils.data.DataLoader], **kwargs):
        """The test loop ruunning a single test epoch.

        Args:
            model: model
            test_loader: The dataloader yielding the test batches.

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
            with autocast(enabled=self.mixed_precision, dtype=self.precision):
                # if self.precision == torch.bfloat16:
                # tensor_dict_to_dtype(batch, self.precision)

                labels = batch.pop("labels")

                outputs = model(**batch)
                loss = self.criterion(outputs, labels)

            # TODO(User): what do you want to log items every epoch end?
            tot_batch_logits.append(outputs.to(metric_on_device))
            tot_batch_labels.append(labels.to(metric_on_device))

            log_output = {"loss": loss}
            # avoid gradients in stored/accumulated values -> prevents potential OOM
            self._current_val_return = apply_to_collection(
                log_output, torch.Tensor, lambda x: x.detach().to(metric_on_device)
            )

            def on_test_batch_end(eval_out, batch, batch_idx):
                pass

            on_test_batch_end(outputs, batch, batch_idx)

            if self.device_id == 0:
                self._format_iterable(iterable, self._current_val_return, "test")
            eval_step += 1

        # TODO(User): Create any form you want to output to wandb!
        def on_test_epoch_end(tot_batch_logits, tot_batch_labels, metric_device, **kwargs):
            # if you want to see all_reduce example, see `fsdp_train.py`'s eval_loop
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

            if self.device_id == 0:
                # example 4 gpus : [gpu0[tensor],gpu1[tensor],gpu2[tensor],gpu3[tensor]]
                logits_gathered_data = torch.cat(logits_gathered_data, dim=0)
                labels_gathered_data = torch.cat(labels_gathered_data, dim=0)
                epoch_loss = self.criterion(logits_gathered_data, labels_gathered_data)
                epoch_rmse = torch.sqrt(epoch_loss)
                self.logger.info(f"RMSE Loss is {epoch_rmse:0.10f}")
                if self.precision == torch.bfloat16:
                    pred = logits_gathered_data.to(torch.float32).cpu().numpy()
                else:
                    pred = logits_gathered_data.cpu().numpy()
                # distributed will shuffle the data for each GPU
                # so you won't be able to find the source specified here up to scaler.
                np_outputs = np.concatenate([pred, labels_gathered_data.cpu().numpy()], axis=1)
                pd_result = pd.DataFrame(np_outputs, columns=["pred", "labels"])
                pd_result.to_excel("./ds_result.xlsx", index=False)

        on_test_epoch_end(tot_batch_logits, tot_batch_labels, metric_on_device, **kwargs)


def main(hparams: InferenceArguments):
    # reference: https://www.kaggle.com/code/anitarostami/lstm-multivariate-forecasting
    setproctitle(os.environ.get("WANDB_PROJECT", "torch-trainer"))
    seed_everything(hparams.seed)
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    rank = int(os.environ.get("RANK", -1))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    logger.info(
        f"Start running basic deepspeed example on total {world_size} computers, {rank}'s process on {local_rank}'s gpu."
    )

    assert world_size > -1 and rank > -1 and local_rank > -1, "Your distributed environ is wrong, plz check and retry!"

    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed("nccl", rank=rank, world_size=world_size)

    # I'm not saved MinMaxScaler, so, have to re-calculate, stupid thing...🤣
    df_train = pd.read_csv("./raw_data/LSTM-Multivariate_pollution.csv", header=0, encoding="utf-8")
    # Kaggle author Test Final RMSE: 0.06539
    df_eval = pd.read_csv(hparams.data_path, header=0, encoding="utf-8")

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
    logger.info(df_test_scaled.head())

    df_test_scaled = np.array(df_test_scaled)

    n_future = 1
    n_past = 11

    np_labels = df_test_scaled[n_past:]
    index_column = np.arange(np_labels.shape[0]).reshape(-1, 1)
    np_idx_labels = np.concatenate([index_column, np_labels], axis=1)

    #  Test Sets
    x = []
    y = []
    for i in range(n_past, len(df_test_scaled) - n_future + 1):
        x.append(df_test_scaled[i - n_past : i, 1 : df_test_scaled.shape[1]])
        y.append(df_test_scaled[i + n_future - 1 : i + n_future, 0])
    x_test, y_test = np.array(x), np.array(y)

    logger.info("X_test shape : {}      y_test shape : {} ".format(x_test.shape, y_test.shape))

    test_dataset = NumpyDataset(
        x_test,
        y_test,
        feature_column_name=hparams.feature_column_name,
        labels_column_name=hparams.labels_column_name,
    )

    if hparams.group_by_length:
        custom_test_sampler = DistributedLengthGroupedSampler(
            batch_size=hparams.per_device_test_batch_size,
            dataset=test_dataset,
            rank=rank,
            seed=hparams.seed,
            shuffle=False,
            model_input_name=test_dataset.length_column_name,
        )
    else:
        custom_test_sampler = DistributedSampler(test_dataset, seed=hparams.seed, rank=rank, shuffle=False)

    # DataLoader's shuffle: one device get random indices dataset in every epoch
    # example np_dataset is already set (feature)7:1(label), so, it can be all shuffle `True` between sampler and dataloader
    test_dataloader = CustomDataLoader(
        dataset=test_dataset,
        feature_column_name=hparams.feature_column_name,
        labels_column_name=hparams.labels_column_name,
        batch_size=hparams.per_device_test_batch_size,
        sampler=custom_test_sampler,
        num_workers=hparams.num_workers,
        drop_last=hparams.dataloader_drop_last,
        pin_memory=True,
        persistent_workers=True,
    )

    # Instantiate objects
    model = Net().cuda(local_rank)

    state = {"model": model}
    load_checkpoint_for_infer(
        state,
        checkpoint_filepath=hparams.model_path,
        model_file_name="mp_rank_00_model_states.pt",
        device=f"cuda:{local_rank}",
        logger=logger,
    )
    # Since the deepspeed lr scheduler is, after all, just a generic object-inherited custom scheduler, Only authorize the use of torch scheduler.
    # Also, the ZeroOptimizer.param_groups address is the same as the torch scheduler.optimizer.param_groups address.
    # Therefore, there is absolutely no reason to use the lr_scheduler provided by Deepspeed.

    # in deepspeed, precision is just using model log for .pt file
    precision_dict = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    precision = torch.float32
    if precision in ["fp16", "float16"]:
        precision = precision_dict["fp16"]
    elif precision in ["bf16" or "bfloat16"]:
        precision = precision_dict["bf16"]

    model = deepspeed.init_inference(
        model=state["model"],
        mp_size=world_size,
        dtype=precision,
        injection_policy={Net: ("fc")},
        replace_with_kernel_inject=False,
    )

    # monitor: ReduceLROnPlateau scheduler is stepped using loss, so monitor input train or val loss
    eval_metric = None
    criterion = torch.nn.MSELoss()

    # TODO(User): input your eval_metric
    eval_metric = None
    trainer = DSTrainer(
        device_id=local_rank,
        criterion=criterion,
        eval_metric=eval_metric,
        precision=hparams.model_dtype,
        cmd_logger=logger,
        metric_on_cpu=hparams.metric_on_cpu,
    )

    trainer.test_loop(model=model, test_loader=test_dataloader)


if __name__ == "__main__":
    assert torch.distributed.is_available(), "DDP is only multi gpu!! check plz!"
    assert torch.cuda.is_available(), "CPU training is not allowed."
    parser = ArgumentParser()
    parser.add_arguments(InferenceArguments, dest="training_args")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    args = dataclass_to_namespace(args, "training_args")

    main(args)
