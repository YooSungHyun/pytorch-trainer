import os
from collections.abc import Mapping
from functools import partial
from typing import Any, Iterable, List, Literal, Optional, Tuple, Union, cast
import time
import torch
from utils.comfy import apply_to_collection, save_checkpoint, load_checkpoint, tensor_dict_to_device, web_log_every_n
from tqdm import tqdm

precision_dict = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


class Trainer:
    def __init__(
        self,
        total_global_step: int,
        precision="fp32",
        cmd_logger=None,
        web_logger=None,
        max_epochs: Optional[int] = 1000,
        max_steps: Optional[int] = None,
        grad_accum_steps: int = 1,
        limit_train_batches: Union[int, float] = float("inf"),
        limit_val_batches: Union[int, float] = float("inf"),
        validation_frequency: int = 1,
        use_distributed_sampler: bool = True,
        checkpoint_dir: str = "./checkpoints",
        checkpoint_frequency: int = 1,
        chk_addr_dict: dict = None,
        non_blocking: bool = True,
        log_every_n: int = 1,
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
            use_distributed_sampler: Wraps the sampler of each dataloader with a respective distributed-aware sampler
                in case of distributed training.
            checkpoint_dir: Directory to store checkpoints to.
            checkpoint_frequency: How many epochs to run before each checkpoint is written.
            non_blocking: async data transfer cpu to gpu or reverse. (if ddp, true is recommanded)
        """
        if precision in ["fp16", "float16"]:
            self.precision = precision_dict["fp16"]
        elif precision in ["bf16" or "bfloat16"]:
            self.precision = precision_dict["bf16"]
        else:
            self.precision = precision_dict["fp32"]

        self.logger = cmd_logger
        self.web_logger = web_logger
        self.chk_addr_dict = chk_addr_dict

        self.global_step = 0
        self.step = 0
        self.total_global_step = total_global_step
        self.grad_accum_steps: int = grad_accum_steps
        self.current_epoch = 0

        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.should_stop = False

        self.non_blocking = non_blocking

        # ensures limit_X_batches is either int or inf
        if not isinstance(limit_train_batches, int):
            assert limit_train_batches == float("inf")

        if not isinstance(limit_val_batches, int):
            assert limit_val_batches == float("inf")

        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.validation_frequency = validation_frequency
        self.use_distributed_sampler = use_distributed_sampler
        self._current_train_return: Union[torch.Tensor, Mapping[str, Any]] = {}
        self._current_val_return: Optional[Union[torch.Tensor, Mapping[str, Any]]] = {}

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_frequency = checkpoint_frequency

        self.non_blocking = non_blocking
        self.log_every_n = log_every_n

    def fit(
        self,
        model,
        optimizer,
        scheduler_cfg: Optional[Mapping],
        criterion,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        trainable_loss=None,
        ckpt_path: Optional[str] = None,
    ):
        """The main entrypoint of the trainer, triggering the actual training.

        Args:
            model: model
            train_loader: the training dataloader. Has to be an iterable returning batches.
            val_loader: the validation dataloader. Has to be an iterable returning batches.
                If not specified, no validation will run.
            ckpt_path: Path to previous checkpoints to resume training from.
                If specified, will always look for the latest checkpoint within the given directory.

        """
        # assemble state (current epoch and global step will be added in save)
        state = {
            "model": model,
            "optimizer": optimizer,
            "scheduler_cfg": scheduler_cfg,
            "trainable_loss": trainable_loss,
        }

        # load last checkpoint if available
        if ckpt_path is not None and os.path.isdir(ckpt_path):
            latest_checkpoint_path = self.get_latest_checkpoint(self.checkpoint_dir)
            if latest_checkpoint_path is not None:
                state.update(self.load(state, latest_checkpoint_path))
                # check if we even need to train here
                if self.max_epochs is not None and self.current_epoch >= self.max_epochs:
                    self.should_stop = True

        self.criterion = criterion
        while not self.should_stop:
            self.train_loop(
                state["model"],
                state["optimizer"],
                scheduler_cfg=state["scheduler_cfg"],
                train_loader=train_loader,
                limit_batches=self.limit_train_batches,
            )

            if self.should_validate:
                self.eval_loop(state["model"], val_loader, limit_batches=self.limit_val_batches)

            self.step_scheduler(state["scheduler_cfg"], level="epoch", current_value=self.current_epoch)

            self.current_epoch += 1

            # stopping condition on epoch level
            if self.max_epochs is not None and self.current_epoch >= self.max_epochs:
                self.should_stop = True

            self.save(state)

        # reset for next fit call
        self.should_stop = False

    def train_loop(
        self,
        model,
        optimizer,
        scheduler_cfg: Optional[Mapping],
        train_loader: torch.utils.data.DataLoader,
        limit_batches: Union[int, float] = float("inf"),
        trainable_loss=None,
    ):
        """The training loop running a single training epoch.

        Args:
            model: model to train
            optimizer: the optimizer
            train_loader: The dataloader yielding the training batches.
            limit_batches: Limits the batches during this training epoch.
                If greater than the number of batches in the ``train_loader``, this has no effect.
            scheduler_cfg: The learning rate scheduler configuration.
        """

        def on_train_epoch_start():
            model.train()

        on_train_epoch_start()
        iterable = self.progbar_wrapper(
            train_loader, total=min(self.total_global_step, limit_batches), desc=f"Epoch {self.current_epoch}"
        )

        for batch_idx, batch in enumerate(iterable):
            tensor_dict_to_device(batch, "cpu", non_blocking=self.non_blocking)
            # end epoch if stopping training completely or max batches for this epoch reached
            if self.should_stop or batch_idx >= limit_batches:
                break

            def on_train_batch_start(batch, batch_idx):
                pass

            on_train_batch_start(batch, batch_idx)
            # check if optimizer should step in gradient accumulation
            should_optim_step = self.global_step % self.grad_accum_steps == 0
            global_loss = 0
            if should_optim_step:
                # currently only supports a single optimizer
                def on_before_optimizer_step(optimizer, opt_idx):
                    pass

                on_before_optimizer_step(optimizer, 0)
                # optimizer step runs train step internally through closure
                loss = self.training_step(model=model, batch=batch, batch_idx=batch_idx)
                global_loss += loss
                mean_global_loss = global_loss / self.grad_accum_steps
                # global_step loss check
                web_log_every_n(
                    self.web_logger,
                    {
                        "train/global_step_loss": mean_global_loss,
                        "train/global_step": self.global_step,
                        "train/epoch": self.current_epoch,
                    },
                    self.global_step,
                    self.log_every_n,
                )
                optimizer.step()

                def on_before_zero_grad(optimizer):
                    pass

                on_before_zero_grad(optimizer)
                optimizer.zero_grad()
            else:
                # gradient accumulation -> no optimizer step
                loss = self.training_step(model=model, batch=batch, batch_idx=batch_idx)
                global_loss += loss

            def on_train_batch_end(outputs, batch, batch_idx):
                pass

            on_train_batch_end(self._current_train_return, batch, batch_idx)
            # this guard ensures, we only step the scheduler once per global step
            if should_optim_step:
                web_log_every_n(
                    self.web_logger,
                    {
                        "train/optim_0_learning_rate": optimizer.param_groups[0]["lr"],
                        "train/global_step": self.global_step,
                        "train/epoch": self.current_epoch,
                    },
                    self.global_step,
                    self.log_every_n,
                )
                self.step_scheduler(scheduler_cfg, level="step", current_value=self.global_step)

            # add output values to progress bar
            self._format_iterable(iterable, self._current_train_return, "train")
            self.step += 1
            # only increase global step if optimizer stepped
            self.global_step += int(should_optim_step)

            # stopping criterion on step level
            if self.max_steps is not None and self.global_step >= self.max_steps:
                self.should_stop = True
                break

        def on_train_epoch_end():
            pass

        on_train_epoch_end()

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

        def on_validation_model_eval():
            pass

        torch.set_grad_enabled(False)

        def on_validation_epoch_start():
            pass

        iterable = self.progbar_wrapper(val_loader, total=min(len(val_loader), limit_batches), desc="Validation")
        eval_step = 0
        tot_batch_result = list()
        tot_batch_labels = list()
        for batch_idx, batch in enumerate(iterable):
            # end epoch if stopping training completely or max batches for this epoch reached
            if self.should_stop or batch_idx >= limit_batches:
                break

            def on_validation_batch_start(batch, batch_idx):
                pass

            on_validation_batch_start(batch, batch_idx)

            # TODO(User): If you needs more labels than 1, must change this line (make your labels)
            labels = batch.pop("labels")

            outputs = model(**batch)
            loss = self.criterion(outputs, labels)

            tot_batch_result.append(outputs)
            tot_batch_labels.append(labels)

            outputs = {"loss": loss}
            # avoid gradients in stored/accumulated values -> prevents potential OOM
            outputs = apply_to_collection(outputs, torch.Tensor, lambda x: x.detach())

            def on_validation_batch_end(eval_out, batch, batch_idx):
                pass

            on_validation_batch_end(outputs, batch, batch_idx)
            self._current_val_return = outputs
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

        def on_validation_epoch_end(tot_batch_result, tot_batch_labels):
            tot_batch_result = torch.stack(tot_batch_result)
            tot_batch_labels = torch.stack(tot_batch_labels)
            epoch_loss = self.criterion(tot_batch_result, tot_batch_labels)
            epoch_rmse = torch.sqrt(epoch_loss)
            # epoch monitoring is must doing every epoch
            web_log_every_n(
                self.web_logger, {"eval/loss": epoch_rmse, "eval/epoch": self.current_epoch}, self.current_epoch, 1
            )

        on_validation_epoch_end(tot_batch_result, tot_batch_labels)

        def on_validation_model_train():
            pass

        on_validation_model_train()
        torch.set_grad_enabled(True)

    def training_step(self, model, batch: Any, batch_idx: int) -> torch.Tensor:
        """A single training step, running forward and backward. The optimizer step is called separately, as this is
        given as a closure to the optimizer step.

        Args:
            model: model to train
            batch: the batch to run the forward on
            batch_idx: index of the current batch w.r.t the current epoch

        """
        # TODO(User): If you needs more labels than 1, must change this line (make your labels)
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

    def step_scheduler(
        self,
        scheduler_cfg: Optional[Mapping],
        level: Literal["step", "epoch"],
        current_value: int,
    ) -> None:
        """Steps the learning rate scheduler if necessary.

        Args:
            scheduler_cfg: The learning rate scheduler configuration.
            level: whether we are trying to step on epoch- or step-level
            current_value: Holds the current_epoch if ``level==epoch``, else holds the ``global_step``

        """

        # no scheduler
        if scheduler_cfg is None:
            return

        # wrong interval (step vs. epoch)
        if scheduler_cfg["interval"] != level:
            return

        # right interval, but wrong step wrt frequency
        if current_value % cast(int, scheduler_cfg["frequency"]) != 0:
            return

        # assemble potential monitored values
        possible_monitor_vals = {None: None}
        if isinstance(self._current_train_return, torch.Tensor):
            possible_monitor_vals.update("train_loss", self._current_train_return)
        elif isinstance(self._current_train_return, Mapping):
            possible_monitor_vals.update({"train_" + k: v for k, v in self._current_train_return.items()})

        if isinstance(self._current_val_return, torch.Tensor):
            possible_monitor_vals.update("val_loss", self._current_val_return)
        elif isinstance(self._current_val_return, Mapping):
            possible_monitor_vals.update({"val_" + k: v for k, v in self._current_val_return.items()})

        try:
            monitor = possible_monitor_vals[cast(Optional[str], scheduler_cfg["monitor"])]
        except KeyError as ex:
            possible_keys = list(possible_monitor_vals.keys())
            raise KeyError(
                f"monitor {scheduler_cfg['monitor']} is invalid. Possible values are {possible_keys}."
            ) from ex

        # rely on model hook for actual step
        if monitor:
            scheduler_cfg["scheduler"].step(monitor)
        else:
            scheduler_cfg["scheduler"].step()

    @property
    def should_validate(self) -> bool:
        """Whether to currently run validation."""
        return self.current_epoch % self.validation_frequency == 0

    def progbar_wrapper(self, iterable: Iterable, total: int, **kwargs: Any):
        """Wraps the iterable with tqdm for global rank zero.

        Args:
            iterable: the iterable to wrap with tqdm
            total: the total length of the iterable, necessary in case the number of batches was limited.

        """
        # in cpu, it just one
        return tqdm(iterable, total=total, **kwargs)

    def load(self, state: Optional[Mapping], path: str) -> None:
        """Loads a checkpoint from a given file into state.

        Args:
            state: a mapping contaning model, optimizer and lr scheduler
            path: the path to load the checkpoint from

        """
        if state is None:
            state = {}

        load_checkpoint(state, checkpoint_filepath=path, logger=self.logger)
        self.global_step = state.pop("global_step")
        self.step = state.pop("step")
        self.current_epoch = state.pop("current_epoch")

        if state:
            self.logger.info(f"Unused Checkpoint Values: {state}, returned")

        return state

    def save(self, state: Optional[Mapping]) -> None:
        """Saves a checkpoint to the ``checkpoint_dir``

        Args:
            state: A mapping containing model, optimizer and lr scheduler.

        """
        if state is None:
            state = {}

        state.update(global_step=self.global_step, current_epoch=self.current_epoch, step=self.step)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        save_checkpoint(
            **state,
            checkpoint_filepath=os.path.join(self.checkpoint_dir, f"epoch-{self.current_epoch:04d}.ckpt"),
            logger=self.logger,
        )

    @staticmethod
    def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
        """Returns the latest checkpoint from the ``checkpoint_dir``

        Args:
            checkpoint_dir: the directory to search for checkpoints

        """
        if not os.path.isdir(checkpoint_dir):
            return None

        items = sorted(os.listdir(checkpoint_dir))

        if not items:
            return None

        return os.path.join(checkpoint_dir, items[-1])

    @staticmethod
    def _format_iterable(
        prog_bar, candidates: Optional[Union[torch.Tensor, Mapping[str, Union[torch.Tensor, float, int]]]], prefix: str
    ):
        """Adds values as postfix string to progressbar.

        Args:
            prog_bar: a progressbar (on global rank zero) or an iterable (every other rank).
            candidates: the values to add as postfix strings to the progressbar.
            prefix: the prefix to add to each of these values.

        """
        if isinstance(prog_bar, tqdm) and candidates is not None:
            postfix_str = ""
            float_candidates = apply_to_collection(candidates, torch.Tensor, lambda x: x.item())
            if isinstance(candidates, torch.Tensor):
                postfix_str += f" {prefix}_loss: {float_candidates:.20f}"
            elif isinstance(candidates, Mapping):
                for k, v in float_candidates.items():
                    postfix_str += f" {prefix}_{k}: {v:.20f}"

            if postfix_str:
                prog_bar.set_postfix_str(postfix_str)
