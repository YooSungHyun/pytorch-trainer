import torch


def load_checkpoint(state: dict, checkpoint_filepath: str, logger=None):
    """Load model checkpoint.

    Args:
        checkpoint_filepath: checkpoint filename including path.
        model: torch model.
        optimizer: torch optimizer.

    Returns:
        state: dict
    """
    model_path, additional_state = state["model"].load_checkpoint(checkpoint_filepath)
    # optimizer
    state["optimizer"] = state["model"].optimizer

    # epoch
    current_epoch = (
        additional_state["current_epoch"]
        if ("current_epoch" in additional_state.keys() and additional_state["current_epoch"] is not None)
        else 0
    )
    state.update({"current_epoch": current_epoch})

    # global_step
    assert state["model"].global_steps == additional_state["global_step"], "save global_steps is mismatch. chk plz"
    state.update({"global_step": state["model"].global_steps})

    # step
    step = (
        additional_state["step"] if "step" in additional_state.keys() and additional_state["step"] is not None else 0
    )
    state.update({"step": step})

    # dtype
    dtype = (
        additional_state["dtype"]
        if "dtype" in additional_state.keys() and additional_state["dtype"] is not None
        else torch.float32
    )
    state.update({"dtype": dtype})

    # scheduler_cfg
    if (
        "scheduler_cfg" in state.keys()
        and state["scheduler_cfg"] is not None
        and "scheduler" in additional_state.keys()
        and additional_state["scheduler"] is not None
    ):
        state["scheduler_cfg"]["scheduler"].load_state_dict(additional_state["scheduler"])

    # trainable_loss
    if (
        "trainable_loss" in state.keys()
        and state["trainable_loss"] is not None
        and "trainable_loss" in additional_state.keys()
        and additional_state["trainable_loss"] is not None
    ):
        state["trainable_loss"].load_state_dict(additional_state["trainable_loss"], strict=False)

    if logger is not None:
        logger.info(f"Loaded checkpoint '{model_path}' epoch: {current_epoch} global_step: {state['global_step']}")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer,
    scheduler_cfg: dict,
    current_epoch: int,
    global_step: int,
    step: int,
    dtype: torch.dtype,
    checkpoint_filepath: str,
    trainable_loss=None,
    logger=None,
):
    """Save model epoch, global step, state dict.

    Args:
        model: torch model.
        epoch: epoch.
        global_step: global step.
        checkpoint_filepath: checkpoint filename including path.
    """
    # assert os.path.isfile(checkpoint_filepath)
    # deepspeed saved model and optimizer automatically
    if logger is not None:
        logger.info("Saving model and optimizer state at epoch {} to {}".format(current_epoch, checkpoint_filepath))

    scheduler_state_dict = None
    if scheduler_cfg["scheduler"] is not None:
        if hasattr(scheduler_cfg["scheduler"], "module"):
            scheduler_state_dict = scheduler_cfg["scheduler"].module.state_dict()
        else:
            scheduler_state_dict = scheduler_cfg["scheduler"].state_dict()

    loss_state_dict = None
    if trainable_loss is not None:
        if hasattr(trainable_loss, "module"):
            loss_state_dict = trainable_loss.module.state_dict()
        else:
            loss_state_dict = trainable_loss.state_dict()

    model.save_checkpoint(
        checkpoint_filepath,
        client_state={
            "current_epoch": current_epoch,
            "global_step": global_step,
            "step": step,
            "scheduler": scheduler_state_dict,
            "trainable_loss": loss_state_dict,
            "dtype": dtype,
        },
    )
