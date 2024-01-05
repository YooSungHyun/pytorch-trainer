import os
import torch


def load_checkpoint(
    state: dict,
    checkpoint_filepath: str,
    device: str = "cpu",
    logger=None,
    load_bias: bool = True,
):
    """Load model checkpoint.

    Args:
        checkpoint_filepath: checkpoint filename including path.
        model: torch model.
        optimizer: torch optimizer.

    Returns:
        state: dict
    """
    assert os.path.isfile(checkpoint_filepath)
    checkpoint_dict = torch.load(checkpoint_filepath, map_location=device)

    # epoch
    current_epoch = (
        checkpoint_dict["current_epoch"]
        if ("current_epoch" in checkpoint_dict.keys() and checkpoint_dict["current_epoch"] is not None)
        else 0
    )
    state.update({"current_epoch": current_epoch})

    # global_step
    global_step = (
        checkpoint_dict["global_step"]
        if "global_step" in checkpoint_dict.keys() and checkpoint_dict["global_step"] is not None
        else 0
    )
    state.update({"global_step": global_step})

    # step
    step = checkpoint_dict["step"] if "step" in checkpoint_dict.keys() and checkpoint_dict["step"] is not None else 0
    state.update({"step": step})

    # dtype
    dtype = (
        checkpoint_dict["dtype"]
        if "dtype" in checkpoint_dict.keys() and checkpoint_dict["dtype"] is not None
        else torch.float32
    )
    state.update({"dtype": dtype})

    # optimizer
    if (
        "optimizer" in state.keys()
        and state["optimizer"] is not None
        and "optimizer" in checkpoint_dict.keys()
        and checkpoint_dict["optimizer"] is not None
    ):
        state["optimizer"].load_state_dict(checkpoint_dict["optimizer"])

    # scheduler_cfg
    if (
        "scheduler_cfg" in state.keys()
        and state["scheduler_cfg"] is not None
        and "scheduler" in checkpoint_dict.keys()
        and checkpoint_dict["scheduler"] is not None
    ):
        state["scheduler_cfg"]["scheduler"].load_state_dict(checkpoint_dict["scheduler"])

    # trainable_loss
    if (
        "grad_scaler" in state.keys()
        and state["grad_scaler"] is not None
        and "grad_scaler" in checkpoint_dict.keys()
        and checkpoint_dict["grad_scaler"] is not None
    ):
        state.update({"grad_scaler": checkpoint_dict["grad_scaler"]})

    # trainable_loss
    if (
        "trainable_loss" in state.keys()
        and state["trainable_loss"] is not None
        and "trainable_loss" in checkpoint_dict.keys()
        and checkpoint_dict["trainable_loss"] is not None
    ):
        state["trainable_loss"].load_state_dict(checkpoint_dict["trainable_loss"], strict=False)

    # model
    assert "model" in checkpoint_dict, "checkpoint_dict must have model state dict."
    saved_state_dict = checkpoint_dict["model"]
    # 불러온 체크포인트와의 shape등을 비교하여, 맞지 않으면 기존에 초기화된걸 쓰기 위한 로직
    # saved_state_dict은 불러올 모델, state_dict 초기화된 모델을 의미한다.
    if hasattr(state["model"], "module"):
        state_dict = state["model"].module.state_dict()
    else:
        state_dict = state["model"].state_dict()

    # 먼가 다르면 초기화된 weight를 쓰고, 아니면 불러온다.
    new_state_dict = {}
    for k in state_dict.keys():
        if k in saved_state_dict.keys() and state_dict[k].size() == saved_state_dict[k].size():
            if not load_bias and k.split(".")[-1] == "bias":
                new_state_dict[k] = state_dict[k]
            else:
                new_state_dict[k] = saved_state_dict[k]

    if hasattr(state["model"], "module"):
        state["model"].module.load_state_dict(new_state_dict)
    else:
        state["model"].load_state_dict(new_state_dict)

    if logger is not None:
        logger.info(f"Loaded checkpoint '{checkpoint_filepath}' epoch: {current_epoch} global_step: {global_step}")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim,
    scheduler_cfg: dict,
    current_epoch: int,
    global_step: int,
    step: int,
    dtype: torch.dtype,
    checkpoint_filepath: str,
    grad_scaler=None,
    trainable_loss=None,
    logger=None,
):
    """Save model epoch, global step, state dict and optimizer.

    Args:
        model: torch model.
        optimizer: torch optimizer.
        epoch: epoch.
        global_step: global step.
        checkpoint_filepath: checkpoint filename including path.
    """
    # assert os.path.isfile(checkpoint_filepath)
    if logger is not None:
        logger.info("Saving model and optimizer state at epoch {} to {}".format(current_epoch, checkpoint_filepath))

    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    optimizer_state_dict = None
    if optimizer is not None:
        if hasattr(optimizer, "module"):
            optimizer_state_dict = optimizer.module.state_dict()
        else:
            optimizer_state_dict = optimizer.state_dict()

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

    torch.save(
        {
            "current_epoch": current_epoch,
            "global_step": global_step,
            "step": step,
            "model": state_dict,
            "optimizer": optimizer_state_dict,
            "scheduler": scheduler_state_dict,
            "grad_scaler": grad_scaler,
            "trainable_loss": loss_state_dict,
            "dtype": dtype,
        },
        checkpoint_filepath,
    )
