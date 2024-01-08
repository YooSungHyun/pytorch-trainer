import torch
import os


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


def load_checkpoint_for_infer(
    state: dict,
    checkpoint_filepath: str,
    model_file_name: str = "mp_rank_00_model_states.pt",
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
    if not os.path.isfile(checkpoint_filepath):
        assert model_file_name, "if you input just model main dir, must input just, model_file_name!"
        tag = None
        latest_path = os.path.join(checkpoint_filepath, "latest")
        if os.path.isfile(latest_path):
            with open(latest_path, "r") as fd:
                tag = fd.read().strip()
        checkpoint_filepath = os.path.join(checkpoint_filepath, tag, model_file_name)
    if logger:
        logger.info(f"Will loaded {checkpoint_filepath} model")
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
    assert "module" in checkpoint_dict, "checkpoint_dict must have model state dict."
    saved_state_dict = checkpoint_dict["module"]
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
